#!/usr/bin/env bash
# =============================================================================
# One-click runner for the @1024 TCT_NGC method suite:
#   fetch checkpoint + fetch dataset -> train -> eval -> AUTO-ANALYZE (figures + verdict).
# A collaborator runs ONE command and gets the result bundle WITH analysis; no manual scp,
# no separate download/analyze steps. (Set --skip-fetch / --analyze 0 to opt out of either.)
#
# ----------------------------------------------------------------------------------------
# QUICK START (collaborator, fresh clone). You only need ONE secret: the ModelScope token
# for the PRIVATE dataset. The pretrained checkpoint is PUBLIC (auto-downloads, no token).
#
#   export MODELSCOPE_API_TOKEN='<token the authors give you>'   # private dataset only
#   bash tools/run_method_suite.sh --list                        # preview arm->config->lr (no run)
#   bash tools/run_method_suite.sh --gpus 0,1,2,3,4,5,6,7 --batch 2   # eff16 -> lr3e-4 (tuned)
#
# What it does automatically:
#   1. fetch_checkpoints.sh -> checkpoints/wedetect_tiny.pth from PUBLIC HF fushh7/WeDetect
#      (via hf-mirror; no token). Override to ModelScope with CKPT_MS_REPO=<owner/name> if needed.
#   2. fetch_dataset.sh      -> data/TCT_NGC_1024/ from ModelScope doilion/TCT_NGC_1024_TAR
#      (needs MODELSCOPE_API_TOKEN). Set TCT_NGC_1024_ROOT to use an existing local copy.
#   3. train -> eval (base organ-macro + per-class + exclude-neg; novel 9-class ZS) -> analyze.
#   4. everything lands in results/<stamp>/ (per-arm metrics JSON + analysis/ figures + verdict).
#      Hand back to the authors:  tar czf results_<stamp>.tgz results/<stamp>   (no weights inside).
#
# Required env on a fresh machine: MODELSCOPE_API_TOKEN (dataset). Optional: TCT_NGC_1024_ROOT,
# CKPT_MS_REPO, PYBIN (wedetect-env python). Everything else has working defaults.
# ----------------------------------------------------------------------------------------
# Two modules under test:
#   * Module 1 (region interaction): RegionAttributeFusion  -> arms attr_mean vs attr
#   * Module 2 (learned text fix):   de-coned relational distillation
#                                     -> arms baseline / decone / reldistill / decone_reldistill
# Arms run SEQUENTIALLY on one GPU group. fp32 throughout (BiomedCLIP @1024 nans under AMP).
# Finished runs are skipped. Each arm's results land in a weight-free bundle (see below).
#
# Knobs — CLI flags (take precedence) OR env vars:
#   --gpus 0,1        / GPUS        GPU ids (comma list). NPROC = number of ids.
#   --batch N         / BATCH       per-GPU train batch; "" = config value (@1024 default 8).
#   --configs "a b"   / CONFIGS     which arms to run (see CFG map).
#   --amp 0|1         / AMP         0 = fp32 (default). 1 = --amp (will likely nan @1024).
#   --eval 0|1        / EVAL        1 = base + novel eval after each train.
#   --port N          / PORT        DDP master port (DIFFERENT per concurrent run!).
#   --auto-lr 0|1     / AUTO_LR     1 = scale lr by eff batch (default). 0 = config lr.
#   --skip-fetch      / SKIP_FETCH  skip checkpoint + dataset auto-download.
#   --analyze 0|1     / ANALYZE     1 = auto-run analyze_suite.py after eval (figures + verdict).
#   --list            dry-run: print arm -> config -> work_dir -> eff-batch -> lr -> outputs, then exit.
#   BASE_LR=3e-4 BASE_BATCH=16      native lr at native eff batch; lr = BASE_LR*NPROC*BATCH/BASE_BATCH.
#   PYBIN / TCT_NGC_1024_ROOT       wedetect python / @1024 dataset root (base+novel).
#
# LR auto-scaling (linear rule = mmengine auto_scale_lr): tuned for lr 3e-4 @ eff batch 16
# (2 GPUs x 8). 4x8=eff32->6e-4; 2x4=eff8->1.5e-4. train.py has no --auto-scale-lr, so we scale here.
#
# 8x H800 guidance:
#   * Recommended (preserve the tuned recipe): keep eff-batch ~16-32 and use the cards for
#     THROUGHPUT — run 4 instances x 2 GPUs, each native eff16/lr3e-4, splitting the arms:
#       for g in 0,1 2,3 4,5 6,7; do p=296${g%%,*}0; \
#         CONFIGS=... GPUS=$g PORT=$p bash tools/run_method_suite.sh & done
#   * Or one arm on all 8 (fast per-arm): --gpus 0,1,2,3,4,5,6,7 --batch 2  (eff16, lr3e-4).
#     NB: very large eff-batch (e.g. 8x16=128) pushes lr to 2.4e-3 — AdamW linear scaling is
#     not guaranteed stable there; prefer eff-batch <= ~32 unless you add a longer warmup.
#
# Results bundle (weight-free, for "ran remote, only got results back"):
#   results/<stamp>/<arm>/{base_metrics.json, novel_metrics.json, train_curve.log, meta.txt}
#   results/<stamp>/summary.tsv     -> `tar czf results_<stamp>.tgz results/<stamp>` = all analysis needs.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."                                          # -> repo root

# ---- env defaults (overridden by CLI flags below) ----
GPUS="${GPUS:-0,1}"; BATCH="${BATCH:-}"; AUTO_LR="${AUTO_LR:-1}"
BASE_LR="${BASE_LR:-3e-4}"; BASE_BATCH="${BASE_BATCH:-16}"
PORT="${PORT:-29610}"; SEED="${SEED:-}"; AMP="${AMP:-0}"; EVAL="${EVAL:-1}"
SKIP_FETCH="${SKIP_FETCH:-0}"; ANALYZE="${ANALYZE:-1}"; LIST=0
PYBIN="${PYBIN:-$HOME/anaconda3/envs/wedetect/bin/python}"
DATA_ROOT_1024="${TCT_NGC_1024_ROOT:-$(pwd)/data/TCT_NGC_1024}"
NOVEL_ROOT="${NOVEL_ROOT:-$DATA_ROOT_1024}"
CONFIGS="${CONFIGS:-baseline attr_mean attr decone reldistill decone_reldistill}"

# ---- CLI flags take precedence over env ----
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --configs) CONFIGS="$2"; shift 2;;
    --amp) AMP="$2"; shift 2;;
    --eval) EVAL="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --auto-lr) AUTO_LR="$2"; shift 2;;
    --skip-fetch) SKIP_FETCH=1; shift;;
    --analyze) ANALYZE="$2"; shift 2;;
    --list) LIST=1; shift;;
    -h|--help) sed -n '2,46p' "$0"; exit 0;;
    *) echo "unknown arg: $1 (try --help)"; exit 2;;
  esac
done

NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
PRE="config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_"
declare -A CFG=(
  [baseline]="${PRE}res1024_biomedclip_2gpu.py"               # frozen text, no module
  [attr_mean]="${PRE}attr_mean_morph6_biomedclip_2gpu.py"     # attribute MEAN (control)
  [attr]="${PRE}attr_b4_morph6_biomedclip_2gpu.py"            # ★ region-adaptive (module 1)
  [decone]="${PRE}whiten_biomedclip_2gpu.py"                  # de-cone only (training-free)
  [reldistill]="${PRE}reldistill_res1024_biomedclip_2gpu.py"  # distill only (no whitening)
  [decone_reldistill]="${PRE}decone_reldistill_res1024_biomedclip_2gpu.py"  # ★ module 2
  # --- optional / combine / legacy arms (run by name) ---
  [stitch]="${PRE}stitch_res1024_biomedclip_2gpu.py"          # ★ combine A: late-fusion (visual_fuse)
  [stitchb]="${PRE}stitchb_res1024_biomedclip_2gpu.py"        # ★ combine B: shared per-attr text
  [attr_b5]="${PRE}attr_b5_morph6_biomedclip_2gpu.py"         # de-cone + region-adaptive (combo)
  [p2]="${PRE}p2_biomedclip_2gpu.py"
  [p2_nwd]="${PRE}p2_nwd_biomedclip_2gpu.py"
)
# Arms whose head classifies from per-attribute text (head.attr_fusion). Novel eval repoints
# that to the 9-class novel per-attr cache (--attr-text); backbone novel text is ignored there.
declare -A ATTR_ARM=( [attr_mean]=1 [attr]=1 [attr_b5]=1 [stitch]=1 [stitchb]=1 )

# Base eval on the held-out TEST split (test_base_clean), NOT the config-default val split.
# The configs' test_dataloader points at val_dev_disjoint (used for checkpoint selection);
# without this the base headline would be a VAL number while novel is on TEST -> inconsistent,
# and VAL runs ~5pp below TEST so @1024 wrongly looks worse than @640. Paper protocol = TEST.
BASE_ANN="${BASE_ANN:-annotations/instances_test_base_clean_dev30.json}"

# Novel zero-shot assets (9 classes, instances_test_novel_merged_9.json, cat 0-8).
NOVEL_ANN="annotations/instances_test_novel_merged_9.json"
NOVEL_TEXT_JSON="data/texts/tct_ngc_novel_merged_9.json"               # PSC prompts (backbone)
NOVEL_TEXT_EMB="data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth"  # PSC-keyed 9x512
NOVEL_ATTR_TEXT="data/texts/tct_ngc_morph6_novel9_per_attr_biomedclip.pth"  # [9,6,512]

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
SUITE_LOG="logs/suite_${STAMP}.log"
RESULTS_DIR="results/${STAMP}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUITE_LOG"; }

work_dir_of() {  # $1=cfg
  local wd; wd=$(grep -oP 'work_dir\s*=\s*"\K[^"]+' "$1" 2>/dev/null)
  [ -z "$wd" ] && wd="./work_dirs/$(basename "${1%.py}")"
  echo "$wd"
}

# resolve per-GPU batch (explicit BATCH else config value) + scaled lr -> echo "batch|eff|lr"
resolve_batch_lr() {  # $1=cfg
  local b="$BATCH"
  [ -z "$b" ] && b=$(PYTHONPATH=. "$PYBIN" -c \
    "from mmengine.config import Config;print(Config.fromfile('$1').train_dataloader.batch_size)" 2>/dev/null)
  local eff=$((NPROC * ${b:-0})) lr="(config)"
  [ "$AUTO_LR" = 1 ] && [ -n "$b" ] && \
    lr=$(awk -v base="$BASE_LR" -v e="$eff" -v bb="$BASE_BATCH" 'BEGIN{printf "%.6g", base*e/bb}')
  echo "${b:-?}|$eff|$lr"
}

list_arms() {
  echo "=== suite plan (dry-run) — GPUs=$GPUS (NPROC=$NPROC), AMP=$AMP, EVAL=$EVAL ==="
  printf "%-18s %-58s %-7s %-7s %-9s %s\n" arm config-exists batch eff lr work_dir
  for name in $CONFIGS; do
    local cfg="${CFG[$name]:-<unknown>}"
    local exists="MISSING"; [ -f "$cfg" ] && exists="ok"
    local bl; bl=$(resolve_batch_lr "$cfg" 2>/dev/null)
    local b="${bl%%|*}" rest="${bl#*|}"; local eff="${rest%%|*}" lr="${rest#*|}"
    printf "%-18s %-58s %-7s %-7s %-9s %s\n" "$name" "$exists:$(basename "$cfg")" \
           "$b" "$eff" "$lr" "$(work_dir_of "$cfg")"
  done
  echo "--- eval outputs per arm -> $RESULTS_DIR/<arm>/{base,novel}_metrics.json + train_curve.log + meta.txt"
  echo "--- attribute arms (need novel --attr-text): ${!ATTR_ARM[*]}"
}

write_bundle_meta() {  # $1=name $2=cfg $3=wd $4=batch $5=eff $6=lr $7=ckpt
  local bdir="$RESULTS_DIR/$1"; mkdir -p "$bdir"
  { echo "arm=$1"; echo "config=$2"; echo "work_dir=$3";
    echo "nproc=$NPROC gpus=$GPUS batch=$4 eff_batch=$5 lr=$6 amp=$AMP seed=${SEED:-cfg}";
    echo "checkpoint=$(basename "${7:-none}")";
    echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo n/a) host=$(hostname) date=$(date '+%F %T')";
  } > "$bdir/meta.txt"
  # weight-free training trace: loss points + RelDistill self-check (skip the param dumps)
  local tl="logs/$1_${STAMP}.train.log"
  [ -f "$tl" ] && grep -E 'Epoch\(train\)|RelDistill diag|nan|loss_rel_distill' "$tl" > "$bdir/train_curve.log" 2>/dev/null || true
}

train_eval_one() {
  local name="$1" cfg="${CFG[$name]:-}"
  if [ -z "$cfg" ]; then log "[skip] unknown config name: $name"; return; fi
  if [ ! -f "$cfg" ]; then log "[skip] $name: config not found ($cfg)"; return; fi
  local wd; wd=$(work_dir_of "$cfg")
  local bl; bl=$(resolve_batch_lr "$cfg"); local b="${bl%%|*}"; local eff="${bl#*|}"; eff="${eff%%|*}"; local lr="${bl##*|}"
  local bdir="$RESULTS_DIR/$name"; mkdir -p "$bdir"

  # ---- train (skip if a final checkpoint already exists) ----
  if ls "$wd"/best_*.pth "$wd"/epoch_12.pth >/dev/null 2>&1; then
    log "[done ] $name: final ckpt in $wd -> skip train"
  else
    log "[train] $name | $cfg | GPUs=$GPUS x$NPROC batch=$b eff=$eff lr=$lr amp=$AMP port=$PORT"
    local opts=(--launcher pytorch --resume auto); [ "$AMP" = 1 ] && opts+=(--amp)
    [ -n "$SEED" ] && opts+=(--seed "$SEED")
    local cfgo=()
    [ -n "$BATCH" ] && cfgo+=("train_dataloader.batch_size=$BATCH")
    [ "$AUTO_LR" = 1 ] && [ "$lr" != "(config)" ] && cfgo+=("optim_wrapper.optimizer.lr=$lr")
    [ ${#cfgo[@]} -gt 0 ] && opts+=(--cfg-options "${cfgo[@]}")
    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. "$PYBIN" -m torch.distributed.run \
      --nproc_per_node="$NPROC" --master_port="$PORT" \
      train.py "$cfg" "${opts[@]}" 2>&1 | tee "logs/${name}_${STAMP}.train.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      log "[FAIL ] $name training exited non-zero -> continue to next config"
      write_bundle_meta "$name" "$cfg" "$wd" "$b" "$eff" "$lr" ""; return
    fi
  fi

  # ---- eval (base organ-macro + per-class + exclude; novel 9-class ZS). Explicit ckpt;
  # PIPESTATUS check (the `| tee` would otherwise mask an OOM/crash as success). ----
  local ckpt=""
  if [ "$EVAL" = 1 ]; then
    local g="${GPUS%%,*}"   # single GPU for eval
    ckpt=$(ls -t "$wd"/best_*.pth "$wd"/epoch_12.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then log "[eval-skip] $name: no checkpoint in $wd";
      write_bundle_meta "$name" "$cfg" "$wd" "$b" "$eff" "$lr" ""; return; fi

    log "[eval ] $name base (TEST split, organ-macro + per-class + exclude-neg) @ $(basename "$ckpt")"
    CUDA_VISIBLE_DEVICES="$g" PYTHONPATH=. "$PYBIN" test_exclude_negative.py \
      --config "$cfg" --checkpoint "$ckpt" --metric organ \
      --data-root "$DATA_ROOT_1024" --ann-file "$BASE_ANN" \
      --metrics-out "$bdir/base_metrics.json" 2>&1 | tee "logs/${name}_${STAMP}.eval_base.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && log "[FAIL ] $name base eval exited non-zero"

    local attr_opt=()
    [ -n "${ATTR_ARM[$name]:-}" ] && attr_opt=(--attr-text "$NOVEL_ATTR_TEXT")
    log "[eval ] $name novel (9-class ZS, root $NOVEL_ROOT)${attr_opt:+ [+per-attr novel text]}"
    CUDA_VISIBLE_DEVICES="$g" PYTHONPATH=. "$PYBIN" tools/eval_novel_split.py \
      --config "$cfg" --checkpoint "$ckpt" --data-root "$NOVEL_ROOT" \
      --ann-file "$NOVEL_ANN" --text-json "$NOVEL_TEXT_JSON" --text-emb "$NOVEL_TEXT_EMB" \
      --work-dir "$wd/novel_eval" --metrics-out "$bdir/novel_metrics.json" "${attr_opt[@]}" 2>&1 \
      | tee "logs/${name}_${STAMP}.eval_novel.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && log "[FAIL ] $name novel eval exited non-zero"
  fi
  write_bundle_meta "$name" "$cfg" "$wd" "$b" "$eff" "$lr" "$ckpt"
  log "[done ] $name complete -> $bdir"
}

# ---- metric scrapers: prefer the JSON bundle, fall back to log grep ----
_jget() { PYTHONPATH=. "$PYBIN" -c "import json,sys; d=json.load(open('$1')); \
  k=[x for x in d if x.endswith('$2')]; print(d[k[-1]] if k else '')" 2>/dev/null; }
base_mAP() {  # organ-macro headline
  local j="$RESULTS_DIR/$1/base_metrics.json"
  [ -f "$j" ] && { _jget "$j" "overall/macro_mAP"; return; }
  grep -oE 'coco/overall/macro_mAP: [0-9.]+' "logs/$1_${STAMP}.eval_base.log" 2>/dev/null | tail -1 | awk '{print $NF}'
}
novel_mAP() {
  local j="$RESULTS_DIR/$1/novel_metrics.json"
  [ -f "$j" ] && { _jget "$j" "bbox_mAP"; return; }
  grep -oE 'coco/bbox_mAP: [0-9.]+' "logs/$1_${STAMP}.eval_novel.log" 2>/dev/null | tail -1 | awk '{print $NF}'
}

write_summary() {
  mkdir -p "$RESULTS_DIR"
  local tsv="$RESULTS_DIR/summary.tsv"
  { printf 'arm\tbase_macro_mAP\tnovel_mAP\n'
    for name in $CONFIGS; do printf '%s\t%s\t%s\n' "$name" "$(base_mAP "$name")" "$(novel_mAP "$name")"; done
  } | tee "$tsv"
  cp "$tsv" "logs/suite_${STAMP}.summary.tsv" 2>/dev/null || true
  log "--- decisive deltas ---"
  local a am; a=$(base_mAP attr); am=$(base_mAP attr_mean)
  [ -n "$a" ] && [ -n "$am" ] && \
    log "  region-adaptive (attr - attr_mean), base: $(awk -v x="$a" -v y="$am" 'BEGIN{printf "%+.4f", x-y}')"
  local nb nd nr ndr; nb=$(novel_mAP baseline); nd=$(novel_mAP decone)
  nr=$(novel_mAP reldistill); ndr=$(novel_mAP decone_reldistill)
  [ -n "$ndr" ] && [ -n "$nb" ] && \
    log "  text-fix (decone_reldistill - baseline), novel: $(awk -v x="$ndr" -v y="$nb" 'BEGIN{printf "%+.4f", x-y}')"
  [ -n "$nd" ] && [ -n "$nb" ] && log "    decone-only - baseline, novel: $(awk -v x="$nd" -v y="$nb" 'BEGIN{printf "%+.4f", x-y}')"
  [ -n "$nr" ] && [ -n "$nb" ] && log "    distill-only - baseline, novel: $(awk -v x="$nr" -v y="$nb" 'BEGIN{printf "%+.4f", x-y}')"
  log "summary -> $tsv  | bundle -> $RESULTS_DIR/  (tar czf results_${STAMP}.tgz $RESULTS_DIR)"
}

# ---- dry-run preview + exit ----
if [ "$LIST" = 1 ]; then list_arms; exit 0; fi

# ---- auto-fetch dataset + pretrained checkpoint (idempotent; skip if present) ----
if [ "$SKIP_FETCH" != 1 ]; then
  log "=== ensuring pretrained checkpoint (fetch_checkpoints.sh) ==="
  PYBIN="$PYBIN" bash tools/fetch_checkpoints.sh 2>&1 | tee -a "$SUITE_LOG" \
    || log "[WARN] fetch_checkpoints failed; ensure checkpoints/wedetect_tiny.pth exists"
  log "=== ensuring dataset at $DATA_ROOT_1024 (fetch_dataset.sh) ==="
  TCT_NGC_1024_ROOT="$DATA_ROOT_1024" PYBIN="$PYBIN" bash tools/fetch_dataset.sh 2>&1 | tee -a "$SUITE_LOG" \
    || log "[WARN] fetch_dataset failed; assuming data is present at $DATA_ROOT_1024"
fi

log "=== suite start: [$CONFIGS] on GPUs $GPUS (${NPROC} proc), eval=$EVAL amp=$AMP -> $RESULTS_DIR ==="
for name in $CONFIGS; do train_eval_one "$name"; done
[ "$EVAL" = 1 ] && write_summary

# ---- auto-analysis: figures + analysis.md straight into the result bundle (one-click) ----
if [ "$EVAL" = 1 ] && [ "$ANALYZE" = 1 ]; then
  log "=== auto-analysis (analyze_suite.py -> figures + analysis.md) ==="
  PYTHONPATH=. "$PYBIN" tools/analyze_suite.py --results-dir "$RESULTS_DIR" \
    --out-dir "$RESULTS_DIR/analysis" 2>&1 | tee -a "$SUITE_LOG" \
    || log "[WARN] analyze_suite failed; raw metrics still in $RESULTS_DIR/<arm>/*.json"
fi
log "=== suite done. bundle: $RESULTS_DIR/ (metrics + analysis/) | logs: logs/<name>_${STAMP}.* | $SUITE_LOG ==="
log "    -> hand back: tar czf results_${STAMP}.tgz $RESULTS_DIR"
