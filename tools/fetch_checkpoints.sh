#!/usr/bin/env bash
# Auto-fetch the pretrained START checkpoint(s) so a collaborator needs ZERO manual scp.
# Every arm's config has `load_from = checkpoints/wedetect_tiny.pth`; this pulls it (and any
# other listed ckpt) into checkpoints/. Idempotent: skips files already present.
#
# Default source: PUBLIC HuggingFace `fushh7/WeDetect` (the canonical WeDetect weights, per
# CLAUDE.md). HF_ENDPOINT is forced to the hf-mirror so it works on any machine without a
# token / without direct huggingface.co access.
# Override to ModelScope (same infra + token as the dataset): set CKPT_MS_REPO=<owner/name>.
#
# Usage:
#   bash tools/fetch_checkpoints.sh                                  # -> checkpoints/wedetect_tiny.pth from HF
#   CKPT_MS_REPO=doilion/WeDetect-ckpts MODELSCOPE_API_TOKEN=... bash tools/fetch_checkpoints.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYBIN="${PYBIN:-$HOME/anaconda3/envs/wedetect/bin/python}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
CKPT_FILES="${CKPT_FILES:-wedetect_tiny.pth}"          # space-separated list
HF_REPO="${CKPT_HF_REPO:-fushh7/WeDetect}"
MS_REPO="${CKPT_MS_REPO:-}"                            # if set, prefer ModelScope
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # public mirror; works tokenless
log() { printf '[fetch_ckpt] %s\n' "$*"; }

mkdir -p "$CKPT_DIR"
need=()
for f in $CKPT_FILES; do [ -s "$CKPT_DIR/$f" ] || need+=("$f"); done
if [ ${#need[@]} -eq 0 ]; then log "all checkpoints present in $CKPT_DIR -> skip"; exit 0; fi
log "missing: ${need[*]} -> downloading (HF_ENDPOINT=$HF_ENDPOINT)"

if [ -n "$MS_REPO" ]; then
  # ---- ModelScope source (same token as the dataset) ----
  MS="$(dirname "$PYBIN")/modelscope"; [ -x "$MS" ] || MS="$(command -v modelscope || command -v ms || true)"
  [ -n "$MS" ] || { log "ERROR: modelscope CLI not found for CKPT_MS_REPO=$MS_REPO"; exit 1; }
  TOKEN="${MODELSCOPE_API_TOKEN:-${MS_TOKEN:-}}"
  [ -n "$TOKEN" ] && { "$MS" login --token "$TOKEN" >/dev/null 2>&1 || true; export MODELSCOPE_API_TOKEN="$TOKEN"; }
  for f in "${need[@]}"; do
    log "modelscope download $MS_REPO :: $f -> $CKPT_DIR"
    # current CLI: positional repo + --repo-type model + --local-dir; legacy: --model + --local_dir
    if "$MS" download --help 2>&1 | grep -q -- '--model'; then
      "$MS" download --model "$MS_REPO" "$f" --local_dir "$CKPT_DIR"
    else
      "$MS" download --repo-type model "$MS_REPO" "$f" --local-dir "$CKPT_DIR"
    fi
  done
else
  # ---- default: public HuggingFace via mirror (no token) ----
  for f in "${need[@]}"; do
    log "hf_hub_download $HF_REPO :: $f -> $CKPT_DIR"
    "$PYBIN" - "$HF_REPO" "$f" "$CKPT_DIR" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download
repo, fname, dst = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(repo_id=repo, filename=fname)   # honors HF_ENDPOINT mirror, public repo
os.makedirs(dst, exist_ok=True)
tgt = os.path.join(dst, fname)
if os.path.abspath(p) != os.path.abspath(tgt):
    shutil.copy(p, tgt)
print("downloaded ->", tgt)
PY
  done
fi

# ---- final assertion ----
for f in $CKPT_FILES; do
  [ -s "$CKPT_DIR/$f" ] || { log "ERROR: $CKPT_DIR/$f still missing after fetch"; exit 1; }
done
log "done -> $CKPT_DIR ($CKPT_FILES)"
