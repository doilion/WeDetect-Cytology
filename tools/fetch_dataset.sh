#!/usr/bin/env bash
# Idempotent one-click loader for the TCT_NGC_1024 dataset.
#
# It does NOT reinvent the download: it delegates to the canonical bootstrap that
# ships with the (private) ModelScope dataset doilion/TCT_NGC_1024_TAR ->
#   scripts/download_tct_ngc_1024_modelscope_tar.sh  (download + sha256 + extract + verify)
# This wrapper only adds what a one-click suite needs: idempotent skip, `ms` CLI
# resolution, and private-repo auth via a token.
#
# Usage (collaborator):
#   export MODELSCOPE_API_TOKEN=<temp token>     # private repo; or run `ms login` first
#   bash tools/fetch_dataset.sh                  # -> ./data/TCT_NGC_1024
#   TCT_NGC_1024_ROOT=/data/TCT_NGC_1024 bash tools/fetch_dataset.sh
set -euo pipefail
cd "$(dirname "$0")/.."                                  # repo root (scripts/ + data/ resolve)

ROOT="${TCT_NGC_1024_ROOT:-data/TCT_NGC_1024}"
PYBIN="${PYBIN:-$HOME/anaconda3/envs/wedetect/bin/python}"
REPO_ID="${TCT_NGC_1024_MS_REPO:-doilion/TCT_NGC_1024_TAR}"
# Private dataset: pass a token via MODELSCOPE_API_TOKEN (preferred) or MS_TOKEN.
TOKEN="${MODELSCOPE_API_TOKEN:-${MS_TOKEN:-}}"
BOOTSTRAP="scripts/download_tct_ngc_1024_modelscope_tar.sh"
ORGANS=(Serous_effusion TCT_CCD Thyroid_gland Urine respiratory_tract)

log() { printf '[fetch_dataset] %s\n' "$*"; }

# ---- idempotent skip --------------------------------------------------------
have=1
for o in "${ORGANS[@]}"; do [ -d "$ROOT/images/$o" ] || have=0; done
if [ "$have" = 1 ] && [ -f "$ROOT/annotations/instances_test_novel_merged_9.json" ]; then
  log "dataset already present at $ROOT (images/ + annotations/) -> skip"
  exit 0
fi
log "dataset missing/incomplete at $ROOT -> fetching $REPO_ID"

[ -f "$BOOTSTRAP" ] || { log "ERROR: $BOOTSTRAP not found (needed for download)"; exit 1; }

# ---- ensure the `ms` CLI is on PATH (the bootstrap calls bare `ms`) ----------
MSDIR="$(dirname "$PYBIN")"
if ! command -v ms >/dev/null 2>&1; then
  if [ -x "$MSDIR/ms" ]; then
    export PATH="$MSDIR:$PATH"
  else
    log "ms CLI not found -> $PYBIN -m pip install modelscope"
    "$PYBIN" -m pip install -q -U modelscope 2>/dev/null || pip install -q -U modelscope || {
      log "ERROR: could not install modelscope. Install it, or point TCT_NGC_1024_ROOT"
      log "       at a host that already has the dataset."
      exit 1
    }
    [ -x "$MSDIR/ms" ] && export PATH="$MSDIR:$PATH"
  fi
fi
command -v ms >/dev/null 2>&1 || { log "ERROR: ms CLI still not on PATH after install"; exit 1; }

# ---- private-repo auth ------------------------------------------------------
if [ -n "$TOKEN" ]; then
  log "ms login with provided token"
  ms login --token "$TOKEN" >/dev/null 2>&1 || log "WARN: ms login failed; relying on MODELSCOPE_API_TOKEN env"
  export MODELSCOPE_API_TOKEN="$TOKEN"   # the SDK also reads this directly
else
  log "no token in MODELSCOPE_API_TOKEN/MS_TOKEN; assuming an existing 'ms login' session"
fi

# ---- delegate download + checksum + extract + verify ------------------------
log "delegating to $BOOTSTRAP -> $ROOT"
REPO_ID="$REPO_ID" bash "$BOOTSTRAP" "$ROOT"

# ---- final layout assertion -------------------------------------------------
for o in "${ORGANS[@]}"; do
  [ -d "$ROOT/images/$o" ] || { log "ERROR: missing images/$o after fetch"; exit 1; }
done
[ -f "$ROOT/annotations/instances_test_novel_merged_9.json" ] || {
  log "ERROR: novel ann missing after fetch"; exit 1; }
log "done -> $ROOT (images/ + annotations/ ready)"
