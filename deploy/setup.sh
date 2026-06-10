#!/usr/bin/env bash
# One-click WeDetect (TCT_NGC cytology OVD) setup for a FRESH machine + PRIVATE HF data.
# Installs the conda env, downloads base weights + the private dataset, writes data-root env hints.
#
# Prereqs: conda, a CUDA 11.8 GPU, and HF access to the private dataset
#   (ask the owner Doilion111 to grant your HF account, then `huggingface-cli login`).
#
# Usage:
#   ENV_NAME=wedetect DATA_DIR=/abs/path/for/data  bash deploy/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ENV_NAME="${ENV_NAME:-wedetect}"
DATA_DIR="${DATA_DIR:-$REPO/data}"

echo "=== WeDetect setup  | env=$ENV_NAME  data=$DATA_DIR ==="

echo "[1/5] install conda env"
bash "$HERE/install_env.sh" "$ENV_NAME"
PY="$(conda run -n "$ENV_NAME" which python)"

echo "[2/5] HF auth check (private dataset Doilion111/tct-ngc-native)"
if [ -z "${HF_TOKEN:-}" ] && ! conda run -n "$ENV_NAME" huggingface-cli whoami >/dev/null 2>&1; then
  echo "  !! Not authenticated. Do BOTH then re-run:"
  echo "     - ask the owner (Doilion111) to grant your HF account access to the dataset"
  echo "     - conda run -n $ENV_NAME huggingface-cli login   (or: export HF_TOKEN=hf_xxx)"
  exit 1
fi

echo "[3/5] download base weights + private dataset -> $DATA_DIR"
PYTHONPATH="$REPO" "$PY" "$HERE/download_hf.py" --what all --data-dir "$DATA_DIR"

echo "[4/5] write data-root env hints"
PYTHONPATH="$REPO" "$PY" "$HERE/patch_data_root.py" --data-dir "$DATA_DIR"

echo "[5/5] smoke: import + build a config"
cd "$REPO"
PYTHONPATH="$REPO" "$PY" -c "import wedetect; from mmengine.config import Config; \
Config.fromfile('config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py'); \
print('config build OK')"

cat <<EOF

=== SETUP DONE ===
Most TCT configs train on a letterboxed CACHE, not the native images. Build the cache
your config needs ONCE (native -> 640 or 1024 letterbox; takes ~30-90 min):

  conda activate $ENV_NAME
  source deploy/data_roots.env
  TCT_NGC_DATA_ROOT=$DATA_DIR/TCT_NGC python tools/cache_tct_ngc_640.py \\
    --size 640  --out-root $DATA_DIR/TCT_NGC_640  \\
    --splits train_dev_disjoint_dev30 val_dev_disjoint_dev30 --workers 12
  # (use --size 1024 --out-root $DATA_DIR/TCT_NGC_1024 for the 1024 line)

Then run an experiment, e.g. the BiomedCLIP M1 baseline on 2 GPUs:

  PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=2 --master_port=29500 \\
    train.py config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py \\
    --launcher pytorch --amp

See deploy/README.md for eval + the resolution/method configs.
EOF
