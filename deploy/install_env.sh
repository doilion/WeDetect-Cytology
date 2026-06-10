#!/usr/bin/env bash
# One-shot conda environment install for WeDetect (TCT_NGC cytology OVD), fresh machine.
# Recreates the exact working env: python 3.10 / torch 2.1.0+cu118 / mmcv 2.1.0 /
# mmdet 3.3.0 / mmengine 0.10.7. Requires a CUDA 11.8-capable GPU + conda.
#
# Usage:  bash deploy/install_env.sh [ENV_NAME]   (default ENV_NAME=wedetect)
set -euo pipefail

ENV_NAME="${1:-wedetect}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

echo "[1/4] create conda env '$ENV_NAME' (python 3.10)"
conda create -y -n "$ENV_NAME" python=3.10

# Use the env's pip directly (no need to `conda activate` inside a script).
PY="$(conda run -n "$ENV_NAME" which python)"
PIP="$PY -m pip"

echo "[2/4] install torch 2.1.0 + cu118 (matching CUDA 11.8)"
$PIP install --no-input \
  torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

echo "[3/4] install mmengine/mmcv/mmdet + the rest from requirements.txt"
# mmcv is pinned to the prebuilt cu118/torch2.1 wheel URL inside requirements.txt,
# so a plain -r install resolves it without compiling.
$PIP install --no-input -r "$HERE/requirements.txt"

echo "[4/4] verify import"
PYTHONPATH="$REPO_ROOT" $PY -c "import torch, mmcv, mmdet, mmengine, wedetect; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmengine', mmengine.__version__); \
print('wedetect import OK')"

echo "DONE. Activate with:  conda activate $ENV_NAME"
