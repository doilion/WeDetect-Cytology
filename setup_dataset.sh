#!/usr/bin/env bash
# One-shot dataset setup for a collaborator running WeDetect TCT_NGC experiments.
#
# Downloads the private HF dataset, reconstructs the TCT_CCD images from their tar
# (TCT_CCD's ~35k images exceed HF's 10k-files-per-directory limit, so the overflow
# ships as one tar), verifies the file count, and places the init checkpoint.
# After this, the local tree is byte-identical to the original TCT_NGC and every
# annotation path resolves — just point the configs at it.
#
# Prereq: `hf auth login` with access to the private repo.
# Usage:
#   bash setup_dataset.sh <owner>/tct-ngc-native /abs/path/to/TCT_NGC
set -euo pipefail
REPO="${1:?usage: setup_dataset.sh <owner>/tct-ngc-native /abs/dest/TCT_NGC}"
DEST="${2:?usage: setup_dataset.sh <owner>/tct-ngc-native /abs/dest/TCT_NGC}"

# Private repo MUST come from the real hub, not a download mirror (hf-mirror.com).
export HF_ENDPOINT="https://huggingface.co"

echo ">> [1/4] downloading $REPO -> $DEST  (resumable: rerun this script if it drops)"
pip install -U "huggingface_hub[hf_xet]" >/dev/null 2>&1 || true
hf download "$REPO" --repo-type dataset --local-dir "$DEST"

echo ">> [2/4] reconstructing TCT_CCD from its tar (if present)"
if [ -f "$DEST/TCT_CCD_remaining.tar" ]; then
    tar xf "$DEST/TCT_CCD_remaining.tar" -C "$DEST"
    echo "   extracted (you may delete $DEST/TCT_CCD_remaining.tar afterwards to save space)"
else
    echo "   no tar found (already extracted, or none needed)"
fi

echo ">> [3/4] verifying image count"
n=$(find "$DEST/images" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
echo "   images on disk: $n  (expected 171333)"

echo ">> [4/4] placing init checkpoint"
if [ -f "$DEST/weights/wedetect_tiny.pth" ]; then
    mkdir -p checkpoints
    cp -n "$DEST/weights/wedetect_tiny.pth" checkpoints/wedetect_tiny.pth
    echo "   checkpoints/wedetect_tiny.pth ready"
fi

cat <<EOF

DONE. Final step — make the configs find the data:
  export TCT_NGC_DATA_ROOT="$DEST"
  # optional caches, if you build/download them:
  # export TCT_NGC_640_ROOT="/path/to/TCT_NGC_640"
  # export TCT_NGC_1024_ROOT="/path/to/TCT_NGC_1024"

Then train, e.g.:
  bash dist_train.sh config/wedetect_tiny_tct_ngc_dev30_2gpu.py 2
EOF
