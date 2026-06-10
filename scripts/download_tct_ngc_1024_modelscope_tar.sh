#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-doilion/TCT_NGC_1024_TAR}"
TARGET_DIR="${1:-${TARGET_DIR:-TCT_NGC_1024}}"
MAX_WORKERS="${MAX_WORKERS:-8}"

usage() {
  cat <<'EOF'
Download and restore the TCT_NGC_1024 ModelScope tar dataset.

Usage:
  bash scripts/download_tct_ngc_1024_modelscope_tar.sh [TARGET_DIR]

Environment:
  REPO_ID       ModelScope dataset repo id. Default: doilion/TCT_NGC_1024_TAR
  TARGET_DIR   Output dataset directory. Default: TCT_NGC_1024
  MAX_WORKERS  ModelScope download workers. Default: 8

For a private dataset, log in first:
  ms login

Or provide a temporary token through the environment:
  export MODELSCOPE_API_TOKEN='...'
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v ms >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: ModelScope CLI "ms" was not found.
Install/activate an environment that provides the ModelScope CLI first.
EOF
  exit 127
fi

mkdir -p "$TARGET_DIR"

echo "==== Downloading ${REPO_ID} -> ${TARGET_DIR} ===="
ms download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$TARGET_DIR" \
  --max-workers "$MAX_WORKERS"

cd "$TARGET_DIR"

required_files=(
  "annotations/instances_train_dev_disjoint_dev30.json"
  "annotations/instances_val_dev_disjoint_dev30.json"
  "annotations/instances_test_base_clean_dev30.json"
  "annotations/instances_test_novel_merged_9.json"
  "archives/images_Serous_effusion.tar"
  "archives/images_TCT_CCD.tar"
  "archives/images_Thyroid_gland.tar"
  "archives/images_Urine.tar"
  "archives/images_respiratory_tract.tar"
  "SHA256SUMS"
)

for path in "${required_files[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "error: required file is missing after download: $path" >&2
    exit 1
  fi
done

echo "==== Verifying archives ===="
sha256sum -c SHA256SUMS

echo "==== Extracting archives ===="
mkdir -p images
tar -xf archives/images_Serous_effusion.tar -C images
tar -xf archives/images_TCT_CCD.tar -C images
tar -xf archives/images_Thyroid_gland.tar -C images
tar -xf archives/images_Urine.tar -C images
tar -xf archives/images_respiratory_tract.tar -C images

echo "==== Verifying restored layout ===="
for dir in Serous_effusion TCT_CCD Thyroid_gland Urine respiratory_tract; do
  if [[ ! -d "images/$dir" ]]; then
    echo "error: extracted image directory is missing: images/$dir" >&2
    exit 1
  fi
done

echo "TCT_NGC_1024 is ready at: $(pwd)"
