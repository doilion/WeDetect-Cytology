#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="$1"
GPU_ID="$2"
CONFIG_FILE="$3"
WORK_DIR="$4"
TEXT_FILE="$5"

LOG_DIR="work_dirs/logs"
EVAL_DIR="work_dirs/eval"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_run.log"

mkdir -p "${LOG_DIR}" "${EVAL_DIR}"

CONDA_HOOK="${CONDA_HOOK:-$HOME/anaconda3/etc/profile.d/conda.sh}"
source "${CONDA_HOOK}"
conda activate wedetect
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python train.py "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python test_exclude_negative.py \
    --config "${CONFIG_FILE}" \
    --checkpoint "${WORK_DIR}/best_coco_bbox_mAP_epoch_*.pth" \
    --work-dir "${EVAL_DIR}/test_exclude_negative_${EXP_NAME}" \
    2>&1 | tee -a "${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python eval_novel.py \
    --config "${CONFIG_FILE}" \
    --checkpoint "${WORK_DIR}/best_coco_bbox_mAP_epoch_*.pth" \
    --text "${TEXT_FILE}" \
    --out-dir "${EVAL_DIR}/test_novel_${EXP_NAME}" \
    --device cuda:0 \
    2>&1 | tee -a "${LOG_FILE}"
