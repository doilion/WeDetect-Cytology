# Running WeDetect TCT_NGC experiments (collaborator runbook)

This repo holds the code; the dataset and the init checkpoint live in a **private
HuggingFace dataset repo** (access granted to you by the owner). Below is the full
chain to reproduce the native (full-resolution) TCT_NGC experiment line.

## 1. Code

Already done if you can read this — you cloned the repo. The small derived text
caches (`data/texts/*.pth/.pt`) and `xlm-roberta-base/` config+tokenizer travel
with git, so you do **not** need to rebuild them.

## 2. Environment

```bash
conda create -n wedetect python=3.10 -y && conda activate wedetect
# Core (versions that this repo is pinned to):
#   pytorch==2.5.1+cu124, transformers==4.57.1
#   mmcv==2.1.0, mmdet==3.3.0, mmengine==0.10.7
# BiomedCLIP configs (*_biomedclip*) additionally need: open_clip_torch>=2.23.0
PYTHONPATH=. python -c "import wedetect; print('import ok')"
```

## 3. Get the data + init checkpoint from HuggingFace

**Easy path — one command** (after `hf auth login`):
```bash
bash setup_dataset.sh <owner>/tct-ngc-native /abs/path/to/TCT_NGC
```
It downloads, extracts the TCT_CCD tar, verifies the file count, and places the
checkpoint. The rest of this section is the manual equivalent.

```bash
pip install -U huggingface_hub        # 1.x; Xet fast-transfer is built in, no env var needed
# NOTE: a PRIVATE repo cannot be served by hf-mirror.com — pin the real hub for
# both login and download (override any HF_ENDPOINT=hf-mirror.com in your shell):
HF_ENDPOINT=https://huggingface.co hf auth login          # token the owner shared / your own

# Pull the whole private dataset repo (images + annotations + weights/ + one tar):
HF_ENDPOINT=https://huggingface.co hf download <owner>/tct-ngc-native --repo-type dataset \
    --local-dir /SOME/LOCAL/PATH/TCT_NGC

# TCT_CCD's ~35k images live in ONE directory, over HF's 10,000-files-per-dir
# limit, so the overflow ships as a single tar instead of loose files. Extract it
# in place to recreate images/TCT_CCD/images/train30000/* (local FS has no limit):
[ -f /SOME/LOCAL/PATH/TCT_NGC/TCT_CCD_remaining.tar ] && \
    tar xf /SOME/LOCAL/PATH/TCT_NGC/TCT_CCD_remaining.tar -C /SOME/LOCAL/PATH/TCT_NGC/

# The init weight arrives at  /SOME/LOCAL/PATH/TCT_NGC/weights/wedetect_tiny.pth
mkdir -p checkpoints
cp /SOME/LOCAL/PATH/TCT_NGC/weights/wedetect_tiny.pth checkpoints/wedetect_tiny.pth
```

## 4. Point configs at your data

Configs resolve data roots from environment variables first, then fall back to
repo-local `data/TCT_NGC*` directories. Set the variables in your shell:

```bash
export TCT_NGC_DATA_ROOT=/SOME/LOCAL/PATH/TCT_NGC
# Optional caches, if available:
export TCT_NGC_640_ROOT=/SOME/LOCAL/PATH/TCT_NGC_640
export TCT_NGC_1024_ROOT=/SOME/LOCAL/PATH/TCT_NGC_1024
```

## 5. Train / evaluate

```bash
# Train (2-GPU native config; adjust GPU count):
bash dist_train.sh config/wedetect_tiny_tct_ngc_dev30_2gpu.py 2

# Filtered dev eval used for reporting:
PYTHONPATH=. python tools/evaluate_ngc_filtered.py \
    --config config/wedetect_tiny_tct_ngc_dev30_2gpu.py \
    --checkpoint work_dirs/.../best_*.pth \
    --min-annotations 100 --exclude-keywords negative normal nilm impurity
```

Notes:
- Configs starting `wedetect_tiny_tct_ngc_dev30_*` / `wedetect_tiny_tct_ngc_dev_*`
  use the native dataset (this repo). `*_cache640_*` configs need the separate
  13 GB 640-cache, which is **not** shipped here.
- `TCT_CCD` under `images/` is a placeholder layout — exclude it from any
  per-case generalization claims.
- The data is patient cytology. Keep it on storage covered by the owner's
  ethics/consent terms; do not redistribute or push to a public repo.
