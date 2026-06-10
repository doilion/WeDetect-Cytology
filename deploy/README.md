# WeDetect (TCT_NGC cytology OVD) — collaborator quick-start

One-click setup to run experiments on a **fresh machine** with the **private** dataset.

## 0. Prereqs
- `conda`, a **CUDA 11.8**-capable NVIDIA GPU (24 GB recommended, e.g. RTX 3090).
- An HF account that the owner (**Doilion111**) has **granted access** to the private
  dataset `Doilion111/tct-ngc-native`.

## 1. Clone + one-click setup
```bash
git clone <this repo>  &&  cd WeDetect

# authenticate to HF (needed for the private dataset)
huggingface-cli login          # or:  export HF_TOKEN=hf_xxx

# install env + download weights/data + write data-root env hints
ENV_NAME=wedetect DATA_DIR=$PWD/data  bash deploy/setup.sh
```
`setup.sh` does: install the conda env (`deploy/install_env.sh`) → download base weights
+ the private native dataset (`deploy/download_hf.py`) → write `deploy/data_roots.env`
with `TCT_NGC_*` paths → smoke-test.

## 2. Build the cache your config needs (once)
Most configs train on a **letterboxed cache**, not native images:
```bash
conda activate wedetect
source deploy/data_roots.env
TCT_NGC_DATA_ROOT=$PWD/data/TCT_NGC python tools/cache_tct_ngc_640.py \
  --size 640 --out-root $PWD/data/TCT_NGC_640 \
  --splits train_dev_disjoint_dev30 val_dev_disjoint_dev30 --workers 12
# use --size 1024 --out-root $PWD/data/TCT_NGC_1024 for the 1024 experiment line
```

## 3. Run an experiment
```bash
# train (2 GPUs). NOTE: use `python -m torch.distributed.run`, NOT bare torchrun.
PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=2 --master_port=29500 \
  train.py config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py \
  --launcher pytorch --amp

# eval (base-class, organ-restricted, the paper metric)
PYTHONPATH=. python tools/eval_organ_restricted.py \
  --config config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py \
  --checkpoint work_dirs/.../best_*.pth
```

## Config map (the lines that matter)
| config | what |
|---|---|
| `..._ochmta_m1_biomedclip_2gpu.py` | **baseline** (M1, frozen BiomedCLIP text, 640) |
| `..._ochmta_m1_xlmr_2gpu.py` | baseline with XLM-R 768d text |
| `..._m1_res1024_biomedclip_2gpu.py` | baseline @1024 cache |
| `..._m1_nwd_biomedclip_2gpu.py` | NWD tiny-object assignment (the "foil") |
| `..._m1_p2_biomedclip_2gpu.py` / `..._xlmr_p2_res1024_2gpu.py` | **stride-4 small-cell method** |

## Notes / gotchas
- Multi-GPU MUST use `python -m torch.distributed.run` (bare `torchrun` hits a wrong python).
- The cached text embeddings + organ mask ship in the repo (`data/texts/`, git-tracked) —
  nothing to download for those.
- `deploy/download_hf.py` / `deploy/setup.sh` carry the HF repo ids; edit the placeholders
  in `download_hf.py` (`HF_TRAINED_REPO`, `HF_CACHE_REPO`) if you publish those too.
- Run `source deploy/data_roots.env` in new shells before training/eval.
