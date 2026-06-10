#!/usr/bin/env python
"""One-click asset download from HuggingFace for WeDetect (TCT_NGC cytology OVD).

Downloads: (1) the base WeDetect-tiny backbone weights (load_from / init), (2) the
TCT_NGC dataset or its letterboxed cache, (3) an optional trained TCT checkpoint for
eval. The cached text embeddings + organ mask + taxonomy already ship inside the repo
(data/texts/, git-tracked) -- nothing to download for those.

>>> FILL IN the HF repo ids below (the dataset upload may still be in progress). <<<

Usage:
    python deploy/download_hf.py --what all      # weights + data + (optional) ckpt
    python deploy/download_hf.py --what weights  # just the base + trained weights
    python deploy/download_hf.py --what data --data-dir /path/to/put/TCT_NGC
"""
import argparse
import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# >>> EDIT THESE once your HuggingFace uploads finish <<<
HF_BASE_WEIGHTS_REPO = "fushh7/WeDetect"          # public original WeDetect (model repo)
HF_BASE_WEIGHTS_FILE = "wedetect_tiny.pth"        # -> checkpoints/wedetect_tiny.pth

HF_DATA_REPO = "Doilion111/tct-ngc-native"         # native full-res TCT_NGC (dataset)
HF_DATA_REPO_TYPE = "dataset"                      # "dataset" (recommended) or "model"
# This dataset is PRIVATE. The collaborator must (1) have been granted access by the
# owner on HF, and (2) authenticate: `huggingface-cli login`  OR  export HF_TOKEN=hf_xxx.
HF_PRIVATE = True
# Optional: a 640/1024 letterboxed cache repo (faster training). Leave "" to skip.
HF_CACHE_REPO = ""                                 # e.g. "<YOUR>/TCT_NGC_640"

# Optional: a trained TCT checkpoint for eval (your fine-tuned model). Leave "" to skip.
HF_TRAINED_REPO = ""                               # e.g. "<YOUR>/WeDetect-TCT"
HF_TRAINED_FILE = "m1_biomedclip_epoch_12.pth"
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
# Token for the PRIVATE dataset: env HF_TOKEN, else huggingface-cli login cache (None).
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def _auth_hint(err: Exception) -> str:
    return (
        f"\nFailed to download {HF_DATA_REPO} (private): {type(err).__name__}: {err}\n"
        "This is a PRIVATE dataset. Make sure:\n"
        "  1) the owner (Doilion111) granted your HF account access to the repo, and\n"
        "  2) you are logged in:  huggingface-cli login   (or  export HF_TOKEN=hf_xxx)\n")


def _need_hf():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # noqa: F401
    except ImportError:
        raise SystemExit("huggingface_hub missing -- run deploy/install_env.sh first.")
    from huggingface_hub import hf_hub_download, snapshot_download
    return hf_hub_download, snapshot_download


def get_weights(hf_hub_download):
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    print(f"[weights] base {HF_BASE_WEIGHTS_REPO}/{HF_BASE_WEIGHTS_FILE} -> {ckpt_dir}")
    p = hf_hub_download(repo_id=HF_BASE_WEIGHTS_REPO, filename=HF_BASE_WEIGHTS_FILE,
                        token=HF_TOKEN)
    dst = ckpt_dir / HF_BASE_WEIGHTS_FILE
    if not dst.exists():
        os.symlink(p, dst)  # HF caches under ~/.cache; link into checkpoints/
    print(f"   -> {dst}")
    if HF_TRAINED_REPO:
        wd = REPO_ROOT / "work_dirs" / "pretrained"
        wd.mkdir(parents=True, exist_ok=True)
        print(f"[weights] trained {HF_TRAINED_REPO}/{HF_TRAINED_FILE} -> {wd}")
        p2 = hf_hub_download(repo_id=HF_TRAINED_REPO, filename=HF_TRAINED_FILE,
                             token=HF_TOKEN)
        dst2 = wd / HF_TRAINED_FILE
        if not dst2.exists():
            os.symlink(p2, dst2)
        print(f"   -> {dst2}")


def get_data(snapshot_download, data_dir: Path):
    # data_dir is the PARENT; native lands at <data-dir>/TCT_NGC.
    target = data_dir / "TCT_NGC"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[data] {HF_DATA_REPO} (PRIVATE dataset) -> {target}  (large download)")
    try:
        snapshot_download(repo_id=HF_DATA_REPO, repo_type=HF_DATA_REPO_TYPE,
                          local_dir=str(target), local_dir_use_symlinks=False,
                          token=HF_TOKEN)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(_auth_hint(e))
    if HF_CACHE_REPO:
        cache_dir = data_dir / Path(HF_CACHE_REPO).name
        print(f"[data] cache {HF_CACHE_REPO} -> {cache_dir}")
        snapshot_download(repo_id=HF_CACHE_REPO, repo_type=HF_DATA_REPO_TYPE,
                          local_dir=str(cache_dir), local_dir_use_symlinks=False,
                          token=HF_TOKEN)
    print(f"\nNEXT:\n  1) (only the native set was downloaded) build a 640/1024 cache if "
          f"your config needs one:\n"
          f"       TCT_NGC_DATA_ROOT={target} python tools/cache_tct_ngc_640.py "
          f"--size 1024 --out-root {data_dir}/TCT_NGC_1024 "
          f"--splits train_dev_disjoint_dev30 val_dev_disjoint_dev30\n"
          f"  2) write data-root env hints and source them:\n"
          f"       python deploy/patch_data_root.py --data-dir {data_dir}\n"
          f"       source deploy/data_roots.env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["all", "weights", "data"], default="all")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"),
        help="PARENT dir for the dataset; native lands at <data-dir>/TCT_NGC "
             "(default ./data).")
    args = ap.parse_args()
    hf_hub_download, snapshot_download = _need_hf()
    if args.what in ("all", "weights"):
        get_weights(hf_hub_download)
    if args.what in ("all", "data"):
        get_data(snapshot_download, Path(args.data_dir))
    print("download_hf.py done.")


if __name__ == "__main__":
    main()
