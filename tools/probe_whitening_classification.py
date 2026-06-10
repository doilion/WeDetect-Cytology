"""Decisive cheap probe (run on a machine with a free GPU): does WHITENING the
text<->image comparison actually improve GT-box classification -- i.e. does the model
'understand' after whitening, or did whitening just spread the embeddings and BREAK the
text<->image alignment?

Background: the BiomedCLIP cytology text collapse is mostly ANISOTROPY (a removable cone)
-- centering/whitening drops within-organ cos 0.79->0.04 and novel<->base 0.70->0.41.
But lower cosine != better classification. This probe answers it empirically WITHOUT a
full detection eval: classify GT-box image features (BN(cls_embed)) against the text,
WITH vs WITHOUT a whitening applied CONSISTENTLY to both image and text.

Decision rule (per split):
  whitened top1 > raw top1  -> whitening preserved/enhanced meaning (model understands)
  whitened top1 < raw top1  -> geometric trick that broke alignment

Run:
  PYTHONPATH=. python tools/probe_whitening_classification.py \
      --checkpoint work_dirs/.../best_*.pth --device cuda:0 --topk 1 --n-per-class 40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.diagnose_novel_align_clean import (  # noqa: E402
    build_detector, sample_by_catid, load_image_letterboxed, gt_center_in_fmap)
from mmengine.config import Config  # noqa: E402
from wedetect.utils.data_paths import get_tct_ngc_640_root  # noqa: E402

_CFG = "config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py"
_CKPT = "work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu_clean/epoch_12.pth"
_DATA_640 = get_tct_ngc_640_root()
_BASE_ANN = str(_DATA_640 / "annotations/instances_val_dev_disjoint_dev30.json")
_IMG_640 = str(_DATA_640 / "images")
_MASK = "data/texts/tct_ngc_class_organ_mask_base30.pt"
_BASE_TXT = "data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth"


@torch.no_grad()
def bn_cls_embed_at_gt(det, img_t, bbox, scale, pad, device, size=640):
    """BN(cls_pred(neck))[L0] at the GT center pixel -- the 512-d space the head
    compares text against. (Same recipe as tools/build_visual_prototype_text.py.)"""
    x = img_t.unsqueeze(0).to(device)
    img_feats = det.neck(det.backbone.forward_image(x))
    hm = det.bbox_head.head_module
    feat0 = img_feats[0]; _, _, H, W = feat0.shape
    fy, fx = gt_center_in_fmap(bbox, scale, pad, H, size)
    ce = hm.cls_contrasts[0].norm(hm.cls_preds[0](feat0))   # [1,512,H,W]
    return ce[0, :, fy, fx].cpu()


def whiten_fit(text_base, topk):
    """Whitening fit on BASE text: remove global mean + top-k anisotropy directions.
    Transform = (x - mu) @ V.t() ; class-agnostic so it also applies to novel text."""
    mu = text_base.mean(0)
    _, _, Vt = torch.linalg.svd(text_base - mu, full_matrices=False)
    return mu, Vt[topk:]


def apply_w(x, mu, V):
    return (x - mu) @ V.t()


def collect(det, ann, img_root, n, device):
    """GT-box BN(cls_embed) features + their category ids."""
    samples = sample_by_catid(ann, img_root, n, 0, 0, 0)
    feats, catids = [], []
    for img_path, bbox, catid in samples:
        try:
            img_t, scale, pad = load_image_letterboxed(img_path)
            feats.append(bn_cls_embed_at_gt(det, img_t, bbox, scale, pad, device))
            catids.append(catid)
        except Exception:  # noqa: BLE001
            continue
    return torch.stack(feats), torch.tensor(catids)


def top1(feats, text, gt_rows, organ_q=None, organ_t=None):
    """global top1; if organ maps given, within-organ top1 (restrict to same organ)."""
    Fn = F.normalize(feats, dim=-1); Tn = F.normalize(text, dim=-1)
    S = Fn @ Tn.t()                                  # [Nq, Ncls]
    if organ_q is None:
        return float((S.argmax(1) == gt_rows).float().mean())
    c = 0
    for q in range(feats.shape[0]):
        same = (organ_t == organ_q[q]).nonzero().squeeze(1)
        c += int(same[S[q, same].argmax()] == gt_rows[q])
    return c / max(feats.shape[0], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_CFG)
    ap.add_argument("--checkpoint", default=_CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--n-per-class", type=int, default=40)
    args = ap.parse_args()

    mask = torch.load(_MASK, map_location="cpu", weights_only=False)
    class_ids = [int(c) for c in mask["class_ids"]]      # base text row order == this
    catid2row = {c: i for i, c in enumerate(class_ids)}
    organ_t = mask["mask"].argmax(1)                     # [30] class-row -> organ

    bt = torch.load(_BASE_TXT, map_location="cpu")
    Tb = torch.stack([torch.as_tensor(bt[k]).float().reshape(-1) for k in bt])  # [30,512]
    mu, V = whiten_fit(Tb, args.topk)

    det = build_detector(Config.fromfile(args.config), args.checkpoint, args.device)
    print("[collect] base GT features...")
    Fb, cb = collect(det, _BASE_ANN, _IMG_640, args.n_per_class, args.device)
    rows = torch.tensor([catid2row[int(c)] for c in cb])
    organ_q = torch.tensor([int(organ_t[r]) for r in rows])

    Fb_w, Tb_w = apply_w(Fb, mu, V), apply_w(Tb, mu, V)
    print(f"\n=== BASE GT-box classification (n={Fb.shape[0]}, whiten fit-on-base, topk={args.topk}) ===")
    for tag, restrict in [("global-30", False), ("within-organ", True)]:
        oq, ot = (organ_q, organ_t) if restrict else (None, None)
        raw = top1(Fb, Tb, rows, oq, ot)
        wht = top1(Fb_w, Tb_w, rows, oq, ot)
        print(f"  {tag:13s}: raw {raw:.3f}  ->  whitened {wht:.3f}   "
              f"({'UP' if wht > raw else 'DOWN'} {wht - raw:+.3f})")
    print("\nRule: whitened > raw => whitening preserves/enhances meaning (model understands);"
          "\n      whitened < raw => geometric trick broke text<->image alignment.")
    print("\nNOVEL extension: load the novel text + novel ann, build novel catid->row by "
          "category NAME match, reuse collect()/apply_w()/top1() with the SAME (mu,V) "
          "(fit on base) -> the class-agnostic whitening is what must generalize to novel.")


if __name__ == "__main__":
    main()
