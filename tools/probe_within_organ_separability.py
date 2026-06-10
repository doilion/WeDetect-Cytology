"""Within-organ region-feature separability probe — go/no-go for Innovation ① (B).

QUESTION: is fine-class information LINEARLY decodable from clean Row1's region
features (pre-BN cls_embed at the GT-center pixel), WITHIN each organ?

  - If a supervised linear probe / NCM recovers within-organ top1 FAR ABOVE the
    model's NATIVE within-organ faithful top1  ->  the discriminative info IS present
    in the region features but thrown away by the frozen cos-0.96 text matching;
    organ-conditioned hard-negative training (①B) has real headroom.  GO.
  - If the probe is ALSO stuck near native  ->  info isn't in the region features =
    encoder / 640-resolution capped; ①B would be futile, do ② (CLIPSelf-gap) or
    higher-res first.  NO-GO.

Reuses model-load + GT sampling from diagnose_novel_align_clean.py. Pure-torch
probe (sklearn unavailable in this env).

Run:
  PYTHONPATH=. python tools/probe_within_organ_separability.py --device cuda:0 \
      --n-per-class 60 --out-dir work_dirs/within_organ_probe
Smoke (CPU, end-to-end):
  PYTHONPATH=. python tools/probe_within_organ_separability.py --smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# importing the diag module also runs register_all_modules() + import wedetect
from tools.diagnose_novel_align_clean import (  # noqa: E402
    _as_vec, build_detector, load_image_letterboxed, gt_center_in_fmap,
    sample_by_catid)
from mmengine.config import Config  # noqa: E402
from wedetect.utils.data_paths import get_tct_ngc_root  # noqa: E402

_DEF_CFG = "config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py"
_DEF_CKPT = ("work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu_clean/"
             "best_coco_overall_macro_mAP_epoch_9.pth")
_DEF_BASE_EMB = "data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth"
_DEF_DATA_ROOT = get_tct_ngc_root()
_DEF_BASE_ANN = str(_DEF_DATA_ROOT / "annotations/instances_test_base_clean_dev30.json")
_DEF_IMG_ROOT = str(_DEF_DATA_ROOT / "images")
_DEF_MASK = "data/texts/tct_ngc_class_organ_mask_base30.pt"


def load_base_vectors(path: str, device: str):
    d = torch.load(path, map_location="cpu")
    if not isinstance(d, dict):
        raise TypeError("expected dict-format base embedding cache")
    keys = list(d.keys())
    vecs = torch.stack([_as_vec(d[k]) for k in keys])  # [30, D]
    assert vecs.shape[0] == 30, f"expected 30 base, got {vecs.shape[0]}"
    return F.normalize(vecs.float(), dim=-1).to(device), keys


def load_organ_map(mask_path: str):
    """Return (organ_of_catid: dict[int,int], organ_names: list[str]).

    mask file: class_ids (category_id order) + mask[30,5] one-hot class->organ.
    """
    m = torch.load(mask_path, map_location="cpu")
    class_ids = m["class_ids"]
    organ_names = m["organ_names"]
    mask = m["mask"]  # [30,5]
    organ_of_catid = {int(class_ids[i]): int(mask[i].argmax()) for i in range(len(class_ids))}
    return organ_of_catid, organ_names


@torch.no_grad()
def feat_and_faithful(detector, image_tensor, bbox, scale, pad, class_vecs, device, size=640):
    """(region_feat[D] L2-normed avg-over-scales pre-BN cls_embed @ GT center,
        faithful_logit[K] over the K base classes)."""
    x = image_tensor.unsqueeze(0).to(device)
    img_feats = detector.neck(detector.backbone.forward_image(x))
    hm = detector.bbox_head.head_module
    w = class_vecs.unsqueeze(0)  # [1,K,D]
    feats, logits = [], []
    for feat, cls_pred, cls_contrast in zip(img_feats, hm.cls_preds, hm.cls_contrasts):
        cls_embed = cls_pred(feat)               # [1,D,H,W]
        _, _, H, W = cls_embed.shape
        fy, fx = gt_center_in_fmap(bbox, scale, pad, H, size)
        feats.append(cls_embed[0, :, fy, fx])    # [D] pre-BN
        lg = cls_contrast(cls_embed, w)          # [1,K,H,W]
        logits.append(lg[0, :, fy, fx])          # [K]
    feat = F.normalize(torch.stack(feats).mean(0), dim=-1)  # [D]
    faithful = torch.stack(logits).mean(0)                  # [K]
    return feat.cpu(), faithful.cpu()


def logreg_cv(X, y, n_class, device, folds=5, epochs=300, lr=0.05, wd=1e-3, seed=0):
    """Pure-torch multinomial logistic regression, stratified-ish k-fold; mean test top1."""
    N, D = X.shape
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    X, y = X[perm], y[perm]
    fold_id = torch.arange(N) % folds
    accs = []
    for f in range(folds):
        te = fold_id == f
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        Xtr, ytr = X[tr].to(device), y[tr].to(device)
        Xte, yte = X[te].to(device), y[te].to(device)
        if len(torch.unique(ytr)) < n_class:
            continue
        clf = torch.nn.Linear(D, n_class).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            F.cross_entropy(clf(Xtr), ytr).backward()
            opt.step()
        with torch.no_grad():
            accs.append((clf(Xte).argmax(1) == yte).float().mean().item())
    return float(np.mean(accs)) if accs else float("nan")


def ncm_cv(X, y, n_class, folds=5, seed=0):
    """Nearest-class-mean (cosine) top1, k-fold — robust non-parametric sanity."""
    N = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    X, y = X[perm], y[perm]
    fold_id = torch.arange(N) % folds
    accs = []
    for f in range(folds):
        te = fold_id == f
        tr = ~te
        means, ok = [], True
        for c in range(n_class):
            mc = X[tr][y[tr] == c]
            if mc.shape[0] == 0:
                ok = False
                break
            means.append(mc.mean(0))
        if not ok or te.sum() == 0:
            continue
        M = F.normalize(torch.stack(means), dim=-1)
        pred = (F.normalize(X[te], dim=-1) @ M.T).argmax(1)
        accs.append((pred == y[te]).float().mean().item())
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_DEF_CFG)
    ap.add_argument("--checkpoint", default=_DEF_CKPT)
    ap.add_argument("--base-emb", default=_DEF_BASE_EMB)
    ap.add_argument("--base-ann", default=_DEF_BASE_ANN)
    ap.add_argument("--img-root", default=_DEF_IMG_ROOT)
    ap.add_argument("--mask", default=_DEF_MASK)
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--out-dir", default="work_dirs/within_organ_probe")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.device, args.n_per_class = "cpu", 3
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    class_vecs, keys = load_base_vectors(args.base_emb, args.device)
    organ_of_catid, organ_names = load_organ_map(args.mask)
    print(f"[setup] base class_vecs {tuple(class_vecs.shape)}, "
          f"{len(organ_names)} organs={organ_names}, device={args.device}")

    detector = build_detector(cfg, args.checkpoint, args.device)
    samples = sample_by_catid(args.base_ann, args.img_root, args.n_per_class,
                              args.seed, 0, 0)
    print(f"[sample] {len(samples)} base GT boxes")

    feats, catids, faiths = [], [], []
    for i, (img_path, bbox, catid) in enumerate(samples):
        try:
            img_t, scale, pad = load_image_letterboxed(img_path)
            ft, fa = feat_and_faithful(detector, img_t, bbox, scale, pad,
                                       class_vecs, args.device)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {Path(img_path).name}: {e}")
            continue
        feats.append(ft); catids.append(catid); faiths.append(fa)
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(samples)}")
    X = torch.stack(feats)                 # [N, D]
    catid = torch.tensor(catids)           # [N]
    faith = torch.stack(faiths)            # [N, 30]
    print(f"[feat] X={tuple(X.shape)}  faith={tuple(faith.shape)}")

    # global cross-organ faithful top1 (context)
    glob_top1 = (faith.argmax(1) == catid).float().mean().item()

    # group by organ
    by_organ = {}
    for j, c in enumerate(catid.tolist()):
        by_organ.setdefault(organ_of_catid[c], []).append(j)

    rows, w_native, w_lp, w_ncm, weights = [], [], [], [], []
    for oid in sorted(by_organ):
        idx = torch.tensor(by_organ[oid])
        local_catids = sorted(set(catid[idx].tolist()))
        n_cls = len(local_catids)
        cid2loc = {c: k for k, c in enumerate(local_catids)}
        Xo = X[idx]
        yo = torch.tensor([cid2loc[int(c)] for c in catid[idx]])
        # native within-organ faithful top1: restrict logits to this organ's catids
        sub = faith[idx][:, local_catids]                 # [n, n_cls]
        native = (sub.argmax(1) == yo).float().mean().item() if n_cls > 1 else 1.0
        if n_cls > 1:
            lp = logreg_cv(Xo, yo, n_cls, args.device, seed=args.seed)
            ncm = ncm_cv(Xo, yo, n_cls, seed=args.seed)
        else:
            lp = ncm = float("nan")
        rows.append(dict(organ=organ_names[oid], n_classes=n_cls, n_samples=len(idx),
                         native_within_top1=round(native, 4),
                         linprobe_within_top1=round(lp, 4),
                         ncm_within_top1=round(ncm, 4),
                         headroom=round((lp - native), 4) if n_cls > 1 else 0.0))
        if n_cls > 1:
            w_native.append(native); w_lp.append(lp); w_ncm.append(ncm)
            weights.append(len(idx))

    print("\n=== WITHIN-ORGAN SEPARABILITY (clean Row1 region features) ===")
    print(f"{'organ':<12}{'#cls':>5}{'#smp':>6}{'native':>9}{'linprobe':>10}{'NCM':>8}{'headroom':>10}")
    for r in rows:
        print(f"{r['organ']:<12}{r['n_classes']:>5}{r['n_samples']:>6}"
              f"{r['native_within_top1']:>9.3f}{r['linprobe_within_top1']:>10.3f}"
              f"{r['ncm_within_top1']:>8.3f}{r['headroom']:>10.3f}")
    macro = lambda a: float(np.mean(a)) if a else float("nan")  # noqa: E731
    print(f"\nmacro (multi-class organs): native {macro(w_native):.3f}  "
          f"linprobe {macro(w_lp):.3f}  NCM {macro(w_ncm):.3f}  "
          f"=> headroom {macro(w_lp)-macro(w_native):+.3f}")
    print(f"global cross-organ faithful top1 (30-way): {glob_top1:.3f}")
    print("\nVERDICT GUIDE: linprobe >> native (headroom big, e.g. >+0.20) => GO (①B has "
          "headroom); linprobe ~ native (both low) => NO-GO (encoder/res capped).")

    summary = dict(global_faithful_top1_30way=round(glob_top1, 4),
                   macro_native=round(macro(w_native), 4),
                   macro_linprobe=round(macro(w_lp), 4),
                   macro_ncm=round(macro(w_ncm), 4),
                   per_organ=rows, n_total=int(X.shape[0]),
                   n_per_class=args.n_per_class, checkpoint=args.checkpoint)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[json] {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
