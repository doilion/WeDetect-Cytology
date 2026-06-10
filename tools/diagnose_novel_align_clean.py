#!/usr/bin/env python
"""GO/NO-GO 诊断:在 clean Row1 ckpt 上重测 novel 区域图特征→正确 novel 文本的对齐。

动机(2026-06-02):整个对齐方向(PARA region-text alignment)押在历史数字
novel image->text cos = -0.178 上,但那来自 2026-05-11 已废弃的 THAF 变体、
clean backbone 从未复现。本脚本用【现役 clean Row1 ckpt】在【clean novel split】
重测,作为 PARA 做不做的硬闸门。

两个指标(每次都算,一次前向):
  1. raw_cos —— L2cos(原始 cls_embed 中心像素, 类向量),复刻 diagnose_thaf_image_encoder.py,
     【与历史 -0.178 直接可比】。但它是 pre-BN 代理,不反映模型真实匹配几何(见下)。
  2. faithful —— 模型真实的 per-class logit = BNContrastiveHead(cls_embed, text)
     (yolo_world_head.py:91-109:BN(x) 而非 L2,再 einsum 归一化文本,*logit_scale.exp()+bias)。
     它的 argmax = 模型在该 GT 位置的【真实预测】。faithful top1_is_base_rate = 真正的
     "novel 框被判成 base 的比例"(历史称 ~99%)。【绝对对齐结论以 faithful 为准。】

类向量【直接从缓存 .pth 加载】,按【category_id 顺序】对齐(嵌入 dict 插入序 == ann
category_id 序),不靠类名匹配(类名与 ann 名不一致,匹配会全 MISS)。启动时打印
emb-key ↔ ann-cat 配对供人工核对顺序(嵌入若重生成换序会在此暴露)。

判读(faithful novel):
  - top1_is_base_rate 高 + top1_acc 低 → 图像端把 novel 拉向 base,对齐瓶颈坐实(PARA 有的放矢)。
  - top1_acc 高 → 瓶颈不在图像编码器对齐,故事需重定位。
  raw_cos novel mean 与历史 -0.178 对比:强负→错位真;≈0/正→"符号翻正"卖点弱。

用法(等空卡):
  PYTHONPATH=. python tools/diagnose_novel_align_clean.py \
    --checkpoint work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu_clean/best_coco_overall_macro_mAP_epoch_9.pth \
    --out-dir work_dirs/.../novel_align_clean --device cuda:0
CPU 自检(不需 GPU,验证端到端能跑):
  PYTHONPATH=. python tools/diagnose_novel_align_clean.py --smoke
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

from mmdet.utils import register_all_modules

import wedetect  # noqa: F401  register custom modules (backbone/head/etc.)

register_all_modules()

from mmengine.config import Config
from mmengine.registry import MODELS
from PIL import Image
import torchvision.transforms.functional as TF

from wedetect.utils.data_paths import get_tct_ngc_root

_DEF_CFG = "config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py"
_DEF_CKPT = ("work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu_clean/"
             "best_coco_overall_macro_mAP_epoch_9.pth")
_DEF_BASE_EMB = "data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth"
_DEF_NOVEL_EMB = "data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth"
_DEF_DATA_ROOT = get_tct_ngc_root()
_DEF_BASE_ANN = str(_DEF_DATA_ROOT / "annotations/instances_test_base_clean_dev30.json")
_DEF_NOVEL_ANN = str(_DEF_DATA_ROOT / "annotations/instances_test_novel_merged_9.json")
_DEF_IMG_ROOT = str(_DEF_DATA_ROOT / "images")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=_DEF_CFG)
    p.add_argument("--checkpoint", default=_DEF_CKPT)
    p.add_argument("--base-emb", default=_DEF_BASE_EMB)
    p.add_argument("--novel-emb", default=_DEF_NOVEL_EMB)
    p.add_argument("--base-ann", default=_DEF_BASE_ANN)
    p.add_argument("--novel-ann", default=_DEF_NOVEL_ANN)
    p.add_argument("--img-root", default=_DEF_IMG_ROOT)
    p.add_argument("--n-per-class", type=int, default=30)
    p.add_argument("--out-dir", default="work_dirs/novel_align_clean_diag")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--smoke", action="store_true",
                   help="CPU self-test: force cpu, n_per_class=1, cap images")
    p.add_argument("--limit-images", type=int, default=0,
                   help="cap GT per split (0=no cap); set by --smoke. NOTE: caps the "
                        "category-ordered list, so a small cap biases toward early "
                        "category_ids — only for quick interim runs, use 0 for the real run.")
    return p.parse_args()


def _as_vec(v) -> torch.Tensor:
    """Coerce a stored embedding (D / 1xD / kxD) to a single [D] vector."""
    t = torch.as_tensor(v).float()
    return t.reshape(-1, t.shape[-1]).mean(0)


def load_class_vectors(base_emb_path: str, novel_emb_path: str, device: str):
    """[39,D] L2-normed class vectors (30 base + 9 novel) in category_id order.

    The cached .pth are dicts keyed by prompt string; their INSERTION ORDER is
    assumed to mirror COCO category_id order. We stack by dict order and index
    later by category_id — never by name. The assumption is only count-checked
    here; main() additionally prints emb-key <-> ann-cat pairs for human eyeball.
    Returns (class_vecs[39,D], n_base, base_keys, novel_keys).
    """
    base = torch.load(base_emb_path, map_location="cpu")
    novel = torch.load(novel_emb_path, map_location="cpu")
    if not isinstance(base, dict) or not isinstance(novel, dict):
        raise TypeError("expected dict-format embedding caches (key=prompt)")
    base_keys, novel_keys = list(base.keys()), list(novel.keys())
    base_vecs = torch.stack([_as_vec(base[k]) for k in base_keys])    # [30, D]
    novel_vecs = torch.stack([_as_vec(novel[k]) for k in novel_keys])  # [9, D]
    n_base = base_vecs.shape[0]
    assert n_base == 30 and novel_vecs.shape[0] == 9, (
        f"expected 30 base + 9 novel, got {n_base}+{novel_vecs.shape[0]}")
    assert base_vecs.shape[1] == novel_vecs.shape[1], "base/novel dim mismatch"
    cv = F.normalize(torch.cat([base_vecs, novel_vecs], 0).float(), dim=-1)
    return cv.to(device), n_base, base_keys, novel_keys


def cat_names_in_id_order(ann_path: str):
    data = json.loads(Path(ann_path).read_text())
    return [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]


def build_detector(cfg: Config, ckpt_path: str, device: str) -> torch.nn.Module:
    model = MODELS.build(cfg.model.copy())
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    incompat = model.load_state_dict(sd, strict=False)
    miss = [k for k in incompat.missing_keys if "text_model" not in k]
    if miss:
        print(f"[warn] {len(miss)} non-text missing keys, first: {miss[:3]}")
    if incompat.unexpected_keys:
        print(f"[warn] {len(incompat.unexpected_keys)} unexpected, "
              f"first: {sorted(incompat.unexpected_keys)[:3]}")
    return model.to(device).eval()


def load_image_letterboxed(img_path: str, size: int = 640):
    img = Image.open(img_path).convert("RGB")
    w0, h0 = img.size
    scale = size / max(h0, w0)
    nw, nh = int(w0 * scale), int(h0 * scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    pl, pt = (size - nw) // 2, (size - nh) // 2
    arr = np.pad(np.array(img, dtype=np.uint8),
                 ((pt, size - nh - pt), (pl, size - nw - pl), (0, 0)),
                 mode="constant", constant_values=114)
    # data_preprocessor is mean=0/std=255 (just /255), so to_tensor's [0,1] matches
    # what the backbone sees post-preprocessor.
    return TF.to_tensor(arr), scale, (pl, pt)


def gt_center_in_fmap(bbox, scale, pad, feat_size, size=640):
    x, y, w, h = bbox
    cx = (x + w / 2) * scale + pad[0]
    cy = (y + h / 2) * scale + pad[1]
    stride = size / feat_size
    fx = max(0, min(feat_size - 1, int(cx / stride)))
    fy = max(0, min(feat_size - 1, int(cy / stride)))
    return fy, fx


@torch.no_grad()
def scores_at_gt(detector, image_tensor, bbox, scale, pad, class_vecs, device, size=640):
    """Per-GT scores over the 39 classes at the bbox-center pixel, avg over FPN scales.

    Returns (raw_cos[K], faithful_logit[K]) on CPU:
      raw_cos       = L2cos(raw cls_embed, class_vecs)  -- comparable to historical THAF -0.178.
      faithful_logit= BNContrastiveHead(cls_embed, text) -- the model's REAL per-class logit
                      (BN on image side + normalized text + logit_scale/bias). argmax = real prediction.
    """
    x = image_tensor.unsqueeze(0).to(device)
    img_feats = detector.neck(detector.backbone.forward_image(x))
    hm = detector.bbox_head.head_module
    w = class_vecs.unsqueeze(0)  # [1, K, D]; cls_contrast re-normalizes w internally
    raw_per_scale, logit_per_scale = [], []
    for feat, cls_pred, cls_contrast in zip(img_feats, hm.cls_preds, hm.cls_contrasts):
        cls_embed = cls_pred(feat)            # [1, D, H, W]
        _, _, H, W = cls_embed.shape
        fy, fx = gt_center_in_fmap(bbox, scale, pad, H, size)
        raw_per_scale.append(cls_embed[0, :, fy, fx])              # [D] (pre-BN)
        logit = cls_contrast(cls_embed, w)                         # [1, K, H, W]
        logit_per_scale.append(logit[0, :, fy, fx])               # [K]
    raw = F.normalize(torch.stack(raw_per_scale).mean(0), dim=-1)  # [D]
    raw_cos = raw @ class_vecs.T                                   # [K]
    faithful = torch.stack(logit_per_scale).mean(0)               # [K]
    return raw_cos.cpu(), faithful.cpu()


def sample_by_catid(ann_path, img_root, n_per_class, seed, idx_offset, limit=0):
    """Sample n_per_class GT per category. Returns list of (img_path, bbox, idx)
    where idx = category_id + idx_offset (global class-vector index)."""
    data = json.loads(Path(ann_path).read_text())
    cat_ids = sorted(c["id"] for c in data["categories"])
    assert cat_ids == list(range(len(cat_ids))), (
        f"{ann_path}: category_ids not contiguous 0..{len(cat_ids)-1}: {cat_ids[:5]}")
    img_name = {im["id"]: im["file_name"] for im in data["images"]}
    by_cat = {cid: [] for cid in cat_ids}
    for a in data["annotations"]:
        by_cat[a["category_id"]].append(a)
    rng = random.Random(seed)
    out = []
    for cid in cat_ids:
        for a in rng.sample(by_cat[cid], min(n_per_class, len(by_cat[cid]))):
            out.append((str(Path(img_root) / img_name[a["image_id"]]),
                        a["bbox"], cid + idx_offset))
    return out[:limit] if limit else out


def agg(recs):
    if not recs:
        return {}
    raw = np.array([r["raw_gt_cos"] for r in recs])
    fit = np.array([r["faithful_gt_logit"] for r in recs])
    return dict(
        n=len(recs),
        raw_cos_mean=float(raw.mean()), raw_cos_std=float(raw.std()),
        raw_top1_acc=sum(r["raw_top1"] == r["gt_idx"] for r in recs) / len(recs),
        raw_top1_is_base_rate=sum(r["raw_top1_is_base"] for r in recs) / len(recs),
        faithful_gt_logit_mean=float(fit.mean()),
        faithful_top1_acc=sum(r["faithful_top1"] == r["gt_idx"] for r in recs) / len(recs),
        faithful_top1_is_base_rate=sum(r["faithful_top1_is_base"] for r in recs) / len(recs),
    )


def main():
    args = parse_args()
    if args.smoke:
        args.device, args.n_per_class, args.limit_images = "cpu", 1, 4
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    class_vecs, n_base, base_keys, novel_keys = load_class_vectors(
        args.base_emb, args.novel_emb, args.device)
    print(f"[setup] class_vecs {tuple(class_vecs.shape)} (n_base={n_base}), device={args.device}")

    # B) order cross-check: emb dict order must mirror ann category_id order.
    base_cats = cat_names_in_id_order(args.base_ann)
    novel_cats = cat_names_in_id_order(args.novel_ann)
    assert len(base_cats) == n_base and len(novel_cats) == len(novel_keys), (
        "ann category count != embedding count — order assumption unsafe")
    print("[order-check] emb key  <->  ann category (eyeball alignment):")
    for i in (0, 1, n_base - 1):
        print(f"   base[{i}] {base_keys[i][:42]!r:46} <-> {base_cats[i]!r}")
    for i in range(len(novel_keys)):
        print(f"   nov [{i}] {novel_keys[i][:42]!r:46} <-> {novel_cats[i]!r}")

    detector = build_detector(cfg, args.checkpoint, args.device)

    samples = {
        "base": sample_by_catid(args.base_ann, args.img_root, args.n_per_class,
                                 args.seed, 0, args.limit_images),
        "novel": sample_by_catid(args.novel_ann, args.img_root, args.n_per_class,
                                  args.seed, n_base, args.limit_images),
    }
    print(f"[sample] base={len(samples['base'])} novel={len(samples['novel'])}")

    records = []
    for tag, samp in samples.items():
        for i, (img_path, bbox, gidx) in enumerate(samp):
            try:
                t, scale, pad = load_image_letterboxed(img_path)
            except (FileNotFoundError, OSError):
                continue
            raw_cos, faithful = scores_at_gt(detector, t, bbox, scale, pad,
                                             class_vecs, args.device)
            r_top1, f_top1 = int(raw_cos.argmax()), int(faithful.argmax())
            records.append(dict(
                split=tag, gt_idx=gidx,
                raw_gt_cos=float(raw_cos[gidx]), raw_top1=r_top1,
                raw_top1_is_base=(r_top1 < n_base),
                faithful_gt_logit=float(faithful[gidx]), faithful_top1=f_top1,
                faithful_top1_is_base=(f_top1 < n_base)))
            if (i + 1) % 100 == 0:
                print(f"  [{tag}] {i+1}/{len(samp)}")
        print(f"  [{tag}] done ({sum(1 for r in records if r['split']==tag)})")

    base_agg = agg([r for r in records if r["split"] == "base"])
    novel_agg = agg([r for r in records if r["split"] == "novel"])
    if not novel_agg or not base_agg:
        raise RuntimeError("no usable GT records for a split — check img-root / ann paths")

    rc = novel_agg["raw_cos_mean"]
    fbase = novel_agg["faithful_top1_is_base_rate"]
    facc = novel_agg["faithful_top1_acc"]
    # raw_cos comparability with historical THAF -0.178
    if rc < -0.05:
        raw_v = f"raw_cos novel={rc:.3f} 强负(历史 THAF=-0.178)→ '符号翻正' 卖点成立"
    elif rc < 0.05:
        raw_v = f"raw_cos novel={rc:.3f} ≈0 → '符号翻正' 弱(无强负可翻)"
    else:
        raw_v = f"raw_cos novel={rc:.3f} 已转正 → '符号翻正' 卖点失效"
    # faithful = the model's REAL behavior (authoritative)
    if fbase > 0.5:
        fit_v = (f"FAITHFUL: {fbase*100:.0f}% novel 框被模型真实预测为 base 类(top1_acc "
                 f"{facc*100:.0f}%)→ 图像端对齐瓶颈【坐实】,PARA 有的放矢(卖'降误分率')")
    elif facc < 0.34:
        fit_v = (f"FAITHFUL: novel top1 多为 novel 但非正确类(acc {facc*100:.0f}%, "
                 f"is_base {fbase*100:.0f}%)→ 器官内细类混淆为主,非'拉向 base'")
    else:
        fit_v = (f"FAITHFUL: novel top1_acc {facc*100:.0f}% 不低 → 瓶颈不在图像编码器对齐,故事需重定位")

    summary = dict(checkpoint=args.checkpoint, device=args.device,
                   historical_thaf_raw_cos=-0.178, n_base=n_base,
                   base=base_agg, novel=novel_agg,
                   verdict_raw=raw_v, verdict_faithful=fit_v)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n==== GO/NO-GO ====")
    print(f"base : raw_cos={base_agg['raw_cos_mean']:.3f} | faithful top1_acc="
          f"{base_agg['faithful_top1_acc']*100:.1f}% is_base={base_agg['faithful_top1_is_base_rate']*100:.1f}%")
    print(f"novel: raw_cos={rc:.3f} (历史 -0.178) | faithful top1_acc="
          f"{facc*100:.1f}% is_base={fbase*100:.1f}% gt_logit={novel_agg['faithful_gt_logit_mean']:.3f}")
    print(f"[raw]      {raw_v}")
    print(f"[faithful] {fit_v}")
    print(f"[json] {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
