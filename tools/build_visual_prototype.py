#!/usr/bin/env python
"""Build visual exemplar class prototypes for novel zero-shot detection.

Replaces the text-derived class vector with the **mean image feature at the
GT bbox region** of N exemplar images, for each novel class. Works with
the existing PseudoLanguageBackbone cache_bank since the contrastive head
treats the class vector as a generic 768-dim embedding regardless of source.

Pipeline (per class):
  1. Sample N GT bboxes from the novel ann file (deterministic seed)
  2. Crop image to bbox + context, resize to 640x640
  3. Forward image through model.backbone.image_model → neck → head.cls_preds[i]
     (this is the same path that produces the 768-dim per-cell image features
     the contrastive head L2-norms and dots with text features)
  4. Spatial mean per FPN scale, then mean across scales → [768] vector
  5. Mean across N exemplars → class prototype
  6. Save as {primary_text → tensor} dict, drop-in replacement for the text emb cache

Output dict key = the same string `LoadText` will look up at inference (i.e. the
first variant in the novel JSON's inner list, same convention as text emb cache).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector

from wedetect.utils import resolve_latest_checkpoint
from wedetect.utils.data_paths import as_posix_dir, get_tct_ngc_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--ann-file",
        required=True,
        help="abs or relative to --data-root, e.g. annotations/instances_test_main_novel.json",
    )
    p.add_argument("--data-root", default=as_posix_dir(get_tct_ngc_root()))
    p.add_argument(
        "--img-prefix",
        default="images/",
        help="prepended to ann image file_name (matches dataset's data_prefix)",
    )
    p.add_argument(
        "--text-json",
        required=True,
        help="novel split prompt JSON; the FIRST variant in each inner list "
        "is used as the dict key (same convention as text emb cache)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260509)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--expand",
        type=float,
        default=1.5,
        help="(legacy path only) bbox expansion ratio for crop context",
    )
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="FPN scale indices to pool over; 0=stride8, 1=stride16, 2=stride32",
    )

    # ── Phase A: ROI prototype path (default ON, geometry matches detector test pipeline) ──
    p.add_argument(
        "--legacy-crop",
        action="store_true",
        default=False,
        help="OLD path: cv2.resize(bbox_crop, 640x640) + spatial global mean. "
             "Default is NEW path: KeepRatioResize+LetterResize + ROIAlign at FPN level.",
    )
    p.add_argument(
        "--roi-size",
        type=int,
        default=7,
        help="ROIAlign output spatial size (square). Default 7 (matches Faster R-CNN convention).",
    )
    p.add_argument(
        "--bg-lambda",
        type=float,
        default=0.0,
        help="Context-ring subtraction weight: r_s = fg_s - bg_lambda * (ring_s - fg_s). "
             "0.0 = pure foreground ROI (default). 0.25-0.5 ablation.",
    )
    p.add_argument(
        "--bg-expand",
        type=float,
        default=1.5,
        help="Ring expansion factor for background subtraction (used only when --bg-lambda > 0).",
    )
    p.add_argument(
        "--roi-expand",
        type=float,
        default=1.0,
        help="Expand bbox by this factor BEFORE ROIAlign (cytology cells are small at native scale; "
             "expanding to 1.5× includes some context, similar to legacy 1.5× crop). "
             "1.0 = exact bbox (default). 1.5 = include 1.5× context like legacy.",
    )
    p.add_argument(
        "--pad-val",
        type=int,
        default=114,
        help="Letterbox padding value (default 114, matches detector test_pipeline).",
    )
    p.add_argument(
        "--save-diag",
        action="store_true",
        default=False,
        help="Save per-class diagnostic stats (intra-cosine, per-exemplar feats) "
             "as <out>.diag.pth for later Phase B clustering.",
    )
    return p.parse_args()


def load_image_and_crop(
    img_path: Path,
    bbox_xywh: list[float],
    expand: float,
    target_size: int,
) -> np.ndarray:
    """Returns BGR uint8 [H, W, 3], same convention as cv2.imread."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    H, W = img.shape[:2]
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2, y + h / 2
    ew, eh = w * expand, h * expand
    x1 = max(0, int(cx - ew / 2))
    y1 = max(0, int(cy - eh / 2))
    x2 = min(W, int(cx + ew / 2))
    y2 = min(H, int(cy + eh / 2))
    if x2 <= x1 or y2 <= y1:
        # Degenerate bbox; fall back to original
        x1, y1, x2, y2 = 0, 0, W, H
    crop = img[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return crop_resized


def preprocess(bgr_img: np.ndarray, device: str) -> torch.Tensor:
    """Match YOLOWDetDataPreprocessor: bgr→rgb, /255, [B,3,H,W] float32."""
    rgb = bgr_img[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return tensor


@torch.no_grad()
def extract_image_embedding(model, img_tensor: torch.Tensor, scales: list[int]) -> torch.Tensor:
    """LEGACY path: forward bbox-crop through backbone → neck → head.cls_preds[i] for each
    i in scales, spatial-mean-pool, then mean across scales → [embed_dim]."""
    img_feats = model.backbone.image_model(img_tensor)
    fpn_feats = model.neck(img_feats)
    head_module = model.bbox_head.head_module
    pooled = []
    for i in scales:
        proj = head_module.cls_preds[i](fpn_feats[i])  # [1, embed_dims, H, W]
        pooled.append(proj.mean(dim=(2, 3)).squeeze(0))  # [embed_dims]
    return torch.stack(pooled, dim=0).mean(dim=0)  # [embed_dims]


# ─────────────────────────── Phase A: ROI prototype path ───────────────────────────
# Helpers below are also in tools/build_savpe_visproto.py:89-165 — kept in sync.
# Idea: full-image letterbox forward + ROIAlign on bbox at FPN feature level,
# instead of crop+resize+global-pool. Eliminates 3 geometry biases:
#   1. crop-resize stretches non-square bboxes
#   2. 1.5× context introduces background into the "single-class" feature
#   3. spatial global mean over crop dilutes signal with ~50% background

def build_test_preprocessing_pipeline(input_size: int, pad_val: int) -> Compose:
    """Same KeepRatioResize + LetterResize the detector uses at test."""
    pipeline_cfg = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="WeDetectKeepRatioResize", scale=(input_size, input_size)),
        dict(
            type="WeDetectLetterResize",
            scale=(input_size, input_size),
            allow_scale_up=False,
            pad_val=dict(img=pad_val),
        ),
    ]
    return Compose(pipeline_cfg)


def transform_bbox_xywh(
    bbox_xywh: List[float],
    scale_factor: Tuple[float, float],
    pad_param: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Apply KeepRatio+Letter transform to (x, y, w, h) bbox → letterbox coords."""
    x, y, w, h = bbox_xywh
    w_r, h_r = float(scale_factor[0]), float(scale_factor[1])
    pad_top, _, pad_left, _ = (float(v) for v in pad_param)
    return x * w_r + pad_left, y * h_r + pad_top, w * w_r, h * h_r


def preprocess_image_via_pipeline(
    pipeline: Compose, img_path: Path
) -> Tuple[torch.Tensor, Tuple[float, float], np.ndarray, Tuple[int, int]]:
    """Run test pipeline on a single image.

    Returns:
        img_tensor: [1, 3, 640, 640] float32 RGB in [0, 1]
        scale_factor: (w_ratio, h_ratio)
        pad_param: np.array([top, bot, left, right])
        ori_shape: (H0, W0) before any resize, for sanity
    """
    data = dict(img_path=str(img_path))
    results = pipeline(data)
    img = results["img"]  # 640×640 BGR uint8
    scale_factor = results["scale_factor"]

    # Reverse-engineer pad_param from ori_shape + scale_factor (WeDetectLetterResize
    # doesn't write pad_param to results when half_pad_param=False).
    ori_shape = results.get("ori_shape", None)
    if ori_shape is None:
        raise RuntimeError(
            "expected 'ori_shape' in pipeline results — LoadImageFromFile should set it"
        )
    H0, W0 = ori_shape
    w_r, h_r = float(scale_factor[0]), float(scale_factor[1])
    no_pad_h = int(round(H0 * h_r))
    no_pad_w = int(round(W0 * w_r))
    target_h, target_w = img.shape[:2]
    padding_h = target_h - no_pad_h
    padding_w = target_w - no_pad_w
    pad_top = int(round(padding_h // 2 - 0.1)) if padding_h > 0 else 0
    pad_bot = padding_h - pad_top
    pad_left = int(round(padding_w // 2 - 0.1)) if padding_w > 0 else 0
    pad_right = padding_w - pad_left
    pad_param = np.array([pad_top, pad_bot, pad_left, pad_right], dtype=np.float32)

    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, scale_factor, pad_param, (H0, W0)


@torch.no_grad()
def extract_image_embedding_roi(
    model,
    img_tensor: torch.Tensor,
    bbox_xyxy_640: torch.Tensor,
    scales: List[int],
    roi_size: int = 7,
    bg_lambda: float = 0.0,
    bbox_xyxy_640_expanded: torch.Tensor = None,
) -> torch.Tensor:
    """NEW path: full-image forward + ROIAlign at FPN feature level.

    Args:
        model: full detector (frozen, eval mode)
        img_tensor: [1, 3, 640, 640] full letterbox image
        bbox_xyxy_640: [1, 4] (x1, y1, x2, y2) in letterbox coords
        scales: FPN scale indices (e.g. [0, 1, 2] for stride 8/16/32)
        roi_size: ROIAlign output spatial size
        bg_lambda: if > 0, subtract background ring contribution
        bbox_xyxy_640_expanded: required if bg_lambda > 0; bbox expanded for ring

    Returns:
        embedding: [embed_dim] (mean over scales, L2-norm NOT applied here —
                                downstream `mean across exemplars` + `F.normalize`
                                applied by caller)
    """
    img_feats = model.backbone.image_model(img_tensor)
    fpn_feats = model.neck(img_feats)
    head_module = model.bbox_head.head_module

    stride_by_scale = {0: 8, 1: 16, 2: 32}
    pooled_per_scale = []
    for i in scales:
        feat = head_module.cls_preds[i](fpn_feats[i])  # [1, D, H_s, W_s]
        stride = stride_by_scale[i]
        # ROIAlign at this scale (torchvision divides bbox coords by 1/spatial_scale
        # internally, so spatial_scale = 1/stride to convert image-coords → feature-coords)
        roi_feat = torchvision.ops.roi_align(
            feat,
            [bbox_xyxy_640.to(feat.device)],
            output_size=roi_size,
            spatial_scale=1.0 / stride,
            aligned=True,
        )  # [1, D, roi_size, roi_size]
        fg_feat = roi_feat.mean(dim=(2, 3)).squeeze(0)  # [D]

        if bg_lambda > 0:
            assert bbox_xyxy_640_expanded is not None, "bg_lambda > 0 needs expanded bbox"
            # Ring = expanded box features - inner box features
            ring_feat_outer = torchvision.ops.roi_align(
                feat,
                [bbox_xyxy_640_expanded.to(feat.device)],
                output_size=roi_size,
                spatial_scale=1.0 / stride,
                aligned=True,
            ).mean(dim=(2, 3)).squeeze(0)
            ring_feat = ring_feat_outer - fg_feat  # background-only feature (approx)
            fg_feat = fg_feat - bg_lambda * ring_feat

        pooled_per_scale.append(fg_feat)

    return torch.stack(pooled_per_scale, dim=0).mean(dim=0)  # [D]


def expand_bbox_xyxy(bbox_xyxy: torch.Tensor, factor: float, max_xy: Tuple[int, int]) -> torch.Tensor:
    """Expand bbox by `factor` around center, clipped to (max_x, max_y)."""
    x1, y1, x2, y2 = bbox_xyxy[0].tolist()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    ew, eh = w * factor, h * factor
    nx1 = max(0, cx - ew / 2)
    ny1 = max(0, cy - eh / 2)
    nx2 = min(max_xy[0], cx + ew / 2)
    ny2 = min(max_xy[1], cy + eh / 2)
    return torch.tensor([[nx1, ny1, nx2, ny2]], dtype=torch.float32)


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    cfg = Config.fromfile(args.config)
    ckpt = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)
    print(f"[init] loading {ckpt}")
    model = init_detector(args.config, ckpt, device=args.device)
    model.eval()

    ann_path = Path(args.ann_file)
    if not ann_path.is_absolute():
        ann_path = Path(args.data_root) / ann_path
    ann = json.loads(ann_path.read_text(encoding="utf-8"))

    cats = {c["id"]: c["name"] for c in ann["categories"]}
    images_by_id = {img["id"]: img for img in ann["images"]}
    by_cat: dict[int, list[dict]] = {cid: [] for cid in cats}
    for a in ann["annotations"]:
        by_cat[a["category_id"]].append(a)

    text_groups = json.loads(Path(args.text_json).read_text(encoding="utf-8"))
    if len(text_groups) != len(cats):
        raise SystemExit(
            f"prompt count {len(text_groups)} != ann categories {len(cats)}; "
            f"text_json must have one entry per category in cat_id order."
        )
    primary_keys = [grp[0] for grp in text_groups]
    cat_id_sorted = sorted(cats.keys())

    # Phase A: default to ROI path (geometry-correct). --legacy-crop opts back to old behavior.
    use_roi = not args.legacy_crop
    if use_roi:
        print(f"[mode] ROI path (KeepRatioResize + LetterResize + ROIAlign), "
              f"roi_size={args.roi_size}, bg_lambda={args.bg_lambda}, bg_expand={args.bg_expand}")
        pipeline = build_test_preprocessing_pipeline(args.input_size, args.pad_val)
    else:
        print(f"[mode] LEGACY path (cv2.resize + global spatial mean), expand={args.expand}")

    prototypes: dict[str, torch.Tensor] = {}
    used_anns: dict[str, list[int]] = {}  # ann_ids used per class (for hold-out)
    diag_per_class: dict[str, dict] = {}  # for --save-diag (Phase B prep)

    for idx, cid in enumerate(cat_id_sorted):
        cname = cats[cid]
        primary = primary_keys[idx]
        anns = by_cat[cid]
        if not anns:
            print(f"[skip] cat_id={cid} {cname!r}: 0 annotations")
            continue
        n = min(args.n_per_class, len(anns))
        chosen_idx = rng.choice(len(anns), size=n, replace=False)
        chosen = [anns[i] for i in chosen_idx]
        feats = []
        per_exemplar_meta = []  # for diag
        for a in chosen:
            img_info = images_by_id[a["image_id"]]
            file_name = img_info["file_name"]
            img_path = Path(args.data_root) / args.img_prefix / file_name
            if not img_path.is_file():
                # fallbacks
                img_path = Path(args.data_root) / file_name
                if not img_path.is_file():
                    img_path = ann_path.parent.parent / file_name

            try:
                if use_roi:
                    # NEW ROI path: full-image letterbox + ROIAlign at FPN level
                    img_tensor, scale_factor, pad_param, ori_shape = preprocess_image_via_pipeline(
                        pipeline, img_path
                    )
                    img_tensor = img_tensor.to(args.device)
                    # bbox xywh → letterbox xywh → xyxy
                    bx, by, bw, bh = transform_bbox_xywh(a["bbox"], scale_factor, pad_param)
                    if bw <= 1 or bh <= 1:
                        print(f"[warn] degenerate bbox for ann {a['id']}: ({bx:.1f}, {by:.1f}, "
                              f"{bw:.1f}, {bh:.1f}) — skipping")
                        continue
                    bbox_xyxy = torch.tensor([[bx, by, bx + bw, by + bh]], dtype=torch.float32)
                    # Phase A.2 fix: expand bbox by --roi-expand BEFORE ROIAlign.
                    # Cytology cells are small at native scale (~30-80px in 640×640 letterbox),
                    # so at stride 8 they're only ~4 feature cells. Legacy crop+resize+global
                    # mean implicitly magnified the cell to fill the receptive field.
                    # Expanding bbox before ROIAlign recovers that benefit while keeping
                    # letterbox geometry correct.
                    if args.roi_expand > 1.0:
                        bbox_xyxy = expand_bbox_xyxy(
                            bbox_xyxy, args.roi_expand, max_xy=(args.input_size, args.input_size)
                        )
                    # Optional bg expansion (ring) for context subtraction
                    bbox_expanded = None
                    if args.bg_lambda > 0:
                        bbox_expanded = expand_bbox_xyxy(
                            bbox_xyxy, args.bg_expand, max_xy=(args.input_size, args.input_size)
                        )
                    emb = extract_image_embedding_roi(
                        model,
                        img_tensor,
                        bbox_xyxy,
                        args.scales,
                        roi_size=args.roi_size,
                        bg_lambda=args.bg_lambda,
                        bbox_xyxy_640_expanded=bbox_expanded,
                    ).cpu()
                else:
                    # LEGACY path: bbox crop + cv2.resize + global mean
                    bgr = load_image_and_crop(
                        img_path, a["bbox"], args.expand, args.input_size
                    )
                    tensor = preprocess(bgr, args.device)
                    emb = extract_image_embedding(model, tensor, args.scales).cpu()
            except FileNotFoundError as e:
                print(f"[warn] missing image for ann {a['id']}: {e}")
                continue
            except RuntimeError as e:
                print(f"[warn] runtime error for ann {a['id']}: {e}")
                continue
            feats.append(emb)
            per_exemplar_meta.append({
                "ann_id": int(a["id"]),
                "image_id": int(a["image_id"]),
                "bbox_orig": [float(v) for v in a["bbox"]],
            })

        if not feats:
            print(f"[skip] {cname!r}: all exemplars failed")
            continue

        feat_stack = torch.stack(feats, dim=0).float()  # [N, D]
        proto = feat_stack.mean(dim=0).contiguous()  # [D]
        prototypes[primary] = proto
        used_anns[primary] = [m["ann_id"] for m in per_exemplar_meta]

        # Diagnostic stats: intra-class cosine
        feat_norm = F.normalize(feat_stack, dim=-1)
        intra_cos = (feat_norm @ feat_norm.T)
        # off-diag mean (consistency measure)
        N = len(feats)
        if N > 1:
            off_diag = (intra_cos.sum() - intra_cos.diag().sum()) / (N * (N - 1))
        else:
            off_diag = torch.tensor(1.0)

        diag_per_class[primary] = {
            "exemplars": per_exemplar_meta,
            "feats": feat_stack,  # [N, D] pre-normalize
            "intra_cos_mean_offdiag": float(off_diag),
            "intra_cos_matrix": intra_cos,
        }

        print(
            f"[ok] cat_id={cid} {cname!r}: {len(feats)}/{n} exemplars → "
            f"proto shape={tuple(proto.shape)}, intra_cos={off_diag:.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prototypes, out_path)
    print(f"\nWrote {out_path} with {len(prototypes)} class prototypes")

    holdout_path = out_path.with_suffix(".holdout_anns.json")
    holdout_path.write_text(json.dumps(used_anns, indent=2), encoding="utf-8")
    print(
        f"Wrote {holdout_path} (ann_ids used as exemplars; exclude from eval to "
        f"avoid leakage if doing strict zero-shot reporting)"
    )

    if args.save_diag:
        diag_path = out_path.with_suffix(".diag.pth")
        torch.save(diag_per_class, diag_path)
        print(f"Wrote {diag_path} (per-class per-exemplar feats + intra-cos for Phase B clustering)")


if __name__ == "__main__":
    main()
