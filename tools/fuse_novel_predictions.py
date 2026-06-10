#!/usr/bin/env python
"""Post-hoc score fusion of two independent novel-zero-shot inference passes.

Avoids the geometric mismatch issue of binary class-level embedding fusion
(see docs/tct_ngc_novel_zero_shot_review_20260509.md §2.3): instead of
mixing text and visual prototype vectors in one inference, we run two
SEPARATE inferences — one with text vectors, one with visual prototypes —
then merge predictions PER CLASS at the detection level.

Per-class routing rule (default; can be overridden via --visproto-classes):
- "Serous effusion-*" → keep text predictions (text already aligns well)
- "respiratory tract-*" / "Thyroid gland-*" / "Bethesda*" → keep visproto preds
- (Urine-* / TCT_CCD-*: text by default; not used in current novel splits)

Pipeline:
  1. Load both bbox.json prediction files
  2. Map each detection's category_id to its primary class name (via novel JSON)
  3. Drop detections whose category does not match the configured source
  4. Concatenate the surviving detections from both sources
  5. Run pycocotools COCO eval on merged predictions vs ground-truth ann file
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--text-preds", required=True, help="bbox.json from text-only eval")
    p.add_argument(
        "--vis-preds", required=True, help="bbox.json from visual prototype eval"
    )
    p.add_argument("--gt-ann", required=True, help="novel ground-truth ann file")
    p.add_argument(
        "--text-source-pattern",
        default=r"(Serous|breast|ovarian)",
        help="regex on primary prompt; matches → keep text prediction for that class",
    )
    p.add_argument("--out", required=True, help="merged bbox.json path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gt = COCO(args.gt_ann)
    cat_id_to_name: dict[int, str] = {c["id"]: c["name"] for c in gt.dataset["categories"]}

    text_pat = re.compile(args.text_source_pattern, re.IGNORECASE)
    text_cat_ids = {
        cid for cid, name in cat_id_to_name.items() if text_pat.search(name)
    }
    vis_cat_ids = set(cat_id_to_name) - text_cat_ids
    print(
        f"[fuse] text classes ({len(text_cat_ids)}): "
        f"{sorted(cat_id_to_name[c] for c in text_cat_ids)}"
    )
    print(
        f"[fuse] visproto classes ({len(vis_cat_ids)}): "
        f"{sorted(cat_id_to_name[c] for c in vis_cat_ids)}"
    )

    text_preds = json.loads(Path(args.text_preds).read_text(encoding="utf-8"))
    vis_preds = json.loads(Path(args.vis_preds).read_text(encoding="utf-8"))

    text_kept = [p for p in text_preds if p["category_id"] in text_cat_ids]
    vis_kept = [p for p in vis_preds if p["category_id"] in vis_cat_ids]
    merged = text_kept + vis_kept
    print(
        f"[fuse] kept {len(text_kept)} text preds + {len(vis_kept)} visproto preds "
        f"= {len(merged)} total (text in: {len(text_preds)}, visproto in: {len(vis_preds)})"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged), encoding="utf-8")
    print(f"[fuse] wrote {out_path}")

    if not merged:
        print("[fuse] empty merged preds — skipping eval")
        return

    dt = gt.loadRes(str(out_path))
    e = COCOeval(gt, dt, "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()

    # Per-class AP
    print("\n[fuse] per-class mAP (IoU=0.5:0.95):")
    precisions = e.eval["precision"]  # [10, 101, K, 4, 3]
    for k, cid in enumerate(sorted(cat_id_to_name)):
        cls_p = precisions[:, :, k, 0, -1]
        valid = cls_p[cls_p > -1]
        ap = float(valid.mean()) if valid.size else float("nan")
        src = "text" if cid in text_cat_ids else "visproto"
        print(f"  cat_id={cid:>2}  AP={ap:.3f}  [{src:<8}]  {cat_id_to_name[cid]}")


if __name__ == "__main__":
    main()
