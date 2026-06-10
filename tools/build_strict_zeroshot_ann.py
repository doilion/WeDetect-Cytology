#!/usr/bin/env python
"""Build strict zero-shot novel eval ann files by excluding exemplar images.

The 5-shot visual prototype build (`tools/build_visual_prototype.py`) uses
5 GT bboxes per novel class from the test set. The `*.holdout_anns.json`
records those ann_ids. For strict zero-shot reporting, we must exclude
those exemplar images entirely from eval (not just the specific bboxes),
because the prototype "saw" those images during construction.

Output: filtered COCO-format ann JSONs at the same directory as input,
named `<basename>.strict_zeroshot.json`.

Usage:
    python tools/build_strict_zeroshot_ann.py \
        --ann-file "$TCT_NGC_DATA_ROOT/annotations/instances_test_main_novel.json" \
        --holdout data/texts/tct_ngc_novel_main_3_visproto_emb_biomedclip_noTHAF.holdout_anns.json \
        --out /tmp/instances_test_main_novel_strict_main_3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ann-file", required=True)
    p.add_argument("--holdout", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ann = json.loads(Path(args.ann_file).read_text(encoding="utf-8"))
    holdout = json.loads(Path(args.holdout).read_text(encoding="utf-8"))

    # Collect all exemplar ann_ids across all novel classes
    exemplar_ann_ids: set[int] = set()
    for cname, ids in holdout.items():
        exemplar_ann_ids.update(int(x) for x in ids)
    print(f"[strict] {len(exemplar_ann_ids)} exemplar ann_ids across {len(holdout)} classes")

    # Find image_ids those exemplars live in
    ann_id_to_image_id = {a["id"]: a["image_id"] for a in ann["annotations"]}
    exemplar_image_ids: set[int] = set()
    missing_anns = 0
    for aid in exemplar_ann_ids:
        if aid in ann_id_to_image_id:
            exemplar_image_ids.add(ann_id_to_image_id[aid])
        else:
            missing_anns += 1
    if missing_anns:
        print(f"[warn] {missing_anns} exemplar ann_ids not found in ann_file")
    print(f"[strict] {len(exemplar_image_ids)} distinct exemplar images")

    # Filter
    orig_images = len(ann["images"])
    orig_anns = len(ann["annotations"])
    ann["images"] = [im for im in ann["images"] if im["id"] not in exemplar_image_ids]
    ann["annotations"] = [a for a in ann["annotations"] if a["image_id"] not in exemplar_image_ids]
    print(f"[strict] images {orig_images} → {len(ann['images'])} (removed {orig_images-len(ann['images'])})")
    print(f"[strict] anns   {orig_anns} → {len(ann['annotations'])} (removed {orig_anns-len(ann['annotations'])})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ann), encoding="utf-8")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
