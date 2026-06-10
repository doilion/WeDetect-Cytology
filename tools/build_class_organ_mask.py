"""Build class×organ valid mask for organ-restricted inference / loss masking.

Given a COCO annotation file and the global taxonomy, produce a mask file:
    {
        'class_names':  [str, ...]      # ordered as ann.categories
        'class_ids':    [int, ...]      # COCO category_id
        'organ_names':  [str, ...]      # ordered as taxonomy.organs
        'organ_to_id':  {name -> int}
        'mask':         FloatTensor[C, O]   # mask[c, o] = 1 if classes[c].organ_id == o
    }

Usage:
    python tools/build_class_organ_mask.py \
        --ann "$TCT_NGC_DATA_ROOT/annotations/instances_test_base_clean_dev30.json" \
        --taxonomy data/texts/tct_ngc_taxonomy.json \
        --out data/texts/tct_ngc_class_organ_mask_base30.pt

    python tools/build_class_organ_mask.py \
        --ann "$TCT_NGC_DATA_ROOT/annotations/instances_test_novel_merged_9.json" \
        --taxonomy data/texts/tct_ngc_taxonomy.json \
        --out data/texts/tct_ngc_class_organ_mask_novel_merged.pt
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ann', required=True, type=Path)
    p.add_argument('--taxonomy', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()

    ann = json.loads(args.ann.read_text())
    tax = json.loads(args.taxonomy.read_text())

    organs = tax['organs']                     # ordered list of organ names
    organ_to_id = tax['organ_to_id']           # name -> int
    classes_meta = tax['classes']              # name -> {organ_id, ...}
    O = len(organs)

    # Sort by category_id to match COCOeval / eval_organ_restricted.py convention.
    sorted_cats = sorted(ann['categories'], key=lambda c: c['id'])
    class_names = [c['name'] for c in sorted_cats]
    class_ids = [c['id'] for c in sorted_cats]
    C = len(class_names)

    mask = torch.zeros(C, O, dtype=torch.float32)
    unmapped = []
    for i, name in enumerate(class_names):
        if name not in classes_meta:
            unmapped.append(name)
            continue
        o = classes_meta[name].get('organ_id')
        if o is None:
            unmapped.append(name)
            continue
        mask[i, o] = 1.0

    if unmapped:
        raise SystemExit(
            f"taxonomy missing organ_id for {len(unmapped)} classes:\n  "
            + "\n  ".join(unmapped))

    # sanity: every class assigned to exactly 1 organ
    assert torch.all(mask.sum(dim=1) == 1.0), 'each class must map to exactly one organ'

    out = {
        'class_names': class_names,
        'class_ids': class_ids,
        'organ_names': organs,
        'organ_to_id': organ_to_id,
        'mask': mask,                          # [C, O] float {0, 1}
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)

    # human-readable report
    print(f"saved: {args.out}")
    print(f"  C={C} classes  O={O} organs  shape={tuple(mask.shape)}")
    per_organ = mask.sum(dim=0).int().tolist()
    for name, n in zip(organs, per_organ):
        print(f"  {name:24s}  {n} classes")


if __name__ == '__main__':
    main()
