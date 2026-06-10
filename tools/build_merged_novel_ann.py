"""Merge novel 4 splits into a single eval file with 9 unique novel classes.

Skips full_5 (= main_3 ∪ pseudo_2, would double-count).

Output category ids are reordered by organ then by source split, so per-organ
grouping is contiguous:

  organ=0 respiratory tract:
    0  respiratory tract-Squamous cell carcinoma   (main_3)
    1  respiratory tract-adenocarcinoma            (pseudo_2)
    2  respiratory tract-Small cell carcinoma      (hard_4)
  organ=1 Serous effusion:
    3  Serous effusion-Breast cancer               (main_3)
    4  Serous effusion-Ovarian cancer              (pseudo_2)
    5  Serous effusion-adenocarcinoma              (hard_4)
  organ=2 Thyroid gland:
    6  Thyroid gland-MTC                            (main_3)
    7  Thyroid gland-Suspicious for Malignancy     (hard_4)
    8  Thyroid gland-Malignant tumour              (hard_4)

Output: {data_root}/annotations/instances_test_novel_merged_9.json

Usage:
    python tools/build_merged_novel_ann.py \
        --ann-root "$TCT_NGC_DATA_ROOT/annotations/" \
        --taxonomy data/texts/tct_ngc_taxonomy.json \
        --out "$TCT_NGC_DATA_ROOT/annotations/instances_test_novel_merged_9.json"
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


SOURCES = [
    ('main_3', 'instances_test_main_novel.json'),
    ('pseudo_2', 'instances_test_pseudo_novel.json'),
    ('hard_4', 'instances_hard_test.json'),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ann-root', required=True, type=Path)
    p.add_argument('--taxonomy', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()

    tax = json.loads(args.taxonomy.read_text())
    classes_meta = tax['classes']
    organs = tax['organs']

    # -- Pass 1: collect all unique classes from sources + their organs --
    seen_classes = {}                       # name -> {source_split, original_cat_id}
    for split_name, fname in SOURCES:
        ann = json.loads((args.ann_root / fname).read_text())
        for cat in ann['categories']:
            name = cat['name']
            if name in seen_classes:
                continue                    # first source wins (deterministic)
            seen_classes[name] = {
                'source': split_name,
                'orig_id': cat['id'],
                'organ_id': classes_meta[name]['organ_id'],
            }

    # -- Order by (organ_id, source priority) --
    src_priority = {'main_3': 0, 'pseudo_2': 1, 'hard_4': 2}
    ordered = sorted(
        seen_classes.items(),
        key=lambda kv: (kv[1]['organ_id'], src_priority[kv[1]['source']], kv[0]),
    )
    name_to_new_id = {name: i for i, (name, _) in enumerate(ordered)}

    new_categories = []
    for i, (name, meta) in enumerate(ordered):
        new_categories.append({
            'id': i,
            'name': name,
            'organ': organs[meta['organ_id']],
            'organ_id': meta['organ_id'],
            'source_split': meta['source'],
        })

    # -- Pass 2: merge images + remap annotations --
    images_by_filename = {}                 # file_name -> image dict (dedup)
    new_anns = []
    seen_ann_ids = set()
    next_img_id = 1
    next_ann_id = 1
    dropped_anns = 0

    for split_name, fname in SOURCES:
        ann = json.loads((args.ann_root / fname).read_text())

        # Build maps from this source: orig_cat_id -> new_id ; orig_img_id -> new_img_id
        cat_id_to_new = {}
        for c in ann['categories']:
            if c['name'] in name_to_new_id:
                cat_id_to_new[c['id']] = name_to_new_id[c['name']]

        img_id_to_new = {}
        for img in ann['images']:
            fn = img['file_name']
            if fn in images_by_filename:
                # already seen via another split → reuse new id
                img_id_to_new[img['id']] = images_by_filename[fn]['id']
                continue
            new_img = dict(img)
            new_img['id'] = next_img_id
            images_by_filename[fn] = new_img
            img_id_to_new[img['id']] = next_img_id
            next_img_id += 1

        for a in ann['annotations']:
            new_cat = cat_id_to_new.get(a['category_id'])
            if new_cat is None:
                # cat not in our 9-novel union (shouldn't happen but defensive)
                dropped_anns += 1
                continue
            new_a = dict(a)
            new_a['id'] = next_ann_id
            new_a['image_id'] = img_id_to_new[a['image_id']]
            new_a['category_id'] = new_cat
            new_anns.append(new_a)
            next_ann_id += 1

    merged_images = list(images_by_filename.values())

    out_ann = {
        'info': {'description': 'TCT_NGC merged novel (9 unique classes)',
                 'sources': [s[1] for s in SOURCES]},
        'categories': new_categories,
        'images': merged_images,
        'annotations': new_anns,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_ann))

    # -- Report --
    print(f"saved: {args.out}")
    print(f"  categories: {len(new_categories)}")
    # Single linear pass over annotations to build per-class counts.
    from collections import Counter
    inst_per_cat = Counter(a['category_id'] for a in new_anns)
    imgs_per_cat = defaultdict(set)
    for a in new_anns:
        imgs_per_cat[a['category_id']].add(a['image_id'])
    for c in new_categories:
        n_inst = inst_per_cat[c['id']]
        n_img = len(imgs_per_cat[c['id']])
        print(f"    {c['id']:2d} organ={c['organ_id']} {c['name']:50s}  src={c['source_split']:9s}  imgs={n_img:5d} insts={n_inst:5d}")
    print(f"  images:      {len(merged_images)}")
    print(f"  annotations: {len(new_anns)}")
    if dropped_anns:
        print(f"  dropped:     {dropped_anns} (cat not in union)")
    # per-organ breakdown
    print(f"  per-organ:")
    by_organ = defaultdict(lambda: {'imgs': set(), 'insts': 0, 'classes': 0})
    for c in new_categories:
        by_organ[c['organ_id']]['classes'] += 1
    for a in new_anns:
        oid = next(c['organ_id'] for c in new_categories if c['id'] == a['category_id'])
        by_organ[oid]['imgs'].add(a['image_id'])
        by_organ[oid]['insts'] += 1
    for oid in sorted(by_organ):
        b = by_organ[oid]
        print(f"    organ={oid} {organs[oid]:24s}  classes={b['classes']}  imgs={len(b['imgs']):5d}  insts={b['insts']:5d}")


if __name__ == '__main__':
    main()
