"""Build per-class metadata + 5-attr emb tensor for Module 2 adapter.

Output: dict saved to .pt
  {
    'class_names':   [str] × C    (sorted by ann cat_id)
    'class_ids':     [int] × C    (matching ann cat_id)
    'organ_ids':    LongTensor[C]
    'axis_ids':     LongTensor[C]     (0 = primary, cervical: 0/1/2/3)
    'rank_along_axis': LongTensor[C]  (-1 if unknown)
    'attr_emb':     FloatTensor[C, A=5, D=512]   BiomedCLIP per-attr cached
    'system_ids':   LongTensor[C]     (0=PSC, 1=Bethesda, 2=TIS, 3=Paris, 4=CervBethesda)
  }

Usage:
  # Base 30:
  python tools/build_class_metadata_tensor.py \\
      --taxonomy data/texts/tct_ngc_taxonomy.json \\
      --attrs-json data/texts/tct_ngc_fullnames_30_attr_train.json \\
      --attr-cache data/texts/tct_ngc_attr_biomedclip_per_attr.pth \\
      --ann "$TCT_NGC_DATA_ROOT/annotations/instances_test_base_clean_dev30.json" \\
      --out data/texts/tct_ngc_class_metadata_base30.pt

  # Novel 9 merged:
  python tools/build_class_metadata_tensor.py \\
      --taxonomy data/texts/tct_ngc_taxonomy.json \\
      --attrs-json data/texts/tct_ngc_attr_main_3_eval.json \\
                   data/texts/tct_ngc_attr_pseudo_2_eval.json \\
                   data/texts/tct_ngc_attr_hard_4_eval.json \\
      --ann "$TCT_NGC_DATA_ROOT/annotations/instances_test_novel_merged_9.json" \\
      --attr-cache data/texts/tct_ngc_attr_biomedclip_per_attr.pth \\
      --out data/texts/tct_ngc_class_metadata_novel_merged.pt
"""
import argparse
import json
from pathlib import Path

import torch

SYSTEM_TO_ID = {
    "PSC": 0,
    "Bethesda": 1,
    "TIS": 2,
    "Paris": 3,
    "CervBethesda": 4,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--taxonomy', required=True, type=Path)
    p.add_argument('--attrs-json', nargs='+', required=True, type=Path,
                   help='5-attr JSON file(s); for novel, pass multiple')
    p.add_argument('--attr-cache', required=True, type=Path)
    p.add_argument('--ann', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()

    tax = json.loads(args.taxonomy.read_text())
    cls_meta = tax['classes']
    attr_cache = torch.load(args.attr_cache, weights_only=False, map_location='cpu')

    ann = json.loads(args.ann.read_text())
    sorted_cats = sorted(ann['categories'], key=lambda c: c['id'])
    class_names = [c['name'] for c in sorted_cats]
    class_ids = [c['id'] for c in sorted_cats]
    C = len(class_names)
    sample_emb = next(iter(attr_cache.values()))
    D = sample_emb.shape[-1]
    A = 5

    # Load all attr JSONs and build name -> [5 prompts] mapping.
    name_to_prompts = {}
    novel_prompts_lists = []
    for attrs_path in args.attrs_json:
        attrs_data = json.loads(attrs_path.read_text())  # list of [5 prompts]
        novel_prompts_lists.append(attrs_data)

    # Try to match by name via ann order if single attr JSON, else by source-of-class
    if len(args.attrs_json) == 1:
        # Single file: should match ann.categories order 1:1
        single = novel_prompts_lists[0]
        if len(single) != C:
            raise SystemExit(
                f'attrs JSON has {len(single)} entries but ann has {C} categories; '
                f'need an attrs JSON per category in cat_id order.')
        for name, prompts in zip(class_names, single):
            name_to_prompts[name] = prompts
    else:
        # Multiple files (novel union): match by class name from source-split mapping
        # via taxonomy's split field.
        for attrs_path, prompts_list in zip(args.attrs_json, novel_prompts_lists):
            # Determine which classes this attrs file covers via filename heuristic
            stem = attrs_path.stem  # e.g., tct_ngc_attr_main_3_eval
            if 'main_3' in stem:
                split_tag = 'novel_main_3'
            elif 'pseudo_2' in stem:
                split_tag = 'novel_pseudo_2'
            elif 'hard_4' in stem:
                split_tag = 'novel_hard_4'
            elif 'base30' in stem:
                split_tag = 'base30'
            else:
                raise SystemExit(f'cannot infer split from filename: {attrs_path}')
            split_classes = [n for n, m in cls_meta.items() if m['split'] == split_tag]
            split_classes.sort(key=lambda n: cls_meta[n]['class_id'])
            if len(split_classes) != len(prompts_list):
                raise SystemExit(
                    f'attrs file {attrs_path} has {len(prompts_list)} entries but '
                    f'taxonomy has {len(split_classes)} classes for split {split_tag}')
            for name, prompts in zip(split_classes, prompts_list):
                name_to_prompts[name] = prompts

    # Build all output tensors in ann-sorted order
    organ_ids = torch.zeros(C, dtype=torch.long)
    axis_ids = torch.zeros(C, dtype=torch.long)
    rank_along_axis = torch.zeros(C, dtype=torch.long)
    system_ids = torch.zeros(C, dtype=torch.long)
    attr_emb = torch.zeros(C, A, D, dtype=torch.float32)

    missing_classes = []
    missing_prompts = []
    for i, name in enumerate(class_names):
        if name not in cls_meta:
            missing_classes.append(name)
            continue
        m = cls_meta[name]
        organ_ids[i] = m['organ_id']
        axis_ids[i] = m.get('axis_id', 0)
        # NOT `or -1` — that would map legitimate rank=0 (cervical NILM /
        # adequacy) to -1 via falsy collapse. Use bare get with default.
        rank_val = m.get('rank_along_axis')
        rank_along_axis[i] = rank_val if rank_val is not None else -1
        sys_name = m.get('system') or 'CervBethesda'  # cervical parser fails
        if name.startswith('TCT_CCD-'):
            sys_name = 'CervBethesda'
        system_ids[i] = SYSTEM_TO_ID.get(sys_name, 4)  # default CervBethesda for unknown

        if name not in name_to_prompts:
            missing_classes.append(name)
            continue
        prompts = name_to_prompts[name]
        if len(prompts) < A:
            raise SystemExit(f'class {name} has {len(prompts)} attrs, expected {A}')
        for a, prompt in enumerate(prompts[:A]):
            if prompt not in attr_cache:
                missing_prompts.append((name, a, prompt[:60]))
                continue
            attr_emb[i, a] = attr_cache[prompt]

    if missing_classes:
        raise SystemExit(f'missing taxonomy/prompts for classes: {missing_classes}')
    if missing_prompts:
        print(f'WARN: {len(missing_prompts)} prompts not in attr cache (will be zero-vector):')
        for name, a, p in missing_prompts[:5]:
            print(f'  {name} attr {a}: {p!r}')

    out = {
        'class_names': class_names,
        'class_ids': class_ids,
        'organ_ids': organ_ids,
        'axis_ids': axis_ids,
        'rank_along_axis': rank_along_axis,
        'system_ids': system_ids,
        'attr_emb': attr_emb,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)

    print(f'saved: {args.out}')
    print(f'  C={C} classes  A={A} attrs  D={D} dim')
    print(f'  attr_emb shape: {tuple(attr_emb.shape)}  norm-mean: {attr_emb.norm(dim=-1).mean().item():.3f}')
    print(f'  organ_ids unique: {organ_ids.unique().tolist()}')
    print(f'  axis_ids unique: {axis_ids.unique().tolist()}')
    print(f'  system_ids unique: {system_ids.unique().tolist()}')


if __name__ == '__main__':
    main()
