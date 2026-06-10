"""Build morph6 class_metadata .pt for non-degenerate ICF training.

Reuses organ_ids/axis_ids/rank_along_axis/system_ids from the existing 1psc
metadata (those are organ-structural and text-content-independent), and only
replaces attr_emb with freshly BiomedCLIP-encoded 6-axis morph6 attributes.

Safety locks:
- Iterates official COCO category order from ann file.
- Asserts each morph6 entry's `official_label` matches both COCO cat name
  AND the existing 1psc metadata's class_names at the same index.

Usage:
  python tools/build_class_metadata_morph6.py \
    --morph6-json data/texts/tct_ngc_attr_morph6_base30.json \
    --donor-meta  data/texts/tct_ngc_class_metadata_base30_1psc.pt \
    --ann         "$TCT_NGC_DATA_ROOT/annotations/instances_test_base_clean_dev30.json" \
    --out         data/texts/tct_ngc_class_metadata_base30_morph6.pt

  python tools/build_class_metadata_morph6.py \
    --morph6-json data/texts/tct_ngc_attr_morph6_novel9.json \
    --donor-meta  data/texts/tct_ngc_class_metadata_novel_merged_1psc.pt \
    --ann         "$TCT_NGC_DATA_ROOT/annotations/instances_test_novel_merged_9.json" \
    --out         data/texts/tct_ngc_class_metadata_novel_merged_morph6.pt
"""
import argparse
import json
from pathlib import Path

# Pre-import torch + monkey-patch for transformers/torch compat in this env
import torch.utils._pytree as _pytree
if not hasattr(_pytree, 'register_pytree_node'):
    import functools
    _old = _pytree._register_pytree_node
    @functools.wraps(_old)
    def _wrap(t, f, u, *a, **kw): return _old(t, f, u)
    _pytree.register_pytree_node = _wrap

import torch

MORPH6_AXES = (
    'cell_size_shape',
    'nuclear_size_shape',
    'nuclear_membrane_contour',
    'chromatin_nucleoli',
    'nc_ratio_cytoplasm',
    'architecture_arrangement',
)
A = len(MORPH6_AXES)  # 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--morph6-json', required=True, type=Path,
                   help='morph6 split JSON (base30 or novel9)')
    p.add_argument('--donor-meta', required=True, type=Path,
                   help='existing 1psc class_metadata.pt to inherit '
                        'organ/axis/system/rank fields from')
    p.add_argument('--ann', required=True, type=Path,
                   help='COCO ann file for safety-lock alignment check')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--biomedclip-name',
                   default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    # ----- load inputs + safety alignment -----
    morph6 = json.loads(args.morph6_json.read_text())
    donor = torch.load(args.donor_meta, weights_only=False, map_location='cpu')
    coco_cats = sorted(json.loads(args.ann.read_text())['categories'],
                       key=lambda c: c['id'])
    C = len(morph6)
    assert C == len(donor['class_names']), \
        f"morph6 len {C} != donor len {len(donor['class_names'])}"
    assert C == len(coco_cats), f"morph6 len {C} != COCO cat count {len(coco_cats)}"

    # Triple-alignment safety lock
    for i in range(C):
        m6_label = morph6[i]['official_label']
        donor_label = donor['class_names'][i]
        coco_label = coco_cats[i]['name']
        if not (m6_label == donor_label == coco_label):
            raise SystemExit(
                f"alignment mismatch at idx {i}:\n"
                f"  morph6.official_label = {m6_label!r}\n"
                f"  donor.class_names     = {donor_label!r}\n"
                f"  coco_cats.name        = {coco_label!r}"
            )
    print(f"[align] all {C} entries triple-aligned (morph6 == donor == COCO)")

    # ----- BiomedCLIP encode 6×C attribute strings -----
    print(f"[encode] loading {args.biomedclip_name}")
    import open_clip
    model, _ = open_clip.create_model_from_pretrained(args.biomedclip_name)
    tokenizer = open_clip.get_tokenizer(args.biomedclip_name)
    model = model.to(args.device).eval()

    flat = []
    for entry in morph6:
        for axis in MORPH6_AXES:
            flat.append(entry['attrs'][axis])
    assert len(flat) == C * A, f"{len(flat)} != {C*A}"
    print(f"[encode] encoding {len(flat)} strings ({C} classes × {A} axes)")

    tokens = tokenizer(flat).to(args.device)
    with torch.no_grad():
        embs = model.encode_text(tokens)
    # L2-normalize per-attribute (standard for cosine-sim contrastive heads)
    embs = embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    embs = embs.detach().cpu().float()
    D = embs.shape[-1]
    assert embs.shape == (C * A, D), f"emb shape {tuple(embs.shape)}"

    attr_emb = embs.reshape(C, A, D).contiguous()  # [C, 6, 512]
    print(f"[encode] attr_emb shape = {tuple(attr_emb.shape)}, dtype = {attr_emb.dtype}")

    # ----- compose output: inherit donor structural fields, replace attr_emb -----
    out = {
        'class_names':     list(donor['class_names']),
        'class_ids':       list(donor['class_ids']),
        'organ_ids':       donor['organ_ids'].clone(),
        'axis_ids':        donor['axis_ids'].clone(),
        'rank_along_axis': donor['rank_along_axis'].clone(),
        'system_ids':      donor['system_ids'].clone(),
        'attr_emb':        attr_emb,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    sz = args.out.stat().st_size / 1024
    print(f"[save] {args.out}  ({sz:.1f} KB,  C={C}  A={A}  D={D})")


if __name__ == '__main__':
    main()
