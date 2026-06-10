#!/usr/bin/env python
"""Build a morph6 per-attribute text cache in the format the attribute head wants.

The attribute head (``AttributeBNContrastiveHead`` via ``head_module.attr_fusion.
attr_text_path``) loads ``pkg['emb']`` = a ``[C, 6, 512]`` tensor of per-class,
per-axis BiomedCLIP text embeddings (see yolo_world_head.py _init_layers). The
base-split file ``tct_ngc_morph6_30_per_attr_biomedclip.pth`` already ships; this
tool builds the matching NOVEL-split file so novel zero-shot eval has a 9-class
per-attr text in merged_9 category order.

Encoding mirrors tools/build_class_metadata_morph6.py EXACTLY (same BiomedCLIP
model, same 6 morph axes, same per-vector L2 norm) so base + novel embeddings
live in one comparable space.

Usage (novel 9-class, merged_9 order):
  PYTHONPATH=. python tools/build_morph6_per_attr.py \
    --morph6-json data/texts/tct_ngc_attr_morph6_novel9.json \
    --out data/texts/tct_ngc_morph6_novel9_per_attr_biomedclip.pth
"""
import argparse
import json
from pathlib import Path

# transformers/torch pytree compat shim for this env (same as the metadata builder)
import torch.utils._pytree as _pytree
if not hasattr(_pytree, "register_pytree_node"):
    import functools
    _old = _pytree._register_pytree_node

    @functools.wraps(_old)
    def _wrap(t, f, u, *a, **kw):
        return _old(t, f, u)

    _pytree.register_pytree_node = _wrap

import torch

MORPH6_AXES = (
    "cell_size_shape",
    "nuclear_size_shape",
    "nuclear_membrane_contour",
    "chromatin_nucleoli",
    "nc_ratio_cytoplasm",
    "architecture_arrangement",
)
A = len(MORPH6_AXES)  # 6


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--morph6-json", required=True, type=Path,
                   help="morph6 split JSON: list of {official_label, attrs:{...6 axes}}")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--biomedclip-name",
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )
    p.add_argument("--device", default="cuda:0",
                   help="cuda:N or cpu (54 short strings -> cpu is fine)")
    args = p.parse_args()

    morph6 = json.loads(args.morph6_json.read_text(encoding="utf-8"))
    C = len(morph6)
    labels = [e["official_label"] for e in morph6]

    # flatten C classes x 6 axes, in axis order, with a presence check
    flat = []
    for entry in morph6:
        for axis in MORPH6_AXES:
            v = entry["attrs"].get(axis, "").strip()
            if not v:
                raise SystemExit(f"empty axis {axis!r} for class {entry['official_label']!r}")
            flat.append(v)
    assert len(flat) == C * A, f"{len(flat)} != {C * A}"

    print(f"[encode] loading {args.biomedclip_name}")
    import open_clip

    model, _ = open_clip.create_model_from_pretrained(args.biomedclip_name)
    tokenizer = open_clip.get_tokenizer(args.biomedclip_name)
    model = model.to(args.device).eval()

    print(f"[encode] {len(flat)} strings ({C} classes x {A} axes)")
    tokens = tokenizer(flat).to(args.device)
    with torch.no_grad():
        embs = model.encode_text(tokens)
    embs = embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-9)  # per-vector L2
    embs = embs.detach().cpu().float()
    D = embs.shape[-1]
    assert embs.shape == (C * A, D), f"emb shape {tuple(embs.shape)}"

    emb = embs.reshape(C, A, D).contiguous()  # [C, 6, 512]
    out = {"emb": emb, "attr_keys": list(MORPH6_AXES), "labels": labels}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[save] {args.out}  (emb={tuple(emb.shape)} D={D}, {C} classes)")
    print(f"[labels] {labels}")


if __name__ == "__main__":
    main()
