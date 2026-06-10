"""Build the 1-PSC class metadata file used by the ICF + 1-PSC ablation.

Output:
    data/texts/tct_ngc_class_metadata_base30_1psc.pt

Schema (matches PseudoMultiAttrLanguageBackbone with num_attrs=1):
    {
        'class_names':      list[str] (30, base30 cat_id order),
        'class_ids':        list[int],
        'organ_ids':        LongTensor [30],
        'axis_ids':         LongTensor [30],
        'rank_along_axis':  LongTensor [30],
        'system_ids':       LongTensor [30],
        'attr_emb':         FloatTensor [30, 1, 512]   # 1-PSC single-prompt emb stacked
    }

Inputs (already tracked in repo / regeneratable):
    data/texts/tct_ngc_class_metadata_base30.pt        — 5-attr metadata (.pt cache,
        rebuild via tools/build_class_metadata_tensor.py if missing)
    data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth — 1-PSC text emb dict
        keyed by prompt string (e.g. "Respiratory tract cytology - Neutrophil"),
        512-d L2-normalized vector

The 5-attr metadata is loaded only to inherit class_names / class_ids /
organ_ids / axis_ids / rank_along_axis / system_ids (everything except
attr_emb). The 1-PSC attr_emb is built by stacking values() of the 1-PSC
embeddings dict in dict-iteration order — order is verified to match the
metadata's cat_id order by tools/build_class_metadata_tensor.py and the
upstream prompt-emb builder.

Usage:
    PYTHONPATH=. python tools/build_class_metadata_1psc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
META_5ATTR = REPO / "data/texts/tct_ngc_class_metadata_base30.pt"
EMB_1PSC = REPO / "data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth"
OUT_1PSC_META = REPO / "data/texts/tct_ngc_class_metadata_base30_1psc.pt"


def main() -> int:
    if not META_5ATTR.is_file():
        print(
            f"missing {META_5ATTR}; rebuild it first via "
            f"tools/build_class_metadata_tensor.py",
            file=sys.stderr,
        )
        return 1
    if not EMB_1PSC.is_file():
        print(
            f"missing {EMB_1PSC}; rebuild it via "
            f"tools/build_biomedclip_text_embeddings.py",
            file=sys.stderr,
        )
        return 1

    m5 = torch.load(META_5ATTR, map_location="cpu", weights_only=False)
    e1 = torch.load(EMB_1PSC, map_location="cpu", weights_only=False)

    if not isinstance(e1, dict):
        print(f"{EMB_1PSC} is not a dict-of-tensors", file=sys.stderr)
        return 1

    psc = torch.stack([t for t in e1.values()]).float()  # [30, 512]
    if psc.shape[0] != len(m5["class_names"]):
        print(
            f"class count mismatch: 5-attr metadata has "
            f"{len(m5['class_names'])} entries, 1-PSC dict has {psc.shape[0]}",
            file=sys.stderr,
        )
        return 1

    out = {
        "class_names": m5["class_names"],
        "class_ids": m5["class_ids"],
        "organ_ids": m5["organ_ids"],
        "axis_ids": m5["axis_ids"],
        "rank_along_axis": m5["rank_along_axis"],
        "system_ids": m5["system_ids"],
        "attr_emb": psc.unsqueeze(1),  # [30, 1, 512]
    }
    torch.save(out, OUT_1PSC_META)

    norms = out["attr_emb"].norm(dim=-1)
    print(f"wrote {OUT_1PSC_META}")
    print(f"  attr_emb shape: {tuple(out['attr_emb'].shape)}")
    print(
        f"  attr_emb norm: min={norms.min():.4f} max={norms.max():.4f} "
        f"mean={norms.mean():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
