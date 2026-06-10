"""Build the 1-PSC novel_merged class metadata for ICF+1-PSC paper eval.

Sister of tools/build_class_metadata_1psc.py (base30 → base30_1psc). Same
pattern: inherit everything except attr_emb from the 5-attr metadata, swap
attr_emb to the 1-PSC BiomedCLIP embeddings (stacked from the per-prompt
dict in dict-iteration order, which the upstream emb-builder guarantees
to match class_id order).

Output:
    data/texts/tct_ngc_class_metadata_novel_merged_1psc.pt

Schema (matches PseudoMultiAttrLanguageBackbone with num_attrs=1):
    {
        'class_names':      list[str] (9, novel_merged order),
        'class_ids':        list[int],
        'organ_ids':        LongTensor [9],
        'axis_ids':         LongTensor [9],
        'rank_along_axis':  LongTensor [9],
        'system_ids':       LongTensor [9],
        'attr_emb':         FloatTensor [9, 1, 512],
    }

Inputs:
    data/texts/tct_ngc_class_metadata_novel_merged.pt   (5-attr metadata,
        for inheriting all non-attr_emb fields)
    data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth (1-PSC dict,
        keyed by full PSC prompt string)

Usage:
    PYTHONPATH=. python tools/build_class_metadata_novel_merged_1psc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
META_5ATTR = REPO / "data/texts/tct_ngc_class_metadata_novel_merged.pt"
EMB_1PSC = REPO / "data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth"
OUT_1PSC_META = REPO / "data/texts/tct_ngc_class_metadata_novel_merged_1psc.pt"


def main() -> int:
    if not META_5ATTR.is_file():
        print(f"missing {META_5ATTR}", file=sys.stderr)
        return 1
    if not EMB_1PSC.is_file():
        print(f"missing {EMB_1PSC}", file=sys.stderr)
        return 1

    m5 = torch.load(META_5ATTR, map_location="cpu", weights_only=False)
    e1 = torch.load(EMB_1PSC, map_location="cpu", weights_only=False)

    if not isinstance(e1, dict):
        print(f"{EMB_1PSC} is not a dict-of-tensors", file=sys.stderr)
        return 1

    psc = torch.stack(list(e1.values())).float()  # [9, 512]
    if psc.shape[0] != len(m5["class_names"]):
        print(
            f"class count mismatch: 5-attr metadata has "
            f"{len(m5['class_names'])} entries, 1-PSC dict has {psc.shape[0]}",
            file=sys.stderr,
        )
        return 1

    out = dict(m5)
    out["attr_emb"] = psc.unsqueeze(1)  # [9, 1, 512]
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
