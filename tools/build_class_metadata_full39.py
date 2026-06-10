"""Build the merged 39-class metadata file for the STEGO-corr loss.

Merges:
    data/texts/tct_ngc_class_metadata_base30.pt        (30 base classes)
    data/texts/tct_ngc_class_metadata_novel_merged.pt  (9 novel classes)

into:
    data/texts/tct_ngc_class_metadata_full39.pt        (39 classes total)

Merge order: by (organ_id, source_order). Both inputs share the same
global organ taxonomy (respiratory=0, serous=1, thyroid=2, urine=3,
tct_ccd=4 — verified by class_name prefix matching against base30), so
novel organ_ids do NOT need remapping. Within each organ, base30
entries come first followed by novel.

Output schema:
    {
        'class_names':       list[str]  (39, merged in organ-sorted order),
        'local_class_ids':   LongTensor[39]   — 0..38 sequential indices
                              (REQUIRED — STEGO loss is position-indexed),
        'taxonomy_class_ids': LongTensor[39] | None
                              (OPTIONAL — global taxonomy IDs; if either
                              input lacks a consistent global namespace we
                              log a warning and set this to None),
        'organ_ids':         LongTensor[39],
        'axis_ids':          LongTensor[39],
        'rank_along_axis':   LongTensor[39],
        'system_ids':        LongTensor[39],
        'attr_emb':          FloatTensor[39, 5, 512],   (BiomedCLIP 5-attr)
        'is_novel':          BoolTensor[39]   — True for the 9 novel entries
    }

Sanity output printed to stdout (consumed by Step 1 verification):
  - F offdiag distribution: min / q25 / q50 / q75 / max
  - within-organ vs cross-organ offdiag means (sign-only, report-only)
  - top-1 nearest-neighbor same-organ rate (HARD gate, must be > 70%)
  - within-organ collision pairs at cos >= 0.97

Usage:
    PYTHONPATH=. python tools/build_class_metadata_full39.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
BASE_META = REPO / "data/texts/tct_ngc_class_metadata_base30.pt"
NOVEL_META = REPO / "data/texts/tct_ngc_class_metadata_novel_merged.pt"
OUT_META = REPO / "data/texts/tct_ngc_class_metadata_full39.pt"


def _organ_prefix(class_name: str) -> str:
    return class_name.split("-", 1)[0].strip()


def main() -> int:
    if not BASE_META.is_file():
        print(f"missing {BASE_META}", file=sys.stderr)
        return 1
    if not NOVEL_META.is_file():
        print(f"missing {NOVEL_META}", file=sys.stderr)
        return 1

    base = torch.load(BASE_META, map_location="cpu", weights_only=False)
    novel = torch.load(NOVEL_META, map_location="cpu", weights_only=False)

    base_n = len(base["class_names"])
    novel_n = len(novel["class_names"])
    if base_n != 30 or novel_n != 9:
        print(
            f"unexpected class counts: base={base_n} (want 30), "
            f"novel={novel_n} (want 9)",
            file=sys.stderr,
        )
        return 1

    # Verify both files share the same organ taxonomy via class-name prefix.
    base_organ_map: dict[str, int] = {}
    for name, oid in zip(base["class_names"], base["organ_ids"].tolist()):
        base_organ_map.setdefault(_organ_prefix(name), oid)
    novel_organ_map: dict[str, int] = {}
    for name, oid in zip(novel["class_names"], novel["organ_ids"].tolist()):
        novel_organ_map.setdefault(_organ_prefix(name), oid)
    for prefix, oid in novel_organ_map.items():
        if prefix not in base_organ_map:
            print(
                f"novel organ prefix {prefix!r} (organ_id={oid}) absent in "
                f"base30; cannot align — abort.",
                file=sys.stderr,
            )
            return 1
        if base_organ_map[prefix] != oid:
            print(
                f"organ_id mismatch for {prefix!r}: base={base_organ_map[prefix]}"
                f" novel={oid}",
                file=sys.stderr,
            )
            return 1

    # Build the merge index list by (organ_id, source) so per-organ blocks
    # are contiguous; within an organ, base entries precede novel entries.
    rows = []
    for src, meta, src_tag in [("base", base, 0), ("novel", novel, 1)]:
        for i, oid in enumerate(meta["organ_ids"].tolist()):
            rows.append((oid, src_tag, i, src))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    class_names: list[str] = []
    organ_ids: list[int] = []
    axis_ids: list[int] = []
    rank_along_axis: list[int] = []
    system_ids: list[int] = []
    is_novel: list[bool] = []
    attr_emb_chunks: list[torch.Tensor] = []
    taxonomy_ok = True
    taxonomy_ids: list[int] = []
    seen_taxonomy: set[int] = set()
    for oid, src_tag, i, src in rows:
        meta = base if src == "base" else novel
        class_names.append(meta["class_names"][i])
        organ_ids.append(int(meta["organ_ids"][i].item()))
        axis_ids.append(int(meta["axis_ids"][i].item()))
        rank_along_axis.append(int(meta["rank_along_axis"][i].item()))
        system_ids.append(int(meta["system_ids"][i].item()))
        is_novel.append(src == "novel")
        attr_emb_chunks.append(meta["attr_emb"][i : i + 1].float())
        # taxonomy_class_ids: best-effort. base uses 0..29 of its own
        # namespace; novel uses 0..8 of its own namespace. We have no
        # global authoritative ID source in the .pt files themselves, so
        # we form taxonomy_ids by offsetting novel by len(base). This is
        # documented; if downstream code wants real global IDs they must
        # come from a separate taxonomy JSON (out of scope for STEGO).
        raw_cid = int(meta["class_ids"][i])
        global_cid = raw_cid if src == "base" else (base_n + raw_cid)
        if global_cid in seen_taxonomy:
            taxonomy_ok = False
        seen_taxonomy.add(global_cid)
        taxonomy_ids.append(global_cid)

    if not taxonomy_ok or len(set(taxonomy_ids)) != 39:
        print(
            "[WARN] taxonomy_class_ids collision detected — emitting None "
            "(STEGO loss is position-indexed and does not need this).",
            file=sys.stderr,
        )
        taxonomy_ids_t: torch.Tensor | None = None
    else:
        taxonomy_ids_t = torch.tensor(taxonomy_ids, dtype=torch.long)

    attr_emb = torch.cat(attr_emb_chunks, dim=0)  # [39, 5, 512]
    if tuple(attr_emb.shape) != (39, 5, 512):
        print(
            f"unexpected attr_emb shape after merge: {tuple(attr_emb.shape)}",
            file=sys.stderr,
        )
        return 1

    out = {
        "class_names": class_names,
        "local_class_ids": torch.arange(39, dtype=torch.long),
        "taxonomy_class_ids": taxonomy_ids_t,
        "organ_ids": torch.tensor(organ_ids, dtype=torch.long),
        "axis_ids": torch.tensor(axis_ids, dtype=torch.long),
        "rank_along_axis": torch.tensor(rank_along_axis, dtype=torch.long),
        "system_ids": torch.tensor(system_ids, dtype=torch.long),
        "attr_emb": attr_emb,
        "is_novel": torch.tensor(is_novel, dtype=torch.bool),
    }
    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT_META)
    print(f"wrote {OUT_META}")
    print(f"  attr_emb shape : {tuple(attr_emb.shape)}")
    print(f"  is_novel count : {int(out['is_novel'].sum().item())} / 39")
    print(f"  organ_ids      : {organ_ids}")
    print(
        f"  taxonomy_class_ids: "
        f"{'None' if taxonomy_ids_t is None else taxonomy_ids_t.tolist()}"
    )

    # ---------- F-matrix sanity stats ----------
    f_mean = F.normalize(attr_emb.mean(dim=1), dim=-1)  # [39, 512]
    F_mat = f_mean @ f_mean.T  # [39, 39]
    eye = torch.eye(39, dtype=torch.bool)
    offdiag = F_mat[~eye]
    print()
    print("=" * 60)
    print("F offdiag stats (BiomedCLIP attr-mean cosine, full39):")
    print(
        f"  min={offdiag.min().item():.4f} "
        f"q25={torch.quantile(offdiag, 0.25).item():.4f} "
        f"q50={torch.quantile(offdiag, 0.50).item():.4f} "
        f"q75={torch.quantile(offdiag, 0.75).item():.4f} "
        f"max={offdiag.max().item():.4f}"
    )
    print(f"  mean={offdiag.mean().item():.4f} "
          f"std={offdiag.std().item():.4f}")

    # within-organ vs cross-organ offdiag means (report-only)
    organ_t = torch.tensor(organ_ids)
    same_organ = organ_t.unsqueeze(0) == organ_t.unsqueeze(1)
    within_mask = same_organ & ~eye
    cross_mask = ~same_organ
    within = F_mat[within_mask]
    cross = F_mat[cross_mask]
    print()
    print("within-organ vs cross-organ offdiag (report-only):")
    print(
        f"  within: n={within.numel()} mean={within.mean().item():.4f} "
        f"q25={torch.quantile(within, 0.25).item():.4f} "
        f"q75={torch.quantile(within, 0.75).item():.4f}"
    )
    print(
        f"  cross : n={cross.numel()} mean={cross.mean().item():.4f} "
        f"q25={torch.quantile(cross, 0.25).item():.4f} "
        f"q75={torch.quantile(cross, 0.75).item():.4f}"
    )
    print(
        f"  mean(within) - mean(cross) = "
        f"{(within.mean() - cross.mean()).item():+.4f} "
        f"(>0 means within is more similar; sign-only signal)"
    )

    # top-1 NN same-organ rate (HARD gate, > 70%)
    F_off = F_mat.clone()
    F_off.fill_diagonal_(-2.0)
    nn_idx = F_off.argmax(dim=1)
    nn_same_organ = (organ_t[nn_idx] == organ_t)
    same_rate = nn_same_organ.float().mean().item()
    print()
    print(
        f"top-1 NN same-organ rate: {same_rate:.4f} (HARD gate > 0.70)"
    )
    if same_rate <= 0.70:
        print(
            "  [HARD-GATE FAIL] top-1 NN same-organ rate <= 0.70 — "
            "BiomedCLIP attr-mean has no organ signal; consider switching "
            "teacher to 1-PSC BiomedCLIP or raising b_quantile to 0.75.",
            file=sys.stderr,
        )

    # within-organ high-collision pairs (cos >= 0.97)
    collisions = []
    for i in range(39):
        for j in range(i + 1, 39):
            if organ_ids[i] == organ_ids[j] and F_mat[i, j].item() >= 0.97:
                collisions.append((i, j, F_mat[i, j].item()))
    print()
    print(f"within-organ collisions @ cos>=0.97: {len(collisions)} pair(s)")
    for i, j, c in collisions:
        print(f"  cos={c:.4f}  ({class_names[i]})  <->  ({class_names[j]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
