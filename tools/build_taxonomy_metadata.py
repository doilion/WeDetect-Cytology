#!/usr/bin/env python
"""Build TCT_NGC taxonomy metadata for OC-HMTA training.

Parses 5-attribute class JSON files (e.g., tct_ngc_fullnames_30_attr_train.json)
and the existing dataset annotations to produce a single canonical taxonomy
lookup:
  - organ assignment per class (5 organs)
  - diagnostic-code parse per class (system, category, subcategory, entity)
  - cross-organ category groups (e.g., all "Malignant" classes across organs)
  - within-organ category groups (e.g., all "Bethesda VI" classes in Thyroid)

Output: data/texts/tct_ngc_taxonomy.json with structure:
  {
    "organs": ["respiratory tract", "Serous effusion", "Thyroid gland",
               "Urine", "TCT_CCD"],
    "organ_to_id": {"respiratory tract": 0, ...},
    "classes": {
      "<class_name>": {
        "class_id": int,            # mapped from train ann cat_id
        "organ": str,
        "organ_id": int,
        "diag_system": str,         # PSC | Bethesda | TIS | Paris | <other>
        "diag_category_roman": str, # "II", "VI", "V", ...
        "diag_category_idx": int,   # 1-based numeric form
        "diag_subcategory": str | null,  # e.g., "MAL-S" in "TIS V: MAL-S"
        "diag_full": str,           # canonical "<system> <category>[:<sub>]"
        "diag_entity": str,         # tail after ": "
        "split": "base30" | "novel_main_3" | "novel_pseudo_2" | "novel_hard_4",
      },
      ...
    },
    "organ_to_classes": {organ: [class_name, ...]},      # within-organ groups
    "diag_full_to_classes": {diag_full: [class_name, ...]},  # cross-organ same-category groups
    "stats": {...},                  # counts, parse coverage
  }

Usage:
  python tools/build_taxonomy_metadata.py \
      --base-attrs data/texts/tct_ngc_fullnames_30_attr_train.json \
      --novel-attrs data/texts/tct_ngc_attr_main_3_eval.json \
                    data/texts/tct_ngc_attr_pseudo_2_eval.json \
                    data/texts/tct_ngc_attr_hard_4_eval.json \
      --base-ann "$TCT_NGC_DATA_ROOT/annotations/instances_test_base_clean_dev30.json" \
      --novel-anns "$TCT_NGC_DATA_ROOT/annotations/instances_test_main_novel.json" \
                   "$TCT_NGC_DATA_ROOT/annotations/instances_test_pseudo_novel.json" \
                   "$TCT_NGC_DATA_ROOT/annotations/instances_hard_test.json" \
      --out data/texts/tct_ngc_taxonomy.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Canonical organ ordering (matches dataset file_name hierarchy)
ORGANS = [
    "respiratory tract",
    "Serous effusion",
    "Thyroid gland",
    "Urine",
    "TCT_CCD",
]
ORGAN_TO_ID = {o: i for i, o in enumerate(ORGANS)}


def organ_of(class_name: str) -> str:
    """Extract organ from organ-prefixed class name.

    Matches the existing convention in `tools/analyze_ngc_disjoint_results.py`.
    """
    return class_name.split("-", 1)[0]


# Diagnostic code parsing regex
# Examples we need to handle:
#   "PSC Category II: Negative — acute neutrophilic inflammation"
#   "Bethesda VI: Malignant — papillary thyroid carcinoma"
#   "TIS Category V: MAL-S — metastatic breast ductal carcinoma"
#   "Paris System NHGUC — benign / non-neoplastic background"
#   "Bethesda I — Nondiagnostic / Unsatisfactory"
#   "PSC Category VI: Malignant — squamous cell carcinoma"

DIAG_PATTERNS = [
    # System (PSC|Bethesda|TIS|Paris ...) [Category]? <roman/numeric>[: <subcat>]? <separator> <entity>
    re.compile(
        r"^(?P<system>PSC|Bethesda|TIS|Paris(?:\s+System)?|TIS|Other)\s+"
        r"(?:Category\s+)?"
        r"(?P<roman>[IVX]+|[0-9]+|HGUC|NHGUC|SHGUC|AUC|LSIL|HSIL)"
        r"(?:\s*:\s*(?P<subcat>[A-Za-z0-9 ,\-\/]+?))?"
        r"\s*[—\-:]\s*(?P<entity>.+)$",
        flags=re.UNICODE,
    ),
]

ROMAN_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}


# Cervical Bethesda is multi-axis (squamous lesion ladder, glandular lesion
# ladder, infection findings, adequacy markers). Single 1D rank would conflate
# semantically distinct entities. Explicit (axis, rank_along_axis) mapping.
# axis_id assignment: squamous=0, glandular=1, infection=2, adequacy=3.
CERVICAL_AXIS_MAP = {
    "TCT_CCD-normal":                    ("squamous",  0),  # II NILM
    "TCT_CCD-ascus":                     ("squamous",  1),  # III ASC-US
    "TCT_CCD-asch":                      ("squamous",  2),  # III ASC-H (slightly higher concern)
    "TCT_CCD-lsil":                      ("squamous",  3),  # IV LSIL
    "TCT_CCD-hsil_scc_omn":              ("squamous",  4),  # V HSIL / SCC / Other malignancies
    "TCT_CCD-agc_adenocarcinoma_em":     ("glandular", 0),  # AGC / Adenocarcinoma endometrial
    "TCT_CCD-vaginalis":                 ("infection", 0),  # Trichomonas
    "TCT_CCD-monilia":                   ("infection", 1),  # Candida
    "TCT_CCD-dysbacteriosis_herpes_act": ("infection", 2),  # mixed flora / HSV / actinomyces
    "TCT_CCD-ec":                        ("adequacy",  0),  # Endocervical cells present
}
CERVICAL_AXIS_TO_ID = {"squamous": 0, "glandular": 1, "infection": 2, "adequacy": 3}
# For non-cervical organs, only a single primary diagnostic axis exists.
PRIMARY_AXIS = "primary"
PRIMARY_AXIS_ID = 0


def parse_diagnostic_code(diag_str: str) -> Dict:
    """Parse diag_code attribute string into structured fields.

    Returns dict with: system, category_roman, category_idx, subcategory,
    diag_full (canonical "<system> <roman>"), diag_entity (tail).

    On parse failure, returns dict with all-None except diag_entity = raw string.
    """
    s = diag_str.strip()
    for pat in DIAG_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        system_raw = m.group("system")
        # Normalize: "Paris System" → "Paris"
        system = system_raw.split()[0] if system_raw else None
        roman = m.group("roman")
        subcat = m.group("subcat")
        if subcat:
            subcat = subcat.strip()
        entity = m.group("entity").strip()
        if roman in ROMAN_TO_INT:
            cat_idx = ROMAN_TO_INT[roman]
        elif roman.isdigit():
            cat_idx = int(roman)
        else:
            # e.g., HGUC, NHGUC — non-numeric Paris classes
            cat_idx = -1  # sentinel
        diag_full_parts = [system, roman]
        if subcat:
            diag_full_parts.append(subcat)
        diag_full = " ".join(diag_full_parts)
        return {
            "system": system,
            "category_roman": roman,
            "category_idx": cat_idx,
            "subcategory": subcat,
            "diag_full": diag_full,
            "diag_entity": entity,
            "parsed": True,
        }
    # Failed to parse — return raw for inspection
    return {
        "system": None,
        "category_roman": None,
        "category_idx": -1,
        "subcategory": None,
        "diag_full": None,
        "diag_entity": s,
        "parsed": False,
    }


def load_class_names_in_cat_id_order(ann_file: Path) -> List[str]:
    """Load class names sorted by cat_id (ascending)."""
    with open(ann_file) as f:
        ann = json.load(f)
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    return [c["name"] for c in cats], [c["id"] for c in cats]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-attrs", required=True,
                   help="5-attr JSON for base 30 classes (in cat_id order)")
    p.add_argument("--novel-attrs", nargs="+", required=True,
                   help="5-attr JSON for novel splits (main_3 / pseudo_2 / hard_4)")
    p.add_argument("--base-ann", required=True,
                   help="COCO ann file to map cat_id → class_name for base")
    p.add_argument("--novel-anns", nargs="+", required=True,
                   help="COCO ann files for novel splits (matching --novel-attrs order)")
    p.add_argument("--novel-split-names", nargs="+",
                   default=["main_3", "pseudo_2", "hard_4"],
                   help="Names for novel splits (matching --novel-anns order)")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    if len(args.novel_attrs) != len(args.novel_anns):
        raise SystemExit(
            f"--novel-attrs ({len(args.novel_attrs)}) must equal --novel-anns "
            f"({len(args.novel_anns)})"
        )
    if len(args.novel_attrs) != len(args.novel_split_names):
        raise SystemExit(
            f"--novel-attrs ({len(args.novel_attrs)}) must equal --novel-split-names "
            f"({len(args.novel_split_names)})"
        )

    classes: Dict[str, Dict] = {}
    parse_failed: List[Tuple[str, str]] = []

    # ── Base 30 ──
    base_attrs = json.load(open(args.base_attrs))
    base_names, base_cat_ids = load_class_names_in_cat_id_order(args.base_ann)
    if len(base_attrs) != len(base_names):
        raise SystemExit(
            f"base attr count {len(base_attrs)} != base ann cat count {len(base_names)}"
        )
    for cls_name, cls_id, attrs in zip(base_names, base_cat_ids, base_attrs):
        if len(attrs) < 5:
            raise SystemExit(f"class {cls_name!r} has {len(attrs)} attrs, expected ≥5")
        organ = organ_of(cls_name)
        if organ not in ORGAN_TO_ID:
            raise SystemExit(f"unknown organ {organ!r} from class {cls_name!r}")
        diag_str = attrs[1]
        diag = parse_diagnostic_code(diag_str)
        if not diag["parsed"]:
            parse_failed.append((cls_name, diag_str))

        # Module 2 ordinal axis assignment:
        # - Cervical (TCT_CCD) has explicit 4-axis structure (squamous/glandular/
        #   infection/adequacy) — see CERVICAL_AXIS_MAP.
        # - Other organs have a single 'primary' diagnostic ladder, with
        #   rank_along_axis = parsed category_idx (II=2, VI=6, ...).
        if cls_name in CERVICAL_AXIS_MAP:
            axis, rank_along_axis = CERVICAL_AXIS_MAP[cls_name]
            axis_id = CERVICAL_AXIS_TO_ID[axis]
        else:
            axis = PRIMARY_AXIS
            axis_id = PRIMARY_AXIS_ID
            rank_along_axis = diag["category_idx"]  # -1 if parse failed

        classes[cls_name] = {
            "class_id": cls_id,
            "organ": organ,
            "organ_id": ORGAN_TO_ID[organ],
            "diag_raw": diag_str,
            **{k: v for k, v in diag.items() if k != "parsed"},
            "axis": axis,
            "axis_id": axis_id,
            "rank_along_axis": rank_along_axis,
            "split": "base30",
        }

    # ── Novel splits ──
    for split_name, attrs_file, ann_file in zip(args.novel_split_names, args.novel_attrs, args.novel_anns):
        novel_attrs = json.load(open(attrs_file))
        novel_names, novel_cat_ids = load_class_names_in_cat_id_order(ann_file)
        # Some novel splits ann files include base classes with cat_id 0..29 plus
        # novel ones at higher cat_ids. Only register classes whose name is novel
        # (not in classes already). Check by name presence.
        novel_only_pairs = [
            (n, i, a) for n, i, a in zip(novel_names, novel_cat_ids, novel_attrs)
            if n not in classes
        ]
        # If novel attr file is novel-only (3/2/4 entries), iterate directly
        if len(novel_attrs) in (2, 3, 4) and len(novel_attrs) < len(novel_names):
            # attrs are novel-only, names list has base + novel — match by tail
            # (heuristic: novel classes have highest cat_ids)
            novel_only_idx = sorted(range(len(novel_names)),
                                    key=lambda j: novel_cat_ids[j])[-len(novel_attrs):]
            novel_only_pairs = [
                (novel_names[j], novel_cat_ids[j], novel_attrs[k])
                for k, j in enumerate(novel_only_idx)
            ]
        for cls_name, cls_id, attrs in novel_only_pairs:
            if len(attrs) < 5:
                raise SystemExit(f"novel class {cls_name!r} has {len(attrs)} attrs")
            organ = organ_of(cls_name)
            if organ not in ORGAN_TO_ID:
                raise SystemExit(f"unknown organ {organ!r} from novel class {cls_name!r}")
            diag_str = attrs[1]
            diag = parse_diagnostic_code(diag_str)
            if not diag["parsed"]:
                parse_failed.append((cls_name, diag_str))

            if cls_name in CERVICAL_AXIS_MAP:
                axis, rank_along_axis = CERVICAL_AXIS_MAP[cls_name]
                axis_id = CERVICAL_AXIS_TO_ID[axis]
            else:
                axis = PRIMARY_AXIS
                axis_id = PRIMARY_AXIS_ID
                rank_along_axis = diag["category_idx"]

            classes[cls_name] = {
                "class_id": cls_id,
                "organ": organ,
                "organ_id": ORGAN_TO_ID[organ],
                "diag_raw": diag_str,
                **{k: v for k, v in diag.items() if k != "parsed"},
                "axis": axis,
                "axis_id": axis_id,
                "rank_along_axis": rank_along_axis,
                "split": f"novel_{split_name}",
            }

    # ── Aggregate views ──
    organ_to_classes: Dict[str, List[str]] = defaultdict(list)
    diag_full_to_classes: Dict[str, List[str]] = defaultdict(list)
    for cls_name, info in classes.items():
        organ_to_classes[info["organ"]].append(cls_name)
        if info["diag_full"]:
            diag_full_to_classes[info["diag_full"]].append(cls_name)

    # ── Stats ──
    parsed_count = sum(1 for c in classes.values() if c.get("diag_full") is not None)
    stats = {
        "total_classes": len(classes),
        "base30_count": sum(1 for c in classes.values() if c["split"] == "base30"),
        "novel_count": sum(1 for c in classes.values() if c["split"].startswith("novel_")),
        "diag_parsed": parsed_count,
        "diag_parse_failed": len(parse_failed),
        "organ_distribution": {o: len(cs) for o, cs in organ_to_classes.items()},
        "diag_full_distribution": {d: len(cs) for d, cs in diag_full_to_classes.items()},
    }

    # ── Write output ──
    out = {
        "organs": ORGANS,
        "organ_to_id": ORGAN_TO_ID,
        "classes": classes,
        "organ_to_classes": dict(organ_to_classes),
        "diag_full_to_classes": dict(diag_full_to_classes),
        "stats": stats,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    print(f"\nStats:")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    if parse_failed:
        print(f"\n⚠ {len(parse_failed)} diag-code parse failures:")
        for cls_name, raw in parse_failed[:5]:
            print(f"  {cls_name!r}: {raw!r}")
        if len(parse_failed) > 5:
            print(f"  ... ({len(parse_failed) - 5} more)")


if __name__ == "__main__":
    main()
