#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from wedetect.utils.data_paths import get_tct_ngc_root


SPLITS = {
    "train": "instances_train.json",
    "train_dev": "instances_train_dev.json",
    "val_dev": "instances_val_dev.json",
    "test_base": "instances_test_base.json",
    "test_novel": "instances_test_novel.json",
    "test_pseudo_novel": "instances_test_pseudo_novel.json",
    "test_main_novel": "instances_test_main_novel.json",
    "hard_test": "instances_hard_test.json",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_counts(ann_dir: Path) -> tuple[dict, dict]:
    counts: dict[str, Counter] = {}
    image_counts: dict[str, dict[str, int]] = {}
    for split, filename in SPLITS.items():
        path = ann_dir / filename
        if not path.exists():
            continue
        data = load_json(path)
        id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
        ann_counter = Counter()
        image_sets = defaultdict(set)
        for ann in data["annotations"]:
            name = id_to_name[ann["category_id"]]
            ann_counter[name] += 1
            image_sets[name].add(ann["image_id"])
        counts[split] = ann_counter
        image_counts[split] = {name: len(ids) for name, ids in image_sets.items()}
    return counts, image_counts


def load_label_map(root: Path) -> dict:
    path = root / "metadata" / "label_map_v2.json"
    if not path.exists():
        return {}
    return load_json(path)


def collect_records(root: Path) -> list[dict]:
    ann_dir = root / "annotations"
    counts, image_counts = split_counts(ann_dir)
    label_map = load_label_map(root)

    names = set(label_map)
    for counter in counts.values():
        names.update(counter)

    records = []
    for name in sorted(names):
        meta = label_map.get(name, {})
        train = counts.get("train", Counter()).get(name, 0)
        train_dev = counts.get("train_dev", Counter()).get(name, 0)
        val_dev = counts.get("val_dev", Counter()).get(name, 0)
        test_base = counts.get("test_base", Counter()).get(name, 0)
        test_novel = counts.get("test_novel", Counter()).get(name, 0)
        pseudo = counts.get("test_pseudo_novel", Counter()).get(name, 0)
        main = counts.get("test_main_novel", Counter()).get(name, 0)
        hard = counts.get("hard_test", Counter()).get(name, 0)
        row = {
            "name": name,
            "id": meta.get("id", ""),
            "role": meta.get("role", ""),
            "ontology": meta.get("ontology", ""),
            "meta_count": meta.get("count", ""),
            "train_anns": train,
            "train_images": image_counts.get("train", {}).get(name, 0),
            "train_dev_anns": train_dev,
            "train_dev_images": image_counts.get("train_dev", {}).get(name, 0),
            "val_dev_anns": val_dev,
            "val_dev_images": image_counts.get("val_dev", {}).get(name, 0),
            "test_base_anns": test_base,
            "test_base_images": image_counts.get("test_base", {}).get(name, 0),
            "test_novel_anns": test_novel,
            "test_novel_images": image_counts.get("test_novel", {}).get(name, 0),
            "test_pseudo_novel_anns": pseudo,
            "test_main_novel_anns": main,
            "hard_test_anns": hard,
        }
        row["val_over_train"] = (val_dev / train) if train else None
        row["test_base_over_train"] = (test_base / train) if train else None
        records.append(row)
    return records


def issue_flags(row: dict) -> list[str]:
    flags = []
    role = row["role"]
    train = row["train_anns"]
    val = row["val_dev_anns"]
    test_base = row["test_base_anns"]
    ratio = row["test_base_over_train"]
    ontology = row["ontology"]

    if role == "base":
        if train == 0 and test_base > 0:
            flags.append("base_test_present_train_absent")
        if 0 < train < 300 and test_base >= 100:
            flags.append("base_train_lt_300_with_test_base_ge_100")
        if val < 100 and ontology != "negative":
            flags.append("val_dev_lt_100_non_negative")
        if ratio is not None and ratio >= 3 and test_base >= 100:
            flags.append("test_base_over_train_ge_3")
        if ratio is not None and ratio >= 1.5 and test_base >= 1000 and ontology != "negative":
            flags.append("test_base_over_train_ge_1.5_large_non_negative")

    if row["name"] == "Urine-HGUC" and role != "base":
        flags.append("semantic_review_hguc_is_high_grade_but_not_base")
    if row["name"] == "Urine-SHGUC" and role == "base" and train < 300:
        flags.append("semantic_review_shguc_base_but_train_sparse")

    return flags


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_ratio(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    header = rows[0]
    out.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(header)) + " |")
    out.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    for row in rows[1:]:
        out.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |")
    return "\n".join(out)


def build_markdown(records: list[dict], root: Path, csv_path: Path) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    base_records = [row for row in records if row["role"] == "base"]
    issue_rows = []
    for row in records:
        flags = issue_flags(row)
        if flags:
            issue_rows.append((row, flags))

    severe = [
        (row, flags)
        for row, flags in issue_rows
        if "test_base_over_train_ge_3" in flags
        or "semantic_review_hguc_is_high_grade_but_not_base" in flags
        or "semantic_review_shguc_base_but_train_sparse" in flags
    ]
    val_low = [
        row
        for row in base_records
        if row["val_dev_anns"] < 100 and row["ontology"] != "negative"
    ]
    high_ratio = sorted(
        [
            row
            for row in base_records
            if row["test_base_anns"] >= 100 and row["train_anns"] > 0
        ],
        key=lambda r: r["test_base_over_train"] or 0,
        reverse=True,
    )[:12]
    urine = [row for row in records if row["name"].startswith("Urine-")]

    lines = [
        "# TCT_NGC Split Audit",
        "",
        f"- Dataset root: `{root}`",
        f"- Generated: `{now}`",
        f"- Full per-class CSV: `{csv_path}`",
        "",
        "## Executive Summary",
        "",
        "The `train_dev` and `val_dev` split is mostly behaving like a 9:1 split of `instances_train.json`. "
        "The main issue is upstream: several base classes have very different distributions between "
        "`instances_train.json` and `instances_test_base.json`.",
        "",
        "The most important problem is `Urine-SHGUC`: it is marked as a base class, but training has only "
        "181 annotations while `test_base` has 2706 annotations. This makes base evaluation for that class "
        "unfair and unstable. Separately, `Urine-HGUC` is a high-grade malignant category but is assigned "
        "to `pseudo_novel`, while the weaker `Urine-SHGUC` category is assigned to `base`; this is clinically "
        "and experimentally questionable unless the intended task is explicitly zero-shot transfer from "
        "suspicious high-grade to definite high-grade carcinoma.",
        "",
        "## High-Risk Findings",
        "",
    ]

    risk_rows = [["class", "role", "ontology", "train", "val_dev", "test_base", "novel", "test/train", "flags"]]
    for row, flags in severe:
        risk_rows.append(
            [
                row["name"],
                row["role"],
                row["ontology"],
                str(row["train_anns"]),
                str(row["val_dev_anns"]),
                str(row["test_base_anns"]),
                str(row["test_novel_anns"]),
                fmt_ratio(row["test_base_over_train"]),
                ", ".join(flags),
            ]
        )
    lines.append(markdown_table(risk_rows))

    lines.extend(
        [
            "",
            "## Base Classes With Highest Test To Train Ratio",
            "",
            "This table highlights base classes whose `test_base` annotation count is large relative to `train`. "
            "Large ratios are not always fatal for negative or background classes, but they are problematic for "
            "diagnostic disease classes.",
            "",
        ]
    )
    ratio_rows = [["class", "ontology", "train", "val_dev", "test_base", "test/train"]]
    for row in high_ratio:
        ratio_rows.append(
            [
                row["name"],
                row["ontology"],
                str(row["train_anns"]),
                str(row["val_dev_anns"]),
                str(row["test_base_anns"]),
                fmt_ratio(row["test_base_over_train"]),
            ]
        )
    lines.append(markdown_table(ratio_rows))

    lines.extend(
        [
            "",
            "## Validation Classes Below 100 Annotations",
            "",
            "These classes make per-class AP estimates noisy on `val_dev`. The current filtered evaluation "
            "rule excludes classes with fewer than 100 validation annotations, plus negative, normal, NILM, "
            "and impurity classes.",
            "",
        ]
    )
    low_rows = [["class", "ontology", "train", "val_dev", "test_base", "reason"]]
    for row in val_low:
        low_rows.append(
            [
                row["name"],
                row["ontology"],
                str(row["train_anns"]),
                str(row["val_dev_anns"]),
                str(row["test_base_anns"]),
                "val_dev < 100",
            ]
        )
    lines.append(markdown_table(low_rows))

    lines.extend(
        [
            "",
            "## Urine Category Detail",
            "",
        ]
    )
    urine_rows = [["class", "role", "ontology", "meta_count", "train", "val_dev", "test_base", "test_novel"]]
    for row in sorted(urine, key=lambda r: str(r["id"])):
        urine_rows.append(
            [
                row["name"],
                row["role"],
                row["ontology"],
                str(row["meta_count"]),
                str(row["train_anns"]),
                str(row["val_dev_anns"]),
                str(row["test_base_anns"]),
                str(row["test_novel_anns"]),
            ]
        )
    lines.append(markdown_table(urine_rows))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Urine-SHGUC` is the clearest split defect among diagnostic base classes. It should not be a "
            "major contributor to the main base metric unless the train/test split is rebuilt.",
            "- `Urine-AUC` is less extreme but still has only 76 validation annotations and a `test_base/train` "
            "ratio above 2, so its validation AP will also be noisy.",
            "- `Thyroid gland-Negative samples` has a high test/train ratio, but it is a negative category and "
            "is already excluded from the filtered diagnostic metric.",
            "- `Urine-HGUC` is semantically high-grade disease but is currently only in novel splits. This can "
            "be valid only if the experiment is explicitly an open-vocabulary or zero-shot high-grade transfer "
            "task. For a closed-set diagnostic detector, it should be moved into base training or merged with "
            "a high-grade urine class.",
            "",
            "## Recommended Actions",
            "",
            "1. Keep the current running experiment for continuity, but report the filtered metric as the main "
            "diagnostic metric.",
            "2. For the next dataset revision, rebuild the base train, val, and test split so that diagnostic "
            "base classes have adequate train and validation support. `Urine-SHGUC` is the first class to fix.",
            "3. Decide the urine label design explicitly:",
            "   - closed-set option: include `Urine-HGUC` in base train and validation;",
            "   - hierarchy option: keep `AUC`, `SHGUC`, and `HGUC` separate but balance all three;",
            "   - pragmatic option: merge `SHGUC` and `HGUC` into one high-grade urine category if visual "
            "separation is too weak or sample counts are limited.",
            "4. Continue excluding negative, normal, NILM, impurity, and validation-count-below-100 classes from "
            "the headline filtered AP until the split is rebuilt.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(get_tct_ngc_root()))
    parser.add_argument(
        "--out-md",
        default="docs/tct_ngc_split_audit_20260429.md",
    )
    parser.add_argument(
        "--out-csv",
        default="work_dirs/tct_ngc_dataset_audit/split_audit_per_class.csv",
    )
    args = parser.parse_args()

    root = Path(args.root)
    records = collect_records(root)
    for row in records:
        row["flags"] = ";".join(issue_flags(row))

    csv_path = Path(args.out_csv)
    md_path = Path(args.out_md)
    write_csv(csv_path, records)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown(records, root, csv_path), encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
