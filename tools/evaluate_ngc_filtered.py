#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

import wedetect  # noqa: F401
from wedetect.utils import load_dev30_negative_classes


# Keyword discovery is a BEST-EFFORT extra; it cannot be trusted on its own
# (it misses 'Urine-NHGUC' which has no keyword, and the bare 'NS' abbreviation).
# The authoritative negatives come from the single source
# (data/texts/tct_ngc_dev30_negatives.json) and are force-excluded below.
DEFAULT_EXCLUDE_KEYWORDS = ("negative", "normal", "nilm", "impurity")


def category_counts(ann_path: Path) -> tuple[Counter, dict[int, str]]:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    id_to_name = {int(cat["id"]): cat["name"] for cat in data["categories"]}
    counts = Counter(int(ann["category_id"]) for ann in data["annotations"])
    return counts, id_to_name


def build_exclude_table(
    class_names: list[str],
    counts_by_coco_id: Counter,
    coco_name_by_id: dict[int, str],
    min_annotations: int,
    exclude_keywords: tuple[str, ...],
    force_exclude: frozenset[str] = frozenset(),
) -> tuple[list[str], list[dict]]:
    coco_id_by_name = {name: cid for cid, name in coco_name_by_id.items()}
    exclude_names: list[str] = []
    table: list[dict] = []

    for internal_idx, name in enumerate(class_names):
        coco_id = coco_id_by_name.get(name)
        ann_count = counts_by_coco_id.get(coco_id, 0) if coco_id is not None else 0
        lower_name = name.lower()
        reasons = []
        if name in force_exclude:
            reasons.append("canonical_negative")
        for keyword in exclude_keywords:
            if keyword in lower_name:
                reasons.append(f"name_contains_{keyword}")
        if ann_count < min_annotations:
            reasons.append(f"ann_count_lt_{min_annotations}")

        excluded = bool(reasons)
        if excluded:
            exclude_names.append(name)

        table.append(
            {
                "internal_idx": internal_idx,
                "coco_id": "" if coco_id is None else coco_id,
                "class_name": name,
                "ann_count": ann_count,
                "excluded": int(excluded),
                "reasons": ";".join(reasons),
            }
        )

    return exclude_names, table


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--min-annotations", type=int, default=100)
    parser.add_argument(
        "--exclude-keywords",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_KEYWORDS),
        help="Case-insensitive substrings that mark background or negative classes.",
    )
    args = parser.parse_args()

    register_all_modules()
    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.work_dir

    ann_path = Path(cfg.val_evaluator.ann_file)
    counts_by_coco_id, coco_name_by_id = category_counts(ann_path)
    class_names = list(cfg.dataset_metainfo.classes)
    exclude_names, table = build_exclude_table(
        class_names=class_names,
        counts_by_coco_id=counts_by_coco_id,
        coco_name_by_id=coco_name_by_id,
        min_annotations=args.min_annotations,
        exclude_keywords=tuple(k.lower() for k in args.exclude_keywords),
        force_exclude=frozenset(load_dev30_negative_classes()),
    )

    work_dir = Path(args.work_dir)
    write_table(work_dir / "filtered_eval_class_table.csv", table)
    (work_dir / "filtered_eval_excluded_classes.json").write_text(
        json.dumps(exclude_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg.test_evaluator = dict(
        type="ExcludeClassCocoMetric",
        ann_file=str(ann_path),
        metric="bbox",
        exclude_class_id=exclude_names,
        classwise=True,
    )
    cfg.val_evaluator = cfg.test_evaluator
    cfg.test_dataloader = cfg.val_dataloader

    print("Filtered evaluation excludes:")
    for row in table:
        if row["excluded"]:
            print(
                f"  {row['internal_idx']:02d} {row['class_name']} "
                f"ann={row['ann_count']} reasons={row['reasons']}"
            )

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in table:
        if row["excluded"]:
            for reason in row["reasons"].split(";"):
                grouped[reason].append(row["class_name"])

    result = {
        "checkpoint": args.checkpoint,
        "ann_file": str(ann_path),
        "min_annotations": args.min_annotations,
        "exclude_keywords": args.exclude_keywords,
        "excluded_classes": exclude_names,
        "excluded_by_reason": dict(grouped),
        "kept_classes": [row["class_name"] for row in table if not row["excluded"]],
        "metrics": metrics,
    }
    out_path = work_dir / "filtered_eval_metrics.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
