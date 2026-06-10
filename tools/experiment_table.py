#!/usr/bin/env python3
"""Master experiment-results table for the TCT_NGC line.

Single source of truth = ``docs/experiment_results.csv`` (git-friendly, one row
per (config, ckpt, eval_tag, split)). A markdown view at
``docs/experiment_results.md`` is regenerated from the CSV.

Subcommands
-----------
ingest    Append one eval result to the CSV (call from an eval driver script).
regen     Rebuild docs/experiment_results.md from the CSV.
pending   Scan work_dirs/ and report which ckpts are not yet in the CSV under
          a given eval tag (default: paper_eval).
show      Print the CSV as an aligned table to stdout.

Typical wiring (in a shell eval driver, after a single base/novel run):

    python tools/experiment_table.py ingest \\
        --eval-workdir "$BASE_WD" \\
        --config "$CONFIG"        \\
        --ckpt   "$EPOCH"         \\
        --eval-tag paper_eval     \\
        --split  base25

The eval-workdir is the directory passed as ``--work-dir`` to
``tools/eval_organ_restricted.py`` / ``test.py`` (mmengine drops
``<TS>/<TS>.json`` inside it).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "docs" / "experiment_results.csv"
MD_PATH = REPO_ROOT / "docs" / "experiment_results.md"

CSV_COLUMNS = [
    "timestamp",
    "config",
    "ckpt",
    "eval_tag",
    "split",
    "all_class_mAP",
    "macro_mAP",
    "inst_weighted_mAP",
    "organ_breakdown",   # JSON-serialized {organ: mAP}
    "eval_workdir",
    "notes",
]


# ---------------------------------------------------------------------------
# Parsing one eval workdir
# ---------------------------------------------------------------------------

def _find_eval_json(eval_workdir: Path) -> Path | None:
    """mmengine writes ``<TS>/<TS>.json`` under the eval workdir. Return the
    most-recent one, or None if missing."""
    if not eval_workdir.is_dir():
        return None
    candidates = sorted(eval_workdir.glob("*/[0-9]*_[0-9]*.json"))
    return candidates[-1] if candidates else None


def parse_eval_workdir(eval_workdir: Path) -> dict | None:
    """Pull the metrics this table cares about from an eval workdir's JSON.

    Returns ``{"all_class_mAP", "macro_mAP", "inst_weighted_mAP",
    "organ_breakdown"}``, or None if no parseable JSON was found.
    """
    js = _find_eval_json(eval_workdir)
    if js is None:
        return None
    try:
        d = json.load(open(js))
    except Exception as e:
        print(f"[warn] could not parse {js}: {e}", file=sys.stderr)
        return None

    def num(k):
        v = d.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    # Per-organ mAP — keys look like ``coco/organ/<organ_name>/mAP``.
    organ = {}
    for k, v in d.items():
        if k.startswith("coco/organ/") and k.endswith("/mAP") and isinstance(v, (int, float)):
            organ_name = k[len("coco/organ/"):-len("/mAP")]
            organ[organ_name] = round(float(v), 4)

    return {
        "all_class_mAP": num("coco/all_class/mAP"),
        # Newer protocol uses overall/macro_mAP; older just exposes
        # overall/macro_mAP under the same key. Fall back to all_class for
        # legacy logs.
        "macro_mAP": num("coco/overall/macro_mAP") or num("coco/all_class/mAP"),
        "inst_weighted_mAP": num("coco/overall/instance_weighted_mAP"),
        "organ_breakdown": organ,
        "eval_json": str(js.relative_to(REPO_ROOT)),
    }


# ---------------------------------------------------------------------------
# CSV read/write
# ---------------------------------------------------------------------------

def _load_csv() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _save_csv(rows: list[dict]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def _row_key(r: dict) -> tuple:
    return (r["config"], r["ckpt"], r["eval_tag"], r["split"])


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def cmd_ingest(args) -> int:
    eval_workdir = Path(args.eval_workdir).resolve()
    metrics = parse_eval_workdir(eval_workdir)
    if metrics is None:
        print(f"[ingest] no eval JSON found under {eval_workdir}", file=sys.stderr)
        return 1

    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "config": Path(args.config).name,
        "ckpt": args.ckpt,
        "eval_tag": args.eval_tag,
        "split": args.split,
        "all_class_mAP": f"{metrics['all_class_mAP']:.4f}" if metrics["all_class_mAP"] is not None else "",
        "macro_mAP": f"{metrics['macro_mAP']:.4f}" if metrics["macro_mAP"] is not None else "",
        "inst_weighted_mAP": f"{metrics['inst_weighted_mAP']:.4f}" if metrics["inst_weighted_mAP"] is not None else "",
        "organ_breakdown": json.dumps(metrics["organ_breakdown"], separators=(",", ":")),
        "eval_workdir": str(eval_workdir.relative_to(REPO_ROOT)) if str(eval_workdir).startswith(str(REPO_ROOT)) else str(eval_workdir),
        "notes": args.notes or "",
    }

    rows = _load_csv()
    key = _row_key(row)
    rows = [r for r in rows if _row_key(r) != key]   # replace existing
    rows.append(row)
    rows.sort(key=lambda r: (r["config"], r["ckpt"], r["eval_tag"], r["split"]))
    _save_csv(rows)

    print(f"[ingest] {row['config']} / {row['ckpt']} / {row['eval_tag']}:{row['split']}  "
          f"macro={row['macro_mAP']}  inst_wt={row['inst_weighted_mAP']}  "
          f"organs={metrics['organ_breakdown']}")

    # Auto-diagnose: run rule-based checks against the newly-ingested row.
    if not args.no_diagnose:
        try:
            from tools import diagnose as _diag  # type: ignore
        except Exception:
            sys.path.insert(0, str(REPO_ROOT))
            try:
                import diagnose as _diag  # type: ignore
            except Exception:
                _diag = None
        if _diag is not None:
            try:
                all_rows = _load_csv()
                flags = _diag.diagnose_row(all_rows, row)
                if flags:
                    print(f"  ⚠ diagnose:")
                    for f in flags:
                        print(f"    - {f}")
            except Exception as e:
                print(f"  [diagnose skipped: {e}]", file=sys.stderr)

    if not args.no_regen:
        cmd_regen(args)
    return 0


# ---------------------------------------------------------------------------
# regen markdown
# ---------------------------------------------------------------------------

def cmd_regen(args=None) -> int:
    rows = _load_csv()
    lines = [
        "# Experiment results (auto-generated)",
        "",
        f"_Source: `{CSV_PATH.relative_to(REPO_ROOT)}` — do not edit this file by hand._",
        f"_Last regen: {dt.datetime.now().isoformat(timespec='seconds')}_",
        "",
    ]
    if not rows:
        lines.append("(no rows yet — run `python tools/experiment_table.py ingest ...`)")
        MD_PATH.write_text("\n".join(lines) + "\n")
        print(f"[regen] {MD_PATH.relative_to(REPO_ROOT)} (empty)")
        return 0

    # Group by eval_tag → split → config rows.
    by_tag: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_tag.setdefault(r["eval_tag"], {}).setdefault(r["split"], []).append(r)

    for tag in sorted(by_tag):
        lines.append(f"## eval_tag = `{tag}`")
        lines.append("")
        for split in sorted(by_tag[tag]):
            srows = by_tag[tag][split]
            lines.append(f"### split = `{split}`")
            lines.append("")
            # collect union of organs
            organs: list[str] = []
            for r in srows:
                try:
                    for k in json.loads(r["organ_breakdown"] or "{}"):
                        if k not in organs:
                            organs.append(k)
                except Exception:
                    pass
            organs.sort()

            header = ["config", "ckpt", "macro", "inst_wt", "all_cls"] + organs
            align = ["---"] + ["---"] + [":---:"] * (len(header) - 2)
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(align) + " |")
            for r in sorted(srows, key=lambda x: (x["config"], x["ckpt"])):
                org = {}
                try:
                    org = json.loads(r["organ_breakdown"] or "{}")
                except Exception:
                    pass
                cells = [
                    r["config"].replace("wedetect_tiny_tct_ngc_dev30_", "…dev30_").replace("_biomedclip_2gpu.py", ""),
                    r["ckpt"],
                    r["macro_mAP"] or "—",
                    r["inst_weighted_mAP"] or "—",
                    r["all_class_mAP"] or "—",
                ]
                for o in organs:
                    v = org.get(o)
                    cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        lines.append("")

    MD_PATH.write_text("\n".join(lines))
    print(f"[regen] {MD_PATH.relative_to(REPO_ROOT)}  ({len(rows)} rows)")
    return 0


# ---------------------------------------------------------------------------
# pending
# ---------------------------------------------------------------------------

def cmd_pending(args) -> int:
    """Report which work_dirs have ckpts not yet in CSV under the given tag.

    Default ("any-ckpt"): a work_dir is pending if no row exists for
    ``(config, eval_tag)`` regardless of which epoch. This matches the common
    workflow where the canonical row picks the *best* (not latest) ckpt.

    With ``--strict-latest``: pending also if the latest epoch_*.pth has no
    row — surfaces work_dirs where new training epochs have been written
    since the last ingest.
    """
    rows = _load_csv()
    have_any: set[tuple[str, str]] = set()             # (config, eval_tag)
    have_exact: set[tuple[str, str, str]] = set()      # (config, ckpt, eval_tag)
    for r in rows:
        have_any.add((r["config"], r["eval_tag"]))
        have_exact.add((r["config"], r["ckpt"], r["eval_tag"]))

    pattern = args.glob or "work_dirs/wedetect_tiny_tct_ngc_dev30_*"
    wd_paths = sorted(glob.glob(str(REPO_ROOT / pattern)))

    def _epoch_num(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except Exception:
            return -1

    pending = []
    for wd in wd_paths:
        wd_path = Path(wd)
        if not wd_path.is_dir():
            continue
        cfg_name = wd_path.name + ".py"
        if not (REPO_ROOT / "config" / cfg_name).exists():
            continue
        ckpts = sorted(wd_path.glob("epoch_*.pth"), key=_epoch_num)
        if not ckpts:
            continue
        latest = ckpts[-1].stem
        any_ingested = (cfg_name, args.eval_tag) in have_any
        latest_ingested = (cfg_name, latest, args.eval_tag) in have_exact
        if args.strict_latest:
            is_pending = not latest_ingested
        else:
            is_pending = not any_ingested
        if is_pending:
            reason = "no ingest" if not any_ingested else f"latest {latest} not ingested"
            pending.append((cfg_name, latest, wd_path.relative_to(REPO_ROOT), reason))

    if not pending:
        msg = "strict-latest" if args.strict_latest else "any-ckpt"
        print(f"[pending] up to date — '{pattern}' clean under tag={args.eval_tag} ({msg} mode)")
        return 0

    print(f"[pending] {len(pending)} work_dirs need eval under tag={args.eval_tag}:")
    for cfg, ckpt, wd, reason in pending:
        print(f"  - config={cfg}  latest_ckpt={ckpt}  ({reason})  work_dir={wd}")
    return 0


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

def cmd_show(args) -> int:
    rows = _load_csv()
    if not rows:
        print("(empty)")
        return 0
    cols = ["config", "ckpt", "eval_tag", "split", "macro_mAP", "inst_weighted_mAP"]
    widths = {c: max(len(c), max((len(r.get(c, "")) for r in rows), default=0)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in sorted(rows, key=lambda r: (r["eval_tag"], r["split"], r["config"], r["ckpt"])):
        print("  ".join(r.get(c, "").ljust(widths[c]) for c in cols))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="append one eval result to the CSV")
    pi.add_argument("--eval-workdir", required=True,
                    help="path passed as --work-dir to the eval script (mmengine drops <TS>/<TS>.json inside)")
    pi.add_argument("--config", required=True, help="config file path or basename")
    pi.add_argument("--ckpt", required=True, help='epoch tag, e.g. "epoch_12"')
    pi.add_argument("--eval-tag", required=True, help='e.g. "paper_eval", "corrected_val"')
    pi.add_argument("--split", required=True, help='e.g. "base25", "novel9", "val30"')
    pi.add_argument("--notes", default="")
    pi.add_argument("--no-regen", action="store_true", help="skip regenerating the markdown view")
    pi.add_argument("--no-diagnose", action="store_true", help="skip auto-running tools/diagnose.py on this row")
    pi.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("regen", help="rebuild docs/experiment_results.md from CSV")
    pr.set_defaults(func=cmd_regen)

    pp = sub.add_parser("pending", help="report work_dirs whose latest ckpt isn't in the CSV")
    pp.add_argument("--eval-tag", default="paper_eval")
    pp.add_argument("--glob", default=None,
                    help='work_dirs glob (default: "work_dirs/wedetect_tiny_tct_ngc_dev30_*")')
    pp.add_argument("--strict-latest", action="store_true",
                    help="also flag work_dirs whose latest epoch_*.pth has no row (default: any-ckpt)")
    pp.set_defaults(func=cmd_pending)

    ps = sub.add_parser("show", help="print the CSV as an aligned table")
    ps.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
