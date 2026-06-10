#!/usr/bin/env python3
"""Rule-based experiment diagnostics for the TCT_NGC line.

Reads ``docs/experiment_results.csv`` (and optionally the training
``vis_data/scalars.json``) and prints red flags. Cheap, deterministic — runs
automatically after every ``experiment_table.py ingest``.

Heuristics (all rules from project memory + corrected paper §A protocol):

  • novel-collapse   : novel macro_mAP < 0.5 × best baseline novel
  • base-regression  : base macro_mAP < baseline base − 0.005
  • organ-collapse   : any novel-split organ < 0.3 × best variant for that organ
  • lr-overlap-spike : val_loss at ep3 spikes vs ep2 (mmengine LinearLR ↔ Cosine overlap)
  • best-ckpt-mismatch: ``best_*.pth`` exists but ingest used a different epoch

Usage:
  python tools/diagnose.py --config <config_file.py>   # one config (both splits)
  python tools/diagnose.py --all                       # everything in the CSV
  python tools/diagnose.py --workdir <work_dir>        # everything for one work_dir
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "docs" / "experiment_results.csv"

# The canonical baseline row for the OC-HMTA / paper §A line.
# All M2 variants / ICF / etc are compared against this.
PAPER_BASELINE_CONFIG = "wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py"
PAPER_BASELINE_CKPT = "epoch_12"
PAPER_BASELINE_TAG = "paper_eval"

# Collapse thresholds.
NOVEL_COLLAPSE_RATIO = 0.5     # novel < 0.5× baseline → flag
ORGAN_COLLAPSE_RATIO = 0.3     # per-organ < 0.3× best → flag
BASE_REGRESSION_DELTA = 0.005  # base mAP regression > 0.5pp → flag


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv() -> list[dict]:
    if not CSV_PATH.exists():
        print(f"[diagnose] {CSV_PATH.relative_to(REPO_ROOT)} not found", file=sys.stderr)
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _organs(row: dict) -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(row.get("organ_breakdown") or "{}").items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def rule_novel_collapse(rows: list[dict], target: dict) -> list[str]:
    """target is a novel9 row; flag if it collapses vs baseline."""
    if target["split"] != "novel9":
        return []
    tgt_macro = _f(target, "macro_mAP")
    if tgt_macro is None:
        return []
    baseline = next((r for r in rows if r["config"] == PAPER_BASELINE_CONFIG
                     and r["split"] == "novel9"
                     and r["eval_tag"] == PAPER_BASELINE_TAG), None)
    if baseline is None:
        return []
    bl_macro = _f(baseline, "macro_mAP")
    if bl_macro is None or bl_macro <= 0:
        return []
    if target["config"] == PAPER_BASELINE_CONFIG:
        return []  # baseline doesn't compare to itself
    if tgt_macro < NOVEL_COLLAPSE_RATIO * bl_macro:
        return [f"novel-collapse: macro {tgt_macro:.4f} < {NOVEL_COLLAPSE_RATIO}× baseline ({bl_macro:.4f}) → DEAD"]
    if tgt_macro < bl_macro - 0.02:
        return [f"novel-regression: macro {tgt_macro:.4f} vs baseline {bl_macro:.4f} (Δ={tgt_macro - bl_macro:+.4f})"]
    return []


def rule_base_regression(rows: list[dict], target: dict) -> list[str]:
    if target["split"] != "base25":
        return []
    tgt = _f(target, "macro_mAP")
    if tgt is None:
        return []
    if target["config"] == PAPER_BASELINE_CONFIG:
        return []
    baseline = next((r for r in rows if r["config"] == PAPER_BASELINE_CONFIG
                     and r["split"] == "base25"
                     and r["eval_tag"] == PAPER_BASELINE_TAG), None)
    if baseline is None:
        return []
    bl = _f(baseline, "macro_mAP")
    if bl is None:
        return []
    if tgt < bl - BASE_REGRESSION_DELTA:
        return [f"base-regression: macro {tgt:.4f} vs baseline {bl:.4f} (Δ={tgt - bl:+.4f})"]
    return []


def rule_organ_collapse(rows: list[dict], target: dict) -> list[str]:
    """A target organ that's much worse than the best variant on that organ."""
    if target["split"] != "novel9":
        return []
    org = _organs(target)
    if not org:
        return []
    # Find best variant per organ
    peer_rows = [r for r in rows if r["split"] == "novel9" and r["eval_tag"] == target["eval_tag"]]
    best_per_organ: dict[str, float] = {}
    for r in peer_rows:
        for k, v in _organs(r).items():
            if v > best_per_organ.get(k, -1):
                best_per_organ[k] = v
    flags = []
    for k, v in org.items():
        peak = best_per_organ.get(k, 0)
        if peak > 0.05 and v < ORGAN_COLLAPSE_RATIO * peak:
            flags.append(f"organ-collapse: {k} {v:.4f} < {ORGAN_COLLAPSE_RATIO}× best ({peak:.4f}) across variants")
    return flags


def rule_lr_overlap_spike(work_dir: Path) -> list[str]:
    """Look at validation events in vis_data/scalars.json (one per epoch) — flag
    if ep3 val_loss spikes vs ep2 by more than 10% of ep2 → ep1 drop.
    Matches the mmengine LinearLR + CosineAnnealingLR overlap bug from memory.
    """
    scalars = sorted(work_dir.glob("*/vis_data/scalars.json"))
    if not scalars:
        return []
    # Use the latest training run
    val_losses: dict[int, float] = {}
    with open(scalars[-1]) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            # val events have an epoch field but no iter (or iter == max)
            if "val_loss" in ev and "epoch" in ev:
                val_losses[int(ev["epoch"])] = float(ev["val_loss"])
    if not all(e in val_losses for e in (1, 2, 3)):
        return []
    e1, e2, e3 = val_losses[1], val_losses[2], val_losses[3]
    drop12 = e1 - e2
    if drop12 <= 0:
        return []
    rise23 = e3 - e2
    if rise23 > 0.1 * drop12:
        return [f"lr-overlap-spike: val_loss ep1={e1:.4f}→ep2={e2:.4f}→ep3={e3:.4f} (ep3 rose {rise23:+.4f}). LinearLR+CosineAnnealingLR overlap likely; CosineAnnealingLR begin=2 fix."]
    return []


def rule_best_ckpt_mismatch(work_dir: Path, ingested_ckpt: str) -> list[str]:
    bests = list(work_dir.glob("best_*epoch_*.pth"))
    if not bests:
        return []
    # parse "best_coco_overall_macro_mAP_epoch_12.pth" → "epoch_12"
    best_ep = bests[-1].stem.split("_epoch_")[-1]
    best_tag = f"epoch_{best_ep}"
    if best_tag != ingested_ckpt:
        return [f"note: training-time best ckpt is {best_tag} ({bests[-1].name}), ingested {ingested_ckpt} (corrected protocol — confirm intentional)"]
    return []


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def diagnose_row(rows: list[dict], row: dict, *, also_workdir: bool = True) -> list[str]:
    flags: list[str] = []
    flags += rule_novel_collapse(rows, row)
    flags += rule_base_regression(rows, row)
    flags += rule_organ_collapse(rows, row)
    if also_workdir and row.get("eval_workdir"):
        # eval_workdir points at e.g. work_dirs/<run>/paper_eval/base25_..._workdir
        # The run dir is two levels up.
        ew = REPO_ROOT / row["eval_workdir"]
        run_dir = ew.parent.parent if ew.is_dir() else None
        if run_dir and run_dir.is_dir():
            flags += rule_lr_overlap_spike(run_dir)
            flags += rule_best_ckpt_mismatch(run_dir, row["ckpt"])
    return flags


def fmt_header(row: dict) -> str:
    cfg = row["config"].replace("wedetect_tiny_tct_ngc_dev30_", "…dev30_")
    return f"{cfg} / {row['ckpt']} / {row['eval_tag']}:{row['split']}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="config filename (with or without path); diagnose all rows")
    g.add_argument("--workdir", help="work_dir name (basename); diagnose all rows whose config matches")
    g.add_argument("--all", action="store_true", help="diagnose every row in the CSV")
    p.add_argument("--quiet-clean", action="store_true", help="suppress 'clean' lines, only print red flags")
    args = p.parse_args(argv)

    rows = _load_csv()
    if not rows:
        return 0

    if args.config:
        target_cfg = Path(args.config).name
        targets = [r for r in rows if r["config"] == target_cfg]
    elif args.workdir:
        wd_basename = Path(args.workdir).name
        targets = [r for r in rows if r["config"] == wd_basename + ".py"]
    else:
        targets = rows

    if not targets:
        print("[diagnose] no matching rows", file=sys.stderr)
        return 0

    any_flag = False
    for row in targets:
        flags = diagnose_row(rows, row)
        if flags:
            any_flag = True
            print(f"⚠ {fmt_header(row)}")
            for f in flags:
                print(f"    - {f}")
        elif not args.quiet_clean:
            print(f"✓ {fmt_header(row)}  (clean)")

    if not any_flag and args.quiet_clean:
        # Stay silent — nothing to surface.
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
