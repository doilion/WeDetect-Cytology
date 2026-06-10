#!/usr/bin/env python
"""Guard: every place that excludes TCT_NGC dev30 negatives agrees with the single source.

Single source of truth:
    data/texts/tct_ngc_dev30_negatives.json  -> negative_class_names

Consumers checked here:
  1. tools/clinical_cost_config_dev30.json  tiers.L0_negative   (must equal the source set,
     and no source name may appear in any other tier).
  2. the two ochmta_m1 configs (biomedclip + xlmr): their resolved
     val_evaluator.exclude_class_names must equal the source set.

The eval scripts (test_exclude_negative.py / eval_summary.py /
tools/evaluate_ngc_filtered.py) read the source via
wedetect.utils.load_dev30_negative_classes, so they cannot drift by construction.

Usage:
    PYTHONPATH=. python tools/check_negative_class_consistency.py
Exit code 0 = all consistent, 1 = mismatch (CI-friendly).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from wedetect.utils import DEV30_NEGATIVES_JSON, load_dev30_negative_classes

REPO_ROOT = Path(__file__).resolve().parents[1]
COST_CONFIG = REPO_ROOT / "tools" / "clinical_cost_config_dev30.json"
CONFIGS = [
    REPO_ROOT / "config" / "wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py",
    REPO_ROOT / "config" / "wedetect_tiny_tct_ngc_dev30_ochmta_m1_xlmr_2gpu.py",
]


def main() -> int:
    canonical = set(load_dev30_negative_classes())
    print(f"[source] {DEV30_NEGATIVES_JSON.name}: {len(canonical)} negatives")
    for n in sorted(canonical):
        print(f"         - {n}")

    failures: list[str] = []

    # 1. cost-config L0_negative == canonical, and canonical names appear nowhere else.
    cost = json.loads(COST_CONFIG.read_text(encoding="utf-8"))
    tiers = cost["tiers"]
    l0 = set(tiers["L0_negative"])
    if l0 != canonical:
        failures.append(
            f"cost-config L0_negative != source. "
            f"missing={sorted(canonical - l0)} extra={sorted(l0 - canonical)}"
        )
    for tier_name, names in tiers.items():
        if tier_name == "L0_negative":
            continue
        leaked = canonical & set(names)
        if leaked:
            failures.append(
                f"cost-config tier {tier_name!r} also lists negative(s): {sorted(leaked)}"
            )

    # 2. resolved config exclude_class_names == canonical.
    try:
        from mmengine.config import Config
    except ImportError:
        print("[warn] mmengine not importable; skipping config resolution check")
    else:
        for cfg_path in CONFIGS:
            cfg = Config.fromfile(str(cfg_path))
            excl = set(cfg.val_evaluator.get("exclude_class_names", []))
            if excl != canonical:
                failures.append(
                    f"{cfg_path.name} exclude_class_names != source. "
                    f"missing={sorted(canonical - excl)} extra={sorted(excl - canonical)}"
                )
            else:
                print(f"[ok]   {cfg_path.name}: exclude_class_names matches source")

    print()
    if failures:
        print("FAIL: negative-class exclusion is inconsistent:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK: all negative-class exclusion sources agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
