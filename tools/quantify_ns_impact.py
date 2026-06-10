#!/usr/bin/env python
"""Quantify the Thyroid-NS exclusion impact on the headline organ-macro.

NS (Bethesda I Nondiagnostic = FNA artifact/impurity) was a BASE class missing from the
OrganRestrictedCocoMetric exclude list, so the reported "base25 organ-macro" counted it in
the Thyroid organ. This re-evaluates a finished checkpoint TWICE on the same predictions-
producing pass set, differing only in the exclude list, and prints both headline macros:
  - base25 = exclude the 5 historical negatives (incl NS)  -> OLD headline
  - base24 = exclude those 5 + Thyroid gland-NS            -> CORRECTED headline
The metric computes each organ AP as a JOINT COCOeval over that organ's cat_ids, so we let
the metric do it (only Thyroid changes; NS is a Thyroid class).

Usage (single GPU; val_dev @640 cache must be local):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/quantify_ns_impact.py \
      --config config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py \
      --checkpoint work_dirs/.../best_coco_overall_macro_mAP_epoch_9.pth
"""
import argparse

import torch  # noqa: F401
from mmengine.config import Config
from mmengine.runner import Runner

import wedetect  # noqa: F401  register modules

OLD5 = [
    "respiratory tract-Impurity",
    "Serous effusion-Negative samples",
    "Thyroid gland-Negative samples",
    "Urine-NHGUC",
    "TCT_CCD-normal",
]
NS = "Thyroid gland-NS"


def run(runner, exclude):
    # change ONLY the exclude list between passes; keep _coco_api (the GT loaded from
    # ann_file) — nulling it makes CocoMetric.process think GT is absent. compute_metrics
    # rebuilds its COCOeval fresh each call anyway, so reuse is correct.
    metric = runner.val_evaluator.metrics[0]
    metric._exclude_class_names = list(exclude)
    return runner.val()


def get(d, *suffixes):
    for k, v in d.items():
        if any(k.endswith(s) for s in suffixes):
            return k, v
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = "work_dirs/_ns_quantify_tmp"
    cfg.load_from = args.checkpoint
    # config already sets val_evaluator.ann_file (the collaborator's single-source refactor);
    # no refresh needed. We override exclude_class_names per pass below.
    assert cfg.val_evaluator.get("ann_file"), "config val_evaluator.ann_file missing"
    cfg.val_dataloader.batch_size = 1
    cfg.val_evaluator.classwise = False
    cfg.val_evaluator.exclude_class_names = list(OLD5)
    cfg.custom_hooks = [h for h in cfg.get("custom_hooks", [])
                        if "CheckBackboneNorm" not in str(h.get("type", ""))]

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint, map_location="cpu")

    print("\n[quantify_ns] PASS 1/2: base25 (exclude 5, NS INCLUDED) ...")
    m25 = run(runner, OLD5)
    print("\n[quantify_ns] PASS 2/2: base24 (exclude 5 + NS) ...")
    m24 = run(runner, OLD5 + [NS])

    km, v25 = get(m25, "overall/macro_mAP")
    _, v24 = get(m24, "overall/macro_mAP")
    kt, t25 = get(m25, "organ/Thyroid_gland/mAP", "organ/Thyroid gland/mAP")
    _, t24 = get(m24, "organ/Thyroid_gland/mAP", "organ/Thyroid gland/mAP")

    print("\n" + "=" * 60)
    print("NS-EXCLUSION IMPACT  (", args.checkpoint.split("/")[-1], ")")
    print("=" * 60)
    print(f"  Thyroid organ AP : base25(incl NS)={t25}  ->  base24(excl NS)={t24}")
    print(f"  HEADLINE 5-organ macro: base25(OLD)={v25}  ->  base24(NEW)={v24}")
    if isinstance(v24, (int, float)) and isinstance(v25, (int, float)):
        print(f"  delta (NEW - OLD): {v24 - v25:+.4f}")
    print("=" * 60)
    print("[full base25 dict]", {k: m25[k] for k in m25 if "overall" in k or "organ/" in k})
    print("[full base24 dict]", {k: m24[k] for k in m24 if "overall" in k or "organ/" in k})


if __name__ == "__main__":
    main()
