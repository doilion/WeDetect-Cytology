#!/usr/bin/env python
"""Test script to evaluate model excluding negative classes."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mmdet.utils import register_all_modules
register_all_modules()

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from wedetect.utils import (
    load_dev30_negative_classes,
    resolve_latest_checkpoint,
    strip_train_only_custom_hooks,
)

# Negative/background class names excluded from the reported (base) metric.
# SINGLE SOURCE: data/texts/tct_ngc_dev30_negatives.json. Previously this file
# hardcoded the stale dev32 Urine names (Urine-NILM / Urine-Negative /
# Urine-Negative Degeneration) which match nothing in dev30 -> Urine-NHGUC (and,
# until 2026-06-09, Thyroid gland-NS) silently leaked into the metric.
NEGATIVE_CLASS_NAMES = load_dev30_negative_classes()


def resolve_ann_file(data_root: str, ann_file: str) -> str:
    ann_path = Path(ann_file)
    if ann_path.is_absolute():
        return str(ann_path)
    return str(Path(data_root) / ann_path)


def refresh_evaluator_ann_file(cfg):
    ann_file = cfg.val_dataloader.dataset.dataset.ann_file
    cfg.val_evaluator.ann_file = resolve_ann_file(cfg.data_root, ann_file)
    cfg.test_evaluator = cfg.val_evaluator


def main():
    parser = argparse.ArgumentParser(description='Test with excluded negative classes')
    parser.add_argument(
        '--config',
        default='config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py',
        help='config file',
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='checkpoint file; defaults to latest best checkpoint in config work_dir',
    )
    parser.add_argument('--exclude-negative', action='store_true', default=True,
                        help='exclude negative classes from evaluation')
    parser.add_argument(
        '--metric', choices=('organ', 'exclude'), default='organ',
        help="base metric. 'organ' (default, RECOMMENDED) keeps the config's "
             "OrganRestrictedCocoMetric with classwise=True -> emits the organ-macro "
             "headline (coco/overall/macro_mAP) + per-class AP (coco/class/<slug>/mAP) + "
             "excludes negatives, all in one. 'exclude' = the legacy flat "
             "ExcludeClassCocoMetric (coco/bbox_mAP, no organ-macro).")
    parser.add_argument(
        '--metrics-out', default=None,
        help='dump the eval metrics dict (runner.test() return) to this JSON path '
             '-> machine-readable per-class + headline for offline analysis (no weights).')
    parser.add_argument('--work-dir', default='./work_dirs/test_exclude_negative',
                        help='work dir for evaluation outputs')
    parser.add_argument('--data-root', default=None,
                        help='override cfg.data_root (e.g. $TCT_NGC_DATA_ROOT '
                             'when running held-out test on the non-cache640 host)')
    parser.add_argument('--ann-file', default=None,
                        help='override the test ann_file relative to --data-root '
                             '(e.g. annotations/instances_test_base_clean.json)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default=None,
                        help='override config entries (mmengine standard form). '
                             'Note: applied AFTER the script rebuilds val_evaluator '
                             '(when --exclude-negative is on, default), so cfg-options '
                             'targeting test_evaluator/val_evaluator survive.')
    parser.add_argument('--outfile-prefix', default=None,
                        help='persist predictions to <prefix>.bbox.json (passes through '
                             'to val_evaluator.outfile_prefix; survives the evaluator '
                             'rebuild done by --exclude-negative)')
    parser.add_argument('--exclude-class-names', default=None,
                        help='comma-separated class names that REPLACE the default '
                             'NEGATIVE_CLASS_NAMES list. Use for dev30 where the '
                             "default has stale entries like 'Urine-NILM' that no "
                             "longer exist after the urine 3-阴性 merge.")
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Set checkpoint
    cfg.load_from = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)
    # At eval, load_from is the trained ckpt and must load fully -> drop train-only
    # hooks that would strip backbone.image_model.* (DINOv3 ViT-Det pyramid / LoRA).
    strip_train_only_custom_hooks(cfg)

    if args.data_root:
        cfg.data_root = args.data_root
        cfg.test_dataloader.dataset.dataset.data_root = args.data_root
        cfg.val_dataloader.dataset.dataset.data_root = args.data_root
        refresh_evaluator_ann_file(cfg)
    if args.ann_file:
        cfg.test_dataloader.dataset.dataset.ann_file = args.ann_file
        cfg.val_dataloader.dataset.dataset.ann_file = args.ann_file
        refresh_evaluator_ann_file(cfg)

    # Resolve which negative classes to exclude — default vs CLI override.
    exclude_names = NEGATIVE_CLASS_NAMES
    if args.exclude_class_names:
        exclude_names = [
            n.strip() for n in args.exclude_class_names.split(',') if n.strip()
        ]

    # Select the base metric. DEFAULT 'organ': keep the config's
    # OrganRestrictedCocoMetric and force classwise=True -> one metric gives the
    # organ-macro headline (coco/overall/macro_mAP), per-class AP (coco/class/<slug>/mAP)
    # AND excludes the negatives. The legacy 'exclude' path downgrades to a flat
    # ExcludeClassCocoMetric (no organ-macro) and is kept only for back-compat.
    cfg_metric_type = cfg.val_evaluator.get('type')
    if args.metric == 'organ' and cfg_metric_type == 'OrganRestrictedCocoMetric':
        cfg.val_evaluator['classwise'] = True
        if args.exclude_negative:
            cfg.val_evaluator['exclude_class_names'] = exclude_names
        cfg.test_evaluator = cfg.val_evaluator
    elif args.exclude_negative:
        if args.metric == 'organ':
            print(f"[warn] --metric organ requested but config val_evaluator is "
                  f"{cfg_metric_type!r} (no OrganRestrictedCocoMetric) -> falling back "
                  f"to flat ExcludeClassCocoMetric (no organ-macro headline).")
        cfg.val_evaluator = dict(
            type='ExcludeClassCocoMetric',
            ann_file=cfg.val_evaluator.ann_file,
            metric='bbox',
            exclude_class_id=exclude_names,  # 排除负样本类别
            classwise=True,
        )
        cfg.test_evaluator = cfg.val_evaluator

    # Apply --cfg-options AFTER the evaluator rebuild so user overrides
    # (e.g. test_evaluator.outfile_prefix=...) survive.
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
        # runner.test() reads cfg.test_evaluator; merge_from_dict may have
        # diverged val_evaluator and test_evaluator (mmengine creates fresh
        # ConfigDict on the merged path). Mirror val → test unless the user
        # explicitly targeted test_evaluator.* in --cfg-options.
        targeted_test = any(str(k).startswith("test_evaluator")
                            for k in args.cfg_options)
        if not targeted_test:
            cfg.test_evaluator = cfg.val_evaluator

    # --outfile-prefix takes precedence: write into the (possibly rebuilt) evaluator.
    if args.outfile_prefix:
        cfg.val_evaluator['outfile_prefix'] = args.outfile_prefix
        cfg.test_evaluator = cfg.val_evaluator

    # Set work dir for this test
    cfg.work_dir = args.work_dir

    # Build runner
    runner = Runner.from_cfg(cfg)

    # Run test; runner.test() returns the eval metrics dict (per-class + headline)
    metrics = runner.test()
    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=float, ensure_ascii=False)
        print(f'[metrics] dumped -> {out}')

if __name__ == '__main__':
    main()
