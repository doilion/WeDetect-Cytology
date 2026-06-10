#!/usr/bin/env python
"""Test script to evaluate model on novel classes.

DEPRECATED for the dev30 line (2026-06-10). The ``novel_metainfo`` below is a
hardcoded 11-class list that INCLUDES the negative ``Thyroid gland-NS`` and
``Thyroid gland-AUC`` (and is missing several real novel classes) — it does NOT
match the current 9-class novel set (``instances_test_novel_merged_9.json``).
The method suite no longer calls this script.

Use the data-driven harness instead, which derives the class set from the ann
file's categories and supports the attribute head via ``--attr-text``:

    PYTHONPATH=. python tools/eval_novel_split.py \\
      --config <cfg> --checkpoint <ckpt> \\
      --data-root "$TCT_NGC_1024_ROOT" \\
      --ann-file annotations/instances_test_novel_merged_9.json \\
      --text-json data/texts/tct_ngc_novel_merged_9.json \\
      --text-emb  data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth \\
      --work-dir  <wd>/novel_eval \\
      [--attr-text data/texts/tct_ngc_morph6_novel9_per_attr_biomedclip.pth]

This file is kept only for the legacy 11-class protocol; do not use it for dev30.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from mmdet.utils import register_all_modules
register_all_modules()

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from wedetect.utils import resolve_latest_checkpoint, strip_train_only_custom_hooks


def assert_text_count_matches(class_text_path: str, expected: int) -> None:
    """Loud failure when the prompt JSON does not 1:1 match the dataset metainfo.

    Without this guard, MMEngine silently feeds N prompts into an M-class evaluator
    (M != N), and CocoMetric scores garbage AP. Hit by the dev32 cache640 default,
    where ``test_class_text_path`` ships 32 fullnames but the novel split has 11
    classes — see audit P1."""
    with open(class_text_path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    if len(groups) != expected:
        raise SystemExit(
            f"prompt count mismatch: {class_text_path} has {len(groups)} entries "
            f"but novel_metainfo declares {expected} classes. "
            f"Pass --text path/to/tct_ngc_novel_{expected}_texts.json with the right count."
        )


def main():
    parser = argparse.ArgumentParser(description='Test on novel classes')
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
    parser.add_argument('--text', default=None,
                        help='class text path (overrides default)')
    parser.add_argument(
        '--data-root',
        default=None,
        help=(
            'override cfg.data_root for the novel split. '
            'Required when the config points at the cache640 dataset, '
            'which only ships train_dev/val_dev (not test_novel_v2.json).'
        ),
    )
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default=None,
        help='override config entries, e.g. swap the attribute head to the 39-class '
             'per-attr text for novel eval, or fit whitening base-only:\n'
             '  model.bbox_head.head_module.attr_fusion.attr_text_path='
             'data/texts/tct_ngc_morph6_39_per_attr_biomedclip.pth\n'
             '  model.bbox_head.head_module.attr_fusion.per_attr_whiten.fit_classes=30')
    args = parser.parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    # Apply --cfg-options early so model/head overrides (e.g. 39-class attr_text_path,
    # whitening fit_classes) take effect before the runner builds the model.
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)
    # At eval, load_from is the trained ckpt and must load fully -> drop train-only
    # hooks that would strip backbone.image_model.* (DINOv3 ViT-Det pyramid / LoRA).
    strip_train_only_custom_hooks(cfg)
    if args.data_root:
        cfg.data_root = args.data_root

    # 定义 novel 类的完整元信息 (与 test_novel_v2.json 中的类别对应)
    novel_metainfo = dict(
        classes=(
            'hsil_scc_omn',
            'monilia',
            'Serous effusion-Ovarian cancer',
            'Serous effusion-adenocarcinoma',
            'Thyroid gland-Suspicious papillary cancer',
            'Thyroid gland-AUC',
            'Thyroid gland-Malignant tumour',
            'Thyroid gland-NS',
            'Urine-HGUC',
            'respiratory tract-Squamous cell cinoma',
            'respiratory tract-Small cell carcinoma',
        )
    )

    class_text_path = (
        args.text
        if args.text
        else cfg.get('test_class_text_path', 'data/texts/tct_ngc_fullnames_32.json')
    )
    assert_text_count_matches(class_text_path, len(novel_metainfo['classes']))

    # 配置 novel 测试数据加载器
    cfg.test_dataloader = dict(
        batch_size=1,  # 使用小 batch size 避免问题
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='MultiModalDataset',
            dataset=dict(
                type='WeCocoDataset',
                data_root=cfg.data_root,
                ann_file='annotations/test_novel_v2.json',
                data_prefix=dict(img=''),
                test_mode=True,
                batch_shapes_cfg=None,
                metainfo=novel_metainfo,
            ),
            class_text_path=class_text_path,
            pipeline=cfg.test_pipeline,
        )
    )

    # 配置评估器
    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=cfg.data_root + 'annotations/test_novel_v2.json',
        metric='bbox',
        classwise=True,
    )

    cfg.work_dir = './work_dirs/test_novel'

    # Build and run
    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    main()
