#!/usr/bin/env python
"""Zero-shot evaluate a checkpoint on a TCT_NGC novel split.

Each split JSON has 2-5 novel categories (cat_ids 21-30) that the dev32 base
model has never seen. We swap PseudoLanguageBackbone's cached embeddings to the
novel prompts and override the test dataloader/evaluator to point at the novel
ann file."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mmdet.utils import register_all_modules

register_all_modules()

from mmengine.config import Config
from mmengine.runner import Runner

from wedetect.utils import resolve_latest_checkpoint, strip_train_only_custom_hooks
from wedetect.utils.data_paths import as_posix_dir, get_tct_ngc_root


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data-root", default=as_posix_dir(get_tct_ngc_root()))
    p.add_argument("--ann-file", required=True,
                   help="relative to --data-root, e.g. annotations/instances_test_main_novel.json")
    p.add_argument("--text-json", required=True)
    p.add_argument("--text-emb", required=True)
    p.add_argument(
        "--attr-text",
        default=None,
        help="ATTRIBUTE-method novel eval: path to the novel per-attribute cache "
        "({'emb': [C,A,D]}, e.g. tct_ngc_morph6_novel9_per_attr_biomedclip.pth). "
        "When given, overrides head.attr_fusion.attr_text_path so the attribute "
        "head outputs C novel channels. The backbone text (--text-emb) is still "
        "swapped for shape consistency but is IGNORED by the attribute head "
        "(visual_fuse_weight=0), so any 9-class class-level cache works there.",
    )
    p.add_argument("--work-dir", required=True)
    p.add_argument(
        "--outfile-prefix",
        default=None,
        help="persist predictions to <prefix>.bbox.json for later score fusion",
    )
    p.add_argument(
        "--metrics-out", default=None,
        help="dump the eval metrics dict (runner.test() return) to this JSON path "
             "-> machine-readable novel per-class + bbox_mAP for offline analysis.")
    p.add_argument(
        "--organ-mask", default="data/texts/tct_ngc_class_organ_mask_novel_merged.pt",
        help="novel organ mask applied at inference (keeps the organ prior, consistent "
             "with base eval). The model carries the 30-class BASE mask from the config; "
             "novel eval must swap it to the matching N-class mask. Auto-disabled if the "
             "file is absent or its class count != the novel split.")
    p.add_argument(
        "--no-mask", action="store_true",
        help="disable inference organ masking for the novel split (ablation / no-prior).")
    return p.parse_args()


def resolve_ann_file(data_root: str, ann_file: str) -> Path:
    ann_path = Path(ann_file)
    if ann_path.is_absolute():
        return ann_path
    return Path(data_root) / ann_path


def derive_metainfo(ann_path: Path) -> tuple[tuple[str, ...], int]:
    with open(ann_path, "r") as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda c: c["id"])
    return tuple(c["name"] for c in cats), len(cats)


def main() -> None:
    args = parse()

    abs_ann = resolve_ann_file(args.data_root, args.ann_file)
    classes, n_classes = derive_metainfo(abs_ann)

    with open(args.text_json, "r") as f:
        n_prompts = len(json.load(f))
    if n_prompts != n_classes:
        raise SystemExit(
            f"prompt count {n_prompts} != novel ann categories {n_classes}; "
            f"build a JSON with one prompt per category in cat_id order."
        )

    cfg = Config.fromfile(args.config)
    cfg.load_from = resolve_latest_checkpoint(args.checkpoint, cfg.work_dir)
    # At eval, load_from is the trained ckpt and must load fully -> drop train-only
    # hooks that would strip backbone.image_model.* (DINOv3 ViT-Det pyramid / LoRA).
    strip_train_only_custom_hooks(cfg)

    # 0) Refuse to run on THAF configs — replacing the text_model with a
    # PseudoLanguageBackbone (step 1 below) would silently discard the
    # trained cross-attention fusion module weights and produce wrong
    # numbers. THAF eval has its own tool that preserves the fusion module.
    text_model_type = cfg.model.backbone.text_model.get("type", "")
    if "Hierarchical" in text_model_type:
        raise SystemExit(
            f"text_model type {text_model_type!r} is a THAF backbone with a "
            f"trained fusion module. Replacing it with PseudoLanguageBackbone "
            f"would discard those weights and yield bogus mAP. Use "
            f"tools/eval_novel_thaf.py instead (no --text-emb arg; the "
            f"trained backbone reads its own attr_emb_cache_path)."
        )

    # 1) Replace text_model with PseudoLanguageBackbone backed by the novel
    # cached embeddings. We do a full replacement (rather than only overriding
    # text_embed_path) because the training config may specify a different
    # backbone family (e.g. PseudoHierarchicalBiomedCLIPLanguageBackbone for
    # THAF) whose constructor doesn't accept `text_embed_path`. The eval
    # cache stores per-class fused vectors keyed by class_name (or by prompt
    # string), which PseudoLanguageBackbone consumes natively.
    # Preserve a learned residual text_adapter (e.g. TextRelationalAdapter from the
    # relational-distillation arms): it is part of the trained weights and must keep
    # applying to the swapped-in novel text, or novel eval would silently use the
    # un-aligned raw text. PseudoLanguageBackbone re-runs the adapter each forward
    # when text_adapter is set, so the novel cached vectors get the learned alignment.
    _orig_adapter = cfg.model.backbone.text_model.get("text_adapter")
    _new_text_model = dict(
        type="PseudoLanguageBackbone",
        text_embed_path=args.text_emb,
    )
    if _orig_adapter is not None:
        _new_text_model["text_adapter"] = _orig_adapter
    cfg.model.backbone.text_model = _new_text_model

    # 1b) ATTRIBUTE method: the attribute head classifies from its OWN per-attr
    # text buffer (head.attr_fusion.attr_text_path), NOT the backbone txt_feats
    # (those are passed as `w` and ignored when visual_fuse_weight=0). So for a
    # novel split we must repoint attr_text_path at the novel per-attr cache; the
    # head then rebuilds with C novel output channels in cat-id order. Any
    # per_attr_whiten in the config re-fits on these C novel rows at build time.
    if args.attr_text:
        af = cfg.model.bbox_head.head_module.get("attr_fusion")
        if af is None:
            raise SystemExit(
                "--attr-text given but this config has no "
                "model.bbox_head.head_module.attr_fusion (not an attribute config)."
            )
        import torch as _torch
        _emb = _torch.load(args.attr_text, map_location="cpu", weights_only=False)
        _at = _emb["emb"] if isinstance(_emb, dict) and "emb" in _emb else _emb
        if int(_at.shape[0]) != n_classes:
            raise SystemExit(
                f"--attr-text has {int(_at.shape[0])} classes but the novel ann "
                f"declares {n_classes}; rebuild the per-attr cache in cat-id order."
            )
        af["attr_text_path"] = args.attr_text

    # 2) Resize test-time class count
    cfg.model.num_test_classes = n_classes
    if "num_classes" in cfg.model.bbox_head.head_module:
        # head_module.num_classes is set to num_training_classes at build time;
        # YOLO-World head uses text_feats[0].shape[0] for actual class count at test,
        # so this override is cosmetic but kept for consistency.
        pass

    # 3) Override dataloader for the novel split
    novel_metainfo = dict(classes=classes)
    cfg.test_dataloader = dict(
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type="MultiModalDataset",
            dataset=dict(
                type="WeCocoDataset",
                data_root=args.data_root,
                test_mode=True,
                ann_file=args.ann_file,
                data_prefix=dict(img="images/"),
                batch_shapes_cfg=None,
                metainfo=novel_metainfo,
            ),
            class_text_path=args.text_json,
            pipeline=cfg.test_pipeline,
        ),
    )

    # 4) Override evaluator
    cfg.test_evaluator = dict(
        type="CocoMetric",
        ann_file=str(abs_ann),
        metric="bbox",
        classwise=True,
    )
    if args.outfile_prefix:
        cfg.test_evaluator["outfile_prefix"] = args.outfile_prefix

    cfg.work_dir = args.work_dir

    runner = Runner.from_cfg(cfg)

    # Organ mask: the model carries the 30-class BASE organ_class_mask (config
    # organ_loss_mask=True), but novel eval has n_classes (e.g. 9) -> inference masking
    # raises "organ_class_mask has 30 classes but cls_scores has N". Swap to the matching
    # novel mask (keeps the organ prior, consistent with base eval) or --no-mask to disable.
    _m = runner.model
    _m = _m.module if hasattr(_m, "module") else _m
    _head = getattr(_m, "bbox_head", None)
    if (_head is not None and hasattr(_head, "set_organ_class_mask")
            and getattr(_head, "organ_class_mask", None) is not None):
        if args.no_mask:
            _head.set_organ_class_mask(None)
            print("[novel] organ mask DISABLED (--no-mask)")
        elif args.organ_mask and os.path.exists(args.organ_mask):
            _pkg = torch.load(args.organ_mask, weights_only=False, map_location="cpu")
            _nmask = _pkg["mask"] if isinstance(_pkg, dict) else _pkg
            # Align mask rows to the eval class order (else the organ prior masks the WRONG
            # classes). Reorder by class_names when available; disable if it can't be aligned.
            _names = list(_pkg["class_names"]) if isinstance(_pkg, dict) and "class_names" in _pkg else None
            if _names is not None and _names != list(classes):
                _idx = {nm: i for i, nm in enumerate(_names)}
                if all(c in _idx for c in classes):
                    _nmask = _nmask[[_idx[c] for c in classes]]
                    print("[novel] reordered organ mask rows to match the eval class order")
                else:
                    print("[novel][warn] organ mask class_names != eval classes (not a "
                          "permutation) -> masking DISABLED")
                    _nmask = None
            if _nmask is None:
                _head.set_organ_class_mask(None)
            elif _nmask.shape[0] != n_classes:
                print(f"[novel][warn] organ mask {args.organ_mask} has {_nmask.shape[0]} "
                      f"classes but novel split has {n_classes} -> masking DISABLED")
                _head.set_organ_class_mask(None)
            else:
                _head.set_organ_class_mask(_nmask)
                print(f"[novel] organ mask -> {args.organ_mask} {tuple(_nmask.shape)} "
                      "(organ prior ON, rows aligned to eval class order)")
        else:
            print(f"[novel] no novel organ mask at {args.organ_mask} -> masking DISABLED")
            _head.set_organ_class_mask(None)

    metrics = runner.test()
    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=float, ensure_ascii=False)
        print(f"[metrics] dumped -> {out}")


if __name__ == "__main__":
    main()
