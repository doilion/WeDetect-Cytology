from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_latest_checkpoint(
    checkpoint: Optional[str],
    work_dir: str | Path,
) -> str:
    """Return ``checkpoint`` if given, else newest ``best_*`` (or ``epoch_*``)
    ckpt under ``work_dir``.

    Search priority (each tier sorted newest-first by mtime):
      1. ``best_coco_overall_macro_mAP_epoch_*.pth`` (corrected-protocol best,
         emitted by trainings whose CheckpointHook uses
         ``save_best='coco/overall/macro_mAP'``: M1, M2, axisstruct, ICF, ...)
      2. ``best_coco_bbox_mAP_epoch_*.pth`` (legacy CocoMetric best,
         emitted by older trainings like clean dev30, noTHAF, THAF, PCW)
      3. ``epoch_*.pth`` (any per-epoch dump as last resort)
    """
    if checkpoint:
        return checkpoint

    work_path = Path(work_dir)
    for pattern in (
        "best_coco_overall_macro_mAP_epoch_*.pth",
        "best_coco_bbox_mAP_epoch_*.pth",
        "epoch_*.pth",
    ):
        candidates = sorted(
            work_path.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError(
        f"No checkpoint found under {work_path}; pass --checkpoint explicitly."
    )


# Custom hooks that only make sense during training and MUST be removed before an
# eval/test run that loads a trained checkpoint via ``cfg.load_from``.
_TRAIN_ONLY_HOOK_TYPES = ("SkipImageBackboneInLoadFromHook",)


def strip_train_only_custom_hooks(cfg) -> list:
    """Drop train-only ``custom_hooks`` from an mmengine eval ``cfg`` (in place).

    Returns the dropped hook type names (for logging); a no-op (``[]``) when none
    are present, so it is safe to call unconditionally in every eval entrypoint.

    Why this exists: ``SkipImageBackboneInLoadFromHook`` strips
    ``backbone.image_model.*`` from ``load_from`` so the warm-start
    ``wedetect_tiny.pth`` cannot overwrite the SSL backbone at TRAIN time. At EVAL,
    ``cfg.load_from`` is the *trained* checkpoint whose ``backbone.image_model.*``
    (the trained ViT-Det pyramid / LoRA adapters / fine-tuned ViT) MUST load in
    full; leaving the hook active strips them and the model evals with a RANDOM
    pyramid -> garbage metrics. DINOv3-ConvNeXt-frozen was accidentally safe (its
    image_model re-loads from ``pretrained_weights`` in ``__init__``); the ViT-S
    ViT-Det pyramid is not. cf. feedback_mmengine_pretrained_load_pitfalls.
    """
    hooks = cfg.get("custom_hooks", None) or []
    kept, dropped = [], []
    for h in hooks:
        htype = h.get("type") if isinstance(h, dict) else getattr(h, "type", None)
        if htype in _TRAIN_ONLY_HOOK_TYPES:
            dropped.append(htype)
        else:
            kept.append(h)
    cfg.custom_hooks = kept
    return dropped
