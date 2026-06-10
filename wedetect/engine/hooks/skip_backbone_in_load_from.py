"""mmengine hook that strips a configurable prefix from `load_from` checkpoints.

Designed for the DINOv3 backbone integration but generally useful: when a new
image_model class loads its own SSL weights at __init__ (e.g. DINOv3 ConvNeXt
or ViT-S/16 from HF safetensors) and the parent config sets
`load_from='checkpoints/wedetect_tiny.pth'` to warm-start neck/head/text-model,
the runner's default behavior would either

  (a) overwrite the SSL backbone weights (Stage A case: same key namespace as
      the legacy ConvNeXt baseline), or
  (b) emit ~180 unexpected-key warnings while still partial-loading the legacy
      neck/head (Stage B case: different image_model key namespace).

This hook intercepts load_from at the `after_load_checkpoint` lifecycle and
removes every state_dict key under `backbone.image_model.` (configurable), so
the SSL backbone weights survive AND only the warm-startable submodules
(neck.*, bbox_head.*, backbone.text_model.*) come from the checkpoint.

NOTE (2026-05-30): mmengine 0.10.7 has NO `before_load_checkpoint` lifecycle
(Hook only exposes `after_load_checkpoint` / `before_save_checkpoint`, and
`Runner.load_checkpoint` calls `after_load_checkpoint` at runner.py:2130 — AFTER
reading the .pth into a dict but BEFORE `_load_checkpoint_to_model` applies it at
:2137). An earlier version of this hook implemented `before_load_checkpoint`,
which Runner never calls, so it was dead code and the DINOv3 SSL weights got
silently overwritten by the ImageNet wedetect_tiny.pth backbone. The method name
below MUST stay `after_load_checkpoint`. cf. memory
feedback_mmengine_pretrained_load_pitfalls + docs/results/problem_analysis_20260530.md.

USAGE:
  In a config, add to `custom_hooks` AND keep the parent's `load_from`:

    custom_imports = dict(imports=["wedetect"], allow_failed_imports=False)
    custom_hooks = [
        dict(type="SkipImageBackboneInLoadFromHook"),
    ]
    # do NOT set load_from = None — let the parent's wedetect_tiny.pth flow
    # through, the hook will strip backbone.image_model.* before load.

This is the principled depth (cf. /code-review finding #15) for the
load_from-overwrites-SSL-weights landmine. Without this hook, the Stage A/B
configs must use `load_from = None` (as currently set), accepting that
neck/head/text-model train from scratch.
"""
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmdet.registry import HOOKS


@HOOKS.register_module()
class SkipImageBackboneInLoadFromHook(Hook):
    """Strip a prefix from the load_from checkpoint before mmengine applies it.

    Args:
        prefixes: One or more key prefixes to drop from the checkpoint's
            state_dict. Default ('backbone.image_model.',) covers the standard
            wedetect detector layout.
    """

    # Run before any other after_load_checkpoint hooks (none in stock mmengine,
    # but we declare HIGH to be future-proof).
    priority = "HIGH"

    def __init__(
        self,
        prefixes: Sequence[str] = ("backbone.image_model.",),
    ) -> None:
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        if not prefixes:
            raise ValueError(
                "SkipImageBackboneInLoadFromHook: 'prefixes' must be non-empty"
            )
        self.prefixes = tuple(prefixes)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """mmengine calls `after_load_checkpoint` AFTER reading the .pth into a
        dict and BEFORE `_load_checkpoint_to_model` applies it (runner.py:2130
        then :2137). We mutate the state_dict in place so the dropped keys never
        reach the model — the SSL backbone loaded in __init__ survives.
        """
        state_dict = checkpoint.get("state_dict", checkpoint)
        before = len(state_dict)
        to_drop = [
            k
            for k in list(state_dict.keys())
            if any(k.startswith(p) for p in self.prefixes)
        ]
        for k in to_drop:
            del state_dict[k]
        kept = len(state_dict)
        print_log(
            f"[SkipImageBackboneInLoadFromHook] dropped {len(to_drop)} keys with "
            f"prefixes {list(self.prefixes)} from load_from checkpoint "
            f"({before} -> {kept} keys remain).",
            logger="current",
        )
