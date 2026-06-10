"""Fail-fast guard: the data_preprocessor normalization must match the backbone.

ROOT CAUSE this prevents (2026-06-03): the BiomedCLIP / DINOv3 ViT backbones do NO
internal normalization -- they rely entirely on the detector's `data_preprocessor`.
Every ViT config inherited the base `YOLOWDetDataPreprocessor(mean=[0,0,0],
std=[255,255,255])`, so the pretrained ViT silently saw [0,1] pixels instead of its
expected CLIP/ImageNet normalization. This degraded every ViT-backbone experiment and
confounded the "ViT backbone is a dead end" conclusion (the ViTs were fed wrong inputs
while the ConvNeXt baseline, pretrained on [0,1], was fed correctly).

This hook makes that class of bug impossible to repeat silently: any normalization-
sensitive backbone declares `EXPECTED_PIXEL_MEAN` / `EXPECTED_PIXEL_STD` (0-255 scale)
as class attributes; at `before_run` we compare them to the active data_preprocessor's
mean/std and `raise RuntimeError` (with the expected vs actual numbers) on mismatch --
so training/eval crashes at step 0, not after hours of wrong-input training.

USAGE (config):
    custom_imports = dict(imports=["wedetect"], allow_failed_imports=False)
    custom_hooks = [dict(type="CheckBackboneNormHook")]

Backbones WITHOUT EXPECTED_PIXEL_MEAN (e.g. the [0,1]-pretrained ConvNeXt baseline)
are skipped -- the hook only guards backbones that explicitly declare an expectation.
"""
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmdet.registry import HOOKS


@HOOKS.register_module()
class CheckBackboneNormHook(Hook):
    """Assert data_preprocessor mean/std match the backbone's expected pixel stats.

    Args:
        atol: Per-channel absolute tolerance (0-255 scale) before flagging a
            mismatch. Default 1.0 (catches the [0,1] default vs ~122 CLIP mean).
    """

    priority = "VERY_HIGH"  # run before training starts, before any heavy setup

    def __init__(self, atol: float = 1.0) -> None:
        self.atol = float(atol)

    def before_run(self, runner) -> None:
        model = getattr(runner, "model", None)
        model = getattr(model, "module", model)  # unwrap DDP
        if model is None:
            return

        # Locate the (optionally nested) image backbone that declares an expectation.
        backbone = getattr(model, "backbone", None)
        image_model = getattr(backbone, "image_model", backbone)
        exp_mean = getattr(image_model, "EXPECTED_PIXEL_MEAN", None)
        exp_std = getattr(image_model, "EXPECTED_PIXEL_STD", None)
        if exp_mean is None or exp_std is None:
            # Backbone does not declare normalization expectations (e.g. ConvNeXt
            # baseline pretrained on [0,1]) -> nothing to check.
            return

        dp = getattr(model, "data_preprocessor", None)
        if dp is None:
            raise RuntimeError(
                "CheckBackboneNormHook: model has no data_preprocessor, cannot "
                f"verify normalization for {type(image_model).__name__}."
            )
        if not getattr(dp, "_enable_normalize", False):
            raise RuntimeError(
                f"CheckBackboneNormHook: {type(image_model).__name__} expects "
                f"normalization mean={tuple(exp_mean)} std={tuple(exp_std)} (0-255 "
                "scale) but the data_preprocessor has normalization DISABLED "
                "(it feeds raw pixels). Add a data_preprocessor override."
            )

        act_mean = dp.mean.detach().flatten().tolist()
        act_std = dp.std.detach().flatten().tolist()
        bad = []
        for ch, (a, e) in enumerate(zip(act_mean, exp_mean)):
            if abs(a - e) > self.atol:
                bad.append(f"mean[{ch}]={a:.3f} (expected {e:.3f})")
        for ch, (a, e) in enumerate(zip(act_std, exp_std)):
            if abs(a - e) > self.atol:
                bad.append(f"std[{ch}]={a:.3f} (expected {e:.3f})")
        if bad:
            raise RuntimeError(
                f"CheckBackboneNormHook: data_preprocessor normalization does NOT "
                f"match {type(image_model).__name__}'s pretrained stats (0-255 scale). "
                f"Mismatches: {', '.join(bad)}. "
                f"Expected mean={tuple(round(x,3) for x in exp_mean)}, "
                f"std={tuple(round(x,3) for x in exp_std)}; "
                f"got mean={[round(x,3) for x in act_mean]}, "
                f"std={[round(x,3) for x in act_std]}. "
                "Fix the config's model.data_preprocessor (likely it inherited the "
                "[0,1] default mean=[0,0,0]/std=[255,255,255])."
            )

        print_log(
            f"[CheckBackboneNormHook] OK: {type(image_model).__name__} normalization "
            f"matches (mean≈{[round(x,1) for x in act_mean]}, "
            f"std≈{[round(x,1) for x in act_std]}).",
            logger="current",
        )
