"""BiomedCLIP ViT-B/16 visual backbone for wedetect (aligned dual-tower).

Uses the IMAGE tower of microsoft/BiomedCLIP, whose embedding space is
pretrained-ALIGNED with the BiomedCLIP TEXT tower that wedetect already uses for
class prompts. Motivation (2026-06-01): our analysis shows novel zero-shot is
bottlenecked by the image encoder being mis-aligned with the (BiomedCLIP) text
space (ConvNeXt-ImageNet image vs BiomedCLIP text = two spaces). Using the
aligned biomedical image tower puts image + text in ONE space → should help
novel generalization. The single-scale ViT detection handicap is addressed by
the same multi-scale option as DINOv3VisionBackbone.

The trunk is a standard timm VisionTransformer (ViT-B/16, 768d, 1 cls prefix),
pretrained at 224. We rebuild it at img_size and load the BiomedCLIP weights with
positional-embedding interpolation (14x14 -> H/16 x W/16). Everything downstream
(ViT-Det simple feature pyramid, optional multi-scale via forward_intermediates,
freeze / partial-FT / LoRA) is inherited unchanged from DINOv3VisionBackbone.

Requires open_clip (wedetect env has 3.3.0) + the BiomedCLIP HF cache. The loader
forces HF offline (the model is cached; the hf-mirror proxy is flaky).
"""
import logging
import os
from typing import Optional, Sequence

# P1 fix: force HF offline BEFORE any import that transitively loads huggingface_hub
# (timm / open_clip). huggingface_hub caches these flags at import time, so setting
# them later (e.g. inside _build_and_load, after `import open_clip`) is too late and
# the flaky hf-mirror still gets hit in the offline/cached environment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import timm
from timm.layers import resample_abs_pos_embed

from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig

from wedetect.models.backbones.dinov3_vit import DINOv3VisionBackbone

_BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@MODELS.register_module()
class BiomedCLIPVisionBackbone(DINOv3VisionBackbone):
    """BiomedCLIP image tower as a wedetect backbone (subclasses DINOv3VisionBackbone).

    Only weight loading differs from the parent (open_clip BiomedCLIP + pos-embed
    interpolation, instead of a DINOv3 safetensors + HF->timm mapper). The pyramid,
    multi-scale forward, freezing and init_weights are inherited.

    Args mirror DINOv3VisionBackbone except `biomedclip_name` replaces
    `timm_model_name`/`pretrained_weights`. ViT-B/16 (embed_dim=768, 1 cls prefix).
    """

    # P2 guard: pixel stats (0-255 scale) these weights were pretrained with =
    # OpenAI/OpenCLIP normalization (BiomedCLIP uses it). CheckBackboneNormHook checks
    # the active data_preprocessor against these so a config that forgets the override
    # (silently feeding [0,1]) fails fast at train start instead of training wrong.
    EXPECTED_PIXEL_MEAN = (122.7709, 116.7460, 104.0937)  # (0.48145,0.45783,0.40821)*255
    EXPECTED_PIXEL_STD = (68.5005, 66.6323, 70.3232)      # (0.26863,0.26130,0.27578)*255

    def __init__(
        self,
        biomedclip_name: str = _BIOMEDCLIP_HF,
        img_size: int = 640,
        out_channels: Sequence[int] = (96, 192, 384, 768),
        frozen: bool = True,
        frozen_blocks: Optional[int] = None,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_targets: Sequence[str] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
        multiscale_layers: Optional[Sequence[int]] = None,
        grad_checkpointing: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        # NOTE: bypass DINOv3VisionBackbone.__init__ (which loads safetensors);
        # replicate its structure here with BiomedCLIP loading, then reuse all its
        # other methods (pyramid forward, freeze, multiscale, init_weights).
        BaseModule.__init__(self, init_cfg=init_cfg)

        out_channels = tuple(out_channels)
        if len(out_channels) != 4:
            raise ValueError(
                f"out_channels must have exactly 4 entries (c1..c4), got {out_channels}")

        self.biomedclip_name = biomedclip_name
        self.timm_model_name = "vit_base_patch16_biomedclip"  # cosmetic, for logs
        self.img_size = img_size
        self.out_channels = out_channels
        self.frozen = frozen
        self.frozen_blocks = frozen_blocks
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha) if lora_alpha is not None else float(lora_rank)
        self.lora_targets = tuple(lora_targets)

        # 1) Build a timm ViT-B/16 at img_size and load BiomedCLIP weights.
        self.vit = self._build_and_load(biomedclip_name, img_size)
        embed_dim = int(self.vit.embed_dim)
        self.embed_dim = embed_dim
        self.num_prefix_tokens = int(getattr(self.vit, "num_prefix_tokens", 1))
        grid = getattr(self.vit.patch_embed, "grid_size", None)
        if grid is not None:
            gh, gw = (grid if isinstance(grid, tuple) else (grid, grid))
            if gh != gw:
                raise ValueError(
                    f"BiomedCLIPVisionBackbone supports square inputs only; got grid {grid}")
            self.patch_side = int(gh)
        else:
            self.patch_side = img_size // 16

        # Optional multi-scale (one ViT block per detection level).
        if multiscale_layers is not None:
            ms = tuple(int(i) for i in multiscale_layers)
            if len(ms) != 4:
                raise ValueError(
                    f"multiscale_layers must have 4 block indices (c1..c4), got {ms}")
            depth = len(self.vit.blocks)
            if any(not (0 <= i < depth) for i in ms):
                raise ValueError(
                    f"multiscale_layers {ms} out of range for ViT depth {depth}")
            self.multiscale_layers = ms
        else:
            self.multiscale_layers = None

        # Prevent mmengine init_weights() cascade from re-randomizing the ViT.
        self.vit.is_init = True
        self.vit._is_init = True
        if grad_checkpointing:
            self.vit.set_grad_checkpointing(True)

        # 2) ViT-Det simple feature pyramid (identical to DINOv3VisionBackbone).
        import torch.nn as nn
        c1, c2, c3, c4 = self.out_channels
        self.up_stride4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=32, num_channels=embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.proj_c1 = nn.Conv2d(embed_dim, c1, kernel_size=1)
        self.up_stride8 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.proj_c2 = nn.Conv2d(embed_dim, c2, kernel_size=1)
        self.proj_c3 = nn.Conv2d(embed_dim, c3, kernel_size=1)
        self.down_stride32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj_c4 = nn.Conv2d(embed_dim, c4, kernel_size=1)

        # 3) LoRA (after weights loaded) + freezing — inherited helpers.
        if self.lora_rank > 0:
            self._inject_lora()
        self._apply_freeze()

    def _build_and_load(self, biomedclip_name: str, img_size: int):
        # HF offline is already forced at module import (top of file), before any
        # huggingface_hub import, so create_model_from_pretrained honors the cache.
        import open_clip

        clip, _ = open_clip.create_model_from_pretrained(biomedclip_name)
        src = dict(clip.visual.trunk.state_dict())
        del clip
        vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0, img_size=img_size)
        new_grid = img_size // 16
        if "pos_embed" in src:
            src["pos_embed"] = resample_abs_pos_embed(
                src["pos_embed"], new_size=(new_grid, new_grid),
                num_prefix_tokens=1, verbose=False)
        res = vit.load_state_dict(src, strict=False)
        print_log(
            f"[BiomedCLIPVisionBackbone] loaded BiomedCLIP ViT-B/16 @{img_size} "
            f"(pos_embed 14x14->{new_grid}x{new_grid}); "
            f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
            logger="current")
        if res.missing_keys or res.unexpected_keys:
            print_log(
                f"  missing[:5]={res.missing_keys[:5]} unexpected[:5]={res.unexpected_keys[:5]}",
                logger="current", level=logging.WARNING)
        return vit
