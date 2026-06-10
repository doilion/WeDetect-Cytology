"""DINOv3 ViT-S/B/L visual backbone for wedetect.

Wraps a timm vit_{small,base,large}_patch16_dinov3 into a wedetect backbone that outputs
a 4-tuple matching the ConvNeXt-Tiny interface (96/192/384/768 channels at stride 4/8/16/32).
This lets CSPRepBiFPANNeck and YOLOWorldHeadModule consume DINOv3 features with ZERO
downstream changes (drop-in replacement at the image_model slot).

Shape adaptation = ViT-Det style simple feature pyramid (Li et al. 2022,
"Exploring Plain Vision Transformer Backbones for Object Detection"):
  ViT produces single-scale stride-16 patch tokens (B, N=H*W/16/16, D=embed_dim).
  After cls + register-token strip, reshape to (B, D, H/16, W/16).
  Stride 4  (c1):  2x deconv -> GroupNorm -> GELU -> 2x deconv  ->  1x1 conv (D -> c1)
  Stride 8  (c2):  2x deconv                                    ->  1x1 conv (D -> c2)
  Stride 16 (c3):  identity                                     ->  1x1 conv (D -> c3)
  Stride 32 (c4):  2x MaxPool                                   ->  1x1 conv (D -> c4)

ViT itself defaults to frozen (audit verdict: frozen linear probe already gives bbox
fine-30 macro F1 = 0.534 for ViT-S/16; the pyramid + projections are trainable).

Weights load from a local HF safetensors via hf_to_timm_vit_dinov3 (no internet needed).
The wedetect env (torch 2.1.0+cu118 / transformers 4.49 / timm 1.0.22) cannot use HF
AutoModel for DINOv3 (transformers >= 4.56 requires torch >= 2.2). See
feedback_dinov3_wedetect_only_timm_path.md memory.
"""
import logging
import math
from typing import Optional, Sequence, Tuple

import timm
import torch
import torch.nn as nn
from torch import Tensor

from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig

from wedetect.models.backbones._dinov3_loaders import hf_to_timm_vit_dinov3


class LoRALinear(nn.Module):
    """Drop-in wrapper adding a trainable low-rank update to a frozen nn.Linear.

    y = base(x) + (alpha / r) * (x A^T) B^T ,  A:(r,in)  B:(out,r).
    B is zero-init so the adapter starts as a no-op (output == base at step 0);
    the base Linear (weight + bias) is frozen, so only A/B (a tiny fraction of the
    layer's params) train. Forward is mode-independent (no dropout/BN), so the
    base ViT can stay in eval() while these adapters still receive gradients.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


@MODELS.register_module()
class DINOv3VisionBackbone(BaseModule):
    """DINOv3 ViT backbone with ViT-Det simple feature pyramid.

    Args:
        timm_model_name: A timm registry name for a DINOv3 ViT (e.g.
            'vit_small_patch16_dinov3' / 'vit_base_patch16_dinov3' /
            'vit_large_patch16_dinov3'). ViT-S/16 is the audit ROI pick.
            Any timm name that has an `embed_dim` attribute after
            create_model() is accepted; the embed dim drives the pyramid's
            input channel count.
        pretrained_weights: Local path to facebook/dinov3-vit* HF safetensors. Required
            for any non-toy use (without it the ViT is random init).
        img_size: Input image size; sets the timm pos_embed length and patch grid.
            Default 640 to match wedetect img_scale.
        out_channels: 4-tuple matching ConvNeXt-Tiny stage dims so the existing neck/head
            don't need any change. Default (96, 192, 384, 768).
        frozen: If True (default), the ViT runs in eval() with requires_grad=False --
            only the pyramid + 1x1 projection convs train.
        frozen_blocks: ViT-Det partial fine-tune. See _frozen_vit_submodules.
        lora_rank: If > 0, enable LoRA PEFT: the whole ViT base is frozen and a
            trainable low-rank adapter (rank=lora_rank) is injected into each
            block's `lora_targets` linears; only the adapters + pyramid + neck/head
            train. Mutually exclusive with full / partial fine-tune (the base stays
            frozen). 0 (default) = no LoRA.
        lora_alpha: LoRA scaling numerator (effective scale = alpha/rank). Defaults
            to lora_rank (scale 1.0).
        lora_targets: Per-block submodule paths to wrap with LoRA. Defaults to all
            four linears (attn.qkv, attn.proj, mlp.fc1, mlp.fc2).
        init_cfg: Standard mmengine init_cfg (unused here; weight loading goes via
            pretrained_weights to keep the HF -> timm key mapper in one place).
    """

    # P2 guard: pixel stats (0-255 scale) DINOv3 weights expect = ImageNet normalization.
    # CheckBackboneNormHook verifies the active data_preprocessor matches these so a
    # config that forgets the override (silently feeding [0,1]) fails fast at train start.
    # BiomedCLIPVisionBackbone overrides these with OpenAI/OpenCLIP stats.
    EXPECTED_PIXEL_MEAN = (123.675, 116.280, 103.530)  # (0.485,0.456,0.406)*255
    EXPECTED_PIXEL_STD = (58.395, 57.120, 57.375)      # (0.229,0.224,0.225)*255

    def __init__(
        self,
        timm_model_name: str = "vit_small_patch16_dinov3",
        pretrained_weights: Optional[str] = None,
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
        super().__init__(init_cfg=init_cfg)

        out_channels = tuple(out_channels)
        if len(out_channels) != 4:
            raise ValueError(
                f"out_channels must have exactly 4 entries (c1..c4), got {out_channels}"
            )

        self.timm_model_name = timm_model_name
        self.img_size = img_size
        self.out_channels = out_channels
        # frozen_blocks (ViT-Det partial fine-tune, 2026-05-31): None = legacy
        # behaviour driven by `frozen` (True=whole ViT frozen, False=full train);
        # K (int>0) = freeze patch_embed + prefix tokens + blocks[:K], train
        # blocks[K:] + final norm + pyramid; 0 = freeze nothing.
        self.frozen = frozen
        self.frozen_blocks = frozen_blocks
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha) if lora_alpha is not None else float(lora_rank)
        self.lora_targets = tuple(lora_targets)

        # 1) Build the timm ViT shell. img_size sets the RoPE length / patch grid.
        self.vit = timm.create_model(
            timm_model_name,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
        )
        # Read embed_dim and patch grid from the built model so we stay aligned with
        # any timm upstream change (e.g. a new variant or a model_name typo would
        # surface here as a clean AttributeError instead of a silent hardcoded mismatch).
        embed_dim = int(self.vit.embed_dim)
        self.embed_dim = embed_dim
        # cls (1) + register_tokens (4) for DINOv3 ViT-S/B/L. Read from the model
        # if available to stay robust to upstream changes.
        self.num_prefix_tokens = int(getattr(self.vit, "num_prefix_tokens", 5))
        # Cache the patch grid side. timm exposes (H/P, W/P) on patch_embed; for
        # the wedetect contract (square img_size), Hp == Wp. Falling back to the
        # arithmetic answer keeps the class robust if patch_embed.grid_size moves.
        grid = getattr(self.vit.patch_embed, "grid_size", None)
        if grid is not None:
            gh, gw = (grid if isinstance(grid, tuple) else (grid, grid))
            if gh != gw:
                raise ValueError(
                    f"DINOv3VisionBackbone supports square inputs only; got patch grid {grid}"
                )
            self.patch_side = int(gh)
        else:
            self.patch_side = img_size // 16

        # Optional multi-scale (2026-06-01): build each detection level from a
        # DIFFERENT ViT block (early -> fine spatial, late -> coarse) instead of
        # replicating the single last-layer map. Opt-in; None = original behaviour.
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

        # 2) Load DINOv3 SSL weights from local HF safetensors (no network).
        if pretrained_weights is not None:
            self._load_pretrained_weights(pretrained_weights)
        # CRITICAL: mark the timm ViT as already initialized so mmengine
        # BaseModule.init_weights() cascade (called by Runner._init_model_weights
        # at train-time) does NOT recurse into self.vit and call timm's
        # VisionTransformer.init_weights(), which would trunc_normal_ every
        # Linear and wipe the SSL weights we just loaded.
        self.vit.is_init = True
        self.vit._is_init = True  # mmengine versions differ in attr name
        # Optional activation checkpointing — halves ViT activation memory so
        # full fine-tune @640 (esp. ViT-B) fits on a 24GB card. timm ViTs honor
        # this in both forward_features and forward_intermediates.
        if grad_checkpointing:
            self.vit.set_grad_checkpointing(True)

        # 3) Simple feature pyramid (ViT-Det). All these layers are trainable.
        #
        # Efficiency trade-off (cf. /code-review xhigh findings #12/#13): the
        # stride-4 path upsamples at the full embed_dim before projecting down,
        # and proj_c3/proj_c4 may be near-identity when out_channels[i] == embed_dim
        # (ViT-S c3=384=embed_dim; ViT-B c4=768=embed_dim). A
        # reduce-then-upsample variant would save ~31x trainable params, but with
        # `frozen=True` the pyramid is the ONLY trainable capacity for the image
        # path, and we want maximum expressive headroom for the detection-time
        # adaptation. We intentionally keep the higher-param ViT-Det template.
        c1, c2, c3, c4 = self.out_channels
        # Stride 4: two stacked 2x deconvs (effective 4x upsample) + 1x1 projection.
        self.up_stride4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=32, num_channels=embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.proj_c1 = nn.Conv2d(embed_dim, c1, kernel_size=1)
        # Stride 8: single 2x deconv + 1x1 projection.
        self.up_stride8 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.proj_c2 = nn.Conv2d(embed_dim, c2, kernel_size=1)
        # Stride 16: identity + 1x1 projection.
        self.proj_c3 = nn.Conv2d(embed_dim, c3, kernel_size=1)
        # Stride 32: 2x maxpool + 1x1 projection.
        self.down_stride32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.proj_c4 = nn.Conv2d(embed_dim, c4, kernel_size=1)

        # 4) LoRA injection (after SSL weights are loaded, so the wrapped base
        # Linears keep their DINOv3 weights). Done before _apply_freeze so the
        # adapters exist when we (re-)enable their grads.
        if self.lora_rank > 0:
            self._inject_lora()

        # 5) Apply ViT freezing (whole / partial / none / LoRA-base) per the flags.
        self._apply_freeze()

    def _inject_lora(self) -> None:
        """Wrap each block's lora_targets linears with a LoRALinear adapter."""
        n = 0
        for blk in self.vit.blocks:
            for tgt in self.lora_targets:
                parent = blk
                parts = tgt.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p, None)
                    if parent is None:
                        break
                if parent is None:
                    continue
                attr = parts[-1]
                lin = getattr(parent, attr, None)
                if isinstance(lin, nn.Linear):
                    setattr(parent, attr, LoRALinear(lin, self.lora_rank, self.lora_alpha))
                    n += 1
        print_log(
            f"[DINOv3VisionBackbone:{self.timm_model_name}] injected LoRA "
            f"(rank={self.lora_rank}, alpha={self.lora_alpha}, scale={self.lora_alpha/self.lora_rank:.2f}) "
            f"into {n} linears across {len(self.vit.blocks)} blocks "
            f"targets={list(self.lora_targets)}",
            logger="current",
        )

    def _load_pretrained_weights(self, pretrained_path: str) -> None:
        from safetensors.torch import load_file

        if not pretrained_path.endswith(".safetensors"):
            raise ValueError(
                f"pretrained_weights must be a .safetensors path, got {pretrained_path}"
            )
        hf_state = load_file(pretrained_path)
        mapped, warns = hf_to_timm_vit_dinov3(hf_state)
        result = self.vit.load_state_dict(mapped, strict=False)
        n_loaded = len(mapped)
        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)
        print_log(
            f"[DINOv3VisionBackbone:{self.timm_model_name}] loaded {n_loaded} keys "
            f"from {pretrained_path}; missing={n_missing} unexpected={n_unexpected}; "
            f"mapper_warns={len(warns)}",
            logger="current",
        )
        if n_missing > 0 or n_unexpected > 0:
            print_log(
                f"  missing[:5]={result.missing_keys[:5]}  "
                f"unexpected[:5]={result.unexpected_keys[:5]}",
                logger="current",
                level=logging.WARNING,
            )

    def init_weights(self) -> None:
        """Skip the timm ViT in the mmengine BaseModule.init_weights() cascade.

        Runner._init_model_weights() walks the model tree and calls init_weights()
        on every child that has it and is not marked is_init=True. timm's
        VisionTransformer.init_weights() does trunc_normal_ + zero-bias which
        would silently wipe the DINOv3 SSL weights loaded in __init__. We mark
        the ViT as already-initialized (both legacy and current sentinel attrs)
        and short-circuit super().init_weights() so the pyramid layers (which are
        plain nn modules without init_weights()) get torch defaults via their
        constructors and the ViT is left untouched.
        """
        # Defensive: re-assert sentinels in case anything cleared them after __init__.
        self.vit.is_init = True
        self.vit._is_init = True
        # We intentionally do NOT call super().init_weights() to avoid the cascade
        # re-entering self.vit even if a future mmengine version stops honoring the
        # is_init guard. Pyramid layers were initialized at construction via
        # nn.ConvTranspose2d / nn.Conv2d / nn.GroupNorm defaults; no further
        # init pass needed.
        self._is_init = True

    def _frozen_vit_submodules(self):
        """ViT submodules to keep frozen (eval + requires_grad=False).

        LoRA (lora_rank>0) -> whole ViT (base runs frozen/eval; adapter grads are
                             re-enabled in _apply_freeze after this blanket freeze).
        frozen_blocks is None -> legacy: whole ViT if `frozen` else nothing.
        frozen_blocks <= 0     -> nothing (full fine-tune).
        frozen_blocks == K > 0 -> patch_embed + blocks[:K] (prefix-token
                                  Parameters are frozen separately in _apply_freeze).
        """
        if self.lora_rank > 0:
            return [self.vit]
        if self.frozen_blocks is None:
            return [self.vit] if self.frozen else []
        if self.frozen_blocks <= 0:
            return []
        return [self.vit.patch_embed] + list(self.vit.blocks)[: self.frozen_blocks]

    def _apply_freeze(self) -> None:
        for m in self._frozen_vit_submodules():
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        # prefix tokens / pos_embed are nn.Parameter (not Modules) -> freeze when
        # partial-freezing the stem.
        if self.frozen_blocks is not None and self.frozen_blocks > 0:
            for attr in ("cls_token", "reg_token", "register_tokens", "pos_embed"):
                t = getattr(self.vit, attr, None)
                if isinstance(t, nn.Parameter):
                    t.requires_grad = False
        # LoRA: the blanket freeze above also froze the adapters (they live inside
        # self.vit) -> re-enable ONLY the low-rank A/B so the SSL base stays frozen
        # while the adapters train.
        if self.lora_rank > 0:
            for mod in self.vit.modules():
                if isinstance(mod, LoRALinear):
                    mod.lora_A.requires_grad = True
                    mod.lora_B.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the frozen ViT submodules in eval mode when the model trains."""
        super().train(mode)
        for m in self._frozen_vit_submodules():
            m.eval()
        return self

    def _forward_multiscale(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Multi-level pyramid: each detection scale from a DIFFERENT ViT block
        (early -> fine spatial / low semantic, late -> coarse / high semantic),
        instead of replicating the single last-layer map. Opt-in via
        multiscale_layers. Frozen blocks contribute features but no grad; unfrozen
        blocks (partial-FT) pass grad through their selected level."""
        # timm builds DINOv3 ViTs as the `Eva` class, which exposes
        # forward_intermediates (NOT get_intermediate_layers). output_fmt='NCHW'
        # returns prefix-stripped [B, embed_dim, side, side] maps, one per index.
        outs = self.vit.forward_intermediates(
            image, indices=list(self.multiscale_layers),
            norm=True, output_fmt="NCHW", intermediates_only=True,
        )
        if len(outs) != 4:
            raise RuntimeError(
                f"get_intermediate_layers returned {len(outs)} maps, expected 4 "
                f"(multiscale_layers={self.multiscale_layers})")
        m1, m2, m3, m4 = outs
        c1 = self.proj_c1(self.up_stride4(m1))      # stride 4   <- earliest block
        c2 = self.proj_c2(self.up_stride8(m2))      # stride 8
        c3 = self.proj_c3(m3)                        # stride 16
        c4 = self.proj_c4(self.down_stride32(m4))   # stride 32  <- latest block
        return c1, c2, c3, c4

    def _assert_input_normalized(self, image: Tensor) -> None:
        """One-time fail-fast net for the 2026-06-03 [0,1]-normalization bug.

        CLIP/ImageNet-pretrained ViTs need NORMALIZED input (mean subtracted, /std),
        whose values span negatives and >1. If the whole batch sits inside [0,1] the
        data_preprocessor normalization is missing and the pretrained features are
        silently wrong. This lives in the backbone (not a hook) so it CANNOT be lost
        to mmengine's custom_hooks list-replacement when a child config overrides it.
        Deterministic: correctly-normalized input always has values outside [0,1].
        """
        if getattr(self, "_input_norm_checked", False):
            return
        self._input_norm_checked = True
        img = image.detach()
        lo, hi = float(img.min()), float(img.max())
        if lo >= -1e-3 and hi <= 1.0 + 1e-3:
            raise RuntimeError(
                f"{type(self).__name__}: input appears UNNORMALIZED (range "
                f"[{lo:.4f},{hi:.4f}] ⊂ [0,1]). This backbone expects normalized "
                f"input (EXPECTED_PIXEL_MEAN/STD); the data_preprocessor is likely "
                f"still the [0,1] default (mean=[0,0,0]/std=[255,255,255]). Add the "
                f"correct data_preprocessor override (CLIP stats for BiomedCLIP, "
                f"ImageNet for DINOv3). See CheckBackboneNormHook / P2 fix."
            )

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        self._assert_input_normalized(image)
        if self.multiscale_layers is not None:
            return self._forward_multiscale(image)
        # timm vit_dinov3 forward_features returns (B, num_prefix + N_patch, embed_dim)
        feats = self.vit.forward_features(image)
        # Strip cls + register tokens -> (B, N, D)
        patch_tokens = feats[:, self.num_prefix_tokens:, :]
        B, N, D = patch_tokens.shape
        # patch grid side cached at __init__ from vit.patch_embed.grid_size --
        # avoids per-forward sqrt + perfect-square check.
        side = self.patch_side
        expected = side * side
        if N != expected:
            raise RuntimeError(
                f"patch tokens N={N} != expected side*side={expected} "
                f"(img_size={self.img_size}, side={side}). The input image was "
                f"not the configured img_size."
            )
        # (B, N, D) -> (B, D, side, side)  (stride 16)
        feat_map = patch_tokens.transpose(1, 2).reshape(B, D, side, side).contiguous()

        # ViT-Det simple feature pyramid -> 4-tuple matching ConvNeXt-Tiny shape contract.
        c1 = self.proj_c1(self.up_stride4(feat_map))      # (B, c1, side*4, side*4)  stride 4
        c2 = self.proj_c2(self.up_stride8(feat_map))      # (B, c2, side*2, side*2)  stride 8
        c3 = self.proj_c3(feat_map)                        # (B, c3, side,   side)   stride 16
        c4 = self.proj_c4(self.down_stride32(feat_map))   # (B, c4, side/2, side/2)  stride 32
        return c1, c2, c3, c4
