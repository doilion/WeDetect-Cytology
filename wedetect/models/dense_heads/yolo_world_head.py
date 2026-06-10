# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
import os
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmdet.registry import MODELS
from .yolov8_head import YOLOv8HeadModule, YOLOv8Head
from .utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer


@MODELS.register_module()
class ContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """

    def __init__(self,
                 embed_dims: int,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class BNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """

    def __init__(self,
                 embed_dims: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape   
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class AttributeBNContrastiveHead(BaseModule):
    """Attribute-adaptive contrastive head (module 2, 2026-06-08). Holds the per-attribute
    class text [C, A, D]; BN's the region feature; then runs RegionAttributeFusion (region-
    conditioned attribute reweighting) INSTEAD of the single mean-class cosine. Drop-in for
    BNContrastiveHead: forward(x, w) computes the attribute logit from its OWN attr_text
    buffer (the OVD vocabulary is fixed per config / per eval split); the passed text w is
    used ONLY for the optional B3+D visual-fusion branch (when visual_fuse_weight>0), else
    ignored. So a caller that passes w=None silently disables that branch. At init the fusion's
    gate is zero -> uniform attribute weights -> identical to a mean-attribute BNContrastive
    classifier (regression-safe), then it learns the instance-adaptive weighting."""

    def __init__(self,
                 embed_dims: int,
                 attr_text: Tensor,
                 num_attrs: int,
                 norm_cfg: ConfigDict,
                 attr_dropout: float = 0.1,
                 balance_weight: float = 0.0,
                 adaptive: bool = True,
                 visual_fuse_weight: float = 0.0,
                 attr_text_adapter: nn.Module = None,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        from .region_attribute_fusion import RegionAttributeFusion
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.fusion = RegionAttributeFusion(embed_dims, num_attrs, attr_dropout,
                                            adaptive=adaptive)
        self.balance_weight = float(balance_weight)
        # Combine flavor B (shared-text): an optional learnable residual adapter applied
        # to the per-attribute text `at` ([C,A,D], TextRelationalAdapter acts on the last
        # dim). Zero-init -> identity at start -> regression-safe. Shared across levels (the
        # SAME module instance is passed to every level's head, so grad accumulates once).
        # When set, forward exposes self.last_class_text = normalize(mean_a at_adapted) so
        # the detector's RelationalDistillationLoss supervises the SAME text the region-
        # adaptive head consumes (Module 1 + Module 2 funnel through one per-attribute text).
        # Held in a 1-element list so it is NOT registered as a submodule of THIS head:
        # the adapter is one shared instance, registered once at the head_module level and
        # referenced by all levels. Registering it per level too would list the same params
        # under multiple paths -> optimizer "param appears in more than one parameter group".
        self._attr_text_adapter = [attr_text_adapter] if attr_text_adapter is not None else None
        self.last_class_text = None
        # B3+D logit fusion (B6): when >0, add lambda * <BN(x), grounded_text> to the
        # attribute logit. `grounded_text` = the de-coned class text the detector passes as w
        # (e.g. VisualPrototypeAnchor-grounded; base classes pulled to their visual prototypes,
        # novel rows pass through unchanged = base-only). 0 (default) -> branch skipped ->
        # byte-identical to the module-2/2+1 head. [2026-06-09]
        self.visual_fuse_weight = float(visual_fuse_weight)
        # [C, A, D] per-attribute class text. persistent=False: the text is fixed by the
        # CONFIG (not learned), so it is NOT checkpointed -> a base-trained (30-class)
        # checkpoint loads cleanly into a novel (39-class) eval config, which supplies its
        # own attr_text; the learned params (attr_gate/gamma/beta/BN) are class-agnostic.
        self.register_buffer('attr_text', attr_text.float().contiguous(), persistent=False)
        self.last_w = None   # cached attribute weights for balance loss / visualization

    def forward(self, x: Tensor, w: Tensor = None) -> Tensor:
        """x: [B, embed, H, W]. s_attr uses self.attr_text. w (the de-coned class text the
        detector passes) is consumed ONLY when visual_fuse_weight>0 (B3+D); else ignored."""
        x = self.norm(x)
        # Combine flavor B: learnably align the per-attribute text before fusion, and
        # expose its class-mean for relational-distillation supervision. Identity at init.
        at = self.attr_text
        if self._attr_text_adapter is not None:
            at = self._attr_text_adapter[0](at)                   # [C,A,D], residual
            self.last_class_text = F.normalize(at.mean(dim=1), dim=-1)  # [C,D], carries grad
        s, attn_w = self.fusion(x, at)               # [B,C,H,W], [B,A,H,W]
        # keep the autograd graph only when the balance loss will consume it; otherwise
        # detach so a [B,A,H,W] graph-attached tensor is not retained for nothing.
        self.last_w = attn_w if self.balance_weight > 0 else attn_w.detach()
        # B3+D: logit-fuse the visual-grounded branch. w = the de-coned class text [B,C,D]
        # the detector passes (VisualPrototypeAnchor-grounded). Same contrastive form as
        # BNContrastiveHead. Guard shapes so a mismatched w silently no-ops instead of erroring.
        if (self.visual_fuse_weight > 0 and w is not None and w.dim() == 3
                and w.shape[1] == s.shape[1] and w.shape[2] == x.shape[1]):
            wn = F.normalize(w.to(x.dtype), dim=-1, p=2)             # [B,C,D]
            s_vis = torch.einsum('bchw,bkc->bkhw', x, wn)           # [B,C,H,W]
            s = s + self.visual_fuse_weight * s_vis
        return s * self.logit_scale.exp() + self.bias

    def get_balance_loss(self):
        """KL(mean attribute weight || uniform) from the LAST forward; 0 if disabled."""
        if self.balance_weight <= 0 or self.last_w is None:
            return self.bias.sum() * 0.0
        from .region_attribute_fusion import RegionAttributeFusion
        return RegionAttributeFusion.balance_loss(self.last_w, self.balance_weight)


@MODELS.register_module()
class RepBNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """

    def __init__(self,
                 embed_dims: int,
                 num_guide_embeds: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.conv = nn.Conv2d(embed_dims, num_guide_embeds, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        x = self.conv(x)
        return x


@MODELS.register_module()
class YOLOWorldHeadModule(YOLOv8HeadModule):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
    """

    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 model_size,
                 attr_fusion=None,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        self.model_size = model_size
        # Module 2 opt-in (2026-06-08): attribute-adaptive classifier. dict with
        #   attr_text_path: *.pth holding {'emb': [C,A,D]} or a raw [C,A,D] tensor
        #   per_attr_whiten: optional dict(topk,alpha,beta,lam) -> module 1 on the text
        #   attr_dropout / balance_weight: module-2 protections
        # None (default) -> standard BNContrastiveHead path, unchanged.
        self.attr_fusion = attr_fusion
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = 256  # Fixed to match pretrained weights


        # change self.in_channels
        #
        # NOTE: historically this HARDCODED the 3 FPN channel widths per
        # model_size, silently overriding whatever in_channels the config
        # passed. That is preserved EXACTLY for the default 3-level path.
        #
        # OPT-IN 4-level (stride-4 / P2) path: when the config requests 4
        # detection levels (num_levels == 4, i.e. featmap_strides has 4
        # entries), we DO NOT override — we respect the config-supplied
        # in_channels verbatim (e.g. [48, 96, 192, 384] for tiny+P2). This
        # keeps the 3-level default byte-for-byte identical while letting the
        # finer head be built from config.
        if self.num_levels == 3:
            if self.model_size == "tiny" or self.model_size == "small":
                self.in_channels = [96, 192, 384] # Small
            if self.model_size == "base":
                self.in_channels = [128, 256, 512] # Base
            if self.model_size == "large":
                self.in_channels = [192, 384, 768] # Larges
        else:
            # 4-level (P2) or any non-default level count: trust config.
            assert len(self.in_channels) == self.num_levels, (
                f'P2/multi-level head: config in_channels '
                f'({self.in_channels}) must have num_levels='
                f'{self.num_levels} entries (one per featmap stride).')

        # Module 2 opt-in: load the per-attribute class text [C, A, D] ONCE (optionally
        # per-attribute de-coned = module 1), shared by every level's attribute head.
        self._attr_text = None
        if self.attr_fusion is not None:
            af = self.attr_fusion
            src = af['attr_text_path']
            if isinstance(src, str):
                pkg = torch.load(src, map_location='cpu', weights_only=False)
                at = pkg['emb'] if isinstance(pkg, dict) and 'emb' in pkg else pkg
            else:
                at = src
            at = torch.as_tensor(at).float()                     # [C, A, D]
            assert at.dim() == 3, f'attr_text must be [C,A,D], got {tuple(at.shape)}'
            assert at.shape[-1] == self.embed_dims, (
                f'attr_text dim {at.shape[-1]} != embed_dims {self.embed_dims}')
            if af.get('per_attr_whiten') is not None:
                from wedetect.models.text_adapters import PerAttrWhitening
                at = PerAttrWhitening(**af['per_attr_whiten'])(at)
            self._attr_text = at
            self._num_attrs = at.shape[1]
            # Combine flavor B: build ONE shared text adapter on the per-attribute text
            # (passed to every level's head -> one set of params, grad accumulates once).
            self._attr_text_adapter = None
            if af.get('attr_text_adapter') is not None:
                self._attr_text_adapter = MODELS.build(af['attr_text_adapter'])

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            if self.attr_fusion is not None:
                self.cls_contrasts.append(
                    AttributeBNContrastiveHead(
                        self.embed_dims,
                        self._attr_text.clone(),
                        self._num_attrs,
                        self.norm_cfg,
                        attr_dropout=self.attr_fusion.get('attr_dropout', 0.1),
                        balance_weight=self.attr_fusion.get('balance_weight', 0.0),
                        adaptive=self.attr_fusion.get('adaptive', True),
                        visual_fuse_weight=self.attr_fusion.get('visual_fuse_weight', 0.0),
                        attr_text_adapter=self._attr_text_adapter))
            elif self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum))
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))


        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        return multi_apply(self.forward_single, img_feats, txt_feats,
                           self.cls_preds, self.reg_preds, self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class RepYOLOWorldHeadModule(YOLOWorldHeadModule):

    def __init__(self,
                 *args,
                 embed_dims: int,
                 num_guide: int,
                 freeze_all: bool = False,
                 **kwargs) -> None:
        super().__init__(*args,
                         embed_dims=embed_dims,
                         use_bn_head=True,
                         use_einsum=False,
                         freeze_all=freeze_all,
                         **kwargs)

        # using rep head
        cls_contrasts = []
        for _ in range(self.num_levels):
            cls_contrasts.append(
                RepBNContrastiveHead(
                    embed_dims=embed_dims,
                    num_guide_embeds=num_guide,
                    norm_cfg=self.norm_cfg
                )
            )
        self.cls_contrasts = nn.ModuleList(cls_contrasts)

    def forward_single(self, img_feat: Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward features from the upstream network."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds

    def forward(self, img_feats: Tuple[Tensor]) -> Tuple[List]:
        assert len(img_feats) == self.num_levels
        return multi_apply(self.forward_single, img_feats, self.cls_preds,
                           self.reg_preds, self.cls_contrasts)


@MODELS.register_module()
class YOLOWorldHead(YOLOv8Head):
    """YOLO-World Head
    """

    def __init__(self, world_size=-1,
                 organ_loss_mask: bool = False,
                 organ_mask_path: Optional[str] = None,
                 organ_disc_loss: Optional[dict] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.world_size = world_size
        # Innovation ① (v1): organ-conditioned hard-negative discriminative loss.
        # opt-in; requires organ_loss_mask=True (reuses per_image_mask as the
        # same-organ group_mask). None (default) -> disabled, head behaves exactly
        # as before (regression-safe).
        self.organ_disc_loss = (
            MODELS.build(organ_disc_loss) if organ_disc_loss is not None else None)
        # Module 1 (OC-HMTA): organ-conditional loss masking.
        # When organ_loss_mask=True, per-image cross-organ class loss is
        # zeroed inside loss_by_feat using self.organ_class_mask indexed by
        # batch_img_metas[b]['organ_id'].
        self.organ_loss_mask_enabled = organ_loss_mask
        # organ_class_mask is registered as a non-persistent buffer so that:
        #   (a) model.cuda() / DDP replication moves it automatically
        #   (b) every rank ends up with the same tensor (loaded in __init__)
        #   (c) it is NOT saved in state_dict (persistent=False) so resumed
        #       runs always re-read the file via config, avoiding stale paths
        if organ_mask_path is not None:
            # Fail fast with a clear message: the buffer is persistent=False so a
            # missing path otherwise only surfaces deep inside the first loss
            # computation (or at resume) with a cryptic error.
            if not os.path.exists(organ_mask_path):
                raise FileNotFoundError(
                    f'organ_mask_path does not exist: {organ_mask_path} '
                    f'(set in config; required when organ_loss_mask=True)')
            mask_pkg = torch.load(organ_mask_path, weights_only=False,
                                  map_location='cpu')
            self.register_buffer('organ_class_mask', mask_pkg['mask'],
                                 persistent=False)            # [C, O]

    def set_organ_class_mask(self, mask):
        """Override or clear the organ_class_mask buffer.

        Used by eval entry scripts to swap the mask after Runner build (e.g.
        base30 mask for base eval vs novel9 mask for novel eval). Pass
        mask=None to disable inference masking (row 1 protocol).

        Note: must be called BEFORE DDP wrapping. The assert catches the case
        where this is invoked on the DDP wrapper instead of the underlying
        module, which would silently mask only rank 0.
        """
        assert not isinstance(self, torch.nn.parallel.DistributedDataParallel), (
            'set_organ_class_mask called on DDP wrapper — would only affect '
            'rank 0. Call on the underlying module (model.module.bbox_head).')
        if 'organ_class_mask' in self._buffers:
            del self._buffers['organ_class_mask']
        if mask is not None:
            self.register_buffer('organ_class_mask', mask, persistent=False)

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""

        outs = self(img_feats, txt_feats)
        # Fast version
        loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats, txt_feats)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        # Module 1 (OC-HMTA): organ-conditional class mask.
        # Build per_image_mask [B, C] in {0, 1} from batch_img_metas BEFORE
        # the assigner so that (a) the assigner ranks anchors using only
        # in-organ class scores (otherwise spurious cross-organ scores could
        # influence positive selection and then get masked out of the loss,
        # leaving them unsupervised), and (b) the same mask is reused for
        # cls-loss masking. Hard-fail on any missing/mismatched input.
        per_image_mask = None
        organ_mask = None
        if getattr(self, 'organ_loss_mask_enabled', False):
            organ_mask = getattr(self, 'organ_class_mask', None)
            if organ_mask is None:
                raise RuntimeError(
                    'organ_loss_mask_enabled=True but head.organ_class_mask is None. '
                    'Set organ_mask_path in the head config or call '
                    'head.set_organ_class_mask(...) before training.')
            if organ_mask.shape[0] != flatten_cls_preds.shape[-1]:
                raise RuntimeError(
                    f'organ_class_mask has {organ_mask.shape[0]} classes but cls_pred '
                    f'has {flatten_cls_preds.shape[-1]}. Regenerate the mask for the '
                    f'training class list (see tools/build_class_organ_mask.py).')
            organ_mask = organ_mask.to(
                device=flatten_cls_preds.device, dtype=flatten_cls_preds.dtype)
            B = flatten_cls_preds.shape[0]
            C = organ_mask.shape[0]
            # Validate + extract organ_ids in one Python pass, then index
            # organ_mask vectorized (cheaper than per-sample mask[b] = ...).
            organ_ids = []
            for b, meta in enumerate(batch_img_metas):
                o = meta.get('organ_id', -1) if isinstance(meta, dict) else -1
                if isinstance(o, torch.Tensor):
                    o = int(o.item())
                # reject bool (int(True)==1 silently maps to organ 1) and None
                if o is None or isinstance(o, bool) or not (0 <= int(o) < organ_mask.shape[1]):
                    raise RuntimeError(
                        f'sample {b} missing valid organ_id in metainfo (got {o!r}); '
                        f'check that OrganExtractor is in the train pipeline and '
                        f'PackDetInputs.meta_keys includes "organ_id".')
                organ_ids.append(int(o))
            organ_ids_t = torch.as_tensor(organ_ids, device=organ_mask.device)
            per_image_mask = organ_mask[:, organ_ids_t].T.contiguous()  # [B, C]

        # Cls scores fed to the assigner. Note: BatchTaskAlignedAssigner only
        # reads pred_scores[b, gt_label[b,g], a] for each (b, g, a) triple
        # (see batch_task_aligned_assigner.py:358-363). Since GT classes are
        # always in-organ for TCT_NGC (single-organ-per-image), cross-organ
        # scores are NEVER consulted by the assigner — pre-masking them would
        # be a mathematical no-op. Loss-side masking alone is sufficient.
        assigner_cls_scores = flatten_cls_preds.detach().sigmoid()

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            assigner_cls_scores, self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls_elem = self.loss_cls(flatten_cls_preds, assigned_scores)  # [B, A, C]

        if per_image_mask is not None:
            # Per-image rescale by C / valid_C keeps the per-image cls-BCE
            # magnitude comparable across organs of different cardinalities
            # (so e.g. Serous 2-class images don't get systematically weaker
            # gradients than TCT_CCD 10-class images). NOTE: only the BCE
            # numerator is rescaled; assigned_scores_sum (the denominator)
            # is naturally restricted to in-organ matches by the assigner,
            # so it is not biased by organ class count.
            C = organ_mask.shape[0]
            valid_C = per_image_mask.sum(dim=1).clamp(min=1.0)            # [B]
            scale = (C / valid_C).view(B, 1, 1)
            loss_cls_elem = loss_cls_elem * per_image_mask.unsqueeze(1) * scale

        loss_cls = loss_cls_elem.sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            # .expand + .contiguous() instead of .repeat: cheaper (no per-elem
            # copy), and avoids a suspected DDP-time aliasing issue where
            # .repeat() produced a tensor whose bool True-count diverged from
            # 4 * num_pos, breaking the downstream reshape([-1, 4]).
            fg_mask_pre_prior = fg_mask_pre_prior.contiguous()
            prior_bbox_mask = (fg_mask_pre_prior.unsqueeze(-1)
                                .expand(-1, -1, 4).contiguous())
            # Hard invariant: True count must equal 4 * num_pos, else reshape
            # to [-1, 4] silently mis-shapes downstream tensors.
            assert prior_bbox_mask.sum().item() == int(num_pos.item()) * 4, (
                f'prior_bbox_mask True={prior_bbox_mask.sum().item()} '
                f'!= 4*num_pos={int(num_pos.item())*4}; '
                f'fg_mask shape={tuple(fg_mask_pre_prior.shape)}')
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        loss_dict = dict(loss_cls=loss_cls * num_imgs * world_size,
                         loss_bbox=loss_bbox * num_imgs * world_size,
                         loss_dfl=loss_dfl * num_imgs * world_size)
        # Innovation ① (v1): organ-conditioned hard-negative discriminative loss.
        # Reuses per_image_mask (built above when organ_loss_mask=True) as the
        # same-organ group_mask; same num_imgs*world_size scaling as the others.
        if self.organ_disc_loss is not None:
            if per_image_mask is None:
                raise RuntimeError(
                    'organ_disc_loss requires organ_loss_mask=True so that '
                    'per_image_mask (the same-organ group_mask) is available.')
            loss_dict['loss_organ_disc'] = self.organ_disc_loss(
                flatten_cls_preds, assigned_scores, fg_mask_pre_prior,
                per_image_mask, avg_factor=assigned_scores_sum) * num_imgs * world_size
        # Module 2: attribute-weight balance loss (KL(mean attr weight || uniform)) from
        # each AttributeBNContrastiveHead's last forward. Added ONLY when balance_weight>0
        # (the regularizer ablation); a no-op for B4/B5 (balance_weight=0). [audit 2026-06-08]
        head_module = getattr(self, 'head_module', None)
        if head_module is not None:
            for i, cc in enumerate(getattr(head_module, 'cls_contrasts', [])):
                if getattr(cc, 'balance_weight', 0.0) > 0:
                    loss_dict[f'loss_attr_balance_{i}'] = (
                        cc.get_balance_loss() * num_imgs * world_size)
        return loss_dict

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)


        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

        # Organ-conditional class mask (inference-time clinical prior).
        # If self.organ_class_mask is set (via config organ_mask_path or via
        # head.set_organ_class_mask), zero cross-organ class scores per-image
        # BEFORE top-k / NMS. Symmetric to training-side hard-fail semantics:
        # mask present but shape-mismatched, or any sample missing organ_id,
        # is an error (silent skip would produce row-1 numbers under what
        # the caller believes is a row-2/3 eval).
        organ_mask = getattr(self, 'organ_class_mask', None)
        if organ_mask is not None:
            if organ_mask.shape[0] != flatten_cls_scores.shape[-1]:
                raise RuntimeError(
                    f'organ_class_mask has {organ_mask.shape[0]} classes but '
                    f'cls_scores has {flatten_cls_scores.shape[-1]}. '
                    f'Pass head.set_organ_class_mask(None) to disable masking, '
                    f'or supply a mask matching the test class list.')
            organ_mask = organ_mask.to(
                device=flatten_cls_scores.device,
                dtype=flatten_cls_scores.dtype)
            # Validate per-sample organ_id + build per_image_mask vectorized.
            organ_ids = []
            for b, meta in enumerate(batch_img_metas):
                o = meta.get('organ_id', -1) if isinstance(meta, dict) else -1
                if isinstance(o, torch.Tensor):
                    o = int(o.item())
                # reject bool (int(True)==1 silently maps to organ 1) and None
                if o is None or isinstance(o, bool) or not (0 <= int(o) < organ_mask.shape[1]):
                    raise RuntimeError(
                        f'sample {b} has invalid organ_id={o!r} in img_meta; '
                        f'check OrganExtractor is in test_pipeline and '
                        f'PackDetInputs.meta_keys includes "organ_id".')
                organ_ids.append(int(o))
            organ_ids_t = torch.as_tensor(organ_ids, device=organ_mask.device)
            per_image_mask = organ_mask[:, organ_ids_t].T.contiguous()  # [B, C]
            # Out-of-place multiply so we never mutate a tensor a future
            # caller might re-read.
            flatten_cls_scores = flatten_cls_scores * per_image_mask.unsqueeze(1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)
    
        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]
        # 8400
        # print(flatten_cls_scores.shape)
        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list
