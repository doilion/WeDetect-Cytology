"""Innovation ② (locked 2026-06-03): organ-modulated region-gap distillation.

Root cause it attacks: the detector's region features land OUTSIDE the BiomedCLIP
image-text aligned space, so region<->text similarity is weak (gt_logit ~ -11.2)
and predictions are under-confident. This loss pulls the detector's GT-region
features INTO that aligned space by distilling from a frozen, EXTERNAL BiomedCLIP
image encoder (teacher): for each GT box we crop the region, encode it with the
frozen BiomedCLIP image tower (whose embedding is aligned with the BiomedCLIP TEXT
the head already matches against), and align the detector's RoIAlign-pooled region
feature (via a small learned projection) to that teacher embedding.

Backbone-agnostic by design: the teacher is a SEPARATE frozen BiomedCLIP, so this
works whether the detector backbone is ConvNeXt (Row1) or a BiomedCLIP ViT -- it
does NOT depend on the detector having a ViT crop-CLS path (that is what decouples
② from the C re-test gate).

Differentiation (BIBM extension framing):
- vs Xu 2026 (MCDC/MAPL self-distillation): their teacher is the model's OWN
  decayed similarity matrix (goal = spread classes apart); ours is an EXTERNAL
  frozen medical VLM crop embedding (goal = close the image-region gap). Different
  teacher source AND objective.
- vs ViLD/F-VLM region distillation: those distill from natural-image CLIP at image
  level; ours is medical BiomedCLIP, region-level, organ-modulated, on cytology.
- organ modulation (per-organ loss weight) is ours alone (single-specimen prior
  work has no organ structure).
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

# OpenAI/OpenCLIP normalization (0-1 scale) — BiomedCLIP teacher expects this.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@MODELS.register_module()
class OrganModulatedRegionGapDistill(nn.Module):
    """Region-gap distillation from a frozen external BiomedCLIP image teacher.

    Args:
        student_dim: channel count of the detector feature map RoIAlign pools from
            (= img_feats[roi_level] channels).
        teacher_dim: BiomedCLIP image-embedding dim (512 for ViT-B/16).
        loss_weight: scalar weight on the returned loss.
        crop_size: side length the GT crop is resized to before the teacher (224).
        roi_level: which img_feats pyramid level to RoIAlign from (0 = finest).
        max_rois: cap on GT boxes used per step (subsampled) to bound teacher cost.
            None = use all.
        num_organs: if given, a learnable per-organ log-weight modulates the loss
            (softplus-positive, mean-1 normalized); None = uniform.
        biomedclip_name: open_clip hub id for the teacher.
    """

    def __init__(self,
                 student_dim: int,
                 teacher_dim: int = 512,
                 loss_weight: float = 1.0,
                 crop_size: int = 224,
                 roi_level: int = 0,
                 max_rois: Optional[int] = 256,
                 num_organs: Optional[int] = None,
                 biomedclip_name: str = _BIOMEDCLIP_HF) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.crop_size = crop_size
        self.roi_level = roi_level
        self.max_rois = max_rois
        self.biomedclip_name = biomedclip_name
        self.proj = nn.Linear(student_dim, teacher_dim)
        # optional learnable per-organ modulation (softplus -> positive)
        if num_organs is not None:
            self.organ_logits = nn.Parameter(torch.zeros(num_organs))
        else:
            self.organ_logits = None
        self.register_buffer("_clip_mean",
                             torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_clip_std",
                             torch.tensor(_CLIP_STD).view(1, 3, 1, 1), persistent=False)
        self._teacher_ref = ()  # non-registered (kept out of state_dict / optimizer)

    # ---- teacher management -------------------------------------------------
    def set_teacher(self, image_encoder: nn.Module) -> None:
        """Inject a frozen image encoder (call .encode_image-style forward).

        Stored in a tuple so it is NOT registered as a submodule (excluded from
        state_dict + optimizer). Moved to the data device lazily in compute().
        """
        image_encoder.eval()
        for p in image_encoder.parameters():
            p.requires_grad_(False)
        self._teacher_ref = (image_encoder,)

    def build_and_set_teacher(self) -> None:
        """Load the frozen BiomedCLIP image tower via open_clip (HF offline is
        forced at biomedclip_vit import time)."""
        import open_clip
        clip, _ = open_clip.create_model_from_pretrained(self.biomedclip_name)
        self.set_teacher(clip.visual)  # .visual returns the projected image embed

    @property
    def has_teacher(self) -> bool:
        return len(self._teacher_ref) == 1

    # ---- core (pure tensors, unit-testable) ---------------------------------
    def _distill(self, student_feats: torch.Tensor, teacher_emb: torch.Tensor,
                 organ_ids: Optional[torch.Tensor] = None,
                 avg_factor: Optional[float] = None) -> torch.Tensor:
        """1 - cos(proj(student), teacher), organ-weighted. teacher_emb is detached."""
        if student_feats.shape[0] == 0:
            return self.proj.weight.sum() * 0.0  # zero loss, keep grad path
        s = F.normalize(self.proj(student_feats), dim=-1)
        t = F.normalize(teacher_emb.detach().float(), dim=-1)
        per = 1.0 - (s * t).sum(dim=-1)  # [N]
        if organ_ids is not None and self.organ_logits is not None:
            w = F.softplus(self.organ_logits)
            w = w / w.mean().clamp(min=1e-6)        # mean-1 normalized
            per = per * w[organ_ids]
        denom = avg_factor if avg_factor is not None else per.shape[0]
        return self.loss_weight * per.sum() / max(float(denom), 1.0)

    # ---- full plumbing (RoIAlign + crop + teacher) --------------------------
    @torch.no_grad()
    def _teacher_encode(self, crops: torch.Tensor) -> torch.Tensor:
        teacher = self._teacher_ref[0]
        if next(teacher.parameters()).device != crops.device:
            teacher.to(crops.device)
        # fp16 autocast on CUDA: the frozen teacher is inference-only, so half
        # precision ~2x-faster with negligible effect on a distillation target.
        # Always return fp32 so the downstream cos / proj math is stable.
        if crops.is_cuda:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                emb = teacher(crops)
        else:
            emb = teacher(crops)
        return emb.float()

    def compute(self, img_feats, batch_inputs, batch_gt_instances,
                batch_organ_ids, det_mean, det_std) -> dict:
        """Build (student_feats, teacher_emb, organ_ids) then return loss dict.

        Args:
            img_feats: tuple of detector feature maps (neck output).
            batch_inputs: [B,3,H,W] preprocessor-normalized images (RGB).
            batch_gt_instances: list of InstanceData with .bboxes (xyxy, input-pixel
                coords) — from unpack_gt_instances.
            batch_organ_ids: list[int] per image (from img_metas['organ_id']); used
                for the optional per-organ modulation. None -> organ id 0 for all.
            det_mean, det_std: the detector data_preprocessor mean/std ([3,1,1] or
                broadcastable), used to invert normalization before re-normalizing
                crops for the teacher.
        """
        from torchvision.ops import roi_align

        device = batch_inputs.device
        feat = img_feats[self.roi_level]
        _, _, H, W = batch_inputs.shape
        scale = feat.shape[-1] / float(W)  # feat-map / input

        # gather GT boxes per image with batch index + organ id
        boxes_list, organ_list = [], []
        for b, inst in enumerate(batch_gt_instances):
            bb = inst.bboxes
            if bb is None or bb.shape[0] == 0:
                boxes_list.append(torch.zeros((0, 5), device=device))
                continue
            bidx = torch.full((bb.shape[0], 1), float(b), device=device)
            boxes_list.append(torch.cat([bidx, bb.to(device).float()], dim=1))  # [Ni,5]
            oid = int(batch_organ_ids[b]) if batch_organ_ids is not None else 0
            organ_list.append(torch.full((bb.shape[0],), oid, device=device, dtype=torch.long))
        rois = torch.cat(boxes_list, dim=0)  # [N,5] (batch_idx, x1,y1,x2,y2)
        if rois.shape[0] == 0:
            return {'loss_region_gap': self.proj.weight.sum() * 0.0}
        organ_ids = torch.cat(organ_list, dim=0) if organ_list else None

        # optional subsample to bound teacher cost
        if self.max_rois is not None and rois.shape[0] > self.max_rois:
            # deterministic stride sample (avoid Math.random / per-step RNG concerns)
            idx = torch.linspace(0, rois.shape[0] - 1, self.max_rois,
                                 device=device).long()
            rois = rois[idx]
            if organ_ids is not None:
                organ_ids = organ_ids[idx]

        # student region features (RoIAlign on detector feature map)
        student_feats = roi_align(
            feat, rois, output_size=1, spatial_scale=scale,
            sampling_ratio=2, aligned=True).flatten(1)  # [N, student_dim]

        # crops for the teacher: invert detector norm -> [0,255]-ish raw -> CLIP norm.
        # Vectorized: roi_align extracts + bilinearly resizes ALL crops in a single
        # kernel (sub-pixel correct, same sampler as the student features above),
        # replacing the old per-box python loop (128x int()-coord GPU->CPU syncs +
        # interpolate launches) that dominated the per-iter cost.
        det_mean = det_mean.to(device).view(1, 3, 1, 1)
        det_std = det_std.to(device).view(1, 3, 1, 1)
        raw = batch_inputs * det_std + det_mean          # back to ~[0,255]
        crops = roi_align(
            raw, rois, output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0, sampling_ratio=2, aligned=True)  # [N,3,crop,crop] in [0,255]
        crops = (crops / 255.0 - self._clip_mean) / self._clip_std

        teacher_emb = self._teacher_encode(crops)
        loss = self._distill(student_feats, teacher_emb, organ_ids,
                             avg_factor=float(student_feats.shape[0]))
        return {'loss_region_gap': loss}
