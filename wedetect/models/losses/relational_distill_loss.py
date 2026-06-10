"""Innovation D (2026-06-05): image->text relational distillation.

Motivation (3-prompt collapse study, docs/results/figures/text_collapse_3sets_
20260605.png): frozen BiomedCLIP text of fine cytology classes is COLLAPSED
(within-organ cos 0.80, max 0.97) and no prompt design fixes it (5attr/morph6
collapse worse). But the detector's IMAGE region features ARE discriminative
(within-organ linear-probe 0.90). So we distill the RELATIONAL geometry of image
region prototypes into the learnable text adapter (TextRelationalAdapter): the
adapted-text within-organ pairwise cosine matrix is pulled toward the image-
prototype within-organ cosine matrix.

Design:
- Teacher = an EMA bank of per-class image region prototypes (RoIAlign over GT
  boxes on the detector's finest neck level, same recipe as innovation ②). The
  bank gives a stable, all-class cosine target even though a single batch rarely
  contains all 30 classes. The bank is a stop-grad buffer -> the IMAGE side gets
  NO gradient from this loss (it only improves via the main det losses).
- Student = the adapted text g(frozen_text). Only the adapter trains here.
- Loss = MSE between the two within-organ [C,C] cosine matrices. Cross-organ
  pairs are excluded (the organ prior masks them at inference, so their text
  geometry is irrelevant). Cosine is dimension-free, so the 96-d image space and
  512-d text space are compared directly without any projection.

Differentiation: not prompt engineering (proven dead) and not absolute-vector
cross-modal alignment (ViLD/F-VLM, our failed ②/THAF). We transfer dimension-
free cosine STRUCTURE from the strong image side to the collapsed text side.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class RelationalDistillationLoss(nn.Module):
    """Within-organ image->text relational (cosine-matrix) distillation.

    Args:
        num_classes: number of base classes C (30 for dev30).
        student_dim: channels of ``img_feats[roi_level]`` (= head in_channels[0],
            96 for tiny neck level-0 / stride-8). Must match the RoIAlign source.
        roi_level: which neck level to RoIAlign region features from (0 = finest).
        organ_mask_path: ``*.pt`` holding ``mask`` [C, O]; argmax gives class->organ.
        momentum: EMA momentum for the per-class prototype bank.
        loss_weight: scalar weight on the returned loss (calibrate from the smoke
            log: the raw MSE is ~1e-2, so ~10 brings it onto the cls-loss scale
            late in training when cls has saturated).
        max_rois: cap GT boxes used per step (deterministic stride subsample).
        warmup_steps: steps to let the EMA bank stabilize before the loss is
            applied (returns a zero, grad-pathed loss until then).
    """

    def __init__(self,
                 num_classes: int = 30,
                 student_dim: int = 96,
                 roi_level: int = 0,
                 teacher_source: str = "neck",
                 organ_mask_path: str = "data/texts/tct_ngc_class_organ_mask_base30.pt",
                 momentum: float = 0.9,
                 loss_weight: float = 10.0,
                 max_rois: Optional[int] = 256,
                 warmup_steps: int = 100) -> None:
        super().__init__()
        # Fail fast on a mistyped teacher_source: the detector silently treats
        # anything that is not exactly "neck"/"cls_embed_bn" as pre-BN cls_embed
        # (yolo_world.py), so a typo like "cls_embedd" would quietly train on the
        # wrong teacher feature. Whitelist the three supported sources here.
        allowed_sources = {"neck", "cls_embed", "cls_embed_bn"}
        if teacher_source not in allowed_sources:
            raise ValueError(
                f"teacher_source={teacher_source!r} not in {sorted(allowed_sources)}; "
                f"check the config (typos silently fall back to pre-BN cls_embed).")
        self.num_classes = num_classes
        self.roi_level = roi_level
        # which image feature the DETECTOR should RoIAlign for the teacher bank:
        #   'neck'         -> neck level-0 (96-d, low-level; within-organ cos 0.68
        #                     converged / 0.98 early — WEAK teacher, D-v1 default)
        #   'cls_embed'    -> cls_pred(neck) level-0 (512-d, pre-BN; cos 0.21 — strong)
        #   'cls_embed_bn' -> BN(cls_pred(neck)) level-0 (512-d; cos 0.26, == the space
        #                     where the head compares text<->image — D-v2, principled)
        # The detector reads this attr and passes the matching feature map; student_dim
        # MUST match (96 for 'neck', 512 for the cls_embed variants).
        self.teacher_source = teacher_source
        self.momentum = momentum
        self.loss_weight = loss_weight
        self.max_rois = max_rois
        self.warmup_steps = warmup_steps
        # EMA prototype bank + per-class update counter (buffers -> follow .to(device),
        # excluded from the optimizer; the loss module has NO trainable params).
        self.register_buffer("prototype_bank", torch.zeros(num_classes, student_dim))
        self.register_buffer("bank_count", torch.zeros(num_classes))
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        # class -> organ id, loaded here so the loss is self-contained (does not
        # depend on the head carrying an organ_class_mask buffer).
        pkg = torch.load(organ_mask_path, map_location="cpu", weights_only=False)
        organ = pkg["mask"].argmax(dim=1).long()  # [C]
        self.register_buffer("class_organ_map", organ)

    # ---- EMA bank update (no grad) -----------------------------------------
    @torch.no_grad()
    def _update_bank(self, region_feats: torch.Tensor, labels: torch.Tensor) -> None:
        for c in labels.unique().tolist():
            if c < 0 or c >= self.num_classes:
                continue
            mean_c = region_feats[labels == c].mean(dim=0)
            if self.bank_count[c] == 0:
                self.prototype_bank[c] = mean_c
            else:
                self.prototype_bank[c] = (
                    self.momentum * self.prototype_bank[c]
                    + (1.0 - self.momentum) * mean_c)
            self.bank_count[c] += 1

    @staticmethod
    def _zero(ref: torch.Tensor) -> dict:
        # zero loss that still touches the student graph (no DDP unused-param /
        # missing-key issues) — ref is txt_feats (= adapter output).
        return {"loss_rel_distill": ref.sum() * 0.0}

    # ---- core (RoIAlign -> bank -> within-organ cos-matrix MSE) -------------
    def compute(self, img_feats, txt_feats, batch_gt_instances, input_hw) -> dict:
        """Args:
            img_feats: tuple of neck feature maps; level ``roi_level`` is RoIAlign'd.
            txt_feats: [B, C, D] adapted text (student; carries grad to the adapter).
            batch_gt_instances: list[InstanceData] with ``.bboxes`` (xyxy, input-
                pixel coords) and ``.labels`` (class ids).
            input_hw: (H, W) of ``batch_inputs`` (for the RoIAlign spatial scale).
        """
        from torchvision.ops import roi_align

        # guard: this loss assumes plain [B, C, D] text with C == num_classes.
        if txt_feats.dim() != 3 or txt_feats.shape[1] != self.num_classes:
            return self._zero(txt_feats)

        feat = img_feats[self.roi_level]
        # Guard the student_dim<->teacher-channel contract: the EMA bank was
        # allocated with student_dim columns, so a teacher feature with a
        # different channel count would otherwise crash deep inside _update_bank
        # (or silently broadcast). Surface it here with the actionable hint.
        if feat.shape[1] != self.prototype_bank.shape[1]:
            raise ValueError(
                f"teacher feature has {feat.shape[1]} channels but student_dim="
                f"{self.prototype_bank.shape[1]} (teacher_source={self.teacher_source!r}); "
                f"set student_dim to match the RoIAlign source "
                f"(96 for 'neck', 512 for the cls_embed variants).")
        device = feat.device
        W = int(input_hw[1])
        scale = feat.shape[-1] / float(W)  # feat-map / input

        boxes_list, label_list = [], []
        for b, inst in enumerate(batch_gt_instances):
            bb = getattr(inst, "bboxes", None)
            lb = getattr(inst, "labels", None)
            if bb is None or lb is None or bb.shape[0] == 0:
                continue
            bidx = torch.full((bb.shape[0], 1), float(b), device=device)
            boxes_list.append(torch.cat([bidx, bb.to(device).float()], dim=1))  # [Ni,5]
            label_list.append(lb.to(device).long())
        if not boxes_list:
            return self._zero(txt_feats)
        rois = torch.cat(boxes_list, dim=0)      # [N,5] (batch_idx, x1,y1,x2,y2)
        labels = torch.cat(label_list, dim=0)    # [N]

        if self.max_rois is not None and rois.shape[0] > self.max_rois:
            idx = torch.linspace(0, rois.shape[0] - 1, self.max_rois,
                                 device=device).long()
            rois = rois[idx]
            labels = labels[idx]

        region_feats = roi_align(
            feat, rois, output_size=1, spatial_scale=scale,
            sampling_ratio=2, aligned=True).flatten(1)  # [N, student_dim]

        self._update_bank(region_feats.detach(), labels)
        self._step += 1

        valid = self.bank_count > 0  # [C]
        if int(self._step) < self.warmup_steps or int(valid.sum()) < 2:
            return self._zero(txt_feats)

        # teacher: image-prototype within-organ cosine matrix (stop-grad)
        bank = F.normalize(self.prototype_bank, dim=-1).detach()  # [C, d]
        img_cos = bank @ bank.t()                                 # [C, C]
        # student: adapted-text within-organ cosine matrix (carries grad to adapter)
        txt = F.normalize(txt_feats.mean(dim=0).float(), dim=-1)  # [C, D]
        txt_cos = txt @ txt.t()                                   # [C, C]

        organ = self.class_organ_map.to(device)
        within = (organ[:, None] == organ[None, :]) & valid[:, None] & valid[None, :]
        within.fill_diagonal_(False)
        denom = within.sum().clamp(min=1)

        # One-time teacher-direction diagnostic: is the IMAGE prototype within-organ
        # cos actually LOWER (less collapsed) than the text? If image > text, the
        # teacher is worse than the student and D pulls text the WRONG way.
        if not getattr(self, "_diag_done", False) and float(denom) > 0:
            wf = within.float()
            img_w = float((img_cos * wf).sum() / denom)
            txt_w = float((txt_cos * wf).sum() / denom)
            print(f"[RelDistill diag @step{int(self._step)}] within-organ cos: "
                  f"image-proto={img_w:.3f}  text={txt_w:.3f}  "
                  f"gap(text-image)={txt_w - img_w:+.3f}  "
                  f"valid_classes={int(valid.sum())}/{self.num_classes}", flush=True)
            self._diag_done = True

        loss = ((txt_cos - img_cos) ** 2 * within.float()).sum() / denom
        return {"loss_rel_distill": self.loss_weight * loss}
