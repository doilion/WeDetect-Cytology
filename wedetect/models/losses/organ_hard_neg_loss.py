"""Innovation ① (v1): organ-conditioned hard-negative discriminative loss.

The standard per-class sigmoid-BCE classification loss under-trains the HARD
within-organ distinctions: most of its "push away from other classes" signal is
spent on easy cross-organ pairs (a thyroid cell vs a urine cell are trivially
different), so genuinely confusable same-organ fine-classes stay entangled. The
within-organ separability probe (tools/probe_within_organ_separability.py)
confirmed the discriminative information IS present in the region features
(linear-probe within-organ top1 0.90 vs the model's native 0.79, headroom
concentrated in Thyroid/Urine) but is left unused.

This loss adds, for each POSITIVE anchor, a softmax cross-entropy (or a
hardest-negative margin) computed ONLY over the classes of the SAME organ
(``group_mask``) -- so the gradient is spent exactly on the confusable same-organ
competitors. Zero new parameters (operates in logit space); reuses the head's
existing ``per_image_mask``.

Positioning vs prior lab work (Xu 2026, MCDC "instance-separability contrastive"):
that loss is GLOBAL (all classes, no group structure, single-specimen blood
cells). Ours restricts the negative SET to the same organ/superclass -- the
multi-organ structure is the new ingredient. (BIBM extension framing.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class OrganHardNegContrastiveLoss(nn.Module):
    """Within-organ (within-group) hard-negative discriminative loss.

    Args:
        loss_weight: scalar weight on the returned loss.
        mode: 'ce' = within-organ softmax cross-entropy (default; the softmax
            inherently up-weights the hardest competitor); 'margin' = explicit
            hardest-in-organ-negative margin loss.
        margin: margin for mode='margin' (logit-gap the GT must exceed).
        eps: numerical floor (unused by 'ce', kept for API symmetry).
    """

    def __init__(self, loss_weight: float = 0.5, mode: str = 'ce',
                 margin: float = 0.2, eps: float = 1e-12) -> None:
        super().__init__()
        assert mode in ('ce', 'margin'), f"unknown mode {mode!r}"
        self.loss_weight = loss_weight
        self.mode = mode
        self.margin = margin
        self.eps = eps

    def forward(self,
                cls_preds: torch.Tensor,
                assigned_scores: torch.Tensor,
                fg_mask: torch.Tensor,
                group_mask: torch.Tensor,
                avg_factor=None) -> torch.Tensor:
        """
        Args:
            cls_preds: [B, A, C] region-text logits (logit_scale*cos + bias).
            assigned_scores: [B, A, C] soft one-hot (label) * alignment metric.
            fg_mask: [B, A] bool positive-anchor mask.
            group_mask: [B, C] {0,1} same-organ class membership per image.
            avg_factor: scalar normalizer; defaults to the positive-weight sum.
        Returns:
            scalar loss (already multiplied by loss_weight).
        """
        if fg_mask.sum() == 0:
            # No positives this batch: return a 0 that still carries a grad path.
            return cls_preds.sum() * 0.0

        neg_floor = torch.finfo(cls_preds.dtype).min

        pos_logits = cls_preds[fg_mask]                                  # [P, C]
        pos_group = group_mask.unsqueeze(1).expand_as(cls_preds)[fg_mask]  # [P, C]
        pos_gt = assigned_scores.argmax(dim=-1)[fg_mask]                 # [P]
        pos_w = assigned_scores.sum(dim=-1)[fg_mask].clamp(min=0.0)      # [P]

        # Mask out cross-organ classes so the discrimination is within-organ only.
        masked = pos_logits.masked_fill(pos_group <= 0, neg_floor)      # [P, C]

        if self.mode == 'ce':
            per = F.cross_entropy(masked, pos_gt, reduction='none')     # [P]
        else:  # 'margin'
            gt_logit = pos_logits.gather(1, pos_gt[:, None]).squeeze(1)  # [P]
            neg = masked.scatter(1, pos_gt[:, None], neg_floor)         # drop GT col
            hard_neg = neg.max(dim=1).values                            # [P]
            has_neg = hard_neg > (neg_floor / 2)  # False for single-class organs
            per = torch.zeros_like(gt_logit)
            per[has_neg] = F.relu(
                self.margin - (gt_logit[has_neg] - hard_neg[has_neg]))

        loss = (per * pos_w).sum()
        if avg_factor is None:
            avg_factor = pos_w.sum().clamp(min=1.0)
        loss = loss / avg_factor
        return self.loss_weight * loss
