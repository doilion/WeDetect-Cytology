# Copyright (c) Tencent Inc. All rights reserved.
# NOTE (2026-05-30): OrganOrdinalLoss (M2 aux) and TextCorrespondenceLoss (STEGO-corr)
# archived to legacy/ — text-side fusion is architecturally inert for base
# (docs/method_design/text_branch_verdict_20260530.md).
from .dynamic_loss import CoVMSELoss
from .iou_loss import mmyoloIoULoss
from .organ_hard_neg_loss import OrganHardNegContrastiveLoss
from .region_gap_distill_loss import OrganModulatedRegionGapDistill
from .relational_distill_loss import RelationalDistillationLoss

__all__ = ['CoVMSELoss', 'OrganHardNegContrastiveLoss',
           'OrganModulatedRegionGapDistill', 'RelationalDistillationLoss']
