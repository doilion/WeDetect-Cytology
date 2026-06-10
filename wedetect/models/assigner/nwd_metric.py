"""Normalized Wasserstein Distance (NWD) similarity for tiny-object assignment.

Drop-in scale-stable replacement for IoU in ``BatchTaskAlignedAssigner`` and in the
bbox regression loss. IoU is geometrically unstable for tiny boxes -- a 1px shift on a
~22px cell collapses IoU -- so TaskAligned assignment starves tiny cells of positive
samples and corrupts the region-text contrastive supervision they receive, hurting
recall. NWD models each box as a 2D Gaussian ``N(center, diag((w/2)^2, (h/2)^2))`` and
compares two boxes by the closed-form 2-Wasserstein distance between their Gaussians,
which degrades gracefully with small offsets and is scale-stable.

Reference: Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object
Detection" (arXiv:2110.13389).

``C`` (normalization constant) is DATA-DRIVEN: ~ mean object pixel size in the *input*
(640) coordinate space the assigner/loss operate in. For TCT_NGC dev30 @640,
mean sqrt(w*h) = 30.3 -> default ``C = 30.0``. Do NOT copy the aerial default (12.8);
see project memory on data-driven hyperparameters.
"""
import torch
from torch import Tensor

# Mean object size sqrt(w*h) of TCT_NGC dev30 train boxes in the 640 input space.
NWD_DEFAULT_C: float = 30.0


def _wasserstein2(c1: Tensor, wh1: Tensor, c2: Tensor, wh2: Tensor) -> Tensor:
    """Squared 2-Wasserstein distance between two axis-aligned box-Gaussians.

    For N(mu, diag((w/2)^2, (h/2)^2)) the closed form reduces to a Euclidean distance
    in the (cx, cy, w/2, h/2) space::

        W2^2 = ||c1 - c2||^2 + ||wh1/2 - wh2/2||^2
    """
    center_term = ((c1 - c2) ** 2).sum(-1)
    size_term = (((wh1 - wh2) * 0.5) ** 2).sum(-1)
    return center_term + size_term


def _xyxy_to_center_wh(bbox: Tensor):
    """xyxy -> (center[..., 2], wh[..., 2]); wh clamped to >= 0."""
    center = (bbox[..., 0:2] + bbox[..., 2:4]) * 0.5
    wh = (bbox[..., 2:4] - bbox[..., 0:2]).clamp(min=0)
    return center, wh


def nwd_similarity(bbox1: Tensor, bbox2: Tensor, C: float = NWD_DEFAULT_C,
                   eps: float = 1e-7) -> Tensor:
    """Batched all-pairs NWD similarity (mirrors ``yolov6_iou_calculator``).

    Args:
        bbox1 (Tensor): shape (batch_size, num_gt, 4), xyxy in input-pixel coords.
        bbox2 (Tensor): shape (batch_size, num_priors, 4), xyxy.
        C (float): normalization constant (~ mean object size, same pixel units).
        eps (float): numerical floor inside the sqrt.
    Returns:
        Tensor: NWD similarity in (0, 1], shape (batch_size, num_gt, num_priors).
    """
    b1 = bbox1.unsqueeze(2)  # [bs, num_gt, 1, 4]
    b2 = bbox2.unsqueeze(1)  # [bs, 1, num_priors, 4]
    c1, wh1 = _xyxy_to_center_wh(b1)
    c2, wh2 = _xyxy_to_center_wh(b2)
    w2 = _wasserstein2(c1, wh1, c2, wh2)  # [bs, num_gt, num_priors]
    return torch.exp(-torch.sqrt(w2 + eps) / C)


def nwd_similarity_aligned(pred: Tensor, target: Tensor, C: float = NWD_DEFAULT_C,
                           eps: float = 1e-7) -> Tensor:
    """Pairwise (element-aligned) NWD similarity for the bbox regression loss.

    Args:
        pred (Tensor): shape (N, 4), xyxy.
        target (Tensor): shape (N, 4), xyxy.
    Returns:
        Tensor: NWD similarity in (0, 1], shape (N,).
    """
    c1, wh1 = _xyxy_to_center_wh(pred)
    c2, wh2 = _xyxy_to_center_wh(target)
    w2 = _wasserstein2(c1, wh1, c2, wh2)  # [N]
    return torch.exp(-torch.sqrt(w2 + eps) / C)
