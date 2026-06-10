"""Module 2 (2026-06-08): region-conditioned attribute reweighting — the core of the
attribute-adaptive OVD classifier.

A cytology class text is built from A morphological attributes (cell size, nuclear shape,
chromatin, N:C ratio, ...). The standard head collapses them to ONE class prototype and
uses a FIXED attribute weighting for every cell. But the per-attribute analysis (2026-06-08)
showed the attributes have very different discriminability (organ axis within-organ cos
0.92 = non-discriminative; morphology axes 0.62-0.77), and which attribute is salient
varies per cell. So we let EACH spatial location (candidate cell) predict per-attribute
weights and re-weight the per-attribute region-text similarities:

    L[b,c,a,h,w] = <region[b,:,h,w], attr_text[c,a,:]>      # per-attribute logit
    w[b,a,h,w]   = softmax_a( Conv1x1(region) )             # per-location attr weights (shared over classes)
    s[b,c,h,w]   = sum_a w[b,a,h,w] * L[b,c,a,h,w]          # instance-adaptive class logit

Leakage guardrail: the weights are SHARED across classes (they only choose which attribute
AXIS matters at a location, never the class) and only re-weight a class's OWN attributes —
so they cannot fabricate a class identity; train/test use the same rule -> legitimate
instance-adaptive scoring, not label leakage.

Protections (audit 2026-06-08): (1) zero-init gate -> starts as a uniform attribute mean
(== the plain mean-pooled classifier), then learns; (2) per-attribute gamma/beta logit
calibration (different attribute axes have different cosine ranges); (3) attribute dropout
(stops collapse onto a single dominant axis); (4) `balance_loss` (keep batch-mean weights
off a single axis); (5) returns `w` for visualisation/interpretability.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class RegionAttributeFusion(nn.Module):
    """Per-location attribute reweighting over per-attribute region-text logits.

    Args:
        embed_dims: channel dim of the (already BN'd) region feature == attr text dim.
        num_attrs: A (5 or 6).
        attr_dropout: per-image attribute-drop prob during training (>=1 axis kept).
    """

    def __init__(self, embed_dims: int, num_attrs: int, attr_dropout: float = 0.1,
                 adaptive: bool = True) -> None:
        super().__init__()
        self.num_attrs = int(num_attrs)
        self.attr_dropout = float(attr_dropout)
        # adaptive=False -> B1 control: FIXED uniform attribute weights (no per-location gate),
        # i.e. classify against the (calibrated) mean attribute prototype. This isolates the
        # per-location adaptive gate as the innovation -> B3(adaptive) vs B1(mean) measures it.
        # The attr_gate is the ONLY adaptive part; gamma/beta (global per-axis calibration)
        # stay in both arms so they cancel in the comparison.
        self.adaptive = bool(adaptive)
        if self.adaptive:
            self.attr_gate = nn.Conv2d(embed_dims, num_attrs, kernel_size=1)
            nn.init.zeros_(self.attr_gate.weight)   # zero-init -> uniform weights at start
            nn.init.zeros_(self.attr_gate.bias)
        self.attr_gamma = nn.Parameter(torch.ones(num_attrs))   # per-attr logit scale
        self.attr_beta = nn.Parameter(torch.zeros(num_attrs))   # per-attr logit shift

    def forward(self, x: torch.Tensor, attr_text: torch.Tensor):
        """x: [B, D, H, W] region feature (BN'd by the head, NOT re-normalized here, to
        match BNContrastiveHead). attr_text: [C, A, D] per-attribute class text.
        Returns (s [B, C, H, W] class logits, w [B, A, H, W] attribute weights)."""
        B, D, H, W = x.shape
        at = F.normalize(attr_text.to(x.dtype), dim=-1)              # [C, A, D]
        # per-attribute logits: each location vs each class's each attribute
        L = torch.einsum('bdhw,cad->bcahw', x, at)                   # [B, C, A, H, W]
        L = L * self.attr_gamma.view(1, 1, -1, 1, 1) + self.attr_beta.view(1, 1, -1, 1, 1)
        if self.adaptive:
            # per-location attribute weights (shared across classes; uniform at init)
            gate = self.attr_gate(x)                                 # [B, A, H, W]
            if self.training and self.attr_dropout > 0:
                keep = torch.rand(B, self.num_attrs, 1, 1, device=x.device) > self.attr_dropout
                keep = keep | (keep.sum(dim=1, keepdim=True) == 0)   # keep >=1 axis per image
                gate = gate.masked_fill(~keep, float('-inf'))
            w = gate.softmax(dim=1)                                  # [B, A, H, W]
        else:
            # B1 control: fixed uniform weights -> mean over attributes (= mean-attr classifier)
            w = x.new_full((B, self.num_attrs, H, W), 1.0 / self.num_attrs)
        s = torch.einsum('bahw,bcahw->bchw', w, L)                  # [B, C, H, W]
        return s, w

    @staticmethod
    def balance_loss(w: torch.Tensor, weight: float = 0.01) -> torch.Tensor:
        """KL(mean_w || uniform): keep the BATCH-MEAN attribute weights from collapsing onto
        one axis (does NOT force per-location uniformity — a location may legitimately use a
        single attribute). w: [B, A, H, W]."""
        A = w.shape[1]
        mean_w = w.mean(dim=(0, 2, 3)).clamp(min=1e-8)               # [A]
        mean_w = mean_w / mean_w.sum()
        kl = (mean_w * (mean_w.log() + math.log(A))).sum()   # device-safe scalar log
        return weight * kl


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, H, W, C, A = 2, 512, 16, 16, 30, 6
    m = RegionAttributeFusion(embed_dims=D, num_attrs=A, attr_dropout=0.0).eval()
    x = torch.randn(B, D, H, W)
    attr = F.normalize(torch.randn(C, A, D), dim=-1)

    s, w = m(x, attr)
    assert s.shape == (B, C, H, W), s.shape
    assert w.shape == (B, A, H, W), w.shape

    # zero-init gate -> uniform weights -> s == mean over attributes of the per-attr logits
    at = F.normalize(attr, dim=-1)
    L = torch.einsum('bdhw,cad->bcahw', x, at)           # gamma=1,beta=0 at init
    s_mean = L.mean(dim=2)                                # uniform attribute mean
    assert torch.allclose(s, s_mean, atol=1e-5), "init must equal the mean-attr classifier"
    assert torch.allclose(w, torch.full_like(w, 1.0 / A), atol=1e-6), "init weights uniform"

    # grad flows to the gate + calibration
    m.train()
    s2, w2 = m(x, attr)
    s2.sum().backward()
    assert m.attr_gate.weight.grad is not None and m.attr_gamma.grad is not None
    # balance loss is finite + non-negative
    bl = RegionAttributeFusion.balance_loss(w2.detach())
    assert bl.item() >= -1e-6
    print(f"[ok] RegionAttributeFusion: shapes ok, init==mean-attr classifier (uniform w), "
          f"grad->gate/gamma, balance_loss={bl.item():.4f}")
