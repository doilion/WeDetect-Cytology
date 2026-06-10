"""Whitening text de-cone (2026-06-08): the parameter-free baseline of the de-collapse
ladder (D-v2 / PrototypeGroundedTextAdapter is the learnable version).

The cytology OVD text collapse is mostly removable ANISOTROPY (a cone): within-organ cos
0.79 -> 0.04 after removing the global mean + top-1 principal direction. This module does
exactly that geometric de-cone on the frozen text prompts, plugged into the detector's
text_decone hook. It has NO learnable parameters -- the trainable part is the IMAGE side,
which learns to align in the de-coned space (this is whitening-IN-training; applying it
only at inference to a cone-trained model is a weak / false-negative-prone test).

Fit ONLINE on the current txt_feats (= the given prompts): base at train, base+novel at
test -> at test the novel prompts are de-coned together with base (the base+novel-fit that
geometrically drops novel<->base 0.70->0.31). The top-1 cone direction is robust to adding
the 9 novel prompts, so the base train(30-fit)/test(39-fit) mismatch is small.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class WhiteningTextDecone(nn.Module):
    """Parameter-free anisotropy de-cone of the OVD text prompts.

    Args:
        topk: number of leading principal directions (the cone) to project out.
    """

    def __init__(self, topk: int = 1, alpha: float = 1.0, beta: float = 1.0,
                 lam: float = 0.5, fit_classes: int = None) -> None:
        """SOFT / RESIDUAL de-cone (2026-06-08 audit): a HARD top-k removal pushes
        within-organ cos to ~0 (near-orthogonal) and flattens the medical class
        hierarchy + can break the CLIP image-text alignment. So expose strengths and
        blend back the raw text:
            decone = normalize((t - alpha*mu) - beta * proj_topk)
            out    = normalize((1 - lam) * raw + lam * decone)
        Args:
            topk: # principal cone directions to attenuate.
            alpha: mean-removal strength (0..1).
            beta:  top-k-removal strength (0..1).
            lam:   residual blend (0 = raw text untouched, 1 = full de-cone). Sweep
                   0.25/0.5/0.75/1.0; LOWER preserves CLIP alignment + medical hierarchy.
            fit_classes: if set, estimate the de-cone basis (mu + top-k dirs) on the FIRST
                   `fit_classes` rows ONLY (= the base classes), then apply it to ALL rows.
                   This is the BASE-ONLY whitening of the transductive control: at novel eval
                   the de-cone never reads the novel prompts' statistics (defends against the
                   "is this transductive?" reviewer question). None = fit on all given prompts
                   (the base+novel open-vocab inference setting). No effect at base eval
                   (fit_classes == #classes).
        Validate with the REGION-TEXT MARGIN in the trained detector, NOT text-text cos.
        """
        super().__init__()
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lam = float(lam)
        self.fit_classes = int(fit_classes) if fit_classes is not None else None

    def forward(self, txt_feats: torch.Tensor) -> torch.Tensor:
        """txt_feats: [C, D] or [B, C, D]. Returns the (soft) de-coned text, same shape/dim."""
        squeeze = txt_feats.dim() == 2
        t = txt_feats.unsqueeze(0) if squeeze else txt_feats   # [B, C, D]
        cls = t[0].float()                                     # [C, D] (== across batch)
        if cls.shape[0] <= self.topk or self.lam == 0.0:       # too few classes / disabled
            return txt_feats
        # the de-cone is a per-VOCABULARY transform; the detector broadcasts the same class
        # text across the batch, so fitting from t[0] is exact. Guard the assumption cheaply
        # (O(C*D)) so a future per-instance-text caller fails loud instead of mis-projecting
        # rows 1..B-1 with row-0's basis. [code-review 2026-06-08]
        assert t.shape[0] == 1 or torch.allclose(t[0], t[1], atol=1e-4), \
            "WhiteningTextDecone expects batch-broadcast class text (rows must be identical)"
        # fit the de-cone basis on base-only rows if fit_classes is set (transductive control);
        # else on all given prompts. The basis is then APPLIED to all rows (base+novel).
        fit = cls
        if self.fit_classes is not None and self.topk < self.fit_classes < cls.shape[0]:
            fit = cls[: self.fit_classes]                      # base-only statistics
        mu = fit.mean(0, keepdim=True)                         # [1, D]
        with torch.no_grad():                                  # principal cone dirs (full-centered)
            _, _, Vt = torch.linalg.svd(fit - mu, full_matrices=False)
            Vk = Vt[: self.topk]                               # [k, D]
        mu = mu.to(t.dtype); Vk = Vk.to(t.dtype)
        centered = t - self.alpha * mu                         # soft mean removal
        decone = centered - self.beta * ((centered @ Vk.t()) @ Vk)  # soft top-k removal
        decone = F.normalize(decone, dim=-1)
        out = F.normalize((1.0 - self.lam) * F.normalize(t, dim=-1)
                          + self.lam * decone, dim=-1)         # residual blend with raw
        return out.squeeze(0) if squeeze else out


@MODELS.register_module()
class PerAttrWhitening(nn.Module):
    """Module 1: per-attribute soft de-cone. Applies the soft/residual WhiteningTextDecone
    to EACH attribute axis of [C, A, D] INDEPENDENTLY. The per-attribute analysis
    (2026-06-08) showed each axis has its own anisotropy + medical structure (organ axis
    within-organ cos 0.92 with a clear cone; morphology axes 0.62-0.77 with spread
    anisotropy where centering does most of the work), so mean-pooling first would erase
    it. Parameter-free; the image side learns to align in the per-attr de-coned space."""

    def __init__(self, topk: int = 1, alpha: float = 1.0, beta: float = 1.0,
                 lam: float = 0.5, fit_classes: int = None) -> None:
        super().__init__()
        self.decone = WhiteningTextDecone(topk=topk, alpha=alpha, beta=beta, lam=lam,
                                          fit_classes=fit_classes)

    def forward(self, attr_text: torch.Tensor) -> torch.Tensor:
        """attr_text: [C, A, D] -> [C, A, D] (each attribute axis de-coned over classes)."""
        A = attr_text.shape[1]
        return torch.stack([self.decone(attr_text[:, a, :]) for a in range(A)], dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    C, D = 30, 512
    # a "coned" text: all vectors share a strong common direction (anisotropy)
    cone = F.normalize(torch.randn(1, D), dim=-1)
    text = F.normalize(0.9 * cone + 0.1 * torch.randn(C, D), dim=-1)

    def mean_pair_cos(x):
        xn = F.normalize(x, dim=-1)
        c = xn @ xn.t()
        return float(c[~torch.eye(len(x), dtype=torch.bool)].mean())

    base = mean_pair_cos(text)
    # lam=0 -> raw passthrough (disabled)
    assert torch.allclose(WhiteningTextDecone(lam=0.0)(text), text), "lam=0 must be a no-op"
    # increasing lam -> progressively MORE de-coned (lower pairwise cos), monotone
    full = mean_pair_cos(WhiteningTextDecone(topk=1, alpha=1, beta=1, lam=1.0)(text))
    soft = mean_pair_cos(WhiteningTextDecone(topk=1, alpha=1, beta=1, lam=0.5)(text))
    assert full < soft < base, f"expect full {full:.3f} < soft {soft:.3f} < raw {base:.3f}"
    # dim preserved + batched + few-class guard
    out = WhiteningTextDecone(lam=0.5)(text)
    assert out.shape == text.shape
    assert WhiteningTextDecone(lam=0.5)(text.unsqueeze(0).expand(4, -1, -1)).shape == (4, C, D)
    assert torch.allclose(WhiteningTextDecone(topk=5)(text[:3]), text[:3])  # C<=topk -> no-op
    print(f"[ok] WhiteningTextDecone soft/residual: raw {base:.3f} -> soft(lam.5) {soft:.3f} "
          f"-> full(lam1) {full:.3f}; lam=0 no-op; dim/batch/few-class ok")

    # PerAttrWhitening: [C, A, D] each axis de-coned independently
    A = 6
    attr = F.normalize(0.9 * cone.unsqueeze(1) + 0.1 * torch.randn(C, A, D), dim=-1)
    pa = PerAttrWhitening(topk=1, lam=0.5)(attr)
    assert pa.shape == (C, A, D), pa.shape
    b0 = mean_pair_cos(attr[:, 0, :]); a0 = mean_pair_cos(pa[:, 0, :])
    assert a0 < b0, f"per-attr axis-0 should de-cone ({b0:.3f}->{a0:.3f})"
    print(f"[ok] PerAttrWhitening: [C,A,D] preserved, per-axis de-cone {b0:.3f}->{a0:.3f}")
