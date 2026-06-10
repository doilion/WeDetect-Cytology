"""D-v2 (2026-06-08): prototype-grounded text de-collapse via text-as-query cross-attention.

This is the ITERATED version of relational distillation (D-v1), born from analyzing D-v1's
failure rather than discarding the idea:

  D-v1 failure mode (analyzed)         -> D-v2 fix (here)
  --------------------------------------------------------------------------------
  gauge-freedom: the relational cos-     -> trained with the DETECTION cls loss (not a
    matrix loss is rotation-invariant,      relational loss), which directly anchors the
    so the text drifts off image            ABSOLUTE text<->image alignment -> no free
    alignment while satisfying the loss     rotation to exploit.
  text collapse is ANISOTROPY (a cone)   -> the text (query) cross-attends to the bank of
    that D-v1 never explicitly removed      discriminative image prototypes (within-organ
                                            cos ~0.21) and is pulled OUT of the cone toward
                                            those directions.
  ICF image-as-query leaks (classifier   -> the bank holds CLASS-LEVEL EMA prototypes, NOT
    peeks at the query instance)            the current instance, and is stop-grad ->
                                            no instance leakage.

Output is the de-coned text g(t) = t + alpha * CrossAttn(t, bank). The output projection
is ZERO-INIT so the residual is exactly 0 and g(t) == t at start (regression-safe; identical
to the frozen-text head until the adapter learns). NOTE (corrected per 2026-06-08 audit):
with o_proj zero-init, o_proj's OWN gradient is non-zero from step 1 (= grad . attn_input)
so it activates immediately, but q/k/v get exactly-zero gradient until o_proj moves off zero
-> a ~2-step warmup before the cross-attention starts learning (immaterial to convergence).
A non-zero o_proj init (e.g. std=0.01) would activate q/k/v from step 1 but break the exact
identity-at-init, so we keep zero-init and accept the 2-step warmup. The contrastive head
L2-normalizes text internally, so this returns the UN-normalized residual sum (same contract
as TextRelationalAdapter).

Expected-failure log (the iteration targets, watch these when it trains):
  F1 "collapse to prototype": if the residual ||g(t) - t|| grows large and attn just copies
     the bank, g(t) -> visual prototype (which we know does not move detection mAP).
     Mitigation: keep alpha small, weight-decay o_proj, monitor cos(g(t), bank_c) + ||g-t||.
  F2 "base flat": base detection mAP is recall-bound, so even a better classifier may not
     move it -> report base CLASS-MACRO (rare classes) + lean on novel as the headline.
  F3 "novel capped": the bank has no novel prototype, so novel text attends to base only;
     the image encoder also maps novel->base. Expect partial novel gain, not a fix.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class PrototypeGroundedTextAdapter(nn.Module):
    """Text-as-query cross-attention to an EMA image-prototype bank (D-v2).

    Args:
        dim: text / prototype channel dim (512 for BiomedCLIP, 768 for XLM-R). The bank
            and the text MUST share this dim (the head compares them in the same space).
        num_classes: C (30 for dev30). Bank is [C, dim].
        num_heads: multi-head cross-attention heads.
        momentum: EMA momentum for the prototype bank.
        gate_init: initial value of the residual gate (0.0 -> identity at start).
    """

    def __init__(self,
                 dim: int = 512,
                 num_classes: int = 30,
                 num_heads: int = 4,
                 momentum: float = 0.9,
                 alpha: float = 1.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.momentum = momentum

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        # zero-init the OUTPUT proj + a scalar gate -> CrossAttn contributes 0 at start,
        # so g(t) == t exactly (identity). Both are plain nn params (survive mmengine
        # init_weights since this is a leaf nn.Module with no init override).
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        self.alpha = float(alpha)

        # EMA prototype bank (buffers -> follow .to(device), excluded from optimizer).
        self.register_buffer("bank", torch.zeros(num_classes, dim))
        self.register_buffer("bank_count", torch.zeros(num_classes))

    # ---- EMA bank update (no grad; stop-grad teacher -> no leakage) -----------
    @torch.no_grad()
    def update_bank(self, region_feats: torch.Tensor, labels: torch.Tensor) -> None:
        """region_feats: [N, dim] GT-region image features (e.g. BN(cls_embed)).
        labels: [N] class ids. DDP: all-gather region_feats/labels across ranks so the EMA
        bank sees the FULL distributed batch -- otherwise buffer broadcast keeps only rank0's
        shard and the bank learns from 1/world_size of the data. [audit 2026-06-08]"""
        from .visual_prototype_anchor import _all_gather_var
        region_feats = _all_gather_var(region_feats)
        labels = _all_gather_var(labels)
        for c in labels.unique().tolist():
            if c < 0 or c >= self.num_classes:
                continue
            mean_c = region_feats[labels == c].mean(dim=0)
            if self.bank_count[c] == 0:
                self.bank[c] = mean_c
            else:
                self.bank[c] = self.momentum * self.bank[c] + (1 - self.momentum) * mean_c
            self.bank_count[c] += 1

    # ---- text-as-query cross-attention to the bank ----------------------------
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """text: [C, dim] or [B, C, dim]. Returns de-coned text, same shape.
        Identity until the bank has been populated (warmup-safe)."""
        squeeze = text.dim() == 2
        t = text.unsqueeze(0) if squeeze else text          # [B, C, dim]
        B, C, D = t.shape

        valid = self.bank_count > 0                          # [num_classes]
        if int(valid.sum()) < 1:                             # no prototypes yet -> identity
            return text
        # K/V from the (stop-grad, L2-normed) valid prototypes; Q from the text.
        bank = F.normalize(self.bank[valid].detach(), dim=-1)  # [Cv, dim]
        Cv = bank.shape[0]

        def heads(x, n):
            return x.reshape(n, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q = heads(self.q_proj(t), B)                         # [B, h, C, hd]
        k = heads(self.k_proj(bank).unsqueeze(0).expand(B, -1, -1), B)  # [B, h, Cv, hd]
        v = heads(self.v_proj(bank).unsqueeze(0).expand(B, -1, -1), B)  # [B, h, Cv, hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale        # [B, h, C, Cv]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, C, D)    # [B, C, dim]
        out = self.o_proj(out)

        g = t + self.alpha * out                             # zero-init o_proj -> 0 at start
        return g.squeeze(0) if squeeze else g


# ---- standalone unit test (no mmengine needed): shapes, identity, grad isolation ----
if __name__ == "__main__":
    torch.manual_seed(0)
    C, D = 30, 512
    ad = PrototypeGroundedTextAdapter(dim=D, num_classes=C, num_heads=4)
    text = F.normalize(torch.randn(C, D), dim=-1)

    # 1) identity before any bank update (no prototypes) -> exact passthrough
    out0 = ad(text)
    assert torch.allclose(out0, text), "should be identity with empty bank"

    # 2) identity at init even WITH a bank (zero-init o_proj + gate=0 -> residual 0)
    feats = F.normalize(torch.randn(200, D), dim=-1)
    labels = torch.randint(0, C, (200,))
    ad.update_bank(feats, labels)
    out1 = ad(text)
    assert torch.allclose(out1, text, atol=1e-6), "should be identity at init (gate=0)"

    # 3) after o_proj trains away from zero, output moves + grad flows to the adapter
    #    (q/k/v/o_proj), NOT the bank (stop-grad buffer)
    with torch.no_grad():
        ad.o_proj.weight.normal_(std=0.1)
        ad.o_proj.bias.normal_(std=0.1)
    out2 = ad(text)
    assert not torch.allclose(out2, text), "trained o_proj should change the text"
    loss = (out2 ** 2).sum()
    loss.backward()
    assert ad.q_proj.weight.grad is not None and ad.q_proj.weight.grad.abs().sum() > 0
    assert ad.bank.grad is None, "bank is a stop-grad buffer -> no grad"
    # 4) batched input shape
    assert ad(text.unsqueeze(0).expand(3, -1, -1)).shape == (3, C, D)
    print("[ok] PrototypeGroundedTextAdapter: identity@init, grad->adapter not bank, shapes ok")
