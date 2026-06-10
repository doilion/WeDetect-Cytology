"""D-v2 v2 (2026-06-08): multi-prototype, class-owned visual prototype anchoring.

Iterated from D-v1 (relational, gauge-freedom) and D-v1.5 (PrototypeGroundedTextAdapter:
single-mean bank, GLOBAL attention). Per the 2026-06-08 design review:

  - MULTI-PROTOTYPE bank [C, K, D]: a single class mean blurs multi-modal cell appearance;
    K prototypes (online nearest-prototype EMA) keep the class's modes.
  - CLASS-OWNED attention: text_c attends ONLY to bank[c, :K] (its own class's prototypes),
    NOT to all C*K prototypes. Global attention pulls same-organ fine-classes together;
    class-owned grounds each class in its own visual modes (de-coning/spreading is module 1's
    job, not this one's).
  - BASE-ONLY: novel classes have no GT -> no prototype -> they pass through UNCHANGED
    (identity). So this is a BASE visual-anchoring branch; novel is protected (its gains come
    from per-attr whitening + attribute weighting, NOT from being pulled to a base prototype).
  - DDP all-gather: the bank is a buffer; without gathering region features/labels across
    ranks the bank would only see rank0's data (broadcast_buffers overwrites the rest).

Intended use: a SEPARATE class-level logit-fusion branch (s_final = s_attr + lam * s_visual),
NOT a transform inside the per-attribute pipeline (a class-level prototype would flatten the
per-attribute structure). The head integration (B6) is deferred; this module + its bank are
unit-tested standalone here.
"""
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@torch.no_grad()
def _all_gather_var(t: torch.Tensor):
    """All-gather variable-length tensors across DDP ranks (no_grad). Returns the
    concatenation of every rank's tensor; identity if DDP is not initialized."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return t
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, t.detach().cpu())
    return torch.cat([g.to(t.device) for g in gathered], dim=0)


@MODELS.register_module()
class VisualPrototypeAnchor(nn.Module):
    """Class-owned multi-prototype visual anchoring of class text (base-only).

    Args:
        dim: text / prototype channel dim (512 BiomedCLIP).
        num_classes: # BASE classes (bank size; novel are not banked).
        num_protos: K prototypes per class.
        num_heads: attention heads.
        momentum: EMA momentum for prototype update.
        alpha: residual strength g(t)=t+alpha*attn (small; sweep 0.25/0.5).
    """

    def __init__(self, dim: int = 512, num_classes: int = 30, num_protos: int = 3,
                 num_heads: int = 4, momentum: float = 0.9, alpha: float = 0.5) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.K, self.num_classes = dim, int(num_protos), int(num_classes)
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.momentum, self.alpha = momentum, float(alpha)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.o_proj.weight); nn.init.zeros_(self.o_proj.bias)  # identity start
        self.register_buffer("bank", torch.zeros(num_classes, self.K, dim))
        self.register_buffer("bank_count", torch.zeros(num_classes, self.K))

    @torch.no_grad()
    def update_bank(self, region_feats: torch.Tensor, labels: torch.Tensor) -> None:
        """region_feats [N, D], labels [N]. DDP all-gather -> nearest-prototype EMA."""
        region_feats = _all_gather_var(region_feats)
        labels = _all_gather_var(labels)
        for c in labels.unique().tolist():
            if c < 0 or c >= self.num_classes:
                continue
            feats = region_feats[labels == c]                 # [n_c, D]
            empty = self.bank_count[c] == 0                    # [K]
            if empty.any():                                    # seed empty prototypes first
                for k in empty.nonzero().squeeze(1).tolist():
                    if feats.shape[0] == 0:
                        break
                    self.bank[c, k] = feats[0]
                    self.bank_count[c, k] += 1
                    feats = feats[1:]
                if feats.shape[0] == 0:
                    continue
            # assign remaining feats to nearest prototype of class c, EMA per prototype
            d = torch.cdist(feats, self.bank[c])               # [n_c, K]
            assign = d.argmin(dim=1)                           # [n_c]
            for k in assign.unique().tolist():
                mean_k = feats[assign == k].mean(dim=0)
                self.bank[c, k] = self.momentum * self.bank[c, k] + (1 - self.momentum) * mean_k
                self.bank_count[c, k] += 1

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """text [C, D] (or [B, C, D]). Class-owned attention to each class's own K prototypes;
        BASE-ONLY: classes with an empty bank (novel) pass through unchanged."""
        squeeze = text.dim() == 2
        t = text.unsqueeze(0) if squeeze else text            # [B, C, D]
        B, C, D = t.shape
        out = t
        n_base = min(C, self.num_classes)
        valid = (self.bank_count[:n_base].sum(dim=1) > 0)      # [n_base] which base classes banked
        if int(valid.sum()) > 0:
            tb = t[:, :n_base, :]                              # [B, n_base, D]
            bank = F.normalize(self.bank[:n_base].detach(), dim=-1)   # [n_base, K, D]

            def heads(x, n_tok):
                return x.reshape(B, n_base, n_tok, self.num_heads, self.head_dim
                                 ).permute(0, 1, 3, 2, 4)      # [B, n_base, h, tok, hd]
            q = heads(self.q_proj(tb).unsqueeze(2), 1)         # [B,n_base,h,1,hd]
            k = heads(self.k_proj(bank).unsqueeze(0).expand(B, -1, -1, -1), self.K)
            v = heads(self.v_proj(bank).unsqueeze(0).expand(B, -1, -1, -1), self.K)
            scores = q @ k.transpose(-2, -1) * self.scale      # [B,n_base,h,1,K]
            # mask UNSEEDED prototypes (bank_count==0): their key is a zero vector and would
            # otherwise dilute the softmax with a content-free slot. A class is only in this
            # branch if >=1 of its K prototypes is seeded, so no all-(-inf) row. [code-review]
            unseeded = (self.bank_count[:n_base] == 0).view(1, n_base, 1, 1, self.K)
            scores = scores.masked_fill(unseeded, float('-inf'))
            attn = scores.softmax(dim=-1)                      # [B,n_base,h,1,K]
            o = (attn @ v).permute(0, 1, 3, 2, 4).reshape(B, n_base, D)     # [B,n_base,D]
            o = self.o_proj(o)
            anchored = tb + self.alpha * o
            # only overwrite the banked (base) classes; novel/un-banked stay = original text
            mask = valid.view(1, n_base, 1)
            out = t.clone()
            out[:, :n_base, :] = torch.where(mask, anchored, tb)
        return out.squeeze(0) if squeeze else out


if __name__ == "__main__":
    torch.manual_seed(0)
    C, D, K = 30, 512, 3
    m = VisualPrototypeAnchor(dim=D, num_classes=C, num_protos=K, alpha=0.5)
    text = F.normalize(torch.randn(C, D), dim=-1)

    # 1) empty bank -> identity (everything novel/un-banked)
    assert torch.allclose(m(text), text), "empty bank must be identity"

    # 2) populate bank for the first 20 classes (base); 21..29 stay novel
    feats = F.normalize(torch.randn(400, D), dim=-1)
    labels = torch.randint(0, 20, (400,))
    m.update_bank(feats, labels)
    assert (m.bank_count[:20].sum(1) > 0).all() and (m.bank_count[20:].sum() == 0)

    # 3) o_proj zero-init -> still identity at start even WITH a bank
    assert torch.allclose(m(text), text, atol=1e-6), "identity at init (o_proj=0)"

    # 4) open o_proj -> base classes (0..19) change, novel (20..29) stay identical
    with torch.no_grad():
        m.o_proj.weight.normal_(std=0.1); m.o_proj.bias.normal_(std=0.1)
    out = m(text)
    base_changed = not torch.allclose(out[:20], text[:20], atol=1e-4)
    novel_same = torch.allclose(out[20:], text[20:], atol=1e-6)
    assert base_changed and novel_same, (base_changed, novel_same)

    # 5) grad to adapter not bank
    out = m(text); out.sum().backward()
    assert m.q_proj.weight.grad is not None and m.bank.grad is None
    print(f"[ok] VisualPrototypeAnchor: empty=identity, multi-proto bank [C,{K},D], "
          f"base-only (novel untouched), identity@init, grad->adapter not bank")
