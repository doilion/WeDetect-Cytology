"""Design A — Image-Conditional Fusion (ICF).

Replaces THAF's class-agnostic learnable fusion_query with an image-derived
context vector. The fused per-class embedding now varies per image, breaking
the mean-pool basin of attraction that killed THAF (α→0 collapse, memory
DEAD-11 / feedback_thaf_fusion_bypassed.md).

Architecture (F2 — with per-attribute expert MLPs, ~2.4M params):

    image_top_feat [B, C_img, H, W]
        -> adaptive_avg_pool2d 1x1                 # global context
        -> Linear image_proj : C_img -> D          # match text dim
        -> image_ctx [B, D]

    attr_emb [B, C, A, D]      (cached BiomedCLIP 5-attr embeddings)
        -> for a in range(A): attr_experts[a](attr_emb[:,:,a,:])
                                                   # per-attribute MLP
        -> attr_adapted [B, C, A, D]
        + attr_type_pe positional emb
        -> K, V flattened to [B*C, A, D]

    Q := image_ctx broadcast over C classes -> [B*C, 1, D]
    cross-attn(Q, K, V)
        -> fused [B*C, 1, D]
    output_proj + L2-norm
        -> [B, C, D]

NO residual blend back to attr_mean (THAF's α gate gave optimizer an escape
hatch). NO 4D expansion MLP (kept small for stable cold start). Pre-norm
attention for training stability.

The per-attribute experts are STATIC routing — attribute index decides which
expert handles it. Unlike M2 Stage 2 per-organ MoE (which split training
data by organ and failed novel zero-shot), every training sample contributes
to every attribute expert, so novel-class attribute_a vectors are processed
by an expert trained on the full data distribution.

Collapse diagnostics consumed by ICFCollapseGuard hook (3 metrics):
  - fused_pairwise_cos_mean: mean off-diagonal cosine between fused class
        vectors of the SAME class across images in the batch. Should be in
        [0.5, 0.95]. > 0.99 means image-invariant fusion = mean-pool collapse.
  - attn_entropy_mean: mean entropy of attention over 5 attribute keys.
        Should be below log(5) ≈ 1.609; > 1.58 means uniform = mean pool.
  - cos_to_attr_mean_mean: cosine between fused output and (normalized)
        attr_mean. Should be < 0.95; > 0.97 means fused == mean direction.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class ImageConditionalFusion(BaseModule):
    """Image-conditional cross-attention pooling over text attributes."""

    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 768,
        num_attrs: int = 5,
        attr_hidden: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        attr_dropout: float = 0.0,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg)
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.num_attrs = num_attrs
        # attr_dropout: per-image, per-attribute Bernoulli drop probability.
        # During training, for each sample in the batch we independently sample
        # which of the A=num_attrs attributes to mask out via the cross-attn
        # ``key_padding_mask`` argument — masked attributes contribute zero to
        # the softmax-normalized fused output. At least one attribute always
        # survives per sample (if all sampled True, one is randomly un-masked).
        # Acts as a regularizer that forces the image-conditional cross-attn to
        # learn from any 4 (or 3) of the 5 attributes, which empirically helps
        # zero-shot generalization to novel classes whose individual attributes
        # may be unevenly informative (e.g. Thyroid MTC's attr_4 dominates).
        # Applied ONLY in ``forward`` (main path); ``forward_text_only`` (STEGO
        # anchor path) intentionally skips dropout to keep the teacher path
        # clean. Eval mode bypasses dropout entirely (standard ``self.training``
        # check).
        self.attr_dropout = float(attr_dropout)

        # (a) Image global context -> text dim
        self.image_proj = nn.Linear(image_dim, text_dim)

        # (b) Per-attribute expert MLPs (F2). 5 parallel attribute-specific
        # 2-layer MLPs, each ≈ 131K params. Static routing: attribute index
        # decides which expert. NOT routing-style MoE.
        self.attr_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, attr_hidden),
                nn.GELU(),
                nn.Linear(attr_hidden, text_dim),
            )
            for _ in range(num_attrs)
        ])

        # (c) Per-attribute positional embedding (so cross-attn can
        # distinguish which of the 5 attributes a key came from)
        self.attr_type_pe = nn.Embedding(num_attrs, text_dim)

        # (d) Pre-norm both sides of cross-attention
        self.norm_q = nn.LayerNorm(text_dim)
        self.norm_kv = nn.LayerNorm(text_dim)

        # (e) Single cross-attn block — Q from image context, K/V from 5 attrs
        self.cross_attn = nn.MultiheadAttention(
            text_dim, num_heads, dropout=dropout, batch_first=True
        )

        # (f) Output projection (NO 4D expansion, NO residual back to attr_mean)
        self.output_proj = nn.Linear(text_dim, text_dim)

        self._init_weights()

        # Live diagnostic buffer (populated each forward, consumed by
        # ICFCollapseGuard hook; cleared on each forward call).
        self._diag: Dict[str, torch.Tensor] = {}

    def _init_weights(self) -> None:
        # image_proj / output_proj: orthogonal init at scale 0.5 — non-identity
        # but not too aggressive. image_proj at init gives a usable image
        # context vector even before training.
        nn.init.orthogonal_(self.image_proj.weight, gain=0.5)
        nn.init.zeros_(self.image_proj.bias)
        nn.init.orthogonal_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)

        # Per-attribute experts: orthogonal init scaled 0.5 so each expert
        # starts as a non-trivial transform of its attribute.
        for expert in self.attr_experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

        # attr_type positional emb: small Gaussian (standard transformer style)
        nn.init.normal_(self.attr_type_pe.weight, std=0.02)

        # Cross-attn weights: xavier with moderate gain (0.5). Smaller gains
        # (e.g. 0.1, which THAF used) leave Q ≈ 0 at init -> attention is
        # uniform -> output becomes image-invariant noise. Larger gain (0.5)
        # keeps the Q-K similarity strong enough that img_ctx differences
        # actually shift the attention pattern at init.
        nn.init.xavier_uniform_(self.cross_attn.in_proj_weight, gain=0.5)
        nn.init.xavier_uniform_(self.cross_attn.out_proj.weight, gain=0.5)
        if self.cross_attn.in_proj_bias is not None:
            nn.init.zeros_(self.cross_attn.in_proj_bias)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(
        self,
        image_top_feat: torch.Tensor,
        attr_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse per-class attribute embeddings conditioned on image context.

        Args:
            image_top_feat: [B, C_img, H, W] pre-neck deepest backbone feature
                (ConvNext-tiny stage 4 = 768d, stride 32).
            attr_emb: [B, C, A, D] cached per-class 5-attribute embeddings
                (L2-normalized upstream by PseudoMultiAttrLanguageBackbone
                running with pool_mode='none').

        Returns:
            [B, C, D] image-conditional class embeddings, L2-normalized.
        """
        if attr_emb.dim() != 4:
            raise ValueError(
                f'ImageConditionalFusion expects attr_emb shape [B, C, A, D] '
                f'(raw 5-attr embeddings); got shape {tuple(attr_emb.shape)}. '
                f'Check that text_model has pool_mode="none" set in the '
                f'PseudoMultiAttrLanguageBackbone config.'
            )
        B, C, A, D = attr_emb.shape
        assert A == self.num_attrs, (
            f'expected {self.num_attrs} attrs, got A={A}')
        assert D == self.text_dim, (
            f'attr_emb dim {D} != text_dim {self.text_dim}')

        # 1. Image -> global context vector in text space
        img_global = F.adaptive_avg_pool2d(image_top_feat, 1).flatten(1)
        if img_global.shape[1] != self.image_dim:
            raise ValueError(
                f'image_top_feat has {img_global.shape[1]} channels, expected '
                f'image_dim={self.image_dim}. Check backbone stage choice.')
        img_ctx = self.image_proj(img_global)                       # [B, D]

        # 2. Per-attribute expert MLPs (F2)
        attr_adapted = torch.stack(
            [self.attr_experts[a](attr_emb[:, :, a, :]) for a in range(A)],
            dim=2,
        )                                                            # [B, C, A, D]

        # 3. Add per-attribute positional embedding to K/V
        attr_pe = self.attr_type_pe.weight.view(1, 1, A, D)
        kv = (attr_adapted + attr_pe).reshape(B * C, A, D)

        # 4. Build Q (image_ctx broadcast to C classes -> [B*C, 1, D])
        q = img_ctx.unsqueeze(1).expand(B, C, D).reshape(B * C, 1, D)

        # 4b. Optional per-sample attribute dropout for the cross-attention.
        # Drops whole attribute slots (same drop pattern across all C classes
        # of a sample, since the per-image cross-attn is a sample-level
        # decision). Drop is realized via ``key_padding_mask`` so softmax
        # truly excludes the masked attribute, not just zeros its V.
        key_padding_mask: Optional[torch.Tensor] = None
        if self.training and self.attr_dropout > 0:
            # drop[b, a] = True means attr a is dropped for sample b
            drop = (
                torch.rand(B, A, device=attr_emb.device) < self.attr_dropout
            )
            # Ensure at least one attribute survives per sample. If all A
            # attrs were sampled True, randomly un-mask one of them.
            all_dropped = drop.all(dim=1)
            if all_dropped.any():
                rand_keep = torch.randint(
                    0, A, (B,), device=attr_emb.device
                )
                # mask out the random-keep column for all-dropped samples only
                idx_b = torch.arange(B, device=attr_emb.device)[all_dropped]
                drop[idx_b, rand_keep[all_dropped]] = False
            # Broadcast to [B, C, A] then collapse the (B, C) axes to
            # match cross_attn's [B*C, A] key_padding_mask convention.
            key_padding_mask = drop.unsqueeze(1).expand(B, C, A).reshape(
                B * C, A
            )

        # 5. Cross-attention
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        fused, attn_weights = self.cross_attn(
            q_norm, kv_norm, kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )                                                            # [B*C, 1, D], [B*C, 1, A]
        fused = fused.squeeze(1)                                     # [B*C, D]

        # 6. Output projection + L2 norm
        out = self.output_proj(fused).reshape(B, C, D)
        out = F.normalize(out, dim=-1)

        # Stash live tensors for the collapse-guard hook (no detach — we
        # want grad-flow checks to be possible). Each forward overwrites.
        # We do NOT keep the adapted attr stack on the live diag (too big);
        # only the small final tensors needed for 3 scalar metrics.
        self._diag = {
            'fused_emb': out,
            'attn_weights': attn_weights.squeeze(1).reshape(B, C, A),
            'attr_mean_normed': F.normalize(attr_emb.mean(dim=2), dim=-1),
        }
        return out

    def forward_text_only(
        self, attr_emb: torch.Tensor
    ) -> torch.Tensor:
        """Image-free anchor path for STEGO-corr loss (Plan v3).

        Structurally bypasses the image branch (image_proj, image input,
        norm_q) and the Q/K projections of ``cross_attn`` so that
        ``W_Q``, ``W_K``, ``b_Q``, ``b_K`` never appear in the autograd
        graph. The path is the analytic solution of ``cross_attn`` when
        attention is uniform over the A attribute keys: for each head
        the head-wise output is ``W_O_head( mean_a W_V_head(kv_a) )``,
        so concatenating heads gives ``W_O( V.mean(dim=A) )`` with
        ``V = F.linear(kv, W_V_slice, b_V_slice)``.

        Two slight divergences from the main forward's data flow,
        forced by the uniform-attn worst-case framing:

        1. We skip ``attr_type_pe`` here. The PE serves to differentiate
           attribute SLOTS so the cross-attention can attend
           preferentially to a subset of the 5 attrs; in the uniform-attn
           limit (mean over all attrs), PE collapses to a class-
           independent constant. Worse, at init the PE (Gaussian std
           0.02, magnitude ≈0.45) DOMINATES the per-class
           ``attr_experts`` output (orthogonal gain 0.5, magnitude
           ≈0.02), so an unmasked V.mean would map all 39 classes onto
           a single direction (pairwise cos > 0.99 — clamp saturates,
           STEGO grad = 0). Skipping PE restores the per-class
           discriminative structure (init pairwise cos ≈ 0.46–0.89,
           ~97% of off-diag S in (0, 0.8)). Main forward still trains
           PE; STEGO simply does not.
        2. We apply ``norm_kv`` to the V input even though main forward
           feeds V the unnormalized ``kv`` (norm_kv is K-only there).
           Reason: with the orthogonal-init experts giving tiny
           magnitudes, the LayerNorm pulls per-token variance back to
           1 so V.mean is not noise. norm_kv has ~1024 params and is
           already trained by the main forward via K; the small extra
           STEGO gradient is harmless cross-regularization.

        Receive grad (YES):
          - ``attr_experts[*].weight/bias``
          - ``norm_kv.{weight, bias}``
          - ``cross_attn.in_proj_weight[2D:3D]``  (W_V slice)
          - ``cross_attn.in_proj_bias[2D:3D]``   (b_V slice)
          - ``cross_attn.out_proj.{weight, bias}`` (W_O)
          - ``output_proj.{weight, bias}``
        Receive None / no grad (NO):
          - ``image_model.*``, ``image_proj.*``, ``norm_q.*``
          - ``attr_type_pe.weight``  (skipped on this path; main forward
            still trains it via the cross-attn K branch)
          - ``cross_attn.in_proj_weight[0:2D]``  (W_Q + W_K slice)
          - ``cross_attn.in_proj_bias[0:2D]``   (b_Q + b_K slice)

        Does NOT write ``self._diag`` (so ICFCollapseGuard keeps reading
        the most recent main-forward batch diagnostic).

        Args:
            attr_emb: ``[B, C, A, D]`` raw teacher attribute embeddings.
                B=1 is the typical call site for the STEGO 39-class anchor;
                larger B is supported.

        Returns:
            ``[B, C, D]`` L2-normalized student embeddings.
        """
        if attr_emb.dim() != 4:
            raise ValueError(
                f"forward_text_only expects [B, C, A, D]; got "
                f"{tuple(attr_emb.shape)}"
            )
        B, C, A, D = attr_emb.shape
        if A != self.num_attrs:
            raise ValueError(
                f"expected {self.num_attrs} attrs, got A={A}"
            )
        if D != self.text_dim:
            raise ValueError(
                f"attr_emb dim {D} != text_dim {self.text_dim}"
            )

        # 1. Per-attribute experts (shared with main forward).
        attr_adapted = torch.stack(
            [self.attr_experts[a](attr_emb[:, :, a, :]) for a in range(A)],
            dim=2,
        )  # [B, C, A, D]

        # 2. Skip attr_type_pe (see docstring): PE collapses to a class-
        #    independent constant under uniform attention, and at init
        #    its magnitude dominates the experts' output, causing a 39-
        #    class direction collapse.
        kv = attr_adapted.reshape(B * C, A, D)

        # 3. LayerNorm before V (slight divergence from main forward —
        #    see docstring). Necessary because the tiny experts output
        #    magnitude would otherwise leave V.mean as essentially noise.
        kv = self.norm_kv(kv)

        # 4. Manual V projection — slice W_V / b_V out of the packed
        #    QKV proj; Q and K projections are not invoked, so their
        #    parameters stay outside the autograd graph for this call.
        Wv = self.cross_attn.in_proj_weight[2 * D : 3 * D, :]
        bv = (
            self.cross_attn.in_proj_bias[2 * D : 3 * D]
            if self.cross_attn.in_proj_bias is not None
            else None
        )
        V = F.linear(kv, Wv, bv)  # [B*C, A, D]

        # 5. Uniform attention closed form: per-head softmax of zero scores
        #    is uniform over A, so head-wise mean over A reproduces the
        #    cross_attn output. Multi-head safe: concat-of-heads(mean V_h)
        #    == mean V because the head split is along the last axis only.
        attended = V.mean(dim=1)  # [B*C, D]

        # 6. W_O (= cross_attn.out_proj) — shared with main forward.
        out_attn = self.cross_attn.out_proj(attended)  # [B*C, D]

        # 7. Output projection + L2 norm — shared with main forward.
        out = self.output_proj(out_attn).reshape(B, C, D)
        out = F.normalize(out, dim=-1)
        return out

    @torch.no_grad()
    def get_collapse_diagnostics(self) -> Dict[str, float]:
        """Three scalar health metrics consumed by ICFCollapseGuard hook.

        All metrics are well-scaled (in [0, 1] or [0, log(A)]) so thresholds
        in the guard hook don't depend on D.
        """
        diag: Dict[str, float] = {}
        if not self._diag:
            return diag

        fused = self._diag['fused_emb'].detach()                     # [B, C, D]
        attn = self._diag['attn_weights'].detach()                   # [B, C, A]
        attr_mean = self._diag['attr_mean_normed'].detach()          # [B, C, D]

        # (1) Mean pairwise cosine between fused vectors of the SAME class
        # across different images in the batch. Image-conditional fusion
        # should produce different outputs for different images of the same
        # class; collapse to mean-pool would produce identical outputs
        # (cos = 1). Range: [-1, 1], healthy ~ [0.5, 0.95], red > 0.99.
        B_, C_, D_ = fused.shape
        if B_ >= 2:
            # cos_matrix[b, a, c] = cos(fused[b, c], fused[a, c]); shape [B, B, C]
            cos_matrix = torch.einsum('bcd,acd->bac', fused, fused)
            eye = torch.eye(B_, dtype=torch.bool, device=fused.device)
            off_diag_mask = (~eye).unsqueeze(-1).expand_as(cos_matrix)
            mean_pairwise_cos = cos_matrix[off_diag_mask].mean()
            diag['fused_pairwise_cos_mean'] = float(mean_pairwise_cos.item())
            diag['fused_pairwise_dist_mean'] = float(
                (1.0 - mean_pairwise_cos).item())
        else:
            diag['fused_pairwise_cos_mean'] = float('nan')
            diag['fused_pairwise_dist_mean'] = float('nan')

        # (2) Mean attention entropy over the 5 attribute keys.
        # log(num_attrs) is uniform = effectively mean pool.
        entropy = -(attn * attn.clamp(min=1e-9).log()).sum(dim=-1)
        diag['attn_entropy_mean'] = float(entropy.mean().item())

        # (3) Mean cosine to attr_mean direction.
        # 1.0 means fused output == mean pool direction.
        cos = (fused * attr_mean).sum(dim=-1)
        diag['cos_to_attr_mean_mean'] = float(cos.mean().item())

        return diag
