_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_res1024_biomedclip_2gpu.py"]

# D-v2 (2026-06-08): prototype-grounded text de-collapse, @1024 BiomedCLIP.
#
# The iterated version of relational distillation (D-v1). Instead of D-v1's relational
# cos-matrix loss (which had gauge-freedom -> the text drifted off image alignment), the
# text (query) cross-attends to an EMA bank of per-class image-region prototypes and the
# de-coned text is trained END-TO-END with the detection cls loss. The cls loss supplies
# the absolute text<->image alignment anchor D-v1 lacked (no free rotation), and the bank
# is stop-grad + class-level (not the current instance) so there is no leakage.
# Motivation: the cytology text collapse is mostly removable ANISOTROPY (a cone); this
# learns an image-grounded de-cone that should also de-collapse novel prompts at inference
# (the base-fit bank de-cones novel relative to the base appearance manifold).
#
# opt-in via the detector's text_decone hook; everything else (1024 cache, frozen
# BiomedCLIP text, organ mask, seed=42) inherited. Regression-safe: o_proj is zero-init
# so g(t) == t until the adapter learns. A/B partner = the m1_res1024_biomedclip baseline.
#
# Expected-failure log (iterate, don't kill — see the module docstring): F1 collapse to
# prototype (monitor ||g(t)-t||), F2 base flat (recall-bound -> report class-macro),
# F3 novel capped (image encoder maps novel->base; bank has no novel prototype).

# MEMORY (2026-06-08 audit): _gt_region_features does a 2nd cls_preds[0] pass for the bank,
# a transient [B,512,128,128] tensor (~268MB fp32 / ~134MB with --amp, freed after the bank
# update). The inherited @1024 batch is near the 3090 limit -> if this OOMs, uncomment the
# train_batch=4 below (matches the P2 config). Prefer --amp.
# train_batch = 4

# 2026-06-09: switched from PrototypeGroundedTextAdapter (D-v1.5: single-mean bank, GLOBAL
# attention -> novel text also attends base prototypes, which the Procrustes no-transfer
# finding says hurts novel) to VisualPrototypeAnchor (D-v2 v2): multi-prototype, CLASS-OWNED
# attention (text_c attends only bank[c]), and BASE-ONLY (novel classes have no bank -> pass
# through UNCHANGED = identity -> self-gating, cannot hurt novel). This is the "D" arm of the
# experiment matrix and the visual-grounding leg of the base/novel division of labor.
model = dict(
    text_decone=dict(
        type="VisualPrototypeAnchor",
        dim=512,          # BiomedCLIP text + cls_embed_bn share the 512-d contrastive space
        num_classes=30,   # bank size = # base classes (GT labels are base)
        num_protos=3,     # K prototypes/class -> keep a class's multi-modal cell appearance
        num_heads=4,
        alpha=0.5,        # residual strength g(t)=t+alpha*attn (small; sweep 0.25/0.5)
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_dv2_biomedclip_2gpu"
