_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b5_morph6_biomedclip_2gpu.py"]

# B3+D (B6, 2026-06-09): the FULL model = "Deconed Attribute-Adaptive Classifier + visual
# prototype grounding". B3 (per-attr whitening + instance-adaptive attribute weighting) PLUS
# a visual-grounded branch fused at the LOGIT level inside the attribute head:
#       s_final = s_attr  +  lambda * <BN(x), VisualPrototypeAnchor(text)>
#   - VisualPrototypeAnchor (text_decone) grounds the per-class text in an EMA bank of the
#     class's own visual prototypes (class-owned, multi-proto, BASE-ONLY -> novel rows pass
#     through unchanged, so dv2 is self-gated off on novel and cannot hurt it).
#   - head_module.attr_fusion.visual_fuse_weight = lambda mixes that grounded-text logit into
#     the attribute logit. lambda=0 -> byte-identical to B3 (regression-safe). Sweep {0.3,0.5,1.0}.
#
# This is the base/novel division of labor: novel handled by the de-coned attribute text
# (text side), base ADDITIONALLY grounded in vision. Decisive matrix readout:
#   B3+D base  > B3 base   (visual grounding adds on the seen classes)
#   B3+D novel ~ B3 novel  (dv2 self-gates off on novel -> no harm)
# If B3+D base ~ B3 base (recall cap eats it), dv2 drops to an ablation; don't pre-judge.
#
# Bank update REUSES the detector text_decone plumbing (_decone_text + _gt_region_features) ->
# no new detector code. The 2nd cls_preds[0] pass for the bank is a transient [B,512,128,128]
# tensor; prefer --amp, drop train batch to 4 if it OOMs @1024.

model = dict(
    text_decone=dict(
        type="VisualPrototypeAnchor",
        dim=512,          # BiomedCLIP text + cls_embed_bn share the 512-d contrastive space
        num_classes=30,   # bank size = # base classes (GT labels are base)
        num_protos=3,
        num_heads=4,
        alpha=0.5,
    ),
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                visual_fuse_weight=0.5,   # lambda; 0 == B3. Sweep {0.3,0.5,1.0}.
                # NOTE: s_vis is fused PRE-logit_scale, sharing the scale with the
                # gamma/beta-calibrated s_attr -> the EFFECTIVE lambda drifts from the
                # nominal as attr_gamma departs from 1. Log s_attr.abs().mean() vs
                # s_vis.abs().mean() at step 1; the {0.3,0.5,1.0} sweep brackets the drift.
            ),
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b5_dv2_morph6_biomedclip_2gpu"
