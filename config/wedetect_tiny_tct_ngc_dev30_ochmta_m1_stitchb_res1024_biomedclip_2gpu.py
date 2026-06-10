_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b5_morph6_biomedclip_2gpu.py"]

# COMBINE flavor B (shared-text): Module 1 + Module 2 funnel through ONE per-attribute text.
#
# Your insight: region-adaptive computes s[c] = Σ_a w_a·(x·at[c,a]) = x·(Σ_a w_a·at[c,a]),
# i.e. an effective per-position CLASS-LEVEL text. So instead of late-fusing a SEPARATE
# class text (flavor A `stitch`, via visual_fuse_weight), we put the learned alignment
# directly on the per-attribute text `at` that the region-adaptive head already uses.
#
# Inherited from attr_b5 (Module 1): RegionAttributeFusion + per_attr_whiten (fixed de-cone on `at`).
# Added (Module 2, on the SAME text):
#   - attr_fusion.attr_text_adapter = TextRelationalAdapter  -> learnable residual on `at`
#     (identity at init; order = de-cone -> adapter -> region-adaptive fusion).
#   - relational_distill_loss = RelationalDistillationLoss     -> supervises the adapter's
#     class-mean text (normalize(mean_a at_adapted)) toward the image region-prototype
#     within-organ cosine matrix (the detector auto-routes this student when the attribute
#     head exposes last_class_text). Image teacher is stop-grad; the adapter is trained by
#     this loss + the main cls loss.
#
# Regression-safe at init (zero-init adapter = identity, gate starts at attribute mean).
# Watch step-0 print `image-proto cos vs text cos` for well-posedness. Opt-in; run after the
# individual arms show both modules earn it. fp32 (no AMP).

model = dict(
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                attr_text_adapter=dict(
                    type="TextRelationalAdapter", dim=512, hidden_dim=256, alpha=0.1,
                ),
            ),
        ),
    ),
    relational_distill_loss=dict(
        type="RelationalDistillationLoss",
        num_classes=30,
        teacher_source="cls_embed_bn",   # discriminative teacher (smoke: neck is collapsed 0.967 > text)
        student_dim=512,
        roi_level=0,
        organ_mask_path="data/texts/tct_ngc_class_organ_mask_base30.pt",
        momentum=0.9,
        loss_weight=10.0,
        max_rois=256,
        warmup_steps=100,
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_stitchb_res1024_biomedclip_2gpu"
