_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b5_morph6_biomedclip_2gpu.py"]

# COMBINED model = Module 1 + Module 2 in ONE detector (the "缝合" / stitch).
#
# Why it composes (the key insight): the region-adaptive attribute head computes, at
# every position, s[c] = Σ_a w_a·(x·at[c,a]) = x·(Σ_a w_a·at[c,a]) — i.e. an effective
# per-position CLASS-LEVEL text vector (an instance-weighted blend of the per-attribute
# vectors). So both modules ultimately act on class-level text and can stack.
#
# Inherited from attr_b5 (Module 1): RegionAttributeFusion (region-adaptive weighting)
#   + per_attr_whiten (de-cone on the per-attribute text).
# Added here (Module 2): the learned relational-distillation path on the CLASS-LEVEL text
#   (TextRelationalAdapter + WhiteningTextDecone + RelationalDistillationLoss), wired into
#   the attribute head via visual_fuse_weight>0 — the head's forward(x, w) consumes that
#   de-coned, relational-distilled class text `w` as an additive logit term. This is LATE
#   logit fusion of the two modules' classification signals (both regression-safe at init:
#   zero-init adapter = identity, visual_fuse adds a small term, gate starts at mean).
#
# (A cleaner "shared-text" flavor would instead put the relational-distill adapter directly
# on the per-attribute text `at`, so Module 1's effective class text is ALREADY aligned —
# that needs a few lines of code. This config is the config-only late-fusion version.)
#
# Opt-in: run only after the individual arms show both modules earn their place.

organ_mask_path = "data/texts/tct_ngc_class_organ_mask_base30.pt"

model = dict(
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                visual_fuse_weight=0.5,   # consume the class-level distilled text (0 == Module 1 only)
            ),
        ),
    ),
    backbone=dict(
        text_model=dict(
            text_adapter=dict(
                type="TextRelationalAdapter", dim=512, hidden_dim=256, alpha=0.1,
            ),
        ),
    ),
    text_decone=dict(type="WhiteningTextDecone", topk=1),
    relational_distill_loss=dict(
        type="RelationalDistillationLoss",
        num_classes=30,
        teacher_source="cls_embed_bn",   # discriminative teacher (smoke: neck is collapsed 0.967 > text)
        student_dim=512,
        roi_level=0,
        organ_mask_path=organ_mask_path,
        momentum=0.9,
        loss_weight=10.0,
        max_rois=256,
        warmup_steps=100,
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_stitch_res1024_biomedclip_2gpu"
