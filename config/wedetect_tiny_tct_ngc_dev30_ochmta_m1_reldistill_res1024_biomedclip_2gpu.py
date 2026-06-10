_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_res1024_biomedclip_2gpu.py"]

# Visual->Text Relational Distillation @1024 (sibling of the @640 reldistill, re-based
# on the @1024 PSC baseline so it is directly comparable with the rest of the @1024
# method suite). This is the "distill-only" control arm of Module 2 — the early version
# WITHOUT whitening (the one that had the gauge-freedom problem); decone_reldistill adds
# the de-cone that conditions it.
#
# Two opt-in pieces (same as the @640 version):
#   (1) backbone.text_model.text_adapter = TextRelationalAdapter — zero-init residual MLP
#       g(.) on the frozen cached text (identity at init -> regression-safe; only g trains).
#   (2) relational_distill_loss = RelationalDistillationLoss — param-free: builds an EMA
#       bank of per-class image region prototypes (RoIAlign on neck level-0, 96-d) and pulls
#       the adapted-text within-organ cosine matrix toward the image-prototype one
#       (cross-organ pairs masked by the organ prior).
#
# fp32 (no AMP) per the suite default — BiomedCLIP @1024 historically diverges under AMP.
# loss_weight is provisional: read loss_rel_distill off the smoke log and recalibrate onto
# the cls-loss scale. At step 0 the loss prints `image-proto cos vs text cos` — if the image
# teacher is MORE collapsed than the text, it is pulling the wrong way (well-posedness check).

model = dict(
    backbone=dict(
        text_model=dict(
            text_adapter=dict(
                type="TextRelationalAdapter",
                dim=512,
                hidden_dim=256,
                alpha=0.1,
            ),
        ),
    ),
    relational_distill_loss=dict(
        type="RelationalDistillationLoss",
        num_classes=30,
        # Teacher = BN(cls_embed) at level-0 (512-d), the space the head actually compares
        # text<->image in. Smoke (2026-06-10) showed the alternative `neck` teacher is
        # COLLAPSED (within-organ cos 0.967 >> text 0.503) -> it would pull text the wrong
        # way; cls_embed_bn is discriminative (~0.26 < text) -> correct direction.
        teacher_source="cls_embed_bn",
        student_dim=512,   # cls_embed channels (= embed_dims), matches cls_embed_bn teacher
        roi_level=0,
        organ_mask_path="data/texts/tct_ngc_class_organ_mask_base30.pt",
        momentum=0.9,
        loss_weight=10.0,
        max_rois=256,
        warmup_steps=100,
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_reldistill_res1024_biomedclip_2gpu"
