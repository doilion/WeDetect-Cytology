_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py"]

# Innovation D (2026-06-05): image->text relational distillation.
#
# Identical to Row1 (M1, BiomedCLIP + 1-PSC, organ-loss-mask) in every other
# respect. Adds two opt-in pieces:
#   (1) backbone.text_model.text_adapter = TextRelationalAdapter — a zero-init
#       residual MLP g(.) on the frozen cached text (identity at init -> the
#       cls path is regression-safe at start; only g trains).
#   (2) relational_distill_loss = RelationalDistillationLoss — param-free; builds
#       an EMA bank of per-class image region prototypes (RoIAlign on neck level-0,
#       same recipe as ②) and pulls the adapted-text within-organ cosine matrix
#       toward the image-prototype within-organ cosine matrix (cross-organ masked).
#
# Motivation: the 3-prompt collapse study (fullnames/5attr/morph6 all within-organ
# cos 0.80-0.88, max ~0.96) proved prompt engineering is dead; the only principled
# text-side lever is to import the discriminative IMAGE relational structure
# (linear-probe 0.90) into the collapsed text. loss_weight is provisional — read
# loss_rel_distill off the smoke log and recalibrate onto the cls-loss scale.

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
        student_dim=96,   # neck level-0 channels (= head in_channels[0], stride-8)
        roi_level=0,
        organ_mask_path="data/texts/tct_ngc_class_organ_mask_base30.pt",
        momentum=0.9,
        loss_weight=10.0,
        max_rois=256,
        warmup_steps=100,
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_reldistill_biomedclip_2gpu"
