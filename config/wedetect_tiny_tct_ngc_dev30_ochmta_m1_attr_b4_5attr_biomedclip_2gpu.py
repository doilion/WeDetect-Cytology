_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b4_morph6_biomedclip_2gpu.py"]

# B4 / 5-attr variant (2026-06-08): module 2 only, but with the 5-ATTRIBUTE text
# (specimen-organ, PSC-diagnosis, 3x morphology) instead of morph6. The per-attribute
# analysis showed the specimen-organ axis is non-discriminative within-organ (cos 0.92) —
# so this variant tests whether module 2 LEARNS to down-weight it (a nice interpretability
# story), at the cost of mixing non-morphology axes. Ablate vs the morph6 B4.

model = dict(
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                attr_text_path="data/texts/tct_ngc_5attr_30_per_attr_biomedclip.pth",
            ),
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b4_5attr_biomedclip_2gpu"
