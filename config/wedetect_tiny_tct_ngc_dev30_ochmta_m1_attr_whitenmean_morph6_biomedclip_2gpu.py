_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b5_morph6_biomedclip_2gpu.py"]

# B1 (2026-06-09): MODULE 1 ONLY = per-attribute whitening + FIXED mean attribute pooling.
# This is attr_b5 (whiten + adaptive) with the per-location adaptive gate turned OFF
# (attr_fusion.adaptive=False -> uniform attribute weights = classify against the whitened
# mean attribute prototype). It completes the de-collapse ablation matrix:
#   B0 = morph6mean                  : raw mean,      no whiten, no adaptive
#   B1 = this (attr_whitenmean)      : whitened mean, whiten,    no adaptive   <- "whiten only"
#   B2 = attr_b4_morph6              : raw per-attr,  no whiten, adaptive
#   B3 = attr_b5_morph6              : whitened,      whiten,    adaptive
# Decisive: B1>B0 => whitening is not decorative ; B3>B1 => the adaptive gate is not decorative.
# Everything else (per_attr_whiten lam=0.5, morph6 text, @1024) inherited from B5.

model = dict(
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                adaptive=False,   # fixed uniform (mean) weights; per_attr_whiten kept from B5
                balance_weight=0.0,
            ),
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_whitenmean_morph6_biomedclip_2gpu"
