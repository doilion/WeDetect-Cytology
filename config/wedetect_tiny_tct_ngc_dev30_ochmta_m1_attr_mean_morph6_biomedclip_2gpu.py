_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b4_morph6_biomedclip_2gpu.py"]

# B0 (2026-06-09): the @1024 MEAN baseline of the de-collapse matrix. Raw per-attribute morph6
# text, FIXED uniform (mean) pooling (attr_fusion.adaptive=False), NO whitening. Routed through
# the attribute head so the gamma/beta per-axis calibration is SHARED with B1/B2/B3 and cancels
# in the comparisons -> the deltas isolate exactly whitening (B1) and the adaptive gate (B2).
#   B0 = this           : raw mean,      no whiten, no adaptive
#   B1 = attr_whitenmean : whitened mean, whiten,    no adaptive
#   B2 = attr_b4_morph6  : raw per-attr,  no whiten, adaptive
#   B3 = attr_b5_morph6  : whitened,      whiten,    adaptive
# Use THIS (not the @640 m1_morph6mean config) as B0 for the @1024 matrix -> no resolution
# confound. Decisive: B1>B0 whitening, B2>B0 adaptive, B3>B1 & B3>B2 they compose.

model = dict(
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                adaptive=False,        # fixed uniform (mean) attribute weights
                # no per_attr_whiten (B4 base has none) -> raw mean
            ),
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_mean_morph6_biomedclip_2gpu"
