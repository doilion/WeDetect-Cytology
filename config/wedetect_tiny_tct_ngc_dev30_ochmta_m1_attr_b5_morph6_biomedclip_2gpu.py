_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b4_morph6_biomedclip_2gpu.py"]

# B5 (2026-06-08): module 2 + MODULE 1 (per-attribute soft de-cone), morph6, @1024.
# Adds per_attr_whiten on top of B4 -> tests whether de-coning each morphology attribute
# axis (soft/residual, lam=0.5, audited to preserve the medical class hierarchy) helps on
# top of the instance-adaptive attribute weighting. B5 vs B4 isolates module 1's value.
# Sweep lam {0.25,0.5,0.75,1.0} via per_attr_whiten.lam.

model = dict(
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                # attr_text_path / attr_dropout inherited from B4
                per_attr_whiten=dict(topk=1, alpha=1.0, beta=1.0, lam=0.5),
            ),
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b5_morph6_biomedclip_2gpu"
