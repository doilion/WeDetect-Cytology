_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_res1024_biomedclip_2gpu.py"]

# B4 (2026-06-08): MODULE 2 ONLY — region-conditioned attribute reweighting, morph6 (6
# pure-morphology attributes), @1024 BiomedCLIP. The go/no-go for the whole attribute-
# adaptive method: does letting each cell instance weight the 6 morphology attributes
# beat the FIXED mean-attribute classifier (B0)?  At init the fusion gate is zero -> uniform
# weights -> IDENTICAL to a mean-attribute BNContrastive classifier (regression-safe), then
# it learns the instance-adaptive weighting. No per-attr whitening (module 1) and no D-v2
# here -> a clean ablation of module 2 alone (B5 adds module 1, B6 adds D-v2).
#
# NOTE: base-split only (attr_text holds the 30 base classes). Novel eval needs a 39-class
# per-attr file + a sibling config. Build attr text: tools encoded
# data/texts/tct_ngc_morph6_30_per_attr_biomedclip.pth = [30, 6, 512].

model = dict(
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                attr_text_path="data/texts/tct_ngc_morph6_30_per_attr_biomedclip.pth",
                attr_dropout=0.1,
                balance_weight=0.0,          # B4: off; ablate {0, 0.01} later
                # per_attr_whiten omitted -> module 2 only (module 1 enters at B5)
            ),
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b4_morph6_biomedclip_2gpu"
