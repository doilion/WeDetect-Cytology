_base_ = ["./wedetect_tiny_tct_ngc_dev30_icf_biomedclip_2gpu.py"]

# Row 7 - clean retrain with morph6 attribute redesign.
#
# This is the FIRST experiment that exercises non-degenerate ICF cross-attention
# (num_attrs=6 instead of 1) on clean-data + morphology-only text prompts.
#
# Key swap vs parent (icf_biomedclip 5-attr old):
#   - class_metadata: 5-attr label-mixed (specimen/PSC/IHC/diff-dx/morphology)
#       -> 6-attr pure morphology (cell_size_shape / nuclear_size_shape /
#          nuclear_membrane_contour / chromatin_nucleoli / nc_ratio_cytoplasm /
#          architecture_arrangement)
#   - cross_modal_fusion.num_attrs: 5 -> 6
#   - ICFCollapseGuard.attn_entropy_max: 1.58 (≈log(5)) -> 1.76 (≈log(6) × 0.98)
#     6-attribute max entropy is log(6) = 1.7918; keep the same 0.98 ratio.
#
# Inherits unchanged:
#   - M1 organ-class mask (clinical restriction)
#   - OrganExtractor pipeline + OrganRestrictedCocoMetric
#   - 12 ep, lr 3e-4 begin=2 cosine, batch 8x2 GPUs
#   - BiomedCLIP encoder (text_channels=512, image_dim=768)
#   - load_from = checkpoints/wedetect_tiny.pth

class_metadata_path = "data/texts/tct_ngc_class_metadata_base30_morph6.pt"

model = dict(
    backbone=dict(
        text_model=dict(
            class_metadata_path=class_metadata_path,
        ),
        cross_modal_fusion=dict(
            num_attrs=6,
        ),
    ),
)

custom_hooks = [
    dict(
        type="ICFCollapseGuard",
        check_interval=500,
        check_at_val=True,
        halt_on_red=False,
        pairwise_cos_max=0.99,
        attn_entropy_max=1.76,   # log(6) ≈ 1.79; keep ~0.98 of theoretical max
        cos_to_mean_max=0.97,
    ),
]

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_icf_morph6_biomedclip_2gpu_clean"
