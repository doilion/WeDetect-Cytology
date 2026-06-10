_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py"]

# Design A — Image-Conditional Fusion (ICF).
#
# Background: all "trainable text-side adapter" experiments (THAF / PCW / M2)
# are Pareto-dominated by M1-5attr static mean pool on novel zero-shot (see
# docs/tct_ngc_ochmta_corrected_protocol_problems_20260514.md and the
# bypass-experiment Table C). Common failure mode: cls loss does not push
# fusion output to differ from mean pool, optimizer finds α=0 / identity /
# mean-pool basin.
#
# ICF breaks this basin by making fusion image-conditional: the cross-
# attention query is derived from the deepest backbone feature (ConvNext
# stage 4, 768d, stride 32) instead of a class-agnostic learnable vector.
# Fusion output for the same class differs across images, so the "mean pool"
# function (image-invariant) is no longer in the function family ICF can
# represent — optimizer must actually learn image-conditional attribute
# selection to reduce loss.
#
# Inherits all M1 components unchanged:
#   - YOLOWorldHead.organ_loss_mask + organ_class_mask (clinical organ-
#     conditional class restriction)
#   - OrganExtractor pipeline + organ_id meta_keys
#   - OrganRestrictedCocoMetric val/test evaluator
#   - CheckpointHook.save_best='coco/overall/macro_mAP'
#   - 12 ep, lr 3e-4 begin=2 cosine, batch 8x2 GPUs
#   - load_from = checkpoints/wedetect_tiny.pth (same start as M1 / M2)
#
# Swaps:
#   - text_model: PseudoLanguageBackbone (1-PSC, [B, C, D])
#     -> PseudoMultiAttrLanguageBackbone with adapter=None + pool_mode='none'
#        (returns raw 5-attribute [B, C, A, D])
#   - cross_modal_fusion: NEW ImageConditionalFusion module on the backbone
#
# Trainable parameters added: ~2.4M
#   image_proj 393K + 5x attr_experts 655K + cross_attn 1.0M + output_proj 262K
# (Compare to THAF 7M / M2 HTA ~1M; budget intentionally small.)
#
# Diagnostic protection: ICFCollapseGuard hook monitors
#   - fused_pairwise_cos_mean  (same class, different images — must vary)
#   - attn_entropy_mean        (attention not uniform = real selection)
#   - cos_to_attr_mean_mean    (fused not collapsing to mean-pool direction)

# Default points to random-attr cache (shape [30, 5, 512]) so this parent
# is standalone-runnable as an "ICF + random text" baseline. Active children
# (icf_1psc_biomedclip / icf_randomattr_biomedclip / icf_morph6_biomedclip)
# override this path with their own class_metadata file.
class_metadata_path = "data/texts/tct_ngc_class_metadata_base30_randomattr.pt"
text_channels = 512

model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoMultiAttrLanguageBackbone",
            class_metadata_path=class_metadata_path,
            adapter=None,
            pool_mode="none",
        ),
        cross_modal_fusion=dict(
            type="ImageConditionalFusion",
            text_dim=text_channels,
            image_dim=768,         # ConvNext-tiny stage 4 dim
            num_attrs=5,
            attr_hidden=128,
            num_heads=8,
            dropout=0.0,
        ),
    ),
)

# ICFCollapseGuard: log every 500 iter + before each val. Warn-only;
# manual halt if diagnostics stay red after warmup.
custom_hooks = [
    dict(
        type="ICFCollapseGuard",
        check_interval=500,
        check_at_val=True,
        halt_on_red=False,
        pairwise_cos_max=0.99,
        attn_entropy_max=1.58,
        cos_to_mean_max=0.97,
    ),
]

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_icf_biomedclip_2gpu"
