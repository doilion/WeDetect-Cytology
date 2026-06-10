_base_ = ["./wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu.py"]

# Ablation control: BiomedCLIP text encoder + SINGLE prompt per class
# (NO THAF fusion module, NO 5-attribute hierarchical text).
#
# Pairs with `wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu` to isolate
# two confounded factors in the THAF BiomedCLIP base mAP gain (+1.7pp over
# clean dev30):
#   (a) text-encoder swap            XLM-R 768d → BiomedCLIP 512d
#   (b) text input format            1 prompt   → 5-attr fused
#
# Together with `wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu`
# (XLM-R + 1 prompt) and `wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu`
# (BiomedCLIP + 5-attr THAF, fusion bypassed by alpha→0), this 3rd point
# in the (encoder, text-format) grid decomposes the gain:
#   encoder-swap effect = (this) − clean dev30
#   5-attr text effect  = THAF BiomedCLIP − (this)
#   THAF fusion effect  ≈ 0  (already shown by alpha→0 in Phase 3.5 diag)
#
# Design choices:
#   1. Inherits the clean retrain base (LR begin=2 / T_max=11 fix), so
#      schedule is identical to the clean XLM-R baseline — encoder is the
#      only variable.
#   2. Reuses `data/texts/tct_ngc_fullnames_30.json` (single string per
#      class) → JSON keys match the cache keys produced by
#      `tools/build_biomedclip_text_embeddings.py`.
#   3. text_model: PseudoLanguageBackbone (NO fusion module, dim-agnostic).
#      bbox_head.head_module.embed_dims overridden 768 → 512 to match the
#      BiomedCLIP output dim. Same situation as THAF BiomedCLIP — the
#      cls_preds.* (768d) layer in checkpoints/wedetect_tiny.pth shape-
#      mismatches and stays randomly initialized for 512d output.
#   4. Train pipeline transforms inherited unchanged: `LoadText` (not
#      `HierarchicalLoadText`) — exactly the same path as clean dev30.

text_embed_path = "data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth"
text_channels = 512

model = dict(
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoLanguageBackbone",
            text_embed_path=text_embed_path,
        ),
    ),
    bbox_head=dict(
        head_module=dict(embed_dims=text_channels),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu"