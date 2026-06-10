_base_ = ["./wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu.py"]

import os as _os


def _resolve_data_root(env_name, rel_path):
    return _os.environ.get(env_name, rel_path).rstrip("/") + "/"

# Encoder-agnostic organ-mask ablation: same as M1 BiomedCLIP (Row 1) but
# with the original XLM-Roberta text encoder (768d cache instead of 512d
# BiomedCLIP cache). Purpose: isolate "organ mask gain" from "BiomedCLIP
# encoder swap gain" by holding the encoder constant.
#
# Reference comparisons:
#   - XLM-R baseline (no mask):  work_dirs/.../disjoint_clean_2gpu/ best ep9
#       → existing paper-eval numbers in baseline_eval_summary.txt
#   - XLM-R + M1 (this config):  to be trained, ~5-7h DDP 2 GPU
#   - BiomedCLIP baseline (no mask):  noTHAF (0.327 / 0.108 novel)
#   - BiomedCLIP + M1 (Row 1):         0.337 / 0.150 novel (+4.2pp from mask)
#
# Expected: XLM-R + M1 novel ≥ XLM-R baseline novel + 3pp → confirms mask is
# encoder-agnostic (gain comes from training-loss structure, not encoder).
#
# Inherits everything from disjoint_clean (XLM-R 768d cache, 1-PSC text):
#   - text_model = PseudoLanguageBackbone (XLM-R 768d cache)
#   - 12 epochs, lr 3e-4, batch 16x2 GPU, cosine begin=2 T_max=11
#   - load_from = checkpoints/wedetect_tiny.pth
# Adds (identical to M1 BiomedCLIP recipe):
#   - bbox_head.organ_loss_mask=True + organ_mask_path
#   - OrganExtractor in train/test pipelines + organ_id in meta_keys
#   - OrganRestrictedCocoMetric val_evaluator (so val mAP matches paper protocol)

organ_mask_path = "data/texts/tct_ngc_class_organ_mask_base30.pt"
taxonomy_path = "data/texts/tct_ngc_taxonomy.json"

model = dict(
    bbox_head=dict(
        type="YOLOWorldHead",
        organ_loss_mask=True,
        organ_mask_path=organ_mask_path,
    ),
)

img_scale = (640, 640)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10,
    ),
    dict(type="WeDetectKeepRatioResize", scale=img_scale),
    dict(
        type="WeDetectLetterResize",
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114),
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="LoadText"),
    dict(type="OrganExtractor", taxonomy_path=taxonomy_path, strict=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id", "img_path", "ori_shape", "img_shape",
            "scale_factor", "pad_param", "texts",
            "organ_id", "organ_name",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="WeDetectKeepRatioResize", scale=img_scale),
    dict(
        type="WeDetectLetterResize",
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114),
    ),
    dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
    dict(type="LoadText"),
    dict(type="OrganExtractor", taxonomy_path=taxonomy_path, strict=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id", "img_path", "ori_shape", "img_shape",
            "scale_factor", "pad_param", "texts",
            "organ_id", "organ_name",
        ),
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# SINGLE SOURCE: data/texts/tct_ngc_dev30_negatives.json (loader:
# wedetect.utils.load_dev30_negative_classes). 6 negatives incl. Thyroid gland-NS
# (Bethesda I Nondiagnostic, added 2026-06-09). Read the JSON directly to keep
# config parsing decoupled from the heavy `wedetect` package import.
import json as _json
NEGATIVE_CLASS_NAMES = _json.load(
    open("data/texts/tct_ngc_dev30_negatives.json", encoding="utf-8")
)["negative_class_names"]
_data_root = _resolve_data_root("TCT_NGC_640_ROOT", "data/TCT_NGC_640")
_val_ann_file = "annotations/instances_val_dev_disjoint_dev30.json"
val_evaluator = dict(
    _delete_=True,
    type="OrganRestrictedCocoMetric",
    ann_file=_data_root + _val_ann_file,
    metric="bbox",
    classwise=False,
    organ_mask_path=organ_mask_path,
    exclude_class_names=NEGATIVE_CLASS_NAMES,
)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="coco/overall/macro_mAP",
        rule="greater",
        max_keep_ckpts=-1,
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_xlmr_2gpu"
