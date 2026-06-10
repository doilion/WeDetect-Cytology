_base_ = ["./wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py"]

import os as _os


def _resolve_data_root(env_name, rel_path):
    return _os.environ.get(env_name, rel_path).rstrip("/") + "/"

# OC-HMTA Module 1 only — Organ-Conditional class loss masking.
#
# Identical to noTHAF (BiomedCLIP + 1-PSC) in every other respect:
#   - text encoder frozen (PseudoLanguageBackbone, cached 512d BiomedCLIP emb)
#   - image encoder = ConvNext-tiny + FPN, trained
#   - load_from = checkpoints/wedetect_tiny.pth (same start)
#   - 30 class text broadcast, head outputs 30-channel cls_pred
#   - LR + epochs + batch sizes inherited (begin=2 cosine bug fix from clean dev30)
#
# Differences:
#   - YOLOWorldHead.organ_loss_mask = True
#   - YOLOWorldHead.organ_mask_path = .../tct_ngc_class_organ_mask_base30.pt
#     → per-sample cross-organ class BCE is zeroed in loss_by_feat,
#       so the model is never penalized for low score on cross-organ classes
#       (and never rewarded either). image encoder is freed from cross-organ
#       disambiguation (DEAD-7 partial fix).
#   - Both train_pipeline and test_pipeline insert `OrganExtractor` before
#     PackDetInputs, and PackDetInputs.meta_keys includes 'organ_id'/'organ_name'
#     so head sees organ via batch_img_metas[b]['organ_id'].
#
# This is row 3 of the ablation table (M1 only). Row 4 will add Module 2
# (diag-code text adapter), row 5 will add Module 3 (organ aux head if it helps).

organ_mask_path = "data/texts/tct_ngc_class_organ_mask_base30.pt"
taxonomy_path = "data/texts/tct_ngc_taxonomy.json"

model = dict(
    bbox_head=dict(
        type="YOLOWorldHead",
        organ_loss_mask=True,
        organ_mask_path=organ_mask_path,
    ),
)

# ── Pipeline overrides ──
# Inject OrganExtractor before PackDetInputs in train + test pipelines, and
# extend PackDetInputs.meta_keys to carry organ_id through to head.
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
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pad_param",
            "texts",
            "organ_id",
            "organ_name",
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
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pad_param",
            "texts",
            "organ_id",
            "organ_name",
        ),
    ),
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline)
)
test_dataloader = val_dataloader

# Align val evaluator with paper protocol (same metric as final test eval):
#   OrganRestrictedCocoMetric reports per-organ AP + overall macro + all-class
#   flat + instance-weighted. Excludes 5 negative classes from COCOeval
#   catIds for parity with test_exclude_negative.py.
# Inference already applies organ mask (head.organ_class_mask buffer); this
# only changes the AGGREGATION/EXCLUSION at metric time so that val mAP
# during training is directly comparable to paper-reported test numbers.
# SINGLE SOURCE: data/texts/tct_ngc_dev30_negatives.json (loader:
# wedetect.utils.load_dev30_negative_classes). 6 negatives incl. Thyroid gland-NS
# (Bethesda I Nondiagnostic = FNA artifact/acellular -> impurity, not a disease
# class; added 2026-06-09 -- previously omitted so the headline organ-macro counted
# it as a real Thyroid class). Read the JSON directly here to keep config parsing
# decoupled from the heavy `wedetect` package import. All attr/res1024 experiment
# configs inherit this val_evaluator, so editing the JSON propagates everywhere.
import json as _json
NEGATIVE_CLASS_NAMES = _json.load(
    open("data/texts/tct_ngc_dev30_negatives.json", encoding="utf-8")
)["negative_class_names"]
# Inherited paths from base config chain. Re-state explicitly because mmengine
# config does not resolve {{_base_.}} substitution in nested dict values.
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

# OrganRestrictedCocoMetric outputs `coco/overall/macro_mAP` rather than the
# inherited `coco/bbox_mAP`. The inherited CheckpointHook key would crash at
# the first val end (KeyError 'coco/bbox_mAP'). Override save_best to the
# corrected metric key (paper-protocol aligned). NOTE: M1 / M1-5attr平均 /
# M1+M2-完整方法 trained before this override existed — they used old
# val_evaluator (CocoMetric), so their inherited save_best='coco/bbox_mAP'
# was valid. This override only affects future trainings of M2 / Row 5
# (axisstruct) / Row 4d-fix variants.
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="coco/overall/macro_mAP",
        rule="greater",
        max_keep_ckpts=-1,
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu"
