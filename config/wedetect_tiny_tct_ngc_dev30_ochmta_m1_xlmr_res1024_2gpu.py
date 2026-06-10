_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_xlmr_2gpu.py"]

import os as _os


def _resolve_data_root(env_name, rel_path):
    return _os.environ.get(env_name, rel_path).rstrip("/") + "/"

# XLM-R base @ 1024 (2026-06-07). The REAL WeDetect foundation: XLM-Roberta 768-d text
# (cached PseudoLanguageBackbone, the original WeDetect text family) -- NOT the
# experimental BiomedCLIP swap. PLAIN model on NATIVE TCT_NGC images at 1024 (no
# stride-4 / NWD / distillation). This is the canonical base our small-cell method is
# built on + compared against. Cells: native median 137px -> 36px at 1024 (vs 22px@640).
# Fixed seed=42 (clean single-run A/B). Launch with --amp.

data_root = _resolve_data_root("TCT_NGC_DATA_ROOT", "data/TCT_NGC")  # NATIVE, not 640 cache
train_ann_file = "annotations/instances_train_dev_disjoint_dev30.json"
val_ann_file = "annotations/instances_val_dev_disjoint_dev30.json"
img_scale = (1024, 1024)
train_batch = 8
organ_mask_path = "data/texts/tct_ngc_class_organ_mask_base30.pt"
taxonomy_path = "data/texts/tct_ngc_taxonomy.json"

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PhotoMetricDistortion", brightness_delta=32,
         contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2), hue_delta=10),
    dict(type="WeDetectKeepRatioResize", scale=img_scale),
    dict(type="WeDetectLetterResize", scale=img_scale, allow_scale_up=True, pad_val=dict(img=114)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="LoadText"),
    dict(type="OrganExtractor", taxonomy_path=taxonomy_path, strict=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
         "scale_factor", "pad_param", "texts", "organ_id", "organ_name")),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="WeDetectKeepRatioResize", scale=img_scale),
    dict(type="WeDetectLetterResize", scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
    dict(type="LoadText"),
    dict(type="OrganExtractor", taxonomy_path=taxonomy_path, strict=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
         "scale_factor", "pad_param", "texts", "organ_id", "organ_name")),
]

train_dataloader = dict(
    batch_size=train_batch,
    dataset=dict(dataset=dict(data_root=data_root, ann_file=train_ann_file), pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=train_batch,
    dataset=dict(dataset=dict(data_root=data_root, ann_file=val_ann_file), pipeline=test_pipeline),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator

randomness = dict(seed=42, deterministic=False)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_xlmr_res1024_2gpu"
