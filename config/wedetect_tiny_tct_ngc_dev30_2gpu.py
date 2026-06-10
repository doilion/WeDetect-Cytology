_base_ = ["default_runtime.py"]

import os as _os


def _resolve_data_root(env_name, rel_path):
    return _os.environ.get(env_name, rel_path).rstrip("/") + "/"

# Bump dist watchdog timeout from default 30 min to 3 h (10800 s). Ep1 val
# COCOeval over 30 classes × ~13K val images takes ~35 min on rank 0; rank 1
# was timing out in post-eval BROADCAST under the default. mmengine 0.10.7
# expects int seconds here, not timedelta. The previous top-level
# `dist_cfg = ...` outside env_cfg was dead — mmengine only reads env_cfg.
env_cfg = dict(
    dist_cfg=dict(backend="nccl", timeout=10800),
)

# TCT_NGC dev split: train on train_dev and tune only on val_dev.
data_root = _resolve_data_root("TCT_NGC_DATA_ROOT", "data/TCT_NGC")
train_ann_file = "annotations/instances_train_dev.json"
val_ann_file = "annotations/instances_val_dev.json"
train_class_text_path = "data/texts/tct_ngc_fullnames_30.json"
test_class_text_path = train_class_text_path

# dev30: Urine NILM (16) / Negative (17) / Negative Degeneration (20) merged
# into a single Urine-NHGUC (Paris System NHGUC). 32 → 30 classes; cat_ids
# now contiguous [0..29]. See tools/remap_dev32_to_dev30.py for the mapping.
base_classes = (
    "respiratory tract-Neutrophil",
    "respiratory tract-Alveolar macrophages",
    "respiratory tract-Ciliated columnar epithelial cells",
    "respiratory tract-Lymphocyte",
    "respiratory tract-Impurity",
    "respiratory tract-Squamous epithelial cells",
    "respiratory tract-Diseased cells",
    "Serous effusion-Negative samples",
    "Serous effusion-Diseased cells",
    "Thyroid gland-PTC",
    "Thyroid gland-SPTC",
    "Thyroid gland-NS",
    "Thyroid gland-Macrophages",
    "Thyroid gland-AUC",
    "Thyroid gland-Negative samples",
    "Thyroid gland-FC",
    "Urine-NHGUC",
    "Urine-SHGUC",
    "Urine-AUC",
    "Urine-HGUC",
    "TCT_CCD-normal",
    "TCT_CCD-ascus",
    "TCT_CCD-asch",
    "TCT_CCD-lsil",
    "TCT_CCD-hsil_scc_omn",
    "TCT_CCD-agc_adenocarcinoma_em",
    "TCT_CCD-vaginalis",
    "TCT_CCD-monilia",
    "TCT_CCD-dysbacteriosis_herpes_act",
    "TCT_CCD-ec",
)

all_classes = base_classes
dataset_metainfo = dict(classes=base_classes)

num_classes = 30
num_training_classes = 30
max_epochs = 12
close_mosaic_epochs = 2
save_epoch_intervals = 1
text_channels = 768
neck_embed_channels = [128, 256, 512]
neck_num_heads = [4, 8, 16]
base_lr = 2.25e-4
weight_decay = 0.05
train_batch_size_per_gpu = 12

find_unused_parameters = True

model_test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    score_thr=0.001,
    nms=dict(type="nms", iou_threshold=0.7),
    max_per_img=300,
)

tal_topk = 10
tal_alpha = 0.5
tal_beta = 2.0

loss_cls_weight = 1.0
loss_bbox_weight = 5.0
loss_dfl_weight = 1.5 / 4

custom_imports = dict(imports=["wedetect"], allow_failed_imports=False)

model = dict(
    type="YOLOWorldDetector",
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type="YOLOWDetDataPreprocessor",
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="MultiModalYOLOBackbone",
        image_model=dict(
            type="ConvNextVisionBackbone",
            model_name="tiny",
            frozen_modules=[],
        ),
        text_model=dict(
            type="XLMRobertaLanguageBackbone",
            model_name="./xlm-roberta-base/",
            model_size="tiny",
            frozen_modules=("all",),
        ),
    ),
    neck=dict(
        type="CSPRepBiFPANNeck",
        model_size="tiny",
    ),
    bbox_head=dict(
        type="YOLOWorldHead",
        head_module=dict(
            type="YOLOWorldHeadModule",
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes,
            model_size="tiny",
            in_channels=[96, 192, 384],
        ),
        prior_generator=dict(
            type="MlvlPointGenerator",
            offset=0.5,
            strides=[8, 16, 32],
        ),
        bbox_coder=dict(type="WeDetectDistancePointBBoxCoder"),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            reduction="none",
            loss_weight=loss_cls_weight,
        ),
        loss_bbox=dict(
            type="mmyoloIoULoss",
            iou_mode="ciou",
            bbox_format="xyxy",
            reduction="sum",
            loss_weight=loss_bbox_weight,
            return_iou=False,
        ),
        loss_dfl=dict(
            type="DistributionFocalLoss",
            reduction="mean",
            loss_weight=loss_dfl_weight,
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type="BatchTaskAlignedAssigner",
            num_classes=num_training_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9,
        )
    ),
    test_cfg=model_test_cfg,
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
        ),
    ),
]

train_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="WeCocoDataset",
        data_root=data_root,
        test_mode=False,
        ann_file=train_ann_file,
        data_prefix=dict(img="images/"),
        filter_cfg=None,
        metainfo=dataset_metainfo,
    ),
    class_text_path=train_class_text_path,
    pipeline=train_pipeline,
)

val_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="WeCocoDataset",
        data_root=data_root,
        test_mode=True,
        ann_file=val_ann_file,
        data_prefix=dict(img="images/"),
        batch_shapes_cfg=None,
        metainfo=dataset_metainfo,
    ),
    class_text_path=test_class_text_path,
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="yolow_collate"),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + val_ann_file,
    metric="bbox",
)

test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=10.0),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.01,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
    ),
]

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=save_epoch_intervals,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev_2gpu"
load_from = "checkpoints/wedetect_tiny.pth"
resume = False
