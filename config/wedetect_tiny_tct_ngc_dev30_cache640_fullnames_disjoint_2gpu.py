_base_ = ["./wedetect_tiny_tct_ngc_dev30_fullnames_1gpu.py"]

import os as _os


def _resolve_data_root(env_name, rel_path):
    return _os.environ.get(env_name, rel_path).rstrip("/") + "/"

# dev30 (Urine NHGUC merged) full-name baseline on the patient-disjoint
# 640x640 cached split. Two RTX 4090. cat_ids contiguous [0..29].
data_root = _resolve_data_root("TCT_NGC_640_ROOT", "data/TCT_NGC_640")
train_ann_file = "annotations/instances_train_dev_disjoint_dev30.json"
val_ann_file = "annotations/instances_val_dev_disjoint_dev30.json"

train_batch_size_per_gpu = 16
base_lr = 3.0e-4
max_epochs = 12
warmup_iters = 1500

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=8,
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            ann_file=train_ann_file,
        ),
    ),
)
val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            ann_file=val_ann_file,
        ),
    ),
)
test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(lr=base_lr))

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=warmup_iters,
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

val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator

# Keep all 12 epoch checkpoints so we can later compute val loss curve per epoch
# (旧 baseline max_keep_ckpts=3 导致 ep1-9 ckpt 被删，无法回算 val loss 曲线)
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=-1,
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu"
