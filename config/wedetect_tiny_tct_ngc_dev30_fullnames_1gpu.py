_base_ = ["./wedetect_tiny_tct_ngc_dev30_2gpu.py"]

# TCT_NGC dev30 split: Urine NILM/Negative/Negative Degen merged → Urine-NHGUC.
# Uses 30 full-name prompts (NHGUC re-described per Paris System) and cached
# text embeddings so the text encoder is not trained or run online.
train_class_text_path = "data/texts/tct_ngc_fullnames_30.json"
test_class_text_path = train_class_text_path
text_embed_path = "data/texts/tct_ngc_fullnames_30_embeddings_wedetect_tiny.pth"

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
train_batch_size_per_gpu = 16
base_lr = 1.5e-4
max_epochs = 12
warmup_iters = 1500

model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    backbone=dict(
        text_model=dict(
            _delete_=True,
            type="PseudoLanguageBackbone",
            text_embed_path=text_embed_path,
        ),
    ),
    bbox_head=dict(
        head_module=dict(num_classes=num_training_classes),
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_training_classes),
    ),
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=8,
    dataset=dict(
        class_text_path=train_class_text_path,
        dataset=dict(metainfo=dataset_metainfo),
    ),
)
val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    dataset=dict(
        class_text_path=test_class_text_path,
        dataset=dict(metainfo=dataset_metainfo),
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

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_fullnames_1gpu"
