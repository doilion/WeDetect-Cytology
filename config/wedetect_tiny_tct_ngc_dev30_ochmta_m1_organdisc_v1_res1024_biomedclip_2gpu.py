_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_res1024_biomedclip_2gpu.py"]

# Innovation ① @1024: organ-conditioned hard-negative discriminative loss (zero new
# params, logit-space) on the BiomedCLIP @1024 baseline. Attacks within-organ class
# separability (rare-class class-macro), NOT base recall -> a class-macro/novel-side
# module, paired with a recall module (stride-4) for the headline. Same module block as
# the 640 organdisc_v1; only the resolution baseline differs. opt-in; baseline unchanged.
model = dict(
    bbox_head=dict(
        organ_disc_loss=dict(
            type="OrganHardNegContrastiveLoss",
            loss_weight=0.5,
            mode="ce",
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_organdisc_v1_res1024_biomedclip_2gpu"
