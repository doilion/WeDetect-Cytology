_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_res1024_biomedclip_2gpu.py"]

# Innovation ② @1024: organ-modulated region-gap distillation from a frozen EXTERNAL
# BiomedCLIP image teacher, on the BiomedCLIP @1024 baseline. roi_align region features
# -> frozen-teacher region embeddings, 1-cos aligned, organ-modulated. Backbone-agnostic;
# loaded at detector __init__ (build_and_set_teacher). Same module block as the 640
# regiongap_v1; only the resolution baseline differs. opt-in; baseline unchanged.
custom_imports = dict(imports=["wedetect"], allow_failed_imports=False)

model = dict(
    region_gap_distill_loss=dict(
        type="OrganModulatedRegionGapDistill",
        student_dim=96,
        teacher_dim=512,
        roi_level=0,
        crop_size=224,
        max_rois=64,
        num_organs=5,
        loss_weight=1.0,
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_regiongap_v1_res1024_biomedclip_2gpu"
