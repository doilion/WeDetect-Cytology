_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_res1024_biomedclip_2gpu.py"]

# Whitening-in-training (2026-06-08): the PARAMETER-FREE baseline of the de-collapse
# ladder (baseline -> whiten -> D-v2). A fixed geometric de-cone of the frozen text
# (remove global mean + top-1 cone direction, keep dim), plugged into the SAME text_decone
# hook as D-v2. No learnable params -> the image side learns to align in the de-coned space
# (this is whitening-IN-training; the inference-only probe is a weak / false-negative test).
# Purpose: isolate "how much of D-v2's gain is just de-coning" vs "the learnable image
# grounding". opt-in; everything else (1024 cache, frozen BiomedCLIP text, seed=42) inherited.

model = dict(
    text_decone=dict(
        type="WhiteningTextDecone",
        topk=1,        # remove the top-1 cone direction (ablate 1/2)
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_whiten_biomedclip_2gpu"
