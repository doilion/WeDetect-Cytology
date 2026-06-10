_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_xlmr_res1024_2gpu.py"]

import os as _os


def _resolve_data_root(env_name, rel_path):
    return _os.environ.get(env_name, rel_path).rstrip("/") + "/"

# OUR METHOD: opt-in stride-4 (P2) coarse-to-fine detection head, on the REAL
# WeDetect base (XLM-R 768-d text) @ 1024 native (2026-06-07).
#
# Same P2 flags as the BiomedCLIP P2 config -- the flags are IMAGE-side (the
# stride-4 sampling slots); the 768-d text dim comes from the xlmr_res1024 base.
# Headline A/B = this vs the XLM-R@1024 baseline (same seed=42, same batch).
#
#   - neck.out_p2=True (+p2_channels=48): emit pan_out3 (stride4, 48ch) =
#     pan_out2(stride8) upsampled + fused with raw x3 (stride4 backbone).
#   - head 4-level: in_channels=[48,96,192,384], featmap_strides=[4,8,16,32],
#     prior_generator.strides=[4,8,16,32]. Tiny cells get their own slots
#     instead of colliding in the stride-8 grid (the recall-ceiling raiser).
#   - batch lowered to 4 (the stride-4 cls map at 1024 is the memory hog).
#     NOTE: final batch will be set to the verified P2 max + MATCHED on the
#     XLM-R baseline for a fair, method-favorable A/B. Launch with --amp.
#   - (text/organ-gated sparse inference = a later inference-time add; this
#     trains the dense stride-4 head first.)

train_batch = 4

model = dict(
    neck=dict(out_p2=True, p2_channels=48),
    bbox_head=dict(
        head_module=dict(
            in_channels=[48, 96, 192, 384],
            featmap_strides=[4, 8, 16, 32],
        ),
        prior_generator=dict(
            type="MlvlPointGenerator", offset=0.5, strides=[4, 8, 16, 32]),
    ),
)

# Use the prebuilt 1024 cache (= native letterboxed to 1024, the SAME data) so this
# runs CONCURRENTLY with the native-loading base WITHOUT the IO contention that made
# two native-1024 runs 3x slower. KeepRatioResize/LetterResize @1024 are no-ops on the
# already-1024 cache images. Fair A/B vs the base (identical 1024 pixels).
_cache_root = _resolve_data_root("TCT_NGC_1024_ROOT", "data/TCT_NGC_1024")
train_dataloader = dict(
    batch_size=train_batch,
    dataset=dict(dataset=dict(data_root=_cache_root)))
val_dataloader = dict(
    batch_size=train_batch,
    dataset=dict(dataset=dict(data_root=_cache_root)))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=_cache_root + "annotations/instances_val_dev_disjoint_dev30.json")
test_evaluator = val_evaluator

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_xlmr_p2_res1024_2gpu"
