_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_res1024_biomedclip_2gpu.py"]

# OPT-IN stride-4 (P2) detection head @ 1024 (2026-06-07).
#
# Adds a 4th, finer detection level (stride 4) on top of the M1 res1024
# baseline (ConvNeXt-tiny + frozen BiomedCLIP text + organ mask + seed=42).
# Motivation: at 1024, native median cell ~36px and ~45% of cells are <32px;
# the stride-8 grid (128x128 cells @ 1024) packs dense pap-cytology cells into
# the same prediction slot. The stride-4 level (256x256 cells @ 1024) gives
# tiny cells their own slots instead of colliding.
#
# This config ONLY flips opt-in flags — the base config / shared neck+head
# code default to the 3-level path, so the running res1024 baseline is
# unaffected.
#
# Changes vs the res1024 baseline:
#   - neck.out_p2 = True (+ p2_channels=48): emits pan_out3 (stride4, 48ch)
#     from a fuse of pan_out2 (stride8) upsampled + raw x3 (stride4 backbone).
#   - head 4-level: in_channels=[48,96,192,384], featmap_strides=[4,8,16,32],
#     prior_generator.strides=[4,8,16,32]. The head's _init_layers respects a
#     4-level config (num_levels==4) instead of the hardcoded 3-level tiny
#     widths. ContrastiveHead/cls_preds/reg_preds are built for 4 levels.
#   - train_batch lowered 8 -> 4: the stride-4 cls map at 1024 is
#     [B, embed_dims, 256, 256] which is the memory hog. Launch with --amp.
#
# Keep seed=42 (inherited) so this is a clean A/B vs the res1024 baseline:
# any delta is attributable to the P2 head, not seed noise.

# Smaller batch to fit the stride-4 head at 1024 in 24GB. The 256x256 cls/reg
# maps (B,48 -> B,256/B,512 -> B,30) dominate activation memory.
train_batch = 4

model = dict(
    neck=dict(
        out_p2=True,
        p2_channels=48,
    ),
    bbox_head=dict(
        head_module=dict(
            # 4-level (P2) widths, finest first: [stride4, stride8, stride16, stride32].
            # stride4=p2_channels(48), stride8/16/32 = tiny FPN widths (96/192/384).
            in_channels=[48, 96, 192, 384],
            featmap_strides=[4, 8, 16, 32],
        ),
        # MlvlPointGenerator must produce priors for the same 4 strides so the
        # flattened prior count matches the 4-level cls/reg maps in loss_by_feat
        # (otherwise the masked_select reshape assert would fire).
        prior_generator=dict(
            type="MlvlPointGenerator", offset=0.5, strides=[4, 8, 16, 32]),
    ),
)

# Re-state the batch override on both loaders (mmengine merges nested dicts but
# the base sets batch_size explicitly, so override it explicitly here too).
train_dataloader = dict(batch_size=train_batch)
val_dataloader = dict(batch_size=train_batch)
test_dataloader = val_dataloader

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_p2_biomedclip_2gpu"
