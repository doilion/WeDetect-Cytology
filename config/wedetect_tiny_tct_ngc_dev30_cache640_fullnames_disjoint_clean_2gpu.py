_base_ = ["./wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu.py"]

# Clean retrain of dev30 — same taxonomy / data / batch sizes as the original
# `wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu` but with the
# two known training-time bugs patched, to get an ablation-grade baseline:
#   1. NCCL timeout (already in env_cfg upstream); training must NOT bridge
#      a 2GPU→1GPU switch, so any GPU fault aborts the run instead of
#      letting `--resume auto` continue on a different effective batch size.
#   2. LR scheduler overlap: the original LinearLR (warmup, by_iter end=1500)
#      ran into CosineAnnealingLR(begin=1, by_epoch=True, T_max=12), causing
#      the warmup to be multiplied by an already-active cosine. Memory note
#      `feedback_mmengine_lr_schedule_overlap.md` documents the val_loss
#      ep3 spike on dev32 from the same overlap. Fix: cosine begin=2 with
#      T_max=max_epochs-1 so cosine spans ep2..ep12 cleanly.

max_epochs = 12
warmup_iters = 1500
base_lr = 3.0e-4

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
        begin=2,
        end=max_epochs,
        T_max=max_epochs - 1,
        by_epoch=True,
    ),
]

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu"
