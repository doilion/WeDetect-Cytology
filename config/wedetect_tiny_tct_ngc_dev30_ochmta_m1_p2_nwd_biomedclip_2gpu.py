_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_p2_biomedclip_2gpu.py"]

# stride-4 P2 head + NWD tiny-object assignment, @1024 BiomedCLIP (2026-06-08).
#
# Combines the two tiny-cell levers (both opt-in, both already in the codebase):
#   - P2 (from the base config): a stride-4 detection level so dense tiny cells get
#     their OWN prediction slots instead of colliding in the stride-8 grid (raises the
#     recall ceiling).
#   - NWD (added here): the TaskAligned assigner scores priors with the scale-stable
#     Normalized Wasserstein Distance instead of IoU, so the NEW stride-4 slots (and the
#     others) are not starved of positives by IoU collapse on ~22-36px cells.
# Hypothesis: finer slots + scale-stable assignment of those slots > P2 or NWD alone.
# Everything else (1024 cache, frozen BiomedCLIP text, organ mask, seed=42, batch=4)
# is inherited from the P2 config. A/B partner: the P2-only and NWD-only rows.

model = dict(
    train_cfg=dict(
        assigner=dict(
            use_nwd=True,      # scale-stable Gaussian assignment for tiny cells
            use_ciou=False,
            # nwd_C = mean sqrt(w*h) of dev30 boxes AT THIS RESOLUTION. @640 it is ~30;
            # this config trains @1024 (boxes 1.6x larger) -> ~48. (audit 2026-06-08:
            # @640's 30 was a resolution mismatch.) Sweep 36/48/60 in the go/no-go.
            nwd_C=48.0,
        ),
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_p2_nwd_biomedclip_2gpu"
