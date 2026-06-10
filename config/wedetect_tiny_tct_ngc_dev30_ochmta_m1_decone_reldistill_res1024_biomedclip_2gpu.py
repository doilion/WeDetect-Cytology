_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_reldistill_res1024_biomedclip_2gpu.py"]

# ★ Module 2 (the proposed text module): De-coned Relational Distillation @1024.
#
# Adds a fixed geometric de-cone (WhiteningTextDecone, top-1 cone direction removed)
# on top of the relational-distillation adapter. Key wiring fact (yolo_world.py loss
# path): the detector applies `text_decone` AFTER the learned text_adapter, so BOTH the
# main cls loss AND the relational distillation loss see DE-CONED text. That removes the
# dominant anisotropy direction that made the bare relational loss (reldistill, no
# whitening) gauge-degenerate — the within-organ cosine matrix was dominated by the
# collapse cone, so matching cosines mostly matched the cone instead of the fine
# structure. De-coning first conditions the image->text relational target onto the
# discriminative within-organ geometry.
#
# Ablation role: baseline -> decone (training-free, statistics-only) -> reldistill
# (learned, no whitening, gauge problem) -> THIS (decone + reldistill = both). Everything
# else (adapter, loss, @1024, fp32) inherited.

model = dict(
    text_decone=dict(
        type="WhiteningTextDecone",
        topk=1,        # remove the top-1 cone direction (ablate 1/2 if needed)
    ),
)

work_dir = "./work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_decone_reldistill_res1024_biomedclip_2gpu"
