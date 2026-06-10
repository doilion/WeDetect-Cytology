_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b4_morph6_novel_biomedclip_2gpu.py"]

# B3 NOVEL-EVAL config + transductive whitening control. = attr_b5 (whiten + adaptive) on
# the 39-class (30 base + 9 novel) per-attribute text. Load the BASE-trained attr_b5 (B3)
# checkpoint here (attr_text is persistent=False + the learned params are class-agnostic).
#
# By default the per-attribute de-cone is fit on the GIVEN prompts (all 39 = base+novel),
# which IS the open-vocabulary inference setting (you have the novel prompts at test time).
# For the transductive-defense control, re-run with BASE-ONLY whitening (fit the de-cone on
# the 30 base prompts only, applied to all 39):
#   PYTHONPATH=. python test_novel.py --config <this> --checkpoint <attr_b5 base best> \
#       --data-root "$TCT_NGC_DATA_ROOT" \
#       --cfg-options model.bbox_head.head_module.attr_fusion.per_attr_whiten.fit_classes=30
# If the base-only and base+novel-prompt novel numbers are close, the gain is not from
# peeking at the novel statistics (defuses the "is this transductive?" reviewer question).

model = dict(
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                per_attr_whiten=dict(topk=1, alpha=1.0, beta=1.0, lam=0.5),
                # fit_classes omitted -> None -> base+novel-prompt whitening (OVD inference).
                # Override to 30 (#base) for the base-only transductive control (see above).
            ),
        ),
    ),
)
