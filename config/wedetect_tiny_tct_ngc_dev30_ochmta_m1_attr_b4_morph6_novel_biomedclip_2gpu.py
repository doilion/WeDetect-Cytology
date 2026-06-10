_base_ = ["./wedetect_tiny_tct_ngc_dev30_ochmta_m1_attr_b4_morph6_biomedclip_2gpu.py"]

# B4 NOVEL-EVAL config: same model as B4 morph6 but the attribute head holds the 39-class
# (30 base + 9 novel) per-attribute text, so the head outputs 39 logits and novel eval does
# not crash. The learned params (attr_gate/gamma/beta/BN/logit_scale) are class-agnostic and
# attr_text is persistent=False, so the BASE-trained B4 checkpoint loads cleanly here.
# Use with: test_novel.py --config <this> --checkpoint <B4 base best> --data-root /home1/.../TCT_NGC

model = dict(
    bbox_head=dict(
        head_module=dict(
            attr_fusion=dict(
                attr_text_path="data/texts/tct_ngc_morph6_39_per_attr_biomedclip.pth",
            ),
        ),
    ),
)
