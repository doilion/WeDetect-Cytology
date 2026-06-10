#!/usr/bin/env python
"""Orthogonal Procrustes alignment of visual prototypes to text embedding space.

The geometric mismatch documented in
`docs/tct_ngc_novel_zero_shot_review_20260509.md` showed that mixing text
vectors (XLM-Roberta output) and visual prototypes (image encoder cls_preds
output) inside one inference causes the visual side to systematically out-cosine
the text side, crushing text-routed classes (Breast 0.454 → 0.000).

Solution: find an orthogonal rotation R such that R · vis_proto_base lies
in the same direction as text_emb_base for the 30 base classes (where we
have BOTH a text embedding AND a visual prototype). Apply the same R to
novel-class visual prototypes — they end up "in the text geometry" and can
be safely mixed with text class vectors in one inference.

Math (orthogonal Procrustes / Wahba's problem):
    Let A = text_emb_base   ∈ ℝ^{N×D}   (rows = base class vectors)
    Let B = vis_proto_base  ∈ ℝ^{N×D}
    Solve  R = argmin ‖B Rᵀ − A‖_F  s.t. R Rᵀ = I
    Closed form: U Σ Vᵀ = svd(Aᵀ B)  ⇒  R = U Vᵀ
    Then apply: vis_calibrated = vis_proto · Rᵀ
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--text-emb",
        default="data/texts/tct_ngc_fullnames_30_embeddings_wedetect_tiny.pth",
        help="base 30-class text embedding cache (XLM-Roberta output)",
    )
    p.add_argument(
        "--vis-proto-base",
        default="data/texts/tct_ngc_base30_visproto_train.pth",
        help="base 30-class visual prototype cache (from training set)",
    )
    p.add_argument(
        "--vis-proto-novel",
        nargs="+",
        required=True,
        help="one or more novel split visproto .pth files to calibrate",
    )
    p.add_argument(
        "--out-dir",
        default="data/texts/",
        help="where to write calibrated novel visproto .pth files",
    )
    p.add_argument(
        "--out-r",
        default="data/texts/procrustes_R.pth",
        help="save the rotation matrix R for reproducibility",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    text_d: dict[str, torch.Tensor] = torch.load(args.text_emb, map_location="cpu")
    vis_base_d: dict[str, torch.Tensor] = torch.load(args.vis_proto_base, map_location="cpu")

    # Match by exact key — both should use the prompt string as key
    common_keys = sorted(set(text_d) & set(vis_base_d))
    if len(common_keys) < 20:
        text_only = sorted(set(text_d) - set(vis_base_d))
        vis_only = sorted(set(vis_base_d) - set(text_d))
        raise SystemExit(
            f"too few common keys ({len(common_keys)}). "
            f"text-only first 3: {text_only[:3]}; vis-only first 3: {vis_only[:3]}"
        )
    print(f"[procrustes] {len(common_keys)} base classes paired (text ↔ vis_proto)")

    # Stack into [N, D] matrices
    A = torch.stack([text_d[k].float() for k in common_keys], dim=0).numpy()    # text
    B = torch.stack([vis_base_d[k].float() for k in common_keys], dim=0).numpy() # visual
    N, D = A.shape
    assert B.shape == (N, D), f"shape mismatch text {A.shape} vs vis {B.shape}"

    # Optional: pre-normalize each row (Procrustes typically run on unit-norm rows)
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-9)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-9)

    # Solve R: min ‖B Rᵀ − A‖
    # M = Aᵀ B  has SVD  U Σ Vᵀ ; optimal R = U Vᵀ
    M = A_norm.T @ B_norm
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt  # shape (D, D), orthogonal

    # Diagnostics: how good is the alignment on base classes?
    B_aligned = B_norm @ R.T  # apply rotation
    base_cos = (B_aligned * A_norm).sum(axis=1)  # per-class cosine
    print(
        f"[procrustes] base-class self-cosine after alignment: "
        f"mean={base_cos.mean():.3f}  min={base_cos.min():.3f}  max={base_cos.max():.3f}"
    )
    pre_cos = (B_norm * A_norm).sum(axis=1)
    print(
        f"[procrustes] base-class self-cosine BEFORE alignment: "
        f"mean={pre_cos.mean():.3f}  min={pre_cos.min():.3f}  max={pre_cos.max():.3f}"
    )

    R_t = torch.from_numpy(R).float()
    Path(args.out_r).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"R": R_t, "common_keys": common_keys}, args.out_r)
    print(f"[procrustes] wrote rotation matrix to {args.out_r}")

    # Apply R to each novel split's visproto
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for novel_path in args.vis_proto_novel:
        novel_d: dict[str, torch.Tensor] = torch.load(novel_path, map_location="cpu")
        calibrated = {}
        for k, v in novel_d.items():
            # NOTE: we do NOT pre-L2-normalize the novel vector here, since
            # the contrastive head L2-normalizes class vectors before cosine.
            # We just rotate.
            v_np = v.float().numpy()
            v_rot = v_np @ R.T
            calibrated[k] = torch.from_numpy(v_rot).float().contiguous()
        in_name = Path(novel_path).stem
        out_name = (
            in_name.replace("_visproto_emb", "_visproto_calibrated_emb")
            if "_visproto_emb" in in_name
            else f"{in_name}_calibrated"
        )
        out_path = out_dir / f"{out_name}.pth"
        torch.save(calibrated, out_path)
        print(f"[procrustes] {len(calibrated)} novel classes calibrated → {out_path}")


if __name__ == "__main__":
    main()
