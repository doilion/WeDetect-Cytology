#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from wedetect.models.backbones.mm_backbone import XLMRobertaLanguageBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cached text embeddings for PseudoLanguageBackbone. "
        "Supports CLIP-style prompt ensembling (CuPL, Pratt et al. 2023) when "
        "the input JSON has multiple variants per class — each variant is "
        "encoded separately and averaged into a single per-class embedding "
        "stored under the first variant's text key. Optional anisotropy "
        "reduction (Mu et al. 2017, 'all-but-the-top') subtracts the global "
        "class-mean before saving."
    )
    parser.add_argument("--texts", required=True, help="Class text JSON file (list-of-lists; inner list = variants for one class).")
    parser.add_argument("--out", required=True, help="Output .pth file.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/wedetect_tiny.pth",
        help="WeDetect checkpoint containing backbone.text_model weights.",
    )
    parser.add_argument("--model-name", default="./xlm-roberta-base/")
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--per-variant-l2-norm",
        action="store_true",
        help="L2-normalize each variant embedding before within-class averaging "
        "(standard CLIP zero-shot recipe; reduces dominance of high-norm "
        "variants). Only meaningful when inner lists have >1 variant.",
    )
    parser.add_argument(
        "--anisotropy-reduce",
        action="store_true",
        help="After per-class averaging, subtract the mean class embedding from "
        "every class and L2-normalize. Targets the well-known anisotropy of "
        "transformer text-embedding spaces where a shared component dominates "
        "cosine similarity (Mu et al. 2017; Ethayarajh 2019).",
    )
    return parser.parse_args()


def load_text_model(args: argparse.Namespace) -> XLMRobertaLanguageBackbone:
    model = XLMRobertaLanguageBackbone(
        model_name=args.model_name,
        model_size=args.model_size,
        frozen_modules=("all",),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "backbone.text_model."
    text_state = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not text_state:
        raise RuntimeError(f"No {prefix} weights found in {args.checkpoint}")

    incompatible = model.load_state_dict(text_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected text-model load result: {incompatible}")

    model.to(args.device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    text_groups = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    texts = list(itertools.chain.from_iterable(text_groups))
    slash_texts = [text for text in texts if "/" in text]
    if slash_texts:
        raise ValueError(
            "Prompt text cannot contain slash because PseudoLanguageBackbone "
            f"uses split lookup. Examples: {slash_texts[:3]}"
        )
    if len(texts) != len(set(texts)):
        repeated = sorted({text for text in texts if texts.count(text) > 1})
        raise ValueError(f"Duplicate prompt strings would collide: {repeated}")

    model = load_text_model(args)
    with torch.no_grad():
        embeddings = model([texts]).squeeze(0).detach().cpu().float()

    has_ensemble = any(len(grp) > 1 for grp in text_groups)
    if not has_ensemble:
        # Single-variant-per-class: keep the legacy {text: embed} mapping.
        embed_dict = {
            text: embeddings[idx].contiguous()
            for idx, text in enumerate(texts)
        }
        out_per_class = embeddings
    else:
        # CuPL / CLIP-style ensembling: per-class average over variants.
        # Optionally L2-normalize each variant before averaging (standard
        # CLIP zero-shot recipe).
        cursor = 0
        per_class_vecs = []
        primary_keys = []
        for group_idx, group in enumerate(text_groups):
            grp_embs = embeddings[cursor : cursor + len(group)]
            cursor += len(group)
            if args.per_variant_l2_norm:
                grp_embs = torch.nn.functional.normalize(grp_embs, p=2, dim=-1)
            class_vec = grp_embs.mean(dim=0)
            per_class_vecs.append(class_vec)
            primary_keys.append(group[0])
        out_per_class = torch.stack(per_class_vecs, dim=0)

        if args.anisotropy_reduce:
            global_mean = out_per_class.mean(dim=0, keepdim=True)
            out_per_class = out_per_class - global_mean
            out_per_class = torch.nn.functional.normalize(out_per_class, p=2, dim=-1)

        embed_dict = {
            primary_keys[i]: out_per_class[i].contiguous()
            for i in range(len(primary_keys))
        }
        n_variants = sum(len(grp) for grp in text_groups)
        n_classes = len(text_groups)
        print(
            f"[ensemble] averaged {n_variants} variants across {n_classes} classes "
            f"(per-variant-l2-norm={args.per_variant_l2_norm}, "
            f"anisotropy-reduce={args.anisotropy_reduce})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embed_dict, out_path)
    print(f"Wrote {out_path} with {len(embed_dict)} embeddings")


if __name__ == "__main__":
    main()
