#!/usr/bin/env python
"""Pre-cache per-attribute embeddings for the THAF training pipeline.

Why: in Phase 3 the text encoder is fully frozen and only the fusion module
trains. Re-encoding 5 × ~80 sampled-class strings on every batch wastes
compute. Pre-encode every attribute string once (39 classes × 5 attrs ≈ 195
strings) so the PseudoHierarchical*LanguageBackbone can index a tensor
table at training time.

Two encoder variants:
    --encoder xlmr        XLM-Roberta (768d), uses dev30 ckpt's text-branch weights
    --encoder biomedclip  BiomedCLIP-PubMedBERT (512d), loaded via open_clip from HF Hub

Output:
    data/texts/tct_ngc_attr_xlmr_per_attr.pth        dict[str, Tensor[768]]
    data/texts/tct_ngc_attr_biomedclip_per_attr.pth  dict[str, Tensor[512]]

Cache keys are the **exact** attribute strings used by
HierarchicalRandomLoadText output. The training pipeline must emit the same
strings; mismatch → KeyError at runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wedetect.models.backbones.mm_backbone import XLMRobertaLanguageBackbone


ATTR_FIELDS_ORDERED = (
    "organ_specimen",
    "diagnostic_code",
    "cytomorphology",
    "background_and_immunoprofile",
    "key_distinguishing_feature",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--attr-json",
        default="data/texts/tct_ngc_fullnames_39_attr.json",
    )
    p.add_argument(
        "--encoder",
        choices=("xlmr", "biomedclip"),
        default="xlmr",
        help="text encoder family for the cache",
    )
    # XLM-R only
    p.add_argument(
        "--checkpoint",
        default="checkpoints/wedetect_tiny.pth",
        help="(XLM-R only) WeDetect ckpt holding backbone.text_model.* weights",
    )
    p.add_argument("--model-name-xlmr", default="./xlm-roberta-base/")
    p.add_argument("--model-size", default="tiny")
    # BiomedCLIP only
    p.add_argument(
        "--biomedclip-name",
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )
    # Common
    p.add_argument(
        "--out",
        default=None,
        help="output .pth path; default depends on encoder choice",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--include-padding",
        action="store_true",
        help="also encode the empty string '' (used by HierarchicalRandomLoadText "
        "padding_to_max=True). Default False since we plan to disable padding.",
    )
    args = p.parse_args()
    if args.out is None:
        args.out = f"data/texts/tct_ngc_attr_{args.encoder}_per_attr.pth"
    return args


def load_text_model_xlmr(args: argparse.Namespace) -> XLMRobertaLanguageBackbone:
    model = XLMRobertaLanguageBackbone(
        model_name=args.model_name_xlmr,
        model_size=args.model_size,
        frozen_modules=("all",),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    prefix = "backbone.text_model."
    text_state = {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }
    if not text_state:
        raise RuntimeError(f"no {prefix}* keys in {args.checkpoint}")
    incompat = model.load_state_dict(text_state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"unexpected load result: {incompat}")
    model.to(args.device)
    model.eval()
    return model


def encode_xlmr(model: XLMRobertaLanguageBackbone, texts: list[str]) -> torch.Tensor:
    with torch.no_grad():
        # XLMRobertaLanguageBackbone.forward expects List[List[str]] = [batch][num]
        return model([texts]).squeeze(0).detach().cpu().float()


def encode_biomedclip(args: argparse.Namespace, texts: list[str]) -> torch.Tensor:
    import open_clip

    print(f"[encode] loading BiomedCLIP from {args.biomedclip_name}")
    model, _ = open_clip.create_model_from_pretrained(args.biomedclip_name)
    tokenizer = open_clip.get_tokenizer(args.biomedclip_name)
    model.to(args.device).eval()
    tokens = tokenizer(texts).to(args.device)
    with torch.no_grad():
        embs = model.encode_text(tokens)  # [N, 512]
        # Match the L2-normalized convention used downstream by the fusion module
        embs = embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return embs.detach().cpu().float()


def main() -> None:
    args = parse_args()
    attr = json.loads(Path(args.attr_json).read_text(encoding="utf-8"))

    # Collect every unique attr string (deduped, preserves insertion order)
    seen: dict[str, None] = {}
    for cls, fields in attr.items():
        for f in ATTR_FIELDS_ORDERED:
            v = fields.get(f, "").strip()
            if not v:
                raise SystemExit(f"empty field {cls!r}.{f!r}")
            seen[v] = None
    if args.include_padding:
        seen[""] = None
    texts = list(seen.keys())
    print(f"[encode] {len(texts)} unique attr strings to encode "
          f"(39 classes × 5 attrs - dedup; include_padding={args.include_padding}; "
          f"encoder={args.encoder})")

    if args.encoder == "xlmr":
        model = load_text_model_xlmr(args)
        embeddings = encode_xlmr(model, texts)
        expected_dim = 768
    elif args.encoder == "biomedclip":
        embeddings = encode_biomedclip(args, texts)
        expected_dim = 512
    else:
        raise ValueError(f"unknown encoder {args.encoder!r}")

    if embeddings.shape != (len(texts), expected_dim):
        raise RuntimeError(
            f"unexpected embedding shape {tuple(embeddings.shape)} "
            f"for {len(texts)} texts (expected ({len(texts)}, {expected_dim}))"
        )

    cache = {t: embeddings[i].contiguous() for i, t in enumerate(texts)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"[save] {out_path}  ({len(cache)} entries, dim={expected_dim})")


if __name__ == "__main__":
    main()
