#!/usr/bin/env python
"""Encode 5-attribute structured prompts via XLM-Roberta with 4 aggregation
strategies, for Phase 2.1 cos-heatmap diagnostic of hierarchical attribute
training (item 19 / component A).

Input JSON (data/texts/tct_ngc_fullnames_39_attr.json) schema:
    {
        "<class_name>": {
            "organ_specimen": "...",
            "diagnostic_code": "...",
            "cytomorphology": "...",
            "background_and_immunoprofile": "...",
            "key_distinguishing_feature": "..."
        },
        ...  # 30 base + 9 novel = 39 classes
    }

Output (per strategy):
    data/texts/tct_ngc_attr_<strategy>_emb.pth   dict[class_name -> Tensor]
    data/texts/tct_ngc_attr_base30.json          [["<class_name>"], ...]   (cos-heatmap input)
    data/texts/tct_ngc_attr_novel9.json          same, 9 novel class names

Strategies:
    concat            5 fields encoded separately, concatenated -> 3840-dim
    sum               5 fields encoded separately, summed -> 768-dim
    weighted-sum      α₁=organ 0.10  α₂=diag 0.15  α₃=morph 0.30
                      α₄=bg 0.05     α₅=distinguish 0.40    -> 768-dim
    only-distinguish  only key_distinguishing_feature -> 768-dim
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT = Path(os.environ.get("TCT_NGC_DATA_ROOT", REPO_ROOT / "data" / "TCT_NGC"))

from wedetect.models.backbones.mm_backbone import XLMRobertaLanguageBackbone


ATTR_FIELDS = (
    "organ_specimen",
    "diagnostic_code",
    "cytomorphology",
    "background_and_immunoprofile",
    "key_distinguishing_feature",
)
WEIGHTED_ALPHAS = (0.10, 0.15, 0.30, 0.05, 0.40)  # parallel to ATTR_FIELDS

STRATEGIES = ("concat", "sum", "weighted-sum", "only-distinguish")

# Canonical base 30 / novel 9 split (matches dev30 ann files)
BASE30_NAMES_FROM_ANN = "instances_test_base_clean_dev30.json"
NOVEL_NAMES_FROM_ATTR = (
    # 5 main novel (test_novel.json)
    "respiratory tract-adenocarcinoma",
    "Serous effusion-Ovarian cancer",
    "respiratory tract-Squamous cell carcinoma",
    "Serous effusion-Breast cancer",
    "Thyroid gland-MTC",
    # 4 extra novel (from hard_4 split)
    "respiratory tract-Small cell carcinoma",
    "Serous effusion-adenocarcinoma",
    "Thyroid gland-Suspicious for Malignancy",
    "Thyroid gland-Malignant tumour",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attr-json", default="data/texts/tct_ngc_fullnames_39_attr.json")
    p.add_argument("--checkpoint", default="checkpoints/wedetect_tiny.pth")
    p.add_argument("--model-name", default="./xlm-roberta-base/")
    p.add_argument("--model-size", default="tiny")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="data/texts/")
    p.add_argument(
        "--base-ann",
        default=str(DATA_ROOT / "annotations/instances_test_base_clean_dev30.json"),
        help="used only to derive the 30 base class names",
    )
    return p.parse_args()


def load_text_model(args: argparse.Namespace) -> XLMRobertaLanguageBackbone:
    model = XLMRobertaLanguageBackbone(
        model_name=args.model_name,
        model_size=args.model_size,
        frozen_modules=("all",),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    prefix = "backbone.text_model."
    text_state = {
        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    if not text_state:
        raise RuntimeError(f"no {prefix}* keys in {args.checkpoint}")
    incompatible = model.load_state_dict(text_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected load result: {incompatible}")
    model.to(args.device)
    model.eval()
    return model


def encode_texts(model: XLMRobertaLanguageBackbone, texts: list[str]) -> torch.Tensor:
    with torch.no_grad():
        # Backbone expects [batch][text]; we use batch=1.
        return model([texts]).squeeze(0).detach().cpu().float()  # [N, 768]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attr: dict[str, dict[str, str]] = json.loads(
        Path(args.attr_json).read_text(encoding="utf-8")
    )
    if len(attr) != 39:
        raise SystemExit(f"expected 39 attr entries, got {len(attr)}")

    # Validate every class has all 5 fields populated (non-empty strings)
    for cls, fields in attr.items():
        missing = [f for f in ATTR_FIELDS if not fields.get(f, "").strip()]
        if missing:
            raise SystemExit(f"class {cls!r} missing/empty fields: {missing}")

    base_ann = json.loads(Path(args.base_ann).read_text(encoding="utf-8"))
    base_names = sorted(c["name"] for c in base_ann["categories"])
    if len(base_names) != 30:
        raise SystemExit(f"base ann has {len(base_names)} classes, expected 30")
    for n in base_names:
        if n not in attr:
            raise SystemExit(f"base class {n!r} not in attr file")

    novel_names = list(NOVEL_NAMES_FROM_ATTR)
    for n in novel_names:
        if n not in attr:
            raise SystemExit(f"novel class {n!r} not in attr file")

    all_names = base_names + novel_names  # 30 + 9 = 39
    if set(all_names) != set(attr):
        extra = set(attr) - set(all_names)
        raise SystemExit(f"unaccounted attr classes: {extra}")

    # Flatten all 5 attrs × 39 classes into a single batch (more efficient)
    flat_texts: list[str] = []
    for cls in all_names:
        for f in ATTR_FIELDS:
            flat_texts.append(attr[cls][f].strip())
    print(f"[encode] {len(flat_texts)} attr strings ({len(all_names)} classes × 5 fields)")

    model = load_text_model(args)
    flat_embs = encode_texts(model, flat_texts)  # [195, 768]
    flat_embs = flat_embs.view(len(all_names), len(ATTR_FIELDS), -1)  # [39, 5, 768]
    print(f"[encode] per-attr embs: {tuple(flat_embs.shape)}")

    # Apply each aggregation strategy
    out_dicts: dict[str, dict[str, torch.Tensor]] = {}

    # 1. concat: [39, 5*768=3840]
    concat = flat_embs.reshape(len(all_names), -1).contiguous()
    out_dicts["concat"] = {n: concat[i] for i, n in enumerate(all_names)}

    # 2. sum: [39, 768]
    summed = flat_embs.sum(dim=1).contiguous()
    out_dicts["sum"] = {n: summed[i] for i, n in enumerate(all_names)}

    # 3. weighted-sum: [39, 768]
    alphas = torch.tensor(WEIGHTED_ALPHAS).view(1, len(ATTR_FIELDS), 1)
    weighted = (flat_embs * alphas).sum(dim=1).contiguous()
    out_dicts["weighted-sum"] = {n: weighted[i] for i, n in enumerate(all_names)}

    # 4. only-distinguish: just the 5th field (index 4)
    only_dist = flat_embs[:, ATTR_FIELDS.index("key_distinguishing_feature"), :].contiguous()
    out_dicts["only-distinguish"] = {n: only_dist[i] for i, n in enumerate(all_names)}

    for strat in STRATEGIES:
        out_path = out_dir / f"tct_ngc_attr_{strat}_emb.pth"
        torch.save(out_dicts[strat], out_path)
        sample_v = next(iter(out_dicts[strat].values()))
        print(f"[save] {out_path}  ({len(out_dicts[strat])} classes, dim={sample_v.shape[-1]})")

    # Companion JSONs — list-of-list of class names for cos-heatmap tool reuse
    base_json = [[n] for n in base_names]
    novel_json = [[n] for n in novel_names]
    (out_dir / "tct_ngc_attr_base30.json").write_text(
        json.dumps(base_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "tct_ngc_attr_novel9.json").write_text(
        json.dumps(novel_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[save] {out_dir/'tct_ngc_attr_base30.json'}  (30 names)")
    print(f"[save] {out_dir/'tct_ngc_attr_novel9.json'} (9 names)")


if __name__ == "__main__":
    main()
