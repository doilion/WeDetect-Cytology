#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_prompt_keys(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    keys: list[str] = []
    for idx, group in enumerate(data):
        if not isinstance(group, list) or not group:
            raise ValueError(f"prompt group {idx} must be a non-empty list")
        key = group[0]
        if not isinstance(key, str) or not key:
            raise ValueError(f"prompt group {idx} has an invalid first prompt")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("prompt keys must be unique")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed random embeddings for PseudoLanguageBackbone."
    )
    parser.add_argument("--text-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()

    keys = load_prompt_keys(Path(args.text_json))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    vectors = F.normalize(
        torch.randn(len(keys), args.dim, generator=generator, dtype=torch.float32),
        dim=1,
    )
    embeddings = {
        key: vectors[idx].contiguous()
        for idx, key in enumerate(keys)
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)
    print(
        f"wrote {out_path} with {len(embeddings)} embeddings, "
        f"dim={args.dim}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
