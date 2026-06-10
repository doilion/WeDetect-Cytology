"""Build merged 9-class novel text prompts JSON + BiomedCLIP embedding cache.

Concatenates the existing per-split text/emb files in the same class order as
the merged ann (built by `tools/build_merged_novel_ann.py`).

Inputs (all already exist):
    data/texts/tct_ngc_novel_main_3.json                  (3 prompts)
    data/texts/tct_ngc_novel_main_3_emb_biomedclip.pth
    data/texts/tct_ngc_novel_pseudo_2.json                (2 prompts)
    data/texts/tct_ngc_novel_pseudo_2_emb_biomedclip.pth
    data/texts/tct_ngc_novel_hard_4.json                  (4 prompts)
    data/texts/tct_ngc_novel_hard_4_emb_biomedclip.pth
    data/TCT_NGC/annotations/instances_test_novel_merged_9.json   (cat_id order)

Outputs:
    data/texts/tct_ngc_novel_merged_9.json                (9 prompts in cat_id order)
    data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth  (dict {prompt: Tensor[512]})

Class order must match merged ann categories:
    0  respiratory tract-Squamous cell carcinoma  ← main_3
    1  respiratory tract-adenocarcinoma            ← pseudo_2
    2  respiratory tract-Small cell carcinoma     ← hard_4
    3  Serous effusion-Breast cancer              ← main_3
    4  Serous effusion-Ovarian cancer             ← pseudo_2
    5  Serous effusion-adenocarcinoma             ← hard_4
    6  Thyroid gland-MTC                           ← main_3
    7  Thyroid gland-Malignant tumour             ← hard_4
    8  Thyroid gland-Suspicious for Malignancy    ← hard_4
"""
import argparse
import json
import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("TCT_NGC_DATA_ROOT", REPO_ROOT / "data" / "TCT_NGC"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ann', type=Path,
                   default=DATA_ROOT / 'annotations/instances_test_novel_merged_9.json')
    p.add_argument('--text-dir', type=Path, default=Path('data/texts'))
    p.add_argument('--out-json', type=Path,
                   default=Path('data/texts/tct_ngc_novel_merged_9.json'))
    p.add_argument('--out-emb', type=Path,
                   default=Path('data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth'))
    args = p.parse_args()

    # ── Load merged ann to learn class names in cat_id order ──
    ann = json.loads(args.ann.read_text())
    target_names = [c['name'] for c in sorted(ann['categories'], key=lambda c: c['id'])]
    target_sources = {c['name']: c['source_split'] for c in ann['categories']}

    # ── Per-source class-name lists (in source ann order) ──
    src_anns = {
        'main_3': DATA_ROOT / 'annotations/instances_test_main_novel.json',
        'pseudo_2': DATA_ROOT / 'annotations/instances_test_pseudo_novel.json',
        'hard_4': DATA_ROOT / 'annotations/instances_hard_test.json',
    }
    src_class_order = {}                # source -> [class_name, ...]
    for src, ann_path in src_anns.items():
        d = json.loads(Path(ann_path).read_text())
        src_class_order[src] = [c['name'] for c in sorted(d['categories'], key=lambda c: c['id'])]

    # ── Load source text JSON + emb (per source, parallel order) ──
    src_text = {}                       # source -> [prompt, ...]
    src_emb = {}                        # source -> dict {prompt: Tensor[512]}
    for src in src_anns:
        json_path = args.text_dir / f'tct_ngc_novel_{src}.json'
        emb_path = args.text_dir / f'tct_ngc_novel_{src}_emb_biomedclip.pth'
        prompts = json.loads(json_path.read_text())
        prompts_flat = [p[0] if isinstance(p, list) else p for p in prompts]
        emb = torch.load(emb_path, weights_only=False, map_location='cpu')
        src_text[src] = prompts_flat
        src_emb[src] = emb

    # ── Assemble in target order ──
    merged_prompts = []
    merged_emb = {}
    for class_name in target_names:
        src = target_sources[class_name]
        src_classes = src_class_order[src]
        src_idx = src_classes.index(class_name)
        prompt = src_text[src][src_idx]
        # Look up emb by the prompt key
        if prompt not in src_emb[src]:
            raise SystemExit(
                f"prompt {prompt!r} not found in {src} emb cache keys: "
                f"{list(src_emb[src].keys())[:3]}...")
        merged_prompts.append([prompt])           # mmdet text fmt expects list[list[str]]
        merged_emb[prompt] = src_emb[src][prompt]

    args.out_json.write_text(json.dumps(merged_prompts, indent=2))
    torch.save(merged_emb, args.out_emb)
    print(f"saved: {args.out_json}  (9 prompts)")
    print(f"saved: {args.out_emb}  ({len(merged_emb)} embeddings, dim={next(iter(merged_emb.values())).shape[-1]})")
    print("class order:")
    for i, (name, prompt) in enumerate(zip(target_names, merged_prompts)):
        print(f"  {i}  {name:50s}  src={target_sources[name]:9s}  prompt[:60]={prompt[0][:60]!r}")


if __name__ == '__main__':
    main()
