#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
from PIL import Image, ImageDraw, ImageFont


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:80]


def load_font(size: int = 14):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "simsun.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, box, label: str, color: tuple[int, int, int], font) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    text_bbox = draw.textbbox((x1, max(0, y1 - 16)), label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((text_bbox[0], text_bbox[1]), label, fill=(255, 255, 255), font=font)


def make_panel(
    image_path: Path,
    gt_anns: list[dict],
    pred_instances,
    categories_by_id: dict[int, dict],
    label_to_name: list[str],
    target_cat_id: int,
    score_thr: float,
    topk: int,
    out_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    gt_panel = image.copy()
    pred_panel = image.copy()
    font = load_font()

    gt_draw = ImageDraw.Draw(gt_panel)
    pred_draw = ImageDraw.Draw(pred_panel)

    for ann in gt_anns:
        x, y, w, h = ann["bbox"]
        name = categories_by_id[ann["category_id"]]["name"]
        color = (0, 170, 60) if ann["category_id"] == target_cat_id else (150, 150, 150)
        draw_box(gt_draw, [x, y, x + w, y + h], f"GT {name}", color, font)

    scores = pred_instances.scores.float().detach().cpu()
    keep = scores >= score_thr
    pred_instances = pred_instances[keep]
    if len(pred_instances) > topk:
        pred_instances = pred_instances[pred_instances.scores.float().topk(topk).indices]
    pred_instances = pred_instances.cpu()

    for bbox, label, score in zip(
        pred_instances.bboxes,
        pred_instances.labels,
        pred_instances.scores,
    ):
        class_name = label_to_name[int(label)]
        draw_box(
            pred_draw,
            bbox.tolist(),
            f"P {class_name} {float(score):.2f}",
            (220, 40, 40),
            font,
        )

    width, height = image.size
    canvas = Image.new("RGB", (width * 2, height + 24), (255, 255, 255))
    canvas.paste(gt_panel, (0, 24))
    canvas.paste(pred_panel, (width, 24))
    header = ImageDraw.Draw(canvas)
    header.text((6, 4), "Ground truth", fill=(0, 0, 0), font=font)
    header.text((width + 6, 4), f"Prediction score >= {score_thr}", fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--score-thr", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--data-root",
        default=None,
        help="override cfg.data_root (e.g. $TCT_NGC_DATA_ROOT for test_base viz)",
    )
    parser.add_argument(
        "--ann-file",
        default=None,
        help="override the COCO annotation path (relative to --data-root if not absolute); "
        "swaps the source for both GT iteration and image lookup",
    )
    parser.add_argument(
        "--out-label",
        default="val",
        help="written into manifest.json as `source` so val/test viz can be told apart",
    )
    parser.add_argument(
        "--class-regex",
        default=None,
        help="if set, only categories whose name matches this regex (re.search) are visualized",
    )
    args = parser.parse_args()

    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    test_pipeline = Compose(cfg.test_pipeline)

    cfg_data_root = Path(cfg.val_dataloader.dataset.dataset.data_root)
    data_root = Path(args.data_root) if args.data_root else cfg_data_root
    image_prefix = data_root / cfg.val_dataloader.dataset.dataset.data_prefix["img"]

    if args.ann_file:
        ann_arg = Path(args.ann_file)
        ann_path = ann_arg if ann_arg.is_absolute() else data_root / ann_arg
    else:
        ann_path = Path(cfg.val_evaluator.ann_file)

    class_regex = re.compile(args.class_regex) if args.class_regex else None

    coco = json.loads(ann_path.read_text(encoding="utf-8"))

    categories = sorted(coco["categories"], key=lambda item: item["id"])
    categories_by_id = {cat["id"]: cat for cat in categories}
    class_names = list(cfg.dataset_metainfo.classes)
    label_to_name = class_names

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    images_by_id = {img["id"]: img for img in coco["images"]}
    image_ids_by_cat: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
        image_ids_by_cat[ann["category_id"]].add(ann["image_id"])

    texts = json.loads(Path(cfg.test_class_text_path).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    manifest = []

    for cat in categories:
        if class_regex and not class_regex.search(cat["name"]):
            continue
        image_ids = sorted(image_ids_by_cat[cat["id"]])
        if not image_ids:
            continue
        chosen = rng.sample(image_ids, min(args.samples_per_class, len(image_ids)))
        class_dir = out_dir / f"{class_names.index(cat['name']):02d}_{safe_name(cat['name'])}"
        for image_id in chosen:
            info = images_by_id[image_id]
            image_path = image_prefix / info["file_name"]
            data_info = dict(img_id=image_id, img_path=str(image_path), texts=texts)
            data_info = test_pipeline(data_info)
            data_batch = dict(
                inputs=data_info["inputs"].unsqueeze(0).to(args.device),
                data_samples=[data_info["data_samples"]],
            )
            with autocast(enabled=args.amp), torch.no_grad():
                output = model.test_step(data_batch)[0]

            out_path = class_dir / f"{image_id}_{Path(info['file_name']).name}"
            make_panel(
                image_path=image_path,
                gt_anns=anns_by_image[image_id],
                pred_instances=output.pred_instances,
                categories_by_id=categories_by_id,
                label_to_name=label_to_name,
                target_cat_id=cat["id"],
                score_thr=args.score_thr,
                topk=args.topk,
                out_path=out_path,
            )
            manifest.append(
                {
                    "category": cat["name"],
                    "image_id": image_id,
                    "image": str(image_path),
                    "output": str(out_path),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_obj = {
        "source": args.out_label,
        "data_root": str(data_root),
        "ann_file": str(ann_path),
        "checkpoint": args.checkpoint,
        "score_thr": args.score_thr,
        "samples_per_class": args.samples_per_class,
        "class_regex": args.class_regex,
        "samples": manifest,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(manifest)} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
