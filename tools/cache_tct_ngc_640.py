#!/usr/bin/env python
"""Create a 640x640 letterboxed cache for TCT_NGC COCO annotations.

The training configs already resize every original image to 640x640 before it
reaches the model. This script moves that deterministic resize and padding step
out of the dataloader and rewrites the COCO boxes into cached-image coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from wedetect.utils.data_paths import get_tct_ngc_root, get_tct_ngc_640_root

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=str(get_tct_ngc_root()),
        help="Original TCT_NGC root containing images and annotations.",
    )
    parser.add_argument(
        "--out-root",
        default=str(get_tct_ngc_640_root()),
        help="Output root for cached images and rewritten annotations.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_dev", "val_dev"],
        help="Annotation suffixes, for example train_dev val_dev.",
    )
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit per split. Do not use for full training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing cached images.",
    )
    return parser.parse_args()


def letterbox_params(width: int, height: int, size: int) -> dict[str, float | int]:
    keep_ratio = size / max(width, height)
    resized_w = int(width * keep_ratio)
    resized_h = int(height * keep_ratio)
    scale_x = resized_w / width
    scale_y = resized_h / height

    padding_w = size - resized_w
    padding_h = size - resized_h
    left = int(round(padding_w // 2 - 0.1))
    top = int(round(padding_h // 2 - 0.1))
    right = padding_w - left
    bottom = padding_h - top
    return {
        "resized_w": resized_w,
        "resized_h": resized_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def transform_bbox(
    bbox: list[float], params: dict[str, float | int], size: int
) -> tuple[list[float], float]:
    x, y, w, h = bbox
    scale_x = float(params["scale_x"])
    scale_y = float(params["scale_y"])
    left = float(params["left"])
    top = float(params["top"])

    x1 = x * scale_x + left
    y1 = y * scale_y + top
    x2 = (x + w) * scale_x + left
    y2 = (y + h) * scale_y + top

    x1 = min(max(x1, 0.0), float(size))
    y1 = min(max(y1, 0.0), float(size))
    x2 = min(max(x2, 0.0), float(size))
    y2 = min(max(y2, 0.0), float(size))

    new_w = max(0.0, x2 - x1)
    new_h = max(0.0, y2 - y1)
    return [x1, y1, new_w, new_h], new_w * new_h


def rewrite_annotation(
    source_root: Path,
    out_root: Path,
    split: str,
    size: int,
    limit: int | None,
) -> dict[str, tuple[int, int]]:
    ann_path = source_root / "annotations" / f"instances_{split}.json"
    with ann_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    if limit is not None:
        keep_images = data["images"][:limit]
        keep_ids = {img["id"] for img in keep_images}
        data["images"] = keep_images
        data["annotations"] = [
            ann for ann in data["annotations"] if ann["image_id"] in keep_ids
        ]

    params_by_id: dict[int, dict[str, float | int]] = {}
    files: dict[str, tuple[int, int]] = {}
    for img in data["images"]:
        width = int(img["width"])
        height = int(img["height"])
        params = letterbox_params(width, height, size)
        params_by_id[int(img["id"])] = params
        files[str(img["file_name"])] = (width, height)
        img["width"] = size
        img["height"] = size

    for ann in data["annotations"]:
        params = params_by_id[int(ann["image_id"])]
        ann["bbox"], ann["area"] = transform_bbox(ann["bbox"], params, size)

    out_ann_dir = out_root / "annotations"
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_ann_path = out_ann_dir / f"instances_{split}.json"
    with out_ann_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(
        f"Wrote {out_ann_path} with {len(data['images'])} images and "
        f"{len(data['annotations'])} annotations",
        flush=True,
    )
    return files


def resize_one(task: tuple[str, str, int, int, int, int, bool]) -> tuple[str, bool, str]:
    src, dst, width, height, size, quality, overwrite = task
    dst_path = Path(dst)
    if dst_path.exists() and not overwrite:
        return src, False, "exists"

    image = cv2.imread(src, cv2.IMREAD_COLOR)
    if image is None:
        return src, False, "missing_or_unreadable"
    if image.shape[1] != width or image.shape[0] != height:
        width = image.shape[1]
        height = image.shape[0]

    params = letterbox_params(width, height, size)
    resized_w = int(params["resized_w"])
    resized_h = int(params["resized_h"])
    interpolation = cv2.INTER_AREA if size / max(width, height) < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    padded = cv2.copyMakeBorder(
        resized,
        int(params["top"]),
        int(params["bottom"]),
        int(params["left"]),
        int(params["right"]),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if padded.shape[:2] != (size, size):
        return src, False, f"bad_shape_{padded.shape[:2]}"

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_path), padded, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return src, ok, "written" if ok else "write_failed"


def write_images(
    source_root: Path,
    out_root: Path,
    files: dict[str, tuple[int, int]],
    size: int,
    workers: int,
    quality: int,
    overwrite: bool,
) -> None:
    tasks = []
    for file_name, (width, height) in sorted(files.items()):
        src = source_root / "images" / file_name
        dst = out_root / "images" / file_name
        tasks.append((str(src), str(dst), width, height, size, quality, overwrite))

    print(f"Resizing {len(tasks)} unique images with {workers} workers", flush=True)
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(resize_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            _, ok, status = future.result()
            counts[status] = counts.get(status, 0) + 1
            if not ok and status != "exists":
                counts["failed"] = counts.get("failed", 0) + 1
            if index == 1 or index % 1000 == 0 or index == len(futures):
                print(f"processed {index}/{len(futures)} images; {counts}", flush=True)

    if counts.get("failed", 0):
        raise RuntimeError(f"Image cache had failures: {counts}")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    if not (source_root / "images").is_dir():
        raise FileNotFoundError(source_root / "images")
    os.makedirs(out_root, exist_ok=True)

    all_files: dict[str, tuple[int, int]] = {}
    for split in args.splits:
        files = rewrite_annotation(source_root, out_root, split, args.size, args.limit)
        all_files.update(files)

    write_images(
        source_root=source_root,
        out_root=out_root,
        files=all_files,
        size=args.size,
        workers=args.workers,
        quality=args.quality,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
