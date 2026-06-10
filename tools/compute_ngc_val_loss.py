#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import re
from copy import deepcopy
from itertools import islice
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmdet.utils import register_all_modules


EPOCH_RE = re.compile(r"epoch_(\d+)\.pth$")


def checkpoint_epoch(path: str) -> int:
    match = EPOCH_RE.search(Path(path).name)
    return int(match.group(1)) if match else 10**9


def scalar(value: torch.Tensor | list | tuple) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.mean()
    return sum(v.mean() for v in value)


def build_val_loss_loader_cfg(cfg: Config, batch_size: int, num_workers: int):
    dataloader_cfg = deepcopy(cfg.val_dataloader)
    dataloader_cfg.batch_size = batch_size
    dataloader_cfg.num_workers = num_workers
    dataloader_cfg.persistent_workers = num_workers > 0
    dataloader_cfg.collate_fn = dict(type="yolow_collate")
    dataloader_cfg.dataset.dataset.test_mode = False
    return dataloader_cfg


def loss_from_collated_batch(model, data_batch: dict) -> dict:
    data = model.data_preprocessor(data_batch, training=True)
    model.bbox_head.num_classes = model.num_train_classes
    img_feats, txt_feats = model.extract_feat(data["inputs"], data["data_samples"])
    return model.bbox_head.loss(img_feats, txt_feats, data["data_samples"])


def evaluate_checkpoint(runner: Runner, checkpoint: str, dataloader) -> dict[str, float]:
    load_checkpoint(runner.model, checkpoint, map_location="cpu")
    model = runner.model
    # YOLOWorldHead emits bbox distribution logits only in training mode, and
    # those logits are required for DFL loss. no_grad avoids optimizer state
    # changes; this computes a validation-set training loss.
    model.train()

    totals = {"loss": 0.0, "loss_cls": 0.0, "loss_bbox": 0.0, "loss_dfl": 0.0}
    num_batches = 0
    with torch.no_grad():
        for data_batch in dataloader:
            losses = loss_from_collated_batch(model, data_batch)
            row = {key: float(scalar(losses[key]).detach().cpu()) for key in totals if key != "loss"}
            row["loss"] = row["loss_cls"] + row["loss_bbox"] + row["loss_dfl"]
            for key, value in row.items():
                totals[key] += value
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError("validation dataloader produced no batches")
    return {key: value / num_batches for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    register_all_modules()
    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    cfg.load_from = None
    cfg.resume = False
    cfg.work_dir = str(Path(args.out).resolve().parent / "val_loss_work_dir")

    checkpoints = sorted(glob.glob(args.checkpoint_glob), key=checkpoint_epoch)
    if not checkpoints:
        raise FileNotFoundError(args.checkpoint_glob)

    runner = Runner.from_cfg(cfg)
    dataloader = runner.build_dataloader(
        build_val_loss_loader_cfg(cfg, args.batch_size, args.num_workers)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in checkpoints:
        epoch = checkpoint_epoch(checkpoint)
        print(f"computing val loss for epoch {epoch}: {checkpoint}", flush=True)
        limited_dataloader = dataloader
        if args.max_batches > 0:
            limited_dataloader = islice(dataloader, args.max_batches)
        metrics = evaluate_checkpoint(runner, checkpoint, limited_dataloader)
        row = {"epoch": epoch, **metrics}
        rows.append(row)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(row, flush=True)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
