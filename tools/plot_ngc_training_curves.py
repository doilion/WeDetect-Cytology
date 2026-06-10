#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FLOAT_RE = r"(?:[-+]?(?:nan|inf)|[0-9.eE+-]+)"


TRAIN_RE = re.compile(
    r"Epoch\(train\)\s+\[(?P<epoch>\d+)\]\[\s*(?P<iter>\d+)/(?P<iters>\d+)\].*?"
    rf"lr:\s*(?P<lr>{FLOAT_RE}).*?"
    rf"time:\s*(?P<time>{FLOAT_RE}).*?"
    rf"loss:\s*(?P<loss>{FLOAT_RE}).*?"
    rf"loss_cls:\s*(?P<loss_cls>{FLOAT_RE}).*?"
    rf"loss_bbox:\s*(?P<loss_bbox>{FLOAT_RE}).*?"
    rf"loss_dfl:\s*(?P<loss_dfl>{FLOAT_RE})"
)

VAL_RE = re.compile(
    r"Epoch\(val\).*?coco/bbox_mAP:\s*(?P<map>[0-9.eE+-]+).*?"
    r"coco/bbox_mAP_50:\s*(?P<map50>[0-9.eE+-]+)"
)


def parse_train_log(path: Path) -> tuple[list[dict], list[dict]]:
    train_by_position: dict[tuple[int, int], dict] = {}
    val_rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRAIN_RE.search(line)
        if match:
            row = match.groupdict()
            epoch = int(row["epoch"])
            it = int(row["iter"])
            iters = int(row["iters"])
            row["epoch"] = epoch
            row["iter"] = it
            row["iters"] = iters
            for key in ("lr", "time", "loss", "loss_cls", "loss_bbox", "loss_dfl"):
                row[key] = float(row[key])
            train_by_position[(epoch, it)] = row
            continue

        match = VAL_RE.search(line)
        if match:
            val_rows.append(
                {
                    "epoch": len(val_rows) + 1,
                    "bbox_mAP": float(match.group("map")),
                    "bbox_mAP_50": float(match.group("map50")),
                }
            )
    train_rows = sorted(
        train_by_position.values(), key=lambda row: (row["epoch"], row["iter"])
    )
    epoch_lengths: dict[int, int] = {}
    for row in train_rows:
        epoch_lengths[row["epoch"]] = max(epoch_lengths.get(row["epoch"], 0), row["iters"])
    epoch_offsets: dict[int, int] = {}
    offset = 0
    for epoch in sorted(epoch_lengths):
        epoch_offsets[epoch] = offset
        offset += epoch_lengths[epoch]
    for row in train_rows:
        row["global_iter"] = epoch_offsets[row["epoch"]] + row["iter"]
    return train_rows, val_rows


def read_val_loss(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            {
                key: int(value) if key == "epoch" else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(f)
        ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def per_epoch_train_means(train_rows: list[dict]) -> list[dict]:
    """Average train_rows within each epoch, so train and val loss share an x-axis.
    The training log emits one row per ``log_interval`` iters; we just take the
    arithmetic mean of (loss, loss_cls, loss_bbox, loss_dfl) per epoch.
    """
    by_epoch: dict[int, list[dict]] = {}
    for row in train_rows:
        by_epoch.setdefault(row["epoch"], []).append(row)
    out = []
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        n = len(rows)
        out.append({
            "epoch": epoch,
            "loss": sum(r["loss"] for r in rows) / n,
            "loss_cls": sum(r["loss_cls"] for r in rows) / n,
            "loss_bbox": sum(r["loss_bbox"] for r in rows) / n,
            "loss_dfl": sum(r["loss_dfl"] for r in rows) / n,
        })
    return out


def make_two_panel_plot(out_path, train_rows, val_loss_rows, val_rows):
    """Legacy 2-panel plot: top = train loss components vs global iter,
    bottom = val loss components + val mAP on twinx."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    ax = axes[0]
    if train_rows:
        x = [row["global_iter"] for row in train_rows]
        ax.plot(x, [row["loss"] for row in train_rows], label="train loss", lw=1.8)
        ax.plot(x, [row["loss_cls"] for row in train_rows], label="train loss cls", lw=1.0)
        ax.plot(x, [row["loss_bbox"] for row in train_rows], label="train loss bbox", lw=1.0)
        ax.plot(x, [row["loss_dfl"] for row in train_rows], label="train loss dfl", lw=1.0)
    ax.set_title("Training Loss")
    ax.set_xlabel("global iter")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if val_loss_rows:
        epochs = [row["epoch"] for row in val_loss_rows]
        ax.plot(epochs, [row["loss"] for row in val_loss_rows], marker="o", label="val loss")
        ax.plot(epochs, [row["loss_cls"] for row in val_loss_rows], marker="o", label="val loss cls")
        ax.plot(epochs, [row["loss_bbox"] for row in val_loss_rows], marker="o", label="val loss bbox")
        ax.plot(epochs, [row["loss_dfl"] for row in val_loss_rows], marker="o", label="val loss dfl")
        ax.set_ylabel("loss")
        ax.legend(loc="upper left")
        ax.set_title("Validation Loss And mAP")
    else:
        # No val_loss available: hide the unused left "loss" axis entirely so the
        # mAP curve on the twinx isn't read as a rising loss.
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title("Validation mAP")
    if val_rows:
        ax2 = ax.twinx()
        epochs = [row["epoch"] for row in val_rows]
        ax2.plot(epochs, [row["bbox_mAP"] for row in val_rows], color="black", marker="x", label="bbox mAP")
        ax2.set_ylabel("mAP")
        ax2.legend(loc="lower right")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"wrote {out_path}")


def epoch_end_global_iters(train_rows: list[dict]) -> dict[int, int]:
    """For each epoch, return the largest global_iter seen (~ end-of-epoch).
    Used to align per-epoch val measurements with the per-iter train x-axis.
    """
    out: dict[int, int] = {}
    for row in train_rows:
        out[row["epoch"]] = max(out.get(row["epoch"], 0), row["global_iter"])
    return out


def make_three_panel_plot(out_path, train_rows, val_loss_rows, val_rows):
    """Three-panel breakdown: (1) train loss components per-iter on log scale;
    (2) train per-iter total + val total at epoch boundaries (same x=global_iter,
    same units) for the underfit/overfit gap; (3) val mAP alone."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

    end_iter = epoch_end_global_iters(train_rows)

    # Panel 1: train loss components per-iter (log y) — show that bbox/dfl are
    # already low at ep1 thanks to the COCO-pretrained backbone, and only cls
    # has real headroom. Per-iter keeps the noise visible (no averaging artifact).
    ax = axes[0]
    if train_rows:
        x = [r["global_iter"] for r in train_rows]
        ax.plot(x, [r["loss"] for r in train_rows], label="total", lw=1.5, alpha=0.8)
        ax.plot(x, [r["loss_cls"] for r in train_rows], label="cls", lw=1.0, alpha=0.8)
        ax.plot(x, [r["loss_bbox"] for r in train_rows], label="bbox", lw=1.0, alpha=0.8)
        ax.plot(x, [r["loss_dfl"] for r in train_rows], label="dfl", lw=1.0, alpha=0.8)
        ax.set_yscale("log")
    ax.set_title("Train loss per iter (log y) — pretrained bbox/dfl already low; cls dominates the drop")
    ax.set_xlabel("global iter")
    ax.set_ylabel("loss (log)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    # Panel 2: train per-iter total + val total at epoch boundaries (linear y) —
    # the canonical underfit/overfit diagnostic. Same units, same x-axis (no
    # twinx). Val is per-epoch, so it shows up as discrete markers at the
    # global_iter where each epoch ends; train keeps its iter-level noise.
    ax = axes[1]
    if train_rows:
        x = [r["global_iter"] for r in train_rows]
        ax.plot(x, [r["loss"] for r in train_rows],
                label="train loss (per iter)", lw=1.0, alpha=0.6)
    if val_loss_rows and end_iter:
        vx, vy = [], []
        for r in val_loss_rows:
            ep = r["epoch"]
            if ep in end_iter:
                vx.append(end_iter[ep])
                vy.append(r["loss"])
        ax.plot(vx, vy, marker="s", label="val loss (per epoch)", lw=2.0, color="C1")
    ax.set_title("Train (per-iter) vs val (per-epoch) total loss — gap = cls bottleneck on val")
    ax.set_xlabel("global iter")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Panel 3: val mAP alone (linear y, no twinx). x is global_iter for axis
    # consistency with panels 1 & 2; we annotate the epoch at each marker.
    ax = axes[2]
    if val_rows and end_iter:
        gx, mp, mp50, eps = [], [], [], []
        for r in val_rows:
            ep = r["epoch"]
            if ep in end_iter:
                gx.append(end_iter[ep])
                mp.append(r["bbox_mAP"])
                mp50.append(r["bbox_mAP_50"])
                eps.append(ep)
        ax.plot(gx, mp, marker="x", color="black", lw=2, label="val bbox mAP")
        ax.plot(gx, mp50, marker=".", color="dimgray", lw=1.2, label="val bbox mAP50")
        # mark best epoch
        best_idx = max(range(len(mp)), key=lambda i: mp[i])
        ax.scatter([gx[best_idx]], [mp[best_idx]], s=140, facecolors="none",
                   edgecolors="red", linewidths=2,
                   label=f"best ep{eps[best_idx]} ({mp[best_idx]:.3f})")
        # epoch tick marks on top axis for readability
        ax2 = ax.secondary_xaxis(
            "top",
            functions=(
                lambda gi: gi / (end_iter[max(end_iter)] / max(end_iter)),
                lambda ep: ep * (end_iter[max(end_iter)] / max(end_iter)),
            ),
        )
        ax2.set_xlabel("epoch")
    ax.set_title("Val mAP per epoch — peak marks the right max_epochs")
    ax.set_xlabel("global iter")
    ax.set_ylabel("mAP")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-loss-csv")
    parser.add_argument(
        "--three-panel", action="store_true",
        help="emit loss_curves_3panel.png with (a) train loss components on log y, "
             "(b) train vs val total overlay (linear, same units), (c) val mAP alone. "
             "Default behavior (no flag) still writes the legacy 2-panel loss_curves.png.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = parse_train_log(log_path)
    val_loss_rows = read_val_loss(Path(args.val_loss_csv) if args.val_loss_csv else None)

    write_csv(out_dir / "train_loss_from_log.csv", train_rows)
    write_csv(out_dir / "val_map_from_log.csv", val_rows)

    make_two_panel_plot(
        out_dir / "loss_curves.png", train_rows, val_loss_rows, val_rows,
    )
    if args.three_panel:
        make_three_panel_plot(
            out_dir / "loss_curves_3panel.png", train_rows, val_loss_rows, val_rows,
        )


if __name__ == "__main__":
    main()
