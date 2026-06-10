#!/usr/bin/env python
"""Compare per-class AP between two MMEngine eval logs (e.g. val_dev vs test_base)."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROW_RE = re.compile(
    r"^\|\s+(?P<name>[^|]+?)\s+\|\s+(?P<map>[0-9.]+|nan)\s+\|"
    r"\s+(?P<map50>[0-9.]+|nan)\s+\|\s+(?P<map75>[0-9.]+|nan)\s+\|"
)

SUPER_COLOR = {
    "respiratory tract": "#4C9AFF",
    "Serous effusion": "#7AC274",
    "Thyroid gland": "#F2A93B",
    "Urine": "#E0584C",
    "TCT_CCD": "#9E7BD0",
}


def supercategory(name: str) -> str:
    for prefix in SUPER_COLOR:
        if name.startswith(prefix):
            return prefix
    return "other"


def parse_log(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if name == "category":
            continue
        try:
            ap = float(m.group("map"))
        except ValueError:
            continue
        if not math.isfinite(ap):
            continue
        out[name] = ap
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val-log", required=True)
    p.add_argument("--test-log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-label", default="val_dev (selection)")
    p.add_argument("--test-label", default="test_base_clean (held-out)")
    p.add_argument("--title", default="Per-class mAP — val vs test (epoch 12 best)")
    args = p.parse_args()

    val = parse_log(Path(args.val_log))
    test = parse_log(Path(args.test_log))
    common = sorted(set(val) & set(test), key=lambda k: test[k])  # sort by test asc → worst at top
    if not common:
        raise SystemExit("no overlapping classes between the two logs")

    n = len(common)
    val_aps = np.array([val[c] for c in common])
    test_aps = np.array([test[c] for c in common])
    deltas = test_aps - val_aps
    colors = [SUPER_COLOR.get(supercategory(c), "#888888") for c in common]

    fig, ax = plt.subplots(figsize=(13, max(6, 0.45 * n)))
    y = np.arange(n)
    bar_h = 0.38
    ax.barh(y + bar_h / 2, val_aps, height=bar_h, color=colors, alpha=0.45,
            edgecolor="black", linewidth=0.4, hatch="///")
    ax.barh(y - bar_h / 2, test_aps, height=bar_h, color=colors, alpha=0.95,
            edgecolor="black", linewidth=0.4)

    for i, (v, t, d) in enumerate(zip(val_aps, test_aps, deltas)):
        ax.text(v + 0.005, i + bar_h / 2, f"val {v:.3f}", va="center",
                fontsize=7, color="#444444")
        ax.text(t + 0.005, i - bar_h / 2, f"test {t:.3f}", va="center",
                fontsize=8, fontweight="bold")

    # Right margin: delta column
    xmax = max(val_aps.max(), test_aps.max()) + 0.18
    delta_x = xmax - 0.05
    for i, d in enumerate(deltas):
        color = "#c0392b" if d < -0.05 else ("#27ae60" if d > 0.02 else "#666666")
        ax.text(delta_x, i, f"Δ {d:+.3f}", va="center", ha="right",
                fontsize=8, fontweight="bold", color=color)

    ax.set_yticks(list(y))
    ax.set_yticklabels(common, fontsize=8)
    ax.set_xlabel("mAP (IoU=0.50:0.95)")
    ax.set_xlim(0, xmax)
    ax.set_title(args.title)

    mean_val = float(val_aps.mean())
    mean_test = float(test_aps.mean())
    ax.axvline(mean_val, color="#444444", linestyle=":", linewidth=0.9)
    ax.axvline(mean_test, color="black", linestyle="--", linewidth=0.9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in SUPER_COLOR.items()
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#666666", alpha=0.45,
                                 hatch="///", label=args.val_label))
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#666666", alpha=0.95,
                                 label=args.test_label))
    handles.append(plt.Line2D([0], [0], color="#444444", linestyle=":",
                              label=f"mean val mAP = {mean_val:.3f}"))
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=f"mean test mAP = {mean_test:.3f}"))
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(
        f"wrote {out} ({n} classes, mean val={mean_val:.4f} test={mean_test:.4f}, "
        f"test-val={mean_test - mean_val:+.4f})"
    )


if __name__ == "__main__":
    main()
