#!/usr/bin/env python
"""Analyze a method-suite run from its LOGS ONLY (no checkpoints needed).

Designed for the "ran on someone else's machine, only got the results back" case:
point it at the `logs/` directory produced by tools/run_method_suite.sh and it
emits figures + a written verdict. Everything is parsed from:
  - logs/suite_<stamp>.summary.tsv          (arm, base_macro_mAP, novel_mAP)
  - logs/<arm>_<stamp>.eval_base.log        (overall macro + per-class table)
  - logs/<arm>_<stamp>.eval_novel.log       (novel bbox_mAP + per-class table)
  - logs/<arm>_<stamp>.train.log            (loss curves + RelDistill diagnostic)
Any missing piece is skipped gracefully (you can run with only summary.tsv).

Usage:
  PYTHONPATH=. python tools/analyze_suite.py                      # newest suite in logs/
  python tools/analyze_suite.py --logs-dir logs --stamp 20260610_xxxxxx
  python tools/analyze_suite.py --out-dir docs/results/suite_analysis
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# per-class classwise table row: | name | mAP | mAP_50 | mAP_75 |  (same as plot_classwise_ap)
ROW_RE = re.compile(
    r"^\|\s+(?P<name>[^|]+?)\s+\|\s+(?P<map>[0-9.]+|nan)\s+\|"
    r"\s+(?P<map50>[0-9.]+|nan)\s+\|")
DIAG_RE = re.compile(
    r"image-proto=(?P<img>[0-9.]+)\s+text=(?P<txt>[0-9.]+)\s+gap\(text-image\)=(?P<gap>[-+0-9.]+)")
ITER_RE = re.compile(r"Epoch\(train\)\s+\[(?P<ep>\d+)\]\[\s*(?P<it>\d+)/(?P<tot>\d+)\]")

ORGAN_COLOR = {
    "respiratory tract": "#4C9AFF", "Serous effusion": "#7AC274",
    "Thyroid gland": "#F2A93B", "Urine": "#E0584C", "TCT_CCD": "#9E7BD0",
}
# the canonical arm order + which deltas decide each module
ARM_ORDER = ["baseline", "attr_mean", "attr", "decone", "reldistill",
             "decone_reldistill", "stitch", "stitchb"]


def num_after(text: str, key: str):
    m = re.findall(re.escape(key) + r":\s*([0-9.]+)", text)
    return float(m[-1]) if m else None


def parse_summary(tsv: Path) -> dict:
    out = {}
    if not tsv.exists():
        return out
    for ln in tsv.read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 3:
            out[p[0]] = (_f(p[1]), _f(p[2]))
    return out


def _f(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_classwise(log: Path) -> list:
    rows = []
    if not log.exists():
        return rows
    for ln in log.read_text(errors="ignore").splitlines():
        m = ROW_RE.match(ln)
        if not m or m.group("name").strip() == "category":
            continue
        try:
            ap = float(m.group("map"))
        except ValueError:
            continue
        if math.isfinite(ap):
            rows.append((m.group("name").strip(), ap))
    # dedup keep last (re-eval prints multiple tables)
    d = {}
    for n, ap in rows:
        d[n] = ap
    return list(d.items())


def parse_metrics_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import json
        return json.loads(path.read_text())
    except Exception:
        return {}


def json_overall(d: dict, suffix: str):
    ks = [k for k in d if k.endswith(suffix)]
    return _f(d[ks[-1]]) if ks else None


def json_perclass(d: dict) -> list:
    # OrganRestrictedCocoMetric dumps coco/class/<slug>/mAP per kept class.
    out = []
    for k, v in d.items():
        m = re.search(r"class/(?P<slug>.+?)/mAP$", k)
        if m and _f(v) is not None:
            out.append((m.group("slug").replace("_", " "), _f(v)))
    return out


def parse_train(log: Path):
    if not log.exists():
        return None
    its, loss, lcls, lrel = [], [], [], []
    diag = None
    glob_it = 0
    for ln in log.read_text(errors="ignore").splitlines():
        if "RelDistill diag" in ln:
            m = DIAG_RE.search(ln)
            if m:
                diag = (float(m.group("img")), float(m.group("txt")), float(m.group("gap")))
        m = ITER_RE.search(ln)
        if not m:
            continue
        glob_it += 1
        its.append(glob_it)
        loss.append(num_after(ln, "loss"))
        lcls.append(num_after(ln, "loss_cls"))
        lrel.append(num_after(ln, "loss_rel_distill"))
    if not its:
        return None
    return dict(it=its, loss=loss, loss_cls=lcls, loss_rel_distill=lrel, diag=diag)


def organ_of(name: str) -> str:
    for k in ORGAN_COLOR:
        if name.startswith(k):
            return k
    return "other"


# ---------------------------------------------------------------- figures ----
def fig_headline(summary, out: Path):
    arms = [a for a in ARM_ORDER if a in summary]
    if not arms:
        return None
    base = [summary[a][0] if summary[a][0] is not None else 0 for a in arms]
    nov = [summary[a][1] if summary[a][1] is not None else 0 for a in arms]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(arms)), 5))
    ax.bar(x - 0.2, base, 0.4, label="base organ-macro mAP", color="#3B6FB5")
    ax.bar(x + 0.2, nov, 0.4, label="novel mAP (9-class ZS)", color="#C0603B")
    for i, (b, n) in enumerate(zip(base, nov)):
        ax.text(i - 0.2, b + 0.003, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, n + 0.003, f"{n:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("mAP")
    ax.set_title("Method suite — headline mAP per arm")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_per_class(summary_logs, split: str, out: Path):
    # summary_logs: dict arm -> list[(class, ap)]
    arms = [a for a in ARM_ORDER if a in summary_logs and summary_logs[a]]
    if len(arms) < 1:
        return None
    classes = list(dict.fromkeys(c for a in arms for c, _ in summary_logs[a]))
    if not classes:
        return None
    data = np.array([[dict(summary_logs[a]).get(c, np.nan) for c in classes] for a in arms])
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(classes)), 0.6 * len(arms) + 2))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0,
                   vmax=np.nanmax(data) if np.isfinite(np.nanmax(data)) else 1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=9)
    for i in range(len(arms)):
        for j in range(len(classes)):
            v = data[i, j]
            if math.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.5 * np.nanmax(data) else "black")
    ax.set_title(f"Per-class AP ({split})")
    fig.colorbar(im, ax=ax, shrink=0.7, label="AP")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_train(trains, out: Path):
    arms = [a for a in ARM_ORDER if a in trains and trains[a]]
    if not arms:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for a in arms:
        t = trains[a]
        axes[0].plot(t["it"], t["loss"], label=a, alpha=0.8, lw=1)
        rel = [r for r in t["loss_rel_distill"] if r is not None]
        if any(r and r > 0 for r in (t["loss_rel_distill"] or [])):
            axes[1].plot(t["it"], t["loss_rel_distill"], label=a, alpha=0.8, lw=1)
    axes[0].set_title("total loss"); axes[0].set_xlabel("logged iter"); axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("loss_rel_distill (reldistill arms)"); axes[1].set_xlabel("logged iter")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


# ---------------------------------------------------------------- verdict ----
def verdict_line(name, val, eps=0.005):
    if val is None:
        return f"  {name}: (missing)"
    tag = "✅ helps" if val > eps else ("➖ TIE" if val >= -eps else "🔻 hurts")
    return f"  {name}: {val:+.4f}  {tag}"


def write_markdown(summary, base_cw, nov_cw, trains, out: Path):
    L = ["# Method suite analysis (from logs only)\n"]
    L.append("## Headline mAP\n")
    L.append("| arm | base organ-macro | novel (9-cls ZS) |")
    L.append("|---|---|---|")
    for a in [x for x in ARM_ORDER if x in summary]:
        b, n = summary[a]
        L.append(f"| {a} | {b if b is not None else '-'} | {n if n is not None else '-'} |")
    L.append("\n## Go / no-go (the decisive deltas)\n")

    def g(a, idx):
        return summary.get(a, (None, None))[idx]
    d_gate = None
    if g("attr", 0) is not None and g("attr_mean", 0) is not None:
        d_gate = g("attr", 0) - g("attr_mean", 0)
    d_text = None
    if g("decone_reldistill", 1) is not None and g("baseline", 1) is not None:
        d_text = g("decone_reldistill", 1) - g("baseline", 1)
    L.append("Module 1 — region-adaptive gate (clean: both morph6 text):")
    L.append(verdict_line("attr − attr_mean (base)", d_gate))
    L.append("\nModule 2 — de-coned relational distillation (clean: both fullnames text):")
    L.append(verdict_line("decone_reldistill − baseline (novel)", d_text))
    # attribution of module 2
    if g("decone", 1) is not None and g("baseline", 1) is not None:
        L.append(verdict_line("  decone-only − baseline (novel)", g("decone", 1) - g("baseline", 1)))
    if g("reldistill", 1) is not None and g("baseline", 1) is not None:
        L.append(verdict_line("  reldistill-only − baseline (novel)", g("reldistill", 1) - g("baseline", 1)))
    # combine
    for c in ("stitch", "stitchb"):
        if g(c, 0) is not None:
            best_single = max(x for x in [g("attr", 0), g("decone_reldistill", 0)] if x is not None) \
                if any(g(k, 0) is not None for k in ("attr", "decone_reldistill")) else None
            if best_single is not None:
                L.append(verdict_line(f"{c} − best single module (base)", g(c, 0) - best_single))

    # reldistill teacher well-posedness
    L.append("\n## RelDistill teacher well-posedness (step-100 diagnostic)\n")
    L.append("Want image-proto cos < text cos (teacher more discriminative -> pulls text the right way).")
    for a in [x for x in ARM_ORDER if x in trains and trains[x] and trains[x].get("diag")]:
        img, txt, gap = trains[a]["diag"]
        tag = "✅ right direction" if gap < 0 else "🔻 WRONG (teacher more collapsed)"
        L.append(f"  {a}: image-proto={img:.3f} text={txt:.3f} gap={gap:+.3f}  {tag}")

    # per-class movers (base + novel). For BASE keep it in-family: only the morph6
    # attribute arms are compared against attr_mean (comparing the fullnames arms to a
    # morph6 baseline would be the cross-family mistake). Novel uses one novel text -> vs baseline.
    attr_family = {"attr", "attr_b5", "stitch", "stitchb"}
    for split, cw in (("base", base_cw), ("novel", nov_cw)):
        bl = "attr_mean" if split == "base" else "baseline"
        movers_arms = [x for x in ARM_ORDER if x in cw and x != bl and cw[x]
                       and (split == "novel" or x in attr_family)]
        if cw.get(bl) and movers_arms:
            L.append(f"\n## Biggest {split} per-class movers vs {bl}"
                     + ("  (attribute-family arms only)" if split == "base" else "") + "\n")
            base_d = dict(cw[bl])
            for a in movers_arms:
                d = {c: ap - base_d.get(c, 0.0) for c, ap in cw[a]}
                top = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:3]
                bot = sorted(d.items(), key=lambda kv: kv[1])[:3]
                L.append(f"- **{a}**: up {', '.join(f'{c} {v:+.3f}' for c, v in top)} | "
                         f"down {', '.join(f'{c} {v:+.3f}' for c, v in bot)}")

    L.append("\n## Caveats (read before claiming a win)\n")
    L.append("- base headline is recall-capped (AR_small≈0.17); a classification gain may not move base mAP.")
    L.append("- novel is image-encoder-capped (novel imgs map into base regions); text fixes have a ceiling.")
    L.append("- `attr`/`attr_mean` use morph6 text; the others use fullnames — only compare WITHIN a family.")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=None,
                   help="a suite result bundle (results/<stamp>/) — PREFERRED: reads machine-"
                        "readable {base,novel}_metrics.json (no weights, robust). "
                        "Default: auto-detect newest results/<stamp>/, else fall back to --logs-dir.")
    p.add_argument("--logs-dir", default="logs", help="fallback: parse raw mmengine logs")
    p.add_argument("--stamp", default=None, help="suite stamp; default = newest")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    # ---- prefer a JSON result bundle (results/<stamp>/) ----
    rdir = None
    if args.results_dir:
        rdir = Path(args.results_dir)
    else:
        cands = sorted((REPO / "results").glob("*/summary.tsv"), key=lambda f: f.stat().st_mtime) \
            if (REPO / "results").exists() else []
        if cands:
            rdir = cands[-1].parent

    if rdir and (rdir / "summary.tsv").exists():
        stamp = args.stamp or rdir.name
        out = Path(args.out_dir) if args.out_dir else REPO / "docs" / "results" / f"suite_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"[analyze] bundle={rdir}  out={out}")
        summary = parse_summary(rdir / "summary.tsv")
        base_cw, nov_cw, trains = {}, {}, {}
        for a in ARM_ORDER:
            bj = parse_metrics_json(rdir / a / "base_metrics.json")
            nj = parse_metrics_json(rdir / a / "novel_metrics.json")
            bc = json_perclass(bj)
            if bc:
                base_cw[a] = bc
            nc = json_perclass(nj)              # novel per-class if the metric dumped it
            if nc:
                nov_cw[a] = nc
            t = parse_train(rdir / a / "train_curve.log")
            if t:
                trains[a] = t
        made = []
        for f in [fig_headline(summary, out / "headline_mAP.png"),
                  fig_per_class(base_cw, "base", out / "per_class_base.png"),
                  fig_per_class(nov_cw, "novel", out / "per_class_novel.png"),
                  fig_train(trains, out / "training_loss.png")]:
            if f:
                made.append(f); print(f"  wrote {f}")
        md = write_markdown(summary, base_cw, nov_cw, trains, out / "analysis.md")
        print(f"  wrote {md}\n[analyze] done (bundle): {len(made)} figures + analysis.md in {out}")
        return

    # ---- fall back to raw logs ----
    logs = Path(args.logs_dir)
    stamp = args.stamp
    if stamp is None:
        cands = sorted(logs.glob("suite_*.summary.tsv"), key=lambda f: f.stat().st_mtime)
        if not cands:
            raise SystemExit(f"no result bundle and no suite_*.summary.tsv in {logs}")
        stamp = cands[-1].name[len("suite_"):-len(".summary.tsv")]
    out = Path(args.out_dir) if args.out_dir else REPO / "docs" / "results" / f"suite_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[analyze] stamp={stamp}  logs={logs}  out={out}")

    summary = parse_summary(logs / f"suite_{stamp}.summary.tsv")
    base_cw = {a: parse_classwise(logs / f"{a}_{stamp}.eval_base.log") for a in ARM_ORDER}
    nov_cw = {a: parse_classwise(logs / f"{a}_{stamp}.eval_novel.log") for a in ARM_ORDER}
    trains = {a: parse_train(logs / f"{a}_{stamp}.train.log") for a in ARM_ORDER}
    base_cw = {k: v for k, v in base_cw.items() if v}
    nov_cw = {k: v for k, v in nov_cw.items() if v}
    trains = {k: v for k, v in trains.items() if v}

    made = []
    for f in [
        fig_headline(summary, out / "headline_mAP.png"),
        fig_per_class(base_cw, "base", out / "per_class_base.png"),
        fig_per_class(nov_cw, "novel", out / "per_class_novel.png"),
        fig_train(trains, out / "training_loss.png"),
    ]:
        if f:
            made.append(f); print(f"  wrote {f}")
    md = write_markdown(summary, base_cw, nov_cw, trains, out / "analysis.md")
    print(f"  wrote {md}")
    print(f"[analyze] done: {len(made)} figures + analysis.md in {out}")


if __name__ == "__main__":
    main()
