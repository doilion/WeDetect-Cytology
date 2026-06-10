#!/usr/bin/env python
"""Compute 4 clinical evaluation metrics from a persisted COCO predictions JSON.

Inputs:
  --preds          <work_dir>/.../preds.bbox.json    (list of {image_id, bbox, score, category_id})
  --ann            COCO-style GT annotation JSON
  --cost-config    tools/clinical_cost_config.json   (severity tiers + cost rules)
  --out            output directory

Outputs (under --out):
  clinical_metrics.json     - all numeric metrics + meta
  summary.md                - human-readable headline table
  confusion_matrix.csv      - true_class x pred_class match counts
  pr_screening_all.csv      - PR curve for image-level all-positive screening
  pr_screening_highrisk.csv - PR curve for image-level L2/L3-only screening

Metrics:
  M1 image-level binary screening      (dual: all-positive + high-risk L2/L3)
  M2 cost-weighted error               (greedy 1-to-1 IoU match, clinical-conservative)
  M3 sensitivity @ specificity targets (0.90, 0.95, 0.99) on high-risk screening
  M4 top-K box recall                  (K=1, 5, 10, 20)

Matching rule (M2 + M4):
  Predictions sorted by score descending, greedy 1-to-1 against GT boxes via IoU >= 0.5.
  Each pred and each GT is matched at most once. Unmatched GTs count as missed
  detections (cost = 10 * tier of GT). Unmatched preds count as overcall FPs.
  'ignored' classes (background cytology) are filtered out before matching.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _roc_pr_curves(y_true: np.ndarray, y_score: np.ndarray):
    """Compute ROC and PR curves from binary labels + continuous scores.

    Numpy-only (no sklearn dep). Returns:
      fpr, tpr, thresholds_roc        (with leading (0, 0, +inf))
      precision, recall, thresholds_pr (descending threshold)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    desc = np.argsort(-y_score, kind="stable")
    y_t = y_true[desc]
    y_s = y_score[desc]
    distinct = np.where(np.diff(y_s))[0]
    keep = np.concatenate([distinct, [len(y_s) - 1]])
    tps = np.cumsum(y_t)[keep]
    fps = (1 + keep) - tps
    n_pos = int(y_t.sum())
    n_neg = len(y_t) - n_pos
    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps, dtype=float)
    thresholds = y_s[keep]
    # ROC curve with leading (0, 0)
    fpr_roc = np.concatenate([[0.0], fpr])
    tpr_roc = np.concatenate([[0.0], tpr])
    thr_roc = np.concatenate([[np.inf], thresholds])
    # PR curve
    denom = tps + fps
    precision = np.where(denom > 0, tps / np.maximum(denom, 1), 1.0)
    recall = tpr.copy()
    return fpr_roc, tpr_roc, thr_roc, precision, recall, thresholds


def auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, *_ = _roc_pr_curves(y_true, y_score)
    return float(np.trapz(tpr, fpr))


def auprc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision: sum_n (R_n - R_{n-1}) * P_n  (sklearn convention)."""
    _, _, _, precision, recall, _ = _roc_pr_curves(y_true, y_score)
    if len(precision) == 0:
        return float("nan")
    # Prepend (R=0, P=precision[0]) so the first step is (R[0] - 0) * P[0]
    rec = np.concatenate([[0.0], recall])
    pre = np.concatenate([precision[:1], precision])
    return float(np.sum(np.diff(rec) * pre[1:]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True,
                        help="bbox.json predictions (COCO list format)")
    parser.add_argument("--ann", required=True, help="GT annotation JSON")
    parser.add_argument("--cost-config", required=True,
                        help="tools/clinical_cost_config.json")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top-k-list", type=int, nargs="+",
                        default=[1, 5, 10, 20])
    return parser.parse_args()


def load_tier_assignment(cost_config_path: Path):
    """Returns (tier_of_name, raw_cfg_dict).

    tier_of_name[name] = 0/1/2/3 for L0/L1/L2/L3, or None if class is 'ignored'.
    """
    cfg = json.loads(cost_config_path.read_text(encoding="utf-8"))
    tiers = cfg["tiers"]
    out = {}
    for name in tiers["L0_negative"]:
        out[name] = 0
    for name in tiers["L1_atypical"]:
        out[name] = 1
    for name in tiers["L2_suspicious"]:
        out[name] = 2
    for name in tiers["L3_malignant"]:
        out[name] = 3
    for name in tiers["ignored"]:
        out[name] = None
    return out, cfg


def cost_value(true_tier, pred_tier) -> float:
    """Clinical-conservative cost rule.

    Sentinels:
      pred_tier == "NO_PRED"   no prediction matched this GT (miss)
      true_tier == "NO_GT"     prediction had no matching GT (false positive)
    """
    if true_tier == "NO_GT":
        # FP: prediction without GT; cost ~ severity of pred
        if pred_tier == 0:
            return 0.0       # FP on negative class is harmless
        return float(pred_tier)
    if pred_tier == "NO_PRED":
        # Miss: no detection for this GT
        if true_tier == 0:
            return 0.0       # missed negative is fine
        return 10.0 * float(true_tier)
    if true_tier == 0:
        # GT is negative
        if pred_tier == 0:
            return 0.0
        return float(pred_tier)        # overcall, cost = pred_tier (mild)
    # GT is positive (1/2/3)
    if pred_tier == 0:
        return 10.0 * float(true_tier)  # called positive negative = severe miss
    # both positive: subtype mismatch
    return float(abs(int(true_tier) - int(pred_tier)))


def iou_xywh(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1 = max(ax, bx); iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match(preds, gts, iou_thr: float):
    """1-to-1 greedy by score desc.

    Args:
      preds: list of dicts with keys 'bbox' [x,y,w,h], 'score', 'category_id'.
      gts:   list of dicts with keys 'bbox' [x,y,w,h], 'category_id'.

    Returns:
      matches: list[(pred_idx, gt_idx)]
      unmatched_pred_idx: list[int]
      unmatched_gt_idx:   list[int]
    """
    pred_order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    gt_used = set()
    matches = []
    unmatched_pred = []
    for pi in pred_order:
        best_iou = iou_thr
        best_gt = None
        pb = preds[pi]["bbox"]
        for gj in range(len(gts)):
            if gj in gt_used:
                continue
            iou = iou_xywh(pb, gts[gj]["bbox"])
            if iou >= best_iou:
                best_iou = iou
                best_gt = gj
        if best_gt is not None:
            matches.append((pi, best_gt))
            gt_used.add(best_gt)
        else:
            unmatched_pred.append(pi)
    unmatched_gt = [gj for gj in range(len(gts)) if gj not in gt_used]
    return matches, unmatched_pred, unmatched_gt


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    tier_of_name, cost_cfg = load_tier_assignment(Path(args.cost_config))

    ann_data = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    cat_id_to_name = {c["id"]: c["name"] for c in ann_data["categories"]}
    cat_id_to_tier = {
        cid: tier_of_name.get(name) for cid, name in cat_id_to_name.items()
    }

    # Sanity: all GT classes should be classified by the cost config
    unassigned = [cat_id_to_name[cid] for cid in cat_id_to_name
                  if cat_id_to_name[cid] not in tier_of_name]
    if unassigned:
        raise SystemExit(
            f"cost-config does not classify these GT classes: {unassigned}. "
            f"Add them to one of L0/L1/L2/L3/ignored in {args.cost_config}.")

    gts_by_image: dict[int, list] = defaultdict(list)
    for a in ann_data["annotations"]:
        gts_by_image[a["image_id"]].append(a)

    image_ids = [im["id"] for im in ann_data["images"]]

    preds_list = json.loads(Path(args.preds).read_text(encoding="utf-8"))
    preds_by_image: dict[int, list] = defaultdict(list)
    for p in preds_list:
        preds_by_image[p["image_id"]].append(p)

    # ===== M1: image-level binary screening (dual) =====
    pos_set_all = {cid for cid, t in cat_id_to_tier.items() if t in (1, 2, 3)}
    pos_set_highrisk = {cid for cid, t in cat_id_to_tier.items() if t in (2, 3)}

    M1 = {}
    pr_dump = {}
    for label, pos_set in (("all", pos_set_all), ("highrisk", pos_set_highrisk)):
        gt_pos_lst, pred_score_lst = [], []
        for img_id in image_ids:
            gt_pos_lst.append(any(
                a["category_id"] in pos_set for a in gts_by_image[img_id]))
            scores = [p["score"] for p in preds_by_image[img_id]
                      if p["category_id"] in pos_set]
            pred_score_lst.append(max(scores) if scores else 0.0)
        gt_arr = np.array(gt_pos_lst, dtype=int)
        score_arr = np.array(pred_score_lst, dtype=float)
        if 0 < gt_arr.sum() < len(gt_arr):
            M1[f"screening_auroc_{label}"] = auroc_score(gt_arr, score_arr)
            M1[f"screening_auprc_{label}"] = auprc_score(gt_arr, score_arr)
            _, _, _, prec, rec, thr = _roc_pr_curves(gt_arr, score_arr)
            pr_dump[label] = (prec, rec, thr)
        else:
            M1[f"screening_auroc_{label}"] = float("nan")
            M1[f"screening_auprc_{label}"] = float("nan")
            pr_dump[label] = None

    for label, pr in pr_dump.items():
        if pr is None:
            continue
        prec, rec, thr = pr
        with (out_dir / f"pr_screening_{label}.csv").open(
                "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "precision", "recall"])
            for i in range(len(thr)):
                w.writerow([f"{thr[i]:.6f}", f"{prec[i]:.6f}", f"{rec[i]:.6f}"])

    # ===== M3: sensitivity @ specificity targets (use highrisk) =====
    M3 = {}
    pos_set = pos_set_highrisk
    gt_arr = np.array([
        any(a["category_id"] in pos_set for a in gts_by_image[i])
        for i in image_ids], dtype=int)
    score_arr = np.array([
        max([p["score"] for p in preds_by_image[i]
             if p["category_id"] in pos_set] or [0.0])
        for i in image_ids])
    if 0 < gt_arr.sum() < len(gt_arr):
        fpr, tpr, thr_roc, *_ = _roc_pr_curves(gt_arr, score_arr)
        spec = 1.0 - fpr
        for spec_target in (0.90, 0.95, 0.99):
            mask = spec >= spec_target
            if mask.any():
                idx = np.where(mask)[0]
                best = idx[np.argmax(tpr[idx])]
                M3[f"sensitivity_at_spec_{spec_target}"] = float(tpr[best])
                thr_v = thr_roc[best]
                M3[f"threshold_at_spec_{spec_target}"] = (
                    float(thr_v) if np.isfinite(thr_v) else None)
            else:
                M3[f"sensitivity_at_spec_{spec_target}"] = float("nan")
                M3[f"threshold_at_spec_{spec_target}"] = None
    else:
        for spec_target in (0.90, 0.95, 0.99):
            M3[f"sensitivity_at_spec_{spec_target}"] = float("nan")
            M3[f"threshold_at_spec_{spec_target}"] = None

    # ===== M2: cost-weighted error =====
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    total_cost = 0.0
    n_images_counted = 0
    for img_id in image_ids:
        gts_eval = [g for g in gts_by_image[img_id]
                    if cat_id_to_tier.get(g["category_id"]) is not None]
        preds_eval = [p for p in preds_by_image[img_id]
                      if cat_id_to_tier.get(p["category_id"]) is not None]
        matches, unmatched_p, unmatched_g = greedy_match(
            preds_eval, gts_eval, args.iou_threshold)
        img_cost = 0.0
        for pi, gj in matches:
            t_cat = gts_eval[gj]["category_id"]
            p_cat = preds_eval[pi]["category_id"]
            t_name = cat_id_to_name[t_cat]
            p_name = cat_id_to_name[p_cat]
            t_tier = cat_id_to_tier[t_cat]
            p_tier = cat_id_to_tier[p_cat]
            img_cost += cost_value(t_tier, p_tier)
            confusion[(t_name, p_name)] += 1
        for gj in unmatched_g:
            t_cat = gts_eval[gj]["category_id"]
            t_name = cat_id_to_name[t_cat]
            t_tier = cat_id_to_tier[t_cat]
            img_cost += cost_value(t_tier, "NO_PRED")
            confusion[(t_name, "NO_PRED")] += 1
        for pi in unmatched_p:
            p_cat = preds_eval[pi]["category_id"]
            p_name = cat_id_to_name[p_cat]
            p_tier = cat_id_to_tier[p_cat]
            img_cost += cost_value("NO_GT", p_tier)
            confusion[("NO_GT", p_name)] += 1
        total_cost += img_cost
        n_images_counted += 1

    M2 = {
        "cost_weighted_total": float(total_cost),
        "cost_weighted_mean_per_image":
            float(total_cost / n_images_counted)
            if n_images_counted else float("nan"),
        "n_images_counted": int(n_images_counted),
    }

    with (out_dir / "confusion_matrix.csv").open(
            "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_class", "pred_class", "count"])
        for (t, p), n in sorted(confusion.items()):
            w.writerow([t, p, n])

    # ===== M4: top-K box recall =====
    # Sum-based recall over all GT boxes across images (NOT per-image mean):
    # this avoids the inflation that comes from giving empty-GT images a vacuous
    # recall of 1.0, which is a real concern on negative-heavy splits like ours.
    # Empty-GT images are skipped entirely (still counted in n_imgs_skipped for
    # transparency).
    M4_recall = {}
    n_imgs_skipped = 0
    for K in args.top_k_list:
        total_matched = 0
        total_gts = 0
        for img_id in image_ids:
            gts_eval = [g for g in gts_by_image[img_id]
                        if cat_id_to_tier.get(g["category_id"]) is not None]
            if not gts_eval:
                continue
            preds_eval = sorted(
                [p for p in preds_by_image[img_id]
                 if cat_id_to_tier.get(p["category_id"]) is not None],
                key=lambda x: -x["score"])[:K]
            matches, _, _ = greedy_match(preds_eval, gts_eval,
                                         args.iou_threshold)
            total_matched += len(matches)
            total_gts += len(gts_eval)
        M4_recall[str(K)] = (total_matched / total_gts) if total_gts else 0.0
    # Count once how many images have no eval-tier GT
    n_imgs_skipped = sum(
        1 for img_id in image_ids
        if not [g for g in gts_by_image[img_id]
                if cat_id_to_tier.get(g["category_id"]) is not None])
    M4 = {
        "top_k_recall": M4_recall,
        "top_k_recall_doc": ("sum-based recall over all GT boxes; "
                             "empty-GT images skipped (not counted as 1.0)"),
        "n_imgs_with_no_eval_gt_skipped_in_M4": int(n_imgs_skipped),
    }

    metrics = {
        **M1, **M2, **M3, **M4,
        "iou_threshold": args.iou_threshold,
        "n_images": len(image_ids),
        "n_gt_anns": sum(len(v) for v in gts_by_image.values()),
        "n_pred_anns": len(preds_list),
        "cost_config_version": cost_cfg["version"],
        "cost_config_path": str(args.cost_config),
        "ann_path": str(args.ann),
        "preds_path": str(args.preds),
    }
    (out_dir / "clinical_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary markdown
    md = ["# Clinical metrics summary\n"]
    md.append("## Image-level screening (M1)\n")
    md.append("| view        | AUROC  | AUPRC  |")
    md.append("|---          |---:    |---:    |")
    md.append(f"| all-positive | {M1['screening_auroc_all']:.4f} | {M1['screening_auprc_all']:.4f} |")
    md.append(f"| high-risk    | {M1['screening_auroc_highrisk']:.4f} | {M1['screening_auprc_highrisk']:.4f} |")
    md.append("")
    md.append("## Cost-weighted error (M2, lower is better)\n")
    md.append(f"- mean cost per image: **{M2['cost_weighted_mean_per_image']:.3f}**")
    md.append(f"- total cost: {M2['cost_weighted_total']:.1f}")
    md.append(f"- images counted: {M2['n_images_counted']}")
    md.append("")
    md.append("## Sensitivity @ Specificity (M3, high-risk screening)\n")
    md.append("| target spec | sens   | threshold |")
    md.append("|---:         |---:    |---:       |")
    for s in (0.90, 0.95, 0.99):
        sens = M3[f"sensitivity_at_spec_{s}"]
        thr = M3[f"threshold_at_spec_{s}"]
        thr_s = f"{thr:.4f}" if thr is not None else "n/a"
        md.append(f"| {s:.2f}        | {sens:.4f} | {thr_s}    |")
    md.append("")
    md.append("## Top-K box recall (M4)\n")
    md.append("| K  | recall |")
    md.append("|---:|---:    |")
    for k, v in M4_recall.items():
        md.append(f"| {k:>2} | {v:.4f} |")
    md.append("")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
