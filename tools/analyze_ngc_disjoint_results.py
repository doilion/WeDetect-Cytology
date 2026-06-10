#!/usr/bin/env python
"""Per-class analysis CSV for the TCT_NGC dev32 patient-disjoint baseline.

One pass over the train/val/test ann jsons + the val/test classwise eval logs +
the cached text embeddings, producing a single CSV row per category with:
  - ann / image / case counts in each split
  - bbox area median + small-object fraction (val and test)
  - per-class AP (mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l) from val and test logs
  - prompt cosine top-3 nearest neighbours from the cached text embeddings
  - evaluated flag (False for the 7 NEGATIVE_NAMES)
  - provenance_ok flag (False for TCT_CCD-* — path lacks real WSI/case info)

case parsing
------------
Non-TCT_CCD images use the layout
    <organ>/<batch>/<case>/<basename>.jpg
so case == path.parts[2]. TCT_CCD images use a placeholder shard layout
(`TCT_CCD/images/{train30000,val}/<basename>.jpg`) without WSI info — we record
NaN there and flip provenance_ok to False, mirroring the convention documented
in feedback_novel_prompts_pending and project_tct_ccd_no_wsi.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import torch


NEGATIVE_NAMES = {
    "respiratory tract-Impurity",
    "Serous effusion-Negative samples",
    "Thyroid gland-Negative samples",
    "Urine-NILM",
    "Urine-Negative",
    "Urine-Negative Degeneration",
    "TCT_CCD-normal",
}


# matches MMEngine classwise table lines (6 numeric cols)
ROW_RE = re.compile(
    r"^\|\s+(?P<name>[^|]+?)\s+\|"
    r"\s+(?P<map>[0-9.]+|nan)\s+\|"
    r"\s+(?P<map50>[0-9.]+|nan)\s+\|"
    r"\s+(?P<map75>[0-9.]+|nan)\s+\|"
    r"\s+(?P<maps>[0-9.]+|nan|-1\.000)\s+\|"
    r"\s+(?P<mapm>[0-9.]+|nan|-1\.000)\s+\|"
    r"\s+(?P<mapl>[0-9.]+|nan|-1\.000)\s+\|"
)


def to_float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return v


def parse_classwise_log(path: Path) -> dict[str, dict[str, float]]:
    """Returns {class_name: {mAP, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l}}."""
    out: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if name == "category":
            continue
        out[name] = dict(
            mAP=to_float(m.group("map")),
            mAP_50=to_float(m.group("map50")),
            mAP_75=to_float(m.group("map75")),
            mAP_s=to_float(m.group("maps")),
            mAP_m=to_float(m.group("mapm")),
            mAP_l=to_float(m.group("mapl")),
        )
    return out


def is_tct_ccd(class_name: str) -> bool:
    return class_name.startswith("TCT_CCD-")


def organ_of(class_name: str) -> str:
    return class_name.split("-", 1)[0]


def parse_case(file_name: str, class_name: str) -> str | None:
    if is_tct_ccd(class_name):
        return None  # provenance unreliable
    parts = Path(file_name).parts
    return parts[2] if len(parts) >= 3 else None


def split_stats(
    ann_path: Path,
    cat_id_to_name: dict[int, str],
) -> tuple[
    dict[int, int],         # ann_count
    dict[int, set[int]],    # image_ids
    dict[int, set[str]],    # case ids (None-aware: TCT_CCD will have empty set)
    dict[int, list[float]], # bbox areas (val/test only)
]:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    img_to_path = {im["id"]: im["file_name"] for im in data["images"]}

    ann_count: dict[int, int] = defaultdict(int)
    images: dict[int, set[int]] = defaultdict(set)
    cases: dict[int, set[str]] = defaultdict(set)
    areas: dict[int, list[float]] = defaultdict(list)
    for ann in data["annotations"]:
        cid = ann["category_id"]
        ann_count[cid] += 1
        images[cid].add(ann["image_id"])
        cls = cat_id_to_name.get(cid, "")
        case = parse_case(img_to_path.get(ann["image_id"], ""), cls)
        if case is not None:
            cases[cid].add(case)
        areas[cid].append(float(ann["area"]))
    return ann_count, images, cases, areas


def fmt(v: float, digits: int = 3) -> str:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "nan"
    return f"{v:.{digits}f}"


def build_prompt_neighbors(
    text_emb_path: Path,
    text_json_path: Path,
    sorted_cat_ids: list[int],
    cat_id_to_name: dict[int, str],
) -> dict[int, list[tuple[str, float]]]:
    """Returns {cat_id: [(neighbor_class_name, cosine), ...3]}.

    `tct_ngc_fullnames_32.json` lists prompts positionally aligned with the
    ann file's sorted-by-id category list (cat_ids are NOT contiguous —
    they skip 21,22,24..30). So prompt index i ↔ sorted_cat_ids[i].
    """
    emb_dict = torch.load(text_emb_path, map_location="cpu")
    fullnames = json.loads(text_json_path.read_text(encoding="utf-8"))
    if len(fullnames) != len(sorted_cat_ids):
        raise SystemExit(
            f"text-json has {len(fullnames)} prompts but ann has "
            f"{len(sorted_cat_ids)} categories; positional alignment broken"
        )

    vecs = []
    for prompt_group in fullnames:
        prompt = prompt_group[0]
        if prompt not in emb_dict:
            raise SystemExit(f"prompt missing from emb dict: {prompt!r}")
        vecs.append(emb_dict[prompt].float())
    mat = torch.stack(vecs, dim=0)
    mat = mat / mat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    sim = mat @ mat.T  # (N, N)

    out: dict[int, list[tuple[str, float]]] = {}
    n = sim.shape[0]
    for i in range(n):
        s = sim[i].clone()
        s[i] = -float("inf")
        top = s.topk(3)
        cid = sorted_cat_ids[i]
        out[cid] = [
            (cat_id_to_name[sorted_cat_ids[int(j)]], float(v))
            for j, v in zip(top.indices.tolist(), top.values.tolist())
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", required=True)
    parser.add_argument("--val-ann", required=True)
    parser.add_argument("--test-ann", required=True)
    parser.add_argument("--val-log", required=True)
    parser.add_argument("--test-log", required=True)
    parser.add_argument("--text-emb", required=True)
    parser.add_argument("--text-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Use the val ann as the canonical category list (train/test should match).
    val_data = json.loads(Path(args.val_ann).read_text(encoding="utf-8"))
    cats = sorted(val_data["categories"], key=lambda c: c["id"])
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    sorted_cat_ids = [c["id"] for c in cats]

    train_ann, train_img, train_case, _ = split_stats(Path(args.train_ann), cat_id_to_name)
    val_ann, val_img, val_case, val_areas = split_stats(Path(args.val_ann), cat_id_to_name)
    test_ann, test_img, test_case, test_areas = split_stats(Path(args.test_ann), cat_id_to_name)

    val_ap = parse_classwise_log(Path(args.val_log))
    test_ap = parse_classwise_log(Path(args.test_log))

    neighbors = build_prompt_neighbors(
        Path(args.text_emb), Path(args.text_json), sorted_cat_ids, cat_id_to_name
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "class_id", "class_name", "organ", "evaluated", "provenance_ok",
        "train_anns", "train_images", "train_cases",
        "val_anns",   "val_images",   "val_cases",
        "test_anns",  "test_images",  "test_cases",
        "bbox_area_median_val", "p_small_val",
        "bbox_area_median_test", "p_small_test",
        "val_AP", "val_AP50", "val_AP75", "val_AP_s", "val_AP_m", "val_AP_l",
        "test_AP", "test_AP50", "test_AP75", "test_AP_s", "test_AP_m", "test_AP_l",
        "delta_AP",
        "prompt_neighbor_1", "prompt_cos_1",
        "prompt_neighbor_2", "prompt_cos_2",
        "prompt_neighbor_3", "prompt_cos_3",
    ]

    rows = []
    SMALL_AREA = 32 * 32
    for cat in cats:
        cid, name = cat["id"], cat["name"]
        provenance_ok = not is_tct_ccd(name)

        def median_p_small(areas: list[float]) -> tuple[float, float]:
            if not areas:
                return float("nan"), float("nan")
            return statistics.median(areas), sum(1 for a in areas if a < SMALL_AREA) / len(areas)

        med_v, p_s_v = median_p_small(val_areas[cid])
        med_t, p_s_t = median_p_small(test_areas[cid])

        v = val_ap.get(name, {})
        t = test_ap.get(name, {})
        v_map = v.get("mAP", float("nan"))
        t_map = t.get("mAP", float("nan"))
        delta = (v_map - t_map) if (math.isfinite(v_map) and math.isfinite(t_map)) else float("nan")

        nb = neighbors[cid]
        rows.append({
            "class_id": cid,
            "class_name": name,
            "organ": organ_of(name),
            "evaluated": int(name not in NEGATIVE_NAMES),
            "provenance_ok": int(provenance_ok),
            "train_anns": train_ann.get(cid, 0),
            "train_images": len(train_img.get(cid, set())),
            "train_cases": len(train_case.get(cid, set())) if provenance_ok else "nan",
            "val_anns": val_ann.get(cid, 0),
            "val_images": len(val_img.get(cid, set())),
            "val_cases": len(val_case.get(cid, set())) if provenance_ok else "nan",
            "test_anns": test_ann.get(cid, 0),
            "test_images": len(test_img.get(cid, set())),
            "test_cases": len(test_case.get(cid, set())) if provenance_ok else "nan",
            "bbox_area_median_val": fmt(med_v, 0),
            "p_small_val": fmt(p_s_v, 3),
            "bbox_area_median_test": fmt(med_t, 0),
            "p_small_test": fmt(p_s_t, 3),
            "val_AP": fmt(v_map, 3), "val_AP50": fmt(v.get("mAP_50", float("nan")), 3),
            "val_AP75": fmt(v.get("mAP_75", float("nan")), 3),
            "val_AP_s": fmt(v.get("mAP_s", float("nan")), 3),
            "val_AP_m": fmt(v.get("mAP_m", float("nan")), 3),
            "val_AP_l": fmt(v.get("mAP_l", float("nan")), 3),
            "test_AP": fmt(t_map, 3), "test_AP50": fmt(t.get("mAP_50", float("nan")), 3),
            "test_AP75": fmt(t.get("mAP_75", float("nan")), 3),
            "test_AP_s": fmt(t.get("mAP_s", float("nan")), 3),
            "test_AP_m": fmt(t.get("mAP_m", float("nan")), 3),
            "test_AP_l": fmt(t.get("mAP_l", float("nan")), 3),
            "delta_AP": fmt(delta, 3),
            "prompt_neighbor_1": nb[0][0], "prompt_cos_1": fmt(nb[0][1], 3),
            "prompt_neighbor_2": nb[1][0], "prompt_cos_2": fmt(nb[1][1], 3),
            "prompt_neighbor_3": nb[2][0], "prompt_cos_3": fmt(nb[2][1], 3),
        })

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
