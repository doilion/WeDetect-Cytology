"""OrganRestrictedCocoMetric: per-organ AP + macro/all-class/instance-weighted aggregation.

Used together with inference-time organ_class_mask (applied in
YOLOWorldHead.predict_by_feat) to evaluate organ-conditional detection.

The metric:
  1) Runs vanilla COCOeval once on the full prediction set → all-class flat mAP.
  2) Per organ, filters COCO params.catIds to that organ's class subset and runs
     COCOeval again → AP_organ.
  3) Reports:
       overall_macro    = mean(AP_organ over organs that have ≥1 class)
       all_class        = standard COCOeval mAP (treats all classes equally)
       instance_weighted = Σ AP_organ * (n_inst_organ / n_inst_total)

The macro is the paper's main number (un-biased by class-imbalance across organs).
The all-class is for sanity (should be close to macro if classes are balanced).
Instance-weighted is for transparency.

Required init args:
    organ_mask_path: path to .pt file produced by tools/build_class_organ_mask.py
                     containing {'class_names', 'organ_names', 'mask'}

Class-name alignment between the metric's organ_mask file and the dataset's
ann_file is enforced — both must list classes in the same order. If they
diverge the metric raises immediately.
"""
import copy
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from contextlib import redirect_stdout
from io import StringIO
from typing import List

import numpy as np
import torch
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS
from mmengine.fileio import dump, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable


def _capture_summarize(coco_eval, logger, label):
    buf = StringIO()
    with redirect_stdout(buf):
        coco_eval.summarize()
    txt = buf.getvalue().strip()
    if txt:
        logger.info(f'\n-- COCO summary [{label}] --\n{txt}')


@METRICS.register_module()
class OrganRestrictedCocoMetric(CocoMetric):
    """COCO metric with per-organ AP breakdown.

    Args:
        organ_mask_path: path to mask .pt (see tools/build_class_organ_mask.py)
        *args, **kwargs: forwarded to CocoMetric
    """

    def __init__(self, organ_mask_path: str,
                 exclude_class_names: list | None = None,
                 *args, **kwargs):
        kwargs.setdefault('classwise', True)
        super().__init__(*args, **kwargs)
        self.organ_mask_path = organ_mask_path
        mask_pkg = torch.load(organ_mask_path, weights_only=False)
        self._mask_class_names: List[str] = mask_pkg['class_names']
        self._mask_class_ids: List[int] = mask_pkg['class_ids']
        self._organ_names: List[str] = mask_pkg['organ_names']
        self._mask: torch.Tensor = mask_pkg['mask']    # [C, O]
        # class_idx → organ_idx (each row of mask must have exactly one 1).
        # Assert so an all-zero row doesn't silently map a class to organ 0.
        row_sums = self._mask.sum(dim=1)
        if not torch.all(row_sums == 1.0):
            raise ValueError(
                f'OrganRestrictedCocoMetric: mask file {organ_mask_path} has '
                f'rows that do not sum to 1 (row_sums unique values: '
                f'{row_sums.unique().tolist()}). Each class must map to '
                f'exactly one organ.')
        self._class_to_organ = self._mask.argmax(dim=1).tolist()
        self._exclude_class_names = list(exclude_class_names or [])
        self._logger = MMLogger.get_current_instance()
        self._logger.info(
            f'[OrganRestrictedCocoMetric] loaded mask {self._mask.shape}: '
            f'{len(self._mask_class_names)} classes × {len(self._organ_names)} organs')
        if self._exclude_class_names:
            self._logger.info(
                f'[OrganRestrictedCocoMetric] excluding from COCOeval: '
                f'{self._exclude_class_names}')

    def _assert_class_alignment(self):
        """Cross-check organ_mask class list against dataset_meta / cat_ids."""
        if hasattr(self, 'dataset_meta') and 'classes' in self.dataset_meta:
            ds_classes = list(self.dataset_meta['classes'])
            if ds_classes != self._mask_class_names:
                raise ValueError(
                    'OrganRestrictedCocoMetric: dataset classes vs mask classes mismatch.\n'
                    f'  dataset[:5]={ds_classes[:5]}\n'
                    f'  mask[:5]={self._mask_class_names[:5]}\n'
                    '  Re-run tools/build_class_organ_mask.py with the same ann file.')
        # Also cross-check COCO category_ids (catches the case where two ann
        # files have identical class *names* but different COCO ids).
        if self.cat_ids is not None and self._mask_class_ids:
            if list(self.cat_ids) != list(self._mask_class_ids):
                raise ValueError(
                    'OrganRestrictedCocoMetric: COCO cat_ids vs mask class_ids mismatch.\n'
                    f'  cat_ids[:5]={list(self.cat_ids)[:5]}\n'
                    f'  mask_ids[:5]={self._mask_class_ids[:5]}\n'
                    '  Regenerate the mask against this ann file.')

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        self._assert_class_alignment()

        gts, preds = zip(*results) if results else ([], [])
        if not preds:
            logger.error('OrganRestrictedCocoMetric: empty predictions — '
                         'reporting all zeros (model produced no detections).')
            zero = OrderedDict()
            for k in ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'):
                zero[f'all_class/{k}'] = 0.0
            zero['overall/macro_mAP'] = 0.0
            zero['overall/all_class_mAP'] = 0.0
            zero['overall/instance_weighted_mAP'] = 0.0
            return zero

        # Standard scaffolding (mirrors CocoMetric.compute_metrics top)
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            logger.info('Building COCO API from ground truth dicts')
            coco_json = self.gt_to_coco_json(gt_dicts=list(gts), outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json)

        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # Invariant: cat_ids must align 1:1 with dataset_meta['classes']
        # (CocoMetric guarantees this since cat_ids is derived from class
        # names, but make it explicit so that downstream `self.cat_ids[i]`
        # indexing remains correct under future refactors).
        if len(self.cat_ids) != len(self.dataset_meta.get('classes', [])):
            raise RuntimeError(
                f'cat_ids length {len(self.cat_ids)} != dataset classes '
                f'length {len(self.dataset_meta.get("classes", []))}; '
                f'OrganRestrictedCocoMetric assumes 1:1 alignment.')

        # Resolve excluded class names → cat_ids
        excluded_cat_ids = set()
        ds_classes = list(self.dataset_meta.get('classes', []))
        for nm in self._exclude_class_names:
            try:
                cls_idx = ds_classes.index(nm)
            except ValueError:
                logger.warning(
                    f'[OrganRestrictedCocoMetric] exclude name {nm!r} not in dataset classes')
                continue
            excluded_cat_ids.add(self.cat_ids[cls_idx])
        kept_cat_ids = [c for c in self.cat_ids if c not in excluded_cat_ids]
        # cls_idx of kept (for classwise table) — same dataset-class order, just filtered
        kept_cls_idx = [i for i, c in enumerate(self.cat_ids) if c not in excluded_cat_ids]

        # Persist prediction file (same as CocoMetric.results2json path)
        result_files = self.results2json(list(preds), outfile_prefix)
        bbox_json = result_files['bbox']
        predictions = load(bbox_json)
        if not predictions:
            logger.error('OrganRestrictedCocoMetric: no predictions to score')
            if tmp_dir:
                tmp_dir.cleanup()
            return OrderedDict()
        coco_dt = self._coco_api.loadRes(predictions)

        eval_results: 'OrderedDict[str, float]' = OrderedDict()

        # --- (1) All-class flat ---
        flat = COCOeval(self._coco_api, coco_dt, 'bbox')
        flat.params.catIds = kept_cat_ids
        flat.params.imgIds = self.img_ids
        flat.params.maxDets = list(self.proposal_nums)
        flat.params.iouThrs = self.iou_thrs
        flat.evaluate(); flat.accumulate()
        _capture_summarize(flat, logger, 'all-class')

        for k, idx in [('mAP', 0), ('mAP_50', 1), ('mAP_75', 2),
                       ('mAP_s', 3), ('mAP_m', 4), ('mAP_l', 5)]:
            eval_results[f'all_class/{k}'] = float(f'{flat.stats[idx]:.4f}')

        # --- (2) Per-organ AP ---
        # Group kept cat_ids by organ.
        organ_to_catids = defaultdict(list)
        for cls_idx in kept_cls_idx:
            organ = self._class_to_organ[cls_idx]
            organ_to_catids[organ].append(self.cat_ids[cls_idx])

        organ_results = []                              # [(organ_name, n_cls, n_inst, mAP, AP50, AP75)]
        organ_stats: list[float] = []                   # macro inputs
        organ_weights: list[float] = []                 # instance-weighted inputs

        # Count GT instance per organ (over kept categories only)
        gt_inst_per_organ = defaultdict(int)
        kept_cat_id_set = set(kept_cat_ids)
        for ann_id in self._coco_api.getAnnIds():
            ann = self._coco_api.anns[ann_id]
            if ann['category_id'] not in kept_cat_id_set:
                continue
            try:
                cls_idx = self.cat_ids.index(ann['category_id'])
            except ValueError:
                continue
            gt_inst_per_organ[self._class_to_organ[cls_idx]] += 1

        for organ_idx, organ_name in enumerate(self._organ_names):
            cats = organ_to_catids.get(organ_idx, [])
            if not cats:
                continue                                # organ has no classes in this dataset
            n_inst = int(gt_inst_per_organ.get(organ_idx, 0))
            if n_inst == 0:
                # Organ has classes but no GT instances in this test set.
                # Report for transparency but exclude from macro / inst-weighted
                # (COCOeval would return -1 / NaN here and pollute aggregates).
                logger.info(f'[organ={organ_name}] 0 GT instances — skipping AP, '
                            f'not included in macro/inst-weighted aggregates')
                organ_results.append((organ_name, len(cats), 0,
                                      float('nan'), float('nan'), float('nan')))
                continue
            sub = COCOeval(self._coco_api, coco_dt, 'bbox')
            sub.params.catIds = cats
            sub.params.imgIds = self.img_ids
            sub.params.maxDets = list(self.proposal_nums)
            sub.params.iouThrs = self.iou_thrs
            sub.evaluate(); sub.accumulate()
            _capture_summarize(sub, logger, f'organ={organ_name}')
            ap = float(sub.stats[0])
            ap50 = float(sub.stats[1])
            ap75 = float(sub.stats[2])
            organ_results.append((organ_name, len(cats), n_inst, ap, ap50, ap75))
            organ_stats.append(ap)
            organ_weights.append(n_inst)

            slug = organ_name.replace(' ', '_').replace('-', '_')
            eval_results[f'organ/{slug}/mAP'] = round(ap, 4)
            eval_results[f'organ/{slug}/mAP_50'] = round(ap50, 4)
            eval_results[f'organ/{slug}/mAP_75'] = round(ap75, 4)
            eval_results[f'organ/{slug}/n_instances'] = n_inst
            eval_results[f'organ/{slug}/n_classes'] = len(cats)

        # --- (3) Aggregates ---
        if organ_stats:
            overall_macro = float(np.mean(organ_stats))
        else:
            overall_macro = float('nan')

        total_inst = sum(organ_weights)
        if total_inst > 0:
            inst_w = float(sum(a * w for a, w in zip(organ_stats, organ_weights)) / total_inst)
        else:
            inst_w = float('nan')

        eval_results['overall/macro_mAP'] = round(overall_macro, 4)
        eval_results['overall/all_class_mAP'] = eval_results['all_class/mAP']
        eval_results['overall/instance_weighted_mAP'] = round(inst_w, 4)

        # --- Pretty table ---
        table_data = [['domain', 'Cs', 'instances', 'mAP', 'AP50', 'AP75']]
        for name, ncls, ninst, ap, ap50, ap75 in organ_results:
            table_data.append([name, str(ncls), str(ninst),
                               f'{ap:.4f}', f'{ap50:.4f}', f'{ap75:.4f}'])
        table_data.append(['-'] * 6)
        table_data.append(['overall macro', '-', str(total_inst),
                           f'{overall_macro:.4f}', '-', '-'])
        table_data.append(['all-class flat', str(len(kept_cat_ids)), str(total_inst),
                           f'{eval_results["all_class/mAP"]:.4f}',
                           f'{eval_results["all_class/mAP_50"]:.4f}',
                           f'{eval_results["all_class/mAP_75"]:.4f}'])
        table_data.append(['instance-weighted', '-', str(total_inst),
                           f'{inst_w:.4f}', '-', '-'])
        logger.info('\n[OrganRestrictedCocoMetric]\n' + AsciiTable(table_data).table)

        # Also dump per-class AP (sanity). 'precisions' indexes by position
        # of each cat_id in flat.params.catIds (= kept_cat_ids).
        if self.classwise:
            precisions = flat.eval['precision']         # [T, R, K, A, M] with K=len(kept)
            # Robust IoU-threshold lookup — don't hardcode index 0 / 5 since
            # users may override self.iou_thrs. Falls back to None if absent.
            iou_thrs = np.asarray(self.iou_thrs)
            i50 = int(np.where(np.isclose(iou_thrs, 0.5))[0][0]) \
                if np.any(np.isclose(iou_thrs, 0.5)) else None
            i75 = int(np.where(np.isclose(iou_thrs, 0.75))[0][0]) \
                if np.any(np.isclose(iou_thrs, 0.75)) else None
            rows = [['class', 'organ', 'mAP', 'AP50', 'AP75']]
            for k_idx, cat_id in enumerate(kept_cat_ids):
                cls_idx = self.cat_ids.index(cat_id)
                name = self._coco_api.loadCats(cat_id)[0]['name']
                organ = self._organ_names[self._class_to_organ[cls_idx]]
                p = precisions[:, :, k_idx, 0, -1]
                p = p[p > -1]
                ap = float(np.mean(p)) if p.size else float('nan')
                if i50 is not None:
                    p50 = precisions[i50, :, k_idx, 0, -1]
                    p50 = p50[p50 > -1]
                    ap50 = float(np.mean(p50)) if p50.size else float('nan')
                else:
                    ap50 = float('nan')
                if i75 is not None:
                    p75 = precisions[i75, :, k_idx, 0, -1]
                    p75 = p75[p75 > -1]
                    ap75 = float(np.mean(p75)) if p75.size else float('nan')
                else:
                    ap75 = float('nan')
                rows.append([name, organ, f'{ap:.4f}', f'{ap50:.4f}', f'{ap75:.4f}'])
                slug = name.replace(' ', '_').replace('-', '_')
                eval_results[f'class/{slug}/mAP'] = round(ap, 4)
            logger.info('\n[Per-class AP]\n' + AsciiTable(rows).table)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results
