# Copyright (c) OpenMMLab. All rights reserved.
import copy
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from contextlib import redirect_stdout
from io import StringIO
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmdet.evaluation import CocoMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval
try:
    from mmdet.datasets.api_wrappers import COCOevalMP
except ImportError:
    COCOevalMP = None
from mmdet.registry import METRICS


def _log_coco_summary(coco_eval, logger) -> None:
    """Capture COCOeval.summarize() stdout and log via MMLogger."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        coco_eval.summarize()
    summary = buffer.getvalue().strip()
    if summary:
        logger.info('\n' + summary)

@METRICS.register_module()
class ExcludeClassCocoMetric(CocoMetric):
    """排除特定类别实例的COCO评估器

    这个评估器可以:
    1. 排除特定类别的实例不参与评估（保留其他类别的实例）
    2. 从结果文件和评估过程中彻底移除该类别

    Args:
        exclude_class_id (int, str, list): 要排除的类别ID或名称，支持单个或多个
            - int: 单个类别内部索引
            - str: 单个类别名称
            - list: 多个类别ID或名称的列表
        replace_with_second_best (bool): 是否用次优类别替换排除类别的预测
        *args, **kwargs: CocoMetric的其他参数
    """

    def __init__(self, exclude_class_id, replace_with_second_best=False, *args, **kwargs):
        # 确保启用classwise以便查看每个类别的指标
        kwargs['classwise'] = True
        super().__init__(*args, **kwargs)

        # 支持单个或多个类别排除
        if isinstance(exclude_class_id, (int, str)):
            self.exclude_class_ids_input = [exclude_class_id]
        else:
            self.exclude_class_ids_input = list(exclude_class_id)

        # 存储解析后的排除信息（内部索引和COCO ID）
        self.exclude_class_ids = []  # 内部索引列表
        self.exclude_class_names = []  # 类别名称列表
        self.exclude_coco_ids = []  # COCO ID列表

        # 兼容旧API
        self.exclude_class_id = None
        self.exclude_class_name = None
        self.exclude_coco_id = None

        self.replace_with_second_best = replace_with_second_best
        self.logger = MMLogger.get_current_instance()

        # 标记是否已经准备好
        self.prepared = False

        # 统计计数器
        self.total_instances = 0
        self.excluded_instances = 0

        # 添加缺失的属性
        self.use_mp_eval = False

        self.logger.info(
            f'创建排除类别评估器，排除类别ID/名称: {self.exclude_class_ids_input}, 替换为次优: {replace_with_second_best}')

    def prepare(self):
        """准备评估，打印必要信息并设置排除类别的信息"""
        # 打印dataset_meta
        self.logger.info("========================= Dataset Meta 信息 =========================")
        if hasattr(self, 'dataset_meta'):
            for key, value in self.dataset_meta.items():
                self.logger.info(f"{key}: {value}")
        else:
            self.logger.warning("没有找到dataset_meta!")

        # 打印COCO类别映射信息
        if self._coco_api is not None:
            self.logger.info("========================= COCO API 类别信息 =========================")
            cats = self._coco_api.cats
            self.logger.info(f"COCO原始类别信息: {cats}")

            # 打印cat_ids映射
            if self.cat_ids is not None:
                self.logger.info(f"内部cat_ids映射: {self.cat_ids}")

                # 打印原始ID和内部索引的对应关系
                self.logger.info("类别ID映射关系:")
                for internal_idx, coco_id in enumerate(self.cat_ids):
                    cat_name = self._coco_api.cats[coco_id]['name']
                    self.logger.info(f"内部索引: {internal_idx}, COCO ID: {coco_id}, 类别名称: {cat_name}")

        # 解析所有要排除的类别
        self.exclude_class_ids = []
        self.exclude_class_names = []
        self.exclude_coco_ids = []

        for exclude_input in self.exclude_class_ids_input:
            if isinstance(exclude_input, str):
                # 字符串类型，通过名称查找
                if hasattr(self, 'dataset_meta') and 'classes' in self.dataset_meta:
                    class_names = self.dataset_meta['classes']
                    try:
                        internal_idx = class_names.index(exclude_input)
                        self.exclude_class_ids.append(internal_idx)
                        self.exclude_class_names.append(exclude_input)

                        if self.cat_ids is not None:
                            coco_id = self.cat_ids[internal_idx]
                            self.exclude_coco_ids.append(coco_id)
                            self.logger.info(
                                f'类别名称 "{exclude_input}" 对应的内部索引为 {internal_idx}, COCO ID为 {coco_id}')
                        else:
                            self.logger.info(f'类别名称 "{exclude_input}" 对应的内部索引为 {internal_idx}')
                    except ValueError:
                        self.logger.warning(f'找不到类别名称 "{exclude_input}"，跳过该类别')
                else:
                    self.logger.warning(f"没有类别名称信息，无法解析类别名称 {exclude_input}")
            else:
                # 整数类型，直接作为内部索引
                internal_idx = exclude_input
                self.exclude_class_ids.append(internal_idx)

                if hasattr(self, 'dataset_meta') and 'classes' in self.dataset_meta:
                    if 0 <= internal_idx < len(self.dataset_meta['classes']):
                        class_name = self.dataset_meta['classes'][internal_idx]
                        self.exclude_class_names.append(class_name)
                        if self.cat_ids is not None:
                            coco_id = self.cat_ids[internal_idx]
                            self.exclude_coco_ids.append(coco_id)
                            self.logger.info(
                                f'内部索引 {internal_idx} 对应类别名称 "{class_name}", COCO ID为 {coco_id}')

        # 兼容旧API（单个类别的情况）
        if len(self.exclude_class_ids) > 0:
            self.exclude_class_id = self.exclude_class_ids[0]
            self.exclude_class_name = self.exclude_class_names[0] if self.exclude_class_names else None
            self.exclude_coco_id = self.exclude_coco_ids[0] if self.exclude_coco_ids else None

        # 标记为已准备
        self.prepared = True

        # 确认最终的排除信息
        self.logger.info(
            f"最终确认的排除类别信息 - 内部索引: {self.exclude_class_ids}, COCO IDs: {self.exclude_coco_ids}, 类别名称: {self.exclude_class_names}")

    def process(self, _data_batch, data_samples):
        """处理一批次的数据和预测结果，排除特定类别的实例"""
        # 如果是第一次调用，首先准备
        if not self.prepared:
            if self.cat_ids is None and self._coco_api is not None:
                self.cat_ids = self._coco_api.get_cat_ids(
                    cat_names=self.dataset_meta['classes'])
            self.prepare()

        # 如果是第一个批次，打印一个数据样本的结构
        if len(self.results) == 0 and data_samples:
            first_sample = data_samples[0]
            self.logger.info("========================= 数据样本结构 =========================")
            self.logger.info(f"数据样本键: {list(first_sample.keys())}")

            # 打印pred_instances的信息
            if 'pred_instances' in first_sample:
                pred = first_sample['pred_instances']
                self.logger.info(f"pred_instances键: {list(pred.keys())}")
                self.logger.info(f"预测标签示例: {pred['labels'][:5] if len(pred['labels']) > 5 else pred['labels']}")

            # 打印instances的信息(Ground Truth)
            if 'instances' in first_sample:
                self.logger.info(
                    f"GT instances信息: {first_sample['instances'][:2] if len(first_sample['instances']) > 2 else first_sample['instances']}")

        if not data_samples:
            return

        # 深拷贝，避免修改原始数据
        data_samples_copy = copy.deepcopy(data_samples)

        # 处理每个图片的数据
        for data_sample in data_samples_copy:
            result = dict()
            result['img_id'] = data_sample['img_id']

            # 处理预测结果
            pred = data_sample['pred_instances']
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            bboxes = pred['bboxes'].cpu().numpy()

            # 找出不是排除类别的预测（支持多个排除类别）
            keep_indices = np.ones(len(labels), dtype=bool)
            for exc_id in self.exclude_class_ids:
                keep_indices &= (labels != exc_id)

            # 过滤预测结果
            labels_filtered = labels[keep_indices]
            scores_filtered = scores[keep_indices]
            bboxes_filtered = bboxes[keep_indices]

            # 记录过滤的实例数
            excluded_count = len(labels) - len(labels_filtered)
            self.excluded_instances += excluded_count
            self.total_instances += len(labels)

            # 更新结果
            result['labels'] = labels_filtered
            result['scores'] = scores_filtered
            result['bboxes'] = bboxes_filtered

            # 如果有masks，也过滤
            if 'masks' in pred:
                masks = pred['masks']
                if isinstance(masks, torch.Tensor):
                    # 简单的mask处理，避免依赖encode_mask_results
                    masks = masks.detach().cpu().numpy()
                result['masks'] = [masks[i] for i in range(len(masks)) if keep_indices[i]]

            # 处理GT信息
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']

            # 如果使用数据集中的标注
            if self._coco_api is None and 'instances' in data_sample:
                instances = data_sample['instances']
                filtered_instances = []

                for instance in instances:
                    # 支持多个排除类别
                    if instance['bbox_label'] not in self.exclude_class_ids:
                        filtered_instances.append(instance)

                gt['anns'] = filtered_instances
            else:
                # 使用外部提供的coco api
                gt['anns'] = data_sample.get('instances', [])

            # 添加到结果列表
            self.results.append((gt, result))

    def results2json(self, results, outfile_prefix):
        """将结果转换为COCO json格式，排除特定类别"""
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None

        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']

            # bbox结果
            for i, label in enumerate(labels):
                # 跳过排除的类别（支持多个）
                if label in self.exclude_class_ids:
                    continue

                # 检查标签是否在有效范围内
                if label >= len(self.cat_ids):
                    self.logger.warning(f"标签 {label} 超出了cat_ids范围 {len(self.cat_ids)}，跳过")
                    continue

                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])

                # 获取正确的COCO类别ID
                # 需要处理标签可能已经被重新映射的情况
                if label >= self.exclude_class_id:
                    # 如果有cat_ids并且标签已经重新映射
                    coco_cat_id = self.cat_ids[label]
                else:
                    coco_cat_id = self.cat_ids[label]

                data['category_id'] = coco_cat_id
                bbox_json_results.append(data)

            # segm结果
            if segm_json_results is None:
                continue

            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                # 跳过排除的类别（支持多个）
                if label in self.exclude_class_ids:
                    continue

                # 检查标签是否在有效范围内
                if label >= len(self.cat_ids):
                    self.logger.warning(f"标签 {label} 超出了cat_ids范围 {len(self.cat_ids)}，跳过")
                    continue

                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])

                # 获取COCO类别ID
                coco_cat_id = self.cat_ids[label]

                data['category_id'] = coco_cat_id
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def compute_metrics(self, results):
        """计算评估指标，排除特定类别的指标"""
        # 确保已准备
        if not self.prepared:
            if self.cat_ids is None and self._coco_api is not None:
                self.cat_ids = self._coco_api.get_cat_ids(
                    cat_names=self.dataset_meta['classes'])
            self.prepare()

        logger = MMLogger.get_current_instance()

        # 根据配置动态生成COCO指标名称到索引的映射
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
        }

        # COCOeval 默认最多统计前三个 maxDets（与self.proposal_nums对应）
        proposal_nums = list(getattr(self, 'proposal_nums', []))
        if len(proposal_nums) == 0:
            proposal_nums = [100, 300, 1000]

        max_det_entries = proposal_nums[:3]
        last_max_det = max_det_entries[-1]

        for idx, max_det in enumerate(max_det_entries):
            coco_metric_names[f'AR@{max_det}'] = 6 + idx

        area_base_index = 6 + len(max_det_entries)
        coco_metric_names[f'AR_s@{last_max_det}'] = area_base_index
        coco_metric_names[f'AR_m@{last_max_det}'] = area_base_index + 1
        coco_metric_names[f'AR_l@{last_max_det}'] = area_base_index + 2

        # 分离gt和预测结果
        gts, preds = zip(*results)

        # 设置临时目录或使用指定的输出前缀
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        # 初始化coco api（如果需要）
        if self._coco_api is None:
            logger.info('正在将ground truth转换为coco格式...')
            coco_json_path = self.gt_to_coco_json(gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # 处理lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # 转换预测为coco格式并保存为json
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info(f'结果已保存到 {osp.dirname(outfile_prefix)}')
            return eval_results

        # 计算各个指标
        for metric in self.metrics:
            logger.info(f'正在评估 {metric}...')

            # 快速评估召回率
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # 评估proposal、bbox和segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} 不在results中')

            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # 移除bbox以避免使用box area而不是mask area
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error('整个数据集的测试结果为空。')
                break

            # 创建评估器
            if self.use_mp_eval and COCOevalMP is not None:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            # 设置参数
            # 排除特定类别ID - 非常重要（支持多个排除类别）
            filtered_cat_ids = [cat_id for cat_id in self.cat_ids if cat_id not in self.exclude_coco_ids]
            coco_eval.params.catIds = filtered_cat_ids

            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # 执行评估
            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                _log_coco_summary(coco_eval, logger)
                if self.metric_items is None:
                    metric_items = [
                        *(f'AR@{max_det}' for max_det in max_det_entries),
                        f'AR_s@{last_max_det}',
                        f'AR_m@{last_max_det}',
                        f'AR_l@{last_max_det}'
                    ]
                else:
                    metric_items = self.metric_items

                for item in metric_items:
                    if item not in coco_metric_names:
                        logger.warning(
                            f'Unknown metric item "{item}" for proposal evaluation, ' \
                            'skip recording its value.')
                        continue
                    val = float(f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                _log_coco_summary(coco_eval, logger)

                # 计算每个类别的AP
                if self.classwise:
                    precisions = coco_eval.eval['precision']

                    # 由于我们排除了类别，需要调整类别对应关系（支持多个排除类别）
                    results_per_category = []
                    cat_ids_no_excluded = [cat_id for cat_id in self.cat_ids if cat_id not in self.exclude_coco_ids]

                    for idx, cat_id in enumerate(cat_ids_no_excluded):
                        # 在COCO API中查找类别名称
                        nm = self._coco_api.loadCats(cat_id)[0]

                        # 获取该类别的precision
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]

                        ap = np.mean(precision) if precision.size else float('nan')
                        t = [f'{nm["name"]}', f'{round(ap, 3)}']
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # IoU@50和IoU@75的AP
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            ap = np.mean(precision) if precision.size else float('nan')
                            t.append(f'{round(ap, 3)}')

                        # 不同尺寸目标的AP
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            ap = np.mean(precision) if precision.size else float('nan')
                            t.append(f'{round(ap, 3)}')

                        results_per_category.append(tuple(t))

                    # 生成表格
                    num_columns = len(results_per_category[0])
                    results_flatten = list(itertools.chain(*results_per_category))
                    headers = [
                        'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                        'mAP_m', 'mAP_l'
                    ]
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                # 记录指标
                if self.metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]
                else:
                    metric_items = self.metric_items

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                # 输出AP值
                ap = coco_eval.stats[:6]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f}')

        # 清理临时目录
        if tmp_dir is not None:
            tmp_dir.cleanup()

        # 输出统计信息
        logger.info(
            f'总实例数: {self.total_instances}, 排除实例数: {self.excluded_instances}, '
            f'参与评估实例数: {self.total_instances - self.excluded_instances}')

        return eval_results

    def evaluate(self, size):
        """执行评估并输出统计"""
        metrics = super().evaluate(size)

        # 输出最终统计
        self.logger.info(
            f'评估完成: 总实例数: {self.total_instances}, 排除实例数: {self.excluded_instances}, '
            f'参与评估实例数: {self.total_instances - self.excluded_instances}')

        # 重置计数
        self.total_instances = 0
        self.excluded_instances = 0

        return metrics
