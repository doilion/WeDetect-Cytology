#!/usr/bin/env python
"""
实验总结脚本 - 自动评估 Base 和 Novel 类并生成 Markdown 报告

用法:
    python eval_summary.py --exp_name m1_biomedclip --checkpoint work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu/best_*.pth

    或者指定配置文件:
    python eval_summary.py --config config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py --checkpoint work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu/best_*.pth

注意:本脚本的负样本/类布局逻辑源自旧 32 类 dev32 线;dev30 30 类评测请优先用
    test_exclude_negative.py / test_novel.py / tools/evaluate_ngc_filtered.py。
"""

import argparse
import os
import sys
import json
import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.utils import register_all_modules
register_all_modules()

from mmengine.config import Config
from mmdet.apis import init_detector
from mmengine.dataset import Compose
from mmengine.runner import Runner


# 负样本类别 — SINGLE SOURCE: data/texts/tct_ngc_dev30_negatives.json.
# (Was the stale dev32 layout with Urine-NILM/Negative/Negative Degeneration, which
#  match nothing in dev30 and silently leaked Urine-NHGUC + Thyroid gland-NS.)
from wedetect.utils import load_dev30_negative_classes
from wedetect.utils.data_paths import as_posix_dir, get_tct_ngc_root

# 配置 — DATA_ROOT 可通过环境变量覆盖，默认 repo-local data/TCT_NGC。
DATA_ROOT = as_posix_dir(get_tct_ngc_root())
BASE_ANN_FILE = DATA_ROOT + "annotations/test_base_v2.json"
NOVEL_ANN_FILE = DATA_ROOT + "annotations/test_novel_v2.json"
NEGATIVE_CLASS_NAMES = load_dev30_negative_classes()


def evaluate_base_classes(cfg, checkpoint, exclude_negative=True):
    """评估 Base 类"""
    print("\n" + "=" * 60)
    print("评估 Base 类 (20类)")
    print("=" * 60)

    cfg = cfg.copy()
    cfg.load_from = checkpoint

    if exclude_negative:
        cfg.val_evaluator = dict(
            type='ExcludeClassCocoMetric',
            ann_file=BASE_ANN_FILE,
            metric='bbox',
            exclude_class_id=NEGATIVE_CLASS_NAMES,
            classwise=True,
        )
        cfg.test_evaluator = cfg.val_evaluator

    cfg.work_dir = './work_dirs/eval_temp'
    os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    return metrics


def evaluate_novel_classes(cfg, checkpoint):
    """评估 Novel 类 (零样本) — 当前对 dev32 32 类不可用，见审计 P1/P2。

    NOVEL_LABEL_TO_CAT_ID 与 32 类索引语义已脱钩：原来 idx 20..30 对应 11 个 novel
    类，现在 32 类下 20..30 是 Urine-Negative Degeneration / Urine-HGUC / TCT_CCD-*。
    要恢复 novel eval：(a) 提供 tct_ngc_novel_11_texts.json 严格 11 prompt，
    (b) 重写 NOVEL_LABEL_TO_CAT_ID 映射到 11 类。
    """
    raise NotImplementedError(
        "evaluate_novel_classes 与 dev32 32 类布局不兼容，会输出错误的 novel AP。"
        "见 docs/tct_ngc_dev32_fullnames_baseline_report_20260506.md 与审计 P2。"
    )

    # 加载模型
    model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')

    # 加载当前 32 类文本
    with open('data/texts/tct_ngc_fullnames_32.json', 'r') as f:
        texts = json.load(f)
    texts = [[t[0]] for t in texts] + [[' ']]

    # Reparameterize
    model.reparameterize(texts)

    # 加载 novel 测试集
    coco_gt = COCO(NOVEL_ANN_FILE)
    img_ids = coco_gt.getImgIds()

    # 构建 pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    # 推理
    results = []
    print(f"推理 {len(img_ids)} 张图片...")

    for img_id in tqdm(img_ids, desc="Novel推理"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = f"{DATA_ROOT}{img_info['file_name']}"

        data_info = dict(img_id=img_id, img_path=img_path, texts=texts)
        data_info = test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info['inputs'].unsqueeze(0).cuda(),
            data_samples=[data_info['data_samples']]
        )

        with torch.no_grad():
            output = model.test_step(data_batch)[0]
            pred = output.pred_instances

        scores = pred.scores.cpu().numpy()
        bboxes = pred.bboxes.cpu().numpy()
        labels = pred.labels.cpu().numpy()

        for i in range(len(labels)):
            label = int(labels[i])
            if label in NOVEL_LABEL_TO_CAT_ID:
                x1, y1, x2, y2 = bboxes[i]
                results.append({
                    'image_id': img_id,
                    'category_id': NOVEL_LABEL_TO_CAT_ID[label],
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(scores[i])
                })

    # COCO 评估
    os.makedirs('work_dirs/eval_temp', exist_ok=True)
    pred_file = 'work_dirs/eval_temp/novel_pred.json'
    with open(pred_file, 'w') as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 提取每类 AP
    per_class_ap = {}
    precisions = coco_eval.eval['precision']
    cat_ids = coco_gt.getCatIds()

    for idx, cat_id in enumerate(cat_ids):
        cat_name = coco_gt.cats[cat_id]['name']
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else 0.0

        precision_50 = precisions[0, :, idx, 0, -1]
        precision_50 = precision_50[precision_50 > -1]
        ap_50 = np.mean(precision_50) if precision_50.size else 0.0

        ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        per_class_ap[cat_name] = {
            'ap': ap, 'ap_50': ap_50, 'samples': len(ann_ids)
        }

    return {
        'mAP': coco_eval.stats[0],
        'mAP_50': coco_eval.stats[1],
        'mAP_75': coco_eval.stats[2],
        'per_class': per_class_ap
    }


def generate_report(exp_name, checkpoint, base_metrics, novel_metrics, output_dir):
    """生成 Markdown 报告"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"summary_{exp_name}_{timestamp}.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 实验总结: {exp_name}\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Checkpoint**: `{checkpoint}`\n\n")

        f.write("---\n\n")

        # 整体结果
        f.write("## 1. 整体性能\n\n")
        f.write("| 评估集 | mAP | mAP_50 | mAP_75 |\n")
        f.write("|--------|-----|--------|--------|\n")

        base_map = base_metrics.get('coco/bbox_mAP', 0)
        base_map50 = base_metrics.get('coco/bbox_mAP_50', 0)
        base_map75 = base_metrics.get('coco/bbox_mAP_75', 0)
        f.write(f"| Base (排除负样本) | {base_map:.3f} | {base_map50:.3f} | {base_map75:.3f} |\n")

        f.write(f"| Novel (零样本) | {novel_metrics['mAP']:.3f} | {novel_metrics['mAP_50']:.3f} | {novel_metrics['mAP_75']:.3f} |\n")

        f.write("\n---\n\n")

        # Novel 类详细结果
        f.write("## 2. Novel 类详细结果\n\n")
        f.write("| 类别 | AP | AP_50 | 样本数 |\n")
        f.write("|------|-----|-------|--------|\n")

        for cat_name, metrics in sorted(novel_metrics['per_class'].items(),
                                        key=lambda x: x[1]['ap'], reverse=True):
            f.write(f"| {cat_name} | {metrics['ap']:.4f} | {metrics['ap_50']:.4f} | {metrics['samples']} |\n")

        f.write("\n---\n\n")

        # 关键发现
        f.write("## 3. 关键发现\n\n")

        # 找出最好和最差的类
        novel_sorted = sorted(novel_metrics['per_class'].items(),
                             key=lambda x: x[1]['ap'], reverse=True)

        best_novel = novel_sorted[0] if novel_sorted else None
        worst_novel = [x for x in novel_sorted if x[1]['ap'] == 0]

        if best_novel:
            f.write(f"- **最佳 Novel 类**: {best_novel[0]} (AP: {best_novel[1]['ap']:.4f})\n")

        if worst_novel:
            f.write(f"- **失败类 (AP=0)**: {', '.join([x[0] for x in worst_novel])}\n")

        f.write(f"- **Base mAP**: {base_map:.3f}\n")
        f.write(f"- **Novel mAP**: {novel_metrics['mAP']:.3f}\n")

        f.write("\n---\n\n")
        f.write(f"*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"\n报告已保存至: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='实验总结脚本')
    parser.add_argument('--exp_name', default='experiment', help='实验名称')
    parser.add_argument(
        '--config',
        default='config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py',
        help='配置文件',
    )
    parser.add_argument('--checkpoint', required=True, help='模型检查点路径 (支持通配符)')
    parser.add_argument('--output_dir', default='work_dirs/summaries', help='报告输出目录')
    parser.add_argument('--skip_base', action='store_true', help='跳过 Base 类评估')
    parser.add_argument('--skip_novel', action='store_true', help='跳过 Novel 类评估')

    args = parser.parse_args()

    # 处理通配符
    checkpoints = glob.glob(args.checkpoint)
    if not checkpoints:
        print(f"错误: 找不到 checkpoint: {args.checkpoint}")
        sys.exit(1)
    checkpoint = sorted(checkpoints)[-1]  # 取最新的
    print(f"使用 checkpoint: {checkpoint}")

    # 加载配置
    cfg = Config.fromfile(args.config)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 评估 Base 类
    base_metrics = {}
    if not args.skip_base:
        base_metrics = evaluate_base_classes(cfg, checkpoint)

    # 评估 Novel 类
    novel_metrics = {'mAP': 0, 'mAP_50': 0, 'mAP_75': 0, 'per_class': {}}
    if not args.skip_novel:
        novel_metrics = evaluate_novel_classes(cfg, checkpoint)

    # 生成报告
    report_path = generate_report(
        args.exp_name, checkpoint, base_metrics, novel_metrics, args.output_dir
    )

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    print(f"Base mAP (排除负样本): {base_metrics.get('coco/bbox_mAP', 0):.3f}")
    print(f"Novel mAP (零样本): {novel_metrics['mAP']:.3f}")
    print(f"报告: {report_path}")


if __name__ == '__main__':
    main()
