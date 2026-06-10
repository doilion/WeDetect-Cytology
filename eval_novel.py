#!/usr/bin/env python
"""Manual evaluation script for novel classes with proper label mapping."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

register_all_modules()

# Novel label indices (20-30) map to novel test set category IDs (1-11)
NOVEL_LABEL_TO_CAT_ID = {
    20: 1,   # hsil_scc -> hsil_scc_omn
    21: 2,   # monilia
    22: 3,   # Serous effusion-Ovarian cancer
    23: 4,   # Serous effusion-adenocarcinoma
    24: 5,   # Thyroid gland-Suspicious papillary cancer
    25: 6,   # Thyroid gland-AUC
    26: 7,   # Thyroid gland-Malignant tumour
    27: 8,   # Thyroid gland-NS
    28: 9,   # Urine-HGUC
    29: 10,  # respiratory tract-Squamous cell cinoma
    30: 11,  # respiratory tract-Small cell carcinoma
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate novel classes with proper label mapping')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument(
        '--text',
        required=True,
        help='class text file (31 classes) used for reparameterize')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/test_novel',
        help='output dir for predictions')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='device for inference, e.g. cuda:0')
    parser.add_argument(
        '--ann-file',
        default=None,
        help='novel test annotation file (default: data_root/annotations/test_novel_v2.json)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    ann_file = args.ann_file or str(
        Path(cfg.data_root) / 'annotations' / 'test_novel_v2.json')

    print('Loading model...')
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)

    with open(args.text, 'r') as f:
        texts = json.load(f)
    # add background class with same number of synonyms
    num_synonyms = len(texts[0]) if texts else 1
    texts = texts + [[' '] * num_synonyms]

    print('Reparameterizing model with 31 classes...')
    model.reparameterize(texts)

    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()

    print(f'Total novel test images: {len(img_ids)}')
    print(f'Novel categories: {coco_gt.cats}')

    test_pipeline = Compose(cfg.test_pipeline)

    results = []
    print('Running inference...')
    for img_id in tqdm(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = str(Path(cfg.data_root) / img_info['file_name'])

        data_info = dict(img_id=img_id, img_path=img_path, texts=texts)
        data_info = test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info['inputs'].unsqueeze(0).to(args.device),
            data_samples=[data_info['data_samples']]
        )

        with torch.no_grad():
            output = model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        scores = pred_instances.scores.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()

        for i in range(len(labels)):
            label = int(labels[i])
            if label in NOVEL_LABEL_TO_CAT_ID:
                x1, y1, x2, y2 = bboxes[i]
                w, h = x2 - x1, y2 - y1

                results.append({
                    'image_id': img_id,
                    'category_id': NOVEL_LABEL_TO_CAT_ID[label],
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(scores[i])
                })

    print(f'Total predictions for novel classes: {len(results)}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / 'novel_predictions.json'
    with open(pred_file, 'w') as f:
        json.dump(results, f)

    print('\nRunning COCO evaluation...')
    coco_dt = coco_gt.loadRes(str(pred_file))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print('\n' + '=' * 60)
    print('Per-class AP for Novel Classes:')
    print('=' * 60)

    precisions = coco_eval.eval['precision']
    cat_ids = coco_gt.getCatIds()

    for idx, cat_id in enumerate(cat_ids):
        cat_name = coco_gt.cats[cat_id]['name']
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = float(np.mean(precision)) if precision.size else 0.0

        precision_50 = precisions[0, :, idx, 0, -1]
        precision_50 = precision_50[precision_50 > -1]
        ap_50 = float(np.mean(precision_50)) if precision_50.size else 0.0

        ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        num_anns = len(ann_ids)

        print(
            f'{cat_name:45s}  AP: {ap:.4f}  AP50: {ap_50:.4f}  '
            f'(samples: {num_anns})')

    print('=' * 60)


if __name__ == '__main__':
    main()
