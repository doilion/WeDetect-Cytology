# Copyright (c) Tencent Inc. All rights reserved.
from typing import Sequence

import torch
from mmengine.dataset import COLLATE_FUNCTIONS


@COLLATE_FUNCTIONS.register_module()
def yolow_collate(data_batch: Sequence,
                  use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    batch_bboxes_labels = []
    batch_masks = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks.to(
                dtype=torch.bool, device=gt_bboxes.device)
            batch_masks.append(masks)
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0)
        }
    }
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)

    if hasattr(data_batch[0]['data_samples'], 'texts'):
        batch_texts = [meta['data_samples'].texts for meta in data_batch]
        collated_results['data_samples']['texts'] = batch_texts

    if hasattr(data_batch[0]['data_samples'], 'is_detection'):
        # detection flag
        batch_detection = [meta['data_samples'].is_detection
                           for meta in data_batch]
        collated_results['data_samples']['is_detection'] = torch.tensor(
            batch_detection)

    # OC-HMTA Module 1: forward per-sample organ_id / organ_name through the
    # training-time fast-path. Without this, YOLOWDetDataPreprocessor below
    # rebuilds img_metas from scratch and the OrganExtractor's metainfo is lost.
    # All-or-none across the batch: if any sample carries organ_id, every
    # sample must (otherwise the head-side mask logic would silently use a
    # default for the missing ones).
    organ_ids_per_sample = [
        meta['data_samples'].metainfo.get('organ_id', None)
        for meta in data_batch
    ]
    n_with = sum(1 for o in organ_ids_per_sample if o is not None)
    if 0 < n_with < len(organ_ids_per_sample):
        raise RuntimeError(
            f'organ_id present on {n_with}/{len(data_batch)} samples in this '
            f'batch — partial coverage is not supported. Make sure every '
            f'dataset in the dataloader injects OrganExtractor or none does.')
    if n_with == len(organ_ids_per_sample):
        collated_results['data_samples']['organ_id'] = organ_ids_per_sample
        collated_results['data_samples']['organ_name'] = [
            meta['data_samples'].metainfo.get('organ_name', '')
            for meta in data_batch
        ]

    return collated_results
