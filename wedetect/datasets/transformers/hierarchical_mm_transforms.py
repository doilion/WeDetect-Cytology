"""Hierarchical multi-attribute text transforms for THAF.

Adapted from the YOLO-World-Medical dataset transform implementation.
transformers/mm_transforms.py:315-507. Two changes vs the source:

  - num_attr_types defaults to 5 (we use 5 attr fields, not 4)
  - removed `padding_value` flag — padding to max is handled the same way
    as in the source: replicate `[padding_value] * num_attr_types`

Output schema:
    results['texts'] = List[List[str]]
        outer list: num_sampled_classes
        inner list: num_attr_types attribute strings (in canonical order;
                    matches attr_type_embed[0..4] in the fusion module)

After yolow_collate (wedetect/datasets/utils.py:9), batched texts arrive at
the model as `List[List[List[str]]]` = [batch][num_classes][num_attr_types].
"""
from __future__ import annotations

import json
import random
from typing import Tuple

import numpy as np
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class HierarchicalRandomLoadText:
    """Training-time transform: sample positive + negative classes, emit
    nested list-of-list of attribute strings."""

    def __init__(
        self,
        text_path: str = None,
        num_attr_types: int = 5,
        num_neg_samples: Tuple[int, int] = (80, 80),
        max_num_samples: int = 80,
        padding_to_max: bool = False,
        padding_value: str = "",
    ) -> None:
        self.num_attr_types = num_attr_types
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, "r", encoding="utf-8") as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert "texts" in results or hasattr(self, "class_texts"), (
            "No texts found in results."
        )
        class_texts = results.get("texts", getattr(self, "class_texts", None))
        num_classes = len(class_texts)

        if "gt_labels" in results:
            gt_label_tag = "gt_labels"
        elif "gt_bboxes_labels" in results:
            gt_label_tag = "gt_bboxes_labels"
        else:
            raise ValueError("No valid labels found in results.")

        positive_labels = set(results[gt_label_tag])
        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(
                random.sample(list(positive_labels), k=self.max_num_samples)
            )

        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples),
        )

        candidate_neg_labels = [
            idx for idx in range(num_classes) if idx not in positive_labels
        ]
        negative_labels = random.sample(candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results["gt_bboxes"]), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag]):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label]
        results["gt_bboxes"] = results["gt_bboxes"][gt_valid_mask]
        if "gt_ignore_flags" in results:
            results["gt_ignore_flags"] = results["gt_ignore_flags"][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

        if "instances" in results:
            retaged_instances = []
            for inst in results["instances"]:
                label = inst["bbox_label"]
                if label in label2ids:
                    inst["bbox_label"] = label2ids[label]
                    retaged_instances.append(inst)
            results["instances"] = retaged_instances

        texts: list[list[str]] = []
        for label in sampled_labels:
            cls_attrs = class_texts[label]
            if isinstance(cls_attrs, list) and len(cls_attrs) == self.num_attr_types:
                texts.append(list(cls_attrs))
            elif isinstance(cls_attrs, list) and len(cls_attrs) > 0:
                attrs = list(cls_attrs[: self.num_attr_types])
                while len(attrs) < self.num_attr_types:
                    attrs.append(attrs[-1])
                texts.append(attrs)
            else:
                single = cls_attrs[0] if isinstance(cls_attrs, list) else str(cls_attrs)
                texts.append([single] * self.num_attr_types)

        if self.padding_to_max:
            num_valid = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid
            if num_padding > 0:
                pad = [self.padding_value] * self.num_attr_types
                texts.extend([pad] * num_padding)

        results["texts"] = texts
        return results


@TRANSFORMS.register_module()
class HierarchicalLoadText:
    """Inference-time transform: pass through all classes, emit list-of-list
    of attribute strings (no sampling)."""

    def __init__(
        self,
        text_path: str = None,
        num_attr_types: int = 5,
        padding_value: str = "",
    ) -> None:
        self.num_attr_types = num_attr_types
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, "r", encoding="utf-8") as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert "texts" in results or hasattr(self, "class_texts"), (
            "No texts found in results."
        )
        class_texts = results.get("texts", getattr(self, "class_texts", None))

        texts: list[list[str]] = []
        for cls_attrs in class_texts:
            if isinstance(cls_attrs, list) and len(cls_attrs) == self.num_attr_types:
                texts.append(list(cls_attrs))
            elif isinstance(cls_attrs, list) and len(cls_attrs) > 0:
                attrs = list(cls_attrs[: self.num_attr_types])
                while len(attrs) < self.num_attr_types:
                    attrs.append(attrs[-1])
                texts.append(attrs)
            else:
                single = cls_attrs[0] if isinstance(cls_attrs, list) else str(cls_attrs)
                texts.append([single] * self.num_attr_types)

        results["texts"] = texts
        return results
