# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from .yolo_detector import YOLODetector
from mmdet.registry import MODELS


@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 ordinal_loss=None,
                 corr_loss=None,
                 region_gap_distill_loss=None,
                 relational_distill_loss=None,
                 text_decone=None,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)
        # Innovation ② (locked 2026-06-03): organ-modulated region-gap distillation
        # from a frozen EXTERNAL BiomedCLIP image teacher. opt-in; None (default) ->
        # disabled, detector behaves exactly as before (regression-safe). Backbone-
        # agnostic (teacher is external), so it works on ConvNeXt and BiomedCLIP-ViT.
        if region_gap_distill_loss is not None:
            from mmdet.registry import MODELS
            self.region_gap_distill_loss = MODELS.build(region_gap_distill_loss)
            self.region_gap_distill_loss.build_and_set_teacher()
        else:
            self.region_gap_distill_loss = None
        # Innovation D (2026-06-05): image->text relational distillation. opt-in;
        # None (default) -> disabled. The learnable text adapter lives in the text
        # backbone (PseudoLanguageBackbone.text_adapter); this loss is param-free and
        # supervises that adapter from an EMA bank of image region prototypes.
        if relational_distill_loss is not None:
            from mmdet.registry import MODELS
            self.relational_distill_loss = MODELS.build(relational_distill_loss)
        else:
            self.relational_distill_loss = None
        # D-v2 (2026-06-08): a swappable TEXT DE-COLLAPSE transform applied to txt_feats
        # before the head. Both the learnable PrototypeGroundedTextAdapter (text-as-query
        # cross-attention to an EMA image-prototype bank) and a fixed whitening de-cone
        # plug in here via the same hook (_decone_text). None (default) -> txt_feats
        # passes through unchanged (regression-safe). The transform that carries an
        # update_bank() gets its bank EMA-updated from this batch's GT regions.
        if text_decone is not None:
            from mmdet.registry import MODELS
            self.text_decone = MODELS.build(text_decone)
        else:
            self.text_decone = None
        # OC-HMTA Module 2: optional within-organ ordinal aux loss.
        # Config: ordinal_loss=dict(type='OrganOrdinalLoss', ...).
        # Detector reads adapted text emb from extract_feat output and feeds
        # to ordinal_loss inside loss(). None = M1 / row 3 / row 3.5 path.
        if ordinal_loss is not None:
            from mmdet.registry import MODELS
            self.ordinal_loss = MODELS.build(ordinal_loss)
        else:
            self.ordinal_loss = None

        # STEGO-corr: optional text-only correspondence distillation on full39.
        # Config: corr_loss=dict(type='TextCorrespondenceLoss', ...).
        # Anchors the ICF text-only forward to BiomedCLIP pairwise structure
        # over base+novel; novel positions get gradient every iter.
        if corr_loss is not None:
            from mmdet.registry import MODELS
            self.corr_loss = MODELS.build(corr_loss)
            fusion = getattr(self.backbone, 'cross_modal_fusion', None)
            if fusion is None:
                raise RuntimeError(
                    'corr_loss requires backbone.cross_modal_fusion '
                    '(ImageConditionalFusion) — none found.')
            self.corr_loss.set_fusion(fusion)
        else:
            self.corr_loss = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        # D-v2: de-collapse the text (+ EMA-update the prototype bank from GT regions)
        # BEFORE the head, so the cls loss trains the de-cone with an absolute-alignment
        # anchor (fixes D-v1's gauge-freedom). No-op if text_decone is None.
        txt_feats = self._decone_text(txt_feats, img_feats, batch_inputs,
                                      batch_data_samples, update_bank=True)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)

        # OC-HMTA Module 2: pull auxiliary losses from text-side adapter.
        # PseudoMultiAttrLanguageBackbone exposes get_aux_losses() (anti-
        # collapse regularizers) and via self.ordinal_loss exposes the
        # within-organ ordinal aux. Both keyed independently in the loss dict.
        text_model = self.backbone.text_model
        adapter = getattr(text_model, 'adapter', None)
        if adapter is not None and hasattr(adapter, 'get_aux_losses'):
            losses.update(adapter.get_aux_losses())
        if hasattr(self, 'ordinal_loss') and self.ordinal_loss is not None:
            adapted_emb = txt_feats  # [B, C, D] post-adapter
            ord_losses = self.ordinal_loss(
                adapted_emb,
                text_model.class_organ_ids,
                text_model.class_axis_ids,
                text_model.class_ranks,
            )
            losses.update(ord_losses)

        # STEGO-corr: text-only forward over full39 teacher embeddings,
        # independent of the current batch's image/text inputs.
        if getattr(self, 'corr_loss', None) is not None:
            losses.update(self.corr_loss())

        # Innovation ②: organ-modulated region-gap distillation from frozen BiomedCLIP.
        if getattr(self, 'region_gap_distill_loss', None) is not None:
            from mmengine.structures import InstanceData
            # YOLO fast-train path: batch_data_samples is the SAME dict the head
            # consumes (NOT a SampleList). bboxes_labels is a flat tensor
            # [N,6] = (img_idx, cls, x1,y1,x2,y2); img_metas is a list of dicts.
            # (unpack_gt_instances would iterate the dict's string keys -> crash.)
            bl = batch_data_samples['bboxes_labels']
            img_metas = batch_data_samples['img_metas']
            B = batch_inputs.shape[0]
            batch_gt_instances = [
                InstanceData(bboxes=bl[bl[:, 0] == i, 2:6]) for i in range(B)]
            organ_ids = [m.get('organ_id', 0) for m in img_metas]
            dp = self.data_preprocessor
            losses.update(self.region_gap_distill_loss.compute(
                img_feats, batch_inputs, batch_gt_instances, organ_ids,
                dp.mean, dp.std))

        # Innovation D: image->text relational distillation. Needs per-GT-box
        # (bbox, class) to build per-class image prototypes; txt_feats (= adapted
        # text) is the student. Same flat bboxes_labels parsing as ②, plus labels.
        if getattr(self, 'relational_distill_loss', None) is not None:
            from mmengine.structures import InstanceData
            bl = batch_data_samples['bboxes_labels']
            B = batch_inputs.shape[0]
            batch_gt_instances = [
                InstanceData(bboxes=bl[bl[:, 0] == i, 2:6],
                             labels=bl[bl[:, 0] == i, 1].long())
                for i in range(B)]
            # Pick the teacher feature map the loss should RoIAlign. The
            # teacher-space analysis (2026-06-05) showed neck-L0 is a weak teacher
            # (within-organ cos 0.68 conv / 0.98 early) while cls_embed / BN(cls_embed)
            # are far more discriminative (0.21 / 0.26 << text 0.81). The loss is a
            # stop-grad teacher, so cls_pred/BN get NO gradient from D.
            ts = getattr(self.relational_distill_loss, 'teacher_source', 'neck')
            if ts == 'neck':
                tfeats = img_feats
            else:
                hm = self.bbox_head.head_module
                ce = hm.cls_preds[0](img_feats[0])           # [B,512,H,W] pre-BN
                if ts == 'cls_embed_bn':
                    # apply the head BN in EVAL mode: we only want the current
                    # running-stat normalization, NOT a second stat update (the
                    # head's own forward already updates these BN running stats).
                    # Only BNContrastiveHead carries a .norm; a plain
                    # ContrastiveHead (logit_scale/bias) does not, so fail loud
                    # here instead of a deep AttributeError mid-training.
                    contrast = hm.cls_contrasts[0]
                    if not hasattr(contrast, 'norm'):
                        raise ValueError(
                            "teacher_source='cls_embed_bn' needs a BNContrastiveHead "
                            f"(with a .norm BN layer); head is {type(contrast).__name__}. "
                            "Use teacher_source='cls_embed' for a plain ContrastiveHead.")
                    bnmod = contrast.norm
                    was_train = bnmod.training
                    bnmod.eval()
                    ce = bnmod(ce)
                    bnmod.train(was_train)
                tfeats = (ce,)                                # roi_level=0 indexes this
            # Combine flavor B (shared-text): if the attribute head exposes an adapted
            # per-attribute class text (TextRelationalAdapter on `at`), supervise THAT so
            # the relational loss shapes the same text the region-adaptive head consumes.
            # Else (Module-2 / flavor-A configs) supervise the backbone class text.
            student_text = txt_feats
            hm = self.bbox_head.head_module
            ct = hm.cls_contrasts[0] if hasattr(hm, 'cls_contrasts') else None
            head_text = getattr(ct, 'last_class_text', None) if ct is not None else None
            if head_text is not None:
                student_text = head_text.unsqueeze(0)          # [C,D] -> [1,C,D]
            losses.update(self.relational_distill_loss.compute(
                tfeats, student_text, batch_gt_instances,
                batch_inputs.shape[-2:]))

        return losses

    # ---- D-v2 text de-collapse hook (shared by loss() and predict()) ----------
    def _decone_text(self, txt_feats, img_feats, batch_inputs,
                     batch_data_samples, update_bank: bool):
        """Apply the opt-in text de-collapse transform to txt_feats. For a transform
        with an EMA prototype bank (PrototypeGroundedTextAdapter), compute t' with the
        PRE-update bank, THEN update the bank from this batch's GT regions -> the current
        image never leaks into its own classifier (no instance leakage)."""
        decone = getattr(self, 'text_decone', None)
        if decone is None:
            return txt_feats
        t_prime = decone(txt_feats)                       # uses current (pre-update) bank
        if update_bank and hasattr(decone, 'update_bank'):
            feats, labels = self._gt_region_features(
                img_feats, batch_inputs, batch_data_samples)
            # ALWAYS call update_bank on every rank (even with 0 GT -> empty tensors): its
            # DDP all_gather is a collective, so skipping it on an empty-GT rank would
            # deadlock the other ranks. [code-review 2026-06-08]
            if feats is None:
                D = txt_feats.shape[-1]
                feats = txt_feats.new_zeros((0, D))
                labels = txt_feats.new_zeros((0,), dtype=torch.long)
            decone.update_bank(feats, labels)
        return t_prime

    @torch.no_grad()
    def _gt_region_features(self, img_feats, batch_inputs, batch_data_samples):
        """BN(cls_pred(neck))[L0] RoIAlign'd at GT boxes -> ([N, 512] region feats,
        [N] labels): the same 512-d contrastive space the head compares text against
        (cls_embed_bn). Returns (None, None) if this batch has no GT boxes. No-grad: the
        bank is a stop-grad teacher (the image side is trained only by the main losses)."""
        from torchvision.ops import roi_align
        # YOLO fast-train path: batch_data_samples is a dict with a flat bboxes_labels
        # [N,6] = (img_idx, cls, x1,y1,x2,y2); fall back to no-op for other shapes.
        if not (isinstance(batch_data_samples, dict)
                and 'bboxes_labels' in batch_data_samples):
            return None, None
        bl = batch_data_samples['bboxes_labels']
        if bl.shape[0] == 0:
            return None, None
        hm = self.bbox_head.head_module
        # cls_preds[0] is a Sequential of ConvModules with BN; run it in EVAL mode for the
        # bank extraction so its running stats are NOT updated here (the real head forward
        # in bbox_head.loss updates them once, as intended). [audit 2026-06-08]
        cp = hm.cls_preds[0]
        # snapshot EACH submodule's training flag and restore it (not just the Sequential's
        # top-level flag): under freeze_all the BN submodules are individually .eval()'d, so a
        # blanket cp.train(cp_was) would un-freeze them. [code-review 2026-06-08]
        cp_states = [(m, m.training) for m in cp.modules()]
        cp.eval()
        ce = cp(img_feats[0])                             # [B,512,H,W] pre-BN (no stat update)
        for _m, _was in cp_states:
            _m.train(_was)
        contrast = hm.cls_contrasts[0]
        if hasattr(contrast, 'norm'):                     # BNContrastiveHead -> apply BN
            bn = contrast.norm
            was = bn.training
            bn.eval(); ce = bn(ce); bn.train(was)
        # roi_align takes ONE spatial_scale for both axes; pre-scale x/y by their OWN
        # feat/input ratios so non-square inputs (H!=W) sample on-target. [code-review]
        sx = ce.shape[-1] / float(batch_inputs.shape[-1])     # width ratio
        sy = ce.shape[-2] / float(batch_inputs.shape[-2])     # height ratio
        rois = torch.cat([bl[:, :1], bl[:, 2:6]], dim=1).float()   # [N,5] batch_idx,x1,y1,x2,y2
        rois[:, [1, 3]] *= sx                                 # x1,x2
        rois[:, [2, 4]] *= sy                                 # y1,y2
        feats = roi_align(ce, rois, output_size=1, spatial_scale=1.0,
                          sampling_ratio=2, aligned=True).flatten(1)  # [N,512]
        return feats, bl[:, 1].long()

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # D-v2: same de-collapse transform at inference (bank is fixed -> no update).
        # For novel, the (base-fit) bank de-cones the novel prompts relative to base.
        txt_feats = self._decone_text(txt_feats, img_feats, batch_inputs,
                                      batch_data_samples, update_bank=False)

        # self.bbox_head.num_classes = self.num_test_classes
        # If the head uses an attribute-fusion classifier (AttributeBNContrastiveHead), its
        # output channel count is the held attr_text class count, NOT the passed prompts
        # (which it ignores). Derive num_classes from attr_text so predict_by_feat reshapes
        # the logits with the right class count. [audit 2026-06-08]
        _ccs = getattr(self.bbox_head.head_module, 'cls_contrasts', None)
        if _ccs is not None and hasattr(_ccs[0], 'attr_text'):
            self.bbox_head.num_classes = _ccs[0].attr_text.shape[0]
        else:
            self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)


    def _forward(
            self,
            batch_inputs: Tensor,
            text_feats: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """

        print('EXPORT ONNX MODEL HERE!!!!')
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 text_feats)

        results = self.bbox_head.forward(img_feats, txt_feats)
        return results



    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        # breakpoint()
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        
        return img_feats, txt_feats
    



@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    # def _forward(
    #         self,
    #         batch_inputs: Tensor,
    #         batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
    #     """Network forward process. Usually includes backbone, neck and head
    #     forward without any post-processing.
    #     """
    #     img_feats, txt_feats = self.extract_feat(batch_inputs,
    #                                              batch_data_samples)
    #     if self.reparameterized:
    #         results = self.bbox_head.forward(img_feats)
    #     else:
    #         results = self.bbox_head.forward(img_feats, txt_feats)
    #     return results

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """

        print('EXPORT ONNX MODEL HERE!!!!')
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
