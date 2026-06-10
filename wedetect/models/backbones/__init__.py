# Copyright (c) Tencent Inc. All rights reserved.
# YOLO Multi-Modal Backbone (Vision Language)
# Vision: YOLOv8 CSPDarknet
# Language: CLIP Text Encoder (12-layer transformer)
from .mm_backbone import (
    MultiModalYOLOBackbone,
    HuggingVisionBackbone,
    HuggingCLIPLanguageBackbone,
    PseudoLanguageBackbone,
    XLMRobertaLanguageBackbone,
    ConvNextVisionBackbone,
    HuggingCLIPVisionBackbone,
    )
# NOTE (2026-05-30): THAF / PCW / ConcatProj / SAVPE language-backbone
# experiments archived to legacy/ (text-side fusion is architecturally inert for
# base — see docs/method_design/text_branch_verdict_20260530.md). The live line
# uses PseudoLanguageBackbone / PseudoMultiAttrLanguageBackbone (both registered
# via the mm_backbone import above) + ImageConditionalFusion.
from .dinov3_vit import DINOv3VisionBackbone
from .biomedclip_vit import BiomedCLIPVisionBackbone

__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'PseudoLanguageBackbone',
    'ConvNextVisionBackbone',
    'XLMRobertaLanguageBackbone',
    'HuggingCLIPVisionBackbone',
    'DINOv3VisionBackbone',
    'BiomedCLIPVisionBackbone',
]
