# NOTE (2026-05-30): BiDirectionalICF (BiICF) and HierarchicalTextAdapter (THAF)
# archived to legacy/ — text-side fusion is architecturally inert for base
# (docs/method_design/text_branch_verdict_20260530.md). Live: ImageConditionalFusion.
from .image_conditional_fusion import ImageConditionalFusion
from .relational_distill import TextRelationalAdapter
from .prototype_grounded_adapter import PrototypeGroundedTextAdapter
from .whitening_decone import WhiteningTextDecone, PerAttrWhitening
from .visual_prototype_anchor import VisualPrototypeAnchor

__all__ = [
    'ImageConditionalFusion',
    'TextRelationalAdapter',
    'PrototypeGroundedTextAdapter',
    'WhiteningTextDecone',
    'PerAttrWhitening',
    'VisualPrototypeAnchor',
]
