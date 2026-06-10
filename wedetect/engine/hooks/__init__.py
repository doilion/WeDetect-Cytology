# NOTE (2026-05-30): AdapterCollapseGuard (THAF) and STEGOCorrEMAHook (corr-loss)
# archived to legacy/ — see docs/method_design/text_branch_verdict_20260530.md.
from .check_backbone_norm_hook import CheckBackboneNormHook
from .icf_collapse_guard import ICFCollapseGuard
from .skip_backbone_in_load_from import SkipImageBackboneInLoadFromHook

__all__ = [
    'CheckBackboneNormHook',
    'ICFCollapseGuard',
    'SkipImageBackboneInLoadFromHook',
]
