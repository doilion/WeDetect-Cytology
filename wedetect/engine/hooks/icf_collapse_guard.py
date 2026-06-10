"""mmengine hook that monitors ImageConditionalFusion (Design A) health.

Reads diagnostic state from the fusion module after every N training iters
(and at every val end) and logs scalar collapse indicators. Mirrors the
AdapterCollapseGuard pattern used for M2 HierarchicalTextAdapter.

Three indicators (all in well-bounded ranges so thresholds are stable):
  - fused_pairwise_cos_mean:  mean cosine between fused vectors of the SAME
        class across different images in the batch. Image-conditional fusion
        produces different outputs for different images -> healthy < 0.99.
        > 0.99 = image-invariant collapse (THAF-style mean-pool failure).
  - attn_entropy_mean:        mean entropy of attention over A=5 attribute
        keys. log(5) = 1.609 is uniform = mean-pool equivalent. > 1.58
        means attention is essentially uniform.
  - cos_to_attr_mean_mean:    mean cosine between fused output and
        normalized attr_mean. 1.0 means fused == mean-pool direction.
        > 0.97 = fused has collapsed onto the mean-pool subspace.
"""
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmdet.registry import HOOKS


@HOOKS.register_module()
class ICFCollapseGuard(Hook):
    """Monitor ImageConditionalFusion for collapse and optionally halt.

    Args:
        check_interval: log every N training iters. Default 500.
        check_at_val: also log at the start of validation.
        halt_on_red: if True, raises RuntimeError when any threshold is
                     exceeded. If False, just logs WARNING.
        pairwise_cos_max: fused_pairwise_cos_mean upper bound (cross-image
                          cosine for same class — should be < 0.99 in
                          a healthy image-conditional fusion).
        attn_entropy_max: attn_entropy_mean upper bound (should be < log(5)
                          = 1.609; > 1.58 means uniform attention).
        cos_to_mean_max:  cos_to_attr_mean_mean upper bound (should be
                          < 0.97; higher = fused collapses to mean pool).
    """

    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        check_interval: int = 500,
        check_at_val: bool = True,
        halt_on_red: bool = False,
        pairwise_cos_max: float = 0.99,
        attn_entropy_max: float = 1.58,
        cos_to_mean_max: float = 0.97,
    ):
        self.check_interval = check_interval
        self.check_at_val = check_at_val
        self.halt_on_red = halt_on_red
        self.thresholds = {
            'fused_pairwise_cos': ('fused_pairwise_cos_mean', pairwise_cos_max, 'max'),
            'attn_entropy':       ('attn_entropy_mean',       attn_entropy_max, 'max'),
            'cos_to_attr_mean':   ('cos_to_attr_mean_mean',   cos_to_mean_max,  'max'),
        }

    def _get_fusion(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        backbone = getattr(model, 'backbone', None)
        if backbone is None:
            return None
        return getattr(backbone, 'cross_modal_fusion', None)

    def _check(self, runner, tag: str):
        fusion = self._get_fusion(runner)
        if fusion is None or not hasattr(fusion, 'get_collapse_diagnostics'):
            return
        diag = fusion.get_collapse_diagnostics()
        if not diag:
            return

        logger = MMLogger.get_current_instance()
        msg_parts = [f'[ICFCollapseGuard {tag}]']
        red = []

        for label, (diag_key, threshold, kind) in self.thresholds.items():
            val = diag.get(diag_key)
            if val is None or val != val:  # skip NaN
                continue
            if kind == 'max':
                bad = val > threshold
            elif kind == 'min':
                bad = val < threshold
            else:
                continue
            marker = 'X' if bad else 'OK'
            sym = '<=' if kind == 'max' else '>='
            msg_parts.append(f'{label}={val:.3f}({sym}{threshold}) {marker}')
            if bad:
                red.append((label, val, threshold, kind))

        # also log pairwise distance for readability
        dist = diag.get('fused_pairwise_dist_mean')
        if dist is not None and dist == dist:
            msg_parts.append(f'pairwise_dist={dist:.3f}')

        logger.info(' | '.join(msg_parts))

        if red and self.halt_on_red:
            details = '\n  '.join(
                f'{name}: value={v:.4f}, threshold={t} ({k})'
                for name, v, t, k in red
            )
            raise RuntimeError(
                f'ICFCollapseGuard halted training due to:\n  {details}'
            )

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (batch_idx + 1) % self.check_interval != 0:
            return
        self._check(runner, f'iter {runner.iter + 1}')

    def before_val_epoch(self, runner):
        if not self.check_at_val:
            return
        self._check(runner, f'pre-val ep {runner.epoch + 1}')
