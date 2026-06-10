"""Single source of truth loader for TCT_NGC dev30 negative/background classes.

These non-diagnostic / benign classes are excluded from detection (base) mAP. The
canonical list lives in ``data/texts/tct_ngc_dev30_negatives.json``; this loader
resolves it relative to the repo root so it works regardless of process cwd.

All eval scripts import :func:`load_dev30_negative_classes` from here; the two
``ochmta_m1`` configs read the same JSON directly (``json.load``). Keep
``tools/clinical_cost_config_dev30.json`` ``tiers.L0_negative`` in sync with this
file (checked by ``tools/check_negative_class_consistency.py``).
"""
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEV30_NEGATIVES_JSON = _REPO_ROOT / "data" / "texts" / "tct_ngc_dev30_negatives.json"


def load_dev30_negative_classes(path: str | Path | None = None) -> list[str]:
    """Return the canonical dev30 negative class-name list (single source).

    Args:
        path: Optional override for the JSON path; defaults to the repo-root
            ``data/texts/tct_ngc_dev30_negatives.json``.
    """
    p = Path(path) if path is not None else DEV30_NEGATIVES_JSON
    with open(p, encoding="utf-8") as f:
        return list(json.load(f)["negative_class_names"])
