"""Centralized TCT_NGC data-root resolution.

Tools and configs should not hardcode machine-local absolute paths. Use these
helpers as defaults so a different host can override via env var without
editing code:

    from wedetect.utils.data_paths import get_tct_ngc_root, get_tct_ngc_640_root
    parser.add_argument('--data-root', default=str(get_tct_ngc_root()))

Override:  export TCT_NGC_DATA_ROOT=/path/to/TCT_NGC
           export TCT_NGC_640_ROOT=/path/to/TCT_NGC_640
           export TCT_NGC_1024_ROOT=/path/to/TCT_NGC_1024
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "TCT_NGC"
_DEFAULT_640_ROOT = _REPO_ROOT / "data" / "TCT_NGC_640"
_DEFAULT_1024_ROOT = _REPO_ROOT / "data" / "TCT_NGC_1024"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def as_posix_dir(path: str | Path) -> str:
    """Return a POSIX-style directory string with one trailing slash."""
    s = Path(path).as_posix()
    return s if s.endswith("/") else s + "/"


def get_tct_ngc_root() -> Path:
    """Full-res TCT_NGC root (env TCT_NGC_DATA_ROOT, else repo data/TCT_NGC)."""
    return _env_path("TCT_NGC_DATA_ROOT", _DEFAULT_ROOT)


def get_tct_ngc_640_root() -> Path:
    """640-cache root (env TCT_NGC_640_ROOT, else repo data/TCT_NGC_640)."""
    return _env_path("TCT_NGC_640_ROOT", _DEFAULT_640_ROOT)


def get_tct_ngc_1024_root() -> Path:
    """1024-cache root (env TCT_NGC_1024_ROOT, else repo data/TCT_NGC_1024)."""
    return _env_path("TCT_NGC_1024_ROOT", _DEFAULT_1024_ROOT)
