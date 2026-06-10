#!/usr/bin/env python
"""Write environment-variable data-root hints for WeDetect configs.

Configs now resolve dataset roots from environment variables instead of
machine-local absolute paths:

    TCT_NGC_DATA_ROOT
    TCT_NGC_640_ROOT
    TCT_NGC_1024_ROOT

This script is kept for deploy/setup.py compatibility. It does not rewrite
source files.
"""
import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dir containing TCT_NGC / TCT_NGC_640 / TCT_NGC_1024")
    ap.add_argument("--out", default="deploy/data_roots.env",
                    help="shell snippet to write")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    exports = {
        "TCT_NGC_DATA_ROOT": data_dir / "TCT_NGC",
        "TCT_NGC_640_ROOT": data_dir / "TCT_NGC_640",
        "TCT_NGC_1024_ROOT": data_dir / "TCT_NGC_1024",
    }
    lines = [f'export {key}="{path}"' for key, path in exports.items()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    print("Run this in future shells before training/eval:")
    print(f"  source {out}")


if __name__ == "__main__":
    main()
