#!/usr/bin/env python3
"""Delete only reproducible temporary Wan motion experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from AAA_my_test.wan_motion_utils import OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--delete-smoke", action="store_true")
    parser.add_argument("--delete-visualizations", action="store_true")
    args = parser.parse_args()
    candidates = list(args.root.rglob("*.tmp"))
    if args.delete_smoke and (args.root / "smoke").exists():
        candidates.extend(path for path in (args.root / "smoke").rglob("*") if path.is_file())
    if args.delete_visualizations:
        for pattern in ("*.mp4", "*.png", "*.jpg"):
            candidates.extend(args.root.rglob(pattern))
    unique = sorted(set(candidates))
    total = sum(path.stat().st_size for path in unique if path.exists())
    for path in unique:
        path.unlink(missing_ok=True)
    for directory in sorted((path for path in args.root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    print(f"Deleted {len(unique)} reproducible files ({total / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
