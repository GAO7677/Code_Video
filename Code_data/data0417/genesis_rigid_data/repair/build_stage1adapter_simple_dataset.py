#!/usr/bin/env python3
# 用途：从 Genesis raw train 构建精简的 stage1adapter simple train 数据目录。
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.stage1adapter_simple_builder import BuilderConfig, process_dataset, resolve_dataset_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build simple-motion Genesis stage1adapter train packages from raw train data."
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        required=True,
        help="Either <dataset_root> or <dataset_root>/train",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Defaults to <dataset_root>/stage1adapter",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root, _ = resolve_dataset_roots(args.raw_root)
    output_root = args.output_root or (dataset_root / "stage1adapter")
    summary = process_dataset(
        BuilderConfig(
            raw_root=args.raw_root,
            output_root=output_root,
            overwrite=bool(args.overwrite),
            max_samples=int(args.max_samples),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
