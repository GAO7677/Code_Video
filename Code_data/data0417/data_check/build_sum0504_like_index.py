#!/usr/bin/env python3
# 用途：包装调用 sum0504 路径索引重建脚本，把 Genesis raw train/rigid 整理成 sum0504 风格目录。
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA0417_ROOT = SCRIPT_DIR.parent
REBUILD_SCRIPT = DATA0417_ROOT / "genesis_rigid_data" / "repair" / "rebuild_sum0504_index.py"

DEFAULT_RAW_TRAIN_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version_1_genesis_rigid_data_all_cases_sum0504_like"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_train_root", type=Path, default=DEFAULT_RAW_TRAIN_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        str(args.python_bin),
        str(REBUILD_SCRIPT),
        "--raw_train_root",
        str(args.raw_train_root.resolve()),
        "--output_root",
        str(args.output_root.resolve()),
    ]
    if args.dry_run:
        cmd.append("--dry_run")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
