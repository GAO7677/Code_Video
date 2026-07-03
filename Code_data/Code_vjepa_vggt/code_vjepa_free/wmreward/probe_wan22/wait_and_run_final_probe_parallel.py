#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
SCRIPT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait until a feature root contains the expected number of probe_features.pt files, "
            "then launch the parallel final probe pipeline."
        )
    )
    parser.add_argument(
        "--watch-root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/extracted/wan21_t2v_1p3b_final/full_default"
        ),
    )
    parser.add_argument("--expected-count", type=int, default=1902)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/datasets/generated/manifests/generated_probe_pairs_final.csv"
        ),
    )
    parser.add_argument("--visible-gpus", default="1,6,7")
    parser.add_argument("--extract-presets", default="full_default,late_focus,late_dense")
    parser.add_argument("--index-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    return parser.parse_args()


def count_feature_files(root: Path) -> int:
    return sum(1 for _ in root.rglob("probe_features.pt"))


def main() -> None:
    args = parse_args()
    watch_root = args.watch_root.expanduser().resolve()

    while True:
        current_count = count_feature_files(watch_root)
        print(
            f"[wait] watch_root={watch_root} current={current_count} expected={args.expected_count}",
            flush=True,
        )
        if current_count >= args.expected_count:
            break
        time.sleep(args.poll_seconds)

    cmd = [
        args.python_bin,
        SCRIPT_ROOT / "run_wan21_final_probe_parallel_pipeline.py",
        "--manifest-csv",
        args.manifest_csv.expanduser().resolve(),
        "--visible-gpus",
        args.visible_gpus,
        "--extract-presets",
        args.extract_presets,
    ]
    if args.index_root is not None:
        cmd.extend(["--index-root", args.index_root.expanduser().resolve()])
    if args.results_root is not None:
        cmd.extend(["--results-root", args.results_root.expanduser().resolve()])

    print("[run]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], check=True)


if __name__ == "__main__":
    sys.exit(main())
