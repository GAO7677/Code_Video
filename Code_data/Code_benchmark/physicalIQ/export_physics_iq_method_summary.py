#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

BENCHMARK_CODE = Path("/home/gaoya/Code_Video/physics-IQ-benchmark-main/code")
if str(BENCHMARK_CODE) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_CODE))

from calculate_iq_score import calculate_iq_score  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Export Physics-IQ per-method summary CSV.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--method", action="append", default=[], help="METHOD_NAME:TASK:CONTEXT_FRAMES")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for item in args.method:
        method_name, task, context_frames = item.split(":", 2)
        eval_csv = args.output_root / "eval_outputs" / "results" / f"{method_name}.csv"
        generated_dir = args.output_root / "generated_videos" / method_name
        eval_output_dir = args.output_root / "eval_outputs"
        num_videos_generated = len(list(generated_dir.glob("*.mp4")))
        if not eval_csv.exists():
            raise FileNotFoundError(f"Missing eval CSV: {eval_csv}")
        physics_iq_score, physical_variance = calculate_iq_score(str(eval_csv))
        rows.append(
            {
                "method": method_name,
                "task": task,
                "context_frames": context_frames,
                "num_cases_expected": 198,
                "num_videos_generated": num_videos_generated,
                "eval_csv": str(eval_csv),
                "physics_iq_score_mean": physics_iq_score,
                "physical_variance_mean": physical_variance,
                "generated_dir": str(generated_dir),
                "eval_output_dir": str(eval_output_dir),
                "notes": "",
            }
        )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "task",
        "context_frames",
        "num_cases_expected",
        "num_videos_generated",
        "eval_csv",
        "physics_iq_score_mean",
        "physical_variance_mean",
        "generated_dir",
        "eval_output_dir",
        "notes",
    ]
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(args.summary_csv)


if __name__ == "__main__":
    main()
