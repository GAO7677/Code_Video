from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "method", "run", "dataset", "prompt", "image_source", "num_videos",
    "candidates_per_case", "context_frames", "output_frames", "fps", "duration_s",
    "score_original", "score_verified", "score_spatiotemporal_view",
    "score_spatial_view", "score_weighted_spatial_view", "score_mse_view",
    "result_dir", "metrics_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert one Physics-IQ metrics row.")
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--output-frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=24.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.metrics_file.read_text())
    csv_path = args.table_root / "physicsiq_metrics.csv"
    rows = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({field: row.get(field, "") for field in FIELDS})
    row = {
        "method": args.method,
        "run": args.run,
        "dataset": "Physics-IQ Verified evaluator / Original score",
        "prompt": args.prompt,
        "image_source": args.image_source,
        "num_videos": 198,
        "candidates_per_case": args.candidates,
        "context_frames": 1,
        "output_frames": args.output_frames,
        "fps": args.fps,
        "duration_s": args.output_frames / args.fps,
        "score_original": data["final_score_origround"],
        "score_verified": 100.0 * data["final_score_view"],
        "score_spatiotemporal_view": data["score_spatiotemporal_view"],
        "score_spatial_view": data["score_spatial_view"],
        "score_weighted_spatial_view": data["score_weighted_spatial_view"],
        "score_mse_view": data["score_mse_view"],
        "result_dir": str(args.result_dir),
        "metrics_file": str(args.metrics_file),
    }
    rows = [old for old in rows if old["run"] != args.run]
    rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.table_root / "physicsiq_metrics.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Physics-IQ Metrics\n\n")
        handle.write("官方 evaluator 指标汇总。Original/Verified 为百分制。\n\n")
        columns = ["Method", "Run", "Prompt", "Image", "N", "BoN", "Frames", "FPS", "Original", "Verified"]
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for item in rows:
            values = [
                item["method"], item["run"], item["prompt"], item["image_source"],
                str(item["num_videos"]), str(item["candidates_per_case"]),
                str(item["output_frames"]), str(item["fps"]),
                f"{float(item['score_original']):.2f}", f"{float(item['score_verified']):.2f}",
            ]
            handle.write("| " + " | ".join(values) + " |\n")
        handle.write("\n## Artifacts\n\n")
        for item in rows:
            handle.write(f"- `{item['run']}`: `{item['result_dir']}`; metrics: `{item['metrics_file']}`\n")


if __name__ == "__main__":
    main()
