#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a flat metadata index for extracted Wan probe features.")
    parser.add_argument(
        "--feature_root",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/extracted",
    )
    parser.add_argument(
        "--output_csv",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/probe_index.csv",
    )
    parser.add_argument(
        "--output_jsonl",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/probe_index.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    feature_root = Path(args.feature_root)
    rows = []

    for meta_path in sorted(feature_root.glob("*/meta.json")):
        sample_dir = meta_path.parent
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        row = {
            "sample_id": meta["sample_id"],
            "pair_id": meta.get("pair_id", ""),
            "role": meta.get("role", ""),
            "basename": meta.get("basename", ""),
            "prompt": meta["prompt"],
            "seed": meta["seed"],
            "source_surprise_score": meta.get("source_surprise_score", ""),
            "num_inference_steps": meta["num_inference_steps"],
            "capture_step_indices": ",".join(str(x) for x in meta.get("capture_step_indices", [])),
            "capture_layers": ",".join(str(x) for x in meta.get("capture_layers", [])),
            "capture_branches": meta.get("capture_branches", ""),
            "feature_path": str(sample_dir / "probe_features.pt"),
            "meta_path": str(meta_path),
        }
        if not (sample_dir / "probe_features.pt").exists():
            row["feature_path"] = str(sample_dir / "probe_forward_smoke.pt")
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "pair_id",
        "role",
        "basename",
        "prompt",
        "seed",
        "source_surprise_score",
        "num_inference_steps",
        "capture_step_indices",
        "capture_layers",
        "capture_branches",
        "feature_path",
        "meta_path",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(output_csv)
    print(output_jsonl)


if __name__ == "__main__":
    main()
