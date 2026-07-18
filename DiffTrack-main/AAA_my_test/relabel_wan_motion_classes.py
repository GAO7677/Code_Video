#!/usr/bin/env python3
"""Relabel motion classes from full trajectories without rerunning Wan inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from AAA_my_test.wan_motion_utils import OUTPUT_ROOT, atomic_write_json, classify_region_tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-dir", type=Path, default=OUTPUT_ROOT / "tracks_base")
    parser.add_argument("--results-dir", type=Path, default=OUTPUT_ROOT / "batch_base")
    args = parser.parse_args()
    changes = []
    for metadata_path in sorted(args.tracks_dir.glob("case_*_base.json")):
        metadata = json.loads(metadata_path.read_text())
        with np.load(metadata_path.with_suffix(".npz")) as loaded:
            tracks = loaded["tracks"]
            visibility = loaded["visibility"]
        region_labels = {}
        for region in metadata["regions"]:
            point_slice = slice(region["point_start"], region["point_end"])
            label = classify_region_tracks(
                tracks[:, point_slice], visibility[:, point_slice], region["region_type"]
            )
            if region["motion_class"] != label:
                changes.append(
                    {
                        "sample_key": metadata["sample_key"],
                        "region_name": region["region_name"],
                        "old": region["motion_class"],
                        "new": label,
                    }
                )
                region["motion_class"] = label
            region_labels[region["region_name"]] = label
        atomic_write_json(metadata_path, metadata)

        result_paths = list(args.results_dir.glob(f"worker_*/{metadata['sample_key']}/step_*.json"))
        result_paths += list(args.results_dir.glob(f"worker_*/{metadata['sample_key']}/metrics.json"))
        for result_path in result_paths:
            payload = json.loads(result_path.read_text())
            for row in payload["rows"]:
                row["motion_class"] = region_labels[row["region_name"]]
            atomic_write_json(result_path, payload)
    atomic_write_json(args.results_dir / "motion_class_relabel_report.json", changes)
    print(f"Relabeled {len(changes)} regions")


if __name__ == "__main__":
    main()
