#!/usr/bin/env python3
"""Select within-group case pairs whose rigid-motion amplitudes are too similar."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.10, help="Maximum symmetric relative difference in [0, 2].")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def symmetric_relative_difference(first: float, second: float) -> float:
    denominator = 0.5 * (first + second)
    if denominator <= 1e-12:
        return 0.0 if abs(first - second) <= 1e-12 else 2.0
    return abs(first - second) / denominator


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 2.0:
        raise ValueError("--threshold must be between 0 and 2")
    root = args.dataset_root.resolve()
    amplitude = read_json(root / "reports/motion_amplitude.json")
    similarity_path = root / "reports/trajectory_similarity.json"
    similarity = read_json(similarity_path) if similarity_path.is_file() else {"groups": {}}
    output = (args.output_json or root / "reports/motion_amplitude_pairs.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    group_reports: dict[str, Any] = {}
    total_selected = 0
    for group, group_summary in amplitude["groups"].items():
        case_ids = list(group_summary["case_ids_descending"])
        trajectory_group = similarity.get("groups", {}).get(group, {})
        similarity_ids = trajectory_group.get("case_ids", [])
        similarity_matrix = trajectory_group.get("similarity_matrix", [])
        similarity_index = {case_id: index for index, case_id in enumerate(similarity_ids)}
        pairs: list[dict[str, Any]] = []
        for case_a, case_b in itertools.combinations(case_ids, 2):
            amplitude_a = float(amplitude["cases"][case_a]["motion_amplitude_m"])
            amplitude_b = float(amplitude["cases"][case_b]["motion_amplitude_m"])
            difference = symmetric_relative_difference(amplitude_a, amplitude_b)
            trajectory_score = None
            if case_a in similarity_index and case_b in similarity_index:
                trajectory_score = similarity_matrix[similarity_index[case_a]][similarity_index[case_b]]
            control_a = amplitude["cases"][case_a].get("control", {})
            control_b = amplitude["cases"][case_b].get("control", {})
            pairs.append(
                {
                    "case_a": case_a,
                    "case_b": case_b,
                    "control_a": control_a,
                    "control_b": control_b,
                    "amplitude_a_m": amplitude_a,
                    "amplitude_b_m": amplitude_b,
                    "absolute_difference_m": abs(amplitude_a - amplitude_b),
                    "symmetric_relative_difference": difference,
                    "difference_percent": difference * 100.0,
                    "trajectory_similarity": trajectory_score,
                    "selected": difference <= args.threshold,
                }
            )
        pairs.sort(key=lambda item: item["symmetric_relative_difference"])
        selected = [pair for pair in pairs if pair["selected"]]
        total_selected += len(selected)
        group_reports[group] = {
            "case_count": len(case_ids),
            "pair_count": len(pairs),
            "selected_pair_count": len(selected),
            "selected_pairs": selected,
            "all_pairs": pairs,
        }

    report = {
        "schema_version": "physv_motion_amplitude_pair_filter_v1",
        "dataset_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": {
            "name": "symmetric_relative_rigid_motion_difference",
            "formula": "abs(A-B) / ((A+B)/2)",
            "amplitude_source": "motion_amplitude.json: total_rigid_motion_length",
            "range": [0.0, 2.0],
            "interpretation": "Lower values indicate more similar scene-level rigid-motion amplitude.",
        },
        "threshold": args.threshold,
        "threshold_percent": args.threshold * 100.0,
        "selected_pair_count": total_selected,
        "groups": group_reports,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={output}")
    print(f"threshold={args.threshold:.4f} selected_pairs={total_selected}")
    for group, summary in group_reports.items():
        print(f"{group}: {summary['selected_pair_count']}/{summary['pair_count']}")


if __name__ == "__main__":
    main()
