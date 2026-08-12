#!/usr/bin/env python3
"""Build compact Baseline/Top/Bottom/Random/All Stage-3 frame montages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


DEFAULT_METRIC_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage3_metrics/head_scope_baseline_fast"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage3_interim_analysis/representatives"
)
SCOPES = (
    "top100",
    "bottom100",
    "random100_layer_matched_draw0",
    "all720",
)
SCOPE_LABELS = {
    "top100": "Top100",
    "bottom100": "Bottom100",
    "random100_layer_matched_draw0": "Random100",
    "all720": "All720",
}
FLOWS = ("self_only", "incoming_only", "outgoing_only")
FLOW_LABELS = {
    "self_only": "M1 R->R",
    "incoming_only": "M2 C->R",
    "outgoing_only": "M3 R->C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("seed", type=int)
    parser.add_argument("region")
    parser.add_argument("--metric-root", type=Path, default=DEFAULT_METRIC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def frames(path: str) -> list[Any]:
    capture = cv2.VideoCapture(path)
    result = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result.append(frame)
    capture.release()
    if not result:
        raise RuntimeError(f"could not decode {path}")
    return result


def put_label(image: Any, label: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 25), (10, 10, 10), -1)
    cv2.putText(image, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    report_path = args.metric_root / args.case / f"seed_{args.seed}" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    target_scope = "all_objects" if args.region == "all_objects" else "single_object"
    selected = [
        row
        for row in report["records"]
        if row["target_scope"] == target_scope and row["region"] == args.region
    ]
    lookup = {(row["mask_mode"], row["head_scope"]): row for row in selected}
    baseline = frames(report["baseline_path"])
    indices = [0, round((len(baseline) - 1) * 0.25), round((len(baseline) - 1) * 0.5), round((len(baseline) - 1) * 0.75), len(baseline) - 1]
    output_dir = args.output_dir / args.case / f"seed_{args.seed}" / args.region
    output_dir.mkdir(parents=True, exist_ok=True)
    for flow in FLOWS:
        rows = [("Baseline", baseline, None)]
        for scope in SCOPES:
            record = lookup[(flow, scope)]
            rows.append((SCOPE_LABELS[scope], frames(record["path"]), record))
        rendered_rows = []
        for row_label, video, record in rows:
            cells = []
            for index in indices:
                frame = video[min(index, len(video) - 1)]
                frame = cv2.resize(frame, (320, 176), interpolation=cv2.INTER_AREA)
                frame = frame.copy()
                if record is None:
                    label = f"{row_label}  F{index:02d}"
                else:
                    categories = record["metrics"]["category_scores_0_100"]
                    outside = 100.0 * record["metrics"]["outside_objects"]["mae_0_1"]
                    label = (
                        f"{row_label} F{index:02d}  target={categories['target_local']:.2f} "
                        f"outside={outside:.2f}"
                    )
                put_label(frame, label)
                cells.append(frame)
            rendered_rows.append(cv2.hconcat(cells))
        montage = cv2.vconcat(rendered_rows)
        title = f"{args.case} seed={args.seed} {args.region} {FLOW_LABELS[flow]}"
        header = montage[:34].copy()
        header[:] = (25, 25, 25)
        cv2.putText(header, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        montage = cv2.vconcat([header, montage])
        cv2.imwrite(str(output_dir / f"{flow}.jpg"), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(output_dir)


if __name__ == "__main__":
    main()
