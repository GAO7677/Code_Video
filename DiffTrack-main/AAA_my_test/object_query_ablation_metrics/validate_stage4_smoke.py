#!/usr/bin/env python3
"""Hard-fail validation for the Stage-4 real-video and metric smoke gate."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL = "attention_matrix_ablation_temporal_direction_v2_dose"
DOSE_KEYS = (
    "attention_mass",
    "attention_mass_query_sum",
    "removed_value_norm",
    "removed_value_norm_query_sum",
    "original_output_norm",
    "removed_to_output_ratio",
    "target_query_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant_dir", type=Path)
    parser.add_argument("--metrics-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def video_frame_count(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(probe.stdout.strip())

    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open smoke video: {path}")
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def main() -> None:
    args = parse_args()
    variant = args.variant_dir.resolve()
    manifest = read_json(variant / "manifest.json")
    complete = read_json(variant / "complete.json")
    if manifest.get("protocol") != PROTOCOL or complete.get("protocol") != PROTOCOL:
        raise RuntimeError("smoke output does not use the frozen v2 directional protocol")
    provenance = manifest.get("implementation_provenance") or {}
    if (
        provenance.get("combined_sha256") != manifest.get("code_hash")
        or len(provenance.get("files_sha256") or {}) < 2
    ):
        raise RuntimeError("smoke manifest lacks valid joint implementation provenance")

    video = variant / "generated.mp4"
    frames = video_frame_count(video)
    if frames != 49:
        raise RuntimeError(f"smoke video is not 49 frames: {frames}")

    with np.load(variant / "dose_metrics.npz") as arrays:
        for key in DOSE_KEYS:
            if key not in arrays or tuple(arrays[key].shape) != (40, 2, 30, 24):
                raise RuntimeError(f"missing or malformed smoke dose: {key}")
        entries = manifest.get("selected_entries") or []
        mask = np.zeros((30, 24), dtype=bool)
        for row in entries:
            mask[int(row["block"]), int(row["head"])] = True
        for key in DOSE_KEYS:
            values = arrays[key][:, :, mask]
            if values.size != len(entries) * 80 or not np.isfinite(values).all():
                raise RuntimeError(f"incomplete finite smoke dose coverage: {key}")

    case = str(manifest["case"])
    seed_dir = f"seed_{int(manifest['seed']):05d}"
    required_reports = (
        args.metrics_root / "head_scope_baseline_fast" / case / seed_dir / "report.json",
        args.metrics_root / "head_scope_trajectory" / case / seed_dir / "report.json",
        args.metrics_root
        / "head_scope_trajectory"
        / case
        / seed_dir
        / "object_survival_report.json",
        args.metrics_root / "head_scope_complete25" / case / seed_dir / "report.json",
    )
    for report in required_reports:
        if not report.is_file() or report.stat().st_size == 0:
            raise FileNotFoundError(f"missing Stage-4 smoke metric report: {report}")
    complete25 = read_json(required_reports[-1])
    if int(complete25.get("ablation_count", 0)) < 1:
        raise RuntimeError("complete25 smoke report contains no ablation")
    record = complete25["records"][0]
    object_a = (record.get("objects") or {}).get("object_A") or {}
    for key in ("perceptual", "shape_vs_baseline"):
        if key not in object_a:
            raise RuntimeError(f"complete25 smoke object_A record lacks {key}")
    if "baseline" not in (record.get("outside_object_lpips") or {}):
        raise RuntimeError("complete25 smoke record lacks baseline outside-object LPIPS")
    print(
        json.dumps(
            {
                "status": "PASS",
                "variant": str(variant),
                "frames": 49,
                "selected_heads": len(manifest["selected_entries"]),
                "dose_events": len(manifest["selected_entries"]) * 80,
                "metric_reports": [str(path) for path in required_reports],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
