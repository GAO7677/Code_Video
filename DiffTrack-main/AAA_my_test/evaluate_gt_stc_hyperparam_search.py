#!/usr/bin/env python3
"""Offline GT pixel-MSE and CoTracker-metric ranking for a GT-STC sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np


ANCHORS = np.arange(0, 49, 4, dtype=np.int64)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def target_metric(path: Path, target: str) -> dict[str, Any]:
    if not path.is_file():
        return {}
    for row in read_json(path).get("metrics", []):
        if row.get("target") == target:
            return row
    return {}


def frame_mse(
    reference: np.ndarray, candidate: np.ndarray, masks: np.ndarray | None = None
) -> float | None:
    difference = (
        candidate.astype(np.float32) / 255.0
        - reference.astype(np.float32) / 255.0
    )
    squared = np.square(difference).mean(axis=-1)
    values: list[float] = []
    for index in range(len(squared)):
        selected = squared[index] if masks is None else squared[index][masks[index]]
        if selected.size:
            values.append(float(selected.mean(dtype=np.float64)))
    return float(np.mean(values, dtype=np.float64)) if values else None


def pixel_metrics(
    source: np.ndarray,
    candidate: np.ndarray,
    masks_othw: np.ndarray,
    target_index: int,
    dilation_pixels: int,
) -> dict[str, float | None]:
    if len(source) < 49 or len(candidate) < 49:
        raise RuntimeError("GT-STC MSE requires 49 source and generated frames")
    future = ANCHORS[1:]
    height, width = masks_othw.shape[-2:]
    reference = np.stack(
        [
            cv2.resize(frame[..., :3], (width, height), interpolation=cv2.INTER_AREA)
            for frame in source[future]
        ]
    )
    generated = np.stack(
        [
            cv2.resize(frame[..., :3], (width, height), interpolation=cv2.INTER_AREA)
            for frame in candidate[future]
        ]
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_pixels + 1, 2 * dilation_pixels + 1)
    )
    target_masks = np.stack(
        [
            cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
            for mask in masks_othw[target_index, 1:]
        ]
    )
    all_objects = masks_othw[:, 1:].any(axis=0)
    outside_masks = np.stack(
        [
            cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) == 0
            for mask in all_objects
        ]
    )
    return {
        "target_tube_mse_0_1": frame_mse(reference, generated, target_masks),
        "outside_object_mse_0_1": frame_mse(reference, generated, outside_masks),
        "full_frame_mse_0_1": frame_mse(reference, generated),
        "anchor_frames": future.tolist(),
        "dilation_pixels": int(dilation_pixels),
    }


def ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    trajectory = row["trajectory"]
    passed = bool(trajectory.get("quality_pass"))
    track_loss = finite(trajectory.get("future_track_loss_score_0_100"))
    ade = finite(trajectory.get("ade_d0"))
    raw_ade = finite(trajectory.get("raw_ade_d0"))
    pixel = row["pixel"]
    target_mse = finite(pixel.get("target_tube_mse_0_1"))
    outside_mse = finite(pixel.get("outside_object_mse_0_1"))
    return (
        0 if passed else 1,
        track_loss if track_loss is not None else math.inf,
        ade if passed and ade is not None else math.inf,
        raw_ade if raw_ade is not None else math.inf,
        target_mse if target_mse is not None else math.inf,
        outside_mse if outside_mse is not None else math.inf,
    )


def fmt(value: Any, digits: int = 5) -> str:
    number = finite(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tube-root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--dilation-pixels", type=int, default=16)
    args = parser.parse_args()

    tube_dir = args.tube_root / args.case
    tube = np.load(tube_dir / "tube.npz", allow_pickle=True)
    names = [str(value) for value in tube["region_names"].tolist()]
    if args.target not in names:
        raise ValueError(f"unknown target {args.target!r}; available={names}")
    target_index = names.index(args.target)
    manifest = read_json(tube_dir / "manifest.json")
    source_path = Path(str(manifest["source_video"]))
    source = np.asarray(iio.imread(source_path))[:49, ..., :3]
    generation_root = args.output_root / "generations" / args.case / f"seed_{args.seed:05d}"

    rows: list[dict[str, Any]] = []
    for video_path in sorted(generation_root.glob("*/generated.mp4")):
        variant_dir = video_path.parent
        candidate = np.asarray(iio.imread(video_path))[:49, ..., :3]
        pixel = pixel_metrics(
            source,
            candidate,
            tube["masks_othw"].astype(bool),
            target_index,
            args.dilation_pixels,
        )
        trajectory = target_metric(variant_dir / "trajectory_metrics.json", args.target)
        variant_manifest = (
            read_json(variant_dir / "manifest.json")
            if (variant_dir / "manifest.json").is_file()
            else {}
        )
        report = {
            "protocol": "gt_stc_offline_selection_metrics_v1",
            "case": args.case,
            "target": args.target,
            "variant": variant_dir.name,
            "video": str(video_path),
            "loss_mode": variant_manifest.get("loss_mode", "baseline"),
            "guidance_scale": variant_manifest.get("guidance_scale", 0.0),
            "guidance_step_range_inclusive": variant_manifest.get(
                "guidance_step_range_inclusive"
            ),
            "trajectory": trajectory,
            "pixel": pixel,
        }
        (variant_dir / "offline_selection_metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows.append(report)

    baseline = next((row for row in rows if row["variant"] == "baseline"), None)
    guided = sorted(
        (row for row in rows if row["variant"] != "baseline"), key=ranking_key
    )
    summary = {
        "protocol": "gt_stc_hyperparam_search_ranking_v1",
        "selection_order": [
            "trajectory quality gate pass",
            "future Track Loss",
            "gated ADE/D0",
            "raw ADE/D0 (failed-gate diagnostic only)",
            "target-tube GT MSE",
            "outside-object GT MSE",
        ],
        "case": args.case,
        "target": args.target,
        "baseline": baseline,
        "guided_ranking": guided,
        "acceptable_winner": guided[0] if guided and guided[0]["trajectory"].get("quality_pass") else None,
    }
    (args.output_root / "search_ranking.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# GT-STC Stage 1A Ranking",
        "",
        f"- Case: `{args.case}`; target: `{args.target}`; seed: `{args.seed}`.",
        "- MSE and CoTracker metrics are offline selection criteria only.",
        "- Ranking is lexicographic; failed trajectory gates cannot be an acceptable winner.",
        "",
        "| Rank | Variant | Gate | Track Loss | ADE/D0 | Raw ADE/D0 | Target MSE | Outside MSE | Full MSE |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    display = ([baseline] if baseline else []) + guided
    for index, row in enumerate(display):
        trajectory = row["trajectory"]
        pixel = row["pixel"]
        rank = "Base" if row["variant"] == "baseline" else str(guided.index(row) + 1)
        lines.append(
            "| "
            + " | ".join(
                [
                    rank,
                    f"`{row['variant']}`",
                    "PASS" if trajectory.get("quality_pass") else "FAIL",
                    fmt(trajectory.get("future_track_loss_score_0_100"), 2),
                    fmt(trajectory.get("ade_d0")),
                    fmt(trajectory.get("raw_ade_d0")),
                    fmt(pixel.get("target_tube_mse_0_1"), 7),
                    fmt(pixel.get("outside_object_mse_0_1"), 7),
                    fmt(pixel.get("full_frame_mse_0_1"), 7),
                ]
            )
            + " |"
        )
    (args.output_root / "STAGE1A_RANKING.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "guided": len(guided), "acceptable": summary["acceptable_winner"] is not None}))


if __name__ == "__main__":
    main()
