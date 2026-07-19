#!/usr/bin/env python3
"""Evaluate native and transferred Q/K tracks against each generated video's CoTracker tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from AAA_my_test import analyze_stage1b_kubric_generation as probe
from AAA_my_test.run_lorav2v_toy_analysis_worker import load_cotracker, run_cotracker


DEFAULT_GT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_gt_attention_transfer_l23_s39"
)
DEFAULT_GENERATED_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk"
)
MODEL_COORDINATES = {"stage1b": "stretch", "lora": "cover_crop", "baseline": "cover_crop"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 25:
        raise RuntimeError(f"expected 25 frames in {path}, got {len(frames)}")
    return np.stack(frames)


def metrics(
    predictions: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    anchors: np.ndarray,
    query_latent_index: int,
    clean_prefix_latents: int,
) -> dict[str, float | int]:
    target = tracks[anchors]
    valid = visibility[anchors].copy()
    valid &= visibility[int(anchors[query_latent_index])][None]
    valid[:clean_prefix_latents] = False
    valid &= np.isfinite(predictions).all(axis=-1)
    error = np.linalg.norm(predictions - target, axis=-1)[valid]
    return {
        "comparisons": int(error.size),
        "mean_error_px": float(error.mean()),
        "median_error_px": float(np.median(error)),
        "pck16": float((error <= 16).mean() * 100),
        "pck32": float((error <= 32).mean() * 100),
        "pck64": float((error <= 64).mean() * 100),
    }


def match_record(predictions: np.ndarray, manifest: dict, source: str) -> probe.MatchRecord:
    grid = tuple(int(value) for value in manifest["token_grid"])
    return probe.MatchRecord(
        method=source,
        layer=23,
        step_index=39,
        timestep=0.0,
        sigma=None,
        grid=grid,
        clean_prefix_latents=int(manifest["clean_prefix_latents"]),
        query_latent_index=int(manifest["query_latent_index"]),
        predictions=predictions,
        probabilities=np.full((grid[0], predictions.shape[1], grid[1] * grid[2]), np.nan),
    )


def source_predictions(
    args: argparse.Namespace, model: str, case_key: str, own_case: Path
) -> dict[str, tuple[np.ndarray, dict]]:
    own_manifest = json.loads((own_case / "manifest.json").read_text(encoding="utf-8"))
    own = np.load(own_case / "predicted_tracks.npz")["qk_layer23_step039_predictions"]
    result = {"own": (own, own_manifest)}
    coordinate = MODEL_COORDINATES[model]
    for mode in ("framewise", "whole"):
        variant = args.gt_root / f"gt_{mode}_{coordinate}" / "cases" / case_key
        manifest = json.loads((variant / "manifest.json").read_text(encoding="utf-8"))
        predictions = np.load(variant / "predicted_tracks.npz")[
            "qk_layer23_step039_predictions"
        ]
        result[mode] = (predictions, manifest)
    return result


def main() -> None:
    args = parse_args()
    cotracker = load_cotracker(args.device)
    rows = []
    for model in MODEL_COORDINATES:
        for case_dir in sorted((args.generated_root / model / "cases").iterdir()):
            if not (case_dir / "complete.json").is_file():
                continue
            case_key = case_dir.name
            output = args.gt_root / "comparisons" / model / "cases" / case_key
            cotracker_path = output / "generated_cotracker.npz"
            frames = read_video(case_dir / "generated.mp4")
            own_manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
            query_points = np.asarray(own_manifest["query_points"], dtype=np.float32)
            query_frame = int(own_manifest["query_pixel_frame"])
            if cotracker_path.is_file() and not args.overwrite:
                data = np.load(cotracker_path)
                tracks, visibility = data["tracks"], data["visibility"]
            else:
                output.mkdir(parents=True, exist_ok=True)
                tracks, visibility = run_cotracker(
                    cotracker, frames, query_points, query_frame, args.device
                )
                np.savez_compressed(
                    cotracker_path,
                    tracks=tracks,
                    visibility=visibility,
                    query_points=query_points,
                )
            sources = source_predictions(args, model, case_key, case_dir)
            anchors = np.asarray(own_manifest["latent_anchor_pixel_frames"], dtype=np.int64)
            for source, (predictions, manifest) in sources.items():
                for region in own_manifest["query_regions"]:
                    point_slice = slice(int(region["point_start"]), int(region["point_end"]))
                    sliced_predictions = predictions[:, point_slice]
                    row = {
                        "model": model,
                        "case_key": case_key,
                        "source": source,
                        "region_name": region["region_name"],
                        **metrics(
                            sliced_predictions,
                            tracks[:, point_slice],
                            visibility[:, point_slice],
                            anchors,
                            int(manifest["query_latent_index"]),
                            int(manifest["clean_prefix_latents"]),
                        ),
                    }
                    rows.append(row)
                    record = match_record(sliced_predictions, manifest, source)
                    video_path = (
                        output
                        / "regions"
                        / region["region_name"]
                        / f"tracks_{source}_vs_cotracker.mp4"
                    )
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    probe.draw_track_video(
                        frames,
                        anchors,
                        record,
                        tracks[:, point_slice],
                        visibility[:, point_slice],
                        video_path,
                        int(args.fps),
                    )
            print(f"complete {model}/{case_key}", flush=True)
    (args.gt_root / "comparison_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
