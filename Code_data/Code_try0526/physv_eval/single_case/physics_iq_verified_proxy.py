from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from ..case_inputs import EvalCase, coerce_eval_case
from ..records import load_payload, stable_path_id
from .common import emit_result, result_record
from .physics_iq import _read_video, _write_video


DEFAULT_BENCHMARK_ROOT = Path(
    "/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified"
)
DEFAULT_OFFICIAL_REPO_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_phys_papers_compare/google-deepmind-physics-iq-benchmark"
)
SUPPORTED_FPS = (8, 16, 24, 30)
VIEWS = ("left", "center", "right")
RATIO_EPS = 1e-8


@dataclass(frozen=True)
class BenchmarkView:
    scene: str
    view: str
    take1_id: int
    take2_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the official Physics-IQ Verified per-view formula to the generated-only "
            "prefix of one V2V result. This is a diagnostic proxy, not an official leaderboard score."
        )
    )
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--benchmark-hint", type=Path, default=None)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--official-repo-root", type=Path, default=DEFAULT_OFFICIAL_REPO_ROOT)
    parser.add_argument("--threshold-value", type=int, default=10)
    parser.add_argument("--aligned-video-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_official_api(official_repo_root: Path) -> tuple[Any, Any, Any]:
    repo_root = official_repo_root.expanduser().resolve()
    if not (repo_root / "physiq" / "calculate_and_write_metrics_to_csv.py").is_file():
        raise FileNotFoundError(f"Physics-IQ official repository is incomplete: {repo_root}")
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from physiq.binary_mask_generator import generate_mask
    from physiq.calculate_and_write_metrics_to_csv import (
        ViewPaths,
        compute_view_metrics,
        load_view,
    )

    return generate_mask, ViewPaths, (load_view, compute_view_metrics)


def _load_benchmark_index(official_repo_root: Path) -> dict[tuple[int, str], BenchmarkView]:
    metadata_path = official_repo_root / "descriptions" / "data.yaml"
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    scenes = payload.get("scenes") if isinstance(payload, Mapping) else None
    if not isinstance(scenes, list):
        raise ValueError(f"Invalid Physics-IQ metadata: {metadata_path}")

    index: dict[tuple[int, str], BenchmarkView] = {}
    for scene_payload in scenes:
        if not isinstance(scene_payload, Mapping):
            continue
        scene = scene_payload.get("scene")
        takes = scene_payload.get("takes")
        if not isinstance(scene, str) or not isinstance(takes, list):
            continue
        take_by_number = {
            int(take["take"]): take
            for take in takes
            if isinstance(take, Mapping) and isinstance(take.get("take"), int)
        }
        if 1 not in take_by_number or 2 not in take_by_number:
            continue
        for view in VIEWS:
            take1_id = int(take_by_number[1][view])
            take2_id = int(take_by_number[2][view])
            index[(take1_id, view)] = BenchmarkView(
                scene=scene,
                view=view,
                take1_id=take1_id,
                take2_id=take2_id,
            )
    return index


def _iter_hints(
    case: EvalCase,
    benchmark_hint: Path | str | None,
) -> list[str]:
    hints: list[str] = []
    if benchmark_hint is not None:
        hints.append(str(benchmark_hint))
    if case.metadata is not None:
        for key in (
            "_json_path",
            "json_path",
            "input_json",
            "case_json",
            "source_video",
            "input_video",
        ):
            value = case.metadata.get(key)
            if isinstance(value, str) and value:
                hints.append(value)
    hints.append(str(case.video_path))
    return hints


def _resolve_benchmark_view(
    case: EvalCase,
    *,
    benchmark_hint: Path | str | None,
    index: Mapping[tuple[int, str], BenchmarkView],
) -> BenchmarkView:
    candidates: list[tuple[int, str, str]] = []
    pattern = re.compile(r"(?<!\d)(\d{4})_perspective-(left|center|right)")
    for hint in _iter_hints(case, benchmark_hint):
        for match in pattern.finditer(hint):
            candidates.append((int(match.group(1)), match.group(2), hint))

    valid = {(case_id, view): index[(case_id, view)] for case_id, view, _ in candidates if (case_id, view) in index}
    if len(valid) == 1:
        return next(iter(valid.values()))
    if not valid:
        shown = "; ".join(_iter_hints(case, benchmark_hint)[:3])
        raise ValueError(
            "Could not map this case to a Physics-IQ Verified take-1 ID and view. "
            f"Expected a name such as 0107_perspective-center; hints: {shown}"
        )
    raise ValueError(f"Ambiguous Physics-IQ benchmark IDs in case metadata: {sorted(valid)}")


def _resolve_fps(video_fps: float, benchmark_root: Path) -> int:
    rounded = int(round(video_fps))
    if rounded not in SUPPORTED_FPS or abs(video_fps - rounded) > 0.25:
        raise ValueError(
            f"Generated video FPS={video_fps:.6g} is not one of the official dataset rates "
            f"{SUPPORTED_FPS}; resample explicitly before scoring."
        )
    required_dirs = [
        benchmark_root / "split-videos" / "testing" / f"{rounded}FPS",
        benchmark_root / "video-masks" / "real" / f"{rounded}FPS",
    ]
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing Physics-IQ Verified FPS directory: {path}")
    return rounded


def _reference_paths(
    benchmark_root: Path,
    benchmark_view: BenchmarkView,
    fps: int,
) -> tuple[Path, Path, Path, Path]:
    scene = benchmark_view.scene
    view = benchmark_view.view
    take1 = f"{benchmark_view.take1_id:04d}"
    take2 = f"{benchmark_view.take2_id:04d}"
    videos = benchmark_root / "split-videos" / "testing" / f"{fps}FPS"
    masks = benchmark_root / "video-masks" / "real" / f"{fps}FPS"
    paths = (
        videos / f"{take1}_testing-videos_{fps}FPS_perspective-{view}_take-1_trimmed-{scene}.mp4",
        videos / f"{take2}_testing-videos_{fps}FPS_perspective-{view}_take-2_trimmed-{scene}.mp4",
        masks / f"{take1}_video-masks_{fps}FPS_perspective-{view}_take-1_trimmed-{scene}.mp4",
        masks / f"{take2}_video-masks_{fps}FPS_perspective-{view}_take-2_trimmed-{scene}.mp4",
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Physics-IQ Verified reference file: {path}")
    return paths


def _default_output_dir(video_path: Path, benchmark_view: BenchmarkView) -> Path:
    case_id = f"{benchmark_view.take1_id:04d}_{benchmark_view.view}_{stable_path_id(video_path)}"
    return Path("/tmp/gaoya/physics_iq_verified_proxy") / case_id


def _component_scores(metrics: Mapping[str, Any]) -> dict[str, float]:
    raw = {
        "spatiotemporal_iou": float(np.mean(metrics["spatiotemporal_iou_v1"])),
        "spatial_iou": float(metrics["spatial_iou_v1"]),
        "weighted_spatial_iou": float(metrics["weighted_spatial_iou_v1"]),
        "mse": float(np.mean(metrics["v1_mse"])),
        "physical_variance_spatiotemporal_iou": float(
            np.mean(metrics["variance_spatiotemporal_iou"])
        ),
        "physical_variance_spatial_iou": float(metrics["variance_spatial"]),
        "physical_variance_weighted_spatial_iou": float(
            metrics["variance_weighted_spatial"]
        ),
        "physical_variance_mse": float(np.mean(metrics["variance_mse"])),
    }
    ratios = {
        "spatiotemporal_iou": raw["spatiotemporal_iou"]
        / (raw["physical_variance_spatiotemporal_iou"] + RATIO_EPS),
        "spatial_iou": raw["spatial_iou"]
        / (raw["physical_variance_spatial_iou"] + RATIO_EPS),
        "weighted_spatial_iou": raw["weighted_spatial_iou"]
        / (raw["physical_variance_weighted_spatial_iou"] + RATIO_EPS),
        "mse": raw["mse"] / (raw["physical_variance_mse"] + RATIO_EPS),
    }
    mse_score = np.inf if ratios["mse"] == 0.0 else ratios["mse"] ** -1
    scores = {
        "score_spatiotemporal_iou": float(np.clip(ratios["spatiotemporal_iou"], 0.0, 1.0)),
        "score_spatial_iou": float(np.clip(ratios["spatial_iou"], 0.0, 1.0)),
        "score_weighted_spatial_iou": float(
            np.clip(ratios["weighted_spatial_iou"], 0.0, 1.0)
        ),
        "score_mse": float(np.clip(mse_score, 0.0, 1.0)),
    }
    return {**raw, **scores}


def score_case(
    case: EvalCase | Path | str | Mapping[str, Any],
    *,
    benchmark_hint: Path | str | None = None,
    context_frames: int = 8,
    benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT,
    official_repo_root: Path | str = DEFAULT_OFFICIAL_REPO_ROOT,
    threshold_value: int = 10,
    aligned_video_dir: Path | str | None = None,
) -> dict[str, Any]:
    if context_frames < 0:
        raise ValueError(f"context_frames must be >= 0, got {context_frames}")

    normalized = coerce_eval_case(case)
    candidate_path = normalized.video_path.expanduser().resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Generated video not found: {candidate_path}")

    benchmark_root_path = Path(benchmark_root).expanduser().resolve()
    official_repo_path = Path(official_repo_root).expanduser().resolve()
    index = _load_benchmark_index(official_repo_path)
    benchmark_view = _resolve_benchmark_view(
        normalized,
        benchmark_hint=benchmark_hint,
        index=index,
    )

    all_frames, candidate_fps = _read_video(candidate_path)
    if context_frames >= len(all_frames):
        raise ValueError(
            f"Generated video has {len(all_frames)} frames, cannot remove {context_frames} context frames"
        )
    generated_frames = all_frames[context_frames:]
    fps = _resolve_fps(candidate_fps, benchmark_root_path)
    expected_official_frames = 5 * fps
    if len(generated_frames) > expected_official_frames:
        generated_frames = generated_frames[:expected_official_frames]

    output_dir = (
        Path(aligned_video_dir).expanduser().resolve()
        if aligned_video_dir is not None
        else _default_output_dir(candidate_path, benchmark_view)
    )
    generated_dir = output_dir / "generated_only"
    generated_mask_dir = output_dir / "generated_masks"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_mask_dir.mkdir(parents=True, exist_ok=True)

    generated_name = (
        f"{benchmark_view.take1_id:04d}_perspective-{benchmark_view.view}_"
        f"trimmed-{benchmark_view.scene}.mp4"
    )
    generated_path = generated_dir / generated_name
    _write_video(generated_frames, generated_path, float(fps))

    generate_mask, ViewPaths, official_metric_api = _load_official_api(official_repo_path)
    load_view, compute_view_metrics = official_metric_api
    generate_mask(
        str(generated_path),
        str(generated_mask_dir / generated_name),
        False,
        int(threshold_value),
    )
    generated_mask_path = generated_mask_dir / (
        f"{benchmark_view.take1_id:04d}_video-masks_{fps}FPS_"
        f"perspective-{benchmark_view.view}_take-1_trimmed-{benchmark_view.scene}.mp4"
    )
    if not generated_mask_path.is_file():
        raise RuntimeError(f"Official mask generator did not create: {generated_mask_path}")

    real1_path, real2_path, mask1_path, mask2_path = _reference_paths(
        benchmark_root_path,
        benchmark_view,
        fps,
    )
    view_paths = ViewPaths(
        real_v1=str(real1_path),
        real_v2=str(real2_path),
        generated=str(generated_path),
        mask_v1=str(mask1_path),
        mask_v2=str(mask2_path),
        mask_generated=str(generated_mask_path),
    )
    frame_count = len(generated_frames)
    frames = load_view(view_paths, 0, frame_count, frame_count)
    if frames is None:
        raise RuntimeError(f"Could not load official reference frames: {real1_path}")
    metrics = compute_view_metrics(frames)
    components = _component_scores(metrics)
    component_score_keys = (
        "score_spatiotemporal_iou",
        "score_spatial_iou",
        "score_weighted_spatial_iou",
        "score_mse",
    )
    score_01 = float(np.mean([components[key] for key in component_score_keys]))
    score_100 = round(score_01 * 100.0, 2)
    duration_sec = frame_count / float(fps)

    return {
        "score": score_100,
        "physics_iq_verified_proxy_score": score_100,
        "score_01": round(score_01, 8),
        "score_unit": "0_to_100",
        "official": False,
        "official_formula": True,
        "official_protocol_compatible": False,
        "method": "physics_iq_verified_single_view_prefix_proxy",
        "benchmark_scene": benchmark_view.scene,
        "benchmark_view": benchmark_view.view,
        "benchmark_take1_id": benchmark_view.take1_id,
        "benchmark_take2_id": benchmark_view.take2_id,
        "context_frames_removed": int(context_frames),
        "num_frames_compared": int(frame_count),
        "expected_official_frames": int(expected_official_frames),
        "temporal_coverage_fraction": round(frame_count / expected_official_frames, 8),
        "compare_duration_sec": round(duration_sec, 8),
        "fps": int(fps),
        "views_compared": 1,
        "official_views_required": 3,
        "official_runs_required": 4,
        "generated_only_video": str(generated_path),
        "generated_mask": str(generated_mask_path),
        "take1_reference_video": str(real1_path),
        "take2_reference_video": str(real2_path),
        "take1_reference_mask": str(mask1_path),
        "take2_reference_mask": str(mask2_path),
        "threshold_value": int(threshold_value),
        **{key: round(float(value), 8) for key, value in components.items()},
        "score_formula": (
            "100 * mean(clip(model_iou/(take1_take2_iou+1e-8),0,1) for 3 IoUs, "
            "clip((model_mse/(take1_take2_mse+1e-8))^-1,0,1))"
        ),
        "official_code_reused": [
            "physiq.binary_mask_generator.generate_mask",
            "physiq.calculate_and_write_metrics_to_csv.load_view",
            "physiq.calculate_and_write_metrics_to_csv.compute_view_metrics",
        ],
        "protocol_gaps": [
            f"generated future is {duration_sec:.3f}s, official protocol requires exactly 5s",
            "only one camera view is available; official score averages left, center, and right",
            "this result is one generation run; leaderboard reporting uses four independent runs",
        ],
        "notes": (
            "Diagnostic score using the official Physics-IQ Verified per-view component formula "
            "and two-take physical-variance normalization. It is not a Physics-IQ Verified "
            "leaderboard score because the current V2V output is a short single-view prefix."
        ),
    }


def _load_cli_case(args: argparse.Namespace) -> tuple[EvalCase, Path | None]:
    if args.input_json is not None:
        payload = load_payload(args.input_json)
        payload["_json_path"] = str(args.input_json)
        return coerce_eval_case(payload), args.input_json
    if args.video is None:
        raise ValueError("Either --input-json or --video is required")
    return EvalCase(video_path=args.video), args.benchmark_hint


def main() -> None:
    args = parse_args()
    case, inferred_hint = _load_cli_case(args)
    result = score_case(
        case,
        benchmark_hint=args.benchmark_hint or inferred_hint,
        context_frames=args.context_frames,
        benchmark_root=args.benchmark_root,
        official_repo_root=args.official_repo_root,
        threshold_value=args.threshold_value,
        aligned_video_dir=args.aligned_video_dir,
    )
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
