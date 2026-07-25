#!/usr/bin/env python3
"""Export the exact PMF/Physics-IQ input clips recorded in result JSON files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PHYSV_EVAL_PARENT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/block17_pmf_piq_metric_inputs_tmp"
)
DEFAULT_RESULTS = {
    "baseline": Path(
        "/data/gaoya/AAA_test_video/0623/test/v2v_wan/wan_lora/baseline/"
        "physicIQ_Solid_Mechanics_0107_perspective-center_trimmed-marble-run-y.json"
    ),
    "self_attn_zero_block17": Path(
        "/data/gaoya/AAA_test_video/0623/test/v2v_wan/wan_lora/"
        "self_attn_zero_block17/"
        "physicIQ_Solid_Mechanics_0107_perspective-center_trimmed-marble-run-y.json"
    ),
}
METRIC_VIDEO_KEYS = {
    "physics_iq_with_context": {
        "prediction": "scored_output_video",
        "reference": "scored_source_video",
        "side_by_side": "compare_side_by_side",
    },
    "physics_iq_without_context": {
        "prediction": "scored_output_video",
        "reference": "scored_source_video",
        "side_by_side": "compare_side_by_side",
    },
    "pmf_with_context": {
        "prediction": "pred_used_for_pmf",
        "reference": "gt_used_for_pmf",
        "side_by_side": "compare_side_by_side",
    },
    "pmf_without_context": {
        "prediction": "pred_used_for_pmf",
        "reference": "gt_used_for_pmf",
        "side_by_side": "compare_side_by_side",
    },
}


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"recorded metric artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def metric_summary(metric: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "score",
        "method",
        "official",
        "context_mode",
        "context_frames_used",
        "output_start_frame",
        "source_start_frame",
        "num_frames_compared",
        "compare_fps",
        "output_spatial_size",
        "target_size",
        "mse_mean",
        "spatiotemporal_iou_mean",
        "spatial_iou",
        "weighted_spatial_iou",
        "score_formula",
        "frame_alignment",
        "spatial_alignment",
    )
    return {field: metric[field] for field in fields if field in metric}


def recompute_metric(
    metric_name: str,
    payload: dict[str, Any],
    recorded_metric: dict[str, Any],
    metric_dir: Path,
) -> dict[str, Any]:
    if str(PHYSV_EVAL_PARENT) not in sys.path:
        sys.path.insert(0, str(PHYSV_EVAL_PARENT))

    case = dict(payload)
    case["video"] = str(payload["output_video"])
    reference_video = recorded_metric["reference_video"]
    context_mode = recorded_metric["context_mode"]
    context_frames = int(recorded_metric["context_frames_used"])
    scorer_output_dir = metric_dir / "recomputed_scorer_output"

    if metric_name.startswith("physics_iq_"):
        from physv_eval.single_case.physics_iq import score_case

        result = score_case(
            case,
            source_video_path=reference_video,
            threshold_value=int(recorded_metric["threshold_value"]),
            downsample_factor=int(recorded_metric["downsample_factor"]),
            context_mode=context_mode,
            context_frames=context_frames,
            aligned_video_dir=scorer_output_dir,
        )
    else:
        from physv_eval.single_case.pmf import score_case

        result = score_case(
            case,
            source_video_path=reference_video,
            context_mode=context_mode,
            context_frames=context_frames,
            device="cpu",
            aligned_video_dir=scorer_output_dir,
        )

    if "score" in recorded_metric:
        recorded_score = float(recorded_metric["score"])
        recomputed_score = float(result["score"])
        if abs(recorded_score - recomputed_score) > 5.0e-4:
            raise RuntimeError(
                f"{metric_name}: recorded score {recorded_score} != "
                f"recomputed score {recomputed_score}"
            )
    (metric_dir / "recomputed_metric.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def export_variant(label: str, result_json: Path, output_root: Path) -> dict[str, Any]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    variant_dir = output_root / label
    copy_required(result_json, variant_dir / "result_metrics.json")

    for source_key, output_name in (
        ("output_video", "generated_output_49f.mp4"),
        ("input_video", "conditioning_context_8f.mp4"),
    ):
        source_value = payload.get(source_key)
        if isinstance(source_value, str):
            copy_required(Path(source_value), variant_dir / output_name)

    exported_metrics: dict[str, Any] = {}
    for metric_name, video_keys in METRIC_VIDEO_KEYS.items():
        recorded_metric = payload.get(metric_name)
        if not isinstance(recorded_metric, dict):
            raise KeyError(f"{result_json}: missing metric {metric_name}")
        metric_dir = variant_dir / metric_name
        metric = recompute_metric(
            metric_name,
            payload,
            recorded_metric,
            metric_dir,
        )
        videos: dict[str, str] = {}
        for output_label, payload_key in video_keys.items():
            source_value = metric.get(payload_key)
            if not isinstance(source_value, str):
                raise KeyError(f"{result_json}: {metric_name} missing {payload_key}")
            destination = metric_dir / f"{output_label}.mp4"
            copy_required(Path(source_value), destination)
            videos[output_label] = str(destination.relative_to(output_root))
        exported_metrics[metric_name] = {
            **metric_summary(metric),
            "videos": videos,
        }

    return {
        "label": label,
        "source_result_json": str(result_json),
        "prompt": payload.get("input_caption"),
        "metrics": exported_metrics,
    }


def export_frozen_control(
    baseline_result_json: Path,
    output_root: Path,
) -> dict[str, Any]:
    if str(PHYSV_EVAL_PARENT) not in sys.path:
        sys.path.insert(0, str(PHYSV_EVAL_PARENT))
    from physv_eval.single_case.physics_iq import _read_video, _write_video

    baseline = json.loads(baseline_result_json.read_text(encoding="utf-8"))
    baseline_output_path = Path(baseline["output_video"])
    baseline_output_frames, baseline_output_fps = _read_video(
        baseline_output_path
    )
    context_frames = 8
    total_frames = 49
    if len(baseline_output_frames) < context_frames:
        raise RuntimeError(
            f"baseline output has fewer than {context_frames} frames"
        )

    variant_dir = output_root / "controlled_frozen_after_context"
    frozen_video = variant_dir / "generated_output_49f.mp4"
    frozen_frames = (
        list(baseline_output_frames[:context_frames])
        + [baseline_output_frames[context_frames - 1].copy()]
        * (total_frames - context_frames)
    )
    _write_video(frozen_frames, frozen_video, baseline_output_fps)
    _write_video(
        list(baseline_output_frames[:context_frames]),
        variant_dir / "conditioning_context_8f.mp4",
        baseline_output_fps,
    )

    payload = {
        "video": str(frozen_video),
        "output_video": str(frozen_video),
        "input_caption": baseline.get("input_caption"),
        "context_frames": context_frames,
        "effective_context_frames": context_frames,
    }
    exported_metrics: dict[str, Any] = {}
    for metric_name, video_keys in METRIC_VIDEO_KEYS.items():
        template = dict(baseline[metric_name])
        template.pop("score", None)
        template.pop("physics_iq_score", None)
        template.pop("pmf_score", None)
        metric_dir = variant_dir / metric_name
        metric = recompute_metric(metric_name, payload, template, metric_dir)
        videos: dict[str, str] = {}
        for output_label, payload_key in video_keys.items():
            destination = metric_dir / f"{output_label}.mp4"
            copy_required(Path(metric[payload_key]), destination)
            videos[output_label] = str(destination.relative_to(output_root))
        exported_metrics[metric_name] = {
            **metric_summary(metric),
            "videos": videos,
        }

    return {
        "label": "controlled_frozen_after_context",
        "source_result_json": None,
        "prompt": baseline.get("input_caption"),
        "construction": (
            "Baseline output frames 0-7 followed by 41 repeats of baseline "
            "output frame 7."
        ),
        "metrics": exported_metrics,
    }


def readme_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# Exact PMF and Physics-IQ Inputs",
        "",
        "Case: `physicIQ_Solid_Mechanics_0107...marble-run-y`",
        "",
        "Each metric directory contains the exact prediction/reference clips recorded by",
        "the scorer plus a side-by-side rendering. They were deterministically regenerated",
        "from each variant's original 49-frame output because the original intermediate",
        "artifact paths collided across ablation folders. `with_context` uses all 49 frames;",
        "`without_context` drops the first 8 frames from prediction and reference and",
        "uses the remaining 41 frames.",
        "",
        "`controlled_frozen_after_context` keeps the Baseline output's first 8 context",
        "frames and then repeats frame 7 for all 41 generated frames. It preserves the",
        "model output resolution and is a direct no-motion control.",
        "",
        "## Scores",
        "",
        "| Variant | PIQ ctx | PIQ no-ctx | PMF ctx | PMF no-ctx |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in manifest["variants"]:
        metrics = variant["metrics"]
        lines.append(
            f"| {variant['label']} "
            f"| {metrics['physics_iq_with_context']['score']} "
            f"| {metrics['physics_iq_without_context']['score']} "
            f"| {metrics['pmf_with_context']['score']} "
            f"| {metrics['pmf_without_context']['score']} |"
        )
    lines.extend(
        [
            "",
            "Physics-IQ here is the local single-view approximation, not the official",
            "multi-view Physics-IQ benchmark score.",
            "",
            "In the self-attention-zero result, all three Physics-IQ motion-overlap terms",
            "are 1.0 because both thresholded motion masks are empty. The low RGB MSE then",
            "produces a score near 100, even though this does not establish correct motion.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    variants = [
        export_variant(label, result_json, output_root)
        for label, result_json in DEFAULT_RESULTS.items()
    ]
    variants.append(
        export_frozen_control(DEFAULT_RESULTS["baseline"], output_root)
    )
    manifest = {
        "case": "physicIQ_Solid_Mechanics_0107_perspective-center_trimmed-marble-run-y",
        "variants": variants,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(readme_text(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "variants": len(variants),
                "metric_inputs": len(variants) * len(METRIC_VIDEO_KEYS),
                "videos_per_variant": 2 + len(METRIC_VIDEO_KEYS) * 3,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
