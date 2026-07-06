from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physv_eval.single_case.physics_iq import score_case as score_physics_iq
from physv_eval.single_case.pmf import score_case as score_pmf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Physics-IQ and PMF for both with/without-context modes for one case JSON, "
            "write the four metric payloads back into the case JSON, and optionally sync the "
            "matching entry inside the sibling result.json."
        )
    )
    parser.add_argument(
        "--case-json",
        type=Path,
        required=True,
        help="Per-case JSON next to the generated output video.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/context_metric_backfill"),
        help="Root directory for aligned videos and compare exports.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=None,
        help="Optional explicit context frame count. Defaults to case metadata when available.",
    )
    parser.add_argument(
        "--pmf-device",
        default="cpu",
        help="Torch device for PMF, for example cpu or cuda.",
    )
    parser.add_argument(
        "--physics-iq-downsample-factor",
        type=int,
        default=4,
        help="Downsample factor forwarded into single_case.physics_iq.",
    )
    parser.add_argument(
        "--physics-iq-threshold-value",
        type=int,
        default=10,
        help="Motion threshold forwarded into single_case.physics_iq.",
    )
    parser.add_argument(
        "--skip-result-json",
        action="store_true",
        help="Do not sync the matching entry inside sibling result.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_source_candidate(candidate: str | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate)
    return path if path.is_file() else None


def resolve_source_video(case_payload: dict[str, Any], case_json_path: Path) -> Path:
    direct_candidates = [
        case_payload.get("source_video"),
        case_payload.get("reference_video"),
    ]
    physics_payload = case_payload.get("physics_iq")
    if isinstance(physics_payload, dict):
        direct_candidates.append(physics_payload.get("reference_video"))

    input_json_value = case_payload.get("input_json")
    if isinstance(input_json_value, str) and input_json_value:
        input_json_path = Path(input_json_value)
        if input_json_path.is_file():
            input_json_payload = load_json(input_json_path)
            direct_candidates.extend(
                [
                    input_json_payload.get("source_video"),
                    input_json_payload.get("gt_full"),
                    input_json_payload.get("reference_video"),
                ]
            )

    for candidate in direct_candidates:
        resolved = _normalize_source_candidate(candidate if isinstance(candidate, str) else None)
        if resolved is not None:
            return resolved

    input_video_value = case_payload.get("input_video")
    if isinstance(input_video_value, str) and input_video_value:
        input_video_path = Path(input_video_value)
        # Common PhysV layout:
        #   .../sample_xxxx/source_video/context_video_8f.mp4
        # -> .../sample_xxxx/source_video.mp4
        if (
            input_video_path.name.startswith("context_video_")
            and input_video_path.parent.name == "source_video"
            and input_video_path.parent.parent.exists()
        ):
            candidate = input_video_path.parent.parent / "source_video.mp4"
            if candidate.is_file():
                return candidate
        # Fallback for other datasets that keep source video beside the context clip.
        candidate = input_video_path.parent / "source_video.mp4"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve GT/source video for case JSON: {case_json_path}"
    )


def resolve_context_frames(case_payload: dict[str, Any], explicit_context_frames: int | None) -> int | None:
    if explicit_context_frames is not None:
        return int(explicit_context_frames)
    model_args = case_payload.get("model_args")
    if isinstance(model_args, dict):
        value = model_args.get("context_frames")
        if isinstance(value, int):
            return int(value)
    value = case_payload.get("context_frames")
    if isinstance(value, int):
        return int(value)
    return None


def build_artifact_dir(
    *,
    artifacts_root: Path,
    case_payload: dict[str, Any],
    case_json_path: Path,
) -> Path:
    method_name = str(case_payload.get("method") or case_json_path.parent.name)
    case_name = case_json_path.stem
    return artifacts_root / method_name / case_name


def compute_four_metrics(
    *,
    case_payload: dict[str, Any],
    case_json_path: Path,
    artifacts_root: Path,
    source_video_path: Path,
    context_frames: int | None,
    pmf_device: str,
    physics_iq_downsample_factor: int,
    physics_iq_threshold_value: int,
) -> dict[str, Any]:
    case_artifact_dir = build_artifact_dir(
        artifacts_root=artifacts_root,
        case_payload=case_payload,
        case_json_path=case_json_path,
    )
    case_artifact_dir.mkdir(parents=True, exist_ok=True)

    physics_iq_with_context = score_physics_iq(
        case_payload,
        source_video_path=source_video_path,
        context_mode="with_context",
        context_frames=context_frames,
        threshold_value=physics_iq_threshold_value,
        downsample_factor=physics_iq_downsample_factor,
        aligned_video_dir=case_artifact_dir / "physics_iq_with_context",
    )
    physics_iq_without_context = score_physics_iq(
        case_payload,
        source_video_path=source_video_path,
        context_mode="without_context",
        context_frames=context_frames,
        threshold_value=physics_iq_threshold_value,
        downsample_factor=physics_iq_downsample_factor,
        aligned_video_dir=case_artifact_dir / "physics_iq_without_context",
    )
    pmf_with_context = score_pmf(
        case_payload,
        source_video_path=source_video_path,
        context_mode="with_context",
        context_frames=context_frames,
        device=pmf_device,
        aligned_video_dir=case_artifact_dir / "pmf_with_context",
    )
    pmf_without_context = score_pmf(
        case_payload,
        source_video_path=source_video_path,
        context_mode="without_context",
        context_frames=context_frames,
        device=pmf_device,
        aligned_video_dir=case_artifact_dir / "pmf_without_context",
    )

    return {
        "physics_iq_with_context": physics_iq_with_context,
        "physics_iq_without_context": physics_iq_without_context,
        "pmf_with_context": pmf_with_context,
        "pmf_without_context": pmf_without_context,
        "context_metric_summary": {
            "reference_video": str(source_video_path),
            "context_frames": None if context_frames is None else int(context_frames),
            "scores": {
                "physics_iq_with_context": float(physics_iq_with_context["score"]),
                "physics_iq_without_context": float(physics_iq_without_context["score"]),
                "pmf_with_context": float(pmf_with_context["score"]),
                "pmf_without_context": float(pmf_without_context["score"]),
            },
        },
    }


def update_result_json(
    *,
    result_json_path: Path,
    output_video: str,
    metric_payload: dict[str, Any],
) -> bool:
    result_payload = load_json(result_json_path)
    entries = result_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{result_json_path}: entries is not a list")

    updated = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("output_video") != output_video:
            continue
        entry.update(metric_payload)
        updated = True
        break

    if not updated:
        return False

    backfill = result_payload.get("context_metric_backfill")
    if not isinstance(backfill, dict):
        backfill = {}
    backfill.update(
        {
            "metric_keys": [
                "physics_iq_with_context",
                "physics_iq_without_context",
                "pmf_with_context",
                "pmf_without_context",
            ],
            "last_case_output_video": output_video,
        }
    )
    result_payload["context_metric_backfill"] = backfill
    save_json(result_json_path, result_payload)
    return True


def main() -> None:
    args = parse_args()
    case_json_path = args.case_json.expanduser().resolve()
    case_payload = load_json(case_json_path)
    source_video_path = resolve_source_video(case_payload, case_json_path)
    context_frames = resolve_context_frames(case_payload, args.context_frames)

    metric_payload = compute_four_metrics(
        case_payload=case_payload,
        case_json_path=case_json_path,
        artifacts_root=args.artifacts_root.expanduser().resolve(),
        source_video_path=source_video_path,
        context_frames=context_frames,
        pmf_device=str(args.pmf_device),
        physics_iq_downsample_factor=int(args.physics_iq_downsample_factor),
        physics_iq_threshold_value=int(args.physics_iq_threshold_value),
    )

    case_payload.update(metric_payload)
    save_json(case_json_path, case_payload)

    result_json_updated = False
    result_json_path = case_json_path.parent / "result.json"
    output_video = case_payload.get("output_video")
    if (
        not args.skip_result_json
        and isinstance(output_video, str)
        and output_video
        and result_json_path.is_file()
    ):
        result_json_updated = update_result_json(
            result_json_path=result_json_path,
            output_video=output_video,
            metric_payload=metric_payload,
        )

    print(
        json.dumps(
            {
                "case_json": str(case_json_path),
                "result_json": str(result_json_path) if result_json_path.is_file() else None,
                "result_json_updated": result_json_updated,
                "reference_video": str(source_video_path),
                "context_frames": context_frames,
                "scores": metric_payload["context_metric_summary"]["scores"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
