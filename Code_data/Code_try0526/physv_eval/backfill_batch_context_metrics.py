from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physv_eval.backfill_case_context_metrics import (
    compute_four_metrics,
    load_json,
    resolve_context_frames,
    resolve_source_video,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-backfill four context-aware metrics into per-case JSON files and sibling "
            "result.json entries under one prediction root."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Prediction root containing per-step subdirectories with result.json files.",
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
        help="Optional explicit context frame count override.",
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
        "--limit",
        type=int,
        default=None,
        help="Optional number of cases to process, useful for spot checks.",
    )
    return parser.parse_args()


def iter_result_json_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/result.json") if path.is_file())


def update_case_and_entry(
    *,
    entry: dict[str, Any],
    result_dir: Path,
    artifacts_root: Path,
    explicit_context_frames: int | None,
    pmf_device: str,
    physics_iq_downsample_factor: int,
    physics_iq_threshold_value: int,
) -> dict[str, Any] | None:
    output_video = entry.get("output_video")
    if not isinstance(output_video, str) or not output_video:
        return None

    case_json_path = Path(output_video).with_suffix(".json")
    if not case_json_path.is_file():
        return None

    case_payload = load_json(case_json_path)
    source_video_path = resolve_source_video(case_payload, case_json_path)
    context_frames = resolve_context_frames(case_payload, explicit_context_frames)
    metric_payload = compute_four_metrics(
        case_payload=case_payload,
        case_json_path=case_json_path,
        artifacts_root=artifacts_root,
        source_video_path=source_video_path,
        context_frames=context_frames,
        pmf_device=pmf_device,
        physics_iq_downsample_factor=physics_iq_downsample_factor,
        physics_iq_threshold_value=physics_iq_threshold_value,
    )

    case_payload.update(metric_payload)
    save_json(case_json_path, case_payload)
    entry.update(metric_payload)
    return {
        "case_json": str(case_json_path),
        "result_dir": str(result_dir),
        "output_video": output_video,
        "reference_video": str(source_video_path),
        "context_frames": context_frames,
        "scores": metric_payload["context_metric_summary"]["scores"],
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    artifacts_root = args.artifacts_root.expanduser().resolve()
    result_paths = iter_result_json_paths(root)
    if not result_paths:
        raise FileNotFoundError(f"No result.json files found under {root}")

    processed_records: list[dict[str, Any]] = []
    missing_case_json: list[str] = []
    processed_count = 0

    for result_path in result_paths:
        result_payload = load_json(result_path)
        entries = result_payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"{result_path}: entries is not a list")

        updated_in_file = 0
        for entry in entries:
            if args.limit is not None and processed_count >= int(args.limit):
                break
            if not isinstance(entry, dict):
                continue
            output_video = entry.get("output_video")
            if not isinstance(output_video, str) or not output_video:
                continue
            case_json_path = Path(output_video).with_suffix(".json")
            if not case_json_path.is_file():
                missing_case_json.append(str(case_json_path))
                continue

            summary = update_case_and_entry(
                entry=entry,
                result_dir=result_path.parent,
                artifacts_root=artifacts_root,
                explicit_context_frames=args.context_frames,
                pmf_device=str(args.pmf_device),
                physics_iq_downsample_factor=int(args.physics_iq_downsample_factor),
                physics_iq_threshold_value=int(args.physics_iq_threshold_value),
            )
            if summary is None:
                continue
            processed_records.append(summary)
            processed_count += 1
            updated_in_file += 1
            print(
                f"[{processed_count}] {case_json_path} "
                f"piq_w={summary['scores']['physics_iq_with_context']:.2f} "
                f"piq_wo={summary['scores']['physics_iq_without_context']:.2f} "
                f"pmf_w={summary['scores']['pmf_with_context']:.6f} "
                f"pmf_wo={summary['scores']['pmf_without_context']:.6f}"
            )

        if updated_in_file > 0:
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
                    "entries_updated": updated_in_file,
                }
            )
            result_payload["context_metric_backfill"] = backfill
            save_json(result_path, result_payload)
            print(f"saved {result_path} updated_entries={updated_in_file}")

        if args.limit is not None and processed_count >= int(args.limit):
            break

    summary_payload = {
        "root": str(root),
        "artifacts_root": str(artifacts_root),
        "pmf_device": str(args.pmf_device),
        "context_frames_override": args.context_frames,
        "physics_iq_downsample_factor": int(args.physics_iq_downsample_factor),
        "physics_iq_threshold_value": int(args.physics_iq_threshold_value),
        "result_json_count": len(result_paths),
        "processed_count": processed_count,
        "missing_case_json_count": len(missing_case_json),
        "missing_case_json": missing_case_json,
        "records": processed_records,
    }
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
