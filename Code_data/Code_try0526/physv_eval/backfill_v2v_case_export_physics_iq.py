from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .single_case.physics_iq import score_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill single-view approximate Physics-IQ scores into "
            "v2v_case_export_test5 metrics.json files and summarize per-method averages."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing per-case folders with metrics.json.",
    )
    parser.add_argument(
        "--tmp-output-root",
        type=Path,
        default=Path("/tmp/gaoya/physics_iq_single_case/v2v_case_export_test5_backfill"),
        help="Temporary directory used to store aligned scoring videos.",
    )
    parser.add_argument("--threshold-value", type=int, default=10)
    parser.add_argument("--downsample-factor", type=int, default=4)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_compact(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with path.open("r+", encoding="utf-8") as handle:
        handle.seek(0)
        handle.write(text)
        handle.truncate()


def _resolve_source_video(metrics_path: Path, payload: dict[str, Any]) -> Path:
    input_json_value = payload.get("input_json")
    if not isinstance(input_json_value, str) or not input_json_value:
        raise ValueError(f"{metrics_path}: missing input_json")
    input_json = Path(input_json_value)
    input_payload = _load_json(input_json)
    source_value = input_payload.get("source_video")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError(f"{input_json}: missing source_video")
    source_video = Path(source_value)
    if not source_video.is_file():
        raise FileNotFoundError(f"source_video not found: {source_video}")
    return source_video


def _build_aligned_dir(tmp_root: Path, case_dir: Path, method: str) -> Path:
    return tmp_root / case_dir.name / method


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    tmp_root = args.tmp_output_root.resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)

    metrics_paths = sorted(path for path in root.glob("*/metrics.json") if path.is_file())
    print(f"Found {len(metrics_paths)} metrics.json files under {root}")

    basename_scores: dict[str, list[float]] = defaultdict(list)
    case_summaries: list[dict[str, Any]] = []

    for metrics_index, metrics_path in enumerate(metrics_paths, start=1):
        print(f"[{metrics_index}/{len(metrics_paths)}] processing {metrics_path}")
        payload = _load_json(metrics_path)
        case_dir = metrics_path.parent
        source_video = _resolve_source_video(metrics_path, payload)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"{metrics_path}: rows is not a list")

        scored_rows = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            copied_video_value = row.get("copied_video")
            method_value = row.get("method")
            if not isinstance(copied_video_value, str) or not copied_video_value:
                continue
            if not isinstance(method_value, str) or not method_value:
                continue
            copied_video = Path(copied_video_value)
            if not copied_video.is_file():
                continue

            aligned_dir = _build_aligned_dir(tmp_root, case_dir, method_value)
            result = score_case(
                str(copied_video),
                source_video_path=str(source_video),
                threshold_value=args.threshold_value,
                downsample_factor=args.downsample_factor,
                aligned_video_dir=aligned_dir,
            )

            row["physics_iq_score"] = result.get("physics_iq_score")
            row["physics_iq_official"] = result.get("official")
            row["physics_iq_method"] = result.get("method")
            row["physics_iq_reference_video"] = result.get("reference_video")
            row["physics_iq_scored_output_video"] = result.get("scored_output_video")
            row["physics_iq_scored_source_video"] = result.get("scored_source_video")
            row["physics_iq_video_codec"] = result.get("video_codec")
            row["physics_iq_mse_mean"] = result.get("mse_mean")
            row["physics_iq_spatiotemporal_iou_mean"] = result.get("spatiotemporal_iou_mean")
            row["physics_iq_spatial_iou"] = result.get("spatial_iou")
            row["physics_iq_weighted_spatial_iou"] = result.get("weighted_spatial_iou")
            row["physics_iq_raw_score"] = result.get("raw_score")
            row["physics_iq_num_frames_compared"] = result.get("num_frames_compared")
            row["physics_iq_compare_duration_sec"] = result.get("compare_duration_sec")
            row["physics_iq_compare_fps"] = result.get("compare_fps")
            row["physics_iq_output_fps"] = result.get("output_fps")
            row["physics_iq_source_fps"] = result.get("source_fps")
            row["physics_iq_output_duration_sec"] = result.get("output_duration_sec")
            row["physics_iq_source_duration_sec"] = result.get("source_duration_sec")
            row["physics_iq_target_size"] = result.get("target_size")
            row["physics_iq_downsample_factor"] = result.get("downsample_factor")
            row["physics_iq_threshold_value"] = result.get("threshold_value")
            row["physics_iq_frame_alignment"] = result.get("frame_alignment")

            score_value = result.get("physics_iq_score")
            if isinstance(score_value, (int, float)):
                basename_scores[method_value].append(float(score_value))
                scored_rows += 1

        payload["physics_iq_backfill"] = {
            "metric_name": "physics_iq_single_view_approx",
            "source_video": str(source_video),
            "tmp_output_root": str(tmp_root / case_dir.name),
            "threshold_value": args.threshold_value,
            "downsample_factor": args.downsample_factor,
            "num_rows_scored": scored_rows,
        }
        _write_json_compact(metrics_path, payload)
        case_summaries.append(
            {
                "case_dir": str(case_dir),
                "metrics_json": str(metrics_path),
                "source_video": str(source_video),
                "num_rows": len(rows),
                "num_rows_scored": scored_rows,
            }
        )

    basename_summary = []
    for method_name, scores in basename_scores.items():
        avg_score = sum(scores) / len(scores)
        basename_summary.append(
            {
                "method": method_name,
                "count": len(scores),
                "avg_physics_iq_score": round(avg_score, 6),
                "max_physics_iq_score": round(max(scores), 6),
                "min_physics_iq_score": round(min(scores), 6),
            }
        )
    basename_summary.sort(key=lambda item: item["avg_physics_iq_score"], reverse=True)

    summary_payload = {
        "root": str(root),
        "num_cases": len(metrics_paths),
        "tmp_output_root": str(tmp_root),
        "threshold_value": args.threshold_value,
        "downsample_factor": args.downsample_factor,
        "case_summaries": case_summaries,
        "method_average_ranking": basename_summary,
        "best_method": basename_summary[0] if basename_summary else None,
    }
    summary_path = tmp_root / "physics_iq_backfill_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote summary: {summary_path}")
    if basename_summary:
        top = basename_summary[0]
        print(
            "Best method by average Physics-IQ score: "
            f"{top['method']} avg={top['avg_physics_iq_score']:.6f} count={top['count']}"
        )


if __name__ == "__main__":
    main()
