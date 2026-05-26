#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from rerank_video.video_utils import ensure_dir, to_jsonable, write_json


DATASET_ROOT = Path("/data/gaoya/dataset/AnteaWu-PDI-Dataset")
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526")
RAW_RUN_ROOT = OUTPUT_ROOT / "runs" / "pdi_gt15_official_eval"
BENCHMARK_ROOT = OUTPUT_ROOT / "PDI-Bench"
BENCHMARK_OUTPUT_ROOT = BENCHMARK_ROOT / "output"
BENCHMARK_RESULT_ROOT = BENCHMARK_ROOT / "result"
METHOD_NAME = "GT"


def _extract(text: str, pattern: str, cast: type | None = None) -> Any:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return cast(value) if cast else value


def parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")
    return {
        "pdi_score": _extract(text, r"FINAL PDI SCORE:\s*([0-9.]+)", float),
        "grade": _extract(text, r"OVERALL GRADE:\s*(.+)"),
        "scale_component": _extract(text, r"Scale Component .*?:\s*([0-9.]+)", float),
        "traj_component": _extract(text, r"Trajectory Component .*?:\s*([0-9.]+)", float),
        "epsilon_rigidity": _extract(text, r"Epsilon Rigidity:\s*([0-9.]+)", float),
        "rigidity_strategy": _extract(text, r"Rigidity Strategy:\s*(.+)"),
        "vp_component": _extract(text, r"VP Component .*?:\s*([0-9.]+)", float),
        "ra_math_pass": _extract(text, r"RA Math Pass:\s*(True|False)"),
        "ra_mllm_success": _extract(text, r"RA MLLM Success:\s*(True|False)"),
        "ra_mllm_score": _extract(text, r"RA MLLM Score:\s*([0-9]+)", int),
        "ra_overall_pass": _extract(text, r"RA Overall Pass:\s*(True|False)"),
    }


def parse_bool(value: Any) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def load_metadata() -> dict[tuple[str, str], dict[str, str]]:
    metadata_path = DATASET_ROOT / "metadata.csv"
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["provider"] != "GT":
                continue
            task = row["task"]
            if task == "Partial_Occlusion":
                task = "partial_occlusion"
            rows[(task, row["prompt"])] = row
    return rows


def discover_gt_reports() -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for report_path in sorted(RAW_RUN_ROOT.glob("*/*/*_pdi_report.txt")):
        task = report_path.parent.parent.name
        clip_name = report_path.parent.name
        rows.append((task, clip_name, report_path))
    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 GT reports, found {len(rows)} under {RAW_RUN_ROOT}")
    return rows


def latest_summary_path() -> Path:
    candidates = sorted(RAW_RUN_ROOT.glob("pdi_results_gt_*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No GT summary file found under {RAW_RUN_ROOT}")
    return candidates[-1]


def copy_video(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def save_first_frame(src_video_path: Path, image_path: Path) -> Path:
    ensure_dir(image_path.parent)
    cap = cv2.VideoCapture(str(src_video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from {src_video_path}")
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.save(image_path)
    return image_path


def grade_letter(grade: str | None) -> str:
    if not grade:
        return ""
    return grade.split()[0]


def build_clip_payload(
    *,
    task: str,
    clip_name: str,
    metadata_row: dict[str, str],
    src_video_path: Path,
    dst_video_path: Path,
    first_frame_path: Path,
    report_path: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark": "PDI-Bench",
        "method": METHOD_NAME,
        "provider": "GT",
        "task": task,
        "clip_name": clip_name,
        "prompt": metadata_row["prompt"],
        "file_path": metadata_row["file_path"],
        "source": src_video_path,
        "first_frame": first_frame_path,
        "source_video_path": src_video_path,
        "copied_video_path": dst_video_path,
        "raw_report_path": report_path,
        "metrics": {
            **metrics,
            "grade_letter": grade_letter(metrics.get("grade")),
            "ra_math_pass_bool": parse_bool(metrics.get("ra_math_pass")),
            "ra_mllm_success_bool": parse_bool(metrics.get("ra_mllm_success")),
            "ra_overall_pass_bool": parse_bool(metrics.get("ra_overall_pass")),
        },
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def resolve_gt_video_path(task: str, clip_name: str, metadata_row: dict[str, str]) -> Path:
    candidate = DATASET_ROOT / "GT" / task / f"{clip_name}.mp4"
    if candidate.is_file():
        return candidate
    metadata_path = DATASET_ROOT / metadata_row["file_path"]
    if metadata_path.is_file():
        return metadata_path
    raise FileNotFoundError(f"Unable to resolve GT video for task={task}, clip={clip_name}")


def export_gt() -> None:
    metadata_index = load_metadata()
    report_rows = discover_gt_reports()
    summary_path = latest_summary_path()

    method_output_root = BENCHMARK_OUTPUT_ROOT / METHOD_NAME
    ensure_dir(method_output_root)
    ensure_dir(BENCHMARK_RESULT_ROOT)

    per_clip_rows: list[dict[str, Any]] = []
    for task, clip_name, report_path in report_rows:
        metadata_row = metadata_index.get((task, clip_name))
        if metadata_row is None:
            raise KeyError(f"Missing metadata row for GT clip: task={task}, clip={clip_name}")

        src_video_path = resolve_gt_video_path(task, clip_name, metadata_row)
        dst_dir = method_output_root / task
        dst_video_path = dst_dir / f"{clip_name}.mp4"
        dst_json_path = dst_dir / f"{clip_name}.json"
        first_frame_path = dst_dir / f"{clip_name}.first_frame.png"

        metrics = parse_report(report_path)
        copy_video(src_video_path, dst_video_path)
        save_first_frame(src_video_path, first_frame_path)
        payload = build_clip_payload(
            task=task,
            clip_name=clip_name,
            metadata_row=metadata_row,
            src_video_path=src_video_path,
            dst_video_path=dst_video_path,
            first_frame_path=first_frame_path,
            report_path=report_path,
            metrics=metrics,
        )
        write_json(dst_json_path, payload)
        per_clip_rows.append(payload)

    pdi_scores = [float(row["metrics"]["pdi_score"]) for row in per_clip_rows]
    scale_scores = [float(row["metrics"]["scale_component"]) for row in per_clip_rows]
    traj_scores = [float(row["metrics"]["traj_component"]) for row in per_clip_rows]
    rigid_scores = [float(row["metrics"]["epsilon_rigidity"]) for row in per_clip_rows]
    vp_scores = [float(row["metrics"]["vp_component"]) for row in per_clip_rows]
    grade_counts = Counter(grade_letter(str(row["metrics"]["grade"])) for row in per_clip_rows)
    ra_math_pass = sum(1 for row in per_clip_rows if row["metrics"]["ra_math_pass_bool"] is True)
    ra_overall_pass = sum(1 for row in per_clip_rows if row["metrics"]["ra_overall_pass_bool"] is True)

    result_row = {
        "benchmark": "PDI-Bench",
        "method": METHOD_NAME,
        "provider": "GT",
        "num_videos": len(per_clip_rows),
        "mean_pdi_score": f"{mean(pdi_scores):.6f}",
        "mean_scale_component": f"{mean(scale_scores):.6f}",
        "mean_traj_component": f"{mean(traj_scores):.6f}",
        "mean_epsilon_rigidity": f"{mean(rigid_scores):.6f}",
        "mean_vp_component": f"{mean(vp_scores):.6f}",
        "grade_A_count": grade_counts.get("A", 0),
        "grade_B_count": grade_counts.get("B", 0),
        "grade_C_count": grade_counts.get("C", 0),
        "ra_math_pass_count": ra_math_pass,
        "ra_overall_pass_count": ra_overall_pass,
        "summary_report_path": str(summary_path),
    }

    result_csv_path = BENCHMARK_RESULT_ROOT / "metrics.csv"
    with result_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_row.keys()))
        writer.writeheader()
        writer.writerow(result_row)

    manifest_path = BENCHMARK_ROOT / "manifest_gt.json"
    write_json(
        manifest_path,
        {
            "benchmark": "PDI-Bench",
            "methods_exported": [METHOD_NAME],
            "output_root": BENCHMARK_OUTPUT_ROOT,
            "result_csv": result_csv_path,
            "raw_run_root": RAW_RUN_ROOT,
            "raw_summary_path": summary_path,
            "clips": [
                {
                    "task": row["task"],
                    "clip_name": row["clip_name"],
                    "copied_video_path": row["copied_video_path"],
                    "json_path": str((method_output_root / row["task"] / f"{row['clip_name']}.json")),
                }
                for row in per_clip_rows
            ],
        },
    )

    print(
        json.dumps(
            to_jsonable(
                {
                    "benchmark_root": BENCHMARK_ROOT,
                    "result_csv": result_csv_path,
                    "num_exported_clips": len(per_clip_rows),
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    export_gt()
