#!/usr/bin/env python3
"""Merge existing FPS-20 and newly generated FPS-15 full-video runs for viewing."""

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any


OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/physv_qwen3vl")
DEFAULT_COMPARISON = OUTPUT_ROOT / "0613_phyco_frame_compare_chinese_prompt.jsonl"
DEFAULT_FPS20 = OUTPUT_ROOT / "0613_phyco_chinese_caption_prompt.jsonl"
DEFAULT_FPS15 = OUTPUT_ROOT / "0613_phyco_chinese_caption_prompt_fps15.jsonl"
DEFAULT_OUTPUT = OUTPUT_ROOT / "0613_phyco_frame_compare_chinese_prompt_fps15.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--fps20-results", type=Path, default=DEFAULT_FPS20)
    parser.add_argument("--fps15-results", type=Path, default=DEFAULT_FPS15)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomically(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} has a row without a case_id")
        if case_id in indexed:
            raise ValueError(f"{label} has duplicate case_id: {case_id}")
        indexed[case_id] = row
    return indexed


def require_equal(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise ValueError(f"Mismatch for {label}: {left!r} != {right!r}")


def normalized_text(value: Any) -> str:
    return "".join(str(value or "").split())


def require_video_params(row: dict[str, Any], label: str) -> dict[str, Any]:
    params = row.get("video_params")
    if not isinstance(params, dict):
        raise ValueError(f"Missing video_params for {label}")
    for key in ("fps", "max_frames", "max_pixels"):
        if key not in params:
            raise ValueError(f"Missing video_params.{key} for {label}")
    return params


def full_variant(row: dict[str, Any], label: str, source: str) -> dict[str, Any]:
    if row.get("status") != "ok":
        raise ValueError(f"Cannot merge failed result for {row.get('case_id')}: {row.get('status')}")
    return {
        "label": label,
        "frame_count": None,
        "video": row["video"],
        "video_params": copy.deepcopy(row["video_params"]),
        "video_info": copy.deepcopy(row["video_info"]),
        "response_raw": row.get("response_raw"),
        "response_final": row.get("response_final"),
        "status": row["status"],
        "elapsed_seconds": row.get("elapsed_seconds"),
        "thinking_disabled": row.get("thinking_disabled"),
        "source": source,
    }


def main() -> None:
    args = parse_args()
    paths = (args.comparison, args.fps20_results, args.fps15_results)
    if args.output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("Output must be a new derived comparison file")

    comparison_rows = load_jsonl(args.comparison)
    fps20_rows = load_jsonl(args.fps20_results)
    fps15_rows = load_jsonl(args.fps15_results)
    fps20_by_case = index_rows(fps20_rows, "FPS-20 results")
    fps15_by_case = index_rows(fps15_rows, "FPS-15 results")
    comparison_by_case = index_rows(comparison_rows, "comparison results")

    expected_cases = set(comparison_by_case)
    require_equal(set(fps20_by_case), expected_cases, "FPS-20 case IDs")
    require_equal(set(fps15_by_case), expected_cases, "FPS-15 case IDs")

    merged_rows = copy.deepcopy(comparison_rows)
    for row in merged_rows:
        case_id = row["case_id"]
        fps20 = fps20_by_case[case_id]
        fps15 = fps15_by_case[case_id]
        variants = row.get("variants")
        if not isinstance(variants, dict) or "full" not in variants:
            raise ValueError(f"Comparison row is missing full variant: {case_id}")

        for key in ("dataset", "video", "question", "thinking_disabled"):
            require_equal(fps20.get(key), fps15.get(key), f"{case_id} {key}")
        require_equal(row.get("dataset"), fps20.get("dataset"), f"{case_id} comparison dataset")
        require_equal(row.get("source_video"), fps20.get("video"), f"{case_id} comparison video")
        require_equal(row.get("question"), fps20.get("question"), f"{case_id} comparison question")

        fps20_params = require_video_params(fps20, f"FPS-20 {case_id}")
        fps15_params = require_video_params(fps15, f"FPS-15 {case_id}")
        require_equal(int(fps20_params["max_frames"]), int(fps15_params["max_frames"]), f"{case_id} max_frames")
        require_equal(int(fps20_params["max_pixels"]), int(fps15_params["max_pixels"]), f"{case_id} max_pixels")
        require_equal(float(fps20_params["fps"]), 20.0, f"{case_id} FPS-20 target")
        require_equal(float(fps15_params["fps"]), 15.0, f"{case_id} FPS-15 target")

        existing_full = variants["full"]
        require_equal(existing_full.get("video"), fps20.get("video"), f"{case_id} existing full video")
        require_equal(existing_full.get("status"), "ok", f"{case_id} existing full status")
        if normalized_text(existing_full.get("response_raw")) != normalized_text(fps20.get("response_raw")):
            raise ValueError(f"Mismatch for {case_id} existing full raw output")
        if normalized_text(existing_full.get("response_final")) != normalized_text(fps20.get("response_final")):
            raise ValueError(f"Mismatch for {case_id} existing full final output")
        # Preserve the prior page's audited FPS-20 response and replay byte-for-byte.
        existing_full["label"] = "完整视频 / FPS 20"
        existing_full["source"] = "full_video_fps20"
        existing_full["video_params"] = copy.deepcopy(fps20_params)
        variants["full_fps15"] = full_variant(fps15, "完整视频 / FPS 15", "full_video_fps15")
        row["fps_comparison"] = {
            "full_fps20_variant": "full",
            "full_fps15_variant": "full_fps15",
            "changed_parameter": "video_params.fps",
        }

    write_jsonl_atomically(args.output, merged_rows)
    print(f"cases={len(merged_rows)}")
    print("full_video_fps20=20")
    print("full_video_fps15=15")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
