#!/usr/bin/env python3
"""Combine full-video and first-8-frame Qwen JSONL rows for the compare viewer."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("video_metadata")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
        except (SyntaxError, ValueError):
            pass
    return {}


def variant(row: dict[str, Any], label: str, source: str, frame_count: int | None) -> dict[str, Any]:
    info = metadata(row)
    result = {
        "label": label,
        "frame_count": frame_count,
        "video": row.get("video"),
        "video_info": {"shape": [len(info.get("frames_indices", [])), 3, 360, 640]},
        "response_raw": row.get("raw_output", row.get("caption", "")),
        "response_final": row.get("caption", ""),
        "status": row.get("status", "unknown"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "thinking_disabled": True,
        "source": source,
        "video_request": row.get("video_request"),
        "video_metadata": info,
        "physical_gpu": row.get("physical_gpu"),
    }
    if row.get("error"):
        result["error"] = row["error"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full = {row["case_id"]: row for row in read_rows(args.full)}
    prefixes = {row["case_id"].removesuffix("_first8"): row for row in read_rows(args.prefix)}
    case_ids = [case_id for case_id in full if case_id in prefixes]
    if not case_ids:
        raise ValueError("No overlapping case IDs")
    if any(full[case_id].get("status") != "ok" or prefixes[case_id].get("status") != "ok" for case_id in case_ids):
        raise ValueError("Comparison rows must all be successful")

    rows = []
    for case_id in case_ids:
        full_row = full[case_id]
        prefix_row = prefixes[case_id]
        rows.append(
            {
                "dataset": "physv_v2v_0819",
                "case_id": case_id,
                "source_video": full_row.get("video"),
                "question": full_row.get("prompt", ""),
                "prompt_source": "/home/gaoya/Code_Video/Code_vlm_wan/prompts/physv_qwen_object_contact_geometry_en.txt",
                "variants": {
                    "prefix_8": variant(prefix_row, "前 8 帧", "generated_first8_frames", 8),
                    "full": variant(full_row, "完整视频", "generated_full_video", len(metadata(full_row).get("frames_indices", []))),
                },
                "frame_counts": [8],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"comparison={args.output} cases={len(rows)} variants_per_case=2")


if __name__ == "__main__":
    main()
