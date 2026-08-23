#!/usr/bin/env python3
"""Call Qwen's official API on original-plus-overlay side-by-side videos."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_qwen_api_context8_vs_full_0819 import (
    DEFAULT_API_BASE,
    DEFAULT_KEY_FILE,
    encode_frame_list,
    load_rows,
    read_api_key,
    request_caption,
    upsert_row,
)


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
SIDE_BY_SIDE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_side_by_side"
)
OUTPUT_PATH = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_side_by_side_api/demo_results.jsonl"
)
CONTEXT_PROMPT_PATH = Path(
    "/home/gaoya/Code_Video/Dataset_physv_v2v_0819/prompts/"
    "physv_context8_side_by_side_future_prediction_en.txt"
)
FULL_PROMPT_PATH = Path(
    "/home/gaoya/Code_Video/Dataset_physv_v2v_0819/prompts/"
    "physv_full_side_by_side_observed_continuation_en.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--side-by-side-root", type=Path, default=SIDE_BY_SIDE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--context-prompt", type=Path, default=CONTEXT_PROMPT_PATH)
    parser.add_argument("--full-prompt", type=Path, default=FULL_PROMPT_PATH)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--max-full-frames", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def load_side_by_side_manifest(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid side-by-side manifest: {path}")
    return {str(case["sample_id"]): case for case in cases}


def case_metadata(case_id: str, dataset_root: Path) -> tuple[Path, dict[str, Any]]:
    path = (
        dataset_root
        / "testjsons"
        / "v2v_jsons"
        / "physv_v2v_0819_all_cycles"
        / f"{case_id}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def make_row(
    *,
    case_id: str,
    metadata_path: Path,
    metadata: dict[str, Any],
    side_by_side_case: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": "physv_side_by_side_api_v1",
        "dataset": "physv_v2v_0819",
        "case_id": case_id,
        "case_json": str(metadata_path),
        "title": metadata.get("title"),
        "taxonomy": metadata.get("taxonomy"),
        "task_type": metadata.get("task_type"),
        "source_group": metadata.get("source_group"),
        "control": metadata.get("control"),
        "visual_input": {
            "layout": "left=original Cycles RGB; right=red trajectory + dynamic-object GT mask overlay",
            "source": "side-by-side rendered video",
            "side_by_side_manifest": str(args.side_by_side_root / "manifest.json"),
        },
        "source_basis": "Only the side-by-side video frames and prompt were sent to the model.",
        "caption_inputs_exclude": [
            "case metadata",
            "physics_supervision.npz",
            "contacts.json",
            "raw/trajectories.npz",
            "raw/masks.npz",
            "raw/depth.npz",
        ],
        "model": args.model,
        "api_provider": "DashScope OpenAI-compatible Chat Completions",
        "api_base": args.api_base,
        "context": {
            "video_rel": side_by_side_case["context8"]["relative_path"],
            "status": "pending",
        },
        "full": {
            "video_rel": side_by_side_case["source"]["relative_path"],
            "status": "pending",
        },
        "prompts": {
            "context_file": str(args.context_prompt),
            "full_file": str(args.full_prompt),
        },
        "status": "pending",
    }


def run_window(
    *,
    row: dict[str, Any],
    result_key: str,
    window: str,
    video_path: Path,
    prompt: str,
    args: argparse.Namespace,
    api_key: str,
) -> None:
    max_frames = 8 if window == "context8" else args.max_full_frames
    frame_urls, frame_metadata = encode_frame_list(
        video_path, max_frames=max_frames, jpeg_quality=args.jpeg_quality
    )
    result: dict[str, Any] = {
        "window": window,
        "video": str(video_path),
        "video_rel": row[result_key]["video_rel"],
        "video_request": {
            "mode": "ordered_jpeg_frame_list",
            "layout": "left=original Cycles RGB; right=trajectory + dynamic-object GT mask overlay",
            "fps_sent": args.fps,
            "max_pixels": args.max_pixels,
            **frame_metadata,
        },
        "prompt": prompt,
        "status": "pending",
    }
    print(
        f"requesting={window} decoded={frame_metadata['decoded_frame_count']} "
        f"sent={frame_metadata['sent_frame_count']} model={args.model}",
        flush=True,
    )
    result.update(
        request_caption(
            api_key=api_key,
            api_base=args.api_base,
            model=args.model,
            video_content={
                "type": "video",
                "video": frame_urls,
                "fps": args.fps,
                "max_pixels": args.max_pixels,
            },
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
    )
    row[result_key] = result
    print(
        f"{window}={result['status']} elapsed={result.get('request_elapsed_seconds')}s",
        flush=True,
    )
    if result["status"] == "ok":
        print(result.get("text", ""), flush=True)


def resolve_case_ids(args: argparse.Namespace, available: dict[str, dict[str, Any]]) -> list[str]:
    if args.all_cases and args.case_ids:
        raise ValueError("Use either --all-cases or --case-id, not both")
    case_ids = sorted(available) if args.all_cases else list(args.case_ids or ["v2v_gap_038"])
    for case_id in case_ids:
        if case_id not in available:
            raise ValueError(f"Missing side-by-side video for case: {case_id}")
    return case_ids


def main() -> int:
    args = parse_args()
    side_by_side_root = args.side_by_side_root.resolve()
    available = load_side_by_side_manifest(side_by_side_root)
    case_ids = resolve_case_ids(args, available)
    context_prompt = args.context_prompt.read_text(encoding="utf-8").strip()
    full_prompt = args.full_prompt.read_text(encoding="utf-8").strip()
    api_key = read_api_key(args.key_file)
    if len(api_key) < 20:
        raise ValueError("API key is unexpectedly short")

    existing = {row.get("case_id"): row for row in load_rows(args.output)}
    print(f"model={args.model}", flush=True)
    print(f"cases={len(case_ids)} input=side_by_side output={args.output}", flush=True)
    for number, case_id in enumerate(case_ids, start=1):
        metadata_path, metadata = case_metadata(case_id, args.dataset_root)
        side_case = available[case_id]
        row = existing.get(case_id) or make_row(
            case_id=case_id,
            metadata_path=metadata_path,
            metadata=metadata,
            side_by_side_case=side_case,
            args=args,
        )
        row["model"] = args.model
        row["api_base"] = args.api_base
        row["last_run_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        context_path = side_by_side_root / side_case["context8"]["relative_path"]
        full_path = side_by_side_root / side_case["source"]["relative_path"]
        print(f"[{number}/{len(case_ids)}] {case_id}", flush=True)
        run_window(
            row=row,
            result_key="context",
            window="context8",
            video_path=context_path,
            prompt=context_prompt,
            args=args,
            api_key=api_key,
        )
        row["status"] = "partial"
        upsert_row(args.output, row)
        run_window(
            row=row,
            result_key="full",
            window="full",
            video_path=full_path,
            prompt=full_prompt,
            args=args,
            api_key=api_key,
        )
        context_ok = row.get("context", {}).get("status") == "ok"
        full_ok = row.get("full", {}).get("status") == "ok"
        row["status"] = "ok" if context_ok and full_ok else "partial"
        upsert_row(args.output, row)
        existing[case_id] = row

    print(f"results={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
