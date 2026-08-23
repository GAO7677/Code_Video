#!/usr/bin/env python3
"""Caption all test70 full Cycles videos and write qwen38vl_caption to JSON."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from run_qwen_api_context8_vs_full_0819 import (
    DEFAULT_API_BASE,
    DEFAULT_KEY_FILE,
    encode_frame_list,
    read_api_key,
    request_caption,
)


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_LIST = DATASET_ROOT / "testjsons/physv_v2v_0819_all_cycles_test70_ctx8.txt"
DEFAULT_PROMPT = Path(
    "/home/gaoya/Code_Video/Dataset_physv_v2v_0819/prompts/describe_this_video.txt"
)
DEFAULT_LOG = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_qwen38vl_caption_test70/run_log.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-list", type=Path, default=DEFAULT_LIST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def load_json_paths(path: Path) -> list[Path]:
    paths = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths) != 70:
        raise ValueError(f"Expected 70 JSON paths in {path}, got {len(paths)}")
    for json_path in paths:
        if not json_path.is_file():
            raise FileNotFoundError(json_path)
    return paths


def main() -> int:
    args = parse_args()
    json_paths = load_json_paths(args.input_list)
    if args.limit:
        json_paths = json_paths[: args.limit]
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    api_key = read_api_key(args.key_file)
    if len(api_key) < 20:
        raise ValueError("API key is unexpectedly short")

    print(f"cases={len(json_paths)} model={args.model}", flush=True)
    print(f"prompt={args.prompt}", flush=True)
    print("input=source_video (full rgb_cycles.mp4)", flush=True)
    for number, json_path in enumerate(json_paths, start=1):
        record = json.loads(json_path.read_text(encoding="utf-8"))
        case_id = str(record.get("sample_id") or json_path.stem)
        if record.get("qwen38vl_caption") and not args.overwrite:
            print(f"[{number}/{len(json_paths)}] {case_id} skipped=existing", flush=True)
            continue
        video_path = Path(str(record.get("source_video", "")))
        if not video_path.is_file():
            raise FileNotFoundError(f"{case_id}: {video_path}")

        started = time.time()
        try:
            frame_urls, frame_metadata = encode_frame_list(
                video_path,
                max_frames=args.max_frames,
                jpeg_quality=args.jpeg_quality,
            )
            print(
                f"[{number}/{len(json_paths)}] {case_id} "
                f"requesting frames={frame_metadata['sent_frame_count']}",
                flush=True,
            )
            result = request_caption(
                api_key=api_key,
                api_base=args.api_base,
                model=args.model,
                video_content={
                    "type": "video",
                    "video": frame_urls,
                    "fps": args.fps,
                },
                prompt=prompt,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
            if result.get("status") != "ok":
                raise RuntimeError(json.dumps(result, ensure_ascii=False))
            record["qwen38vl_caption"] = result["text"]
            atomic_write_json(json_path, record)
            log_record = {
                "case_id": case_id,
                "json_path": str(json_path),
                "video_path": str(video_path),
                "status": "ok",
                "prompt": prompt,
                "model": args.model,
                "frame_metadata": frame_metadata,
                "usage": result.get("usage"),
                "request_elapsed_seconds": result.get("request_elapsed_seconds"),
                "total_elapsed_seconds": round(time.time() - started, 3),
            }
            append_log(args.log, log_record)
            print(
                f"[{number}/{len(json_paths)}] {case_id} saved="
                f"{json_path} elapsed={result.get('request_elapsed_seconds')}s",
                flush=True,
            )
        except Exception as exc:
            log_record = {
                "case_id": case_id,
                "json_path": str(json_path),
                "video_path": str(video_path),
                "status": "error",
                "error": str(exc),
                "total_elapsed_seconds": round(time.time() - started, 3),
            }
            append_log(args.log, log_record)
            print(f"[{number}/{len(json_paths)}] {case_id} error={exc}", flush=True)
            if "Arrearage" in str(exc) or "overdue-payment" in str(exc):
                print("fatal API billing error; stopping batch", flush=True)
                return 2

    print(f"log={args.log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
