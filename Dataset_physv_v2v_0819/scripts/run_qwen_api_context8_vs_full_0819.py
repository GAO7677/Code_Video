#!/usr/bin/env python3
"""Run Qwen official API captions on exact context/full frame lists.

The OpenAI-compatible DashScope API cannot see local paths.  This runner sends
an ordered list of JPEG data URLs as a video, which preserves the exact 8-frame
context and allows the full Cycles video to be represented by all decoded
frames.  API credentials are read from a file and are never written to the
result file or printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import requests


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_CASE = "v2v_gap_038"
DEFAULT_MODEL = "qwen3-vl-plus"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_KEY_FILE = Path("/home/gaoya/qwen_api.txt")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_context8_vs_full/demo_results.jsonl"
)
CONTEXT_PROMPT_PATH = Path(
    "/home/gaoya/Code_Video/Code_vlm_wan/prompts/physv_qwen_object_contact_geometry_en.txt"
)
FULL_PROMPT_PATH = Path(
    "/home/gaoya/Code_Video/Dataset_physv_v2v_0819/prompts/physv_full_observed_continuation_en.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--case-id", default=DEFAULT_CASE)
    parser.add_argument("--window", choices=("context8", "full", "both"), default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--context-prompt", type=Path, default=CONTEXT_PROMPT_PATH)
    parser.add_argument("--full-prompt", type=Path, default=FULL_PROMPT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--max-full-frames", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def read_api_key(path: Path) -> str:
    """Read a raw key or KEY=value file without exposing its value."""

    if not path.is_file():
        raise FileNotFoundError(path)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("bearer "):
            line = line[7:].strip()
        if "=" in line:
            name, value = line.split("=", 1)
            if name.strip().lower() in {
                "qwen_api_key",
                "dashscope_api_key",
                "chatgpt_dashscope_api_key",
                "api_key",
            }:
                line = value.strip()
        return line.strip().strip("'\"")
    raise ValueError(f"No API key found in {path}")


def case_paths(case_id: str, dataset_root: Path) -> tuple[Path, dict[str, Any], Path, Path]:
    if Path(case_id).name != case_id or case_id in {"", ".", ".."}:
        raise ValueError(f"Invalid case ID: {case_id!r}")
    metadata_path = (
        dataset_root
        / "testjsons"
        / "v2v_jsons"
        / "physv_v2v_0819_all_cycles"
        / f"{case_id}.json"
    )
    case = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_root = dataset_root / "samples" / case_id
    context_video = sample_root / "context" / "context8_cycles.mp4"
    full_video = sample_root / "videos" / "rgb_cycles.mp4"
    for path in (metadata_path, context_video, full_video):
        if not path.is_file():
            raise FileNotFoundError(path)
    return metadata_path, case, context_video, full_video


def _read_video_frames(video_path: Path) -> tuple[list[Any], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[Any] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No decodable frames in video: {video_path}")
    return frames, {
        "source_fps": fps,
        "declared_frame_count": declared_count,
        "decoded_frame_count": len(frames),
        "width": width,
        "height": height,
        "duration_seconds": round(len(frames) / fps, 6) if fps else None,
    }


def _uniform_select(frames: list[Any], max_frames: int) -> list[Any]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    indices = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frames[index] for index in indices]


def encode_frame_list(
    video_path: Path,
    *,
    max_frames: int,
    jpeg_quality: int,
) -> tuple[list[str], dict[str, Any]]:
    frames, metadata = _read_video_frames(video_path)
    selected = _uniform_select(frames, max_frames)
    urls: list[str] = []
    for frame in selected:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for {video_path}")
        urls.append("data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"))
    metadata = {
        **metadata,
        "sent_frame_count": len(urls),
        "frame_selection": "all_decoded_frames"
        if len(selected) == len(frames)
        else "uniform_subsample",
        "jpeg_quality": jpeg_quality,
    }
    return urls, metadata


def extract_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip() if content else ""


def request_caption(
    *,
    api_key: str,
    api_base: str,
    model: str,
    video_content: dict[str, Any],
    prompt: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [video_content, {"type": "text", "text": prompt}],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempts = max(1, retries + 1)
    last_error: dict[str, Any] | None = None
    started = time.time()
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(api_base, headers=headers, json=payload, timeout=timeout)
            try:
                response_json = response.json()
            except ValueError:
                response_json = {"raw": response.text[:2000]}
            if response.ok:
                choices = response_json.get("choices") or []
                finish_reason = choices[0].get("finish_reason") if choices else None
                return {
                    "status": "ok",
                    "text": extract_text(response_json),
                    "model_returned": response_json.get("model"),
                    "finish_reason": finish_reason,
                    "usage": response_json.get("usage"),
                    "request_elapsed_seconds": round(time.time() - started, 3),
                }
            last_error = {
                "status": "error",
                "http_status": response.status_code,
                "error": response_json,
                "attempt": attempt,
            }
            if response.status_code not in {408, 409, 429} and response.status_code < 500:
                break
        except requests.RequestException as exc:
            last_error = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "attempt": attempt,
            }
        if attempt < attempts:
            time.sleep(min(5 * attempt, 15))
    return {
        **(last_error or {"status": "error", "error": "unknown API error"}),
        "request_elapsed_seconds": round(time.time() - started, 3),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def upsert_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item for item in load_rows(path) if item.get("case_id") != row.get("case_id")]
    rows.append(row)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def make_row(metadata_path: Path, case: dict[str, Any], case_id: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "physv_context8_vs_full_api_v1",
        "dataset": "physv_v2v_0819",
        "source_basis": "RGB Cycles frames only; no metadata or physics truth was sent to the model",
        "caption_inputs_exclude": [
            "metadata.json",
            "physics_supervision.npz",
            "contacts.json",
            "raw/trajectories.npz",
            "raw/masks.npz",
            "raw/depth.npz",
        ],
        "case_id": case_id,
        "case_json": str(metadata_path),
        "title": case.get("title"),
        "taxonomy": case.get("taxonomy"),
        "task_type": case.get("task_type"),
        "source_group": case.get("source_group"),
        "control": case.get("control"),
        "video_variant": "cycles_pbr",
        "model": args.model,
        "api_provider": "DashScope OpenAI-compatible Chat Completions",
        "api_base": args.api_base,
        "context": {"video_rel": "context/context8_cycles.mp4", "status": "pending"},
        "full": {"video_rel": "videos/rgb_cycles.mp4", "status": "pending"},
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
    video_content = {"type": "video", "video": frame_urls, "fps": args.fps}
    result = {
        "window": window,
        "video": str(video_path),
        "video_rel": "context/context8_cycles.mp4"
        if window == "context8"
        else "videos/rgb_cycles.mp4",
        "video_request": {
            "mode": "ordered_jpeg_frame_list",
            "fps_sent": args.fps,
            **frame_metadata,
        },
        "prompt": prompt,
        "status": "pending",
    }
    print(
        f"requesting={window} decoded={frame_metadata['decoded_frame_count']} "
        f"sent={frame_metadata['sent_frame_count']} payload_frames=ordered_jpeg_list",
        flush=True,
    )
    result.update(
        request_caption(
            api_key=api_key,
            api_base=args.api_base,
            model=args.model,
            video_content=video_content,
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
    )
    row[result_key] = result
    if result["status"] == "ok":
        print(f"{window}=ok elapsed={result['request_elapsed_seconds']}s", flush=True)
        print(result.get("text", ""), flush=True)
    else:
        print(
            f"{window}=error http_status={result.get('http_status')} "
            f"error_type={result.get('error_type')} elapsed={result.get('request_elapsed_seconds')}s",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    metadata_path, case, context_video, full_video = case_paths(args.case_id, args.dataset_root)
    context_prompt = args.context_prompt.read_text(encoding="utf-8").strip()
    full_prompt = args.full_prompt.read_text(encoding="utf-8").strip()
    api_key = read_api_key(args.key_file)
    if len(api_key) < 20:
        raise ValueError("API key is unexpectedly short")

    existing = next((row for row in load_rows(args.output) if row.get("case_id") == args.case_id), None)
    row = existing or make_row(metadata_path, case, args.case_id, args)
    row["model"] = args.model
    row["api_base"] = args.api_base
    row["last_run_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if args.window in {"context8", "both"}:
        run_window(
            row=row,
            result_key="context",
            window="context8",
            video_path=context_video,
            prompt=context_prompt,
            args=args,
            api_key=api_key,
        )
        upsert_row(args.output, row)
    if args.window in {"full", "both"}:
        run_window(
            row=row,
            result_key="full",
            window="full",
            video_path=full_video,
            prompt=full_prompt,
            args=args,
            api_key=api_key,
        )
        upsert_row(args.output, row)

    context_ok = row.get("context", {}).get("status") == "ok"
    full_ok = row.get("full", {}).get("status") == "ok"
    row["status"] = "ok" if context_ok and full_ok else "partial"
    upsert_row(args.output, row)
    print(f"result_status={row['status']}", flush=True)
    print(f"results={args.output}", flush=True)
    return 0 if row["status"] in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
