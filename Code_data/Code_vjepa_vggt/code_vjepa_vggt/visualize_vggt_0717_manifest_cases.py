#!/usr/bin/env python3
"""Run VGGT on a small 0717 PyBullet manifest subset and build a video gallery.

The script intentionally works directly from the 0717 ``manifest.json`` rather
than constructing a training dataset.  It selects the first N records that
belong to the dataset's stable-hash train split, reads the prefix frames used
by training, runs the official VGGT adapter, and writes compact MP4 previews
for tracks, depth, and world points.

Two independent processes can run this script safely at the same time:

  --mode context8  --frame-count 8
  --mode prefix49  --frame-count 49

After both jobs finish, invoke ``--build-gallery`` to create ``index.html``.
Large outputs belong under /data/gaoya/agent-data rather than the code tree.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu

from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter


DEFAULT_MANIFEST = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5/manifest.json"
)
DEFAULT_MODEL = Path("/data/gaoya/ckpt/facebook-VGGT-1B")
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/vggt_0717_train10_context8_prefix49"
)
DEFAULT_INPUT_HW = (420, 728)
PALETTE = (
    (255, 82, 82),
    (55, 214, 119),
    (66, 156, 255),
    (255, 196, 58),
    (208, 96, 255),
    (40, 220, 220),
    (255, 136, 68),
    (180, 180, 180),
)


def stable_split(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    if value < 0.90:
        return "train"
    if value < 0.95:
        return "val"
    return "test"


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_train_rows(manifest_path: Path, *, split: str, limit: int) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError(f"manifest must contain a list: {manifest_path}")
    selected: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for raw_row in manifest:
        if not isinstance(raw_row, dict):
            continue
        family = str(raw_row.get("family_key", "")).strip()
        case_id = str(raw_row.get("case_id", "")).strip()
        video = Path(str(raw_row.get("video", "")))
        if not family or not case_id or not video.is_file():
            continue
        row_split = stable_split(f"{family}/{case_id}")
        split_counts[row_split] = split_counts.get(row_split, 0) + 1
        if row_split != split:
            continue
        row = dict(raw_row)
        row["stable_split"] = row_split
        row["stable_key"] = f"{family}/{case_id}"
        selected.append(row)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise RuntimeError(
            f"only found {len(selected)} usable {split} rows in {manifest_path}; "
            f"split counts before truncation={split_counts}"
        )
    return selected


def read_prefix(video_path: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    reader = VideoReader(str(video_path), ctx=cpu(0))
    source_count = int(len(reader))
    if source_count < frame_count:
        raise ValueError(f"{video_path} has {source_count} frames, requested {frame_count}")
    indices = np.arange(frame_count, dtype=np.int64)
    frames = np.asarray(reader.get_batch(indices).asnumpy(), dtype=np.uint8)
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    return frames, indices, fps, source_count


def write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] uint8 frames, got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def add_header(frames: np.ndarray, title: str, context_frames: int) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.uint8).copy()
    bar_height = 38
    output = np.zeros(
        (frames.shape[0], frames.shape[1] + bar_height, frames.shape[2], 3),
        dtype=np.uint8,
    )
    output[:, bar_height:] = frames
    for frame_id in range(frames.shape[0]):
        output[frame_id, :bar_height] = (22, 26, 35)
        role = "CONTEXT" if frame_id < context_frames else "49F PREFIX"
        text = f"{title} | frame {frame_id:02d}/{frames.shape[0] - 1:02d} | {role}"
        cv2.putText(
            output[frame_id],
            text,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (238, 243, 250),
            1,
            cv2.LINE_AA,
        )
    return output


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float32)
    low = float(np.nanpercentile(values[valid], 5.0))
    high = float(np.nanpercentile(values[valid], 95.0))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.nanmin(values[valid]))
        high = low + 1.0
    return np.clip((values - low) / max(high - low, 1.0e-6), 0.0, 1.0)


def colorize_depth(depth_thw: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    normalized = robust_normalize(depth_thw)
    rendered: list[np.ndarray] = []
    target_h, target_w = output_hw
    for frame in normalized:
        bgr = cv2.applyColorMap((frame * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (target_h, target_w):
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rendered.append(rgb)
    return np.stack(rendered, axis=0)


def colorize_world_points(world_points_thwc: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    points = np.asarray(world_points_thwc, dtype=np.float32)
    channels: list[np.ndarray] = []
    for channel in range(3):
        if channel < points.shape[-1]:
            channels.append(robust_normalize(points[..., channel]))
        else:
            channels.append(np.zeros(points.shape[:-1], dtype=np.float32))
    rgb = np.stack(channels, axis=-1)
    target_h, target_w = output_hw
    rendered: list[np.ndarray] = []
    for frame in rgb:
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rendered.append(np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8))
    return np.stack(rendered, axis=0)


def _as_numpy(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    array = value.detach().float().cpu().numpy()
    while array.ndim > 0 and array.shape[-1] == 1:
        array = array[..., 0]
    return np.asarray(array)


def render_track_overlay(
    frames: np.ndarray,
    tracks_tqx2: np.ndarray,
    visibility_tq: np.ndarray,
    model_hw: tuple[int, int],
    context_frames: int,
) -> np.ndarray:
    raw = np.asarray(frames, dtype=np.uint8)
    tracks = np.asarray(tracks_tqx2, dtype=np.float32)
    visibility = np.asarray(visibility_tq, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"unexpected track shape {tracks.shape}")
    if visibility.ndim != 2:
        raise ValueError(f"unexpected visibility shape {visibility.shape}")
    frame_count = min(int(raw.shape[0]), int(tracks.shape[0]))
    query_count = int(tracks.shape[1])
    model_h, model_w = model_hw
    raw_h, raw_w = int(raw.shape[1]), int(raw.shape[2])
    scale_x = raw_w / max(float(model_w), 1.0)
    scale_y = raw_h / max(float(model_h), 1.0)
    output: list[np.ndarray] = []
    for frame_id in range(frame_count):
        image = raw[frame_id].copy()
        for query_id in range(query_count):
            color = PALETTE[query_id % len(PALETTE)]
            points: list[tuple[int, int]] = []
            for previous_id in range(frame_id + 1):
                if previous_id >= tracks.shape[0] or query_id >= visibility.shape[1]:
                    continue
                point = tracks[previous_id, query_id]
                visible = float(visibility[previous_id, query_id]) >= 0.5
                if visible and np.all(np.isfinite(point)):
                    points.append(
                        (
                            int(round(float(point[0]) * scale_x)),
                            int(round(float(point[1]) * scale_y)),
                        )
                    )
            for left, right in zip(points[:-1], points[1:]):
                cv2.line(
                    image,
                    left,
                    right,
                    (int(color[2]), int(color[1]), int(color[0])),
                    2,
                    cv2.LINE_AA,
                )
            if points:
                x, y = points[-1]
                cv2.circle(
                    image,
                    (x, y),
                    6,
                    (int(color[2]), int(color[1]), int(color[0])),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"q{query_id}",
                    (x + 8, max(y - 7, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (int(color[2]), int(color[1]), int(color[0])),
                    1,
                    cv2.LINE_AA,
                )
        output.append(image)
    rendered = np.stack(output, axis=0)
    return add_header(rendered, "VGGT tracks", context_frames)


def normalize_prediction_shape(value: np.ndarray | None, *, kind: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if kind == "depth":
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 3:
            return None
    elif kind == "world":
        if array.ndim != 4 or array.shape[-1] < 3:
            return None
    return np.asarray(array)


def model_forward(
    adapter: VGGTTrackAdapter,
    frames: np.ndarray,
    *,
    device: torch.device,
    autocast_bf16: bool,
) -> Any:
    input_tensor = torch.from_numpy(frames).to(device=device, dtype=torch.float32).div_(255.0)
    input_tensor = input_tensor.unsqueeze(0)
    try:
        with torch.inference_mode():
            if autocast_bf16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return adapter(input_tensor)
            return adapter(input_tensor)
    finally:
        del input_tensor


def run_one_case(
    adapter: VGGTTrackAdapter,
    row: dict[str, Any],
    *,
    index: int,
    mode: str,
    frame_count: int,
    output_root: Path,
    device: torch.device,
    autocast_bf16: bool,
    overwrite: bool,
) -> dict[str, Any]:
    case_id = str(row["case_id"])
    case_dir = output_root / mode / "cases" / f"{index:02d}_{safe_name(case_id)}"
    result_path = case_dir / "result.json"
    if result_path.is_file() and not overwrite:
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("status") == "ok":
                print(f"[{mode}] skip {index + 1:02d}/10 {case_id} (already complete)", flush=True)
                return cached
        except Exception:
            pass

    case_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(str(row["video"]))
    try:
        frames, frame_indices, fps, source_count = read_prefix(video_path, frame_count)
        output = model_forward(
            adapter,
            frames,
            device=device,
            autocast_bf16=autocast_bf16,
        )
        tracks = _as_numpy(output.tracks)
        visibility = _as_numpy(output.visibility)
        confidence = _as_numpy(output.confidence)
        tracks = None if tracks is None else tracks[0] if tracks.shape[0] == 1 else tracks
        visibility = None if visibility is None else visibility[0] if visibility.shape[0] == 1 else visibility
        confidence = None if confidence is None else confidence[0] if confidence.shape[0] == 1 else confidence
        if tracks is None or visibility is None:
            raise RuntimeError("VGGT returned no tracks or visibility")

        input_video = add_header(frames, f"{case_id} | {mode}", 8)
        input_video_path = case_dir / "input_window.mp4"
        write_mp4(input_video_path, input_video, fps)
        tracks_video = render_track_overlay(
            frames,
            tracks,
            visibility,
            tuple(int(value) for value in output.image_hw),
            8,
        )
        tracks_video_path = case_dir / "vggt_tracks.mp4"
        write_mp4(tracks_video_path, tracks_video, fps)

        depth = normalize_prediction_shape(_as_numpy(output.depth), kind="depth")
        world_points = normalize_prediction_shape(_as_numpy(output.world_points), kind="world")
        depth_video_path: Path | None = None
        world_video_path: Path | None = None
        if depth is not None:
            depth_video_path = case_dir / "vggt_depth.mp4"
            write_mp4(depth_video_path, colorize_depth(depth, frames.shape[1:3]), fps)
        if world_points is not None:
            world_video_path = case_dir / "vggt_world_points.mp4"
            write_mp4(world_video_path, colorize_world_points(world_points, frames.shape[1:3]), fps)

        payload: dict[str, Any] = {
            "status": "ok",
            "mode": mode,
            "frame_count": int(frame_count),
            "context_frames": 8,
            "frame_indices": [int(value) for value in frame_indices.tolist()],
            "source_frames": int(source_count),
            "fps": float(fps),
            "source_video": str(video_path),
            "case_id": case_id,
            "family_key": str(row.get("family_key", "")),
            "caption": str(row.get("caption", "")),
            "stable_split": str(row.get("stable_split", "train")),
            "stable_key": str(row.get("stable_key", "")),
            "model_path": str(adapter.model_path),
            "vggt_input_hw": [int(value) for value in output.image_hw],
            "tracks_shape": [int(value) for value in tracks.shape],
            "visibility_shape": [int(value) for value in visibility.shape],
            "confidence_shape": None if confidence is None else [int(value) for value in confidence.shape],
            "depth_shape": None if depth is None else [int(value) for value in depth.shape],
            "world_points_shape": None if world_points is None else [int(value) for value in world_points.shape],
            "videos": {
                "input_window": str(input_video_path.relative_to(output_root)),
                "vggt_tracks": str(tracks_video_path.relative_to(output_root)),
                "vggt_depth": None if depth_video_path is None else str(depth_video_path.relative_to(output_root)),
                "vggt_world_points": None if world_video_path is None else str(world_video_path.relative_to(output_root)),
            },
        }
        write_json(result_path, payload)
        print(
            f"[{mode}] done {index + 1:02d}/10 {case_id} "
            f"tracks={tuple(tracks.shape)} depth={None if depth is None else tuple(depth.shape)}",
            flush=True,
        )
        del output
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return payload
    except Exception as exc:
        error_payload = {
            "status": "error",
            "mode": mode,
            "frame_count": int(frame_count),
            "case_id": case_id,
            "family_key": str(row.get("family_key", "")),
            "source_video": str(video_path),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(result_path, error_payload)
        print(f"[{mode}] ERROR {index + 1:02d}/10 {case_id}: {exc}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return error_payload


def run_mode(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    rows = load_train_rows(manifest_path, split=args.split, limit=int(args.limit))
    selection = {
        "manifest": str(manifest_path),
        "split": args.split,
        "selection_order": "manifest order after stable-hash split filtering",
        "limit": int(args.limit),
        "rows": rows,
    }
    write_json(output_root / "selection.json", selection)
    mode_dir = output_root / args.mode
    write_json(
        mode_dir / "run_config.json",
        {
            "mode": args.mode,
            "frame_count": int(args.frame_count),
            "context_frames": 8,
            "model_path": str(Path(args.model_path).resolve()),
            "device": str(args.device),
            "vggt_input_hw": [int(args.input_h), int(args.input_w)],
            "autocast_bf16": bool(args.autocast_bf16),
        },
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
    print(
        f"loading VGGT from {args.model_path} on {device}; "
        f"mode={args.mode} frames={args.frame_count} cases={len(rows)}",
        flush=True,
    )
    adapter = VGGTTrackAdapter(
        model_path=str(Path(args.model_path).resolve()),
        num_queries=int(args.num_queries),
        device=str(device),
        input_hw=(int(args.input_h), int(args.input_w)),
        trainable=False,
    )
    if adapter.model is None:
        raise RuntimeError(f"failed to load VGGT model from {args.model_path}")
    parameter_dtype = str(next(adapter.model.parameters()).dtype)
    write_json(mode_dir / "model_info.json", {"parameter_dtype": parameter_dtype})
    print(f"VGGT loaded; first_parameter_dtype={parameter_dtype}", flush=True)
    for index, row in enumerate(rows):
        run_one_case(
            adapter,
            row,
            index=index,
            mode=args.mode,
            frame_count=int(args.frame_count),
            output_root=output_root,
            device=device,
            autocast_bf16=bool(args.autocast_bf16),
            overwrite=bool(args.overwrite),
        )
    print(f"[{args.mode}] finished; outputs under {mode_dir}", flush=True)


def video_panel(title: str, subtitle: str, source: str | None) -> str:
    if not source:
        return (
            '<figure class="missing"><figcaption><strong>'
            + html.escape(title)
            + "</strong><span>not returned</span></figcaption></figure>"
        )
    return (
        "<figure><figcaption><strong>"
        + html.escape(title)
        + "</strong><span>"
        + html.escape(subtitle)
        + "</span></figcaption><video controls muted playsinline preload=\"metadata\">"
        + f'<source src="{html.escape(source)}" type="video/mp4"></video></figure>'
    )


def build_gallery(output_root: Path) -> None:
    selection_path = output_root / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"missing selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = selection["rows"]
    modes = ("context8", "prefix49")
    by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in modes:
        by_mode[mode] = {}
        for result_path in sorted((output_root / mode / "cases").glob("*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            by_mode[mode][str(result.get("case_id"))] = result

    cards: list[str] = []
    all_ok = True
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        context = by_mode["context8"].get(case_id, {"status": "pending", "case_id": case_id})
        prefix = by_mode["prefix49"].get(case_id, {"status": "pending", "case_id": case_id})
        if context.get("status") != "ok" or prefix.get("status") != "ok":
            all_ok = False
        family = html.escape(str(row.get("family_key", "")))
        caption = html.escape(str(row.get("caption", "")))
        context_videos = context.get("videos", {}) if context.get("status") == "ok" else {}
        prefix_videos = prefix.get("videos", {}) if prefix.get("status") == "ok" else {}
        context_error = html.escape(str(context.get("error", "pending")))
        prefix_error = html.escape(str(prefix.get("error", "pending")))
        cards.append(
            f"""
            <article class="case" id="case-{index:02d}">
              <header><div><span class="badge">{family}</span>
              <code>{html.escape(case_id)}</code></div><p>{caption}</p></header>
              <div class="status"><span>8-frame context: {html.escape(str(context.get('status')))}</span>
              <span>49-frame prefix: {html.escape(str(prefix.get('status')))}</span></div>
              <div class="comparison">
                <section><h3>8 帧 context</h3>
                  {video_panel('输入窗口', '训练 context · 8 consecutive frames', context_videos.get('input_window'))}
                  {video_panel('VGGT tracks', 'query trajectories over the 8 frames', context_videos.get('vggt_tracks'))}
                  {video_panel('VGGT depth', 'depth visualization', context_videos.get('vggt_depth'))}
                  {video_panel('World points', 'RGB-normalized world coordinates', context_videos.get('vggt_world_points'))}
                  <p class="error">{context_error if context.get('status') != 'ok' else ''}</p>
                </section>
                <section><h3>49 帧 prefix</h3>
                  {video_panel('输入窗口', 'training window · first 49 consecutive frames', prefix_videos.get('input_window'))}
                  {video_panel('VGGT tracks', 'query trajectories over the 49 frames', prefix_videos.get('vggt_tracks'))}
                  {video_panel('VGGT depth', 'depth visualization', prefix_videos.get('vggt_depth'))}
                  {video_panel('World points', 'RGB-normalized world coordinates', prefix_videos.get('vggt_world_points'))}
                  <p class="error">{prefix_error if prefix.get('status') != 'ok' else ''}</p>
                </section>
              </div>
            </article>
            """
        )

    status = "complete" if all_ok else "partial / pending"
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>0717 PyBullet × VGGT · train 10 cases</title>
<style>
  :root {{ color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --muted:#8b949e; --accent:#58a6ff; }}
  * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:#e6edf3; font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  main {{ width:min(1800px, 96vw); margin:0 auto; padding:24px 0 60px; }}
  h1 {{ margin:0 0 6px; font-size:25px; }} h2 {{ margin:28px 0 10px; }} h3 {{ margin:0 0 10px; color:var(--accent); font-size:18px; }}
  .meta {{ color:var(--muted); margin:0 0 22px; }} .case {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; margin:18px 0; }}
  header {{ display:flex; align-items:baseline; justify-content:space-between; gap:18px; border-bottom:1px solid var(--line); padding-bottom:10px; }}
  header p {{ margin:0; color:#c9d1d9; text-align:right; }} .badge {{ display:inline-block; padding:2px 8px; border-radius:99px; background:#1f6feb33; color:#79c0ff; margin-right:8px; }}
  code {{ color:#f0f6fc; }} .status {{ display:flex; gap:20px; color:var(--muted); padding:10px 0; }}
  .comparison {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
  section {{ min-width:0; }} figure {{ margin:10px 0; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#0d1117; }}
  figcaption {{ display:flex; justify-content:space-between; gap:12px; padding:7px 10px; }} figcaption span {{ color:var(--muted); font-size:12px; text-align:right; }}
  video {{ width:100%; display:block; background:#000; max-height:440px; }} .missing {{ opacity:.55; padding:2px; }} .error {{ color:#ff7b72; white-space:pre-wrap; }}
  @media (max-width:1000px) {{ .comparison {{ grid-template-columns:1fr; }} header {{ display:block; }} header p {{ text-align:left; margin-top:8px; }} }}
</style></head><body><main>
<h1>0717 PyBullet train split · VGGT visualization</h1>
<p class="meta">model: {html.escape(str(DEFAULT_MODEL))} · first {len(rows)} stable-hash train cases · status: {status}<br>
Each case compares the first 8 consecutive context frames with the first 49 consecutive training frames. VGGT input is resized to 420×728; tracks, depth, and world-point previews are generated from the same model forward.</p>
{''.join(cards)}
</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")
    write_json(
        output_root / "gallery_status.json",
        {
            "status": status,
            "cases": len(rows),
            "context8_results": len(by_mode["context8"]),
            "prefix49_results": len(by_mode["prefix49"]),
        },
    )
    print(f"gallery written to {output_root / 'index.html'} ({status})", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", choices=("context8", "prefix49"), default="context8")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-h", type=int, default=DEFAULT_INPUT_HW[0])
    parser.add_argument("--input-w", type=int, default=DEFAULT_INPUT_HW[1])
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--autocast-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-gallery", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if args.build_gallery:
        build_gallery(output_root)
        return
    if int(args.frame_count) <= 0:
        raise SystemExit("--frame-count must be positive")
    if args.mode == "context8" and int(args.frame_count) != 8:
        raise SystemExit("context8 mode must use --frame-count 8")
    if args.mode == "prefix49" and int(args.frame_count) != 49:
        raise SystemExit("prefix49 mode must use --frame-count 49")
    run_mode(args)


if __name__ == "__main__":
    main()
