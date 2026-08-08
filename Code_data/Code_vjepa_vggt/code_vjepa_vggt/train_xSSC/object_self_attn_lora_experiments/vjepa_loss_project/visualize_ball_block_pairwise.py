#!/usr/bin/env python3
"""Rank pairwise V-JEPA differences and render native-rectangle overlays."""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader

import compute_vjepa2_feature_mse as vjepa_common


DEFAULT_INPUT_DIR = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block"
)
DEFAULT_OUTPUT_BASE = Path(
    "/data/gaoya/agent-data/outputs/vjepa_ball_block_pairwise"
)
DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt"
)
DEFAULT_VJEPA2_DIR = Path("/home/gaoya/Code_Video/vjepa2_tinyvae_mse/vjepa2")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16
TUBELET_SIZE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare every video pair with native-rectangle V-JEPA tokens."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vjepa2-dir", type=Path, default=DEFAULT_VJEPA2_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default="vjepa2.1-vitl-384")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--input-height", type=int, default=384)
    parser.add_argument("--input-width", type=int, default=672)
    parser.add_argument("--overlay-alpha", type=float, default=0.52)
    parser.add_argument("--color-low-percentile", type=float, default=5.0)
    parser.add_argument("--color-high-percentile", type=float, default=95.0)
    parser.add_argument("--contact-columns", type=int, default=7)
    parser.add_argument("--contact-tile-width", type=int, default=448)
    parser.add_argument("--video-quality", type=int, default=7)
    return parser.parse_args()


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return text or "video"


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def discover_videos(input_dir: Path) -> list[Path]:
    videos = sorted(
        path.resolve()
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if len(videos) < 2:
        raise RuntimeError(f"Need at least two videos under {input_dir}, found {len(videos)}")
    return videos


def read_metadata(path: Path, num_frames: int) -> dict[str, Any]:
    reader = VideoReader(str(path))
    if len(reader) < num_frames:
        raise ValueError(f"{path} has {len(reader)} frames; need {num_frames}")
    first = reader[0].asnumpy()
    return {
        "path": str(path),
        "name": path.name,
        "stem": path.stem,
        "source_frames": int(len(reader)),
        "used_frames": int(num_frames),
        "fps": float(reader.get_avg_fps()),
        "height": int(first.shape[0]),
        "width": int(first.shape[1]),
    }


def preprocess_native_rect(
    frames: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    if frames.ndim != 4 or int(frames.shape[1]) != 3:
        raise ValueError(f"Expected [T,3,H,W], got {tuple(frames.shape)}")
    frames = frames.float().div_(255.0)
    frames = F.interpolate(
        frames,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    mean = torch.tensor(IMAGENET_MEAN, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=frames.dtype).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def unwrap_features(features: Any) -> torch.Tensor:
    if isinstance(features, (list, tuple)):
        features = features[-1]
    if not torch.is_tensor(features) or features.ndim != 3:
        raise TypeError(f"Unexpected V-JEPA output: {type(features)!r}")
    return features


@torch.inference_mode()
def extract_video_features(
    encoder,
    video_path: Path,
    feature_path: Path,
    *,
    num_frames: int,
    input_height: int,
    input_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    reader = VideoReader(str(video_path))
    indices = np.arange(num_frames, dtype=np.int64)
    frames_np = reader.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
    model_frames = torch.cat([frames, frames[-1:]], dim=0)
    if int(model_frames.shape[0]) % TUBELET_SIZE:
        raise RuntimeError("Model frame count must be divisible by tubelet size")
    video = preprocess_native_rect(
        model_frames,
        height=input_height,
        width=input_width,
    ).to(device=device, dtype=dtype)
    features = unwrap_features(encoder(video)).detach().float()
    temporal_tokens = int(model_frames.shape[0]) // TUBELET_SIZE
    grid_height = input_height // PATCH_SIZE
    grid_width = input_width // PATCH_SIZE
    expected_tokens = temporal_tokens * grid_height * grid_width
    if int(features.shape[1]) != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} tokens for {video_path.name}, "
            f"got {tuple(features.shape)}"
        )
    features = features.reshape(
        temporal_tokens,
        grid_height,
        grid_width,
        int(features.shape[-1]),
    )
    features = F.normalize(features, dim=-1, eps=1e-6)
    feature_array = features.cpu().to(torch.float16).numpy()
    np.save(feature_path, feature_array, allow_pickle=False)
    return {
        "feature_path": str(feature_path),
        "feature_shape": list(feature_array.shape),
        "model_frames": int(model_frames.shape[0]),
        "duplicated_last_frame": True,
        "temporal_tokens": temporal_tokens,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "feature_dim": int(feature_array.shape[-1]),
    }


def compute_pair_map(feature_a: Path, feature_b: Path) -> np.ndarray:
    a = np.load(feature_a, mmap_mode="r")
    b = np.load(feature_b, mmap_mode="r")
    if a.shape != b.shape:
        raise ValueError(f"Feature shape mismatch: {a.shape} vs {b.shape}")
    maps = np.empty(a.shape[:-1], dtype=np.float32)
    for temporal_index in range(int(a.shape[0])):
        delta = (
            np.asarray(a[temporal_index], dtype=np.float32)
            - np.asarray(b[temporal_index], dtype=np.float32)
        )
        maps[temporal_index] = np.einsum(
            "hwd,hwd->hw", delta, delta, optimize=True
        )
    return maps


def heatmap_overlay(
    frame: np.ndarray,
    feature_map: np.ndarray,
    *,
    color_min: float,
    color_max: float,
    alpha: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    resized = cv2.resize(feature_map, (width, height), interpolation=cv2.INTER_CUBIC)
    normalized = np.clip(
        (resized - color_min) / max(color_max - color_min, 1e-8),
        0.0,
        1.0,
    )
    color = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    return cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0.0)


def label_frame(
    frame: np.ndarray,
    *,
    name: str,
    frame_index: int,
    pair_score: float,
) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 48), (8, 12, 16), -1)
    cv2.putText(
        result,
        f"{name} | frame {frame_index:02d} | pair mean {pair_score:.6f}",
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 247, 239),
        2,
        cv2.LINE_AA,
    )
    return result


def build_contact_sheet(
    frames: list[np.ndarray],
    *,
    columns: int,
    tile_width: int,
) -> np.ndarray:
    if not frames:
        raise ValueError("No frames for contact sheet")
    source_height, source_width = frames[0].shape[:2]
    tile_height = int(round(source_height * tile_width / source_width))
    rows = int(math.ceil(len(frames) / columns))
    gap = 4
    sheet = np.full(
        (
            rows * tile_height + max(0, rows - 1) * gap,
            columns * tile_width + max(0, columns - 1) * gap,
            3,
        ),
        (11, 14, 18),
        dtype=np.uint8,
    )
    for index, frame in enumerate(frames):
        tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        top = row * (tile_height + gap)
        left = column * (tile_width + gap)
        sheet[top : top + tile_height, left : left + tile_width] = tile
    return sheet


def render_source_media(
    metadata: dict[str, Any],
    *,
    output_root: Path,
    num_frames: int,
    contact_columns: int,
    contact_tile_width: int,
    video_quality: int,
) -> dict[str, Any]:
    source_dir = output_root / "sources" / safe_name(str(metadata["stem"]))
    source_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(metadata["path"]))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {metadata['path']}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_path = source_dir / "raw_first_49_frames.mp4"
    writer = imageio.get_writer(
        str(video_path),
        fps=fps,
        codec="libx264",
        quality=video_quality,
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    frames: list[np.ndarray] = []
    try:
        for frame_index in range(num_frames):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Decode ended before frame {frame_index}: {metadata['path']}"
                )
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
    finally:
        writer.close()
        capture.release()
    sheet = build_contact_sheet(
        frames,
        columns=contact_columns,
        tile_width=contact_tile_width,
    )
    sheet_path = source_dir / "raw_all_49_frames.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Failed to write {sheet_path}")
    result = {
        "name": metadata["name"],
        "source_path": metadata["path"],
        "raw_video": str(video_path),
        "raw_contact_sheet": str(sheet_path),
        "fps": fps,
        "frames": num_frames,
    }
    json_dump(source_dir / "source.json", result)
    return result


def render_pair(
    pair: dict[str, Any],
    *,
    output_root: Path,
    color_min: float,
    color_max: float,
    alpha: float,
    num_frames: int,
    contact_columns: int,
    contact_tile_width: int,
    video_quality: int,
) -> dict[str, Any]:
    pair_dir = output_root / "pairs" / pair["pair_id"]
    pair_dir.mkdir(parents=True, exist_ok=True)
    maps = np.load(pair["map_path"], mmap_mode="r")
    capture_a = cv2.VideoCapture(pair["video_a"])
    capture_b = cv2.VideoCapture(pair["video_b"])
    if not capture_a.isOpened() or not capture_b.isOpened():
        raise RuntimeError(f"Could not open pair videos: {pair['pair_id']}")
    fps_a = float(capture_a.get(cv2.CAP_PROP_FPS))
    fps_b = float(capture_b.get(cv2.CAP_PROP_FPS))
    if abs(fps_a - fps_b) > 1e-3:
        raise RuntimeError(f"FPS mismatch for {pair['pair_id']}: {fps_a} vs {fps_b}")
    video_path = pair_dir / "comparison_overlay.mp4"
    writer = imageio.get_writer(
        str(video_path),
        fps=fps_a,
        codec="libx264",
        quality=video_quality,
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    contact_frames: list[np.ndarray] = []
    try:
        for frame_index in range(num_frames):
            ok_a, frame_a = capture_a.read()
            ok_b, frame_b = capture_b.read()
            if not ok_a or not ok_b:
                raise RuntimeError(
                    f"Decode ended before frame {frame_index} for {pair['pair_id']}"
                )
            if frame_a.shape != frame_b.shape:
                raise RuntimeError(
                    f"Frame shape mismatch for {pair['pair_id']}: "
                    f"{frame_a.shape} vs {frame_b.shape}"
                )
            token_index = min(frame_index // TUBELET_SIZE, int(maps.shape[0]) - 1)
            feature_map = np.asarray(maps[token_index], dtype=np.float32)
            overlay_a = heatmap_overlay(
                frame_a,
                feature_map,
                color_min=color_min,
                color_max=color_max,
                alpha=alpha,
            )
            overlay_b = heatmap_overlay(
                frame_b,
                feature_map,
                color_min=color_min,
                color_max=color_max,
                alpha=alpha,
            )
            overlay_a = label_frame(
                overlay_a,
                name=pair["name_a"],
                frame_index=frame_index,
                pair_score=pair["score"],
            )
            overlay_b = label_frame(
                overlay_b,
                name=pair["name_b"],
                frame_index=frame_index,
                pair_score=pair["score"],
            )
            comparison = np.concatenate([overlay_a, overlay_b], axis=1)
            writer.append_data(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
            contact_frames.append(comparison)
    finally:
        writer.close()
        capture_a.release()
        capture_b.release()
    sheet = build_contact_sheet(
        contact_frames,
        columns=contact_columns,
        tile_width=contact_tile_width,
    )
    sheet_path = pair_dir / "all_49_frames_heatmap.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Failed to write {sheet_path}")
    details = dict(pair)
    details.update(
        {
            "comparison_video": str(video_path),
            "contact_sheet": str(sheet_path),
            "fps": fps_a,
            "rendered_frames": num_frames,
        }
    )
    json_dump(pair_dir / "pair.json", details)
    return details


def relative_to(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def build_html(
    output_root: Path,
    source_media: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> Path:
    source_cards = []
    for item in source_media:
        video_rel = html.escape(relative_to(item["raw_video"], output_root))
        sheet_rel = html.escape(relative_to(item["raw_contact_sheet"], output_root))
        name = html.escape(str(item["name"]))
        source_cards.append(
            f'''<article class="source-card"><h3>{name}</h3>
  <video controls preload="metadata" src="{video_rel}"></video>
  <a href="{sheet_rel}" target="_blank"><img loading="lazy" src="{sheet_rel}" alt="raw all 49 frames"></a>
</article>'''
        )
    cards = []
    for item in results:
        video_rel = html.escape(relative_to(item["comparison_video"], output_root))
        sheet_rel = html.escape(relative_to(item["contact_sheet"], output_root))
        name_a = html.escape(item["name_a"])
        name_b = html.escape(item["name_b"])
        cards.append(
            f'''<article class="card" data-score="{item['score']:.12f}">
  <header><span class="rank">#{item['rank']:02d}</span><h2>{name_a} vs {name_b}</h2>
  <code>mean normalized L2 = {item['score']:.8f}</code></header>
  <video controls preload="metadata" src="{video_rel}"></video>
  <a href="{sheet_rel}" target="_blank"><img loading="lazy" src="{sheet_rel}" alt="all 49 frame heatmaps"></a>
</article>'''
        )
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ball-block V-JEPA pairwise differences</title>
<style>
:root{{--ink:#14211d;--paper:#f2efe5;--rust:#bc4b2d;--moss:#315f4b;--line:#c9c3b2}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(135deg,#efe8d7,#f8f6ee 55%,#e3ebdf);color:var(--ink);font-family:Georgia,serif}}
.hero{{padding:44px clamp(20px,5vw,72px) 28px;border-bottom:1px solid var(--line)}}
h1{{margin:0 0 10px;font-size:clamp(34px,5vw,72px);line-height:.95}} .hero p{{max-width:920px;font:16px/1.55 ui-monospace,monospace}}
.tools{{position:sticky;top:0;z-index:2;padding:12px clamp(20px,5vw,72px);background:#f2efe5e8;backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}}
button{{border:1px solid var(--ink);background:transparent;padding:9px 14px;margin-right:8px;font:600 13px ui-monospace,monospace;cursor:pointer}} button:hover{{background:var(--ink);color:white}}
#grid{{padding:28px clamp(16px,4vw,60px) 70px;display:grid;gap:24px}}
.source-section{{padding:32px clamp(16px,4vw,60px);border-bottom:1px solid var(--line)}} .source-section>h2{{font-size:34px;margin:0 0 8px}} .source-section>p{{font:14px/1.5 ui-monospace,monospace;margin:0 0 20px}}
.source-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}} .source-card{{background:#fffdf7;border:1px solid var(--line);padding:14px}} .source-card h3{{margin:0 0 10px;font:700 15px ui-monospace,monospace}}
.card{{background:#fffdf7;border:1px solid var(--line);box-shadow:0 10px 30px #29352a18;padding:18px}}
.card header{{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:14px}} .card h2{{margin:0;font-size:23px}} .rank{{font:700 22px ui-monospace,monospace;color:var(--rust)}} code{{margin-left:auto;color:var(--moss)}}
video,img{{display:block;width:100%;background:#111}} img{{margin-top:14px;border:1px solid var(--line)}}
</style></head><body>
<section class="hero"><h1>V-JEPA pairwise<br>feature differences</h1>
<p>First 49 frames, native rectangle 384x672, final frame duplicated only for the 2-frame tubelet. Shared color scale across all pairs. Default order: largest mean normalized-token L2 first.</p></section>
<nav class="tools"><button onclick="sortCards(false)">Largest first</button><button onclick="sortCards(true)">Smallest first</button><button onclick="replayAll()">Replay all</button></nav>
<section class="source-section"><h2>Raw source videos and frames</h2><p>No heatmap or text overlay. Each video and 7x7 contact sheet contains the same first 49 frames used by V-JEPA.</p><div class="source-grid">{''.join(source_cards)}</div></section>
<main id="grid">{''.join(cards)}</main>
<script>
function sortCards(ascending){{const grid=document.getElementById('grid');const cards=[...grid.children];cards.sort((a,b)=>(Number(a.dataset.score)-Number(b.dataset.score))*(ascending?1:-1));cards.forEach((card,index)=>{{card.querySelector('.rank').textContent='#'+String(index+1).padStart(2,'0');grid.appendChild(card)}})}}
function replayAll(){{document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play()}})}}
</script></body></html>'''
    output_path = output_root / "index.html"
    output_path.write_text(page, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    if args.num_frames != 49:
        raise ValueError("This visualization is defined for the first 49 frames")
    if args.input_height % PATCH_SIZE or args.input_width % PATCH_SIZE:
        raise ValueError("Input height and width must be divisible by patch size 16")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")
    if not 0.0 <= args.color_low_percentile < args.color_high_percentile <= 100.0:
        raise ValueError("Invalid color percentiles")
    input_dir = args.input_dir.expanduser().resolve()
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = DEFAULT_OUTPUT_BASE / stamp
    else:
        output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    feature_root = output_root / "features"
    map_root = output_root / "pair_maps"
    feature_root.mkdir()
    map_root.mkdir()

    device = vjepa_common.resolve_device(args.device)
    dtype = vjepa_common.dtype_for(device, args.dtype)
    videos = discover_videos(input_dir)
    metadata = [read_metadata(path, args.num_frames) for path in videos]
    reference = metadata[0]
    for item in metadata[1:]:
        for key in ("fps", "height", "width"):
            if abs(float(item[key]) - float(reference[key])) > 1e-3:
                raise RuntimeError(f"Video metadata mismatch for {key}: {item} vs {reference}")

    encoder, _ = vjepa_common.load_encoder(args, device=device, dtype=dtype)
    feature_records: dict[str, dict[str, Any]] = {}
    for index, (video_path, item) in enumerate(zip(videos, metadata), start=1):
        feature_path = feature_root / f"{safe_name(video_path.stem)}.npy"
        print(f"[feature {index}/{len(videos)}] {video_path.name}", flush=True)
        details = extract_video_features(
            encoder,
            video_path,
            feature_path,
            num_frames=args.num_frames,
            input_height=args.input_height,
            input_width=args.input_width,
            device=device,
            dtype=dtype,
        )
        feature_records[video_path.name] = {**item, **details}
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pairs: list[dict[str, Any]] = []
    color_values: list[np.ndarray] = []
    for pair_index, (video_a, video_b) in enumerate(
        itertools.combinations(videos, 2), start=1
    ):
        name_a, name_b = video_a.name, video_b.name
        pair_id = f"{safe_name(video_a.stem)}__vs__{safe_name(video_b.stem)}"
        feature_a = Path(feature_records[name_a]["feature_path"])
        feature_b = Path(feature_records[name_b]["feature_path"])
        maps = compute_pair_map(feature_a, feature_b)
        map_path = map_root / f"{pair_id}.npy"
        np.save(map_path, maps, allow_pickle=False)
        score = float(maps.mean(dtype=np.float64))
        print(f"[pair {pair_index}/28] {name_a} vs {name_b}: {score:.8f}", flush=True)
        pairs.append(
            {
                "pair_id": pair_id,
                "name_a": name_a,
                "name_b": name_b,
                "video_a": str(video_a),
                "video_b": str(video_b),
                "feature_a": str(feature_a),
                "feature_b": str(feature_b),
                "map_path": str(map_path),
                "score": score,
                "map_min": float(maps.min()),
                "map_max": float(maps.max()),
            }
        )
        color_values.append(maps.reshape(-1))
    all_values = np.concatenate(color_values)
    color_min = float(np.percentile(all_values, args.color_low_percentile))
    color_max = float(np.percentile(all_values, args.color_high_percentile))
    pairs.sort(key=lambda item: item["score"], reverse=True)

    source_media: list[dict[str, Any]] = []
    for source_index, item in enumerate(metadata, start=1):
        print(f"[raw {source_index}/{len(metadata)}] {item['name']}", flush=True)
        source_media.append(
            render_source_media(
                item,
                output_root=output_root,
                num_frames=args.num_frames,
                contact_columns=args.contact_columns,
                contact_tile_width=args.contact_tile_width,
                video_quality=args.video_quality,
            )
        )

    rendered: list[dict[str, Any]] = []
    for rank, pair in enumerate(pairs, start=1):
        pair["rank"] = rank
        print(f"[render {rank}/{len(pairs)}] {pair['pair_id']}", flush=True)
        rendered.append(
            render_pair(
                pair,
                output_root=output_root,
                color_min=color_min,
                color_max=color_max,
                alpha=args.overlay_alpha,
                num_frames=args.num_frames,
                contact_columns=args.contact_columns,
                contact_tile_width=args.contact_tile_width,
                video_quality=args.video_quality,
            )
        )
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "dtype": str(dtype),
        "num_videos": len(videos),
        "num_pairs": len(rendered),
        "source_frames_used": args.num_frames,
        "model_frames": args.num_frames + 1,
        "input_shape": [args.num_frames + 1, args.input_height, args.input_width],
        "native_rectangle": True,
        "center_crop": False,
        "color_scale": {
            "low_percentile": args.color_low_percentile,
            "high_percentile": args.color_high_percentile,
            "minimum": color_min,
            "maximum": color_max,
        },
        "source_media": source_media,
        "features": feature_records,
        "ranking": rendered,
    }
    json_dump(output_root / "results.json", manifest)
    page = build_html(output_root, source_media, rendered)
    print(f"Results: {output_root}", flush=True)
    print(f"Page: {page}", flush=True)


if __name__ == "__main__":
    main()
