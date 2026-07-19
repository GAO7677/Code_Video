#!/usr/bin/env python3
"""Track shared SAM2 query points on PhysicIQ source videos and render overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from AAA_my_test.analyze_stage1b_kubric_generation import point_colors
from AAA_my_test.sam2_region_query_utils import load_region_cache, region_metadata


DEFAULT_DATASET = Path("/data/gaoya/agent-data/datasets/physiciq_selected_qk")
DEFAULT_CACHE = Path("/data/gaoya/agent-data/cache/physiciq_selected_sam2_regions")
DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk/gt_cotracker")
DEFAULT_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--query-frame", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_video(path: Path, height: int, width: int) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"cannot read source video: {path}")
    return np.stack(frames), fps


def run_cotracker(
    model,
    frames: np.ndarray,
    points: np.ndarray,
    query_frame: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    native_height, native_width = frames.shape[1:3]
    video = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
    video = F.interpolate(video, size=(384, 512), mode="bilinear", align_corners=True)
    queries = torch.from_numpy(points).float()
    queries[:, 0] *= 512 / native_width
    queries[:, 1] *= 384 / native_height
    query_times = torch.full((len(points), 1), float(query_frame))
    queries = torch.cat((query_times, queries), dim=-1).unsqueeze(0).to(device)
    with torch.inference_mode():
        tracks, visibility = model(
            video.unsqueeze(0).to(device),
            queries=queries,
            backward_tracking=False,
        )
    tracks = tracks[0].float().cpu().numpy()
    tracks[..., 0] *= max(native_width - 1, 1) / 511
    tracks[..., 1] *= max(native_height - 1, 1) / 383
    return tracks, visibility[0].float().cpu().numpy() > 0.5


def render_tracks(
    frames: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    output_path: Path,
    fps: float,
    label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = point_colors(tracks.shape[1])
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame_index, rgb in enumerate(frames):
            canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for point_index, color in enumerate(colors):
                start = max(0, frame_index - 24)
                for previous in range(start, frame_index):
                    if visibility[previous, point_index] and visibility[previous + 1, point_index]:
                        p0 = tuple(np.rint(tracks[previous, point_index]).astype(int))
                        p1 = tuple(np.rint(tracks[previous + 1, point_index]).astype(int))
                        cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
                if visibility[frame_index, point_index]:
                    point = tuple(np.rint(tracks[frame_index, point_index]).astype(int))
                    cv2.circle(canvas, point, 5, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(canvas, point, 3, color, -1, cv2.LINE_AA)
            text = f"GT source | CoTracker | {label} | frame {frame_index}"
            cv2.putText(canvas, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 3)
            cv2.putText(canvas, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(COTRACKER_ROOT))
    from cotracker.predictor import CoTrackerPredictor

    model = CoTrackerPredictor(
        checkpoint=str(args.checkpoint), offline=True, v2=False, window_len=60
    ).to(args.device).eval().requires_grad_(False)
    manifests = sorted((args.dataset_root / "cases").glob("case_*/case_manifest.json"))
    for index, manifest_path in enumerate(manifests, start=1):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_key = str(payload["case_key"])
        case_dir = args.output_dir / "cases" / case_key
        complete = case_dir / "complete.json"
        if complete.is_file() and not args.overwrite:
            print(f"[{index}/{len(manifests)}] reuse {case_key}", flush=True)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        cache = load_region_cache(args.cache_root, case_key)
        frames, fps = read_video(
            Path(payload["base"]["source_video"]), int(args.height), int(args.width)
        )
        query_frame = min(int(args.query_frame), len(frames) - 1)
        points = cache.query_points.astype(np.float32)
        tracks, visibility = run_cotracker(model, frames, points, query_frame, args.device)
        np.savez_compressed(
            case_dir / "cotracker_tracks.npz",
            tracks=tracks,
            visibility=visibility,
            query_points=points,
            query_frame=np.asarray(query_frame),
        )
        render_tracks(
            frames, tracks, visibility, case_dir / "tracks_all.mp4", fps, "all regions"
        )
        for region in cache.regions:
            point_slice = slice(region.point_start, region.point_end)
            render_tracks(
                frames,
                tracks[:, point_slice],
                visibility[:, point_slice],
                case_dir / "regions" / region.region_name / "tracks_cotracker.mp4",
                fps,
                region.region_name,
            )
        manifest = {
            "case_key": case_key,
            "input_json": payload["base"]["input_json"],
            "source_video": payload["base"]["source_video"],
            "query_frame": query_frame,
            "query_points": points.tolist(),
            "query_regions": [region_metadata(region) for region in cache.regions],
            "source_frames": len(frames),
            "fps": fps,
            "height": int(args.height),
            "width": int(args.width),
        }
        (case_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        complete.write_text(json.dumps({"case_key": case_key}, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(manifests)}] complete {case_key}: {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
