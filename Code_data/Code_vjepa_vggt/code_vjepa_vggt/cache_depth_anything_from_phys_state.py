from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch


DEFAULT_DATASET_ROOT = "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500"
DEFAULT_OUTPUT_DIR = "/data/gaoya/AAA_test_video/0623/train/train0624/depth_anything_cache"
DEFAULT_DEPTH_SCRIPT = "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_depth_anything_on_video.py"
DEFAULT_CHECKPOINT = "/data/gaoya/ckpt/LiheYoung-depth_anything_vitl14_raw/checkpoints/depth_anything_vitl14.pth"
DEFAULT_RAW_ROOT = "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--depth-script", default=DEFAULT_DEPTH_SCRIPT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--keep-depth-video", action="store_true")
    return parser.parse_args()


def _infer_source_video(meta: dict, raw_root: Path) -> Path:
    sample_dir = Path(str(meta["sample_dir"]))
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"raw sample_dir not found: {sample_dir}")
    for candidate_name in ("video.mp4", "source_video.mp4", "context_video.mp4"):
        candidate = sample_dir / candidate_name
        if candidate.is_file():
            return candidate
    nested = sample_dir / "source_video" / "context_video_8f.mp4"
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"no supported source video found under {sample_dir}")


def _read_video_gray(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32) / 255.0)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return np.stack(frames, axis=0)


def _write_h264_mp4(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    try:
        for frame in frames_rgb:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed for {path} with code {ret}")
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_root = dataset_root / args.split
    output_dir = Path(args.output_dir).expanduser().resolve()
    depth_script = Path(args.depth_script).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(split_root.glob("*.json"))
    if int(args.num_shards) < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if not (0 <= int(args.shard_index) < int(args.num_shards)):
        raise ValueError(
            f"--shard-index must satisfy 0 <= shard_index < num_shards, "
            f"got shard_index={args.shard_index}, num_shards={args.num_shards}"
        )
    meta_paths = meta_paths[int(args.shard_index) :: int(args.num_shards)]
    if args.limit is not None:
        meta_paths = meta_paths[: max(int(args.limit), 0)]

    records = []
    for meta_path in meta_paths:
        cache_path = output_dir / f"{meta_path.stem}.depth_anything.pt"
        if cache_path.is_file() and not args.overwrite:
            records.append({"meta_path": str(meta_path), "cache_path": str(cache_path), "status": "skipped_existing"})
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source_video = _infer_source_video(meta, raw_root)
        tmp_depth_video = output_dir / f"{meta_path.stem}.depth_anything.mp4"
        cmd = [
            "bash",
            "-lc",
            (
                f"CUDA_VISIBLE_DEVICES={args.gpu} "
                "PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:"
                "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt:"
                "/home/gaoya/MimicBrush-main/depthanything "
                "/home/gaoya/miniconda3/envs/wan-cu128/bin/python "
                f"{depth_script} "
                f"--input-video {source_video} "
                f"--output-video {tmp_depth_video} "
                f"--checkpoint {checkpoint}"
            ),
        ]
        subprocess.run(cmd, check=True)
        depth_frames = _read_video_gray(tmp_depth_video)
        torch.save(
            {
                "depth_frames": torch.from_numpy(depth_frames).float(),
                "frame_indices": list(range(int(depth_frames.shape[0]))),
                "source_video": str(source_video),
                "output_file": str(cache_path),
                "q_low": 5.0,
                "q_high": 95.0,
                "dtype": "float32",
            },
            cache_path,
        )
        if tmp_depth_video.is_file() and not args.keep_depth_video:
            tmp_depth_video.unlink()
        records.append({"meta_path": str(meta_path), "cache_path": str(cache_path), "status": "written"})

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
