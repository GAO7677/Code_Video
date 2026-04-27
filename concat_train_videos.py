#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    import cv2
    import numpy as np
except Exception as e:
    print(f"[ERROR] 需要安装 opencv-python 和 numpy: {e}", file=sys.stderr)
    raise


def find_videos(root: Path) -> list[Path]:
    primary = sorted(root.glob("train_scene_*/video/*.mp4"))
    if primary:
        return primary
    return sorted(root.glob("**/video/*.mp4"))


def get_scene_title(video_path: Path) -> str:
    for p in [video_path.parent, *video_path.parents]:
        if p.name.startswith("train_scene_"):
            return p.name
    if video_path.parent.name == "video":
        return video_path.parent.parent.name
    return video_path.stem


def round_even(x: int) -> int:
    x = int(x)
    return x if x % 2 == 0 else x + 1


def safe_fps(v: float, default: float = 25.0) -> float:
    if v is None or v <= 1e-3 or v != v:
        return default
    # 防止一些奇怪值
    if v > 240:
        return default
    return float(v)


def collect_media_info(videos: Iterable[Path], default_fps: float) -> tuple[int, int, float]:
    max_w = 0
    max_h = 0
    fps_list: list[float] = []

    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[WARN] 无法打开视频，跳过信息读取: {vp}")
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = safe_fps(cap.get(cv2.CAP_PROP_FPS), default=default_fps)
        cap.release()

        max_w = max(max_w, w)
        max_h = max(max_h, h)
        if fps > 0:
            fps_list.append(fps)

    if max_w == 0 or max_h == 0:
        raise RuntimeError("没有读到任何有效视频尺寸信息。")

    out_fps = default_fps if default_fps > 0 else (fps_list[0] if fps_list else 25.0)
    return round_even(max_w), round_even(max_h), out_fps


def resize_and_pad(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    new_w = min(new_w, out_w)
    new_h = min(new_h, out_h)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def draw_title(frame: np.ndarray, title: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    bar_h = max(44, int(h * 0.09))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
    alpha = 0.45
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, min(w, h) / 900.0)
    thickness = max(1, int(round(min(w, h) / 400.0)))

    (tw, th), baseline = cv2.getTextSize(title, font, font_scale, thickness)
    tx = max(10, (w - tw) // 2)
    ty = max(th + 8, (bar_h + th) // 2 - baseline // 2)

    # 黑边提升可读性
    cv2.putText(out, title, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, title, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def process_and_append_video(
    writer: cv2.VideoWriter,
    video_path: Path,
    title: str,
    out_w: int,
    out_h: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频，跳过: {video_path}")
        return 0

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_and_pad(frame, out_w, out_h)
        frame = draw_title(frame, title)
        writer.write(frame)
        written += 1

    cap.release()
    return written


def make_writer(output: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    # 优先 mp4v，兼容性通常较好
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if writer.isOpened():
        return writer

    # 兜底 avi/XVID
    alt_output = output.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(alt_output), fourcc, fps, (width, height))
    if writer.isOpened():
        print(f"[WARN] mp4 编码器不可用，改为输出 AVI: {alt_output}")
        return writer

    raise RuntimeError("无法创建 VideoWriter，请检查 OpenCV 是否带视频编码支持。")


def main() -> int:
    parser = argparse.ArgumentParser(description="不依赖 ffmpeg/ffprobe：读取 train_scene 视频，写标题并拼接为一个视频")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Dataset_test/physxnet_proxy_dataset_v03232/train"),
        help="train 根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp.mp4"),
        help="输出路径",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="输出 fps；默认 25",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not root.exists():
        print(f"[ERROR] 目录不存在: {root}", file=sys.stderr)
        return 2

    videos = find_videos(root)
    if not videos:
        print(f"[ERROR] 在 {root} 下没有找到任何 video/*.mp4", file=sys.stderr)
        return 3

    print(f"[INFO] 共找到 {len(videos)} 个视频文件")
    for i, vp in enumerate(videos[:10], 1):
        print(f"  {i:03d}. {vp}")
    if len(videos) > 10:
        print("  ...")

    width, height, fps = collect_media_info(videos, default_fps=args.fps)
    print(f"[INFO] 输出分辨率: {width}x{height}")
    print(f"[INFO] 输出帧率: {fps}")
    print("[INFO] 输出音频: 不保留（纯视频）")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(output, fps, width, height)

    total_frames = 0
    real_output = output
    if output.suffix.lower() == ".mp4" and not writer.isOpened():
        real_output = output.with_suffix(".avi")

    try:
        for idx, src in enumerate(videos, 1):
            title = get_scene_title(src)
            print(f"[INFO] 处理中 {idx}/{len(videos)}: {src.name} -> 标题: {title}")
            n = process_and_append_video(writer, src, title, width, height)
            total_frames += n
            print(f"[INFO]   写入 {n} 帧")
    finally:
        writer.release()

    if total_frames == 0:
        print("[ERROR] 没有成功写入任何帧。", file=sys.stderr)
        return 4

    print(f"[DONE] 已生成: {real_output}")
    print(f"[DONE] 总帧数: {total_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""

python /home/gaoya/Code_Video/concat_train_videos.py \
  --root /data/gaoya/AAA_test_video/Dataset_test/physxnet_proxy_dataset_v03232/train \
  --output /home/gaoya/Code_Video/Code_data/tmp.mp4
  
  
  
  """