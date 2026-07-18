import argparse
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def read_frame(cap: cv2.VideoCapture):
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (240, 44), (0, 0, 0), -1)
    cv2.putText(labeled, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def side_by_side_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = min(left.shape[0], right.shape[0])
    width = min(left.shape[1], right.shape[1])
    left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
    left = add_label(left, "PAG")
    right = add_label(right, "Baseline")
    spacer = np.full((height, 12, 3), 255, dtype=np.uint8)
    return np.concatenate([left, spacer, right], axis=1)


def encode_h264(frames, output_path: Path, fps: float, width: int, height: int) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {output_path} with exit code {return_code}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pag_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pag_videos = sorted(args.pag_dir.glob("video_*.mp4"))
    baseline_videos = sorted(args.baseline_dir.glob("video_*.mp4"))
    count = min(len(pag_videos), len(baseline_videos))
    if count == 0:
        raise RuntimeError("No overlapping PAG/baseline videos found.")

    for idx in range(count):
        pag_path = args.pag_dir / f"video_{idx}.mp4"
        baseline_path = args.baseline_dir / f"video_{idx}.mp4"
        if not pag_path.exists() or not baseline_path.exists():
            continue

        pag_cap = open_video(pag_path)
        baseline_cap = open_video(baseline_path)

        fps = pag_cap.get(cv2.CAP_PROP_FPS) or baseline_cap.get(cv2.CAP_PROP_FPS) or 8
        pag_width = int(pag_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        pag_height = int(pag_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base_width = int(baseline_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        base_height = int(baseline_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_height = min(pag_height, base_height)
        frame_width = min(pag_width, base_width) * 2 + 12

        output_path = args.output_dir / f"compare_{idx}.mp4"
        frames = []
        try:
            while True:
                pag_frame = read_frame(pag_cap)
                baseline_frame = read_frame(baseline_cap)
                if pag_frame is None or baseline_frame is None:
                    break
                frames.append(side_by_side_frame(pag_frame, baseline_frame))
        finally:
            pag_cap.release()
            baseline_cap.release()

        if not frames:
            raise RuntimeError(f"No frames found for pair {pag_path} and {baseline_path}")

        encode_h264(frames, output_path, fps, frame_width, frame_height)


if __name__ == "__main__":
    main()
