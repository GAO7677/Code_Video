import argparse
import math
import subprocess
from pathlib import Path

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
    cv2.rectangle(labeled, (0, 0), (min(frame.shape[1], 320), 44), (0, 0, 0), -1)
    cv2.putText(labeled, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


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


def parse_input(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Expected LABEL=PATH, got: {item}")
    label, path_str = item.split("=", 1)
    return label, Path(path_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Repeated LABEL=PATH entries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--gap", type=int, default=12)
    args = parser.parse_args()

    entries = [parse_input(item) for item in args.input]
    captures = [(label, path, open_video(path)) for label, path in entries]
    fps = captures[0][2].get(cv2.CAP_PROP_FPS) or 8
    tile_width = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for _, _, cap in captures)
    tile_height = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for _, _, cap in captures)
    rows = math.ceil(len(captures) / args.cols)
    canvas_width = args.cols * tile_width + (args.cols - 1) * args.gap
    canvas_height = rows * tile_height + (rows - 1) * args.gap
    blank_tile = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)

    frames = []
    try:
        while True:
            tiles = []
            for label, _, cap in captures:
                frame = read_frame(cap)
                if frame is None:
                    tiles = None
                    break
                frame = resize_frame(frame, tile_width, tile_height)
                tiles.append(add_label(frame, label))
            if tiles is None:
                break

            while len(tiles) < rows * args.cols:
                tiles.append(blank_tile.copy())

            row_frames = []
            for row in range(rows):
                start = row * args.cols
                end = start + args.cols
                row_tiles = tiles[start:end]
                row_frames.append(np.concatenate(row_tiles, axis=1) if args.gap == 0 else np.concatenate(
                    [item for pair in zip(row_tiles, [np.full((tile_height, args.gap, 3), 255, dtype=np.uint8)] * (args.cols - 1) + [None]) for item in ([pair[0]] if pair[1] is None else [pair[0], pair[1]])],
                    axis=1,
                ))

            if args.gap == 0 or rows == 1:
                canvas = np.concatenate(row_frames, axis=0) if rows > 1 else row_frames[0]
            else:
                row_gap = np.full((args.gap, canvas_width, 3), 255, dtype=np.uint8)
                canvas_parts = []
                for idx, row_frame in enumerate(row_frames):
                    canvas_parts.append(row_frame)
                    if idx != len(row_frames) - 1:
                        canvas_parts.append(row_gap)
                canvas = np.concatenate(canvas_parts, axis=0)

            frames.append(canvas)
    finally:
        for _, _, cap in captures:
            cap.release()

    if not frames:
        raise RuntimeError("No frames were assembled for output.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_h264(frames, args.output, fps, canvas_width, canvas_height)


if __name__ == "__main__":
    main()
