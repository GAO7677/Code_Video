from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from torchvision.transforms import Compose


DEPTH_ANYTHING_ROOT = Path("/home/gaoya/MimicBrush-main/depthanything")
if str(DEPTH_ANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_ROOT))

from depth_anything.dpt import DepthAnything  # type: ignore
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize  # type: ignore


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument(
        "--checkpoint",
        default="/data/gaoya/ckpt/LiheYoung-depth_anything_vitl14_raw/checkpoints/depth_anything_vitl14.pth",
    )
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl"], default="vitl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--fps", type=float, default=None)
    return parser.parse_args()


def _build_model(checkpoint: str, encoder: str, device: torch.device) -> DepthAnything:
    model = DepthAnything(MODEL_CONFIGS[encoder])
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[depth-anything] missing keys: {len(missing)}")
    if unexpected:
        print(f"[depth-anything] unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()
    return model


def _build_transform(input_size: int) -> Compose:
    return Compose(
        [
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])),
            PrepareForNet(),
        ]
    )


def _depth_to_rgb(depth_hw: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_hw)
    if np.any(valid):
        lo = float(np.nanpercentile(depth_hw[valid], 5.0))
        hi = float(np.nanpercentile(depth_hw[valid], 95.0))
        if hi - lo < 1.0e-6:
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((np.where(valid, depth_hw, lo) - lo) / (hi - lo + 1.0e-6), 0.0, 1.0)
    gray = (norm * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return frames, fps


def _write_video(path: Path, frames_rgb: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
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
    try:
        assert proc.stdin is not None
        for frame_rgb in frames_rgb:
            proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed for {path} with code {ret}")
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    input_video = Path(args.input_video).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    device = torch.device(args.device)

    model = _build_model(args.checkpoint, args.encoder, device)
    transform = _build_transform(int(args.input_size))

    frames_rgb, src_fps = _read_video(input_video)
    fps = float(args.fps) if args.fps is not None else (src_fps if src_fps > 0 else 30.0)
    out_frames: list[np.ndarray] = []

    for frame_rgb in frames_rgb:
        h, w = frame_rgb.shape[:2]
        sample = transform({"image": frame_rgb.astype(np.float32) / 255.0})
        image = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
        depth = model(image)
        depth = torch.nn.functional.interpolate(
            depth[:, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        depth_np = depth[0].detach().float().cpu().numpy()
        out_frames.append(_depth_to_rgb(depth_np))

    _write_video(output_video, out_frames, fps=fps)
    print(f"[depth-anything] input={input_video}")
    print(f"[depth-anything] output={output_video}")
    print(f"[depth-anything] frames={len(out_frames)} fps={fps:.3f}")


if __name__ == "__main__":
    main()
