from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from code_vjepa_vggt.utils.depth_target_branch import (
    build_depth_target_comparison,
    scalar_depth_map_to_rgb,
    scalar_depth_to_box_map,
)
from code_vjepa_vggt.utils.npz_io import load_npz_tensor_dict


DEFAULT_SAMPLE = "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500/val/sample_000301_w000.npz"
DEFAULT_OUTPUT = "/data/gaoya/agent-data/outputs/depth_gt_compare_sample_000301_w000"
DEFAULT_DEPTH_SCRIPT = "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_depth_anything_on_video.py"
DEFAULT_DEPTH_CKPT = "/data/gaoya/ckpt/LiheYoung-depth_anything_vitl14_raw/checkpoints/depth_anything_vitl14.pth"
DEFAULT_RAW_VIDEO = "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F1_single_object/sample_000301/source_video/context_video_8f.mp4"

PALETTE = [
    (255, 80, 80),
    (80, 220, 120),
    (80, 140, 255),
    (255, 200, 70),
    (200, 100, 255),
    (40, 220, 220),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-npz", default=DEFAULT_SAMPLE)
    parser.add_argument("--raw-context-video", default=DEFAULT_RAW_VIDEO)
    parser.add_argument("--depth-video", default=None)
    parser.add_argument("--depth-script", default=DEFAULT_DEPTH_SCRIPT)
    parser.add_argument("--depth-checkpoint", default=DEFAULT_DEPTH_CKPT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--depth-target-state-index", type=int, default=2)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--gpu", default="5")
    return parser.parse_args()


def _write_h264_mp4(path: Path, frames_rgb: np.ndarray, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = (
        subprocess.check_output(
            [
                "/home/gaoya/miniconda3/envs/wan-cu128/bin/python",
                "-c",
                "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
            ],
            text=True,
        )
        .strip()
    )
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
        str(int(fps)),
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


def _run_depth_anything(raw_video: Path, output_video: Path, depth_script: Path, checkpoint: Path, gpu: str) -> None:
    if output_video.is_file():
        return
    cmd = [
        "bash",
        "-lc",
        (
            f"CUDA_VISIBLE_DEVICES={gpu} "
            "PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:"
            "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt:"
            "/home/gaoya/MimicBrush-main/depthanything "
            "/home/gaoya/miniconda3/envs/wan-cu128/bin/python "
            f"{depth_script} "
            f"--input-video {raw_video} "
            f"--output-video {output_video} "
            f"--checkpoint {checkpoint}"
        ),
    ]
    subprocess.run(cmd, check=True)


def _read_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0)


def _frames_to_tensor(frames_rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float() / 255.0


def _draw_box_rgb(image: np.ndarray, box_xyxy_px: np.ndarray, color_rgb: tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy_px.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image, (x0, y0), (x1, y1), color_bgr, 2)
    cv2.putText(image, label, (x0 + 2, max(y0 + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)


def _render_comparison_frames(
    background_rgb: np.ndarray,
    state_maps: torch.Tensor,
    depth_anything_maps: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    boxes_valid: torch.Tensor,
) -> np.ndarray:
    frames = []
    num_frames = int(state_maps.shape[0])
    height, width = int(state_maps.shape[1]), int(state_maps.shape[2])
    for t in range(num_frames):
        state_rgb = scalar_depth_map_to_rgb(state_maps[t].cpu().numpy())
        depth_rgb = scalar_depth_map_to_rgb(depth_anything_maps[t].cpu().numpy())
        bg = background_rgb[t].copy()
        for obj_idx in range(int(boxes_xyxy.shape[1])):
            if not bool(boxes_valid[t, obj_idx].item()):
                continue
            color = PALETTE[obj_idx % len(PALETTE)]
            x0, y0, x1, y1 = boxes_xyxy[t, obj_idx].cpu().numpy().astype(np.float32)
            box_px = np.array([x0 * width, y0 * height, x1 * width, y1 * height], dtype=np.float32)
            _draw_box_rgb(bg, box_px, color, f"obj{obj_idx}")
            _draw_box_rgb(state_rgb, box_px, color, f"obj{obj_idx}")
            _draw_box_rgb(depth_rgb, box_px, color, f"obj{obj_idx}")
        panel = np.concatenate([bg, state_rgb, depth_rgb], axis=1)
        cv2.putText(panel, "context frame + boxes", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "state GT -> HxW", (width + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "Depth Anything pooled GT -> HxW", (2 * width + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(panel)
    return np.stack(frames, axis=0)


def main() -> None:
    args = _parse_args()
    sample_npz = Path(args.sample_npz).expanduser().resolve()
    raw_context_video = Path(args.raw_context_video).expanduser().resolve()
    depth_script = Path(args.depth_script).expanduser().resolve()
    depth_checkpoint = Path(args.depth_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.depth_video is not None and str(args.depth_video).strip():
        depth_video = Path(args.depth_video).expanduser().resolve()
    else:
        depth_video = output_dir / f"{sample_npz.stem}__depth_anything_context8f_h264.mp4"
        _run_depth_anything(raw_context_video, depth_video, depth_script, depth_checkpoint, args.gpu)

    tensors = load_npz_tensor_dict(sample_npz)
    context_boxes = tensors["context_boxes"].float().unsqueeze(0)
    context_states = tensors["context_states"].float().unsqueeze(0)
    object_indices = torch.arange(min(int(context_boxes.shape[2]), int(args.aux_max_objects)), dtype=torch.long).unsqueeze(0)

    depth_frames_rgb = _read_video_rgb(depth_video)
    raw_frames_rgb = _read_video_rgb(raw_context_video)
    depth_gray = _frames_to_tensor(depth_frames_rgb).mean(dim=1, keepdim=False).unsqueeze(0)

    comparison = build_depth_target_comparison(
        context_states=context_states,
        context_boxes=context_boxes,
        object_indices=object_indices,
        depth_target_state_index=int(args.depth_target_state_index),
        depth_maps_norm=depth_gray,
        latent_frames=int(args.latent_frames),
    )

    state_map = scalar_depth_to_box_map(
        comparison.state_depth_latent,
        comparison.matched_boxes[:, comparison.frame_indices_for_latent],
        comparison.matched_boxes_valid[:, comparison.frame_indices_for_latent],
        image_hw=(raw_frames_rgb.shape[1], raw_frames_rgb.shape[2]),
    )[0]
    depth_anything_map = scalar_depth_to_box_map(
        comparison.depth_anything_latent,
        comparison.matched_boxes[:, comparison.frame_indices_for_latent],
        comparison.matched_boxes_valid[:, comparison.frame_indices_for_latent],
        image_hw=(raw_frames_rgb.shape[1], raw_frames_rgb.shape[2]),
    )[0]
    bg_frames = raw_frames_rgb[comparison.frame_indices_for_latent.cpu().numpy()]
    boxes_latent = comparison.matched_boxes[0, comparison.frame_indices_for_latent]
    valid_latent = comparison.matched_boxes_valid[0, comparison.frame_indices_for_latent]
    panel_frames = _render_comparison_frames(bg_frames, state_map, depth_anything_map, boxes_latent, valid_latent)

    panel_video = output_dir / f"{sample_npz.stem}__state_vs_depth_anything_panel.mp4"
    _write_h264_mp4(panel_video, panel_frames, fps=2)

    summary = {
        "sample_npz": str(sample_npz),
        "raw_context_video": str(raw_context_video),
        "depth_video": str(depth_video),
        "latent_frames": int(args.latent_frames),
        "frame_indices_for_latent": comparison.frame_indices_for_latent.tolist(),
        "state_depth_latent_shape": list(comparison.state_depth_latent.shape),
        "depth_anything_latent_shape": list(comparison.depth_anything_latent.shape),
        "state_depth_latent_values": comparison.state_depth_latent[0, :, :, 0].tolist(),
        "depth_anything_latent_values": comparison.depth_anything_latent[0, :, :, 0].tolist(),
        "matched_boxes_shape": list(comparison.matched_boxes.shape),
        "panel_video": str(panel_video),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
