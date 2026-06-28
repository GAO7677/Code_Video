from __future__ import annotations

import argparse
import base64
import http.server
import io
import json
import socketserver
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from code_vjepa_vggt.adapters.sam2_motion import SAM2MotionTracker
from code_vjepa_vggt.utils.depth_target_branch import (
    group_last,
    scalar_depth_map_to_rgb,
    scalar_depth_to_box_map,
)
from code_vjepa_vggt.utils.npz_io import load_npz_tensor_dict


DEFAULT_SAMPLE = "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500/val/sample_000301_w000.npz"
DEFAULT_OUTPUT = "/data/gaoya/agent-data/outputs/depth_sam_mask_pool_sample_000301_w000"
DEFAULT_DEPTH_SCRIPT = "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_depth_anything_on_video.py"
DEFAULT_DEPTH_CKPT = "/data/gaoya/ckpt/LiheYoung-depth_anything_vitl14_raw/checkpoints/depth_anything_vitl14.pth"
DEFAULT_PORT = 8781

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
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--depth-video", default=None)
    parser.add_argument("--depth-script", default=DEFAULT_DEPTH_SCRIPT)
    parser.add_argument("--depth-checkpoint", default=DEFAULT_DEPTH_CKPT)
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--depth-target-state-index", type=int, default=2)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--depth-gpu", default="5")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--serve", action="store_true")
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


def _frames_to_rgb_uint8(context_frames_tchw: torch.Tensor) -> np.ndarray:
    return (
        context_frames_tchw.clamp(0.0, 1.0)
        .permute(0, 2, 3, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _run_depth_anything(input_video: Path, output_video: Path, depth_script: Path, checkpoint: Path, gpu: str) -> None:
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
            f"--input-video {input_video} "
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


def _read_video_gray(path: Path) -> np.ndarray:
    rgb = _read_video_rgb(path)
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb], axis=0)
    return gray.astype(np.float32) / 255.0


def _pool_depth_from_masks_median(depth_maps: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if depth_maps.ndim != 3:
        raise ValueError(f"depth_maps must have shape [T,H,W], got {list(depth_maps.shape)}")
    if masks.ndim != 4:
        raise ValueError(f"masks must have shape [T,O,H,W], got {list(masks.shape)}")
    num_frames, height, width = depth_maps.shape
    if tuple(masks.shape[0:3])[:1] != (num_frames,):
        raise ValueError("mask/depth frame mismatch")
    out = torch.zeros((num_frames, masks.shape[1], 1), dtype=torch.float32)
    for t in range(num_frames):
        depth_hw = depth_maps[t].detach().cpu().numpy()
        for o in range(masks.shape[1]):
            mask_hw = masks[t, o].detach().cpu().numpy() > 0.5
            if not mask_hw.any():
                continue
            vals = depth_hw[mask_hw]
            finite = np.isfinite(vals)
            if finite.any():
                out[t, o, 0] = float(np.median(vals[finite]))
    return out


def _box_valid(box_xyxy: np.ndarray) -> bool:
    return bool(float(box_xyxy[2] - box_xyxy[0]) > 1.0e-6 and float(box_xyxy[3] - box_xyxy[1]) > 1.0e-6)


def _build_sam_masks_from_gt_boxes(
    tracker: SAM2MotionTracker,
    context_frames_tchw: torch.Tensor,
    context_boxes_to_use: torch.Tensor,
) -> torch.Tensor:
    frames_np = context_frames_tchw.cpu().numpy()
    num_frames, _, height, width = frames_np.shape
    num_objects = int(context_boxes_to_use.shape[1])
    masks = torch.zeros((num_frames, num_objects, height, width), dtype=torch.float32)
    for t in range(num_frames):
        frame = frames_np[t]
        for o in range(num_objects):
            box = context_boxes_to_use[t, o].cpu().numpy().astype(np.float32)
            if not _box_valid(box):
                continue
            box_px = np.asarray(
                [box[0] * width, box[1] * height, box[2] * width, box[3] * height],
                dtype=np.float32,
            )
            mask = tracker._refine_box_to_mask(frame, box_px)
            if mask is None:
                continue
            masks[t, o] = torch.from_numpy(mask.astype(np.float32))
    return masks


def _object_indices(context_boxes: torch.Tensor, aux_max_objects: int) -> torch.Tensor:
    num_objects = min(int(context_boxes.shape[1]), int(aux_max_objects))
    return torch.arange(num_objects, dtype=torch.long)


def _gather_object_state_depth(context_states: torch.Tensor, obj_idx: torch.Tensor, depth_idx: int) -> torch.Tensor:
    depth = context_states[:, obj_idx, depth_idx : depth_idx + 1]
    return depth


def _draw_box_rgb(image: np.ndarray, box_xyxy_px: np.ndarray, color_rgb: tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy_px.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image, (x0, y0), (x1, y1), color_bgr, 2)
    cv2.putText(image, label, (x0 + 2, max(y0 + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)


def _overlay_mask(image_rgb: np.ndarray, mask_hw: np.ndarray, color_rgb: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = image_rgb.copy()
    if mask_hw.dtype != np.bool_:
        mask_hw = mask_hw > 0
    if not mask_hw.any():
        return out
    color = np.asarray(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    out_f = out.astype(np.float32)
    out_f[mask_hw] = out_f[mask_hw] * (1.0 - alpha) + color * alpha
    return np.clip(out_f, 0.0, 255.0).astype(np.uint8)


def _render_panel_frames(
    raw_frames_rgb: np.ndarray,
    boxes_norm: torch.Tensor,
    sam_masks: torch.Tensor,
    state_map: torch.Tensor,
    box_map: torch.Tensor,
    mask_map: torch.Tensor,
) -> np.ndarray:
    frames = []
    num_frames, height, width = raw_frames_rgb.shape[0], raw_frames_rgb.shape[1], raw_frames_rgb.shape[2]
    for t in range(num_frames):
        left = raw_frames_rgb[t].copy()
        for o in range(int(boxes_norm.shape[1])):
            box = boxes_norm[t, o].cpu().numpy().astype(np.float32)
            if not _box_valid(box):
                continue
            color = PALETTE[o % len(PALETTE)]
            left = _overlay_mask(left, sam_masks[t, o].cpu().numpy() > 0.5, color, alpha=0.30)
            _draw_box_rgb(
                left,
                np.asarray([box[0] * width, box[1] * height, box[2] * width, box[3] * height], dtype=np.float32),
                color,
                f"obj{o}",
            )
        state_rgb = scalar_depth_map_to_rgb(state_map[t].cpu().numpy())
        box_rgb = scalar_depth_map_to_rgb(box_map[t].cpu().numpy())
        mask_rgb = scalar_depth_map_to_rgb(mask_map[t].cpu().numpy())
        panel = np.concatenate([left, state_rgb, box_rgb, mask_rgb], axis=1)
        cv2.putText(panel, "raw + GT box + SAM mask", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "state GT", (width + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "Depth Anything + box pool", (2 * width + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "Depth Anything + SAM mask pool", (3 * width + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(panel)
    return np.stack(frames, axis=0)


def _pil_data_url(image_rgb: np.ndarray) -> str:
    image = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _build_html(summary: dict, output_dir: Path) -> Path:
    preview_cards = []
    for item in summary["frame_previews"]:
        preview_cards.append(
            f"""
            <figure>
              <img src="{item['src']}" style="width:100%; border:1px solid #ccc; background:#000;" />
              <figcaption>{item['caption']}</figcaption>
            </figure>
            """
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Depth Anything SAM Mask Pool Viewer</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:16px; }}
    video {{ width:100%; border:1px solid #ccc; background:#000; }}
    img {{ display:block; }}
    pre {{ background:#fff; border:1px solid #ddd; padding:16px; white-space:pre-wrap; }}
    figure {{ margin:0; }}
    figcaption {{ font-size:12px; color:#444; margin-top:4px; }}
  </style>
</head>
<body>
  <h1>Depth Anything: GT Box Pool vs SAM Mask Pool</h1>
  <p>流程：对当前 sample 的 <code>context_frames</code> 跑 Depth Anything；再用同一帧的 GT box 提示 SAM2 image predictor 生成对应 mask；最后分别用 <code>GT box</code> 和 <code>SAM mask</code> 对深度图做 median pooling，并和 <code>state depth</code> 对比。</p>
  <p><b>Sample:</b> {summary['sample_npz']}</p>
  <div class="grid">
    <figure>
      <video controls preload="metadata" playsinline src="{summary['panel_video_name']}"></video>
      <figcaption>四列对比视频：raw+SAM / state / box pool / SAM mask pool</figcaption>
    </figure>
    <figure>
      <video controls preload="metadata" playsinline src="{summary['depth_video_name']}"></video>
      <figcaption>Depth Anything context 8f 输出视频</figcaption>
    </figure>
  </div>
  <h2>Frame Previews</h2>
  <div class="grid">
    {''.join(preview_cards)}
  </div>
  <h2>Summary</h2>
  <pre>{json.dumps(summary['metrics'], indent=2, ensure_ascii=False)}</pre>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    args = _parse_args()
    sample_npz = Path(args.sample_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_script = Path(args.depth_script).expanduser().resolve()
    depth_checkpoint = Path(args.depth_checkpoint).expanduser().resolve()

    tensors = load_npz_tensor_dict(sample_npz)
    context_frames = tensors["context_frames"].float()
    context_boxes = tensors["context_boxes"].float()
    context_states = tensors["context_states"].float()
    num_objects = min(int(context_boxes.shape[1]), int(args.aux_max_objects))
    obj_idx = _object_indices(context_boxes, num_objects)
    boxes_used = context_boxes[:, obj_idx]

    context_video_path = output_dir / f"{sample_npz.stem}__context8f.mp4"
    if not context_video_path.is_file():
        _write_h264_mp4(context_video_path, _frames_to_rgb_uint8(context_frames), fps=8)

    if args.depth_video is not None and str(args.depth_video).strip():
        depth_video_path = Path(args.depth_video).expanduser().resolve()
    else:
        depth_video_path = output_dir / f"{sample_npz.stem}__depth_anything_context8f_h264.mp4"
        _run_depth_anything(context_video_path, depth_video_path, depth_script, depth_checkpoint, str(args.depth_gpu))

    raw_frames_rgb = _read_video_rgb(context_video_path)
    depth_gray = torch.from_numpy(_read_video_gray(depth_video_path)).float()

    tracker = SAM2MotionTracker(device=str(args.device), enable_text_prompt=False)
    sam_masks = _build_sam_masks_from_gt_boxes(tracker, context_frames, boxes_used)

    valid_boxes = ((boxes_used[..., 2] - boxes_used[..., 0]) > 1.0e-6) & ((boxes_used[..., 3] - boxes_used[..., 1]) > 1.0e-6)
    box_pool_framewise = torch.zeros((context_boxes.shape[0], num_objects, 1), dtype=torch.float32)
    for t in range(int(context_boxes.shape[0])):
        for o in range(num_objects):
            if not bool(valid_boxes[t, o].item()):
                continue
            x0, y0, x1, y1 = boxes_used[t, o].tolist()
            px0 = max(0, min(int(np.floor(x0 * depth_gray.shape[2])), depth_gray.shape[2] - 1))
            py0 = max(0, min(int(np.floor(y0 * depth_gray.shape[1])), depth_gray.shape[1] - 1))
            px1 = max(px0 + 1, min(int(np.ceil(x1 * depth_gray.shape[2])), depth_gray.shape[2]))
            py1 = max(py0 + 1, min(int(np.ceil(y1 * depth_gray.shape[1])), depth_gray.shape[1]))
            roi = depth_gray[t, py0:py1, px0:px1]
            if roi.numel() > 0:
                box_pool_framewise[t, o, 0] = roi.median()

    mask_pool_framewise = _pool_depth_from_masks_median(depth_gray, sam_masks)
    state_framewise = _gather_object_state_depth(context_states, obj_idx, int(args.depth_target_state_index))

    state_latent = group_last(state_framewise.unsqueeze(0), int(args.latent_frames))[0]
    box_latent = group_last(box_pool_framewise.unsqueeze(0), int(args.latent_frames))[0]
    mask_latent = group_last(mask_pool_framewise.unsqueeze(0), int(args.latent_frames))[0]

    state_map = scalar_depth_to_box_map(
        state_latent.unsqueeze(0),
        group_last(boxes_used.unsqueeze(0), int(args.latent_frames))[0].unsqueeze(0),
        group_last(valid_boxes.unsqueeze(0).unsqueeze(-1).float(), int(args.latent_frames))[0, ..., 0].bool().unsqueeze(0),
        image_hw=(raw_frames_rgb.shape[1], raw_frames_rgb.shape[2]),
    )[0]
    box_map = scalar_depth_to_box_map(
        box_latent.unsqueeze(0),
        group_last(boxes_used.unsqueeze(0), int(args.latent_frames))[0].unsqueeze(0),
        group_last(valid_boxes.unsqueeze(0).unsqueeze(-1).float(), int(args.latent_frames))[0, ..., 0].bool().unsqueeze(0),
        image_hw=(raw_frames_rgb.shape[1], raw_frames_rgb.shape[2]),
    )[0]
    mask_map = scalar_depth_to_box_map(
        mask_latent.unsqueeze(0),
        group_last(boxes_used.unsqueeze(0), int(args.latent_frames))[0].unsqueeze(0),
        group_last(valid_boxes.unsqueeze(0).unsqueeze(-1).float(), int(args.latent_frames))[0, ..., 0].bool().unsqueeze(0),
        image_hw=(raw_frames_rgb.shape[1], raw_frames_rgb.shape[2]),
    )[0]

    latent_group = int(context_frames.shape[0]) // int(args.latent_frames)
    latent_indices = [i * latent_group + (latent_group - 1) for i in range(int(args.latent_frames))]
    panel_frames = _render_panel_frames(
        raw_frames_rgb=np.asarray(raw_frames_rgb)[latent_indices],
        boxes_norm=boxes_used[latent_indices],
        sam_masks=sam_masks[latent_indices],
        state_map=state_map,
        box_map=box_map,
        mask_map=mask_map,
    )
    panel_video_path = output_dir / f"{sample_npz.stem}__depth_box_vs_sam_mask_panel.mp4"
    _write_h264_mp4(panel_video_path, panel_frames, fps=2)

    frame_previews = []
    for idx, frame in enumerate(panel_frames):
        frame_previews.append(
            {
                "caption": f"latent frame {idx} (source t={latent_indices[idx]})",
                "src": _pil_data_url(frame),
            }
        )

    metrics = {
        "latent_indices": latent_indices,
        "state_latent_values": state_latent[:, :, 0].tolist(),
        "box_latent_values": box_latent[:, :, 0].tolist(),
        "sam_mask_latent_values": mask_latent[:, :, 0].tolist(),
        "sam_mask_pixel_count_per_frame_object": sam_masks.sum(dim=(-1, -2)).tolist(),
        "box_minus_mask_l1_mean": float((box_latent - mask_latent).abs().mean().item()),
        "state_minus_mask_l1_mean": float((state_latent - mask_latent).abs().mean().item()),
        "state_minus_box_l1_mean": float((state_latent - box_latent).abs().mean().item()),
        "shapes": {
            "context_frames": list(context_frames.shape),
            "context_boxes": list(context_boxes.shape),
            "sam_masks": list(sam_masks.shape),
            "depth_gray": list(depth_gray.shape),
            "state_latent": list(state_latent.shape),
            "box_latent": list(box_latent.shape),
            "sam_mask_latent": list(mask_latent.shape),
        },
    }
    summary = {
        "sample_npz": str(sample_npz),
        "panel_video_name": panel_video_path.name,
        "depth_video_name": depth_video_path.name,
        "frame_previews": frame_previews,
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = _build_html(summary, output_dir)
    print(f"report: {html_path}")

    if args.serve:
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *handler_args, **handler_kwargs):
                super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

        class ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        with ReusableTCPServer(("0.0.0.0", int(args.port)), Handler) as httpd:
            print(f"serving report at http://0.0.0.0:{int(args.port)}/index.html")
            httpd.serve_forever()


if __name__ == "__main__":
    main()
