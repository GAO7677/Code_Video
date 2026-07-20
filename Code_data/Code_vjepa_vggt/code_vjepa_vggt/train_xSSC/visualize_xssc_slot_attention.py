#!/usr/bin/env python3
"""Visualize xSSC slot-attention maps over context video frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as train
from code_vjepa_vggt.utils.video_io import (
    read_video_prefix,
)


def _resolve_video_path(payload: dict[str, object], json_path: Path) -> Path:
    for key in ("source_video", "input_video"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (json_path.parent / path).resolve()
            return path.resolve()
    raise ValueError(f"{json_path} does not contain source_video or input_video")


def _colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(heatmap.min())
    denom = float(heatmap.max())
    if denom > 1.0e-8:
        heatmap = heatmap / denom
    r = np.clip(1.5 * heatmap, 0.0, 1.0)
    g = np.clip(1.5 * (1.0 - np.abs(heatmap - 0.5) * 2.0), 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - heatmap), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _overlay(frame: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    base = frame.astype(np.float32) / 255.0
    color = _colorize_heatmap(heatmap)
    mask = np.clip(heatmap[..., None], 0.0, 1.0)
    out = base * (1.0 - alpha * mask) + color * (alpha * mask)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _cover_crop_to_tensor(
    frames: np.ndarray,
    *,
    target_hw: tuple[int, int],
    cover_crop_hw: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, int | float | list[int]]]:
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    _, _, src_h, src_w = tensor.shape
    crop_h, crop_w = int(cover_crop_hw[0]), int(cover_crop_hw[1])
    out_h, out_w = int(target_hw[0]), int(target_hw[1])
    scale = max(crop_h / float(src_h), crop_w / float(src_w))
    resized_h = max(crop_h, int(round(src_h * scale)))
    resized_w = max(crop_w, int(round(src_w * scale)))
    resized = F.interpolate(
        tensor,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    crop_top = max(0, (resized_h - crop_h) // 2)
    crop_left = max(0, (resized_w - crop_w) // 2)
    cropped = resized[:, :, crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]
    if (crop_h, crop_w) != (out_h, out_w):
        cropped = F.interpolate(cropped, size=(out_h, out_w), mode="bilinear", align_corners=False)
    cropped = cropped / 255.0 * 2.0 - 1.0
    metadata = {
        "source_hw": [int(src_h), int(src_w)],
        "target_hw": [out_h, out_w],
        "cover_crop_hw": [crop_h, crop_w],
        "cover_scale": float(scale),
        "resized_hw": [int(resized_h), int(resized_w)],
        "cover_crop_yxhw_in_resized": [int(crop_top), int(crop_left), crop_h, crop_w],
    }
    return cropped.permute(1, 0, 2, 3).contiguous(), metadata


def _project_preprocessed_box_to_source(
    *,
    box_yxhw: tuple[int, int, int, int],
    preprocess: dict[str, int | float | list[int]],
) -> tuple[int, int, int, int]:
    box_y, box_x, box_h, box_w = [int(v) for v in box_yxhw]
    crop_y, crop_x, _, _ = [int(v) for v in preprocess["cover_crop_yxhw_in_resized"]]  # type: ignore[index]
    scale = float(preprocess["cover_scale"])
    src_h, src_w = [int(v) for v in preprocess["source_hw"]]  # type: ignore[index]
    y0 = int(round((crop_y + box_y) / scale))
    x0 = int(round((crop_x + box_x) / scale))
    y1 = int(round((crop_y + box_y + box_h) / scale))
    x1 = int(round((crop_x + box_x + box_w) / scale))
    y0 = max(0, min(src_h, y0))
    y1 = max(0, min(src_h, y1))
    x0 = max(0, min(src_w, x0))
    x1 = max(0, min(src_w, x1))
    return y0, x0, max(0, y1 - y0), max(0, x1 - x0)


def _preprocess_xssc(context_video: torch.Tensor, input_size: int) -> torch.Tensor:
    frames = context_video.permute(0, 2, 1, 3, 4).float()
    batch, time_steps, channels, height, width = frames.shape
    crop_size = min(int(height), int(width))
    top = (int(height) - crop_size) // 2
    left = (int(width) - crop_size) // 2
    frames = frames[..., top : top + crop_size, left : left + crop_size]
    frames = frames.reshape(batch * time_steps, channels, crop_size, crop_size)
    frames = F.interpolate(
        frames,
        size=(int(input_size), int(input_size)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    frames = (frames + 1.0).mul(127.5).clamp(0.0, 255.0)
    mean = frames.new_tensor(train.XSSC_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = frames.new_tensor(train.XSSC_IMAGENET_STD).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames.view(batch, time_steps, channels, int(input_size), int(input_size))


@torch.no_grad()
def _extract_slots_and_attention(model, context_video: torch.Tensor):
    xssc_video = _preprocess_xssc(context_video, model.xssc_input_size)
    model.xssc.eval()
    batch, time_steps, _, _, _ = xssc_video.shape
    flat_video = xssc_video.flatten(0, 1)
    autocast_enabled = flat_video.device.type == "cuda"
    with torch.autocast(
        device_type=flat_video.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        feature = model.xssc.encode_backbone(flat_video).detach()
        _, _, feature_h, feature_w = feature.shape
        encoded = feature.permute(0, 2, 3, 1)
        encoded = model.xssc.encode_posit_embed(encoded).flatten(1, 2)
        encoded = model.xssc.encode_project(encoded)
        encoded = encoded.view(batch, time_steps, encoded.shape[1], encoded.shape[2])

        slots = None
        attentions = []
        for frame_id in range(time_steps):
            if frame_id == 0:
                query = model.xssc.initializ(batch)
            else:
                query = model.xssc.transit(slots, encoded[:, : frame_id + 1])
            num_iter = None if frame_id == 0 else 1
            current_slots, current_attention = model.xssc.aggregat(
                encoded[:, frame_id], query, num_iter=num_iter
            )
            slots = (
                current_slots[:, None]
                if slots is None
                else torch.cat((slots, current_slots[:, None]), dim=1)
            )
            attentions.append(current_attention.view(batch, model.xssc_num_slots, feature_h, feature_w))

    attention = torch.stack(attentions, dim=1)
    return slots, attention


def _save_contact_sheet(images: list[np.ndarray], output_path: Path) -> None:
    pil_images = [Image.fromarray(image) for image in images]
    widths, heights = zip(*(image.size for image in pil_images))
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    x = 0
    for image in pil_images:
        canvas.paste(image, (x, 0))
        x += image.size[0]
    canvas.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--input-cover-crop-height", type=int, default=512)
    parser.add_argument("--input-cover-crop-width", type=int, default=896)
    parser.add_argument("--context-frames", type=int, default=train.XSSC_NUM_CONTEXT_FRAMES)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--overlay-video",
        type=Path,
        default=None,
        help="Optional generated video to receive the slot overlays. When omitted, overlays source frames.",
    )
    parser.add_argument(
        "--overlay-name",
        default=None,
        help="Optional label written to metadata, e.g. clean/zero/noise/shuffle_slot.",
    )
    args = parser.parse_args()

    case_json = args.case_json.expanduser().resolve()
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    video_path = _resolve_video_path(payload, case_json)
    frames, frame_indices = read_video_prefix(video_path, int(args.context_frames))
    context_video_single, preprocess_metadata = _cover_crop_to_tensor(
        frames,
        target_hw=(int(args.height), int(args.width)),
        cover_crop_hw=(int(args.input_cover_crop_height), int(args.input_cover_crop_width)),
    )
    context_video = context_video_single.unsqueeze(0).to(
        device=torch.device(args.device),
        dtype=torch.bfloat16,
    )

    model = SimpleNamespace()
    xssc, slot_dim, num_slots = train._load_xssc_model(
        xssc_root=train.DEFAULT_XSSC_ROOT,
        config_path=train.DEFAULT_XSSC_CONFIG,
        checkpoint_path=train.DEFAULT_XSSC_CHECKPOINT,
        device=torch.device(args.device),
    )
    model.xssc = xssc
    model.xssc_slot_dim = slot_dim
    model.xssc_num_slots = num_slots
    model.xssc_input_size = 256

    slots, attention = _extract_slots_and_attention(model, context_video)
    attention = attention[0].float().cpu()
    if args.overlay_video is None:
        overlay_frame_uint8 = frames.astype(np.uint8)
        overlay_space = "source_video"
    else:
        overlay_frame_uint8 = iio.imread(args.overlay_video.expanduser().resolve()).astype(np.uint8)
        if overlay_frame_uint8.ndim != 4 or overlay_frame_uint8.shape[-1] != 3:
            raise ValueError(
                f"overlay video must decode to [T,H,W,3], got {overlay_frame_uint8.shape}"
            )
        overlay_space = "generated_video"

    output_dir = args.output_dir.expanduser().resolve()
    frames_dir = output_dir / "frames"
    videos_dir = output_dir / "videos"
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    target_h, target_w = int(args.height), int(args.width)
    crop_size = min(target_h, target_w)
    top = (target_h - crop_size) // 2
    left = (target_w - crop_size) // 2
    overlay_h, overlay_w = int(overlay_frame_uint8.shape[1]), int(overlay_frame_uint8.shape[2])
    if args.overlay_video is None:
        overlay_top, overlay_left, overlay_crop_h, overlay_crop_w = _project_preprocessed_box_to_source(
            box_yxhw=(top, left, crop_size, crop_size),
            preprocess=preprocess_metadata,
        )
    else:
        if (overlay_h, overlay_w) != (target_h, target_w):
            raise ValueError(
                "generated overlay video resolution must match Wan output "
                f"{(target_h, target_w)}, got {(overlay_h, overlay_w)}"
            )
        overlay_top, overlay_left, overlay_crop_h, overlay_crop_w = top, left, crop_size, crop_size
    overlay_count = min(int(attention.shape[0]), int(overlay_frame_uint8.shape[0]))
    metadata = {
        "case_json": str(case_json),
        "source_video": str(video_path),
        "overlay_video": None if args.overlay_video is None else str(args.overlay_video.expanduser().resolve()),
        "overlay_name": args.overlay_name,
        "overlay_space": overlay_space,
        "frame_indices": [int(v) for v in frame_indices.tolist()],
        "attention_shape": list(attention.shape),
        "slots_shape": list(slots.shape),
        "overlay_size": [overlay_h, overlay_w],
        "preprocess": preprocess_metadata,
        "xssc_crop_box_yxhw_in_preprocessed": [top, left, crop_size, crop_size],
        "xssc_crop_box_yxhw_in_overlay": [
            overlay_top,
            overlay_left,
            overlay_crop_h,
            overlay_crop_w,
        ],
        "time_axis": {
            "type": "context_to_generated_prefix" if args.overlay_video is not None else "context_frames",
            "overlay_frame_count": int(overlay_frame_uint8.shape[0]),
            "attention_frame_count": int(attention.shape[0]),
            "strictly_aligned_overlay_frames": int(overlay_count),
            "note": (
                "For generated-video overlays, xSSC slot attention has 8 context-frame "
                "steps aligned to generated frames 0..7. Later generated frames are left "
                "unoverlaid because there is no per-generated-frame xSSC attention."
                if args.overlay_video is not None
                else "xSSC slot attention has 8 context-frame steps aligned to frame_indices, not 49 generated/latent frames."
            ),
        },
        "slot_videos": {},
    }

    for slot_id in range(int(num_slots)):
        slot_frames = []
        for frame_id in range(int(overlay_frame_uint8.shape[0])):
            if frame_id < overlay_count:
                heat = attention[frame_id, slot_id].unsqueeze(0).unsqueeze(0)
                heat = F.interpolate(
                    heat,
                    size=(overlay_crop_h, overlay_crop_w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()
                full_heat = np.zeros((overlay_h, overlay_w), dtype=np.float32)
                full_heat[
                    overlay_top : overlay_top + overlay_crop_h,
                    overlay_left : overlay_left + overlay_crop_w,
                ] = heat
                full_heat = full_heat - float(full_heat.min())
                if float(full_heat.max()) > 1.0e-8:
                    full_heat = full_heat / float(full_heat.max())
                overlaid = _overlay(overlay_frame_uint8[frame_id], full_heat, alpha=float(args.alpha))
            else:
                overlaid = overlay_frame_uint8[frame_id]
            if frame_id < overlay_count:
                frame_path = frames_dir / f"slot{slot_id:02d}_frame{frame_id:02d}.jpg"
                Image.fromarray(overlaid).save(frame_path, quality=95)
            slot_frames.append(overlaid)
        video_path_out = videos_dir / f"slot{slot_id:02d}_attention_overlay.mp4"
        iio.imwrite(video_path_out, np.stack(slot_frames, axis=0), fps=int(args.fps), codec="libx264")
        _save_contact_sheet(slot_frames, frames_dir / f"slot{slot_id:02d}_contact_sheet.jpg")
        metadata["slot_videos"][f"slot{slot_id:02d}"] = str(video_path_out)

    (output_dir / "slot_attention_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
