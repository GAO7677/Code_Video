#!/usr/bin/env python3
"""Visualize the real Wan+DINOv3-xSSC training input preprocessing pipeline."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F


TRAIN_XSSC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_XSSC_ROOT.parent
PACKAGE_PARENT = REPO_ROOT.parent
EXPERIMENT = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
sys.path.insert(0, str(PACKAGE_PARENT))
sys.path.insert(0, str(TRAIN_XSSC_ROOT))
sys.path.insert(0, str(EXPERIMENT / "third_party/dinov3"))
sys.path.insert(0, str(EXPERIMENT / "upstream"))

from code_vjepa_vggt.train_xSSC import train_xssc_context_slots_dinov3 as train_xssc  # noqa: E402
from compare_movi_c_gt_vs_gdino_sam2 import slot_overlay  # noqa: E402
from visualize_movi_c_sam2_amg import (  # noqa: E402
    draw_selected_boxes,
    overlay_masks,
    resolve_sam2_config_name,
    select_xssc_candidates,
)


DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/dinov3_wan_xssc_train_preprocess_smoke"
)
PALETTE = np.asarray(
    [
        [239, 68, 68],
        [59, 130, 246],
        [34, 197, 94],
        [250, 204, 21],
        [168, 85, 247],
        [6, 182, 212],
        [249, 115, 22],
        [236, 72, 153],
        [132, 204, 22],
        [20, 184, 166],
        [244, 114, 182],
    ],
    dtype=np.uint8,
)


def build_parser() -> argparse.ArgumentParser:
    parser = train_xssc.build_parser()
    group = parser.add_argument_group("preprocess_visualization")
    group.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    group.add_argument("--num_visual_samples", type=int, default=3)
    group.add_argument("--visual_indices", default="")
    group.add_argument("--device", default="cuda:0")
    group.add_argument("--fps", type=float, default=3.0)
    group.add_argument("--webp_quality", type=int, default=92)
    return parser


def to_uint8_video(context_video: torch.Tensor) -> np.ndarray:
    frames = context_video.permute(1, 2, 3, 0).float()
    frames = (frames + 1.0).mul(127.5).round().clamp(0, 255)
    return frames.to(torch.uint8).cpu().numpy()


def center_crop_frames(frames: torch.Tensor) -> tuple[torch.Tensor, dict]:
    _, _, _, height, width = frames.shape
    crop_size = min(int(height), int(width))
    top = (int(height) - crop_size) // 2
    left = (int(width) - crop_size) // 2
    cropped = frames[..., top : top + crop_size, left : left + crop_size]
    return cropped, {
        "crop_size": crop_size,
        "top": top,
        "left": left,
        "height": int(height),
        "width": int(width),
    }


def preprocess_xssc_exact(context_video: torch.Tensor, input_size: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict]:
    frames = context_video.unsqueeze(0).permute(0, 2, 1, 3, 4).float()
    cropped, crop_info = center_crop_frames(frames)
    batch, time_steps, channels, crop_size, _ = cropped.shape
    resized = F.interpolate(
        cropped.reshape(batch * time_steps, channels, crop_size, crop_size),
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    resized_rgb = (resized + 1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    mean = resized.new_tensor(train_xssc.base.XSSC_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(train_xssc.base.XSSC_IMAGENET_STD).view(1, 3, 1, 1)
    normalized = (resized.mul(127.5).add(127.5).clamp(0, 255) - mean) / std
    xssc_video = normalized.view(batch, time_steps, channels, input_size, input_size)
    cropped_rgb = (cropped[0] + 1.0).mul(127.5).round().clamp(0, 255)
    cropped_rgb = cropped_rgb.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    resized_rgb = resized_rgb.view(batch, time_steps, channels, input_size, input_size)[0]
    resized_rgb = resized_rgb.permute(0, 2, 3, 1).cpu().numpy()
    return xssc_video, cropped_rgb, resized_rgb, crop_info


def draw_crop_rect(frame: np.ndarray, crop_info: dict) -> np.ndarray:
    output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    left = int(crop_info["left"])
    top = int(crop_info["top"])
    size = int(crop_info["crop_size"])
    cv2.rectangle(output, (left, top), (left + size - 1, top + size - 1), (0, 255, 255), 3)
    cv2.putText(
        output,
        "center crop",
        (left + 8, max(24, top + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def make_mask_box_panel(image: np.ndarray, selected: list[dict]) -> np.ndarray:
    if selected:
        masked = overlay_masks(image, selected)
        boxed = draw_selected_boxes(masked, selected)
        return boxed
    return image.copy()


def write_image(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_WEBP_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError(f"could not write {path}")


def write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.ascontiguousarray(frames.astype(np.uint8))
    height, width = frames.shape[1:3]
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(path),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, round(image.shape[1] * (height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def make_pipeline_strip(
    raw: np.ndarray,
    crop_rect: np.ndarray,
    cropped: np.ndarray,
    xssc_input: np.ndarray,
    amg_panel: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    panels = [
        add_label(resize_to_height(raw, 256), "dataset RGB"),
        add_label(resize_to_height(crop_rect, 256), "crop window"),
        add_label(resize_to_height(cropped, 256), "cropped"),
        add_label(xssc_input, "xSSC 256"),
        add_label(amg_panel, "AMG boxes"),
        add_label(overlay, "slot overlay"),
    ]
    return np.concatenate(panels, axis=1)


def build_generator(args: argparse.Namespace, device: torch.device):
    grounded_sam2_root = "/home/gaoya/Grounded-SAM-2-main"
    if grounded_sam2_root not in sys.path:
        sys.path.insert(0, grounded_sam2_root)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    sam2 = build_sam2(
        resolve_sam2_config_name(args.xssc_sam2_config),
        str(args.xssc_sam2_checkpoint),
        device=str(device),
        mode="eval",
    )
    return SAM2AutomaticMaskGenerator(sam2)


def choose_indices(dataset, args: argparse.Namespace) -> list[int]:
    if str(args.visual_indices).strip():
        return [int(item) for item in str(args.visual_indices).replace(",", " ").split()]
    if all(hasattr(dataset, name) for name in ("source_lengths", "source_names")):
        indices = []
        start = 0
        for length in dataset.source_lengths:
            if len(indices) >= int(args.num_visual_samples):
                break
            indices.append(start + min(max(length // 2, 0), length - 1))
            start += length
        if indices:
            return indices[: int(args.num_visual_samples)]
    return list(range(min(int(args.num_visual_samples), len(dataset))))


def boxes_from_masks(selected: list[dict], num_slots: int, num_frames: int) -> torch.Tensor:
    if selected:
        masks = np.stack([item["segmentation"].astype(bool) for item in selected], axis=0)
    else:
        masks = np.zeros((0, 256, 256), dtype=bool)
    boxes = train_xssc.masks_to_repeated_boxes(masks, num_slots, num_frames)
    return torch.from_numpy(boxes[None]).float()


@torch.no_grad()
def extract_slots_attention(model, video: torch.Tensor, boxes: torch.Tensor, amp_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    batch, time_steps, _, _, _ = video.shape
    flat_video = video.flatten(0, 1)
    with torch.autocast(device_type=flat_video.device.type, dtype=amp_dtype, enabled=flat_video.device.type == "cuda"):
        feature = model.encode_backbone(flat_video).detach()
        encoded = feature.permute(0, 2, 3, 1)
        encoded = model.encode_posit_embed(encoded).flatten(1, 2)
        encoded = model.encode_project(encoded)
        encoded = encoded.view(batch, time_steps, encoded.shape[1], encoded.shape[2])
        boxes = boxes.to(device=encoded.device, dtype=encoded.dtype)
        slots = None
        attentions = []
        for frame_id in range(time_steps):
            if frame_id == 0:
                query = model.initializ(boxes[:, 0])
            else:
                query = model.transit(slots, encoded[:, : frame_id + 1])
            num_iter = None if frame_id == 0 else 1
            current_slots, current_attention = model.aggregat(
                encoded[:, frame_id],
                query,
                num_iter=num_iter,
            )
            current_slots = current_slots[:, None]
            slots = current_slots if slots is None else torch.cat((slots, current_slots), dim=1)
            attentions.append(current_attention)
        attention = torch.stack(attentions, dim=1)
        patch_side = int(round(attention.shape[-1] ** 0.5))
        attention = attention.view(batch, time_steps, attention.shape[2], patch_side, patch_side)
    return slots, attention, feature


def filter_args_from_train_args(args: argparse.Namespace) -> SimpleNamespace:
    return train_xssc._amg_filter_args_from_args(args)


def html_page(cases: list[dict], metadata: dict) -> str:
    cards = []
    for case in cases:
        safe_title = html.escape(case["title"])
        cards.append(
            f"""
            <section class="case">
              <h2>{safe_title}</h2>
              <div class="meta">
                <span>index {case['index']}</span>
                <span>source {html.escape(case['source'])}</span>
                <span>train video {case['train_video_shape']}</span>
                <span>raw {case['raw_shape']}</span>
                <span>xSSC {case['xssc_shape']}</span>
                <span>boxes {case['box_shape']}</span>
                <span>slots {case['slot_shape']}</span>
                <span>attention {case['attention_shape']}</span>
                <span>selected AMG masks {case['selected_masks']}</span>
              </div>
              <img class="strip" src="{case['strip']}" alt="preprocess strip for {safe_title}">
              <div class="videos">
                <figure><video src="{case['train_video']}" controls muted loop></video><figcaption>Wan train video</figcaption></figure>
                <figure><video src="{case['raw_video']}" controls muted loop></video><figcaption>dataset context RGB</figcaption></figure>
                <figure><video src="{case['crop_video']}" controls muted loop></video><figcaption>center-cropped context</figcaption></figure>
                <figure><video src="{case['xssc_video']}" controls muted loop></video><figcaption>xSSC 256 input</figcaption></figure>
                <figure><video src="{case['slot_video']}" controls muted loop></video><figcaption>xSSC slot overlay</figcaption></figure>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DINOv3 xSSC Wan Train Preprocess</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:#101214; color:#e8ecef; font:14px system-ui,sans-serif; letter-spacing:0; }}
    header {{ position:sticky; top:0; z-index:3; padding:14px 18px; background:#16191c; border-bottom:1px solid #343a40; }}
    h1 {{ margin:0 0 6px; font-size:20px; }}
    .summary {{ color:#aeb6bf; }}
    main {{ max-width:1800px; margin:0 auto; padding:18px; }}
    .case {{ padding:18px 0 26px; border-bottom:1px solid #30363d; }}
    h2 {{ margin:0 0 10px; font-size:17px; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:8px; color:#bac2cb; margin:0 0 12px; }}
    .meta span {{ border:1px solid #3b424a; border-radius:6px; padding:5px 8px; background:#191d21; }}
    .strip {{ width:100%; height:auto; display:block; background:#000; border:1px solid #343a40; }}
    .videos {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-top:12px; }}
    figure {{ margin:0; min-width:0; }}
    video {{ display:block; width:100%; background:#000; border:1px solid #343a40; }}
    figcaption {{ padding:6px 2px; color:#b8c0c9; font-size:12px; }}
    @media(max-width:1300px) {{ .videos {{ grid-template-columns:repeat(3,1fr); }} }}
    @media(max-width:900px) {{ .videos {{ grid-template-columns:repeat(2,1fr); }} }}
    @media(max-width:650px) {{ main {{ padding:10px; }} .videos {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>DINOv3 xSSC Wan training preprocess</h1>
    <div class="summary">{html.escape(json.dumps(metadata, ensure_ascii=False))}</div>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""


def main() -> None:
    args = train_xssc.tvn.prepare_args(build_parser().parse_args())
    args.no_context_ratio = 0.0
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = torch.bfloat16

    dataset = train_xssc.base.build_dataset(args)
    indices = choose_indices(dataset, args)
    checkpoint = train_xssc.resolve_latest_xssc_checkpoint(
        args.xssc_checkpoint,
        args.xssc_checkpoint_latest_dir,
    )
    xssc, slot_dim, num_slots = train_xssc._load_dinov3_xssc_model(
        xssc_root=args.xssc_root,
        config_path=args.xssc_config,
        checkpoint_path=checkpoint,
        dinov3_root=args.dinov3_root,
        dinov3_checkpoint=args.dinov3_checkpoint,
        device=device,
    )
    generator = build_generator(args, device)
    filter_args = filter_args_from_train_args(args)

    output_dir = args.output_dir.resolve()
    asset_root = output_dir / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    cases = []
    for position, index in enumerate(indices, start=1):
        sample = dataset[int(index)]
        context_video = sample["context_video"]
        if context_video.ndim != 4:
            raise ValueError(f"context_video must be [C,T,H,W], got {tuple(context_video.shape)}")
        train_video = sample.get("video", context_video)
        if not isinstance(train_video, torch.Tensor) or train_video.ndim != 4:
            raise ValueError(f"video must be [C,T,H,W], got {type(train_video)!r}")
        train_rgb = to_uint8_video(train_video)
        raw_rgb = to_uint8_video(context_video)
        xssc_video, cropped_rgb, xssc_rgb, crop_info = preprocess_xssc_exact(
            context_video,
            int(args.xssc_input_size),
        )
        xssc_first = xssc_rgb[0]
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            annotations = generator.generate(xssc_first)
        selected = select_xssc_candidates(
            annotations,
            xssc_first.shape[0] * xssc_first.shape[1],
            filter_args,
            image=xssc_first,
        )
        boxes = boxes_from_masks(selected, num_slots, int(xssc_video.shape[1]))
        slots, attention, _ = extract_slots_attention(
            xssc,
            xssc_video.to(device=device),
            boxes.to(device=device),
            amp_dtype,
        )
        labels = attention[0].float().cpu().numpy().argmax(axis=1).astype(np.uint8)
        overlay = slot_overlay(xssc_rgb, labels)
        amg_panel = make_mask_box_panel(xssc_first, selected)
        crop_rect = draw_crop_rect(raw_rgb[0], crop_info)
        strip = make_pipeline_strip(
            raw_rgb[0],
            crop_rect,
            cropped_rgb[0],
            xssc_first,
            amg_panel,
            overlay[0],
        )

        case_dir = asset_root / f"case_{position:02d}_index_{int(index):06d}"
        write_image(case_dir / "pipeline_frame0.webp", strip, args.webp_quality)
        write_video(case_dir / "wan_train_video.mp4", train_rgb, args.fps)
        write_video(case_dir / "raw_context.mp4", raw_rgb, args.fps)
        write_video(case_dir / "center_crop.mp4", cropped_rgb, args.fps)
        write_video(case_dir / "xssc_input_256.mp4", xssc_rgb, args.fps)
        write_video(case_dir / "xssc_slot_overlay.mp4", overlay, args.fps)

        metadata = dict(sample.get("metadata", {}))
        source = str(metadata.get("dataset_source", "unknown"))
        title = f"case {position:02d} | {source}"
        case_payload = {
            "title": title,
            "index": int(index),
            "source": source,
            "train_video_shape": list(train_video.shape),
            "raw_shape": list(context_video.shape),
            "xssc_shape": list(xssc_video.shape),
            "box_shape": list(boxes.shape),
            "slot_shape": list(slots.shape),
            "attention_shape": list(attention.shape),
            "selected_masks": len(selected),
            "crop_info": crop_info,
            "metadata": metadata,
            "strip": str((case_dir / "pipeline_frame0.webp").relative_to(output_dir)),
            "train_video": str((case_dir / "wan_train_video.mp4").relative_to(output_dir)),
            "raw_video": str((case_dir / "raw_context.mp4").relative_to(output_dir)),
            "crop_video": str((case_dir / "center_crop.mp4").relative_to(output_dir)),
            "xssc_video": str((case_dir / "xssc_input_256.mp4").relative_to(output_dir)),
            "slot_video": str((case_dir / "xssc_slot_overlay.mp4").relative_to(output_dir)),
        }
        (case_dir / "metadata.json").write_text(json.dumps(case_payload, indent=2, ensure_ascii=False) + "\n")
        cases.append(case_payload)
        print(f"[{position}/{len(indices)}] index={index} source={source} selected_masks={len(selected)}")

    report = {
        "indices": indices,
        "checkpoint": checkpoint,
        "dinov3_checkpoint": args.dinov3_checkpoint,
        "slot_dim": slot_dim,
        "num_slots": num_slots,
        "dataset_stats": getattr(dataset, "dataset_stats", None),
    }
    (output_dir / "metadata.json").write_text(json.dumps({"report": report, "cases": cases}, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "index.html").write_text(html_page(cases, report))
    print(f"viewer={output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
