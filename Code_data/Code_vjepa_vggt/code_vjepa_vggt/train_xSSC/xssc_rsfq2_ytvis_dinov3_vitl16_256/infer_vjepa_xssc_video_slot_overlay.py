#!/usr/bin/env python3
"""Run V-JEPA xSSC on a video and render fixed-color slot overlays."""

from argparse import ArgumentParser
import html
import json
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

from train_ddp_ytvis_hq import (  # noqa: E402
    checkpoint_load_summary,
    load_matching_checkpoint,
)
from visualize_vjepa_xssc_downstream_one_train_step import (  # noqa: E402
    FONT,
    FONT_SMALL,
    add_header,
    fit_width,
    set_seed,
    write_h264,
)


DEFAULT_CONFIG = ROOT / (
    "upstream/config-randsfq/"
    "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512.py"
)
DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/agent-data/checkpoints/"
    "xssc_vjepa2_1_video_noncausal_ytvis_hq_bs64_steps10000/"
    "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512/42/"
    "step-007000.pth"
)
DEFAULT_INPUT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/"
    "industrial_s1_scale2_merged_h264_batch1500/val/F1_single_object/"
    "sample_000301/source_video.mp4"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/vjepa_xssc_video_slot_overlay"
)
IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)
SLOT_COLORS = np.asarray(
    [
        [239, 68, 68],
        [59, 130, 246],
        [34, 197, 94],
        [250, 204, 21],
        [168, 85, 247],
        [6, 182, 212],
        [249, 115, 22],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-video", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum decoded frames; use 0 for the complete video.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resize-mode", choices=("center-crop", "padding"), default="center-crop")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=None)
    return parser.parse_args()


def read_video(path, max_frames, stride):
    if stride < 1:
        raise ValueError("frame-stride must be >= 1")
    metadata = iio.immeta(path, plugin="pyav")
    frames = []
    for raw_index, frame in enumerate(iio.imiter(path, plugin="pyav")):
        if raw_index % stride:
            continue
        frames.append(np.asarray(frame)[..., :3])
        if max_frames > 0 and len(frames) >= max_frames:
            break
    if len(frames) < 2:
        raise ValueError(f"input video must yield at least two frames, got {len(frames)}")
    return np.stack(frames).astype(np.uint8), metadata


def preprocess_frames(frames, size, resize_mode):
    processed = []
    for frame in frames:
        height, width = frame.shape[:2]
        if resize_mode == "center-crop":
            side = min(height, width)
            y0 = (height - side) // 2
            x0 = (width - side) // 2
            image = Image.fromarray(frame[y0 : y0 + side, x0 : x0 + side])
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        else:
            scale = min(size / width, size / height)
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            resized = Image.fromarray(frame).resize(new_size, Image.Resampling.BILINEAR)
            image = Image.new("RGB", (size, size), (0, 0, 0))
            image.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
        processed.append(np.asarray(image))
    return np.stack(processed).astype(np.uint8)


def normalize_frames(frames):
    video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float()[None]
    return (video - IMAGENET_MEAN) / IMAGENET_STD


def load_model(config_file, checkpoint, device):
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(config_file)
    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    load_report = load_matching_checkpoint(
        model,
        checkpoint,
        exclude_patterns=(),
        allowed_missing_patterns=cfg.checkpoint_allowed_missing,
        expected_source_variant=cfg.variant_name,
        expected_source_step=7000,
    )
    model.freez(cfg.freez, verbose=False)
    return cfg, model.to(device).eval(), load_report


@torch.inference_mode()
def infer_slot_labels(model, video, device, amp_dtype):
    with torch.autocast("cuda", dtype=amp_dtype):
        output = model(batch={"video": video.to(device)})
    attention = output["attentd"]
    if attention.ndim != 5:
        raise RuntimeError(f"expected attentd [B,T,S,H,W], got {tuple(attention.shape)}")
    labels = attention[0].argmax(dim=1).to(torch.uint8).cpu().numpy()
    return labels, tuple(attention.shape)


def upsample_labels(labels, height, width):
    tensor = torch.from_numpy(labels.astype(np.int64))[:, None].float()
    return (
        torch.nn.functional.interpolate(tensor, size=(height, width), mode="nearest")
        .squeeze(1)
        .to(torch.uint8)
        .numpy()
    )


def slot_boundaries(labels):
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary


def render_overlay(frame, labels, alpha):
    colors = SLOT_COLORS[labels % len(SLOT_COLORS)]
    result = np.rint(
        frame.astype(np.float32) * (1 - alpha) + colors.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)
    result[slot_boundaries(labels)] = 255
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    for slot_id in range(len(SLOT_COLORS)):
        ys, xs = np.where(labels == slot_id)
        if len(xs) < labels.size * 0.012:
            continue
        x = int(np.median(xs))
        y = int(np.median(ys))
        draw.text(
            (x, y),
            f"S{slot_id}",
            fill=tuple(int(value) for value in SLOT_COLORS[slot_id]),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            font=FONT,
            anchor="mm",
        )
    return np.asarray(image)


def add_legend(frame):
    height = 34
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + height), (15, 18, 22))
    image.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(image)
    cell = max(1, frame.shape[1] // len(SLOT_COLORS))
    for slot_id, color in enumerate(SLOT_COLORS):
        x = slot_id * cell + 5
        y = frame.shape[0] + 9
        draw.rectangle((x, y, x + 14, y + 14), fill=tuple(int(value) for value in color))
        draw.text((x + 19, y), f"S{slot_id}", fill=(225, 230, 235), font=FONT_SMALL)
    return np.asarray(image)


def build_page(report):
    legend = "".join(
        f'<span><i style="background:rgb({r},{g},{b})"></i>S{slot}</span>'
        for slot, (r, g, b) in enumerate(SLOT_COLORS.tolist())
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA xSSC slot overlay</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#0d1117;color:#e6edf3;font:14px system-ui,sans-serif}}header{{position:sticky;top:0;z-index:3;padding:14px 22px;background:rgba(13,17,23,.96);border-bottom:1px solid #30363d}}h1{{margin:0 0 4px;font-size:20px}}main{{max-width:1450px;margin:auto;padding:18px}}.card{{padding:15px;border:1px solid #30363d;border-radius:8px;background:#161b22}}video{{width:100%;display:block;background:#000;border-radius:6px}}.sub{{color:#8b949e}}.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}}.legend span{{display:flex;align-items:center;gap:5px}}.legend i{{display:inline-block;width:15px;height:15px;border-radius:3px}}code{{color:#79c0ff;overflow-wrap:anywhere}}button{{position:fixed;right:20px;bottom:20px;border:1px solid #388bfd;border-radius:7px;padding:10px 15px;color:white;background:#1f6feb;font-weight:650;cursor:pointer}}table{{width:100%;border-collapse:collapse;margin-top:12px}}td{{padding:8px;border-bottom:1px solid #30363d}}td:first-child{{color:#8b949e;width:180px}}
</style></head><body><header><h1>V-JEPA xSSC 彩色 Slot Overlay</h1><div class="sub">V-JEPA2.1 ViT-L/16 video encoder → noncausal xSSC step-7000 → decoder slot assignment</div></header><main><section class="card"><video id="video" controls muted loop autoplay playsinline src="comparison.mp4" poster="poster.jpg"></video><div class="legend">{legend}</div><table><tr><td>输入视频</td><td><code>{html.escape(report['input_video'])}</code></td></tr><tr><td>有效模型输入</td><td>{report['processed_frames']} frames × 256×256，{html.escape(report['resize_mode'])}，fps={report['output_fps']:.3f}</td></tr><tr><td>V-JEPA/xSSC shape</td><td><code>{html.escape(str(report['shapes']))}</code></td></tr><tr><td>Checkpoint</td><td><code>{html.escape(report['checkpoint'])}</code></td></tr></table><p class="sub">颜色仅表示本次 forward 中的 slot index，不表示语义类别。V-JEPA tubelet=2，每个 tubelet 的 slot map 在视频中覆盖对应的两帧；白线是 slot 边界。</p></section></main><button id="replay">重新播放</button><script>document.getElementById('replay').onclick=()=>{{const v=document.getElementById('video');v.currentTime=0;v.play()}}</script></body></html>"""


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    input_video = args.input_video.resolve()
    config_file = args.cfg_file.resolve()
    checkpoint = args.checkpoint.resolve()
    for path in (input_video, config_file, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_raw, video_metadata = read_video(input_video, args.max_frames, args.frame_stride)
    frames = preprocess_frames(frames_raw, 256, args.resize_mode)
    original_frame_count = len(frames)
    padded = original_frame_count % 2
    if padded:
        frames_model = np.concatenate([frames, frames[-1:]], axis=0)
    else:
        frames_model = frames

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device("cuda", 0)
    cfg, model, load_report = load_model(config_file, checkpoint, device)
    video = normalize_frames(frames_model)
    slot_labels, attention_shape = infer_slot_labels(
        model, video, device, getattr(torch, cfg.amp_dtype)
    )
    labels_256 = upsample_labels(slot_labels, 256, 256)
    labels_per_frame = np.repeat(labels_256, 2, axis=0)[:original_frame_count]

    source_fps = float(video_metadata.get("fps", 30.0)) / args.frame_stride
    output_fps = source_fps if args.fps is None else float(args.fps)
    comparison_frames = []
    overlay_frames = []
    for frame_index, (frame, labels) in enumerate(zip(frames, labels_per_frame)):
        tubelet_index = frame_index // 2
        source = add_header(
            frame,
            [
                "MODEL INPUT",
                f"frame {frame_index} · {args.resize_mode}/resize 256×256",
            ],
            color=(121, 192, 255),
        )
        overlay = render_overlay(frame, labels, args.alpha)
        overlay = add_header(
            overlay,
            ["xSSC SLOT OVERLAY", f"tubelet {tubelet_index} · 7 fixed slot colors"],
            color=(255, 196, 107),
        )
        width = max(source.shape[1], overlay.shape[1])
        comparison_frames.append(
            add_legend(np.concatenate([fit_width(source, width), fit_width(overlay, width)], axis=1))
        )
        overlay_frames.append(add_legend(overlay))

    write_h264(output_dir / "comparison.mp4", comparison_frames, output_fps)
    write_h264(output_dir / "slot_overlay.mp4", overlay_frames, output_fps)
    Image.fromarray(comparison_frames[0]).save(output_dir / "poster.jpg", quality=92)
    report = {
        "input_video": str(input_video),
        "config_file": str(config_file),
        "checkpoint": str(checkpoint),
        "checkpoint_load": checkpoint_load_summary(load_report),
        "temporal_mode": cfg.temporal_mode,
        "tubelet_size": 2,
        "tubelet_label_policy": cfg.tubelet_label_policy,
        "decoded_source_shape": list(frames_raw.shape),
        "processed_frames": original_frame_count,
        "padded_model_frames": len(frames_model),
        "resize_mode": args.resize_mode,
        "frame_stride": args.frame_stride,
        "source_fps": float(video_metadata.get("fps", 30.0)),
        "output_fps": output_fps,
        "alpha": args.alpha,
        "seed": args.seed,
        "shapes": {
            "model_input": list(video.shape),
            "decoder_attention": list(attention_shape),
            "tubelet_slot_labels": list(slot_labels.shape),
            "frame_slot_labels": list(labels_per_frame.shape),
        },
        "slot_colors_rgb": SLOT_COLORS.tolist(),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "index.html").write_text(build_page(report))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
