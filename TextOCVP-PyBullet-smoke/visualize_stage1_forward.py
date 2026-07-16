#!/usr/bin/env python3
"""Visualize exact raw clips and one official SAVi Stage 1 forward pass."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
TEXTOCVP_SRC = TEXTOCVP_ROOT / "src"
sys.path.insert(0, str(TEXTOCVP_SRC))
os.chdir(TEXTOCVP_ROOT)

from data.PyBullet import PyBullet  # noqa: E402
from lib.setup_model import load_checkpoint, setup_model  # noqa: E402


PALETTE = np.asarray(
    [
        [230, 57, 70],
        [29, 154, 108],
        [43, 116, 189],
        [244, 162, 54],
        [138, 79, 191],
        [0, 168, 181],
        [241, 91, 181],
        [126, 130, 122],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tensor_shape(value):
    if torch.is_tensor(value):
        return list(value.shape)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [tensor_shape(item) for item in value]
    if isinstance(value, dict):
        return {key: tensor_shape(item) for key, item in value.items()}
    return type(value).__name__


def register_shape_hooks(model):
    records = {}
    handles = []
    modules = {
        "01_conv_encoder": model.encoder,
        "02_encoder_position_embedding": model.encoder_pos_embedding,
        "03_encoder_pointwise_mlp": model.encoder_mlp,
        "04_slot_attention": model.slot_attention,
        "05_slot_transition": model.transition_module,
        "06_decoder_position_embedding": model.decoder_pos_embedding,
        "07_conv_decoder": model.decoder,
    }

    def make_hook(name):
        def hook(_module, inputs, kwargs, output):
            record = records.setdefault(
                name,
                {
                    "input": {
                        "args": tensor_shape(inputs),
                        "kwargs": tensor_shape(kwargs),
                    },
                    "output": tensor_shape(output),
                    "calls": 0,
                },
            )
            record["calls"] += 1

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name), with_kwargs=True))
    return records, handles


def to_uint8(video):
    video = video.detach().cpu().clamp(0, 1)
    if video.ndim == 5:
        video = video.permute(0, 1, 3, 4, 2)
    elif video.ndim == 4:
        video = video.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Expected [B,T,C,H,W] or [T,C,H,W], got {tuple(video.shape)}")
    return video.mul(255).round().byte().numpy()


def write_h264(path, frames, fps=30):
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def slot_overlay(input_frame, masks):
    labels = masks[:, 0].argmax(axis=0)
    colors = PALETTE[labels % len(PALETTE)]
    return np.clip(input_frame.astype(np.float32) * 0.55 + colors * 0.45, 0, 255).astype(np.uint8)


def add_external_header(panels, title, panel_names):
    scale = 4
    resized = [
        cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for panel in panels
    ]
    body = np.concatenate(resized, axis=1)
    header = np.full((64, body.shape[1], 3), 247, dtype=np.uint8)
    panel_width = resized[0].shape[1]
    for index, name in enumerate(panel_names):
        cv2.putText(
            header,
            name,
            (index * panel_width + 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        header,
        title,
        (12, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, body], axis=0)


def make_forward_video(input_video, reconstruction, masks, metadata):
    frames = []
    for time_index in range(input_video.shape[0]):
        overlay = slot_overlay(input_video[time_index], masks[time_index])
        title = (
            f"raw frame {metadata['frame_ids'][time_index]} | "
            f"clip start {metadata['start_frame']} | slot argmax overlay"
        )
        frames.append(
            add_external_header(
                [input_video[time_index], reconstruction[time_index], overlay],
                title,
                ["Exact SAVi input", "SAVi reconstruction", "8-slot masks"],
            )
        )
    return frames


def build_index(output_dir, samples, summary):
    shape_rows = []
    for name, record in summary["module_shapes"].items():
        shape_rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            f"<td><code>{html.escape(json.dumps(record['input']))}</code></td>"
            f"<td><code>{html.escape(json.dumps(record['output']))}</code></td>"
            f"<td>{record['calls']}</td>"
            "</tr>"
        )
    sample_sections = []
    for sample in samples:
        path = html.escape(sample["metadata"]["video_path"])
        frame_ids = html.escape(str(sample["metadata"]["frame_ids"]))
        sample_sections.append(
            f"""
            <section>
              <h2>{html.escape(sample['family'])} / {html.escape(sample['sample_id'])}</h2>
              <p><code>{path}</code></p>
              <p>start={sample['metadata']['start_frame']}; raw frame IDs: <code>{frame_ids}</code></p>
              <div class="videos">
                <figure><video controls loop muted src="{sample['clip_file']}"></video><figcaption>Exact training tensor, 112x64</figcaption></figure>
                <figure class="wide"><video controls loop muted src="{sample['forward_file']}"></video><figcaption>Input / reconstruction / slot masks; title bar is outside image</figcaption></figure>
              </div>
            </section>
            """
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TextOCVP PyBullet Stage 1</title>
<style>
:root {{ color-scheme: light; --ink:#17202a; --line:#cfd5db; --paper:#f4f6f7; --accent:#b42318; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:"IBM Plex Sans","Noto Sans",sans-serif; }}
header, section {{ padding:20px 28px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
h2 {{ margin:0 0 8px; font-size:19px; letter-spacing:0; }}
p {{ margin:6px 0; overflow-wrap:anywhere; }}
.facts {{ display:flex; flex-wrap:wrap; gap:20px; font-weight:600; }}
.videos {{ display:grid; grid-template-columns:minmax(180px,280px) minmax(600px,1fr); gap:20px; margin-top:14px; align-items:start; }}
figure {{ margin:0; }} video {{ width:100%; background:#111; display:block; }}
figcaption {{ margin-top:6px; font-size:13px; }}
table {{ width:100%; border-collapse:collapse; background:white; }}
th,td {{ border:1px solid var(--line); padding:8px; text-align:left; font-size:13px; }}
code {{ font-family:"IBM Plex Mono","Noto Sans Mono",monospace; font-size:12px; }}
@media (max-width:900px) {{ header,section {{ padding:16px; }} .videos {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>TextOCVP SAVi Stage 1: exact raw clips and forward pass</h1>
  <div class="facts"><span>Sampling range: raw frames 0-49</span><span>Clip: 10 contiguous frames</span><span>Input: [B,10,3,64,112]</span><span>Slots: 8x128</span></div>
  <p>Checkpoint: <code>{html.escape(summary['checkpoint'])}</code></p>
</header>
<section>
  <h2>Observed module shapes</h2>
  <table><thead><tr><th>Module</th><th>Input</th><th>Output</th><th>Calls</th></tr></thead><tbody>{''.join(shape_rows)}</tbody></table>
</section>
{''.join(sample_sections)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = PyBullet(
        root=args.dataset_root,
        split="train",
        num_frames=10,
        img_size=(64, 112),
        random_start=True,
        frame_stride=1,
        sampling_frame_range=(0, 49),
        max_samples=args.num_samples,
    )
    loaded = [dataset[index] for index in range(len(dataset))]
    videos = torch.stack([sample[0] for sample in loaded], dim=0)
    metadata = [sample[1] for sample in loaded]

    params = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    model = setup_model(params["model"])
    model = load_checkpoint(
        checkpoint_path=str(args.checkpoint),
        model=model,
        only_model=True,
        map_cpu=True,
    )
    device = torch.device(f"cuda:{args.gpu}")
    model = model.eval().to(device)
    shape_records, handles = register_shape_hooks(model)
    with torch.inference_mode():
        output = model(x=videos.to(device), num_imgs=videos.shape[1], decode=True)
    for handle in handles:
        handle.remove()

    input_uint8 = to_uint8(videos)
    recon_uint8 = to_uint8(output["recons_imgs"])
    masks = output["masks"].detach().cpu().numpy()
    sample_reports = []
    for index, item_metadata in enumerate(metadata):
        sample_id = Path(item_metadata["video_path"]).parent.name
        family = Path(item_metadata["video_path"]).parents[1].name
        clip_name = f"{index:02d}_{family}_{sample_id}_training_clip.mp4"
        forward_name = f"{index:02d}_{family}_{sample_id}_stage1_forward.mp4"
        write_h264(args.output_dir / clip_name, input_uint8[index], fps=30)
        forward_frames = make_forward_video(
            input_uint8[index], recon_uint8[index], masks[index], item_metadata
        )
        write_h264(args.output_dir / forward_name, forward_frames, fps=10)
        sample_reports.append(
            {
                "sample_id": sample_id,
                "family": family,
                "metadata": item_metadata,
                "clip_file": clip_name,
                "forward_file": forward_name,
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "sampling_frame_range": [0, 49],
        "train_start_range": [0, 40],
        "input_shape": list(videos.shape),
        "model_outputs": tensor_shape(output),
        "module_shapes": shape_records,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "samples": sample_reports,
    }
    (args.output_dir / "forward_shapes.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    build_index(args.output_dir, sample_reports, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
