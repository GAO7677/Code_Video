#!/usr/bin/env python3
"""Visualize a pixel-space SAVi checkpoint on indexed and external videos."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
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

from data.Stage1Indexed import Stage1Indexed  # noqa: E402
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
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--dataset-mode", choices=("pybullet", "kubric", "mixed"), default="pybullet")
    parser.add_argument("--split", choices=("valid", "val"), default="valid")
    parser.add_argument("--external-json", type=Path, action="append", default=[])
    parser.add_argument("--external-json-list", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--height", type=int, default=216)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--fps", type=float, default=10.0)
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
                    "input": {"args": tensor_shape(inputs), "kwargs": tensor_shape(kwargs)},
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


def write_h264(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
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
    body = np.concatenate(panels, axis=1)
    header = np.full((72, body.shape[1], 3), 247, dtype=np.uint8)
    panel_width = panels[0].shape[1]
    for index, name in enumerate(panel_names):
        cv2.putText(
            header,
            name,
            (index * panel_width + 12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        header,
        title,
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, body], axis=0)


def make_forward_video(input_video, reconstruction, masks, metadata, num_slots):
    frames = []
    for time_index in range(input_video.shape[0]):
        overlay = slot_overlay(input_video[time_index], masks[time_index])
        title = (
            f"source frame {metadata['frame_ids'][time_index]} | "
            f"clip start {metadata['start_frame']} | {num_slots}-slot argmax overlay"
        )
        frames.append(
            add_external_header(
                [input_video[time_index], reconstruction[time_index], overlay],
                title,
                ["Exact model input", "SAVi reconstruction", "Slot masks"],
            )
        )
    return frames


class ExternalStage1Dataset(Stage1Indexed):
    """Use Stage1Indexed's exact sampling and resize path for external videos."""

    def __init__(self, json_paths, num_frames, img_size):
        self.dataset_mode = "external"
        self.split = "valid"
        self.num_frames = int(num_frames)
        self.img_size = tuple(int(value) for value in img_size)
        self.frame_stride = 1
        self.random_start = False
        self.preprocess_mode = "resize"
        self.vjepa_short_side = 438
        self.vjepa_crop_size = 384
        self.records = []
        for json_path in json_paths:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            video_path = Path(payload["source_video"])
            if not video_path.is_file():
                raise FileNotFoundError(f"External source_video does not exist: {video_path}")
            self.records.append(
                {
                    "source": "physiq_external",
                    "video_path": str(video_path),
                    "metadata_path": str(json_path),
                    "group": "physiq_external",
                    "sample_id": json_path.stem,
                    "sampling_frame_range": [0, 49],
                }
            )


def load_external_json_paths(args):
    paths = list(args.external_json)
    if args.external_json_list:
        for line in args.external_json_list.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                paths.append(Path(value))
    resolved = []
    seen = set()
    for path in paths:
        path = path.expanduser().resolve()
        if path not in seen:
            if not path.is_file():
                raise FileNotFoundError(f"External input JSON does not exist: {path}")
            resolved.append(path)
            seen.add(path)
    return resolved


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


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
        metadata = sample["metadata"]
        sample_sections.append(
            f"""
            <article data-source="{html.escape(metadata['source'])}">
              <h2>{html.escape(metadata['source'])} / {html.escape(metadata['sample_id'])}</h2>
              <p><code>{html.escape(metadata['video_path'])}</code></p>
              <p>source resolution HxW={metadata['source_resolution_hw']}; start={metadata['start_frame']};
                 frames=<code>{html.escape(str(metadata['frame_ids']))}</code>; MSE={sample['mse']:.8f}</p>
              <div class="videos">
                <figure><video controls loop muted preload="metadata" src="{html.escape(sample['clip_file'])}"></video><figcaption>Exact resized Stage 1 input</figcaption></figure>
                <figure class="wide"><video controls loop muted preload="metadata" src="{html.escape(sample['forward_file'])}"></video><figcaption>Input / reconstruction / slot argmax; title outside image</figcaption></figure>
              </div>
            </article>
            """
        )
    input_shape = summary["observed_batch_input_shape"]
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PyBullet SAVi Stage 1 validation</title>
<style>
:root {{ color-scheme:light; --ink:#17202a; --line:#cfd5db; --paper:#f4f6f7; --card:#fff; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:"IBM Plex Sans","Noto Sans",sans-serif; }}
header,section,article {{ padding:20px 28px; border-bottom:1px solid var(--line); }}
article {{ background:var(--card); margin:18px 24px; border:1px solid var(--line); }}
h1 {{ margin:0 0 8px; font-size:28px; }} h2 {{ margin:0 0 8px; font-size:19px; }}
p {{ margin:6px 0; overflow-wrap:anywhere; }}
.facts {{ display:flex; flex-wrap:wrap; gap:20px; font-weight:600; }}
.videos {{ display:grid; grid-template-columns:minmax(240px,384px) minmax(600px,1fr); gap:20px; margin-top:14px; align-items:start; }}
figure {{ margin:0; }} video {{ width:100%; background:#111; display:block; }} figcaption {{ margin-top:6px; font-size:13px; }}
table {{ width:100%; border-collapse:collapse; background:white; }} th,td {{ border:1px solid var(--line); padding:8px; text-align:left; font-size:13px; }}
code {{ font-family:"IBM Plex Mono","Noto Sans Mono",monospace; font-size:12px; }}
@media (max-width:900px) {{ header,section,article {{ padding:16px; margin:0; }} .videos {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Pixel-space SAVi Stage 1 validation and external tests</h1>
  <div class="facts"><span>Samples: {len(samples)}</span><span>Input batch: {input_shape}</span><span>Slots: {summary['num_slots']}x{summary['slot_dim']}</span><span>Mean MSE: {summary['mean_mse']:.8f}</span></div>
  <p>Checkpoint: <code>{html.escape(summary['checkpoint'])}</code></p>
  <p>Validation uses deterministic source frames 20..29 when at least 50 frames are available. External videos use the identical sampling and resize path.</p>
</header>
<section><h2>Observed first-batch module shapes</h2><table><thead><tr><th>Module</th><th>Input</th><th>Output</th><th>Calls</th></tr></thead><tbody>{''.join(shape_rows)}</tbody></table></section>
{''.join(sample_sections)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def process_batch(model, device, batch, output_dir, args, num_slots):
    videos = torch.stack([item[0] for item in batch], dim=0)
    metadata = [item[1] for item in batch]
    with torch.inference_mode():
        output = model(x=videos.to(device), num_imgs=videos.shape[1], decode=True)

    recons = output["recons_imgs"].detach().cpu()
    masks = output["masks"].detach().cpu().numpy()
    input_uint8 = to_uint8(videos)
    recon_uint8 = to_uint8(recons)
    reports = []
    for index, item_metadata in enumerate(metadata):
        source = safe_name(item_metadata["source"])
        sample_id = safe_name(item_metadata["sample_id"])
        relative_root = Path(source) / sample_id
        clip_path = relative_root / "model_input.mp4"
        forward_path = relative_root / "stage1_forward_overlay.mp4"
        write_h264(output_dir / clip_path, input_uint8[index], args.fps)
        write_h264(
            output_dir / forward_path,
            make_forward_video(
                input_uint8[index], recon_uint8[index], masks[index], item_metadata, num_slots
            ),
            args.fps,
        )
        mse = torch.mean((recons[index] - videos[index].cpu()) ** 2).item()
        reports.append(
            {
                "metadata": item_metadata,
                "clip_file": clip_path.as_posix(),
                "forward_file": forward_path.as_posix(),
                "mse": mse,
            }
        )
    return reports, tensor_shape(output), list(videos.shape)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    model_params = params["model"]["model_params"]
    expected_resolution = tuple(model_params["encoder"]["encoder_params"]["resolution"])
    requested_resolution = (args.height, args.width)
    if requested_resolution != expected_resolution:
        raise ValueError(
            f"Requested HxW={requested_resolution}, but checkpoint model expects {expected_resolution}"
        )

    validation = Stage1Indexed(
        index_root=args.index_root,
        dataset_mode=args.dataset_mode,
        split=args.split,
        num_frames=args.num_frames,
        img_size=requested_resolution,
        frame_stride=1,
        random_start=False,
        preprocess_mode="resize",
        max_samples=args.max_validation_samples,
    )
    external_paths = load_external_json_paths(args)
    external = ExternalStage1Dataset(external_paths, args.num_frames, requested_resolution)
    items = [
        ("validation", validation, index) for index in range(len(validation))
    ] + [("external", external, index) for index in range(len(external))]

    model = setup_model(params["model"])
    model = load_checkpoint(
        checkpoint_path=str(args.checkpoint), model=model, only_model=True, map_cpu=True
    )
    device = torch.device(f"cuda:{args.gpu}")
    model = model.eval().to(device)
    shape_records, shape_handles = register_shape_hooks(model)

    reports = []
    failures = []
    first_output_shapes = None
    first_input_shape = None
    batch = []
    for position, (kind, dataset, index) in enumerate(items, start=1):
        try:
            batch.append(dataset[index])
        except Exception as error:  # Keep the remaining validation cases inspectable.
            failures.append({"kind": kind, "index": index, "error": repr(error)})
        if len(batch) >= args.batch_size or position == len(items):
            if not batch:
                continue
            batch_reports, output_shapes, input_shape = process_batch(
                model, device, batch, args.output_dir, args, model_params["num_slots"]
            )
            reports.extend(batch_reports)
            if first_output_shapes is None:
                first_output_shapes = output_shapes
                first_input_shape = input_shape
                for handle in shape_handles:
                    handle.remove()
                shape_handles.clear()
            print(f"processed={len(reports)}/{len(items)} failures={len(failures)}", flush=True)
            batch = []

    for handle in shape_handles:
        handle.remove()
    mean_mse = float(np.mean([item["mse"] for item in reports])) if reports else float("nan")
    source_counts = {}
    source_mse = {}
    for source in sorted({item["metadata"]["source"] for item in reports}):
        values = [item["mse"] for item in reports if item["metadata"]["source"] == source]
        source_counts[source] = len(values)
        source_mse[source] = float(np.mean(values))
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "experiment_config": str(args.experiment_config.resolve()),
        "index_root": str(args.index_root.resolve()),
        "external_jsons": [str(path) for path in external_paths],
        "device": str(device),
        "seed": args.seed,
        "num_slots": model_params["num_slots"],
        "slot_dim": model_params["slot_dim"],
        "resolution_hw": list(requested_resolution),
        "num_frames": args.num_frames,
        "sampling_frame_range": [0, 49],
        "observed_batch_input_shape": first_input_shape,
        "first_batch_model_outputs": first_output_shapes,
        "module_shapes": shape_records,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "successful_samples": len(reports),
        "requested_samples": len(items),
        "source_counts": source_counts,
        "mean_mse": mean_mse,
        "source_mean_mse": source_mse,
        "failures": failures,
        "samples": reports,
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    build_index(args.output_dir, reports, summary)
    print(json.dumps({key: summary[key] for key in (
        "checkpoint", "device", "successful_samples", "requested_samples",
        "source_counts", "mean_mse", "source_mean_mse", "failures"
    )}, indent=2))


if __name__ == "__main__":
    main()
