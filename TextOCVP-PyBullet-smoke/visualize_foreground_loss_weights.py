#!/usr/bin/env python3
"""Visualize proposed foreground-weighted SAVi reconstruction loss maps."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import decord
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


PB_VIZ_PATH = Path(
    "/home/gaoya/Code_Video/phys_state_video/scripts/export_raw_multimodal_viz.py"
)
STATIC_KUBRIC_TYPES = {"dome", "cube_platform", "wall", "ground", "pool_table"}
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=216)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--foreground-weight", type=float, default=1.0)
    parser.add_argument("--dilation-pixels", type=int, default=3)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-pybullet-cases", type=int, default=5)
    parser.add_argument("--max-kubric-cases", type=int, default=9)
    return parser.parse_args()


def load_pb_viz_module():
    spec = importlib.util.spec_from_file_location("physv_pb_viz", PB_VIZ_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import PyBullet visualization helpers from {PB_VIZ_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_group_representatives(records, limit):
    groups = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    selected = [sorted(groups[group], key=lambda item: item["sample_id"])[0] for group in sorted(groups)]
    return selected[:limit]


def deterministic_frame_ids(video_length, num_frames):
    range_start = 0
    range_end = min(49, video_length - 1)
    max_start = range_end - num_frames + 1
    if max_start < range_start:
        raise ValueError(f"Video with {video_length} frames cannot provide {num_frames} frames")
    start = (range_start + max_start) // 2
    return start + np.arange(num_frames, dtype=np.int64)


def decode_video(video_path, frame_ids, size_hw, interpolation="bilinear"):
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    frames = torch.from_numpy(reader.get_batch(frame_ids).asnumpy()).float()
    frames = frames.permute(0, 3, 1, 2)
    mode = interpolation
    kwargs = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs = {"align_corners": False, "antialias": True}
    frames = F.interpolate(frames, size=size_hw, mode=mode, **kwargs)
    return frames.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy(), len(reader)


def pybullet_case(record, args, pb_viz, cache_dir):
    sample_dir = Path(record["video_path"]).parent
    full_case = pb_viz.load_case(sample_dir, args.width, args.height)
    frame_ids = deterministic_frame_ids(full_case.frames.shape[0], args.num_frames)
    time_keys = {
        "positions",
        "quats",
        "linear_velocities",
        "angular_velocities",
        "frame_times",
    }
    sliced_states = {
        key: value[frame_ids] if key in time_keys else value
        for key, value in full_case.states.items()
    }
    case = pb_viz.CaseData(
        sample_dir=full_case.sample_dir,
        meta=full_case.meta,
        states=sliced_states,
        frames=full_case.frames[frame_ids],
        frame_width=args.width,
        frame_height=args.height,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    groundtruth = pb_viz.render_pybullet_groundtruth(case, args.width, args.height, cache_dir)
    instance_map = np.asarray(groundtruth["segmentation_object_idx"], dtype=np.int32)
    frames = np.clip(case.frames * 255.0, 0, 255).astype(np.uint8)
    return frames, instance_map, frame_ids, {
        "mask_source": "PyBullet replay segmentation buffer",
        "foreground_objects": [item["name"] for item in case.meta["objects"]],
        "background_objects": ["plane", "backdrop"],
    }


def kubric_instance_map(segmentation_frames, metadata):
    object_data = metadata["object_data"]
    object_types = object_data["type"]
    segmentation_ids = [int(value) for value in object_data["segmentation_id"]]
    color_map = metadata["segmentation_color_map"]
    available_ids = [value for value in segmentation_ids if str(value) in color_map]
    ignored_ids = [value for value in segmentation_ids if str(value) not in color_map]
    colors = np.asarray([color_map[str(value)] for value in available_ids], dtype=np.float32)
    static_ids = {
        segmentation_id
        for segmentation_id, object_type in zip(segmentation_ids, object_types)
        if object_type in STATIC_KUBRIC_TYPES
    }
    foreground_ids = [value for value in available_ids if value not in static_ids]
    foreground_lookup = {value: index for index, value in enumerate(foreground_ids)}
    output = np.full(segmentation_frames.shape[:3], -1, dtype=np.int32)
    for time_index, frame in enumerate(segmentation_frames):
        pixels = frame.astype(np.float32).reshape(-1, 3)
        nearest = np.empty((pixels.shape[0],), dtype=np.int32)
        chunk_size = 65536
        for start in range(0, pixels.shape[0], chunk_size):
            chunk = pixels[start : start + chunk_size]
            distance = ((chunk[:, None, :] - colors[None, :, :]) ** 2).sum(axis=-1)
            nearest[start : start + chunk_size] = distance.argmin(axis=1)
        decoded_ids = np.asarray(available_ids, dtype=np.int32)[nearest].reshape(frame.shape[:2])
        for segmentation_id, foreground_index in foreground_lookup.items():
            output[time_index][decoded_ids == segmentation_id] = foreground_index
    return output, foreground_ids, sorted(static_ids), ignored_ids


def kubric_case(record, args):
    video_path = Path(record["video_path"])
    segmentation_path = video_path.parent / "segmentation.mp4"
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"Missing Kubric segmentation video: {segmentation_path}")
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    frame_ids = deterministic_frame_ids(len(reader), args.num_frames)
    frames, _ = decode_video(video_path, frame_ids, (args.height, args.width), "bilinear")
    segmentation, _ = decode_video(
        segmentation_path, frame_ids, (args.height, args.width), "nearest"
    )
    metadata = json.loads(Path(record["metadata_path"]).read_text(encoding="utf-8"))
    instance_map, foreground_ids, static_ids, ignored_ids = kubric_instance_map(
        segmentation, metadata
    )
    object_data = metadata["object_data"]
    id_to_type = {
        int(segmentation_id): object_type
        for segmentation_id, object_type in zip(
            object_data["segmentation_id"], object_data["type"]
        )
    }
    return frames, instance_map, frame_ids, {
        "mask_source": "Kubric segmentation.mp4 nearest-color instance IDs",
        "foreground_objects": [id_to_type[value] for value in foreground_ids],
        "background_objects": [id_to_type[value] for value in static_ids],
        "ignored_segmentation_ids_without_color": ignored_ids,
    }


def dilate_foreground(instance_map, dilation_pixels):
    foreground = instance_map >= 0
    if dilation_pixels <= 0:
        return foreground
    kernel_size = 2 * dilation_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return np.stack(
        [cv2.dilate(frame.astype(np.uint8), kernel, iterations=1) > 0 for frame in foreground]
    )


def colorize_instances(instance_map):
    panels = np.full((*instance_map.shape, 3), 20, dtype=np.uint8)
    for index in np.unique(instance_map):
        if index >= 0:
            panels[instance_map == index] = PALETTE[index % len(PALETTE)]
    return panels


def colorize_weights(weight_maps, background_weight, foreground_weight):
    denominator = max(foreground_weight - background_weight, 1e-8)
    normalized = np.clip((weight_maps - background_weight) / denominator, 0, 1)
    outputs = []
    for frame in normalized:
        bgr = cv2.applyColorMap((frame * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)
        outputs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(outputs)


def add_header(panels, title, panel_names):
    body = np.concatenate(panels, axis=1)
    header = np.full((76, body.shape[1], 3), 247, dtype=np.uint8)
    panel_width = panels[0].shape[1]
    for index, name in enumerate(panel_names):
        cv2.putText(
            header,
            name,
            (index * panel_width + 10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (24, 24, 24),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        header,
        title,
        (10, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, body], axis=0)


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


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def render_record(record, args, pb_viz):
    source = record["source"]
    sample_id = record["sample_id"]
    relative_dir = Path(source) / safe_name(record["group"]) / safe_name(sample_id)
    if source == "pybullet":
        frames, instance_map, frame_ids, details = pybullet_case(
            record, args, pb_viz, args.output_dir / relative_dir / "mask_cache"
        )
    elif source == "kubric":
        frames, instance_map, frame_ids, details = kubric_case(record, args)
    else:
        raise ValueError(f"Unsupported source={source}")

    foreground = dilate_foreground(instance_map, args.dilation_pixels)
    weight_maps = np.where(
        foreground, args.foreground_weight, args.background_weight
    ).astype(np.float32)
    instance_panels = colorize_instances(instance_map)
    heatmaps = colorize_weights(
        weight_maps, args.background_weight, args.foreground_weight
    )
    overlays = np.clip(frames.astype(np.float32) * 0.55 + heatmaps * 0.45, 0, 255).astype(
        np.uint8
    )
    video_frames = []
    foreground_fractions = []
    for index in range(len(frames)):
        foreground_fraction = float(foreground[index].mean())
        foreground_fractions.append(foreground_fraction)
        title = (
            f"source={source} | frame={int(frame_ids[index])} | "
            f"foreground={foreground_fraction * 100:.2f}% | "
            f"loss weights: fg={args.foreground_weight:g}, bg={args.background_weight:g}"
        )
        video_frames.append(
            add_header(
                [frames[index], instance_panels[index], heatmaps[index], overlays[index]],
                title,
                [
                    "Exact RGB input",
                    "Foreground instances",
                    f"Loss heatmap: blue={args.background_weight:g}, red={args.foreground_weight:g}",
                    "Heatmap overlay",
                ],
            )
        )
    relative_video = relative_dir / "foreground_loss_weight_overlay.mp4"
    write_h264(args.output_dir / relative_video, video_frames, args.fps)
    return {
        **record,
        **details,
        "frame_ids": frame_ids.tolist(),
        "foreground_fraction_per_frame": foreground_fractions,
        "mean_foreground_fraction": float(np.mean(foreground_fractions)),
        "foreground_weight": args.foreground_weight,
        "background_weight": args.background_weight,
        "dilation_pixels": args.dilation_pixels,
        "output_video": relative_video.as_posix(),
    }


def build_index(output_dir, reports, args):
    cards = []
    for report in reports:
        cards.append(
            f"""
            <article>
              <h2>{html.escape(report['source'])} / {html.escape(report['group'])} / {html.escape(report['sample_id'])}</h2>
              <p><code>{html.escape(report['video_path'])}</code></p>
              <p>mask source: {html.escape(report['mask_source'])}; mean foreground after dilation: {report['mean_foreground_fraction'] * 100:.2f}%</p>
              <p>foreground objects: <code>{html.escape(str(report['foreground_objects']))}</code>; background objects: <code>{html.escape(str(report['background_objects']))}</code></p>
              <video controls loop muted preload="metadata" src="{html.escape(report['output_video'])}"></video>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAVi foreground loss weights</title><style>
:root {{ --paper:#f1f3f2; --ink:#18211d; --line:#c7ceca; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:"IBM Plex Sans","Noto Sans",sans-serif; }}
header,article {{ padding:20px 28px; border-bottom:1px solid var(--line); }}
article {{ background:#fff; margin:18px 24px; border:1px solid var(--line); }}
h1 {{ margin:0 0 10px; }} h2 {{ margin:0 0 8px; font-size:19px; }} p {{ margin:6px 0; overflow-wrap:anywhere; }}
video {{ display:block; width:100%; margin-top:14px; background:#111; }} code {{ font-size:12px; }}
@media(max-width:800px) {{ header,article {{ padding:14px; margin:0; }} }}
</style></head><body><header><h1>Foreground-weighted reconstruction loss maps</h1>
<p>Weights: foreground={args.foreground_weight:g}, background={args.background_weight:g}; foreground dilation={args.dilation_pixels}px at {args.height}x{args.width}.</p>
<p>Kubric uses native instance segmentation. PyBullet uses state/camera replay and the PyBullet segmentation buffer. Masks are targets for loss only and are not model inputs.</p>
</header>{''.join(cards)}</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    if not 0 <= args.background_weight <= args.foreground_weight:
        raise ValueError("Expected 0 <= background weight <= foreground weight")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pybullet_records = read_records(args.index_root / "pybullet" / "handoff_monitor.jsonl")
    kubric_records = read_records(args.index_root / "kubric" / "handoff_monitor.jsonl")
    records = select_group_representatives(
        pybullet_records, args.max_pybullet_cases
    ) + select_group_representatives(kubric_records, args.max_kubric_cases)
    pb_viz = load_pb_viz_module()
    reports = []
    failures = []
    for index, record in enumerate(records, start=1):
        try:
            report = render_record(record, args, pb_viz)
            reports.append(report)
            print(f"processed={index}/{len(records)} {record['source']}:{record['sample_id']}", flush=True)
        except Exception as error:
            failures.append({"record": record, "error": repr(error)})
            print(f"FAILED={index}/{len(records)} {record['source']}:{record['sample_id']} {error!r}", flush=True)
    summary = {
        "resolution_hw": [args.height, args.width],
        "num_frames": args.num_frames,
        "foreground_weight": args.foreground_weight,
        "background_weight": args.background_weight,
        "dilation_pixels": args.dilation_pixels,
        "successful_cases": len(reports),
        "requested_cases": len(records),
        "failures": failures,
        "cases": reports,
    }
    (args.output_dir / "loss_weight_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    build_index(args.output_dir, reports, args)
    print(json.dumps({key: summary[key] for key in (
        "successful_cases", "requested_cases", "foreground_weight",
        "background_weight", "dilation_pixels", "failures"
    )}, indent=2))


if __name__ == "__main__":
    main()
