#!/usr/bin/env python3
"""Visualize the exact latent mask targets used by V-JEPA Stage 1 mask loss."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


TEXTOCVP_SRC = Path("/home/gaoya/Code_Video/TextOCVP-master/src")
sys.path.insert(0, str(TEXTOCVP_SRC))
from data.Stage1Indexed import Stage1Indexed  # noqa: E402


INSTANCE_PALETTE = np.asarray(
    [
        [230, 57, 70],
        [29, 154, 108],
        [43, 116, 189],
        [138, 79, 191],
        [0, 168, 181],
        [241, 91, 181],
    ],
    dtype=np.float32,
)
STATIC_COLOR = np.asarray([244, 162, 54], dtype=np.float32)
BACKGROUND_COLOR = np.asarray([20, 25, 29], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--union-weight", type=float, default=0.20)
    parser.add_argument("--instance-weight", type=float, default=0.10)
    parser.add_argument("--static-weight", type=float, default=0.02)
    parser.add_argument("--background-weight", type=float, default=0.01)
    parser.add_argument("--unused-weight", type=float, default=0.01)
    parser.add_argument("--focal-bce-weight", type=float, default=0.25)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-ramp", type=float, default=1.0)
    parser.add_argument("--max-pybullet-cases", type=int, default=5)
    parser.add_argument("--max-kubric-cases", type=int, default=9)
    parser.add_argument("--fps", type=float, default=5.0)
    return parser.parse_args()


def select_group_indices(dataset, limit):
    groups = defaultdict(list)
    for index, record in enumerate(dataset.records):
        groups[record["group"]].append((record["sample_id"], index))
    return [sorted(groups[group])[0][1] for group in sorted(groups)[:limit]]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def resize_latent(mask, size=(384, 384)):
    mask = torch.as_tensor(mask)
    return F.interpolate(
        mask[None, None].float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0, 1).numpy()


def make_target_panel(instance_masks, instance_valid, dynamic_union, static, instance_mode):
    height, width = dynamic_union.shape
    background = np.clip(1.0 - dynamic_union - static, 0.0, 1.0)
    panel = background[..., None] * BACKGROUND_COLOR
    if instance_mode:
        for index in np.flatnonzero(instance_valid):
            occupancy = instance_masks[index][..., None]
            panel += occupancy * INSTANCE_PALETTE[index % len(INSTANCE_PALETTE)]
    else:
        panel += dynamic_union[..., None] * INSTANCE_PALETTE[0]
    panel += static[..., None] * STATIC_COLOR
    return np.clip(panel, 0, 255).astype(np.uint8)


def colorize_coefficient_map(values, maximum):
    if maximum <= 0 or float(values.max()) <= 0:
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    normalized = np.clip(values / maximum, 0, 1)
    bgr = cv2.applyColorMap((normalized * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def add_header(panels, title, panel_names):
    body = np.concatenate(panels, axis=1)
    header = np.full((82, body.shape[1], 3), 247, dtype=np.uint8)
    panel_width = panels[0].shape[1]
    for index, name in enumerate(panel_names):
        cv2.putText(
            header,
            name,
            (index * panel_width + 10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (24, 24, 24),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        header,
        title,
        (10, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (62, 62, 62),
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


def make_skipped_panel(shape, lines):
    panel = np.full(shape, 34, dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (28, 168 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76 if index == 0 else 0.58,
            (238, 238, 238),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    return panel


def render_sample(dataset, index, args):
    video, metadata = dataset[index]
    targets = metadata.pop("_mask_targets")
    frames = (
        video.permute(0, 2, 3, 1).clamp(0, 1).mul(255).round().byte().numpy()
    )
    supervised = bool(targets["mask_supervision_valid"].item())
    instance_mode = supervised and bool(targets["instance_supervision_valid"].item())
    mode = "instance" if instance_mode else ("union-only" if supervised else "skipped")
    dynamic_coefficient = args.union_weight + (
        args.instance_weight if instance_mode else 0.0
    )
    global_scale = args.mask_loss_weight * args.mask_ramp
    dynamic_coefficient *= global_scale
    static_coefficient = args.static_weight * global_scale
    background_coefficient = args.background_weight * global_scale
    maximum = max(
        dynamic_coefficient,
        static_coefficient,
        background_coefficient,
        1e-8,
    )

    dynamic_instances = targets["dynamic_instance_masks"].numpy()
    instance_valid = targets["dynamic_instance_valid"].numpy().astype(bool)
    dynamic_union = targets["dynamic_union_mask"][:, 0].numpy()
    static_geometry = targets["static_geometry_mask"][:, 0].numpy()
    output_frames = []
    dynamic_fractions = []
    static_fractions = []
    for frame_index, frame in enumerate(frames):
        latent_index = frame_index // 2
        if supervised:
            dynamic = resize_latent(dynamic_union[latent_index])
            static = resize_latent(static_geometry[latent_index])
            instances = np.stack(
                [
                    resize_latent(dynamic_instances[latent_index, instance_index])
                    for instance_index in range(args.max_instances)
                ]
            )
            background = np.clip(1.0 - dynamic - static, 0.0, 1.0)
            coefficients = (
                dynamic * dynamic_coefficient
                + static * static_coefficient
                + background * background_coefficient
            )
            target_panel = make_target_panel(
                instances, instance_valid, dynamic, static, instance_mode
            )
            heatmap = colorize_coefficient_map(coefficients, maximum)
            overlay = np.clip(frame.astype(np.float32) * 0.55 + heatmap * 0.45, 0, 255).astype(
                np.uint8
            )
            dynamic_fraction = float(dynamic.mean())
            static_fraction = float(static.mean())
        else:
            target_panel = make_skipped_panel(
                frame.shape,
                ["MASK LOSS SKIPPED", "No PyBullet segmentation target"],
            )
            heatmap = make_skipped_panel(
                frame.shape,
                ["COEFFICIENT MAP = 0", "Feature reconstruction loss still active"],
            )
            overlay = frame.copy()
            dynamic_fraction = 0.0
            static_fraction = 0.0
        dynamic_fractions.append(dynamic_fraction)
        static_fractions.append(static_fraction)
        source_frame = metadata["frame_ids"][frame_index]
        title = (
            f"source={metadata['source']} | frame={source_frame} | tubelet={latent_index} "
            f"covers clip frames {2 * latent_index},{2 * latent_index + 1} | mode={mode} | "
            f"dyn={dynamic_fraction * 100:.2f}% static={static_fraction * 100:.2f}%"
        )
        output_frames.append(
            add_header(
                [frame, target_panel, heatmap, overlay],
                title,
                [
                    "Exact V-JEPA RGB input",
                    "24x24 latent target (upsampled)",
                    "Mask component coefficient proxy",
                    "Coefficient heatmap overlay",
                ],
            )
        )

    relative_path = (
        Path(metadata["source"])
        / safe_name(metadata["group"])
        / safe_name(metadata["sample_id"])
        / "vjepa_mask_loss_weight_overlay.mp4"
    )
    write_h264(args.output_dir / relative_path, output_frames, args.fps)
    relative_poster = relative_path.with_name("poster.jpg")
    imageio.imwrite(args.output_dir / relative_poster, output_frames[0], quality=92)
    return {
        **metadata,
        "mode": mode,
        "mask_supervised": supervised,
        "dynamic_coefficient": dynamic_coefficient if supervised else 0.0,
        "static_coefficient": static_coefficient if supervised else 0.0,
        "background_coefficient": background_coefficient if supervised else 0.0,
        "unused_slot_coefficient": args.unused_weight * global_scale if supervised else 0.0,
        "mean_dynamic_fraction": float(np.mean(dynamic_fractions)),
        "mean_static_fraction": float(np.mean(static_fractions)),
        "output_video": relative_path.as_posix(),
        "poster": relative_poster.as_posix(),
    }


def build_index(output_dir, reports, args):
    cards = []
    for report in reports:
        status = "MASK LOSS ACTIVE" if report["mask_supervised"] else "MASK LOSS SKIPPED"
        cards.append(
            f"""
            <article class="{'active' if report['mask_supervised'] else 'skipped'}">
              <h2>{html.escape(report['source'])} / {html.escape(report['group'])} / {html.escape(report['sample_id'])}</h2>
              <p class="status">{status}; mode={html.escape(report['mode'])}</p>
              <p><code>{html.escape(report['video_path'])}</code></p>
              <p>dynamic coefficient={report['dynamic_coefficient']:.3f}; static={report['static_coefficient']:.3f}; background={report['background_coefficient']:.3f}; unused-slot={report['unused_slot_coefficient']:.3f}</p>
              <p>mean latent occupancy: dynamic={report['mean_dynamic_fraction'] * 100:.2f}%; static={report['mean_static_fraction'] * 100:.2f}%</p>
              <video controls loop muted playsinline preload="auto" poster="{html.escape(report['poster'])}" src="{html.escape(report['output_video'])}?v=2"></video>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA mask-loss weight heatmaps</title><style>
:root {{ --paper:#eef1ef; --ink:#17201c; --line:#c5cec8; --active:#176b48; --skip:#9a3412; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:"IBM Plex Sans","Noto Sans",sans-serif; }}
header,article {{ padding:20px 28px; }} header {{ border-bottom:1px solid var(--line); }}
article {{ background:#fff; margin:18px 24px; border:1px solid var(--line); }} article.active {{ border-left:5px solid var(--active); }} article.skipped {{ border-left:5px solid var(--skip); }}
h1 {{ margin:0 0 10px; }} h2 {{ margin:0 0 8px; font-size:19px; }} p {{ margin:6px 0; overflow-wrap:anywhere; }}
.status {{ font-weight:700; }} .active .status {{ color:var(--active); }} .skipped .status {{ color:var(--skip); }}
video {{ display:block; width:100%; margin-top:14px; background:#111; }} code {{ font-size:12px; }}
@media(max-width:800px) {{ header,article {{ padding:14px; margin:0; }} }}
</style></head><body><header><h1>V-JEPA Stage 1 mask-loss targets and coefficients</h1>
<p>Exact preprocessing: 10 RGB frames -> 384x384 center crop; mask occupancy -> tubelet 2 x patch 16 -> 5x24x24 latent target.</p>
<p>Full-ramp component coefficients: union={args.union_weight:g}, instance={args.instance_weight:g}, static={args.static_weight:g}, background={args.background_weight:g}, unused={args.unused_weight:g}, focal-BCE={args.focal_bce_weight:g}; global mask-loss weight={args.mask_loss_weight:g}.</p>
<p>Dynamic coefficient is {args.union_weight + args.instance_weight:g} for instance-supervised cases and {args.union_weight:g} for union-only cases. This map is a component-coefficient proxy: Dice, Focal-BCE, Hungarian matching, and unused-slot penalties are not literal per-pixel multipliers.</p>
<p>PyBullet is intentionally shown as skipped because the current training index has no precomputed segmentation target.</p>
</header>{''.join(cards)}</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_kwargs = {
        "index_root": args.index_root,
        "split": "valid",
        "num_frames": args.num_frames,
        "img_size": (384, 384),
        "frame_stride": 1,
        "random_start": False,
        "preprocess_mode": "vjepa",
        "load_masks": True,
        "max_mask_instances": args.max_instances,
        "mask_temporal_stride": 2,
        "mask_spatial_stride": 16,
    }
    datasets = {
        source: Stage1Indexed(dataset_mode=source, **dataset_kwargs)
        for source in ("pybullet", "kubric")
    }
    requests = [
        ("pybullet", index)
        for index in select_group_indices(datasets["pybullet"], args.max_pybullet_cases)
    ] + [
        ("kubric", index)
        for index in select_group_indices(datasets["kubric"], args.max_kubric_cases)
    ]
    reports = []
    failures = []
    for position, (source, index) in enumerate(requests, start=1):
        try:
            reports.append(render_sample(datasets[source], index, args))
            print(f"processed={position}/{len(requests)} {source}", flush=True)
        except Exception as error:
            failures.append({"source": source, "index": index, "error": repr(error)})
            print(f"FAILED={position}/{len(requests)} {source} {error!r}", flush=True)
    summary = {
        "successful_cases": len(reports),
        "requested_cases": len(requests),
        "failures": failures,
        "weights": {
            "union": args.union_weight,
            "instance": args.instance_weight,
            "static": args.static_weight,
            "background": args.background_weight,
            "unused": args.unused_weight,
            "focal_bce": args.focal_bce_weight,
            "global": args.mask_loss_weight,
            "ramp": args.mask_ramp,
        },
        "cases": reports,
    }
    (args.output_dir / "vjepa_mask_weight_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    build_index(args.output_dir, reports, args)
    print(json.dumps({key: summary[key] for key in ("successful_cases", "requested_cases", "failures", "weights")}, indent=2))


if __name__ == "__main__":
    main()
