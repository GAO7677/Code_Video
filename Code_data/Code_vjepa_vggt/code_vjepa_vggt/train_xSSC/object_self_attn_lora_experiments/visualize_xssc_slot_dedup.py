#!/usr/bin/env python3
"""Visualize xSSC slot-track de-duplication shapes and merge decisions."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
TRAIN_XSSC_ROOT = ROOT.parent
PROJECT_ROOT = TRAIN_XSSC_ROOT.parent
PACKAGE_PARENT = PROJECT_ROOT.parent
DEFAULT_CONFIG = ROOT / "configs/formal_full_sa_slot_dedup_merge_gpu67.json"
DEFAULT_OUTPUT_DIR = Path("/data/gaoya/agent-data/outputs/xssc_slot_dedup_shape_heatmaps")

for item in (PACKAGE_PARENT, TRAIN_XSSC_ROOT, ROOT):
    text = str(item)
    if text not in sys.path:
        sys.path.insert(0, text)

import launch_slot_dedup_from_config as config_launcher  # noqa: E402
import train_xssc_object_self_attn_lora as object_train  # noqa: E402
import train_xssc_object_self_attn_lora_slot_dedup as dedup_train  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--indices", default="")
    parser.add_argument("--max-scan", type=int, default=24)
    parser.add_argument("--prefer-merged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--xssc-checkpoint-override",
        type=Path,
        default=None,
        help="Use this xSSC .pth instead of the checkpoint resolved from the training config.",
    )
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--wan-hidden-dim", type=int, default=3072)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_train_args(config_path: Path, output_dir: Path) -> tuple[argparse.Namespace, dict]:
    raw, sources = config_launcher.base.load_config(config_path)
    config = config_launcher.validate_config(raw, config_path.expanduser().resolve().parent)
    command = config_launcher.build_command(config, output_dir / "_unused_train_output")
    script_index = next(
        index for index, token in enumerate(command) if str(token).endswith(".py")
    )
    train_argv = [str(item) for item in command[script_index + 1 :]]

    diffsynth_root = str(config["paths"]["diffsynth_root"])
    project_root = str(config["paths"]["project_root"])
    for item in (project_root, diffsynth_root):
        if item and item not in sys.path:
            sys.path.insert(0, item)
    train_args = object_train.tvn.prepare_args(
        dedup_train.build_parser().parse_args(train_argv)
    )
    train_args.no_context_ratio = 0.0
    metadata = {
        "config_sources": sources,
        "resolved_config": config,
        "launch_command_preview": shlex.join(command),
    }
    return train_args, metadata


def to_uint8_video(context_video: torch.Tensor) -> np.ndarray:
    frames = context_video.permute(1, 2, 3, 0).float()
    frames = (frames + 1.0).mul(127.5).round().clamp(0, 255)
    return frames.to(torch.uint8).cpu().numpy()


def preprocess_xssc_exact(
    context_video: torch.Tensor,
    input_size: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict[str, int]]:
    frames = context_video.unsqueeze(0).permute(0, 2, 1, 3, 4).float()
    batch, time_steps, channels, height, width = frames.shape
    crop_size = min(int(height), int(width))
    top = (int(height) - crop_size) // 2
    left = (int(width) - crop_size) // 2
    cropped = frames[..., top : top + crop_size, left : left + crop_size]
    resized = F.interpolate(
        cropped.reshape(batch * time_steps, channels, crop_size, crop_size),
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    resized_rgb = (resized + 1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    mean = resized.new_tensor(object_train.base.XSSC_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(object_train.base.XSSC_IMAGENET_STD).view(1, 3, 1, 1)
    normalized = (resized.mul(127.5).add(127.5).clamp(0, 255) - mean) / std
    xssc_video = normalized.view(batch, time_steps, channels, input_size, input_size)
    cropped_rgb = (cropped[0] + 1.0).mul(127.5).round().clamp(0, 255)
    cropped_rgb = cropped_rgb.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    resized_rgb = resized_rgb.view(batch, time_steps, channels, input_size, input_size)[0]
    resized_rgb = resized_rgb.permute(0, 2, 3, 1).cpu().numpy()
    crop_info = {
        "input_height": int(height),
        "input_width": int(width),
        "crop_size": int(crop_size),
        "crop_top": int(top),
        "crop_left": int(left),
    }
    return xssc_video, cropped_rgb, resized_rgb, crop_info


@torch.no_grad()
def extract_slots_attention(
    model: torch.nn.Module,
    video: torch.Tensor,
    boxes: torch.Tensor,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    model.eval()
    batch, time_steps, _, _, _ = video.shape
    flat_video = video.flatten(0, 1)
    shape_trace: dict[str, list[int]] = {"flat_xssc_video": list(flat_video.shape)}
    with torch.autocast(
        device_type=flat_video.device.type,
        dtype=amp_dtype,
        enabled=flat_video.device.type == "cuda",
    ):
        feature = model.encode_backbone(flat_video).detach()
        shape_trace["backbone_feature"] = list(feature.shape)
        encoded = feature.permute(0, 2, 3, 1)
        shape_trace["encoded_permuted"] = list(encoded.shape)
        encoded = model.encode_posit_embed(encoded).flatten(1, 2)
        shape_trace["encoded_flattened"] = list(encoded.shape)
        encoded = model.encode_project(encoded)
        shape_trace["encoded_projected"] = list(encoded.shape)
        encoded = encoded.view(batch, time_steps, encoded.shape[1], encoded.shape[2])
        shape_trace["encoded_temporal"] = list(encoded.shape)
        boxes = boxes.to(device=encoded.device, dtype=encoded.dtype)

        slots = None
        attentions: list[torch.Tensor] = []
        query_shapes: list[list[int]] = []
        current_slot_shapes: list[list[int]] = []
        current_attention_shapes: list[list[int]] = []
        for frame_id in range(time_steps):
            if frame_id == 0:
                query = model.initializ(boxes[:, 0])
            else:
                query = model.transit(slots, encoded[:, : frame_id + 1])
            query_shapes.append(list(query.shape))
            num_iter = None if frame_id == 0 else 1
            current_slots, current_attention = model.aggregat(
                encoded[:, frame_id],
                query,
                num_iter=num_iter,
            )
            current_slot_shapes.append(list(current_slots.shape))
            current_attention_shapes.append(list(current_attention.shape))
            current_slots = current_slots[:, None]
            slots = current_slots if slots is None else torch.cat((slots, current_slots), dim=1)
            attentions.append(current_attention)
        attention = torch.stack(attentions, dim=1)
        patch_side = int(round(attention.shape[-1] ** 0.5))
        attention = attention.view(
            batch,
            time_steps,
            attention.shape[2],
            patch_side,
            patch_side,
        )
    if slots is None:
        raise RuntimeError("xSSC received zero context frames")
    shape_trace["initial_or_transition_query_per_frame"] = query_shapes[0]
    shape_trace["current_slots_per_frame"] = current_slot_shapes[0]
    shape_trace["current_attention_per_frame"] = current_attention_shapes[0]
    shape_trace["attention_grid"] = list(attention.shape)
    return slots, attention, shape_trace


def build_boxes(
    args: argparse.Namespace,
    video: torch.Tensor,
    num_slots: int,
) -> tuple[torch.Tensor, list[int]]:
    if str(args.xssc_box_source) == "zeros":
        batch, time_steps = int(video.shape[0]), int(video.shape[1])
        return video.new_zeros(batch, time_steps, num_slots, 4), [0] * batch
    builder = object_train.AMGBoxBuilder(
        sam2_config=args.xssc_sam2_config,
        sam2_checkpoint=args.xssc_sam2_checkpoint,
        cache_dir=args.xssc_box_cache_dir,
        filter_args=object_train._amg_filter_args_from_args(args),
    )
    boxes = builder(video, num_slots)
    return boxes, list(builder.last_selected_counts)


def groups_from_similarity(
    similarity: torch.Tensor,
    threshold: float,
    min_keep: int,
) -> list[list[int]]:
    return dedup_train._connected_components_from_similarity(
        similarity,
        threshold=float(threshold),
        min_keep=int(min_keep),
    )


def groups_to_payload(groups: list[list[int]], similarity: np.ndarray) -> list[dict[str, Any]]:
    payload = []
    for group in groups:
        pairs = []
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                pairs.append(
                    {
                        "pair": [int(left), int(right)],
                        "similarity": float(similarity[int(left), int(right)]),
                    }
                )
        payload.append(
            {
                "representative": int(group[0]),
                "members": [int(item) for item in group],
                "duplicates": [int(item) for item in group[1:]],
                "max_pair_similarity": max(
                    [item["similarity"] for item in pairs],
                    default=1.0,
                ),
                "pairs": pairs,
            }
        )
    return payload


def save_similarity_heatmap(
    path: Path,
    matrix: np.ndarray,
    title: str,
    groups: list[list[int]],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(5.0, min(8.0, matrix.shape[0] * 0.58))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=160)
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("slot id")
    ax.set_ylabel("slot id")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if abs(float(value)) > 0.65 else "black"
            weight = "bold" if row != col and value >= threshold else "normal"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color,
                fontweight=weight,
            )
    for group in groups:
        if len(group) <= 1:
            continue
        for left in group:
            for right in group:
                rect = plt.Rectangle(
                    (right - 0.5, left - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#111827",
                    linewidth=1.2,
                )
                ax.add_patch(rect)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def slot_overlay_video(
    frames_rgb: np.ndarray,
    attention: torch.Tensor,
    *,
    groups: list[list[int]] | None = None,
    mode: str = "before",
) -> np.ndarray:
    labels = attention[0].float().cpu().numpy().argmax(axis=1).astype(np.int32)
    remap = {slot: slot for slot in range(PALETTE.shape[0])}
    hidden = set()
    if groups is not None:
        for group in groups:
            rep = int(group[0])
            for item in group[1:]:
                if mode == "mask":
                    hidden.add(int(item))
                else:
                    remap[int(item)] = rep
    output = []
    for frame_id, frame in enumerate(frames_rgb):
        label_small = labels[min(frame_id, labels.shape[0] - 1)].copy()
        for source, target in remap.items():
            if source != target:
                label_small[label_small == source] = target
        label_map = cv2.resize(
            label_small.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        color = PALETTE[label_map % len(PALETTE)].copy()
        if hidden:
            hidden_mask = np.zeros_like(label_small, dtype=bool)
            for slot_id in hidden:
                hidden_mask |= labels[min(frame_id, labels.shape[0] - 1)] == slot_id
            hidden_mask = cv2.resize(
                hidden_mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            color[hidden_mask] = np.asarray([150, 150, 150], dtype=np.uint8)
        blended = cv2.addWeighted(frame, 0.58, color, 0.42, 0.0)
        output.append(blended)
    return np.stack(output, axis=0)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"could not write image: {path}")


def write_video(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb = np.ascontiguousarray(frames_rgb.astype(np.uint8))
    height, width = frames_rgb.shape[1:3]
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
    _, stderr = proc.communicate(frames_rgb.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def pick_indices(dataset, args: argparse.Namespace) -> list[int]:
    if str(args.indices).strip():
        return [int(item) for item in str(args.indices).replace(",", " ").split()]
    if all(hasattr(dataset, name) for name in ("source_lengths", "source_names")):
        chosen = []
        start = 0
        for length in dataset.source_lengths:
            chosen.append(start + min(max(int(length) // 2, 0), int(length) - 1))
            start += int(length)
        return chosen[: max(1, int(args.num_cases))]
    return list(range(min(max(1, int(args.num_cases)), len(dataset))))


def shape_rows_to_html(rows: list[tuple[str, Any, str]]) -> str:
    rendered = []
    for name, shape, note in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td><code>{html.escape(str(shape))}</code></td>"
            f"<td>{html.escape(str(note))}</td>"
            "</tr>"
        )
    return "".join(rendered)


def html_page(cases: list[dict], report: dict) -> str:
    case_sections = []
    for case in cases:
        merge_items = []
        for group in case["merge_groups"]:
            if group["duplicates"]:
                merge_items.append(
                    "<li>"
                    f"rep slot {group['representative']} &lt;= {group['members']} "
                    f"(duplicates {group['duplicates']})"
                    "</li>"
                )
        if not merge_items:
            merge_items.append("<li>No duplicate group above threshold.</li>")
        pair_rows = []
        for group in case["merge_groups"]:
            for pair in group["pairs"]:
                pair_rows.append(
                    "<tr>"
                    f"<td>{pair['pair'][0]}-{pair['pair'][1]}</td>"
                    f"<td>{pair['similarity']:.4f}</td>"
                    f"<td>{group['representative']}</td>"
                    "</tr>"
                )
        pair_table = "".join(pair_rows) or (
            "<tr><td colspan='3'>No off-diagonal pair selected.</td></tr>"
        )
        case_sections.append(
            f"""
            <section class="case">
              <h2>{html.escape(case['title'])}</h2>
              <div class="chips">
                <span>index {case['index']}</span>
                <span>source {html.escape(case['source'])}</span>
                <span>selected AMG boxes {case['selected_boxes']}</span>
                <span>slot groups {case['num_groups']}</span>
                <span>effective object tokens {case['effective_tokens']}</span>
              </div>
              <h3>Shape trace</h3>
              <table><thead><tr><th>step</th><th>shape</th><th>note</th></tr></thead><tbody>{shape_rows_to_html(case['shape_rows'])}</tbody></table>
              <h3>Merged slots</h3>
              <ul>{''.join(merge_items)}</ul>
              <table class="compact"><thead><tr><th>pair</th><th>similarity</th><th>representative</th></tr></thead><tbody>{pair_table}</tbody></table>
              <div class="heatmaps">
                <figure><img src="{case['before_heatmap']}" alt="before similarity"><figcaption>before dedup: slot-track cosine</figcaption></figure>
                <figure><img src="{case['after_heatmap']}" alt="after similarity"><figcaption>after dedup: fixed 11-slot tensor after merge/mask</figcaption></figure>
                <figure><img src="{case['active_heatmap']}" alt="active after similarity"><figcaption>after dedup: active representatives only</figcaption></figure>
              </div>
              <div class="videos">
                <figure><video src="{case['raw_video']}" controls muted preload="metadata"></video><figcaption>context frames</figcaption></figure>
                <figure><video src="{case['before_overlay']}" controls muted preload="metadata"></video><figcaption>before: original slot labels</figcaption></figure>
                <figure><video src="{case['after_overlay']}" controls muted preload="metadata"></video><figcaption>after: duplicate labels remapped to representative</figcaption></figure>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC Slot Dedup Shape Heatmaps</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #101214; color: #edf2f7; font: 14px system-ui, sans-serif; letter-spacing: 0; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 14px 18px; background: #16191c; border-bottom: 1px solid #333b44; }}
    h1 {{ margin: 0 0 6px; font-size: 21px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    h3 {{ margin: 16px 0 8px; font-size: 15px; color: #dbe7f3; }}
    main {{ max-width: 1800px; margin: 0 auto; padding: 18px; }}
    code {{ color: #d5f5ff; }}
    .summary {{ color: #b9c3cc; }}
    .case {{ padding: 18px 0 30px; border-bottom: 1px solid #30363d; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .chips span {{ border: 1px solid #39424b; border-radius: 6px; padding: 5px 8px; background: #181d22; color: #cad3dd; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0 12px; }}
    th, td {{ border: 1px solid #303942; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #192027; color: #dce6ef; }}
    td {{ background: #12171c; color: #cbd5df; }}
    .compact {{ max-width: 680px; }}
    ul {{ margin: 8px 0 12px 20px; color: #ccd6df; }}
    .heatmaps {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .videos {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }}
    figure {{ margin: 0; min-width: 0; }}
    img, video {{ display: block; width: 100%; background: #000; border: 1px solid #333b44; }}
    figcaption {{ padding: 6px 2px; color: #b8c0c9; font-size: 12px; }}
    @media (max-width: 1000px) {{ .heatmaps, .videos {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>xSSC slot-track de-duplication</h1>
    <div class="summary">{html.escape(json.dumps(report, ensure_ascii=False))}</div>
  </header>
  <main>{''.join(case_sections)}</main>
</body>
</html>
"""


def run_case(
    *,
    sample: dict,
    index: int,
    position: int,
    dataset_source: str,
    train_args: argparse.Namespace,
    xssc: torch.nn.Module,
    num_slots: int,
    slot_dim: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    output_dir: Path,
    fps: float,
    wan_hidden_dim: int,
) -> dict:
    context_video = sample["context_video"]
    if context_video.ndim != 4:
        raise ValueError(f"context_video must be [C,T,H,W], got {tuple(context_video.shape)}")
    raw_rgb = to_uint8_video(context_video)
    xssc_video, _, _, crop_info = preprocess_xssc_exact(
        context_video,
        int(train_args.xssc_input_size),
    )
    xssc_video = xssc_video.to(device=device)
    boxes, selected_counts = build_boxes(train_args, xssc_video, num_slots)
    slots, attention, xssc_shape_trace = extract_slots_attention(
        xssc,
        xssc_video,
        boxes.to(device=device),
        amp_dtype,
    )
    similarity = dedup_train.compute_slot_track_similarity(
        slots,
        metric=train_args.xssc_slot_dedup_similarity_metric,
    )
    groups = groups_from_similarity(
        similarity[0],
        threshold=float(train_args.xssc_slot_dedup_similarity_threshold),
        min_keep=int(train_args.xssc_slot_dedup_min_keep),
    )
    deduped_slots, keep_mask, dedup_stats = dedup_train.deduplicate_xssc_slot_tracks(
        slots,
        mode=train_args.xssc_slot_dedup_mode,
        threshold=float(train_args.xssc_slot_dedup_similarity_threshold),
        similarity_metric=train_args.xssc_slot_dedup_similarity_metric,
        min_keep=int(train_args.xssc_slot_dedup_min_keep),
    )
    after_similarity = dedup_train.compute_slot_track_similarity(
        deduped_slots,
        metric=train_args.xssc_slot_dedup_similarity_metric,
    )
    keep_np = keep_mask[0].detach().cpu().numpy().astype(bool)
    active_similarity = after_similarity[0].detach().cpu().numpy()[keep_np][:, keep_np]
    if active_similarity.size == 0:
        active_similarity = np.zeros((1, 1), dtype=np.float32)

    case_dir = output_dir / "assets" / f"case_{position:02d}_index_{int(index):06d}"
    before_heatmap = case_dir / "slot_similarity_before.png"
    after_heatmap = case_dir / "slot_similarity_after_fixed_11.png"
    active_heatmap = case_dir / "slot_similarity_after_active.png"
    before_np = similarity[0].detach().cpu().numpy()
    after_np = after_similarity[0].detach().cpu().numpy()
    threshold = float(train_args.xssc_slot_dedup_similarity_threshold)
    save_similarity_heatmap(before_heatmap, before_np, "before dedup", groups, threshold)
    save_similarity_heatmap(after_heatmap, after_np, "after dedup fixed S=11", groups, threshold)
    save_similarity_heatmap(
        active_heatmap,
        active_similarity,
        "after dedup active reps only",
        [[index] for index in range(active_similarity.shape[0])],
        threshold,
    )

    raw_video = case_dir / "context_frames.mp4"
    before_overlay = case_dir / "slot_overlay_before.mp4"
    after_overlay = case_dir / "slot_overlay_after_merge.mp4"
    write_video(raw_video, raw_rgb, fps)
    write_video(before_overlay, slot_overlay_video(raw_rgb, attention), fps)
    write_video(
        after_overlay,
        slot_overlay_video(
            raw_rgb,
            attention,
            groups=groups,
            mode=str(train_args.xssc_slot_dedup_mode),
        ),
        fps,
    )

    batch = int(slots.shape[0])
    time_steps = int(slots.shape[1])
    retained_slots = int(keep_mask[0].sum().item())
    effective_tokens = time_steps * retained_slots
    shape_rows = [
        ("dataset context_video", list(context_video.shape), "[C,T,H,W] from the same training dataset item"),
        ("batched context_video", [1, *list(context_video.shape)], "[B,C,T,H,W] entering xSSC branch"),
        ("permute for xSSC", [batch, time_steps, 3, crop_info["input_height"], crop_info["input_width"]], "[B,T,C,H,W]"),
        ("center crop", [batch, time_steps, 3, crop_info["crop_size"], crop_info["crop_size"]], f"top={crop_info['crop_top']}, left={crop_info['crop_left']}"),
        ("resize/normalize", list(xssc_video.shape), "xSSC input [B,T,C,256,256]"),
        ("AMG pseudo boxes", list(boxes.shape), "[B,T,S,4], frame-0 boxes repeated over T"),
        ("flat xSSC frames", xssc_shape_trace["flat_xssc_video"], "[B*T,C,256,256]"),
        ("DINOv3 backbone feature", xssc_shape_trace["backbone_feature"], "before xSSC encode_project"),
        ("patch tokens before project", xssc_shape_trace["encoded_flattened"], "256 spatial patches for 256x256 ViT/16"),
        ("patch tokens after project", xssc_shape_trace["encoded_projected"], "xSSC feature dimension"),
        ("temporal patch tokens", xssc_shape_trace["encoded_temporal"], "[B,T,Npatch,D]"),
        ("initializ/transit query", xssc_shape_trace["initial_or_transition_query_per_frame"], "[B,S,512] query for one frame"),
        ("aggregat current slots", xssc_shape_trace["current_slots_per_frame"], "[B,S,512] per frame"),
        ("xSSC slots before dedup", list(slots.shape), "[B,T,S,512]"),
        ("slot-track similarity", list(similarity.shape), "[B,S,S]"),
        ("dedup keep_mask", list(keep_mask.shape), f"retained {retained_slots}/{num_slots} slot tracks"),
        ("xSSC slots after dedup", list(deduped_slots.shape), "shape preserved; duplicate tracks zeroed or merged into representative"),
        ("LayerNorm", list(deduped_slots.shape), "[B,T,S,512]"),
        ("Linear projection", [batch, time_steps, num_slots, wan_hidden_dim], f"512 -> Wan hidden dim {wan_hidden_dim}"),
        ("time embedding", [1, time_steps, 1, wan_hidden_dim], "added after projection"),
        ("apply keep_mask again", [batch, time_steps, num_slots, wan_hidden_dim], "prevents masked duplicates from reappearing via time embedding"),
        ("object tokens", [batch, time_steps * num_slots, wan_hidden_dim], f"fixed token count; effective nonzero tokens {effective_tokens}"),
    ]
    payload = {
        "title": f"case {position:02d} | {dataset_source}",
        "index": int(index),
        "source": dataset_source,
        "selected_boxes": int(selected_counts[0]) if selected_counts else 0,
        "num_groups": len(groups),
        "effective_tokens": effective_tokens,
        "shape_rows": shape_rows,
        "merge_groups": groups_to_payload(groups, before_np),
        "keep_mask": keep_np.astype(int).tolist(),
        "dedup_stats": dedup_stats,
        "threshold": threshold,
        "similarity_metric": str(train_args.xssc_slot_dedup_similarity_metric),
        "dedup_mode": str(train_args.xssc_slot_dedup_mode),
        "raw_video": str(raw_video.relative_to(output_dir)),
        "before_overlay": str(before_overlay.relative_to(output_dir)),
        "after_overlay": str(after_overlay.relative_to(output_dir)),
        "before_heatmap": str(before_heatmap.relative_to(output_dir)),
        "after_heatmap": str(after_heatmap.relative_to(output_dir)),
        "active_heatmap": str(active_heatmap.relative_to(output_dir)),
        "crop_info": crop_info,
        "sample_metadata": dict(sample.get("metadata", {})),
    }
    (case_dir / "case_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and args.force:
        for path in output_dir.glob("*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_args, config_metadata = load_train_args(args.config, output_dir)
    if args.xssc_checkpoint_override is not None:
        override = args.xssc_checkpoint_override.expanduser().resolve()
        if not override.is_file():
            raise FileNotFoundError(f"xSSC checkpoint override does not exist: {override}")
        train_args.xssc_checkpoint = str(override)
        config_metadata["xssc_checkpoint_override"] = str(override)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = torch.bfloat16

    dataset = object_train.base.build_dataset(train_args)
    requested_indices = pick_indices(dataset, args)
    checkpoint = object_train.resolve_latest_xssc_checkpoint(
        train_args.xssc_checkpoint,
        train_args.xssc_checkpoint_latest_dir,
    )
    xssc, slot_dim, num_slots = object_train._load_dinov3_xssc_model(
        xssc_root=train_args.xssc_root,
        config_path=train_args.xssc_config,
        checkpoint_path=checkpoint,
        dinov3_root=train_args.dinov3_root,
        dinov3_checkpoint=train_args.dinov3_checkpoint,
        device=device,
    )
    cases = []
    fallback_cases = []
    scanned = 0
    candidate_indices = list(requested_indices)
    if not str(args.indices).strip() and args.prefer_merged:
        candidate_indices.extend(
            index for index in range(min(len(dataset), int(args.max_scan)))
            if index not in set(candidate_indices)
        )
    for index in candidate_indices:
        if len(cases) >= int(args.num_cases):
            break
        scanned += 1
        sample = dataset[int(index)]
        metadata = dict(sample.get("metadata", {}))
        source = str(metadata.get("dataset_source", "unknown"))
        payload = run_case(
            sample=sample,
            index=int(index),
            position=len(cases) + len(fallback_cases) + 1,
            dataset_source=source,
            train_args=train_args,
            xssc=xssc,
            num_slots=num_slots,
            slot_dim=slot_dim,
            device=device,
            amp_dtype=amp_dtype,
            output_dir=output_dir,
            fps=float(args.fps),
            wan_hidden_dim=int(args.wan_hidden_dim),
        )
        has_merge = any(group["duplicates"] for group in payload["merge_groups"])
        print(
            f"[scan {scanned}] index={index} source={source} "
            f"retained={sum(payload['keep_mask'])}/{num_slots} has_merge={has_merge}",
            flush=True,
        )
        if args.prefer_merged and not str(args.indices).strip() and not has_merge:
            fallback_cases.append(payload)
            continue
        cases.append(payload)
    while len(cases) < int(args.num_cases) and fallback_cases:
        cases.append(fallback_cases.pop(0))
    cases = cases[: int(args.num_cases)]

    report = {
        "config": str(args.config.expanduser().resolve()),
        "output_dir": str(output_dir),
        "xssc_checkpoint": checkpoint,
        "dinov3_checkpoint": train_args.dinov3_checkpoint,
        "dedup_mode": str(train_args.xssc_slot_dedup_mode),
        "similarity_metric": str(train_args.xssc_slot_dedup_similarity_metric),
        "similarity_threshold": float(train_args.xssc_slot_dedup_similarity_threshold),
        "min_keep": int(train_args.xssc_slot_dedup_min_keep),
        "slot_shape_contract": f"[B,T,{num_slots},{slot_dim}] -> [B,T,{num_slots},{slot_dim}] -> [B,{8 * num_slots},{int(args.wan_hidden_dim)}]",
        "note": "Dedup preserves fixed S slots; duplicate tracks are merged/masked before projection and keep_mask is applied again after time embedding.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "report": report,
                "config_metadata": config_metadata,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        html_page(cases, report),
        encoding="utf-8",
    )
    print(f"viewer={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
