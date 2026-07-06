#!/usr/bin/env python3
"""
Run command example:

CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/analyze_local_vjepa.py \
  --device cuda:0 \
  --port 8791 \
  --serve
"""
from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image

from code_vjepa_free.vjepa_guidance.motion_masks import MotionMaskResult, compute_all_motion_masks
from code_vjepa_free.vjepa_guidance.vjepa_surprise import (
    VJEPASurpriseEnergy,
    _window_video,
    add_vjepa_repo_to_path,
    generate_causal_masks,
    prepare_video_for_vjepa,
)


DEFAULT_OUT_DIR = Path("/data/gaoya/agent-data/outputs/local_vjepa_diagnostics/0613pybullet_sample_001460_w002")
DEFAULT_VJEPA_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vith.pt")
DEFAULT_CHOSEN_SCHEME = "background_residual"

DEFAULT_VIDEOS: list[tuple[str, Path]] = [
    (
        "train0705_step002500::baseline",
        Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step002500/baseline/step-002500/0613pybullet_sample_001460_w002.mp4"),
    ),
    (
        "train0705_step002500::guided",
        Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step002500/guided/step-002500/0613pybullet_sample_001460_w002.mp4"),
    ),
    (
        "train0705_step005000::baseline",
        Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step005000/baseline/step-005000/0613pybullet_sample_001460_w002.mp4"),
    ),
    (
        "train0705_step005000::guided",
        Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step005000/guided/step-005000/0613pybullet_sample_001460_w002.mp4"),
    ),
    (
        "wan22_early_lora_step000500::baseline",
        Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/wan22_early_lora_step000500/baseline/0613pybullet_sample_001460_w002.mp4"),
    ),
    (
        "wan22_early_lora_step000500::guided",
        Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/wan22_early_lora_step000500/guided/0613pybullet_sample_001460_w002.mp4"),
    ),
    (
        "wan22_official_ti2v5b::baseline",
        Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/wan22_official_ti2v5b/baseline/0613pybullet_sample_001460_w002.mp4"),
    ),
    (
        "wan22_official_ti2v5b::guided",
        Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/wan22_official_ti2v5b/guided/0613pybullet_sample_001460_w002.mp4"),
    ),
]


@dataclass
class LocalSchemeScore:
    local_score: float
    coverage: float
    threshold: float


@dataclass
class VideoDiagnostic:
    label: str
    family: str
    variant: str
    video_path: str
    served_video_path: str
    raw_strip_path: str
    summary_frame_path: str
    global_score: float
    schemes: dict[str, LocalSchemeScore]
    best_scheme: str
    best_local_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare global vs local V-JEPA surprise on existing A/B videos.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vjepa-model", type=str, default="vith")
    parser.add_argument("--vjepa-ckpt", type=Path, default=DEFAULT_VJEPA_CKPT)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--diff-quantile", type=float, default=0.85)
    parser.add_argument("--flow-quantile", type=float, default=0.80)
    parser.add_argument("--dilate-px", type=int, default=14)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()


def load_video_rgb(video_path: Path) -> np.ndarray:
    reader = VideoReader(str(video_path), ctx=cpu(0))
    return reader.get_batch(np.arange(len(reader))).asnumpy()


def video_to_btchw(video_thwc_u8: np.ndarray) -> torch.Tensor:
    video = torch.from_numpy(video_thwc_u8).permute(3, 0, 1, 2).float()
    video = video / 127.5 - 1.0
    return video.unsqueeze(0).contiguous()


def _even_indices(total: int, count: int) -> np.ndarray:
    if total <= count:
        return np.arange(total, dtype=np.int64)
    return np.round(np.linspace(0, total - 1, count)).astype(np.int64)


def build_raw_strip(video_thwc_u8: np.ndarray, out_path: Path, *, count: int = 8) -> None:
    idx = _even_indices(video_thwc_u8.shape[0], count)
    frames = [Image.fromarray(video_thwc_u8[int(i)]) for i in idx]
    widths = [frame.width for frame in frames]
    heights = [frame.height for frame in frames]
    canvas = Image.new("RGB", (sum(widths), max(heights)), (255, 255, 255))
    cursor = 0
    for frame in frames:
        canvas.paste(frame, (cursor, 0))
        cursor += frame.width
    canvas.save(out_path)


def _overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
    base = frame_rgb.astype(np.float32)
    tint = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = base * (1.0 - 0.45 * alpha) + tint * (0.45 * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_binary_mask(mask: np.ndarray, *, fg_color: tuple[int, int, int], bg_color: tuple[int, int, int]) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[...] = np.array(bg_color, dtype=np.uint8)
    out[binary > 0] = np.array(fg_color, dtype=np.uint8)
    return out


def _overlay_background_region(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    motion = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
    background = 1.0 - motion
    base = frame_rgb.astype(np.float32)
    tinted = np.array([70.0, 110.0, 210.0], dtype=np.float32).reshape(1, 1, 3)
    out = base * (1.0 - 0.35 * background) + tinted * (0.35 * background)
    out = out * (0.85 + 0.15 * motion)
    return np.clip(out, 0, 255).astype(np.uint8)


def _heat_to_rgb(heat: np.ndarray) -> np.ndarray:
    heat_u8 = np.clip(heat * 255.0, 0.0, 255.0).astype(np.uint8)
    color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)  # type: ignore[name-defined]
    return color_bgr[..., ::-1]


def _resize_heat(heat: np.ndarray, frame_hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(heat.astype(np.float32), (frame_hw[1], frame_hw[0]), interpolation=cv2.INTER_LINEAR)  # type: ignore[name-defined]


def build_summary_panel(
    video_thwc_u8: np.ndarray,
    masks: dict[str, MotionMaskResult],
    aggregated_heat: dict[str, np.ndarray],
    out_path: Path,
    *,
    chosen_scheme: str = DEFAULT_CHOSEN_SCHEME,
) -> None:
    future_index = min(video_thwc_u8.shape[0] - 1, max(0, video_thwc_u8.shape[0] // 2))
    base_frame = video_thwc_u8[future_index]
    if chosen_scheme not in masks:
        raise KeyError(f"unknown chosen scheme: {chosen_scheme}")

    colors = {
        "frame_diff": (255, 64, 64),
        "background_residual": (70, 220, 120),
        "hybrid": (255, 180, 0),
    }
    chosen_mask = masks[chosen_scheme].mask[future_index]
    chosen_binary = _render_binary_mask(
        chosen_mask,
        fg_color=colors[chosen_scheme],
        bg_color=(242, 238, 230),
    )
    chosen_overlay = _overlay_mask(base_frame, chosen_mask, color=colors[chosen_scheme])
    chosen_background = _overlay_background_region(base_frame, chosen_mask)

    tiles: list[Image.Image] = [
        Image.fromarray(base_frame),
        Image.fromarray(chosen_binary),
        Image.fromarray(chosen_overlay),
        Image.fromarray(chosen_background),
    ]
    labels = [
        "raw frame",
        f"chosen motion region (binary, {chosen_scheme})",
        f"chosen motion region overlay ({chosen_scheme})",
        "background region overlay (complement of chosen region)",
    ]

    for name in ("frame_diff", "background_residual", "hybrid"):
        overlay = _overlay_mask(base_frame, masks[name].mask[future_index], color=colors[name])
        tiles.append(Image.fromarray(overlay))
        labels.append(f"candidate mask overlay: {name}")
    for name in ("frame_diff", "background_residual", "hybrid"):
        heat = _resize_heat(aggregated_heat[name], base_frame.shape[:2])
        heat_rgb = _heat_to_rgb(heat)
        blend = np.clip(0.55 * base_frame.astype(np.float32) + 0.45 * heat_rgb.astype(np.float32), 0, 255).astype(np.uint8)
        tiles.append(Image.fromarray(blend))
        labels.append(f"token surprise heatmap: {name}")

    tile_w = base_frame.shape[1]
    tile_h = base_frame.shape[0]
    pad = 12
    caption_h = 34
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * (tile_h + caption_h) + (rows + 1) * pad), (248, 244, 236))
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        for index, (tile, label) in enumerate(zip(tiles, labels)):
            row = index // cols
            col = index % cols
            x = pad + col * (tile_w + pad)
            y = pad + row * (tile_h + caption_h + pad)
            canvas.paste(tile.resize((tile_w, tile_h)), (x, y))
            draw.text((x, y + tile_h + 8), label, fill=(20, 20, 20))
    except Exception:
        for index, tile in enumerate(tiles):
            row = index // cols
            col = index % cols
            x = pad + col * (tile_w + pad)
            y = pad + row * (tile_h + caption_h + pad)
            canvas.paste(tile.resize((tile_w, tile_h)), (x, y))
    canvas.save(out_path)


def compute_local_scores(
    video_btchw: torch.Tensor,
    motion_mask: np.ndarray,
    *,
    energy: VJEPASurpriseEnergy,
    window_size: int,
    context_frames: int,
    stride: int,
) -> tuple[float, float, np.ndarray]:
    add_vjepa_repo_to_path()
    from src.masks.utils import apply_masks

    device = energy.device
    encoder = energy.encoder
    target_encoder = energy.target_encoder
    predictor = energy.predictor

    model_dtype = next(encoder.parameters()).dtype
    x = prepare_video_for_vjepa(video_btchw.to(device), img_size=energy.img_size).to(dtype=model_dtype)
    pieces = _window_video(x, window_size=window_size, stride=stride)
    patch_size = int(encoder.patch_size if isinstance(encoder.patch_size, int) else encoder.patch_size[0])
    tubelet_size = int(encoder.tubelet_size if isinstance(encoder.tubelet_size, int) else encoder.tubelet_size[0])
    grid_h = energy.img_size // patch_size
    grid_w = energy.img_size // patch_size
    future_depth = max(1, (window_size - context_frames) // tubelet_size)

    global_scores: list[float] = []
    local_scores: list[float] = []
    heat_accum = torch.zeros((grid_h, grid_w), dtype=torch.float32, device=device)
    heat_count = 0

    for chunk_id in range(pieces.shape[0]):
        start = chunk_id * stride
        chunk = pieces[chunk_id : chunk_id + 1]
        masks_enc, masks_pred = generate_causal_masks(
            batch_size=1,
            img_size=energy.img_size,
            frames_per_clip=window_size,
            encoder=encoder,
            context_frames=context_frames,
            device=chunk.device,
        )
        target_tokens = target_encoder(chunk)
        if isinstance(target_tokens, (list, tuple)):
            target_tokens = target_tokens[-1]
        target_tokens = F.layer_norm(target_tokens, (target_tokens.shape[-1],))
        context_tokens = encoder(chunk, masks_enc)
        predicted = predictor(context_tokens, masks_enc, masks_pred)
        predicted = F.layer_norm(predicted, (predicted.shape[-1],))
        masked_target = apply_masks(target_tokens, masks_pred, concat=False)[0]
        token_surprise = 1.0 - F.cosine_similarity(predicted, masked_target, dim=-1)
        token_map = token_surprise.view(future_depth, grid_h, grid_w)
        global_scores.append(float(token_map.mean().item()))

        future_mask = motion_mask[start + context_frames : start + window_size]
        future_mask_t = torch.from_numpy(future_mask).to(device=device, dtype=torch.float32)
        future_mask_t = future_mask_t.unsqueeze(0).unsqueeze(0)
        weights = F.interpolate(
            future_mask_t,
            size=(future_depth, grid_h, grid_w),
            mode="trilinear",
            align_corners=False,
        )[0, 0]
        weights = weights.clamp(0.0, 1.0)
        if float(weights.sum().item()) <= 1.0e-6:
            local_scores.append(float(token_map.mean().item()))
        else:
            local_scores.append(float((token_map * weights).sum().item() / weights.sum().item()))
        heat_accum += token_map.mean(dim=0)
        heat_count += 1

    heat_map = (heat_accum / max(1, heat_count)).detach().cpu().numpy().astype(np.float32)
    return float(np.mean(global_scores)), float(np.mean(local_scores)), heat_map


def summarize_pairwise_separation(rows: list[VideoDiagnostic]) -> dict[str, Any]:
    grouped: dict[str, dict[str, VideoDiagnostic]] = {}
    for row in rows:
        grouped.setdefault(row.family, {})[row.variant] = row

    scheme_names = ("frame_diff", "background_residual", "hybrid")
    summary_rows: list[dict[str, Any]] = []
    aggregate: dict[str, list[float]] = {name: [] for name in scheme_names}
    global_deltas: list[float] = []

    for family, variants in sorted(grouped.items()):
        baseline = variants.get("baseline")
        guided = variants.get("guided")
        if baseline is None or guided is None:
            continue
        global_delta = float(guided.global_score - baseline.global_score)
        global_deltas.append(abs(global_delta))
        item: dict[str, Any] = {"family": family, "global_delta": global_delta}
        for name in scheme_names:
            local_delta = float(guided.schemes[name].local_score - baseline.schemes[name].local_score)
            aggregate[name].append(abs(local_delta))
            item[f"{name}_delta"] = local_delta
        summary_rows.append(item)

    scheme_rank = []
    global_abs_mean = float(np.mean(global_deltas)) if global_deltas else 0.0
    for name in scheme_names:
        local_abs_mean = float(np.mean(aggregate[name])) if aggregate[name] else 0.0
        ratio = local_abs_mean / max(global_abs_mean, 1.0e-6)
        scheme_rank.append(
            {
                "name": name,
                "mean_abs_local_delta": local_abs_mean,
                "mean_abs_global_delta": global_abs_mean,
                "amplification_vs_global": ratio,
            }
        )
    scheme_rank.sort(key=lambda row: row["amplification_vs_global"], reverse=True)
    return {"pair_rows": summary_rows, "scheme_rank": scheme_rank}


def build_html(rows: list[VideoDiagnostic], pairwise: dict[str, Any], out_html: Path) -> None:
    scheme_rank = pairwise["scheme_rank"]
    pair_rows = pairwise["pair_rows"]
    top_scheme = scheme_rank[0]["name"] if scheme_rank else "n/a"
    chosen_scheme = DEFAULT_CHOSEN_SCHEME

    rank_cards = "".join(
        f"""
        <div class="rank-card">
          <div class="rank-name">{html.escape(row['name'])}</div>
          <div class="rank-main">|Δlocal| / |Δglobal| = {row['amplification_vs_global']:.3f}</div>
          <div class="rank-sub">mean |Δlocal| = {row['mean_abs_local_delta']:.5f}</div>
        </div>
        """
        for row in scheme_rank
    )

    pair_table_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(row['family'])}</td>
          <td>{row['global_delta']:+.5f}</td>
          <td>{row['frame_diff_delta']:+.5f}</td>
            <td>{row['background_residual_delta']:+.5f}</td>
            <td>{row['hybrid_delta']:+.5f}</td>
        </tr>
        """
        for row in pair_rows
    )

    cards = []
    for row in rows:
        scheme_stats = "".join(
            f"<div class='pill'>{html.escape(name)} {score.local_score:.4f} | cov {score.coverage * 100:.1f}%</div>"
            for name, score in row.schemes.items()
        )
        cards.append(
            f"""
            <div class="card">
              <div class="card-head">
                <div>
                  <div class="label">{html.escape(row.label)}</div>
                  <div class="sub">{html.escape(row.video_path)}</div>
                </div>
                <div class="global">global {row.global_score:.4f}</div>
              </div>
              <div class="pill-row">
                <div class="pill">best local scheme: {html.escape(row.best_scheme)} ({row.best_local_score:.4f})</div>
                {scheme_stats}
              </div>
              <div class="media-grid">
                <div>
                  <video controls loop muted preload="metadata" width="560">
                    <source src="{html.escape(Path(row.served_video_path).name)}" type="video/mp4">
                  </video>
                </div>
                <div>
                  <img class="panel" src="{html.escape(Path(row.summary_frame_path).name)}" />
                  <img class="strip" src="{html.escape(Path(row.raw_strip_path).name)}" />
                </div>
              </div>
            </div>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>Local vs Global V-JEPA Diagnostic</title>
  <style>
    body {{ margin: 0; background: #f6f2ea; color: #1e1e1e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .lead {{ color: #5f5648; line-height: 1.6; margin-bottom: 20px; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 14px; margin-bottom: 20px; }}
    .rank-card {{ background: #fff; border: 1px solid #e0d7ca; border-radius: 12px; padding: 14px 16px; }}
    .rank-name {{ font-weight: 700; font-size: 16px; margin-bottom: 6px; }}
    .rank-main {{ font-size: 18px; }}
    .rank-sub {{ color: #6c6558; margin-top: 4px; }}
    .table-wrap {{ background: #fff; border: 1px solid #e0d7ca; border-radius: 12px; padding: 16px; margin-bottom: 22px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #ece5d8; }}
    th {{ background: #faf6ef; }}
    .card {{ background: #fff; border: 1px solid #e0d7ca; border-radius: 14px; padding: 16px; margin-bottom: 18px; }}
    .card-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; margin-bottom: 12px; }}
    .label {{ font-size: 18px; font-weight: 700; }}
    .sub {{ color: #756d60; font-size: 12px; margin-top: 4px; word-break: break-all; }}
    .global {{ font-size: 20px; font-weight: 700; white-space: nowrap; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .pill {{ background: #f4eee3; border: 1px solid #ded3c3; border-radius: 999px; padding: 6px 10px; font-size: 12px; }}
    .media-grid {{ display: grid; grid-template-columns: 580px minmax(480px, 1fr); gap: 18px; align-items: start; }}
    video {{ width: 100%; border-radius: 10px; background: #000; }}
    img.panel {{ width: 100%; border-radius: 10px; border: 1px solid #ddd2c2; display: block; margin-bottom: 10px; }}
    img.strip {{ width: 100%; border-radius: 10px; border: 1px solid #ddd2c2; display: block; }}
    .note {{ background: #fff9e9; border: 1px solid #ead9a2; border-radius: 10px; padding: 12px 14px; margin-bottom: 20px; color: #5d512e; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Local vs Global V-JEPA Diagnostic</h1>
    <div class="lead">
      同一个 case 上，比较 `global token mean surprise` 与三种 `motion-region masked token mean surprise`。
      当前用已有 8 个 A/B 视频做 sensitivity check。这里的“更有用”暂时定义为：在 baseline/guided 对比里，局部指标是否比全局指标有更大的分离度。
    </div>
    <div class="note">
      当前页面里实际拿来强调展示的“选中运动区域”是 <b>{html.escape(chosen_scheme)}</b>。
      当前最强分离方案也是 <b>{html.escape(top_scheme)}</b>。
      先看每张卡右侧第一排：第 2 张是二值运动区域，第 3 张是运动区域 overlay，第 4 张是背景区域 overlay。
      颜色约定：绿色 = 选中运动区域，蓝色 = 背景区域，红色/橙色 = 其他候选 mask。
    </div>
    <div class="summary">{rank_cards}</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>family</th>
            <th>Δglobal</th>
            <th>Δframe_diff local</th>
            <th>Δbackground_residual local</th>
            <th>Δhybrid local</th>
          </tr>
        </thead>
        <tbody>{pair_table_rows}</tbody>
      </table>
    </div>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    out_html.write_text(html_text, encoding="utf-8")


def _safe_stem(label: str) -> str:
    return label.replace("::", "__").replace("/", "_")


def ensure_media_link(src: Path, dst_dir: Path, *, dst_name: str) -> Path:
    dst = dst_dir / dst_name
    if dst.exists() or dst.is_symlink():
        return dst
    try:
        dst.symlink_to(src)
    except OSError:
        import shutil

        shutil.copy2(src, dst)
    return dst


def analyze_video(
    label: str,
    video_path: Path,
    *,
    energy: VJEPASurpriseEnergy,
    args: argparse.Namespace,
    out_dir: Path,
) -> VideoDiagnostic:
    family, variant = label.split("::", 1)
    video_thwc_u8 = load_video_rgb(video_path)
    video_btchw = video_to_btchw(video_thwc_u8)
    masks = compute_all_motion_masks(
        video_thwc_u8,
        diff_quantile=args.diff_quantile,
        flow_quantile=args.flow_quantile,
        dilate_px=args.dilate_px,
        blur_ksize=args.blur_ksize,
    )

    raw_strip_path = out_dir / f"{family}_{variant}_strip.png"
    summary_frame_path = out_dir / f"{family}_{variant}_summary.png"
    build_raw_strip(video_thwc_u8, raw_strip_path)

    global_score = None
    scheme_scores: dict[str, LocalSchemeScore] = {}
    aggregated_heat: dict[str, np.ndarray] = {}
    for name, motion in masks.items():
        global_value, local_value, heat_map = compute_local_scores(
            video_btchw,
            motion.mask,
            energy=energy,
            window_size=args.window_size,
            context_frames=args.context_frames,
            stride=args.stride,
        )
        if global_score is None:
            global_score = global_value
        scheme_scores[name] = LocalSchemeScore(
            local_score=float(local_value),
            coverage=float(motion.coverage),
            threshold=float(motion.threshold),
        )
        aggregated_heat[name] = heat_map

    assert global_score is not None
    build_summary_panel(
        video_thwc_u8,
        masks,
        aggregated_heat,
        summary_frame_path,
        chosen_scheme=DEFAULT_CHOSEN_SCHEME,
    )
    best_name, best_score = min(
        ((name, score.local_score) for name, score in scheme_scores.items()),
        key=lambda item: item[1],
    )
    return VideoDiagnostic(
        label=label,
        family=family,
        variant=variant,
        video_path=str(video_path),
        served_video_path=str(out_dir / f"{_safe_stem(label)}.mp4"),
        raw_strip_path=str(raw_strip_path),
        summary_frame_path=str(summary_frame_path),
        global_score=float(global_score),
        schemes=scheme_scores,
        best_scheme=best_name,
        best_local_score=float(best_score),
    )


def save_summary(rows: list[VideoDiagnostic], pairwise: dict[str, Any], out_path: Path) -> None:
    payload = {
        "rows": [
            {
                "label": row.label,
                "family": row.family,
                "variant": row.variant,
                "video_path": row.video_path,
                "served_video_path": row.served_video_path,
                "raw_strip_path": row.raw_strip_path,
                "summary_frame_path": row.summary_frame_path,
                "global_score": row.global_score,
                "best_scheme": row.best_scheme,
                "best_local_score": row.best_local_score,
                "schemes": {
                    name: {
                        "local_score": score.local_score,
                        "coverage": score.coverage,
                        "threshold": score.threshold,
                    }
                    for name, score in row.schemes.items()
                },
            }
            for row in rows
        ],
        "pairwise": pairwise,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    energy = VJEPASurpriseEnergy(
        model_name=str(args.vjepa_model),
        device=str(args.device),
        local_torchhub=True,
        checkpoint_path=Path(args.vjepa_ckpt).expanduser().resolve(),
    )

    rows: list[VideoDiagnostic] = []
    for label, video_path in DEFAULT_VIDEOS:
        if not video_path.is_file():
            raise FileNotFoundError(f"missing input video: {video_path}")
        ensure_media_link(video_path, out_dir, dst_name=f"{_safe_stem(label)}.mp4")
        print(f"[analyze] {label}")
        rows.append(analyze_video(label, video_path, energy=energy, args=args, out_dir=out_dir))

    pairwise = summarize_pairwise_separation(rows)
    summary_path = out_dir / "summary.json"
    html_path = out_dir / "index.html"
    save_summary(rows, pairwise, summary_path)
    build_html(rows, pairwise, html_path)

    scheme_rank = pairwise["scheme_rank"]
    if scheme_rank:
        best = scheme_rank[0]
        print(
            "[summary] best separation scheme:",
            best["name"],
            f"|Δlocal|/|Δglobal|={best['amplification_vs_global']:.3f}",
        )
    print(f"[summary] html: {html_path}")
    print(f"[summary] json: {summary_path}")

    if args.serve:
        import http.server
        import socketserver

        os.chdir(out_dir)
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("0.0.0.0", int(args.port)), handler) as httpd:
            print(f"Serving at http://localhost:{int(args.port)}/index.html")
            httpd.serve_forever()


if __name__ == "__main__":
    main()
