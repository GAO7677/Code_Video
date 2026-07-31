#!/usr/bin/env python3
"""Check whether xSSC slots separate objects and motion signatures in videos."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
for item in (ROOT, PROJECT_ROOT, PACKAGE_PARENT):
    text = str(item)
    if text not in sys.path:
        sys.path.insert(0, text)

import analyze_official_xssc_dynamics_raft as base  # noqa: E402
from code_vjepa_vggt.utils.video_io import read_video_prefix, preprocess_video_rgb_uint8  # noqa: E402


DEFAULT_JSONS = [
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_000301_w000.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001460_w002.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_000331_w001.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001455_w000.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_000336_w001.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_000336_w001.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px.json",
]
DEFAULT_OUTPUT_DIR = Path("/data/gaoya/agent-data/outputs/xssc_object_slot_separation_cases")

PALETTE = np.asarray(
    [
        [230, 57, 70],
        [69, 123, 157],
        [42, 157, 143],
        [244, 162, 97],
        [131, 56, 236],
        [255, 183, 3],
        [17, 138, 178],
        [239, 71, 111],
        [6, 214, 160],
        [255, 127, 80],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="append", default=None)
    parser.add_argument("--official-root", type=Path, default=Path("/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--xssc-input-size", type=int, default=256)
    parser.add_argument("--xssc-batch-size", type=int, default=16)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--skip-raft", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def safe_id(text: str) -> str:
    return base.safe_id(text)


def resolve_source_video(payload: dict[str, Any], json_path: Path) -> Path:
    value = payload.get("source_video") or payload.get("input_video")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{json_path} has no source_video/input_video")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (json_path.parent / path).resolve()
    return path.resolve()


def read_cases(json_paths: list[str], max_cases: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = []
    duplicates = []
    seen_json = {}
    seen_source = {}
    for raw_position, raw in enumerate(json_paths, start=1):
        json_path = Path(raw).expanduser().resolve()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source_video = resolve_source_video(payload, json_path)
        if str(json_path) in seen_json:
            duplicates.append({"type": "json", "current": raw_position, "first": seen_json[str(json_path)], "path": str(json_path)})
            continue
        seen_json[str(json_path)] = raw_position
        if str(source_video) in seen_source:
            duplicates.append({"type": "source_video", "current": raw_position, "first": seen_source[str(source_video)], "path": str(source_video)})
            continue
        seen_source[str(source_video)] = raw_position
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        cases.append(
            {
                "json": str(json_path),
                "case_id": safe_id(json_path.stem),
                "source_video": str(source_video),
                "caption": str(payload.get("input_caption", "")),
                "raw_position": raw_position,
            }
        )
        if max_cases > 0 and len(cases) >= max_cases:
            break
    return cases, duplicates


def read_source_video(video_path: Path, num_frames: int, height: int, width: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    frames, frame_indices = read_video_prefix(video_path, num_frames)
    if len(frames) <= 0:
        raise RuntimeError(f"no readable frames: {video_path}")
    video = preprocess_video_rgb_uint8(
        frames,
        (height, width),
        resize_mode="cover_crop",
        cover_crop_hw=(height, width),
    )
    return video, frames, frame_indices


def attention_to_hard_labels(attention: np.ndarray, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    labels_low = attention.argmax(axis=1).astype(np.int32)
    labels = np.stack(
        [
            cv2.resize(item, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
            for item in labels_low
        ],
        axis=0,
    )
    return labels_low, labels


def overlay_all_slots(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    colors = PALETTE[labels % len(PALETTE)]
    return (rgb.astype(np.float32) * (1.0 - alpha) + colors.astype(np.float32) * alpha).round().clip(0, 255).astype(np.uint8)


def overlay_slot_grid(rgb: np.ndarray, labels: np.ndarray, num_slots: int) -> np.ndarray:
    frames = []
    cols = min(4, max(1, int(num_slots)))
    rows = int(math.ceil(max(1, int(num_slots)) / cols))
    for t in range(len(rgb)):
        panels = []
        for slot_id in range(num_slots):
            mask = labels[t] == slot_id
            color = PALETTE[slot_id % len(PALETTE)]
            panel = np.ascontiguousarray((rgb[t].astype(np.float32) * 0.38).astype(np.uint8))
            panel[mask] = (panel[mask].astype(np.float32) * 0.25 + color.astype(np.float32) * 0.75).round().clip(0, 255).astype(np.uint8)
            cv2.putText(panel, f"S{slot_id}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(int(v) for v in color.tolist()), 2, cv2.LINE_AA)
            panels.append(panel)
        while len(panels) < rows * cols:
            panels.append(np.zeros_like(panels[0]))
        grid_rows = [
            np.concatenate(panels[row * cols : (row + 1) * cols], axis=1)
            for row in range(rows)
        ]
        frames.append(np.concatenate(grid_rows, axis=0))
    return np.stack(frames, axis=0)


def plot_matrix(path: Path, matrix: np.ndarray, title: str, labels: list[str], cmap: str = "coolwarm", vmin=-1.0, vmax=1.0) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 4.5), dpi=150)
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            ax.text(x, y, "nan" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_slot_curves(path: Path, curves: dict[str, np.ndarray], title: str, active_slots: list[int]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), dpi=150, sharex=True)
    names = ["d_adj", "slot_flow", "centroid_speed"]
    ylabels = ["D_adj", "RAFT slot-flow", "centroid speed"]
    for ax, name, ylabel in zip(axes, names, ylabels):
        values = curves[name]
        for slot_id in range(values.shape[1]):
            alpha = 1.0 if slot_id in active_slots else 0.28
            linewidth = 1.7 if slot_id in active_slots else 0.8
            ax.plot(values[:, slot_id], label=f"S{slot_id}", color=PALETTE[slot_id % len(PALETTE)] / 255.0, alpha=alpha, linewidth=linewidth)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].set_title(title, fontsize=10)
    axes[-1].set_xlabel("adjacent transition t -> t+1")
    axes[0].legend(ncol=7, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pair_matrices(path: Path, d_pair_by_slot: np.ndarray, active_slots: list[int], title: str) -> None:
    show_slots = active_slots[:4] if active_slots else list(range(min(4, d_pair_by_slot.shape[0])))
    cols = len(show_slots)
    fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 3.6), dpi=150, squeeze=False)
    for ax, slot_id in zip(axes[0], show_slots):
        image = ax.imshow(d_pair_by_slot[slot_id], cmap="magma", aspect="auto")
        ax.set_title(f"S{slot_id}", fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("frame")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cosine_matrix(flat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    flat = flat / np.maximum(norm, 1e-12)
    return flat @ flat.T


def corr_matrix(values: np.ndarray, method: str = "spearman") -> np.ndarray:
    n = values.shape[1]
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if method == "spearman":
                out[i, j] = base.spearman(values[:, i], values[:, j])
            else:
                out[i, j] = base.pearson(values[:, i], values[:, j])
    return out


def pair_vector_corr(pair_by_slot: np.ndarray) -> np.ndarray:
    n = pair_by_slot.shape[0]
    triu = np.triu_indices(pair_by_slot.shape[1], k=1)
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            out[i, j] = base.spearman(pair_by_slot[i][triu], pair_by_slot[j][triu])
    return out


def attention_centroids(attention: np.ndarray) -> np.ndarray:
    return base.attention_centroids(attention)


def d_pair_by_slot(slots: np.ndarray) -> np.ndarray:
    normalized = slots / np.maximum(np.linalg.norm(slots, axis=-1, keepdims=True), 1e-12)
    sims = np.einsum("tsd,usd->stu", normalized, normalized)
    return 1.0 - sims


def slot_flow_from_attention(flow: np.ndarray | None, attention: np.ndarray) -> np.ndarray:
    if flow is None:
        return np.zeros((attention.shape[0] - 1, attention.shape[1]), dtype=np.float32)
    flow_mag = np.linalg.norm(flow, axis=-1)
    flow_low = base.downsample_flow_mag(flow_mag, tuple(attention.shape[-2:]))
    attn = attention[:-1] / np.maximum(attention[:-1].sum(axis=(2, 3), keepdims=True), 1e-12)
    return (attn * flow_low[:, None]).sum(axis=(2, 3))


def centroid_speed_from_attention(attention: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids = attention_centroids(attention)
    speed = np.linalg.norm(centroids[1:] - centroids[:-1], axis=-1)
    return centroids, speed


def mean_centroid_distance(centroids: np.ndarray) -> np.ndarray:
    num_slots = centroids.shape[1]
    out = np.zeros((num_slots, num_slots), dtype=np.float64)
    for i in range(num_slots):
        for j in range(num_slots):
            out[i, j] = float(np.linalg.norm(centroids[:, i] - centroids[:, j], axis=-1).mean())
    return out


def active_slot_summary(
    labels_low: np.ndarray,
    d_adj: np.ndarray,
    slot_flow: np.ndarray,
    centroids: np.ndarray,
    centroid_speed: np.ndarray,
    num_slots: int,
) -> list[dict[str, Any]]:
    total = float(labels_low.shape[1] * labels_low.shape[2])
    rows = []
    for slot_id in range(num_slots):
        area = (labels_low == slot_id).mean(axis=(1, 2))
        active_frames = float(np.mean(area > 0.01))
        mean_area = float(area.mean())
        mean_flow = float(slot_flow[:, slot_id].mean()) if len(slot_flow) else 0.0
        mean_d = float(d_adj[:, slot_id].mean()) if len(d_adj) else 0.0
        std_d = float(d_adj[:, slot_id].std()) if len(d_adj) else 0.0
        travel = float(centroid_speed[:, slot_id].sum()) if len(centroid_speed) else 0.0
        object_score = active_frames * (mean_flow + 0.15 * travel / max(1, len(centroid_speed))) * min(1.0, mean_area / 0.08)
        rows.append(
            {
                "slot": int(slot_id),
                "mean_area": mean_area,
                "active_frames": active_frames,
                "mean_slot_flow": mean_flow,
                "mean_d_adj": mean_d,
                "std_d_adj": std_d,
                "centroid_travel": travel,
                "object_score": float(object_score),
            }
        )
    return rows


def select_object_slots(summary: list[dict[str, Any]]) -> list[int]:
    candidates = [
        row
        for row in summary
        if row["active_frames"] >= 0.35 and 0.006 <= row["mean_area"] <= 0.72
    ]
    candidates = sorted(candidates, key=lambda row: row["object_score"], reverse=True)
    return [int(row["slot"]) for row in candidates[:4]]


def verdict_for_slots(
    selected: list[int],
    residual_track_cos: np.ndarray,
    d_adj_corr: np.ndarray,
    d_pair_corr: np.ndarray,
    centroid_distance: np.ndarray,
) -> dict[str, Any]:
    if len(selected) < 2:
        return {
            "level": "weak",
            "text": "少于两个稳定活动 slot，不能支持两个物体被分开。",
            "top_pair": [],
        }
    a, b = selected[0], selected[1]
    metrics = {
        "residual_track_cos": float(residual_track_cos[a, b]),
        "d_adj_spearman": float(d_adj_corr[a, b]),
        "d_pair_spearman": float(d_pair_corr[a, b]),
        "centroid_distance": float(centroid_distance[a, b]),
    }
    feature_distinct = metrics["residual_track_cos"] < 0.88
    motion_distinct = metrics["d_adj_spearman"] < 0.82 and metrics["d_pair_spearman"] < 0.88
    spatial_distinct = metrics["centroid_distance"] > 1.5
    if feature_distinct and motion_distinct and spatial_distinct:
        level = "strong"
        text = "top-2 slot 在空间、feature residual、D_adj/D_pair 上都有明显差异，支持两个物体/运动被区分。"
    elif spatial_distinct and (feature_distinct or motion_distinct):
        level = "partial"
        text = "top-2 slot 空间上可分，但 feature 或运动曲线仍偏相似，存在部分雷同/共享动态风险。"
    else:
        level = "merge-risk"
        text = "top-2 slot 的空间或运动签名相似度过高，存在 merge/冗余风险。"
    return {"level": level, "text": text, "top_pair": [int(a), int(b)], "metrics": metrics}


def analyze_slots(slots: np.ndarray, attention: np.ndarray, flow: np.ndarray | None, labels_low: np.ndarray) -> dict[str, Any]:
    num_slots = slots.shape[1]
    d_adj = np.linalg.norm(slots[1:] - slots[:-1], axis=-1) / math.sqrt(slots.shape[-1])
    slot_residual = slots - slots.mean(axis=0, keepdims=True)
    raw_track_cos = cosine_matrix(slots.transpose(1, 0, 2).reshape(num_slots, -1))
    residual_track_cos = cosine_matrix(slot_residual.transpose(1, 0, 2).reshape(num_slots, -1))
    d_adj_corr = corr_matrix(d_adj, "spearman")
    pair_by_slot = d_pair_by_slot(slots)
    pair_corr = pair_vector_corr(pair_by_slot)
    slot_flow = slot_flow_from_attention(flow, attention)
    slot_flow_corr = corr_matrix(slot_flow, "spearman") if len(slot_flow) else np.eye(num_slots)
    centroids, centroid_speed = centroid_speed_from_attention(attention)
    centroid_dist = mean_centroid_distance(centroids)
    summary = active_slot_summary(labels_low, d_adj, slot_flow, centroids, centroid_speed, num_slots)
    selected = select_object_slots(summary)
    verdict = verdict_for_slots(selected, residual_track_cos, d_adj_corr, pair_corr, centroid_dist)
    return {
        "d_adj": d_adj,
        "slot_flow": slot_flow,
        "centroid_speed": centroid_speed,
        "pair_by_slot": pair_by_slot,
        "raw_track_cos": raw_track_cos,
        "residual_track_cos": residual_track_cos,
        "d_adj_corr": d_adj_corr,
        "d_pair_corr": pair_corr,
        "slot_flow_corr": slot_flow_corr,
        "centroid_distance": centroid_dist,
        "slot_summary": summary,
        "selected_slots": selected,
        "verdict": verdict,
    }


def render_case_weight(
    case_dir: Path,
    model_name: str,
    rgb: np.ndarray,
    slots: np.ndarray,
    attention: np.ndarray,
    flow: np.ndarray | None,
) -> dict[str, Any]:
    model_dir = case_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    labels_low, labels = attention_to_hard_labels(attention, rgb.shape[1])
    analysis = analyze_slots(slots, attention, flow, labels_low)

    all_overlay = overlay_all_slots(rgb, labels)
    slot_grid = overlay_slot_grid(rgb, labels, slots.shape[1])
    base.write_video(model_dir / "all_slot_overlay.mp4", all_overlay, fps=8.0)
    base.write_video(model_dir / "per_slot_grid_overlay.mp4", slot_grid, fps=8.0)

    slot_labels = [f"S{i}" for i in range(slots.shape[1])]
    plot_slot_curves(
        model_dir / "slot_dynamics_curves.png",
        {
            "d_adj": analysis["d_adj"],
            "slot_flow": analysis["slot_flow"],
            "centroid_speed": analysis["centroid_speed"],
        },
        f"{model_name} slot dynamics",
        analysis["selected_slots"],
    )
    plot_pair_matrices(model_dir / "d_pair_matrices.png", analysis["pair_by_slot"], analysis["selected_slots"], f"{model_name} D_pair")
    plot_matrix(model_dir / "residual_track_cos.png", analysis["residual_track_cos"], "slot residual feature-track cosine", slot_labels)
    plot_matrix(model_dir / "d_adj_corr.png", analysis["d_adj_corr"], "slot D_adj Spearman", slot_labels)
    plot_matrix(model_dir / "d_pair_corr.png", analysis["d_pair_corr"], "slot D_pair Spearman", slot_labels)
    plot_matrix(model_dir / "centroid_distance.png", analysis["centroid_distance"], "mean centroid distance", slot_labels, cmap="viridis", vmin=None, vmax=None)

    compact = {
        "model_name": model_name,
        "selected_slots": analysis["selected_slots"],
        "verdict": analysis["verdict"],
        "slot_summary": analysis["slot_summary"],
        "assets": {
            "all_slot_overlay": f"{model_name}/all_slot_overlay.mp4",
            "per_slot_grid_overlay": f"{model_name}/per_slot_grid_overlay.mp4",
            "slot_dynamics_curves": f"{model_name}/slot_dynamics_curves.png",
            "d_pair_matrices": f"{model_name}/d_pair_matrices.png",
            "residual_track_cos": f"{model_name}/residual_track_cos.png",
            "d_adj_corr": f"{model_name}/d_adj_corr.png",
            "d_pair_corr": f"{model_name}/d_pair_corr.png",
            "centroid_distance": f"{model_name}/centroid_distance.png",
        },
        "matrices": {
            "residual_track_cos": analysis["residual_track_cos"].tolist(),
            "d_adj_corr": analysis["d_adj_corr"].tolist(),
            "d_pair_corr": analysis["d_pair_corr"].tolist(),
            "slot_flow_corr": analysis["slot_flow_corr"].tolist(),
            "centroid_distance": analysis["centroid_distance"].tolist(),
        },
    }
    np.savez_compressed(
        model_dir / "slot_separation_arrays.npz",
        slots=slots.astype(np.float16),
        attention=attention.astype(np.float16),
        d_adj=analysis["d_adj"].astype(np.float32),
        slot_flow=analysis["slot_flow"].astype(np.float32),
        d_pair_by_slot=analysis["pair_by_slot"].astype(np.float32),
        labels_low=labels_low.astype(np.uint8),
    )
    (model_dir / "summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return compact


def build_html(output_dir: Path, metadata: dict[str, Any]) -> None:
    case_sections = []
    for case in metadata["cases"]:
        cards = []
        for model in case["models"]:
            rows = []
            for row in model["slot_summary"]:
                cls = "selected" if int(row["slot"]) in model["selected_slots"] else ""
                rows.append(
                    f"<tr class='{cls}'><td>S{row['slot']}</td><td>{row['mean_area']:.3f}</td><td>{row['active_frames']:.2f}</td>"
                    f"<td>{row['mean_slot_flow']:.3f}</td><td>{row['mean_d_adj']:.3f}</td><td>{row['std_d_adj']:.3f}</td>"
                    f"<td>{row['centroid_travel']:.2f}</td><td>{row['object_score']:.3f}</td></tr>"
                )
            verdict = model["verdict"]
            metrics = verdict.get("metrics", {})
            metrics_text = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            assets = model["assets"]
            cards.append(
                f"""
                <article class="model {html.escape(verdict['level'])}">
                  <h3>{html.escape(model['model_name'])} | {html.escape(verdict['level'])}</h3>
                  <p class="small">{html.escape(verdict['text'])} top_pair={html.escape(str(verdict.get('top_pair', [])))} {html.escape(metrics_text)}</p>
                  <div class="videos">
                    <figure><video src="{case['case_id']}/{assets['all_slot_overlay']}" controls muted preload="metadata"></video><figcaption>all-slot hard assignment overlay</figcaption></figure>
                    <figure><video src="{case['case_id']}/{assets['per_slot_grid_overlay']}" controls muted preload="metadata"></video><figcaption>per-slot overlay grid</figcaption></figure>
                  </div>
                  <div class="plots">
                    <figure><img src="{case['case_id']}/{assets['slot_dynamics_curves']}" loading="lazy"><figcaption>D_adj / RAFT slot-flow / centroid speed</figcaption></figure>
                    <figure><img src="{case['case_id']}/{assets['d_pair_matrices']}" loading="lazy"><figcaption>D_pair matrices for selected slots</figcaption></figure>
                    <figure><img src="{case['case_id']}/{assets['residual_track_cos']}" loading="lazy"><figcaption>feature residual track similarity</figcaption></figure>
                    <figure><img src="{case['case_id']}/{assets['d_adj_corr']}" loading="lazy"><figcaption>D_adj similarity</figcaption></figure>
                    <figure><img src="{case['case_id']}/{assets['d_pair_corr']}" loading="lazy"><figcaption>D_pair similarity</figcaption></figure>
                    <figure><img src="{case['case_id']}/{assets['centroid_distance']}" loading="lazy"><figcaption>slot centroid distance</figcaption></figure>
                  </div>
                  <details><summary>slot table</summary>
                    <table><thead><tr><th>slot</th><th>area</th><th>active</th><th>flow</th><th>D mean</th><th>D std</th><th>travel</th><th>score</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
                  </details>
                </article>
                """
            )
        case_sections.append(
            f"""
            <section class="case">
              <h2>{html.escape(case['case_id'])}</h2>
              <p class="small"><b>source:</b> {html.escape(case['source_video'])}</p>
              <p class="small"><b>caption:</b> {html.escape(case.get('caption', ''))}</p>
              <div class="source"><figure><video src="{case['case_id']}/xssc_input_49f.mp4" controls muted preload="metadata"></video><figcaption>xSSC input 49f</figcaption></figure></div>
              <div class="models">{''.join(cards)}</div>
            </section>
            """
        )
    summary_rows = []
    for row in metadata["verdict_summary"]:
        summary_rows.append(
            f"<tr><td>{html.escape(row['case_id'])}</td><td>{html.escape(row['model_name'])}</td><td>{html.escape(row['level'])}</td>"
            f"<td>{html.escape(str(row.get('top_pair', [])))}</td><td>{row.get('residual_track_cos', float('nan')):.3f}</td>"
            f"<td>{row.get('d_adj_spearman', float('nan')):.3f}</td><td>{row.get('d_pair_spearman', float('nan')):.3f}</td>"
            f"<td>{row.get('centroid_distance', float('nan')):.3f}</td></tr>"
        )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC Slot Object/Motion Separation</title>
  <style>
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#101214; color:#eef2f7; font:13px system-ui,sans-serif; letter-spacing:0; }}
    header {{ position:sticky; top:0; z-index:10; padding:12px 16px; background:#15191d; border-bottom:1px solid #303942; }}
    main {{ max-width:2200px; margin:0 auto; padding:16px; }}
    h1 {{ margin:0 0 6px; font-size:20px; }}
    h2 {{ margin:0 0 8px; font-size:17px; }}
    h3 {{ margin:0 0 6px; font-size:14px; }}
    .small {{ color:#bdc7d1; overflow-wrap:anywhere; }}
    .case {{ padding:18px 0 30px; border-top:1px solid #303942; }}
    .source {{ max-width:420px; margin-bottom:12px; }}
    .models {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(620px,1fr)); gap:12px; }}
    .model {{ border:1px solid #333b44; background:#14191e; padding:10px; border-radius:8px; }}
    .model.strong {{ border-color:#42d392; }}
    .model.partial {{ border-color:#f7c948; }}
    .model.merge-risk,.model.weak {{ border-color:#ff6b6b; }}
    .videos,.plots {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    figure {{ margin:0; min-width:0; }}
    img,video {{ display:block; width:100%; background:#000; border:1px solid #303942; }}
    figcaption {{ color:#aeb8c2; font-size:11px; padding:4px 1px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
    th,td {{ border:1px solid #303942; padding:5px 7px; text-align:left; }}
    th {{ background:#192027; }}
    td {{ background:#12171c; color:#cbd5df; }}
    tr.selected td {{ background:#203326; }}
    @media(max-width:900px) {{ .models,.videos,.plots {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>xSSC slot object/motion separation</h1>
    <div class="small">Question: if two objects differ in motion and attributes, do xSSC slots avoid merging and do slot-level D_adj/D_pair avoid becoming identical?</div>
  </header>
  <main>
    <p class="small">No GT instance masks are used here, so this is evidence rather than proof. A strong case needs two stable active slots with spatially separated centroids, low residual feature-track similarity, and low D_adj/D_pair similarity.</p>
    <h2>Verdict Summary</h2>
    <table><thead><tr><th>case</th><th>weight</th><th>level</th><th>top pair</th><th>feature cos</th><th>D_adj</th><th>D_pair</th><th>centroid dist</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
    {''.join(case_sections)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    json_paths = args.json if args.json else DEFAULT_JSONS
    cases, duplicates = read_cases(json_paths, args.max_cases)
    checkpoints = sorted(args.official_root.expanduser().resolve().glob("*.pth"))
    if len(checkpoints) != 3:
        raise RuntimeError(f"Expected 3 official xSSC weights, found {len(checkpoints)} under {args.official_root}")

    models = []
    for checkpoint in checkpoints:
        model, _ = base.build_official_model(checkpoint, device)
        models.append((f"official_{checkpoint.stem}", checkpoint, model))

    raft = None if args.skip_raft else base.build_raft(device, args.raft_iters)
    out_cases = []
    verdict_summary = []
    for case_position, case in enumerate(cases, start=1):
        case_dir = output_dir / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        video_tensor, _, frame_indices = read_source_video(
            Path(case["source_video"]),
            args.num_frames,
            args.height,
            args.width,
        )
        normalized, rgb = base.preprocess_video_for_xssc(video_tensor, args.xssc_input_size)
        base.write_video(case_dir / "xssc_input_49f.mp4", rgb, fps=8.0)
        flow = None if raft is None else base.compute_raft_flow(raft, rgb, device, args.raft_iters)
        model_records = []
        print(f"[case] {case_position}/{len(cases)} {case['case_id']} frames={len(rgb)} flow={'none' if flow is None else flow.shape}", flush=True)
        for model_position, (model_name, checkpoint, model) in enumerate(models, start=1):
            seed = int(args.seed) + case_position * 1000 + model_position * 100 + int(checkpoint.stem.split("-")[0])
            slots, attention = base.extract_official_slots(
                model,
                normalized,
                device,
                seed=seed,
                batch_size=args.xssc_batch_size,
            )
            record = render_case_weight(
                case_dir,
                model_name,
                rgb,
                slots.numpy().astype(np.float32),
                attention.numpy().astype(np.float32),
                flow,
            )
            record["checkpoint"] = str(checkpoint)
            model_records.append(record)
            verdict = record["verdict"]
            metrics = verdict.get("metrics", {})
            verdict_summary.append(
                {
                    "case_id": case["case_id"],
                    "model_name": model_name,
                    "level": verdict["level"],
                    "top_pair": verdict.get("top_pair", []),
                    "residual_track_cos": float(metrics.get("residual_track_cos", float("nan"))),
                    "d_adj_spearman": float(metrics.get("d_adj_spearman", float("nan"))),
                    "d_pair_spearman": float(metrics.get("d_pair_spearman", float("nan"))),
                    "centroid_distance": float(metrics.get("centroid_distance", float("nan"))),
                    "text": verdict["text"],
                }
            )
            print(
                f"[model] {model_name} {case['case_id']} "
                f"level={verdict['level']} pair={verdict.get('top_pair', [])} "
                f"metrics={metrics}",
                flush=True,
            )
        out_cases.append(
            {
                **case,
                "frame_indices": [int(v) for v in frame_indices.tolist()],
                "models": model_records,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "cases": out_cases,
        "duplicates_skipped": duplicates,
        "verdict_summary": verdict_summary,
        "checkpoints": [str(path) for path in checkpoints],
        "note": "This experiment has no GT instance masks. Strong/partial/merge-risk are evidence labels based on slot attention, feature residual track similarity, D_adj/D_pair similarity, and centroid separation.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    build_html(output_dir, metadata)
    print(f"viewer={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
