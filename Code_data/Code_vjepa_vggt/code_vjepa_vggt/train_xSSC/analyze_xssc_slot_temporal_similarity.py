#!/usr/bin/env python3
"""Analyze temporal similarity of xSSC slots for one video case."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as train
from code_vjepa_vggt.train_xSSC.visualize_xssc_slot_attention import (
    _cover_crop_to_tensor,
    _extract_slots_and_attention,
    _resolve_video_path,
)
from code_vjepa_vggt.utils.video_io import read_video_prefix


def _cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    return x @ y.transpose(-1, -2)


def _offdiag_values(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def _plot_heatmap(matrix: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=160)
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("frame")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_adjacent_matrix(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title("Mean adjacent-frame slot cosine")
    ax.set_xlabel("slot at t+1")
    ax.set_ylabel("slot at t")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels([f"s{i}" for i in range(matrix.shape[1])])
    ax.set_yticklabels([f"s{i}" for i in range(matrix.shape[0])])
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_pca(slots: np.ndarray, output_path: Path) -> None:
    time_steps, num_slots, dim = slots.shape
    x = slots.reshape(time_steps * num_slots, dim).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=160)
    colors = plt.cm.tab10(np.linspace(0, 1, num_slots))
    for slot_id in range(num_slots):
        idx = np.arange(slot_id, time_steps * num_slots, num_slots)
        ax.plot(coords[idx, 0], coords[idx, 1], "-o", color=colors[slot_id], label=f"slot{slot_id:02d}")
        for frame_id, point_idx in enumerate(idx):
            ax.text(coords[point_idx, 0], coords[point_idx, 1], str(frame_id), fontsize=7)
    ax.set_title("PCA of xSSC slots across context frames")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_slot_bars(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    slot_ids = [int(row["slot"]) for row in rows]
    emb = [float(row["embedding_adjacent_mean"]) for row in rows]
    attn = [float(row["attention_adjacent_mean"]) for row in rows]
    x = np.arange(len(slot_ids))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ax.bar(x - width / 2, emb, width, label="slot embedding")
    ax.bar(x + width / 2, attn, width, label="attention map")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{i}" for i in slot_ids])
    ax.set_ylabel("adjacent-frame cosine")
    ax.set_title("Per-slot adjacent-frame stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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
    slots_tsc = slots[0].float().cpu()
    attention_tshw = attention[0].float().cpu()
    time_steps = int(slots_tsc.shape[0])
    num_slots = int(slots_tsc.shape[1])

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slot_rows: list[dict[str, float | int | str]] = []
    slot_time_similarity = []
    adjacent_same_values = []
    same_slot_top1 = []
    adjacent_matrices = []

    for slot_id in range(num_slots):
        sim = _cosine_similarity_matrix(slots_tsc[:, slot_id], slots_tsc[:, slot_id]).numpy()
        slot_time_similarity.append(sim)
        adjacent = np.diag(sim, k=1)
        offdiag = _offdiag_values(sim)
        adjacent_same_values.extend(float(v) for v in adjacent)
        attention_flat = attention_tshw[:, slot_id].flatten(1)
        attention_sim = _cosine_similarity_matrix(attention_flat, attention_flat).numpy()
        attention_adjacent = np.diag(attention_sim, k=1)
        attention_offdiag = _offdiag_values(attention_sim)
        row = {
            "slot": int(slot_id),
            "embedding_adjacent_mean": float(adjacent.mean()),
            "embedding_adjacent_min": float(adjacent.min()),
            "embedding_adjacent_max": float(adjacent.max()),
            "embedding_all_pairs_offdiag_mean": float(offdiag.mean()),
            "embedding_all_pairs_offdiag_min": float(offdiag.min()),
            "embedding_all_pairs_offdiag_max": float(offdiag.max()),
            "attention_adjacent_mean": float(attention_adjacent.mean()),
            "attention_adjacent_min": float(attention_adjacent.min()),
            "attention_adjacent_max": float(attention_adjacent.max()),
            "attention_all_pairs_offdiag_mean": float(attention_offdiag.mean()),
            "attention_all_pairs_offdiag_min": float(attention_offdiag.min()),
            "attention_all_pairs_offdiag_max": float(attention_offdiag.max()),
        }
        slot_rows.append(row)
        _plot_heatmap(sim, output_dir / f"slot{slot_id:02d}_time_cosine_heatmap.png", f"slot{slot_id:02d} time cosine")
        _plot_heatmap(
            attention_sim,
            output_dir / f"slot{slot_id:02d}_attention_time_cosine_heatmap.png",
            f"slot{slot_id:02d} attention time cosine",
        )

    for frame_id in range(time_steps - 1):
        mat = _cosine_similarity_matrix(slots_tsc[frame_id], slots_tsc[frame_id + 1]).numpy()
        adjacent_matrices.append(mat)
        for slot_id in range(num_slots):
            top1 = int(mat[slot_id].argmax())
            same_slot_top1.append(top1 == slot_id)
    adjacent_mean_matrix = np.stack(adjacent_matrices, axis=0).mean(axis=0)
    _plot_adjacent_matrix(adjacent_mean_matrix, output_dir / "adjacent_slot_to_slot_mean_cosine.png")
    _plot_pca(slots_tsc.numpy(), output_dir / "slot_pca_temporal_tracks.png")
    _plot_slot_bars(slot_rows, output_dir / "slot_adjacent_stability_bars.png")

    cross_same_frame_values = []
    for frame_id in range(time_steps):
        mat = _cosine_similarity_matrix(slots_tsc[frame_id], slots_tsc[frame_id]).numpy()
        cross_same_frame_values.extend(float(v) for v in _offdiag_values(mat))

    csv_path = output_dir / "slot_temporal_similarity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(slot_rows)

    np.save(output_dir / "slots_tsc.npy", slots_tsc.numpy())
    np.save(output_dir / "slot_time_cosine.npy", np.stack(slot_time_similarity, axis=0))
    np.save(output_dir / "adjacent_slot_to_slot_cosine_mean.npy", adjacent_mean_matrix)

    summary = {
        "case_json": str(case_json),
        "source_video": str(video_path),
        "frame_indices": [int(v) for v in frame_indices.tolist()],
        "preprocess": preprocess_metadata,
        "slots_shape": list(slots_tsc.shape),
        "attention_shape": list(attention_tshw.shape),
        "slot_temporal_similarity_csv": str(csv_path),
        "embedding_same_slot_adjacent_mean": float(np.mean(adjacent_same_values)),
        "embedding_same_slot_adjacent_min": float(np.min(adjacent_same_values)),
        "embedding_same_slot_adjacent_max": float(np.max(adjacent_same_values)),
        "embedding_same_slot_all_pairs_offdiag_mean": float(
            np.mean([row["embedding_all_pairs_offdiag_mean"] for row in slot_rows])
        ),
        "attention_same_slot_adjacent_mean": float(
            np.mean([row["attention_adjacent_mean"] for row in slot_rows])
        ),
        "attention_same_slot_adjacent_min": float(
            np.min([row["attention_adjacent_min"] for row in slot_rows])
        ),
        "attention_same_slot_all_pairs_offdiag_mean": float(
            np.mean([row["attention_all_pairs_offdiag_mean"] for row in slot_rows])
        ),
        "cross_slot_same_frame_mean": float(np.mean(cross_same_frame_values)),
        "cross_slot_same_frame_max": float(np.max(cross_same_frame_values)),
        "same_slot_adjacent_top1_rate": float(np.mean(same_slot_top1)),
        "slot_rows": slot_rows,
        "outputs": {
            "adjacent_matrix_png": str(output_dir / "adjacent_slot_to_slot_mean_cosine.png"),
            "pca_png": str(output_dir / "slot_pca_temporal_tracks.png"),
            "stability_bar_png": str(output_dir / "slot_adjacent_stability_bars.png"),
        },
    }
    summary_path = output_dir / "slot_temporal_similarity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
