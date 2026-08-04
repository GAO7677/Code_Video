#!/usr/bin/env python3
"""Render all 720 ranked three-model attention strips."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


METRIC_ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_all720_neighbor_diagonal_5case"
)
HEAT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_all720_neighbor_diagonal_heatmaps_case001"
)
CASE = "case_001_ball_roll"
MODELS = (("gt", "GT teacher-forced"), ("lora", "LoRA"), ("baseline", "Wan2.2 Baseline"))


def load_block(model: str, block: int) -> tuple[list[int], np.ndarray]:
    path = (
        HEAT_ROOT
        / model
        / "cases"
        / CASE
        / "all_token_qk"
        / f"block{block:02d}_selected_qk.npz"
    )
    with np.load(path) as payload:
        steps = payload["steps_zero_based"].tolist()
        step_index = steps.index(39)
        heads = payload["selected_heads"].tolist()
        matrices = payload["softmax_attention_mass"][step_index].astype(np.float32)
    return heads, matrices


def add_neighbor_boxes(axis, size: int) -> None:
    boundaries = [round(index * size / 7) for index in range(8)]
    for query_frame in range(7):
        y0, y1 = boundaries[query_frame], boundaries[query_frame + 1]
        for key_frame in range(7):
            x0, x1 = boundaries[key_frame], boundaries[key_frame + 1]
            axis.plot(
                [x0 - 0.5, x1 - 0.5],
                [y0 - 0.5, y1 - 0.5],
                color="#f0c96a",
                linewidth=0.28,
                alpha=0.55,
            )
        for key_frame in range(max(0, query_frame - 1), min(7, query_frame + 2)):
            x0, x1 = boundaries[key_frame], boundaries[key_frame + 1]
            y0, y1 = boundaries[query_frame], boundaries[query_frame + 1]
            axis.add_patch(
                Rectangle(
                    (x0 - 0.5, y0 - 0.5),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="#64d8cb",
                    linewidth=0.45,
                    alpha=0.65,
                )
            )
    for boundary in boundaries[1:-1]:
        axis.axhline(boundary - 0.5, color="white", lw=0.3, alpha=0.45)
        axis.axvline(boundary - 0.5, color="white", lw=0.3, alpha=0.45)


def render(row: dict[str, str], matrices_by_model: dict[str, tuple[list[int], np.ndarray]]) -> None:
    block, head = int(row["block"]), int(row["head"])
    matrices = []
    for model, _label in MODELS:
        heads, block_matrices = matrices_by_model[model]
        matrices.append(block_matrices[heads.index(head)])
    logs = [np.log10(np.maximum(matrix, 1e-8)) for matrix in matrices]
    merged = np.concatenate([matrix.ravel() for matrix in logs])
    vmin, vmax = np.quantile(merged, (0.01, 0.997))
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.15), constrained_layout=True)
    image = None
    for axis, matrix, (_model, label) in zip(axes, logs, MODELS):
        image = axis.imshow(
            matrix,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        add_neighbor_boxes(axis, matrix.shape[0])
        centers = [(index + 0.5) * matrix.shape[0] / 7 for index in range(7)]
        axis.set_xticks(centers, [f"F{i}" for i in range(7)], fontsize=6)
        axis.set_yticks(centers, [f"F{i}" for i in range(7)], fontsize=6)
        axis.set_xlabel("key token / frame", fontsize=7)
        axis.set_ylabel("query token / frame", fontsize=7)
        axis.set_title(label, fontsize=10, fontweight="bold")
    figure.colorbar(image, ax=axes, shrink=0.78, label="log10 attention mass")
    figure.suptitle(
        (
            f"Strict rank {int(row['strict_rank']):03d} | L{block:02d}/H{head:02d} | "
            f"score={float(row['neighbor3_allblock_diagonal_score']):.4f} | "
            f"purity={float(row['allblock_diagonal_purity']):.4f} | "
            f"min-purity={float(row['allblock_min_diagonal_purity']):.4f}"
        ),
        fontsize=11,
        fontweight="bold",
    )
    output = HEAT_ROOT / "web" / f"block{block:02d}_head{head:02d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=125, facecolor="#fffdf7")
    plt.close(figure)


def main() -> None:
    summary = METRIC_ROOT / "all720_neighbor_diagonal_summary.csv"
    with summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_block: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_block.setdefault(int(row["block"]), []).append(row)
    completed = 0
    for block in sorted(by_block):
        matrices_by_model = {model: load_block(model, block) for model, _ in MODELS}
        for row in by_block[block]:
            render(row, matrices_by_model)
            completed += 1
            print(
                f"[{completed:03d}/720] L{int(row['block']):02d}/H{int(row['head']):02d}",
                flush=True,
            )
    (HEAT_ROOT / "render.complete").write_text("720\n", encoding="utf-8")


if __name__ == "__main__":
    main()
