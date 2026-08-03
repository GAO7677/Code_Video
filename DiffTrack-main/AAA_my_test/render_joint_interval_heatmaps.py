#!/usr/bin/env python3
"""Render three-model S039 attention strips for Joint-bin samples."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    os.environ.get(
        "INTERVAL_HEATMAP_ROOT",
        "/data/gaoya/agent-data/outputs/"
        "three_model_joint_interval_samples_alltoken_qk_case001",
    )
)
CASE = "case_001_ball_roll"
MODELS = (("gt", "GT teacher-forced"), ("lora", "LoRA"), ("baseline", "Wan2.2 Baseline"))


def load_attention(model: str, block: int, head: int) -> np.ndarray:
    path = ROOT / model / "cases" / CASE / "all_token_qk" / f"block{block:02d}_selected_qk.npz"
    with np.load(path) as payload:
        steps = payload["steps_zero_based"].tolist()
        heads = payload["selected_heads"].tolist()
        step_index = steps.index(39)
        head_index = heads.index(head)
        return payload["softmax_attention_mass"][step_index, head_index].astype(np.float32)


def render(row: dict) -> None:
    block, head = int(row["block"]), int(row["head"])
    matrices = [load_attention(model, block, head) for model, _ in MODELS]
    logs = [np.log10(np.maximum(matrix, 1e-8)) for matrix in matrices]
    merged = np.concatenate([matrix.ravel() for matrix in logs])
    vmin, vmax = np.quantile(merged, (0.01, 0.997))
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.15), constrained_layout=True)
    image = None
    for axis, matrix, (_, label) in zip(axes, logs, MODELS):
        image = axis.imshow(matrix, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        boundaries = [round(index * matrix.shape[0] / 7) for index in range(1, 7)]
        centers = [(index + 0.5) * matrix.shape[0] / 7 for index in range(7)]
        for boundary in boundaries:
            axis.axhline(boundary - 0.5, color="white", lw=0.35, alpha=0.55)
            axis.axvline(boundary - 0.5, color="white", lw=0.35, alpha=0.55)
        axis.set_xticks(centers, [f"F{i}" for i in range(7)], fontsize=6)
        axis.set_yticks(centers, [f"F{i}" for i in range(7)], fontsize=6)
        axis.set_xlabel("key token / frame", fontsize=7)
        axis.set_ylabel("query token / frame", fontsize=7)
        axis.set_title(label, fontsize=10, fontweight="bold")
    fig.colorbar(image, ax=axes, shrink=0.78, label="log10 attention mass")
    fig.suptitle(f'L{block:02d} / H{head:02d} · S039 · {CASE}', fontsize=12, fontweight="bold")
    output = ROOT / "web" / f"block{block:02d}_head{head:02d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="#fffdf7")
    plt.close(fig)


def main() -> None:
    manifest = json.loads((ROOT / "selected_heads.json").read_text(encoding="utf-8"))
    for index, row in enumerate(manifest["combinations"], 1):
        render(row)
        print(f'[{index:02d}/{len(manifest["combinations"])}] L{row["block"]:02d}/H{row["head"]:02d}', flush=True)


if __name__ == "__main__":
    main()
