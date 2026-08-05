#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40")
RANKING = Path("/data/gaoya/agent-data/outputs/neighbor_diagonal_ranking_snapshot/all720-neighbor-diagonal.csv")
SEEDS = (47326, 90094, 32466, 35075, 21890, 49530)
TOP_NS = (30, 50, 100)
REGIONS = ("object_A", "object_B")
EPS = 1e-12


def find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise KeyError(f"None of {names} exists in columns {list(frame.columns)}")


def load_ranking() -> tuple[list[tuple[int, int]], np.ndarray]:
    frame = pd.read_csv(RANKING)
    block_col = find_column(frame, ("block", "block_id", "layer"))
    head_col = find_column(frame, ("head", "head_id"))
    score_col = find_column(frame, ("lora_pck32",))
    frame = frame.sort_values(score_col, ascending=False).head(100).reset_index(drop=True)
    ids = [(int(row[block_col]), int(row[head_col])) for _, row in frame.iterrows()]
    return ids, frame[score_col].to_numpy(dtype=np.float32)


def load_schedule(seed: int, steps: int, ranked_ids: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    capture_dir = ROOT / "seeds" / f"seed_{seed:06d}" / f"steps{steps}" / "captures"
    files = sorted(capture_dir.glob("step_*.npz"))
    records: dict[tuple[int, str], np.ndarray] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as item:
            step = int(item["step"].item())
            branch = str(item["branch"].item())
            blocks = item["blocks"].astype(np.int64)
            heads = item["heads"].astype(np.int64)
            names = [str(value) for value in item["region_names"].tolist()]
            attention = item["attention"].astype(np.float32, copy=False)
        capture_index = {(int(block), int(head)): index for index, (block, head) in enumerate(zip(blocks, heads))}
        missing = [head_id for head_id in ranked_ids if head_id not in capture_index]
        if missing:
            raise RuntimeError(f"{path}: missing ranked heads {missing[:5]} ({len(missing)} total)")
        missing_regions = [name for name in REGIONS if name not in names]
        if missing_regions:
            raise RuntimeError(f"{path}: missing regions {missing_regions}")
        head_order = [capture_index[head_id] for head_id in ranked_ids]
        region_order = [names.index(name) for name in REGIONS]
        ordered = attention[head_order][:, region_order]
        records[(step, branch)] = ordered.reshape(100, len(REGIONS), -1)

    branches = sorted({branch for _, branch in records})
    if not branches:
        raise RuntimeError(f"No captures found in {capture_dir}")
    result: dict[str, np.ndarray] = {}
    for branch in branches:
        missing_steps = [step for step in range(steps) if (step, branch) not in records]
        if missing_steps:
            raise RuntimeError(f"{capture_dir}: branch={branch} missing steps {missing_steps}")
        result[branch] = np.stack([records[(step, branch)] for step in range(steps)], axis=0)
    return result


def normalize(array: np.ndarray) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), EPS)


def save_heatmap(matrix: np.ndarray, title: str, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 4.8))
    image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_xlabel("40-step denoising step")
    axis.set_ylabel("10-step denoising step")
    axis.set_xticks(np.arange(0, 40, 2))
    axis.set_yticks(np.arange(10))
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_ids, ranked_scores = load_ranking()

    primary_samples = {n: [] for n in TOP_NS}
    auxiliary_samples = {n: [] for n in TOP_NS}
    head_samples: list[np.ndarray] = []
    sample_labels: list[tuple[int, str, str]] = []
    detailed_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        maps10 = load_schedule(seed, 10, ranked_ids)
        maps40 = load_schedule(seed, 40, ranked_ids)
        branches = sorted(set(maps10) & set(maps40))
        if not branches:
            raise RuntimeError(f"seed={seed}: no shared CFG branches")
        for branch in branches:
            for region_index, region in enumerate(REGIONS):
                ten = maps10[branch][:, :, region_index]
                forty = maps40[branch][:, :, region_index]
                per_head = np.einsum(
                    "thd,shd->tsh",
                    normalize(ten),
                    normalize(forty),
                    optimize=True,
                )
                head_samples.append(per_head)
                sample_labels.append((seed, branch, region))

                for rank_index, ((block, head), pck32) in enumerate(zip(ranked_ids, ranked_scores)):
                    for step10 in range(10):
                        similarities = per_head[step10, :, rank_index]
                        best40 = int(np.argmax(similarities))
                        detailed_rows.append(
                            {
                                "seed": seed,
                                "branch": branch,
                                "object": region,
                                "rank": rank_index + 1,
                                "block": block,
                                "head": head,
                                "lora_pck32": float(pck32),
                                "step10": step10,
                                "best_step40": best40,
                                "cosine": float(similarities[best40]),
                            }
                        )

                for top_n in TOP_NS:
                    primary_samples[top_n].append(per_head[:, :, :top_n].mean(axis=-1))
                    mean10 = normalize(ten[:, :top_n].mean(axis=1))
                    mean40 = normalize(forty[:, :top_n].mean(axis=1))
                    auxiliary_samples[top_n].append(mean10 @ mean40.T)

    per_head_stack = np.stack(head_samples, axis=0)
    per_head_macro = per_head_stack.mean(axis=0)
    matrices: dict[str, np.ndarray] = {
        "per_head_macro": per_head_macro,
        "rank_blocks": np.asarray([item[0] for item in ranked_ids], dtype=np.int16),
        "rank_heads": np.asarray([item[1] for item in ranked_ids], dtype=np.int16),
        "rank_lora_pck32": ranked_scores,
    }
    top_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "seeds": list(SEEDS),
        "branches": sorted({label[1] for label in sample_labels}),
        "objects": list(REGIONS),
        "sample_count": len(sample_labels),
        "similarity": "cosine on raw attention-probability object-query maps",
        "primary": "mean of per-head cosine similarities",
        "auxiliary": "cosine after averaging head maps",
        "topn": {},
    }

    for top_n in TOP_NS:
        primary = np.stack(primary_samples[top_n]).mean(axis=0)
        auxiliary = np.stack(auxiliary_samples[top_n]).mean(axis=0)
        matrices[f"top{top_n}_mean_per_head_cosine"] = primary
        matrices[f"top{top_n}_cosine_of_mean_map"] = auxiliary
        save_heatmap(primary, f"Top{top_n}: mean per-head cosine", output_dir / f"top{top_n}_mean_per_head_cosine.png")
        save_heatmap(auxiliary, f"Top{top_n}: cosine of mean attention map", output_dir / f"top{top_n}_cosine_of_mean_map.png")
        best_steps = primary.argmax(axis=1)
        best_scores = primary[np.arange(10), best_steps]
        summary["topn"][str(top_n)] = {
            "best_step40": best_steps.tolist(),
            "best_cosine": best_scores.tolist(),
        }
        for step10, (step40, score) in enumerate(zip(best_steps, best_scores)):
            top_rows.append(
                {
                    "aggregation": "mean_per_head_cosine",
                    "top_n": top_n,
                    "step10": step10,
                    "best_step40": int(step40),
                    "cosine": float(score),
                }
            )
        auxiliary_best = auxiliary.argmax(axis=1)
        auxiliary_scores = auxiliary[np.arange(10), auxiliary_best]
        for step10, (step40, score) in enumerate(zip(auxiliary_best, auxiliary_scores)):
            top_rows.append(
                {
                    "aggregation": "cosine_of_mean_map",
                    "top_n": top_n,
                    "step10": step10,
                    "best_step40": int(step40),
                    "cosine": float(score),
                }
            )

    aggregate_head_rows: list[dict[str, object]] = []
    for rank_index, ((block, head), pck32) in enumerate(zip(ranked_ids, ranked_scores)):
        for step10 in range(10):
            similarities = per_head_macro[step10, :, rank_index]
            best40 = int(np.argmax(similarities))
            aggregate_head_rows.append(
                {
                    "rank": rank_index + 1,
                    "block": block,
                    "head": head,
                    "lora_pck32": float(pck32),
                    "step10": step10,
                    "best_step40": best40,
                    "cosine": float(similarities[best40]),
                }
            )

    np.savez(output_dir / "similarity_matrices.npz", **matrices)
    pd.DataFrame(top_rows).to_csv(output_dir / "topn_best_matches.csv", index=False)
    pd.DataFrame(aggregate_head_rows).to_csv(output_dir / "per_head_best_matches.csv", index=False)
    pd.DataFrame(detailed_rows).to_csv(output_dir / "per_head_best_matches_by_sample.csv", index=False)

    fig, axis = plt.subplots(figsize=(11, 5.5))
    colors = {30: "#cf4a30", 50: "#297f87", 100: "#d29b2e"}
    for top_n in TOP_NS:
        matrix = matrices[f"top{top_n}_mean_per_head_cosine"]
        best = matrix.argmax(axis=1)
        axis.plot(range(10), best, marker="o", linewidth=2.2, label=f"Top{top_n}", color=colors[top_n])
    axis.plot(range(10), np.arange(10) * 4, linestyle="--", color="#555555", alpha=0.6, label="linear 4x reference")
    axis.set_xticks(range(10))
    axis.set_yticks(range(0, 40, 2))
    axis.set_xlabel("10-step denoising step")
    axis.set_ylabel("best matching 40-step denoising step")
    axis.set_title("10-step to 40-step object-query attention alignment")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "best_match_curves.png", dpi=180)
    plt.close(fig)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# 10-step vs 40-step Object-Query Attention Alignment",
        "",
        f"- Seeds: {', '.join(map(str, SEEDS))}",
        f"- Samples averaged: {len(sample_labels)} = seed x CFG branch x object",
        "- Ranking: fixed LoRA PCK@32 physical block/head ranking",
        "- Primary score: cosine per physical head, then averaged over heads and samples",
        "- Auxiliary score: average TopN attention maps first, then cosine",
        "",
        "## Primary best matching 40-step index",
        "",
        "| 10-step index | Top30 | Top50 | Top100 |",
        "|---:|---:|---:|---:|",
    ]
    for step10 in range(10):
        values = []
        for top_n in TOP_NS:
            matrix = matrices[f"top{top_n}_mean_per_head_cosine"]
            best40 = int(matrix[step10].argmax())
            values.append(f"S{best40:02d} ({matrix[step10, best40]:.4f})")
        lines.append(f"| S{step10:02d} | " + " | ".join(values) + " |")
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
