#!/usr/bin/env python3
"""Compute reverse 40-step -> 10-step object-query attention matches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[47326])
    return parser.parse_args()


def normalized_attention(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        attention = data["attention"].astype(np.float32).transpose(1, 0, 2, 3, 4)
        blocks = data["blocks"].astype(np.int64)
        heads = data["heads"].astype(np.int64)
        objects = data["region_names"].astype(str)
    flat = attention.reshape(attention.shape[0], attention.shape[1], -1)
    norms = np.linalg.norm(flat, axis=-1, keepdims=True)
    flat = flat / np.maximum(norms, np.float32(1e-12))
    return flat, blocks, heads, objects


def write_aggregate(root: Path, output: Path) -> None:
    matrix_path = root / "analysis" / "similarity_matrices.npz"
    with np.load(matrix_path, allow_pickle=False) as data:
        macro = data["per_head_macro"].astype(np.float32)
        blocks = data["rank_blocks"].astype(np.int64)
        heads = data["rank_heads"].astype(np.int64)
        pck = data["rank_lora_pck32"].astype(np.float32)
        topn = {
            count: data[f"top{count}_mean_per_head_cosine"].astype(np.float32)
            for count in (30, 50, 100)
        }
    with (output / "per_head_best_matches_reverse.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "block", "head", "lora_pck32", "step40", "best_step10", "cosine"])
        for step40 in range(40):
            for rank in range(len(blocks)):
                scores = macro[:, step40, rank]
                best = int(np.argmax(scores))
                writer.writerow([rank + 1, blocks[rank], heads[rank], pck[rank], step40, best, scores[best]])
    rows = []
    with (output / "topn_best_matches_reverse.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["top_n", "step40", "best_step10", "cosine"])
        for count, matrix in topn.items():
            for step40 in range(40):
                best = int(np.argmax(matrix[:, step40]))
                score = float(matrix[best, step40])
                writer.writerow([count, step40, best, score])
                rows.append((count, step40, best, score))
    lines = [
        "# Reverse 40-step -> 10-step Object-Query Attention Alignment",
        "",
        "For every 40-step index and physical head, choose the most similar 10-step index.",
        "",
        "| 40-step | Top30 | Top50 | Top100 |",
        "|---:|---:|---:|---:|",
    ]
    lookup = {(count, step40): (best, score) for count, step40, best, score in rows}
    for step40 in range(40):
        cells = []
        for count in (30, 50, 100):
            best, score = lookup[count, step40]
            cells.append(f"S{best:02d} ({score:.4f})")
        lines.append(f"| S{step40:02d} | " + " | ".join(cells) + " |")
    (output / "RESULTS_REVERSE.md").write_text("\n".join(lines) + "\n")


def write_samples(root: Path, output: Path, seeds: list[int]) -> None:
    matrix_path = root / "analysis" / "similarity_matrices.npz"
    with np.load(matrix_path, allow_pickle=False) as data:
        rank_pck = data["rank_lora_pck32"].astype(np.float32)
    rows = []
    for seed in seeds:
        seed_root = root / "seeds" / f"seed_{seed:06d}"
        for branch in ("conditional", "unconditional"):
            ten = []
            metadata = None
            for step10 in range(10):
                loaded = normalized_attention(seed_root / "steps10" / "captures" / f"step_{step10:02d}__{branch}.npz")
                ten.append(loaded[0])
                metadata = loaded[1:]
            ten_stack = np.stack(ten, axis=0)
            assert metadata is not None
            blocks, heads, objects = metadata
            for step40 in range(40):
                forty, forty_blocks, forty_heads, forty_objects = normalized_attention(
                    seed_root / "steps40" / "captures" / f"step_{step40:02d}__{branch}.npz"
                )
                if not (
                    np.array_equal(blocks, forty_blocks)
                    and np.array_equal(heads, forty_heads)
                    and np.array_equal(objects, forty_objects)
                ):
                    raise RuntimeError("Capture metadata mismatch")
                cosine = np.einsum("tohd,ohd->toh", ten_stack, forty, optimize=True)
                best_steps = np.argmax(cosine, axis=0)
                best_scores = np.take_along_axis(cosine, best_steps[None, ...], axis=0)[0]
                for object_index, object_name in enumerate(objects):
                    for rank, (block, head) in enumerate(zip(blocks, heads)):
                        rows.append(
                            [
                                seed,
                                branch,
                                object_name,
                                rank + 1,
                                int(block),
                                int(head),
                                float(rank_pck[rank]),
                                step40,
                                int(best_steps[object_index, rank]),
                                float(best_scores[object_index, rank]),
                            ]
                        )
    with (output / "per_head_best_matches_by_sample_reverse.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["seed", "branch", "object", "rank", "block", "head", "lora_pck32", "step40", "best_step10", "cosine"]
        )
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output = args.root / "analysis" / "reverse"
    output.mkdir(parents=True, exist_ok=True)
    write_aggregate(args.root, output)
    write_samples(args.root, output, args.seeds)
    print(output)


if __name__ == "__main__":
    main()
