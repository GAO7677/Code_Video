#!/usr/bin/env python3
"""Inject pooled all-token Q/K capture into the existing ToyDataset workers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/home/gaoya/Code_Video/DiffTrack-main")
REFERENCE_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
)
for path in (PROJECT_ROOT, REFERENCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AAA_my_test import analyze_stage1b_kubric_generation as probe
from selected_qk_matrix import pool_selected_qk_matrices


def selected_heads_from_environment() -> dict[int, tuple[int, ...]]:
    text = os.environ.get(
        "ALLTOKEN_COMBINATIONS",
        "20:9,20:17,26:7,18:11,24:6,19:0",
    )
    grouped: dict[int, list[int]] = {}
    for item in text.split(","):
        block, head = (int(value) for value in item.split(":"))
        grouped.setdefault(block, []).append(head)
    return {block: tuple(sorted(set(heads))) for block, heads in grouped.items()}


SELECTED_HEADS = selected_heads_from_environment()
OUTPUT_BINS = 512
QUERY_CHUNK = 64
METRICS_ONLY = os.environ.get("ALLTOKEN_METRICS_ONLY", "0") == "1"


def uniform_diagonal_metrics(attention: np.ndarray) -> dict[str, float]:
    """Summarize same-spatial-position attention without retaining the matrix."""
    count = int(attention.shape[-1])
    frame = np.floor(np.arange(count) * 7 / count).astype(np.int64)
    frame_ids = [np.flatnonzero(frame == index) for index in range(7)]
    query_mask = frame == 1
    diagonal = np.empty((count, 7), dtype=np.float32)
    for query in range(count):
        source = frame_ids[int(frame[query])]
        position = np.searchsorted(source, query) / max(len(source) - 1, 1)
        for target_time, keys in enumerate(frame_ids):
            center = int(round(position * (len(keys) - 1)))
            selected = keys[max(0, center - 1) : min(len(keys), center + 2)]
            diagonal[query, target_time] = attention[query, selected].sum()
    total = diagonal.sum(axis=1)
    distribution = diagonal / np.maximum(total[:, None], 1e-12)
    entropy = -(
        distribution * np.log(np.maximum(distribution, 1e-12))
    ).sum(axis=1) / math.log(7)
    return {
        "queryframe_diagonal_mass": float(total[query_mask].mean()),
        "queryframe_diagonal_frame_entropy": float(entropy[query_mask].mean()),
        "queryframe_joint": float((total * entropy)[query_mask].mean()),
        "queryframe_balanced_diagonal": float((7 * diagonal.min(axis=1))[query_mask].mean()),
        "alltoken_diagonal_mass": float(total.mean()),
        "alltoken_diagonal_frame_entropy": float(entropy.mean()),
        "alltoken_joint": float((total * entropy).mean()),
        "alltoken_balanced_diagonal": float((7 * diagonal.min(axis=1)).mean()),
    }


OriginalCapture = probe.GenerationCapture
original_save_records = probe.save_records
active_capture: "StableAllTokenCapture | None" = None


class StableAllTokenCapture(OriginalCapture):
    """Reuse the validated recorder while adding reference-style pooled matrices."""

    def __init__(self, *args, **kwargs):
        global active_capture
        super().__init__(*args, **kwargs)
        self.all_token_captures: dict[tuple[int, int], tuple] = {}
        self.diagonal_metric_rows: list[dict[str, float | int]] = []
        active_capture = self

    def _consume_qk(self, layer, q, k):
        selected = SELECTED_HEADS.get(int(layer))
        key = (int(layer), int(self.current_step))
        if selected and key not in self.all_token_captures:
            batch, sequence, heads, head_dim = q.shape
            if batch != 1:
                raise RuntimeError(f"all-token Q/K expects batch one, got {q.shape}")
            if self.current_grid is None:
                raise RuntimeError("all-token Q/K capture has no latent grid")
            q_flat = q.reshape(batch, sequence, heads * head_dim)
            k_flat = k.reshape(batch, sequence, heads * head_dim)
            captured = pool_selected_qk_matrices(
                q_flat,
                k_flat,
                num_heads=int(heads),
                selected_heads=selected,
                output_bins=OUTPUT_BINS,
                query_chunk=QUERY_CHUNK,
                temporal_bins=int(self.current_grid[0]),
            )
            if METRICS_ONLY:
                attention = captured[1].astype(np.float32)
                for index, head in enumerate(selected):
                    self.diagonal_metric_rows.append(
                        {
                            "block": int(layer),
                            "head": int(head),
                            "step": int(self.current_step),
                            **uniform_diagonal_metrics(attention[index]),
                        }
                    )
            else:
                self.all_token_captures[key] = captured
        super()._consume_qk(layer, q, k)

    def save_all_token(self, case_dir: Path) -> None:
        output = case_dir / "all_token_qk"
        output.mkdir(parents=True, exist_ok=True)
        if METRICS_ONLY:
            metrics_path = output / "uniform_diagonal_metrics.csv"
            with metrics_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(self.diagonal_metric_rows[0])
                )
                writer.writeheader()
                writer.writerows(self.diagonal_metric_rows)
            (output / "complete.json").write_text(
                json.dumps(
                    {
                        "mode": "uniform_diagonal_metrics_only",
                        "rows": len(self.diagonal_metric_rows),
                        "blocks": sorted(SELECTED_HEADS),
                        "steps": sorted({row["step"] for row in self.diagonal_metric_rows}),
                        "output_bins": OUTPUT_BINS,
                        "query_chunk": QUERY_CHUNK,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return
        for block, selected_heads in SELECTED_HEADS.items():
            steps = sorted(
                step for captured_block, step in self.all_token_captures
                if captured_block == block
            )
            if not steps:
                continue
            raw = np.stack(
                [self.all_token_captures[(block, step)][0] for step in steps]
            )
            attention = np.stack(
                [self.all_token_captures[(block, step)][1] for step in steps]
            )
            temporal = np.stack(
                [self.all_token_captures[(block, step)][2] for step in steps]
            )
            metadata = self.all_token_captures[(block, steps[0])][3]
            matrix_path = output / f"block{block:02d}_selected_qk.npz"
            np.savez_compressed(
                matrix_path,
                steps_zero_based=np.asarray(steps, dtype=np.int16),
                selected_heads=np.asarray(selected_heads, dtype=np.int16),
                raw_qk_mean=raw,
                softmax_attention_mass=attention,
                temporal_matrix=temporal,
            )
            summary = {
                "block": block,
                "selected_heads": list(selected_heads),
                "steps_zero_based": steps,
                "latent_grid": list(self.current_grid or ()),
                "matrix_npz": matrix_path.name,
                "raw_qk": "mean QK^T/sqrt(d) over each pooled query/key bin",
                "softmax_attention": (
                    "exact all-key softmax followed by key-bin mass aggregation "
                    "and query-bin averaging"
                ),
                "temporal_matrix": (
                    "exact-softmax mass aggregated by latent query/key time"
                ),
                "metadata": metadata,
            }
            (output / f"block{block:02d}_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        (output / "complete.json").write_text(
            json.dumps(
                {
                    "blocks": sorted(SELECTED_HEADS),
                    "combinations": [
                        {"block": block, "head": head}
                        for block, heads in SELECTED_HEADS.items()
                        for head in heads
                    ],
                    "steps": sorted({step for _, step in self.all_token_captures}),
                    "output_bins": OUTPUT_BINS,
                    "query_chunk": QUERY_CHUNK,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def save_records_with_all_token(output_dir, records):
    original_save_records(output_dir, records)
    if active_capture is None:
        raise RuntimeError("all-token capture instance was not created")
    active_capture.save_all_token(Path(output_dir))


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-kind", choices=("gt", "lora", "baseline"), required=True)
    return parser.parse_known_args()


def main() -> None:
    custom, remaining = parse_wrapper_args()
    probe.GenerationCapture = StableAllTokenCapture
    probe.save_records = save_records_with_all_token
    sys.argv = [sys.argv[0], *remaining]
    if custom.model_kind == "gt":
        from AAA_my_test import analyze_wan_gt_toy_worker as worker
    else:
        from AAA_my_test import run_lorav2v_toy_analysis_worker as worker
        if custom.model_kind == "baseline" and "--base-model-only" not in sys.argv:
            sys.argv.append("--base-model-only")
    worker.main()


if __name__ == "__main__":
    main()
