#!/usr/bin/env python3
"""Measure per-head specialization from exact moving-ball query attention."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_case_moving_ball_heads import (
    _circle_tokens,
    _generated_video,
    _track_latent_ball,
)


ROLE_LABELS = {
    "S": "帧内空间",
    "T": "球轨迹传播",
    "P": "固定位置时间对齐",
    "C": "首帧/历史上下文",
    "G": "全局聚合",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiblock-root", type=Path, required=True)
    parser.add_argument("--block17-root", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", default="0,5,11,17,19,29")
    parser.add_argument("--models", default="wan_lora,xssc,physrvg")
    return parser.parse_args()


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return order.astype(np.float64) / float(len(values) - 1)


def _feature_rows(
    attention: np.ndarray,
    *,
    query_coords: np.ndarray,
    trajectory_tokens: list[np.ndarray],
) -> tuple[dict[str, np.ndarray], float]:
    heads, frames, grid_h, grid_w = attention.shape
    token_count = frames * grid_h * grid_w
    query_t = int(query_coords[0, 0])
    flat = attention.reshape(heads, token_count).astype(np.float64)
    probability = np.clip(flat, 1.0e-30, None)
    entropy = -(probability * np.log(probability)).sum(1) / math.log(token_count)
    temporal = attention.sum(axis=(2, 3)).astype(np.float64)
    time_ids = np.arange(frames)
    same_frame = temporal[:, query_t]
    first_frame = temporal[:, 0]
    past = temporal[:, :query_t].sum(1)
    future = temporal[:, query_t + 1 :].sum(1)
    history_bias = past - future
    mean_time_distance = (
        temporal * np.abs(time_ids[None, :] - query_t)
    ).sum(1)

    local_mask = np.zeros((frames, grid_h, grid_w), dtype=bool)
    y0 = max(0, int(query_coords[:, 1].min()) - 1)
    y1 = min(grid_h, int(query_coords[:, 1].max()) + 2)
    x0 = max(0, int(query_coords[:, 2].min()) - 1)
    x1 = min(grid_w, int(query_coords[:, 2].max()) + 2)
    local_mask[query_t, y0:y1, x0:x1] = True
    local_mass = flat[:, local_mask.reshape(-1)].sum(1)

    aligned_mask = np.zeros((frames, grid_h, grid_w), dtype=bool)
    for time in range(frames):
        if time == query_t:
            continue
        for _, row, column in query_coords:
            aligned_mask[time, int(row), int(column)] = True
    aligned_count = int(aligned_mask.sum())
    aligned_mass = flat[:, aligned_mask.reshape(-1)].sum(1)
    aligned_enrichment = aligned_mass / (aligned_count / token_count)

    cross_ball_ids = np.unique(
        np.concatenate(
            [
                ids
                for latent_t, ids in enumerate(trajectory_tokens)
                if latent_t != query_t
            ]
        )
    )
    cross_ball_mass = flat[:, cross_ball_ids].sum(1)
    cross_ball_enrichment = cross_ball_mass / (
        len(cross_ball_ids) / token_count
    )

    normalized = flat / np.maximum(
        np.linalg.norm(flat, axis=1, keepdims=True), 1.0e-30
    )
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~np.eye(heads, dtype=bool)]
    mean_cosine = float(off_diagonal.mean())
    return (
        {
            "entropy": entropy,
            "same_frame_mass": same_frame,
            "local_mass": local_mass,
            "first_frame_mass": first_frame,
            "history_bias": history_bias,
            "mean_time_distance": mean_time_distance,
            "aligned_enrichment": aligned_enrichment,
            "cross_ball_enrichment": cross_ball_enrichment,
        },
        mean_cosine,
    )


def _role_scores(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "S": 0.5
        * (
            _rank01(features["same_frame_mass"])
            + _rank01(features["local_mass"])
        ),
        "T": 0.7 * _rank01(features["cross_ball_enrichment"])
        + 0.3 * _rank01(features["mean_time_distance"]),
        "P": _rank01(features["aligned_enrichment"]),
        "C": 0.5
        * (
            _rank01(features["first_frame_mass"])
            + _rank01(features["history_bias"])
        ),
        "G": _rank01(features["entropy"]),
    }


def _sample_labels(scores: dict[str, np.ndarray]) -> list[str]:
    roles = list(ROLE_LABELS)
    matrix = np.stack([scores[role] for role in roles], axis=1)
    return [roles[index] for index in matrix.argmax(axis=1)]


def _summary_path(block_root: Path, model: str, case: str) -> Path:
    return block_root / "matrices" / model / case / "summary.json"


def main() -> None:
    args = parse_args()
    multiblock_root = args.multiblock_root.expanduser().resolve()
    block17_root = args.block17_root.expanduser().resolve()
    generated_root = (
        args.generated_root.expanduser().resolve()
        if args.generated_root is not None
        else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = [int(value) for value in args.blocks.split(",") if value.strip()]
    models = tuple(
        value.strip() for value in args.models.split(",") if value.strip()
    )
    supported_models = {"wan_lora", "xssc", "physrvg"}
    if not models or len(set(models)) != len(models):
        raise ValueError("model list must be non-empty and unique")
    if not set(models).issubset(supported_models):
        raise ValueError(
            f"unsupported models {sorted(set(models) - supported_models)}"
        )
    sample_count = len(models) * 4
    stable_sample_count = math.ceil((2.0 / 3.0) * sample_count)

    block_roots = {
        block: (
            block17_root
            if block == 17
            else multiblock_root / f"block{block:02d}"
        )
        for block in blocks
    }

    trajectories: dict[str, tuple[list[np.ndarray], tuple[int, int]]] = {}
    for model in models:
        if generated_root is None:
            video = _generated_video(block17_root / "matrices", model, args.case)
        else:
            matches = sorted((generated_root / model).glob(f"**/{args.case}.mp4"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one generated video for {model}/{args.case}, "
                    f"found {matches}"
                )
            video = matches[0]
        trajectory, frame_shape = _track_latent_ball(video, temporal_tokens=13)
        grid = (13, 16, 28)
        trajectories[model] = (
            [
                _circle_tokens(
                    latent_t=latent_t,
                    circle=circle,
                    frame_shape=frame_shape,
                    grid=grid,
                )
                for latent_t, circle in enumerate(trajectory)
            ],
            frame_shape,
        )

    block_samples: dict[int, list[dict[str, Any]]] = {block: [] for block in blocks}
    for block in blocks:
        for model in models:
            summary_path = _summary_path(block_roots[block], model, args.case)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(summary["block_id"]) != block:
                raise RuntimeError(
                    f"{summary_path} records block {summary['block_id']}, expected {block}"
                )
            trajectory_tokens, _ = trajectories[model]
            for entry in summary["steps"]:
                matrix_path = (
                    summary_path.parent
                    / str(entry["directory"])
                    / str(entry["matrix_npz"])
                )
                with np.load(matrix_path) as arrays:
                    attention = arrays["attention"]
                    query_coords = arrays["query_coords"]
                features, cosine = _feature_rows(
                    attention,
                    query_coords=query_coords,
                    trajectory_tokens=trajectory_tokens,
                )
                scores = _role_scores(features)
                block_samples[block].append(
                    {
                        "model": model,
                        "step": int(entry["step_number_one_based"]),
                        "features": features,
                        "scores": scores,
                        "labels": _sample_labels(scores),
                        "mean_head_cosine": cosine,
                    }
                )

    block_records: list[dict[str, Any]] = []
    head_records: list[dict[str, Any]] = []
    raw_clarity: dict[int, float] = {}
    for block in blocks:
        samples = block_samples[block]
        heads = len(samples[0]["labels"])
        aggregate_features = {
            key: np.stack([sample["features"][key] for sample in samples]).mean(0)
            for key in samples[0]["features"]
        }
        aggregate_scores = _role_scores(aggregate_features)
        roles = list(ROLE_LABELS)
        model_aggregate_labels = {}
        for model in models:
            model_samples = [
                sample for sample in samples if sample["model"] == model
            ]
            model_features = {
                key: np.stack(
                    [sample["features"][key] for sample in model_samples]
                ).mean(0)
                for key in aggregate_features
            }
            model_aggregate_labels[model] = _sample_labels(
                _role_scores(model_features)
            )
        step_aggregate_labels = {}
        for step in (5, 15, 25, 35):
            step_samples = [
                sample for sample in samples if sample["step"] == step
            ]
            step_features = {
                key: np.stack(
                    [sample["features"][key] for sample in step_samples]
                ).mean(0)
                for key in aggregate_features
            }
            step_aggregate_labels[step] = _sample_labels(
                _role_scores(step_features)
            )
        score_matrix = np.stack(
            [aggregate_scores[role] for role in roles], axis=1
        )
        score_order = np.argsort(score_matrix, axis=1)
        aggregate_primary = [
            roles[int(index)] for index in score_order[:, -1]
        ]
        aggregate_secondary = [
            roles[int(index)] for index in score_order[:, -2]
        ]
        primary_scores = np.take_along_axis(
            score_matrix, score_order[:, -1:], axis=1
        )[:, 0]
        secondary_scores = np.take_along_axis(
            score_matrix, score_order[:, -2:-1], axis=1
        )[:, 0]
        margins = primary_scores - secondary_scores
        labels_by_sample = np.asarray([sample["labels"] for sample in samples])
        modal_labels = []
        stability = []
        for head in range(heads):
            counts = Counter(labels_by_sample[:, head].tolist())
            role, count = counts.most_common(1)[0]
            modal_labels.append(role)
            stability.append(count / len(samples))
        stability_array = np.asarray(stability)
        cosine = float(
            np.mean([sample["mean_head_cosine"] for sample in samples])
        )
        stable_fraction = float(np.mean(stability_array >= (2.0 / 3.0)))
        median_stability = float(np.median(stability_array))
        median_margin = float(np.median(margins))
        diversity = len(set(modal_labels))
        clarity = (
            stable_fraction
            * median_stability
            * max(0.0, 1.0 - cosine)
            * (0.5 + median_margin)
        )
        raw_clarity[block] = clarity
        block_records.append(
            {
                "block": block,
                "role_diversity": diversity,
                "stable_head_fraction": stable_fraction,
                "median_role_stability": median_stability,
                "mean_head_cosine": cosine,
                "median_role_margin": median_margin,
                "raw_clarity": clarity,
            }
        )
        for head in range(heads):
            primary_role = aggregate_primary[head]
            secondary_role = aggregate_secondary[head]
            aggregate_stability = float(
                np.mean(labels_by_sample[:, head] == primary_role)
            )
            model_modes = [
                model_aggregate_labels[model][head] for model in models
            ]
            step_modes = [
                step_aggregate_labels[step][head] for step in (5, 15, 25, 35)
            ]
            model_consistency = float(
                np.mean(np.asarray(model_modes) == primary_role)
            )
            step_consistency = float(
                np.mean(np.asarray(step_modes) == primary_role)
            )
            if aggregate_stability >= (2.0 / 3.0) and margins[head] >= 0.10:
                classification = f"明确{primary_role}"
            elif aggregate_stability >= 0.50:
                classification = f"{primary_role}/{secondary_role}混合"
            else:
                classification = f"不稳定{primary_role}/{secondary_role}混合"
            row = {
                "block": block,
                "head": head,
                "role": modal_labels[head],
                "role_label": ROLE_LABELS[modal_labels[head]],
                "role_stability": stability[head],
                "role_margin": float(margins[head]),
                "aggregate_primary_role": primary_role,
                "aggregate_primary_role_label": ROLE_LABELS[primary_role],
                "aggregate_primary_score": float(primary_scores[head]),
                "aggregate_secondary_role": secondary_role,
                "aggregate_secondary_role_label": ROLE_LABELS[secondary_role],
                "aggregate_secondary_score": float(secondary_scores[head]),
                "aggregate_role_stability": aggregate_stability,
                "model_role_consistency": model_consistency,
                "step_role_consistency": step_consistency,
                "classification": classification,
            }
            row.update(
                {
                    key: float(values[head])
                    for key, values in aggregate_features.items()
                }
            )
            row.update(
                {
                    f"{role.lower()}_score": float(aggregate_scores[role][head])
                    for role in roles
                }
            )
            head_records.append(row)

    baseline_clarity = raw_clarity[17]
    for row in block_records:
        relative = (
            row["raw_clarity"] / baseline_clarity
            if baseline_clarity > 0
            else float("nan")
        )
        row["clarity_relative_to_block17"] = relative
        if relative >= 0.85:
            row["verdict"] = "明确"
        elif relative >= 0.60:
            row["verdict"] = "中等"
        else:
            row["verdict"] = "较弱"

    block_csv = output_dir / "multiblock_head_clarity.csv"
    with block_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(block_records[0]))
        writer.writeheader()
        writer.writerows(block_records)
    head_csv = output_dir / "multiblock_head_roles.csv"
    with head_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(head_records[0]))
        writer.writeheader()
        writer.writerows(head_records)

    block_table = []
    detail_sections = []
    for row in block_records:
        block_table.append(
            f"| {row['block']:02d} | {row['role_diversity']} | "
            f"{row['stable_head_fraction']:.0%} | "
            f"{row['median_role_stability']:.0%} | "
            f"{row['mean_head_cosine']:.3f} | "
            f"{row['median_role_margin']:.3f} | "
            f"{row['clarity_relative_to_block17']:.2f}× | "
            f"{row['verdict']} |"
        )
        current = [record for record in head_records if record["block"] == row["block"]]
        role_rows = []
        for role, label in ROLE_LABELS.items():
            ranked = sorted(
                current,
                key=lambda record: record[f"{role.lower()}_score"],
                reverse=True,
            )[:3]
            role_rows.append(
                f"| {label} | "
                + ", ".join(
                    f"H{record['head']:02d}"
                    f"({record[f'{role.lower()}_score']:.2f})"
                    for record in ranked
                )
                + " |"
            )
        all_head_rows = [
            f"| H{record['head']:02d} | {record['classification']} | "
            f"{record['aggregate_primary_role']} "
            f"{record['aggregate_primary_score']:.2f} | "
            f"{record['aggregate_secondary_role']} "
            f"{record['aggregate_secondary_score']:.2f} | "
            f"{record['aggregate_role_stability']:.0%} | "
            f"{record['model_role_consistency']:.0%} | "
            f"{record['step_role_consistency']:.0%} |"
            for record in current
        ]
        detail_sections.append(
            f"""## Block {row['block']:02d}

| 功能 | 得分最高的 Head |
|---|---|
{chr(10).join(role_rows)}

| Head | 分类 | 主角色 | 次角色 | 样本稳定性 | 模型一致性 | 去噪步一致性 |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(all_head_rows)}
"""
        )

    markdown = f"""# Multi-Block Exact Ball-Query Head Specialization

Case: `{args.case}`. Models: `{list(models)}`. Blocks: `{blocks}`. Each sample uses the exact mean
attention from four moving-ball query patches to all 5824 key tokens. Statistics
cover the listed model(s) and denoise steps 5/15/25/35.

| Block | Role types | Stable heads | Median stability | Head cosine | Role margin | Relative clarity | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(block_table)}

`Head cosine` is mean pairwise cosine similarity between the 24 attention maps;
lower means less redundancy. `Stable heads` retain the same dominant role in at
least {stable_sample_count} of {sample_count} model/step samples. Relative clarity is a heuristic normalized to
Block 17 and should be read together with the raw columns.

Classification uses the highest aggregate rank-based role score. `明确` requires
at least {stable_sample_count}/{sample_count} samples to agree and a primary-secondary score margin of at least
0.10. Other heads are explicitly marked as mixed or unstable; these labels are
descriptive heuristics rather than causal proofs of head function.

{chr(10).join(detail_sections)}
"""
    markdown_path = output_dir / "multiblock_head_specialization.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path = output_dir / "multiblock_head_specialization.json"
    json_path.write_text(
        json.dumps(
            {
                "case": args.case,
                "blocks": block_records,
                "heads": head_records,
                "role_labels": ROLE_LABELS,
                "method": {
                    "query": "exact four moving-ball patches",
                    "keys": 5824,
                    "models": list(models),
                    "steps": [5, 15, 25, 35],
                    "clarity_reference": "Block 17",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "markdown": str(markdown_path),
                "block_csv": str(block_csv),
                "head_csv": str(head_csv),
                "json": str(json_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
