#!/usr/bin/env python3
"""Classify all DiT heads from exact moving-object query attention maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


MODELS = ("wan_lora", "xssc", "physrvg")
ROLES = ("S", "T", "P", "C", "G")
ROLE_LABELS = {
    "S": "within-frame spatial",
    "T": "moving-object trajectory",
    "P": "fixed-position temporal",
    "C": "history/context",
    "G": "global aggregation",
}
FEATURE_NAMES = (
    "entropy",
    "same_frame_mass",
    "local_mass",
    "first_frame_mass",
    "history_bias",
    "mean_time_distance",
    "aligned_enrichment",
    "cross_ball_enrichment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--blocks", default=",".join(str(i) for i in range(30)))
    parser.add_argument("--steps", default="5,15,25,35")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(len(values) - 1, 1)


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


def _features_from_maps(
    attention: np.ndarray, query_coords: np.ndarray
) -> dict[str, np.ndarray]:
    heads, query_times, key_times, grid_h, grid_w = attention.shape
    if query_times != key_times:
        raise ValueError(f"query/key time mismatch: {attention.shape}")
    token_count = key_times * grid_h * grid_w
    time_ids = np.arange(key_times)
    values = {name: [] for name in FEATURE_NAMES}

    for query_time in range(query_times):
        probability = attention[:, query_time].astype(np.float64)
        probability /= np.maximum(
            probability.sum(axis=(1, 2, 3), keepdims=True), 1.0e-30
        )
        flat = probability.reshape(heads, token_count)
        temporal = probability.sum(axis=(2, 3))
        current = query_coords[query_coords[:, 0] == query_time]
        if not len(current):
            raise ValueError(f"query coordinates do not cover time {query_time}")

        local_mask = np.zeros((key_times, grid_h, grid_w), dtype=bool)
        y0 = max(0, int(current[:, 1].min()) - 1)
        y1 = min(grid_h, int(current[:, 1].max()) + 2)
        x0 = max(0, int(current[:, 2].min()) - 1)
        x1 = min(grid_w, int(current[:, 2].max()) + 2)
        local_mask[query_time, y0:y1, x0:x1] = True

        aligned_mask = np.zeros_like(local_mask)
        for key_time in range(key_times):
            if key_time == query_time:
                continue
            aligned_mask[
                key_time, current[:, 1].astype(int), current[:, 2].astype(int)
            ] = True

        trajectory_mask = np.zeros_like(local_mask)
        for key_time in range(key_times):
            if key_time == query_time:
                continue
            key_coords = query_coords[query_coords[:, 0] == key_time]
            trajectory_mask[
                key_time,
                key_coords[:, 1].astype(int),
                key_coords[:, 2].astype(int),
            ] = True

        safe = np.maximum(flat, 1.0e-30)
        values["entropy"].append(
            -(safe * np.log(safe)).sum(axis=1) / math.log(token_count)
        )
        values["same_frame_mass"].append(temporal[:, query_time])
        values["local_mass"].append(flat[:, local_mask.reshape(-1)].sum(axis=1))
        values["first_frame_mass"].append(temporal[:, 0])
        values["history_bias"].append(
            temporal[:, :query_time].sum(axis=1)
            - temporal[:, query_time + 1 :].sum(axis=1)
        )
        values["mean_time_distance"].append(
            (temporal * np.abs(time_ids - query_time)[None]).sum(axis=1)
        )
        values["aligned_enrichment"].append(
            flat[:, aligned_mask.reshape(-1)].sum(axis=1)
            / (aligned_mask.sum() / token_count)
        )
        values["cross_ball_enrichment"].append(
            flat[:, trajectory_mask.reshape(-1)].sum(axis=1)
            / (trajectory_mask.sum() / token_count)
        )

    return {
        name: np.stack(samples, axis=0).mean(axis=0)
        for name, samples in values.items()
    }


def _load_step(
    root: Path, model: str, case: str, block: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    summary_path = (
        root / f"block{block:02d}" / "matrices" / model / case / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in summary["steps"]
        if int(item["step_number_one_based"]) == step
    )
    with np.load(summary_path.parent / entry["directory"] / entry["maps_npz"]) as data:
        attention = data["attention"].astype(np.float32)
        heads = data["selected_heads"].astype(int)
        coords = data["query_coords"].astype(int)
    if heads.tolist() != list(range(24)):
        raise ValueError(f"{summary_path}: expected heads 0..23")
    return attention, coords


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.maps_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    blocks = tuple(int(value) for value in args.blocks.split(","))
    steps = tuple(int(value) for value in args.steps.split(","))
    head_rows: list[dict] = []
    block_rows: list[dict] = []
    representatives: dict[str, dict[str, dict[str, int]]] = {}

    for model in MODELS:
        representatives[model] = {}
        for block in blocks:
            step_features = []
            step_scores = []
            step_roles = []
            for step in steps:
                attention, coords = _load_step(
                    root, model, args.case, block, step
                )
                features = _features_from_maps(attention, coords)
                scores = _role_scores(features)
                matrix = np.stack([scores[role] for role in ROLES], axis=1)
                step_features.append(features)
                step_scores.append(matrix)
                step_roles.append(matrix.argmax(axis=1))

            mean_features = {
                name: np.stack(
                    [sample[name] for sample in step_features], axis=0
                ).mean(axis=0)
                for name in FEATURE_NAMES
            }
            mean_scores = np.stack(step_scores, axis=0).mean(axis=0)
            labels = np.stack(step_roles, axis=0)
            primary = mean_scores.argmax(axis=1)
            sorted_scores = np.sort(mean_scores, axis=1)
            margins = sorted_scores[:, -1] - sorted_scores[:, -2]
            stability = np.asarray(
                [
                    (labels[:, head] == primary[head]).mean()
                    for head in range(24)
                ]
            )

            role_indices, selected_heads = linear_sum_assignment(-mean_scores.T)
            selected = {
                ROLES[int(role)]: int(head)
                for role, head in zip(role_indices, selected_heads)
            }
            representatives[model][str(block)] = selected

            counts = Counter(ROLES[index] for index in primary)
            block_rows.append(
                {
                    "model": model,
                    "block": block,
                    **{f"{role}_count": counts[role] for role in ROLES},
                    "mean_stability": float(stability.mean()),
                    "clear_heads": int(
                        ((stability >= 0.75) & (margins >= 0.10)).sum()
                    ),
                }
            )
            for head in range(24):
                role = ROLES[int(primary[head])]
                if stability[head] >= 0.75 and margins[head] >= 0.10:
                    confidence = "clear"
                elif stability[head] <= 0.50 or margins[head] < 0.03:
                    confidence = "unstable"
                else:
                    confidence = "mixed"
                row = {
                    "model": model,
                    "block": block,
                    "head": head,
                    "role": role,
                    "role_label": ROLE_LABELS[role],
                    "confidence": confidence,
                    "step_stability": float(stability[head]),
                    "role_margin": float(margins[head]),
                }
                row.update(
                    {
                        f"{candidate.lower()}_score": float(
                            mean_scores[head, index]
                        )
                        for index, candidate in enumerate(ROLES)
                    }
                )
                row.update(
                    {
                        name: float(mean_features[name][head])
                        for name in FEATURE_NAMES
                    }
                )
                head_rows.append(row)

    _write_csv(output / "allblock_head_roles.csv", head_rows)
    _write_csv(output / "allblock_role_counts.csv", block_rows)
    payload = {
        "case": args.case,
        "blocks": list(blocks),
        "steps": list(steps),
        "roles": ROLE_LABELS,
        "method": {
            "query": "moving 2x2 object-token query at each latent time",
            "aggregation": "mean feature and role score over query times and denoise steps",
            "scores": (
                "same rank-based S/T/P/C/G formulas as the existing all-block "
                "analysis; roles are relative specializations within one block"
            ),
            "clear": "step stability >= 0.75 and primary margin >= 0.10",
            "unstable": "step stability <= 0.50 or primary margin < 0.03",
            "map_entropy_note": (
                "entropy is computed after averaging the 2x2 object queries"
            ),
        },
        "representatives": representatives,
        "heads": head_rows,
        "blocks_summary": block_rows,
    }
    (output / "allblock_head_roles.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output / "allblock_head_roles.json")


if __name__ == "__main__":
    main()
