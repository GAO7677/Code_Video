#!/usr/bin/env python3
"""Aggregate complete DiffTrack toy-dataset probe results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def best_entry(matrix: np.ndarray) -> tuple[int, int, float]:
    layer, step = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(layer), int(step), float(matrix[layer, step])


def main() -> None:
    args = parse_args()
    states = sorted(args.result_dir.glob("*/step_state.npz"))
    if not states:
        raise FileNotFoundError(f"No step_state.npz files under {args.result_dir}")

    matrices: dict[str, list[np.ndarray]] = {"qk": [], "feature": []}
    rows = []
    for state_path in states:
        sample_dir = state_path.parent
        if not (sample_dir / "complete.json").exists():
            raise RuntimeError(f"Incomplete sample: {sample_dir.name}")
        state = np.load(state_path)
        completed_steps = state["completed_steps"]
        if not np.array_equal(completed_steps, np.arange(50)):
            raise RuntimeError(f"Unexpected completed steps for {sample_dir.name}: {completed_steps}")

        for descriptor, state_key in (("qk", "qk_pck"), ("feature", "feature_pck")):
            matrix = state[state_key]
            if matrix.shape != (30, 50) or not np.isfinite(matrix).all():
                raise RuntimeError(f"Invalid {state_key} matrix for {sample_dir.name}: {matrix.shape}")
            matrices[descriptor].append(matrix)
            layer, step, value = best_entry(matrix)
            step49_layer = int(np.argmax(matrix[:, 49]))
            rows.append(
                {
                    "sample_id": sample_dir.name,
                    "descriptor": descriptor,
                    "global_best_pck8": value,
                    "global_best_layer": layer,
                    "global_best_inverse_step": step,
                    "step49_best_pck8": float(matrix[step49_layer, 49]),
                    "step49_best_layer": step49_layer,
                    "fixed_layer17_step49_pck8": float(matrix[17, 49]),
                }
            )

    frame = pd.DataFrame(rows)
    base_values = frame[frame["sample_id"].str.endswith("_base")].set_index("descriptor")
    frame["global_best_delta_from_base"] = frame.apply(
        lambda row: row["global_best_pck8"] - base_values.loc[row["descriptor"], "global_best_pck8"], axis=1
    )
    frame["fixed_layer17_step49_delta_from_base"] = frame.apply(
        lambda row: row["fixed_layer17_step49_pck8"]
        - base_values.loc[row["descriptor"], "fixed_layer17_step49_pck8"],
        axis=1,
    )
    frame.to_csv(args.result_dir / "per_sample_summary.csv", index=False)

    aggregate = {"num_unique_videos": len(states), "descriptors": {}}
    average_arrays = {}
    for descriptor, values in matrices.items():
        stack = np.stack(values)
        mean_matrix = stack.mean(axis=0)
        std_matrix = stack.std(axis=0)
        layer, step, value = best_entry(mean_matrix)
        step49_layer = int(np.argmax(mean_matrix[:, 49]))
        aggregate["descriptors"][descriptor] = {
            "mean_surface_best_pck8": value,
            "mean_surface_best_layer": layer,
            "mean_surface_best_inverse_step": step,
            "step49_best_mean_pck8": float(mean_matrix[step49_layer, 49]),
            "step49_best_layer": step49_layer,
            "fixed_layer17_step49_mean_pck8": float(mean_matrix[17, 49]),
            "fixed_layer17_step49_std_pck8": float(std_matrix[17, 49]),
        }
        average_arrays[f"{descriptor}_mean"] = mean_matrix
        average_arrays[f"{descriptor}_std"] = std_matrix

    (args.result_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    np.savez_compressed(args.result_dir / "aggregate_surfaces.npz", **average_arrays)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
