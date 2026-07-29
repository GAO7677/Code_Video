#!/usr/bin/env python3
"""Rank common stable T heads by aggregated trajectory features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from classify_fulltoken_moving_heads import _rank
from export_head_role_raw_features import (
    EXPECTED_STEPS,
    OBJECT_NAMES,
    _load_case,
    _trajectory_validity,
)


MODELS = ("wan_lora", "xssc", "physrvg")
FEATURES = (
    "trajectory_selectivity_log2",
    "trajectory_enrichment",
    "mean_time_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--seed-snapshot", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-visible-times", type=int, default=8)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.8)
    return parser.parse_args()


def input_cases(path: Path) -> list[str]:
    return [
        Path(line.strip()).expanduser().resolve().stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_t_heads(path: Path) -> list[tuple[int, int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    heads = sorted(
        (int(row["block"]), int(row["head"]))
        for row in rows
        if row["role"] == "T"
    )
    if not heads or len(set(heads)) != len(heads):
        raise RuntimeError(f"invalid common T-head list: {len(heads)} rows")
    return heads


def dense_rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int32)
    return ranks


def accumulate(
    total: np.ndarray,
    count: np.ndarray,
    values: np.ndarray,
) -> None:
    finite = np.isfinite(values)
    total[finite] += values[finite]
    count[finite] += 1


def main() -> None:
    args = parse_args()
    capture_root = args.capture_root.expanduser().resolve()
    query_root = args.query_root.expanduser().resolve()
    snapshot = json.loads(
        args.seed_snapshot.expanduser().resolve().read_text(encoding="utf-8")
    )
    cases = input_cases(args.input_list.expanduser().resolve())
    t_heads = load_t_heads(args.heads.expanduser().resolve())
    feature_indices = {name: OBJECT_NAMES.index(name) for name in FEATURES}

    output_rows = []
    for model in MODELS:
        seeds = [int(value) for value in snapshot[model]]
        raw_sum = {name: np.zeros((30, 24), dtype=np.float64) for name in FEATURES}
        raw_count = {
            name: np.zeros((30, 24), dtype=np.int64) for name in FEATURES
        }
        rank_sum = {name: np.zeros((30, 24), dtype=np.float64) for name in FEATURES}
        rank_count = {
            name: np.zeros((30, 24), dtype=np.int64) for name in FEATURES
        }
        valid_samples = 0

        for seed_index, seed in enumerate(seeds, start=1):
            query_path = (
                query_root / model / f"seed-{seed:06d}" / "query_map.json"
            )
            query_map = json.loads(
                query_path.read_text(encoding="utf-8")
            )["cases"]
            seed_root = capture_root / model / f"seed-{seed:06d}"
            for case in cases:
                valid, _, _ = _trajectory_validity(
                    query_map,
                    case,
                    minimum_visible_times=args.minimum_visible_times,
                    minimum_valid_ratio=args.minimum_valid_ratio,
                )
                if not valid:
                    continue
                _, object_steps = _load_case(seed_root, model, case)
                valid_samples += 1
                for step_index in range(len(EXPECTED_STEPS)):
                    for name, feature_index in feature_indices.items():
                        values = object_steps[step_index, ..., feature_index]
                        accumulate(raw_sum[name], raw_count[name], values)
                        ranked = _rank(values)
                        accumulate(rank_sum[name], rank_count[name], ranked)
            print(
                f"[t-feature-rank] {model} seed {seed_index}/{len(seeds)} "
                f"valid_samples={valid_samples}",
                flush=True,
            )

        raw_mean = {
            name: np.divide(
                raw_sum[name],
                raw_count[name],
                out=np.full((30, 24), np.nan, dtype=np.float64),
                where=raw_count[name] > 0,
            )
            for name in FEATURES
        }
        rank_mean = {
            name: np.divide(
                rank_sum[name],
                rank_count[name],
                out=np.full((30, 24), np.nan, dtype=np.float64),
                where=rank_count[name] > 0,
            )
            for name in FEATURES
        }
        selected = {
            name: np.asarray(
                [rank_mean[name][block, head] for block, head in t_heads]
            )
            for name in FEATURES
        }
        order = {name: dense_rank_desc(values) for name, values in selected.items()}

        for index, (block, head) in enumerate(t_heads):
            row: dict[str, object] = {
                "model": model,
                "block": block,
                "head": head,
                "valid_samples": valid_samples,
            }
            for name in FEATURES:
                row[f"{name}_observations"] = int(raw_count[name][block, head])
                row[f"{name}_raw_mean"] = raw_mean[name][block, head]
                row[f"{name}_rank_mean"] = rank_mean[name][block, head]
                row[f"{name}_t_rank"] = int(order[name][index])
            output_rows.append(row)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[t-feature-rank] wrote {len(output_rows)} rows to {output}")


if __name__ == "__main__":
    main()
