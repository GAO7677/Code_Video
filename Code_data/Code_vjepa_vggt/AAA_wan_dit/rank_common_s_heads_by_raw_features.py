#!/usr/bin/env python3
"""Rank common stable S heads by aggregated local and same-frame features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from classify_fulltoken_moving_heads import _rank


MODELS = ("wan_lora", "xssc", "physrvg")
FEATURES = ("local_enrichment", "same_frame_mass")
EXPECTED_STEPS = (5, 15, 25, 35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--seed-snapshot", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def input_cases(path: Path) -> list[str]:
    return [
        Path(line.strip()).expanduser().resolve().stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_s_heads(path: Path) -> list[tuple[int, int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    heads = sorted(
        (int(row["block"]), int(row["head"]))
        for row in rows
        if row["role"] == "S"
    )
    if len(heads) != 159 or len(set(heads)) != len(heads):
        raise RuntimeError(f"expected 159 unique common S heads, found {len(heads)}")
    return heads


def load_case(
    capture_root: Path,
    *,
    model: str,
    seed: int,
    case: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    seed_root = capture_root / model / f"seed-{seed:06d}"
    paths = sorted(
        seed_root.glob(
            f"block*/matrices/{model}/{case}/block*_fulltoken_moving.npz"
        )
    )
    if len(paths) != 30:
        raise RuntimeError(
            f"{model}/seed-{seed:06d}/{case}: found {len(paths)}/30 blocks"
        )
    blocks = []
    names: tuple[str, ...] | None = None
    observed = []
    for path in paths:
        observed.append(int(path.name.split("_", 1)[0].replace("block", "")))
        with np.load(path, allow_pickle=False) as data:
            steps = tuple(int(value) for value in data["steps_one_based"])
            current_names = tuple(data["full_feature_names"].astype(str))
            full = data["full_features"].astype(np.float32)
        if steps != EXPECTED_STEPS or full.shape != (4, 24, len(current_names)):
            raise RuntimeError(f"unexpected feature schema: {path}")
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError(f"inconsistent feature names: {path}")
        blocks.append(full)
    if observed != list(range(30)) or names is None:
        raise RuntimeError(f"invalid block ordering for {model}/{seed}/{case}")
    return np.stack(blocks, axis=1), names


def dense_rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int32)
    return ranks


def main() -> None:
    args = parse_args()
    capture_root = args.capture_root.expanduser().resolve()
    snapshot = json.loads(
        args.seed_snapshot.expanduser().resolve().read_text(encoding="utf-8")
    )
    cases = input_cases(args.input_list.expanduser().resolve())
    s_heads = load_s_heads(args.heads.expanduser().resolve())
    if len(cases) != 20:
        raise RuntimeError(f"expected 20 cases, found {len(cases)}")

    output_rows = []
    for model in MODELS:
        seeds = [int(value) for value in snapshot[model]]
        if len(seeds) != 22:
            raise RuntimeError(f"{model}: expected 22 seeds, found {len(seeds)}")
        raw_sum = {name: np.zeros((30, 24), dtype=np.float64) for name in FEATURES}
        rank_sum = {name: np.zeros((30, 24), dtype=np.float64) for name in FEATURES}
        observations = 0
        for seed_index, seed in enumerate(seeds, start=1):
            for case in cases:
                full, names = load_case(
                    capture_root,
                    model=model,
                    seed=seed,
                    case=case,
                )
                index = {name: names.index(name) for name in FEATURES}
                for step_index in range(full.shape[0]):
                    for name in FEATURES:
                        values = full[step_index, ..., index[name]]
                        raw_sum[name] += values
                        rank_sum[name] += _rank(values)
                    observations += 1
            print(
                f"[s-feature-rank] {model} seed {seed_index}/{len(seeds)} "
                f"observations={observations}",
                flush=True,
            )
        expected_observations = len(seeds) * len(cases) * len(EXPECTED_STEPS)
        if observations != expected_observations:
            raise RuntimeError(
                f"{model}: {observations} observations, expected {expected_observations}"
            )
        raw_mean = {name: values / observations for name, values in raw_sum.items()}
        rank_mean = {name: values / observations for name, values in rank_sum.items()}
        selected = {
            name: np.asarray([rank_mean[name][block, head] for block, head in s_heads])
            for name in FEATURES
        }
        order = {name: dense_rank_desc(values) for name, values in selected.items()}
        for index, (block, head) in enumerate(s_heads):
            output_rows.append(
                {
                    "model": model,
                    "block": block,
                    "head": head,
                    "observations": observations,
                    "local_enrichment_raw_mean": raw_mean["local_enrichment"][
                        block, head
                    ],
                    "local_enrichment_rank_mean": rank_mean["local_enrichment"][
                        block, head
                    ],
                    "local_enrichment_s_rank": int(
                        order["local_enrichment"][index]
                    ),
                    "same_frame_mass_raw_mean": raw_mean["same_frame_mass"][
                        block, head
                    ],
                    "same_frame_mass_rank_mean": rank_mean["same_frame_mass"][
                        block, head
                    ],
                    "same_frame_mass_s_rank": int(
                        order["same_frame_mass"][index]
                    ),
                }
            )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[s-feature-rank] wrote {len(output_rows)} rows to {output}")


if __name__ == "__main__":
    main()
