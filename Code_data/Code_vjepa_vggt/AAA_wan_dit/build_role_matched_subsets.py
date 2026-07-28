#!/usr/bin/env python3
"""Build reproducible S/T/C Head subsets with audited depth matching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from common22_public_head_targets import load_public_head_targets


ROLES = ("S", "T", "C")
BANDS = ((0, 6), (6, 12), (12, 18), (18, 24), (24, 30))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate-pool", type=int, default=20000)
    parser.add_argument("--exact-common-block-replicates", type=int, default=2)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(targets: tuple[tuple[int, int], ...]) -> np.ndarray:
    blocks = np.asarray([block for block, _ in targets], dtype=np.float64)
    ordered = np.sort(blocks) / 29.0
    counts = np.asarray(
        [np.sum((blocks >= start) & (blocks < stop)) for start, stop in BANDS],
        dtype=np.float64,
    )
    counts /= float(len(targets))
    moments = np.asarray(
        [
            blocks.mean() / 29.0,
            blocks.std() / 15.0,
            np.quantile(blocks, 0.25) / 29.0,
            np.quantile(blocks, 0.50) / 29.0,
            np.quantile(blocks, 0.75) / 29.0,
        ],
        dtype=np.float64,
    )
    # Ordered depths approximate a one-dimensional Wasserstein match, while
    # band counts and moments prevent a superficially similar mean from hiding
    # different early/middle/late-layer support.
    return np.concatenate((ordered, counts, moments))


def _candidate_pool(
    values: list[tuple[int, int]],
    *,
    k: int,
    count: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[tuple[int, int], ...]], np.ndarray]:
    if len(values) < k:
        raise ValueError(f"Cannot sample k={k} from {len(values)} heads")
    candidates: list[tuple[tuple[int, int], ...]] = []
    observed: set[tuple[tuple[int, int], ...]] = set()
    attempts = 0
    maximum_attempts = count * 20
    while len(candidates) < count and attempts < maximum_attempts:
        indices = rng.choice(len(values), size=k, replace=False)
        candidate = tuple(sorted(values[int(index)] for index in indices))
        observed.add(candidate)
        attempts += 1
        if len(observed) > len(candidates):
            candidates.append(candidate)
    if len(candidates) < min(count, 1000):
        raise RuntimeError(
            f"Only generated {len(candidates)} unique candidates for k={k}"
        )
    return candidates, np.stack([_profile(item) for item in candidates])


def _pair_cost(
    profiles: dict[str, np.ndarray],
    indices: dict[str, int],
) -> float:
    selected = [profiles[role][indices[role]] for role in ROLES]
    return float(
        sum(
            np.linalg.norm(selected[left] - selected[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
    )


def _matched_triplets(
    targets: dict[str, list[tuple[int, int]]],
    *,
    k: int,
    replicates: int,
    candidate_count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    candidates: dict[str, list[tuple[tuple[int, int], ...]]] = {}
    profiles: dict[str, np.ndarray] = {}
    trees: dict[str, cKDTree] = {}
    for role in ROLES:
        candidates[role], profiles[role] = _candidate_pool(
            targets[role],
            k=k,
            count=candidate_count,
            rng=rng,
        )
        trees[role] = cKDTree(profiles[role])

    proposals: list[tuple[float, dict[str, int]]] = []
    for anchor in ROLES:
        others = [role for role in ROLES if role != anchor]
        for anchor_index, anchor_profile in enumerate(profiles[anchor]):
            indices = {anchor: anchor_index}
            for role in others:
                _, nearest = trees[role].query(anchor_profile, k=1)
                indices[role] = int(nearest)
            proposals.append((_pair_cost(profiles, indices), indices))
    proposals.sort(key=lambda item: item[0])

    selected: list[dict[str, Any]] = []
    used_subsets = {role: set() for role in ROLES}
    usage = {role: Counter() for role in ROLES}
    while len(selected) < replicates:
        best: tuple[float, float, dict[str, int]] | None = None
        for raw_cost, indices in proposals[: min(len(proposals), 20000)]:
            subset_keys = {
                role: candidates[role][indices[role]] for role in ROLES
            }
            if any(
                subset_keys[role] in used_subsets[role] for role in ROLES
            ):
                continue
            overlap = sum(
                sum(usage[role][target] for target in subset_keys[role])
                for role in ROLES
            )
            objective = raw_cost + 0.025 * float(overlap)
            candidate = (objective, raw_cost, indices)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            raise RuntimeError("Unable to select non-duplicate matched triplets")
        _, raw_cost, indices = best
        record: dict[str, Any] = {
            "matching": "approximate_depth_profile",
            "profile_pairwise_l2_sum": raw_cost,
            "roles": {},
        }
        for role in ROLES:
            subset = candidates[role][indices[role]]
            used_subsets[role].add(subset)
            usage[role].update(subset)
            record["roles"][role] = {
                "targets": subset,
                "profile": _profile(subset),
            }
        selected.append(record)
    return selected


def _aggregate_scores(path: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    output: dict[tuple[str, int, int], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], int(row["block"]), int(row["head"]))
            output[key] = {
                name: float(row[name])
                for name in row
                if name.startswith("score_") or name in {"margin", "support"}
            }
    return output


def _target_record(
    role: str,
    target: tuple[int, int],
    scores: dict[tuple[str, int, int], dict[str, float]],
) -> dict[str, Any]:
    block, head = target
    model_values = {
        model: scores[(model, block, head)]
        for model in ("wan_lora", "xssc", "physrvg")
    }
    return {
        "block": block,
        "head": head,
        "model_statistics": model_values,
        "cross_model_mean_role_score": float(
            np.mean([values[f"score_{role}"] for values in model_values.values()])
        ),
        "cross_model_mean_margin": float(
            np.mean([values["margin"] for values in model_values.values()])
        ),
        "cross_model_mean_support": float(
            np.mean([values["support"] for values in model_values.values()])
        ),
    }


def main() -> None:
    args = parse_args()
    report = args.head_report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    targets, source = load_public_head_targets(report)
    role_targets = {role: targets[role] for role in ROLES}
    scores = _aggregate_scores(report)
    rng = np.random.default_rng(int(args.seed))
    approximate = _matched_triplets(
        role_targets,
        k=int(args.k),
        replicates=int(args.replicates),
        candidate_count=int(args.candidate_pool),
        rng=rng,
    )
    subsets: dict[str, Any] = {}
    for replicate, triplet in enumerate(approximate):
        for role in ROLES:
            values = triplet["roles"][role]
            subset_id = f"{role}_k{args.k:02d}_r{replicate:02d}_depthmatch"
            selected = list(values["targets"])
            subsets[subset_id] = {
                "role": role,
                "k": int(args.k),
                "replicate": replicate,
                "matching": triplet["matching"],
                "triplet_profile_pairwise_l2_sum": triplet[
                    "profile_pairwise_l2_sum"
                ],
                "depth_profile": values["profile"].tolist(),
                "block_histogram": dict(
                    sorted(Counter(block for block, _ in selected).items())
                ),
                "targets": [
                    _target_record(role, target, scores) for target in selected
                ],
            }

    common_blocks = sorted(
        set(block for block, _ in role_targets["S"])
        & set(block for block, _ in role_targets["T"])
        & set(block for block, _ in role_targets["C"])
    )
    if common_blocks != [9, 15, 16, 17, 28]:
        raise RuntimeError(f"Unexpected common S/T/C blocks: {common_blocks}")
    maximum_unique_exact = min(
        int(
            np.prod(
                [
                    sum(target[0] == block for target in role_targets[role])
                    for block in common_blocks
                ]
            )
        )
        for role in ROLES
    )
    if int(args.exact_common_block_replicates) > maximum_unique_exact:
        raise ValueError(
            "Requested exact-block replicates exceed the number of unique "
            f"subsets available to every role: {maximum_unique_exact}"
        )
    for replicate in range(int(args.exact_common_block_replicates)):
        for role in ROLES:
            selected = []
            for block in common_blocks:
                candidates = [
                    target for target in role_targets[role] if target[0] == block
                ]
                selected.append(
                    candidates[int(rng.integers(0, len(candidates)))]
                )
            subset_id = f"{role}_k05_r{replicate:02d}_exactblock"
            subsets[subset_id] = {
                "role": role,
                "k": 5,
                "replicate": replicate,
                "matching": "exact_same_one_head_per_common_block",
                "depth_profile": _profile(tuple(selected)).tolist(),
                "block_histogram": {str(block): 1 for block in common_blocks},
                "targets": [
                    _target_record(role, target, scores) for target in selected
                ],
            }

    payload = {
        "schema_version": 1,
        "experiment": "wan_dit_head_role_dose_control",
        "selection_seed": int(args.seed),
        "source_report": {
            **source,
            "sha256_recomputed": _sha256(report),
        },
        "roles": list(ROLES),
        "approximate_matching": {
            "k": int(args.k),
            "replicates": int(args.replicates),
            "candidate_pool": int(args.candidate_pool),
            "profile": (
                "sorted normalized block depths, five six-block depth-band "
                "proportions, and normalized mean/std/q25/median/q75"
            ),
            "limitation": (
                "S/T/C do not share enough blocks for exact k=8 matching; "
                "residual depth imbalance must remain a statistical covariate."
            ),
        },
        "exact_matching": {
            "k": 5,
            "replicates": int(args.exact_common_block_replicates),
            "maximum_unique_replicates_shared_by_all_roles": maximum_unique_exact,
            "common_blocks": common_blocks,
        },
        "subsets": subsets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
