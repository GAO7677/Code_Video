#!/usr/bin/env python3
"""Analyze paired moving/anchor head roles across cases and random seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_multiblock_ball_query_heads import ROLE_LABELS, _role_scores
from moving_query_attention import FEATURE_NAMES


ROLES = tuple(ROLE_LABELS)
PROTOCOLS = ("moving", "anchor_t2")
BLOCKS = tuple(range(30))
HEADS = tuple(range(24))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _classify(features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aggregate = {name: values.mean(axis=0) for name, values in features.items()}
    scores = _role_scores(aggregate)
    matrix = np.stack([scores[role] for role in ROLES], axis=1)
    order = np.argsort(matrix, axis=1)
    primary = order[:, -1]
    margin = (
        np.take_along_axis(matrix, order[:, -1:], axis=1)[:, 0]
        - np.take_along_axis(matrix, order[:, -2:-1], axis=1)[:, 0]
    )
    step_labels = []
    for step in range(next(iter(features.values())).shape[0]):
        step_scores = _role_scores(
            {name: values[step] for name, values in features.items()}
        )
        step_matrix = np.stack([step_scores[role] for role in ROLES], axis=1)
        step_labels.append(step_matrix.argmax(axis=1))
    step_labels = np.stack(step_labels, axis=0)
    step_consistency = np.asarray(
        [
            np.mean(step_labels[:, head] == primary[head])
            for head in HEADS
        ]
    )
    return primary, margin, step_consistency


def _modal(labels: list[str]) -> tuple[str, float]:
    role, count = Counter(labels).most_common(1)[0]
    return role, count / len(labels)


def _bootstrap_delta(
    transitions: np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    total = int(transitions.sum())
    if total == 0:
        return float("nan"), float("nan")
    probabilities = transitions.reshape(-1).astype(np.float64) / total
    draws = rng.multinomial(total, probabilities, size=resamples).reshape(
        resamples, len(ROLES), len(ROLES)
    )
    moving_consistency = draws.sum(axis=2).max(axis=1) / total
    anchor_consistency = draws.sum(axis=1).max(axis=1) / total
    delta = anchor_consistency - moving_consistency
    low, high = np.percentile(delta, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_seeds = tuple(int(seed) for seed in config["seed_sampling"]["seeds"])
    models = tuple(config["models"])
    samples: list[dict[str, Any]] = []

    for model in models:
        for seed in expected_seeds:
            seed_name = f"seed-{seed:06d}"
            state_path = root / "state" / model / f"{seed_name}.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "complete":
                raise RuntimeError(f"incomplete state: {state_path}")
            capture_root = root / "capture" / model / seed_name
            case_root = capture_root / "block00" / "matrices" / model
            cases = sorted(path.name for path in case_root.iterdir() if path.is_dir())
            if len(cases) != 20:
                raise RuntimeError(f"{model}/{seed_name}: expected 20 cases")
            for case in cases:
                block_results: dict[int, dict[str, Any]] = {}
                for block in BLOCKS:
                    case_dir = (
                        capture_root
                        / f"block{block:02d}"
                        / "matrices"
                        / model
                        / case
                    )
                    summary = json.loads(
                        (case_dir / "summary.json").read_text(encoding="utf-8")
                    )
                    with np.load(case_dir / summary["feature_npz"]) as arrays:
                        anchor_valid = bool(arrays["anchor_t2_valid"])
                        block_results[block] = {
                            "anchor_valid": anchor_valid,
                            "features": {
                                protocol: {
                                    feature: arrays[
                                        f"{protocol}__{feature}"
                                    ].astype(np.float64)
                                    for feature in FEATURE_NAMES
                                }
                                for protocol in PROTOCOLS
                                if protocol == "moving" or anchor_valid
                            },
                        }
                for block, result in block_results.items():
                    for protocol, features in result["features"].items():
                        primary, margin, step_consistency = _classify(features)
                        for head in HEADS:
                            samples.append(
                                {
                                    "model": model,
                                    "seed": seed,
                                    "case": case,
                                    "block": block,
                                    "head": head,
                                    "protocol": protocol,
                                    "role": ROLES[int(primary[head])],
                                    "margin": float(margin[head]),
                                    "step_consistency": float(
                                        step_consistency[head]
                                    ),
                                }
                            )

    _write_csv(output / "sample_roles.csv", samples)
    grouped: dict[
        tuple[str, str, int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in samples:
        grouped[
            (
                row["model"],
                row["protocol"],
                int(row["block"]),
                int(row["head"]),
            )
        ].append(row)

    stability_rows = []
    for key, rows in sorted(grouped.items()):
        model, protocol, block, head = key
        joint_role, joint_consistency = _modal([row["role"] for row in rows])
        case_groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            case_groups[row["case"]].append(row["role"])
        case_modes = []
        within_case_values = []
        for labels in case_groups.values():
            role, consistency = _modal(labels)
            case_modes.append(role)
            within_case_values.append(consistency)
        across_role, across_consistency = _modal(case_modes)
        stability_rows.append(
            {
                "model": model,
                "protocol": protocol,
                "block": block,
                "head": head,
                "valid_samples": len(rows),
                "valid_cases": len(case_groups),
                "joint_dominant_role": joint_role,
                "joint_consistency": joint_consistency,
                "mean_within_case_seed_consistency": float(
                    np.mean(within_case_values)
                ),
                "across_case_dominant_role": across_role,
                "across_case_consistency_after_seed_mode": across_consistency,
                "mean_step_consistency": float(
                    np.mean([row["step_consistency"] for row in rows])
                ),
                "median_role_margin": float(
                    np.median([row["margin"] for row in rows])
                ),
            }
        )
    _write_csv(output / "protocol_stability.csv", stability_rows)

    sample_lookup = {
        (
            row["model"],
            row["seed"],
            row["case"],
            row["block"],
            row["head"],
            row["protocol"],
        ): row
        for row in samples
    }
    stability_lookup = {
        (
            row["model"],
            row["protocol"],
            row["block"],
            row["head"],
        ): row
        for row in stability_rows
    }
    bootstrap = config["analysis"]["paired_comparison"]
    rng = np.random.default_rng(int(bootstrap["bootstrap_seed"]))
    paired_rows = []
    transition_payload = {}
    for model in models:
        transition_payload[model] = {}
        for block in BLOCKS:
            transition_payload[model][str(block)] = {}
            for head in HEADS:
                transitions = np.zeros((len(ROLES), len(ROLES)), dtype=np.int64)
                paired_count = 0
                for seed in expected_seeds:
                    prefix = (model, seed)
                    moving_keys = [
                        key
                        for key in sample_lookup
                        if key[:2] == prefix
                        and key[3] == block
                        and key[4] == head
                        and key[5] == "moving"
                    ]
                    for moving_key in moving_keys:
                        anchor_key = (*moving_key[:5], "anchor_t2")
                        if anchor_key not in sample_lookup:
                            continue
                        moving_role = sample_lookup[moving_key]["role"]
                        anchor_role = sample_lookup[anchor_key]["role"]
                        transitions[
                            ROLES.index(moving_role), ROLES.index(anchor_role)
                        ] += 1
                        paired_count += 1
                moving_stability = stability_lookup[
                    (model, "moving", block, head)
                ]
                anchor_stability = stability_lookup[
                    (model, "anchor_t2", block, head)
                ]
                low, high = _bootstrap_delta(
                    transitions,
                    resamples=int(bootstrap["bootstrap_resamples"]),
                    rng=rng,
                )
                paired_rows.append(
                    {
                        "model": model,
                        "block": block,
                        "head": head,
                        "paired_samples": paired_count,
                        "role_agreement": float(
                            np.trace(transitions) / max(paired_count, 1)
                        ),
                        "moving_joint_consistency": moving_stability[
                            "joint_consistency"
                        ],
                        "anchor_joint_consistency": anchor_stability[
                            "joint_consistency"
                        ],
                        "anchor_minus_moving_consistency": anchor_stability[
                            "joint_consistency"
                        ]
                        - moving_stability["joint_consistency"],
                        "bootstrap_delta_ci95_low": low,
                        "bootstrap_delta_ci95_high": high,
                    }
                )
                transition_payload[model][str(block)][str(head)] = {
                    "roles": list(ROLES),
                    "moving_rows_anchor_columns": transitions.tolist(),
                }
    _write_csv(output / "paired_protocol_comparison.csv", paired_rows)
    (output / "role_transitions.json").write_text(
        json.dumps(transition_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    sections = []
    for model in models:
        current = [row for row in paired_rows if row["model"] == model]
        sections.append(
            f"""## {model}

- Mean paired role agreement: {np.mean([row['role_agreement'] for row in current]):.3f}
- Mean anchor-minus-moving consistency: {np.mean([row['anchor_minus_moving_consistency'] for row in current]):+.3f}
- Heads with positive/negative delta: {sum(row['anchor_minus_moving_consistency'] > 0 for row in current)}/{sum(row['anchor_minus_moving_consistency'] < 0 for row in current)}
"""
        )
    report = f"""# Paired Moving vs Anchor-t2 Head Stability

Models are analyzed independently over 20 cases and 50 sampled seeds. Anchor
statistics exclude model-case-seed samples where the generated video has no
visible object at latent t=2. Roles are descriptive relative specializations,
not causal proof.

{chr(10).join(sections)}
"""
    (output / "paired_query_head_stability.md").write_text(
        report, encoding="utf-8"
    )
    print(output / "paired_query_head_stability.md")


if __name__ == "__main__":
    main()
