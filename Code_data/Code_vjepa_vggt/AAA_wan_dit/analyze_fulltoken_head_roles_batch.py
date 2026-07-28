#!/usr/bin/env python3
"""Aggregate 50-seed full-token head roles and select representative QK captures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from classify_fulltoken_moving_heads import (
    MODEL_NAMES,
    ROLE_COLORS,
    ROLES,
    _classify,
)


ROLE_ORDER = (*ROLES, "M")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the default <output_root>/analysis directory.",
    )
    parser.add_argument(
        "--seed-snapshot",
        type=Path,
        help="JSON mapping model names to the exact completed seeds to analyze.",
    )
    return parser.parse_args()


def _load_sample(seed_root: Path, model: str, case: str) -> dict[str, Any]:
    files = sorted(
        seed_root.glob(
            f"block*/matrices/{model}/{case}/block*_fulltoken_moving.npz"
        )
    )
    if len(files) != 30:
        raise RuntimeError(f"{model}/{case}: found {len(files)}/30 blocks")
    blocks = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            blocks.append(
                {
                    "block": int(path.name.split("_", 1)[0].replace("block", "")),
                    "path": path,
                    "steps": data["steps_one_based"].astype(int),
                    "full_names": data["full_feature_names"].astype(str).tolist(),
                    "object_names": data["object_feature_names"].astype(str).tolist(),
                    "full": data["full_features"].astype(np.float32),
                    "object_by_time": data[
                        "object_features_by_query_time"
                    ].astype(np.float32),
                }
            )
    blocks.sort(key=lambda item: item["block"])
    if [item["block"] for item in blocks] != list(range(30)):
        raise RuntimeError(f"{model}/{case}: block ids are not 0..29")
    steps = blocks[0]["steps"]
    if any(not np.array_equal(item["steps"], steps) for item in blocks):
        raise RuntimeError(f"{model}/{case}: inconsistent denoise steps")
    full = np.stack([item["full"] for item in blocks], axis=1)
    object_by_time = np.stack(
        [item["object_by_time"] for item in blocks], axis=1
    )
    with np.errstate(invalid="ignore"):
        obj = np.nanmean(object_by_time, axis=3)
    object_names = blocks[0]["object_names"]
    for name in ("context_enrichment", "history_bias"):
        feature_index = object_names.index(name)
        with np.errstate(invalid="ignore"):
            obj[..., feature_index] = np.nanmean(
                object_by_time[..., 2:, feature_index], axis=3
            )
    return {
        "model": model,
        "case": case,
        "steps": steps,
        "paths": [item["path"] for item in blocks],
        "full_names": blocks[0]["full_names"],
        "object_names": object_names,
        "full": full,
        "object": obj,
    }


def _trajectory_validity(
    item: dict[str, Any],
    *,
    minimum_times: int,
    minimum_ratio: float,
) -> tuple[bool, int, float]:
    coords = item.get("query_coords_per_time", [])
    visible_times = sum(bool(entries) for entries in coords)
    valid_ratio = float(item.get("track_quality", {}).get("valid_ratio", 0.0))
    return (
        visible_times >= minimum_times and valid_ratio >= minimum_ratio,
        visible_times,
        valid_ratio,
    )


def _render_grid(labels: np.ndarray, title: str, output: Path) -> None:
    role_to_int = {role: index for index, role in enumerate(ROLE_ORDER)}
    values = np.vectorize(role_to_int.get)(labels)
    fig, axis = plt.subplots(figsize=(13.5, 7.0), constrained_layout=True)
    axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([ROLE_COLORS[role] for role in ROLE_ORDER]),
        vmin=-0.5,
        vmax=len(ROLE_ORDER) - 0.5,
    )
    axis.set_xlabel("Head index")
    axis.set_ylabel("DiT block")
    axis.set_xticks(np.arange(labels.shape[1]))
    axis.set_yticks(np.arange(labels.shape[0]))
    axis.set_title(title)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=9,
            color=ROLE_COLORS[role],
            label=role,
        )
        for role in ROLE_ORDER
    ]
    axis.legend(
        handles=handles,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _bootstrap_support(
    agreements: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_count, head_count = agreements.shape
    values = np.empty((resamples, head_count), dtype=np.float32)
    for start in range(0, resamples, 20):
        stop = min(start + 20, resamples)
        indices = rng.integers(0, sample_count, size=(stop - start, sample_count))
        values[start:stop] = agreements[indices].mean(axis=1)
    return (
        np.percentile(values, 2.5, axis=0).astype(np.float32),
        np.percentile(values, 97.5, axis=0).astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    prerequisite = (
        Path(config["storage"]["prerequisite_root"]).expanduser().resolve()
    )
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.seed_snapshot:
        seed_snapshot_path = args.seed_snapshot.expanduser().resolve()
        seed_snapshot = json.loads(
            seed_snapshot_path.read_text(encoding="utf-8")
        )
    else:
        seed_snapshot_path = None
        seed_snapshot = {
            model: [int(value) for value in config["seeds"]]
            for model in config["models"]
        }
    cases = [
        Path(line.strip()).expanduser().resolve().stem
        for line in (root / "input_lists" / "test5_unique20.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    minimum_times = int(
        config["trajectory_validity"]["minimum_visible_latent_times"]
    )
    minimum_ratio = float(
        config["trajectory_validity"]["minimum_track_valid_ratio"]
    )
    bootstrap_resamples = int(config["analysis"]["bootstrap_resamples"])
    bootstrap_seed = int(config["analysis"]["bootstrap_seed"])
    representative_seed = int(config["seeds"][0])
    selection: dict[str, dict[str, dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "config": str(config_path),
        "seed_snapshot": (
            str(seed_snapshot_path) if seed_snapshot_path is not None else None
        ),
        "trajectory_validity": config["trajectory_validity"],
        "models": {},
    }
    csv_rows = []

    for model in config["models"]:
        model_seeds = [int(value) for value in seed_snapshot[model]]
        if not model_seeds:
            raise ValueError(f"seed snapshot contains no seeds for {model}")
        sample_records = []
        for seed in model_seeds:
            query_path = (
                prerequisite
                / "query_maps"
                / model
                / f"seed-{seed:06d}"
                / "query_map.json"
            )
            query_map = json.loads(
                query_path.read_text(encoding="utf-8")
            )["cases"]
            seed_root = root / "capture" / model / f"seed-{seed:06d}"
            for case in cases:
                valid, visible_times, valid_ratio = _trajectory_validity(
                    query_map[case],
                    minimum_times=minimum_times,
                    minimum_ratio=minimum_ratio,
                )
                classified = _classify(
                    _load_sample(seed_root, model, case),
                    trajectory_valid=valid,
                )
                sample_records.append(
                    {
                        "seed": seed,
                        "case": case,
                        "trajectory_valid": valid,
                        "visible_times": visible_times,
                        "track_valid_ratio": valid_ratio,
                        "labels": classified["labels"],
                        "scores": classified["scores"],
                    }
                )

        labels = np.stack([item["labels"] for item in sample_records])
        scores = np.stack([item["scores"] for item in sample_records]).astype(
            np.float32
        )
        valid_mask = np.asarray(
            [item["trajectory_valid"] for item in sample_records], dtype=bool
        )
        finite_scores = np.where(np.isfinite(scores), scores, np.nan)
        with np.errstate(invalid="ignore"):
            mean_scores = np.nanmean(finite_scores, axis=0)
        winner = np.nanargmax(mean_scores, axis=-1)
        aggregate_labels = np.asarray(ROLES, dtype="<U1")[winner]
        sorted_scores = np.sort(mean_scores, axis=-1)
        margin = sorted_scores[..., -1] - sorted_scores[..., -2]
        agreement = labels == aggregate_labels[None, ...]
        trajectory_role = np.isin(aggregate_labels, ("T", "P"))
        support_all = agreement.mean(axis=0)
        support_valid = agreement[valid_mask].mean(axis=0)
        support = np.where(trajectory_role, support_valid, support_all)
        aggregate_labels[
            (margin < float(config["analysis"]["winner_margin_threshold"]))
            | (support < 0.50)
        ] = "M"
        support_all_low, support_all_high = _bootstrap_support(
            agreement.reshape(len(sample_records), -1),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        support_valid_low, support_valid_high = _bootstrap_support(
            agreement[valid_mask].reshape(int(valid_mask.sum()), -1),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + 1,
        )
        support_low = np.where(
            trajectory_role,
            support_valid_low.reshape(30, 24),
            support_all_low.reshape(30, 24),
        )
        support_high = np.where(
            trajectory_role,
            support_valid_high.reshape(30, 24),
            support_all_high.reshape(30, 24),
        )
        _render_grid(
            aggregate_labels,
            (
                f"{len(model_seeds)} seeds x 20 cases | aggregate head roles"
            ),
            output / f"{model}_aggregate_roles.png",
        )

        model_rows = []
        for block in range(30):
            for head in range(24):
                row = {
                    "model": model,
                    "block": block,
                    "head": head,
                    "role": str(aggregate_labels[block, head]),
                    "margin": float(margin[block, head]),
                    "support": float(support[block, head]),
                    "support_ci95_low": float(support_low[block, head]),
                    "support_ci95_high": float(support_high[block, head]),
                    "valid_trajectory_samples": int(valid_mask.sum()),
                    "total_samples": len(sample_records),
                }
                for role_index, role in enumerate(ROLES):
                    row[f"score_{role}"] = float(
                        mean_scores[block, head, role_index]
                    )
                model_rows.append(row)
                csv_rows.append(row)

        model_selection: dict[str, dict[str, Any]] = {}
        representative = [
            item for item in sample_records if item["seed"] == representative_seed
        ]
        top_count = int(config["analysis"]["selected_raw_qk_heads_per_role"])
        for role_index, role in enumerate(ROLES):
            role_scores = mean_scores[..., role_index].copy()
            role_scores[aggregate_labels != role] = -np.inf
            flat_indices = np.argsort(role_scores.reshape(-1))[::-1]
            flat_indices = [
                int(index)
                for index in flat_indices
                if np.isfinite(role_scores.reshape(-1)[index])
            ][:top_count]
            if len(flat_indices) != top_count:
                raise RuntimeError(
                    f"{model}: found {len(flat_indices)}/{top_count} stable {role} heads"
                )
            for rank, flat_index in enumerate(flat_indices, start=1):
                block, head = np.unravel_index(flat_index, role_scores.shape)
                candidates = [
                    item
                    for item in representative
                    if role not in ("T", "P") or item["trajectory_valid"]
                ]
                best = max(
                    candidates,
                    key=lambda item: float(item["scores"][block, head, role_index]),
                )
                role_key = f"{role}_rank{rank}"
                model_selection.setdefault(best["case"], {"roles": {}})["roles"][
                    role_key
                ] = {
                    "block": int(block),
                    "head": int(head),
                    "aggregate_score": float(role_scores[block, head]),
                    "representative_seed": representative_seed,
                }
        selection[model] = model_selection
        report["models"][model] = {
            "seeds": model_seeds,
            "seed_count": len(model_seeds),
            "samples": len(sample_records),
            "valid_trajectory_samples": int(valid_mask.sum()),
            "aggregate_heads": model_rows,
            "role_counts": {
                role: int(np.sum(aggregate_labels == role)) for role in ROLE_ORDER
            },
        }

    with (output / "aggregate_heads.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    np.savez_compressed(
        output / "aggregate_arrays.npz",
        model_names=np.asarray(config["models"]),
    )
    (output / "head_role_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selection_payload = {
        "source": str(output / "head_role_report.json"),
        "policy": (
            "aggregate top-3 per S/T/P/C/G role; representative case chosen at "
            f"seed {representative_seed}"
        ),
        "representative_seed": representative_seed,
        "samples": selection,
    }
    (output / "selected_qk_selection.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output / "head_role_report.json")
    print(output / "selected_qk_selection.json")


if __name__ == "__main__":
    main()
