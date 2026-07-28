#!/usr/bin/env python3
"""Export immutable raw head features and reproducible derived role scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from classify_fulltoken_moving_heads import ROLES, _rank


EXPECTED_BLOCKS = 30
EXPECTED_HEADS = 24
EXPECTED_STEPS = (5, 15, 25, 35)
FULL_NAMES = (
    "entropy",
    "same_frame_mass",
    "local_enrichment",
    "context_enrichment",
    "history_bias",
    "mean_time_distance",
    "aligned_enrichment",
    "exact_self_mass",
)
OBJECT_NAMES = (
    "entropy",
    "same_frame_mass",
    "context_enrichment",
    "history_bias",
    "mean_time_distance",
    "trajectory_enrichment",
    "shift_enrichment",
    "shuffle_enrichment",
    "trajectory_selectivity_log2",
    "fixed_position_enrichment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--seed-snapshot", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-visible-times", type=int, default=8)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.8)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _cases(path: Path) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        case = Path(stripped).expanduser().resolve().stem
        if case not in seen:
            seen.add(case)
            values.append(case)
    if not values:
        raise ValueError(f"No cases found in {path}")
    return values


def _trajectory_validity(
    query_map: dict[str, Any],
    case: str,
    *,
    minimum_visible_times: int,
    minimum_valid_ratio: float,
) -> tuple[bool, int, float]:
    item = query_map[case]
    coords = item.get("query_coords_per_time", [])
    visible_times = sum(bool(entries) for entries in coords)
    valid_ratio = float(item.get("track_quality", {}).get("valid_ratio", 0.0))
    valid = (
        visible_times >= minimum_visible_times
        and valid_ratio >= minimum_valid_ratio
    )
    return valid, int(visible_times), valid_ratio


def _load_case(
    seed_root: Path,
    model: str,
    case: str,
) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(
        seed_root.glob(
            f"block*/matrices/{model}/{case}/block*_fulltoken_moving.npz"
        )
    )
    if len(files) != EXPECTED_BLOCKS:
        raise RuntimeError(
            f"{model}/{seed_root.name}/{case}: "
            f"found {len(files)}/{EXPECTED_BLOCKS} blocks"
        )
    full_blocks: list[np.ndarray] = []
    object_blocks: list[np.ndarray] = []
    observed_blocks: list[int] = []
    for path in files:
        block = int(path.name.split("_", 1)[0].replace("block", ""))
        observed_blocks.append(block)
        with np.load(path, allow_pickle=False) as data:
            steps = tuple(int(value) for value in data["steps_one_based"])
            full_names = tuple(data["full_feature_names"].astype(str))
            object_names = tuple(data["object_feature_names"].astype(str))
            if steps != EXPECTED_STEPS:
                raise RuntimeError(f"Unexpected steps {steps}: {path}")
            if full_names != FULL_NAMES or object_names != OBJECT_NAMES:
                raise RuntimeError(f"Feature schema mismatch: {path}")
            full = data["full_features"].astype(np.float32)
            object_by_time = data[
                "object_features_by_query_time"
            ].astype(np.float32)
        if full.shape != (4, EXPECTED_HEADS, len(FULL_NAMES)):
            raise RuntimeError(f"Unexpected full feature shape {full.shape}: {path}")
        if object_by_time.shape != (
            4,
            EXPECTED_HEADS,
            13,
            len(OBJECT_NAMES),
        ):
            raise RuntimeError(
                f"Unexpected object feature shape {object_by_time.shape}: {path}"
            )
        full_blocks.append(full)
        object_blocks.append(object_by_time)
    if observed_blocks != list(range(EXPECTED_BLOCKS)):
        raise RuntimeError(f"Block ids are not 0..29 for {model}/{case}")

    full_array = np.stack(full_blocks, axis=1)
    object_by_time = np.stack(object_blocks, axis=1)
    with np.errstate(invalid="ignore"):
        object_array = np.nanmean(object_by_time, axis=3)
    for name in ("context_enrichment", "history_bias"):
        index = OBJECT_NAMES.index(name)
        with np.errstate(invalid="ignore"):
            object_array[..., index] = np.nanmean(
                object_by_time[..., 2:, index],
                axis=3,
            )
    return full_array, object_array


def _identifiers(
    *,
    model: str,
    seed: int,
    case: str,
    denoise_step: int,
) -> dict[str, np.ndarray]:
    rows = EXPECTED_BLOCKS * EXPECTED_HEADS
    return {
        "model": np.repeat(model, rows),
        "source_case": np.repeat(case, rows),
        "seed": np.full(rows, seed, dtype=np.int32),
        "denoise_step": np.full(rows, denoise_step, dtype=np.int16),
        "block": np.repeat(
            np.arange(EXPECTED_BLOCKS, dtype=np.int8),
            EXPECTED_HEADS,
        ),
        "head": np.tile(
            np.arange(EXPECTED_HEADS, dtype=np.int8),
            EXPECTED_BLOCKS,
        ),
    }


def _score_arrays(
    full: np.ndarray,
    obj: np.ndarray,
    *,
    trajectory_valid: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    full_index = {name: index for index, name in enumerate(FULL_NAMES)}
    object_index = {name: index for index, name in enumerate(OBJECT_NAMES)}
    ranks = {
        "rank_local_enrichment": _rank(
            full[..., full_index["local_enrichment"]]
        ),
        "rank_same_frame_mass": _rank(
            full[..., full_index["same_frame_mass"]]
        ),
        "rank_trajectory_selectivity_log2": _rank(
            obj[..., object_index["trajectory_selectivity_log2"]]
        ),
        "rank_trajectory_enrichment": _rank(
            obj[..., object_index["trajectory_enrichment"]]
        ),
        "rank_object_mean_time_distance": _rank(
            obj[..., object_index["mean_time_distance"]]
        ),
        "rank_fixed_position_enrichment": _rank(
            obj[..., object_index["fixed_position_enrichment"]]
        ),
        "rank_aligned_enrichment": _rank(
            full[..., full_index["aligned_enrichment"]]
        ),
        "rank_object_context_enrichment": _rank(
            obj[..., object_index["context_enrichment"]]
        ),
        "rank_full_context_enrichment": _rank(
            full[..., full_index["context_enrichment"]]
        ),
        "rank_object_history_bias": _rank(
            obj[..., object_index["history_bias"]]
        ),
        "rank_full_entropy": _rank(full[..., full_index["entropy"]]),
        "rank_full_mean_time_distance": _rank(
            full[..., full_index["mean_time_distance"]]
        ),
        "rank_negative_same_frame_mass": _rank(
            -full[..., full_index["same_frame_mass"]]
        ),
    }
    scores = np.stack(
        (
            0.55 * ranks["rank_local_enrichment"]
            + 0.45 * ranks["rank_same_frame_mass"],
            0.55 * ranks["rank_trajectory_selectivity_log2"]
            + 0.25 * ranks["rank_trajectory_enrichment"]
            + 0.20 * ranks["rank_object_mean_time_distance"],
            0.75 * ranks["rank_fixed_position_enrichment"]
            + 0.25 * ranks["rank_aligned_enrichment"],
            0.55 * ranks["rank_object_context_enrichment"]
            + 0.25 * ranks["rank_full_context_enrichment"]
            + 0.20 * ranks["rank_object_history_bias"],
            0.60 * ranks["rank_full_entropy"]
            + 0.25 * ranks["rank_full_mean_time_distance"]
            + 0.15 * ranks["rank_negative_same_frame_mass"],
        ),
        axis=-1,
    ).astype(np.float32)
    if not trajectory_valid:
        scores[..., ROLES.index("T")] = -np.inf
        scores[..., ROLES.index("P")] = -np.inf
    scores = np.where(np.isfinite(scores), scores, -np.inf).astype(np.float32)
    return ranks, scores


def _write_seed(
    *,
    model: str,
    seed: int,
    cases: list[str],
    capture_root: Path,
    query_root: Path,
    output: Path,
    minimum_visible_times: int,
    minimum_valid_ratio: float,
) -> dict[str, Any]:
    seed_name = f"seed-{seed:06d}"
    query_path = query_root / model / seed_name / "query_map.json"
    query_map = json.loads(query_path.read_text(encoding="utf-8"))["cases"]
    if not set(cases).issubset(query_map):
        missing = sorted(set(cases) - set(query_map))
        raise RuntimeError(f"Query map lacks cases: {missing}")
    seed_root = capture_root / model / seed_name
    raw_chunks: list[pd.DataFrame] = []
    rank_chunks: list[pd.DataFrame] = []
    sample_chunks: list[pd.DataFrame] = []

    for case in cases:
        full_steps, object_steps = _load_case(seed_root, model, case)
        valid, visible_times, valid_ratio = _trajectory_validity(
            query_map,
            case,
            minimum_visible_times=minimum_visible_times,
            minimum_valid_ratio=minimum_valid_ratio,
        )
        step_scores: list[np.ndarray] = []
        for step_index, denoise_step in enumerate(EXPECTED_STEPS):
            full = full_steps[step_index]
            obj = object_steps[step_index]
            ids = _identifiers(
                model=model,
                seed=seed,
                case=case,
                denoise_step=denoise_step,
            )
            ids.update(
                {
                    "trajectory_valid": np.full(
                        EXPECTED_BLOCKS * EXPECTED_HEADS,
                        valid,
                        dtype=bool,
                    ),
                    "trajectory_visible_times": np.full(
                        EXPECTED_BLOCKS * EXPECTED_HEADS,
                        visible_times,
                        dtype=np.int8,
                    ),
                    "trajectory_valid_ratio": np.full(
                        EXPECTED_BLOCKS * EXPECTED_HEADS,
                        valid_ratio,
                        dtype=np.float32,
                    ),
                }
            )
            raw = dict(ids)
            for index, name in enumerate(FULL_NAMES):
                raw[f"full_{name}_raw"] = full[..., index].reshape(-1)
            for index, name in enumerate(OBJECT_NAMES):
                raw[f"object_{name}_raw"] = obj[..., index].reshape(-1)
            raw_chunks.append(pd.DataFrame(raw))

            ranks, scores = _score_arrays(
                full,
                obj,
                trajectory_valid=valid,
            )
            derived = dict(ids)
            for name, values in ranks.items():
                derived[name] = values.reshape(-1)
            for role_index, role in enumerate(ROLES):
                derived[f"score_{role}"] = scores[..., role_index].reshape(-1)
            rank_chunks.append(pd.DataFrame(derived))
            step_scores.append(scores)

        score_steps = np.stack(step_scores)
        mean_scores = score_steps.mean(axis=0)
        winner = mean_scores.argmax(axis=-1)
        sorted_scores = np.sort(mean_scores, axis=-1)
        margin = sorted_scores[..., -1] - sorted_scores[..., -2]
        step_winner = score_steps.argmax(axis=-1)
        consistency = np.mean(step_winner == winner[None, ...], axis=0)
        labels = np.asarray(ROLES, dtype="<U1")[winner]
        labels[(margin < 0.08) | (consistency < 0.75)] = "M"
        sample_ids = _identifiers(
            model=model,
            seed=seed,
            case=case,
            denoise_step=-1,
        )
        sample_ids.pop("denoise_step")
        sample_ids.update(
            {
                "trajectory_valid": np.full(
                    EXPECTED_BLOCKS * EXPECTED_HEADS,
                    valid,
                    dtype=bool,
                ),
                "role": labels.reshape(-1),
                "margin": margin.reshape(-1).astype(np.float32),
                "step_consistency": consistency.reshape(-1).astype(np.float32),
            }
        )
        for role_index, role in enumerate(ROLES):
            sample_ids[f"mean_score_{role}"] = mean_scores[
                ..., role_index
            ].reshape(-1)
        sample_chunks.append(pd.DataFrame(sample_ids))

    raw_frame = pd.concat(raw_chunks, ignore_index=True)
    rank_frame = pd.concat(rank_chunks, ignore_index=True)
    sample_frame = pd.concat(sample_chunks, ignore_index=True)
    expected_raw = len(cases) * len(EXPECTED_STEPS) * EXPECTED_BLOCKS * EXPECTED_HEADS
    expected_samples = len(cases) * EXPECTED_BLOCKS * EXPECTED_HEADS
    if len(raw_frame) != expected_raw or len(rank_frame) != expected_raw:
        raise RuntimeError("Raw/derived row count mismatch")
    if len(sample_frame) != expected_samples:
        raise RuntimeError("Sample classification row count mismatch")
    key = ["model", "source_case", "seed", "denoise_step", "block", "head"]
    if raw_frame.duplicated(key).any() or rank_frame.duplicated(key).any():
        raise RuntimeError("Duplicate raw/derived primary keys")
    sample_key = ["model", "source_case", "seed", "block", "head"]
    if sample_frame.duplicated(sample_key).any():
        raise RuntimeError("Duplicate sample classification primary keys")

    paths = {}
    for kind, frame in (
        ("raw", raw_frame),
        ("ranks_scores", rank_frame),
        ("sample_roles", sample_frame),
    ):
        path = output / kind / f"model={model}" / f"{seed_name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        frame.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        temporary.replace(path)
        paths[kind] = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(frame),
            "bytes": path.stat().st_size,
        }
    return {
        "model": model,
        "seed": seed,
        "cases": len(cases),
        "files": paths,
    }


def main() -> None:
    args = parse_args()
    capture_root = args.capture_root.expanduser().resolve()
    query_root = args.query_root.expanduser().resolve()
    seed_snapshot_path = args.seed_snapshot.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_snapshot = json.loads(seed_snapshot_path.read_text(encoding="utf-8"))
    cases = _cases(input_list)
    started = time.time()
    records: list[dict[str, Any]] = []
    selected_models = set(args.models or seed_snapshot)
    unknown_models = selected_models - set(seed_snapshot)
    if unknown_models:
        raise ValueError(f"Models absent from seed snapshot: {sorted(unknown_models)}")
    selected_seeds = None if args.seeds is None else set(args.seeds)
    for model, seed_values in seed_snapshot.items():
        if model not in selected_models:
            continue
        for seed_value in seed_values:
            seed = int(seed_value)
            if selected_seeds is not None and seed not in selected_seeds:
                continue
            print(f"[raw-export] {model} seed={seed}", flush=True)
            records.append(
                _write_seed(
                    model=model,
                    seed=seed,
                    cases=cases,
                    capture_root=capture_root,
                    query_root=query_root,
                    output=output,
                    minimum_visible_times=int(args.minimum_visible_times),
                    minimum_valid_ratio=float(args.minimum_valid_ratio),
                )
            )
            _atomic_json(
                output / "classification_manifest.partial.json",
                {
                    "status": "running",
                    "completed_partitions": records,
                },
            )
    if not records:
        raise ValueError("Model/seed filters selected no capture partitions")

    schema = {
        "schema_version": 1,
        "primary_key": [
            "model",
            "source_case",
            "seed",
            "denoise_step",
            "block",
            "head",
        ],
        "rank_scope": "within model/source_case/seed/denoise_step over 30x24 heads",
        "rank_direction": "ascending raw value maps to ascending [0,1] rank",
        "rank_ties": "stable input order; no average-rank tie correction",
        "score_formula_version": "fulltoken-moving-role-v1",
        "raw_full_feature_names": list(FULL_NAMES),
        "raw_object_feature_names": list(OBJECT_NAMES),
        "object_aggregation": {
            "default": "nanmean over all 13 query times",
            "context_enrichment": "nanmean over predicted query times 2..12",
            "history_bias": "nanmean over predicted query times 2..12",
        },
        "legacy_compact_capture_limitations": {
            "raw_final_feature_values_available": True,
            "raw_object_values_by_query_time_retained_in_source_npz": True,
            "feature_numerator_denominator_components_available": False,
            "note": (
                "The existing compact capture predates component-level export. "
                "Do not claim that enrichment numerators/denominators can be "
                "reconstructed from these Parquet files."
            ),
        },
    }
    _atomic_json(output / "raw_feature_schema.json", schema)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "capture_root": str(capture_root),
        "query_root": str(query_root),
        "seed_snapshot": {
            "path": str(seed_snapshot_path),
            "sha256": _sha256(seed_snapshot_path),
        },
        "input_list": {
            "path": str(input_list),
            "sha256": _sha256(input_list),
            "cases": cases,
        },
        "minimum_visible_times": int(args.minimum_visible_times),
        "minimum_valid_ratio": float(args.minimum_valid_ratio),
        "partitions": records,
        "elapsed_seconds": time.time() - started,
        "schema": {
            "path": str(output / "raw_feature_schema.json"),
            "sha256": _sha256(output / "raw_feature_schema.json"),
        },
    }
    _atomic_json(output / "classification_manifest.json", manifest)
    (output / "classification_manifest.partial.json").unlink(missing_ok=True)
    print(output / "classification_manifest.json")


if __name__ == "__main__":
    main()
