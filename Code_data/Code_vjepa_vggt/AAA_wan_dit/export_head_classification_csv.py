#!/usr/bin/env python3
"""Export the common-22 head-role results as validated CSV tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_AGGREGATE = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/"
    "partial_analysis/snapshot_20260728T0245Z/common22/aggregate_heads.csv"
)
DEFAULT_S_SPLIT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_feature_split/configs/"
    "s_feature_split_subsets.json"
)
DEFAULT_S59 = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "train_xSSC/object_self_attn_lora_experiments/configs/"
    "same_frame_mass_heads_full59.json"
)
DEFAULT_T70 = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "train_xSSC/object_self_attn_lora_experiments/configs/"
    "common_t_heads_full70.json"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/head_classification_csv/common22_public_stable"
)
DEFAULT_RANKING = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_headwise_50case/"
    "cross_model_robust_rankings.csv"
)

MODELS = ("wan_lora", "xssc", "physrvg")
FINAL_CLASSES = ("S", "T", "P", "C", "G", "M")
EXPECTED_FINAL_COUNTS = {"S": 159, "T": 13, "P": 82, "C": 20, "G": 75, "M": 371}
CLASS_DESCRIPTIONS = {
    "S": "spatial/local same-frame head",
    "T": "temporal trajectory-selective head",
    "P": "fixed-position/aligned-position head",
    "C": "object-context/history-context head",
    "G": "global broad-attention head",
    "M": "mixed, uncertain, or cross-model-disagreement head",
}
MODEL_METRICS = (
    "role",
    "margin",
    "support",
    "support_ci95_low",
    "support_ci95_high",
    "valid_trajectory_samples",
    "total_samples",
    "score_S",
    "score_T",
    "score_P",
    "score_C",
    "score_G",
)
S_FEATURE_METRICS = (
    "local_enrichment_raw_mean",
    "local_enrichment_rank_mean",
    "same_frame_mass_raw_mean",
    "same_frame_mass_rank_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--s-split", type=Path, default=DEFAULT_S_SPLIT)
    parser.add_argument("--s59", type=Path, default=DEFAULT_S59)
    parser.add_argument("--t70", type=Path, default=DEFAULT_T70)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_key(item: dict[str, Any]) -> tuple[int, int]:
    return int(item["block"]), int(item["head"])


def head_id(block: int, head: int) -> str:
    return f"B{block:02d}H{head:02d}"


def depth_name(block: int) -> str:
    if block < 10:
        return "early"
    if block < 20:
        return "middle"
    return "late"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_aggregate(path: Path) -> dict[tuple[int, int], dict[str, dict[str, str]]]:
    by_head: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 2160:
        raise ValueError(f"Expected 2160 aggregate rows, found {len(rows)}")
    for row in rows:
        key = head_key(row)
        model = row["model"]
        if model not in MODELS:
            raise ValueError(f"Unexpected model {model!r} at {key}")
        if model in by_head[key]:
            raise ValueError(f"Duplicate aggregate row for {model} {key}")
        by_head[key][model] = row

    expected_positions = {(block, head) for block in range(30) for head in range(24)}
    if set(by_head) != expected_positions:
        missing = sorted(expected_positions - set(by_head))
        extra = sorted(set(by_head) - expected_positions)
        raise ValueError(f"Invalid head grid; missing={missing[:5]}, extra={extra[:5]}")
    for key, model_rows in by_head.items():
        if set(model_rows) != set(MODELS):
            raise ValueError(f"Head {key} does not contain all three models")
    return by_head


def load_s_features(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = read_json(path)
    rows = payload["full_partition"]["heads"]
    result = {head_key(row): row for row in rows}
    if len(rows) != 159 or len(result) != 159:
        raise ValueError(f"Expected 159 unique common-S feature rows, found {len(result)}")
    counts = Counter(row["feature_class"] for row in rows)
    if counts != Counter({"local_dominant": 100, "same_frame_dominant": 59}):
        raise ValueError(f"Unexpected S subtype counts: {dict(counts)}")
    return result


def load_training_subset(path: Path, expected_count: int) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    payload = read_json(path)
    targets = {head_key(row) for row in payload["targets"]}
    if len(targets) != expected_count or int(payload["num_heads"]) != expected_count:
        raise ValueError(
            f"{path} expected {expected_count} unique targets, found {len(targets)}"
        )
    return payload, targets


def export_ranked_heads(
    ranking_path: Path,
    output_dir: Path,
    classification_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    with ranking_path.open("r", encoding="utf-8", newline="") as handle:
        ranking_rows = list(csv.DictReader(handle))
        ranking_fields = list(ranking_rows[0]) if ranking_rows else []
    if len(ranking_rows) != 720:
        raise ValueError(f"Expected 720 ranked heads, found {len(ranking_rows)}")

    class_by_key = {(row["block"], row["head"]): row for row in classification_rows}
    seen: set[tuple[int, int]] = set()
    merged_rows: list[dict[str, Any]] = []
    for ranking_row in ranking_rows:
        key = head_key(ranking_row)
        if key in seen or key not in class_by_key:
            raise ValueError(f"Invalid or duplicate ranked head {key}")
        seen.add(key)
        classified = class_by_key[key]
        rank = int(ranking_row["robust_rank"])
        merged_rows.append(
            {
                **ranking_row,
                "head_id": classified["head_id"],
                "depth": classified["depth"],
                "final_class": classified["final_class"],
                "s_subtype": classified["s_subtype"],
                "model_role_signature": classified["model_role_signature"],
                "raw_score_winner_signature": classified["raw_score_winner_signature"],
                "raw_score_consensus_candidate": classified["raw_score_consensus_candidate"],
                "in_training_s_same_full59": classified["in_training_s_same_full59"],
                "in_training_t_common_full70": classified["in_training_t_common_full70"],
                "rank_percentile_from_top": rank / 720,
            }
        )
    if sorted(int(row["robust_rank"]) for row in merged_rows) != list(range(1, 721)):
        raise ValueError("robust_rank must contain every integer from 1 through 720")
    merged_rows.sort(key=lambda row: int(row["robust_rank"]))

    classification_fields = [
        "head_id",
        "depth",
        "final_class",
        "s_subtype",
        "model_role_signature",
        "raw_score_winner_signature",
        "raw_score_consensus_candidate",
        "in_training_s_same_full59",
        "in_training_t_common_full70",
        "rank_percentile_from_top",
    ]
    write_csv(
        output_dir / "cross_model_robust_rankings_with_classes.csv",
        merged_rows,
        [*ranking_fields, *classification_fields],
    )

    population = Counter(row["final_class"] for row in merged_rows)
    enrichment_rows: list[dict[str, Any]] = []
    for top_k in (10, 20, 50, 100, 200, 360, 720):
        top_counts = Counter(row["final_class"] for row in merged_rows[:top_k])
        for role in FINAL_CLASSES:
            population_share = population[role] / 720
            count = top_counts[role]
            enrichment_rows.append(
                {
                    "top_k": top_k,
                    "final_class": role,
                    "count": count,
                    "share_percent": 100 * count / top_k,
                    "population_count": population[role],
                    "population_share_percent": 100 * population_share,
                    "expected_count_if_random": top_k * population_share,
                    "enrichment_fold": (count / top_k) / population_share,
                }
            )
    write_csv(
        output_dir / "ranking_class_enrichment.csv",
        enrichment_rows,
        [
            "top_k",
            "final_class",
            "count",
            "share_percent",
            "population_count",
            "population_share_percent",
            "expected_count_if_random",
            "enrichment_fold",
        ],
    )

    class_statistics: list[dict[str, Any]] = []
    for role in FINAL_CLASSES:
        selected = [row for row in merged_rows if row["final_class"] == role]
        ranks = [int(row["robust_rank"]) for row in selected]
        class_statistics.append(
            {
                "final_class": role,
                "count": len(selected),
                "best_rank": min(ranks),
                "median_rank": statistics.median(ranks),
                "mean_rank": statistics.mean(ranks),
                "worst_rank": max(ranks),
                "count_top10": sum(rank <= 10 for rank in ranks),
                "count_top20": sum(rank <= 20 for rank in ranks),
                "count_top50": sum(rank <= 50 for rank in ranks),
                "count_top100": sum(rank <= 100 for rank in ranks),
                "mean_min_pck32": statistics.mean(
                    float(row["min_pck32"]) for row in selected
                ),
                "mean_mean_pck32": statistics.mean(
                    float(row["mean_pck32"]) for row in selected
                ),
            }
        )
    write_csv(
        output_dir / "ranking_class_statistics.csv",
        class_statistics,
        [
            "final_class",
            "count",
            "best_rank",
            "median_rank",
            "mean_rank",
            "worst_rank",
            "count_top10",
            "count_top20",
            "count_top50",
            "count_top100",
            "mean_min_pck32",
            "mean_mean_pck32",
        ],
    )
    return {
        "path": str(ranking_path),
        "sha256": sha256(ranking_path),
        "ranking_rule": (
            "min macro Object PCK@32 across GT/LoRA/Baseline descending; "
            "mean PCK@32 descending; worst error ascending"
        ),
        "top_class_counts": {
            str(top_k): dict(Counter(row["final_class"] for row in merged_rows[:top_k]))
            for top_k in (10, 20, 50, 100)
        },
    }


def build_rows(
    aggregate: dict[tuple[int, int], dict[str, dict[str, str]]],
    s_features: dict[tuple[int, int], dict[str, Any]],
    s59: set[tuple[int, int]],
    t70: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in range(30):
        for head in range(24):
            key = (block, head)
            model_rows = aggregate[key]
            model_roles = [model_rows[model]["role"] for model in MODELS]
            raw_score_winners = [
                max(
                    FINAL_CLASSES[:-1],
                    key=lambda role: float(model_rows[model][f"score_{role}"]),
                )
                for model in MODELS
            ]
            stable = len(set(model_roles)) == 1 and model_roles[0] != "M"
            final_class = model_roles[0] if stable else "M"
            feature = s_features.get(key)

            row: dict[str, Any] = {
                "head_id": head_id(block, head),
                "block": block,
                "head": head,
                "depth": depth_name(block),
                "final_class": final_class,
                "class_description": CLASS_DESCRIPTIONS[final_class],
                "stable_across_models": int(stable),
                "model_role_signature": "/".join(model_roles),
                "raw_score_winner_signature": "/".join(raw_score_winners),
                "raw_score_consensus_candidate": (
                    raw_score_winners[0] if len(set(raw_score_winners)) == 1 else "mixed"
                ),
                "classification_reason": (
                    "same_non_m_role_in_all_three_models"
                    if stable
                    else "mixed_uncertain_or_cross_model_disagreement"
                ),
                "s_subtype": feature["feature_class"] if feature else "",
                "in_training_s_same_full59": int(key in s59),
                "in_training_t_common_full70": int(key in t70),
                "training_subset_ids": ";".join(
                    subset
                    for subset, selected in (
                        ("S_same_full59", key in s59),
                        ("T_common_full70", key in t70),
                    )
                    if selected
                ),
                "cross_model_local_rank_mean": (
                    feature["cross_model_local_rank_mean"] if feature else ""
                ),
                "cross_model_same_frame_rank_mean": (
                    feature["cross_model_same_frame_rank_mean"] if feature else ""
                ),
                "rank_advantage_local_minus_same": (
                    feature["rank_advantage_local_minus_same"] if feature else ""
                ),
            }
            for model in MODELS:
                for metric in MODEL_METRICS:
                    row[f"{model}_{metric}"] = model_rows[model][metric]
                for metric in S_FEATURE_METRICS:
                    row[f"{model}_{metric}"] = (
                        feature["model_statistics"][model][metric] if feature else ""
                    )
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    aggregate = load_aggregate(args.aggregate)
    s_features = load_s_features(args.s_split)
    s59_payload, s59 = load_training_subset(args.s59, 59)
    t70_payload, t70 = load_training_subset(args.t70, 70)
    if s59 & t70:
        raise ValueError(f"S59/T70 training subsets overlap: {sorted(s59 & t70)}")

    rows = build_rows(aggregate, s_features, s59, t70)
    final_counts = Counter(row["final_class"] for row in rows)
    if dict(final_counts) != EXPECTED_FINAL_COUNTS:
        raise ValueError(f"Unexpected final class counts: {dict(final_counts)}")
    stable_s = {(row["block"], row["head"]) for row in rows if row["final_class"] == "S"}
    if stable_s != set(s_features):
        raise ValueError("The S feature partition does not exactly match the stable common-S set")
    if s59 != {key for key, value in s_features.items() if value["feature_class"] == "same_frame_dominant"}:
        raise ValueError("S_same_full59 does not match the complete same-frame-dominant S subtype")

    general_fields = [
        "head_id",
        "block",
        "head",
        "depth",
        "final_class",
        "class_description",
        "stable_across_models",
        "model_role_signature",
        "raw_score_winner_signature",
        "raw_score_consensus_candidate",
        "classification_reason",
        "s_subtype",
        "in_training_s_same_full59",
        "in_training_t_common_full70",
        "training_subset_ids",
        "cross_model_local_rank_mean",
        "cross_model_same_frame_rank_mean",
        "rank_advantage_local_minus_same",
    ]
    detailed_fields = list(general_fields)
    for model in MODELS:
        detailed_fields.extend(f"{model}_{metric}" for metric in MODEL_METRICS)
        detailed_fields.extend(f"{model}_{metric}" for metric in S_FEATURE_METRICS)

    output_dir = args.output_dir
    write_csv(output_dir / "head_classification_all_720.csv", rows, detailed_fields)
    for role in FINAL_CLASSES:
        selected = [row for row in rows if row["final_class"] == role]
        write_csv(
            output_dir / f"heads_{role}_{len(selected)}.csv",
            selected,
            detailed_fields,
        )

    summary_rows: list[dict[str, Any]] = []
    for role in FINAL_CLASSES:
        summary_rows.append(
            {
                "scope": "final_non_overlapping_class",
                "category": role,
                "subcategory": "",
                "count": final_counts[role],
                "description": CLASS_DESCRIPTIONS[role],
            }
        )
    for subtype, count in (("local_dominant", 100), ("same_frame_dominant", 59)):
        summary_rows.append(
            {
                "scope": "stable_S_subtype",
                "category": "S",
                "subcategory": subtype,
                "count": count,
                "description": "disjoint subtype within the 159 stable common-S heads",
            }
        )
    summary_rows.extend(
        [
            {
                "scope": "training_subset",
                "category": s59_payload["role"],
                "subcategory": s59_payload["subset_id"],
                "count": len(s59),
                "description": s59_payload["selection_policy"],
            },
            {
                "scope": "training_subset",
                "category": t70_payload["role"],
                "subcategory": t70_payload["subset_id"],
                "count": len(t70),
                "description": t70_payload["selection_policy"],
            },
        ]
    )
    write_csv(
        output_dir / "head_classification_summary.csv",
        summary_rows,
        ["scope", "category", "subcategory", "count", "description"],
    )

    block_rows: list[dict[str, Any]] = []
    for block in range(30):
        block_heads = [row for row in rows if row["block"] == block]
        counts = Counter(row["final_class"] for row in block_heads)
        block_rows.append(
            {
                "block": block,
                "depth": depth_name(block),
                **{f"count_{role}": counts[role] for role in FINAL_CLASSES},
                "count_s_local_dominant": sum(
                    row["s_subtype"] == "local_dominant" for row in block_heads
                ),
                "count_s_same_frame_dominant": sum(
                    row["s_subtype"] == "same_frame_dominant" for row in block_heads
                ),
                "count_training_s_same_full59": sum(
                    row["in_training_s_same_full59"] for row in block_heads
                ),
                "count_training_t_common_full70": sum(
                    row["in_training_t_common_full70"] for row in block_heads
                ),
            }
        )
    write_csv(
        output_dir / "head_classification_by_block.csv",
        block_rows,
        [
            "block",
            "depth",
            *(f"count_{role}" for role in FINAL_CLASSES),
            "count_s_local_dominant",
            "count_s_same_frame_dominant",
            "count_training_s_same_full59",
            "count_training_t_common_full70",
        ],
    )

    row_by_key = {(row["block"], row["head"]): row for row in rows}
    training_rows: list[dict[str, Any]] = []
    for payload, selected in ((s59_payload, s59), (t70_payload, t70)):
        for block, head in sorted(selected):
            public = row_by_key[(block, head)]
            training_rows.append(
                {
                    "subset_id": payload["subset_id"],
                    "declared_role": payload["role"],
                    "feature_subtype": payload["feature_subtype"],
                    "head_id": head_id(block, head),
                    "block": block,
                    "head": head,
                    "depth": depth_name(block),
                    "final_public_class": public["final_class"],
                    "public_model_role_signature": public["model_role_signature"],
                }
            )
    write_csv(
        output_dir / "training_head_subsets.csv",
        training_rows,
        [
            "subset_id",
            "declared_role",
            "feature_subtype",
            "head_id",
            "block",
            "head",
            "depth",
            "final_public_class",
            "public_model_role_signature",
        ],
    )

    ranking_metadata = export_ranked_heads(args.ranking, output_dir, rows)

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "indexing": "zero-based B{block:02d}H{head:02d}",
        "classification_scope": "22 public seeds completed for all three models",
        "models": list(MODELS),
        "final_class_rule": (
            "A head receives S/T/P/C/G only when all three models have the same non-M role; "
            "all other heads receive M. Final classes are mutually exclusive."
        ),
        "counts": {
            "all_heads": len(rows),
            "final_classes": {role: final_counts[role] for role in FINAL_CLASSES},
            "S_subtypes": {"local_dominant": 100, "same_frame_dominant": 59},
            "training_subsets": {"S_same_full59": len(s59), "T_common_full70": len(t70)},
            "training_subset_overlap": len(s59 & t70),
        },
        "important_distinction": (
            "T_common_full70 is a user-provided training subset and is not equivalent to the "
            "13-head cross-model stable final T class."
        ),
        "sources": {
            "aggregate": {"path": str(args.aggregate), "sha256": sha256(args.aggregate)},
            "s_split": {"path": str(args.s_split), "sha256": sha256(args.s_split)},
            "s59": {"path": str(args.s59), "sha256": sha256(args.s59)},
            "t70": {"path": str(args.t70), "sha256": sha256(args.t70)},
            "ranking": ranking_metadata,
        },
    }
    metadata_path = output_dir / "metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)

    print(json.dumps({"output_dir": str(output_dir), **metadata["counts"]}, indent=2))


if __name__ == "__main__":
    main()
