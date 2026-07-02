#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Train simple ridge probes on Wan probing features.")
    parser.add_argument(
        "--index_csv",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/smoke_forward_outputs_20260702/probe_index.csv",
    )
    parser.add_argument(
        "--output_root",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/probe_results_smoke",
    )
    parser.add_argument(
        "--target_field",
        default="source_surprise_score",
    )
    parser.add_argument(
        "--feature_keys",
        default="h_post_global_mean,delta_h_global_mean,h_post_frame_mean,delta_h_frame_mean",
        help="Comma-separated feature names to evaluate.",
    )
    parser.add_argument(
        "--frame_reduce",
        default="mean",
        choices=["mean", "flatten"],
        help="How to convert frame-level features [T,D] into a probe vector.",
    )
    parser.add_argument(
        "--group_field",
        default="basename",
        help="Grouping field used for GroupKFold splitting.",
    )
    parser.add_argument(
        "--max_splits",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,1.0,10.0,100.0",
    )
    return parser.parse_args()


def load_index_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def tensor_to_vector(value: torch.Tensor, frame_reduce: str) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        if frame_reduce == "mean":
            return array.mean(axis=0)
        if frame_reduce == "flatten":
            return array.reshape(-1)
    raise ValueError(f"Unsupported feature shape {array.shape}")


def extract_samples(
    rows: Iterable[Dict[str, str]],
    feature_keys: List[str],
    target_field: str,
    group_field: str,
    frame_reduce: str,
) -> Dict[Tuple[int, int, str], List[Dict]]:
    combos = defaultdict(list)

    for row in rows:
        feature_path = row["feature_path"]
        target_value = safe_float(row.get(target_field))
        if target_value is None:
            continue

        payload = torch.load(feature_path, map_location="cpu")
        features = payload["features"]
        meta = payload.get("meta", {})
        group_value = row.get(group_field) or meta.get(group_field) or row.get("sample_id")

        for step_key, step_payload in features.items():
            step_idx = int(step_key)
            branches = step_payload.get("branches", {})
            cond_layers = branches.get("cond", {})
            for layer_key, layer_payload in cond_layers.items():
                layer_idx = int(layer_key)
                for feature_key in feature_keys:
                    if feature_key not in layer_payload:
                        continue
                    vector = tensor_to_vector(layer_payload[feature_key], frame_reduce)
                    combos[(step_idx, layer_idx, feature_key)].append(
                        {
                            "sample_id": row["sample_id"],
                            "group": group_value,
                            "target": target_value,
                            "vector": vector,
                        }
                    )

    return combos


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    return corrcoef_safe(rankdata_average(x), rankdata_average(y))


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def mae_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_combo(records: List[Dict], alphas: List[float], max_splits: int) -> Dict:
    n_samples = len(records)
    groups = [record["group"] for record in records]
    unique_groups = sorted(set(groups))
    if n_samples < 4 or len(unique_groups) < 2:
        return {
            "status": "skipped_insufficient_samples",
            "n_samples": n_samples,
            "n_groups": len(unique_groups),
        }

    feature_dims = {record["vector"].shape[0] for record in records}
    if len(feature_dims) != 1:
        return {
            "status": "skipped_inconsistent_feature_dim",
            "n_samples": n_samples,
            "n_groups": len(unique_groups),
        }

    x = np.stack([record["vector"] for record in records], axis=0)
    y = np.asarray([record["target"] for record in records], dtype=np.float64)
    groups_array = np.asarray(groups)

    n_splits = min(max_splits, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    oof_pred = np.full_like(y, fill_value=np.nan, dtype=np.float64)

    for train_idx, test_idx in splitter.split(x, y, groups_array):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=alphas)),
            ]
        )
        model.fit(x[train_idx], y[train_idx])
        oof_pred[test_idx] = model.predict(x[test_idx])

    valid_mask = ~np.isnan(oof_pred)
    y_valid = y[valid_mask]
    pred_valid = oof_pred[valid_mask]

    return {
        "status": "ok",
        "n_samples": n_samples,
        "n_groups": len(unique_groups),
        "feature_dim": int(x.shape[1]),
        "pearson": corrcoef_safe(y_valid, pred_valid),
        "spearman": spearman_safe(y_valid, pred_valid),
        "r2": r2_safe(y_valid, pred_valid),
        "mae": mae_safe(y_valid, pred_valid),
    }


def build_heatmap_tables(rows: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped = defaultdict(dict)
    for row in rows:
        if row["status"] != "ok":
            continue
        feature_key = row["feature_key"]
        grouped[feature_key][f"layer_{row['layer_idx']:02d}"] = grouped[feature_key].get(
            f"layer_{row['layer_idx']:02d}", {}
        )
        grouped[feature_key][f"layer_{row['layer_idx']:02d}"][f"step_{row['step_idx']:02d}"] = row["pearson"]
    return grouped


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    feature_keys = [item.strip() for item in args.feature_keys.split(",") if item.strip()]
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    index_rows = load_index_rows(args.index_csv)

    combos = extract_samples(
        index_rows,
        feature_keys=feature_keys,
        target_field=args.target_field,
        group_field=args.group_field,
        frame_reduce=args.frame_reduce,
    )

    result_rows = []
    for (step_idx, layer_idx, feature_key), records in sorted(combos.items()):
        metrics = evaluate_combo(records, alphas=alphas, max_splits=args.max_splits)
        result_rows.append(
            {
                "step_idx": step_idx,
                "layer_idx": layer_idx,
                "feature_key": feature_key,
                **metrics,
            }
        )

    results_csv = output_root / "probe_metrics.csv"
    fieldnames = [
        "step_idx",
        "layer_idx",
        "feature_key",
        "status",
        "n_samples",
        "n_groups",
        "feature_dim",
        "pearson",
        "spearman",
        "r2",
        "mae",
    ]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    summary = {
        "index_csv": args.index_csv,
        "target_field": args.target_field,
        "feature_keys": feature_keys,
        "frame_reduce": args.frame_reduce,
        "group_field": args.group_field,
        "num_combos": len(result_rows),
        "num_ok_combos": sum(row["status"] == "ok" for row in result_rows),
        "heatmap_tables": build_heatmap_tables(result_rows),
    }
    summary_json = output_root / "probe_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(results_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
