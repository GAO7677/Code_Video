from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


"""
Examples

/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/merge_vjepa_bench.py \
    --bench-json /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff/bench_table.json \
    --feature-root /data/gaoya/agent-data/outputs/vjepa_wan_precheck/precheck_v2_s42_ti2v_5b_fluxff \
    --output-json /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff/vjepa_bench_merged.json \
    --output-csv /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff/vjepa_bench_merged.csv \
    --corr-json /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff/vjepa_bench_correlations.json
"""


DEFAULT_BENCH_JSON = Path(
    "/data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff/bench_table.json"
)
DEFAULT_FEATURE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/vjepa_wan_precheck/precheck_v2_s42_ti2v_5b_fluxff"
)

STATIC_COLUMNS = ["case_id", "input_video_prompt", "input_image", "output_video"]
LAYER_KEYS = ["motion_saliency_mean", "adjacent_affinity_mean", "token_std"]
BENCH_METRIC_KEYS = [
    "wmreward_similarity",
    "wmreward_surprise",
    "pdi_score",
    "videophy2_pc",
    "videophy2_sa",
    "phyground_general_avg",
    "phyground_SA",
    "phyground_PTV",
    "phyground_persistence",
    "cosmos_reason1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge bench results with per-case V-JEPA feature summaries and compute simple correlations."
    )
    parser.add_argument("--bench-json", type=Path, default=DEFAULT_BENCH_JSON)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--corr-json", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def flatten_summary(case_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": case_id}
    layers = summary.get("layers", {})
    motion_values: list[float] = []
    affinity_values: list[float] = []
    token_std_values: list[float] = []

    for layer_name, layer_stats in sorted(layers.items()):
        for key in LAYER_KEYS:
            value = layer_stats.get(key)
            row[f"vjepa_{layer_name}_{key}"] = value
        if isinstance(layer_stats.get("motion_saliency_mean"), (int, float)):
            motion_values.append(float(layer_stats["motion_saliency_mean"]))
        if isinstance(layer_stats.get("adjacent_affinity_mean"), (int, float)):
            affinity_values.append(float(layer_stats["adjacent_affinity_mean"]))
        if isinstance(layer_stats.get("token_std"), (int, float)):
            token_std_values.append(float(layer_stats["token_std"]))

    row["vjepa_num_sampled_frames"] = summary.get("num_sampled_frames")
    row["vjepa_target_frames"] = summary.get("target_frames")
    row["vjepa_motion_mean_all_layers"] = mean_or_none(motion_values)
    row["vjepa_affinity_mean_all_layers"] = mean_or_none(affinity_values)
    row["vjepa_token_std_mean_all_layers"] = mean_or_none(token_std_values)
    if motion_values:
        row["vjepa_motion_last_minus_first"] = motion_values[-1] - motion_values[0]
    else:
        row["vjepa_motion_last_minus_first"] = None
    if affinity_values:
        row["vjepa_affinity_last_minus_first"] = affinity_values[-1] - affinity_values[0]
    else:
        row["vjepa_affinity_last_minus_first"] = None

    add_transition_features(row, layers)
    return row


def add_transition_features(row: dict[str, Any], layers: dict[str, Any]) -> None:
    emergence_layers = [5, 7, 8, 9, 11]
    late_layers = [11, 17, 23]
    for metric_key, short_name in [
        ("motion_saliency_mean", "motion"),
        ("adjacent_affinity_mean", "affinity"),
        ("token_std", "token_std"),
    ]:
        for a, b in zip(emergence_layers[:-1], emergence_layers[1:]):
            va = get_layer_metric(layers, a, metric_key)
            vb = get_layer_metric(layers, b, metric_key)
            row[f"vjepa_{short_name}_{a}_to_{b}_delta"] = safe_sub(vb, va)
            row[f"vjepa_{short_name}_{a}_to_{b}_ratio"] = safe_ratio(vb, va)

        start = get_layer_metric(layers, 5, metric_key)
        end = get_layer_metric(layers, 11, metric_key)
        row[f"vjepa_{short_name}_emergence_5_to_11_delta"] = safe_sub(end, start)
        row[f"vjepa_{short_name}_emergence_5_to_11_slope"] = safe_div(safe_sub(end, start), 6.0)

        late_start = get_layer_metric(layers, 11, metric_key)
        late_end = get_layer_metric(layers, 23, metric_key)
        row[f"vjepa_{short_name}_late_11_to_23_delta"] = safe_sub(late_end, late_start)
        row[f"vjepa_{short_name}_late_11_to_23_slope"] = safe_div(safe_sub(late_end, late_start), 12.0)

        emergence_vals = [get_layer_metric(layers, idx, metric_key) for idx in emergence_layers]
        late_vals = [get_layer_metric(layers, idx, metric_key) for idx in late_layers]
        emergence_clean = [v for v in emergence_vals if v is not None]
        late_clean = [v for v in late_vals if v is not None]
        row[f"vjepa_{short_name}_emergence_mean"] = mean_or_none(emergence_clean)
        row[f"vjepa_{short_name}_late_mean"] = mean_or_none(late_clean)
        row[f"vjepa_{short_name}_late_minus_emergence_mean"] = safe_sub(
            row[f"vjepa_{short_name}_late_mean"],
            row[f"vjepa_{short_name}_emergence_mean"],
        )


def get_layer_metric(layers: dict[str, Any], layer_idx: int, metric_key: str) -> float | None:
    layer_stats = layers.get(f"layer_{layer_idx}")
    if not isinstance(layer_stats, dict):
        return None
    value = layer_stats.get(metric_key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def safe_sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_merged_rows(bench_rows: list[dict[str, Any]], feature_root: Path) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for bench_row in bench_rows:
        case_id = bench_row["case_id"]
        summary_path = feature_root / case_id / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing summary for {case_id}: {summary_path}")
        summary = load_json(summary_path)
        merged_row = dict(bench_row)
        merged_row.update(flatten_summary(case_id, summary))
        merged.append(merged_row)
    return merged


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def build_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vjepa_keys = sorted(key for key in rows[0].keys() if key.startswith("vjepa_"))
    report: dict[str, Any] = {
        "num_cases": len(rows),
        "bench_metrics": {},
    }
    for bench_key in BENCH_METRIC_KEYS:
        pairs: dict[str, dict[str, Any]] = {}
        for vjepa_key in vjepa_keys:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                x = numeric_value(row.get(vjepa_key))
                y = numeric_value(row.get(bench_key))
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
            corr = pearson(xs, ys)
            if corr is None:
                continue
            pairs[vjepa_key] = {
                "pearson": corr,
                "abs_pearson": abs(corr),
                "num_pairs": len(xs),
            }
        top = sorted(pairs.items(), key=lambda item: item[1]["abs_pearson"], reverse=True)[:5]
        report["bench_metrics"][bench_key] = {
            "top_correlations": [{**stats, "vjepa_feature": key} for key, stats in top],
            "all_correlations": pairs,
        }
    return report


def write_json(path: Path, payload: Any) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for key in STATIC_COLUMNS + BENCH_METRIC_KEYS:
        if key in rows[0]:
            keys.append(key)
    extra_keys = sorted(key for key in rows[0].keys() if key not in keys)
    fieldnames = keys + extra_keys
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    bench_rows = load_json(args.bench_json)
    if not isinstance(bench_rows, list) or not bench_rows:
        raise ValueError(f"bench json must be a non-empty list: {args.bench_json}")

    feature_root = args.feature_root.expanduser().resolve()
    merged_rows = build_merged_rows(bench_rows, feature_root)
    corr_report = build_correlations(merged_rows)

    write_json(args.output_json, merged_rows)
    write_csv(args.output_csv, merged_rows)
    write_json(args.corr_json, corr_report)

    print(str(args.output_json.expanduser().resolve()))
    print(str(args.output_csv.expanduser().resolve()))
    print(str(args.corr_json.expanduser().resolve()))


if __name__ == "__main__":
    main()
