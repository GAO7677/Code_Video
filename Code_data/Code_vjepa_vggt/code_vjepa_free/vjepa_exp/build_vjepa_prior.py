from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


"""
Build lightweight V-JEPA prior scores from the merged dense feature table.

These scores are intentionally heuristic and low-capacity. They use the
emergence-zone features suggested by the V-JEPA physics interpretability paper,
plus a late-layer dispersion term that was empirically strong in the current
small run.

Example:

/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/build_vjepa_prior.py \
    --input-json /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff/vjepa_bench_merged_dense.json \
    --output-json /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff/vjepa_prior_scored_dense.json \
    --output-csv /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff/vjepa_prior_scored_dense.csv \
    --summary-json /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff/vjepa_prior_summary_dense.json
"""


BENCH_METRICS = [
    "wmreward_similarity",
    "pdi_score",
    "videophy2_pc",
    "videophy2_sa",
    "phyground_general_avg",
    "cosmos_reason1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build heuristic V-JEPA prior scores from merged dense features.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"expected non-empty list at {path}")
    return data


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return None


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def std_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    if var <= 0.0:
        return None
    return math.sqrt(var)


def zscore_map(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    vals = [(row["case_id"], numeric(row.get(key))) for row in rows]
    clean = [v for _, v in vals if v is not None]
    m = mean_or_none(clean)
    s = std_or_none(clean)
    out: dict[str, float | None] = {}
    for case_id, value in vals:
        if value is None or m is None or s is None or s == 0.0:
            out[case_id] = None
        else:
            out[case_id] = (value - m) / s
    return out


def signed_mean(case_id: str, signed_feature_zmaps: list[tuple[int, dict[str, float | None]]]) -> float | None:
    vals: list[float] = []
    for sign, zmap in signed_feature_zmaps:
        value = zmap.get(case_id)
        if value is None:
            continue
        vals.append(sign * value)
    return mean_or_none(vals)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def build_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_keys = [
        "vjepa_motion_5_to_7_delta",
        "vjepa_motion_7_to_8_delta",
        "vjepa_motion_8_to_9_delta",
        "vjepa_motion_9_to_11_delta",
        "vjepa_layer_8_motion_saliency_mean",
        "vjepa_layer_9_motion_saliency_mean",
        "vjepa_layer_11_motion_saliency_mean",
        "vjepa_layer_8_adjacent_affinity_mean",
        "vjepa_layer_9_adjacent_affinity_mean",
        "vjepa_layer_11_adjacent_affinity_mean",
        "vjepa_layer_23_token_std",
        "vjepa_token_std_late_11_to_23_delta",
    ]
    zmaps = {key: zscore_map(rows, key) for key in feature_keys}

    emergence_signed = [
        (+1, zmaps["vjepa_motion_5_to_7_delta"]),
        (+1, zmaps["vjepa_motion_7_to_8_delta"]),
        (+1, zmaps["vjepa_motion_8_to_9_delta"]),
        (+1, zmaps["vjepa_motion_9_to_11_delta"]),
    ]
    mid_signed = [
        (+1, zmaps["vjepa_layer_8_motion_saliency_mean"]),
        (+1, zmaps["vjepa_layer_9_motion_saliency_mean"]),
        (+1, zmaps["vjepa_layer_11_motion_saliency_mean"]),
        (-1, zmaps["vjepa_layer_8_adjacent_affinity_mean"]),
        (-1, zmaps["vjepa_layer_9_adjacent_affinity_mean"]),
        (-1, zmaps["vjepa_layer_11_adjacent_affinity_mean"]),
    ]
    late_signed = [
        (+1, zmaps["vjepa_layer_23_token_std"]),
        (+1, zmaps["vjepa_token_std_late_11_to_23_delta"]),
    ]

    scored: list[dict[str, Any]] = []
    for row in rows:
        case_id = row["case_id"]
        out = dict(row)
        out["vjepa_prior_emergence_score"] = signed_mean(case_id, emergence_signed)
        out["vjepa_prior_midlayer_score"] = signed_mean(case_id, mid_signed)
        out["vjepa_prior_late_score"] = signed_mean(case_id, late_signed)
        out["vjepa_prior_composite_score"] = mean_or_none(
            [
                v
                for v in [
                    out["vjepa_prior_emergence_score"],
                    out["vjepa_prior_midlayer_score"],
                    out["vjepa_prior_late_score"],
                ]
                if v is not None
            ]
        )
        scored.append(out)
    return scored


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_keys = [
        "vjepa_prior_emergence_score",
        "vjepa_prior_midlayer_score",
        "vjepa_prior_late_score",
        "vjepa_prior_composite_score",
    ]
    out: dict[str, Any] = {"num_cases": len(rows), "scores": {}}
    for score_key in score_keys:
        out["scores"][score_key] = {"bench_correlations": {}}
        for metric_key in BENCH_METRICS:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                x = numeric(row.get(score_key))
                y = numeric(row.get(metric_key))
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
            corr = pearson(xs, ys)
            out["scores"][score_key]["bench_correlations"][metric_key] = {
                "pearson": corr,
                "num_pairs": len(xs),
            }
    return out


def write_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_json)
    scored = build_scores(rows)
    summary = build_summary(scored)
    write_json(args.output_json, scored)
    write_csv(args.output_csv, scored)
    write_json(args.summary_json, summary)
    print(str(args.output_json.expanduser().resolve()))
    print(str(args.output_csv.expanduser().resolve()))
    print(str(args.summary_json.expanduser().resolve()))


if __name__ == "__main__":
    main()
