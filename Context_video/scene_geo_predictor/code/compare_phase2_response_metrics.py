"""Audit the Phase-2 vector recheck against the frozen scalar v4 report.

This is a read-only post-processing command.  It does not load a model or
touch the manifest; it only joins the existing CSV exports and writes
versioned comparison/audit tables into the new response-metric directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    if value in (None, "", "nan", "NaN", "None"):
        return None
    return float(value)


def as_int(value):
    return int(float(value))


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(values)) if values else None


def max_abs(values):
    values = [abs(float(x)) for x in values if x is not None and math.isfinite(float(x))]
    return float(max(values)) if values else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_key(row: dict) -> tuple:
    return (row["arm"], as_int(row["eval_seed"]), row["family"],
            round(float(row["geometry_a"]), 8), round(float(row["geometry_b"]), 8),
            row["group_id"], as_int(row["history_index"]), row["response_layer"], row["split"])


def metric_comparison(new_root: Path, old_root: Path) -> dict:
    new_rows = [row for row in read_csv(new_root / "per_pair_seed_vector.csv")
                if row["epoch"] == "200" and row["arm"] != "analytic_cv"]
    old_rows = read_csv(old_root / "per_pair_seed.csv")
    old_by_key = {pair_key(row): row for row in old_rows}
    grouped = defaultdict(list)
    matched = []
    unmatched = []
    for new in new_rows:
        key = pair_key(new)
        old = old_by_key.get(key)
        if old is None:
            unmatched.append(key)
            continue
        matched.append((new, old))
        grouped[(new["split"], new["family"], new["arm"], new["response_layer"])].append((new, old))

    rows = []
    for (split, family, arm, layer), values in sorted(grouped.items()):
        new, old = zip(*values)
        row = {
            "split": split, "family": family, "arm": arm, "response_layer": layer,
            "row_count_new": len(values), "row_count_old": len(values),
            "old_D_pred_mm_mean": mean([as_float(x["D_pred_m"]) for x in old]) * 1000,
            "new_D_pred_mm_mean": mean([as_float(x["D_pred_m"]) for x in new]) * 1000,
            "old_D_gt_mm_mean": mean([as_float(x["D_gt_m"]) for x in old]) * 1000,
            "new_D_gt_mm_mean": mean([as_float(x["D_gt_m"]) for x in new]) * 1000,
            "max_abs_D_gt_difference_m": max_abs([as_float(a["D_gt_m"]) - as_float(b["D_gt_m"]) for a, b in values]),
            "max_abs_D_pred_difference_m": max_abs([as_float(a["D_pred_m"]) - as_float(b["D_pred_m"]) for a, b in values]),
            "metric_status": "equal_future_NA" if layer == "equal_future" else "vector_and_amplitude_reported",
        }
        if layer == "equal_future":
            row.update({
                "old_equal_D_pred_mm_mean": mean([as_float(x["equal_future_absolute_prediction_delta_m"]) for x in old]) * 1000,
                "new_equal_D_pred_mm_mean": mean([as_float(x["equal_future_absolute_prediction_delta_m"]) for x in new]) * 1000,
            })
        else:
            row.update({
                "old_amplitude_ratio_mean": mean([as_float(x["R_delta"]) for x in old]),
                "old_amplitude_error_mean": mean([as_float(x["R_error"]) for x in old]),
                "new_vector_E_delta_mm_mean": mean([as_float(x["E_delta_m"]) for x in new]) * 1000,
                "new_vector_R_delta_mean": mean([as_float(x["R_delta"]) for x in new]),
                "new_amplitude_ratio_mean": mean([as_float(x["response_magnitude_ratio"]) for x in new]),
                "new_amplitude_error_mean": mean([as_float(x["response_magnitude_error"]) for x in new]),
            })
        rows.append(row)
    write_csv(new_root / "old_new_numeric_comparison.csv", rows)

    gt_diffs = [as_float(new["D_gt_m"]) - as_float(new["D_gt_manifest_m"]) for new in new_rows]
    vector_formula_diffs = []
    for new in new_rows:
        if new["response_layer"] != "equal_future":
            vector_formula_diffs.append(as_float(new["R_delta"]) - as_float(new["E_delta_m"]) / as_float(new["D_gt_m"]))
    audit = {
        "status": "EXECUTED",
        "epoch": 200,
        "new_rows": len(new_rows),
        "old_rows": len(old_rows),
        "matched_rows": len(matched),
        "unmatched_new_rows": len(unmatched),
        "new_splits": sorted({row["split"] for row in new_rows}),
        "locked_test_model_metrics": "NOT_RUN",
        "max_abs_D_gt_vs_manifest_m": max_abs(gt_diffs),
        "max_abs_vector_R_formula_difference": max_abs(vector_formula_diffs),
        "max_abs_D_gt_old_new_m": max_abs([as_float(a["D_gt_m"]) - as_float(b["D_gt_m"]) for a, b in matched]),
        "max_abs_D_pred_old_new_m": max_abs([as_float(a["D_pred_m"]) - as_float(b["D_pred_m"]) for a, b in matched]),
        "point_rows_are_seed_level": True,
        "point_eval_seeds": sorted({as_int(row["eval_seed"]) for row in new_rows if row["arm"] == "point_geometry_resample"}),
        "aggregation_order": "pair/seed -> history/family summaries",
    }
    (new_root / "metric_repro_check.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def sampling_comparison(new_root: Path, old_root: Path) -> None:
    new_rows = [row for row in read_csv(new_root / "sampling_stability_recomputed.csv") if row["epoch"] == "200"]
    old_rows = read_csv(old_root / "sampling_stability.csv")
    old_by_key = {(row["family"], row["split"], row["key"]): row for row in old_rows}
    grouped = defaultdict(list)
    for new in new_rows:
        old = old_by_key.get((new["family"], new["split"], new["key"]))
        if old is not None:
            grouped[(new["split"], new["family"])].append((new, old))
    rows = []
    for (split, family), values in sorted(grouped.items()):
        new, old = zip(*values)
        rows.append({
            "split": split, "family": family, "episode_count": len(values),
            "old_mean_pairwise_m_mean": mean([as_float(x["mean_pairwise_m"]) for x in old]),
            "new_S_pairwise_m_mean": mean([as_float(x["S_pairwise_m"]) for x in new]),
            "new_S_center_m_mean": mean([as_float(x["S_center_m"]) for x in new]),
            "old_max_pairwise_m_max": max([as_float(x["max_pairwise_m"]) for x in old]),
            "new_S_max_pair_m_max": max([as_float(x["S_max_pair_m"]) for x in new]),
            "old_mean_to_seed0_m_mean": mean([as_float(x["mean_to_seed0_m"]) for x in old]),
            "new_S_seed0_m_mean": mean([as_float(x["S_seed0_m"]) for x in new]),
            "max_abs_S_pairwise_old_new_m": max_abs([as_float(a["S_pairwise_m"]) - as_float(b["mean_pairwise_m"]) for a, b in values]),
            "max_abs_S_seed0_old_new_m": max_abs([as_float(a["S_seed0_m"]) - as_float(b["mean_to_seed0_m"]) for a, b in values]),
            "threshold_status": "UNRESOLVED_for_S_center_and_S_pairwise",
        })
    write_csv(new_root / "sampling_old_new_comparison.csv", rows)


def deflector_coverage(new_root: Path) -> None:
    rows = [row for row in read_csv(new_root / "per_pair_seed_vector.csv")
            if row["epoch"] == "200" and row["arm"] == "motion_only" and row["eval_seed"] == "0" and row["family"] == "deflector"]
    histories = read_csv(new_root / "deflector_response_histories.csv")
    history_categories = defaultdict(set)
    for row in histories:
        history_categories[(row["split"], row["group_id"])].add(row["manifest_event_category"])
    out = []
    for split in ("train", "dev"):
        split_rows = [row for row in rows if row["split"] == split]
        total_histories = len({row["group_id"] for row in split_rows})
        for layer in ("strong", "weak", "equal_future"):
            layer_rows = [row for row in split_rows if row["response_layer"] == layer]
            layer_histories = {row["group_id"] for row in layer_rows}
            out.append({"split": split, "response_layer": layer,
                        "physical_pair_count": len({row["pair_id"] for row in layer_rows}),
                        "history_count": len(layer_histories), "total_history_count": total_histories,
                        "history_coverage_fraction": len(layer_histories) / total_histories if total_histories else None,
                        "event_category_counts_for_layer": json.dumps({category: sum(category in history_categories[(split, group)] for group in layer_histories)
                                                                         for category in sorted({category for key in history_categories for category in history_categories[key] if key[0] == split})}),
                        "note": "physical manifest pair counts; point seeds are not multiplied; event categories may overlap when variants in one history differ"})
    write_csv(new_root / "deflector_strong_response_coverage.csv", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    args = parser.parse_args()
    audit = metric_comparison(args.new_root, args.old_root)
    sampling_comparison(args.new_root, args.old_root)
    deflector_coverage(args.new_root)
    code_root = Path(__file__).resolve().parent
    code_files = ["phase2_response_metrics.py", "recheck_physvideo_phase2_response_metrics.py",
                  "evaluate_physvideo_phase2.py", "compare_phase2_response_metrics.py"]
    code_hashes = {name: sha256_file(code_root / name) for name in code_files}
    (args.new_root / "source_hashes.json").write_text(json.dumps({
        "status": "EXECUTED", "code_hashes_sha256": code_hashes,
        "old_evaluation_csv_sha256": {
            "per_pair_seed.csv": sha256_file(args.old_root / "per_pair_seed.csv"),
            "sampling_stability.csv": sha256_file(args.old_root / "sampling_stability.csv"),
        },
    }, indent=2) + "\n")
    mapping_path = args.new_root / "metric_old_new_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["code_hashes_sha256"] = code_hashes
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
