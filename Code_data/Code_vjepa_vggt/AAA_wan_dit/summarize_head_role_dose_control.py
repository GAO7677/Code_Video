#!/usr/bin/env python3
"""Build paired, case-clustered statistics for the matched-head pilot."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metric:
    name: str
    path: tuple[str, ...]
    direction: str


METRICS = (
    Metric("physics_iq_with_context", ("physics_iq_with_context", "score"), "higher"),
    Metric("physics_iq_without_context", ("physics_iq_without_context", "score"), "higher"),
    Metric("pmf_with_context", ("pmf_with_context", "score"), "higher"),
    Metric("pmf_without_context", ("pmf_without_context", "score"), "higher"),
    Metric("wmreward_surprise", ("wmreward", "surprise"), "lower"),
    Metric("vbench_subject_consistency", ("vbench_subject_consistency", "score"), "higher"),
    Metric("vbench_background_consistency", ("vbench_background_consistency", "score"), "higher"),
    Metric("vbench_temporal_flickering", ("vbench_temporal_flickering", "score"), "higher"),
    Metric("vbench_motion_smoothness", ("vbench_motion_smoothness", "score"), "higher"),
    Metric("vbench_dynamic_degree", ("vbench_dynamic_degree", "score"), "higher"),
    Metric("vbench_aesthetic_quality", ("vbench_aesthetic_quality", "score"), "higher"),
    Metric("vbench_imaging_quality", ("vbench_imaging_quality", "score"), "higher"),
    Metric("videophy2_sa", ("videophy2", "sa_score"), "higher"),
    Metric("videophy2_pc", ("videophy2", "pc_score"), "higher"),
    Metric("videophy2_joint_rate", ("videophy2", "joint_pass"), "higher"),
    Metric("videophy2_pc_raw", ("videophy2", "pc_raw_score"), "higher"),
    Metric("cosmos_reason1", ("cosmos_reason1", "score"), "higher"),
)
PRIMARY_METRICS = (
    "physics_iq_with_context",
    "pmf_with_context",
    "wmreward_surprise",
    "videophy2_pc",
    "cosmos_reason1",
)
ROLE_PAIRS = (("S", "T"), ("S", "C"), ("T", "C"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def nested_number(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return float("nan")
        value = value.get(key)
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return float("nan")
    return float(value)


def sidecars(root: Path, cases: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.rglob("*.json"):
        if path.stem not in cases:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        output = Path(str(payload.get("output_video", "")))
        if output.stem != path.stem or not output.is_file():
            continue
        if path.stem in found:
            raise RuntimeError(f"Duplicate sidecar for {path.stem} below {root}")
        found[path.stem] = path
    return found


def bootstrap_case_mean(
    frame: pd.DataFrame,
    value_column: str,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, int, float]:
    clean = frame[["case_id", value_column]].dropna()
    if clean.empty:
        return (float("nan"),) * 3 + (0, 0, float("nan"))
    case_values = clean.groupby("case_id")[value_column].mean().to_numpy(float)
    estimate = float(case_values.mean())
    if len(case_values) == 1:
        return estimate, estimate, estimate, len(clean), 1, float("nan")
    draws = rng.choice(case_values, size=(samples, len(case_values)), replace=True)
    boot = draws.mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    lower_tail = (np.count_nonzero(boot <= 0) + 1) / (len(boot) + 1)
    upper_tail = (np.count_nonzero(boot >= 0) + 1) / (len(boot) + 1)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return (
        estimate,
        float(low),
        float(high),
        len(clean),
        len(case_values),
        float(p_value),
    )


def holm_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.dropna().sort_values()
    running = 0.0
    count = len(finite)
    for rank, (index, value) in enumerate(finite.items()):
        running = max(running, (count - rank) * float(value))
        result.loc[index] = min(1.0, running)
    return result


def load_rows(root: Path) -> pd.DataFrame:
    manifest = json.loads((root / "generation_manifest.json").read_text(encoding="utf-8"))
    cases = {
        case
        for entry in manifest["entries"]
        if entry["kind"] == "ablation"
        for case in entry["videos"]
    }
    rows: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        found = sidecars(Path(entry["result_root"]), cases)
        if set(found) != cases:
            missing = sorted(cases - set(found))
            raise RuntimeError(f"{entry['result_root']} missing sidecars: {missing[:3]}")
        for case_id, path in sorted(found.items()):
            payload = json.loads(path.read_text(encoding="utf-8"))
            row: dict[str, Any] = {
                "kind": entry["kind"],
                "model": entry["model"],
                "seed": int(entry["seed"]),
                "case_id": case_id,
                "sidecar": str(path),
                "output_video": str(payload["output_video"]),
                "input_json": str(payload.get("input_json", "")),
                "input_video": str(payload.get("input_video", "")),
                "caption": str(payload.get("input_caption", "")),
                "subset_id": "",
                "role": "baseline",
                "k": 0,
                "replicate": -1,
                "matching": "baseline",
                "denoise_start": -1,
                "denoise_end": -1,
            }
            if entry["kind"] == "ablation":
                row.update(
                    {
                        "subset_id": entry["subset_id"],
                        "role": entry["subset_id"][0],
                        "k": int(entry["k"]),
                        "replicate": int(
                            entry["subset_id"].split("_r", 1)[1].split("_", 1)[0]
                        ),
                        "matching": (
                            "exact_block"
                            if "exact" in entry["subset_id"]
                            else "approx_depth"
                        ),
                        "denoise_start": int(entry["step_range"][0]),
                        "denoise_end": int(entry["step_range"][1]),
                    }
                )
            row.update({metric.name: nested_number(payload, metric.path) for metric in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def pair_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "seed", "case_id"]
    baseline = frame[frame["kind"] == "baseline"].set_index(keys)
    if baseline.index.duplicated().any():
        raise RuntimeError("Duplicate model/seed/case baselines")
    paired = frame[frame["kind"] == "ablation"].copy()
    reference = baseline.loc[pd.MultiIndex.from_frame(paired[keys])]
    reference.index = paired.index
    for metric in METRICS:
        paired[f"{metric.name}_baseline"] = reference[metric.name]
        paired[f"{metric.name}_delta"] = paired[metric.name] - reference[metric.name]
        sign = 1.0 if metric.direction == "higher" else -1.0
        paired[f"{metric.name}_improvement"] = sign * paired[f"{metric.name}_delta"]
        paired[f"{metric.name}_harm"] = -paired[f"{metric.name}_improvement"]
    return paired


def summarize_roles(
    paired: pd.DataFrame,
    *,
    samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # Average subset replicates first; cases, not videos or heads, are the bootstrap unit.
    replicate_keys = [
        "model", "seed", "case_id", "matching", "k", "denoise_start",
        "denoise_end", "role",
    ]
    value_columns = [f"{metric.name}_harm" for metric in METRICS]
    collapsed = paired.groupby(replicate_keys, as_index=False)[value_columns].mean()
    rows = []
    group_keys = ["model", "matching", "k", "denoise_start", "denoise_end", "role"]
    for key, group in collapsed.groupby(group_keys, sort=True):
        record = dict(zip(group_keys, key))
        record["n_seeds"] = int(group["seed"].nunique())
        for metric in METRICS:
            estimate, low, high, n, n_cases, _ = bootstrap_case_mean(
                group,
                f"{metric.name}_harm",
                samples=samples,
                rng=rng,
            )
            record[f"{metric.name}_harm_mean"] = estimate
            record[f"{metric.name}_harm_ci95_low"] = low
            record[f"{metric.name}_harm_ci95_high"] = high
            record[f"{metric.name}_n"] = n
            record[f"{metric.name}_n_cases"] = n_cases
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_role_contrasts(
    paired: pd.DataFrame,
    *,
    samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # Roles from the same replicate are a frozen depth-matched triplet.
    index = [
        "model", "seed", "case_id", "matching", "k", "replicate",
        "denoise_start", "denoise_end",
    ]
    rows = []
    for metric in METRICS:
        wide = paired.pivot_table(
            index=index,
            columns="role",
            values=f"{metric.name}_harm",
            aggfunc="first",
        ).reset_index()
        for left, right in ROLE_PAIRS:
            if left not in wide or right not in wide:
                continue
            contrast = wide[index].copy()
            contrast["value"] = wide[left] - wide[right]
            # Average matched-subset replicates before case-cluster inference.
            collapsed = contrast.groupby(
                [key for key in index if key != "replicate"],
                as_index=False,
            )["value"].mean()
            group_keys = ["model", "matching", "k", "denoise_start", "denoise_end"]
            for key, group in collapsed.groupby(group_keys, sort=True):
                raw_group = contrast[
                    (contrast.model == key[0])
                    & (contrast.matching == key[1])
                    & (contrast.k == key[2])
                    & (contrast.denoise_start == key[3])
                    & (contrast.denoise_end == key[4])
                ]
                estimate, low, high, n, n_cases, p_value = bootstrap_case_mean(
                    group, "value", samples=samples, rng=rng
                )
                value_std = float(group["value"].std(ddof=1))
                rows.append(
                    {
                        **dict(zip(group_keys, key)),
                        "metric": metric.name,
                        "contrast": f"{left}_minus_{right}",
                        "harm_difference_mean": estimate,
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_p_two_sided": p_value,
                        "standardized_paired_effect": (
                            estimate / value_std
                            if value_std > 0 and math.isfinite(value_std)
                            else float("nan")
                        ),
                        "case_mean_variance": float(
                            group.groupby("case_id")["value"].mean().var(ddof=1)
                        ),
                        "seed_mean_variance": float(
                            group.groupby("seed")["value"].mean().var(ddof=1)
                        ),
                        "matched_subset_mean_variance": float(
                            raw_group.groupby("replicate")["value"].mean().var(ddof=1)
                        ),
                        "n": n,
                        "n_cases": n_cases,
                        "n_seeds": int(group["seed"].nunique()),
                    }
                )
    result = pd.DataFrame(rows)
    correction_keys = [
        "model", "matching", "k", "denoise_start", "denoise_end", "metric"
    ]
    result["holm_p_three_role_contrasts"] = result.groupby(
        correction_keys, group_keys=False
    )["bootstrap_p_two_sided"].apply(holm_adjust)
    return result


def write_readme(
    path: Path,
    frame: pd.DataFrame,
    role_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Matched Head Role Dose-Control Pilot",
        "",
        "This is an exploratory pilot, not a held-out confirmatory result. The same 20",
        "cases contributed to head-role discovery, so causal role labels require a later",
        "held-out replication.",
        "",
        "Positive `*_harm` means the ablation degraded the metric relative to the same",
        "model, seed, and source-case baseline. For WMReward surprise, the sign is",
        "reversed because lower is better.",
        "",
        f"- Per-video rows: {len(frame):,}",
        f"- Role summary rows: {len(role_summary):,}",
        f"- Matched contrast rows: {len(contrast_summary):,}",
        "- Bootstrap unit: source case; subset replicates are averaged before inference.",
        "- k=8 approximate depth matching and k=5 exact same-block matching are separate.",
        "",
        "Primary metrics: " + ", ".join(PRIMARY_METRICS) + ".",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    frame = load_rows(root)
    paired = pair_baselines(frame)
    role_summary = summarize_roles(paired, samples=args.bootstrap_samples, rng=rng)
    contrast_summary = summarize_role_contrasts(
        paired, samples=args.bootstrap_samples, rng=rng
    )
    frame.to_csv(output / "per_video_metrics.csv", index=False)
    paired.to_csv(output / "paired_vs_baseline.csv", index=False)
    role_summary.to_csv(output / "role_harm_case_bootstrap.csv", index=False)
    contrast_summary.to_csv(output / "matched_role_contrasts.csv", index=False)
    coverage = {
        metric.name: int(frame[metric.name].notna().sum()) for metric in METRICS
    }
    (output / "coverage.json").write_text(
        json.dumps(
            {
                "rows": len(frame),
                "expected_rows": 258 * 20,
                "metric_coverage": coverage,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_unit": "source_case",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_readme(output / "README.md", frame, role_summary, contrast_summary)
    print(f"[dose-summary] rows={len(frame)} output={output}")


if __name__ == "__main__":
    main()
