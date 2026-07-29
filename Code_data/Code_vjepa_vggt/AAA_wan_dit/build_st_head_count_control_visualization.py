#!/usr/bin/env python3
"""Build the S/T equal-head-count metric and representative-case report."""

from __future__ import annotations

import csv
import html
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GALLERY = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery"
)
PILOT = GALLERY / "head-role-dose-control-pilot"
MANIFEST = PILOT / "manifest.json"
OUT = PILOT / "metrics" / "s-t-head-count-control"
ALL_HEAD_ROOT = GALLERY / "multiseed"
ALL_HEAD_MANIFEST = ALL_HEAD_ROOT / "manifest.json"
PHASED_CASE_ROOT = GALLERY / "test5-st-phased-seed851" / "cases"

METRICS = [
    ("physics_iq_with_context", "Physics-IQ ctx"),
    ("physics_iq_without_context", "Physics-IQ noctx"),
    ("pmf_with_context", "PMF ctx"),
    ("pmf_without_context", "PMF noctx"),
]
ROLES = ["S", "T", "C"]
ROLE_COLORS = {
    "S": "#17806d",
    "T": "#d08418",
    "C": "#3567a8",
    "ST": "#a14e79",
}
MODEL_ORDER = ["wan_lora", "xssc", "physrvg"]
STAGES = [(0, 10), (10, 20)]
ALL_HEAD_STAGES = [(0, 5), (5, 10), (0, 10), (10, 20), (20, 30)]
MATCHING = {
    "exact_block": ("Exact k=5", 5),
    "approx_depth": ("Depth-matched k=8", 8),
}


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(20260729)
    sampled = rng.choice(arr, size=(3000, len(arr)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return mean, float(low), float(high)


def matching_name(record: dict) -> str:
    value = record.get("matching", "")
    if "exact" in value:
        return "exact_block"
    if "approx" in value or "depth" in value:
        return "approx_depth"
    return value


def case_cluster_values(rows: list[dict], value_key: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is not None and np.isfinite(value):
            grouped[row["case_id"]].append(float(value))
    return [float(np.mean(values)) for values in grouped.values()]


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_paired_rows(manifest: dict) -> tuple[list[dict], dict]:
    baselines = {
        (row["model"], row["seed"], row["case_id"]): row
        for row in manifest["records"]
        if row.get("kind") == "baseline"
    }
    paired = []
    for row in manifest["records"]:
        if row.get("kind") != "ablation" or row.get("role") not in ROLES:
            continue
        baseline = baselines.get((row["model"], row["seed"], row["case_id"]))
        if baseline is None:
            continue
        output = {
            "model": row["model"],
            "seed": row["seed"],
            "case_id": row["case_id"],
            "subset_id": row["subset_id"],
            "role": row["role"],
            "k": row["k"],
            "replicate": row["replicate"],
            "matching": matching_name(row),
            "start": row["start"],
            "end": row["end"],
            "video": row["video"],
            "baseline_video": baseline["video"],
        }
        complete = True
        for metric, _ in METRICS:
            score = row.get("metrics", {}).get(metric)
            base = baseline.get("metrics", {}).get(metric)
            if score is None or base is None:
                complete = False
                break
            output[metric] = float(score)
            output[f"{metric}_baseline"] = float(base)
            output[f"{metric}_delta"] = float(score) - float(base)
            output[f"{metric}_abs_delta"] = abs(float(score) - float(base))
        if complete:
            paired.append(output)
    return paired, baselines


def build_st_pairs(paired: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in paired:
        key = (
            row["model"],
            row["seed"],
            row["case_id"],
            row["matching"],
            row["k"],
            row["replicate"],
            row["start"],
            row["end"],
        )
        grouped[key][row["role"]] = row
    result = []
    for key, roles in grouped.items():
        if "S" not in roles or "T" not in roles:
            continue
        s_row, t_row = roles["S"], roles["T"]
        row = {
            "model": key[0],
            "seed": key[1],
            "case_id": key[2],
            "matching": key[3],
            "k": key[4],
            "replicate": key[5],
            "start": key[6],
            "end": key[7],
            "baseline_video": s_row["baseline_video"],
            "s_video": s_row["video"],
            "t_video": t_row["video"],
        }
        for metric, _ in METRICS:
            row[f"{metric}_baseline"] = s_row[f"{metric}_baseline"]
            row[f"{metric}_s"] = s_row[metric]
            row[f"{metric}_t"] = t_row[metric]
            row[f"{metric}_s_delta"] = s_row[f"{metric}_delta"]
            row[f"{metric}_t_delta"] = t_row[f"{metric}_delta"]
            row[f"{metric}_abs_contrast"] = (
                s_row[f"{metric}_abs_delta"] - t_row[f"{metric}_abs_delta"]
            )
        result.append(row)
    return result


def plot_curves(paired: list[dict], value_suffix: str, filename: str, title: str) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True)
    for metric_index, (metric, label) in enumerate(METRICS):
        for match_index, matching in enumerate(MATCHING):
            ax = axes[metric_index, match_index]
            for role in ROLES:
                means, lows, highs = [], [], []
                for start, end in STAGES:
                    selected = [
                        row
                        for row in paired
                        if row["matching"] == matching
                        and row["role"] == role
                        and row["start"] == start
                        and row["end"] == end
                    ]
                    values = case_cluster_values(
                        selected, f"{metric}_{value_suffix}"
                    )
                    mean, low, high = mean_ci(values)
                    means.append(mean)
                    lows.append(low)
                    highs.append(high)
                x = np.arange(len(STAGES))
                ax.plot(
                    x,
                    means,
                    marker="o",
                    linewidth=2,
                    color=ROLE_COLORS[role],
                    label=role,
                )
                ax.fill_between(
                    x, lows, highs, color=ROLE_COLORS[role], alpha=0.14
                )
            ax.axhline(0, color="#89908c", linewidth=0.8)
            ax.grid(axis="y", alpha=0.22)
            ax.set_xticks(range(len(STAGES)), ["0-10", "10-20"])
            if metric_index == 0:
                ax.set_title(MATCHING[matching][0])
            if match_index == 0:
                ax.set_ylabel(label)
    axes[0, 1].legend(frameon=False, ncol=3, loc="best")
    fig.suptitle(title, fontsize=16)
    fig.supxlabel("Denoising step interval")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def heatmap(
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
    filename: str,
    title: str,
    cbar_label: str,
    color_matrix: np.ndarray | None = None,
) -> None:
    colors = matrix if color_matrix is None else color_matrix
    finite = np.abs(colors[np.isfinite(colors)])
    vmax = float(np.quantile(finite, 0.95)) if len(finite) else 1.0
    vmax = max(vmax, 1e-8)
    fig_width = max(10, len(columns) * 1.25)
    fig_height = max(3.5, len(rows) * 0.58 + 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(colors, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=14)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def normalize_metric_groups(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix.copy()
    for start in range(0, matrix.shape[1], len(STAGES)):
        stop = start + len(STAGES)
        values = np.abs(matrix[:, start:stop])
        finite = values[np.isfinite(values)]
        scale = float(np.max(finite)) if len(finite) else 1.0
        normalized[:, start:stop] = matrix[:, start:stop] / max(scale, 1e-8)
    return normalized


def build_heatmaps(st_pairs: list[dict]) -> None:
    columns = [
        f"{label}\n{start}-{end}"
        for _, label in METRICS
        for start, end in STAGES
    ]
    model_matrix = []
    for model in MODEL_ORDER:
        cells = []
        for metric, _ in METRICS:
            for start, end in STAGES:
                rows = [
                    row
                    for row in st_pairs
                    if row["matching"] == "exact_block"
                    and row["model"] == model
                    and row["start"] == start
                    and row["end"] == end
                ]
                values = case_cluster_values(rows, f"{metric}_abs_contrast")
                cells.append(float(np.mean(values)) if values else np.nan)
        model_matrix.append(cells)
    model_matrix = np.asarray(model_matrix)
    heatmap(
        model_matrix,
        ["Wan+LoRA", "Wan+xSSC", "PhysRVG"],
        columns,
        "exact_st_model_stage_heatmap.png",
        "Exact k=5: |S - baseline| minus |T - baseline|",
        "within-metric normalized; negative = T changes more",
        normalize_metric_groups(model_matrix),
    )

    replicate_rows = []
    labels = []
    for matching, (_, k) in MATCHING.items():
        replicates = sorted(
            {row["replicate"] for row in st_pairs if row["matching"] == matching}
        )
        for replicate in replicates:
            labels.append(f"{matching} r{replicate} (k={k})")
            cells = []
            for metric, _ in METRICS:
                for start, end in STAGES:
                    rows = [
                        row
                        for row in st_pairs
                        if row["matching"] == matching
                        and row["replicate"] == replicate
                        and row["start"] == start
                        and row["end"] == end
                    ]
                    values = case_cluster_values(rows, f"{metric}_abs_contrast")
                    cells.append(float(np.mean(values)) if values else np.nan)
            replicate_rows.append(cells)
    replicate_rows = np.asarray(replicate_rows)
    heatmap(
        replicate_rows,
        labels,
        columns,
        "replicate_stability_heatmap.png",
        "Replicate stability of the S-vs-T absolute-impact contrast",
        "within-metric normalized; negative = T changes more",
        normalize_metric_groups(replicate_rows),
    )


def plot_coverage(manifest: dict) -> list[dict]:
    core = [
        row
        for row in manifest["records"]
        if row.get("kind") in {"ablation", "baseline"}
    ]
    coverage = []
    for definition in manifest["metric_definitions"]:
        name = definition["name"]
        count = sum(row.get("metrics", {}).get(name) is not None for row in core)
        coverage.append(
            {
                "metric": name,
                "label": definition["label"],
                "available": count,
                "expected": len(core),
                "coverage": count / len(core),
            }
        )
    fig, ax = plt.subplots(figsize=(11, 7))
    ordered = sorted(coverage, key=lambda row: row["coverage"])
    labels = [row["label"] for row in ordered]
    values = [100 * row["coverage"] for row in ordered]
    colors = ["#17806d" if value >= 99.5 else "#d08418" for value in values]
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Coverage among 5,040 ablations + 120 baselines (%)")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    ax.set_title("Metric completion status")
    fig.tight_layout()
    fig.savefig(OUT / "metric_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return coverage


def load_phased_all_head() -> tuple[list[dict], list[dict]]:
    score_rows = []
    browser_cases = []
    for path in sorted(PHASED_CASE_ROOT.glob("*/case.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = payload["id"]
        base_url = f"/test5-st-phased-seed851/cases/{case_id}/"
        browser_case = {
            "id": case_id,
            "prompt": payload.get("prompt", ""),
            "source_video": base_url + payload["references"]["source"],
            "phased": {},
        }
        for model in MODEL_ORDER:
            baseline_scores = payload["metric_scores"]["baseline"][model]
            baseline_video = base_url + payload["videos"]["baseline"][model]
            browser_case["phased"][model] = {
                "baseline": {
                    "video": baseline_video,
                    "metrics": {
                        metric: baseline_scores.get(metric)
                        for metric, _ in METRICS
                    },
                },
                "stages": {},
            }
            score_rows.append(
                {
                    "case_id": case_id,
                    "model": model,
                    "role": "baseline",
                    "start": -1,
                    "end": -1,
                    "video": baseline_video,
                    **{
                        metric: baseline_scores.get(metric)
                        for metric, _ in METRICS
                    },
                }
            )
            for start, end in ALL_HEAD_STAGES:
                stage_key = f"{start:02d}_{end:02d}"
                browser_case["phased"][model]["stages"][stage_key] = {}
                for role in ["S", "T", "ST"]:
                    scores = payload["metric_scores"]["stages"][stage_key][model][role]
                    video = (
                        base_url
                        + payload["videos"]["stages"][stage_key][model][role]
                    )
                    record = {
                        "video": video,
                        "metrics": {
                            metric: scores.get(metric)
                            for metric, _ in METRICS
                        },
                    }
                    browser_case["phased"][model]["stages"][stage_key][role] = record
                    score_rows.append(
                        {
                            "case_id": case_id,
                            "model": model,
                            "role": role,
                            "start": start,
                            "end": end,
                            "video": video,
                            **{
                                metric: scores.get(metric)
                                for metric, _ in METRICS
                            },
                        }
                    )
        browser_cases.append(browser_case)
    return score_rows, browser_cases


def add_full_category_videos(browser_cases: list[dict]) -> dict:
    manifest = json.loads(ALL_HEAD_MANIFEST.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in browser_cases}
    variants = ["baseline", "S", "T", "P", "C", "G"]
    for case in manifest["cases"]:
        case_id = case["id"]
        if case_id not in by_id:
            source_name = f"{case_id}__source_video_49f.mp4"
            item = {
                "id": case_id,
                "prompt": case.get("prompt", ""),
                "source_video": f"/multiseed/media/references/{source_name}",
                "phased": {},
            }
            browser_cases.append(item)
            by_id[case_id] = item
        item = by_id[case_id]
        item["full_categories"] = {}
        for seed in manifest["seeds"]:
            seed_key = str(seed)
            seed_videos = manifest["videos"].get(case_id, {}).get(seed_key, {})
            if not seed_videos:
                continue
            item["full_categories"][seed_key] = {}
            for model in MODEL_ORDER:
                item["full_categories"][seed_key][model] = {}
                for variant in variants:
                    relative = seed_videos.get(model, {}).get(variant)
                    if relative:
                        item["full_categories"][seed_key][model][variant] = {
                            "video": f"/multiseed/{relative}",
                            "metrics": None,
                        }
            if seed_key == "851" and item.get("phased"):
                for model in MODEL_ORDER:
                    baseline = item["full_categories"][seed_key][model].get("baseline")
                    if baseline:
                        baseline["metrics"] = item["phased"][model]["baseline"]["metrics"]
        source_name = f"{case_id}__source_video_49f.mp4"
        source_url = f"/multiseed/media/references/{source_name}"
        if (ALL_HEAD_ROOT / "media" / "references" / source_name).is_file():
            item["source_video"] = source_url
    seed_coverage = {}
    for seed in manifest["seeds"]:
        seed_key = str(seed)
        available = 0
        for item in browser_cases:
            for model_rows in item.get("full_categories", {}).get(seed_key, {}).values():
                available += len(model_rows)
        seed_coverage[seed_key] = available
    complete_seeds = [
        seed for seed, count in seed_coverage.items() if count == len(MODEL_ORDER) * len(variants)
    ]
    return {
        "models": MODEL_ORDER,
        "model_labels": manifest["model_names"],
        "role_labels": {
            "baseline": "Baseline",
            "S": "All-S (159)",
            "T": "All-T (13)",
            "ST": "All-S+T (172)",
            "P": "All-P (82)",
            "C": "All-C (20)",
            "G": "All-G (75)",
        },
        "phased_stages": [f"{start:02d}_{end:02d}" for start, end in ALL_HEAD_STAGES],
        "full_seeds": complete_seeds,
        "full_seeds_all": [str(seed) for seed in manifest["seeds"]],
        "full_seed_coverage": seed_coverage,
        "cases": browser_cases,
    }


def plot_all_head_score_curves(rows: list[dict]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(17, 15), sharex=True)
    stage_labels = [f"{start}-{end}" for start, end in ALL_HEAD_STAGES]
    for metric_index, (metric, label) in enumerate(METRICS):
        for model_index, model in enumerate(MODEL_ORDER):
            ax = axes[metric_index, model_index]
            baseline_values = [
                row[metric]
                for row in rows
                if row["model"] == model and row["role"] == "baseline"
            ]
            baseline_mean, baseline_low, baseline_high = mean_ci(baseline_values)
            ax.axhline(
                baseline_mean,
                color="#242827",
                linestyle="--",
                linewidth=1.8,
                label="Baseline",
            )
            ax.axhspan(baseline_low, baseline_high, color="#242827", alpha=0.07)
            for role in ["S", "T", "ST"]:
                means, lows, highs = [], [], []
                for start, end in ALL_HEAD_STAGES:
                    values = [
                        row[metric]
                        for row in rows
                        if row["model"] == model
                        and row["role"] == role
                        and row["start"] == start
                        and row["end"] == end
                    ]
                    mean, low, high = mean_ci(values)
                    means.append(mean)
                    lows.append(low)
                    highs.append(high)
                x = np.arange(len(ALL_HEAD_STAGES))
                ax.plot(
                    x,
                    means,
                    marker="o",
                    linewidth=2,
                    color=ROLE_COLORS[role],
                    label=f"All-{role}",
                )
                ax.fill_between(
                    x, lows, highs, color=ROLE_COLORS[role], alpha=0.12
                )
            ax.grid(axis="y", alpha=0.2)
            ax.set_xticks(range(len(stage_labels)), stage_labels)
            if metric_index == 0:
                ax.set_title({"wan_lora": "Wan+LoRA", "xssc": "Wan+xSSC", "physrvg": "PhysRVG"}[model])
            if model_index == 0:
                ax.set_ylabel(label)
    axes[0, 2].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle("All-head category scores with the unablated baseline", fontsize=16)
    fig.supxlabel("Denoising step interval")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / "all_head_baseline_score_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all_head_delta_heatmap(rows: list[dict]) -> None:
    baseline = {
        (row["model"], row["case_id"]): row
        for row in rows
        if row["role"] == "baseline"
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    columns = [
        f"{role}\n{start}-{end}"
        for role in ["S", "T", "ST"]
        for start, end in ALL_HEAD_STAGES
    ]
    for ax, (metric, label) in zip(axes.flat, METRICS):
        matrix = []
        for model in MODEL_ORDER:
            values = []
            for role in ["S", "T", "ST"]:
                for start, end in ALL_HEAD_STAGES:
                    deltas = [
                        row[metric] - baseline[(model, row["case_id"])][metric]
                        for row in rows
                        if row["model"] == model
                        and row["role"] == role
                        and row["start"] == start
                        and row["end"] == end
                    ]
                    values.append(float(np.mean(deltas)))
            matrix.append(values)
        matrix = np.asarray(matrix)
        vmax = max(float(np.max(np.abs(matrix))), 1e-8)
        image = ax.imshow(
            matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto"
        )
        ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(3), ["Wan+LoRA", "Wan+xSSC", "PhysRVG"])
        ax.set_title(label)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2g}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.72, label="score - baseline")
    fig.suptitle("All-head signed score change (higher is better)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "all_head_signed_delta_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_summary(st_pairs: list[dict]) -> list[dict]:
    rows = []
    for matching in MATCHING:
        for model in ["all"] + MODEL_ORDER:
            for start, end in [(-1, -1)] + STAGES:
                selected = [
                    row
                    for row in st_pairs
                    if row["matching"] == matching
                    and (model == "all" or row["model"] == model)
                    and (start == -1 or (row["start"] == start and row["end"] == end))
                ]
                for metric, label in METRICS:
                    values = case_cluster_values(selected, f"{metric}_abs_contrast")
                    mean, low, high = mean_ci(values)
                    rows.append(
                        {
                            "matching": matching,
                            "model": model,
                            "start": "all" if start == -1 else start,
                            "end": "all" if end == -1 else end,
                            "metric": metric,
                            "metric_label": label,
                            "s_minus_t_abs_impact": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "n_cases": len(values),
                        }
                    )
    return rows


def select_representatives(st_pairs: list[dict], manifest: dict) -> list[dict]:
    exact = [row for row in st_pairs if row["matching"] == "exact_block"]
    case_lookup = {row["id"]: row for row in manifest["cases"]}
    scales = {}
    for metric, _ in METRICS:
        values = np.asarray(
            [row[f"{metric}_abs_contrast"] for row in exact], dtype=float
        )
        scales[metric] = max(float(values.std()), 1e-8)
    for row in exact:
        row["_selection_score"] = float(
            np.mean(
                [
                    row[f"{metric}_abs_contrast"] / scales[metric]
                    for metric, _ in METRICS
                ]
            )
        )
        row["_primary_metric"] = max(
            METRICS,
            key=lambda item: abs(row[f"{item[0]}_abs_contrast"]) / scales[item[0]],
        )[1]

    selected = []
    for direction, reverse in [("总体趋势：T 影响更大", False), ("反例：S 影响更大", True)]:
        used_cases = set()
        for model in MODEL_ORDER:
            candidates = sorted(
                [row for row in exact if row["model"] == model],
                key=lambda row: row["_selection_score"],
                reverse=reverse,
            )
            chosen = next(
                (row for row in candidates if row["case_id"] not in used_cases),
                candidates[0],
            )
            used_cases.add(chosen["case_id"])
            case = case_lookup[chosen["case_id"]]
            item = {
                key: value
                for key, value in chosen.items()
                if not key.startswith("_")
            }
            item.update(
                {
                    "group": direction,
                    "selection_score": chosen["_selection_score"],
                    "primary_metric": chosen["_primary_metric"],
                    "caption": case.get("caption", ""),
                    "source_video": case.get("source_url", ""),
                }
            )
            selected.append(item)
    return selected


def fmt(value: float) -> str:
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"


def metric_table(row: dict) -> str:
    lines = []
    for metric, label in METRICS:
        baseline = row[f"{metric}_baseline"]
        s_value = row[f"{metric}_s"]
        t_value = row[f"{metric}_t"]
        contrast = row[f"{metric}_abs_contrast"]
        winner = "T变化更大" if contrast < 0 else "S变化更大"
        lines.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{fmt(baseline)}</td>"
            f"<td>{fmt(s_value)} <small>({row[f'{metric}_s_delta']:+.4f})</small></td>"
            f"<td>{fmt(t_value)} <small>({row[f'{metric}_t_delta']:+.4f})</small></td>"
            f"<td class='{'t-more' if contrast < 0 else 's-more'}'>{winner} {abs(contrast):.4f}</td>"
            "</tr>"
        )
    return "".join(lines)


def representative_html(rows: list[dict], model_labels: dict) -> str:
    sections = []
    for group in ["总体趋势：T 影响更大", "反例：S 影响更大"]:
        cards = []
        for index, row in enumerate([item for item in rows if item["group"] == group]):
            uid = f"case-{len(sections)}-{index}"
            videos = [
                ("Source", row["source_video"]),
                ("Baseline", row["baseline_video"]),
                ("S 消融", row["s_video"]),
                ("T 消融", row["t_video"]),
            ]
            video_html = "".join(
                f"<figure><figcaption>{label}</figcaption>"
                f"<video preload='metadata' playsinline src='{html.escape(url)}'></video></figure>"
                for label, url in videos
            )
            cards.append(
                f"<article class='case' id='{uid}'>"
                "<div class='case-head'><div>"
                f"<h3>{html.escape(model_labels[row['model']])} · Seed {row['seed']} · "
                f"{row['start']}-{row['end']} · exact r{row['replicate']}</h3>"
                f"<p><code>{html.escape(row['case_id'])}</code></p>"
                f"<p>{html.escape(row['caption'])}</p></div>"
                "<div class='row-controls'>"
                f"<button onclick=\"playRow('{uid}',true)\">从头播放本行</button>"
                f"<button onclick=\"playRow('{uid}',false)\">继续播放本行</button>"
                f"<button onclick=\"pauseRow('{uid}')\">暂停本行</button>"
                "</div></div>"
                f"<div class='videos'>{video_html}</div>"
                "<div class='metric-table'><table><thead><tr><th>指标</th>"
                "<th>Baseline</th><th>S（Δ）</th><th>T（Δ）</th>"
                f"<th>|ΔS|-|ΔT|</th></tr></thead><tbody>{metric_table(row)}</tbody></table></div>"
                f"<p class='selection'>筛选依据：4 指标尺度标准化后的平均 |ΔS|-|ΔT| = "
                f"{row['selection_score']:+.3f}；最突出指标为 {html.escape(row['primary_metric'])}。</p>"
                "</article>"
            )
        sections.append(
            f"<section class='examples'><h2>{group}</h2>"
            "<p class='section-note'>每个模型各选 1 组；同一行严格共享 model、seed、case、阶段和 replicate。</p>"
            + "".join(cards)
            + "</section>"
        )
    return "".join(sections)


def build_html(
    manifest: dict,
    summary: list[dict],
    representatives: list[dict],
    coverage: list[dict],
    all_head_browser: dict,
) -> str:
    exact_overall = {
        row["metric"]: row
        for row in summary
        if row["matching"] == "exact_block"
        and row["model"] == "all"
        and row["start"] == "all"
    }
    summary_rows = "".join(
        "<tr>"
        f"<td>{label}</td><td>{row['s_minus_t_abs_impact']:.4f}</td>"
        f"<td>[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]</td>"
        f"<td>{'T 影响更大' if row['s_minus_t_abs_impact'] < 0 else 'S 影响更大'}</td>"
        "</tr>"
        for metric, label in METRICS
        for row in [exact_overall[metric]]
    )
    complete = sum(item["coverage"] >= 0.995 for item in coverage)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S/T 等 Head 数量控制分析</title>
<style>
:root{{--bg:#f4f5f2;--paper:#fff;--ink:#202423;--muted:#66706b;--line:#cbd1cd;--s:#17806d;--t:#d08418;--link:#176f62}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
header,main{{max-width:1540px;margin:auto;padding:18px 24px}}header{{border-bottom:1px solid var(--line)}}
h1,h2,h3,p{{margin:0}}h1{{font-size:26px}}h2{{font-size:20px;margin-bottom:5px}}h3{{font-size:16px}}
.sub,.section-note,.selection{{color:var(--muted)}}.links{{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}}a{{color:var(--link)}}
.facts{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));border:1px solid var(--line);margin:18px 0;background:var(--paper)}}
.fact{{padding:12px;border-right:1px solid var(--line)}}.fact:last-child{{border-right:0}}.fact strong{{display:block;font-size:19px}}
.analysis{{display:grid;grid-template-columns:minmax(390px,.75fr) minmax(560px,1.25fr);gap:18px;align-items:start;margin:24px 0}}
.text-panel,.table-panel{{background:var(--paper);border:1px solid var(--line);padding:15px}}.text-panel p+p{{margin-top:9px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#e9ece9}}
.plots{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:12px 0 28px}}.plot{{background:var(--paper);border:1px solid var(--line)}}
.plot h3,.plot p{{padding:10px 12px 0}}.plot p{{color:var(--muted);font-size:12px}}.plot img{{display:block;width:100%;height:auto}}
.plot.wide{{grid-column:1/-1}}.examples{{margin:30px 0}}.case{{border-top:2px solid var(--line);padding:16px 0 24px}}
.all-head{{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:24px 0;margin:20px 0 30px}}
.browser-tools{{display:flex;gap:9px;align-items:end;flex-wrap:wrap;margin:13px 0}}.browser-tools label{{display:grid;gap:3px;color:var(--muted);font-size:12px}}
select{{min-width:170px;border:1px solid #9ba59f;background:#fff;padding:7px 9px;font:inherit}}.all-reference{{width:min(420px,100%);margin:8px 0 15px}}
.all-model-row{{border-top:2px solid var(--line);padding:11px 0 18px}}.all-model-head{{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}}
.all-videos{{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:7px;overflow-x:auto}}.all-videos figure{{min-width:150px}}
.video-metrics{{color:#d8dedb;padding:5px 7px;font-size:10px;line-height:1.35}}.pending-score{{color:#d7ad70}}
.case-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}}.case-head code{{font-size:11px;overflow-wrap:anywhere}}
.row-controls{{display:flex;gap:7px;flex-wrap:wrap}}button{{border:1px solid #9ba59f;background:#fff;padding:7px 10px;cursor:pointer;font:inherit}}button:hover{{background:#edf2ef}}
.videos{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin:11px 0}}figure{{margin:0;background:#171918}}figcaption{{color:#fff;padding:6px 8px;font-weight:700}}
video{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}}.metric-table{{overflow:auto}}.metric-table table{{font-size:12px}}
.t-more{{color:var(--t);font-weight:700}}.s-more{{color:var(--s);font-weight:700}}.selection{{font-size:12px;margin-top:6px}}
.downloads{{border-top:1px solid var(--line);padding-top:16px;margin:30px 0}}.downloads a{{margin-right:15px}}
@media(max-width:950px){{.facts,.analysis,.plots,.videos{{grid-template-columns:1fr 1fr}}.analysis .table-panel{{grid-column:1/-1}}}}
@media(max-width:650px){{.facts,.analysis,.plots,.videos{{grid-template-columns:1fr}}.case-head{{display:block}}.row-controls{{margin-top:9px}}}}
</style></head><body>
<header><h1>S/T 等 Head 数量控制：指标与代表 Case</h1>
<p class="sub">严格比较同 model、seed、case、去噪阶段和 replicate；exact k=5 还固定使用相同 block 9/15/16/17/28。更新 {updated}。</p>
<nav class="links"><a href="/">8946 总入口</a><a href="../index.html">17 项动态指标页</a><a href="../../cases/index.html">全部 Case 视频</a></nav></header>
<main>
<section class="facts"><div class="fact"><strong>5 vs 5</strong><span>S/T exact head 数</span></div>
<div class="fact"><strong>20 × 2</strong><span>cases × seeds</span></div><div class="fact"><strong>3</strong><span>模型</span></div>
<div class="fact"><strong>{complete}/17</strong><span>核心视频近完整指标</span></div></section>
<section class="all-head"><h2>Baseline 与分类别 All-head 消融</h2>
<p class="section-note">两组覆盖不同：S/T/S+T 分阶段结果为 Seed 851 × 20 cases，已有完整评分；五类全程结果为 1 个代表 case，计划 {len(all_head_browser["full_seeds_all"])} seeds，目前 {len(all_head_browser["full_seeds"])} seeds 的 Baseline/S/T/P/C/G 六组视频完整。它包含 S(159)、T(13)、P(82)、C(20)、G(75) 的全部分类 head。由于类别 head 数不相等，本节用于观察真实 all-head 效应，不能替代后面的 k=5 等数量因果对照。</p>
<div class="plots"><article class="plot"><h3>Baseline 与 All-S/T/S+T 分数曲线</h3><p>虚线和灰色带为同模型未消融 baseline；彩色线为类别 all-head 消融。</p><img src="all_head_baseline_score_curves.png" alt="all head baseline score curves"></article>
<article class="plot"><h3>All-head 相对 Baseline 的有符号热力图</h3><p>每个指标独立色标；红色为分数提高，蓝色为分数下降。</p><img src="all_head_signed_delta_heatmap.png" alt="all head signed delta heatmap"></article></div>
<h3>逐 Case 浏览</h3><div class="browser-tools">
<label>Case<select id="all-case"></select></label>
<label>结果类型<select id="all-mode"><option value="phased">S/T/S+T 分阶段（有指标）</option><option value="full">S/T/P/C/G 全去噪过程</option></select></label>
<label id="seed-label" hidden>Seed<select id="all-seed"></select></label>
<label id="stage-label">去噪阶段<select id="all-stage"></select></label>
</div><div id="all-prompt" class="section-note"></div><div id="all-source" class="all-reference"></div><div id="all-browser"></div>
</section>
<section class="analysis"><div class="text-panel"><h2>怎么读</h2>
<p><b>绝对影响</b>定义为 <code>|消融分数 - 同 case baseline 分数|</code>。热力图中的 <code>|ΔS|-|ΔT|</code> 小于 0，表示 T 消融令指标变化更大；它衡量“扰动强度”，不等同于质量一定更差。</p>
<p><b>有符号变化</b>直接使用 <code>消融分数 - baseline 分数</code>。本页 4 项指标均为越高越好，因此负值才表示质量下降。</p>
<p><b>当前结论</b>：在 head 数和 block 位置同时控制的 exact k=5 中，四项完整指标的平均绝对变化均是 T 大于 S，且 0-10 步差异更明显。这说明此前 S/T 差异不能仅由 head 数量不平衡解释。但它仍是 5-head 小子集结论，不能直接外推到 all-S 159 heads 与 all-T 13 heads 的非线性联合消融。</p>
</div><div class="table-panel"><h2>Exact k=5 总体配对结果</h2>
<table><thead><tr><th>指标</th><th>|ΔS|-|ΔT|</th><th>Case bootstrap 95% CI</th><th>方向</th></tr></thead><tbody>{summary_rows}</tbody></table>
</div></section>
<section><h2>曲线分析</h2><p class="section-note">阴影为按 case 聚合后的 bootstrap 95% CI；每个 case 内先平均模型、seed 和 replicate，避免重复测量虚增样本量。</p>
<div class="plots"><article class="plot"><h3>绝对指标影响</h3><p>数值越大，消融相对 baseline 造成的变化越大。</p><img src="absolute_impact_curves.png" alt="absolute impact curves"></article>
<article class="plot"><h3>有符号质量变化</h3><p>0 为 baseline；负值表示该指标下降。</p><img src="signed_delta_curves.png" alt="signed delta curves"></article></div></section>
<section><h2>热力图分析</h2><p class="section-note">所有热力图都显示 |ΔS|-|ΔT|；蓝色负值表示 T 影响更大，红色正值表示 S 影响更大。</p>
<div class="plots"><article class="plot"><h3>模型 × 去噪阶段</h3><p>Exact k=5，定位模型和阶段差异。</p><img src="exact_st_model_stage_heatmap.png" alt="model stage heatmap"></article>
<article class="plot"><h3>Replicate 稳定性</h3><p>Exact k=5 与 depth-matched k=8 的重复采样对照。</p><img src="replicate_stability_heatmap.png" alt="replicate stability heatmap"></article>
<article class="plot wide"><h3>17 项指标覆盖率</h3><p>橙色项目仍是部分覆盖，不能与四项接近完整的 CPU 指标作同等强度结论。</p><img src="metric_coverage.png" alt="metric coverage"></article></div></section>
{representative_html(representatives, manifest["model_labels"])}
<section class="downloads"><h2>数据下载</h2><p>
<a href="paired_complete_metrics.csv">完整指标逐视频配对 CSV</a>
<a href="st_exact_and_depth_pairs.csv">S/T 成对 CSV</a>
<a href="aggregate_summary.csv">聚合与置信区间 CSV</a>
<a href="all_head_phased_metrics.csv">All-head 分阶段指标 CSV</a>
<a href="all_head_browser.json">All-head 视频索引 JSON</a>
<a href="representative_cases.json">代表 Case JSON</a></p></section>
</main>
<script>
function playRow(id,restart){{document.querySelectorAll(`#${{id}} video`).forEach(video=>{{if(restart)video.currentTime=0;video.play().catch(()=>{{}})}})}}
function pauseRow(id){{document.querySelectorAll(`#${{id}} video`).forEach(video=>video.pause())}}
let AH=null;
const ah=id=>document.getElementById(id);
function metricText(item){{if(!item||!item.metrics)return"<span class='pending-score'>指标尚未计算</span>";const labels={{physics_iq_with_context:"PIQ-c",physics_iq_without_context:"PIQ-n",pmf_with_context:"PMF-c",pmf_without_context:"PMF-n"}};return Object.entries(labels).map(([key,label])=>`${{label}} ${{item.metrics[key]===null||item.metrics[key]===undefined?"Pending":Number(item.metrics[key]).toFixed(3)}}`).join("<br>")}}
function ahFigure(label,item){{return item?`<figure><figcaption>${{label}}</figcaption><video preload="none" muted playsinline src="${{item.video}}"></video><div class="video-metrics">${{metricText(item)}}</div></figure>`:`<figure><figcaption>${{label}}</figcaption><div class="video-metrics pending-score">Pending</div></figure>`}}
function updateAllCases(){{const mode=ah("all-mode").value,current=ah("all-case").value,available=AH.cases.filter(item=>mode==="phased"?Object.keys(item.phased||{{}}).length:Object.keys(item.full_categories||{{}}).length);ah("all-case").innerHTML=available.map(item=>`<option value="${{item.id}}">${{item.id}}</option>`).join("");if(available.some(item=>item.id===current))ah("all-case").value=current}}
function renderAllHead(){{if(!AH)return;const mode=ah("all-mode").value,selected=AH.cases.find(item=>item.id===ah("all-case").value),stage=ah("all-stage").value,seed=mode==="phased"?"851":ah("all-seed").value;ah("stage-label").hidden=mode!=="phased";ah("seed-label").hidden=mode==="phased";if(!selected)return;ah("all-prompt").textContent=selected.prompt;ah("all-source").innerHTML=`<figure><figcaption>Source / GT</figcaption><video preload="metadata" muted playsinline src="${{selected.source_video}}"></video></figure>`;const variants=mode==="phased"?["baseline","S","T","ST"]:["baseline","S","T","P","C","G"];ah("all-browser").innerHTML=AH.models.map((model,index)=>{{const values=mode==="phased"?{{baseline:selected.phased[model].baseline,...selected.phased[model].stages[stage]}}:((selected.full_categories||{{}})[seed]||{{}})[model]||{{}};const id=`all-model-${{index}}`;return`<section class="all-model-row" id="${{id}}"><div class="all-model-head"><h3>${{AH.model_labels[model]}} · Seed ${{seed}}</h3><div class="row-controls"><button onclick="playRow('${{id}}',true)">从头播放本行</button><button onclick="pauseRow('${{id}}')">暂停本行</button></div></div><div class="all-videos">${{variants.map(role=>ahFigure(AH.role_labels[role],values[role])).join("")}}</div></section>`}}).join("")}}
fetch("all_head_browser.json").then(response=>response.json()).then(data=>{{AH=data;ah("all-stage").innerHTML=data.phased_stages.map(value=>`<option value="${{value}}">${{value.slice(0,2)}}-${{value.slice(3)}}</option>`).join("");ah("all-seed").innerHTML=data.full_seeds.map(value=>`<option value="${{value}}">${{value}}</option>`).join("");ah("all-seed").value="851";updateAllCases();["all-case","all-stage","all-seed"].forEach(id=>ah(id).addEventListener("change",renderAllHead));ah("all-mode").addEventListener("change",()=>{{updateAllCases();renderAllHead()}});renderAllHead()}})
</script></body></html>"""


def integrate_entry() -> None:
    root = GALLERY / "index.html"
    text = root.read_text(encoding="utf-8")
    href = "head-role-dose-control-pilot/metrics/s-t-head-count-control/index.html"
    if href not in text:
        marker = "<section><h2>指标分析</h2><div class='entries'>"
        entry = (
            f"<a class='entry' href='{href}' "
            "data-search='s/t 等 head 数量控制 指标曲线 热力图 代表case exact block k5'>"
            "<span class='entry-title'>S/T 等 Head 数量控制分析</span>"
            "<span class='entry-description'>Exact k=5 与 depth-matched k=8 的曲线、热力图、覆盖率和代表视频。</span>"
            f"<code>/{href}</code></a>"
        )
        if marker not in text:
            raise RuntimeError("Cannot find metrics section in gallery root index")
        text = text.replace(marker, marker + entry, 1)
        text = text.replace("18个入口", "19个入口", 1)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = re.sub(r"更新 \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", f"更新 {stamp}", text, count=1)
    root.write_text(text, encoding="utf-8")

    metrics_index = PILOT / "metrics" / "index.html"
    text = metrics_index.read_text(encoding="utf-8")
    link = '<a href="s-t-head-count-control/index.html">S/T 等数量专题分析</a>'
    if link not in text:
        needle = '<nav class="links">'
        text = text.replace(needle, needle + link, 1)
        metrics_index.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paired, _ = build_paired_rows(manifest)
    st_pairs = build_st_pairs(paired)
    summary = aggregate_summary(st_pairs)
    representatives = select_representatives(st_pairs, manifest)
    all_head_rows, all_head_cases = load_phased_all_head()
    all_head_browser = add_full_category_videos(all_head_cases)

    plot_curves(
        paired,
        "abs_delta",
        "absolute_impact_curves.png",
        "Absolute metric impact relative to the paired baseline",
    )
    plot_curves(
        paired,
        "delta",
        "signed_delta_curves.png",
        "Signed score change relative to the paired baseline",
    )
    build_heatmaps(st_pairs)
    plot_all_head_score_curves(all_head_rows)
    plot_all_head_delta_heatmap(all_head_rows)
    coverage = plot_coverage(manifest)

    paired_columns = list(paired[0].keys())
    st_columns = [
        key for key in st_pairs[0].keys() if not key.startswith("_")
    ]
    write_csv(OUT / "paired_complete_metrics.csv", paired, paired_columns)
    write_csv(OUT / "st_exact_and_depth_pairs.csv", st_pairs, st_columns)
    write_csv(OUT / "aggregate_summary.csv", summary, list(summary[0].keys()))
    write_csv(
        OUT / "all_head_phased_metrics.csv",
        all_head_rows,
        list(all_head_rows[0].keys()),
    )
    (OUT / "all_head_browser.json").write_text(
        json.dumps(all_head_browser, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "representative_cases.json").write_text(
        json.dumps(representatives, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "index.html").write_text(
        build_html(manifest, summary, representatives, coverage, all_head_browser),
        encoding="utf-8",
    )
    integrate_entry()
    print(f"paired rows: {len(paired)}")
    print(f"S/T pairs: {len(st_pairs)}")
    print(f"all-head phased metric rows: {len(all_head_rows)}")
    print(f"representative cases: {len(representatives)}")
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
