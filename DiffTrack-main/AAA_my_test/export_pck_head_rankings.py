#!/usr/bin/env python3
"""Export per-step, per-head PCK@32 rankings for the five comparison series."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case")
HEADWISE_SUMMARY = ROOT / "block_step_head_summary.csv"
COMBINED_SUMMARY = ROOT / "three_model_combined_summary.csv"
LEGACY_ROOT = Path("/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50")
OUTPUT_DIR = Path("/data/gaoya/agent-data/outputs/pck_head_rankings")
JSON_PATH = OUTPUT_DIR / "pck_head_rankings.json"
MD_PATH = OUTPUT_DIR / "pck_head_rankings.md"

STEPS = tuple(range(40))
BLOCKS = tuple(range(30))
HEADS = tuple(range(24))
EXPECTED_HEADS = len(BLOCKS) * len(HEADS)
REFERENCE_SERIES = ("gt", "lora", "baseline", "combined")
SERIES = ("legacy_s039", *REFERENCE_SERIES)
LABELS = {
    "legacy_s039": "Legacy S039",
    "gt": "GT teacher-forced",
    "lora": "LoRA",
    "baseline": "Wan2.2 Baseline",
    "combined": "Three-model combined",
}
PROVENANCE = {
    "common": {
        "metric": "PCK@32, in percent",
        "physical_heads": "30 blocks x 24 heads = 720 physical heads",
        "source_steps": "40 denoising/source steps, S000-S039",
        "views": {
            "s039": "rank each series by its S039 PCK@32",
            "all_steps_mean": (
                "for every physical head, arithmetic mean of its S000-S039 "
                "PCK@32 values, then rank the 720 heads again"
            ),
        },
        "pairwise": (
            "Top-K overlap and Pearson/Spearman correlations are computed after "
            "aligning the same 720 physical (block, head) IDs inside each view. "
            "They are not paired by case or seed."
        ),
    },
    "legacy_s039": {
        "runs": "6 cases x 50 unique seeds = 300 completed runs",
        "cases": [
            "0613pybullet_sample_000301_w000",
            "0613pybullet_sample_000331_w001",
            "0613pybullet_sample_001455_w000",
            "0613pybullet_sample_000336_w001",
            "0613pybullet_sample_001460_w002",
            "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed",
        ],
        "seed_file": "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds.txt",
        "source_video_root": "/data/gaoya/AAA_test_video/0623/test/v2v/basemodel/wan2p2_ti2v5B_frame49",
        "input_json_root": "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons",
        "model": "Wan2.2 TI2V 5B, legacy DiffSynth backend",
        "generation": "704x1280, 49 frames, 40 sampling steps, seed-specific generation, CFG 5.0, sample_shift 5.0",
        "query_protocol": (
            "first pixel frame / first latent frame object queries; "
            "query_pixel_frame=0, query_latent_index=0, latent anchor frames 0,4,...,48"
        ),
        "query_region_distribution": (
            "50 runs with 8 query points / 1 object region; 250 runs with "
            "16 query points / 2 object regions"
        ),
        "region_protocol": (
            "GroundingDINO first-frame boxes -> SAM2 video propagation; object regions only"
        ),
        "capture_protocol": (
            "self-attention post RMSNorm and post 3D RoPE, pre flash-attention; "
            "positive conditional first call only; per-target-frame Q-to-K argmax; "
            "no averaging across heads"
        ),
        "aggregation": (
            "micro aggregate: sum correct32/comparisons/error_sum over all 300 runs, "
            "then PCK@32 = 100 * correct32 / comparisons"
        ),
        "effective_comparisons": (
            "45,156 visible object-query point/latent comparisons per S/B/H; "
            "1,806,240 comparisons per physical head when all 40 steps are summed"
        ),
        "source_scripts": [
            "/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_ti2v_firstlatent_pck_worker.py",
            "/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/aggregate_legacy_ti2v_firstlatent_pck50.py",
            "/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/legacy_ti2v_firstlatent_common.py",
        ],
    },
    "three_model_reference": {
        "runs": "3 models x 50 case directories; no seed sweep, manifests use seed=42",
        "validation": (
            "validation.json PASS: each of gt/lora/baseline has 50 cases, "
            "50 complete cases, 50 validated cases"
        ),
        "models": {
            "gt": "Wan2.2-TI2V-5B, GT teacher-forced / clean-prefix protocol",
            "lora": "Wan OpenVid 0613 PyBullet LoRA, checkpoint step-000500",
            "baseline": "Wan2.2 TI2V 5B baseline, context-aware protocol without LoRA",
        },
        "generation": "512x896, 40 sampling steps, context_pixel_frames=8, seed=42",
        "query_protocol": (
            "SAM2 region cache, objects scope only; query_pixel_frame=4, "
            "query_latent_index=1, latent anchor frames 0,4,...,24, "
            "future latent indices 2-6"
        ),
        "query_region_distribution": (
            "per-case object region counts vary across the 50 cases: "
            "15 cases with 2 regions (16 query points), 20 cases with 3 regions "
            "(24 query points), and 15 cases with 4 regions (32 query points)"
        ),
        "capture_protocol": (
            "video self-attention post RMSNorm and post 3D RoPE, pre flash-attention; "
            "headwise direct-token argmax; no averaging across heads"
        ),
        "per_model_object_scope": {
            "gt": "50 valid object cases, 3,881 comparisons per S/B/H",
            "lora": "49 valid object cases, 3,893 comparisons per S/B/H",
            "baseline": "49 valid object cases, 3,064 comparisons per S/B/H",
        },
        "macro_vs_pooled": (
            "rankings use macro_pck32, the unweighted mean of per-case PCK values; "
            "pooled_pck32 exists in source CSV but is not used for these rankings"
        ),
        "combined": (
            "Three-model combined is an equal-weight arithmetic mean of the GT, "
            "LoRA, and Baseline macro metrics for the same step/block/head; "
            "object-scope valid_cases=148 of total_cases=150 and comparisons=10,838 per S/B/H"
        ),
        "source_scripts": [
            "/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/aggregate_allblocks_allsteps_headwise_50case.py",
            "/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/aggregate_three_model_combined_rankings.py",
            "/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/launch_wan_gt_toy_analysis_multigpu.py",
            "/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/launch_lorav2v_toy_analysis_multigpu.py",
            "/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/launch_wan22_baseline_toy_analysis_multigpu.py",
        ],
    },
}


def finite(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def step_id(step: int) -> str:
    return f"S{step:03d}"


def normalize_row(row: dict, pck_key: str, error_key: str) -> dict:
    record = {
        "block": int(row["block"] if "block" in row else row["layer"]),
        "head": int(row["head"]),
        "pck32": finite(row[pck_key]),
        "mean_error_px": finite(row.get(error_key)),
        "comparisons": int(row.get("comparisons", 0)),
    }
    for field in ("cases", "valid_cases", "total_cases"):
        if row.get(field) not in (None, ""):
            record[field] = int(row[field])
    for field in ("timestep", "sigma"):
        value = finite(row.get(field))
        if value is not None:
            record[field] = value
    return record


def rank_rows(rows: list[dict], series: str, step: int, label: str | None = None) -> dict:
    keys = {(row["block"], row["head"]) for row in rows}
    expected = {(block, head) for block in BLOCKS for head in HEADS}
    if keys != expected:
        missing = sorted(expected - keys)[:5]
        extra = sorted(keys - expected)[:5]
        raise ValueError(
            f"{series}/{label or step_id(step)} has {len(rows)} rows; "
            f"missing={missing}, extra={extra}"
        )
    if any(row["pck32"] is None for row in rows):
        raise ValueError(f"{series}/{label or step_id(step)} contains non-finite PCK@32")

    ordered = sorted(
        rows,
        key=lambda row: (
            -row["pck32"],
            row["block"],
            row["head"],
        ),
    )
    ranked = []
    for rank, row in enumerate(ordered, start=1):
        item = {"rank": rank, **row}
        item["head_id"] = f"L{row['block']:02d}/H{row['head']:02d}"
        ranked.append(item)
    return {
        "step": step if label is None else None,
        "step_id": label or step_id(step),
        "head_count": EXPECTED_HEADS,
        "sorted_by": [
            {"field": "pck32", "direction": "desc"},
            {"field": "block", "direction": "asc"},
            {"field": "head", "direction": "asc"},
        ],
        "ranked_heads": ranked,
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    left_centered = left_array - left_array.mean()
    right_centered = right_array - right_array.mean()
    denominator = float(
        np.sqrt(
            np.dot(left_centered, left_centered)
            * np.dot(right_centered, right_centered)
        )
    )
    if denominator <= 0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def pairwise_comparison(
    left_id: str,
    right_id: str,
    left_step: dict,
    right_step: dict,
) -> dict:
    left_rows = {(row["block"], row["head"]): row for row in left_step["ranked_heads"]}
    right_rows = {(row["block"], row["head"]): row for row in right_step["ranked_heads"]}
    keys = sorted(left_rows.keys() & right_rows.keys())
    left_values = [left_rows[key]["pck32"] for key in keys]
    right_values = [right_rows[key]["pck32"] for key in keys]
    deltas = np.asarray(left_values, dtype=np.float64) - np.asarray(right_values, dtype=np.float64)
    overlaps = {}
    for top_k in (10, 30, 50, 100):
        left_top = {key for key in keys if left_rows[key]["rank"] <= top_k}
        right_top = {key for key in keys if right_rows[key]["rank"] <= top_k}
        common_count = len(left_top & right_top)
        union_count = len(left_top | right_top)
        overlaps[f"Top{top_k}"] = {
            "k": top_k,
            "common_count": common_count,
            "coverage_pct": 100.0 * common_count / top_k,
            "jaccard": common_count / union_count if union_count else 0.0,
            "union_count": union_count,
        }
    return {
        "left_series": left_id,
        "right_series": right_id,
        "pair_count": len(keys),
        "pearson_pck32": pearson(left_values, right_values),
        "spearman_pck32": pearson(average_ranks(left_values), average_ranks(right_values)),
        "mean_delta_left_minus_right": float(deltas.mean()),
        "mean_abs_delta_pck32": float(np.abs(deltas).mean()),
        "overlaps": overlaps,
    }


def build_view_pairwise_comparisons(views: dict) -> dict:
    comparisons = {}
    for view_key, view in views.items():
        available = list(view["series"])
        step_pairs = {}
        for left_index, left_id in enumerate(available):
            for right_id in available[left_index + 1 :]:
                pair_id = f"{left_id}__{right_id}"
                step_pairs[pair_id] = pairwise_comparison(
                    left_id,
                    right_id,
                    view["series"][left_id],
                    view["series"][right_id],
                )
        comparisons[view_key] = step_pairs
    return comparisons


def load_reference_series() -> dict[str, dict[str, list[dict]]]:
    series_rows = {series: {step_id(step): [] for step in STEPS} for series in REFERENCE_SERIES}
    with HEADWISE_SUMMARY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != "objects":
                continue
            model = row["model"]
            if model not in ("gt", "lora", "baseline"):
                continue
            step = int(row["step"])
            series_rows[model][step_id(step)].append(
                normalize_row(row, "macro_pck32", "macro_mean_error_px")
            )

    with COMBINED_SUMMARY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != "objects":
                continue
            step = int(row["step"])
            series_rows["combined"][step_id(step)].append(
                normalize_row(row, "macro_pck32", "macro_mean_error_px")
            )
    return series_rows


def load_legacy_steps() -> dict[str, list[dict]]:
    counts_path = LEGACY_ROOT / "aggregate" / "combined_counts.npz"
    with np.load(counts_path) as arrays:
        correct = np.asarray(arrays["correct32"], dtype=np.float64)
        comparisons = np.asarray(arrays["comparisons"], dtype=np.float64)
        error_sum = np.asarray(arrays["error_sum"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pck = np.divide(
            100.0 * correct,
            comparisons,
            out=np.full_like(correct, np.nan),
            where=comparisons > 0,
        )
        mean_error = np.divide(
            error_sum,
            comparisons,
            out=np.full_like(error_sum, np.nan),
            where=comparisons > 0,
        )
    result = {}
    for step in STEPS:
        rows = []
        for block in BLOCKS:
            for head in HEADS:
                pck_value = pck[step, block, head]
                error_value = mean_error[step, block, head]
                rows.append(
                    {
                        "block": block,
                        "head": head,
                        "pck32": float(pck_value) if np.isfinite(pck_value) else None,
                        "mean_error_px": float(error_value)
                        if np.isfinite(error_value)
                        else None,
                        "comparisons": int(comparisons[step, block, head]),
                    }
                )
        result[step_id(step)] = rows
    return result


def average_rows(step_rows: dict[str, list[dict]], series: str) -> list[dict]:
    by_head = {}
    for rows in step_rows.values():
        for row in rows:
            key = (row["block"], row["head"])
            by_head.setdefault(key, []).append(row)
    averaged = []
    for (block, head), rows in sorted(by_head.items()):
        pck_values = [row["pck32"] for row in rows if row["pck32"] is not None]
        error_values = [row["mean_error_px"] for row in rows if row["mean_error_px"] is not None]
        if len(pck_values) != len(step_rows):
            raise ValueError(f"{series}/all_steps_mean has missing PCK@32 values for L{block}/H{head}")
        averaged.append(
            {
                "block": block,
                "head": head,
                "pck32": float(np.mean(pck_values)),
                "mean_error_px": float(np.mean(error_values)) if error_values else None,
                "comparisons": int(sum(row["comparisons"] for row in rows)),
                "step_count": len(rows),
            }
        )
    return averaged


def build_payload() -> dict:
    reference_rows = load_reference_series()
    raw_rows = {**reference_rows, "legacy_s039": load_legacy_steps()}
    view_rows = {
        "s039": {series: raw_rows[series]["S039"] for series in SERIES},
        "all_steps_mean": {
            series: average_rows(raw_rows[series], series) for series in SERIES
        },
    }
    view_labels = {
        "s039": "S039",
        "all_steps_mean": "所有 Step 平均",
    }
    views = {}
    for view_key, series_rows in view_rows.items():
        ranked_series = {
            series: rank_rows(
                rows,
                series,
                39,
                label=None if view_key == "s039" else "all_steps_mean",
            )
            for series, rows in series_rows.items()
        }
        views[view_key] = {
            "label": view_labels[view_key],
            "source_steps": ["S039"] if view_key == "s039" else [step_id(step) for step in STEPS],
            "series": ranked_series,
        }
    pairwise_comparisons = build_view_pairwise_comparisons(views)
    series_metadata = {
        "legacy_s039": {
            "label": LABELS["legacy_s039"],
            "scope": "S039 only for the S039 view; all 40 source steps for the average view",
            "aggregation": "micro aggregate over 6 cases x 50 seeds per source step",
            "source_files": [str(LEGACY_ROOT / "aggregate" / "combined_counts.npz")],
            "provenance": PROVENANCE["legacy_s039"],
        },
        "gt": {
            "label": LABELS["gt"],
            "scope": "objects",
            "aggregation": "50-case per-step macro PCK@32; arithmetic mean across S000-S039 in the average view",
            "source_files": [str(HEADWISE_SUMMARY)],
            "provenance": PROVENANCE["three_model_reference"],
        },
        "lora": {
            "label": LABELS["lora"],
            "scope": "objects",
            "aggregation": "50-case per-step macro PCK@32; arithmetic mean across S000-S039 in the average view",
            "source_files": [str(HEADWISE_SUMMARY)],
            "provenance": PROVENANCE["three_model_reference"],
        },
        "baseline": {
            "label": LABELS["baseline"],
            "scope": "objects",
            "aggregation": "50-case per-step macro PCK@32; arithmetic mean across S000-S039 in the average view",
            "source_files": [str(HEADWISE_SUMMARY)],
            "provenance": PROVENANCE["three_model_reference"],
        },
        "combined": {
            "label": LABELS["combined"],
            "scope": "objects",
            "aggregation": "equal-weight mean of GT, LoRA, and Baseline macro PCK@32 per step; arithmetic mean across steps in the average view",
            "source_files": [str(COMBINED_SUMMARY)],
            "provenance": PROVENANCE["three_model_reference"],
        },
    }
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "PCK@32",
        "unit": "percent",
        "head_identity": "(block, head)",
        "dimensions": {
            "source_steps": 40,
            "views": 2,
            "blocks": 30,
            "heads": 24,
            "heads_per_view": EXPECTED_HEADS,
        },
        "sort_rule": "PCK@32 descending; block/head ascending for deterministic ties",
        "series_order": list(SERIES),
        "series": series_metadata,
        "views": views,
        "available_series_by_view": {
            view_key: list(view["series"]) for view_key, view in views.items()
        },
        "pairwise_comparisons_by_view": pairwise_comparisons,
        "provenance": PROVENANCE,
    }


def write_markdown(payload: dict) -> None:
    series = payload["series"]
    lines = [
        "# PCK@32 Head Rankings",
        "",
        "这份 JSON 只保留两个展示口径：`S039` 和 `所有 Step 平均`。",
        "每个口径包含 Legacy S039、GT teacher-forced、LoRA、Wan2.2 Baseline、三模型综合五组数据，",
        "每组都按相同的 720 个物理 `(Block, Head)` 重新排序。",
        "这份导出对应的是 50-case 三模型 headwise 汇总加 6-case × 50-seed 的 Legacy 汇总；",
        "它不同于旧的 `neighbor-diagonal-ranking?v=4` 页面，后者只是在 S039 上对 5 个 case 做三模型对角线统计。",
        "",
        "## 文件结构",
        "",
        "- JSON：`pck_head_rankings.json`",
        "- 本说明：`pck_head_rankings.md`",
        "- 排名数据：`views.<view_id>.series.<series_id>.ranked_heads`",
        "- 两两比较：`pairwise_comparisons_by_view.<view_id>`",
        "",
        "每个 `ranked_heads` 都是长度 720、从 rank 1 到 rank 720 排好的数组。",
        "读取某个 view 的 Top30 时，直接取数组前 30 项。",
        "",
        "## 两个展示口径",
        "",
        "| view id | 含义 | 使用的数据 |",
        "|---|---|---|",
        "| `s039` | 当前最终时间步 | S039 原始排名 |",
        "| `all_steps_mean` | 所有时间步平均 | 对每个物理 Head 的 S000–S039 PCK@32 做算术平均后重新排序 |",
        "",
        "平均 view 平均的是每个 Head 的 PCK 数值，不是各个 step 的 rank；因此得到的是“平均性能排序”。",
        "Legacy 的平均 view 使用 Legacy 原始 40-step 聚合中的同样计算；S039 view 使用你指定的 Legacy S039。",
        "",
        "## 统计追溯与执行条件",
        "",
        "### 共同设置",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| metric | {payload['metric']} |",
        f"| views | {payload['dimensions']['views']} 个视图（`S039` 与 `所有 Step 平均`） |",
        f"| source steps | {payload['dimensions']['source_steps']} 个 step（S000-S039） |",
        f"| physical heads | {payload['dimensions']['blocks']} 个 block × {payload['dimensions']['heads']} 个 head = {payload['dimensions']['heads_per_view']} 个物理 Head |",
        "| pairwise 对齐单位 | 同一视图内的 720 个物理 `(Block, Head)`；不按 case 或 seed 配对 |",
        "",
        "### Legacy S039",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| runs | {PROVENANCE['legacy_s039']['runs']} |",
        f"| cases | 6 个 case：`0613pybullet_sample_000301_w000`、`0613pybullet_sample_000331_w001`、`0613pybullet_sample_001455_w000`、`0613pybullet_sample_000336_w001`、`0613pybullet_sample_001460_w002`、`physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed` |",
        f"| seed file | `{PROVENANCE['legacy_s039']['seed_file']}`，50 个唯一 seed |",
        "| temporal protocol | 704×1280，49 帧，40 sampling steps，seed-specific generation，CFG 5.0，sample_shift 5.0 |",
        "| query protocol | first pixel frame / first latent frame object queries；query_pixel_frame=0，query_latent_index=0，latent anchor frames 0,4,...,48 |",
        f"| query region distribution | {PROVENANCE['legacy_s039']['query_region_distribution']} |",
        "| region / capture / matching | GroundingDINO first-frame boxes + SAM2 传播；self-attention post RMSNorm / post 3D RoPE / pre flash-attention；positive conditional first call only；per-target-frame Q-to-K argmax；no head averaging |",
        f"| aggregation | {PROVENANCE['legacy_s039']['aggregation']} |",
        f"| effective comparisons | {PROVENANCE['legacy_s039']['effective_comparisons']} |",
        "| source scripts | `run_legacy_ti2v_firstlatent_pck_worker.py`、`aggregate_legacy_ti2v_firstlatent_pck50.py`、`legacy_ti2v_firstlatent_common.py` |",
        "",
        "### 三模型参考系列",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| runs | {PROVENANCE['three_model_reference']['runs']} |",
        f"| validation | {PROVENANCE['three_model_reference']['validation']} |",
        f"| model protocol | GT={PROVENANCE['three_model_reference']['models']['gt']}；LoRA={PROVENANCE['three_model_reference']['models']['lora']}；Baseline={PROVENANCE['three_model_reference']['models']['baseline']} |",
        f"| generation | {PROVENANCE['three_model_reference']['generation']} |",
        f"| query protocol | {PROVENANCE['three_model_reference']['query_protocol']} |",
        f"| query region distribution | {PROVENANCE['three_model_reference']['query_region_distribution']} |",
        f"| capture / matching | {PROVENANCE['three_model_reference']['capture_protocol']} |",
        f"| per-model object scope | GT={PROVENANCE['three_model_reference']['per_model_object_scope']['gt']}；LoRA={PROVENANCE['three_model_reference']['per_model_object_scope']['lora']}；Baseline={PROVENANCE['three_model_reference']['per_model_object_scope']['baseline']} |",
        f"| macro vs pooled | {PROVENANCE['three_model_reference']['macro_vs_pooled']} |",
        f"| combined | {PROVENANCE['three_model_reference']['combined']} |",
        "| source scripts | `aggregate_allblocks_allsteps_headwise_50case.py`、`aggregate_three_model_combined_rankings.py`、`launch_wan_gt_toy_analysis_multigpu.py`、`launch_lorav2v_toy_analysis_multigpu.py`、`launch_wan22_baseline_toy_analysis_multigpu.py` |",
        "",
        "## 五组数据",
        "",
        "| series id | 显示名称 | 统计口径 |",
        "|---|---|---|",
    ]
    for series_id in payload["series_order"]:
        item = series[series_id]
        lines.append(f"| `{series_id}` | {item['label']} | {item['aggregation']} |")

    lines.extend(
        [
            "",
            "## 排序规则",
            "",
            "1. PCK@32 从高到低排序。",
            "2. PCK@32 相同时，按 Block、Head 升序排列，保证结果稳定。",
            "3. 平均误差保留在记录中用于分析，但不改变 PCK 排名。",
            "4. `rank` 是当前 view、当前系列内的连续排名，范围为 1–720。",
            "",
            "## Head 记录字段",
            "",
            "| 字段 | 含义 |",
            "|---|---|",
            "| `rank` | 当前 view、当前系列内的 PCK 排名 |",
            "| `block` | Block 编号，0–29 |",
            "| `head` | Head 编号，0–23 |",
            "| `head_id` | 可读标识，例如 `L17/H08` |",
            "| `pck32` | PCK@32 百分比 |",
            "| `mean_error_px` | 平均像素误差 |",
            "| `comparisons` | 该记录参与的有效比较数；平均 view 为 40 step 合计 |",
            "| `step_count` | 平均 view 中参与平均的 step 数 |",
            "",
            "## 读取示例",
            "",
            "```python",
            "import json",
            "",
            "with open(\"pck_head_rankings.json\", encoding=\"utf-8\") as handle:",
            "    data = json.load(handle)",
            "",
            "top30_s039 = data[\"views\"][\"s039\"][\"series\"][\"combined\"][\"ranked_heads\"][:30]",
            "top30_mean = data[\"views\"][\"all_steps_mean\"][\"series\"][\"combined\"][\"ranked_heads\"][:30]",
            "```",
            "",
            "## 所有排序之间的重叠度与相关性",
            "",
            "JSON 的 `pairwise_comparisons_by_view` 保存两个 view 内五组数据的全部两两比较。",
            "每个 view 有 10 个系列对，共同使用 720 个物理 `(Block, Head)`。",
            "Top-K 交集按各自排序的前 K 个 Head 计算，K 为 10、30、50、100。",
            "`coverage_pct` 是交集占 K 的百分比，`jaccard` 是交集除以并集。",
            "Pearson/Spearman 是对齐后的 720 个实际 PCK@32 值计算；Spearman 使用平均秩处理并列值。",
            "`mean_delta_left_minus_right` 表示左系列 PCK 减右系列 PCK，单位为百分点。",
            "",
            "每个 pair 的 JSON 字段示例：",
            "",
            "```text",
            "pairwise_comparisons_by_view.s039.gt__combined.overlaps.Top30.common_count",
            "pairwise_comparisons_by_view.all_steps_mean.gt__combined.pearson_pck32",
            "pairwise_comparisons_by_view.all_steps_mean.gt__combined.spearman_pck32",
            "```",
            "",
        ]
    )
    pairwise = payload["pairwise_comparisons_by_view"]
    for view_key in ("s039", "all_steps_mean"):
        lines.extend(
            [
                f"### {payload['views'][view_key]['label']}",
                "",
                "| 比较组合 | Top10 | Top30 | Top50 | Top100 | Pearson | Spearman |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in pairwise[view_key].values():
            left = LABELS[comparison["left_series"]]
            right = LABELS[comparison["right_series"]]
            overlap_text = []
            for top_k in (10, 30, 50, 100):
                item = comparison["overlaps"][f"Top{top_k}"]
                overlap_text.append(f"{item['common_count']}/{top_k} ({item['jaccard']:.3f})")
            lines.append(
                f"| {left} × {right} | {overlap_text[0]} | {overlap_text[1]} | "
                f"{overlap_text[2]} | {overlap_text[3]} | "
                f"{comparison['pearson_pck32']:.4f} | {comparison['spearman_pck32']:.4f} |"
            )
        lines.append("")

    lines.extend(["表中 `a/b (j)` 表示 `a` 个公共 Head / `b` 个 Top-K，括号内是 Jaccard。", "", "## 来源", ""])
    for series_id in payload["series_order"]:
        item = series[series_id]
        lines.append(f"- `{series_id}`：" + "；".join(item["source_files"]))
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    write_markdown(payload)
    print(JSON_PATH)
    print(MD_PATH)
    print(json.dumps(payload["available_series_by_view"], ensure_ascii=False))


if __name__ == "__main__":
    main()
