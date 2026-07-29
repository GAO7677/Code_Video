#!/usr/bin/env python3
"""Build block-depth distributions for Wan DiT head roles and feature subtypes."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CLASSIFICATION_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/"
    "head_classification"
)
SNAPSHOT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/"
    "partial_analysis/snapshot_20260728T0245Z/common22"
)
S_SPLIT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_feature_split/"
    "configs/s_feature_split_subsets.json"
)
GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_fulltoken_moving_pilot/gallery"
)
OUTPUT = GALLERY_ROOT / "head-role-depth-distribution"

MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
ROLES = ("S", "T", "P", "C", "G", "M")
COMMON_ROLES = ("S", "T", "P", "C", "G")
EXPECTED_COMMON_COUNTS = {"S": 159, "T": 13, "P": 82, "C": 20, "G": 75}
ROLE_LABELS = {
    "S": "S · 空间局部",
    "T": "T · 运动轨迹",
    "P": "P · 固定位置",
    "C": "C · 上下文",
    "G": "G · 全局",
    "M": "M · 混合/不稳定",
}
ROLE_PLOT_LABELS = {
    "S": "S · Spatial",
    "T": "T · Trajectory",
    "P": "P · Position",
    "C": "C · Context",
    "G": "G · Global",
    "M": "M · Mixed",
}
ROLE_COLORS = {
    "S": "#00897b",
    "T": "#d1495b",
    "P": "#7b61a8",
    "C": "#e39a27",
    "G": "#3478b8",
    "M": "#8b918e",
}
DEPTH_BANDS = {
    "Early B00-09": range(0, 10),
    "Middle B10-19": range(10, 20),
    "Late B20-29": range(20, 30),
}

# These are score components, not additional official role labels.  A head is
# assigned to exactly one subtype by its largest cross-sample mean rank.
ROLE_COMPONENTS = {
    "S": (
        ("local_enrichment", "Local enrichment"),
        ("same_frame_mass", "Same-frame mass"),
    ),
    "T": (
        ("trajectory_selectivity_log2", "Trajectory selectivity"),
        ("trajectory_enrichment", "Trajectory enrichment"),
        ("object_mean_time_distance", "Mean time distance"),
    ),
    "P": (
        ("fixed_position_enrichment", "Fixed-position enrichment"),
        ("aligned_enrichment", "Aligned enrichment"),
    ),
    "C": (
        ("object_context_enrichment", "Object-context enrichment"),
        ("full_context_enrichment", "Full-context enrichment"),
        ("object_history_bias", "History bias"),
    ),
    "G": (
        ("full_entropy", "Attention entropy"),
        ("full_mean_time_distance", "Mean time distance"),
        ("negative_same_frame_mass", "Low same-frame mass"),
    ),
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def block_counts(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    groups: tuple[str, ...] | list[str],
) -> dict[str, list[int]]:
    output = {group: [0] * 30 for group in groups}
    for row in rows:
        group = str(row[group_key])
        if group in output:
            output[group][int(row["block"])] += 1
    return output


def derive_common_rows(
    aggregate_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    labels = {
        (row["model"], int(row["block"]), int(row["head"])): row["role"]
        for row in aggregate_rows
    }
    rows = []
    for block in range(30):
        for head in range(24):
            roles = [labels[(model, block, head)] for model in MODELS]
            if len(set(roles)) == 1 and roles[0] in COMMON_ROLES:
                rows.append(
                    {
                        "role": roles[0],
                        "block": str(block),
                        "head": str(head),
                    }
                )
    counts = Counter(row["role"] for row in rows)
    observed = {role: int(counts[role]) for role in COMMON_ROLES}
    if observed != EXPECTED_COMMON_COUNTS:
        raise RuntimeError(
            f"Unexpected common stable role counts: {observed}; "
            f"expected {EXPECTED_COMMON_COUNTS}"
        )
    return rows


def depth_counts(values: dict[str, list[int]]) -> dict[str, dict[str, int]]:
    return {
        group: {
            band: int(sum(counts[block] for block in blocks))
            for band, blocks in DEPTH_BANDS.items()
        }
        for group, counts in values.items()
    }


def feature_rank_means(
    manifest: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    columns = [
        "block",
        "head",
        "trajectory_valid",
        *[
            f"rank_{component}"
            for components in ROLE_COMPONENTS.values()
            for component, _ in components
        ],
    ]
    trajectory_columns = {
        "rank_trajectory_selectivity_log2",
        "rank_trajectory_enrichment",
        "rank_object_mean_time_distance",
        "rank_fixed_position_enrichment",
        "rank_aligned_enrichment",
    }
    paths_by_model: dict[str, list[Path]] = {model: [] for model in MODELS}
    for partition in manifest["partitions"]:
        model = partition["model"]
        if model in paths_by_model:
            paths_by_model[model].append(
                Path(partition["files"]["ranks_scores"]["path"])
            )

    output: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        paths = sorted(paths_by_model[model])
        if len(paths) != 22:
            raise RuntimeError(f"{model}: expected 22 rank partitions, got {len(paths)}")
        sums = {
            column: np.zeros((30, 24), dtype=np.float64)
            for column in columns
            if column.startswith("rank_")
        }
        counts = {
            column: np.zeros((30, 24), dtype=np.int64)
            for column in sums
        }
        for path in paths:
            frame = pd.read_parquet(path, columns=columns)
            block = frame["block"].to_numpy(dtype=np.int64)
            head = frame["head"].to_numpy(dtype=np.int64)
            trajectory_valid = frame["trajectory_valid"].to_numpy(dtype=bool)
            for column in sums:
                values = frame[column].to_numpy(dtype=np.float64)
                valid = np.isfinite(values)
                if column in trajectory_columns:
                    valid &= trajectory_valid
                np.add.at(sums[column], (block[valid], head[valid]), values[valid])
                np.add.at(counts[column], (block[valid], head[valid]), 1)
        output[model] = {
            column.removeprefix("rank_"): np.divide(
                sums[column],
                counts[column],
                out=np.full((30, 24), np.nan, dtype=np.float64),
                where=counts[column] > 0,
            )
            for column in sums
        }
        print(
            f"[head-depth] aggregated feature ranks for {model}: "
            f"{len(paths)} seeds",
            flush=True,
        )
    return output


def assign_subtypes(
    role_rows: list[dict[str, str]],
    feature_means: dict[str, dict[str, np.ndarray]],
    *,
    common: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in role_rows:
        role = row["role"]
        if role not in ROLE_COMPONENTS:
            continue
        block = int(row["block"])
        head = int(row["head"])
        components = ROLE_COMPONENTS[role]
        models = MODELS if common else (row["model"],)
        means = []
        for component, _ in components:
            values = [
                feature_means[model][component][block, head]
                for model in models
            ]
            means.append(float(np.nanmean(values)))
        winner = int(np.nanargmax(means))
        sorted_means = sorted(means, reverse=True)
        output.append(
            {
                "scope": "common" if common else row["model"],
                "role": role,
                "block": block,
                "head": head,
                "subtype": components[winner][0],
                "subtype_label": components[winner][1],
                "subtype_margin": (
                    float(sorted_means[0] - sorted_means[1])
                    if len(sorted_means) > 1
                    else float("nan")
                ),
                "component_rank_means": {
                    component: value
                    for (component, _), value in zip(components, means)
                },
            }
        )
    return output


def audit_s_partition(common_subtypes: list[dict[str, Any]]) -> dict[str, Any]:
    frozen = json.loads(S_SPLIT.read_text(encoding="utf-8"))["full_partition"]
    frozen_map = {
        (int(row["block"]), int(row["head"])): row["feature_class"]
        for row in frozen["heads"]
    }
    computed_map = {
        (int(row["block"]), int(row["head"])): (
            "local_dominant"
            if row["subtype"] == "local_enrichment"
            else "same_frame_dominant"
        )
        for row in common_subtypes
        if row["role"] == "S"
    }
    mismatches = [
        {"block": key[0], "head": key[1], "frozen": frozen_map[key], "computed": value}
        for key, value in computed_map.items()
        if frozen_map.get(key) != value
    ]
    if set(frozen_map) != set(computed_map) or mismatches:
        raise RuntimeError(
            "Computed common-S subtype partition differs from the frozen partition: "
            f"keys={len(computed_map)}/{len(frozen_map)} mismatches={len(mismatches)}"
        )
    return {
        "status": "exact_match",
        "heads": len(computed_map),
        "local_dominant": int(
            sum(value == "local_dominant" for value in computed_map.values())
        ),
        "same_frame_dominant": int(
            sum(value == "same_frame_dominant" for value in computed_map.values())
        ),
        "frozen_source": str(S_SPLIT),
    }


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat_rows = []
    for row in rows:
        item = dict(row)
        if isinstance(item.get("component_rank_means"), dict):
            item["component_rank_means"] = json.dumps(
                item["component_rank_means"], ensure_ascii=False, sort_keys=True
            )
        flat_rows.append(item)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def role_heatmaps(
    model_counts: dict[str, dict[str, list[int]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        len(MODELS),
        1,
        figsize=(17.5, 9.4),
        constrained_layout=True,
    )
    image = None
    for axis, model in zip(axes, MODELS):
        values = np.asarray([model_counts[model][role] for role in ROLES])
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap="YlGnBu",
            vmin=0,
            vmax=24,
        )
        axis.set_title(MODEL_LABELS[model], loc="left", fontweight="bold")
        axis.set_yticks(
            range(len(ROLES)),
            [ROLE_PLOT_LABELS[role] for role in ROLES],
        )
        axis.set_xticks(range(30), [f"{block:02d}" for block in range(30)])
        axis.set_xlabel("DiT block")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = int(values[row, column])
                if value:
                    axis.text(
                        column,
                        row,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color="white" if value >= 13 else "#16211d",
                    )
    assert image is not None
    fig.colorbar(image, ax=axes, label="Head count in block", shrink=0.84)
    fig.suptitle(
        "Per-model aggregate head-role distribution (22 seeds × 20 cases)",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def common_role_plot(common_counts: dict[str, list[int]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(17.5, 6.3), constrained_layout=True)
    bottom = np.zeros(30, dtype=np.int64)
    blocks = np.arange(30)
    for role in COMMON_ROLES:
        values = np.asarray(common_counts[role])
        axis.bar(
            blocks,
            values,
            bottom=bottom,
            width=0.82,
            color=ROLE_COLORS[role],
            label=f"{ROLE_PLOT_LABELS[role]} · n={int(values.sum())}",
        )
        bottom += values
    axis.axvspan(-0.5, 9.5, color="#1f77b4", alpha=0.045)
    axis.axvspan(9.5, 19.5, color="#e39a27", alpha=0.045)
    axis.axvspan(19.5, 29.5, color="#2f8f67", alpha=0.045)
    axis.axvline(9.5, color="#737b77", linewidth=0.8)
    axis.axvline(19.5, color="#737b77", linewidth=0.8)
    axis.set_xticks(blocks, [f"{block:02d}" for block in blocks])
    axis.set_xlabel("DiT block")
    axis.set_ylabel("Cross-model common stable heads")
    axis.set_title(
        "Cross-model common stable role distribution",
        loc="left",
        fontweight="bold",
    )
    axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def depth_band_plot(
    model_depth: dict[str, dict[str, dict[str, int]]],
    common_depth: dict[str, dict[str, int]],
    output: Path,
) -> None:
    scopes = [*MODELS, "common"]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.3), constrained_layout=True)
    band_colors = ("#3478b8", "#e39a27", "#2f8f67")
    for axis, scope in zip(axes, scopes):
        values = common_depth if scope == "common" else model_depth[scope]
        roles = list(COMMON_ROLES if scope == "common" else ROLES)
        left = np.zeros(len(roles), dtype=np.float64)
        for color, band in zip(band_colors, DEPTH_BANDS):
            totals = np.asarray(
                [sum(values[role].values()) for role in roles],
                dtype=np.float64,
            )
            counts = np.asarray([values[role][band] for role in roles], dtype=np.float64)
            ratios = np.divide(
                counts,
                totals,
                out=np.zeros_like(counts),
                where=totals > 0,
            )
            axis.barh(roles, ratios, left=left, color=color, label=band)
            left += ratios
        axis.set_xlim(0, 1)
        axis.set_xticks((0, 0.5, 1), ("0", "50%", "100%"))
        axis.set_title(
            "Cross-model common" if scope == "common" else MODEL_LABELS[scope],
            fontweight="bold",
        )
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].legend(
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0, 1.24),
        frameon=False,
    )
    fig.suptitle("Role composition across Early / Middle / Late depth bands", y=1.06)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def subtype_heatmaps(
    common_subtypes: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        len(COMMON_ROLES),
        1,
        figsize=(17.5, 15.5),
        constrained_layout=True,
    )
    image = None
    for axis, role in zip(axes, COMMON_ROLES):
        role_rows = [row for row in common_subtypes if row["role"] == role]
        labels = [label for _, label in ROLE_COMPONENTS[role]]
        keys = [key for key, _ in ROLE_COMPONENTS[role]]
        counts = block_counts(role_rows, group_key="subtype", groups=keys)
        values = np.asarray([counts[key] for key in keys])
        vmax = max(1, int(values.max()))
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap="BuPu",
            vmin=0,
            vmax=vmax,
        )
        axis.set_title(
            f"{ROLE_PLOT_LABELS[role]} · n={len(role_rows)}",
            loc="left",
            fontweight="bold",
        )
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xticks(range(30), [f"{block:02d}" for block in range(30)])
        axis.set_xlabel("DiT block")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = int(values[row, column])
                if value:
                    axis.text(
                        column,
                        row,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color="white" if value >= max(2, 0.65 * vmax) else "#20192b",
                    )
    assert image is not None
    fig.suptitle(
        "Cross-model common heads: mutually exclusive feature-dominance subtypes",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def table_html(
    values: dict[str, list[int]],
    labels: dict[str, str],
) -> str:
    headers = "".join(f"<th>B{block:02d}</th>" for block in range(30))
    rows = []
    for key, counts in values.items():
        cells = "".join(
            f"<td class='n{min(int(value), 9)}'>{int(value) or ''}</td>"
            for value in counts
        )
        rows.append(
            f"<tr><th>{html.escape(labels.get(key, key))}</th>{cells}"
            f"<td class='total'>{sum(counts)}</td></tr>"
        )
    return (
        "<div class='table-scroll'><table class='block-table'><thead><tr>"
        f"<th>类别</th>{headers}<th>总数</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def main_table_section(
    model_counts: dict[str, dict[str, list[int]]],
) -> str:
    tables = []
    for index, model in enumerate(MODELS):
        tables.append(
            f"<div class='model-table' data-model='{model}'"
            f"{'' if index == 0 else ' hidden'}>"
            f"{table_html(model_counts[model], ROLE_LABELS)}</div>"
        )
    options = "".join(
        f"<option value='{model}'>{MODEL_LABELS[model]}</option>" for model in MODELS
    )
    return f"""
<section>
  <div class="section-head"><div><h2>精确 Block 计数：各模型主类别</h2>
  <p>每个模型独立分类720个head；每个单元格是该block中该类别的head数量。</p></div>
  <label class="select-label">模型<select id="model-select">{options}</select></label></div>
  {''.join(tables)}
</section>"""


def subtype_sections(common_subtypes: list[dict[str, Any]]) -> str:
    sections = []
    for role in COMMON_ROLES:
        rows = [row for row in common_subtypes if row["role"] == role]
        keys = [key for key, _ in ROLE_COMPONENTS[role]]
        labels = {key: label for key, label in ROLE_COMPONENTS[role]}
        counts = block_counts(rows, group_key="subtype", groups=keys)
        totals = ", ".join(f"{labels[key]}={sum(counts[key])}" for key in keys)
        sections.append(
            f"<article class='subtype-group'><h3>{ROLE_LABELS[role]}</h3>"
            f"<p>{html.escape(totals)}</p>{table_html(counts, labels)}</article>"
        )
    return "".join(sections)


def build_page(
    *,
    model_counts: dict[str, dict[str, list[int]]],
    common_counts: dict[str, list[int]],
    common_subtypes: list[dict[str, Any]],
    s_audit: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    role_totals = {
        role: int(sum(common_counts[role])) for role in COMMON_ROLES
    }
    stat_items = "".join(
        f"<div class='stat'><span>{ROLE_LABELS[role]}</span>"
        f"<strong>{role_totals[role]}</strong></div>"
        for role in COMMON_ROLES
    )
    common_table = table_html(common_counts, ROLE_LABELS)
    subtype_html = subtype_sections(common_subtypes)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Head 类别与子类别 Block 深度分布</title>
<style>
:root{{--bg:#f5f6f4;--ink:#202523;--muted:#626c67;--line:#c9cfcb;--accent:#176f62;--paper:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;letter-spacing:0}}
header,main{{max-width:1500px;margin:auto;padding:18px 26px}}header{{border-bottom:1px solid var(--line)}}
h1,h2,h3,p{{margin:0}}h1{{font-size:26px}}h2{{font-size:19px}}h3{{font-size:15px}}p{{color:var(--muted)}}
.topline{{display:flex;align-items:center;justify-content:space-between;gap:16px}}.back{{color:var(--accent);font-weight:700;text-decoration:none}}
.meta{{margin-top:5px}}section{{padding:24px 0;border-bottom:1px solid var(--line)}}.lead{{max-width:1100px;margin-top:11px}}
.stats{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));border:1px solid var(--line);margin-top:18px;background:var(--paper)}}
.stat{{padding:12px 15px;border-right:1px solid var(--line)}}.stat:last-child{{border-right:0}}.stat span{{display:block;color:var(--muted);font-size:12px}}.stat strong{{font-size:24px}}
.plots{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}}figure{{margin:0;background:var(--paper);border:1px solid var(--line);padding:10px}}
figure.wide{{grid-column:1/-1}}figcaption{{font-weight:700;margin:2px 3px 8px}}img{{display:block;width:100%;height:auto}}
.section-head{{display:flex;justify-content:space-between;align-items:end;gap:18px;margin-bottom:12px}}.select-label{{display:flex;align-items:center;gap:8px;color:var(--muted)}}
select{{padding:7px 28px 7px 9px;background:#fff;border:1px solid #9da8a2;color:var(--ink);font:inherit}}
.table-scroll{{overflow-x:auto;background:var(--paper);border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%}}
.block-table{{min-width:1260px;font-size:11px}}th,td{{border-right:1px solid #e1e5e2;border-bottom:1px solid #e1e5e2;text-align:center;padding:5px 4px;white-space:nowrap}}
.block-table th:first-child{{position:sticky;left:0;background:#f1f3f1;z-index:1;text-align:left;min-width:155px}}thead th{{background:#eef1ef}}td.total{{font-weight:800;background:#eef1ef}}
.n1{{background:#edf6f3}}.n2{{background:#d8ece6}}.n3{{background:#bee1d7}}.n4{{background:#9dd3c6}}.n5{{background:#79c2b1}}.n6{{background:#55aa99}}.n7{{background:#338c7e;color:white}}.n8,.n9{{background:#176f62;color:white}}
.subtype-group{{margin-top:16px}}.subtype-group p{{margin:3px 0 8px}}.note{{border-left:4px solid #e39a27;padding:8px 12px;margin-top:14px;background:#fff}}
.downloads{{display:flex;flex-wrap:wrap;gap:9px;margin-top:12px}}.downloads a{{color:var(--accent);font-weight:700}}
code{{font-size:12px;overflow-wrap:anywhere}}@media(max-width:900px){{header,main{{padding:15px}}.stats{{grid-template-columns:1fr 1fr}}.stat{{border-bottom:1px solid var(--line)}}.plots{{grid-template-columns:1fr}}figure.wide{{grid-column:auto}}.section-head{{align-items:start;flex-direction:column}}}}
</style></head><body>
<header><div class="topline"><h1>Head 类别与子类别 Block 深度分布</h1>
<a class="back" href="../index.html">返回可视化总入口</a></div>
<p class="meta">冻结口径：22 seeds × 20 cases × 3 models · 30 blocks × 24 heads · 更新 {updated}</p></header>
<main>
<section><h2>统计口径</h2><p class="lead">主类别来自现有聚合分类：S/T/P/C/G 按四个去噪步的角色分数、margin和跨样本support确定，M表示混合或不稳定。公共稳定head要求同一(block, head)在三个模型中属于同一非M类别。子类别不是新的官方角色标签，而是把每个公共稳定head按所属角色内部“跨样本、跨模型平均特征rank最高”的分数组成项做互斥归类，用来观察角色内部的深度结构。</p>
<div class="stats">{stat_items}</div>
<p class="note">S 子类别已与冻结分区逐head校验：{s_audit["status"]}，Local={s_audit["local_dominant"]}、Same-frame={s_audit["same_frame_dominant"]}。T/P/C/G 的子类别采用同样的唯一argmax规则，属于描述性分解，尚未替代主分类。</p></section>
<section><h2>主类别深度分布</h2>
<div class="plots"><figure class="wide"><figcaption>三个模型各自的主类别 × Block</figcaption><img src="per_model_role_block_heatmaps.png" alt="per model role block heatmaps"></figure>
<figure><figcaption>三模型公共稳定类别 × Block</figcaption><img src="common_role_block_distribution.png" alt="common role block distribution"></figure>
<figure><figcaption>Early / Middle / Late 比例</figcaption><img src="role_depth_band_composition.png" alt="role depth band composition"></figure></div></section>
{main_table_section(model_counts)}
<section><h2>精确 Block 计数：三模型公共稳定类别</h2>
<p>同一个公共head只属于一个主类别；总计{sum(role_totals.values())}个公共稳定head。</p>{common_table}</section>
<section><h2>公共稳定 Head 的互斥子类别</h2>
<p>每个角色内部按公式组成特征的平均rank最大项唯一归类；数值代表该block中的head数量。</p>
<figure class="wide" style="margin-top:14px"><figcaption>五类公共稳定head的子类别 × Block</figcaption><img src="common_subtype_block_heatmaps.png" alt="common subtype block heatmaps"></figure>
{subtype_html}</section>
<section><h2>数据文件</h2><p>页面统计可直接复算，未从图片反推数据。</p>
<div class="downloads"><a href="distribution_data.json">完整 JSON</a><a href="per_model_role_by_block.csv">模型主类别 CSV</a><a href="common_role_by_block.csv">公共类别 CSV</a><a href="feature_subtype_heads.csv">逐 Head 子类别 CSV</a><a href="common_subtype_by_block.csv">子类别 Block CSV</a></div>
<p style="margin-top:10px">主分类：<code>{html.escape(str(SNAPSHOT_ROOT / "aggregate_heads.csv"))}</code><br>
公共稳定head：从上述主分类逐(block, head)求三个模型相同的非M角色<br>
原始rank分区：<code>{html.escape(str(CLASSIFICATION_ROOT))}</code>（{len(manifest["partitions"])} partitions）</p></section>
</main>
<script>
const select=document.getElementById("model-select");
select.addEventListener("change",()=>document.querySelectorAll(".model-table").forEach(table=>table.hidden=table.dataset.model!==select.value));
</script></body></html>"""


def write_count_csv(
    path: Path,
    scoped_counts: dict[str, dict[str, list[int]]],
) -> None:
    rows = []
    for scope, groups in scoped_counts.items():
        for group, counts in groups.items():
            row = {"scope": scope, "group": group, "total": sum(counts)}
            row.update({f"block_{block:02d}": count for block, count in enumerate(counts)})
            rows.append(row)
    save_rows(path, rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (CLASSIFICATION_ROOT / "classification_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.get("status") != "complete":
        raise RuntimeError("Head-classification feature export is not complete")

    aggregate_rows = load_csv(SNAPSHOT_ROOT / "aggregate_heads.csv")
    common_rows = derive_common_rows(aggregate_rows)
    if len(aggregate_rows) != len(MODELS) * 30 * 24:
        raise RuntimeError(f"Unexpected aggregate head count: {len(aggregate_rows)}")
    common_keys = [
        (int(row["block"]), int(row["head"]), row["role"]) for row in common_rows
    ]
    if len(common_keys) != len(set(common_keys)):
        raise RuntimeError("Common head list contains duplicate role keys")

    model_counts = {
        model: block_counts(
            [row for row in aggregate_rows if row["model"] == model],
            group_key="role",
            groups=ROLES,
        )
        for model in MODELS
    }
    common_counts = block_counts(
        common_rows,
        group_key="role",
        groups=COMMON_ROLES,
    )
    feature_means = feature_rank_means(manifest)
    model_subtypes = assign_subtypes(
        aggregate_rows,
        feature_means,
        common=False,
    )
    common_subtypes = assign_subtypes(
        common_rows,
        feature_means,
        common=True,
    )
    s_audit = audit_s_partition(common_subtypes)

    model_depth = {
        model: depth_counts(model_counts[model]) for model in MODELS
    }
    common_depth = depth_counts(common_counts)
    subtype_counts = {
        role: block_counts(
            [row for row in common_subtypes if row["role"] == role],
            group_key="subtype",
            groups=[key for key, _ in ROLE_COMPONENTS[role]],
        )
        for role in COMMON_ROLES
    }

    role_heatmaps(model_counts, OUTPUT / "per_model_role_block_heatmaps.png")
    common_role_plot(common_counts, OUTPUT / "common_role_block_distribution.png")
    depth_band_plot(
        model_depth,
        common_depth,
        OUTPUT / "role_depth_band_composition.png",
    )
    subtype_heatmaps(
        common_subtypes,
        OUTPUT / "common_subtype_block_heatmaps.png",
    )

    write_count_csv(
        OUTPUT / "per_model_role_by_block.csv",
        model_counts,
    )
    write_count_csv(
        OUTPUT / "common_role_by_block.csv",
        {"common": common_counts},
    )
    write_count_csv(
        OUTPUT / "common_subtype_by_block.csv",
        {
            f"common_{role}": counts
            for role, counts in subtype_counts.items()
        },
    )
    save_rows(
        OUTPUT / "feature_subtype_heads.csv",
        [*model_subtypes, *common_subtypes],
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "models": list(MODELS),
            "seeds_per_model": 22,
            "cases": 20,
            "blocks": 30,
            "heads_per_block": 24,
            "rank_partitions": len(manifest["partitions"]),
        },
        "sources": {
            "aggregate_roles": str(SNAPSHOT_ROOT / "aggregate_heads.csv"),
            "common_heads": (
                "derived from aggregate_roles: same non-M role across all "
                "three models"
            ),
            "rank_manifest": str(
                CLASSIFICATION_ROOT / "classification_manifest.json"
            ),
            "frozen_s_partition": str(S_SPLIT),
        },
        "subtype_policy": {
            "definition": (
                "Within a head's assigned role, choose the score-component "
                "feature with the largest mean rank."
            ),
            "aggregation": (
                "Mean over 22 seeds, 20 cases and four denoise steps; T/P "
                "features use trajectory-valid samples only; common subtypes "
                "then average the three model means."
            ),
            "overlap": "none within each role",
            "tie_policy": "stable component order (exact ties were not observed)",
            "role_components": {
                role: [key for key, _ in components]
                for role, components in ROLE_COMPONENTS.items()
            },
        },
        "s_partition_audit": s_audit,
        "per_model_role_by_block": model_counts,
        "per_model_role_depth_bands": model_depth,
        "common_role_by_block": common_counts,
        "common_role_depth_bands": common_depth,
        "common_subtype_by_block": subtype_counts,
        "common_subtype_heads": common_subtypes,
    }
    (OUTPUT / "distribution_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "index.html").write_text(
        build_page(
            model_counts=model_counts,
            common_counts=common_counts,
            common_subtypes=common_subtypes,
            s_audit=s_audit,
            manifest=manifest,
        ),
        encoding="utf-8",
    )
    print(f"[head-depth] output={OUTPUT}")


if __name__ == "__main__":
    main()
