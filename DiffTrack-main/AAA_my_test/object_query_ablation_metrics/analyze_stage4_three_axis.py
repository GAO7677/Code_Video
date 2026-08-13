#!/usr/bin/env python3
"""Build the exhaustive Head × Time × Flow comparison tables for Stage 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import analyze_stage4_existing_cases as base


METRICS = (
    ("removed_value_norm_query_sum", "删除 AV 总量", "dose"),
    ("target_local", "Target ROI MAE×100", "object-composite"),
    ("center_ade", "Center-ADE/D0", "trajectory"),
    ("center_fde", "Center-FDE/D0", "trajectory"),
    ("velocity", "Velocity Error", "motion"),
    ("pck10_failure", "1−PCK@10%", "trajectory"),
    ("track_loss", "Track Loss", "trajectory-guardrail"),
    ("gate_failure", "Trajectory Gate Failure", "trajectory-guardrail"),
    ("identity_failure", "Identity Failure %", "identity"),
    ("area_failure", "Area Failure %", "shape"),
    ("mask_absence", "Mask Absence %", "survival"),
    ("disappearance", "Disappearance %", "survival"),
    ("terminal_missing", "Terminal Missing %", "survival"),
    ("outside_static", "Outside MAE×100", "background-proxy"),
    ("other_ade", "Other-object ADE/D0", "cross-object"),
)
TEMPORAL_CONTRASTS = (
    ("future", "past", "Future − Past"),
    ("same", "future", "Same − Future"),
    ("same", "past", "Same − Past"),
)
FLOW_CONTRASTS = (
    ("M1", "M2", "M1 − M2"),
    ("M1", "M3", "M1 − M3"),
    ("M2", "M3", "M2 − M3"),
)
MAIN_SCOPES = ("top100", "bottom100")
SHORT_LABELS = {
    "target_local": "ROI像素",
    "center_ade": "ADE",
    "center_fde": "FDE",
    "velocity": "速度",
    "pck10_failure": "PCK失败",
    "track_loss": "跟踪丢失",
    "gate_failure": "轨迹门控失败",
    "identity_failure": "身份失败",
    "area_failure": "面积异常",
    "mask_absence": "空mask",
    "disappearance": "对象消失",
    "terminal_missing": "末段消失",
    "outside_static": "对象外像素",
    "other_ade": "其他对象ADE",
}


def evidence(row: dict[str, Any]) -> str:
    case_rows = row["case_rows"]
    count = len(case_rows)
    if count < 3:
        return f"🔴 仅 {count} case"
    differences = [float(item["difference"]) for item in case_rows]
    if all(value > 0 for value in differences):
        return f"🟡 {count}/{count} 同向↑"
    if all(value < 0 for value in differences):
        return f"🟡 {count}/{count} 同向↓"
    if all(value >= 0 for value in differences) and any(value > 0 for value in differences):
        return f"🟡 {count}/{count} 非负↑（含零）"
    if all(value <= 0 for value in differences) and any(value < 0 for value in differences):
        return f"🟡 {count}/{count} 非正↓（含零）"
    if all(value == 0 for value in differences):
        return f"🔴 {count}/{count} 均为零"
    return f"🔴 方向混合（{sum(value > 0 for value in differences)}/{count}↑）"


def fmt(value: Any, digits: int = 3) -> str:
    return base.fmt(value, digits)


def effect(row: dict[str, Any]) -> str:
    ratio = row["ratio"]
    ratio_text = f"{fmt(ratio)}×" if base.finite(ratio) else "N/A"
    return f"Δ {fmt(row['difference'])}; {ratio_text}"


def cell_summary(
    records: list[dict[str, Any]], metric: str, filters: dict[str, str]
) -> dict[str, Any]:
    rows = [
        record
        for record in records
        if all(record.get(key) == value for key, value in filters.items())
        and base.finite(record["metrics"].get(metric))
    ]
    by_case: dict[str, list[float]] = {}
    for record in rows:
        by_case.setdefault(record["case"], []).append(float(record["metrics"][metric]))
    case_values = {case: float(np.mean(values)) for case, values in by_case.items()}
    return {
        "mean": base.finite_mean(case_values.values()),
        "case_count": len(case_values),
        "unit_count": len(rows),
    }


def gate_summary(
    records: list[dict[str, Any]], filters: dict[str, str]
) -> dict[str, Any]:
    rows = [record for record in records if all(record.get(k) == v for k, v in filters.items())]
    values = [record["metrics"].get("quality_pass") is True for record in rows]
    failed = sum(not value for value in values)
    return {
        "failed": failed,
        "total": len(values),
        "failure_rate": failed / len(values) if values else None,
    }


def compare(
    records: list[dict[str, Any]],
    metric: str,
    varying: str,
    left: str,
    right: str,
    filters: dict[str, str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    row = base.paired_contrast(
        records,
        metric,
        varying,
        left,
        right,
        filters,
        20_000,
        rng,
    )
    row["evidence"] = evidence(row)
    return row


def build_report(records: list[dict[str, Any]], coverage: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(20260813)
    for record in records:
        quality_pass = record["metrics"].get("quality_pass")
        record["metrics"]["gate_failure"] = (
            0.0 if quality_pass is True else 100.0 if quality_pass is False else None
        )
    cube = []
    for scope in MAIN_SCOPES:
        for direction in base.DIRECTIONS:
            for flow in base.FLOWS:
                filters = {"scope": scope, "direction": direction, "flow": flow}
                cube.append(
                    {
                        "scope": scope,
                        "direction": direction,
                        "flow": flow,
                        "metrics": {
                            metric: cell_summary(records, metric, filters)
                            for metric, _, _ in METRICS
                        },
                        "gate": gate_summary(records, filters),
                    }
                )

    head = []
    for metric, label, category in METRICS:
        for direction in base.DIRECTIONS:
            for flow in base.FLOWS:
                head.append(
                    {
                        "metric": metric,
                        "metric_label": label,
                        "category": category,
                        "direction": direction,
                        "flow": flow,
                        "contrast": compare(
                            records,
                            metric,
                            "scope",
                            "top100",
                            "bottom100",
                            {"direction": direction, "flow": flow},
                            rng,
                        ),
                    }
                )

    temporal = []
    for metric, label, category in METRICS:
        for scope in MAIN_SCOPES:
            for flow in base.FLOWS:
                for left, right, contrast_label in TEMPORAL_CONTRASTS:
                    temporal.append(
                        {
                            "metric": metric,
                            "metric_label": label,
                            "category": category,
                            "scope": scope,
                            "flow": flow,
                            "contrast_label": contrast_label,
                            "contrast": compare(
                                records,
                                metric,
                                "direction",
                                left,
                                right,
                                {"scope": scope, "flow": flow},
                                rng,
                            ),
                        }
                    )

    flow = []
    for metric, label, category in METRICS:
        for scope in MAIN_SCOPES:
            for direction in base.DIRECTIONS:
                for left, right, contrast_label in FLOW_CONTRASTS:
                    flow.append(
                        {
                            "metric": metric,
                            "metric_label": label,
                            "category": category,
                            "scope": scope,
                            "direction": direction,
                            "contrast_label": contrast_label,
                            "contrast": compare(
                                records,
                                metric,
                                "flow",
                                left,
                                right,
                                {"scope": scope, "direction": direction},
                                rng,
                            ),
                        }
                    )
    return {"coverage": coverage, "cube": cube, "head": head, "temporal": temporal, "flow": flow}


def concise_metric(metric: str, row: dict[str, Any]) -> str:
    left = float(row["left_mean"])
    right = float(row["right_mean"])
    difference = float(row["difference"])
    symbol = ">" if difference > 0 else "<" if difference < 0 else "="
    if metric in {
        "track_loss",
        "gate_failure",
        "identity_failure",
        "area_failure",
        "mask_absence",
        "disappearance",
        "terminal_missing",
        "pck10_failure",
    }:
        suffix = f"Δ{difference:+.2f}pp"
    elif metric in {"center_ade", "center_fde", "other_ade"}:
        suffix = f"Δ{difference:+.3f}D0"
    elif metric == "velocity":
        suffix = f"Δ{difference:+.3f}D0/frame"
    else:
        suffix = f"Δ{difference:+.3f}"
    return f"{SHORT_LABELS[metric]} {left:.3f}{symbol}{right:.3f}（{suffix}）"


def grouped_control_rows(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        by_metric = {row["metric"]: row["contrast"] for row in group}
        dose = by_metric["removed_value_norm_query_sum"]
        consistent = []
        insufficient = []
        for metric, _, category in METRICS:
            if category == "dose":
                continue
            current = by_metric[metric]
            if current["evidence"].startswith("🟡"):
                consistent.append(concise_metric(metric, current))
            else:
                insufficient.append(SHORT_LABELS[metric])
        output.append(
            {
                **dict(zip(fields, key)),
                "left": dose["left"],
                "right": dose["right"],
                "dose": (
                    f"{float(dose['left_mean']):.2f}"
                    f"{'>' if dose['difference'] > 0 else '<' if dose['difference'] < 0 else '='}"
                    f"{float(dose['right_mean']):.2f}（{float(dose['ratio']):.2f}×）"
                    if base.finite(dose["left_mean"])
                    and base.finite(dose["right_mean"])
                    and base.finite(dose["ratio"])
                    else "N/A"
                ),
                "dose_evidence": dose["evidence"],
                "consistent": consistent,
                "insufficient": insufficient,
                "case_count": dose["case_count"],
            }
        )
    return output


def controlled_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    planned = 999
    quality_total = sum(int(row["total"]) for row in coverage["quality_gate"])
    quality_failed = sum(int(row["failed"]) for row in coverage["quality_gate"])
    quality_passed = quality_total - quality_failed

    def pick(axis: str, metric: str, **filters: str) -> dict[str, Any]:
        matches = [
            row["contrast"]
            for row in report[axis]
            if row["metric"] == metric
            and all(row.get(key) == value for key, value in filters.items())
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one Stage4 contrast for {axis}/{metric}/{filters}, got {len(matches)}"
            )
        return matches[0]

    def value(metric: str, axis: str, **filters: str) -> str:
        row = pick(axis, metric, **filters)
        ci = row["bootstrap_95_ci"]
        if metric in {
            "track_loss",
            "gate_failure",
            "identity_failure",
            "area_failure",
            "mask_absence",
            "disappearance",
            "terminal_missing",
            "pck10_failure",
        }:
            unit = " pp"
            digits = 2
        elif metric in {"center_ade", "center_fde", "other_ade"}:
            unit = " D0"
            digits = 3
        elif metric == "velocity":
            unit = " D0/frame"
            digits = 3
        else:
            unit = ""
            digits = 3
        left = fmt(row["left_mean"], digits)
        right = fmt(row["right_mean"], digits)
        difference = fmt(row["difference"], digits)
        low = fmt(ci[0], digits)
        high = fmt(ci[1], digits)
        return (
            f"{left} vs {right}; Δ={difference}{unit}; CI [{low}, {high}]; "
            f"{row['evidence']}; n={row['case_count']} cases/{row['unit_count']} pairs"
        )

    dose_filters = {"contrast_label": "Future − Past"}
    lines = [
        "# Stage 4 结果与控制变量证据审计",
        "",
        "## 1. 当前覆盖与统计单位",
        "",
        f"- 计划新生成：**{planned}** 个 variants；当前纳入：**{coverage['records']} / {planned} ({100.0 * coverage['records'] / planned:.2f}%)**；尚缺 **{planned - coverage['records']}**。",
        f"- 独立统计单位：**{len(coverage['cases'])} 个 case**。共有 **{len(coverage['case_seeds'])} 个嵌套 case-seed**；{coverage['records']} 个视频不是 {coverage['records']} 个独立样本。",
        f"- 轨迹质量门控：通过 **{quality_passed}/{quality_total}**，失败 **{quality_failed}/{quality_total} ({100.0 * quality_failed / quality_total:.2f}%)**。ADE/FDE/速度只在通过者上计算；Track Loss 与 Disappearance 保留失败结果。",
        f"- Head coverage：Top100={coverage['scope_counts'].get('top100', 0)}，Bottom100={coverage['scope_counts'].get('bottom100', 0)}，Random100={coverage['scope_counts'].get('random100_layer_matched_draw0', 0)}，All720=0。",
        "- 聚合顺序：先在同一 case 内平均 object 和 seed，再对 case 等权；每个对比只改变 Head、时间方向、信息流三轴中的一个。",
        "",
        "| Case | Seed | 已纳入视频 |",
        "|---|---:|---:|",
    ]
    for row in coverage["records_by_case_seed"]:
        lines.append(f"| `{row['case']}` | {row['seed']} | {row['records']} |")

    lines.extend(
        [
            "",
            "> 证据等级：`🟡 3/3 同向`只表示三个 pilot case 方向一致，不等于统计显著。n=3 时双侧 exact sign-flip 的最小 p=0.25；当前 CI 是描述性 case-bootstrap CI，不进行 BH-FDR 或总体机制宣称。",
            "",
            "## 2. 冻结主问题 T1–T3",
            "",
            "表中均为 Left vs Right；正 Δ 表示 Left 相对同 seed Baseline 的干预效应更大。",
            "",
            "| 主问题 | 指标 | 严格配对结果 | 判定 |",
            "|---|---|---|---|",
            f"| T1 · Top100-M1 Future vs Past | Center-ADE | {value('center_ade', 'temporal', scope='top100', flow='M1', **dose_filters)} | **不支持 Future 更强**：均值反而更小，且 case 方向混合。 |",
            f"| T1 · Top100-M1 Future vs Past | Track Loss | {value('track_loss', 'temporal', scope='top100', flow='M1', **dose_filters)} | 方向混合，不能形成轨迹破坏结论。 |",
            f"| T1 · Top100-M1 Future vs Past | Disappearance | {value('disappearance', 'temporal', scope='top100', flow='M1', **dose_filters)} | 方向混合，不能形成对象存活结论。 |",
            f"| T1 · Top100-M1 Future vs Past | 删除 AV 总量 | {value('removed_value_norm_query_sum', 'temporal', scope='top100', flow='M1', **dose_filters)} | Future 删除量在 3/3 cases 更大，但没有转化成稳定 outcome 差异。 |",
            f"| T2 · Top100-M2 Future vs Past | Velocity Error | {value('velocity', 'temporal', scope='top100', flow='M2', **dose_filters)} | 方向混合；当前不能支持更强物理/轨迹效应。 |",
            f"| T2 · Top100-M2 Future vs Past | Identity Failure | {value('identity_failure', 'temporal', scope='top100', flow='M2', **dose_filters)} | 三个 case 非负，属于**身份/存活 pilot 信号**；不是 GT 物理证据。 |",
            f"| T2 · Top100-M2 Future vs Past | Disappearance | {value('disappearance', 'temporal', scope='top100', flow='M2', **dose_filters)} | 三个 case 非负，属于**身份/存活 pilot 信号**。 |",
            f"| T2 · Top100-M2 Future vs Past | 删除 AV 总量 | {value('removed_value_norm_query_sum', 'temporal', scope='top100', flow='M2', **dose_filters)} | Future 删除量更大，时间方向与 dose 仍混杂。 |",
            f"| T3 · Top100-M3 Future vs Past | Other-object ADE | {value('other_ade', 'temporal', scope='top100', flow='M3', **dose_filters)} | 方向混合，**不支持稳定跨对象传播**。 |",
            f"| T3 · Top100-M3 Future vs Past | Outside MAE | {value('outside_static', 'temporal', scope='top100', flow='M3', **dose_filters)} | 方向混合；且像素变化不能等同背景运动。 |",
            f"| T3 · Top100-M3 Future vs Past | 删除 AV 总量 | {value('removed_value_norm_query_sum', 'temporal', scope='top100', flow='M3', **dose_filters)} | Future 删除量更大，但 cross-object outcome 没有稳定同向。 |",
            "",
            "## 3. 三轴控制变量结论",
            "",
            "### 3.1 只改变 Head group",
            "",
            "| 固定条件 | Dose | 轨迹/跟踪 | 身份/存活 | 结论 |",
            "|---|---|---|---|---|",
            f"| M1-Future，Top100 vs Bottom100 | {value('removed_value_norm_query_sum', 'head', direction='future', flow='M1')} | ADE: {value('center_ade', 'head', direction='future', flow='M1')}<br>Velocity: {value('velocity', 'head', direction='future', flow='M1')}<br>Track Loss: {value('track_loss', 'head', direction='future', flow='M1')} | Identity: {value('identity_failure', 'head', direction='future', flow='M1')}<br>Disappearance: {value('disappearance', 'head', direction='future', flow='M1')} | 当前最稳定的 Top100>Bottom100 组合；但 Top100 dose 约为 Bottom100 的 7.32×，不能归因为单位信息更关键。 |",
            f"| M2-Same，Top100 vs Bottom100 | {value('removed_value_norm_query_sum', 'head', direction='same', flow='M2')} | Track Loss: {value('track_loss', 'head', direction='same', flow='M2')} | Identity: {value('identity_failure', 'head', direction='same', flow='M2')}<br>Disappearance: {value('disappearance', 'head', direction='same', flow='M2')} | Bottom100 删除量和破坏都更大，反证“Top100 总是更重要”。 |",
            "",
            "**Head 轴结论：** latest3350 Top100 对 M1/R→R 更富集；Bottom100 对部分 M2/C→R 更富集。现有结果支持的是**信息流选择性**，不是 Top100 的全局优越性。两组比较均存在强 dose 差异，尚无每单位贡献的因果结论。",
            "",
            "### 3.2 只改变时间方向",
            "",
            "| 固定条件 | Dose · Future vs Past | 轨迹/跟踪 | 身份/存活 | 结论 |",
            "|---|---|---|---|---|",
            f"| Top100-M1 | {value('removed_value_norm_query_sum', 'temporal', scope='top100', flow='M1', **dose_filters)} | ADE: {value('center_ade', 'temporal', scope='top100', flow='M1', **dose_filters)}<br>Track: {value('track_loss', 'temporal', scope='top100', flow='M1', **dose_filters)} | Disappearance: {value('disappearance', 'temporal', scope='top100', flow='M1', **dose_filters)} | outcome 方向混合，是“Future 总更重要”的直接反例。 |",
            f"| Top100-M2 | {value('removed_value_norm_query_sum', 'temporal', scope='top100', flow='M2', **dose_filters)} | Velocity: {value('velocity', 'temporal', scope='top100', flow='M2', **dose_filters)} | Identity: {value('identity_failure', 'temporal', scope='top100', flow='M2', **dose_filters)}<br>Disappearance: {value('disappearance', 'temporal', scope='top100', flow='M2', **dose_filters)} | 只有身份/存活出现初步 Future>Past；轨迹证据不足。 |",
            f"| Bottom100-M1 | {value('removed_value_norm_query_sum', 'temporal', scope='bottom100', flow='M1', **dose_filters)} | Velocity: {value('velocity', 'temporal', scope='bottom100', flow='M1', **dose_filters)}<br>Track: {value('track_loss', 'temporal', scope='bottom100', flow='M1', **dose_filters)} | Identity: {value('identity_failure', 'temporal', scope='bottom100', flow='M1', **dose_filters)}<br>Disappearance: {value('disappearance', 'temporal', scope='bottom100', flow='M1', **dose_filters)} | 多项反而 Future<Past，说明时间效应依赖 Head×Flow。 |",
            "",
            "**时间轴结论：** 没有跨 Head group 和 M1/M2/M3 都成立的 Future/Past/Same 排序。当前只能报告局部交互；不能把 `Future` 普遍解释为状态传播主通道。",
            "",
            "### 3.3 只改变信息流 M1/M2/M3",
            "",
            "| 固定条件 | Dose | 轨迹/跟踪 | 身份/存活 | 结论 |",
            "|---|---|---|---|---|",
            f"| Top100-Same，M1 vs M2 | {value('removed_value_norm_query_sum', 'flow', scope='top100', direction='same', contrast_label='M1 − M2')} | Track: {value('track_loss', 'flow', scope='top100', direction='same', contrast_label='M1 − M2')} | Identity: {value('identity_failure', 'flow', scope='top100', direction='same', contrast_label='M1 − M2')}<br>Disappearance: {value('disappearance', 'flow', scope='top100', direction='same', contrast_label='M1 − M2')} | M1 对对象身份/存活的关联最稳定；但 M1 dose 约为 M2 的 5.46×。 |",
            f"| Top100-Same，M1 vs M3 | {value('removed_value_norm_query_sum', 'flow', scope='top100', direction='same', contrast_label='M1 − M3')} | Track: {value('track_loss', 'flow', scope='top100', direction='same', contrast_label='M1 − M3')} | Identity: {value('identity_failure', 'flow', scope='top100', direction='same', contrast_label='M1 − M3')}<br>Disappearance: {value('disappearance', 'flow', scope='top100', direction='same', contrast_label='M1 − M3')} | 同样支持 M1 与对象自身稳定性的关系；仍受 dose 混杂。 |",
            f"| Bottom100-Future，M1 vs M2 | {value('removed_value_norm_query_sum', 'flow', scope='bottom100', direction='future', contrast_label='M1 − M2')} | ADE: {value('center_ade', 'flow', scope='bottom100', direction='future', contrast_label='M1 − M2')}<br>Velocity: {value('velocity', 'flow', scope='bottom100', direction='future', contrast_label='M1 − M2')}<br>Track: {value('track_loss', 'flow', scope='bottom100', direction='future', contrast_label='M1 − M2')} | Identity: {value('identity_failure', 'flow', scope='bottom100', direction='future', contrast_label='M1 − M2')}<br>Disappearance: {value('disappearance', 'flow', scope='bottom100', direction='future', contrast_label='M1 − M2')} | 负 Δ 表示 M2 更强：Bottom100-Future 的 C→R 对目标轨迹/存活有稳定 pilot 影响；M2 dose 约为 M1 的 5.72×。 |",
            "",
            "**Flow 轴结论：** M1/R→R 与对象自身身份、存活和部分轨迹连续性的关系最清楚；M2/C→R 在 Bottom100-Future 条件下也强烈影响目标轨迹与存活。M3/R→C 的 Other-object ADE 未形成稳定方向，因此当前不能宣称已经证明对象状态向其他对象传播。",
            "",
            "## 4. 能下与不能下的结论",
            "",
            "| 结论 | 当前证据 | 证据状态 |",
            "|---|---|---|",
            "| latest3350 Top100 在 M1/R→R 上比 Bottom100 删除更多 contribution，并造成更强对象破坏 | M1-Future 的 dose、ADE、velocity、track、identity、disappearance 均为 3/3 同向 | 🟡 pilot 支持；强 dose 混杂 |",
            "| Top100 总比 Bottom100 重要 | M2-Same 中 Bottom100 的 dose、track、identity、disappearance 更高 | 🔴 被当前结果反证 |",
            "| Future 普遍比 Past/Same 重要 | Top100-M1 方向混合；Bottom100-M1 多项 Future<Past | 🔴 不支持 |",
            "| M1 专门编码对象身份/轨迹 | Top100-Same 下 M1 对身份/存活较强，但其 dose 同时更大 | 🟡 有必要性关联；🔴 尚无语义专属性 |",
            "| M2 是物理交互输入 | Bottom100-Future 改变目标轨迹/存活，但当前没有合格的完整 GT contact/post-contact 主检验 | 🔴 证据不足 |",
            "| M3 稳定把对象状态广播给其他对象 | T3 Other-object ADE 与 Outside MAE 均为 case 方向混合 | 🔴 证据不足 |",
            "| 纯外观变化已有跨 case 结论 | complete25 目前只覆盖同一独立 case 的两个 seed | 🔴 证据不足 |",
            "| 当前结果具有总体统计显著性 | 只有 3 个独立 case，且 999 矩阵尚未完成 | 🔴 不允许宣称 |",
            "",
            "## 5. 后续 Gate",
            "",
            "1. 先补齐剩余 315 个 variants，特别是 `000331`、Random100、All-time 和 All720 sentinel，重新冻结 Stage 4A 报告。",
            "2. 用 Stage 4A 的 case-level 方差和预先给定 MDE 做 power 分析；在选择 Stage 4B case 数前人工确认。",
            "3. Stage 4B 使用未参与当前页面挑选的 held-out cases；确认性统计以 case 为独立单位。",
            "4. 若要回答纯外观、背景运动或物理正确性，必须分别补足跨 case complete25、背景运动指标和 simulator GT 交互指标，不能用 ROI/Outside MAE 替代。",
            "",
            "## 6. 可追溯入口",
            "",
            "- 全部三轴数值：`STAGE4_THREE_AXIS_FULL_TABLES.md`",
            "- 原始统计 JSON：`three_axis_report.json`",
            "- 预注册主问题报告：`STAGE4_EXISTING_CASES_ANALYSIS.md` 与 `report.json`",
            "- 代表性视频：`http://localhost:8092/object-query-information-flow-stage4-representatives?v=2`",
        ]
    )
    return "\n".join(lines)


def metric_sections(
    rows: list[dict[str, Any]], kind: str
) -> list[str]:
    lines: list[str] = []
    for metric, label, _ in METRICS:
        selected = [row for row in rows if row["metric"] == metric]
        lines.extend([f"### {label}", ""])
        if kind == "head":
            lines.extend(
                [
                    "| 时间 | Flow | Top100 | Bottom100 | Top−Bottom | Evidence | Cases/units |",
                    "|---|---|---:|---:|---|---|---:|",
                ]
            )
            for row in selected:
                value = row["contrast"]
                lines.append(
                    f"| {row['direction'].title()} | {row['flow']} | {fmt(value['left_mean'])} | "
                    f"{fmt(value['right_mean'])} | {effect(value)} | {value['evidence']} | "
                    f"{value['case_count']}/{value['unit_count']} |"
                )
        elif kind == "temporal":
            lines.extend(
                [
                    "| Head | Flow | 比较 | Left | Right | Effect | Evidence | Cases/units |",
                    "|---|---|---|---:|---:|---|---|---:|",
                ]
            )
            for row in selected:
                value = row["contrast"]
                lines.append(
                    f"| {base.SCOPE_LABELS[row['scope']]} | {row['flow']} | {row['contrast_label']} | "
                    f"{fmt(value['left_mean'])} | {fmt(value['right_mean'])} | {effect(value)} | "
                    f"{value['evidence']} | {value['case_count']}/{value['unit_count']} |"
                )
        else:
            lines.extend(
                [
                    "| Head | 时间 | 比较 | Left | Right | Effect | Evidence | Cases/units |",
                    "|---|---|---|---:|---:|---|---|---:|",
                ]
            )
            for row in selected:
                value = row["contrast"]
                lines.append(
                    f"| {base.SCOPE_LABELS[row['scope']]} | {row['direction'].title()} | "
                    f"{row['contrast_label']} | {fmt(value['left_mean'])} | {fmt(value['right_mean'])} | "
                    f"{effect(value)} | {value['evidence']} | {value['case_count']}/{value['unit_count']} |"
                )
        lines.append("")
    return lines


def markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# Stage 4 三轴全面比较",
        "",
        "## 1. 三个实验变量",
        "",
        "| 变量轴 | 水平 | 精确含义 |",
        "|---|---|---|",
        "| Head group | Top100 / Bottom100 | latest3350、step39、micro PCK@32 排名的前100/后100个 layer-head |",
        "| 时间方向 | Same / Future / Past | Same: `t_k=t_q`；Future: `t_k<t_q`，历史 source→未来 query；Past: `t_k>t_q`，未来 source→过去 query |",
        "| 信息流 | M1 / M2 / M3 | M1: R K/V→R Query；M2: C K/V→R Query；M3: R K/V→C Query |",
        "",
        "主实验矩阵共有 `2 × 3 × 3 = 18` 个条件。Random100 是补充控制，不进入本主矩阵。",
        "",
        "## 2. 统计和证据标记",
        "",
        f"- 当前覆盖 **{len(coverage['cases'])} cases、{len(coverage['case_seeds'])} case-seeds、{coverage['records']} variants**。",
        "- case 是最高独立单位；同一 case 内先平均 seed 和 object，再对 case 等权。",
        "- `🟡 n/n 同向`：所有当前 case 的差值同号，只能称为初步一致，不能称统计显著。",
        "- `🔴 方向混合/仅2 case`：证据不足，不形成方向结论。",
        "- Effect 中 `Δ=Left−Right`；倍数为 `Left/Right`。所有结果均相对同 seed Baseline，数值大表示干预效应大，不表示相对 GT 更差。",
        "",
        "## 3. 18 条件总览",
        "",
        "ADE/Velocity 只包含轨迹门控通过者；Track Loss/Identity/Disappearance 保留破坏性失败；Other-ADE 只对单对象消融有定义。",
        "",
        "| Head | 时间 | Flow | AV dose | ROI MAE | ADE | Velocity | Track Loss | Identity % | Disappear % | Outside MAE | Other ADE | Gate fail |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cube"]:
        metrics = row["metrics"]
        gate = row["gate"]
        lines.append(
            f"| {base.SCOPE_LABELS[row['scope']]} | {row['direction'].title()} | {row['flow']} | "
            + " | ".join(
                fmt(metrics[name]["mean"])
                for name in (
                    "removed_value_norm_query_sum",
                    "target_local",
                    "center_ade",
                    "velocity",
                    "track_loss",
                    "identity_failure",
                    "disappearance",
                    "outside_static",
                    "other_ade",
                )
            )
            + f" | {fmt(100.0 * gate['failure_rate'], 1)}% ({gate['failed']}/{gate['total']}) |"
        )

    lines.extend(
        [
            "",
            "## 4. 轴一：Top100 vs Bottom100",
            "",
            "固定时间方向和信息流，只改变 head group。以下均为严格配对有限 cohort。",
            "",
        ]
    )
    lines.extend(metric_sections(report["head"], "head"))
    lines.extend(
        [
            "## 5. 轴二：Future / Past / Same",
            "",
            "固定 head group 和信息流，只改变时间方向。每个指标列出全部三组两两比较。",
            "",
        ]
    )
    lines.extend(metric_sections(report["temporal"], "temporal"))
    lines.extend(
        [
            "## 6. 轴三：M1 / M2 / M3",
            "",
            "固定 head group 和时间方向，只改变被删除的信息流。M1/M2/M3 的 query 集合不同，必须和 AV dose 同时解释。",
            "",
        ]
    )
    lines.extend(metric_sections(report["flow"], "flow"))
    return "\n".join(lines)


def main() -> None:
    records, coverage = base.collect(base.DEFAULT_ROOT)
    report = build_report(records, coverage)
    output = base.DEFAULT_ROOT / "stage4_current_analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "three_axis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path = output / "STAGE4_THREE_AXIS_FULL_TABLES.md"
    path.write_text(markdown(report), encoding="utf-8")
    controlled_path = output / "STAGE4_CONTROLLED_VARIABLE_CONCLUSIONS.md"
    controlled_path.write_text(controlled_markdown(report), encoding="utf-8")
    print(path)
    print(controlled_path)


if __name__ == "__main__":
    main()
