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
    lines = [
        "# Stage 4 控制变量结论",
        "",
        "## 怎么读",
        "",
        "每一行只改变一个变量，另外两个变量固定：",
        "",
        "- Head实验：固定时间方向和M，只比较 Top100 与 Bottom100。",
        "- 时间实验：固定Head和M，只比较 Future、Past、Same。",
        "- 信息流实验：固定Head和时间方向，只比较 M1、M2、M3。",
        "",
        f"当前有 **{len(coverage['cases'])}个独立case、{len(coverage['case_seeds'])}个case-seed、{coverage['records']}个视频**。",
        "`一致变化`仅表示当前3个case方向一致，是pilot证据，不是统计显著；`证据不足`表示case方向混合或可用case不足。",
        "",
        "## 指标对应什么影响",
        "",
        "| 影响类型 | 指标 | 数值变大表示什么 |",
        "|---|---|---|",
        "| 干预强度 | 删除AV总量 | 实际删除的ΣA·V contribution更多；不是输出质量 |",
        "| 目标轨迹 | ADE、FDE、速度、PCK失败 | 目标对象相对Baseline的轨迹/速度变化更大 |",
        "| 轨迹破坏 | 跟踪丢失、轨迹门控失败 | 对象无法被可靠追踪的比例更高 |",
        "| 身份/形状 | 身份失败、面积异常 | 对象更不像Baseline，或面积明显异常 |",
        "| 对象存活 | 空mask、对象消失、末段消失 | 对象不存在或在视频后段消失得更多 |",
        "| 背景代理 | 对象外像素 | 对象之外区域的像素变化更大；不等同背景运动 |",
        "| 跨对象轨迹 | 其他对象ADE | 消融目标A后，其他对象的轨迹变化更大 |",
        "",
        "## A. 只改变Head：Top100 vs Bottom100",
        "",
        "控制：时间方向、M和case/seed/object相同。",
        "",
        "| 固定时间 | 固定M | 删除量 Top vs Bottom | 当前一致变化的输出指标 | 证据不足的指标 |",
        "|---|---|---|---|---|",
    ]
    for row in grouped_control_rows(report["head"], ("direction", "flow")):
        lines.append(
            f"| {str(row['direction']).title()} | {row['flow']} | {row['dose']} {row['dose_evidence']} | "
            f"{'；'.join(row['consistent']) or '无'} | {'、'.join(row['insufficient']) or '无'} |"
        )

    lines.extend(
        [
            "",
            "### Head轴结论",
            "",
            "1. 固定M1和任一时间方向时，Top100删除量都显著高于Bottom100；轨迹、身份和存活指标中，M1-Future最一致。",
            "2. 固定M2/M3时，Top100删除量反而低于Bottom100；因此不能笼统说Top100比Bottom100重要。",
            "3. 当前最准确的解释是：Top100偏向R→R通信，Bottom100偏向C→R和R→C通信。",
            "",
            "## B. 只改变时间方向：Future vs Past vs Same",
            "",
            "控制：Head、M和case/seed/object相同。下表列出三组完整两两比较。",
            "",
            "| 固定Head | 固定M | 唯一变化 | 删除量 Left vs Right | 当前一致变化的输出指标 | 证据不足的指标 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in grouped_control_rows(
        report["temporal"], ("scope", "flow", "contrast_label")
    ):
        lines.append(
            f"| {base.SCOPE_LABELS[row['scope']]} | {row['flow']} | {row['contrast_label']} | "
            f"{row['dose']} {row['dose_evidence']} | {'；'.join(row['consistent']) or '无'} | "
            f"{'、'.join(row['insufficient']) or '无'} |"
        )

    lines.extend(
        [
            "",
            "### 时间轴结论",
            "",
            "1. Future/Past/Same没有跨Head和M都成立的固定强弱顺序。",
            "2. Top100-M2和Top100-M3中，Future相对Past一致增加身份失败和对象消失；但轨迹/速度或其他对象ADE没有一致变化。",
            "3. Bottom100-M1中方向相反：Future相对Past的一致变化是身份失败和消失减少。说明时间效应依赖Head group。",
            "4. 三个方向删除量不等；没有dose matching之前，不能把差异完全归因于时间方向。",
            "",
            "## C. 只改变信息流：M1 vs M2 vs M3",
            "",
            "控制：Head、时间方向和case/seed/object相同。下表列出三组完整两两比较。",
            "",
            "| 固定Head | 固定时间 | 唯一变化 | 删除量 Left vs Right | 当前一致变化的输出指标 | 证据不足的指标 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in grouped_control_rows(
        report["flow"], ("scope", "direction", "contrast_label")
    ):
        lines.append(
            f"| {base.SCOPE_LABELS[row['scope']]} | {str(row['direction']).title()} | "
            f"{row['contrast_label']} | {row['dose']} {row['dose_evidence']} | "
            f"{'；'.join(row['consistent']) or '无'} | {'、'.join(row['insufficient']) or '无'} |"
        )

    lines.extend(
        [
            "",
            "### 信息流轴结论",
            "",
            "1. Top100-Same中，M1相对M2/M3一致增加ROI变化、跟踪丢失、身份失败、对象消失和对象外像素变化；说明M1与对象自身稳定性关系最明确。",
            "2. Top100-Past中，M1相对M2/M3也一致增加身份失败和对象消失；相对M3还增加ADE。",
            "3. Top100-Future中，M1虽然均值最大，但M1与M2/M3的多数输出指标在case间方向不一致，所以不能仅凭均值下结论。",
            "4. Bottom100-Future中，M2相对M1一致增加轨迹、速度、身份、消失和对象外像素变化；Bottom100-Past中M3相对M2更影响身份/存活。",
            "5. M1/M2/M3的query集合与删除量不同，flow比较仍需结合dose解释。",
            "",
            "## 最简结论",
            "",
            "| 问题 | 控制变量后的回答 |",
            "|---|---|",
            "| Top100是否总比Bottom100重要？ | 否。Top100主要在M1更强；Bottom100在M2/M3删除量更大。 |",
            "| 哪个组合最稳定影响对象？ | Top100-M1-Future：轨迹、跟踪、身份、消失均相对Bottom100更强。 |",
            "| Future是否总比Past重要？ | 否。结果依赖Head和M；Top-M2/M3的身份/消失偏Future，Bottom-M1反而偏Past。 |",
            "| M1主要影响什么？ | 当前最一致的是对象自身轨迹连续性、身份和存活；纯外观仍缺complete25证据。 |",
            "| M2主要影响什么？ | Bottom100-Future/Same出现较强目标对象变化，但GT物理交互指标不足，暂不能定义为物理约束输入。 |",
            "| M3主要影响什么？ | 会影响部分身份/存活指标，但Other-object ADE不稳定，尚不能证明稳定跨对象传播。 |",
            "| 哪些结论证据不足？ | Future/Past/Same统一排序、纯外观、背景运动、跨对象传播、物理正确性。 |",
            "",
            "完整原始数值和cases/paired units见 `STAGE4_THREE_AXIS_FULL_TABLES.md`。",
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
