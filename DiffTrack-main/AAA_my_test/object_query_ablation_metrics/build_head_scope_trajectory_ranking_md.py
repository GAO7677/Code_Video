#!/usr/bin/env python3
"""Build a complete, auditable Markdown ranking from a trajectory report."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Callable


MODE_LABELS = {
    "self_only": ("M1", "All-time", "delete A[R_tq,R_tk] for every tq,tk"),
    "self_same": ("M1", "Same", "delete A[R_tq,R_tk] where tk=tq"),
    "self_future": ("M1", "Future", "delete A[R_tq,R_tk] where tk<tq"),
    "self_past": ("M1", "Past", "delete A[R_tq,R_tk] where tk>tq"),
    "incoming_only": ("M2", "All-time", "delete A[R_tq,C_tk] for every tq,tk"),
    "incoming_same": ("M2", "Same", "delete A[R_tq,C_tk] where tk=tq"),
    "incoming_future": ("M2", "Future", "delete A[R_tq,C_tk] where tk<tq"),
    "incoming_past": ("M2", "Past", "delete A[R_tq,C_tk] where tk>tq"),
    "outgoing_only": ("M3", "All-time", "delete A[C_tq,R_tk] for every tq,tk"),
    "outgoing_same": ("M3", "Same", "delete A[C_tq,R_tk] where tk=tq"),
    "outgoing_future": ("M3", "Future", "delete A[C_tq,R_tk] where tk<tq"),
    "outgoing_past": ("M3", "Past", "delete A[C_tq,R_tk] where tk>tq"),
}

HEAD_LABELS = {
    "top100": "Top100 PCK Heads",
    "bottom100": "Bottom100 PCK Heads",
    "all720": "All720 Heads",
}

TARGET_LABELS = {
    ("single_object", "object_A"): "Object A",
    ("single_object", "object_B"): "Object B",
    ("all_objects", ""): "All objects",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="default: TRAJECTORY_METRICS_COMPLETE_RANKING.md beside report",
    )
    return parser.parse_args()


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def target_label(record: dict[str, Any]) -> str:
    key = (record["target_scope"], str(record.get("region") or ""))
    return TARGET_LABELS.get(key, "/".join(part for part in key if part))


def mode_label(record: dict[str, Any]) -> str:
    family, temporal, _ = MODE_LABELS.get(
        record["mask_mode"], (record["mask_mode"], "", "")
    )
    return f"{family}-{temporal}" if temporal != "All-time" else family


def selected_object_metrics(record: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = record["metrics"]
    return [metrics["objects"][name] for name in metrics["selected_objects"]]


def target_mean(record: dict[str, Any], getter: Callable[[dict[str, Any]], Any]) -> float | None:
    if not record["metrics"].get("quality_pass"):
        return None
    values = [getter(item) for item in selected_object_metrics(record)]
    if not values or not all(finite(value) for value in values):
        return None
    return fmean(float(value) for value in values)


def derived_metrics(record: dict[str, Any]) -> dict[str, float | None]:
    pck = lambda alpha: target_mean(  # noqa: E731
        record, lambda item: item.get("pck_normalized", {}).get(alpha)
    )
    return {
        "center_ade": target_mean(record, lambda item: item.get("center_ade_norm")),
        "center_fde": target_mean(record, lambda item: item.get("center_fde_norm")),
        "velocity": target_mean(
            record, lambda item: item.get("velocity_vector_error_norm_per_frame")
        ),
        "point_ade": target_mean(record, lambda item: item.get("point_ade_norm")),
        "pck05": pck("0.05"),
        "pck10": pck("0.1"),
        "pck20": pck("0.2"),
    }


METRICS = {
    "center_ade": {
        "label": "Center-ADE",
        "unit": "%D0",
        "stronger": "越大",
        "reverse": True,
        "digits": 3,
    },
    "center_fde": {
        "label": "Center-FDE",
        "unit": "%D0",
        "stronger": "越大",
        "reverse": True,
        "digits": 3,
    },
    "velocity": {
        "label": "Velocity vector error",
        "unit": "%D0/frame",
        "stronger": "越大",
        "reverse": True,
        "digits": 3,
    },
    "point_ade": {
        "label": "Point-ADE",
        "unit": "%D0",
        "stronger": "越大",
        "reverse": True,
        "digits": 3,
    },
    "pck05": {
        "label": "PCK@5%",
        "unit": "%",
        "stronger": "越小",
        "reverse": False,
        "digits": 2,
    },
    "pck10": {
        "label": "PCK@10%",
        "unit": "%",
        "stronger": "越小",
        "reverse": False,
        "digits": 2,
    },
    "pck20": {
        "label": "PCK@20%",
        "unit": "%",
        "stronger": "越小",
        "reverse": False,
        "digits": 2,
    },
}


def metric_display(key: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    spec = METRICS[key]
    scaled = 100.0 * value
    return f"{scaled:.{spec['digits']}f}"


def assign_ranks(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    ranks: dict[str, dict[str, int]] = {}
    for key, spec in METRICS.items():
        valid = [record for record in records if record["_derived"][key] is not None]
        valid.sort(
            key=lambda record: (
                -record["_derived"][key]
                if spec["reverse"]
                else record["_derived"][key],
                record["variant_id"],
            )
        )
        if key == "center_ade":
            ranks[key] = {
                record["variant_id"]: int(record["trajectory_rank_within_case_seed"])
                for record in valid
            }
            continue
        metric_ranks = {}
        previous_value = None
        previous_rank = None
        for position, record in enumerate(valid, start=1):
            value = record["_derived"][key]
            rank = previous_rank if value == previous_value else position
            metric_ranks[record["variant_id"]] = rank
            previous_value = value
            previous_rank = rank
        ranks[key] = metric_ranks
    return ranks


def experiment_cell(record: dict[str, Any]) -> str:
    return f"`{record['variant_id']}`"


def overlay_link(record: dict[str, Any], output: Path) -> str:
    overlay = Path(record["overlay_path"])
    relative = os.path.relpath(overlay, output.parent)
    return f"[overlay]({relative})"


def target_coverage(record: dict[str, Any]) -> str:
    parts = []
    metrics = record["metrics"]
    for name in metrics["selected_objects"]:
        item = metrics["objects"][name]
        common = item.get("common_center_valid_frames", 0)
        baseline = item.get("baseline_center_valid_frames", 0)
        coverage = 100.0 * float(item.get("common_center_coverage", 0.0))
        parts.append(f"{name[-1]}:{common}/{baseline} ({coverage:.1f}%)")
    return "; ".join(parts)


def compact_id(record: dict[str, Any]) -> str:
    return " · ".join(
        (
            target_label(record),
            mode_label(record),
            HEAD_LABELS.get(record["head_scope"], record["head_scope"]),
        )
    )


def build_markdown(report: dict[str, Any], report_path: Path, output: Path) -> str:
    records = report["records"]
    for record in records:
        record["_derived"] = derived_metrics(record)
    ranks = assign_ranks(records)
    valid = [record for record in records if record["_derived"]["center_ade"] is not None]
    for record in valid:
        source_value = record["metrics"].get("target_center_ade_norm")
        derived_value = record["_derived"]["center_ade"]
        if not finite(source_value) or not math.isclose(
            float(source_value), float(derived_value), rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError(
                f"target Center-ADE mismatch for {record['variant_id']}: "
                f"report={source_value}, derived={derived_value}"
            )
    valid.sort(key=lambda record: ranks["center_ade"][record["variant_id"]])
    invalid = [record for record in records if record not in valid]
    invalid.sort(key=lambda record: record["variant_id"])

    report_rel = os.path.relpath(report_path, output.parent)
    lines = [
        f"# Object Query 消融：完整轨迹指标排序",
        "",
        f"- Case：`{report['case']}`",
        f"- Seed：`{report['seed']}`",
        f"- Reference：{report['reference']}",
        f"- 原始报告：[`report.json`]({report_rel})",
        f"- 统计范围：**1 个 case × 1 个 seed × {report['expected_ablation_count']} 个消融视频**；不是跨 case/seed 平均",
        f"- 轨迹已提取：**{report['tracked_ablation_count']}/{report['expected_ablation_count']}**；通过质量门控并参与排序：**{report['ranked_ablation_count']}**；N/A：**{len(invalid)}**",
        "- Dashboard：[8092 可视化页面](http://localhost:8092/wan22-ti2v-legacy-physiciq67-samples?v=20&case=0613pybullet_sample_001460_w002&seed=47326)",
        "",
        "## 1. 排序口径",
        "",
        "主排名只使用相同 seed 未消融 Baseline 为 reference 的对象中心轨迹变化：",
        "",
        "```text",
        "c(t) = median of >=4 visible CoTracker points of that object",
        "Center-ADE_norm = mean_t ||c_abl(t) - c_base(t)||_2 / D0",
        "Trajectory impact (%D0) = 100 * mean_selected_objects(Center-ADE_norm)",
        "```",
        "",
        "`D0` 是 F00 对象 mask bbox 对角线。单对象实验只统计被消融对象；`all_objects` 对 A/B 做对象级宏平均。`%D0` 不是 0–100 分数，100 表示平均中心位移等于一个 F00 对象对角线，数值可以超过 100。所有指标都表示相对 Baseline 的差异强度，不表示生成质量，也不表示相对 simulator GT 更差。",
        "",
        "质量门控要求每个被选对象至少有 4 个共同中心有效帧，且 `common_valid / baseline_valid >= 80%`。任一被选对象失败时整条实验记为 `N/A`，不参与任何名次；`N/A` 不能解释为没有影响。",
        "",
        "## 2. 指标定义与排序方向",
        "",
        "| 指标 | 精确计算形式 | 本文单位 | 影响排序方向 | 解释与限制 |",
        "|---|---|---:|---|---|",
        "| Center-ADE（主指标） | `mean_t ||c_abl(t)-c_base(t)|| / D0` | `%D0` | 越大影响越强 | 全时段对象中心轨迹平均偏移；平均会弱化偶发单帧偏移 |",
        "| Center-FDE | `||c_abl(t*)-c_base(t*)|| / D0` | `%D0` | 越大影响越强 | `t*` 是最后共同中心有效帧，不一定恒为 F48 |",
        "| Velocity vector error | `mean_t ||[(c_abl(t+4)-c_abl(t))/4]-[(c_base(t+4)-c_base(t))/4]|| / D0` | `%D0/frame` | 越大影响越强 | 比较四帧差分速度向量，联合反映速度大小和方向 |",
        "| Point-ADE | 所有共同可见 CoTracker 点的 `mean ||p_abl-p_base|| / D0` | `%D0` | 越大影响越强 | 保留表面点级差异，但滚动、形变和点身份漂移会使其变大 |",
        "| PCK@5/10/20% | `mean 1[||p_abl-p_base|| < alpha*D0]` | `%` | **越小影响越强** | 与 Baseline 点轨迹仍落在阈值内的比例；这里不是 attention PCK head 的排名分数 |",
        "",
        "所有 `all_objects` 辅助指标都先在每个对象内计算，再对 A/B 做等权宏平均；不是把大小不同的对象点直接混池。不同指标不合成为一个未经验证的综合分数。",
        "",
        "## 3. 消融 ID 与信息流",
        "",
        "`R_t` 是对象 tube 在 latent 时刻 `t` 的 token 集合，`C_t` 是其余 token。M1/M2/M3 都是在 post-softmax attention entry 上置零且不重新归一化。",
        "",
        "| ID | 实现族 | 被删除的信息流 | 诊断含义 |",
        "|---|---|---|---|",
        "| M1 | `self_*` | `R K/V -> R Query` | 对象 tube 内部自支持/跨时传播 |",
        "| M2 | `incoming_*` | `C K/V -> R Query` | 环境、背景和其他对象向目标对象输入信息 |",
        "| M3 | `outgoing_*` | `R K/V -> C Query` | 目标对象向其他 token 广播信息 |",
        "",
        "| 时序后缀 | 条件 | 精确含义 |",
        "|---|---|---|",
        "| All-time | 所有 `tq,tk` | 删除该通路的全部时序配对 |",
        "| Same | `tk=tq` | 只删除同一 latent 帧配对；不是仅矩阵主对角线 `q=k` |",
        "| Future | `tk<tq` | 删除历史 K/V 向未来 Query 的传播 |",
        "| Past | `tk>tq` | 删除未来 K/V 向过去 Query 的反向控制 |",
        "",
        "| Head scope | 精确范围 |",
        "|---|---|",
        "| Top100 | 冻结 S039 PCK 排名第 1–100 的 layer-head |",
        "| Bottom100 | 同一冻结排名第 621–720 的等数量低 PCK 对照 |",
        "| All720 | 30 layers × 24 heads 的全部 720 个 layer-head |",
        "",
        "共同干预条件：这里的 Same/Future/Past 描述的是视频 latent 序列中的 `tq/tk` 关系；不是 diffusion 去噪 step。每个选中的矩阵块都会在 S000–S039 全部 40 个去噪步以及 conditional/unconditional 两个 CFG 分支上应用。对象区域使用由 Baseline 轨迹冻结得到的 `R_tube`，因此各消融视频共享同一套 token 选择，不会由消融结果反向改变 mask。",
        "",
        "## 4. 快速结论（仅限当前 case/seed）",
        "",
    ]

    target_counts = []
    for target in ("Object A", "Object B", "All objects"):
        target_records = [record for record in records if target_label(record) == target]
        target_valid = [
            record
            for record in target_records
            if record["_derived"]["center_ade"] is not None
        ]
        target_counts.append(f"{target} {len(target_valid)}/{len(target_records)}")
    lines.append("质量通过率：" + "；".join(target_counts) + "。")
    lines.append("")

    for rank, record in enumerate(valid[:10], start=1):
        lines.append(
            f"{rank}. **{compact_id(record)}**："
            f"Center-ADE = {metric_display('center_ade', record['_derived']['center_ade'])}%D0"
        )
    lines.extend(
        [
            "",
            "当前有效排名主要由 Object A 的大位移实验占据，但 Object A 只有部分实验通过轨迹门控，`all_objects` 又要求 A/B 同时通过。高影响视频更容易因对象外观、身份或可跟踪性改变而成为 N/A，因此有效条目的排序存在不可忽略的选择偏差。",
            "",
            "这是描述性单样本排序，不能据此声称某个 M/时序/head scope 在数据集层面稳定更重要；需要跨 case、跨 seed 的共同有效 cohort 才能作总体结论。",
            "",
            "## 5. 各指标 Top-10 影响排序",
            "",
        ]
    )

    for key, spec in METRICS.items():
        ordered = sorted(
            valid,
            key=lambda record: ranks[key].get(record["variant_id"], 10**9),
        )
        lines.extend(
            [
                f"### {spec['label']}（{spec['stronger']}表示影响更强）",
                "",
                f"| Rank | Target · Ablation · Heads | {spec['label']} ({spec['unit']}) | Overlay |",
                "|---:|---|---:|---|",
            ]
        )
        for record in ordered[:10]:
            lines.append(
                f"| {ranks[key][record['variant_id']]} | {compact_id(record)} | "
                f"{metric_display(key, record['_derived'][key])} | "
                f"{overlay_link(record, output)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 6. 完整数值表（67 个质量通过实验）",
            "",
            "以下按主指标 Center-ADE 排序。PCK 三列越小表示与 Baseline 点轨迹差异越强，其余误差列越大表示影响越强。",
            "",
            "| ADE Rank | Experiment | Target | Ablation | Heads | Center-ADE %D0 | Center-FDE %D0 | Velocity %D0/frame | Point-ADE %D0 | PCK@5% | PCK@10% | PCK@20% | Coverage | Overlay |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for record in valid:
        derived = record["_derived"]
        lines.append(
            f"| {ranks['center_ade'][record['variant_id']]} | {experiment_cell(record)} | "
            f"{target_label(record)} | {mode_label(record)} | "
            f"{HEAD_LABELS.get(record['head_scope'], record['head_scope'])} | "
            f"{metric_display('center_ade', derived['center_ade'])} | "
            f"{metric_display('center_fde', derived['center_fde'])} | "
            f"{metric_display('velocity', derived['velocity'])} | "
            f"{metric_display('point_ade', derived['point_ade'])} | "
            f"{metric_display('pck05', derived['pck05'])} | "
            f"{metric_display('pck10', derived['pck10'])} | "
            f"{metric_display('pck20', derived['pck20'])} | "
            f"{target_coverage(record)} | {overlay_link(record, output)} |"
        )

    lines.extend(
        [
            "",
            "## 7. 完整名次矩阵",
            "",
            "同一实验在每个指标下独立排序。PCK 已按“越小影响越强”排列；其余指标按“越大影响越强”排列。",
            "",
            "| Experiment | ADE Rank | FDE Rank | Velocity Rank | Point-ADE Rank | PCK@5 Rank | PCK@10 Rank | PCK@20 Rank |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in valid:
        variant = record["variant_id"]
        lines.append(
            f"| {experiment_cell(record)} | {ranks['center_ade'][variant]} | "
            f"{ranks['center_fde'][variant]} | {ranks['velocity'][variant]} | "
            f"{ranks['point_ade'][variant]} | {ranks['pck05'][variant]} | "
            f"{ranks['pck10'][variant]} | {ranks['pck20'][variant]} |"
        )

    lines.extend(
        [
            "",
            "## 8. 按消融目标的 Center-ADE 完整排序",
            "",
        ]
    )
    target_order = ("Object A", "Object B", "All objects")
    for target in target_order:
        subset = [record for record in records if target_label(record) == target]
        subset_valid = [record for record in subset if record["_derived"]["center_ade"] is not None]
        subset_valid.sort(key=lambda record: -record["_derived"]["center_ade"])
        lines.extend(
            [
                f"<details><summary>{target}：{len(subset_valid)}/{len(subset)} 可排名</summary>",
                "",
                "| Local rank | Ablation | Heads | Center-ADE %D0 | Global rank |",
                "|---:|---|---|---:|---:|",
            ]
        )
        for local_rank, record in enumerate(subset_valid, start=1):
            lines.append(
                f"| {local_rank} | {mode_label(record)} | "
                f"{HEAD_LABELS.get(record['head_scope'], record['head_scope'])} | "
                f"{metric_display('center_ade', record['_derived']['center_ade'])} | "
                f"{ranks['center_ade'][record['variant_id']]} |"
            )
        lines.extend(["", "</details>", ""])

    failed_objects = Counter()
    for record in invalid:
        for name in record["metrics"]["selected_objects"]:
            if not record["metrics"]["objects"][name].get("quality_pass"):
                failed_objects[name] += 1
    failure_summary = ", ".join(
        f"{name}={count}" for name, count in sorted(failed_objects.items())
    )
    lines.extend(
        [
            "## 9. N/A 与质量门控明细",
            "",
            f"共 {len(invalid)} 条不参与排名；失败对象计数（同一 all_objects 实验可涉及多个对象）：{failure_summary or '无'}。轨迹丢失可能正是强消融导致的外观/身份/遮挡变化，也可能是 tracker 失败，因此只能标为不可可靠量化，不能填 0。",
            "",
            "| Experiment | Target | Ablation | Heads | 失败对象与覆盖率 | 最后共同有效帧 | Overlay |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for record in invalid:
        failed = []
        last_frames = []
        metrics = record["metrics"]
        for name in metrics["selected_objects"]:
            item = metrics["objects"][name]
            common = item.get("common_center_valid_frames", 0)
            baseline = item.get("baseline_center_valid_frames", 0)
            coverage = 100.0 * float(item.get("common_center_coverage", 0.0))
            if not item.get("quality_pass"):
                failed.append(f"{name}: {common}/{baseline} ({coverage:.1f}%)")
            last_frame = item.get("last_common_visible_frame")
            last_frames.append(
                f"{name}: {'N/A' if last_frame is None else f'F{last_frame}'}"
            )
        lines.append(
            f"| {experiment_cell(record)} | {target_label(record)} | "
            f"{mode_label(record)} | {HEAD_LABELS.get(record['head_scope'], record['head_scope'])} | "
            f"{'; '.join(failed)} | {'; '.join(last_frames)} | {overlay_link(record, output)} |"
        )

    lines.extend(
        [
            "",
            "## 10. 复现",
            "",
            "重新从已有 `report.json` 生成本文档（不运行模型）：",
            "",
            "```bash",
            "cd /home/gaoya/Code_Video/DiffTrack-main",
            "/data/gaoya/miniconda3/envs/wan/bin/python \\",
            "  AAA_my_test/object_query_ablation_metrics/build_head_scope_trajectory_ranking_md.py \\",
            f"  {report_path}",
            "```",
            "",
            "重新提取轨迹和指标需运行 `bench.sh ... --head-scope-trajectory`；该步骤会使用 CoTracker GPU 推理。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    output = (
        args.output.resolve()
        if args.output
        else report_path.parent / "TRAJECTORY_METRICS_COMPLETE_RANKING.md"
    )
    report = json.loads(report_path.read_text())
    required = {"case", "seed", "records", "expected_ablation_count"}
    missing = required - report.keys()
    if missing:
        raise ValueError(f"report is missing required fields: {sorted(missing)}")
    markdown = build_markdown(report, report_path, output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(markdown, encoding="utf-8")
    temporary.replace(output)
    print(f"[pass] wrote {output} ({len(report['records'])} records)")


if __name__ == "__main__":
    main()
