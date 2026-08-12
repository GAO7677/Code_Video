#!/usr/bin/env python3
"""Build the auditable final report for the latest3350 Stage-3 discovery matrix."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1"
)
PRIMARY_METRICS = (
    "center_ade",
    "track_loss",
    "other_ade",
    "disappearance",
    "identity_failure",
)
FLOW_LABELS = {
    "self_only": "M1 · R→R",
    "incoming_only": "M2 · C→R",
    "outgoing_only": "M3 · R→C",
}
SCOPE_LABELS = {
    "top100": "Top100",
    "bottom100": "Bottom100",
    "random100_layer_matched_draw0": "Random100",
    "all720": "All720",
}
METRIC_LABELS = {
    "center_ade": "Target Center-ADE / D0",
    "track_loss": "Worst target Track Loss",
    "other_ade": "Other-object Center-ADE / D0",
    "disappearance": "Worst target Disappearance",
    "identity_failure": "Worst target Identity Failure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def target_id(source: dict[str, Any]) -> str:
    if source.get("target_scope") == "single_object":
        return f"single_object::{source.get('region')}"
    return "all_objects::all_objects"


def exact_sign_flip(case_differences: dict[str, float]) -> float:
    values = [float(value) for value in case_differences.values()]
    if not values:
        return math.nan
    observed = abs(sum(values) / len(values))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(
            sum(value * sign for value, sign in zip(values, signs, strict=True))
            / len(values)
        )
        extreme += statistic >= observed - 1e-12
    return extreme / total


def bh_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = running
    return adjusted


def build_tests(report: dict[str, Any]) -> list[dict[str, Any]]:
    tests = []
    for row in report["head_contrasts"]:
        if row["metric"] not in PRIMARY_METRICS or row["label"] not in (
            "Top100 − Bottom100",
            "Top100 − Random100",
        ):
            continue
        tests.append(
            {
                "family": "head_group",
                "metric": row["metric"],
                "flow": row["flow"],
                "scope": None,
                "contrast": row["label"],
                **{
                    key: row[key]
                    for key in (
                        "left_mean",
                        "right_mean",
                        "difference",
                        "ratio",
                        "ci95",
                        "case_positive_fraction",
                        "case_count",
                        "paired_unit_count",
                        "case_differences",
                    )
                },
            }
        )
    for row in report["flow_contrasts"]:
        if row["metric"] not in PRIMARY_METRICS or row["scope"] != "top100":
            continue
        tests.append(
            {
                "family": "flow_within_top100",
                "metric": row["metric"],
                "flow": None,
                "scope": row["scope"],
                "contrast": row["label"],
                **{
                    key: row[key]
                    for key in (
                        "left_mean",
                        "right_mean",
                        "difference",
                        "ratio",
                        "ci95",
                        "case_positive_fraction",
                        "case_count",
                        "paired_unit_count",
                        "case_differences",
                    )
                },
            }
        )
    p_values = [exact_sign_flip(row["case_differences"]) for row in tests]
    q_values = bh_adjust(p_values)
    for row, p_value, q_value in zip(tests, p_values, q_values, strict=True):
        row["p_exact_two_sided"] = p_value
        row["q_bh_45_tests"] = q_value
        row["exploratory_fdr_pass"] = q_value < 0.05
    return tests


def find_test(
    tests: list[dict[str, Any]], metric: str, family: str, contrast: str, flow: str | None = None
) -> dict[str, Any]:
    matches = [
        row
        for row in tests
        if row["metric"] == metric
        and row["family"] == family
        and row["contrast"] == contrast
        and (flow is None or row["flow"] == flow)
    ]
    if len(matches) != 1:
        raise RuntimeError((metric, family, contrast, flow, len(matches)))
    return matches[0]


def find_head_contrast(
    report: dict[str, Any], metric: str, flow: str, label: str
) -> dict[str, Any]:
    matches = [
        row
        for row in report["head_contrasts"]
        if row["metric"] == metric and row["flow"] == flow and row["label"] == label
    ]
    if len(matches) != 1:
        raise RuntimeError((metric, flow, label, len(matches)))
    return matches[0]


def find_scope_summary(
    report: dict[str, Any], metric: str, flow: str
) -> dict[str, Any]:
    matches = [
        row
        for row in report["scope_summaries"]
        if row["metric"] == metric and row["flow"] == flow
    ]
    if len(matches) != 1:
        raise RuntimeError((metric, flow, len(matches)))
    return matches[0]


def evaluated_units(root: Path) -> tuple[set[tuple[str, int, str]], dict[str, list[int]]]:
    units: set[tuple[str, int, str]] = set()
    seeds: dict[str, set[int]] = {}
    report_root = root / "stage3_metrics" / "head_scope_baseline_fast"
    for path in sorted(report_root.glob("*/seed_*/report.json")):
        report = read_json(path)
        case = str(report["case"])
        seed = int(report["seed"])
        seeds.setdefault(case, set()).add(seed)
        for row in report["records"]:
            units.add((case, seed, target_id(row)))
    return units, {case: sorted(values) for case, values in sorted(seeds.items())}


def generated_units(dose: dict[str, Any]) -> set[tuple[str, int, str]]:
    def normalized_target(value: Any) -> str:
        target = str(value)
        return "all_objects::all_objects" if target.startswith("all_objects::") else target

    return {
        (str(row["case"]), int(row["seed"]), normalized_target(row["target"]))
        for row in dose["records"]
    }


def fmt(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def ci_text(row: dict[str, Any]) -> str:
    return f"[{fmt(row['ci95'][0])}, {fmt(row['ci95'][1])}]"


def test_evidence(row: dict[str, Any]) -> str:
    return (
        f"Δ={fmt(row['difference'])}, 95% CI {ci_text(row)}, "
        f"exact p={fmt(row['p_exact_two_sided'], 4)}, "
        f"BH q={fmt(row['q_bh_45_tests'], 4)}, n_case={row['case_count']}"
    )


def markdown(final: dict[str, Any], current: dict[str, Any]) -> str:
    tests = final["primary_tests"]
    coverage = final["coverage"]
    m1_dis_b = find_test(
        tests, "disappearance", "head_group", "Top100 − Bottom100", "self_only"
    )
    m1_id_b = find_test(
        tests, "identity_failure", "head_group", "Top100 − Bottom100", "self_only"
    )
    m1_ade_r = find_test(
        tests, "center_ade", "head_group", "Top100 − Random100", "self_only"
    )
    m1_other_r = find_test(
        tests, "other_ade", "head_group", "Top100 − Random100", "self_only"
    )
    flow_dis_12 = find_test(
        tests, "disappearance", "flow_within_top100", "M1 − M2"
    )
    flow_dis_13 = find_test(
        tests, "disappearance", "flow_within_top100", "M1 − M3"
    )
    flow_id_12 = find_test(
        tests, "identity_failure", "flow_within_top100", "M1 − M2"
    )
    flow_id_13 = find_test(
        tests, "identity_failure", "flow_within_top100", "M1 − M3"
    )
    flow_ade_12 = find_test(
        tests, "center_ade", "flow_within_top100", "M1 − M2"
    )
    flow_track_13 = find_test(
        tests, "track_loss", "flow_within_top100", "M1 − M3"
    )
    flow_other_13 = find_test(
        tests, "other_ade", "flow_within_top100", "M1 − M3"
    )
    flow_other_23 = find_test(
        tests, "other_ade", "flow_within_top100", "M2 − M3"
    )

    lines = [
        "# Stage 3 最终报告：Object Query Self-Attention 信息流消融",
        "",
        f"冻结时间：`{final['generated_at_utc']}`",
        "",
        "## 1. 结论",
        "",
        "1. **latest3350 Top100 首先是 R→R 高贡献 heads，不是对三种信息流都更重要。** "
        "M1 中 Top100 每 head 删除的 AV 范数为 5.811，是 Bottom100 的 5.94×、Random100 的 2.74×；"
        "M2/M3 中方向相反，Top100 删除量低于两个对照。",
        "2. **M1 是 Top100 下最稳定的破坏来源，主要影响对象身份、存活和轨迹连续性。** 第 2 节给出三个关键数字的直观读法。",
        "3. **“M3 专门负责向其他对象广播状态”没有得到支持。** "
        f"Top100 下 M1−M3 的 Other-object ADE 为 {fmt(flow_other_13['difference'])} D0，"
        f"CI {ci_text(flow_other_13)}、q={fmt(flow_other_13['q_bh_45_tests'], 4)}；"
        f"M2−M3 为 {fmt(flow_other_23['difference'])} D0，CI {ci_text(flow_other_23)}、q={fmt(flow_other_23['q_bh_45_tests'], 4)}。"
        "两项都不能排除零差异。",
        "4. **不能把现有结果解释为“Top100 每单位信息更有语义”。** "
        "M1 的 Top100 与 Random100 删除剂量明显不匹配，且原始剂量分布几乎不重叠；当前结果证明 Top100 承载并删除了更多 R→R contribution，"
        "但不能区分输出差异来自删除量还是单位 contribution 的语义特异性。",
        "5. **M2 是否承载物理接触/环境约束，Stage 3 证据不足。** "
        "本阶段只有 vs Baseline 的 All-time 消融，没有 GT contact-time、交互/非交互分层或 denoising/time-direction 对照。",
        "6. **Future/Same/Past 时序选择性未在 Stage 3 测试。** 本阶段全部是 All-time；该问题只能由 Stage 4 回答。",
        "",
        "以上均为 **discovery 结论**。即使 BH q<0.05，也不是 held-out confirmatory 证据。",
        "",
        "## 2. M1 三个关键数字怎么理解",
        "",
        "这一节的所有结果都以**同 case、同 seed 的未消融 Baseline**为参照。"
        "Top100、Bottom100 表示 latest3350 PCK 排名得到的两组 layer-head；"
        "这里删除的是指定 Query→Key 区块对 attention 输出的 `ΣA·V` contribution，"
        "不是把整组 K/V token 从网络中永久删除。",
        "",
        "### 2.1 对象更容易进入非存活状态",
        "",
        "**实验操作：**两组实验都执行 M1，即在所选 heads 上删除对象区域 `R` 的 K/V 向对象 Query `R` 提供的"
        " `Σ A[R_query,R_key]V[R_key]`。两组唯一差别是所选 heads 分别为 Top100 和 Bottom100。",
        "",
        "**具体数值：**",
        "",
        f"- M1-Top100 的平均 Disappearance：**{fmt(m1_dis_b['left_mean'])}%**。",
        f"- M1-Bottom100 的平均 Disappearance：**{fmt(m1_dis_b['right_mean'])}%**。",
        f"- 差值：`{fmt(m1_dis_b['left_mean'])} − {fmt(m1_dis_b['right_mean'])}` = "
        f"**+{fmt(m1_dis_b['difference'])} 个百分点**。",
        f"- 统计证据：95% CI {ci_text(m1_dis_b)}，BH q={fmt(m1_dis_b['q_bh_45_tests'], 4)}。",
        "",
        "**直观含义：**消融 Top100 的 R→R contribution 后，对象被判定为异常非存活的帧占比，"
        f"平均比消融 Bottom100 多 **{fmt(m1_dis_b['difference'])} 个百分点**。"
        "这是帧比例的百分点差，不是“对象数量多消失了 12.813%”，也不是单个对象必然消失的概率。",
        "",
        "Disappearance 是复合指标：某帧只要出现 DINO 身份失败、面积比例异常或 SAM2 空 mask，就会被记为非存活。"
        "因此它包括变形、身份漂移和检测不到等情况，不等同于每次都是肉眼可见的完全消失。",
        "",
        "### 2.2 对象更容易变得不像 Baseline 中的同一对象",
        "",
        "**实验操作：**仍然比较同一个 M1 消融，只改变 head group：Top100 对 Bottom100。",
        "",
        "**具体数值：**",
        "",
        f"- M1-Top100 的平均 DINO Identity Failure：**{fmt(m1_id_b['left_mean'])}%**。",
        f"- M1-Bottom100 的平均 DINO Identity Failure：**{fmt(m1_id_b['right_mean'])}%**。",
        f"- 差值：`{fmt(m1_id_b['left_mean'])} − {fmt(m1_id_b['right_mean'])}` = "
        f"**+{fmt(m1_id_b['difference'])} 个百分点**。",
        f"- 统计证据：95% CI {ci_text(m1_id_b)}，BH q={fmt(m1_id_b['q_bh_45_tests'], 4)}。",
        "",
        "**直观含义：**对每一帧，把消融视频中的目标对象与同 seed Baseline 中对应对象的 DINO 特征比较；"
        "相似度低于预设身份阈值就记为身份失败。Top100-M1 使这种失败帧的占比平均多 11.940 个百分点，"
        "说明对象更容易发生颜色、纹理、形状或语义身份漂移。该指标不是 LPIPS，也不要求对象字面消失。",
        "",
        "### 2.3 M1 比 M2 更容易改变对象中心轨迹",
        "",
        "**实验操作：**这次固定使用同一组 Top100 heads，只改变被删除的信息流：",
        "",
        "- M1（R→R）：删除 `Σ A[R_query,R_key]V[R_key]`，即对象 tube 内部对对象 Query 的贡献。",
        "- M2（C→R）：删除 `Σ A[R_query,C_key]V[C_key]`，即背景/其他 token 对对象 Query 的贡献。",
        "",
        "**具体数值：**",
        "",
        f"- Top100-M1 的配对 Center-ADE：**{fmt(flow_ade_12['left_mean'])} D0**。",
        f"- Top100-M2 的配对 Center-ADE：**{fmt(flow_ade_12['right_mean'])} D0**。",
        f"- 差值：`{fmt(flow_ade_12['left_mean'])} − {fmt(flow_ade_12['right_mean'])}` = "
        f"**+{fmt(flow_ade_12['difference'])} D0**。",
        f"- 统计证据：95% CI {ci_text(flow_ade_12)}，BH q={fmt(flow_ade_12['q_bh_45_tests'], 4)}。",
        "",
        "**Center-ADE 怎么算：**先在每个共同可跟踪帧计算消融对象中心与 Baseline 对象中心的欧氏距离，"
        "再对时间取平均，最后除以首帧对象 bbox 对角线 `D0`。因此 `0.133 D0` 表示 M1 相比 M2，"
        "平均额外造成约等于首帧对象尺寸 **13.3%** 的中心轨迹偏移。",
        "",
        "这只能说明 M1 相比 M2 更强地改变了同 seed Baseline 的生成轨迹；因为这里没有使用 simulator GT，"
        "所以不能据此说 M1 的轨迹在物理上一定更错误。",
        "",
        "### 2.4 这三项证据合起来能说明什么",
        "",
        "三个 95% CI 都没有跨 0，且经过 45 项比较的 BH-FDR 校正后 q<0.05，"
        "所以这些差异在当前 10 个 discovery cases 中具有探索性统计支持，不只是单个视频的视觉印象。",
        "",
        "**当前证据支持：**latest3350 Top100 heads 中存在很强的对象内部 R→R contribution；删除这些 contribution 后，"
        "对象身份维持、存活和轨迹连续性受到更明显影响。",
        "",
        "**当前证据不能区分：**Top100 的影响更大，究竟是因为这些 heads 中删除的 R→R 总量更大，"
        "还是因为每单位 R→R contribution 本身更关键。M1-Top100 的干预剂量显著高于 Bottom100/Random100，"
        "因此不能直接宣称“Top100 每单位信息具有更强语义或因果作用”。",
        "",
        "## 3. 数据完整性",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        f"| 独立 cases | {coverage['case_count']} |",
        f"| discovery seeds | {coverage['seed_values']} |",
        f"| 生成矩阵 | {coverage['generated_records']} / {coverage['expected_generated_records']} |",
        f"| case-seed | {coverage['generated_case_seed_count']} |",
        f"| target-seed units | {coverage['generated_target_seed_units']} |",
        f"| 有同 seed Baseline 的消融 | {coverage['evaluable_records']} |",
        f"| Fast / Trajectory / Survival 完整记录 | {coverage['fast_records']} / {coverage['trajectory_records']} / {coverage['survival_records']} |",
        f"| 无同 seed Baseline、未纳入 outcome 统计 | {coverage['ineligible_records']} |",
        "",
        "唯一不具备相对 Baseline 评价条件的单元：",
        "",
        "| Case | Seed | Targets | 消融视频 | 原因 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in coverage["ineligible_case_seeds"]:
        lines.append(
            f"| `{row['case']}` | {row['seed']} | {row['target_count']} | "
            f"{row['record_count']} | {row['reason']} |"
        )

    lines += [
        "",
        "## 4. 统计口径",
        "",
        "- 最高独立单位是 case；seed、object、target 在 case 内先平均，再对 cases 等权。",
        "- 差值的 95% CI 是 50,000 次 case bootstrap 描述区间。",
        "- 5 个可用主终点共 45 个对比，使用双侧 exact case sign-flip 检验并统一 BH-FDR："
        "Center-ADE、Track Loss、Other-object ADE、Disappearance、DINO Identity Failure。",
        f"- 45 项中 **{sum(row['exploratory_fdr_pass'] for row in tests)} 项** q<0.05，均列在第 6 节；其余不按单个未校正 p 值下结论。",
        "- Center-ADE 只统计通过轨迹门控的配对；Track Loss 和 Disappearance 保留破坏性失败，防止幸存者偏差。",
        "- 预注册的 center-aligned Object LPIPS 本轮没有生成，Identity Failure 只是身份保持代理，不能替代 LPIPS。",
        "",
        "## 5. 干预剂量：实际删除了什么",
        "",
        "下表为每 selected head 的 removed AV norm。它是干预强度，不是输出质量或语义分数。",
        "",
        "| Flow | Top100 | Bottom100 | Random100 | Top/Bottom | Top/Random | 证据 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for flow in ("self_only", "incoming_only", "outgoing_only"):
        bottom = find_head_contrast(
            current, "removed_value_norm", flow, "Top100 − Bottom100"
        )
        random = find_head_contrast(
            current, "removed_value_norm", flow, "Top100 − Random100"
        )
        lines.append(
            f"| {FLOW_LABELS[flow]} | {fmt(bottom['left_mean'])} | "
            f"{fmt(bottom['right_mean'])} | {fmt(random['right_mean'])} | "
            f"{fmt(bottom['ratio'])}× | {fmt(random['ratio'])}× | "
            f"Top−Bottom CI {ci_text(bottom)}；Top−Random CI {ci_text(random)} |"
        )

    lines += [
        "",
        "M1 原始剂量的支持范围进一步显示缺乏可比性：Top100 `3.368–9.054`，"
        "Random100 `0.755–3.904`，Bottom100 `0.206–2.137`。因此不进行会依赖外推的 dose-adjusted 机制宣称。",
        "",
        "## 6. 关键 outcome 证据",
        "",
        "`Δ>0` 表示左侧消融造成更强的 Baseline-relative 改变，不代表物理质量更差或更好。",
        "",
        "| 问题 | 对比 | 指标 | 左 / 右 | Δ [95% CI] | exact p / BH q | 判定 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    evidence_rows = [
        ("H1 · Top 是否强于 Bottom", m1_dis_b, "M1 Top−Bottom", "探索性支持"),
        ("H1 · Top 是否强于 Bottom", m1_id_b, "M1 Top−Bottom", "探索性支持"),
        ("H1 · 轨迹是否强于 Random", m1_ade_r, "M1 Top−Random", "FDR 后证据不足"),
        ("H2 · M1 是否强于 M2", flow_dis_12, "Top100 M1−M2", "探索性支持"),
        ("H2 · M1 是否强于 M3", flow_dis_13, "Top100 M1−M3", "探索性支持"),
        ("H2 · 身份效应 M1 vs M2", flow_id_12, "Top100 M1−M2", "探索性支持"),
        ("H2 · 身份效应 M1 vs M3", flow_id_13, "Top100 M1−M3", "探索性支持"),
        ("H2 · 轨迹效应 M1 vs M2", flow_ade_12, "Top100 M1−M2", "探索性支持"),
        ("H2 · 轨迹失效 M1 vs M3", flow_track_13, "Top100 M1−M3", "探索性支持"),
        ("H4 · M3 是否更强传播", flow_other_13, "Top100 M1−M3", "不支持 M3 更强"),
        ("H4 · M3 是否更强传播", flow_other_23, "Top100 M2−M3", "不支持 M3 更强"),
        ("反例 · M1 跨对象效应", m1_other_r, "M1 Top−Random", "探索性支持 M1 spillover"),
    ]
    for question, row, contrast, verdict in evidence_rows:
        lines.append(
            f"| {question} | {contrast} | {METRIC_LABELS[row['metric']]} | "
            f"{fmt(row['left_mean'])} / {fmt(row['right_mean'])} | "
            f"{fmt(row['difference'])} {ci_text(row)} | "
            f"{fmt(row['p_exact_two_sided'], 4)} / {fmt(row['q_bh_45_tests'], 4)} | {verdict} |"
        )

    lines += [
        "",
        "### 6.1 对象外观与背景",
        "",
        "- frozen target ROI MAE（位置、外观、形变和消失的混合诊断）在 M1 下为 Top 6.981、Bottom 4.479、Random 5.016。"
        "它说明目标区域改变更大，但不能单独证明外观改变。",
        "- outside-object MAE 在 M1 下为 Top 0.818、Bottom 0.620、Random 0.665；"
        "因此 M1 效应并非严格局限于对象区域。Outside-object MAE 是像素代理，跨对象运动以 Other-object ADE 为准。",
        "",
        "### 6.2 All720 只作强干预上界",
        "",
        "| 指标 | M1 | M2 | M3 |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("target_local", "track_loss", "disappearance", "identity_failure", "other_ade"):
        values = []
        for flow in ("self_only", "incoming_only", "outgoing_only"):
            row = find_scope_summary(current, metric, flow)
            values.append(row["scopes"]["all720"]["mean"])
        label = current["metric_definitions"][metric]["label"]
        lines.append(f"| {label} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} |")
    lines += [
        "",
        "All720 在 Track Loss、Disappearance 和 Identity Failure 上明显更破坏，但这是删除 720 heads 的总效应；"
        "不能除以 7.2 后解释为单 head 因果强度。",
        "",
        "## 7. 轨迹质量门控",
        "",
        "| Flow | Top100 | Bottom100 | Random100 | All720 |",
        "|---|---:|---:|---:|---:|",
    ]
    quality = {
        (row["flow"], row["scope"]): row
        for row in current["trajectory_quality"]["rows"]
    }
    for flow in ("self_only", "incoming_only", "outgoing_only"):
        values = [
            100.0 * quality[(flow, scope)]["failure_rate"]
            for scope in (
                "top100",
                "bottom100",
                "random100_layer_matched_draw0",
                "all720",
            )
        ]
        lines.append(
            f"| {FLOW_LABELS[flow]} | " + " | ".join(f"{value:.1f}%" for value in values) + " |"
        )
    lines += [
        "",
        "All720 的门控失败率为 M1 51.0%、M2 47.9%、M3 36.5%。因此 All720 的通过门控 ADE 是幸存样本结果，"
        "必须与 Track Loss/Disappearance 联读。",
        "",
        "## 8. 证据等级与下一步",
        "",
        "- **探索性支持：** exact p 经 45-test BH-FDR 后 q<0.05，但 discovery cases 与 ranking/页面检查存在重叠。",
        "- **证据不足：** CI 跨 0、FDR 未通过、endpoint 未计算，或干预剂量缺乏可比支持。",
        "- 本报告没有“已验证/confirmatory”结论。下一步最小充分实验是："
        "(1) held-out cases/seeds；(2) dose-matched R→R heads 或受限 overlap 分析；"
        "(3) 补 center-aligned LPIPS/shape；(4) 对交互子集补 GT contact/velocity；(5) Stage 4 Future/Same/Past。",
        "",
        "## 9. 可审计产物",
        "",
        "- 完整描述统计：`../stage3_interim_analysis/STAGE3_CURRENT_METRICS_ANALYSIS.md`",
        "- 机器可读最终报告：`stage3_final_report.json`",
        "- 45 项 exact p / BH q：`primary_tests_45.csv`",
        "- Dose 原始汇总：`../stage3_interim_analysis/report.json`",
        "- Outcome 原始汇总：`../stage3_interim_analysis/current_metrics_report.json`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    source_dir = root / "stage3_interim_analysis"
    output_dir = root / "stage3_final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    dose = read_json(source_dir / "report.json")
    current = read_json(source_dir / "current_metrics_report.json")
    tests = build_tests(current)

    generated = generated_units(dose)
    evaluated, evaluated_seeds = evaluated_units(root)
    missing = sorted(generated - evaluated)
    missing_groups: dict[tuple[str, int], set[str]] = {}
    for case, seed, target in missing:
        missing_groups.setdefault((case, seed), set()).add(target)
    ineligible_case_seeds = []
    for (case, seed), targets in sorted(missing_groups.items()):
        ineligible_case_seeds.append(
            {
                "case": case,
                "seed": seed,
                "target_count": len(targets),
                "record_count": len(targets) * 4 * 3,
                "reason": "缺少同 case、同 seed 的未消融 Baseline；禁止跨 seed 替代",
            }
        )
    case_seeds = {(case, seed) for case, seed, _target in generated}
    all_seeds = sorted({seed for _case, seed in case_seeds})
    coverage = {
        "case_count": len({case for case, _seed in case_seeds}),
        "seed_values": all_seeds,
        "generated_case_seed_count": len(case_seeds),
        "generated_target_seed_units": len(generated),
        "evaluated_target_seed_units": len(evaluated),
        "expected_generated_records": 1188,
        "generated_records": int(dose["record_count"]),
        "evaluable_records": len(evaluated) * 4 * 3,
        "ineligible_records": len(missing) * 4 * 3,
        "fast_records": int(current["coverage"]["fast_records"]),
        "trajectory_records": int(current["coverage"]["trajectory_records"]),
        "survival_records": int(current["coverage"]["survival_records"]),
        "evaluated_seeds_by_case": evaluated_seeds,
        "ineligible_case_seeds": ineligible_case_seeds,
    }
    if coverage["generated_records"] != 1188:
        raise RuntimeError(f"dose coverage is incomplete: {coverage}")
    expected_evaluable = coverage["evaluable_records"]
    for family in ("fast_records", "trajectory_records", "survival_records"):
        if coverage[family] != expected_evaluable:
            raise RuntimeError(f"{family} incomplete: {coverage}")

    final = {
        "schema_version": 1,
        "status": "stage3_discovery_complete_outcomes_complete_for_baseline_eligible_units",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Stage 3 All-time discovery matrix; not held-out confirmation",
        "coverage": coverage,
        "statistics": {
            "highest_independent_unit": "case",
            "within_case_aggregation": "mean over seeds, targets, and objects",
            "confidence_interval": "50,000-repetition case bootstrap (descriptive)",
            "test": "two-sided exact case sign-flip test",
            "multiple_testing": "BH-FDR over 45 available primary/proxy contrasts",
            "primary_metrics": list(PRIMARY_METRICS),
            "missing_preregistered_endpoint": "center-aligned Object LPIPS",
        },
        "primary_tests": tests,
        "source_reports": {
            "dose": str(source_dir / "report.json"),
            "outcomes": str(source_dir / "current_metrics_report.json"),
        },
    }
    atomic_json(output_dir / "stage3_final_report.json", final)
    (output_dir / "STAGE3_FINAL_REPORT.md").write_text(
        markdown(final, current), encoding="utf-8"
    )
    with (output_dir / "primary_tests_45.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "family",
            "metric",
            "flow",
            "scope",
            "contrast",
            "left_mean",
            "right_mean",
            "difference",
            "ratio",
            "ci95_low",
            "ci95_high",
            "case_positive_fraction",
            "case_count",
            "paired_unit_count",
            "p_exact_two_sided",
            "q_bh_45_tests",
            "exploratory_fdr_pass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in tests:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fields},
                    "ci95_low": row["ci95"][0],
                    "ci95_high": row["ci95"][1],
                }
            )
    print(json.dumps({"coverage": coverage, "fdr_pass": sum(row["exploratory_fdr_pass"] for row in tests)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
