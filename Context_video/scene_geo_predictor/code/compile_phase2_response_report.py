"""Compile the seed-42 Phase-2 response recheck into one Markdown report.

The CSV/JSON/NPZ exports remain machine-readable attachments.  This command
only reads them and rewrites the human-facing ``reviewed_conclusion.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


EPOCH_ORDER = {"50": 0, "100": 1, "200": 2, "analytic_cv": 3}
SPLIT_ORDER = {"train": 0, "dev": 1}
ARM_ORDER = {"motion_only": 0, "point_geometry_resample": 1,
             "finite_surface_geometry": 2, "analytic_cv": 3}
FAMILY_ORDER = {"aperture": 0, "deflector": 1, "support_edge": 2}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text())


def fmt(value, digits: int = 3, suffix: str = "") -> str:
    if value in (None, "", "None", "nan", "NaN"):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}{suffix}"


def fmt_mm(value, digits: int = 2) -> str:
    if value in (None, "", "None", "nan", "NaN"):
        return "N/A"
    return fmt(float(value) * 1000.0, digits, " mm")


def fmt_sci(value, digits: int = 3, suffix: str = "") -> str:
    if value in (None, "", "None", "nan", "NaN"):
        return "N/A"
    return f"{float(value):.{digits}e}{suffix}"


def pct(value, digits: int = 1) -> str:
    if value in (None, "", "None", "nan", "NaN"):
        return "N/A"
    return fmt(float(value) * 100.0, digits, "%")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_无记录。_\n"
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def link(root: Path, name: str, label: str | None = None) -> str:
    target = root / name
    return f"[{label or name}](<{target}>)"


def sort_summary(row: dict):
    return (EPOCH_ORDER.get(str(row["epoch"]), 99), SPLIT_ORDER.get(row["split"], 99),
            FAMILY_ORDER.get(row["family"], 99), ARM_ORDER.get(row["arm"], 99))


def summary_rows(summary: list[dict]) -> list[list[str]]:
    rows = []
    for row in sorted(summary, key=sort_summary):
        rows.append([
            str(row["epoch"]), row["split"], row["family"], row["arm"],
            fmt_mm(row["ADE_m_mean"]), fmt_mm(row["FDE_m_mean"]),
            fmt(row["interval_velocity_MAE_mps_mean"], 4, " m/s"),
            str(row["strong_pair_count"]), str(row["strong_history_count"]),
            fmt_mm(float(row["strong_E_delta_mm_mean"]) / 1000.0),
            fmt(row["strong_R_delta_mean"], 3), pct(row["strong_pass_fraction"]),
            str(row["weak_pair_count"]), fmt_mm(float(row["weak_E_delta_mm_mean"]) / 1000.0),
            fmt(row["weak_R_delta_mean"], 3), str(row["negative_pair_count"]),
            fmt(float(row["equal_D_pred_mm_mean"]) / 1000.0, 2, " mm"),
            pct(row["equal_pass_fraction"]),
        ])
    return rows


def response_rows(oldnew: list[dict], summary: list[dict]) -> list[list[str]]:
    summary_lookup = {(str(row["epoch"]), row["split"], row["family"], row["arm"], "strong"): row
                      for row in summary if str(row["epoch"]) == "200"}
    rows = []
    selected = [row for row in oldnew if row["response_layer"] == "strong"]
    selected.sort(key=lambda row: (SPLIT_ORDER.get(row["split"], 99), FAMILY_ORDER.get(row["family"], 99),
                                   ARM_ORDER.get(row["arm"], 99)))
    for row in selected:
        key = ("200", row["split"], row["family"], row["arm"], "strong")
        current = summary_lookup.get(key, {})
        rows.append([
            row["split"], row["family"], row["arm"], row["row_count_new"],
            fmt(row["old_amplitude_ratio_mean"], 3), fmt(row["old_amplitude_error_mean"], 3),
            fmt_mm(row["new_vector_E_delta_mm_mean"] and float(row["new_vector_E_delta_mm_mean"]) / 1000.0),
            fmt(row["new_vector_R_delta_mean"], 3), pct(current.get("strong_pass_fraction")),
            fmt(row["new_amplitude_ratio_mean"], 3), fmt(row["new_amplitude_error_mean"], 3),
        ])
    return rows


def equal_rows(oldnew: list[dict]) -> list[list[str]]:
    selected = [row for row in oldnew if row["response_layer"] == "equal_future"]
    selected.sort(key=lambda row: (SPLIT_ORDER.get(row["split"], 99), FAMILY_ORDER.get(row["family"], 99),
                                   ARM_ORDER.get(row["arm"], 99)))
    return [[row["split"], row["family"], row["arm"], row["row_count_new"],
             fmt(row["old_equal_D_pred_mm_mean"], 2, " mm"),
             fmt(row["new_equal_D_pred_mm_mean"], 2, " mm")]
            for row in selected]


def sampling_rows(rows: list[dict]) -> list[list[str]]:
    rows = sorted(rows, key=lambda row: (SPLIT_ORDER.get(row["split"], 99), FAMILY_ORDER.get(row["family"], 99)))
    return [[row["split"], row["family"], row["episode_count"],
             fmt(row["new_S_center_m_mean"], 4, " m"), fmt(row["new_S_pairwise_m_mean"], 4, " m"),
             fmt(row["new_S_max_pair_m_max"], 4, " m"), fmt(row["new_S_seed0_m_mean"], 4, " m"),
             fmt(row["max_abs_S_pairwise_old_new_m"], 2, " m"), row["threshold_status"]]
            for row in rows]


def coverage_rows(rows: list[dict]) -> list[list[str]]:
    rows = sorted(rows, key=lambda row: (SPLIT_ORDER.get(row["split"], 99),
                                         {"strong": 0, "weak": 1, "equal_future": 2}.get(row["response_layer"], 99)))
    return [[row["split"], row["response_layer"], row["physical_pair_count"], row["history_count"],
             row["total_history_count"], pct(row["history_coverage_fraction"]), row["event_category_counts_for_layer"]]
            for row in rows]


def event_rows(rows: list[dict]) -> list[list[str]]:
    selected = [row for row in rows if row["epoch"] == "200" and row["event_source"] == "manifest_event_category"]
    selected.sort(key=lambda row: (SPLIT_ORDER.get(row["split"], 99), ARM_ORDER.get(row["arm"], 99), row["event_category"]))
    return [[row["split"], row["arm"], row["event_category"], row["episode_seed_count"],
             row["unique_episode_count"], row["history_count"], fmt_mm(row["strong_E_delta_mm_mean"] and float(row["strong_E_delta_mm_mean"]) / 1000.0),
             fmt_mm(row["weak_E_delta_mm_mean"] and float(row["weak_E_delta_mm_mean"]) / 1000.0)] for row in selected]


def penetration_rows(rows: list[dict]) -> list[list[str]]:
    selected = [row for row in rows if row["epoch"] == "200" and row["family"] == "aperture"]
    selected.sort(key=lambda row: (SPLIT_ORDER.get(row["split"], 99), {"gt": 0, "pred": 1}.get(row["truth"], 2),
                                   ARM_ORDER.get(row["arm"], 99), row["collider"]))
    return [[row["split"], row["truth"], row["arm"], row["collider"], row["unique_episode_denominator"],
             row["episode_seed_denominator"], row["frame_denominator"], row["cases_gt5mm"],
             pct(row["case_gt5mm_rate"]), row["frames_gt5mm"], pct(row["frame_gt5mm_rate"]),
             fmt(float(row["p95_case_max_depth_m"]) * 1000.0, 2, " mm"),
             fmt(float(row["max_depth_m"]) * 1000.0, 2, " mm")] for row in selected]


def build_report(root: Path) -> str:
    status = read_json(root / "recheck_status.json")
    definition = read_json(root / "metric_definition.json")
    selftests = read_json(root / "metric_numeric_selftests.json")
    mapping = read_json(root / "metric_old_new_mapping.json")
    source_hashes = read_json(root / "source_hashes.json")
    repro = read_json(root / "metric_repro_check.json")
    prediction_source = read_json(root / "prediction_source.json")
    summary = read_csv(root / "train_dev_response_summary.csv")
    oldnew = read_csv(root / "old_new_numeric_comparison.csv")
    sampling = read_csv(root / "sampling_old_new_comparison.csv")
    coverage = read_csv(root / "deflector_strong_response_coverage.csv")
    event_summary = read_csv(root / "deflector_event_coverage_summary.csv")
    penetration = read_csv(root / "penetration_summary.csv")
    command = (root / "recheck_command.txt").read_text().strip()

    files = [
        ("recheck_status.json", "执行状态"), ("recheck_command.txt", "实际回算命令"),
        ("metric_definition.json", "指标定义"), ("metric_numeric_selftests.json", "数值自检"),
        ("metric_old_new_mapping.json", "旧新字段映射"), ("metric_repro_check.json", "逐行复现审计"),
        ("source_hashes.json", "来源与代码 hash"), ("prediction_source.json", "checkpoint/数据来源"),
        ("predictions_train_dev_epoch50_100_200.npz", "完整预测轨迹 archive"),
        ("per_episode_seed_vector.csv", "逐 episode/seed 指标"), ("per_pair_seed_vector.csv", "逐 pair/seed 向量指标"),
        ("train_dev_response_summary.csv", "train/dev/epoch 汇总"), ("checkpoint_curves_vector.csv", "学习曲线"),
        ("old_new_numeric_comparison.csv", "旧新数值对比"), ("response_by_history.csv", "history 分层结果"),
        ("sampling_metric_definitions.json", "采样公式"), ("sampling_stability_recomputed.csv", "逐 episode 采样稳定性"),
        ("sampling_old_new_comparison.csv", "采样旧新对比"), ("deflector_event_coverage.csv", "deflector 逐变体事件覆盖"),
        ("deflector_event_coverage_summary.csv", "deflector 事件汇总"), ("deflector_strong_response_coverage.csv", "deflector 强响应覆盖"),
        ("penetration_case_breakdown.csv", "逐 collider 穿透明细"), ("penetration_summary.csv", "穿透汇总与原始分母"),
    ]
    report = []
    report.append("# physVideo Phase 2：seed42 配对指标回算与复现准备（完整合并报告）\n")
    report.append("日期：2026-09-21  \n本文件是本轮唯一的人类可读主报告；CSV/JSON/NPZ 是其可审计附件。旧评测目录不被覆盖。\n")
    report.append("## 1. 执行状态\n")
    report.append(md_table(["项目", "状态/数值"], [
        ["总体", status["status"]], ["seed", str(status["seed"])],
        ["train/dev records", f'{status["train_records"]} / {status["dev_records"]}'],
        ["epochs", ", ".join(map(str, status["epochs"]))],
        ["optimizer updates", str(status["optimizer_updates"])],
        ["data modified", str(status["data_modified"])],
        ["seed43/44", status["seed43_44"]], ["locked-test model metrics", status["locked_test_model_metrics"]],
        ["point eval seeds", ", ".join(map(str, status["point_eval_seeds"]))],
    ]))
    report.append("实际命令：\n\n```text\n" + command + "\n```\n")
    report.append("本轮未训练、未读取 locked-test 做模型预测、未启动 DiT；analytic_cv 仅评测。\n")

    report.append("## 2. 指标定义与数值自检\n")
    report.append("主配对指标固定为同一时间轴、固定 A/B 顺序的逐时刻向量差分：\n\n```text\ntrue_delta[t] = gt_A[t] - gt_B[t]\npred_delta[t] = pred_A[t] - pred_B[t]\nD_gt = mean_t ||true_delta[t]||_2\nD_pred = mean_t ||pred_delta[t]||_2\nE_delta = mean_t ||pred_delta[t] - true_delta[t]||_2\nR_delta = E_delta / D_gt\n```\n\n")
    report.append("幅度列单独保留：`response_magnitude_ratio = D_pred/D_gt`，`response_magnitude_error = |D_pred-D_gt|/D_gt`；等未来 pair 的三项比例均为 N/A，只报告 `D_pred`。\n\n")
    report.append(md_table(["审计项", "结果"], [
        ["selftests", "全部通过" if selftests["all_pass"] else "失败"],
        ["D_gt vs manifest 最大差", fmt_sci(repro["max_abs_D_gt_vs_manifest_m"], 3, " m")],
        ["向量 R 公式最大差", fmt_sci(repro["max_abs_vector_R_formula_difference"], 3)],
        ["旧/新 D_pred 最大差", fmt_sci(repro["max_abs_D_pred_old_new_m"], 3, " m")],
        ["pair/seed 行逐行匹配", f'{repro["matched_rows"]}/{repro["new_rows"]}'],
        ["point 汇总顺序", repro["aggregation_order"]],
    ]))
    report.append("固定自检覆盖：预测=GT、同预测、反向同幅度、响应时刻错位、共同平移、等未来零分母。\n")

    report.append("## 3. train/dev 全量汇总（epoch50/100/200 与 analytic_cv）\n")
    report.append("ADE/FDE/E_delta 使用 mm；速度误差使用 m/s；strong 门为逐 pair `R_delta <= 0.25`。point 的行数按 16 个采样 seed 展开，motion/finite 为确定性单次预测。\n\n")
    report.append(md_table(["epoch", "split", "family", "arm", "ADE", "FDE", "vel MAE", "strong pairs", "strong histories", "strong E", "strong R", "pass", "weak pairs", "weak E", "weak R", "equal pairs", "equal D_pred", "equal pass"], summary_rows(summary)))
    report.append("完整 34 列原始汇总见 " + link(root, "train_dev_response_summary.csv") + "；逐 pair/seed 原始结果见 " + link(root, "per_pair_seed_vector.csv") + "。\n")

    report.append("## 4. epoch200 旧幅度指标与新向量指标对照\n")
    report.append("旧 v4 的 `R_delta/E_delta` 实际对应幅度比/幅度误差；以下同时列出旧幅度与新向量结果。\n\n")
    report.append(md_table(["split", "family", "arm", "rows", "old amp ratio", "old amp error", "new vector E", "new vector R", "new pass", "new amp ratio", "new amp error"], response_rows(oldnew, summary)))
    report.append("\n等未来 pair：\n\n")
    report.append(md_table(["split", "family", "arm", "rows", "old D_pred", "new D_pred"], equal_rows(oldnew)))
    report.append("逐行旧新数值对比见 " + link(root, "old_new_numeric_comparison.csv") + "。\n")

    report.append("## 5. 采样稳定性\n")
    report.append("\n```text\nS_center   = mean_{s,t} ||pred[s,t] - mean_s pred[s,t]||_2\nS_pairwise = mean_{s<s',t} ||pred[s,t] - pred[s',t]||_2\nS_max_pair = max_{s<s'} mean_t ||pred[s,t] - pred[s',t]||_2\nS_seed0    = mean_{s>0,t} ||pred[s,t] - pred[0,t]||_2\n```\n\n")
    report.append(md_table(["split", "family", "episodes", "S_center mean", "S_pairwise mean", "S_max_pair max", "S_seed0 mean", "old/new pairwise diff", "threshold"], sampling_rows(sampling)))
    report.append("旧 `mean_pairwise_m` 与新 `S_pairwise`、旧 `mean_to_seed0_m` 与新 `S_seed0` 逐 episode 一致；`S_center/S_pairwise` 的 5.5 mm 门槛对应关系仍为 `UNRESOLVED`。\n")

    report.append("## 6. deflector 事件与强响应覆盖\n")
    report.append(md_table(["split", "layer", "physical pairs", "histories", "total histories", "coverage", "event categories"], coverage_rows(coverage)))
    report.append("\nmanifest 事件覆盖（epoch200）：\n\n")
    report.append(md_table(["split", "arm", "event category", "episode-seed rows", "unique episodes", "histories", "strong E", "weak E"], event_rows(event_summary)))
    report.append("dev 只有 1 个 physical strong pair / 1 个 history，train 为 11 / 6；manifest 推荐窗口 `0.25--0.90 s` 没有事件，有限 OBB 代理另有少量推荐窗口事件。逐变体时间、目标 collider 和事件后帧数见 " + link(root, "deflector_event_coverage.csv") + "。\n")

    report.append("## 7. aperture 穿透分解与原始分母\n")
    report.append("以下为 epoch200、aperture、GT 与三臂 prediction 的逐 collider 结果；case 分母是 collider-observation，frame 分母为实际输出帧数。\n\n")
    report.append(md_table(["split", "truth", "arm", "collider", "unique episodes", "episode-seed denom", "frame denom", "cases >5mm", "case rate", "frames >5mm", "frame rate", "p95 max", "max depth"], penetration_rows(penetration)))
    report.append("GT 同口径 >5 mm 穿透为 0。完整逐 episode 明细见 " + link(root, "penetration_case_breakdown.csv") + "；汇总及三种原始分母见 " + link(root, "penetration_summary.csv") + "。\n")

    report.append("## 8. 来源、hash 与预测 archive\n")
    report.append("数据、训练 receipt、checkpoint hash 与 CPU/eval forward 记录见 " + link(root, "prediction_source.json") + "。代码和旧 CSV hash 见 " + link(root, "source_hashes.json") + "；字段映射见 " + link(root, "metric_old_new_mapping.json") + "。\n\n")
    report.append(md_table(["来源", "SHA256"], [[name, digest] for name, digest in source_hashes["code_hashes_sha256"].items()] + [[name, digest] for name, digest in source_hashes["old_evaluation_csv_sha256"].items()]))
    report.append("冻结 checkpoint 的完整 train/dev 轨迹 archive：" + link(root, "predictions_train_dev_epoch50_100_200.npz") + "。\n")

    report.append("## 9. 结论与后续边界\n")
    report.append("1. **有限面的三类轨迹收益保留**：seed42 dev 的 ADE/FDE 均低于 motion 和 point，但这是单 seed、开发集、且几何编码器参数量不相同的结果。\n2. **原向量响应仅部分支持**：aperture finite 聚合通过；deflector 只有一个 dev strong pair；support_edge finite 的 `R_delta` 仍高于 0.25。不能写成三类场景均已验证。\n3. **主要风险**：support_edge 强响应、point 的方向/时刻误差、deflector 覆盖不足、finite aperture 穿透率高于 point；弱响应比例受小分母影响，须同时看绝对 `E_delta`。\n4. **复现准备**：可冻结当前训练配置和本版评测器作为 seed43/44 复现基线；这不等于冻结有限面有效性的科学结论。seed43/44 与 locked-test 在本轮明确未运行。\n")
    report.append("## 10. 附件索引\n")
    report.append(md_table(["文件", "用途"], [[link(root, name), purpose] for name, purpose in files]))
    report.append("\n状态分类：`EXECUTED` 为本轮实际完成；`IMPLEMENTED_NOT_RUN` 为后续 seed43/44 与 locked-test；`BLOCKED` 无。\n")
    return "\n".join(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    text = build_report(args.root.resolve())
    (args.root / "reviewed_conclusion.md").write_text(text)
    print(json.dumps({"status": "EXECUTED", "path": str((args.root / "reviewed_conclusion.md").resolve()),
                      "summary_rows": len(read_csv(args.root / "train_dev_response_summary.csv")),
                      "penetration_rows": len(read_csv(args.root / "penetration_summary.csv"))}, indent=2))


if __name__ == "__main__":
    main()
