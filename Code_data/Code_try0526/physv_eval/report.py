from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .datasets import GROUP_SPECS, iter_group_jsons
from .paths import A_OUTPUT, ABC_REPORT_ROOT, DATA_ROOT
from .records import load_payload, metric_value, resolve_video_path

PHYSICSIQ_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark")
PHYSICSIQ_OUTPUT = PHYSICSIQ_ROOT / "output"
PHYSICSIQ_BENCHMARK = "physics-iq-benchmark"
PHYSICSIQ_METHODS = ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]
PHYSICSIQ_METHOD_LABELS = {
    "GT": "GT",
    "wan22-5B-TI2V": "Wan2.2-5B TI2V",
    "VACE_1p3B_TI2V": "VACE 1.3B TI2V",
    "VACE_1p3B_ctx08": "VACE 1.3B ctx=8",
}
PHYGENBENCH_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/phygenbench")
PHYGENBENCH_OUTPUT = PHYGENBENCH_ROOT / "output"
PHYGENBENCH_BENCHMARK = "phygenbench"
PHYGENBENCH_METHODS = ["wan22-5B-TI2V", "VACE_1p3B_TI2V"]
PHYGENBENCH_METHOD_LABELS = {
    "wan22-5B-TI2V": "Wan2.2-5B TI2V",
    "VACE_1p3B_TI2V": "VACE 1.3B TI2V",
}

METRIC_LABELS = {
    "official_pdi": "Official PDI",
    "scale_component": "Scale",
    "traj_component": "Trajectory",
    "epsilon_rigidity": "Rigidity",
    "vp_component": "VP",
    "wmreward_surprise": "WMReward Surprise",
    "cosmos_reason1": "Cosmos Reason1",
    "vjepa_predictive_alignment": "V-JEPA AlignAux",
    "vjepa_temporal_relation_raw_error": "V-JEPA RelRaw",
    "vjepa_delta_relation_raw_error": "V-JEPA DeltaRel",
    "vjepa_delta_profile_error": "V-JEPA DeltaProf",
    "videophy2_auto_sa": "VideoPhy-2 SA",
    "videophy2_auto_pc": "VideoPhy-2 PC",
    "videophy2_auto_joint": "VideoPhy-2 Joint",
}

LOWER_IS_BETTER = {
    "official_pdi",
    "scale_component",
    "traj_component",
    "epsilon_rigidity",
    "vp_component",
    "wmreward_surprise",
    "vjepa_temporal_relation_raw_error",
    "vjepa_delta_relation_raw_error",
    "vjepa_delta_profile_error",
}

GROUP_C_METRICS = [
    "official_pdi",
    "scale_component",
    "traj_component",
    "epsilon_rigidity",
    "vp_component",
    "wmreward_surprise",
    "cosmos_reason1",
    "vjepa_temporal_relation_raw_error",
    "vjepa_delta_relation_raw_error",
    "vjepa_delta_profile_error",
    "videophy2_auto_sa",
    "videophy2_auto_pc",
    "videophy2_auto_joint",
]

REPRESENTATIVE_METRICS = [
    "official_pdi",
    "wmreward_surprise",
    "cosmos_reason1",
    "vjepa_temporal_relation_raw_error",
    "vjepa_delta_relation_raw_error",
    "vjepa_delta_profile_error",
    "videophy2_auto_sa",
    "videophy2_auto_pc",
    "videophy2_auto_joint",
]

REPRESENTATIVE_METRIC_TITLES = {
    "official_pdi": "Official PDI ↓",
    "wmreward_surprise": "WMReward Surprise ↓",
    "cosmos_reason1": "Cosmos ↑",
    "vjepa_temporal_relation_raw_error": "RelRaw ↓",
    "vjepa_delta_relation_raw_error": "DeltaRel ↓",
    "vjepa_delta_profile_error": "DeltaProf ↓",
    "videophy2_auto_sa": "SA ↑",
    "videophy2_auto_pc": "PC ↑",
    "videophy2_auto_joint": "Joint ↑",
}

GROUP_C_GT_OVERRIDES = {
    "bus": A_OUTPUT / "GT" / "Dynamic_Tracking" / "bus.json",
}

PREVIEW_ROOT = ABC_REPORT_ROOT / "preview_videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve unified PhysV ABC report.")
    parser.add_argument("--port", type=int, default=18708)
    return parser.parse_args()


def fv(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def metric_td(name: str, value: Any, *, is_best: bool = False) -> str:
    classes = f"num metric metric-{name}"
    if is_best:
        classes += " best"
    return f"<td class='{classes}'>{fv(value)}</td>"


def text_td(value: Any, classes: str = "") -> str:
    cls = f" class='{classes}'" if classes else ""
    return f"<td{cls}>{value}</td>"


def build_metric_legend() -> str:
    cards = []
    cards.append(_metric_card("official_pdi", "Official PDI", "↓ lower is better", "`metric_results.official_pdi`"))
    cards.append(_metric_card("wmreward_surprise", "WMReward Surprise", "↓ lower is better", "`metric_results.wmreward_jepa.surprise`"))
    cards.append(_metric_card("cosmos_reason1", "Cosmos Reason1", "↑ higher is better", "`metric_results.cosmos_reason1.score`"))
    cards.append(_metric_card("vjepa_temporal_relation_raw_error", "V-JEPA RelRaw", "↓ lower is better", "`metric_results.vjepa_proxy.details.temporal_relation_raw_error`"))
    cards.append(_metric_card("vjepa_delta_relation_raw_error", "V-JEPA DeltaRel", "↓ lower is better", "`metric_results.vjepa_proxy.details.delta_relation_raw_error`"))
    cards.append(_metric_card("vjepa_delta_profile_error", "V-JEPA DeltaProf", "↓ lower is better", "`metric_results.vjepa_proxy.details.delta_profile_error`"))
    cards.append(_metric_card("videophy2_auto_sa", "VideoPhy-2 SA", "↑ higher is better", "`metric_results.videophy2_auto.sa_score`"))
    cards.append(_metric_card("videophy2_auto_pc", "VideoPhy-2 PC", "↑ higher is better", "`metric_results.videophy2_auto.pc_score`"))
    cards.append(_metric_card("videophy2_auto_joint", "VideoPhy-2 Joint", "↑ higher is better", "`1[SA>=4 & PC>=4]`"))
    return f"""
    <section class="legend">
      <h2>指标图例</h2>
      <div class="legend-sub">元数据列与指标列分开渲染；PDI 拆成总分和 4 个子指标；WMReward 保持官方 surprise 口径；Cosmos Reason1 是 cookbook 中 Reason1 physical-plausibility prompt 的 1-5 分；V-JEPA 主看连续原始结构误差。<code>AlignAux</code> 已移到每组下方的折叠诊断区，只作为辅助诊断，不建议单独拿来排方法优劣。Joint 表示 <code>SA&gt;=4 且 PC&gt;=4</code> 的通过率。PhyGround 当前不进主表，原因见报告说明。</div>
      <div class="metric-grid">{''.join(cards)}</div>
    </section>
    """


def _metric_card(name: str, title: str, direction: str, field_name: str) -> str:
    return f"""
      <div class="metric-card metric-{name}">
        <div class="metric-name">{title}</div>
        <div class="metric-dir">{direction}</div>
        <div class="metric-field">{field_name}</div>
      </div>
    """


def section_header(group_id: str, title: str, desc: str) -> str:
    return f"""
    <section class="group-block">
      <div class="group-head">
        <div>
          <div class="group-tag">{group_id}</div>
          <h2>{title}</h2>
          <div class="group-desc">{desc}</div>
        </div>
      </div>
    """


def section_footer() -> str:
    return "</section>"


def standard_table(thead: str, rows: list[str]) -> str:
    return f"<table>{thead}<tbody>{''.join(rows)}</tbody></table>"


def aux_details_table(title: str, rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr>{text_td(label, 'label-cell')}{metric_td('vjepa_predictive_alignment', value)}</tr>"
        for label, value in rows
    )
    return f"""
    <details class="aux-details">
      <summary>{title}</summary>
      <div class="aux-copy"><code>V-JEPA AlignAux</code> 依赖公共子空间对齐，只保留作辅助诊断，不参与主表最优高亮。</div>
      <table class="aux-table">
        <thead>
          <tr>
            <th>Label</th>
            <th class="metric metric-vjepa_predictive_alignment">V-JEPA AlignAux ↑</th>
          </tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </details>
    """


def best_metric_mask(rows: list[dict[str, Any]], metric_names: list[str]) -> list[dict[str, bool]]:
    best: dict[str, float] = {}
    for metric_name in metric_names:
        values = [row.get(metric_name) for row in rows if row.get(metric_name) is not None]
        if not values:
            continue
        best[metric_name] = min(values) if metric_name in LOWER_IS_BETTER else max(values)

    masks: list[dict[str, bool]] = []
    for row in rows:
        row_mask: dict[str, bool] = {}
        for metric_name in metric_names:
            value = row.get(metric_name)
            row_mask[metric_name] = value is not None and metric_name in best and value == best[metric_name]
        masks.append(row_mask)
    return masks


def build_group_a() -> str:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for json_path in iter_group_jsons("A"):
        payload = load_payload(json_path)
        method = str(payload.get("method") or payload.get("provider") or json_path.parts[-3])
        by_method[method].append(payload)

    row_data = []
    for method in sorted(by_method):
        payloads = by_method[method]
        row_data.append(
            {
                "method": method,
                "count": len(payloads),
                "official_pdi": mean_or_none(_metric_list(payloads, "official_pdi")),
                "scale_component": mean_or_none(_metric_list(payloads, "scale_component")),
                "traj_component": mean_or_none(_metric_list(payloads, "traj_component")),
                "epsilon_rigidity": mean_or_none(_metric_list(payloads, "epsilon_rigidity")),
                "vp_component": mean_or_none(_metric_list(payloads, "vp_component")),
                "wmreward_surprise": mean_or_none(_metric_list(payloads, "wmreward_surprise")),
                "cosmos_reason1": mean_or_none(_metric_list(payloads, "cosmos_reason1")),
                "vjepa_temporal_relation_raw_error": mean_or_none(_metric_list(payloads, "vjepa_temporal_relation_raw_error")),
                "vjepa_delta_relation_raw_error": mean_or_none(_metric_list(payloads, "vjepa_delta_relation_raw_error")),
                "vjepa_delta_profile_error": mean_or_none(_metric_list(payloads, "vjepa_delta_profile_error")),
                "videophy2_auto_sa": mean_or_none(_metric_list(payloads, "videophy2_auto_sa")),
                "videophy2_auto_pc": mean_or_none(_metric_list(payloads, "videophy2_auto_pc")),
                "videophy2_auto_joint": mean_or_none(_metric_list(payloads, "videophy2_auto_joint")),
            }
        )
    masks = best_metric_mask(row_data, GROUP_C_METRICS)

    rows = []
    for row, mask in zip(row_data, masks):
        rows.append(
            "<tr>"
            f"{text_td(row['method'], 'label-cell')}"
            f"{text_td(row['count'], 'num')}"
            f"{metric_td('official_pdi', row['official_pdi'], is_best=mask['official_pdi'])}"
            f"{metric_td('scale_component', row['scale_component'], is_best=mask['scale_component'])}"
            f"{metric_td('traj_component', row['traj_component'], is_best=mask['traj_component'])}"
            f"{metric_td('epsilon_rigidity', row['epsilon_rigidity'], is_best=mask['epsilon_rigidity'])}"
            f"{metric_td('vp_component', row['vp_component'], is_best=mask['vp_component'])}"
            f"{metric_td('wmreward_surprise', row['wmreward_surprise'], is_best=mask['wmreward_surprise'])}"
            f"{metric_td('cosmos_reason1', row['cosmos_reason1'], is_best=mask['cosmos_reason1'])}"
            f"{metric_td('vjepa_temporal_relation_raw_error', row['vjepa_temporal_relation_raw_error'], is_best=mask['vjepa_temporal_relation_raw_error'])}"
            f"{metric_td('vjepa_delta_relation_raw_error', row['vjepa_delta_relation_raw_error'], is_best=mask['vjepa_delta_relation_raw_error'])}"
            f"{metric_td('vjepa_delta_profile_error', row['vjepa_delta_profile_error'], is_best=mask['vjepa_delta_profile_error'])}"
            f"{metric_td('videophy2_auto_sa', row['videophy2_auto_sa'], is_best=mask['videophy2_auto_sa'])}"
            f"{metric_td('videophy2_auto_pc', row['videophy2_auto_pc'], is_best=mask['videophy2_auto_pc'])}"
            f"{metric_td('videophy2_auto_joint', row['videophy2_auto_joint'], is_best=mask['videophy2_auto_joint'])}"
            "</tr>"
        )

    thead = """
    <thead>
      <tr>
        <th colspan="2">Method Metadata</th>
        <th colspan="5">Official PDI Breakdown</th>
        <th colspan="8">Predictive Metrics</th>
      </tr>
      <tr>
        <th>Method</th>
        <th>N</th>
        <th class="metric metric-official_pdi">Official PDI ↓</th>
        <th class="metric metric-scale_component">Scale ↓</th>
        <th class="metric metric-traj_component">Trajectory ↓</th>
        <th class="metric metric-epsilon_rigidity">Rigidity ↓</th>
        <th class="metric metric-vp_component">VP ↓</th>
        <th class="metric metric-wmreward_surprise">WMReward Surprise ↓</th>
        <th class="metric metric-cosmos_reason1">Cosmos ↑</th>
        <th class="metric metric-vjepa_temporal_relation_raw_error">V-JEPA RelRaw ↓</th>
        <th class="metric metric-vjepa_delta_relation_raw_error">V-JEPA DeltaRel ↓</th>
        <th class="metric metric-vjepa_delta_profile_error">V-JEPA DeltaProf ↓</th>
        <th class="metric metric-videophy2_auto_sa">VideoPhy-2 SA ↑</th>
        <th class="metric metric-videophy2_auto_pc">VideoPhy-2 PC ↑</th>
        <th class="metric metric-videophy2_auto_joint">VideoPhy-2 Joint ↑</th>
      </tr>
    </thead>
    """
    aux_rows = [(row["method"], mean_or_none(_metric_list(by_method[row["method"]], "vjepa_predictive_alignment"))) for row in row_data]
    return (
        section_header("A", GROUP_SPECS["A"].title, GROUP_SPECS["A"].description)
        + standard_table(thead, rows)
        + aux_details_table("展开查看 A 组 AlignAux 诊断", aux_rows)
        + section_footer()
    )


def build_group_d() -> str:
    by_method: dict[str, list[dict[str, Any]]] = {}
    gt_paths = sorted((PHYSICSIQ_OUTPUT / "GT" / PHYSICSIQ_BENCHMARK).glob("*.json"))
    gt_sample_ids = {path.stem for path in gt_paths}
    full_compare_count = 0
    if gt_sample_ids:
        full_compare_count = sum(
            1
            for sample_id in gt_sample_ids
            if all((PHYSICSIQ_OUTPUT / method / PHYSICSIQ_BENCHMARK / f"{sample_id}.json").is_file() for method in PHYSICSIQ_METHODS)
        )

    for method in PHYSICSIQ_METHODS:
        method_paths = sorted((PHYSICSIQ_OUTPUT / method / PHYSICSIQ_BENCHMARK).glob("*.json"))
        by_method[method] = [load_payload(path) for path in method_paths]

    row_data = []
    for method in PHYSICSIQ_METHODS:
        payloads = by_method.get(method, [])
        row_data.append(
            {
                "method": PHYSICSIQ_METHOD_LABELS[method],
                "count": len(payloads),
                "official_pdi": mean_or_none(_metric_list(payloads, "official_pdi")),
                "scale_component": mean_or_none(_metric_list(payloads, "scale_component")),
                "traj_component": mean_or_none(_metric_list(payloads, "traj_component")),
                "epsilon_rigidity": mean_or_none(_metric_list(payloads, "epsilon_rigidity")),
                "vp_component": mean_or_none(_metric_list(payloads, "vp_component")),
                "wmreward_surprise": mean_or_none(_metric_list(payloads, "wmreward_surprise")),
                "cosmos_reason1": mean_or_none(_metric_list(payloads, "cosmos_reason1")),
                "vjepa_temporal_relation_raw_error": mean_or_none(_metric_list(payloads, "vjepa_temporal_relation_raw_error")),
                "vjepa_delta_relation_raw_error": mean_or_none(_metric_list(payloads, "vjepa_delta_relation_raw_error")),
                "vjepa_delta_profile_error": mean_or_none(_metric_list(payloads, "vjepa_delta_profile_error")),
                "videophy2_auto_sa": mean_or_none(_metric_list(payloads, "videophy2_auto_sa")),
                "videophy2_auto_pc": mean_or_none(_metric_list(payloads, "videophy2_auto_pc")),
                "videophy2_auto_joint": mean_or_none(_metric_list(payloads, "videophy2_auto_joint")),
            }
        )
    masks = best_metric_mask(row_data, GROUP_C_METRICS)

    rows = []
    for row, mask in zip(row_data, masks):
        rows.append(
            "<tr>"
            f"{text_td(row['method'], 'label-cell')}"
            f"{text_td(row['count'], 'num')}"
            f"{render_metric_cells(row, mask)}"
            "</tr>"
        )

    thead = """
    <thead>
      <tr>
        <th colspan="2">Method Metadata</th>
        <th colspan="5">Official PDI Breakdown</th>
        <th colspan="8">Predictive Metrics</th>
      </tr>
      <tr>
        <th>Method</th>
        <th>N</th>
        <th class="metric metric-official_pdi">Official PDI ↓</th>
        <th class="metric metric-scale_component">Scale ↓</th>
        <th class="metric metric-traj_component">Trajectory ↓</th>
        <th class="metric metric-epsilon_rigidity">Rigidity ↓</th>
        <th class="metric metric-vp_component">VP ↓</th>
        <th class="metric metric-wmreward_surprise">WMReward Surprise ↓</th>
        <th class="metric metric-cosmos_reason1">Cosmos ↑</th>
        <th class="metric metric-vjepa_temporal_relation_raw_error">V-JEPA RelRaw ↓</th>
        <th class="metric metric-vjepa_delta_relation_raw_error">V-JEPA DeltaRel ↓</th>
        <th class="metric metric-vjepa_delta_profile_error">V-JEPA DeltaProf ↓</th>
        <th class="metric metric-videophy2_auto_sa">VideoPhy-2 SA ↑</th>
        <th class="metric metric-videophy2_auto_pc">VideoPhy-2 PC ↑</th>
        <th class="metric metric-videophy2_auto_joint">VideoPhy-2 Joint ↑</th>
      </tr>
    </thead>
    """
    aux_rows = [
        (PHYSICSIQ_METHOD_LABELS[method], mean_or_none(_metric_list(by_method[method], "vjepa_predictive_alignment")))
        for method in PHYSICSIQ_METHODS
    ]
    description = (
        f"Physics-IQ 多物理现象 benchmark。当前 GT 共 {len(gt_sample_ids)} 个 case，"
        f"其中 {full_compare_count} 个 case 已具备 4-way 方法对比；本组直接复用主表指标做方法级汇总。"
    )
    return (
        section_header("D", "Physics-IQ", description)
        + standard_table(thead, rows)
        + aux_details_table("展开查看 D 组 AlignAux 诊断", aux_rows)
        + section_footer()
    )


def build_group_e() -> str:
    by_method: dict[str, list[dict[str, Any]]] = {}
    sample_sets: list[set[str]] = []
    for method in PHYGENBENCH_METHODS:
        method_paths = sorted((PHYGENBENCH_OUTPUT / method / PHYGENBENCH_BENCHMARK).glob("*.json"))
        by_method[method] = [load_payload(path) for path in method_paths]
        sample_sets.append({path.stem for path in method_paths})

    full_compare_count = len(set.intersection(*sample_sets)) if sample_sets else 0

    row_data = []
    for method in PHYGENBENCH_METHODS:
        payloads = by_method.get(method, [])
        row_data.append(
            {
                "method": PHYGENBENCH_METHOD_LABELS[method],
                "count": len(payloads),
                "official_pdi": mean_or_none(_metric_list(payloads, "official_pdi")),
                "scale_component": mean_or_none(_metric_list(payloads, "scale_component")),
                "traj_component": mean_or_none(_metric_list(payloads, "traj_component")),
                "epsilon_rigidity": mean_or_none(_metric_list(payloads, "epsilon_rigidity")),
                "vp_component": mean_or_none(_metric_list(payloads, "vp_component")),
                "wmreward_surprise": mean_or_none(_metric_list(payloads, "wmreward_surprise")),
                "cosmos_reason1": mean_or_none(_metric_list(payloads, "cosmos_reason1")),
                "vjepa_temporal_relation_raw_error": mean_or_none(_metric_list(payloads, "vjepa_temporal_relation_raw_error")),
                "vjepa_delta_relation_raw_error": mean_or_none(_metric_list(payloads, "vjepa_delta_relation_raw_error")),
                "vjepa_delta_profile_error": mean_or_none(_metric_list(payloads, "vjepa_delta_profile_error")),
                "videophy2_auto_sa": mean_or_none(_metric_list(payloads, "videophy2_auto_sa")),
                "videophy2_auto_pc": mean_or_none(_metric_list(payloads, "videophy2_auto_pc")),
                "videophy2_auto_joint": mean_or_none(_metric_list(payloads, "videophy2_auto_joint")),
            }
        )
    masks = best_metric_mask(row_data, GROUP_C_METRICS)

    rows = []
    for row, mask in zip(row_data, masks):
        rows.append(
            "<tr>"
            f"{text_td(row['method'], 'label-cell')}"
            f"{text_td(row['count'], 'num')}"
            f"{render_metric_cells(row, mask)}"
            "</tr>"
        )

    thead = """
    <thead>
      <tr>
        <th colspan="2">Method Metadata</th>
        <th colspan="5">Official PDI Breakdown</th>
        <th colspan="8">Predictive Metrics</th>
      </tr>
      <tr>
        <th>Method</th>
        <th>N</th>
        <th class="metric metric-official_pdi">Official PDI ↓</th>
        <th class="metric metric-scale_component">Scale ↓</th>
        <th class="metric metric-traj_component">Trajectory ↓</th>
        <th class="metric metric-epsilon_rigidity">Rigidity ↓</th>
        <th class="metric metric-vp_component">VP ↓</th>
        <th class="metric metric-wmreward_surprise">WMReward Surprise ↓</th>
        <th class="metric metric-cosmos_reason1">Cosmos ↑</th>
        <th class="metric metric-vjepa_temporal_relation_raw_error">V-JEPA RelRaw ↓</th>
        <th class="metric metric-vjepa_delta_relation_raw_error">V-JEPA DeltaRel ↓</th>
        <th class="metric metric-vjepa_delta_profile_error">V-JEPA DeltaProf ↓</th>
        <th class="metric metric-videophy2_auto_sa">VideoPhy-2 SA ↑</th>
        <th class="metric metric-videophy2_auto_pc">VideoPhy-2 PC ↑</th>
        <th class="metric metric-videophy2_auto_joint">VideoPhy-2 Joint ↑</th>
      </tr>
    </thead>
    """
    aux_rows = [
        (PHYGENBENCH_METHOD_LABELS[method], mean_or_none(_metric_list(by_method[method], "vjepa_predictive_alignment")))
        for method in PHYGENBENCH_METHODS
    ]
    description = (
        f"PhyGenBench 物理视频生成 benchmark。当前只纳入已经完整生成且可本地复现的两条 TI2V 链路，"
        f"共 {full_compare_count} 个 case 可直接做方法级对比；FLUX 首帧和未完成的 ctx08 不作为主表可比方法。"
    )
    return (
        section_header("E", "PhyGenBench", description)
        + standard_table(thead, rows)
        + aux_details_table("展开查看 E 组 AlignAux 诊断", aux_rows)
        + section_footer()
    )


def build_group_b1() -> str:
    row_data = []
    for json_path in iter_group_jsons("B1"):
        payload = load_payload(json_path)
        params = payload.get("parameters", {})
        row_data.append(
            {
                "label": json_path.stem,
                "meta": [params.get("restitution", "-"), params.get("lateral_friction", "-"), params.get("ball_mass_kg", "-")],
                **metric_values(payload),
            }
        )
    return _sample_group_section("B1", "Scenario", ["e", "μ", "m"], row_data)


def build_group_b2() -> str:
    row_data = []
    for json_path in iter_group_jsons("B2"):
        payload = load_payload(json_path)
        row_data.append(
            {
                "label": json_path.stem,
                "meta": [payload.get("description", payload.get("experiment", "-"))],
                **metric_values(payload),
            }
        )
    return _sample_group_section("B2", "Scenario", ["Description"], row_data)


def build_group_b3() -> str:
    row_data = []
    for json_path in iter_group_jsons("B3"):
        payload = load_payload(json_path)
        row_data.append(
            {
                "label": payload.get("scenario", json_path.stem),
                "meta": [payload.get("appearance_variant", "-")],
                **metric_values(payload),
            }
        )
    return _sample_group_section("B3", "Base Scenario", ["Appearance"], row_data)


def build_group_c() -> str:
    shuffle_payloads: list[dict[str, Any]] = []
    noshuffle_payloads: list[dict[str, Any]] = []
    originals_in_c = {json_path.stem: json_path for json_path in iter_group_jsons("C") if json_path.stem.endswith("_original")}

    for json_path in iter_group_jsons("C"):
        if not json_path.stem.endswith("_shuffled"):
            continue
        shuffle_payloads.append(load_payload(json_path))
        noshuffle_payloads.append(load_payload(_resolve_group_c_original_path(json_path, originals_in_c)))

    row_data = [
        _aggregate_metric_row("noshuffle", noshuffle_payloads),
        _aggregate_metric_row("shuffle", shuffle_payloads),
    ]
    masks = best_metric_mask(row_data, GROUP_C_METRICS)

    rows = []
    for row, mask in zip(row_data, masks):
        rows.append(
            "<tr>"
            f"{text_td(row['method'], 'label-cell')}"
            f"{text_td(row['count'], 'num')}"
            f"{render_metric_cells(row, mask)}"
            "</tr>"
        )

    thead = """
    <thead>
      <tr>
        <th colspan="2">Method Metadata</th>
        <th colspan="5">Official PDI Breakdown</th>
        <th colspan="8">Predictive Metrics</th>
      </tr>
      <tr>
        <th>Method</th>
        <th>N</th>
        <th class="metric metric-official_pdi">Official PDI ↓</th>
        <th class="metric metric-scale_component">Scale ↓</th>
        <th class="metric metric-traj_component">Trajectory ↓</th>
        <th class="metric metric-epsilon_rigidity">Rigidity ↓</th>
        <th class="metric metric-vp_component">VP ↓</th>
        <th class="metric metric-wmreward_surprise">WMReward Surprise ↓</th>
        <th class="metric metric-cosmos_reason1">Cosmos ↑</th>
        <th class="metric metric-vjepa_temporal_relation_raw_error">V-JEPA RelRaw ↓</th>
        <th class="metric metric-vjepa_delta_relation_raw_error">V-JEPA DeltaRel ↓</th>
        <th class="metric metric-vjepa_delta_profile_error">V-JEPA DeltaProf ↓</th>
        <th class="metric metric-videophy2_auto_sa">VideoPhy-2 SA ↑</th>
        <th class="metric metric-videophy2_auto_pc">VideoPhy-2 PC ↑</th>
        <th class="metric metric-videophy2_auto_joint">VideoPhy-2 Joint ↑</th>
      </tr>
    </thead>
    """
    spec = GROUP_SPECS["C"]
    description = (
        f"{spec.description} 当前按 `shuffle` / `noshuffle` 两种方法聚合显示均值；"
        " `noshuffle` 优先使用 C 组显式 original，其余样本回溯到对应 A/B 原始结果。"
    )
    aux_rows = [
        ("noshuffle", mean_or_none(_metric_list(noshuffle_payloads, "vjepa_predictive_alignment"))),
        ("shuffle", mean_or_none(_metric_list(shuffle_payloads, "vjepa_predictive_alignment"))),
    ]
    return (
        section_header("C", spec.title, description)
        + standard_table(thead, rows)
        + aux_details_table("展开查看 C 组 AlignAux 诊断", aux_rows)
        + section_footer()
    )


def metric_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "official_pdi": metric_value(payload, "official_pdi"),
        "scale_component": metric_value(payload, "scale_component"),
        "traj_component": metric_value(payload, "traj_component"),
        "epsilon_rigidity": metric_value(payload, "epsilon_rigidity"),
        "vp_component": metric_value(payload, "vp_component"),
        "wmreward_surprise": metric_value(payload, "wmreward_surprise"),
        "cosmos_reason1": metric_value(payload, "cosmos_reason1"),
        "vjepa_temporal_relation_raw_error": metric_value(payload, "vjepa_temporal_relation_raw_error"),
        "vjepa_delta_relation_raw_error": metric_value(payload, "vjepa_delta_relation_raw_error"),
        "vjepa_delta_profile_error": metric_value(payload, "vjepa_delta_profile_error"),
        "videophy2_auto_sa": metric_value(payload, "videophy2_auto_sa"),
        "videophy2_auto_pc": metric_value(payload, "videophy2_auto_pc"),
        "videophy2_auto_joint": metric_value(payload, "videophy2_auto_joint"),
    }


def render_metric_cells(row: dict[str, Any], mask: dict[str, bool]) -> str:
    return "".join(
        [
            metric_td("official_pdi", row["official_pdi"], is_best=mask["official_pdi"]),
            metric_td("scale_component", row["scale_component"], is_best=mask["scale_component"]),
            metric_td("traj_component", row["traj_component"], is_best=mask["traj_component"]),
            metric_td("epsilon_rigidity", row["epsilon_rigidity"], is_best=mask["epsilon_rigidity"]),
            metric_td("vp_component", row["vp_component"], is_best=mask["vp_component"]),
            metric_td("wmreward_surprise", row["wmreward_surprise"], is_best=mask["wmreward_surprise"]),
            metric_td("cosmos_reason1", row["cosmos_reason1"], is_best=mask["cosmos_reason1"]),
            metric_td("vjepa_temporal_relation_raw_error", row["vjepa_temporal_relation_raw_error"], is_best=mask["vjepa_temporal_relation_raw_error"]),
            metric_td("vjepa_delta_relation_raw_error", row["vjepa_delta_relation_raw_error"], is_best=mask["vjepa_delta_relation_raw_error"]),
            metric_td("vjepa_delta_profile_error", row["vjepa_delta_profile_error"], is_best=mask["vjepa_delta_profile_error"]),
            metric_td("videophy2_auto_sa", row["videophy2_auto_sa"], is_best=mask["videophy2_auto_sa"]),
            metric_td("videophy2_auto_pc", row["videophy2_auto_pc"], is_best=mask["videophy2_auto_pc"]),
            metric_td("videophy2_auto_joint", row["videophy2_auto_joint"], is_best=mask["videophy2_auto_joint"]),
        ]
    )


def _sample_group_section(group_id: str, label1: str, extra_headers: list[str], row_data: list[dict[str, Any]]) -> str:
    meta_colspan = 1 + len(extra_headers)
    meta_headers = [f"<th>{label1}</th>"] + [f"<th>{header}</th>" for header in extra_headers]
    masks = best_metric_mask(row_data, GROUP_C_METRICS)
    rows = []
    for row, mask in zip(row_data, masks):
        meta_cells = "".join(
            [text_td(row["label"], "label-cell")] + [text_td(value, "num" if isinstance(value, (int, float)) else "") for value in row["meta"]]
        )
        rows.append("<tr>" + meta_cells + render_metric_cells(row, mask) + "</tr>")

    thead = f"""
    <thead>
      <tr>
        <th colspan="{meta_colspan}">Sample Metadata</th>
        <th colspan="5">Official PDI Breakdown</th>
        <th colspan="8">Predictive Metrics</th>
      </tr>
      <tr>
        {''.join(meta_headers)}
        <th class="metric metric-official_pdi">Official PDI ↓</th>
        <th class="metric metric-scale_component">Scale ↓</th>
        <th class="metric metric-traj_component">Trajectory ↓</th>
        <th class="metric metric-epsilon_rigidity">Rigidity ↓</th>
        <th class="metric metric-vp_component">VP ↓</th>
        <th class="metric metric-wmreward_surprise">WMReward Surprise ↓</th>
        <th class="metric metric-cosmos_reason1">Cosmos ↑</th>
        <th class="metric metric-vjepa_temporal_relation_raw_error">V-JEPA RelRaw ↓</th>
        <th class="metric metric-vjepa_delta_relation_raw_error">V-JEPA DeltaRel ↓</th>
        <th class="metric metric-vjepa_delta_profile_error">V-JEPA DeltaProf ↓</th>
        <th class="metric metric-videophy2_auto_sa">VideoPhy-2 SA ↑</th>
        <th class="metric metric-videophy2_auto_pc">VideoPhy-2 PC ↑</th>
        <th class="metric metric-videophy2_auto_joint">VideoPhy-2 Joint ↑</th>
      </tr>
    </thead>
    """
    spec = GROUP_SPECS[group_id]
    aux_rows = [(row["label"], metric_value(payload, "vjepa_predictive_alignment")) for row, payload in zip(row_data, [load_payload(p) for p in iter_group_jsons(group_id)])]
    return (
        section_header(group_id, spec.title, spec.description)
        + standard_table(thead, rows)
        + aux_details_table(f"展开查看 {group_id} 组 AlignAux 诊断", aux_rows)
        + section_footer()
    )


def _metric_list(payloads: list[dict[str, Any]], name: str) -> list[float]:
    values = [metric_value(payload, name) for payload in payloads]
    return [float(value) for value in values if value is not None]


def _aggregate_metric_row(method: str, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": method,
        "count": len(payloads),
        "official_pdi": mean_or_none(_metric_list(payloads, "official_pdi")),
        "scale_component": mean_or_none(_metric_list(payloads, "scale_component")),
        "traj_component": mean_or_none(_metric_list(payloads, "traj_component")),
        "epsilon_rigidity": mean_or_none(_metric_list(payloads, "epsilon_rigidity")),
        "vp_component": mean_or_none(_metric_list(payloads, "vp_component")),
        "wmreward_surprise": mean_or_none(_metric_list(payloads, "wmreward_surprise")),
        "cosmos_reason1": mean_or_none(_metric_list(payloads, "cosmos_reason1")),
        "vjepa_temporal_relation_raw_error": mean_or_none(_metric_list(payloads, "vjepa_temporal_relation_raw_error")),
        "vjepa_delta_relation_raw_error": mean_or_none(_metric_list(payloads, "vjepa_delta_relation_raw_error")),
        "vjepa_delta_profile_error": mean_or_none(_metric_list(payloads, "vjepa_delta_profile_error")),
        "videophy2_auto_sa": mean_or_none(_metric_list(payloads, "videophy2_auto_sa")),
        "videophy2_auto_pc": mean_or_none(_metric_list(payloads, "videophy2_auto_pc")),
        "videophy2_auto_joint": mean_or_none(_metric_list(payloads, "videophy2_auto_joint")),
    }


def _resolve_group_c_original_path(shuffled_path: Path, originals_in_c: dict[str, Path]) -> Path:
    stem = shuffled_path.stem
    if not stem.endswith("_shuffled"):
        raise ValueError(f"Expected shuffled C sample, got: {shuffled_path}")

    prefix, sample_key = stem.split("_", 1)
    sample_key = sample_key[: -len("_shuffled")]

    explicit_original = originals_in_c.get(f"{prefix}_{sample_key}_original")
    if explicit_original is not None:
        return explicit_original

    if prefix == "gt":
        return _resolve_group_c_gt_original_path(sample_key)
    if prefix == "sim":
        return _resolve_group_c_sim_original_path(sample_key)
    raise ValueError(f"Unsupported C sample prefix in {shuffled_path.name}")


def _resolve_group_c_gt_original_path(sample_key: str) -> Path:
    override = GROUP_C_GT_OVERRIDES.get(sample_key)
    if override is not None:
        return override

    matches = sorted((A_OUTPUT / "GT").rglob(f"{sample_key}.json"))
    if not matches:
        raise FileNotFoundError(f"No GT original json found for C sample {sample_key!r}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous GT original json for C sample {sample_key!r}: {matches}")
    return matches[0]


def _resolve_group_c_sim_original_path(sample_key: str) -> Path:
    for candidate in [
        DATA_ROOT / "videos" / "ball_block" / f"{sample_key}.json",
        DATA_ROOT / "videos" / "jepa_sensitivity" / f"{sample_key}.json",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No sim original json found for C sample {sample_key!r}")


def build_representative_samples() -> str:
    physicsiq_case = _first_physicsiq_full_compare_case()
    phygenbench_case = _first_phygenbench_full_compare_case()
    case_specs = [
        {
            "group": "A",
            "title": "A 组方法对比: Dynamic Tracking / bus",
            "description": "同一提示和场景下对比 GT、VACE 和 Wan，适合直观看方法级差异，以及哪些指标与主表排序一致。",
            "samples": [
                {"label": "GT", "json_path": A_OUTPUT / "GT" / "Dynamic_Tracking" / "bus.json"},
                {"label": "VACE TI2V", "json_path": A_OUTPUT / "VACE_1p3B_TI2V" / "Dynamic_Tracking" / "bus.json"},
                {"label": "VACE ctx08", "json_path": A_OUTPUT / "VACE_1p3B_ctx08" / "Dynamic_Tracking" / "bus.json"},
                {"label": "Wan 2.2-5B", "json_path": A_OUTPUT / "wan22-5B-TI2V" / "Dynamic_Tracking" / "bus.json"},
            ],
        },
        {
            "group": "B1",
            "title": "B1 组物理参数: 基准质量 vs 极轻球",
            "description": "只改球质量，比较基准样本和极轻球样本，适合看 Official PDI 与预测类指标对明显动力学变化的响应。",
            "samples": [
                {"label": "baseline e07_mu05_m1", "json_path": DATA_ROOT / "videos" / "ball_block" / "e07_mu05_m1.json"},
                {"label": "light ball e07_mu05_m01", "json_path": DATA_ROOT / "videos" / "ball_block" / "e07_mu05_m01.json"},
            ],
        },
        {
            "group": "B2",
            "title": "B2 组运动敏感性: 重力从月球到超重",
            "description": "固定外观和碰撞设置，只改重力 4.9 / 9.81 / 19.6 m/s²，适合看指标是否能响应连续运动速度轮廓变化。",
            "samples": [
                {"label": "grav_050", "json_path": DATA_ROOT / "videos" / "jepa_sensitivity" / "grav_050.json"},
                {"label": "grav_098", "json_path": DATA_ROOT / "videos" / "jepa_sensitivity" / "grav_098.json"},
                {"label": "grav_200", "json_path": DATA_ROOT / "videos" / "jepa_sensitivity" / "grav_200.json"},
            ],
        },
        {
            "group": "B3",
            "title": "B3 组外观敏感性: 同一轨迹，不同渲染",
            "description": "完全相同的物理轨迹，只换外观风格和光照，适合看哪些指标仍然受 appearance 干扰。",
            "samples": [
                {"label": "default", "json_path": DATA_ROOT / "videos" / "ball_block_appearance" / "e07_mu05_m1_v1_default.json"},
                {"label": "dark blue", "json_path": DATA_ROOT / "videos" / "ball_block_appearance" / "e07_mu05_m1_v2_dark_blue.json"},
                {"label": "warm bright", "json_path": DATA_ROOT / "videos" / "ball_block_appearance" / "e07_mu05_m1_v3_warm_bright.json"},
            ],
        },
        {
            "group": "C",
            "title": "C 组时序打乱: 原始 vs shuffled",
            "description": "同样的单帧内容，只打乱帧顺序。这个 case 最适合直接看时序相关指标是否真在响应顺序破坏。",
            "samples": [
                {"label": "original", "json_path": _resolve_group_c_original_path(DATA_ROOT / "videos" / "shuffle_test" / "gt_bus_shuffled.json", {})},
                {"label": "shuffled", "json_path": DATA_ROOT / "videos" / "shuffle_test" / "gt_bus_shuffled.json"},
            ],
        },
    ]
    if physicsiq_case is not None:
        case_specs.append(
            {
                "group": "D",
                "title": f"D 组方法对比: Physics-IQ / {physicsiq_case}",
                "description": "同一个 Physics-IQ case 下并排放 GT、Wan、VACE TI2V 和 VACE ctx08，方便直接查看 D 组视频与指标是否一致。",
                "samples": [
                    {"label": PHYSICSIQ_METHOD_LABELS[method], "json_path": PHYSICSIQ_OUTPUT / method / PHYSICSIQ_BENCHMARK / f"{physicsiq_case}.json"}
                    for method in PHYSICSIQ_METHODS
                ],
            }
        )
    if phygenbench_case is not None:
        case_specs.append(
            {
                "group": "E",
                "title": f"E 组方法对比: PhyGenBench / {phygenbench_case}",
                "description": "同一个 PhyGenBench case 下并排放 Wan 和 VACE TI2V，方便看这套开放链路在物理 plausibility 指标上的相对排序。",
                "samples": [
                    {
                        "label": PHYGENBENCH_METHOD_LABELS[method],
                        "json_path": PHYGENBENCH_OUTPUT / method / PHYGENBENCH_BENCHMARK / f"{phygenbench_case}.json",
                    }
                    for method in PHYGENBENCH_METHODS
                ],
            }
        )
    cards = "".join(_representative_case_card(case_spec) for case_spec in case_specs)
    return f"""
    <section class="samples-section">
      <div class="group-head">
        <div>
          <div class="group-tag">Samples</div>
          <h2>代表性样本</h2>
          <div class="group-desc">这里不再看组均值，而是挑选几组最有解释力的具体视频，直接把视频和对应指标并排放在一起，方便判断某个指标到底在“看什么”。</div>
        </div>
      </div>
      <div class="sample-case-list">{cards}</div>
    </section>
    """


def _representative_case_card(case_spec: dict[str, Any]) -> str:
    sample_rows = [_representative_sample_row(sample_spec["label"], sample_spec["json_path"]) for sample_spec in case_spec["samples"]]
    masks = best_metric_mask(sample_rows, REPRESENTATIVE_METRICS)
    video_cards = "".join(_representative_video_card(row) for row in sample_rows)
    metric_rows = []
    for row, mask in zip(sample_rows, masks):
        metric_rows.append(
            "<tr>"
            f"{text_td(row['label'], 'label-cell')}"
            + "".join(metric_td(name, row[name], is_best=mask[name]) for name in REPRESENTATIVE_METRICS)
            + "</tr>"
        )
    headers = "".join(
        f"<th class='metric metric-{name}'>{REPRESENTATIVE_METRIC_TITLES[name]}</th>"
        for name in REPRESENTATIVE_METRICS
    )
    return f"""
    <article class="sample-case">
      <div class="sample-case-head">
        <div class="sample-case-tag">{case_spec['group']}</div>
        <h3>{case_spec['title']}</h3>
        <p>{case_spec['description']}</p>
      </div>
      <div class="sample-video-grid">{video_cards}</div>
      <div class="sample-table-wrap">
        <table class="sample-metrics-table">
          <thead>
            <tr>
              <th>Sample</th>
              {headers}
            </tr>
          </thead>
          <tbody>{''.join(metric_rows)}</tbody>
        </table>
      </div>
    </article>
    """


def _representative_sample_row(label: str, json_path: Path) -> dict[str, Any]:
    payload = load_payload(json_path)
    video_path = resolve_video_path(json_path, payload)
    row = {
        "label": label,
        "video_url": ensure_preview_video(video_path),
        "caption": payload.get("description") or payload.get("caption") or payload.get("scenario") or payload.get("experiment") or json_path.stem,
        "json_name": json_path.name,
    }
    row.update(metric_values(payload))
    return row


def _representative_video_card(row: dict[str, Any]) -> str:
    return f"""
      <figure class="sample-video-card">
        <div class="sample-video-label">{row['label']}</div>
        <video controls muted playsinline preload="metadata" src="{row['video_url']}"></video>
        <figcaption>
          <div class="sample-caption">{row['caption']}</div>
          <div class="sample-json">{row['json_name']}</div>
        </figcaption>
      </figure>
    """


def _video_url(video_path: Path) -> str:
    resolved = video_path.resolve()
    roots = [
        ("dataset_videos", (DATA_ROOT / "videos").resolve()),
        ("pdi_output", A_OUTPUT.resolve()),
        ("physicsiq_output", PHYSICSIQ_OUTPUT.resolve()),
        ("phygenbench_output", PHYGENBENCH_OUTPUT.resolve()),
    ]
    for prefix, root in roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return f"{prefix}/{relative.as_posix()}"
    raise ValueError(f"Video path is outside report static roots: {video_path}")


def ensure_preview_video(video_path: Path) -> str:
    source = video_path.resolve()
    relative_key = _video_url(source)
    preview_path = PREVIEW_ROOT / relative_key
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = _resolve_ffmpeg_exe()
    if ffmpeg_exe is None:
        return relative_key

    if not preview_path.exists() or preview_path.stat().st_mtime < source.stat().st_mtime:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(preview_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"preview_videos/{relative_key}"


def _resolve_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
    except Exception:
        imageio_ffmpeg = None

    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    return None


def build_html() -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PhysV ABCDE Metrics Report</title>
  <style>
    :root {{
      --bg: #0f1417;
      --panel: #162027;
      --panel2: #1c2730;
      --line: #30414f;
      --text: #eaf0f4;
      --muted: #9cb1c0;
      --accent: #f09a5c;
      --pdi: #7db7ff;
      --pdi2: #89d2ff;
      --wmr: #7bd39e;
      --cosmos: #ffb36f;
      --proxy: #f4c96b;
      --proxy2: #ffd97a;
      --proxy3: #ffebb3;
      --proxy4: #ffe9c8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at top left, rgba(240,154,92,0.09), transparent 22%), linear-gradient(180deg, #0d1317, #11181d); color: var(--text); font-family: system-ui, sans-serif; }}
    .page {{ max-width: 1720px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 10px; font-size: 30px; }}
    h2 {{ margin: 0; font-size: 20px; }}
    code {{ background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; }}
    .sub {{ color: var(--muted); margin-bottom: 22px; line-height: 1.65; max-width: 1120px; }}
    .legend {{ margin-bottom: 26px; }}
    .legend-sub {{ color: var(--muted); margin: 6px 0 14px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; }}
    .metric-card {{ border: 1px solid var(--line); border-left-width: 6px; border-radius: 14px; background: var(--panel); padding: 14px 16px; }}
    .metric-card.metric-official_pdi {{ border-left-color: var(--pdi); }}
    .metric-card.metric-wmreward_surprise {{ border-left-color: var(--wmr); }}
    .metric-card.metric-cosmos_reason1 {{ border-left-color: var(--cosmos); }}
    .metric-card.metric-vjepa_predictive_alignment {{ border-left-color: var(--proxy); }}
    .metric-card.metric-vjepa_temporal_relation_raw_error {{ border-left-color: var(--proxy2); }}
    .metric-card.metric-vjepa_delta_relation_raw_error {{ border-left-color: var(--proxy3); }}
    .metric-card.metric-vjepa_delta_profile_error {{ border-left-color: var(--proxy4); }}
    .metric-card.metric-videophy2_auto_sa {{ border-left-color: #bf7bff; }}
    .metric-card.metric-videophy2_auto_pc {{ border-left-color: #f36f6f; }}
    .metric-card.metric-videophy2_auto_joint {{ border-left-color: #ffb84f; }}
    .metric-name {{ font-weight: 700; font-size: 16px; margin-bottom: 4px; }}
    .metric-dir {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .metric-field {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #d9e2ea; font-size: 12px; }}
    .group-block {{ margin-bottom: 26px; }}
    .group-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-end; margin: 0 0 12px; }}
    .group-tag {{ display: inline-block; padding: 4px 8px; border-radius: 999px; background: rgba(240,154,92,0.16); color: var(--accent); font-size: 11px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px; }}
    .group-desc {{ color: var(--muted); margin-top: 6px; line-height: 1.55; max-width: 980px; }}
    .aux-details {{ margin-top: 10px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel2); }}
    .aux-details summary {{ cursor: pointer; padding: 10px 14px; color: var(--muted); font-weight: 700; }}
    .aux-copy {{ color: var(--muted); padding: 0 14px 10px; line-height: 1.5; }}
    .aux-table {{ margin: 0 12px 12px; width: calc(100% - 24px); }}
    .samples-section {{ margin-top: 34px; padding-top: 6px; }}
    .sample-case-list {{ display: grid; gap: 18px; }}
    .sample-case {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
      padding: 16px;
    }}
    .sample-case-head h3 {{ margin: 0 0 8px; font-size: 18px; }}
    .sample-case-head p {{ margin: 0 0 14px; color: var(--muted); line-height: 1.6; max-width: 1080px; }}
    .sample-case-tag {{
      display: inline-block;
      margin-bottom: 8px;
      padding: 4px 9px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      color: #ffd9b8;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
    }}
    .sample-video-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 14px; }}
    .sample-video-card {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: var(--panel);
    }}
    .sample-video-label {{
      padding: 10px 12px;
      font-size: 12px;
      font-weight: 800;
      color: #fff1df;
      border-bottom: 1px solid var(--line);
      background: rgba(240,154,92,0.08);
    }}
    .sample-video-card video {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #0a0f12;
    }}
    .sample-video-card figcaption {{ padding: 10px 12px 12px; }}
    .sample-caption {{ font-size: 12px; line-height: 1.5; color: var(--text); }}
    .sample-json {{ margin-top: 6px; font-size: 11px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .sample-table-wrap {{ overflow-x: auto; }}
    .sample-metrics-table {{ min-width: 900px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; font-size: 12px; }}
    th {{ background: rgba(255,255,255,0.05); text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; }}
    td.label-cell {{ font-weight: 700; color: #fff4e6; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .metric-official_pdi, .metric-scale_component, .metric-traj_component, .metric-epsilon_rigidity, .metric-vp_component {{ color: var(--pdi); }}
    .metric-scale_component, .metric-traj_component, .metric-epsilon_rigidity, .metric-vp_component {{ color: var(--pdi2); }}
    .metric-wmreward_surprise {{ color: var(--wmr); }}
    .metric-cosmos_reason1 {{ color: var(--cosmos); }}
    .metric-vjepa_predictive_alignment {{ color: var(--proxy); }}
    .metric-vjepa_temporal_relation_raw_error {{ color: var(--proxy2); }}
    .metric-vjepa_delta_relation_raw_error {{ color: var(--proxy3); }}
    .metric-vjepa_delta_profile_error {{ color: var(--proxy4); }}
    .metric-videophy2_auto_sa {{ color: #d3a6ff; }}
    .metric-videophy2_auto_pc {{ color: #f8a3a3; }}
    .metric-videophy2_auto_joint {{ color: #ffd08a; }}
    td.best {{
      font-weight: 800;
      background: linear-gradient(180deg, rgba(240,154,92,0.18), rgba(240,154,92,0.08));
      box-shadow: inset 0 0 0 1px rgba(240,154,92,0.45);
      border-radius: 8px;
    }}
    tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
    @media (max-width: 1200px) {{
      .metric-grid {{ grid-template-columns: 1fr; }}
      .group-head {{ flex-direction: column; align-items: stretch; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>PhysV ABCDE Metrics Report</h1>
    <div class="sub">
      页面统一展示 A/B/C 仿真评测、D 组 Physics-IQ 和 E 组 PhyGenBench 的同一套结果：<strong>Official PDI</strong>、<strong>WMReward Surprise</strong>、<strong>Cosmos Reason1</strong>、<strong>V-JEPA 子指标</strong>、<strong>VideoPhy-2 SA / PC / Joint</strong>。
      其中 WMReward 直接采用官方 <code>surprise / loss</code> 口径，越低越好；Cosmos Reason1 复用 cookbook 中 Reason1 physical-plausibility prompt 的 1-5 分，越高越好；V-JEPA 去掉手工加权总分和 margin 截断，直接展示 <code>predictive_alignment / temporal_relation_raw_error / delta_relation_raw_error / delta_profile_error</code>，其中前三项里的后三者是主误差项，<code>predictive_alignment</code> 只保留为辅助诊断；VideoPhy-2 的 SA / PC 是 1-5 离散评分，越高越好；Joint 表示 <code>SA&gt;=4 且 PC&gt;=4</code> 的通过率。PhyGround 当前不进主表，原因是 released <code>phyjudge-9B/infer.py</code> 在本批视频上经常生成超长自由文本而不稳定落出结构化分数，无法保证官方 case 和批处理 case 都稳定同分。
    </div>
    {build_metric_legend()}
    {build_group_a()}
    {build_group_b1()}
    {build_group_b2()}
    {build_group_b3()}
    {build_group_c()}
    {build_group_d()}
    {build_group_e()}
    {build_representative_samples()}
  </div>
</body>
</html>"""


def _first_physicsiq_full_compare_case() -> str | None:
    gt_root = PHYSICSIQ_OUTPUT / "GT" / PHYSICSIQ_BENCHMARK
    for gt_json in sorted(gt_root.glob("*.json")):
        sample_id = gt_json.stem
        if all((PHYSICSIQ_OUTPUT / method / PHYSICSIQ_BENCHMARK / f"{sample_id}.json").is_file() for method in PHYSICSIQ_METHODS):
            return sample_id
    return None


def _first_phygenbench_full_compare_case() -> str | None:
    method_roots = [PHYGENBENCH_OUTPUT / method / PHYGENBENCH_BENCHMARK for method in PHYGENBENCH_METHODS]
    if not all(root.is_dir() for root in method_roots):
        return None
    sample_sets = [{path.stem for path in root.glob("*.json")} for root in method_roots]
    if not sample_sets:
        return None
    common = sorted(set.intersection(*sample_sets))
    return common[0] if common else None


def main() -> None:
    args = parse_args()
    ABC_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (ABC_REPORT_ROOT / "index.html").write_text(build_html(), encoding="utf-8")

    for name, target in [
        ("dataset_videos", DATA_ROOT / "videos"),
        ("pdi_output", A_OUTPUT),
        ("physicsiq_output", PHYSICSIQ_OUTPUT),
        ("phygenbench_output", PHYGENBENCH_OUTPUT),
    ]:
        link = ABC_REPORT_ROOT / name
        if not link.exists():
            link.symlink_to(target)

    print(f"http://127.0.0.1:{args.port}/index.html")
    subprocess.run([sys.executable, "-m", "http.server", str(args.port), "--directory", str(ABC_REPORT_ROOT)], check=False)


if __name__ == "__main__":
    main()
