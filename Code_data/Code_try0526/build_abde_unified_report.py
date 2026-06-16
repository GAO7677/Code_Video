#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/abc_report")
OUT_HTML = OUT_ROOT / "index.html"

ABD_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/ABD_test")
C_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/shuffle_test")
D_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark")
E_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/phygenbench")

METHODS_4WAY = ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]
METHODS_3WAY = ["wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]
METHOD_LABELS_4WAY = {
    "GT": "GT",
    "wan22-5B-TI2V": "Wan2.2-5B TI2V",
    "VACE_1p3B_TI2V": "VACE 1.3B TI2V",
    "VACE_1p3B_ctx08": "VACE 1.3B ctx=8",
}

METRICS = [
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

METRIC_TITLES = {
    "official_pdi": "Official PDI ↓",
    "scale_component": "Scale ↓",
    "traj_component": "Trajectory ↓",
    "epsilon_rigidity": "Rigidity ↓",
    "vp_component": "VP ↓",
    "wmreward_surprise": "WMReward Surprise ↓",
    "cosmos_reason1": "Cosmos ↑",
    "vjepa_temporal_relation_raw_error": "RelRaw ↓",
    "vjepa_delta_relation_raw_error": "DeltaRel ↓",
    "vjepa_delta_profile_error": "DeltaProf ↓",
    "videophy2_auto_sa": "SA ↑",
    "videophy2_auto_pc": "PC ↑",
    "videophy2_auto_joint": "Joint ↑",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric_from_payload(payload: dict[str, Any], metric: str) -> float | None:
    mr = payload.get("metric_results") if isinstance(payload.get("metric_results"), dict) else {}
    if metric == "official_pdi":
        bucket = mr.get("official_pdi") if isinstance(mr, dict) else None
        if isinstance(bucket, dict) and bucket.get("pdi_score") is not None:
            return float(bucket["pdi_score"])
        return payload.get("pdi_score")
    if metric == "scale_component":
        bucket = mr.get("official_pdi") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("scale_component", bucket.get("scale_error"))
        return payload.get("pdi", {}).get("scale_error")
    if metric == "traj_component":
        bucket = mr.get("official_pdi") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("traj_component", bucket.get("traj_error"))
        return payload.get("pdi", {}).get("traj_error")
    if metric == "epsilon_rigidity":
        bucket = mr.get("official_pdi") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("epsilon_rigidity", bucket.get("rigidity_error"))
        return payload.get("pdi", {}).get("rigidity_error")
    if metric == "vp_component":
        bucket = mr.get("official_pdi") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("vp_component", bucket.get("vp_error"))
        return payload.get("pdi", {}).get("vp_error")
    if metric == "wmreward_surprise":
        bucket = mr.get("wmreward_jepa") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("surprise")
        return payload.get("wmreward_jepa")
    if metric == "cosmos_reason1":
        bucket = mr.get("cosmos_reason1") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("score")
        if payload.get("cosmos_reason1_score") is not None:
            return payload.get("cosmos_reason1_score")
        if isinstance(payload.get("cosmos_reason1"), dict):
            return payload["cosmos_reason1"].get("score")
        return None
    if metric == "vjepa_temporal_relation_raw_error":
        bucket = mr.get("vjepa_proxy") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("details", {}).get("temporal_relation_raw_error")
        if isinstance(payload.get("jepa"), dict):
            return payload["jepa"].get("temporal_relation_raw_error")
        return None
    if metric == "vjepa_delta_relation_raw_error":
        bucket = mr.get("vjepa_proxy") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("details", {}).get("delta_relation_raw_error")
        if isinstance(payload.get("jepa"), dict):
            return payload["jepa"].get("delta_relation_raw_error")
        return None
    if metric == "vjepa_delta_profile_error":
        bucket = mr.get("vjepa_proxy") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("details", {}).get("delta_profile_error")
        if isinstance(payload.get("jepa"), dict):
            return payload["jepa"].get("delta_profile_error")
        return None
    if metric == "videophy2_auto_sa":
        bucket = mr.get("videophy2_auto") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("sa_score")
        return payload.get("videophy2_auto_sa")
    if metric == "videophy2_auto_pc":
        bucket = mr.get("videophy2_auto") if isinstance(mr, dict) else None
        if isinstance(bucket, dict):
            return bucket.get("pc_score")
        return payload.get("videophy2_auto_pc")
    if metric == "videophy2_auto_joint":
        sa = metric_from_payload(payload, "videophy2_auto_sa")
        pc = metric_from_payload(payload, "videophy2_auto_pc")
        if sa is None or pc is None:
            return None
        return 1.0 if sa >= 4 and pc >= 4 else 0.0
    raise KeyError(metric)


def summarize(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    row = {"count": len(payloads)}
    for metric in METRICS:
        vals = [metric_from_payload(p, metric) for p in payloads]
        if any(v is None for v in vals):
            row[metric] = None
        else:
            row[metric] = mean([float(v) for v in vals if v is not None])
    return row


def best_masks(rows: list[dict[str, Any]]) -> list[dict[str, bool]]:
    best: dict[str, float] = {}
    for metric in METRICS:
        vals = [row[metric] for row in rows if row.get(metric) is not None]
        if not vals:
            continue
        best[metric] = min(vals) if metric in LOWER_IS_BETTER else max(vals)
    masks: list[dict[str, bool]] = []
    for row in rows:
        masks.append({metric: row.get(metric) is not None and metric in best and row[metric] == best[metric] for metric in METRICS})
    return masks


def render_table(title: str, rows: list[dict[str, Any]]) -> str:
    masks = best_masks(rows)
    body = []
    for row, mask in zip(rows, masks):
        body.append(
            "<tr>"
            f"<td class='label-cell'>{row['method']}</td>"
            f"<td class='num'>{row['count']}</td>"
            + "".join(f"<td class='num {'best' if mask[m] else ''}'>{fv(row[m])}</td>" for m in METRICS)
            + "</tr>"
        )
    return f"""
    <section class="block">
      <h2>{title}</h2>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>N</th>
            {''.join(f'<th>{METRIC_TITLES[m]}</th>' for m in METRICS)}
          </tr>
        </thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </section>
    """


def load_group_from_dir(root: Path, benchmark: str, methods: list[str]) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        mdir = root / method / benchmark
        payloads = [load_json(p) for p in sorted(mdir.glob("*.json"))]
        rows.append({"method": method, **summarize(payloads)})
    return rows


def load_b_group(prefix: str) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS_4WAY:
        mdir = ABD_ROOT / "B" / method
        payloads = []
        for p in sorted(mdir.glob("*.json")):
            payload = load_json(p)
            category = str(payload.get("category") or "")
            if category.startswith(prefix):
                payloads.append(payload)
        rows.append({"method": method, **summarize(payloads)})
    return rows


def load_c_group() -> list[dict[str, Any]]:
    payloads_by_method = {"original": [], "shuffled": []}
    for p in sorted(C_ROOT.glob("*.json")):
        payload = load_json(p)
        stem = p.stem
        if stem.endswith("_original"):
            payloads_by_method["original"].append(payload)
        elif stem.endswith("_shuffled"):
            payloads_by_method["shuffled"].append(payload)
    return [{"method": key, **summarize(value)} for key, value in payloads_by_method.items()]


def load_e_group() -> list[dict[str, Any]]:
    rows = []
    for method in METHODS_3WAY:
        payloads = []
        mdir = E_ROOT / "output" / method / "phygenbench"
        for p in sorted(mdir.glob("*.json")):
            payloads.append(load_json(p))
        rows.append({"method": method, **summarize(payloads)})
    return rows


def render_case_card(title: str, desc: str, items: list[tuple[str, str, str]]) -> str:
    cards = []
    for label, path, kind in items:
        if kind == "video":
            media = f'<video controls muted playsinline preload="metadata" src="{path}"></video>'
        else:
            media = f'<img src="{path}" alt="{label}" />'
        cards.append(f"<figure class='sample-card'><div class='sample-label'>{label}</div>{media}</figure>")
    return f"""
    <section class="case-block">
      <h3>{title}</h3>
      <div class="case-desc">{desc}</div>
      <div class="sample-grid">{''.join(cards)}</div>
    </section>
    """


def build_html() -> str:
    a_rows = load_group_from_dir(ABD_ROOT / "A", "GT", METHODS_4WAY)
    b1_rows = load_b_group("B1")
    b2_rows = load_b_group("B2")
    b3_rows = load_b_group("B3")
    d_rows = load_group_from_dir(ABD_ROOT / "D", "GT", METHODS_4WAY)
    c_rows = load_c_group()
    e_rows = load_e_group()

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ABDE Unified Metrics Report</title>
  <style>
    body {{ margin: 0; background: #0f1417; color: #eaf0f4; font-family: system-ui, sans-serif; }}
    .page {{ max-width: 1840px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 10px; }}
    .sub {{ color: #9cb1c0; margin-bottom: 16px; line-height: 1.6; }}
    .jump {{ margin-bottom: 24px; }}
    .jump a {{ color: #11181d; background: #f09a5c; padding: 8px 12px; border-radius: 10px; text-decoration: none; font-weight: 800; }}
    .block, .case-block {{ margin: 22px 0 30px; }}
    .case-block {{ border-top: 1px solid #30414f; padding-top: 12px; }}
    .case-desc {{ color: #9cb1c0; margin-bottom: 12px; line-height: 1.55; }}
    .sample-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .sample-card {{ margin: 0; border: 1px solid #30414f; border-radius: 14px; overflow: hidden; background: #162027; }}
    .sample-label {{ padding: 10px 12px; border-bottom: 1px solid #30414f; font-weight: 700; background: rgba(255,255,255,0.05); }}
    .sample-card video, .sample-card img {{ display: block; width: 100%; aspect-ratio: 16/9; object-fit: cover; background: #0a0f12; }}
    table {{ width: 100%; border-collapse: collapse; background: #162027; border: 1px solid #30414f; }}
    th, td {{ border-bottom: 1px solid #30414f; padding: 8px 10px; font-size: 12px; }}
    th {{ background: rgba(255,255,255,0.05); text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; }}
    .label-cell {{ font-weight: 700; white-space: nowrap; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .best {{ font-weight: 800; background: rgba(240,154,92,0.18); }}
  </style>
</head>
<body>
<div class="page">
  <h1>ABDE Unified Metrics Report</h1>
  <div class="sub">A / B / D 直接读取 <code>ABD_test</code>；C 读取 <code>shuffle_test</code>；E 读取 <code>phygenbench</code>。每个测试集单独成表，每个方法一行；同一测试集下指标不完整时保持空白。方法统计已并入本页下方。</div>
  {render_table("A / PDI-Bench", a_rows)}
  {render_case_card("A 组代表性 case", "同一提示和场景下对比 GT、Wan 和 VACE，适合看外观跟踪与时序稳定性。", [
      ("GT", str(ABD_ROOT / 'A' / 'GT' / 'Dynamic_Tracking' / 'bus.mp4'), 'video'),
      ("Wan2.2-5B TI2V", str(ABD_ROOT / 'A' / 'wan22-5B-TI2V' / 'Dynamic_Tracking' / 'bus.mp4'), 'video'),
      ("VACE 1.3B TI2V", str(ABD_ROOT / 'A' / 'VACE_1p3B_TI2V' / 'Dynamic_Tracking' / 'bus.mp4'), 'video'),
      ("VACE 1.3B ctx=8", str(ABD_ROOT / 'A' / 'VACE_1p3B_ctx08' / 'Dynamic_Tracking' / 'bus.mp4'), 'video'),
  ])}
  {render_table("B1 / Ball-Block Physics", b1_rows)}
  {render_case_card("B1 组代表性 case", "固定外观，只改恢复系数、摩擦和球质量；这里展示的是整组方法均值，不是单个样本。", [
      ("GT", str(ABD_ROOT / 'B' / 'GT' / '004_B1_ball_block_physics_e07_mu05_m1.mp4'), 'video'),
      ("Wan2.2-5B TI2V", str(ABD_ROOT / 'B' / 'wan22-5B-TI2V' / '004_B1_ball_block_physics_e07_mu05_m1.mp4'), 'video'),
      ("VACE 1.3B TI2V", str(ABD_ROOT / 'B' / 'VACE_1p3B_TI2V' / '004_B1_ball_block_physics_e07_mu05_m1.mp4'), 'video'),
      ("VACE 1.3B ctx=8", str(ABD_ROOT / 'B' / 'VACE_1p3B_ctx08' / '004_B1_ball_block_physics_e07_mu05_m1.mp4'), 'video'),
  ])}
  {render_table("B2 / JEPA Sensitivity", b2_rows)}
  {render_case_card("B2 组代表性 case", "固定外观，只改速度、质量、重力、碰撞与方向；这里展示的是整组方法均值，不是单个样本。", [
      ("GT", str(ABD_ROOT / 'B' / 'GT' / '019_B2_gravity_grav_050.mp4'), 'video'),
      ("Wan2.2-5B TI2V", str(ABD_ROOT / 'B' / 'wan22-5B-TI2V' / '019_B2_gravity_grav_050.mp4'), 'video'),
      ("VACE 1.3B TI2V", str(ABD_ROOT / 'B' / 'VACE_1p3B_TI2V' / '019_B2_gravity_grav_050.mp4'), 'video'),
      ("VACE 1.3B ctx=8", str(ABD_ROOT / 'B' / 'VACE_1p3B_ctx08' / '019_B2_gravity_grav_050.mp4'), 'video'),
  ])}
  {render_table("B3 / Appearance Sensitivity", b3_rows)}
  {render_case_card("B3 组代表性 case", "同一物理轨迹，只改渲染外观和光照；这里展示的是整组方法均值，不是单个样本。", [
      ("GT", str(ABD_ROOT / 'B' / 'GT' / '040_B3_default_render_e07_mu05_m1_v1_default.mp4'), 'video'),
      ("Wan2.2-5B TI2V", str(ABD_ROOT / 'B' / 'wan22-5B-TI2V' / '040_B3_default_render_e07_mu05_m1_v1_default.mp4'), 'video'),
      ("VACE 1.3B TI2V", str(ABD_ROOT / 'B' / 'VACE_1p3B_TI2V' / '040_B3_default_render_e07_mu05_m1_v1_default.mp4'), 'video'),
      ("VACE 1.3B ctx=8", str(ABD_ROOT / 'B' / 'VACE_1p3B_ctx08' / '040_B3_default_render_e07_mu05_m1_v1_default.mp4'), 'video'),
  ])}
  {render_table("D / Physics-IQ", d_rows)}
  {render_case_card("D 组代表性 case", "真实物理现象基准，覆盖流体、力学和光学等类别，适合观察推断类指标的稳定性。", [
      ("GT", str(D_ROOT / 'report_subset' / 'assets' / '0032_perspective-center_trimmed-balls-collide' / 'GT.overlay.mp4'), 'video'),
      ("Wan2.2-5B TI2V", str(D_ROOT / 'report_subset' / 'assets' / '0032_perspective-center_trimmed-balls-collide' / 'wan22-5B-TI2V.overlay.mp4'), 'video'),
      ("VACE 1.3B TI2V", str(D_ROOT / 'report_subset' / 'assets' / '0032_perspective-center_trimmed-balls-collide' / 'VACE_1p3B_TI2V.overlay.mp4'), 'video'),
      ("VACE 1.3B ctx=8", str(D_ROOT / 'report_subset' / 'assets' / '0032_perspective-center_trimmed-balls-collide' / 'VACE_1p3B_ctx08.overlay.mp4'), 'video'),
  ])}
  {render_table("C / Shuffle Sanity", c_rows)}
  {render_case_card("C 组代表性 case", "只破坏帧顺序，不改单帧内容。这里展示原始与 shuffled 两个代表样本。", [
      ("original", str(C_ROOT / 'gt_ball_original.mp4'), 'video'),
      ("shuffled", str(C_ROOT / 'gt_bus_shuffled.mp4'), 'video'),
  ])}
  {render_table("E / PhyGenBench", e_rows)}
  {render_case_card("E 组代表性 case", "开放式物理生成基准，展示 FLUX 首帧和三路视频方法对同一 prompt 的生成。", [
      ("FLUX first frame", str(E_ROOT / 'output' / 'FLUX_1_Kontext' / 'phygenbench' / '001.png'), 'img'),
      ("Wan2.2-5B TI2V", str(E_ROOT / 'output' / 'wan22-5B-TI2V' / 'phygenbench' / '001.mp4'), 'video'),
      ("VACE 1.3B TI2V", str(E_ROOT / 'output' / 'VACE_1p3B_TI2V' / 'phygenbench' / '001.mp4'), 'video'),
      ("VACE 1.3B ctx=8", str(E_ROOT / 'output' / 'VACE_1p3B_ctx08' / 'phygenbench' / '001.ctx08.mp4'), 'video'),
  ])}
</div>
</body>
</html>"""
    return html


def render_table_rows(label: str, rows: list[dict[str, Any]]) -> list[str]:
    masks = best_masks(rows)
    out = []
    for row, mask in zip(rows, masks):
        out.append(
            "<tr>"
            f"<td class='label-cell'>{label} / {row['method']}</td>"
            f"<td class='num'>{row['count']}</td>"
            + "".join(f"<td class='num {'best' if mask[m] else ''}'>{fv(row[m])}</td>" for m in METRICS)
            + "</tr>"
        )
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(), encoding="utf-8")
    print(OUT_HTML)


if __name__ == "__main__":
    main()
