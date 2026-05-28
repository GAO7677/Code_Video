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
from .records import load_payload, metric_value


METRIC_LABELS = {
    "official_pdi": "Official PDI",
    "scale_component": "Scale",
    "traj_component": "Trajectory",
    "epsilon_rigidity": "Rigidity",
    "vp_component": "VP",
    "wmreward_jepa": "WMReward JEPA",
    "vjepa_proxy": "V-JEPA Proxy",
}


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


def metric_td(name: str, value: Any) -> str:
    return f"<td class='num metric metric-{name}'>{fv(value)}</td>"


def text_td(value: Any, classes: str = "") -> str:
    cls = f" class='{classes}'" if classes else ""
    return f"<td{cls}>{value}</td>"


def build_metric_legend() -> str:
    cards = []
    cards.append(_metric_card("official_pdi", "Official PDI", "↓ lower is better", "`metric_results.official_pdi`"))
    cards.append(_metric_card("wmreward_jepa", "WMReward JEPA", "↑ higher is better", "`metric_results.wmreward_jepa`"))
    cards.append(_metric_card("vjepa_proxy", "V-JEPA Proxy", "↑ higher is better", "`metric_results.vjepa_proxy`"))
    return f"""
    <section class="legend">
      <h2>指标图例</h2>
      <div class="legend-sub">元数据列与指标列分开渲染；PDI 拆成总分和 4 个子指标，WMReward / Proxy 单独着色。</div>
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


def build_group_a() -> str:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for json_path in iter_group_jsons("A"):
        payload = load_payload(json_path)
        method = str(payload.get("method") or payload.get("provider") or json_path.parts[-3])
        by_method[method].append(payload)

    rows = []
    for method in sorted(by_method):
        payloads = by_method[method]
        rows.append(
            "<tr>"
            f"{text_td(method, 'label-cell')}"
            f"{text_td(len(payloads), 'num')}"
            f"{metric_td('official_pdi', mean_or_none(_metric_list(payloads, 'official_pdi')))}"
            f"{metric_td('scale_component', mean_or_none(_metric_list(payloads, 'scale_component')))}"
            f"{metric_td('traj_component', mean_or_none(_metric_list(payloads, 'traj_component')))}"
            f"{metric_td('epsilon_rigidity', mean_or_none(_metric_list(payloads, 'epsilon_rigidity')))}"
            f"{metric_td('vp_component', mean_or_none(_metric_list(payloads, 'vp_component')))}"
            f"{metric_td('wmreward_jepa', mean_or_none(_metric_list(payloads, 'wmreward_jepa')))}"
            f"{metric_td('vjepa_proxy', mean_or_none(_metric_list(payloads, 'vjepa_proxy')))}"
            "</tr>"
        )

    thead = """
    <thead>
      <tr>
        <th colspan="2">Method Metadata</th>
        <th colspan="5">Official PDI Breakdown</th>
        <th colspan="2">Predictive Metrics</th>
      </tr>
      <tr>
        <th>Method</th>
        <th>N</th>
        <th class="metric metric-official_pdi">Official PDI ↓</th>
        <th class="metric metric-scale_component">Scale ↓</th>
        <th class="metric metric-traj_component">Trajectory ↓</th>
        <th class="metric metric-epsilon_rigidity">Rigidity ↓</th>
        <th class="metric metric-vp_component">VP ↓</th>
        <th class="metric metric-wmreward_jepa">WMReward JEPA ↑</th>
        <th class="metric metric-vjepa_proxy">V-JEPA Proxy ↑</th>
      </tr>
    </thead>
    """
    return section_header("A", GROUP_SPECS["A"].title, GROUP_SPECS["A"].description) + standard_table(thead, rows) + section_footer()


def build_group_b1() -> str:
    rows = []
    for json_path in iter_group_jsons("B1"):
        payload = load_payload(json_path)
        params = payload.get("parameters", {})
        rows.append(
            "<tr>"
            f"{text_td(json_path.stem, 'label-cell')}"
            f"{text_td(params.get('restitution', '-'), 'num')}"
            f"{text_td(params.get('lateral_friction', '-'), 'num')}"
            f"{text_td(params.get('ball_mass_kg', '-'), 'num')}"
            f"{metric_row(payload)}"
            "</tr>"
        )
    return _sample_group_section("B1", "Scenario", ["e", "μ", "m"], rows)


def build_group_b2() -> str:
    rows = []
    for json_path in iter_group_jsons("B2"):
        payload = load_payload(json_path)
        rows.append(
            "<tr>"
            f"{text_td(json_path.stem, 'label-cell')}"
            f"{text_td(payload.get('description', payload.get('experiment', '-')))}"
            f"{metric_row(payload)}"
            "</tr>"
        )
    return _sample_group_section("B2", "Scenario", ["Description"], rows)


def build_group_b3() -> str:
    rows = []
    for json_path in iter_group_jsons("B3"):
        payload = load_payload(json_path)
        rows.append(
            "<tr>"
            f"{text_td(payload.get('scenario', json_path.stem), 'label-cell')}"
            f"{text_td(payload.get('appearance_variant', '-'))}"
            f"{metric_row(payload)}"
            "</tr>"
        )
    return _sample_group_section("B3", "Base Scenario", ["Appearance"], rows)


def build_group_c() -> str:
    rows = []
    for json_path in iter_group_jsons("C"):
        payload = load_payload(json_path)
        rows.append(
            "<tr>"
            f"{text_td(json_path.stem, 'label-cell')}"
            f"{metric_row(payload)}"
            "</tr>"
        )
    return _sample_group_section("C", "Video", [], rows)


def metric_row(payload: dict[str, Any]) -> str:
    return "".join(
        [
            metric_td("official_pdi", metric_value(payload, "official_pdi")),
            metric_td("scale_component", metric_value(payload, "scale_component")),
            metric_td("traj_component", metric_value(payload, "traj_component")),
            metric_td("epsilon_rigidity", metric_value(payload, "epsilon_rigidity")),
            metric_td("vp_component", metric_value(payload, "vp_component")),
            metric_td("wmreward_jepa", metric_value(payload, "wmreward_jepa")),
            metric_td("vjepa_proxy", metric_value(payload, "vjepa_proxy")),
        ]
    )


def _sample_group_section(group_id: str, label1: str, extra_headers: list[str], rows: list[str]) -> str:
    meta_colspan = 1 + len(extra_headers)
    meta_headers = [f"<th>{label1}</th>"] + [f"<th>{header}</th>" for header in extra_headers]
    thead = f"""
    <thead>
      <tr>
        <th colspan="{meta_colspan}">Sample Metadata</th>
        <th colspan="5">Official PDI Breakdown</th>
        <th colspan="2">Predictive Metrics</th>
      </tr>
      <tr>
        {''.join(meta_headers)}
        <th class="metric metric-official_pdi">Official PDI ↓</th>
        <th class="metric metric-scale_component">Scale ↓</th>
        <th class="metric metric-traj_component">Trajectory ↓</th>
        <th class="metric metric-epsilon_rigidity">Rigidity ↓</th>
        <th class="metric metric-vp_component">VP ↓</th>
        <th class="metric metric-wmreward_jepa">WMReward JEPA ↑</th>
        <th class="metric metric-vjepa_proxy">V-JEPA Proxy ↑</th>
      </tr>
    </thead>
    """
    spec = GROUP_SPECS[group_id]
    return section_header(group_id, spec.title, spec.description) + standard_table(thead, rows) + section_footer()


def _metric_list(payloads: list[dict[str, Any]], name: str) -> list[float]:
    values = [metric_value(payload, name) for payload in payloads]
    return [float(value) for value in values if value is not None]


def build_html() -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PhysV ABC Report</title>
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
      --proxy: #f4c96b;
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
    .metric-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .metric-card {{ border: 1px solid var(--line); border-left-width: 6px; border-radius: 14px; background: var(--panel); padding: 14px 16px; }}
    .metric-card.metric-official_pdi {{ border-left-color: var(--pdi); }}
    .metric-card.metric-wmreward_jepa {{ border-left-color: var(--wmr); }}
    .metric-card.metric-vjepa_proxy {{ border-left-color: var(--proxy); }}
    .metric-name {{ font-weight: 700; font-size: 16px; margin-bottom: 4px; }}
    .metric-dir {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .metric-field {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #d9e2ea; font-size: 12px; }}
    .group-block {{ margin-bottom: 26px; }}
    .group-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-end; margin: 0 0 12px; }}
    .group-tag {{ display: inline-block; padding: 4px 8px; border-radius: 999px; background: rgba(240,154,92,0.16); color: var(--accent); font-size: 11px; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px; }}
    .group-desc {{ color: var(--muted); margin-top: 6px; line-height: 1.55; max-width: 980px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; font-size: 12px; }}
    th {{ background: rgba(255,255,255,0.05); text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; }}
    td.label-cell {{ font-weight: 700; color: #fff4e6; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .metric-official_pdi, .metric-scale_component, .metric-traj_component, .metric-epsilon_rigidity, .metric-vp_component {{ color: var(--pdi); }}
    .metric-scale_component, .metric-traj_component, .metric-epsilon_rigidity, .metric-vp_component {{ color: var(--pdi2); }}
    .metric-wmreward_jepa {{ color: var(--wmr); }}
    .metric-vjepa_proxy {{ color: var(--proxy); }}
    tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
    @media (max-width: 1200px) {{
      .metric-grid {{ grid-template-columns: 1fr; }}
      .group-head {{ flex-direction: column; align-items: stretch; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>PhysV ABC Metrics Report</h1>
    <div class="sub">
      页面只展示三类统一指标名称：<strong>Official PDI</strong>、<strong>WMReward JEPA</strong>、<strong>V-JEPA Proxy</strong>。
      其中 Official PDI 进一步拆为 <code>score / scale / trajectory / rigidity / vp</code> 五列，避免把元数据和指标混在一起。
    </div>
    {build_metric_legend()}
    {build_group_a()}
    {build_group_b1()}
    {build_group_b2()}
    {build_group_b3()}
    {build_group_c()}
  </div>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    ABC_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (ABC_REPORT_ROOT / "index.html").write_text(build_html(), encoding="utf-8")

    for name, target in [("dataset_videos", DATA_ROOT / "videos"), ("pdi_output", A_OUTPUT)]:
        link = ABC_REPORT_ROOT / name
        if not link.exists():
            link.symlink_to(target)

    print(f"http://127.0.0.1:{args.port}/index.html")
    subprocess.run([sys.executable, "-m", "http.server", str(args.port), "--directory", str(ABC_REPORT_ROOT)], check=False)


if __name__ == "__main__":
    main()
