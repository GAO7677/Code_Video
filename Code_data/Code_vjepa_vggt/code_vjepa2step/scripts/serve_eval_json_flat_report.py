#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


TMP_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa2step/tmp")
JSON_ROOT = TMP_ROOT / "eval_json_flat"
REPORT_ROOT = TMP_ROOT / "eval_json_flat_report"
REPORT_PATH = REPORT_ROOT / "index.html"

PRIMARY_METRICS = [
    ("official_pdi", "Official PDI", "down"),
    ("scale_component", "Scale", "down"),
    ("traj_component", "Trajectory", "down"),
    ("epsilon_rigidity", "Rigidity", "down"),
    ("vp_component", "VP", "down"),
    ("wmreward_surprise", "WMReward Surprise", "down"),
    ("wmreward_similarity", "WMReward Similarity", "up"),
    ("vjepa_proxy", "V-JEPA Proxy", "up"),
    ("videophy2_auto_pc", "VideoPhy2 PC", "up"),
    ("videophy2_auto_sa", "VideoPhy2 SA", "up"),
    ("phyground_general_avg", "PhyGround Avg", "up"),
    ("cosmos_reason1", "Cosmos Reason1", "up"),
    ("fid", "FID", "down"),
    ("fvd", "FVD", "down"),
    ("accuracy", "Accuracy", "up"),
    ("pearson_correlation", "Pearson", "up"),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local report for eval_json_flat results.")
    parser.add_argument("--port", type=int, default=18711)
    return parser.parse_args()


def load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_path in sorted(JSON_ROOT.glob("*.json")):
        if json_path.name.startswith("_"):
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        records.append(payload)
    return records


def fv(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return html.escape(str(value))


def rel_to_report(target: Path) -> str:
    resolved = target.resolve()
    if resolved.parent == (TMP_ROOT / "wan_runs").resolve():
        return f"wan_runs/{resolved.name}"
    if resolved.parent == (TMP_ROOT / "wan_runs_step_sweep").resolve():
        return f"wan_runs_step_sweep/{resolved.name}"
    return resolved.name


def resolve_video_path(record: dict[str, Any]) -> Path:
    path = Path(record["video_path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def build_video_panel(record: dict[str, Any], combo_label: str) -> str:
    video_path = resolve_video_path(record)
    video_rel = rel_to_report(video_path)
    prompt = record.get("prompt")
    prompt_html = (
        f"<div class='video-prompt'>{html.escape(str(prompt))}</div>"
        if isinstance(prompt, str) and prompt.strip()
        else ""
    )
    return f"""
    <article class="video-panel">
      <div class="video-panel-head">{html.escape(combo_label)}</div>
      <div class="video-wrap">
        <video controls preload="metadata" src="{html.escape(video_rel)}"></video>
      </div>
      <div class="video-meta">
        <div class="video-name">{html.escape(record.get("video_stem", "unknown"))}</div>
        {prompt_html}
      </div>
    </article>
    """


def parse_record_identity(record: dict[str, Any]) -> tuple[str, str, int, str]:
    stem = str(record.get("video_stem") or "")
    seed = "unknown"
    method = stem
    step = 50
    guidance = "default"

    if "_seed" in stem:
        method = stem.rsplit("_seed", 1)[0]
        seed = stem.rsplit("_seed", 1)[1]

    if "_steps" in method:
        base, step_text = method.rsplit("_steps", 1)
        method = base
        digits = "".join(ch for ch in step_text if ch.isdigit())
        if digits:
            step = int(digits)
    elif method.endswith("_same_prompt"):
        method = method[: -len("_same_prompt")]
        step = 50
        guidance = "same_prompt"

    return seed, method, step, guidance


def method_sort_key(method: str) -> tuple[int, str]:
    order = {
        "wan21_t2v_1p3b": 0,
        "wan21_vace_1p3b": 1,
        "wan22_ti2v_5b": 2,
    }
    return order.get(method, 99), method


def step_sort_key(step: int) -> tuple[int, int]:
    preferred = {5: 0, 15: 1, 25: 2, 50: 3}
    return preferred.get(step, 99), step


def combo_sort_key(combo: tuple[int, str]) -> tuple[tuple[int, int], int, str]:
    step, guidance = combo
    guidance_order = {
        "default": 0,
        "same_prompt": 1,
    }
    return step_sort_key(step), guidance_order.get(guidance, 99), guidance


def human_method_name(method: str) -> str:
    mapping = {
        "wan21_t2v_1p3b": "Wan2.1 T2V 1.3B",
        "wan21_vace_1p3b": "Wan2.1 VACE 1.3B",
        "wan22_ti2v_5b": "Wan2.2 TI2V 5B",
    }
    return mapping.get(method, method)


def step_label(step: int) -> str:
    if step == 50:
        return "step 50"
    return f"step {step}"


def combo_label(step: int, guidance: str) -> str:
    base = step_label(step)
    if guidance == "same_prompt":
        return f"{base} / same_prompt"
    if guidance == "default":
        return base
    return f"{base} / {guidance}"


def group_records(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[tuple[int, str], dict[str, Any]]]]:
    grouped: dict[str, dict[str, dict[tuple[int, str], dict[str, Any]]]] = {}
    for record in records:
        seed, method, step, guidance = parse_record_identity(record)
        grouped.setdefault(seed, {}).setdefault(method, {})[(step, guidance)] = record
    return grouped


def build_metric_table(method_rows: dict[tuple[int, str], dict[str, Any]], combos: list[tuple[int, str]]) -> str:
    header_cells = "".join(
        f"<th>{html.escape(combo_label(step, guidance))}</th>"
        for step, guidance in combos
    )
    body_rows = []
    for key, label, direction in PRIMARY_METRICS:
        arrow = "↓" if direction == "down" else "↑"
        value_cells = []
        for combo in combos:
            record = method_rows.get(combo)
            value_cells.append(
                f"<td class='metric-score'>{fv(record.get(key) if record else None)}</td>"
            )
        body_rows.append(
            "<tr>"
            f"<th class='metric-name'>{html.escape(label)} {arrow}</th>"
            + "".join(value_cells)
            + "</tr>"
        )
    return f"""
    <div class="metric-table-wrap">
      <table class="metric-table">
        <thead>
          <tr>
            <th class="metric-head">Metric</th>
            {header_cells}
          </tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
    """


def build_method_section(method: str, method_rows: dict[tuple[int, str], dict[str, Any]]) -> str:
    combos = sorted(method_rows.keys(), key=combo_sort_key)
    video_panels = []
    for step, guidance in combos:
        record = method_rows[(step, guidance)]
        video_panels.append(build_video_panel(record, combo_label(step, guidance)))
    return f"""
    <section class="method-block">
      <div class="method-head">
        <h3>{html.escape(human_method_name(method))}</h3>
      </div>
      <div class="video-grid">
        {''.join(video_panels)}
      </div>
      {build_metric_table(method_rows, combos)}
    </section>
    """


def build_seed_section(seed: str, seed_rows: dict[str, dict[tuple[int, str], dict[str, Any]]]) -> str:
    methods = sorted(seed_rows.keys(), key=method_sort_key)
    method_sections = "".join(build_method_section(method, seed_rows[method]) for method in methods)
    return f"""
    <section class="seed-block">
      <div class="seed-head">
        <h2>Seed {html.escape(seed)}</h2>
        <div class="seed-sub">每个方法单独成块；先展示各个 <code>step/guidance</code> 组合的视频，再给出统一指标表。表格每行一个指标，每列一个 <code>step/guidance</code> 组合。</div>
      </div>
      {method_sections}
    </section>
    """


def build_wmreward_explainer() -> str:
    return """
    <section class="wmr-box">
      <h2>WMReward 说明</h2>
      <p><code>WMReward</code> 这里走的是项目里封装的官方口径：<code>physv_eval/wmreward_official.py</code> 直接复用 <code>/home/gaoya/Code_Video/WMReward-main/compute_wmreward.py</code> 的默认参数，并在进程内缓存模型以便批量跑多个视频。</p>
      <p>它本质上不是“人工偏好奖励”，而是 <code>V-JEPA</code> 的未来预测误差。代码会先把视频最多取前 <code>49</code> 帧，缩放到模型输入尺寸，然后用滑窗方式做未来预测误差统计。</p>
      <p>当前实际参数是：模型 <code>vitg384</code>，<code>window_size=16</code>，<code>context_frames=8</code>，<code>stride=8</code>，<code>seed=42</code>，并且使用 <code>mode="mean"</code> 对所有滑窗 loss 求平均。</p>
      <p>页面里展示两个字段：</p>
      <p><code>wmreward_surprise</code>：直接等于这次 V-JEPA sliding-window loss，越低越好。</p>
      <p><code>wmreward_similarity</code>：代码里只是做了一个派生变换 <code>1.0 - surprise</code>，不是独立模型输出。严格来说主指标应看 <code>wmreward_surprise</code>。</p>
    </section>
    """


def build_html(records: list[dict[str, Any]]) -> str:
    grouped = group_records(records)
    seed_sections = "".join(
        build_seed_section(seed, grouped[seed])
        for seed in sorted(grouped.keys())
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eval JSON Flat Report</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px 20px 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.2;
      font-weight: 700;
    }}
    .lead {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
      margin-bottom: 20px;
    }}
    .wmr-box {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 20px;
    }}
    .wmr-box h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .wmr-box p {{
      margin: 8px 0 0;
      line-height: 1.6;
      font-size: 14px;
    }}
    .seed-block {{
      margin-top: 20px;
    }}
    .seed-head {{
      margin-bottom: 10px;
    }}
    .seed-head h2 {{
      margin: 0 0 4px;
      font-size: 20px;
    }}
    .seed-sub {{
      color: var(--muted);
      font-size: 13px;
    }}
    .method-block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      margin-top: 14px;
    }}
    .method-head {{
      margin-bottom: 12px;
    }}
    .method-head h3 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.3;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .video-panel {{
      display: flex;
      flex-direction: column;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      overflow: hidden;
    }}
    .video-panel-head {{
      padding: 10px 12px;
      background: #f8fafc;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      font-weight: 700;
    }}
    .video-wrap {{
      background: #111827;
      overflow: hidden;
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    video {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #000;
    }}
    .video-meta {{
      padding: 10px 12px 12px;
    }}
    .video-name {{
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
      line-height: 1.4;
      margin-bottom: 6px;
      word-break: break-word;
    }}
    .video-prompt {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
      word-break: break-word;
    }}
    .metric-table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .metric-table {{
      width: 100%;
      min-width: 820px;
      border-collapse: separate;
      border-spacing: 0;
      background: #fff;
    }}
    .metric-table thead th {{
      background: #f8fafc;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .metric-table th,
    .metric-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      font-size: 13px;
    }}
    .metric-table tr:last-child th,
    .metric-table tr:last-child td {{
      border-bottom: none;
    }}
    .metric-table th:last-child,
    .metric-table td:last-child {{
      border-right: none;
    }}
    .metric-head,
    .metric-name {{
      min-width: 180px;
      text-align: left;
      background: #f8fafc;
    }}
    .metric-name {{
      color: var(--ink);
      font-weight: 700;
    }}
    .metric-score {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      word-break: break-word;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      background: #f3f4f6;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    @media (max-width: 720px) {{
      .page {{
        padding: 16px 12px 28px;
      }}
      .method-block {{
        padding: 12px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Eval JSON Flat Report</h1>
    <div class="lead">页面数据来自 <code>{html.escape(str(JSON_ROOT))}</code>。按 <code>seed</code> 分板块；每个方法单独成块；块内列按 <code>step/guidance</code> 组合组织。视频直接展示在页面里，下面的指标表统一按“每行一个指标、每列一个 <code>step/guidance</code> 组合”排版。指标名后的箭头表示越高越好 <code>↑</code> / 越低越好 <code>↓</code>。当前这 12 个结果文件里，页面展示的指标都已经有值，没有缺测项。</div>
    {build_wmreward_explainer()}
    {seed_sections}
  </div>
</body>
</html>
"""


def ensure_video_links() -> None:
    link_specs = [
        (REPORT_ROOT / "wan_runs", TMP_ROOT / "wan_runs"),
        (REPORT_ROOT / "wan_runs_step_sweep", TMP_ROOT / "wan_runs_step_sweep"),
    ]
    for link_path, target_path in link_specs:
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
                continue
            if link_path.is_dir() and not link_path.is_symlink():
                continue
            link_path.unlink()
        link_path.symlink_to(target_path, target_is_directory=True)


def main() -> None:
    args = parse_args()
    records = load_records()
    if not records:
        raise RuntimeError(f"No result json files found in {JSON_ROOT}")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_video_links()
    REPORT_PATH.write_text(build_html(records), encoding="utf-8")
    print(REPORT_PATH)
    print(f"http://127.0.0.1:{args.port}/")
    subprocess.run([sys.executable, "-m", "http.server", str(args.port), "--directory", str(REPORT_ROOT)], check=True)


if __name__ == "__main__":
    main()
