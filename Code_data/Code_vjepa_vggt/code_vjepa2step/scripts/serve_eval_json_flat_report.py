#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


TMP_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa2step/tmp")
JSON_ROOT = TMP_ROOT / "eval_json_flat"
REPORT_ROOT = TMP_ROOT / "eval_json_flat_report"
REPORT_PATH = REPORT_ROOT / "index.html"

VIDEO_FIELDS = [
    "video_name",
    "video_stem",
    "prompt",
]

PRIMARY_METRICS = [
    ("official_pdi", "Official PDI"),
    ("scale_component", "Scale"),
    ("traj_component", "Trajectory"),
    ("epsilon_rigidity", "Rigidity"),
    ("vp_component", "VP"),
    ("wmreward_surprise", "WMReward Surprise"),
    ("wmreward_similarity", "WMReward Similarity"),
    ("vjepa_proxy", "V-JEPA Proxy"),
    ("videophy2_auto_pc", "VideoPhy2 PC"),
    ("videophy2_auto_sa", "VideoPhy2 SA"),
    ("phyground_general_avg", "PhyGround Avg"),
    ("cosmos_reason1", "Cosmos Reason1"),
    ("fid", "FID"),
    ("fvd", "FVD"),
    ("accuracy", "Accuracy"),
    ("pearson_correlation", "Pearson"),
]

EXTRA_FIELDS = [
    "jepa_score",
    "cse",
    "tse",
    "official_pdi_error",
    "wmreward_error",
    "vjepa_proxy_error",
    "videophy2_auto_pc_error",
    "videophy2_auto_sa_error",
    "phyground_error",
    "cosmos_reason1_error",
    "jepa_error",
    "fid_error",
    "fvd_error",
    "cse_error",
    "tse_error",
    "fid_note",
    "fvd_note",
    "cse_note",
    "tse_note",
    "accuracy_note",
    "pearson_correlation_note",
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
    return os.path.relpath(target.resolve(), REPORT_ROOT.resolve()).replace("\\", "/")


def resolve_video_path(record: dict[str, Any]) -> Path:
    path = Path(record["video_path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def metric_table(record: dict[str, Any]) -> str:
    rows = []
    for key, label in PRIMARY_METRICS:
        rows.append(
            "<tr>"
            f"<td class='metric-label'>{html.escape(label)}</td>"
            f"<td class='metric-value'>{fv(record.get(key))}</td>"
            "</tr>"
        )
    return f"<table class='metrics'><tbody>{''.join(rows)}</tbody></table>"


def extra_table(record: dict[str, Any]) -> str:
    rows = []
    for key in EXTRA_FIELDS:
        value = record.get(key)
        if value is None:
            continue
        rows.append(
            "<tr>"
            f"<td class='extra-key'>{html.escape(key)}</td>"
            f"<td class='extra-value'>{fv(value)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<details class='extra-block'>"
        "<summary>More Fields</summary>"
        f"<table class='extra'><tbody>{''.join(rows)}</tbody></table>"
        "</details>"
    )


def build_card(record: dict[str, Any]) -> str:
    video_path = resolve_video_path(record)
    video_rel = rel_to_report(video_path)
    meta_lines = []
    for key in VIDEO_FIELDS:
        value = record.get(key)
        if value is None:
            continue
        meta_lines.append(
            f"<div><span class='meta-key'>{html.escape(key)}</span>: <span class='meta-val'>{fv(value)}</span></div>"
        )
    meta_lines.append(
        f"<div><span class='meta-key'>video_path</span>: <code>{html.escape(str(video_path))}</code></div>"
    )
    return f"""
    <article class="card">
      <div class="video-wrap">
        <video controls preload="metadata" src="{html.escape(video_rel)}"></video>
      </div>
      <div class="card-body">
        <h2>{html.escape(record.get("video_stem", "unknown"))}</h2>
        <div class="meta">{''.join(meta_lines)}</div>
        {metric_table(record)}
        {extra_table(record)}
      </div>
    </article>
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
    cards = "".join(build_card(record) for record in records)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eval JSON Flat Report</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffaf0;
      --ink: #1f1b17;
      --muted: #6b6157;
      --line: #d8c9b7;
      --accent: #9c5a2e;
      --accent-2: #2f6c63;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      background:
        radial-gradient(circle at top right, rgba(156,90,46,0.08), transparent 28%),
        linear-gradient(180deg, #f7f2e9 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 36px;
      line-height: 1.1;
      letter-spacing: 0.02em;
    }}
    .lead {{
      color: var(--muted);
      max-width: 1000px;
      font-size: 16px;
      line-height: 1.6;
      margin-bottom: 24px;
    }}
    .wmr-box {{
      background: rgba(255,250,240,0.88);
      border: 1px solid var(--line);
      border-left: 6px solid var(--accent);
      border-radius: 18px;
      padding: 18px 20px;
      margin-bottom: 26px;
      box-shadow: 0 10px 30px rgba(60,40,20,0.06);
    }}
    .wmr-box h2 {{
      margin: 0 0 10px;
      font-size: 22px;
    }}
    .wmr-box p {{
      margin: 8px 0;
      line-height: 1.65;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
      gap: 20px;
    }}
    .card {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 14px;
      background: rgba(255,250,240,0.92);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 14px;
      box-shadow: 0 12px 32px rgba(60,40,20,0.08);
    }}
    .video-wrap {{
      background: #e9dece;
      border-radius: 16px;
      overflow: hidden;
      min-height: 240px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    video {{
      width: 100%;
      height: auto;
      display: block;
      background: #000;
    }}
    .card-body h2 {{
      margin: 0 0 10px;
      font-size: 22px;
      line-height: 1.2;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      margin-bottom: 12px;
      word-break: break-word;
    }}
    .meta-key {{
      color: var(--accent-2);
      font-weight: 700;
    }}
    .metrics, .extra {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255,255,255,0.55);
      border-radius: 14px;
      overflow: hidden;
    }}
    .metrics td, .extra td {{
      padding: 7px 10px;
      border-bottom: 1px solid rgba(216,201,183,0.8);
      font-size: 13px;
      vertical-align: top;
    }}
    .metrics tr:last-child td, .extra tr:last-child td {{
      border-bottom: none;
    }}
    .metric-label, .extra-key {{
      color: var(--accent-2);
      font-weight: 700;
      width: 48%;
    }}
    .metric-value, .extra-value {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      word-break: break-word;
    }}
    .extra-block {{
      margin-top: 12px;
    }}
    .extra-block summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 8px;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 12px;
      background: rgba(47,108,99,0.08);
      padding: 1px 5px;
      border-radius: 6px;
    }}
    @media (max-width: 960px) {{
      .card {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Eval JSON Flat Report</h1>
    <div class="lead">页面数据来自 <code>{html.escape(str(JSON_ROOT))}</code>。每张卡片对应一个评测结果 JSON，展示视频预览和当前扁平字段里的指标分数。</div>
    {build_wmreward_explainer()}
    <section class="grid">
      {cards}
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    records = load_records()
    if not records:
        raise RuntimeError(f"No result json files found in {JSON_ROOT}")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_html(records), encoding="utf-8")
    print(REPORT_PATH)
    print(f"http://127.0.0.1:{args.port}/eval_json_flat_report/index.html")
    subprocess.run([sys.executable, "-m", "http.server", str(args.port), "--directory", str(TMP_ROOT)], check=True)


if __name__ == "__main__":
    main()
