#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_SUMMARY = Path(
    "/data/gaoya/AAA_test_video/Output_try0526/runs/pdi_proxy_eval_demo/report/summary.json"
)
DEFAULT_PROXY_SUMMARY = DEFAULT_INPUT_SUMMARY
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526")
DEFAULT_PDI_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/PDI-Bench-main")


@dataclass(frozen=True)
class EvalItem:
    case_id: str
    provider: str
    target_object: str
    prompt: str
    video_path: Path
    gt_video_path: Path | None

    @property
    def unique_stem(self) -> str:
        return f"{self.case_id}__{self.provider}"


def sys_executable() -> str:
    return os.environ.get("PYTHON", "") or shutil.which("python") or "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official PDI-Bench on staged Wan/VACE/GT videos.")
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_INPUT_SUMMARY)
    parser.add_argument("--proxy_summary_json", type=Path, default=DEFAULT_PROXY_SUMMARY)
    parser.add_argument("--pdi_root", type=Path, default=DEFAULT_PDI_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", default="pdi_official_eval_demo")
    parser.add_argument("--providers", nargs="+", default=["wan", "vace", "gt"])
    parser.add_argument("--case_ids", nargs="*", default=None)
    parser.add_argument("--python_bin", default=sys_executable())
    parser.add_argument("--florence_model", type=Path, default=Path("/data/gaoya/ckpt/microsoft-Florence-2-base"))
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--serve_port", type=int, default=8878)
    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Force rerunning official PDI even if report files already exist.")
    return parser.parse_args()


def load_proxy_payload(summary_json: Path) -> dict[str, Any]:
    return json.loads(summary_json.read_text(encoding="utf-8"))


def load_items(proxy_payload: dict[str, Any], providers: set[str], case_filter: set[str] | None) -> list[EvalItem]:
    items: list[EvalItem] = []
    for row in proxy_payload["results"]:
        provider = row["provider"]
        case_id = row["case_id"]
        if provider not in providers:
            continue
        if case_filter is not None and case_id not in case_filter:
            continue
        items.append(
            EvalItem(
                case_id=case_id,
                provider=provider,
                target_object=row["target_object"],
                prompt=row["prompt"],
                video_path=Path(row["video_path"]),
                gt_video_path=Path(row["gt_video_path"]) if row.get("gt_video_path") else None,
            )
        )
    return items


def stage_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")

    def extract(pattern: str, cast: type | None = None) -> Any:
        match = re.search(pattern, text)
        if not match:
            return None
        value = match.group(1)
        return cast(value) if cast else value

    return {
        "pdi_score": extract(r"FINAL PDI SCORE:\s*([0-9.]+)", float),
        "grade": extract(r"OVERALL GRADE:\s*([A-Z+-]+)"),
        "scale_component": extract(r"Scale Component .*?:\s*([0-9.]+)", float),
        "traj_component": extract(r"Trajectory Component .*?:\s*([0-9.]+)", float),
        "epsilon_rigidity": extract(r"Epsilon Rigidity:\s*([0-9.]+)", float),
        "rigidity_strategy": extract(r"Rigidity Strategy:\s*(.+)"),
        "vp_component": extract(r"VP Component .*?:\s*([0-9.]+)", float),
        "ra_math_pass": extract(r"RA Math Pass:\s*(True|False)"),
        "ra_ground_rmse": extract(r"RA Ground RMSE:\s*([0-9.eE+-]+)", float),
        "ra_scale_jump": extract(r"RA Scale Jump:\s*([0-9.eE+-]+)", float),
        "ra_reproj_err": extract(r"RA Reproj Err:\s*([0-9.eE+-]+)", float),
        "ra_overall_pass": extract(r"RA Overall Pass:\s*(True|False)"),
    }


def find_first(patterns: list[str], directory: Path) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def discover_artifacts(report_dir: Path, unique_stem: str) -> dict[str, str | None]:
    curves = find_first([f"{unique_stem}_error_plot.png", f"{unique_stem}_error_curves.png"], report_dir)
    volume = find_first([f"{unique_stem}_volume_plot.png", f"{unique_stem}_volume_stability.png"], report_dir)
    mask = find_first([f"{unique_stem}_mask_frame*.png", f"{unique_stem}_mask_sample.png"], report_dir)
    return {
        "curves_png": str(curves) if curves else None,
        "volume_png": str(volume) if volume else None,
        "mask_png": str(mask) if mask else None,
    }


def build_result_record(item: EvalItem, staged_video: Path, report_dir: Path, report_path: Path) -> dict[str, Any]:
    return {
        "case_id": item.case_id,
        "provider": item.provider,
        "target_object": item.target_object,
        "prompt": item.prompt,
        "original_video_path": str(item.video_path),
        "staged_video_path": str(staged_video),
        "report_dir": str(report_dir),
        "report_path": str(report_path),
        **discover_artifacts(report_dir, item.unique_stem),
        **parse_report(report_path),
    }


def normalize_existing_result(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    report_dir = Path(row["report_dir"])
    normalized.update(discover_artifacts(report_dir, f"{row['case_id']}__{row['provider']}"))
    return normalized


def run_single_eval(
    item: EvalItem,
    pdi_root: Path,
    run_root: Path,
    python_bin: str,
    florence_model: Path,
    cuda_visible_devices: str | None,
    refresh: bool,
) -> dict[str, Any]:
    staged_video = run_root / "staged_inputs" / f"{item.unique_stem}.mp4"
    output_dir = run_root / "official_outputs" / item.case_id / item.provider
    report_dir = output_dir / item.unique_stem
    report_path = report_dir / f"{item.unique_stem}_pdi_report.txt"

    stage_video(item.video_path.resolve(), staged_video)

    if report_path.exists() and not refresh:
        return build_result_record(item, staged_video, report_dir, report_path)

    cmd = [
        python_bin,
        "evaluation/main.py",
        "--input",
        str(staged_video),
        "--text",
        item.target_object,
        "--output_dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pdi_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PDI_FLORENCE_MODEL_ID"] = str(florence_model)
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.run(cmd, cwd=pdi_root, env=env, check=True)
    return build_result_record(item, staged_video, report_dir, report_path)


def relpath_from_report(run_root: Path, target: str | None) -> str:
    if not target:
        return ""
    return os.path.relpath(target, run_root / "report")


def build_case_meta(proxy_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for case in proxy_payload.get("cases", []):
        meta[case["case_id"]] = dict(case)
    for row in proxy_payload.get("results", []):
        case_meta = meta.setdefault(row["case_id"], {"case_id": row["case_id"]})
        if row.get("first_frame_path") and "first_frame_path" not in case_meta:
            case_meta["first_frame_path"] = row["first_frame_path"]
        if row.get("gt_video_path") and "gt_video_path" not in case_meta:
            case_meta["gt_video_path"] = row["gt_video_path"]
        case_meta.setdefault("prompt", row.get("prompt"))
        case_meta.setdefault("target_object", row.get("target_object"))
    return meta


def build_proxy_index(proxy_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["case_id"], row["provider"]): row for row in proxy_payload.get("results", [])}


def score_text(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def provider_card(run_root: Path, row: dict[str, Any], proxy_row: dict[str, Any] | None) -> str:
    provider_name = row["provider"].upper()
    proxy_score = proxy_row.get("geometry_score") if proxy_row else None
    proxy_details = proxy_row.get("geometry_details", {}) if proxy_row else {}
    tracks = proxy_details.get("track_count")
    proxy_scale_error = proxy_details.get("scale_error")
    proxy_rigidity_error = proxy_details.get("rigidity_error")
    proxy_vp_error = proxy_details.get("vp_error")
    proxy_error_total = proxy_details.get("proxy_error_total")
    return f"""
      <article class="provider-card">
        <div class="provider-head">
          <h3>{provider_name}</h3>
          <div class="grade-badge grade-{html.escape(str(row.get('grade', 'na')).lower())}">{html.escape(str(row.get('grade', '-')))}</div>
        </div>
        <video controls preload="metadata" src="{html.escape(relpath_from_report(run_root, row['staged_video_path']))}"></video>
        <div class="metric-panels">
          <section class="metric-panel official-panel">
            <div class="panel-title">官方</div>
            <div class="score-grid">
              <div class="metric">
                <span class="label">PDI 分数 ↓</span>
                <strong>{score_text(row.get('pdi_score'))}</strong>
              </div>
              <div class="metric">
                <span class="label">尺度误差 ε ↓</span>
                <strong>{score_text(row.get('scale_component'))}</strong>
              </div>
              <div class="metric">
                <span class="label">刚性误差 ε ↓</span>
                <strong>{score_text(row.get('epsilon_rigidity'))}</strong>
              </div>
              <div class="metric">
                <span class="label">VP 误差 ε ↓</span>
                <strong>{score_text(row.get('vp_component'))}</strong>
              </div>
            </div>
          </section>
          <section class="metric-panel custom-panel">
            <div class="panel-title">自定义</div>
            <div class="score-grid">
              <div class="metric">
                <span class="label">代理分数 ↑</span>
                <strong>{score_text(proxy_score)}</strong>
              </div>
              <div class="metric">
                <span class="label">代理总误差 ↓</span>
                <strong>{score_text(proxy_error_total)}</strong>
              </div>
              <div class="metric">
                <span class="label">尺度误差 ↓</span>
                <strong>{score_text(proxy_scale_error)}</strong>
              </div>
              <div class="metric">
                <span class="label">刚性误差 ↓</span>
                <strong>{score_text(proxy_rigidity_error)}</strong>
              </div>
              <div class="metric">
                <span class="label">VP 误差 ↓</span>
                <strong>{score_text(proxy_vp_error)}</strong>
              </div>
            </div>
          </section>
        </div>
        <div class="mini-meta">
          <span>轨迹点数: {tracks if tracks is not None else '-'}</span>
        </div>
        <div class="thumb-grid">
          {thumb_html(run_root, row.get('mask_png'), '分割掩码')}
          {thumb_html(run_root, row.get('curves_png'), '误差曲线')}
          {thumb_html(run_root, row.get('volume_png'), '体积曲线')}
        </div>
        <div class="links">
          {link_html(run_root, row.get('report_path'), '文字报告')}
          {link_html(run_root, row.get('curves_png'), '误差曲线')}
          {link_html(run_root, row.get('volume_png'), '体积曲线')}
          {link_html(run_root, row.get('mask_png'), '分割掩码')}
        </div>
      </article>
    """


def thumb_html(run_root: Path, path: str | None, label: str) -> str:
    if not path:
        return ""
    rel = html.escape(relpath_from_report(run_root, path))
    safe_label = html.escape(label)
    return f"""
      <a class="thumb" href="{rel}">
        <img src="{rel}" alt="{safe_label}" />
        <span>{safe_label}</span>
      </a>
    """


def link_html(run_root: Path, path: str | None, label: str) -> str:
    if not path:
        return ""
    rel = html.escape(relpath_from_report(run_root, path))
    safe_label = html.escape(label)
    return f"<a href=\"{rel}\">{safe_label}</a>"


def generate_html(
    results: list[dict[str, Any]],
    run_root: Path,
    port: int,
    proxy_payload: dict[str, Any],
) -> Path:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row["case_id"], []).append(row)

    case_meta = build_case_meta(proxy_payload)
    proxy_index = build_proxy_index(proxy_payload)
    provider_order = {"gt": 0, "wan": 1, "vace": 2}

    sections = []
    for case_id in sorted(grouped):
        rows = sorted(grouped[case_id], key=lambda row: provider_order.get(row["provider"], 99))
        meta = case_meta.get(case_id, {})
        first_frame = meta.get("first_frame_path")
        gt_video = meta.get("gt_video_path")
        cards = []
        for row in rows:
            cards.append(provider_card(run_root, row, proxy_index.get((case_id, row["provider"]))))
        sections.append(
            f"""
            <section class="case-card">
              <div class="case-header">
                <div class="case-copy">
                  <div class="eyebrow">{html.escape(meta.get('category', case_id))}</div>
                  <h2>{html.escape(case_id)}</h2>
                  <div class="input-block">
                    <div><span class="input-label">目标</span> {html.escape(str(meta.get('target_object', '-')))}</div>
                    <div><span class="input-label">提示词</span> {html.escape(str(meta.get('prompt', '-')))}</div>
                  </div>
                </div>
                <div class="reference-panel">
                  <div class="reference-title">输入参考</div>
                  {reference_image(run_root, first_frame)}
                  <div class="reference-links">
                    {link_html(run_root, gt_video, 'GT 视频')}
                  </div>
                </div>
              </div>
              <div class="provider-grid">
                {''.join(cards)}
              </div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PDI 官方评测</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: rgba(255, 251, 244, 0.92);
      --line: #d5c8b4;
      --text: #1d1b18;
      --muted: #6d665c;
      --accent: #9f4f30;
      --good: #216e39;
      --warn: #9a6700;
      --bad: #a40e26;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(159,79,48,0.10), transparent 24%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    .sub {{
      color: var(--muted);
      margin: 0 0 22px;
      line-height: 1.5;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 22px;
    }}
    .legend span {{
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,0.6);
      font-size: 13px;
    }}
    .explain-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .explain-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      background: rgba(255,255,255,0.72);
    }}
    .explain-card h3 {{
      margin: 0 0 10px;
      font-size: 17px;
    }}
    .explain-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
    }}
    .case-card {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 24px;
      padding: 22px;
      margin-bottom: 22px;
      box-shadow: 0 20px 60px rgba(70, 52, 30, 0.08);
    }}
    .case-header {{
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(260px, 0.9fr);
      gap: 20px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 28px;
    }}
    .input-block {{
      display: grid;
      gap: 10px;
      font-size: 14px;
      line-height: 1.5;
    }}
    .input-label {{
      color: var(--muted);
      display: inline-block;
      min-width: 64px;
      font-weight: 600;
    }}
    .reference-panel {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.68);
    }}
    .reference-title {{
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .reference-panel img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid var(--line);
      object-fit: cover;
    }}
    .reference-links, .links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .reference-links a, .links a {{
      color: #0e5a8a;
      text-decoration: none;
      font-size: 13px;
    }}
    .provider-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
    }}
    .provider-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.78);
    }}
    .provider-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }}
    h3 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0.02em;
    }}
    .grade-badge {{
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid currentColor;
    }}
    .grade-a {{ color: var(--good); }}
    .grade-b {{ color: #4e6f16; }}
    .grade-c {{ color: var(--warn); }}
    .grade-f {{ color: var(--bad); }}
    .grade-na {{ color: var(--muted); }}
    video {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #000;
      margin-bottom: 12px;
    }}
    .score-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 10px;
    }}
    .metric-panels {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .metric-panel {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
    }}
    .official-panel {{
      background: rgba(248, 245, 237, 0.92);
    }}
    .custom-panel {{
      background: rgba(252, 248, 241, 0.92);
    }}
    .panel-title {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      background: rgba(250,246,238,0.88);
    }}
    .metric .label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric strong {{
      font-size: 18px;
    }}
    .mini-meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }}
    .thumb-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .thumb {{
      display: block;
      text-decoration: none;
      color: inherit;
    }}
    .thumb img {{
      width: 100%;
      aspect-ratio: 1.2 / 1;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid var(--line);
      display: block;
      background: #f0ece4;
    }}
    .thumb span {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
    }}
    @media (max-width: 1100px) {{
      .provider-grid {{ grid-template-columns: 1fr; }}
      .case-header {{ grid-template-columns: 1fr; }}
      .explain-grid {{ grid-template-columns: 1fr; }}
      .metric-panels {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>PDI 官方评测</h1>
    <p class="sub">
      本地页面地址：<code>http://127.0.0.1:{port}/report/index.html</code><br />
      官方 PDI 以及各个 epsilon 误差项都遵循 <strong>↓ 越低越好</strong>；我们自己的代理分数遵循 <strong>↑ 越高越好</strong>。
    </p>
    <div class="legend">
      <span>按 case 分组展示</span>
      <span>同一 case 的 GT / Wan / VACE 并排对比</span>
      <span>首页直接展示提示词、目标、首帧、视频、掩码和曲线</span>
      <span>官方 PDI ↓ 越低越好</span>
      <span>代理分数 ↑ 越高越好</span>
    </div>
    <section class="explain-grid">
      <article class="explain-card">
        <h3>1. 官方 PDI ↓ 是什么</h3>
        <p>
          官方 PDI 是 PDI-Bench 给出的物理几何一致性分数，用来检查目标物体在时间维度上是否保持合理的透视、尺度和结构。
          它会综合目标物体的尺度变化是否和深度变化一致、运动轨迹是否合理、结构是否保持刚性、以及前景运动是否和背景透视一致。
          这个分数是误差型指标，所以 <strong>越低越好</strong>：越接近 0，说明视频越符合物理和几何直觉；分数越大，说明几何失真越严重。
        </p>
      </article>
      <article class="explain-card">
        <h3>2. 代理分数 ↑ 是怎么来的</h3>
        <p>
          代理分数是我们自己实现的轻量几何评分，不直接依赖完整的官方 3D 重建流程。
          它主要利用 SAM2 分割、CoTracker 点轨迹和单目深度 proxy 来衡量三件事：尺度一致性、刚性稳定性、以及前景运动和背景透视的一致性。
          现在它先计算三个 <strong>误差项</strong>，再做加权求和得到 <strong>代理总误差 ↓</strong>，最后用 <code>exp(-代理总误差)</code> 映射成 <strong>代理分数 ↑</strong>。
          这样如果同一个视频的三个代理误差都更低，代理分数一定更高，不会再出现方向混乱。
        </p>
      </article>
      <article class="explain-card">
        <h3>3. 几个子指标分别表示什么</h3>
        <p>
          <strong>尺度误差 Scale ε</strong>：看物体在图像里变大变小，是否和它的深度前后变化相匹配。<br />
          <strong>刚性误差 Rigidity ε</strong>：看物体内部结构是否稳定，是否出现“呼吸感”、拉伸、局部形变。<br />
          <strong>消失点误差 VP ε</strong>：看前景物体的运动方向，是否和背景场景的透视关系一致。<br />
          这些都是误差项，因此统一是 <strong>越低越好</strong>。
        </p>
      </article>
      <article class="explain-card">
        <h3>4. 等级 A / B / C / F 怎么理解</h3>
        <p>
          这是官方 PDI 分数对应的粗粒度解释。<strong>A</strong> 表示物理和几何上都比较紧，整体可信；<strong>B</strong> 表示有轻微抖动或小问题；
          <strong>C</strong> 表示已经能看到明显失真；<strong>F</strong> 表示几何一致性已经明显崩坏。
          实际比较时，最好优先在 <strong>同一个 case 内部</strong> 比较不同模型，因为不同 case 的运动难度本来就不一样。
        </p>
      </article>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""
    path = run_root / "report" / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path


def reference_image(run_root: Path, path: str | None) -> str:
    if not path:
        return "<div class=\"metric\">No first frame found.</div>"
    rel = html.escape(relpath_from_report(run_root, path))
    return f"<img src=\"{rel}\" alt=\"reference frame\" />"


def main() -> None:
    args = parse_args()
    run_root = args.output_root / "runs" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "report" / "summary.json"

    proxy_payload = load_proxy_payload(args.proxy_summary_json)

    if args.render_only:
        if not summary_path.exists():
            raise SystemExit(f"Official summary not found: {summary_path}")
        results = [normalize_existing_result(row) for row in json.loads(summary_path.read_text(encoding="utf-8"))]
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        providers = set(args.providers)
        case_filter = set(args.case_ids) if args.case_ids else None
        items = load_items(proxy_payload, providers, case_filter)
        if not items:
            raise SystemExit("No evaluation items matched the requested filters.")

        results = []
        for item in items:
            results.append(
                run_single_eval(
                    item,
                    args.pdi_root.resolve(),
                    run_root,
                    args.python_bin,
                    args.florence_model.resolve(),
                    args.cuda_visible_devices,
                    args.refresh,
                )
            )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    html_path = generate_html(results, run_root, args.serve_port, proxy_payload)
    print(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "html": str(html_path),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
