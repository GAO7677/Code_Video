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
DEFAULT_JEPA_SUMMARY = Path(
    "/data/gaoya/AAA_test_video/Output_try0526/runs/pdi_jepa_eval_demo/report/summary.json"
)
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
    parser.add_argument("--jepa_summary_json", type=Path, default=DEFAULT_JEPA_SUMMARY)
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


def stage_report_asset(run_root: Path, target: str | None) -> str:
    if not target:
        return ""
    src = Path(target)
    if not src.is_file():
        return ""
    parent_name = src.parent.name or "asset"
    dst = run_root / "report_assets" / parent_name / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return os.path.relpath(dst, run_root / "report")


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


def build_jepa_index(jepa_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["case_id"], row["provider"]): row for row in jepa_payload.get("results", [])}


def score_text(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def provider_card(
    run_root: Path,
    row: dict[str, Any],
    proxy_row: dict[str, Any] | None,
    jepa_row: dict[str, Any] | None,
) -> str:
    provider_name = row["provider"].upper()
    proxy_score = proxy_row.get("geometry_score") if proxy_row else None
    proxy_details = proxy_row.get("geometry_details", {}) if proxy_row else {}
    jepa_score = jepa_row.get("jepa_score") if jepa_row else None
    jepa_details = jepa_row.get("jepa_details", {}) if jepa_row else {}
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
              <div class="metric">
                <span class="label">JEPA 分数 ↑</span>
                <strong>{score_text(jepa_score)}</strong>
              </div>
              <div class="metric">
                <span class="label">Tok Cos ↑</span>
                <strong>{score_text(jepa_details.get('predictive_alignment'))}</strong>
              </div>
              <div class="metric">
                <span class="label">Rel Err ↓</span>
                <strong>{score_text(jepa_details.get('temporal_relation_error'))}</strong>
              </div>
              <div class="metric">
                <span class="label">Delta Err ↓</span>
                <strong>{score_text(jepa_details.get('delta_l2'))}</strong>
              </div>
            </div>
          </section>
        </div>
        <div class="mini-meta">
          <span>轨迹点数: {tracks if tracks is not None else '-'}</span>
          <span>JEPA context: {html.escape(str(jepa_details.get('context_mode', '-')))}</span>
        </div>
        {context_preview_html(run_root, jepa_details)}
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


def context_preview_html(run_root: Path, jepa_details: dict[str, Any]) -> str:
    context_montage_path = jepa_details.get("context_montage_path")
    if not context_montage_path:
        return ""
    rel_path = stage_report_asset(run_root, context_montage_path)
    if not rel_path:
        return ""
    rel = html.escape(rel_path)
    prefix_frames = jepa_details.get("context_prefix_frames", "-")
    return f"""
      <div class="context-strip">
        <div class="context-strip-title">JEPA Context 帧（前 {html.escape(str(prefix_frames))} 帧）</div>
        <a class="context-strip-link" href="{rel}">
          <img src="{rel}" alt="JEPA context frames" />
        </a>
      </div>
    """


def link_html(run_root: Path, path: str | None, label: str) -> str:
    if not path:
        return ""
    rel = html.escape(relpath_from_report(run_root, path))
    safe_label = html.escape(label)
    return f"<a href=\"{rel}\">{safe_label}</a>"


def metrics_table(
    rows: list[dict[str, Any]],
    proxy_index: dict[tuple[str, str], dict[str, Any]],
    jepa_index: dict[tuple[str, str], dict[str, Any]],
) -> str:
    header = """
      <div class="table-wrap">
        <table class="metric-table">
          <thead>
            <tr>
              <th rowspan="2">方法</th>
              <th rowspan="2">等级</th>
              <th colspan="4">官方</th>
              <th colspan="9">自定义</th>
            </tr>
            <tr>
              <th>PDI ↓</th>
              <th>尺度 ε ↓</th>
              <th>刚性 ε ↓</th>
              <th>VP ε ↓</th>
              <th>代理分数 ↑</th>
              <th>总误差 ↓</th>
              <th>尺度误差 ↓</th>
              <th>刚性误差 ↓</th>
              <th>VP 误差 ↓</th>
              <th>JEPA ↑</th>
              <th>Tok Cos ↑</th>
              <th>Rel Err ↓</th>
              <th>Delta Err ↓</th>
            </tr>
          </thead>
          <tbody>
    """
    body_rows: list[str] = []
    for row in rows:
        proxy_row = proxy_index.get((row["case_id"], row["provider"]))
        proxy_details = proxy_row.get("geometry_details", {}) if proxy_row else {}
        jepa_row = jepa_index.get((row["case_id"], row["provider"]))
        jepa_details = jepa_row.get("jepa_details", {}) if jepa_row else {}
        body_rows.append(
            f"""
            <tr>
              <td>{html.escape(str(row["provider"]).upper())}</td>
              <td>{html.escape(str(row.get("grade", "-")))}</td>
              <td>{score_text(row.get("pdi_score"))}</td>
              <td>{score_text(row.get("scale_component"))}</td>
              <td>{score_text(row.get("epsilon_rigidity"))}</td>
              <td>{score_text(row.get("vp_component"))}</td>
              <td>{score_text(proxy_row.get("geometry_score") if proxy_row else None)}</td>
              <td>{score_text(proxy_details.get("proxy_error_total"))}</td>
              <td>{score_text(proxy_details.get("scale_error"))}</td>
              <td>{score_text(proxy_details.get("rigidity_error"))}</td>
              <td>{score_text(proxy_details.get("vp_error"))}</td>
              <td>{score_text(jepa_row.get("jepa_score") if jepa_row else None)}</td>
              <td>{score_text(jepa_details.get("predictive_alignment"))}</td>
              <td>{score_text(jepa_details.get("temporal_relation_error"))}</td>
              <td>{score_text(jepa_details.get("delta_l2"))}</td>
            </tr>
            """
        )
    footer = """
          </tbody>
        </table>
      </div>
    """
    return header + "".join(body_rows) + footer


def generate_html(
    results: list[dict[str, Any]],
    run_root: Path,
    port: int,
    proxy_payload: dict[str, Any],
    jepa_payload: dict[str, Any],
) -> Path:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row["case_id"], []).append(row)

    case_meta = build_case_meta(proxy_payload)
    proxy_index = build_proxy_index(proxy_payload)
    jepa_index = build_jepa_index(jepa_payload)
    provider_order = {"gt": 0, "wan": 1, "vace": 2}

    sections = []
    for case_id in sorted(grouped):
        rows = sorted(grouped[case_id], key=lambda row: provider_order.get(row["provider"], 99))
        meta = case_meta.get(case_id, {})
        first_frame = meta.get("first_frame_path")
        gt_video = meta.get("gt_video_path")
        cards = []
        for row in rows:
            cards.append(
                provider_card(
                    run_root,
                    row,
                    proxy_index.get((case_id, row["provider"])),
                    jepa_index.get((case_id, row["provider"])),
                )
            )
        table_html = metrics_table(rows, proxy_index, jepa_index)
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
              {table_html}
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
    .table-wrap {{
      overflow-x: auto;
      margin-bottom: 18px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.8);
    }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
    }}
    .metric-table th,
    .metric-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: center;
      font-size: 13px;
      white-space: nowrap;
    }}
    .metric-table thead th {{
      background: rgba(245, 238, 227, 0.95);
      color: #4d4438;
      font-weight: 700;
    }}
    .metric-table thead tr:first-child th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric-table tbody tr:nth-child(odd) {{
      background: rgba(255, 251, 245, 0.82);
    }}
    .metric-table tbody tr:nth-child(even) {{
      background: rgba(250, 246, 238, 0.62);
    }}
    .metric-table tbody tr:hover {{
      background: rgba(239, 230, 214, 0.72);
    }}
    .metric-table th:first-child,
    .metric-table td:first-child {{
      text-align: left;
      font-weight: 700;
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
    .context-strip {{
      margin-bottom: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: rgba(250, 246, 238, 0.75);
    }}
    .context-strip-title {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .context-strip-link {{
      display: block;
      text-decoration: none;
      color: inherit;
    }}
    .context-strip img {{
      width: 100%;
      display: block;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #f0ece4;
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
        <h3>1. 官方指标怎么计算</h3>
        <p>
          <strong>PDI ↓</strong>：PDI-Bench 官方总误差，综合下面几个物理几何误差，<strong>越低越好</strong>。<br />
          <strong>尺度 ε ↓</strong>：比较物体图像尺度变化和深度变化是否匹配，防止物体无故突然变大或变小。<br />
          <strong>刚性 ε ↓</strong>：检查目标内部结构是否稳定，防止“呼吸感”、拉伸、局部变形。<br />
          <strong>VP ε ↓</strong>：比较前景运动方向和背景透视消失点是否一致，防止 3D 运动方向不合理。<br />
          <strong>等级 A / B / C / F</strong>：官方按 PDI 总分给出的粗粒度质量分档。
        </p>
      </article>
      <article class="explain-card">
        <h3>2. 自定义几何指标怎么计算</h3>
        <p>
          <strong>尺度误差 ↓</strong>：用 SAM2 掩码和单目深度 proxy 估计目标尺度与深度的对应关系，越不匹配误差越大。<br />
          <strong>刚性误差 ↓</strong>：用 CoTracker 轨迹检查目标内部点对距离是否稳定，越像刚体误差越小。<br />
          <strong>VP 误差 ↓</strong>：比较前景轨迹方向和背景透视方向，方向越冲突误差越大。<br />
          <strong>代理总误差 ↓</strong>：以上三个误差的加权和。<br />
          <strong>代理分数 ↑</strong>：把代理总误差做 <code>exp(-error)</code> 映射后的分数，所以误差更小一定会对应更高分。
        </p>
      </article>
      <article class="explain-card">
        <h3>3. JEPA 指标怎么计算</h3>
        <p>
          <strong>JEPA 分数 ↑</strong>：我们现在采用的是参考 PhysAlign 整理出的 JEPA 时空关系分数，由 token 预测对齐、时间关系矩阵一致性和时间差分一致性加权得到。<br />
          <strong>Tok Cos ↑</strong>：预测 future token 和真实 future token 的逐 token 余弦相似度均值，越高说明预测内容更贴近真实未来。<br />
          <strong>Rel Err ↓</strong>：先把 future 特征按时间聚合，再按 PhysAlign 的思路构建时间关系 Gram 矩阵；预测矩阵和真实矩阵的 margin-L1 差越小越好。<br />
          <strong>Delta Err ↓</strong>：比较预测时间特征和真实时间特征的一阶差分 L2，越低说明速度 / 节奏变化更接近真实未来。<br />
          当前 demo 的 JEPA context 是 <strong>gt_prefix_video</strong>，也就是用 GT 前缀 clip 作为所有方法共享的真实上下文，再分别比较各方法后续 future，因此比重复首帧更接近真实多帧条件；但它仍然是 demo 级 proxy，不等同于完整 benchmark 的最终绝对物理评测。
        </p>
      </article>
      <article class="explain-card">
        <h3>4. 这一页怎么读</h3>
        <p>
          这页里所有带 <strong>↓</strong> 的量都是误差，都是 <strong>越低越好</strong>；所有带 <strong>↑</strong> 的量都是分数或相似度，都是 <strong>越高越好</strong>。<br />
          比较时最好优先看 <strong>同一个 case 内</strong> 的 GT、Wan、VACE 排序，因为不同 case 的运动难度本来就不一样。<br />
          如果官方 PDI 和自定义 / JEPA 排序不一致，优先说明这些 proxy 还没有完全学到官方或人工偏好的物理标准，这正是后续 rerank 方案要继续验证和校准的部分。
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
    jepa_payload = load_proxy_payload(args.jepa_summary_json) if args.jepa_summary_json.is_file() else {"results": []}

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

    html_path = generate_html(results, run_root, args.serve_port, proxy_payload, jepa_payload)
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
