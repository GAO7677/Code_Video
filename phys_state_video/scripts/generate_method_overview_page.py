#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_RUNS_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1")
DEFAULT_OUTPUT_ROOT = DEFAULT_RUNS_ROOT / "method_overview_v1"
DEFAULT_PORT = 18834

METHOD_SPECS = [
    {
        "id": "baseline_v1",
        "label": "baseline_v1 显式状态基线",
        "desc": "显式 future state 作为主条件，使用 best ckpt 导出的 case 页面。",
        "best_dir": DEFAULT_RUNS_ROOT / "industrial_s1_scale2_baseline_v1" / "viz" / "trained_cases_v1",
        "best_report": DEFAULT_RUNS_ROOT / "industrial_s1_scale2_baseline_v1" / "viz" / "trained_cases_v1" / "report.json",
        "timeline_dir": None,
    },
    {
        "id": "latent_v1",
        "label": "latent_v1 隐式 latent 条件版",
        "desc": "future latent tokens 作为主条件，同时保留显式 head 监督；展示 best ckpt 和训练中 ckpt 时间线页面。",
        "best_dir": DEFAULT_RUNS_ROOT / "industrial_s1_scale2_latent_v1" / "viz" / "training_ckpts" / "cases" / "adapter_best",
        "best_report": DEFAULT_RUNS_ROOT / "industrial_s1_scale2_latent_v1" / "viz" / "training_ckpts" / "cases" / "adapter_best" / "report.json",
        "timeline_dir": DEFAULT_RUNS_ROOT / "industrial_s1_scale2_latent_v1" / "viz" / "training_ckpts",
    },
    {
        "id": "latent_v2",
        "label": "latent_v2 latent-only 生成版",
        "desc": "显式 state 主要用于监督，视频生成主条件切换为 future latent tokens；展示 best ckpt 和训练中 ckpt 时间线页面。",
        "best_dir": DEFAULT_RUNS_ROOT / "industrial_s1_scale2_latent_v2" / "viz" / "training_ckpts" / "cases" / "adapter_best",
        "best_report": DEFAULT_RUNS_ROOT / "industrial_s1_scale2_latent_v2" / "viz" / "training_ckpts" / "cases" / "adapter_best" / "report.json",
        "timeline_dir": DEFAULT_RUNS_ROOT / "industrial_s1_scale2_latent_v2" / "viz" / "training_ckpts",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a unified method overview and best-ckpt comparison page.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def safe_read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def start_server(output_root: Path, port: int) -> int:
    log_path = output_root / f"http_{port}.log"
    pid_path = output_root / f"http_{port}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            return pid
        except Exception:
            pid_path.unlink(missing_ok=True)
    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(
            ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=str(output_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(1.0)
    return proc.pid


def metric_text(report: dict | None) -> str:
    if not report:
        return "暂无 report"
    preview = report.get("aggregate_preview", {})
    parts = []
    if preview.get("predictor_center_error_mean") is not None:
        parts.append(f"predictor center {preview['predictor_center_error_mean']:.3f}")
    if preview.get("video_center_error_mean") is not None:
        parts.append(f"video center {preview['video_center_error_mean']:.3f}")
    eval_metrics = report.get("eval_metrics", {})
    val_metrics = eval_metrics.get("val") if isinstance(eval_metrics, dict) else None
    if val_metrics and val_metrics.get("metrics"):
        parts.append(f"val loss {val_metrics['metrics']['loss']:.4f}")
    return " | ".join(parts) if parts else "暂无摘要指标"


def build_index_html(method_entries: list[dict], port: int) -> str:
    cards = []
    for method in method_entries:
        links = []
        if method.get("best_rel"):
            links.append(f'<a class="link" href="{html.escape(method["best_rel"])}/index.html">best ckpt case 页面</a>')
        else:
            links.append('<span class="pending">best ckpt case 页面尚未就绪</span>')
        if method.get("timeline_rel"):
            links.append(f'<a class="link secondary" href="{html.escape(method["timeline_rel"])}/index.html">训练中 ckpt 时间线</a>')
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">{html.escape(method['id'])}</div>
              <h2>{html.escape(method['label'])}</h2>
              <p class="desc">{html.escape(method['desc'])}</p>
              <p class="meta">{html.escape(method['metrics'])}</p>
              <div class="link-row">
                {''.join(links)}
              </div>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>方法总入口页</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      color: #1f1f1b;
      background: linear-gradient(180deg, #f8f3ea 0%, #ede2d3 100%);
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .card {{
      background: rgba(255, 252, 246, 0.95);
      border: 1px solid #dccfbe;
      border-radius: 18px;
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .card {{
      padding: 18px;
    }}
    .eyebrow {{
      color: #b8642a;
      text-transform: uppercase;
      font-size: 13px;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    .desc, .meta {{
      color: #6f675d;
      line-height: 1.7;
    }}
    .link-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 14px;
    }}
    .link {{
      color: #0f5a52;
      font-weight: 700;
      text-decoration: none;
    }}
    .secondary {{
      color: #7f4f28;
    }}
    .pending {{
      color: #9b8f82;
      font-weight: 700;
    }}
    .compare {{
      display: inline-block;
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>所有方法 case 入口</h1>
      <p>本页汇总当前所有方法版本的 case 可视化入口，并保留一个统一的 best ckpt 同批 case 对比页面。访问地址：<a href="http://127.0.0.1:{port}">http://127.0.0.1:{port}</a></p>
      <a class="link compare" href="compare_best/index.html">打开 best ckpt 同批 case 对比页</a>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>"""


def build_compare_html(methods: list[dict], reports: dict[str, dict]) -> str:
    summary_rows = []
    for method in methods:
        report = reports[method["id"]]
        preview = report.get("aggregate_preview", {})
        val_metrics = (report.get("eval_metrics") or {}).get("val") if isinstance(report.get("eval_metrics"), dict) else None
        summary_rows.append(
            f"""
            <tr>
              <td>{html.escape(method['label'])}</td>
              <td>{preview.get('predictor_center_error_mean', float('nan')):.4f}</td>
              <td>{preview.get('video_center_error_mean', float('nan')):.4f}</td>
              <td>{(val_metrics or {}).get('metrics', {}).get('loss', float('nan')):.4f}</td>
            </tr>
            """
        )

    base_cases = reports[methods[0]["id"]]["cases"]
    case_blocks = []
    for case_idx, base_case in enumerate(base_cases):
        case_id = base_case["case_id"]
        ref_row = f"""
        <div class="ref-grid">
          <section class="ref-card">
            <div class="ref-name">Context</div>
            <video controls preload="metadata" src="../{html.escape(methods[0]['best_rel'])}/{html.escape(base_case['context_video'])}"></video>
          </section>
          <section class="ref-card">
            <div class="ref-name">GT Future</div>
            <video controls preload="metadata" src="../{html.escape(methods[0]['best_rel'])}/{html.escape(base_case['gt_video'])}"></video>
          </section>
        </div>
        """
        method_cards = []
        for method in methods:
            report_case = reports[method["id"]]["cases"][case_idx]
            method_cards.append(
                f"""
                <section class="method-card">
                  <div class="method-name">{html.escape(method['label'])}</div>
                  <video controls preload="metadata" src="../{html.escape(method['best_rel'])}/{html.escape(report_case['generated_video'])}"></video>
                  <video controls preload="metadata" src="../{html.escape(method['best_rel'])}/{html.escape(report_case['condition_video'])}"></video>
                  <div class="metrics">
                    <span>predictor center {report_case['predictor_metrics']['center_error']:.3f}</span>
                    <span>predictor scale {report_case['predictor_metrics']['log_scale_error']:.3f}</span>
                    <span>video center {report_case['video_metrics']['center_error']:.3f}</span>
                    <span>video scale {report_case['video_metrics']['log_scale_error']:.3f}</span>
                  </div>
                </section>
                """
            )
        case_blocks.append(
            f"""
            <article class="case-card">
              <h2>{html.escape(case_id)}</h2>
              <p class="prompt">{html.escape(base_case['prompt'])}</p>
              {ref_row}
              <div class="method-grid">
                {''.join(method_cards)}
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>best ckpt 同批 case 对比页</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      color: #1f1f1b;
      background: linear-gradient(180deg, #f8f3ea 0%, #ede2d3 100%);
    }}
    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .case-card, table, .ref-card, .method-card {{
      background: rgba(255, 252, 246, 0.95);
      border: 1px solid #dccfbe;
      border-radius: 18px;
    }}
    .hero, .case-card {{
      padding: 20px;
      margin-bottom: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 18px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #e6dacb;
      text-align: left;
    }}
    .prompt {{
      color: #6f675d;
      line-height: 1.7;
    }}
    .ref-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .method-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .ref-card, .method-card {{
      padding: 12px;
    }}
    .ref-name, .method-name {{
      margin-bottom: 10px;
      color: #0f5a52;
      font-weight: 700;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #000;
      margin-bottom: 10px;
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .metrics span {{
      background: #f3eadf;
      border-radius: 999px;
      padding: 4px 8px;
      color: #7b4f2d;
      font-size: 12px;
    }}
    @media (max-width: 1100px) {{
      .ref-grid, .method-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>best ckpt 同批 case 对比页</h1>
      <p>本页对比当前所有方法的 best ckpt 在同一批 12 个代表性 case 上的生成结果。每个 case 展示统一的 context、GT future，以及各方法的 generated future 和 predicted conditions。</p>
      <p><a href="../index.html">返回方法总入口页</a></p>
    </section>
    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Predictor Center Mean</th>
          <th>Video Center Mean</th>
          <th>Val Loss</th>
        </tr>
      </thead>
      <tbody>
        {''.join(summary_rows)}
      </tbody>
    </table>
    {''.join(case_blocks)}
  </div>
</body>
</html>"""


def main():
    args = parse_args()
    output_root = args.output_root
    methods_root = output_root / "methods"
    compare_root = output_root / "compare_best"
    output_root.mkdir(parents=True, exist_ok=True)
    methods_root.mkdir(parents=True, exist_ok=True)
    compare_root.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for child in methods_root.iterdir():
            if child.is_symlink() or child.exists():
                if child.is_symlink() or child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)

    valid_methods = []
    reports: dict[str, dict] = {}
    for spec in METHOD_SPECS:
        report = safe_read_json(spec["best_report"])
        best_rel = None
        if report and spec["best_dir"].exists():
            best_link_name = f"{spec['id']}_best"
            ensure_symlink(spec["best_dir"], methods_root / best_link_name)
            best_rel = f"methods/{best_link_name}"

        timeline_rel = None
        if spec.get("timeline_dir") and spec["timeline_dir"].exists():
            timeline_link_name = f"{spec['id']}_timeline"
            ensure_symlink(spec["timeline_dir"], methods_root / timeline_link_name)
            timeline_rel = f"methods/{timeline_link_name}"

        if not report and not timeline_rel:
            continue

        valid_methods.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "desc": spec["desc"],
                "best_rel": best_rel,
                "timeline_rel": timeline_rel,
                "metrics": metric_text(report) if report else "训练中，best report 尚未生成",
            }
        )
        if report:
            reports[spec["id"]] = report

    compare_methods = [method for method in valid_methods if method["id"] in reports]
    if len(compare_methods) >= 2:
        compare_html = build_compare_html(compare_methods, reports)
        (compare_root / "index.html").write_text(compare_html, encoding="utf-8")

    index_html = build_index_html(valid_methods, args.port)
    (output_root / "index.html").write_text(index_html, encoding="utf-8")
    pid = start_server(output_root, args.port)
    print(f"overview: {output_root / 'index.html'}")
    print(f"compare: {compare_root / 'index.html'}")
    print(f"server: http://127.0.0.1:{args.port}")
    print(f"pid: {pid}")


if __name__ == "__main__":
    main()
