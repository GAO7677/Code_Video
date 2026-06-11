#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio_ffmpeg


METHODS = ["wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a static dashboard for ABD_test/B benchmark cases.")
    parser.add_argument(
        "--bench-root",
        default="/data/gaoya/AAA_test_video/Output_try0526/ABD_test/B",
        help="Root directory containing GT and method subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/abd_test_b_dashboard_18883",
        help="Directory to write report.json, index.html, and linked assets.",
    )
    parser.add_argument(
        "--meta-dir",
        default=None,
        help="Optional ABD_test/B/_meta directory to receive selected-case metadata.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional metrics summary csv path. Defaults to <bench-root>/_meta/method_metrics_summary.csv.",
    )
    parser.add_argument("--max-cases", type=int, default=12, help="Maximum number of cases to show.")
    parser.add_argument("--port", type=int, default=18883, help="Local port for http.server.")
    parser.add_argument("--clean", action="store_true", help="Delete output dir before export.")
    parser.add_argument("--no-serve", action="store_true", help="Only export static files, do not start server.")
    return parser.parse_args()


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_server(output_dir: Path, port: int) -> int:
    log_path = output_dir / f"http_{port}.log"
    pid_path = output_dir / f"http_{port}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            if is_port_open(port):
                return pid
        except Exception:
            pid_path.unlink(missing_ok=True)

    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(
            ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=str(output_dir),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(1.0)
    return proc.pid


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary_rows(summary_csv: Path | None) -> list[dict[str, str]]:
    if summary_csv is None or not summary_csv.is_file():
        return []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_case_keys(gt_dir: Path, max_cases: int) -> list[str]:
    rows: list[dict[str, str]] = []
    for json_path in sorted(gt_dir.glob("*.json")):
        payload = load_json(json_path)
        rows.append(
            {
                "case_key": str(payload["case_key"]),
                "category": str(payload["category"]),
            }
        )

    by_category: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["case_key"])

    ordered_categories = sorted(by_category)
    selected: list[str] = []
    for category in ordered_categories:
        selected.append(by_category[category][0])
        if len(selected) >= max_cases:
            return selected

    if len(selected) < max_cases:
        for category in ordered_categories:
            for case_key in by_category[category][1:]:
                selected.append(case_key)
                if len(selected) >= max_cases:
                    return selected
    return selected


def choose_case_meta(gt_dir: Path, max_cases: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for json_path in sorted(gt_dir.glob("*.json")):
        payload = load_json(json_path)
        rows.append(
            {
                "case_key": str(payload["case_key"]),
                "category": str(payload["category"]),
                "clip_name": str(payload.get("clip_name", payload["case_key"])),
            }
        )

    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    ordered_categories = sorted(by_category)
    selected: list[dict[str, str]] = []
    for category in ordered_categories:
        first = dict(by_category[category][0])
        first["reason"] = f"Category coverage pick for {category}."
        selected.append(first)
        if len(selected) >= max_cases:
            return selected

    if len(selected) < max_cases:
        for category in ordered_categories:
            for row in by_category[category][1:]:
                extra = dict(row)
                extra["reason"] = f"Additional case from {category} to fill the dashboard."
                selected.append(extra)
                if len(selected) >= max_cases:
                    return selected
    return selected


def rel_link(output_dir: Path, case_dir: Path, target: Path, name: str) -> str:
    case_dir.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ""
    link_path = case_dir / f"{name}{suffix}"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)
    return str(link_path.relative_to(output_dir))


def ensure_browser_mp4(output_dir: Path, case_dir: Path, target: Path, name: str) -> str:
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / f"{name}.browser.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path.relative_to(output_dir))

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(target),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out_path.relative_to(output_dir))


def render_media_card(title: str, rel_path: str | None, kind: str, note: str) -> str:
    if rel_path:
        if kind == "image":
            body = f'<img loading="lazy" src="{html.escape(rel_path)}" alt="{html.escape(title)}" />'
        else:
            body = f'<video controls preload="metadata" src="{html.escape(rel_path)}"></video>'
    else:
        body = '<div class="media-missing">Not provided</div>'
    return f"""
      <article class="media-card">
        <div class="media-head">
          <h3>{html.escape(title)}</h3>
          <div class="media-note">{html.escape(note)}</div>
        </div>
        {body}
      </article>
    """


def build_html(report: dict[str, Any]) -> str:
    summary_html = ""
    if report.get("metrics_summary"):
        header = "".join(
            f"<th>{html.escape(label)}</th>"
            for label in ["method", "num_videos", "official_pdi", "wmreward_surprise", "cosmos_reason1", "videophy2_auto_pc", "videophy2_auto_joint"]
        )
        rows = []
        for row in report["metrics_summary"]:
            cells = "".join(
                f"<td>{html.escape(str(row.get(key, '')))}</td>"
                for key in ["method", "num_videos", "official_pdi", "wmreward_surprise", "cosmos_reason1", "videophy2_auto_pc", "videophy2_auto_joint"]
            )
            rows.append(f"<tr>{cells}</tr>")
        summary_html = f"""
        <section class="metrics-panel">
          <h2>Metrics Summary</h2>
          <div class="metrics-scroll">
            <table class="metrics-table">
              <thead><tr>{header}</tr></thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
          </div>
        </section>
        """

    case_html: list[str] = []
    for case in report["cases"]:
        input_cards = "".join(
            [
                render_media_card(
                    "Context Image",
                    case.get("context_image_rel"),
                    "image",
                    "TI2V methods use this single first-frame image as input.",
                ),
                render_media_card(
                    "Context Video",
                    case.get("context_video_rel"),
                    "video",
                    "VACE_1p3B_ctx08 uses this source clip as video context.",
                ),
                render_media_card(
                    "GT Full Video",
                    case.get("gt_full_video_rel"),
                    "video",
                    "Ground-truth full source video.",
                ),
            ]
        )
        output_cards = "".join(
            [
                render_media_card(
                    method["label"],
                    method.get("video_rel"),
                    "video",
                    method["note"],
                )
                for method in case["methods"]
            ]
        )
        case_html.append(
            f"""
            <section class="case-card">
              <div class="case-head">
                <div>
                  <div class="eyebrow">{html.escape(case['category'])}</div>
                  <h2>{html.escape(case['case_key'])}</h2>
                </div>
                <div class="meta-chip">{html.escape(case['clip_name'])}</div>
              </div>
              <div class="caption-box">{html.escape(case['caption'])}</div>
              <div class="grid inputs-grid">
                {input_cards}
              </div>
              <div class="grid outputs-grid">
                {output_cards}
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(report['title'])}</title>
  <style>
    :root {{
      --bg0: #f5efe5;
      --bg1: #e7dbc9;
      --panel: rgba(255, 251, 245, 0.96);
      --line: #d6c8b5;
      --ink: #1a1815;
      --muted: #6e665d;
      --accent: #1d5a52;
      --accent2: #b56d36;
      --shadow: 0 16px 40px rgba(63, 47, 30, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at left top, rgba(181, 109, 54, 0.11), transparent 28%),
        radial-gradient(circle at right top, rgba(29, 90, 82, 0.11), transparent 30%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
    }}
    .page {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .case-card, .media-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 22px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 34px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.75;
      max-width: 1180px;
    }}
    .hero .link {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0 0;
    }}
    .summary-card {{
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(250, 245, 237, 0.92);
      border: 1px solid var(--line);
    }}
    .summary-card strong {{
      display: block;
      font-size: 28px;
      color: var(--accent);
      margin-bottom: 4px;
    }}
    .summary-card span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .metrics-panel {{
      margin: 18px 0 22px;
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .metrics-panel h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .metrics-scroll {{
      overflow-x: auto;
    }}
    .metrics-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .metrics-table th,
    .metrics-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    .metrics-table th {{
      color: var(--accent);
      background: rgba(245, 239, 229, 0.88);
    }}
    .case-card {{
      padding: 20px;
      margin-bottom: 22px;
    }}
    .case-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 10px;
    }}
    .case-head h2 {{
      margin: 0;
      font-size: 24px;
    }}
    .eyebrow {{
      color: var(--accent2);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .meta-chip {{
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      background: rgba(245, 239, 229, 0.8);
      white-space: nowrap;
    }}
    .caption-box {{
      margin: 12px 0 18px;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(246, 241, 232, 0.92);
      border: 1px solid var(--line);
      font-size: 16px;
      line-height: 1.7;
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .inputs-grid {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-bottom: 16px;
    }}
    .outputs-grid {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .media-card {{
      padding: 14px;
    }}
    .media-head {{
      margin-bottom: 10px;
    }}
    .media-head h3 {{
      margin: 0 0 4px;
      font-size: 18px;
    }}
    .media-note {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      min-height: 40px;
    }}
    video, img {{
      display: block;
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #000;
    }}
    img {{
      aspect-ratio: 16 / 9;
      object-fit: cover;
      background: #ece4d7;
    }}
    .media-missing {{
      min-height: 200px;
      display: grid;
      place-items: center;
      color: var(--muted);
      border-radius: 14px;
      border: 1px dashed var(--line);
      background: rgba(239, 232, 220, 0.55);
    }}
    @media (max-width: 1120px) {{
      .summary,
      .inputs-grid,
      .outputs-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 760px) {{
      .page {{
        padding: 16px;
      }}
      .summary,
      .inputs-grid,
      .outputs-grid {{
        grid-template-columns: 1fr;
      }}
      .case-head {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .hero h1 {{
        font-size: 28px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(report['title'])}</h1>
      <p>{html.escape(report['intro'])}</p>
      <div class="summary">
        <div class="summary-card"><strong>{report['case_count']}</strong><span>visualized cases</span></div>
        <div class="summary-card"><strong>{report['gt_count']}</strong><span>GT B-group cases available</span></div>
        <div class="summary-card"><strong>{report['method_count']}</strong><span>generated methods compared</span></div>
        <div class="summary-card"><strong><a class="link" href="http://127.0.0.1:{report['port']}">127.0.0.1:{report['port']}</a></strong><span>local page entry</span></div>
      </div>
    </section>
    {summary_html}
    {''.join(case_html)}
  </div>
</body>
</html>
"""


def build_case_record(bench_root: Path, output_dir: Path, case_key: str) -> dict[str, Any]:
    gt_payload = load_json(bench_root / "GT" / f"{case_key}.json")
    wan_payload = load_json(bench_root / "wan22-5B-TI2V" / f"{case_key}.json")
    vace_ti2v_payload = load_json(bench_root / "VACE_1p3B_TI2V" / f"{case_key}.json")
    vace_ctx_payload = load_json(bench_root / "VACE_1p3B_ctx08" / f"{case_key}.json")

    case_dir = output_dir / "cases" / case_key
    context_image_rel = rel_link(output_dir, case_dir, Path(str(wan_payload["input_image"])), "context_image")
    context_video_path = Path(str(vace_ctx_payload["input_context_video"])) if vace_ctx_payload.get("input_context_video") else None
    context_video_rel = (
        ensure_browser_mp4(output_dir, case_dir, context_video_path, "context_video")
        if context_video_path is not None else None
    )
    gt_full_video_rel = ensure_browser_mp4(
        output_dir,
        case_dir,
        Path(str(gt_payload["output_video"])),
        "gt_full_video",
    )

    methods = []
    for method_name, payload, label, note in [
        (
            "wan22-5B-TI2V",
            wan_payload,
            "wan22-5B-TI2V",
            "First-frame TI2V baseline; consumes only the context image.",
        ),
        (
            "VACE_1p3B_TI2V",
            vace_ti2v_payload,
            "VACE_1p3B_TI2V",
            "VACE TI2V baseline; also uses only the context image.",
        ),
        (
            "VACE_1p3B_ctx08",
            vace_ctx_payload,
            "VACE_1p3B_ctx08",
            "Video-conditioned baseline; uses an 8-frame context clip.",
        ),
    ]:
        methods.append(
            {
                "method_name": method_name,
                "label": label,
                "conditioning_mode": payload.get("conditioning_mode"),
                "context_frames": payload.get("context_frames"),
                "note": note,
                "video_rel": rel_link(output_dir, case_dir, Path(str(payload["output_video"])), method_name),
            }
        )

    return {
        "case_key": case_key,
        "category": str(gt_payload["category"]),
        "clip_name": str(gt_payload.get("clip_name", case_key)),
        "caption": str(gt_payload["input_prompt"]),
        "context_image_rel": context_image_rel,
        "context_video_rel": context_video_rel,
        "gt_full_video_rel": gt_full_video_rel,
        "methods": methods,
    }


def write_meta_exports(meta_dir: Path, report: dict[str, Any], output_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = []
    for case in report["cases"]:
        selected_cases.append(
            {
                "case_key": case["case_key"],
                "category": case["category"],
                "clip_name": case["clip_name"],
                "caption": case["caption"],
                "reason": f"Included in ABD_test B dashboard export ({report['case_count']} visualized cases).",
                "methods": [method["method_name"] for method in case["methods"]],
            }
        )
    (meta_dir / "report_subset_selected_cases.json").write_text(
        json.dumps({"cases": selected_cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (meta_dir / "dashboard_entry.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index_html": str(output_dir / "index.html"),
                "report_json": str(output_dir / "report.json"),
                "mode": report["mode"],
                "case_count": report["case_count"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    bench_root = Path(args.bench_root)
    output_dir = Path(args.output_dir)
    meta_dir = Path(args.meta_dir) if args.meta_dir else (bench_root / "_meta")
    summary_csv = Path(args.summary_csv) if args.summary_csv else (meta_dir / "method_metrics_summary.csv")
    gt_dir = bench_root / "GT"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_meta = choose_case_meta(gt_dir, args.max_cases)
    cases = [build_case_record(bench_root, output_dir, row["case_key"]) for row in case_meta]
    metrics_summary = load_summary_rows(summary_csv)

    report = {
        "title": "Dataset_physV B Group Benchmark Dashboard",
        "intro": (
            "This page compares B-group source cases and generated videos. "
            "Each card shows the TI2V context image, the ctx08 context video, the GT full video, "
            "and outputs from wan22-5B-TI2V, VACE_1p3B_TI2V, and VACE_1p3B_ctx08."
        ),
        "mode": "abd_test_b_dashboard",
        "bench_root": str(bench_root),
        "case_count": len(cases),
        "gt_count": len(list(gt_dir.glob('*.json'))),
        "method_count": len(METHODS),
        "port": args.port,
        "summary_csv": str(summary_csv) if summary_csv.exists() else None,
        "metrics_summary": metrics_summary,
        "cases": cases,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(build_html(report), encoding="utf-8")
    write_meta_exports(meta_dir, report, output_dir)

    if not args.no_serve:
        pid = start_server(output_dir, args.port)
        print(f"served http://127.0.0.1:{args.port} pid={pid}")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
