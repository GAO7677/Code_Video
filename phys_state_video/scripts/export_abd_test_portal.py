#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a unified portal page for ABD_test visualizations.")
    parser.add_argument(
        "--abd-root",
        default="/data/gaoya/AAA_test_video/Output_try0526/ABD_test",
        help="Unified ABD_test root.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/abd_test_portal_18884",
        help="Directory to write the unified portal page.",
    )
    parser.add_argument("--port", type=int, default=18884, help="Local port for http.server.")
    parser.add_argument("--clean", action="store_true", help="Delete output dir before export.")
    parser.add_argument("--no-serve", action="store_true", help="Only export static files.")
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


def count_jsons(dir_path: Path) -> int:
    if not dir_path.is_dir():
        return 0
    return len(list(dir_path.glob("*.json")))


def resolve_case_count(report_json: Path | None) -> int:
    if report_json is None or not report_json.is_file():
        return 0
    payload = load_json(report_json)
    if isinstance(payload.get("case_count"), int):
        return int(payload["case_count"])
    cases = payload.get("cases")
    if isinstance(cases, list):
        return len(cases)
    return 0


def build_sections(abd_root: Path) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    b_meta = abd_root / "B" / "_meta"
    b_dashboard_entry = b_meta / "dashboard_entry.json"
    b_dashboard_dir = None
    if b_dashboard_entry.is_file():
        b_dashboard_dir = Path(load_json(b_dashboard_entry)["output_dir"])

    sections.append(
        {
            "slug": "a_full",
            "label": "A Full Report",
            "description": "PDI-Bench 全量方法报告页。",
            "path": str(Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/report")),
            "count": count_jsons(abd_root / "A" / "GT"),
            "methods": ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"],
        }
    )
    sections.append(
        {
            "slug": "a_subset",
            "label": "A Subset Cases",
            "description": "PDI-Bench 代表性 case 子集页。",
            "path": str(Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/report_subset")),
            "count": len(load_json(abd_root / "A" / "_meta" / "report_subset_selected_cases.json").get("cases", [])),
            "methods": ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"],
        }
    )
    sections.append(
        {
            "slug": "b_dashboard",
            "label": "B Dashboard",
            "description": "Dataset_physV B 组对比页，展示 context、GT full video 和 3 个方法输出。",
            "path": str(b_dashboard_dir) if b_dashboard_dir else None,
            "count": resolve_case_count((b_dashboard_dir / "report.json") if b_dashboard_dir else None),
            "methods": ["wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"],
        }
    )
    sections.append(
        {
            "slug": "d_full",
            "label": "D Full Report",
            "description": "Physics-IQ 全量进度报告页。",
            "path": str(Path("/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark/report_progress")),
            "count": count_jsons(abd_root / "D" / "GT"),
            "methods": ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"],
        }
    )
    sections.append(
        {
            "slug": "d_subset",
            "label": "D Subset Cases",
            "description": "Physics-IQ 代表性 case 子集页。",
            "path": str(Path("/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark/report_subset")),
            "count": len(load_json(abd_root / "D" / "_meta" / "report_subset_selected_cases.json").get("cases", [])),
            "methods": ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"],
        }
    )
    return sections


def build_html(report: dict[str, Any]) -> str:
    cards = []
    for section in report["sections"]:
        path = section.get("path")
        href = f"{section['slug']}/index.html" if path else ""
        open_link = (
            f'<a class="link" href="{html.escape(href)}">打开页面</a>'
            if path else '<span class="missing">尚未生成</span>'
        )
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">{html.escape(section['slug'])}</div>
              <h2>{html.escape(section['label'])}</h2>
              <p class="desc">{html.escape(section['description'])}</p>
              <p class="meta">cases={section['count']} | methods={','.join(section['methods'])}</p>
              {open_link}
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report['title'])}</title>
  <style>
    :root {{
      --bg0: #f7f2e9;
      --bg1: #eadfce;
      --panel: rgba(255, 251, 245, 0.96);
      --line: #ddcfbc;
      --ink: #201b17;
      --muted: #6d665d;
      --accent: #0d5b54;
      --accent2: #b96b34;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at left top, rgba(185, 107, 52, 0.12), transparent 24%),
        radial-gradient(circle at right top, rgba(13, 91, 84, 0.12), transparent 28%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
    }}
    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      padding: 18px;
    }}
    .eyebrow {{
      color: var(--accent2);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    h1, h2 {{
      margin: 0 0 10px;
    }}
    .intro, .meta, .access, .desc {{
      color: var(--muted);
      line-height: 1.7;
    }}
    .link {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    .missing {{
      color: #9f4c2b;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(report['title'])}</h1>
      <p class="intro">{html.escape(report['intro'])}</p>
      <p class="access">入口地址：<a href="http://127.0.0.1:{report['port']}">http://127.0.0.1:{report['port']}</a></p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    abd_root = Path(args.abd_root)
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                for sub in child.rglob("*"):
                    pass
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = build_sections(abd_root)
    for section in sections:
        source = section.get("path")
        if source is None or not Path(source).exists():
            continue
        link_path = output_dir / section["slug"]
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                import shutil
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.symlink_to(Path(source))

    report = {
        "title": "ABD_test Unified Portal",
        "intro": "统一入口页，汇总 A / B / D 三组 benchmark 的全量报告页、subset 代表 case 页，以及 B 组专用 dashboard。",
        "port": args.port,
        "sections": sections,
        "mode": "abd_test_portal",
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(build_html(report), encoding="utf-8")

    if not args.no_serve:
        pid = start_server(output_dir, args.port)
        print(f"served http://127.0.0.1:{args.port} pid={pid}")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
