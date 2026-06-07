#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a parent index page for static visualization subpages.")
    parser.add_argument("--output-dir", required=True, help="Directory to write report.json and index.html.")
    parser.add_argument("--title", required=True, help="Portal page title.")
    parser.add_argument("--intro", default="", help="Intro text shown in the hero section.")
    parser.add_argument("--port", type=int, default=None, help="Optional serving port for display only.")
    parser.add_argument(
        "--subpage",
        action="append",
        default=[],
        help="Subpage spec: slug|label|description|path_to_report_json_or_dir",
    )
    return parser.parse_args()


def parse_subpage_spec(spec: str) -> dict:
    parts = spec.split("|", 3)
    if len(parts) != 4:
        raise ValueError(f"invalid --subpage spec: {spec!r}")
    slug, label, description, source_str = [part.strip() for part in parts]
    if not slug:
        raise ValueError(f"empty slug in --subpage spec: {spec!r}")
    source = Path(source_str)
    report_path = source if source.name == "report.json" else source / "report.json"
    payload = {}
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "slug": slug,
        "label": label or slug,
        "description": description,
        "source": str(source),
        "report_path": str(report_path),
        "case_count": int(payload.get("case_count", 0)) if isinstance(payload, dict) else 0,
        "mode": str(payload.get("mode", "")) if isinstance(payload, dict) else "",
        "json_names": list(payload.get("json_names", [])) if isinstance(payload, dict) else [],
    }


def build_html(title: str, intro: str, port: int | None, subpages: list[dict]) -> str:
    cards = []
    for item in subpages:
        meta_bits = []
        if item["mode"]:
            meta_bits.append(f"mode={item['mode']}")
        if item["case_count"]:
            meta_bits.append(f"cases={item['case_count']}")
        if item["json_names"]:
            meta_bits.append("json=" + ",".join(map(str, item["json_names"])))
        meta_text = " | ".join(meta_bits) if meta_bits else "暂无附加统计"
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">{html.escape(item['slug'])}</div>
              <h2>{html.escape(item['label'])}</h2>
              <p class="desc">{html.escape(item['description'])}</p>
              <p class="meta">{html.escape(meta_text)}</p>
              <a class="link" href="{html.escape(item['slug'])}/index.html">打开子页面</a>
            </article>
            """
        )

    access_line = ""
    if port is not None:
        access_line = (
            f'<p class="access">入口地址：'
            f'<a href="http://127.0.0.1:{port}">http://127.0.0.1:{port}</a></p>'
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
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
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p class="intro">{html.escape(intro)}</p>
      {access_line}
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>"""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subpages = [parse_subpage_spec(spec) for spec in args.subpage]
    report = {
        "title": args.title,
        "intro": args.intro,
        "port": args.port,
        "subpages": subpages,
        "mode": "static_subpage_portal",
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(
        build_html(args.title, args.intro, args.port, subpages),
        encoding="utf-8",
    )
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
