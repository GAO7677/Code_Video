from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a simple gallery for Physics-IQ ctx sweep outputs.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Physics-IQ ctx sweep")
    return parser.parse_args()


def relpath(target: Path, base: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def collect_entries(result_root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for ctx_dir in sorted(path for path in result_root.iterdir() if path.is_dir() and path.name.startswith("ctx")):
        step_dirs = sorted(path for path in ctx_dir.iterdir() if path.is_dir() and path.name.startswith("step-"))
        if not step_dirs:
            continue
        step_dir = step_dirs[0]
        json_files = sorted(path for path in step_dir.glob("*.json") if path.name != "result.json")
        if not json_files:
            continue
        case_json = json_files[0]
        payload = json.loads(case_json.read_text(encoding="utf-8"))
        output_video = Path(str(payload["output_video"])).expanduser().resolve()
        input_video = Path(str(payload["input_video"])).expanduser().resolve() if payload.get("input_video") else None
        source_video = Path(str(payload["source_video"])).expanduser().resolve() if payload.get("source_video") else None
        entries.append(
            {
                "ctx": ctx_dir.name,
                "case_json": str(case_json.resolve()),
                "caption": str(payload.get("input_caption", "")),
                "output_video": str(output_video),
                "input_video": "" if input_video is None else str(input_video),
                "source_video": "" if source_video is None else str(source_video),
                "method": str(payload.get("method", "")),
            }
        )
    return entries


def render_html(*, title: str, result_root: Path, entries: list[dict[str, str]]) -> str:
    cards: list[str] = []
    for entry in entries:
        output_rel = relpath(Path(entry["output_video"]), result_root)
        input_rel = relpath(Path(entry["input_video"]), result_root) if entry["input_video"] else ""
        source_rel = relpath(Path(entry["source_video"]), result_root) if entry["source_video"] else ""
        if input_rel:
            context_block = f'<img src="{html.escape(input_rel)}" />'
        else:
            context_block = "<div class='empty'>N/A</div>"
        if source_rel:
            source_block = (
                f'<video controls playsinline preload="metadata" src="{html.escape(source_rel)}"></video>'
            )
        else:
            source_block = "<div class='empty'>N/A</div>"
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(entry['ctx'])}</h2>
              <div class="meta">{html.escape(entry['method'])}</div>
              <div class="meta">{html.escape(entry['case_json'])}</div>
              <p class="caption">{html.escape(entry['caption'])}</p>
              <div class="grid">
                <div>
                  <div class="label">generated</div>
                  <video controls playsinline preload="metadata" src="{html.escape(output_rel)}"></video>
                </div>
                <div>
                  <div class="label">context</div>
                  {context_block}
                </div>
                <div>
                  <div class="label">source</div>
                  {source_block}
                </div>
              </div>
            </section>
            """
        )

    body = "\n".join(cards) if cards else "<p>No completed ctx outputs found yet.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf1;
      --ink: #1f1d18;
      --muted: #6f675c;
      --line: #d8cfbf;
      --accent: #8c4f2b;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #fff9ec, transparent 30%),
        linear-gradient(180deg, #efe6d8 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 42px;
      line-height: 1.05;
      letter-spacing: -0.02em;
    }}
    .sub {{
      margin: 0 0 28px;
      color: var(--muted);
      font-size: 16px;
    }}
    .card {{
      background: color-mix(in srgb, var(--panel) 92%, white 8%);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 20px;
      box-shadow: 0 18px 40px rgba(68, 50, 27, 0.08);
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 28px;
      color: var(--accent);
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
      word-break: break-all;
    }}
    .caption {{
      font-size: 15px;
      line-height: 1.5;
      margin: 10px 0 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    video, img, .empty {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #ddd4c3;
      min-height: 220px;
      object-fit: contain;
    }}
    .empty {{
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 1100px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>{html.escape(title)}</h1>
    <p class="sub">{html.escape(str(result_root))}</p>
    {body}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_html = (args.output_html.expanduser().resolve() if args.output_html else result_root / "index.html")
    entries = collect_entries(result_root)
    output_html.write_text(
        render_html(title=str(args.title), result_root=result_root, entries=entries),
        encoding="utf-8",
    )
    print(json.dumps({"output_html": str(output_html), "num_entries": len(entries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
