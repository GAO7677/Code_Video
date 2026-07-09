from __future__ import annotations

import argparse
import html
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a root portal for all PhysicsIQ formal compare result folders."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="PhysicsIQ formal compare portal")
    return parser.parse_args()


def html_escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def find_result_dirs(root: Path) -> list[tuple[Path, int]]:
    results: list[tuple[Path, int]] = []
    for directory in sorted(root.rglob("*")):
        if not directory.is_dir():
            continue
        if directory.name.startswith("_"):
            continue
        count = len(list(directory.glob("*_input_ctx*.jpg")))
        if count > 0:
            results.append((directory, count))
    return results


def build_html(*, title: str, root: Path, results: list[tuple[Path, int]]) -> str:
    cards: list[str] = []
    for directory, count in results:
        rel = directory.relative_to(root).as_posix()
        gallery_index = directory / "_case_grouped_gallery" / "index.html"
        gallery_rel = gallery_index.relative_to(root).as_posix()
        mp4_count = len(list(directory.glob("*.mp4")))
        json_count = len(list(directory.glob("*.json")))
        cards.append(
            f"""
            <article class="card">
              <div class="card-head">
                <div>
                  <div class="eyebrow">{html_escape(directory.parent.relative_to(root).as_posix() or '.')}</div>
                  <h2>{html_escape(directory.name)}</h2>
                  <p>{html_escape(rel)}</p>
                </div>
                <a class="open-link" href="{html_escape(gallery_rel)}">Open grouped gallery</a>
              </div>
              <div class="stats">
                <div><span>Shared inputs</span><strong>{count}</strong></div>
                <div><span>MP4 files</span><strong>{mp4_count}</strong></div>
                <div><span>JSON files</span><strong>{json_count}</strong></div>
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(title)}</title>
  <style>
    :root {{
      --bg: #f3ede4;
      --paper: #fffdf8;
      --ink: #1f1a16;
      --muted: #6d655d;
      --line: #d8c9b6;
      --accent: #0f4c81;
      --shadow: 0 18px 54px rgba(54, 37, 20, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 76, 129, 0.10), transparent 24rem),
        radial-gradient(circle at top right, rgba(113, 63, 18, 0.10), transparent 24rem),
        linear-gradient(180deg, #faf5ed 0%, var(--bg) 100%);
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
    }}
    .shell {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 28px 22px 40px;
    }}
    .hero {{
      background: rgba(255, 253, 248, 0.88);
      border: 1px solid rgba(216, 201, 182, 0.92);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px 30px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: "IBM Plex Serif", "Noto Serif SC", serif;
      font-size: clamp(30px, 3.6vw, 50px);
      line-height: 1.04;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 0;
      max-width: 1080px;
      color: var(--muted);
      line-height: 1.7;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
      margin-top: 24px;
    }}
    .card {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid rgba(216, 201, 182, 0.92);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    .card-head {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    h2 {{
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      word-break: break-word;
    }}
    .card-head p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      word-break: break-word;
    }}
    .open-link {{
      display: inline-flex;
      width: fit-content;
      text-decoration: none;
      padding: 10px 14px;
      border-radius: 999px;
      background: #e6f0f8;
      color: var(--accent);
      font-weight: 700;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}
    .stats div {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--paper);
      padding: 10px 12px;
    }}
    .stats span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 5px;
    }}
    .stats strong {{
      font-size: 18px;
    }}
    @media (max-width: 720px) {{
      .shell {{
        padding: 18px 12px 28px;
      }}
      .hero {{
        padding: 20px;
        border-radius: 22px;
      }}
      .stats {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>{html_escape(title)}</h1>
      <p>This root portal lists every PhysicsIQ formal-compare result folder that contains shared input images. Each entry links to a grouped gallery where one original case maps to one card and all available methods for that result folder are shown side by side.</p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    results = find_result_dirs(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(title=args.title, root=root, results=results), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
