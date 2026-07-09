from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path


INPUT_SUFFIX = "_input_ctx08"


@dataclass(frozen=True)
class MethodEntry:
    method_name: str
    input_image: str | None
    output_video: str | None
    metrics_json: str | None
    run_log: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a gallery that groups V2V outputs by shared input case."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--title",
        type=str,
        default="PhysicsIQ same-input multi-method compare",
    )
    return parser.parse_args()


def html_escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def rel_to_posix(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def discover_cases(result_root: Path) -> tuple[list[str], dict[str, dict[str, MethodEntry]]]:
    methods = sorted([path for path in result_root.iterdir() if path.is_dir()])
    grouped: dict[str, dict[str, MethodEntry]] = {}

    for method_dir in methods:
        method_name = method_dir.name
        inputs = sorted(method_dir.glob(f"*{INPUT_SUFFIX}.jpg"))
        for input_path in inputs:
            case_stem = input_path.stem[: -len(INPUT_SUFFIX)]
            video_path = method_dir / f"{case_stem}.mp4"
            json_path = method_dir / f"{case_stem}.json"
            log_path = method_dir / f"{case_stem}.log"
            grouped.setdefault(case_stem, {})[method_name] = MethodEntry(
                method_name=method_name,
                input_image=rel_to_posix(input_path, result_root),
                output_video=rel_to_posix(video_path, result_root) if video_path.is_file() else None,
                metrics_json=rel_to_posix(json_path, result_root) if json_path.is_file() else None,
                run_log=rel_to_posix(log_path, result_root) if log_path.is_file() else None,
            )

    method_names = [path.name for path in methods]
    return method_names, grouped


def build_html(
    *,
    title: str,
    method_names: list[str],
    grouped: dict[str, dict[str, MethodEntry]],
) -> str:
    cards_html: list[str] = []
    case_names = sorted(grouped)
    total_methods = len(method_names)

    for idx, case_name in enumerate(case_names, start=1):
        method_entries = grouped[case_name]
        first_entry = next(iter(method_entries.values()))
        input_image = first_entry.input_image
        present_count = sum(1 for entry in method_entries.values() if entry.output_video is not None)
        method_blocks: list[str] = []

        for method_name in method_names:
            entry = method_entries.get(method_name)
            if entry is None:
                method_blocks.append(
                    f"""
                    <div class="method-panel missing">
                      <div class="method-head">
                        <h3>{html_escape(method_name)}</h3>
                        <span class="state missing">missing input</span>
                      </div>
                      <div class="missing-box">No shared input found for this case in this method.</div>
                    </div>
                    """
                )
                continue

            links: list[str] = []
            if entry.output_video is not None:
                links.append(
                    f'<a href="../{html_escape(entry.output_video)}" target="_blank" rel="noreferrer">video</a>'
                )
            if entry.metrics_json is not None:
                links.append(
                    f'<a href="../{html_escape(entry.metrics_json)}" target="_blank" rel="noreferrer">json</a>'
                )
            if entry.run_log is not None:
                links.append(
                    f'<a href="../{html_escape(entry.run_log)}" target="_blank" rel="noreferrer">log</a>'
                )

            if entry.output_video is None:
                body_html = '<div class="missing-box">Input exists, but the final `.mp4` output is missing.</div>'
                state_html = '<span class="state partial">no video</span>'
            else:
                body_html = (
                    f'<video controls preload="metadata" playsinline src="../{html_escape(entry.output_video)}"></video>'
                )
                state_html = '<span class="state ready">ready</span>'

            method_blocks.append(
                f"""
                <div class="method-panel">
                  <div class="method-head">
                    <h3>{html_escape(method_name)}</h3>
                    {state_html}
                  </div>
                  {body_html}
                  <div class="method-links">{''.join(links) if links else '<span class="muted">no side files</span>'}</div>
                </div>
                """
            )

        cards_html.append(
            f"""
            <section class="case-card" id="case-{idx}">
              <div class="case-head">
                <div>
                  <div class="eyebrow">Case {idx:02d}</div>
                  <h2>{html_escape(case_name)}</h2>
                  <p class="case-meta">methods with output: {present_count} / {total_methods}</p>
                </div>
              </div>
              <div class="case-layout">
                <aside class="input-panel">
                  <div class="input-title">Shared input</div>
                  <img src="../{html_escape(input_image)}" alt="{html_escape(case_name)} input">
                </aside>
                <div class="methods-grid">
                  {''.join(method_blocks)}
                </div>
              </div>
            </section>
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
      --bg: #efe8de;
      --paper: #fffdf9;
      --ink: #1c1714;
      --muted: #6b625a;
      --line: #d9caba;
      --accent: #14532d;
      --accent-soft: #e4f4e7;
      --warn: #9a3412;
      --warn-soft: #fff1e6;
      --shadow: 0 18px 54px rgba(54, 37, 20, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(20, 83, 45, 0.10), transparent 26rem),
        radial-gradient(circle at top right, rgba(154, 52, 18, 0.12), transparent 24rem),
        linear-gradient(180deg, #f7f1e8 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
    }}
    .shell {{
      max-width: 1880px;
      margin: 0 auto;
      padding: 28px 22px 40px;
    }}
    .hero {{
      background: rgba(255, 253, 249, 0.86);
      border: 1px solid rgba(217, 202, 186, 0.92);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px 30px;
      backdrop-filter: blur(12px);
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: "IBM Plex Serif", "Noto Serif SC", serif;
      font-size: clamp(30px, 3.6vw, 52px);
      line-height: 1.02;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      max-width: 1100px;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    .summary-chip {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    .case-stack {{
      display: grid;
      gap: 22px;
      margin-top: 24px;
    }}
    .case-card {{
      background: rgba(255, 253, 249, 0.92);
      border: 1px solid rgba(217, 202, 186, 0.9);
      border-radius: 26px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .case-head {{
      padding: 22px 24px 18px;
      border-bottom: 1px solid rgba(217, 202, 186, 0.8);
      background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(247,240,232,0.78));
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
      font-size: 24px;
      line-height: 1.18;
      word-break: break-word;
    }}
    .case-meta {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .case-layout {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      align-items: start;
    }}
    .input-panel {{
      position: sticky;
      top: 14px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 14px;
    }}
    .input-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 10px;
    }}
    img {{
      display: block;
      width: 100%;
      border-radius: 16px;
      background: #111;
      border: 1px solid rgba(217, 202, 186, 0.9);
    }}
    .methods-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      min-width: 0;
    }}
    .method-panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 20px;
      overflow: hidden;
    }}
    .method-panel.missing {{
      border-style: dashed;
    }}
    .method-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 16px 10px;
    }}
    .method-head h3 {{
      margin: 0;
      font-size: 17px;
      line-height: 1.25;
      word-break: break-word;
    }}
    .state {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .state.ready {{
      color: var(--accent);
      background: var(--accent-soft);
    }}
    .state.partial {{
      color: var(--warn);
      background: var(--warn-soft);
    }}
    .state.missing {{
      color: #7c2d12;
      background: #fde7da;
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #15110d;
    }}
    .method-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 12px 16px 16px;
      font-size: 13px;
    }}
    .method-links a {{
      color: #0f4c81;
      text-decoration: none;
      font-weight: 600;
    }}
    .method-links a:hover {{
      text-decoration: underline;
    }}
    .missing-box {{
      margin: 0 16px 16px;
      min-height: 220px;
      border-radius: 16px;
      border: 1px dashed rgba(154, 52, 18, 0.45);
      background: #fff7f1;
      color: var(--muted);
      display: grid;
      place-items: center;
      text-align: center;
      padding: 16px;
      line-height: 1.6;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 1200px) {{
      .case-layout {{
        grid-template-columns: 1fr;
      }}
      .input-panel {{
        position: static;
      }}
    }}
    @media (max-width: 720px) {{
      .shell {{
        padding: 18px 12px 28px;
      }}
      .hero {{
        padding: 20px;
        border-radius: 22px;
      }}
      .methods-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>{html_escape(title)}</h1>
      <p>Each card corresponds to one shared input case. The left column shows the input context image, and the right side places outputs from different methods side by side. Missing methods are explicitly marked so you can see coverage differences immediately.</p>
      <div class="summary">
        <div class="summary-chip">Cases: {len(case_names)}</div>
        <div class="summary-chip">Methods: {len(method_names)}</div>
        <div class="summary-chip">Root: {html_escape(str(result_root_placeholder := Path(".")))}</div>
      </div>
    </section>
    <div class="case-stack">
      {''.join(cards_html)}
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    method_names, grouped = discover_cases(result_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_text = build_html(
        title=args.title,
        method_names=method_names,
        grouped=grouped,
    )
    index_path = output_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    print(index_path)


if __name__ == "__main__":
    main()
