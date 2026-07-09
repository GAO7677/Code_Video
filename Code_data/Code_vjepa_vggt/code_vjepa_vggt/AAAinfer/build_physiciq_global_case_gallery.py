from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path


INPUT_MARKER = "_input_ctx"


@dataclass(frozen=True)
class MethodSource:
    label: str
    rel_dir: str
    abs_dir: Path


@dataclass(frozen=True)
class CaseArtifact:
    input_image_rel: str | None
    video_rel: str | None
    json_rel: str | None
    log_rel: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a giant PhysicsIQ gallery grouped by case across all method directories."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--title",
        type=str,
        default="PhysicsIQ global same-case compare",
    )
    return parser.parse_args()


def html_escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def find_method_sources(root: Path) -> list[MethodSource]:
    sources: list[MethodSource] = []
    for directory in sorted(root.rglob("*")):
        if not directory.is_dir():
            continue
        if directory.name.startswith("_"):
            continue
        if any(directory.glob(f"*{INPUT_MARKER}*.jpg")):
            rel_dir = directory.relative_to(root).as_posix()
            label = rel_dir.replace("/", " | ")
            sources.append(MethodSource(label=label, rel_dir=rel_dir, abs_dir=directory))
    return sources


def collect_cases(root: Path, sources: list[MethodSource]) -> dict[str, dict[str, CaseArtifact]]:
    grouped: dict[str, dict[str, CaseArtifact]] = {}
    for source in sources:
        for input_path in sorted(source.abs_dir.glob(f"*{INPUT_MARKER}*.jpg")):
            case_stem = input_path.stem.split(INPUT_MARKER, 1)[0]
            video_path = source.abs_dir / f"{case_stem}.mp4"
            json_path = source.abs_dir / f"{case_stem}.json"
            log_path = source.abs_dir / f"{case_stem}.log"
            grouped.setdefault(case_stem, {})[source.label] = CaseArtifact(
                input_image_rel=input_path.relative_to(root).as_posix(),
                video_rel=video_path.relative_to(root).as_posix() if video_path.is_file() else None,
                json_rel=json_path.relative_to(root).as_posix() if json_path.is_file() else None,
                log_rel=log_path.relative_to(root).as_posix() if log_path.is_file() else None,
            )
    return grouped


def build_html(
    *,
    title: str,
    sources: list[MethodSource],
    grouped: dict[str, dict[str, CaseArtifact]],
) -> str:
    cards: list[str] = []
    source_labels = [source.label for source in sources]
    case_names = sorted(grouped)

    for idx, case_name in enumerate(case_names, start=1):
        artifacts_by_source = grouped[case_name]
        first_artifact = next(iter(artifacts_by_source.values()))
        input_rel = first_artifact.input_image_rel
        ready_count = sum(1 for artifact in artifacts_by_source.values() if artifact.video_rel is not None)
        panels: list[str] = []

        for source_label in source_labels:
            artifact = artifacts_by_source.get(source_label)
            if artifact is None:
                panels.append(
                    f"""
                    <div class="method-panel missing">
                      <div class="method-head">
                        <h3>{html_escape(source_label)}</h3>
                        <span class="state missing">missing</span>
                      </div>
                      <div class="missing-box">This case is not present in this result directory.</div>
                    </div>
                    """
                )
                continue

            links: list[str] = []
            if artifact.video_rel is not None:
                links.append(f'<a href="../{html_escape(artifact.video_rel)}" target="_blank" rel="noreferrer">video</a>')
            if artifact.json_rel is not None:
                links.append(f'<a href="../{html_escape(artifact.json_rel)}" target="_blank" rel="noreferrer">json</a>')
            if artifact.log_rel is not None:
                links.append(f'<a href="../{html_escape(artifact.log_rel)}" target="_blank" rel="noreferrer">log</a>')

            if artifact.video_rel is None:
                state_html = '<span class="state partial">no video</span>'
                media_html = '<div class="missing-box">Input exists but the final `.mp4` output is missing.</div>'
            else:
                state_html = '<span class="state ready">ready</span>'
                media_html = f'<video controls preload="metadata" playsinline src="../{html_escape(artifact.video_rel)}"></video>'

            panels.append(
                f"""
                <div class="method-panel">
                  <div class="method-head">
                    <h3>{html_escape(source_label)}</h3>
                    {state_html}
                  </div>
                  {media_html}
                  <div class="method-links">{''.join(links) if links else '<span class="muted">no side files</span>'}</div>
                </div>
                """
            )

        cards.append(
            f"""
            <section class="case-card" id="case-{idx}">
              <div class="case-head">
                <div>
                  <div class="eyebrow">Case {idx:02d}</div>
                  <h2>{html_escape(case_name)}</h2>
                  <p class="case-meta">ready outputs: {ready_count} / {len(source_labels)}</p>
                </div>
              </div>
              <div class="case-layout">
                <aside class="input-panel">
                  <div class="input-title">Shared input</div>
                  <img src="../{html_escape(input_rel)}" alt="{html_escape(case_name)} input">
                </aside>
                <div class="methods-grid">
                  {''.join(panels)}
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
      --bg: #efe7dc;
      --paper: #fffdf8;
      --ink: #1d1814;
      --muted: #6d655d;
      --line: #d9cab8;
      --accent: #0f4c81;
      --accent-soft: #e6f0f8;
      --warn: #9a3412;
      --warn-soft: #fff1e6;
      --shadow: 0 18px 54px rgba(54, 37, 20, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 76, 129, 0.10), transparent 24rem),
        radial-gradient(circle at top right, rgba(154, 52, 18, 0.10), transparent 24rem),
        linear-gradient(180deg, #faf4eb 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
    }}
    .shell {{
      max-width: 2280px;
      margin: 0 auto;
      padding: 28px 22px 40px;
    }}
    .hero {{
      background: rgba(255, 253, 248, 0.88);
      border: 1px solid rgba(217, 202, 184, 0.92);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px 30px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: "IBM Plex Serif", "Noto Serif SC", serif;
      font-size: clamp(30px, 3.8vw, 52px);
      line-height: 1.04;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 0;
      max-width: 1180px;
      color: var(--muted);
      line-height: 1.7;
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
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid rgba(217, 202, 184, 0.9);
      border-radius: 26px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .case-head {{
      padding: 22px 24px 18px;
      border-bottom: 1px solid rgba(217, 202, 184, 0.8);
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
      line-height: 1.2;
      word-break: break-word;
    }}
    .case-meta {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .case-layout {{
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
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
      border: 1px solid rgba(217, 202, 184, 0.9);
    }}
    .methods-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
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
      font-size: 16px;
      line-height: 1.28;
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
      background: #17120d;
    }}
    .method-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 12px 16px 16px;
      font-size: 13px;
    }}
    .method-links a {{
      color: var(--accent);
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
    @media (max-width: 1240px) {{
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
      <p>This giant page aggregates every PhysicsIQ result directory under the formal compare root. Each card corresponds to one shared case, and every available result directory is displayed as a method panel on the same card.</p>
      <div class="summary">
        <div class="summary-chip">Cases: {len(case_names)}</div>
        <div class="summary-chip">Method directories: {len(source_labels)}</div>
      </div>
    </section>
    <div class="case-stack">
      {''.join(cards)}
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sources = find_method_sources(root)
    grouped = collect_cases(root, sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    index_path.write_text(
        build_html(title=args.title, sources=sources, grouped=grouped),
        encoding="utf-8",
    )
    print(index_path)


if __name__ == "__main__":
    main()
