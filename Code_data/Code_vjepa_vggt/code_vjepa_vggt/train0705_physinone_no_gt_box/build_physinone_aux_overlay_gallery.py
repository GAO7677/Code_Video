from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single HTML gallery page that embeds all PhysInOne "
            "aux-overlay visualization results under a root directory."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/phisinone_train_forward_aux_overlay"),
        help="Root directory that contains per-case result.json files.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/phisinone_train_forward_aux_overlay/all_cases_gallery.html"),
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include cases whose train/object_count is zero.",
    )
    return parser.parse_args()


def _relpath(target: Path, start: Path) -> str:
    return target.resolve().relative_to(start.resolve()).as_posix()


def _load_cases(root_dir: Path, *, include_empty: bool) -> list[dict]:
    cases: list[dict] = []
    for result_path in sorted(root_dir.rglob("result.json")):
        case_dir = result_path.parent
        data = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = dict(data.get("metrics", {}))
        object_count = float(metrics.get("train/object_count", 0.0))
        if not include_empty and object_count <= 0.0:
            continue
        cases.append(
            {
                "case_dir": case_dir,
                "case_name": str(case_dir.relative_to(root_dir)),
                "sample_key": str(data.get("sample_key", "")),
                "caption": str(data.get("caption", "")),
                "video_path": str(data.get("video_path", "")),
                "context_frame_indices": data.get("context_frame_indices", []),
                "prompt_preview_png": str(data.get("prompt_preview_png", "")),
                "input_overlay_video": str(data.get("input_overlay_video", "")),
                "box_overlay_video": str(data.get("box_overlay_video", "")),
                "track_overlay_video": str(data.get("track_overlay_video", "")),
                "metrics": metrics,
                "object_count": object_count,
            }
        )
    cases.sort(
        key=lambda item: (
            -float(item["object_count"]),
            str(item["case_name"]),
        )
    )
    return cases


def _render_case(case: dict, *, root_dir: Path, output_dir: Path) -> str:
    case_dir = Path(case["case_dir"])
    prompt_src = _relpath(case_dir / case["prompt_preview_png"], output_dir)
    input_video_src = _relpath(case_dir / case["input_overlay_video"], output_dir)
    box_video_src = _relpath(case_dir / case["box_overlay_video"], output_dir)
    track_video_src = _relpath(case_dir / case["track_overlay_video"], output_dir)
    report_href = _relpath(case_dir / "index.html", output_dir)
    status = "non-empty" if float(case["object_count"]) > 0.0 else "empty"
    metrics_json = html.escape(json.dumps(case["metrics"], ensure_ascii=False, indent=2))
    caption = html.escape(str(case["caption"]))
    sample_key = html.escape(str(case["sample_key"]))
    video_path = html.escape(str(case["video_path"]))
    context_frames = html.escape(json.dumps(case["context_frame_indices"], ensure_ascii=False))
    case_name = html.escape(str(case["case_name"]))

    return f"""
    <section class="case-card">
      <div class="case-header">
        <div>
          <h2>{case_name}</h2>
          <p class="meta"><b>status:</b> {status} &nbsp; <b>object_count:</b> {float(case["object_count"]):.1f}</p>
          <p class="meta"><b>sample_key:</b> {sample_key}</p>
          <p class="meta"><b>video_path:</b> {video_path}</p>
          <p class="meta"><b>context frames:</b> {context_frames}</p>
          <p class="caption">{caption}</p>
        </div>
        <div class="actions">
          <a href="{report_href}">open case report</a>
        </div>
      </div>
      <div class="media-grid">
        <figure>
          <img src="{prompt_src}" loading="lazy" />
          <figcaption>Prompt preview</figcaption>
        </figure>
        <figure>
          <video controls preload="none" playsinline src="{input_video_src}"></video>
          <figcaption>Input pre-pipe overlay</figcaption>
        </figure>
        <figure>
          <video controls preload="none" playsinline src="{box_video_src}"></video>
          <figcaption>Aux predicted boxes</figcaption>
        </figure>
        <figure>
          <video controls preload="none" playsinline src="{track_video_src}"></video>
          <figcaption>Aux predicted tracks</figcaption>
        </figure>
      </div>
      <details>
        <summary>Metrics</summary>
        <pre>{metrics_json}</pre>
      </details>
    </section>
    """


def build_html(cases: list[dict], *, root_dir: Path, output_html: Path, include_empty: bool) -> str:
    output_dir = output_html.parent
    body = "".join(
        _render_case(case, root_dir=root_dir, output_dir=output_dir)
        for case in cases
    )
    included_cases = len(cases)
    empty_note = "included" if include_empty else "filtered out"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PhysInOne Aux Overlay Gallery</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe6;
      --panel: #fffdf8;
      --line: #d9d0c2;
      --text: #1f1f1f;
      --muted: #5f5a53;
      --link: #0b5cad;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, #efe6d3 0, transparent 24%),
        linear-gradient(180deg, #f6f1e7 0%, #f2ede3 100%);
    }}
    .page {{
      max-width: 1680px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    .intro {{
      margin: 0 0 24px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .case-list {{
      display: grid;
      gap: 20px;
    }}
    .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    }}
    .case-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    .case-header h2 {{
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.25;
    }}
    .meta {{
      margin: 4px 0;
      color: var(--muted);
      word-break: break-word;
    }}
    .caption {{
      margin: 12px 0 0;
      line-height: 1.6;
    }}
    .actions a {{
      display: inline-block;
      white-space: nowrap;
      color: var(--link);
      text-decoration: none;
      font-weight: 600;
    }}
    .actions a:hover {{
      text-decoration: underline;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 16px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: #ffffff;
    }}
    img, video {{
      display: block;
      width: 100%;
      background: #000;
    }}
    figcaption {{
      padding: 10px 12px;
      font-size: 13px;
      color: var(--muted);
      border-top: 1px solid var(--line);
    }}
    details {{
      margin-top: 14px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    pre {{
      margin: 12px 0 0;
      padding: 14px;
      overflow-x: auto;
      border-radius: 10px;
      background: #faf7f0;
      border: 1px solid var(--line);
      white-space: pre-wrap;
    }}
    @media (max-width: 1100px) {{
      .media-grid {{
        grid-template-columns: 1fr;
      }}
      .case-header {{
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>PhysInOne Aux Overlay Gallery</h1>
    <p class="intro">
      This page aggregates PhysInOne single-sample train-forward auxiliary overlays into one view.
      Cases found: {included_cases}. Zero-object cases are {empty_note}.
      Each card embeds the prompt preview, input overlay, predicted box overlay, and predicted track overlay.
    </p>
    <div class="case-list">
      {body}
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(root_dir, include_empty=bool(args.include_empty))
    if not cases:
        raise RuntimeError(f"no result.json found under {root_dir}")
    html_text = build_html(
        cases,
        root_dir=root_dir,
        output_html=output_html,
        include_empty=bool(args.include_empty),
    )
    output_html.write_text(html_text, encoding="utf-8")
    print(output_html)


if __name__ == "__main__":
    main()
