#!/usr/bin/env python3
"""Build a simple portal from stage0 output sidecar json files."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portal from stage0 sidecar json files.")
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/output"),
    )
    parser.add_argument(
        "--portal_subdir",
        type=Path,
        default=Path("tools/visualization/output_sidecar_portal"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_root(root: Path, path: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def web_path(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.replace(os.sep, "/").lstrip("/")
    return f"/{normalized}"


def ensure_symlink(target: Path, link_path: Path) -> str | None:
    if not target.exists():
        return None
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return link_path.name
        link_path.unlink()
    link_path.symlink_to(target)
    return link_path.name


def sanitize_token(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._")
    return safe or "item"


def resolve_asset_path(
    *,
    raw_path: str,
    benchmark_root: Path,
    portal_dir: Path,
    asset_dir: Path,
    link_name: str,
) -> str | None:
    candidate = Path(raw_path)
    if candidate.exists() and candidate.is_relative_to(benchmark_root):
        return relative_to_root(benchmark_root, candidate)
    if not candidate.exists():
        return None
    linked_name = ensure_symlink(candidate, asset_dir / link_name)
    if not linked_name:
        return None
    return relative_to_root(benchmark_root, asset_dir / linked_name)


def media_html(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    resolved = web_path(path)
    if not resolved:
        return "<div class='missing'>Missing</div>"
    lowered = path.lower()
    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return f"<img loading='lazy' src='{html.escape(resolved)}' alt='input image'>"
    if lowered.endswith(".mp4"):
        return (
            f"<video controls preload='none' muted playsinline>"
            f"<source src='{html.escape(resolved)}' type='video/mp4'>"
            "</video>"
        )
    return f"<a href='{html.escape(resolved)}' target='_blank' rel='noreferrer'>{html.escape(Path(path).name)}</a>"


def render_input_group(paths: list[str]) -> str:
    if not paths:
        return "<div class='media-grid single'><div class='missing'>Missing</div></div>"
    cards = []
    for idx, path in enumerate(paths, start=1):
        cards.append(
            "<div class='media-slot'>"
            f"<div class='slot-head'>Input {idx}</div>"
            f"{media_html(path)}"
            "</div>"
        )
    grid_class = "media-grid multi" if len(paths) > 1 else "media-grid single"
    return f"<div class='{grid_class}'>{''.join(cards)}</div>"


def render_output_slot(path: str | None) -> str:
    return (
        "<div class='media-slot output-slot'>"
        "<div class='slot-head'>Output</div>"
        f"{media_html(path)}"
        "</div>"
    )


def collect_entries(
    *,
    benchmark_root: Path,
    output_root: Path,
    portal_dir: Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    assets_root = portal_dir / "assets" / "records"

    for json_path in sorted(output_root.rglob("*.json")):
        payload = read_json(json_path)
        paths = payload.get("paths", {})
        if not isinstance(paths, dict):
            paths = {}

        model_name = str(payload.get("model_name") or json_path.parent.name)
        dataset = str(payload.get("dataset") or "unknown")
        sample_id = str(payload.get("sample_id") or json_path.stem)
        record_tag = sanitize_token(f"{model_name}__{dataset}__{sample_id}")
        asset_dir = assets_root / record_tag

        raw_input_path = paths.get("input_path")
        input_specs: list[str] = []
        if isinstance(raw_input_path, str) and raw_input_path:
            input_specs = [raw_input_path]
        elif isinstance(raw_input_path, list):
            input_specs = [item for item in raw_input_path if isinstance(item, str) and item]

        input_assets: list[str] = []
        for idx, raw_path in enumerate(input_specs):
            suffix = Path(raw_path).suffix or ".bin"
            linked = resolve_asset_path(
                raw_path=raw_path,
                benchmark_root=benchmark_root,
                portal_dir=portal_dir,
                asset_dir=asset_dir,
                link_name=f"input_{idx:02d}{suffix}",
            )
            if linked:
                input_assets.append(linked)

        output_asset = None
        raw_output_path = paths.get("output_video_path")
        if isinstance(raw_output_path, str) and raw_output_path:
            candidate = Path(raw_output_path)
            if candidate.exists():
                output_asset = relative_to_root(benchmark_root, candidate)
        if output_asset is None:
            sibling_mp4 = json_path.with_suffix(".mp4")
            if sibling_mp4.exists():
                output_asset = relative_to_root(benchmark_root, sibling_mp4)

        entries.append(
            {
                "model_name": model_name,
                "dataset": dataset,
                "sample_id": sample_id,
                "caption": str(payload.get("caption") or ""),
                "status": str(payload.get("status") or ""),
                "input_assets": input_assets,
                "output_asset": output_asset,
                "json_relpath": relative_to_root(benchmark_root, json_path),
            }
        )
    return entries


def render_cards(entries: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for entry in entries:
        model_name = html.escape(entry["model_name"])
        dataset = html.escape(entry["dataset"])
        sample_id = html.escape(entry["sample_id"])
        caption = html.escape(entry["caption"])
        status = html.escape(entry["status"])
        json_relpath = html.escape(entry["json_relpath"])
        input_html = render_input_group(entry.get("input_assets", []))
        output_html = render_output_slot(entry.get("output_asset"))
        chunks.append(
            "<article class='record-card' "
            f"data-model='{model_name.lower()}' "
            f"data-dataset='{dataset.lower()}' "
            f"data-sample-id='{sample_id.lower()}' "
            f"data-caption='{caption.lower()}'>"
            "<div class='meta-row'>"
            f"<span class='badge model'>{model_name}</span>"
            f"<span class='badge dataset'>{dataset}</span>"
            f"<span class='badge status'>{status}</span>"
            "</div>"
            f"<h3>{sample_id}</h3>"
            f"<p class='caption'>{caption}</p>"
            f"<p class='json-path'>{json_relpath}</p>"
            "<div class='record-grid'>"
            f"{input_html}"
            f"{output_html}"
            "</div>"
            "</article>"
        )
    return "".join(chunks)


def build_html(entries: list[dict[str, Any]]) -> str:
    dataset_options = "".join(
        f"<option value='{html.escape(dataset)}'>{html.escape(dataset)}</option>"
        for dataset in sorted({entry['dataset'] for entry in entries})
    )
    model_options = "".join(
        f"<option value='{html.escape(model)}'>{html.escape(model)}</option>"
        for model in sorted({entry['model_name'] for entry in entries})
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage0 Output Sidecars</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1e1b16;
      --muted: #6e675d;
      --line: #d8cfbf;
      --accent: #b5532d;
      --accent-soft: #f3d7c9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(181,83,45,0.10), transparent 26%),
        linear-gradient(180deg, #f9f6ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .shell {{
      width: min(1600px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    .hero {{
      margin-bottom: 24px;
      padding: 24px 28px;
      background: rgba(255,253,248,0.90);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 14px 40px rgba(33, 24, 16, 0.06);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .filters {{
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: 12px;
      margin: 20px 0 24px;
    }}
    input, select {{
      width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
    }}
    .record-list {{
      display: grid;
      gap: 18px;
    }}
    .record-card {{
      padding: 18px;
      background: rgba(255,253,248,0.95);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(33, 24, 16, 0.05);
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      background: #efe7da;
      color: #4f4338;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    .badge.model {{
      background: var(--accent-soft);
      color: #6e2a13;
    }}
    .record-card h3 {{
      margin: 0 0 6px;
      font-size: 18px;
      line-height: 1.25;
    }}
    .caption {{
      margin: 0 0 6px;
      color: var(--ink);
      font-size: 15px;
      line-height: 1.5;
    }}
    .json-path {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }}
    .record-grid {{
      display: grid;
      grid-template-columns: minmax(340px, 1fr) minmax(340px, 1fr);
      gap: 14px;
      align-items: start;
    }}
    .media-grid {{
      display: grid;
      gap: 12px;
    }}
    .media-grid.multi {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .media-slot {{
      background: #fbf8f2;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      min-height: 220px;
    }}
    .slot-head {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      font-weight: 700;
      color: #55493d;
      background: rgba(239, 231, 218, 0.65);
    }}
    .output-slot .slot-head {{
      background: rgba(243, 215, 201, 0.7);
      color: #6e2a13;
    }}
    video, img {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 220px;
      object-fit: contain;
      background: #0d0d0d;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 220px;
      padding: 12px;
      color: var(--muted);
      background: repeating-linear-gradient(
        45deg,
        rgba(216, 207, 191, 0.35),
        rgba(216, 207, 191, 0.35) 10px,
        rgba(255, 253, 248, 0.75) 10px,
        rgba(255, 253, 248, 0.75) 20px
      );
    }}
    @media (max-width: 1080px) {{
      .filters {{
        grid-template-columns: 1fr;
      }}
      .record-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Stage0 Output Sidecar Portal</h1>
      <p>Each card is built directly from one output sidecar json and visualizes <code>paths.input_path</code>, <code>paths.output_video_path</code>, and <code>caption</code>.</p>
    </section>
    <section class="filters">
      <input id="searchBox" type="search" placeholder="Search sample id or caption">
      <select id="modelFilter">
        <option value="">All models</option>
        {model_options}
      </select>
      <select id="datasetFilter">
        <option value="">All datasets</option>
        {dataset_options}
      </select>
    </section>
    <section id="recordList" class="record-list">
      {render_cards(entries)}
    </section>
  </div>
  <script>
    const searchBox = document.getElementById('searchBox');
    const modelFilter = document.getElementById('modelFilter');
    const datasetFilter = document.getElementById('datasetFilter');
    const cards = Array.from(document.querySelectorAll('.record-card'));
    function applyFilters() {{
      const search = searchBox.value.trim().toLowerCase();
      const model = modelFilter.value.toLowerCase();
      const dataset = datasetFilter.value.toLowerCase();
      for (const card of cards) {{
        const matchesModel = !model || card.dataset.model === model;
        const matchesDataset = !dataset || card.dataset.dataset === dataset;
        const haystack = `${{card.dataset.sampleId}} ${{card.dataset.caption}} ${{card.dataset.model}} ${{card.dataset.dataset}}`.toLowerCase();
        const matchesSearch = !search || haystack.includes(search);
        card.style.display = matchesModel && matchesDataset && matchesSearch ? '' : 'none';
      }}
    }}
    searchBox.addEventListener('input', applyFilters);
    modelFilter.addEventListener('change', applyFilters);
    datasetFilter.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    portal_dir = (args.benchmark_root / args.portal_subdir).resolve()
    portal_dir.mkdir(parents=True, exist_ok=True)

    entries = collect_entries(
        benchmark_root=args.benchmark_root,
        output_root=args.output_root,
        portal_dir=portal_dir,
    )
    html_text = build_html(entries)
    index_path = portal_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    summary = {
        "benchmark_root": str(args.benchmark_root),
        "output_root": str(args.output_root),
        "num_entries": len(entries),
        "index_path": str(index_path),
    }
    (portal_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(index_path)


if __name__ == "__main__":
    main()
