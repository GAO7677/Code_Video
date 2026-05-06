#!/usr/bin/env python3
"""Build a portal comparing caption vs null-caption outputs for matched cases."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_ORIG_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V")
DEFAULT_NULL_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")
DEFAULT_DIAGNOSTICS_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_train/train_0419/geometry_diagnostics/_debug_run_tracking"
)
MODEL_RELATIVE_DIR = Path("output") / "VACE_1_3B_V2V" / "context_08f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig_benchmark_root", type=Path, default=DEFAULT_ORIG_BENCHMARK_ROOT)
    parser.add_argument("--null_benchmark_root", type=Path, default=DEFAULT_NULL_BENCHMARK_ROOT)
    parser.add_argument("--diagnostics_root", type=Path, default=DEFAULT_DIAGNOSTICS_ROOT)
    parser.add_argument(
        "--portal_subdir",
        type=Path,
        default=Path("tools/visualization/caption_vs_nullcaption_portal"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative_to_root(root: Path, path: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def web_path(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.replace(os.sep, "/").lstrip("/")
    return f"/{normalized}"


def sanitize_token(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._")
    return safe or "item"


def ensure_clean_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()


def ensure_symlink(target: Path, link_path: Path) -> Path | None:
    if not target.exists():
        return None
    ensure_clean_parent(link_path)
    link_path.symlink_to(target)
    return link_path


def expose_asset(
    *,
    target: str | Path | None,
    benchmark_root: Path,
    assets_dir: Path,
    link_name: str,
) -> str | None:
    if target is None:
        return None
    candidate = Path(str(target))
    if not candidate.exists():
        return None
    if candidate.is_relative_to(benchmark_root):
        return relative_to_root(benchmark_root, candidate)
    linked = ensure_symlink(candidate, assets_dir / link_name)
    if linked is None:
        return None
    return relative_to_root(benchmark_root, linked)


def media_html(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    resolved = web_path(path)
    if not resolved:
        return "<div class='missing'>Missing</div>"
    lowered = path.lower()
    if lowered.endswith(".mp4"):
        return (
            f"<video controls preload='metadata' muted playsinline>"
            f"<source src='{html.escape(resolved)}' type='video/mp4'>"
            "</video>"
        )
    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return f"<img loading='lazy' src='{html.escape(resolved)}' alt='asset'>"
    return f"<a href='{html.escape(resolved)}'>{html.escape(Path(path).name)}</a>"


def text_block(text: str) -> str:
    if not text:
        return "<div class='text-body empty'>(empty caption)</div>"
    return f"<div class='text-body'>{html.escape(text)}</div>"


def render_slot(title: str, body: str, extra_class: str = "") -> str:
    class_attr = "slot"
    if extra_class:
        class_attr += f" {extra_class}"
    return (
        f"<section class='{class_attr}'>"
        f"<div class='slot-head'>{html.escape(title)}</div>"
        f"{body}"
        "</section>"
    )


def load_diagnostics_index(diagnostics_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for diagnostics_path in sorted(diagnostics_root.glob("*/diagnostics.json")):
        payload = read_json(diagnostics_path)
        dataset = payload.get("dataset")
        sample_id = payload.get("sample_id")
        if isinstance(dataset, str) and isinstance(sample_id, str):
            index[(dataset, sample_id)] = payload
    return index


def collect_records(orig_root: Path, null_root: Path, diagnostics_root: Path, portal_dir: Path) -> list[dict[str, Any]]:
    orig_sidecars = orig_root / MODEL_RELATIVE_DIR
    null_sidecars = null_root / MODEL_RELATIVE_DIR
    diag_index = load_diagnostics_index(diagnostics_root)
    assets_root = portal_dir / "assets"
    records: list[dict[str, Any]] = []

    for null_json_path in sorted(null_sidecars.glob("*.json")):
        orig_json_path = orig_sidecars / null_json_path.name
        if not orig_json_path.exists():
            continue

        null_payload = read_json(null_json_path)
        orig_payload = read_json(orig_json_path)
        null_paths = null_payload.get("paths", {})
        orig_paths = orig_payload.get("paths", {})
        if not isinstance(null_paths, dict) or not isinstance(orig_paths, dict):
            continue

        dataset = str(null_payload.get("dataset") or orig_payload.get("dataset") or "unknown")
        sample_id = str(null_payload.get("sample_id") or orig_payload.get("sample_id") or null_json_path.stem)
        record_tag = sanitize_token(f"{dataset}__{sample_id}")
        asset_dir = assets_root / record_tag
        asset_dir.mkdir(parents=True, exist_ok=True)

        orig_output = orig_paths.get("output_video_path")
        null_output = null_paths.get("output_video_path")
        if not isinstance(orig_output, str) or not isinstance(null_output, str):
            continue

        context_path = null_paths.get("context_video_path") or orig_paths.get("context_video_path")
        gt_path = null_paths.get("full_video_path") or orig_paths.get("full_video_path")
        diagnostics = diag_index.get((dataset, sample_id), {})
        artifacts = diagnostics.get("artifacts", {}) if isinstance(diagnostics, dict) else {}
        summary = diagnostics.get("summary", {}) if isinstance(diagnostics, dict) else {}
        analysis = diagnostics.get("analysis", []) if isinstance(diagnostics, dict) else []
        if not isinstance(artifacts, dict):
            artifacts = {}
        if not isinstance(summary, dict):
            summary = {}

        born_track_items = artifacts.get("generated_born_single_track_videos")
        born_track_paths: list[str] = []
        if isinstance(born_track_items, list):
            for idx, item in enumerate(born_track_items[:6], start=1):
                if not isinstance(item, dict):
                    continue
                exposed = expose_asset(
                    target=item.get("path"),
                    benchmark_root=null_root,
                    assets_dir=asset_dir,
                    link_name=f"born_track_{idx:02d}.mp4",
                )
                if exposed:
                    born_track_paths.append(exposed)

        analysis_lines: list[str] = []
        if isinstance(analysis, list):
            for item in analysis[:8]:
                if isinstance(item, str) and item.strip():
                    analysis_lines.append(item.strip())

        record = {
            "dataset": dataset,
            "sample_id": sample_id,
            "orig_caption": str(orig_payload.get("caption") or ""),
            "null_caption": str(null_payload.get("caption") or ""),
            "context_video": expose_asset(
                target=context_path,
                benchmark_root=null_root,
                assets_dir=asset_dir,
                link_name="context_video.mp4",
            ),
            "gt_full_video": expose_asset(
                target=gt_path,
                benchmark_root=null_root,
                assets_dir=asset_dir,
                link_name="gt_full_video.mp4",
            ),
            "orig_output_video": expose_asset(
                target=orig_output,
                benchmark_root=null_root,
                assets_dir=asset_dir,
                link_name="generated_with_caption.mp4",
            ),
            "null_output_video": expose_asset(
                target=null_output,
                benchmark_root=null_root,
                assets_dir=asset_dir,
                link_name="generated_nullcaption.mp4",
            ),
            "comparison_curves_png": expose_asset(
                target=artifacts.get("comparison_curves_png"),
                benchmark_root=null_root,
                assets_dir=asset_dir,
                link_name="comparison_curves.png",
            ),
            "context_diagnostic_video": expose_asset(
                target=artifacts.get("context_diagnostic_video"),
                benchmark_root=null_root,
                assets_dir=asset_dir,
                link_name="context_diagnostic.mp4",
            ),
            "generated_diagnostic_video": expose_asset(
                target=artifacts.get("generated_diagnostic_video"),
                benchmark_root=null_root,
                assets_dir=asset_dir,
                link_name="generated_diagnostic.mp4",
            ),
            "gt_diagnostic_video": expose_asset(
                target=artifacts.get("gt_diagnostic_video"),
                benchmark_root=null_root,
                assets_dir=asset_dir,
                link_name="gt_diagnostic.mp4",
            ),
            "born_track_videos": born_track_paths,
            "diagnostic_summary": json.dumps(summary, ensure_ascii=False, indent=2) if summary else "",
            "analysis_text": "\n".join(analysis_lines),
            "orig_json_relpath": relative_to_root(orig_root, orig_json_path),
            "null_json_relpath": relative_to_root(null_root, null_json_path),
            "diagnostics_relpath": relative_to_root(diagnostics_root, diagnostics_root / f"vace_v2v_ctx08f__{dataset}__{sample_id}" / "diagnostics.json")
            if (diagnostics_root / f"vace_v2v_ctx08f__{dataset}__{sample_id}" / "diagnostics.json").exists()
            else None,
        }
        records.append(record)
    return records


def render_track_strip(paths: list[str]) -> str:
    if not paths:
        return "<div class='missing'>No born-track videos yet</div>"
    return "<div class='track-strip'>" + "".join(
        render_slot(f"generated_born_track_{idx}", media_html(path), "track-slot")
        for idx, path in enumerate(paths, start=1)
    ) + "</div>"


def render_cards(records: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for record in records:
        dataset = html.escape(record["dataset"])
        sample_id = html.escape(record["sample_id"])
        orig_json = html.escape(record["orig_json_relpath"])
        null_json = html.escape(record["null_json_relpath"])
        diagnostics_relpath = html.escape(record["diagnostics_relpath"] or "")
        cards.append(
            "<article class='case-card' "
            f"data-dataset='{dataset.lower()}' "
            f"data-sample-id='{sample_id.lower()}' "
            f"data-caption='{html.escape(record['orig_caption'].lower())}'>"
            "<div class='meta-row'>"
            f"<span class='badge dataset'>{dataset}</span>"
            "<span class='badge compare'>caption vs null-caption</span>"
            "</div>"
            f"<h3>{sample_id}</h3>"
            "<div class='grid context-grid'>"
            f"{render_slot('context_video', media_html(record['context_video']))}"
            f"{render_slot('gt_full_video', media_html(record['gt_full_video']), 'gt-slot')}"
            f"{render_slot('caption_text', text_block(record['orig_caption']), 'text-slot')}"
            f"{render_slot('null_caption_text', text_block(record['null_caption']), 'text-slot')}"
            "</div>"
            "<div class='grid compare-row-grid'>"
            f"{render_slot('generated_with_caption', media_html(record['orig_output_video']), 'caption-slot')}"
            f"{render_slot('generated_nullcaption', media_html(record['null_output_video']), 'null-slot')}"
            "</div>"
            "<div class='grid diagnostic-grid'>"
            f"{render_slot('comparison_curves', media_html(record['comparison_curves_png']), 'curve-slot')}"
            f"{render_slot('context_diagnostic_overlay', media_html(record['context_diagnostic_video']), 'diag-slot')}"
            f"{render_slot('gt_diagnostic_overlay', media_html(record['gt_diagnostic_video']), 'diag-slot')}"
            f"{render_slot('generated_diagnostic_overlay', media_html(record['generated_diagnostic_video']), 'diag-slot')}"
            f"{render_slot('diagnostic_summary', text_block(record['diagnostic_summary']), 'text-slot')}"
            f"{render_slot('analysis', text_block(record['analysis_text']), 'text-slot')}"
            "</div>"
            f"{render_slot('generated_born_tracks', render_track_strip(record['born_track_videos']), 'born-strip-slot')}"
            "<div class='path-grid'>"
            f"<p><strong>caption json</strong><br>{orig_json}</p>"
            f"<p><strong>null json</strong><br>{null_json}</p>"
            f"<p><strong>diagnostics json</strong><br>{diagnostics_relpath or 'Missing'}</p>"
            "</div>"
            "</article>"
        )
    return "".join(cards)


def build_html(records: list[dict[str, Any]]) -> str:
    dataset_options = "".join(
        f"<option value='{html.escape(dataset)}'>{html.escape(dataset)}</option>"
        for dataset in sorted({record["dataset"] for record in records})
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Caption vs Null-Caption</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1e1b16;
      --muted: #6e675d;
      --line: #d8cfbf;
      --caption: #b5532d;
      --caption-soft: #f3d7c9;
      --null: #2f6d57;
      --null-soft: #d6ead9;
      --aux: #e8dfd0;
      --curve-soft: #efe8d9;
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
      width: min(1960px, calc(100vw - 20px));
      margin: 0 auto;
      padding: 14px 0 24px;
    }}
    .hero {{
      margin-bottom: 12px;
      padding: 14px 18px;
      background: rgba(255,253,248,0.90);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 26px rgba(33, 24, 16, 0.05);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .filters {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 8px;
      margin: 10px 0 12px;
    }}
    input, select {{
      width: 100%;
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }}
    .record-list {{
      display: grid;
      gap: 10px;
    }}
    .case-card {{
      padding: 10px;
      background: rgba(255,253,248,0.96);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(33, 24, 16, 0.04);
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 6px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: #efe7da;
      color: #4f4338;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    .badge.compare {{
      background: var(--aux);
      color: #5d4d3a;
    }}
    .case-card h3 {{
      margin: 0 0 8px;
      font-size: 14px;
      line-height: 1.2;
    }}
    .grid {{
      display: grid;
      gap: 8px;
      align-items: start;
      margin-bottom: 8px;
    }}
    .context-grid {{
      grid-template-columns: repeat(2, minmax(220px, 1fr));
    }}
    .compare-row-grid {{
      grid-template-columns: repeat(2, minmax(320px, 1fr));
    }}
    .diagnostic-grid {{
      grid-template-columns: repeat(3, minmax(220px, 1fr));
    }}
    .slot {{
      background: #fbf8f2;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      min-height: 156px;
    }}
    .slot-head {{
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      font-size: 11px;
      font-weight: 700;
      color: #55493d;
      background: rgba(239, 231, 218, 0.65);
    }}
    .caption-slot .slot-head {{
      background: rgba(243, 215, 201, 0.78);
      color: #6e2a13;
    }}
    .null-slot .slot-head {{
      background: rgba(214, 234, 217, 0.82);
      color: #28563c;
    }}
    .gt-slot .slot-head {{
      background: rgba(232, 223, 208, 0.82);
      color: #584936;
    }}
    .curve-slot .slot-head {{
      background: rgba(239, 232, 217, 0.82);
      color: #5c4c39;
    }}
    .born-strip-slot {{
      margin-bottom: 8px;
    }}
    .track-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px;
      padding: 8px;
    }}
    .track-slot {{
      min-height: 132px;
    }}
    video, img {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 156px;
      object-fit: contain;
      background: #0d0d0d;
    }}
    .track-slot video {{
      min-height: 132px;
    }}
    .text-body {{
      min-height: 156px;
      padding: 8px 10px;
      background: #fffdf9;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .text-body.empty {{
      color: var(--muted);
      font-style: italic;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 156px;
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
    .path-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }}
    .path-grid p {{
      margin: 0;
      padding: 8px 10px;
      background: #fffaf2;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      word-break: break-all;
    }}
    @media (max-width: 1100px) {{
      .context-grid, .compare-row-grid, .diagnostic-grid {{
        grid-template-columns: 1fr;
      }}
      .filters, .path-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Caption vs Null-Caption</h1>
      <p>Matched comparisons for the same VACE ctx08 case. Shared assets are exposed under this portal root so the browser can load context, GT, captioned generation, null-caption generation, diagnostic curves, diagnostic overlays, and born-track videos from one place.</p>
    </section>
    <section class="filters">
      <input id="searchBox" type="search" placeholder="Search sample id or caption text">
      <select id="datasetFilter">
        <option value="">All datasets</option>
        {dataset_options}
      </select>
    </section>
    <section id="recordList" class="record-list">
      {render_cards(records)}
    </section>
  </div>
  <script>
    const searchBox = document.getElementById('searchBox');
    const datasetFilter = document.getElementById('datasetFilter');
    const cards = Array.from(document.querySelectorAll('.case-card'));
    function applyFilters() {{
      const search = searchBox.value.trim().toLowerCase();
      const dataset = datasetFilter.value.toLowerCase();
      for (const card of cards) {{
        const matchesDataset = !dataset || card.dataset.dataset === dataset;
        const haystack = `${{card.dataset.sampleId}} ${{card.dataset.caption}} ${{card.dataset.dataset}}`.toLowerCase();
        const matchesSearch = !search || haystack.includes(search);
        card.style.display = matchesDataset && matchesSearch ? '' : 'none';
      }}
    }}
    searchBox.addEventListener('input', applyFilters);
    datasetFilter.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    orig_root = args.orig_benchmark_root.expanduser().resolve()
    null_root = args.null_benchmark_root.expanduser().resolve()
    diagnostics_root = args.diagnostics_root.expanduser().resolve()
    portal_dir = (null_root / args.portal_subdir).resolve()
    portal_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(orig_root, null_root, diagnostics_root, portal_dir)
    html_path = portal_dir / "index.html"
    html_path.write_text(build_html(records), encoding="utf-8")
    write_json(
        portal_dir / "build_summary.json",
        {
            "record_count": len(records),
            "html_path": str(html_path),
            "portal_url_path": f"/{relative_to_root(null_root, html_path)}",
        },
    )
    print(json.dumps({"record_count": len(records), "html_path": str(html_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
