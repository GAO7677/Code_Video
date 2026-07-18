#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML overview page for the 0718 toy dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--page-title", default="0718 Toy Dataset Viewer")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relpath(path: str | Path, start: Path) -> str:
    return os.path.relpath(str(path), str(start)).replace(os.sep, "/")


def _safe(text: str) -> str:
    return html.escape(text, quote=True)


def _video_card(*, label: str, manifest: dict, output_root: Path) -> str:
    video_rel = _relpath(manifest["video"], output_root)
    meta_rel = _relpath(manifest["meta"], output_root)
    phrases = ", ".join(str(item) for item in manifest.get("object_phrases", []))
    return f"""
    <article class="video-card">
      <div class="card-top">
        <span class="chip">{_safe(label)}</span>
        <span class="case-id">{_safe(str(manifest.get("sample_key", "")))}</span>
      </div>
      <video controls preload="metadata" src="{_safe(video_rel)}"></video>
      <div class="card-body">
        <p class="caption">{_safe(str(manifest.get("short_caption", "")))}</p>
        <p class="meta-line"><strong>Objects</strong> {_safe(phrases)}</p>
        <p class="meta-line"><strong>Meta</strong> <a href="{_safe(meta_rel)}">{_safe(Path(str(manifest["meta"])).name)}</a></p>
      </div>
    </article>
    """


def _pair_section(pair_manifest: dict, output_root: Path) -> str:
    attribute = str(pair_manifest["attribute"]).replace("_", " ")
    anchor = _video_card(
        label=f"Anchor · {pair_manifest['anchor_label']}",
        manifest=pair_manifest["anchor"],
        output_root=output_root,
    )
    variant = _video_card(
        label=f"Variant · {pair_manifest['variant_label']}",
        manifest=pair_manifest["variant"],
        output_root=output_root,
    )
    return f"""
    <section class="pair-section">
      <div class="section-head">
        <div>
          <div class="eyebrow">Single-attribute pair</div>
          <h2>{_safe(attribute.title())}</h2>
        </div>
        <p class="section-note">This pair is intended to change only <strong>{_safe(attribute)}</strong>.</p>
      </div>
      <div class="pair-grid">
        {anchor}
        {variant}
      </div>
    </section>
    """


def build_html(dataset_manifest: dict, output_root: Path, page_title: str) -> str:
    base_case = _video_card(label="Base Case", manifest=dataset_manifest["base_case"], output_root=output_root)
    pair_sections = "\n".join(_pair_section(pair, output_root) for pair in dataset_manifest.get("pairs", []))
    rendered_attributes = ", ".join(str(item) for item in dataset_manifest.get("rendered_attributes", []))
    dataset_root = _relpath(dataset_manifest["dataset_root"], output_root)
    manifest_rel = _relpath(Path(dataset_manifest["dataset_root"]) / "dataset_manifest.json", output_root)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_safe(page_title)}</title>
  <style>
    :root {{
      --bg: #f2ede3;
      --panel: rgba(255, 251, 245, 0.86);
      --ink: #1f2430;
      --muted: #645f57;
      --warm: #b85c38;
      --line: rgba(45, 38, 30, 0.12);
      --shadow: 0 18px 48px rgba(61, 43, 27, 0.12);
      --radius: 24px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.8), transparent 36%),
        linear-gradient(180deg, #f6f2ea 0%, #ece4d7 100%);
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }}
    .hero {{
      background: linear-gradient(145deg, rgba(255,255,255,0.78), rgba(244,232,216,0.92));
      border: 1px solid var(--line);
      border-radius: 32px;
      box-shadow: var(--shadow);
      padding: 30px 30px 26px;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      right: -70px;
      top: -60px;
      width: 220px;
      height: 220px;
      border-radius: 999px;
      background: rgba(184, 92, 56, 0.10);
      filter: blur(4px);
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 11px;
      color: var(--warm);
      margin-bottom: 12px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(30px, 5vw, 54px);
      line-height: 0.98;
      font-weight: 700;
    }}
    .hero p {{
      margin: 0;
      max-width: 760px;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.6;
    }}
    .meta-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 20px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      font-size: 13px;
      color: var(--ink);
    }}
    .overview {{
      margin-top: 26px;
      display: grid;
      gap: 24px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 22px;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 28px;
    }}
    .panel a {{
      color: var(--warm);
      text-decoration: none;
    }}
    .panel a:hover {{
      text-decoration: underline;
    }}
    .pair-section {{
      margin-top: 26px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 22px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 28px;
    }}
    .section-note {{
      margin: 0;
      max-width: 360px;
      text-align: right;
      color: var(--muted);
      line-height: 1.5;
      font-size: 15px;
    }}
    .pair-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .single-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
    }}
    .video-card {{
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(45, 38, 30, 0.08);
      border-radius: 22px;
      overflow: hidden;
    }}
    .video-card video {{
      width: 100%;
      display: block;
      background: #000;
      aspect-ratio: 16 / 9;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px 0;
      align-items: center;
    }}
    .case-id {{
      font-size: 12px;
      color: var(--muted);
      text-align: right;
    }}
    .card-body {{
      padding: 14px 16px 18px;
    }}
    .caption {{
      margin: 0 0 10px;
      font-size: 17px;
      line-height: 1.45;
    }}
    .meta-line {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }}
    @media (max-width: 860px) {{
      .pair-grid {{
        grid-template-columns: 1fr;
      }}
      .section-head {{
        align-items: start;
        flex-direction: column;
      }}
      .section-note {{
        max-width: none;
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="eyebrow">Toy Physics Pairs</div>
      <h1>{_safe(page_title)}</h1>
      <p>
        A compact local viewer for the July 18, 2026 toy dataset. The base case is a ball thrown into a wooden
        block, and each pair below is intended to change just one attribute while keeping the interaction template stable.
      </p>
      <div class="meta-strip">
        <span class="chip">Case: {_safe(str(dataset_manifest.get("case_key", "")))}</span>
        <span class="chip">Seed: {_safe(str(dataset_manifest.get("seed", "")))}</span>
        <span class="chip">Resolution: {_safe(str(dataset_manifest.get("width", "")))}x{_safe(str(dataset_manifest.get("height", "")))}</span>
        <span class="chip">Rendered attributes: {_safe(rendered_attributes)}</span>
        <span class="chip"><a href="{_safe(manifest_rel)}">dataset_manifest.json</a></span>
        <span class="chip"><a href="{_safe(dataset_root)}">dataset root</a></span>
      </div>
    </section>

    <section class="overview">
      <div class="panel">
        <div class="eyebrow">Reference</div>
        <h2>Base Case</h2>
        <div class="single-grid">
          {base_case}
        </div>
      </div>
      {pair_sections}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_root = args.output_root or (dataset_root / "html")
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_manifest = _read_json(dataset_root / "dataset_manifest.json")
    html_text = build_html(dataset_manifest, output_root=output_root, page_title=args.page_title)
    (output_root / "index.html").write_text(html_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "output_root": str(output_root),
                "index_html": str(output_root / "index.html"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
