from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return str(value)
    return "N/A"


def _video_index(videos_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(videos_root.rglob("*.mp4")):
        mapping.setdefault(path.name, path)
    return mapping


def _stable_card_id(label: str, video_name: str) -> str:
    safe_label = "".join(ch if ch.isalnum() else "-" for ch in label.lower()).strip("-")
    return f"{safe_label}-{video_name.rsplit('.', 1)[0][:16]}"


def _render_metadata_table(record: dict[str, Any]) -> str:
    rows: list[tuple[str, str]] = [
        ("video_name", str(record.get("video_name", ""))),
        ("label", str(record.get("label", ""))),
        ("resolution", f"{record.get('width', 'N/A')} x {record.get('height', 'N/A')}"),
        ("fps", _coerce_number(record.get("fps"))),
        ("duration_s", _coerce_number(record.get("duration"))),
        ("motion_score", _coerce_number(record.get("motion_score"))),
        ("motion_score_v2", _coerce_number(record.get("motion_score_v2"))),
        ("visual_quality_score", _coerce_number(record.get("visual_quality_score"))),
        ("text_bbox_num", _coerce_number(record.get("text_bbox_num"))),
        ("text_bbox_ratio", _coerce_number(record.get("text_bbox_ratio"))),
    ]
    items = []
    for key, value in rows:
        items.append(
            f"<div class='meta-item'><div class='meta-key'>{html.escape(key)}</div>"
            f"<div class='meta-value'>{html.escape(value)}</div></div>"
        )
    return "".join(items)


def _render_physical_annotation(record: dict[str, Any]) -> str:
    physical = record.get("physical_annotation")
    if not isinstance(physical, dict):
        return "<p class='muted'>No physical_annotation</p>"
    order = [
        "phys_law",
        "n0",
        "n1",
        "n2",
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
        "quantify_n0",
        "quantify_n1",
        "quantify_n2",
    ]
    blocks: list[str] = []
    for key in order:
        value = physical.get(key)
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            rendered = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            rendered = str(value)
        blocks.append(
            "<div class='phy-block'>"
            f"<div class='phy-key'>{html.escape(key)}</div>"
            f"<pre class='phy-value'>{html.escape(rendered)}</pre>"
            "</div>"
        )
    return "".join(blocks)


def _render_card(record: dict[str, Any], video_href: str) -> str:
    label = str(record.get("label", "unknown"))
    video_name = str(record.get("video_name", ""))
    card_id = _stable_card_id(label, video_name)
    caption = str(record.get("captions", "")).strip()
    caption_short = caption if len(caption) <= 320 else caption[:317] + "..."
    raw_json = json.dumps(record, ensure_ascii=False, indent=2)
    return f"""
<article class="card" data-label="{html.escape(label)}" id="{html.escape(card_id)}">
  <div class="card-top">
    <div class="tag">{html.escape(label)}</div>
    <div class="tag subtle">{html.escape(video_name)}</div>
  </div>
  <video controls preload="metadata" playsinline src="{html.escape(video_href)}"></video>
  <div class="section">
    <div class="section-title">Caption</div>
    <p class="caption">{html.escape(caption_short)}</p>
  </div>
  <div class="section">
    <div class="section-title">Video Stats</div>
    <div class="meta-grid">{_render_metadata_table(record)}</div>
  </div>
  <div class="section">
    <div class="section-title">Physical Annotation</div>
    <div class="phy-grid">{_render_physical_annotation(record)}</div>
  </div>
  <details>
    <summary>Raw JSON</summary>
    <pre class="raw-json">{html.escape(raw_json)}</pre>
  </details>
</article>
"""


def _render_index(
    *,
    selected_by_label: dict[str, list[dict[str, Any]]],
    label_counts_available: dict[str, int],
    output_dir: Path,
    dataset_root: Path,
    videos_root: Path,
    cases_per_label: int,
    seed: int,
) -> str:
    nav_items: list[str] = []
    sections: list[str] = []
    for label in sorted(selected_by_label):
        selected = selected_by_label[label]
        available = label_counts_available[label]
        label_id = "".join(ch if ch.isalnum() else "-" for ch in label.lower()).strip("-")
        nav_items.append(
            f"<a class='nav-pill' href='#{html.escape(label_id)}'>{html.escape(label)} "
            f"<span>{len(selected)}/{available}</span></a>"
        )
        cards = []
        for record in selected:
            video_name = str(record["video_name"])
            video_href = f"videos/{video_name}"
            cards.append(_render_card(record, video_href))
        sections.append(
            f"""
<section class="label-section" id="{html.escape(label_id)}">
  <div class="label-header">
    <h2>{html.escape(label)}</h2>
    <div class="label-meta">selected {len(selected)} / available {available}</div>
  </div>
  <div class="card-grid">
    {''.join(cards)}
  </div>
</section>
"""
        )

    summary = {
        "dataset_root": str(dataset_root),
        "videos_root": str(videos_root),
        "labels": len(selected_by_label),
        "cases_per_label_requested": cases_per_label,
        "cases_selected_total": sum(len(v) for v in selected_by_label.values()),
        "seed": seed,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WISA Label Gallery</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --paper: #fffaf4;
      --ink: #1f2933;
      --muted: #6b7280;
      --line: #d8cfc3;
      --accent: #ab3b2f;
      --accent-2: #28666e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(171,59,47,0.10), transparent 22rem),
        radial-gradient(circle at top right, rgba(40,102,110,0.10), transparent 24rem),
        var(--bg);
    }}
    .shell {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 10px 30px rgba(31,41,51,0.06);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 32px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 8px 0;
      color: var(--muted);
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .summary-item {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    .summary-key {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .summary-value {{
      margin-top: 6px;
      font-size: 16px;
      font-weight: 700;
      word-break: break-word;
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0 4px;
    }}
    .nav-pill {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      text-decoration: none;
      color: var(--ink);
      background: #fff;
      font-size: 14px;
    }}
    .nav-pill span {{
      color: var(--muted);
    }}
    .label-section {{
      margin-bottom: 34px;
    }}
    .label-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 14px;
      padding: 0 4px;
    }}
    .label-header h2 {{
      margin: 0;
      font-size: 24px;
    }}
    .label-meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 10px 24px rgba(31,41,51,0.05);
    }}
    .card-top {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .tag {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(171,59,47,0.12);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      word-break: break-word;
    }}
    .tag.subtle {{
      background: rgba(40,102,110,0.10);
      color: var(--accent-2);
      font-weight: 600;
    }}
    video {{
      width: 100%;
      border-radius: 12px;
      background: #000;
      border: 1px solid var(--line);
    }}
    .section {{
      margin-top: 14px;
    }}
    .section-title {{
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .caption {{
      margin: 0;
      line-height: 1.5;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px;
    }}
    .meta-item {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
    }}
    .meta-key {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .meta-value {{
      margin-top: 4px;
      font-size: 14px;
      font-weight: 600;
      word-break: break-word;
    }}
    .phy-grid {{
      display: grid;
      gap: 8px;
    }}
    .phy-block {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
    }}
    .phy-key {{
      font-size: 12px;
      font-weight: 700;
      color: var(--accent-2);
      margin-bottom: 6px;
    }}
    .phy-value, .raw-json {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.4;
    }}
    details {{
      margin-top: 12px;
      border-top: 1px dashed var(--line);
      padding-top: 12px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
    .muted {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>WISA-80K Local Label Gallery</h1>
      <p>Each label is randomly sampled from the currently downloaded local WISA videos and rendered with the original video plus the full metadata and physical annotation stored in <code>wisa-80k.json</code>.</p>
      <div class="summary">
        {''.join(
            f"<div class='summary-item'><div class='summary-key'>{html.escape(k)}</div><div class='summary-value'>{html.escape(str(v))}</div></div>"
            for k, v in summary.items()
        )}
      </div>
      <div class="nav">
        {''.join(nav_items)}
      </div>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local WISA gallery with random samples per label."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/gaoya/dataset/qihoo360-WISA-80K"),
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=Path("/data/gaoya/dataset/qihoo360-WISA-80K/videos"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("/data/gaoya/dataset/qihoo360-WISA-80K/data/wisa-80k.json"),
    )
    parser.add_argument("--cases-per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/wisa_label_gallery"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    videos_root = args.videos_root.expanduser().resolve()
    metadata_path = args.metadata_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selected_videos_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_videos_dir.mkdir(parents=True, exist_ok=True)

    if args.cases_per_label <= 0:
        raise ValueError(f"--cases-per-label must be positive, got {args.cases_per_label}")
    if not videos_root.is_dir():
        raise FileNotFoundError(f"videos root not found: {videos_root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata path not found: {metadata_path}")

    metadata = _read_json(metadata_path)
    if not isinstance(metadata, list):
        raise RuntimeError(f"expected list metadata, got {type(metadata).__name__}")
    video_map = _video_index(videos_root)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, entry in enumerate(metadata):
        if not isinstance(entry, dict):
            continue
        video_name = entry.get("video_name")
        label = entry.get("label")
        if not isinstance(video_name, str) or not isinstance(label, str):
            continue
        local_path = video_map.get(video_name)
        if local_path is None:
            continue
        item = dict(entry)
        item["__row_index__"] = row_index
        item["__local_video_path__"] = str(local_path)
        grouped[label].append(item)

    rng = random.Random(args.seed)
    selected_by_label: dict[str, list[dict[str, Any]]] = {}
    label_counts_available = {label: len(items) for label, items in grouped.items()}
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        selected = items[: min(args.cases_per_label, len(items))]
        for item in selected:
            source = Path(item["__local_video_path__"])
            target = selected_videos_dir / source.name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source)
        selected_by_label[label] = selected

    manifest = {
        "dataset_root": str(dataset_root),
        "videos_root": str(videos_root),
        "metadata_path": str(metadata_path),
        "cases_per_label": int(args.cases_per_label),
        "seed": int(args.seed),
        "labels_total": len(selected_by_label),
        "selected_total": sum(len(v) for v in selected_by_label.values()),
        "available_counts_by_label": label_counts_available,
        "selected_video_names_by_label": {
            label: [str(item["video_name"]) for item in items]
            for label, items in selected_by_label.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    index_html = _render_index(
        selected_by_label=selected_by_label,
        label_counts_available=label_counts_available,
        output_dir=output_dir,
        dataset_root=dataset_root,
        videos_root=videos_root,
        cases_per_label=int(args.cases_per_label),
        seed=int(args.seed),
    )
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index_html": str(output_dir / "index.html"),
                "manifest": str(output_dir / "manifest.json"),
                "labels_total": len(selected_by_label),
                "selected_total": sum(len(v) for v in selected_by_label.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
