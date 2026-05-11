#!/usr/bin/env python3
"""Build a portal for 0002 baseline multi-seed caption vs null-caption comparisons."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any


BENCH_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")
WORK_ROOT = BENCH_ROOT / "tools" / "seed_sweeps" / "0002_baseline_multiseed"
PORTAL_DIR = BENCH_ROOT / "tools" / "visualization" / "0002_seed_sweep_baseline_portal"
ASSETS_DIR = PORTAL_DIR / "assets"
RESULT_INDEX = WORK_ROOT / "result_index.json"
MANIFEST_PATH = WORK_ROOT / "meta" / "manifest.json"


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
    return "/" + path.replace(os.sep, "/").lstrip("/")


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


def expose_asset(target: Path | None, assets_dir: Path, link_name: str) -> str | None:
    if target is None or not target.exists():
        return None
    if target.is_relative_to(BENCH_ROOT):
        return relative_to_root(BENCH_ROOT, target)
    linked = ensure_symlink(target, assets_dir / link_name)
    if linked is None:
        return None
    return relative_to_root(BENCH_ROOT, linked)


def media_html(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    resolved = web_path(path)
    return (
        f"<video controls preload='metadata' muted playsinline>"
        f"<source src='{html.escape(resolved or '')}' type='video/mp4'>"
        "</video>"
    )


def text_html(text: str, empty_label: str = "(empty)") -> str:
    if not text:
        return f"<div class='text-body empty'>{html.escape(empty_label)}</div>"
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


def parse_summary_status(path_str: str | None) -> str:
    if not path_str:
        return "runtime_summary: Missing"
    path = Path(path_str)
    if not path.exists():
        return "runtime_summary: Missing"
    data = read_json(path)
    generated = data.get("num_generated")
    failed = data.get("num_failed")
    skipped = data.get("num_skipped")
    return f"runtime_summary: generated={generated}, failed={failed}, skipped={skipped}"


def load_records() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    rows = []
    if RESULT_INDEX.exists():
        rows = read_json(RESULT_INDEX).get("rows", [])

    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        seed = int(row.get("seed"))
        variant = str(row.get("variant") or "")
        by_seed.setdefault(seed, {})[variant] = row

    for seed in manifest.get("seeds", []):
        seed_i = int(seed)
        seed_tag = f"{seed_i:04d}"
        by_seed.setdefault(seed_i, {})
        for variant in ("caption", "nullcaption"):
            if variant in by_seed[seed_i]:
                continue
            output_dir = WORK_ROOT / "generated" / variant / f"seed_{seed_tag}"
            runtime_dir = WORK_ROOT / "runtime" / variant / f"seed_{seed_tag}"
            summary_paths = sorted((runtime_dir / "metadata").rglob("*_summary.json"))
            videos = sorted(output_dir.glob("*.mp4"))
            jsons = sorted(output_dir.glob("*.json"))
            by_seed[seed_i][variant] = {
                "seed": seed_i,
                "variant": variant,
                "output_dir": str(output_dir),
                "output_video": str(videos[0]) if videos else None,
                "output_json": str(jsons[0]) if jsons else None,
                "runtime_summary": str(summary_paths[0]) if summary_paths else None,
                "log_path": str(WORK_ROOT / "logs" / f"{variant}_seed_{seed_tag}.log"),
            }

    case_meta = read_json(Path(manifest["case_meta"]))
    case_paths = case_meta.get("paths", {})
    context_video = Path(case_paths["context_video_path"])
    gt_full_video = Path(case_paths["full_video_path"])
    caption_text = str(case_meta.get("caption") or "")

    case_assets = ASSETS_DIR / case_meta.get("sample_id", "case_0002")
    case_assets.mkdir(parents=True, exist_ok=True)

    records = []
    for seed in sorted(by_seed):
        cap = by_seed[seed].get("caption", {})
        nul = by_seed[seed].get("nullcaption", {})
        cap_video = Path(cap["output_video"]) if cap.get("output_video") else None
        nul_video = Path(nul["output_video"]) if nul.get("output_video") else None
        records.append(
            {
                "seed": seed,
                "context_video": expose_asset(context_video, case_assets, "context_video.mp4"),
                "gt_full_video": expose_asset(gt_full_video, case_assets, "gt_full_video.mp4"),
                "caption_video": expose_asset(cap_video, case_assets, f"seed_{seed:04d}_caption.mp4"),
                "null_video": expose_asset(nul_video, case_assets, f"seed_{seed:04d}_nullcaption.mp4"),
                "caption_status": parse_summary_status(cap.get("runtime_summary")),
                "null_status": parse_summary_status(nul.get("runtime_summary")),
                "caption_log": cap.get("log_path"),
                "null_log": nul.get("log_path"),
            }
        )

    return {
        "case_name": str(case_meta.get("sample_id") or ""),
        "caption": caption_text,
        "shared_context_frames": manifest.get("shared_context_frames"),
        "shared_full_video_frames": manifest.get("shared_full_video_frames"),
        "shared_fps": manifest.get("shared_fps"),
        "records": records,
    }


def build_html(data: dict[str, Any]) -> str:
    rows = []
    for record in data["records"]:
        status_text = (
            f"seed: {record['seed']}\n"
            f"{record['caption_status']}\n"
            f"{record['null_status']}\n"
            f"caption_log: {record['caption_log'] or 'Missing'}\n"
            f"null_log: {record['null_log'] or 'Missing'}"
        )
        caption_card = (
            render_slot("caption_input_used", text_html(data["caption"]), "meta-slot")
            if data["caption"]
            else ""
        )
        rows.append(
            "<article class='seed-card'>"
            "<div class='seed-head'>"
            f"<span class='badge'>seed {record['seed']}</span>"
            "</div>"
            "<div class='video-grid'>"
            f"{render_slot('context_video_from_meta', media_html(record['context_video']), 'video-slot')}"
            f"{render_slot('gt_full_video', media_html(record['gt_full_video']), 'video-slot gt-slot')}"
            f"{render_slot('generated_with_caption', media_html(record['caption_video']), 'video-slot caption-slot')}"
            f"{render_slot('generated_nullcaption', media_html(record['null_video']), 'video-slot null-slot')}"
            "</div>"
            "<div class='meta-grid'>"
            f"{caption_card}"
            f"{render_slot('run_status', text_html(status_text), 'meta-slot')}"
            "</div>"
            "</article>"
        )

    summary = (
        f"context_frames={data['shared_context_frames']} | "
        f"output_frames={data['shared_full_video_frames']} | "
        f"fps={data['shared_fps']}"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>0002 Baseline Multi-Seed</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1e1b16;
      --muted: #6e675d;
      --line: #d8cfbf;
      --caption-soft: #f3d7c9;
      --caption-ink: #6e2a13;
      --null-soft: #d6ead9;
      --null-ink: #28563c;
      --gt-soft: #e8dfd0;
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
      width: min(2200px, calc(100vw - 24px));
      margin: 0 auto;
      padding: 14px 0 24px;
    }}
    .hero {{
      margin-bottom: 12px;
      padding: 14px 18px;
      background: rgba(255,253,248,0.90);
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .hero h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .seed-list {{
      display: grid;
      gap: 14px;
    }}
    .seed-card {{
      padding: 10px;
      background: rgba(255,253,248,0.96);
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .seed-head {{
      display: flex;
      align-items: center;
      margin-bottom: 8px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(214, 234, 217, 0.82);
      color: var(--null-ink);
      font-size: 11px;
      font-weight: 700;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(340px, 1fr));
      gap: 8px;
      margin-bottom: 8px;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(240px, 1fr));
      gap: 8px;
    }}
    .slot {{
      background: #fffdf8;
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
    .gt-slot .slot-head {{ background: rgba(232, 223, 208, 0.82); }}
    .caption-slot .slot-head {{ background: rgba(243, 215, 201, 0.86); color: var(--caption-ink); }}
    .null-slot .slot-head {{ background: rgba(214, 234, 217, 0.86); color: var(--null-ink); }}
    .meta-slot .slot-head {{ background: rgba(255, 250, 242, 0.95); }}
    video {{
      width: 100%;
      aspect-ratio: 4 / 3;
      display: block;
      background: #000;
    }}
    .text-body {{
      padding: 8px 10px;
      font-family: "IBM Plex Mono", monospace;
      white-space: pre-wrap;
      line-height: 1.4;
      font-size: 11px;
    }}
    .empty {{ color: #8e877d; font-style: italic; }}
    .missing {{
      min-height: 150px;
      display: grid;
      place-items: center;
      color: #8e877d;
      background: rgba(242, 236, 226, 0.72);
      font-size: 12px;
    }}
    @media (max-width: 1700px) {{
      .video-grid {{ grid-template-columns: repeat(2, minmax(320px, 1fr)); }}
      .meta-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 980px) {{
      .shell {{ width: calc(100vw - 14px); }}
      .video-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <h1>0002 Baseline Multi-Seed (caption vs nullcaption)</h1>
      <p>case: {html.escape(data["case_name"])}
{html.escape(summary)}</p>
    </header>
    <section class="seed-list">
      {''.join(rows)}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    data = load_records()
    PORTAL_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    html_content = build_html(data)
    (PORTAL_DIR / "index.html").write_text(html_content, encoding="utf-8")
    write_json(PORTAL_DIR / "portal_data.json", data)
    print(str(PORTAL_DIR / "index.html"))


if __name__ == "__main__":
    main()
