#!/usr/bin/env python3
"""Build a local HTML portal for two-case baseline/checkpoint sweeps."""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/two_case_checkpoint_sweep")
PORTAL_DIR = OUTPUT_ROOT / "visualization" / "checkpoint_sweep_portal"
ASSET_DIR = PORTAL_DIR / "assets" / "reference"
GENERATED_LINK_DIR = PORTAL_DIR / "generated_videos"
STEP_DIR_PATTERN = re.compile(r"step-(\d+)$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_portal_path(path: Path) -> str:
    return os.path.relpath(path, PORTAL_DIR).replace(os.sep, "/")


def ensure_symlink(target: Path, link_path: Path) -> Path | None:
    if not target.exists():
        return None
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return link_path
        link_path.unlink()
    link_path.symlink_to(target)
    return link_path


def video_stats(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"frames": 0, "width": 0, "height": 0, "fps": 0.0}
    stats = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return stats


def collect_models(generated_root: Path) -> list[str]:
    models = []
    for path in generated_root.iterdir():
        if path.is_dir():
            models.append(path.name)

    def sort_key(name: str) -> tuple[int, int, str]:
        if name == "base-ti2v-5b":
            return (0, -1, name)
        match = STEP_DIR_PATTERN.fullmatch(name)
        if match is not None:
            return (1, int(match.group(1)), name)
        return (2, 10**9, name)

    return sorted(models, key=sort_key)


def collect_cases(output_root: Path) -> list[dict[str, Any]]:
    generated_root = output_root / "generated_videos"
    models = collect_models(generated_root)
    case_map: dict[str, dict[str, Any]] = {}

    for model_name in models:
        model_dir = generated_root / model_name
        if not model_dir.is_dir():
            continue
        for sidecar_path in sorted(model_dir.glob("*.json")):
            try:
                payload = read_json(sidecar_path)
            except Exception:
                continue
            sample_id = str(payload.get("sample_id"))
            case_entry = case_map.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "dataset": payload.get("dataset", "unknown"),
                    "caption": payload.get("caption", ""),
                    "paths": payload.get("paths", {}),
                    "generation_params": payload.get("generation_params", {}),
                    "models": {},
                },
            )
            case_entry["models"][model_name] = payload
    cases = list(case_map.values())
    cases.sort(key=lambda item: (str(item["dataset"]).lower(), str(item["sample_id"]).lower()))
    return cases


def build_reference_assets(case: dict[str, Any]) -> dict[str, str]:
    paths = case.get("paths", {})
    case_dir = ASSET_DIR / str(case["sample_id"])
    linked: dict[str, str] = {}
    for source_key, asset_name in (
        ("context_video_path", "context_video.mp4"),
        ("future_gt_video_path", "future_gt_video.mp4"),
        ("full_video_path", "full_video.mp4"),
        ("first_frame_path", "first_frame.png"),
        ("meta_json_path", "meta.json"),
    ):
        raw = paths.get(source_key)
        if not isinstance(raw, str) or not raw:
            continue
        linked_path = ensure_symlink(Path(raw), case_dir / asset_name)
        if linked_path is not None:
            linked[source_key] = relative_portal_path(linked_path)
    return linked


def render_video_slot(title: str, web_path: str | None) -> str:
    if not web_path:
        return (
            "<div class='video-slot'>"
            f"<div class='slot-title'>{html.escape(title)}</div>"
            "<div class='missing'>missing</div>"
            "</div>"
        )
    return (
        "<div class='video-slot'>"
        f"<div class='slot-title'>{html.escape(title)}</div>"
        f"<video controls preload='metadata' muted playsinline src='{html.escape(web_path)}'></video>"
        "</div>"
    )


def render_case_section(case: dict[str, Any], model_names: list[str]) -> str:
    linked_assets = build_reference_assets(case)
    ref_stats = {}
    for key in ("context_video_path", "future_gt_video_path", "full_video_path"):
        raw = case.get("paths", {}).get(key)
        if isinstance(raw, str) and raw:
            ref_stats[key] = video_stats(Path(raw))

    stats_lines = []
    for label, key in (
        ("Context", "context_video_path"),
        ("Future GT", "future_gt_video_path"),
        ("Full Video", "full_video_path"),
    ):
        stats = ref_stats.get(key)
        if not stats:
            continue
        stats_lines.append(
            f"{label}: {int(stats['frames'])}f, {int(stats['height'])}x{int(stats['width'])}, {float(stats['fps']):.2f} fps"
        )

    reference_html = "".join(
        [
            render_video_slot("Context", linked_assets.get("context_video_path")),
            render_video_slot("Future GT", linked_assets.get("future_gt_video_path")),
            render_video_slot("Full Video", linked_assets.get("full_video_path")),
            (
                "<div class='video-slot'>"
                "<div class='slot-title'>First Frame</div>"
                + (
                    f"<img src='{html.escape(linked_assets['first_frame_path'])}' loading='lazy' alt='first frame' />"
                    if "first_frame_path" in linked_assets
                    else "<div class='missing'>missing</div>"
                )
                + "</div>"
            ),
        ]
    )

    generated_cards = []
    for model_name in model_names:
        model_payload = case.get("models", {}).get(model_name)
        if not model_payload:
            generated_cards.append(
                "<article class='generated-card missing-card'>"
                f"<h3>{html.escape(model_name)}</h3>"
                "<div class='missing'>missing</div>"
                "</article>"
            )
            continue
        output_path = Path(model_payload["paths"]["output_video_path"])
        portal_video_path = GENERATED_LINK_DIR / model_name / output_path.name
        web = relative_portal_path(portal_video_path) if output_path.is_file() else None
        video_meta = video_stats(output_path) if output_path.is_file() else {"frames": 0, "width": 0, "height": 0, "fps": 0.0}
        generated_cards.append(
            "<article class='generated-card'>"
            f"<h3>{html.escape(model_name)}</h3>"
            f"<div class='card-meta'>status: {html.escape(str(model_payload.get('status', 'unknown')))}</div>"
            f"<div class='card-meta'>output: {int(video_meta['frames'])}f, {int(video_meta['height'])}x{int(video_meta['width'])}, {float(video_meta['fps']):.2f} fps</div>"
            + (
                f"<video controls preload='metadata' muted playsinline src='{html.escape(web)}'></video>"
                if web
                else "<div class='missing'>missing video</div>"
            )
            + "</article>"
        )

    meta_link = linked_assets.get("meta_json_path")
    meta_html = (
        f"<a href='{html.escape(meta_link)}' target='_blank' rel='noopener'>meta.json</a>"
        if meta_link
        else "meta.json missing"
    )

    return (
        "<section class='case-section'>"
        f"<div class='case-head'><h2>{html.escape(str(case['sample_id']))}</h2>"
        f"<span class='dataset-tag'>{html.escape(str(case['dataset']))}</span></div>"
        f"<p class='caption'>{html.escape(str(case.get('caption', '')))}</p>"
        f"<p class='stats'>{html.escape(' | '.join(stats_lines))}</p>"
        f"<p class='meta-link'>{meta_html}</p>"
        "<div class='reference-grid'>"
        f"{reference_html}"
        "</div>"
        "<div class='generated-grid'>"
        f"{''.join(generated_cards)}"
        "</div>"
        "</section>"
    )


def build_html(output_root: Path) -> str:
    model_names = collect_models(output_root / "generated_videos")
    cases = collect_cases(output_root)
    sections = "".join(render_case_section(case, model_names) for case in cases)
    model_badges = "".join(f"<span class='model-badge'>{html.escape(name)}</span>" for name in model_names)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Two Case Checkpoint Sweep</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111418;
      --panel: #1b2128;
      --panel-2: #242d36;
      --text: #edf2f7;
      --muted: #a9b6c3;
      --accent: #78c0ff;
      --border: #34404d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #0f1318 0%, #161d24 100%);
      color: var(--text);
    }}
    main {{
      max-width: 1800px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero, .case-section {{
      background: rgba(27, 33, 40, 0.96);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
    }}
    h1, h2, h3, p {{ margin-top: 0; }}
    .model-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .model-badge, .dataset-tag {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      background: #213041;
      color: #d9ecff;
      font-size: 13px;
      border: 1px solid #37516d;
    }}
    .case-head {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .caption, .stats, .meta-link {{
      color: var(--muted);
      line-height: 1.55;
    }}
    .reference-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin: 18px 0 20px;
    }}
    .generated-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .video-slot, .generated-card {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
    }}
    .slot-title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 8px;
      color: #d8e2ec;
    }}
    .card-meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    video, img {{
      width: 100%;
      border-radius: 10px;
      background: #000;
      display: block;
    }}
    .missing {{
      min-height: 120px;
      display: grid;
      place-items: center;
      border: 1px dashed #5b6672;
      border-radius: 10px;
      color: #94a3b8;
      background: rgba(0, 0, 0, 0.18);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Two Case Checkpoint Sweep</h1>
      <p class="caption">Baseline plus all zero-padded checkpoints under the mixed 24-frame 384x672 LoRA run. Physics-IQ is capped to 24 generated future frames for quick comparison. The Genesis case keeps its 8-frame context, but because this single-case inference path requires output length to exceed context length, it is generated as 13 frames while still showing both Future GT and Full Video for reference.</p>
      <div class="model-badges">{model_badges}</div>
    </section>
    {sections}
  </main>
</body>
</html>
"""


def main() -> None:
    PORTAL_DIR.mkdir(parents=True, exist_ok=True)
    ensure_symlink(OUTPUT_ROOT / "generated_videos", GENERATED_LINK_DIR)
    html_text = build_html(OUTPUT_ROOT)
    index_path = PORTAL_DIR / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    summary = {
        "output_root": str(OUTPUT_ROOT),
        "portal_path": str(index_path),
        "model_names": collect_models(OUTPUT_ROOT / "generated_videos"),
        "num_cases": len(collect_cases(OUTPUT_ROOT)),
    }
    (PORTAL_DIR / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(index_path)


if __name__ == "__main__":
    main()
