#!/usr/bin/env python3
"""Build a local HTML viewer for the xSSC preprocessing ablation outputs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MODES = [
    ("center_crop", "center crop"),
    ("left_crop", "left crop"),
    ("right_crop", "right crop"),
    ("resize_pad_square", "resize + pad square"),
]


def rel(path: Path, root: Path) -> str:
    return html.escape(path.relative_to(root).as_posix())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def p95_from_hist(hist: np.ndarray, count: int) -> int | None:
    if count <= 0:
        return None
    return int(np.searchsorted(np.cumsum(hist), int(np.ceil(count * 0.95))))


def init_bucket() -> dict[str, Any]:
    return {"sum": 0, "count": 0, "hist": np.zeros(256, dtype=np.int64), "max": 0}


def update_bucket(bucket: dict[str, Any], diff: np.ndarray) -> None:
    bucket["sum"] += int(diff.sum())
    bucket["count"] += int(diff.size)
    bucket["hist"] += np.bincount(diff.reshape(-1), minlength=256)
    bucket["max"] = max(int(bucket["max"]), int(diff.max()))


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    count = int(bucket["count"])
    if count == 0:
        return {"mean_abs": None, "p95_abs": None, "max_abs": None}
    return {
        "mean_abs": float(bucket["sum"]) / count,
        "p95_abs": p95_from_hist(bucket["hist"], count),
        "max_abs": int(bucket["max"]),
    }


def compare_video_to_center(center_path: Path, mode_path: Path, ctx_frames: int = 8) -> dict[str, Any]:
    if not center_path.exists() or not mode_path.exists():
        return {"error": "missing video"}

    cap_a = cv2.VideoCapture(str(center_path))
    cap_b = cv2.VideoCapture(str(mode_path))
    if not cap_a.isOpened() or not cap_b.isOpened():
        return {"error": "failed to open video"}

    buckets = {"all": init_bucket(), "ctx": init_bucket(), "future": init_bucket()}
    frame_idx = 0
    size_a = None
    size_b = None
    mismatch = False

    while True:
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            mismatch = ok_a != ok_b
            break
        if size_a is None:
            size_a = [int(frame_a.shape[1]), int(frame_a.shape[0])]
            size_b = [int(frame_b.shape[1]), int(frame_b.shape[0])]
        if frame_a.shape != frame_b.shape:
            frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(frame_a, frame_b)
        update_bucket(buckets["all"], diff)
        update_bucket(buckets["ctx" if frame_idx < ctx_frames else "future"], diff)
        frame_idx += 1

    cap_a.release()
    cap_b.release()

    stats = {name: finalize_bucket(bucket) for name, bucket in buckets.items()}
    stats["frames_compared"] = frame_idx
    stats["center_size"] = size_a
    stats["mode_size"] = size_b
    stats["frame_count_mismatch"] = mismatch
    return stats


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build(root: Path, output_name: str) -> None:
    cases_path = root / "cases_4.txt"
    cases = [Path(line.strip()) for line in cases_path.read_text().splitlines() if line.strip()]
    mode_dirs = {mode: root / f"step001500_xssc_preprocess_{mode}" for mode, _ in MODES}

    metrics: dict[str, Any] = {
        "root": str(root),
        "modes": [mode for mode, _ in MODES],
        "cases": {},
    }

    for case_path in cases:
        stem = case_path.stem
        center_video = mode_dirs["center_crop"] / f"{stem}.mp4"
        case_metrics: dict[str, Any] = {}
        for mode, _ in MODES:
            mode_video = mode_dirs[mode] / f"{stem}.mp4"
            meta = read_json(mode_dirs[mode] / f"{stem}.json")
            case_metrics[mode] = {
                "video": str(mode_video),
                "json": str(mode_dirs[mode] / f"{stem}.json"),
                "input_ctx": str(mode_dirs[mode] / f"{stem}_input_ctx08.jpg"),
                "xssc_preprocess": meta.get("object_debug", {}).get("xssc_preprocess"),
                "diff_vs_center": (
                    {
                        "all": {"mean_abs": 0.0, "p95_abs": 0, "max_abs": 0},
                        "ctx": {"mean_abs": 0.0, "p95_abs": 0, "max_abs": 0},
                        "future": {"mean_abs": 0.0, "p95_abs": 0, "max_abs": 0},
                        "frames_compared": None,
                    }
                    if mode == "center_crop"
                    else compare_video_to_center(center_video, mode_video)
                ),
            }
        metrics["cases"][stem] = {
            "input_json": str(case_path),
            "modes": case_metrics,
        }

    (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (root / output_name).write_text(render_html(metrics, root), encoding="utf-8")


def render_diff(stats: dict[str, Any]) -> str:
    if "error" in stats:
        return html.escape(stats["error"])
    all_stats = stats.get("all", {})
    future_stats = stats.get("future", {})
    return (
        f"all mean {format_metric(all_stats.get('mean_abs'))} / p95 {format_metric(all_stats.get('p95_abs'))} / "
        f"max {format_metric(all_stats.get('max_abs'))}<br>"
        f"future mean {format_metric(future_stats.get('mean_abs'))} / p95 {format_metric(future_stats.get('p95_abs'))} / "
        f"max {format_metric(future_stats.get('max_abs'))}<br>"
        f"frames {format_metric(stats.get('frames_compared'), 0)}"
    )


def render_debug(debug: Any) -> str:
    if not isinstance(debug, dict):
        return "xSSC preprocess debug missing"
    parts = [f"mode={debug.get('mode')}"]
    if "crop_yxhw" in debug:
        parts.append(f"crop_yxhw={debug['crop_yxhw']}")
    if "resize_hw" in debug:
        parts.append(f"resize_hw={debug['resize_hw']}")
    if "pad_top_left" in debug:
        parts.append(f"pad_top_left={debug['pad_top_left']}")
    return html.escape(" · ".join(parts))


def render_html(metrics: dict[str, Any], root: Path) -> str:
    rows = []
    for stem, case in metrics["cases"].items():
        source_json = html.escape(case["input_json"])
        modes = case["modes"]
        input_ctx = Path(modes["center_crop"]["input_ctx"])
        cells = [
            f"""
            <article class="input">
              <h3>input ctx08</h3>
              <img src="{rel(input_ctx, root)}" alt="input context frames">
              <div class="meta"><code>{source_json}</code></div>
            </article>
            """
        ]
        for mode, label in MODES:
            item = modes[mode]
            video = Path(item["video"])
            cells.append(
                f"""
                <article>
                  <h3>{html.escape(label)}</h3>
                  <video src="{rel(video, root)}" controls loop muted playsinline></video>
                  <div class="meta">{render_diff(item["diff_vs_center"])}</div>
                  <div class="meta small">{render_debug(item["xssc_preprocess"])}</div>
                </article>
                """
            )
        rows.append(
            f"""
            <section>
              <h2>{html.escape(stem)}</h2>
              <div class="case-grid">
                {''.join(cells)}
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC preprocessing ablation · step1500</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1115;
      color: #edf1f7;
    }}
    body {{
      margin: 0;
      background: #0f1115;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 14px 22px;
      background: rgba(15, 17, 21, 0.95);
      border-bottom: 1px solid #2a303a;
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 680;
      letter-spacing: 0;
    }}
    main {{
      padding: 18px 22px 36px;
      display: grid;
      gap: 26px;
    }}
    section {{
      display: grid;
      gap: 12px;
      min-width: 0;
    }}
    h2 {{
      margin: 0;
      font-size: 15px;
      font-weight: 660;
      overflow-wrap: anywhere;
    }}
    .summary {{
      color: #c1c8d4;
      font-size: 13px;
      line-height: 1.5;
      max-width: 1180px;
    }}
    .case-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(240px, 1fr));
      gap: 12px;
      align-items: start;
    }}
    article {{
      min-width: 0;
      overflow: hidden;
      border: 1px solid #2a303a;
      border-radius: 8px;
      background: #171b22;
    }}
    article h3 {{
      margin: 0;
      padding: 9px 10px;
      font-size: 13px;
      font-weight: 650;
      background: #202632;
      border-bottom: 1px solid #2a303a;
    }}
    video, img {{
      display: block;
      width: 100%;
      height: auto;
      background: #050608;
    }}
    .meta {{
      padding: 8px 10px;
      color: #c5ccd8;
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
      border-top: 1px solid #2a303a;
    }}
    .small {{
      color: #9da7b7;
      font-size: 11px;
    }}
    code {{
      color: #dce7ff;
      overflow-wrap: anywhere;
    }}
    a {{
      color: #8db4ff;
    }}
    @media (max-width: 1500px) {{
      .case-grid {{
        grid-template-columns: repeat(3, minmax(250px, 1fr));
      }}
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 14px;
      }}
      .case-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>xSSC preprocessing ablation · step1500 · 4 cases × 4 modes</h1>
  </header>
  <main>
    <div class="summary">
      Only the frozen xSSC input preprocessing is changed. Wan context/video preprocessing remains 512×896 cover-crop.
      The three non-baseline videos are compared pixel-wise against the center-crop baseline; metrics are absolute RGB differences on aligned decoded frames.
      Raw metrics: <a href="metrics.json">metrics.json</a>.
    </div>
    {''.join(rows)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/AAA_physv/AAA_xSSC/xssc_preprocess_ablation_step1500"),
    )
    parser.add_argument("--output-name", default="index.html")
    args = parser.parse_args()
    build(args.root, args.output_name)
    print(args.root / args.output_name)
    print(args.root / "metrics.json")


if __name__ == "__main__":
    main()
