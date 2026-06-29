from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an HTML report for batch single-case Physics-IQ results."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Summary JSON produced by run_single_case_physics_iq_batch.py",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Directory where the static HTML report and assets will be written.",
    )
    return parser.parse_args()


def _safe_name(path: Path) -> str:
    text = str(path).replace("/", "__")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _link_asset(report_dir: Path, source: Path, name: str) -> str:
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ext = source.suffix or ".bin"
    target = assets_dir / f"{name}{ext}"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)
    return str(target.relative_to(report_dir))


def _metric_table(item: dict[str, Any]) -> str:
    rows = [
        ("score", f"{item.get('score'):.2f}" if item.get("score") is not None else "-"),
        ("mse_mean", f"{item.get('mse_mean'):.6f}" if item.get("mse_mean") is not None else "-"),
        (
            "spatiotemporal_iou_mean",
            f"{item.get('spatiotemporal_iou_mean'):.6f}" if item.get("spatiotemporal_iou_mean") is not None else "-",
        ),
        ("spatial_iou", f"{item.get('spatial_iou'):.6f}" if item.get("spatial_iou") is not None else "-"),
        (
            "weighted_spatial_iou",
            f"{item.get('weighted_spatial_iou'):.6f}" if item.get("weighted_spatial_iou") is not None else "-",
        ),
        ("num_frames_compared", str(item.get("num_frames_compared"))),
        ("target_size", "x".join(str(v) for v in item.get("target_size", []))),
        ("video_codec", str(item.get("video_codec"))),
    ]
    html_rows = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>" for key, value in rows
    )
    return f"<table>{html_rows}</table>"


def _build_card(report_dir: Path, item: dict[str, Any], gt_original_rel: str) -> str:
    candidate_path = Path(item["candidate_video"])
    scored_candidate_path = Path(item["scored_output_video"])
    scored_gt_path = Path(item["scored_source_video"])

    candidate_rel = _link_asset(report_dir, candidate_path, f"{item['case_name']}__candidate")
    scored_candidate_rel = _link_asset(report_dir, scored_candidate_path, f"{item['case_name']}__scored_candidate")
    scored_gt_rel = _link_asset(report_dir, scored_gt_path, f"{item['case_name']}__scored_gt")

    title = html.escape(candidate_path.parent.name if candidate_path.parent.name else candidate_path.stem)
    path_text = html.escape(str(candidate_path))
    score = item.get("score")
    score_text = "-" if score is None else f"{score:.2f}"
    return f"""
    <section class="card">
      <div class="head">
        <div>
          <h2>{title}</h2>
          <div class="path">{path_text}</div>
        </div>
        <div class="score">{score_text}</div>
      </div>
      <div class="metrics">{_metric_table(item)}</div>
      <div class="videos">
        <div class="video-block">
          <h3>Original Candidate</h3>
          <video controls preload="metadata" src="{candidate_rel}"></video>
        </div>
        <div class="video-block">
          <h3>Scored Candidate</h3>
          <video controls preload="metadata" src="{scored_candidate_rel}"></video>
        </div>
        <div class="video-block">
          <h3>Scored GT</h3>
          <video controls preload="metadata" src="{scored_gt_rel}"></video>
        </div>
        <div class="video-block">
          <h3>Original GT</h3>
          <video controls preload="metadata" src="{gt_original_rel}"></video>
        </div>
      </div>
    </section>
    """


def build_html(summary: dict[str, Any], report_dir: Path) -> str:
    gt_original_rel = _link_asset(report_dir, Path(summary["gt_video"]), "gt_original")
    cards = "\n".join(_build_card(report_dir, item, gt_original_rel) for item in summary["results"])
    failure_html = ""
    if summary.get("failures"):
        rows = "\n".join(
            f"<tr><td>{html.escape(item['video'])}</td><td>{html.escape(item['error'])}</td></tr>"
            for item in summary["failures"]
        )
        failure_html = f"""
        <section class="failures">
          <h2>Failures</h2>
          <table>
            <tr><th>video</th><th>error</th></tr>
            {rows}
          </table>
        </section>
        """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Single-Case Physics-IQ Report</title>
  <style>
    :root {{
      --bg: #f2efe8;
      --paper: #fffdf8;
      --ink: #1d1b18;
      --muted: #645d55;
      --line: #d8d0c5;
      --accent: #0f766e;
      --accent-soft: #dff4f1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Source Serif 4", "Noto Serif", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 30%),
        linear-gradient(180deg, #f7f4ee 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1600px, calc(100vw - 40px));
      margin: 24px auto 64px;
    }}
    .hero {{
      background: linear-gradient(140deg, rgba(255,255,255,0.92), rgba(249,245,237,0.88));
      border: 1px solid rgba(216,208,197,0.8);
      border-radius: 24px;
      padding: 28px 32px;
      box-shadow: 0 18px 50px rgba(74, 63, 53, 0.08);
      backdrop-filter: blur(8px);
    }}
    h1, h2, h3 {{ margin: 0; }}
    .hero p {{ color: var(--muted); max-width: 1100px; }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .stat {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
    }}
    .stat .k {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat .v {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .cards {{
      display: grid;
      gap: 18px;
      margin-top: 24px;
    }}
    .card {{
      background: rgba(255,253,248,0.92);
      border: 1px solid rgba(216,208,197,0.9);
      border-radius: 24px;
      padding: 22px;
      box-shadow: 0 18px 40px rgba(81, 68, 56, 0.07);
    }}
    .head {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 20px;
      margin-bottom: 18px;
    }}
    .path {{ color: var(--muted); font-size: 13px; margin-top: 8px; word-break: break-all; }}
    .score {{
      min-width: 108px;
      text-align: center;
      padding: 14px 18px;
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 18px;
      font-size: 30px;
      font-weight: 700;
    }}
    .metrics table, .failures table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 18px;
      font-family: "IBM Plex Mono", "Noto Sans Mono", monospace;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .videos {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }}
    .video-block {{
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    video {{
      width: 100%;
      border-radius: 12px;
      background: #000;
      display: block;
    }}
    .failures {{
      margin-top: 24px;
      background: rgba(255,250,250,0.92);
      border: 1px solid #e8c5c5;
      border-radius: 20px;
      padding: 20px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Single-Case Physics-IQ Approx Report</h1>
      <p>
        GT is fixed to <code>{html.escape(summary['gt_video'])}</code>. All candidate videos are rescored against the same
        GT using the aligned/resized single-view approximate Physics-IQ implementation. Each card shows the original
        candidate, the exact H.264-aligned candidate used for scoring, the exact H.264-aligned GT used for scoring,
        and the original GT video.
      </p>
      <div class="hero-grid">
        <div class="stat"><div class="k">Candidates</div><div class="v">{summary['num_candidates']}</div></div>
        <div class="stat"><div class="k">Scored</div><div class="v">{summary['num_scored']}</div></div>
        <div class="stat"><div class="k">Failed</div><div class="v">{summary['num_failed']}</div></div>
        <div class="stat"><div class="k">Downsample</div><div class="v">{summary['downsample_factor']}</div></div>
        <div class="stat"><div class="k">Threshold</div><div class="v">{summary['threshold_value']}</div></div>
      </div>
    </section>
    <section class="cards">
      {cards}
    </section>
    {failure_html}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    report_dir = args.report_dir.resolve()
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_text = build_html(summary, report_dir)
    index_path = report_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    print(index_path)


if __name__ == "__main__":
    main()
