from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a browser-friendly portal for batch metric results, grouped by context_mode."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more batch summary.json files produced by batch_compare_single_case_metrics.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where index.html and merged metadata are written.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relpath_from_portal_dir(target: Path, portal_dir: Path) -> str:
    return os.path.relpath(target.resolve(), start=portal_dir.resolve()).replace("\\", "/")


def choose_gt_video(record: dict) -> Path:
    return Path(str(record["pmf"]["gt_used_for_pmf"]))


def choose_pred_video(record: dict) -> Path:
    return Path(str(record["pmf"]["pred_used_for_pmf"]))


def build_card_html(
    *,
    title: str,
    video_rel: str,
    badge: str,
    subtitle: str,
    details: list[str],
    extra_links: list[tuple[str, str]],
) -> str:
    details_html = "".join(f"<div class='detail'>{html.escape(item)}</div>" for item in details)
    links_html = "".join(
        f"<a class='link' href='{html.escape(href)}' target='_blank' rel='noopener'>{html.escape(label)}</a>"
        for label, href in extra_links
    )
    return f"""
    <article class="card">
      <div class="badge">{html.escape(badge)}</div>
      <h3>{html.escape(title)}</h3>
      <div class="subtitle">{html.escape(subtitle)}</div>
      <video controls playsinline preload="metadata" src="{html.escape(video_rel)}"></video>
      <div class="details">{details_html}</div>
      <div class="links">{links_html}</div>
    </article>
    """


def build_row_html(
    *,
    context_mode: str,
    records: list[dict],
    portal_dir: Path,
) -> str:
    sorted_records = sorted(records, key=lambda item: str(item["relative_parent"]))
    gt_record = sorted_records[0]
    gt_path = choose_gt_video(gt_record)
    gt_rel = relpath_from_portal_dir(gt_path, portal_dir)
    gt_details = [
        f"reference={gt_record['reference_video'] if 'reference_video' in gt_record else gt_record['pmf']['reference_video']}",
        f"context_mode={context_mode}",
        f"context_frames={gt_record['context_frames']}",
    ]
    cards = [
        build_card_html(
            title="GT used for scoring",
            video_rel=gt_rel,
            badge="GT",
            subtitle=f"{context_mode}",
            details=gt_details,
            extra_links=[],
        )
    ]
    for record in sorted_records:
        pred_rel = relpath_from_portal_dir(choose_pred_video(record), portal_dir)
        compare_rel = relpath_from_portal_dir(Path(str(record["annotated_compare"]["compare_side_by_side"])), portal_dir)
        physics_compare_rel = relpath_from_portal_dir(Path(str(record["physics_iq"]["compare_side_by_side"])), portal_dir)
        pmf_compare_rel = relpath_from_portal_dir(Path(str(record["pmf"]["compare_side_by_side"])), portal_dir)
        physics_score = float(record["physics_iq"]["score"])
        pmf_score = float(record["pmf"]["score"])
        pred_frames = int(record["pmf"]["output_frames_after_context_clip"])
        cards.append(
            build_card_html(
                title=str(record["relative_parent"]),
                video_rel=pred_rel,
                badge="Output",
                subtitle=f"physics_iq={physics_score:.2f} | pmf={pmf_score:.6f}",
                details=[
                    f"pred_frames={pred_frames}",
                    f"compare_fps={float(record['pmf']['compare_fps']):.2f}",
                ],
                extra_links=[
                    ("Annotated compare", compare_rel),
                    ("Physics-IQ compare", physics_compare_rel),
                    ("PMF compare", pmf_compare_rel),
                ],
            )
        )
    cards_html = "".join(cards)
    return f"""
    <section class="mode-block">
      <div class="mode-head">
        <h2>{html.escape(context_mode)}</h2>
        <p>One GT card followed by all output videos scored under the same context mode.</p>
      </div>
      <div class="row-inner">
        {cards_html}
      </div>
    </section>
    """


def build_html(
    *,
    grouped_records: dict[str, list[dict]],
    portal_dir: Path,
) -> str:
    row_html = "".join(
        build_row_html(context_mode=mode, records=records, portal_dir=portal_dir)
        for mode, records in grouped_records.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Batch Metric Portal</title>
  <style>
    :root {{
      --bg: #f3f1ea;
      --card: #fffdf8;
      --ink: #201d17;
      --muted: #6b6255;
      --accent: #8f4b2b;
      --line: #ddd2c1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(143, 75, 43, 0.15), transparent 28%),
        linear-gradient(180deg, #f7f4ec 0%, var(--bg) 100%);
    }}
    main {{
      width: min(100vw - 32px, 1800px);
      margin: 24px auto 48px;
    }}
    .hero {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 20px 36px rgba(32, 29, 23, 0.08);
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-size: clamp(30px, 4vw, 54px);
      line-height: 0.95;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 16px;
    }}
    .mode-block {{
      margin-bottom: 24px;
      background: rgba(255, 253, 248, 0.86);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 16px 32px rgba(32, 29, 23, 0.06);
    }}
    .mode-head {{
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
    }}
    .mode-head p {{
      margin: 0;
      color: var(--muted);
    }}
    .row-inner {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: flex-start;
    }}
    .card {{
      width: 320px;
      flex: 0 0 320px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 12px 26px rgba(32, 29, 23, 0.06);
    }}
    .badge {{
      display: inline-block;
      margin-bottom: 10px;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(143, 75, 43, 0.12);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .subtitle {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }}
    video {{
      width: 100%;
      display: block;
      margin-top: 12px;
      border-radius: 14px;
      background: #000;
    }}
    .details {{
      margin-top: 12px;
      display: grid;
      gap: 6px;
    }}
    .detail {{
      color: var(--muted);
      font-size: 13px;
    }}
    .links {{
      margin-top: 12px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .link {{
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
    }}
    .link:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Batch Metric Portal</h1>
      <p>Rows are grouped by context mode. Each row shows one GT video and all output videos, with both Physics-IQ and PMF scores labeled on the output cards.</p>
    </section>
    {row_html}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = [load_payload(path.expanduser().resolve()) for path in args.summary_json]
    grouped_records: dict[str, list[dict]] = {}
    for payload in payloads:
        mode = str(payload["context_mode"])
        grouped_records.setdefault(mode, []).extend(payload["records"])

    ordered = {mode: grouped_records[mode] for mode in sorted(grouped_records)}
    merged = {"groups": ordered}
    (out_dir / "portal_data.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_text = build_html(grouped_records=ordered, portal_dir=out_dir)
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "modes": list(ordered.keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
