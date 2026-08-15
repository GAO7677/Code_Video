#!/usr/bin/env python3
"""Build a cross-noise visibility/loss report from completed trajectory runs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


TIMESTEPS = (100, 300, 500, 700, 900)
CASE_ORDER = ("F1", "F2", "F3")


def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/"
            "object_cotracker_trajectory_noise_sweep_compare/runs"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/"
            "object_cotracker_trajectory_noise_sweep_compare"
        ),
    )
    return parser.parse_args()


def fmean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def load_sweep(runs_root: Path) -> list[dict]:
    sweep = []
    for timestep in TIMESTEPS:
        run_root = runs_root / f"t{timestep:04d}"
        case_rows = []
        for metrics_path in sorted(run_root.glob("cases/*/*/metrics.json")):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["case_key"] = str(metrics_path.parent.relative_to(run_root / "cases")).split("/", 1)[0]
            # The nested case key is the path below cases; keep the complete key
            # for links while exposing the family in the card heading.
            metrics["case_key"] = str(metrics_path.parent.relative_to(run_root / "cases"))
            case_root = metrics_path.parent
            forward = json.loads(
                (case_root / "forward_metrics.json").read_text(encoding="utf-8")
            )
            metrics["training_weighted_loss"] = float(forward["weighted_loss"])
            metrics["scheduler_sigma"] = float(forward["scheduler_sigma"])
            metrics["relative_case_root"] = str(case_root.relative_to(runs_root.parent))
            case_rows.append(metrics)
        if len(case_rows) != 3:
            raise RuntimeError(f"expected 3 cases for t={timestep}, found {len(case_rows)}")
        sweep.append(
            {
                "timestep": timestep,
                "sigma": float(case_rows[0]["scheduler_sigma"]),
                "training_weighted_loss": fmean(case_rows, "training_weighted_loss"),
                "cases": case_rows,
                "old_loss": fmean(case_rows, "old_all_point_trajectory_loss"),
                "coordinate_loss": fmean(case_rows, "visibility_aware_coordinate_loss"),
                "weighted_visibility_loss": fmean(
                    case_rows, "weighted_visibility_preservation_loss"
                ),
                "new_total_loss": fmean(case_rows, "visibility_aware_total_loss"),
                "gt_mask_visible": fmean(case_rows, "gt_visible_fraction"),
                "gt_cotracker_visible": fmean(
                    case_rows, "gt_cotracker_visible_fraction"
                ),
                "pred_visibility": fmean(case_rows, "mean_pred_visibility_probability"),
            }
        )
    return sweep


def case_card(row: dict, run_root_name: str) -> str:
    case_root = Path(row["relative_case_root"])
    overlay = f"{case_root.as_posix()}/object_trajectory_overlay.mp4"
    poster = f"{case_root.as_posix()}/trajectory_future_preview.jpg"
    object_rows = "".join(
        f"<tr><td>O{o['object_index'] + 1} · {html.escape(o['phrase'])}</td>"
        f"<td>{o['gt_visible_fraction']:.1%}</td>"
        f"<td>{o['gt_cotracker_visible_fraction']:.1%}</td>"
        f"<td>{o['mean_pred_visibility_probability']:.1%}</td>"
        f"<td>{o['trajectory_loss']:.6f}</td>"
        f"<td>{o['visibility_aware_coordinate_loss']:.6f}</td>"
        f"<td>{o['visibility_aware_total_loss']:.6f}</td></tr>"
        for o in row["objects"]
    )
    return f"""
<article class="case-card">
  <div class="case-card-head"><h3>{html.escape(row['case_key'])}</h3>
  <span>{html.escape(row['caption'])}</span></div>
  <div class="case-metrics"><b>mask-visible {row['gt_visible_fraction']:.1%}</b>
  <b>CoTracker &gt;0.9 {row['gt_cotracker_visible_fraction']:.1%}</b>
  <b>pred visibility {row['mean_pred_visibility_probability']:.1%}</b>
  <b>old {row['old_all_point_trajectory_loss']:.6f}</b>
  <b>new {row['visibility_aware_total_loss']:.6f}</b></div>
  <video controls muted loop playsinline preload="metadata" poster="{poster}" src="{overlay}"></video>
  <details><summary>Per-object comparison</summary>
    <table><thead><tr><th>object</th><th>mask-visible</th><th>CoTracker&gt;.9</th><th>pred vis</th><th>old</th><th>new coord</th><th>new total</th></tr></thead>
    <tbody>{object_rows}</tbody></table>
  </details>
</article>"""


def render(sweep: list[dict], output_root: Path) -> Path:
    rows = []
    for item in sweep:
        case_cards = "".join(case_card(row, f"t{item['timestep']:04d}") for row in item["cases"])
        rows.append(
            f"""
<section class="noise-section"><header><div><span class="eyebrow">NOISE STAGE</span>
<h2>t={item['timestep']} · sigma={item['sigma']:.4f}</h2>
<p>Training weighted flow loss: <b>{item['training_weighted_loss']:.6f}</b></p></div>
<div class="noise-summary"><span><b>{item['gt_mask_visible']:.1%}</b>GT mask-visible</span>
<span><b>{item['gt_cotracker_visible']:.1%}</b>GT CoTracker&gt;.9</span>
<span><b>{item['pred_visibility']:.1%}</b>pred visibility</span>
<span><b>{item['old_loss']:.6f}</b>old all-point</span>
<span><b>{item['coordinate_loss']:.6f}</b>new coord</span>
<span><b>{item['weighted_visibility_loss']:.6f}</b>0.05 × vis</span>
<span><b>{item['new_total_loss']:.6f}</b>new total</span></div></header>
<div class="case-grid">{case_cards}</div></section>"""
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Noise sweep · visibility and trajectory loss</title><style>
:root{{--bg:#edf2f1;--ink:#132127;--muted:#607076;--line:#c7d1d2;--paper:#fff;--green:#16745a;--old:#ad423b;--blue:#26648b;--gold:#a46f16}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 "IBM Plex Sans","Noto Sans SC",sans-serif}}
header.mast{{background:#172329;color:#f6fafb;border-bottom:5px solid var(--green);padding:28px max(28px,calc((100vw - 1500px)/2)) 24px}}
h1{{font:700 34px/1.1 "IBM Plex Sans Condensed","Noto Sans SC",sans-serif;margin:0 0 8px}}header p{{margin:0;color:#b9c7cb}}main{{max-width:1500px;margin:auto;padding:22px 28px 80px}}
.noise-section{{padding:22px 0 36px;border-bottom:1px solid var(--line)}}.noise-section>header{{display:flex;justify-content:space-between;gap:28px;align-items:end;margin-bottom:14px}}.eyebrow{{color:var(--green);font-size:11px;font-weight:800}}h2{{margin:2px 0;font-size:25px}}.noise-section header p{{color:var(--muted)}}.noise-summary{{display:grid;grid-template-columns:repeat(7,minmax(100px,1fr));background:var(--paper);border:1px solid var(--line)}}.noise-summary span{{padding:9px 10px;border-right:1px solid var(--line);color:var(--muted);font-size:12px}}.noise-summary span:last-child{{border:0}}.noise-summary b{{display:block;color:var(--ink);font-size:16px}}
.case-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.case-card{{background:var(--paper);border-top:4px solid var(--blue);padding:12px}}.case-card:nth-child(2){{border-color:var(--green)}}.case-card:nth-child(3){{border-color:var(--gold)}}.case-card-head h3{{margin:0 0 3px;font-size:17px}}.case-card-head span{{display:block;color:var(--muted);font-size:12px;min-height:36px}}.case-metrics{{display:grid;grid-template-columns:repeat(2,1fr);gap:5px;margin:10px 0;font-size:12px}}.case-metrics b{{padding:6px;background:#f0f4f3;font-weight:600}}video{{display:block;width:100%;aspect-ratio:21/4;object-fit:contain;background:#050708;border:1px solid #29383f}}details{{margin-top:10px}}summary{{cursor:pointer;color:var(--blue);font-weight:700}}table{{width:100%;border-collapse:collapse;margin-top:8px;font-size:11px}}th,td{{padding:5px 4px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);font-weight:600}}
@media(max-width:1150px){{.noise-summary{{grid-template-columns:repeat(4,1fr)}}.case-grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:760px){{.noise-section>header{{display:block}}.noise-summary{{margin-top:12px;grid-template-columns:repeat(2,1fr)}}.case-grid{{grid-template-columns:1fr}}}}
</style></head><body><header class="mast"><h1>Noise Sweep · Visibility and Trajectory Loss</h1><p>同一批 PyBullet F1/F2/F3、24 queries/object；只改变训练输入噪声 t，x0_pred 来自既有 Tiny-VAE sweep。mask-visible 是物理 object-mask gate，CoTracker visibility 仅作点身份可靠性诊断。新版：Lnew = Lcoord + 0.05 Lvis。</p></header><main>{''.join(rows)}</main></body></html>"""
    output_root.mkdir(parents=True, exist_ok=True)
    index = output_root / "index.html"
    index.write_text(document, encoding="utf-8")
    summary = {
        "state": "complete",
        "timesteps": [item["timestep"] for item in sweep],
        "cases_per_timestep": 3,
        "rows": sweep,
        "index": str(index),
    }
    (output_root / "noise_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return index


if __name__ == "__main__":
    args = load_args()
    print(render(load_sweep(args.runs_root), args.output_root))
