#!/usr/bin/env python3
"""Build a paired Pixel/V-JEPA reconstruction-error gallery and report."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def finite_mean(samples: list[dict], key: str) -> float:
    values = [
        sample["metrics"][key]
        for sample in samples
        if key in sample["metrics"] and math.isfinite(sample["metrics"][key])
    ]
    return statistics.fmean(values) if values else float("nan")


def fmt(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.6f}"


def ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    return numerator / max(denominator, 1e-12)


def main() -> None:
    args = parse_args()
    pixel = json.loads((args.root / "pixel" / "pixel_summary.json").read_text())
    vjepa = json.loads((args.root / "vjepa" / "vjepa_summary.json").read_text())
    pixel_by_key = {(s["kind"], s["sample_id"]): s for s in pixel["samples"]}
    vjepa_by_key = {(s["kind"], s["sample_id"]): s for s in vjepa["samples"]}
    if pixel_by_key.keys() != vjepa_by_key.keys():
        raise RuntimeError("Pixel and V-JEPA sample sets differ")

    cards = []
    for key in sorted(pixel_by_key):
        p = pixel_by_key[key]
        v = vjepa_by_key[key]
        rows = []
        for name in sorted(set(p["metrics"]) | set(v["metrics"])):
            rows.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td>{fmt(p['metrics'].get(name, float('nan')))}</td>"
                f"<td>{fmt(v['metrics'].get(name, float('nan')))}</td></tr>"
            )
        cards.append(
            f"<article><h2>{html.escape(key[0])} / {html.escape(key[1])}</h2>"
            f"<p><code>{html.escape(p['source_video'])}</code></p>"
            f"<p>Pixel input: <code>{p['model_input_shape']}</code>; "
            f"V-JEPA input: <code>{v['model_input_shape']}</code></p>"
            "<div class='videos'>"
            f"<section><h3>Pixel-space SAVi, step {pixel['checkpoint_step']}</h3>"
            f"<video controls loop muted preload='metadata' src='pixel/{html.escape(p['video'])}'></video></section>"
            f"<section><h3>V-JEPA-space SAVi, step {vjepa['checkpoint_step']}</h3>"
            f"<video controls loop muted preload='metadata' src='vjepa/{html.escape(v['video'])}'></video></section>"
            "</div><table><thead><tr><th>Metric</th><th>Pixel</th><th>V-JEPA</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></article>"
        )

    kubric_pixel = [s for s in pixel["samples"] if s["kind"] == "kubric_val"]
    kubric_vjepa = [s for s in vjepa["samples"] if s["kind"] == "kubric_val"]
    physiq_pixel = [s for s in pixel["samples"] if s["kind"] == "physiq"]
    physiq_vjepa = [s for s in vjepa["samples"] if s["kind"] == "physiq"]
    aggregate = {
        "pixel": {
            "kubric_global": finite_mean(kubric_pixel, "global_loss"),
            "kubric_dynamic": finite_mean(kubric_pixel, "dynamic_loss"),
            "kubric_static": finite_mean(kubric_pixel, "static_geometry_loss"),
            "kubric_background": finite_mean(kubric_pixel, "background_loss"),
            "physiq_global": finite_mean(physiq_pixel, "global_loss"),
            "physiq_motion_proxy": finite_mean(physiq_pixel, "motion_proxy_loss"),
            "physiq_non_motion_proxy": finite_mean(
                physiq_pixel, "non_motion_proxy_loss"
            ),
        },
        "vjepa": {
            "kubric_global": finite_mean(kubric_vjepa, "global_loss"),
            "kubric_dynamic": finite_mean(kubric_vjepa, "dynamic_loss"),
            "kubric_static": finite_mean(kubric_vjepa, "static_geometry_loss"),
            "kubric_background": finite_mean(kubric_vjepa, "background_loss"),
            "physiq_global": finite_mean(physiq_vjepa, "global_loss"),
            "physiq_motion_proxy": finite_mean(physiq_vjepa, "motion_proxy_loss"),
            "physiq_non_motion_proxy": finite_mean(
                physiq_vjepa, "non_motion_proxy_loss"
            ),
        },
    }
    for metrics in aggregate.values():
        metrics["kubric_dynamic_to_background"] = ratio(
            metrics["kubric_dynamic"], metrics["kubric_background"]
        )
        metrics["kubric_dynamic_to_static"] = ratio(
            metrics["kubric_dynamic"], metrics["kubric_static"]
        )
        metrics["physiq_motion_to_non_motion"] = ratio(
            metrics["physiq_motion_proxy"], metrics["physiq_non_motion_proxy"]
        )
    (args.root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    report = f"""# Pixel/V-JEPA SAVi reconstruction-error analysis

## Evaluation setup

- Pixel checkpoint: `{pixel['checkpoint']}`
- V-JEPA checkpoint: `{vjepa['checkpoint']}`
- Validation indices, fixed seed {pixel['validation_seed']}: `{pixel['validation_indices']}`
- Pixel input: {pixel['model_input_policy']}
- V-JEPA input: {vjepa['model_input_policy']}
- Pixel loss: RGB channel-mean MSE against GT RGB.
- V-JEPA loss: 1408-D feature-mean MSE against frozen V-JEPA target features; this is not RGB reconstruction.
- Kubric regions use GT segmentation. PhysicIQ has no segmentation GT and uses top-20% temporal RGB-difference as a motion proxy.

## Aggregate metrics

| Space | Kubric global | dynamic | static geometry | background | dynamic/background | dynamic/static | PhysicIQ global | motion proxy | non-motion proxy | motion/non-motion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pixel | {fmt(aggregate['pixel']['kubric_global'])} | {fmt(aggregate['pixel']['kubric_dynamic'])} | {fmt(aggregate['pixel']['kubric_static'])} | {fmt(aggregate['pixel']['kubric_background'])} | {fmt(aggregate['pixel']['kubric_dynamic_to_background'])} | {fmt(aggregate['pixel']['kubric_dynamic_to_static'])} | {fmt(aggregate['pixel']['physiq_global'])} | {fmt(aggregate['pixel']['physiq_motion_proxy'])} | {fmt(aggregate['pixel']['physiq_non_motion_proxy'])} | {fmt(aggregate['pixel']['physiq_motion_to_non_motion'])} |
| V-JEPA | {fmt(aggregate['vjepa']['kubric_global'])} | {fmt(aggregate['vjepa']['kubric_dynamic'])} | {fmt(aggregate['vjepa']['kubric_static'])} | {fmt(aggregate['vjepa']['kubric_background'])} | {fmt(aggregate['vjepa']['kubric_dynamic_to_background'])} | {fmt(aggregate['vjepa']['kubric_dynamic_to_static'])} | {fmt(aggregate['vjepa']['physiq_global'])} | {fmt(aggregate['vjepa']['physiq_motion_proxy'])} | {fmt(aggregate['vjepa']['physiq_non_motion_proxy'])} | {fmt(aggregate['vjepa']['physiq_motion_to_non_motion'])} |

The numeric scales between Pixel and V-JEPA are not directly comparable because they measure different target spaces. Compare region ratios within each row.
"""
    (args.root / "analysis_report.md").write_text(report, encoding="utf-8")
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>SAVi reconstruction error</title>
<style>:root{{--bg:#f1f3ef;--ink:#18221c;--line:#b8c0b9;--paper:#fff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Verdana,sans-serif}}header,article{{padding:20px 24px}}header{{border-bottom:1px solid var(--line)}}article{{background:var(--paper);margin:20px;border:1px solid var(--line);border-radius:6px}}code{{overflow-wrap:anywhere;font-size:12px}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}video{{display:block;width:100%;background:#111}}h2,h3{{letter-spacing:0}}h3{{font-size:15px}}table{{border-collapse:collapse;margin-top:16px;width:100%;font-size:12px}}th,td{{border:1px solid var(--line);padding:6px;text-align:right}}th:first-child,td:first-child{{text-align:left}}@media(max-width:900px){{.videos{{grid-template-columns:1fr}}article{{margin:8px;padding:12px}}}}</style></head><body>
<header><h1>Pixel/V-JEPA SAVi reconstruction-error comparison</h1><p>Same 10 Kubric validation samples plus 4 PhysicIQ cases. Heatmap colors share one p99 scale within each model and metric, not across feature spaces.</p><p><a href='analysis_report.md'>Analysis report</a> | <a href='aggregate_metrics.json'>Aggregate JSON</a></p></header>{''.join(cards)}</body></html>"""
    (args.root / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps({"root": str(args.root), "samples": len(cards)}, indent=2))


if __name__ == "__main__":
    main()
