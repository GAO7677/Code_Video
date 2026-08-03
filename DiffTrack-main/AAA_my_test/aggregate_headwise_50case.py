#!/usr/bin/env python3
"""Aggregate fixed-configuration per-head correspondence over 50 cases."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data/gaoya/agent-data/outputs/three_model_headwise_50case")
MODELS = {
    "gt": "GT teacher-forced",
    "stage1b": "Stage1b step-004000",
    "lora": "LoRA step-000500",
    "baseline": "Wan2.2 baseline",
}
METRICS = ("pck8", "pck16", "pck32", "mean_error_px")


def case_rows(path: Path) -> list[dict]:
    grouped = defaultdict(list)
    for row in json.loads(path.read_text(encoding="utf-8")):
        match = re.fullmatch(r"qk_head(\d+)", str(row.get("method", "")))
        if match and int(row.get("comparisons", 0)) > 0:
            scope = "objects" if row.get("region_type") == "object" else "background"
            grouped[(int(match.group(1)), scope)].append(row)
    output = []
    for (head, scope), rows in grouped.items():
        count = sum(int(row["comparisons"]) for row in rows)
        item = {"head": head, "scope": scope, "comparisons": count}
        for metric in METRICS:
            item[metric] = sum(float(row[metric]) * int(row["comparisons"]) for row in rows) / count
        output.append(item)
    return output


def aggregate_model(model: str) -> list[dict]:
    grouped = defaultdict(list)
    for path in sorted((ROOT / model / "cases").glob("case_*/metrics.json")):
        for row in case_rows(path):
            grouped[(row["head"], row["scope"])].append(row)
    output = []
    for (head, scope), rows in grouped.items():
        count = sum(int(row["comparisons"]) for row in rows)
        item = {"model": model, "head": head, "scope": scope, "cases": len(rows), "comparisons": count}
        for metric in METRICS:
            item[f"macro_{metric}"] = sum(float(row[metric]) for row in rows) / len(rows)
            item[f"pooled_{metric}"] = sum(float(row[metric]) * int(row["comparisons"]) for row in rows) / count
        output.append(item)
    return output


def main() -> None:
    rows = [row for model in MODELS for row in aggregate_model(model)]
    if not rows:
        raise RuntimeError(f"no headwise metrics found under {ROOT}")
    with (ROOT / "head_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["model"], row["scope"], row["head"])))
    best = {}
    for model in MODELS:
        candidates = [row for row in rows if row["model"] == model and row["scope"] == "objects"]
        best[model] = sorted(candidates, key=lambda row: (row["macro_pck32"], -row["macro_mean_error_px"]), reverse=True)[:5]
    (ROOT / "best_heads.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 50-case per-head Q@K validation", "",
        "Fixed global-best layer/step; no per-case head selection.", "",
        "| model | rank | head | cases | macro PCK@32 | pooled PCK@32 | macro error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, label in MODELS.items():
        for rank, row in enumerate(best[model], 1):
            lines.append(f"| {label} | {rank} | H{row['head']:02d} | {row['cases']} | {row['macro_pck32']:.2f}% | {row['pooled_pck32']:.2f}% | {row['macro_mean_error_px']:.2f}px |")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_rows = []
    for row in sorted(rows, key=lambda item: (item["model"], item["scope"], item["head"])):
        html_rows.append(f"<tr><td>{MODELS[row['model']]}</td><td>{row['scope']}</td><td>H{row['head']:02d}</td><td>{row['cases']}</td><td>{row['macro_pck8']:.2f}%</td><td>{row['macro_pck16']:.2f}%</td><td>{row['macro_pck32']:.2f}%</td><td>{row['macro_mean_error_px']:.2f}px</td></tr>")
    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Per-head validation</title><style>:root{{--paper:#ebe4d5;--ink:#17211d;--card:#fffdf7;--line:#bdb19b;--rust:#b7422c}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 5% 0,#db735044,transparent 32rem),var(--paper);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}}main{{width:min(1400px,calc(100% - 28px));margin:auto;padding:34px 0 70px}}h1{{font:700 clamp(46px,7vw,88px)/.9 Georgia;margin:0}}table{{width:100%;border-collapse:collapse;background:var(--card);margin-top:20px}}th,td{{border:1px solid var(--line);padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}thead{{position:sticky;top:0;background:var(--ink);color:white}}.note{{border-left:5px solid var(--rust);padding:12px;background:#fffdf7cc}}</style></head><body><main><h1>Head by Head</h1><p class="note">Fixed best layer/step, 50-case validation. Macro metrics weight every case equally; objects and background remain separate.</p><table><thead><tr><th>Model</th><th>Scope</th><th>Head</th><th>Cases</th><th>PCK@8</th><th>PCK@16</th><th>PCK@32</th><th>Error</th></tr></thead><tbody>{''.join(html_rows)}</tbody></table></main></body></html>'''
    (ROOT / "index.html").write_text(page, encoding="utf-8")
    print(f"aggregated {len(rows)} rows: {ROOT / 'index.html'}")


if __name__ == "__main__":
    main()
