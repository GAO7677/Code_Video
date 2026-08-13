#!/usr/bin/env python3
"""Build a lightweight overview page from completed Stage-1 JSON outputs."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--case-gallery", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def cell(value):
    if isinstance(value, float):
        return f"{value:.5f}"
    return html.escape(str(value))


def table(headers, rows):
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(row.get(header, ''))}</td>" for header in headers) + "</tr>"
        for row in rows
    )
    return f"<div class=scroll><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = []
    audit = None
    causality = None
    for json_file in sorted(args.results_root.resolve().rglob("*.json")):
        try:
            payload = json.loads(json_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if payload.get("format") == "xssc_stage1_evaluation_v1":
            for horizon, metrics in payload["metrics"].items():
                evaluations.append(
                    {
                        "representation": payload["representation"],
                        "H": payload["history"],
                        "context": payload["context"],
                        "seed": payload["seed"],
                        "horizon": horizon,
                        "velocity_nrmse": metrics.get("velocity_nrmse"),
                        "center_ade": metrics.get("center_ade"),
                        "center_fde": metrics.get("center_fde"),
                        "bbox_iou": metrics.get("bbox_iou"),
                        "presence_f1": metrics.get("presence_f1"),
                    }
                )
        elif payload.get("format") == "xssc_stage1_representation_audit_v1":
            audit = payload
        elif "records" in payload and "passed" in payload and "atol" in payload:
            causality = payload

    sections = []
    if causality is not None:
        sections.append(
            f"<section><h2>Causality gate</h2><p class={'pass' if causality['passed'] else 'fail'}>"
            f"{'PASS' if causality['passed'] else 'FAIL'} · atol={causality['atol']}</p></section>"
        )
    if audit is not None:
        rows = [
            {"metric": key, **value} for key, value in audit["metrics"].items()
        ]
        sections.append(
            "<section><h2>Temporal stability</h2>"
            + table(["metric", "mean", "std_between_videos"], rows)
            + "</section>"
        )
    if evaluations:
        sections.append(
            "<section><h2>State sufficiency and object context</h2>"
            + table(
                ["representation", "H", "context", "seed", "horizon", "velocity_nrmse", "center_ade", "center_fde", "bbox_iou", "presence_f1"],
                evaluations,
            )
            + "</section>"
        )
    if args.case_gallery is not None:
        gallery = Path(args.case_gallery).resolve()
        relative = Path(os.path.relpath(gallery, output_dir))
        sections.append(
            f"<section><h2>Fixed-identity frame gallery</h2><p><a href=\"{relative.as_posix()}/\">Open gallery</a></p></section>"
        )

    document = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>xSSC Stage 1</title><style>body{{margin:0;background:#11151a;color:#edf1f5;font:14px system-ui}}header,main{{max-width:1450px;margin:auto;padding:22px}}section{{margin:22px 0;border-top:1px solid #343b44;padding-top:14px}}table{{border-collapse:collapse;width:100%;background:#171c22}}th,td{{border:1px solid #343b44;padding:7px 9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}.pass{{color:#4ade80;font-weight:700}}.fail{{color:#fb7185;font-weight:700}}a{{color:#67a9ff}}</style></head><body><header><h1>xSSC Stage 1 · causal object-state audit</h1><p>Q1 stability · Q2 object-history sufficiency · Q3 other-object context</p></header><main>{''.join(sections)}</main></body></html>"""
    (output_dir / "index.html").write_text(document)
    print(json.dumps({"index": str(output_dir / 'index.html'), "evaluation_rows": len(evaluations)}, indent=2))


if __name__ == "__main__":
    main()
