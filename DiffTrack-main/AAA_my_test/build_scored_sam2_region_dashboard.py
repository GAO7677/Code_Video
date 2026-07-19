#!/usr/bin/env python3
"""Build a scored copy of the SAM2 region correspondence dashboard."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import build_stage1b_kubric_analysis_dashboard as base_dashboard


DEFAULT_SOURCE = Path(
    "/data/gaoya/agent-data/outputs/sam2_region_generation_comparison"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/sam2_region_generation_comparison_scored"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--score-root", type=Path, default=DEFAULT_SOURCE / "physv_scores")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_ok(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("status") == "ok" else None


def case_scores(score_root: Path, model: str, case_key: str) -> dict[str, Any]:
    root = score_root / model / case_key
    wm = load_ok(root / "wmreward.json")
    vp = load_ok(root / "videophy2.json")
    cosmos = load_ok(root / "cosmos_reason1.json")
    wm_result = wm.get("result", {}) if wm else {}
    vp_result = vp.get("result", {}) if vp else {}
    cosmos_result = cosmos.get("result", {}) if cosmos else {}
    return {
        "wmreward_surprise": wm_result.get("surprise"),
        "wmreward_similarity": wm_result.get("similarity"),
        "videophy2_sa": vp_result.get("sa", {}).get("score"),
        "videophy2_pc": vp_result.get("pc", {}).get("score"),
        "cosmos_reason1": cosmos_result.get("score"),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, float | None]:
    names = [
        "wmreward_surprise",
        "wmreward_similarity",
        "videophy2_sa",
        "videophy2_pc",
        "cosmos_reason1",
    ]
    summary: dict[str, float | None] = {}
    for name in names:
        values = [float(case["physv"][name]) for case in cases if case["physv"][name] is not None]
        summary[name] = statistics.mean(values) if values else None
        summary[f"{name}_count"] = len(values)
    return summary


def link_assets(output: Path, source: Path, name: str) -> None:
    source_link = source / name
    target = source_link.resolve()
    link = output / name
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(link)
    os.symlink(target, link, target_is_directory=True)


def scored_html() -> str:
    html = base_dashboard.HTML
    html = html.replace(
        '<section class="metrics" id="metrics"></section>',
        '<div class="eyebrow score-heading">Physical plausibility scores</div>'
        '<section class="metrics" id="physv-metrics"></section>'
        '<div class="eyebrow score-heading">Token correspondence scores</div>'
        '<section class="metrics" id="metrics"></section>',
    )
    html = html.replace(
        '.note{font-size:12px;',
        '.score-heading{margin:22px 2px 8px}.note{font-size:12px;',
    )
    marker = "function matrix(method){"
    helper = """function physCard(label,key,digits,direction){const c=currentCase(),m=currentModel(),v=c.physv?.[key],avg=m.physv_summary?.[key],count=m.physv_summary?.[key+'_count']??0;return `<article class=\"card metric\"><span>${label} · ${direction}</span><b>${fmt(v,digits)}</b><small>${m.label} mean ${fmt(avg,digits)} · n=${count}</small></article>`}"""
    html = html.replace(marker, helper + marker)
    render_marker = "document.getElementById('metrics').innerHTML=card("
    render_physv = "document.getElementById('physv-metrics').innerHTML=physCard('WMReward surprise','wmreward_surprise',3,'lower is better')+physCard('VideoPhy2-SA','videophy2_sa',1,'higher is better')+physCard('VideoPhy2-PC','videophy2_pc',1,'higher is better')+physCard('Cosmos-R1','cosmos_reason1',1,'higher is better');"
    html = html.replace(render_marker, render_physv + render_marker)
    return html


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    score_root = args.score_root.resolve()
    payload = json.loads((source / "dashboard_data.json").read_text(encoding="utf-8"))
    for model in payload["models"]:
        for case in model["cases"]:
            case["physv"] = case_scores(score_root, model["name"], case["case_key"])
        model["physv_summary"] = summarize(model["cases"])

    output.mkdir(parents=True, exist_ok=True)
    for model in payload["models"]:
        link_assets(output, source, model["name"])
    (output / "dashboard_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    (output / "index.html").write_text(
        scored_html().replace("__PAYLOAD__", serialized), encoding="utf-8"
    )
    print(f"dashboard: {output / 'index.html'}")


if __name__ == "__main__":
    main()
