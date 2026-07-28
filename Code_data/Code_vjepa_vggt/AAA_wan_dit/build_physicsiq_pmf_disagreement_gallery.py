#!/usr/bin/env python3
"""Visualize same-source ablations ranked oppositely by metric pairs."""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_metric_extreme_pair_gallery import (
    DEFAULT_BATCH_ROOT,
    METRIC_TITLES,
    MODEL_LABELS,
    MODEL_ORDER,
    add_case_ids,
    atomic_text,
    ensure_video_link,
    method_label,
    read_sidecar,
    resolve_source_metadata,
    score_payload,
)
from summarize_stc_bench_metrics import METRICS


DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed/seed851/benchmark-metrics/"
    "metric-extreme-pairs/physics-iq-pmf-disagreement"
)
DEFAULT_BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_seed851_baseline_bench"
)
CONTEXTS = {
    "with_context": {
        "label": "Physics-IQ vs PMF · With context",
        "physics_iq": "physics_iq_with_context",
        "pmf": "pmf_with_context",
        "left_label": "Physics-IQ",
        "right_label": "PMF",
    },
    "without_context": {
        "label": "Physics-IQ vs PMF · Without context",
        "physics_iq": "physics_iq_without_context",
        "pmf": "pmf_without_context",
        "left_label": "Physics-IQ",
        "right_label": "PMF",
    },
    "videophy2_pc_vs_cosmos": {
        "label": "VideoPhy2-PC vs Cosmos-Reason1",
        "physics_iq": "videophy2_pc",
        "pmf": "cosmos_reason1",
        "left_label": "VideoPhy2-PC",
        "right_label": "Cosmos-Reason1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def choose_disagreement(
    model_frame: pd.DataFrame,
    physics_metric: str,
    pmf_metric: str,
) -> tuple[pd.Series, pd.Series, dict[str, float | int]]:
    physics_scale = float(model_frame[physics_metric].std(ddof=1))
    pmf_scale = float(model_frame[pmf_metric].std(ddof=1))
    if physics_scale <= 0 or pmf_scale <= 0:
        raise RuntimeError("Metric scale is zero")
    best_key: tuple[Any, ...] | None = None
    best_payload: tuple[pd.Series, pd.Series, dict[str, float | int]] | None = None
    discordant_pairs = 0
    for case_id, case_frame in model_frame.groupby("case_id", sort=True):
        valid = case_frame.dropna(subset=[physics_metric, pmf_metric])
        for (_, first), (_, second) in itertools.combinations(
            valid.iterrows(),
            2,
        ):
            physics_delta = float(
                first[physics_metric] - second[physics_metric]
            )
            pmf_delta = float(first[pmf_metric] - second[pmf_metric])
            if physics_delta * pmf_delta >= 0:
                continue
            discordant_pairs += 1
            normalized_physics = abs(physics_delta) / physics_scale
            normalized_pmf = abs(pmf_delta) / pmf_scale
            conflict_strength = min(normalized_physics, normalized_pmf)
            geometric_strength = math.sqrt(
                normalized_physics * normalized_pmf
            )
            key = (
                conflict_strength,
                geometric_strength,
                abs(physics_delta),
                abs(pmf_delta),
                str(case_id),
                str(first["variant"]),
                str(second["variant"]),
            )
            if best_key is not None and key <= best_key:
                continue
            physics_preferred, pmf_preferred = (
                (first, second)
                if physics_delta > 0
                else (second, first)
            )
            best_key = key
            best_payload = (
                physics_preferred,
                pmf_preferred,
                {
                    "physics_gap": abs(physics_delta),
                    "pmf_gap": abs(pmf_delta),
                    "physics_gap_sigma": normalized_physics,
                    "pmf_gap_sigma": normalized_pmf,
                    "conflict_strength": conflict_strength,
                    "discordant_pair_count": discordant_pairs,
                },
            )
    if best_payload is None:
        raise RuntimeError(
            f"No Physics-IQ/PMF disagreement for {physics_metric}"
        )
    best_payload[2]["discordant_pair_count"] = discordant_pairs
    return best_payload


def build_record(
    batch_root: Path,
    output_dir: Path,
    context_name: str,
    model: str,
    physics_preferred: pd.Series,
    pmf_preferred: pd.Series,
    baseline: pd.Series,
    stats: dict[str, float | int],
) -> dict[str, Any]:
    context = CONTEXTS[context_name]
    physics_meta = read_sidecar(
        batch_root,
        str(physics_preferred["entry_id"]),
    )
    pmf_meta = read_sidecar(batch_root, str(pmf_preferred["entry_id"]))
    physics_source, physics_prompt = resolve_source_metadata(physics_meta)
    pmf_source, pmf_prompt = resolve_source_metadata(pmf_meta)
    if (
        physics_preferred["case_id"] != pmf_preferred["case_id"]
        or physics_source != pmf_source
    ):
        raise RuntimeError("Disagreement pair does not share one source")
    asset_dir = output_dir / "assets" / context_name / model
    source_target = asset_dir / "source.mp4"
    physics_target = asset_dir / "physics_iq_preferred.mp4"
    pmf_target = asset_dir / "pmf_preferred.mp4"
    ensure_video_link(physics_source, source_target)
    ensure_video_link(
        Path(str(physics_meta["output_video"])),
        physics_target,
    )
    ensure_video_link(Path(str(pmf_meta["output_video"])), pmf_target)
    relative = lambda path: path.relative_to(output_dir).as_posix()
    return {
        "context": context_name,
        "context_label": context["label"],
        "physics_metric": context["physics_iq"],
        "pmf_metric": context["pmf"],
        "left_label": context["left_label"],
        "right_label": context["right_label"],
        "model": model,
        "model_label": MODEL_LABELS[model],
        "case_id": str(physics_preferred["case_id"]),
        "prompt": physics_prompt or pmf_prompt,
        **stats,
        "source_video": relative(source_target),
        "baseline": {
            "entry_id": str(baseline["entry_id"]),
            "scores": score_payload(baseline),
        },
        "physics_preferred": {
            "entry_id": str(physics_preferred["entry_id"]),
            "method": method_label(physics_preferred),
            "video": relative(physics_target),
            "scores": score_payload(physics_preferred),
        },
        "pmf_preferred": {
            "entry_id": str(pmf_preferred["entry_id"]),
            "method": method_label(pmf_preferred),
            "video": relative(pmf_target),
            "scores": score_payload(pmf_preferred),
        },
    }


def build_records(
    batch_root: Path,
    output_dir: Path,
    frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    records = []
    for context_name, context in CONTEXTS.items():
        for model in MODEL_ORDER:
            model_frame = frame[frame["model"] == model]
            physics_preferred, pmf_preferred, stats = choose_disagreement(
                model_frame,
                context["physics_iq"],
                context["pmf"],
            )
            baseline_rows = baseline_frame[
                (baseline_frame["model"] == model)
                & (
                    baseline_frame["case_id"]
                    == physics_preferred["case_id"]
                )
                & (baseline_frame["variant"] == "baseline")
            ]
            if len(baseline_rows) != 1:
                raise RuntimeError(
                    f"Expected one baseline for {model}/"
                    f"{physics_preferred['case_id']}, got "
                    f"{len(baseline_rows)}"
                )
            records.append(
                build_record(
                    batch_root,
                    output_dir,
                    context_name,
                    model,
                    physics_preferred,
                    pmf_preferred,
                    baseline_rows.iloc[0],
                    stats,
                )
            )
    return records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "comparison",
        "model",
        "case_id",
        "conflict_strength",
        "left_metric",
        "right_metric",
        "left_gap",
        "left_gap_sigma",
        "right_gap",
        "right_gap_sigma",
        "left_preferred_method",
        "left_preferred_left_score",
        "left_preferred_right_score",
        "right_preferred_method",
        "right_preferred_left_score",
        "right_preferred_right_score",
        "baseline_left_score",
        "baseline_right_score",
        "discordant_pair_count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "comparison": record["context"],
                    "model": record["model"],
                    "case_id": record["case_id"],
                    "conflict_strength": record["conflict_strength"],
                    "left_metric": record["physics_metric"],
                    "right_metric": record["pmf_metric"],
                    "left_gap": record["physics_gap"],
                    "left_gap_sigma": record["physics_gap_sigma"],
                    "right_gap": record["pmf_gap"],
                    "right_gap_sigma": record["pmf_gap_sigma"],
                    "left_preferred_method": record[
                        "physics_preferred"
                    ]["method"],
                    "left_preferred_left_score": record[
                        "physics_preferred"
                    ]["scores"][record["physics_metric"]],
                    "left_preferred_right_score": record[
                        "physics_preferred"
                    ]["scores"][record["pmf_metric"]],
                    "right_preferred_method": record["pmf_preferred"][
                        "method"
                    ],
                    "right_preferred_left_score": record[
                        "pmf_preferred"
                    ]["scores"][record["physics_metric"]],
                    "right_preferred_right_score": record["pmf_preferred"][
                        "scores"
                    ][record["pmf_metric"]],
                    "baseline_left_score": record["baseline"]["scores"][
                        record["physics_metric"]
                    ],
                    "baseline_right_score": record["baseline"]["scores"][
                        record["pmf_metric"]
                    ],
                    "discordant_pair_count": record[
                        "discordant_pair_count"
                    ],
                }
            )


def build_html(records: list[dict[str, Any]]) -> str:
    metric_meta = [
        {
            "name": metric.name,
            "title": METRIC_TITLES[metric.name],
            "direction": metric.direction,
        }
        for metric in METRICS
    ]
    payload = json.dumps(
        {"metrics": metric_meta, "records": records},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    options = "".join(
        f"<option value='{html.escape(name)}'>"
        f"{html.escape(context['label'])}</option>"
        for name, context in CONTEXTS.items()
    )
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>跨指标评价歧义</title>
<style>
:root{{--bg:#f4f5f2;--panel:#fff;--ink:#202423;--muted:#66706b;--line:#cbd1cd;--piq:#1969a6;--pmf:#a65a18;--good:#14734d;--bad:#a13d35}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:20;background:rgba(244,245,242,.97);border-bottom:1px solid var(--line)}}
.bar,main{{max-width:1800px;margin:auto;padding:14px 22px}}.bar{{display:flex;align-items:end;gap:18px;flex-wrap:wrap}}
h1,h2,p{{margin:0}}h1{{font-size:22px}}h2{{font-size:18px}}.sub,.identity,.note{{color:var(--muted)}}
label{{display:grid;gap:4px;color:var(--muted);font-size:12px}}select,button{{font:inherit;border:1px solid #adb6b0;background:#fff;padding:7px 10px}}
select{{min-width:210px}}button{{cursor:pointer}}.note{{margin:16px 0;padding:10px 12px;background:#fff;border-left:3px solid var(--piq)}}
.model{{padding:17px 0 25px;border-top:1px solid var(--line)}}.model-head{{display:flex;justify-content:space-between;align-items:start;gap:18px;margin-bottom:10px}}
.identity{{overflow-wrap:anywhere}}.strength{{font-weight:700}}.gaps{{font-size:12px;color:var(--muted);text-align:right}}
.videos{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}figure{{margin:0;min-width:0;background:#fff;border:1px solid var(--line)}}
figcaption{{min-height:72px;padding:8px 10px;border-bottom:1px solid var(--line)}}figcaption strong,figcaption span{{display:block}}figcaption span{{font-size:12px;color:var(--muted)}}
.piq-label{{color:var(--piq)}}.pmf-label{{color:var(--pmf)}}video{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}}
.table-wrap{{overflow:auto;margin-top:10px;border:1px solid var(--line);background:#fff}}table{{border-collapse:collapse;width:100%;min-width:900px}}
th,td{{padding:6px 9px;border-bottom:1px solid #e5e9e6;text-align:right}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#edf1ee}}
tr.piq-row{{background:#eef7ff;font-weight:700}}tr.pmf-row{{background:#fff4e8;font-weight:700}}td.good{{color:var(--good);background:#f1faf5}}td.bad{{color:var(--bad);background:#fff6f4}}
.direction{{color:var(--muted);font-weight:400}}.footer{{margin:18px 0;color:var(--muted)}}a{{color:#176f62}}
@media(max-width:900px){{.videos{{grid-template-columns:1fr}}}}
</style></head><body><header><div class="bar"><div><h1>跨指标评价歧义</h1>
<p class="sub">Seed 851 · test_5 · 更新 {updated}</p></div>
<label>评价模式<select id="context">{options}</select></label></div></header>
<main><p class="note">每组保持模型和 source 不变，左、右两段消融分别被两个待比较指标判得更好。冲突强度先按各模型内指标标准差归一化，再取两项分差中较小者；因此选中的pair不是单项极端，而是两个指标都明确反向的case。</p>
<div id="models"></div>
<p class="footer"><a href="metric_disagreement.csv">下载9组歧义清单</a> · <a href="../physics-iq-pmf.html">查看单指标极端对比</a> · <a href="../../">返回完整指标页</a></p></main>
<script id="payload" type="application/json">{payload}</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent);
const select=document.getElementById('context'),root=document.getElementById('models');
const esc=v=>String(v).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const fmt=v=>v===null?'NA':Number(v).toPrecision(5).replace(/(?:\\.0+|(?:(\\.\\d*?[1-9]))0+)$/,'$1');
function delta(value,baseline,direction){{if(value===null||baseline===null)return{{text:'NA',cls:''}};const raw=value-baseline,improvement=direction==='higher'?raw:-raw;return{{text:`${{raw>=0?'+':''}}${{fmt(raw)}}`,cls:improvement>0?'good':improvement<0?'bad':''}};}}
function render(){{
 const records=data.records.filter(item=>item.context===select.value);
 root.innerHTML=records.map((record,index)=>{{
  const p=record.physics_preferred,m=record.pmf_preferred,b=record.baseline;
  const rows=data.metrics.map(metric=>{{
   const a=p.scores[metric.name],z=m.scores[metric.name],base=b.scores[metric.name];
   const da=delta(a,base,metric.direction),dz=delta(z,base,metric.direction);
   const rowClass=metric.name===record.physics_metric?'piq-row':metric.name===record.pmf_metric?'pmf-row':'';
   return `<tr class="${{rowClass}}"><td>${{esc(metric.title)}} <span class="direction">${{metric.direction==='higher'?'↑':'↓'}}</span></td><td>${{fmt(base)}}</td><td>${{fmt(a)}}</td><td class="${{da.cls}}">${{da.text}}</td><td>${{fmt(z)}}</td><td class="${{dz.cls}}">${{dz.text}}</td></tr>`;
  }}).join('');
  return `<section class="model"><div class="model-head"><div><h2>${{esc(record.model_label)}}</h2><p class="identity">Source: ${{esc(record.case_id)}}<br>Prompt: ${{esc(record.prompt)}}</p></div><div class="gaps"><span class="strength">冲突强度 ${{fmt(record.conflict_strength)}}σ</span><br>${{esc(record.left_label)}} gap ${{fmt(record.physics_gap)}} (${{fmt(record.physics_gap_sigma)}}σ)<br>${{esc(record.right_label)}} gap ${{fmt(record.pmf_gap)}} (${{fmt(record.pmf_gap_sigma)}}σ)<br><button data-play="${{index}}" type="button">同步重播本行</button></div></div>
  <div class="videos" data-row="${{index}}"><figure><figcaption><strong>Source / GT</strong><span>相同输入视频</span></figcaption><video controls muted preload="metadata" src="${{esc(record.source_video)}}"></video></figure>
  <figure><figcaption><strong class="piq-label">${{esc(record.left_label)}} 判定更好</strong><span>${{esc(p.method)}} · ${{esc(record.left_label)}} ${{fmt(p.scores[record.physics_metric])}}（Base ${{fmt(b.scores[record.physics_metric])}}）· ${{esc(record.right_label)}} ${{fmt(p.scores[record.pmf_metric])}}（Base ${{fmt(b.scores[record.pmf_metric])}}）</span></figcaption><video controls muted preload="metadata" src="${{esc(p.video)}}"></video></figure>
  <figure><figcaption><strong class="pmf-label">${{esc(record.right_label)}} 判定更好</strong><span>${{esc(m.method)}} · ${{esc(record.left_label)}} ${{fmt(m.scores[record.physics_metric])}}（Base ${{fmt(b.scores[record.physics_metric])}}）· ${{esc(record.right_label)}} ${{fmt(m.scores[record.pmf_metric])}}（Base ${{fmt(b.scores[record.pmf_metric])}}）</span></figcaption><video controls muted preload="metadata" src="${{esc(m.video)}}"></video></figure></div>
  <div class="table-wrap"><table><thead><tr><th>指标</th><th>Baseline</th><th>${{esc(p.method)}}</th><th>Δ vs Base</th><th>${{esc(m.method)}}</th><th>Δ vs Base</th></tr></thead><tbody>${{rows}}</tbody></table></div></section>`;
 }}).join('');
 document.querySelectorAll('[data-play]').forEach(button=>button.addEventListener('click',()=>{{const videos=document.querySelector(`[data-row="${{button.dataset.play}}"]`).querySelectorAll('video');videos.forEach(v=>{{v.pause();v.currentTime=0;}});videos.forEach(v=>v.play().catch(()=>null));}}));
}}
select.addEventListener('change',render);render();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    baseline_root = args.baseline_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    frame = add_case_ids(
        pd.read_csv(batch_root / "analysis" / "per_video_metrics.csv")
    )
    baseline_frame = add_case_ids(
        pd.read_csv(
            baseline_root / "analysis" / "per_video_metrics.csv"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(
        batch_root,
        output_dir,
        frame,
        baseline_frame,
    )
    atomic_text(
        output_dir / "metric_disagreement.json",
        json.dumps(records, ensure_ascii=False, indent=2),
    )
    write_csv(output_dir / "metric_disagreement.csv", records)
    atomic_text(output_dir / "index.html", build_html(records))
    print(
        f"[metric-disagreement] groups={len(records)} "
        f"output={output_dir / 'index.html'}"
    )


if __name__ == "__main__":
    main()
