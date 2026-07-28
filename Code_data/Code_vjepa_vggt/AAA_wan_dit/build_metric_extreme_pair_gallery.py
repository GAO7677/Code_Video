#!/usr/bin/env python3
"""Build same-model, same-source extreme metric video comparisons."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from summarize_stc_bench_metrics import METRICS


DEFAULT_BATCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_st_phased_seed851_bench"
)
DEFAULT_BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_seed851_baseline_bench"
)
DEFAULT_GT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_gt49f_896x512_bench"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed/seed851/benchmark-metrics/metric-extreme-pairs"
)
MODEL_ORDER = ("wan_lora", "xssc", "physrvg")
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
PHYSICS_PMF_METRICS = (
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
)
ROLE_LABELS = {"S": "S-only", "T": "T-only", "ST": "S+T", "C": "C-only"}
METRIC_TITLES = {
    "physics_iq_with_context": "Physics-IQ with context",
    "physics_iq_without_context": "Physics-IQ without context",
    "pmf_with_context": "PMF with context",
    "pmf_without_context": "PMF without context",
    "wmreward_surprise": "WMReward surprise",
    "vbench_subject_consistency": "VBench subject consistency",
    "vbench_background_consistency": "VBench background consistency",
    "vbench_temporal_flickering": "VBench temporal flickering",
    "vbench_motion_smoothness": "VBench motion smoothness",
    "vbench_dynamic_degree": "VBench dynamic degree",
    "vbench_aesthetic_quality": "VBench aesthetic quality",
    "vbench_imaging_quality": "VBench imaging quality",
    "videophy2_sa": "VideoPhy2 semantic adherence",
    "videophy2_pc": "VideoPhy2 physical commonsense",
    "videophy2_joint_rate": "VideoPhy2 joint pass rate",
    "videophy2_pc_raw": "VideoPhy2 physical commonsense raw",
    "cosmos_reason1": "Cosmos-Reason1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def add_case_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["case_id"] = [
        entry_id.split(f"__{variant}__", 1)[1]
        for entry_id, variant in zip(frame["entry_id"], frame["variant"])
    ]
    return frame


def method_label(row: pd.Series) -> str:
    role = ROLE_LABELS.get(str(row["role"]), str(row["role"]))
    return (
        f"{role} "
        f"[{int(row['denoise_start'])},{int(row['denoise_end'])})"
    )


def read_sidecar(batch_root: Path, entry_id: str) -> dict[str, Any]:
    path = batch_root / "cases" / f"{entry_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source_metadata(payload: dict[str, Any]) -> tuple[Path, str]:
    source_payload: dict[str, Any] = {}
    input_json = payload.get("input_json")
    if isinstance(input_json, str):
        input_json_path = Path(input_json).expanduser()
        if input_json_path.is_file():
            source_payload = json.loads(
                input_json_path.read_text(encoding="utf-8")
            )
    source_value = (
        payload.get("source_video")
        or source_payload.get("source_video")
        or payload.get("input_video_original")
    )
    if not isinstance(source_value, str):
        raise KeyError("Cannot resolve source_video from sidecar or input_json")
    prompt_value = (
        payload.get("input_caption")
        or source_payload.get("input_caption")
        or ""
    )
    return Path(source_value).expanduser().resolve(), str(prompt_value)


def ensure_video_link(source: Path, target: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    desired = os.path.relpath(source, target.parent)
    if target.is_symlink() and os.readlink(target) == desired:
        return
    if os.path.lexists(target):
        target.unlink()
    target.symlink_to(desired)


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def score_payload(row: pd.Series) -> dict[str, float | None]:
    return {
        metric.name: finite_number(row[metric.name])
        for metric in METRICS
    }


def choose_pair(
    model_frame: pd.DataFrame,
    metric_name: str,
    direction: str,
) -> tuple[pd.Series, pd.Series, float]:
    candidates = []
    for case_id, case_frame in model_frame.groupby("case_id", sort=True):
        valid = case_frame[case_frame[metric_name].notna()].sort_values(
            [metric_name, "variant"],
            kind="stable",
        )
        if len(valid) < 2:
            continue
        low = valid.iloc[0]
        high = valid.iloc[-1]
        best, worst = (high, low) if direction == "higher" else (low, high)
        gap = abs(float(high[metric_name]) - float(low[metric_name]))
        candidates.append((gap, str(case_id), best, worst))
    if not candidates:
        raise RuntimeError(f"No pair candidates for {metric_name}")
    _, _, best, worst = max(candidates, key=lambda item: (item[0], item[1]))
    gap = abs(float(best[metric_name]) - float(worst[metric_name]))
    return best, worst, gap


def build_record(
    batch_root: Path,
    baseline_root: Path,
    gt_root: Path,
    output_dir: Path,
    metric_name: str,
    direction: str,
    model: str,
    best: pd.Series,
    worst: pd.Series,
    baseline: pd.Series,
    gt: pd.Series,
    gap: float,
) -> dict[str, Any]:
    best_meta = read_sidecar(batch_root, str(best["entry_id"]))
    worst_meta = read_sidecar(batch_root, str(worst["entry_id"]))
    baseline_meta = read_sidecar(baseline_root, str(baseline["entry_id"]))
    gt_meta = read_sidecar(gt_root, str(gt["entry_id"]))
    if best["case_id"] != worst["case_id"]:
        raise RuntimeError("Extreme pair does not share a source case")
    source_best, prompt_best = resolve_source_metadata(best_meta)
    source_worst, prompt_worst = resolve_source_metadata(worst_meta)
    if source_best != source_worst:
        raise RuntimeError(
            f"Source mismatch for {metric_name}/{model}: "
            f"{source_best} != {source_worst}"
        )
    asset_dir = output_dir / "assets" / metric_name / model
    gt_target = asset_dir / "gt_49f_30fps_896x512.mp4"
    baseline_target = asset_dir / "baseline.mp4"
    best_target = asset_dir / "metric_best.mp4"
    worst_target = asset_dir / "metric_worst.mp4"
    ensure_video_link(Path(str(gt_meta["output_video"])), gt_target)
    ensure_video_link(
        Path(str(baseline_meta["output_video"])),
        baseline_target,
    )
    ensure_video_link(Path(str(best_meta["output_video"])), best_target)
    ensure_video_link(Path(str(worst_meta["output_video"])), worst_target)
    relative = lambda path: path.relative_to(output_dir).as_posix()
    prompt = prompt_best or prompt_worst
    return {
        "metric": metric_name,
        "metric_title": METRIC_TITLES[metric_name],
        "direction": direction,
        "model": model,
        "model_label": MODEL_LABELS[model],
        "case_id": str(best["case_id"]),
        "prompt": prompt,
        "selection_gap": gap,
        "source_video_original": str(source_best),
        "gt": {
            "entry_id": str(gt["entry_id"]),
            "method": "GT · 49f @ 30 FPS · 896×512",
            "video": relative(gt_target),
            "scores": score_payload(gt),
        },
        "baseline": {
            "entry_id": str(baseline["entry_id"]),
            "method": f"{MODEL_LABELS[model]} baseline",
            "video": relative(baseline_target),
            "scores": score_payload(baseline),
        },
        "best": {
            "entry_id": str(best["entry_id"]),
            "variant": str(best["variant"]),
            "method": method_label(best),
            "video": relative(best_target),
            "scores": score_payload(best),
        },
        "worst": {
            "entry_id": str(worst["entry_id"]),
            "variant": str(worst["variant"]),
            "method": method_label(worst),
            "video": relative(worst_target),
            "scores": score_payload(worst),
        },
    }


def build_records(
    batch_root: Path,
    baseline_root: Path,
    gt_root: Path,
    output_dir: Path,
    frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    gt_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    records = []
    for metric in METRICS:
        for model in MODEL_ORDER:
            model_frame = frame[frame["model"] == model]
            best, worst, gap = choose_pair(
                model_frame,
                metric.name,
                metric.direction,
            )
            case_id = str(best["case_id"])
            baseline_matches = baseline_frame[
                (baseline_frame["model"] == model)
                & (baseline_frame["case_id"] == case_id)
            ]
            gt_matches = gt_frame[gt_frame["case_id"] == case_id]
            if len(baseline_matches) != 1:
                raise RuntimeError(
                    f"Expected one baseline for {model}/{case_id}, "
                    f"found {len(baseline_matches)}"
                )
            if len(gt_matches) != 1:
                raise RuntimeError(
                    f"Expected one GT for {case_id}, found {len(gt_matches)}"
                )
            records.append(
                build_record(
                    batch_root,
                    baseline_root,
                    gt_root,
                    output_dir,
                    metric.name,
                    metric.direction,
                    model,
                    best,
                    worst,
                    baseline_matches.iloc[0],
                    gt_matches.iloc[0],
                    gap,
                )
            )
    return records


def write_selection_csv(
    output_dir: Path,
    records: list[dict[str, Any]],
    filename: str = "extreme_pair_selection.csv",
) -> None:
    path = output_dir / filename
    fields = (
        "metric",
        "direction",
        "model",
        "case_id",
        "selection_gap",
        "gt_score",
        "baseline_score",
        "best_method",
        "best_entry_id",
        "best_score",
        "worst_method",
        "worst_entry_id",
        "worst_score",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metric = record["metric"]
            writer.writerow(
                {
                    "metric": metric,
                    "direction": record["direction"],
                    "model": record["model"],
                    "case_id": record["case_id"],
                    "selection_gap": record["selection_gap"],
                    "gt_score": record["gt"]["scores"][metric],
                    "baseline_score": record["baseline"]["scores"][metric],
                    "best_method": record["best"]["method"],
                    "best_entry_id": record["best"]["entry_id"],
                    "best_score": record["best"]["scores"][metric],
                    "worst_method": record["worst"]["method"],
                    "worst_entry_id": record["worst"]["entry_id"],
                    "worst_score": record["worst"]["scores"][metric],
                }
            )


def build_html(
    records: list[dict[str, Any]],
    selectable_metrics: tuple[str, ...] | None = None,
    page_title: str = "指标极端消融视频对比",
    method_note: str | None = None,
    selection_filename: str = "extreme_pair_selection.csv",
    selection_label: str = "下载51组选择清单",
    include_all_metrics_link: bool = False,
) -> str:
    selectable_metrics = selectable_metrics or tuple(
        metric.name for metric in METRICS
    )
    visible_records = [
        record
        for record in records
        if record["metric"] in selectable_metrics
    ]
    metric_meta = [
        {
            "name": metric.name,
            "title": METRIC_TITLES[metric.name],
            "direction": metric.direction,
        }
        for metric in METRICS
    ]
    payload = json.dumps(
        {"metrics": metric_meta, "records": visible_records},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    options = "".join(
        f"<option value='{html.escape(metric.name)}'>"
        f"{html.escape(METRIC_TITLES[metric.name])}</option>"
        for metric in METRICS
        if metric.name in selectable_metrics
    )
    note = method_note or (
        "每个模型内保持 source case 不变，在15种消融中取当前指标判定"
        "最好与最差的视频；再选择分差最大的 source case。标签“较好/较差”"
        "只代表当前所选指标，不代表综合视觉质量。GT 统一为49帧、30 FPS、"
        "896×512；表格同时列出同 case 的 GT、模型 baseline 和两种消融。"
    )
    all_metrics_link = (
        ' · <a href="index.html">查看全部17项指标</a>'
        if include_all_metrics_link
        else ""
    )
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(page_title)}</title>
<style>
:root{{--bg:#f4f5f2;--panel:#fff;--ink:#202423;--muted:#68716d;--line:#ccd2ce;--accent:#176f62;--good:#14734d;--bad:#a13d35}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:20;background:rgba(244,245,242,.97);border-bottom:1px solid var(--line)}}
.bar,main{{max-width:1800px;margin:auto;padding:14px 22px}}.bar{{display:flex;align-items:end;gap:18px;flex-wrap:wrap}}
h1,h2,h3,p{{margin:0}}h1{{font-size:22px}}h2{{font-size:18px}}h3{{font-size:14px}}
.sub{{color:var(--muted)}}label{{display:grid;gap:4px;color:var(--muted);font-size:12px}}
select,button{{font:inherit;border:1px solid #aeb7b1;background:#fff;color:var(--ink);padding:7px 10px}}
select{{min-width:320px}}button{{cursor:pointer}}button:hover{{border-color:var(--accent);color:var(--accent)}}
.method-note{{margin:16px 0;padding:10px 12px;border-left:3px solid var(--accent);background:#fff}}
.model{{padding:17px 0 24px;border-top:1px solid var(--line)}}.model-head{{display:flex;align-items:start;justify-content:space-between;gap:16px;margin-bottom:10px}}
.identity{{color:var(--muted);overflow-wrap:anywhere}}.gap{{font-weight:700;color:var(--accent)}}
.videos{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
figure{{margin:0;min-width:0;background:#fff;border:1px solid var(--line)}}figcaption{{padding:8px 10px;min-height:59px;border-bottom:1px solid var(--line)}}
figcaption strong,figcaption span{{display:block}}figcaption span{{color:var(--muted);font-size:12px}}
video{{display:block;width:100%;aspect-ratio:16/9;background:#111;object-fit:contain}}
.score-table-wrap{{overflow:auto;margin-top:10px;border:1px solid var(--line);background:#fff}}
table{{border-collapse:collapse;width:100%;min-width:1100px}}th,td{{padding:6px 9px;border-bottom:1px solid #e5e9e6;text-align:right}}
th:first-child,td:first-child{{text-align:left}}thead th{{background:#edf1ee}}tr.selected{{background:#fff5d9;font-weight:700}}
td.winner{{color:var(--good);background:#f1faf5}}td.loser{{color:var(--bad);background:#fff6f4}}
.direction{{color:var(--muted);font-weight:400}}.download{{margin:18px 0;color:var(--muted)}}
a{{color:var(--accent)}}@media(max-width:900px){{.videos{{grid-template-columns:1fr}}select{{min-width:min(100%,320px)}}}}
</style></head><body>
<header><div class="bar"><div><h1>{html.escape(page_title)}</h1>
<p class="sub">Seed 851 · test_5 · 更新 {updated}</p></div>
<label>指标<select id="metric">{options}</select></label></div></header>
<main><p class="method-note">{html.escape(note)}</p>
<div id="models"></div>
<p class="download"><a href="{html.escape(selection_filename)}">{html.escape(selection_label)}</a>{all_metrics_link} · <a href="../">返回完整指标页</a></p></main>
<script id="payload" type="application/json">{payload}</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent);
const select=document.getElementById('metric');
const root=document.getElementById('models');
const fmt=value=>value===null?'NA':Number(value).toPrecision(5).replace(/(?:\\.0+|(?:(\\.\\d*?[1-9]))0+)$/,'$1');
const esc=value=>String(value).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function scoreClass(a,b,direction){{
  if(a===null||b===null||a===b)return ['', ''];
  const firstWins=direction==='higher'?a>b:a<b;
  return firstWins?['winner','loser']:['loser','winner'];
}}
function render(){{
  const metric=select.value;
  const meta=data.metrics.find(item=>item.name===metric);
  const records=data.records.filter(item=>item.metric===metric);
  root.innerHTML=records.map((record,index)=>{{
    const rows=data.metrics.map(item=>{{
      const gt=record.gt.scores[item.name],base=record.baseline.scores[item.name];
      const a=record.best.scores[item.name],b=record.worst.scores[item.name];
      const classes=scoreClass(a,b,item.direction);
      const raw=a===null||b===null?null:a-b;
      return `<tr class="${{item.name===metric?'selected':''}}"><td>${{esc(item.title)}} <span class="direction">${{item.direction==='higher'?'↑':'↓'}}</span></td><td>${{fmt(gt)}}</td><td>${{fmt(base)}}</td><td class="${{classes[0]}}">${{fmt(a)}}</td><td class="${{classes[1]}}">${{fmt(b)}}</td><td>${{raw===null?'NA':(raw>=0?'+':'')+fmt(raw)}}</td></tr>`;
    }}).join('');
    return `<section class="model"><div class="model-head"><div><h2>${{esc(record.model_label)}}</h2><p class="identity">Source: ${{esc(record.case_id)}}<br>Prompt: ${{esc(record.prompt)}}</p></div><div><span class="gap">${{esc(meta.title)}} 分差 ${{fmt(record.selection_gap)}}</span><br><button type="button" data-play="${{index}}">同步重播本行</button></div></div>
    <div class="videos" data-row="${{index}}">
    <figure><figcaption><strong>GT</strong><span>49f @ 30 FPS · 896×512 · ${{fmt(record.gt.scores[metric])}}</span></figcaption><video controls muted preload="metadata" src="${{esc(record.gt.video)}}"></video></figure>
    <figure><figcaption><strong>${{esc(record.model_label)}} baseline</strong><span>未消融 · ${{fmt(record.baseline.scores[metric])}}</span></figcaption><video controls muted preload="metadata" src="${{esc(record.baseline.video)}}"></video></figure>
    <figure><figcaption><strong>当前指标判定较好</strong><span>${{esc(record.best.method)}} · ${{fmt(record.best.scores[metric])}}</span></figcaption><video controls muted preload="metadata" src="${{esc(record.best.video)}}"></video></figure>
    <figure><figcaption><strong>当前指标判定较差</strong><span>${{esc(record.worst.method)}} · ${{fmt(record.worst.scores[metric])}}</span></figcaption><video controls muted preload="metadata" src="${{esc(record.worst.video)}}"></video></figure>
    </div><div class="score-table-wrap"><table><thead><tr><th>指标</th><th>GT</th><th>${{esc(record.model_label)}} baseline</th><th>${{esc(record.best.method)}}</th><th>${{esc(record.worst.method)}}</th><th>较好消融 - 较差消融</th></tr></thead><tbody>${{rows}}</tbody></table></div></section>`;
  }}).join('');
  document.querySelectorAll('[data-play]').forEach(button=>button.addEventListener('click',()=>{{
    const videos=document.querySelector(`[data-row="${{button.dataset.play}}"]`).querySelectorAll('video');
    videos.forEach(video=>{{video.pause();video.currentTime=0;}});
    Promise.all([...videos].map(video=>video.play().catch(()=>null)));
  }}));
}}
select.addEventListener('change',render);render();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    baseline_root = args.baseline_root.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    frame = add_case_ids(
        pd.read_csv(batch_root / "analysis" / "per_video_metrics.csv")
    )
    baseline_frame = add_case_ids(
        pd.read_csv(
            baseline_root / "analysis" / "per_video_metrics.csv"
        )
    )
    gt_frame = add_case_ids(
        pd.read_csv(gt_root / "analysis" / "per_video_metrics.csv")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(
        batch_root,
        baseline_root,
        gt_root,
        output_dir,
        frame,
        baseline_frame,
        gt_frame,
    )
    atomic_text(
        output_dir / "extreme_pair_selection.json",
        json.dumps(records, ensure_ascii=False, indent=2),
    )
    write_selection_csv(output_dir, records)
    atomic_text(output_dir / "index.html", build_html(records))
    physics_pmf_records = [
        record
        for record in records
        if record["metric"] in PHYSICS_PMF_METRICS
    ]
    write_selection_csv(
        output_dir,
        physics_pmf_records,
        filename="physics_iq_pmf_extreme_pair_selection.csv",
    )
    atomic_text(
        output_dir / "physics-iq-pmf.html",
        build_html(
            records,
            selectable_metrics=PHYSICS_PMF_METRICS,
            page_title="Physics-IQ / PMF 极端消融视频对比",
            method_note=(
                "分别针对 Physics-IQ 与 PMF 的 with/without context 分数，"
                "在每个模型内固定 source，选择15种消融中分差最大的最好/最差"
                "视频。两类指标量纲不同，不直接相减；标签只代表当前所选指标。"
                "GT 统一为49帧、30 FPS、896×512，并与模型 baseline 一起列出。"
            ),
            selection_filename="physics_iq_pmf_extreme_pair_selection.csv",
            selection_label="下载12组选择清单",
            include_all_metrics_link=True,
        ),
    )
    print(
        f"[metric-extreme-gallery] groups={len(records)} "
        f"output={output_dir / 'index.html'}"
    )


if __name__ == "__main__":
    main()
