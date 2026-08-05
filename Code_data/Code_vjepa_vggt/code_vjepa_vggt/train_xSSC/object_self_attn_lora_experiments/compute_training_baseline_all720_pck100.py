#!/usr/bin/env python3
"""Evaluate all 720 Wan2.2 attention heads on the fixed 100-case PCK protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compute_training_baseline_top100_pck100 as prior


base = prior.base
gt = prior.gt
STEPS = prior.STEPS
HEIGHT = prior.HEIGHT
WIDTH = prior.WIDTH
BLOCKS = 30
HEADS_PER_BLOCK = 24
TOTAL_HEADS = BLOCKS * HEADS_PER_BLOCK


def write_page(output_root: Path) -> None:
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>100-Case All720 Mean Head PCK</title><style>
:root{--paper:#eee7d8;--ink:#17251f;--card:#fffdf8;--line:#bdb19c;--green:#176b5c;--rust:#ad452f;--gold:#bc812c;--blue:#245985}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99c5550,transparent 34rem),radial-gradient(circle at 98% 2%,#4c947653,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:10;padding:15px 24px;background:#eee7d8ef;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:3px 0;font-size:clamp(27px,4vw,46px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button{padding:8px 11px;border:1px solid var(--line);background:#fff;font-weight:900;cursor:pointer}button.active{background:var(--ink);color:#fff}.status{font:12px ui-monospace,monospace;color:#58665f}main{width:min(1900px,calc(100% - 18px));margin:auto;padding:18px 0 70px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.card,.panel{padding:12px;border:1px solid var(--line);border-radius:13px;background:var(--card)}.card b{display:block;font-size:27px;color:var(--green)}.card.best{border-top:6px solid var(--gold)}.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:13px 0}.charts img{display:block;width:100%;border:1px solid var(--line);background:#fff}.scroll{overflow:auto;border:1px solid var(--line);border-radius:13px;background:var(--card)}table{border-collapse:collapse;min-width:1180px;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;border:1px solid #d8cfbf;text-align:right}th:first-child,td:first-child{text-align:left}thead th{position:sticky;top:0;background:#19362d;color:#fff}.selected{background:#fff0c9}.best-cell{background:#dcefe5;color:#125548;font-weight:900}.yes{color:var(--rust);font-weight:900}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:13px 0}.missed{font:13px ui-monospace,monospace;line-height:1.7}@media(max-width:900px){header{position:static}.charts,.pair{grid-template-columns:1fr}}
</style></head><body><header><a href="metrics.html">Fixed Top100 指标</a> · <a href="index.html">5-Case Overlay</a> · <a href="http://localhost:8855/">总入口</a><h1>100-Case · All 720 Mean Head PCK</h1><p>Wan2.2 + OpenVid LoRA baseline · 30 blocks × 24 heads · 100 cases 等权平均 · S00/S09/S19/S29/S39</p><div class="tools"><button id="refresh">手动刷新</button><button data-filter="all" class="active">全部720</button><button data-filter="selected">原固定Top100</button><button data-filter="missed">实测Top100漏选</button><span id="status" class="status">读取中</span></div></header><main><section id="cards" class="cards"></section><section id="charts" class="charts"></section><section class="pair"><article class="panel"><h2>重合分析</h2><div id="overlap"></div></article><article class="panel"><h2>实测Top100漏选 Head</h2><div id="missed" class="missed"></div></article></section><section class="panel"><h2>固定选择与实测排序的 Mean Head PCK@32</h2><div id="means"></div></section><h2>全部 Head 排名</h2><div class="scroll"><table><thead><tr><th>实测Rank / Head</th><th>原Top100</th><th>原Rank</th><th>S00</th><th>S09</th><th>S19</th><th>S29</th><th>S39</th><th>All-Noise Mean</th><th>有效Case</th></tr></thead><tbody id="heads"></tbody></table></div></main><script>
const f=v=>v==null?'—':Number(v).toFixed(3);let DATA=null,FILTER='all';function means(rows){return `<div class="scroll"><table><thead><tr><th>Noise</th><th>实测Top30</th><th>实测Top50</th><th>实测Top100</th><th>原Top30</th><th>原Top50</th><th>原Top100</th><th>All720</th></tr></thead><tbody>${rows.map(r=>`<tr><td>S${String(r.step).padStart(2,'0')}</td>${['empirical_top30','empirical_top50','empirical_top100','original_top30','original_top50','original_top100','all720'].map(k=>`<td>${f(r[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}function renderHeads(){if(!DATA)return;let rows=DATA.per_head;if(FILTER==='selected')rows=rows.filter(r=>r.selected_top100);if(FILTER==='missed')rows=rows.filter(r=>r.empirical_rank<=100&&!r.selected_top100);document.getElementById('heads').innerHTML=rows.map(r=>`<tr class="${r.selected_top100?'selected':''}"><td>#${r.empirical_rank} · B${String(r.block).padStart(2,'0')}H${String(r.head).padStart(2,'0')}</td><td class="${r.selected_top100?'yes':''}">${r.selected_top100?'YES':'—'}</td><td>${r.selected_rank??'—'}</td>${[0,9,19,29,39].map(s=>`<td>${f(r[`s${String(s).padStart(2,'0')}`])}</td>`).join('')}<td class="${r.empirical_rank===1?'best-cell':''}">${f(r.all_noise_mean)}</td><td>${Math.min(...Object.values(r.valid_cases))}</td></tr>`).join('')}function render(d){DATA=d;const best=d.per_head[0]||{};document.getElementById('cards').innerHTML=`<article class="card"><span>Cases</span><b>${d.completed_cases}/100</b></article><article class="card"><span>Heads</span><b>${d.per_head.length}/720</b></article><article class="card best"><span>Best Head</span><b>B${String(best.block??0).padStart(2,'0')}H${String(best.head??0).padStart(2,'0')}</b></article><article class="card"><span>原Top100 ∩ 实测Top100</span><b>${d.overlap.top100}/100</b></article><article class="card"><span>原Top100 Mean</span><b>${f(d.selected_mean)}</b></article><article class="card"><span>其余620 Mean</span><b>${f(d.unselected_mean)}</b></article>`;document.getElementById('means').innerHTML=means(d.mean_head);document.getElementById('overlap').innerHTML=`实测 Top30 中原集合命中 <b>${d.overlap.top30}/30</b><br>实测 Top50 中原集合命中 <b>${d.overlap.top50}/50</b><br>实测 Top100 中原集合命中 <b>${d.overlap.top100}/100</b>`;document.getElementById('missed').textContent=d.missed_top100.map(r=>`#${r.empirical_rank} B${String(r.block).padStart(2,'0')}H${String(r.head).padStart(2,'0')} (${f(r.all_noise_mean)})`).join(' · ')||'无';document.getElementById('charts').innerHTML=d.charts?`<img src="${d.charts.curve}?v=${Date.now()}"><img src="${d.charts.heatmap}?v=${Date.now()}">`:'';renderHeads()}async function load(){try{const s=await fetch('metrics720/status.json?'+Date.now()).then(r=>r.json());document.getElementById('status').textContent=`${s.state} · ${s.message}`;const r=await fetch('metrics720/summary.json?'+Date.now());if(r.ok)render(await r.json())}catch(err){document.getElementById('status').textContent=err}}document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');FILTER=b.dataset.filter;renderHeads()});document.getElementById('refresh').onclick=load;load();
</script></body></html>'''
    (output_root / "metrics-all720.html").write_text(page, encoding="utf-8")


def make_targets(selection: Path, ranking: Path) -> list[dict]:
    selected = prior.top100(selection, ranking)
    selected_by_head = {(row["block"], row["head"]): row for row in selected}
    targets = []
    for block in range(BLOCKS):
        for head in range(HEADS_PER_BLOCK):
            old = selected_by_head.get((block, head))
            targets.append({
                "rank": block * HEADS_PER_BLOCK + head + 1,
                "block": block,
                "head": head,
                "selected_top100": old is not None,
                "selected_rank": old["rank"] if old else None,
                "ranking_pck32": old["ranking_pck32"] if old else None,
            })
    if len(targets) != TOTAL_HEADS or sum(row["selected_top100"] for row in targets) != 100:
        raise RuntimeError("Expected 720 total heads and exactly 100 selected heads")
    return targets


def prepare(args) -> None:
    dataset = args.output_root / "metrics100" / "dataset" / "manifest.json"
    if not dataset.is_file():
        raise FileNotFoundError(f"Missing completed Top100 dataset: {dataset}")
    metrics_root = args.output_root / "metrics720"
    (metrics_root / "cases").mkdir(parents=True, exist_ok=True)
    (metrics_root / "logs").mkdir(parents=True, exist_ok=True)
    write_page(args.output_root)
    base.atomic_json(metrics_root / "status.json", {
        "state": "prepared",
        "message": "All720 analysis prepared; reusing completed Top100 rows and SAM2 cache",
    })


def reusable_top100_rows(args, case_key: str, target_by_head: dict) -> list[dict]:
    old_root = args.output_root / "metrics100" / "cases" / case_key
    rows = prior.load_saved_rows(old_root)
    converted = []
    for row in rows:
        target = target_by_head.get((row["block"], row["head"]))
        if target is None or not target["selected_top100"]:
            continue
        converted.append({**row, "rank": target["rank"]})
    return converted


def worker(args) -> None:
    targets = make_targets(args.selection, args.ranking)
    target_by_head = {(row["block"], row["head"]): row for row in targets}
    capture_targets = [row for row in targets if not row["selected_top100"]]
    _, pipe, _ = base.build_training_baseline(args.baseline_config, args.device)
    cotracker = gt.load_cotracker(args.device)
    cases = base.load_case_manifests(args.output_root / "metrics100" / "dataset")
    assigned = [(index, case) for index, case in enumerate(cases) if index % args.num_workers == args.worker_id]
    status_path = args.output_root / "metrics720" / f"status_worker_{args.worker_id}.json"
    done = 0
    for local_index, (global_index, case) in enumerate(assigned, start=1):
        case_key, case_base = case["case_key"], case["base"]
        case_root = args.output_root / "metrics720" / "cases" / case_key
        if (case_root / "complete").is_file():
            done += 1
            continue
        base.atomic_json(status_path, {
            "state": "computing", "gpu": args.gpu_label,
            "message": f"case {local_index}/{len(assigned)} (global {global_index + 1}/100): {case_key}",
        })
        reused_rows = reusable_top100_rows(args, case_key, target_by_head)
        case_capture_targets = capture_targets if len(reused_rows) == len(STEPS) * 100 else targets
        regions, points, _ = base.load_regions(args.cache_root / case_key)
        frames = gt.load_video_prefix(Path(case_base["source_video"]), 49, HEIGHT, WIDTH, "cache")
        context = gt.load_video_prefix(Path(case_base["video"]), 8, HEIGHT, WIDTH, "cache")
        latents = gt.encode_gt_video(pipe, frames, "whole_video")
        shared, positive = gt.prepare_conditioning(
            pipe, prompt=case_base["caption"], context_video=context,
            height=HEIGHT, width=WIDTH, num_frames=49, sampling_steps=40,
            sigma_shift=5.0, cfg_scale=5.0, seed=42,
        )
        prefix = gt.validate_geometry(argparse.Namespace(height=HEIGHT, width=WIDTH), latents, shared, 49)
        capture = base.TopHeadCapture(pipe, case_capture_targets, points, STEPS)
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        capture.install()
        try:
            with torch.inference_mode():
                for step in STEPS:
                    timestep = pipe.scheduler.timesteps[step].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                    noised = latents.clone()
                    noised[:, :, prefix:] = pipe.scheduler.add_noise(
                        latents[:, :, prefix:], shared["noise"][:, :, prefix:], timestep,
                    )
                    noised[:, :, :prefix] = shared["clean_prefix_latents"]
                    shared["latents"] = noised
                    pipe.model_fn(**models, **shared, **positive, timestep=timestep)
        finally:
            capture.remove()
        expected = len(STEPS) * len(case_capture_targets)
        if len(capture.records) != expected:
            raise RuntimeError(f"Captured {len(capture.records)}/{expected} records for {case_key}")
        frame_array = np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames])
        tracks, visibility = gt.run_cotracker(cotracker, frame_array, points, 4, args.device)
        case_rows = reused_rows if len(reused_rows) == len(STEPS) * 100 else []
        for step in STEPS:
            for target in case_capture_targets:
                pck32, valid_objects, comparisons = prior.case_pck(
                    capture.records[(step, target["block"], target["head"])],
                    regions, tracks, visibility,
                )
                if pck32 is None:
                    continue
                case_rows.append({
                    "case_key": case_key, "family": case.get("family", "PyBullet"),
                    "step": step, "rank": target["rank"], "block": target["block"],
                    "head": target["head"], "pck32": pck32,
                    "valid_objects": valid_objects, "comparisons": comparisons,
                })
        prior.write_csv(case_root / "pck_rows.csv", case_rows, [
            "case_key", "family", "step", "rank", "block", "head", "pck32",
            "valid_objects", "comparisons",
        ])
        (case_root / "complete").write_text("complete\n", encoding="utf-8")
        done += 1
        base.atomic_json(status_path, {
            "state": "computing", "gpu": args.gpu_label,
            "message": f"completed {done}/{len(assigned)} assigned cases",
        })
        del capture, latents, shared, positive
        torch.cuda.empty_cache()
    base.atomic_json(status_path, {
        "state": "complete", "gpu": args.gpu_label,
        "message": f"completed {done}/{len(assigned)} assigned cases",
    })


def read_all_rows(metrics_root: Path) -> tuple[list[dict], int]:
    all_rows = []
    completed = 0
    for marker in sorted((metrics_root / "cases").glob("*/complete")):
        rows = prior.load_saved_rows(marker.parent)
        completed += 1
        if rows:
            all_rows.extend(rows)
    return all_rows, completed


def safe_mean(values) -> float | None:
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def aggregate(args) -> None:
    metrics_root = args.output_root / "metrics720"
    targets = make_targets(args.selection, args.ranking)
    all_rows, completed = read_all_rows(metrics_root)
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["step"], row["block"], row["head"])].append(row["pck32"])
    per_head = []
    per_noise = []
    for target in targets:
        result = {**target, "valid_cases": {}}
        noise_means = []
        for step in STEPS:
            values = grouped[(step, target["block"], target["head"])]
            mean = safe_mean(values)
            key = f"s{step:02d}"
            result[key] = mean
            result["valid_cases"][key] = len(values)
            if mean is not None:
                noise_means.append(mean)
            per_noise.append({
                "step": step, "block": target["block"], "head": target["head"],
                "selected_top100": target["selected_top100"],
                "selected_rank": target["selected_rank"], "mean_pck32": mean,
                "std_pck32": float(np.std(values)) if values else None,
                "valid_cases": len(values),
            })
        result["all_noise_mean"] = safe_mean(noise_means)
        per_head.append(result)
    per_head.sort(key=lambda row: (row["all_noise_mean"] is None, -(row["all_noise_mean"] or -1.0)))
    for rank, row in enumerate(per_head, start=1):
        row["empirical_rank"] = rank
    for step in STEPS:
        ranked_step = sorted(
            [row for row in per_noise if row["step"] == step],
            key=lambda row: (row["mean_pck32"] is None, -(row["mean_pck32"] or -1.0)),
        )
        for rank, row in enumerate(ranked_step, start=1):
            row["noise_rank"] = rank
    selected_set = {(row["block"], row["head"]) for row in per_head if row["selected_top100"]}
    overlap = {}
    for count in (30, 50, 100):
        empirical = {(row["block"], row["head"]) for row in per_head[:count]}
        overlap[f"top{count}"] = len(empirical & selected_set)
    missed = [row for row in per_head[:100] if not row["selected_top100"]]
    displaced = [row for row in per_head[100:] if row["selected_top100"]]
    mean_head = []
    for step in STEPS:
        key = f"s{step:02d}"
        row = {"step": step}
        for count in (30, 50, 100):
            row[f"empirical_top{count}"] = safe_mean(item[key] for item in per_head[:count])
            original = [item[key] for item in per_head if item["selected_rank"] is not None and item["selected_rank"] <= count]
            row[f"original_top{count}"] = safe_mean(original)
        row["all720"] = safe_mean(item[key] for item in per_head)
        mean_head.append(row)
    charts = None
    if completed:
        x = np.arange(1, TOTAL_HEADS + 1)
        y = np.asarray([row["all_noise_mean"] for row in per_head], dtype=float)
        selected_mask = np.asarray([row["selected_top100"] for row in per_head], dtype=bool)
        figure, axis = plt.subplots(figsize=(10.5, 5.2))
        axis.plot(x, y, color="#176b5c", linewidth=1.8, label="All heads by measured PCK rank")
        axis.scatter(x[selected_mask], y[selected_mask], s=18, color="#ad452f", alpha=.8, label="Original fixed Top100")
        axis.axvline(100, color="#bc812c", linestyle="--", linewidth=1.3)
        axis.set(xlabel="Measured all-noise PCK rank", ylabel="100-case Mean Head PCK@32 (%)", title="All720 ranking and original Top100 coverage")
        axis.grid(alpha=.22); axis.legend(); figure.tight_layout()
        curve = metrics_root / "all720_rank_curve.png"; figure.savefig(curve, dpi=180); plt.close(figure)
        matrix = np.asarray([[row[f"s{step:02d}"] for step in STEPS] for row in per_head], dtype=float)
        figure, axis = plt.subplots(figsize=(8.6, 12.5))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(STEPS)), [f"S{step:02d}" for step in STEPS])
        axis.set(xlabel="Noise level", ylabel="Measured all-noise rank", title="All720 per-head PCK@32")
        figure.colorbar(image, ax=axis, label="PCK@32 (%)"); figure.tight_layout()
        heatmap = metrics_root / "all720_noise_heatmap.png"; figure.savefig(heatmap, dpi=180); plt.close(figure)
        charts = {
            "curve": str(curve.relative_to(args.output_root)),
            "heatmap": str(heatmap.relative_to(args.output_root)),
        }
    prior.write_csv(metrics_root / "all720_head_ranking.csv", per_head, [
        "empirical_rank", "rank", "block", "head", "selected_top100", "selected_rank",
        "ranking_pck32", "s00", "s09", "s19", "s29", "s39", "all_noise_mean",
        "valid_cases",
    ])
    prior.write_csv(metrics_root / "all720_per_noise_ranking.csv", per_noise, [
        "step", "noise_rank", "block", "head", "selected_top100", "selected_rank",
        "mean_pck32", "std_pck32", "valid_cases",
    ])
    prior.write_csv(metrics_root / "mean_head_pck.csv", mean_head, [
        "step", "empirical_top30", "empirical_top50", "empirical_top100",
        "original_top30", "original_top50", "original_top100", "all720",
    ])
    if args.final:
        prior.write_csv(metrics_root / "raw_case_head_pck.csv", all_rows, [
            "case_key", "family", "step", "rank", "block", "head", "pck32",
            "valid_objects", "comparisons",
        ])
    summary = {
        "completed_cases": completed, "expected_cases": 100,
        "model": "Wan2.2 + OpenVid LoRA training baseline; no trained Top100 modules",
        "protocol": "case-macro PCK@32; Mean Head PCK; S00/S09/S19/S29/S39",
        "overlap": overlap,
        "selected_mean": safe_mean(row["all_noise_mean"] for row in per_head if row["selected_top100"]),
        "unselected_mean": safe_mean(row["all_noise_mean"] for row in per_head if not row["selected_top100"]),
        "mean_head": mean_head, "per_head": per_head,
        "missed_top100": missed, "selected_outside_empirical_top100": displaced,
        "charts": charts,
    }
    base.atomic_json(metrics_root / "summary.json", summary)
    state = "complete" if completed == 100 else "computing"
    base.atomic_json(metrics_root / "status.json", {
        "state": state,
        "message": f"{completed}/100 cases; {TOTAL_HEADS} heads × {len(STEPS)} noise levels",
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "worker", "aggregate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=base.DEFAULT_SELECTION)
    parser.add_argument("--ranking", type=Path, default=base.DEFAULT_RANKING)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-label", default="GPU")
    parser.add_argument("--final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "aggregate":
        aggregate(args)
    elif args.cache_root is None or args.baseline_config is None:
        raise ValueError("worker requires --cache-root and --baseline-config")
    else:
        worker(args)


if __name__ == "__main__":
    main()
