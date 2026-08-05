#!/usr/bin/env python3
"""All720 Mean Head PCK on 100 cases with a shared set of 50 random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import random
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

import compute_training_baseline_all720_pck100 as allh


prior = allh.prior
base = allh.base
gt = allh.gt
STEPS = allh.STEPS
HEIGHT = allh.HEIGHT
WIDTH = allh.WIDTH
TOTAL_HEADS = allh.TOTAL_HEADS
SEED_COUNT = 50
SEED_SAMPLE_STATE = 20260805


def write_page(output_root: Path) -> None:
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All720 · 100 Cases × 50 Seeds</title><style>
:root{--paper:#eee7d8;--ink:#17251f;--card:#fffdf8;--line:#bdb19c;--green:#176b5c;--rust:#ad452f;--gold:#bc812c}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99c5550,transparent 34rem),radial-gradient(circle at 98% 2%,#4c947653,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:10;padding:15px 24px;background:#eee7d8ef;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:3px 0;font-size:clamp(27px,4vw,46px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button{padding:8px 11px;border:1px solid var(--line);background:#fff;font-weight:900;cursor:pointer}button.active{background:var(--ink);color:#fff}.status{font:12px ui-monospace,monospace;color:#58665f}main{width:min(1900px,calc(100% - 18px));margin:auto;padding:18px 0 70px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.card,.panel{padding:12px;border:1px solid var(--line);border-radius:13px;background:var(--card)}.card b{display:block;font-size:27px;color:var(--green)}.card.best{border-top:6px solid var(--gold)}.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:13px 0}.charts img{display:block;width:100%;border:1px solid var(--line);background:#fff}.scroll{overflow:auto;border:1px solid var(--line);border-radius:13px;background:var(--card)}table{border-collapse:collapse;min-width:1350px;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;border:1px solid #d8cfbf;text-align:right}th:first-child,td:first-child{text-align:left}thead th{position:sticky;top:0;background:#19362d;color:#fff}.selected{background:#fff0c9}.yes{color:var(--rust);font-weight:900}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:13px 0}.mono{font:12px ui-monospace,monospace;line-height:1.65;overflow-wrap:anywhere}@media(max-width:900px){header{position:static}.charts,.pair{grid-template-columns:1fr}}
</style></head><body><header><a href="metrics-all720.html">单Seed All720</a> · <a href="metrics.html">Fixed Top100</a> · <a href="index.html">Overlay</a><h1>All720 · 100 Cases × 50 Seeds</h1><p>每 case 内先跨 50 seeds 平均，再跨 100 cases 等权平均 · PCK@32 · S00/S09/S19/S29/S39</p><div class="tools"><button id="refresh">手动刷新</button><button data-filter="all" class="active">全部720</button><button data-filter="selected">原固定Top100</button><button data-filter="missed">实测Top100漏选</button><span id="status" class="status">读取中</span></div></header><main><section id="cards" class="cards"></section><section id="charts" class="charts"></section><section class="pair"><article class="panel"><h2>原 Top100 重合</h2><div id="overlap"></div></article><article class="panel"><h2>固定随机 Seeds</h2><div id="seeds" class="mono"></div></article></section><section class="panel"><h2>Mean Head PCK@32</h2><div id="means"></div></section><h2>全部 Head 排名</h2><div class="scroll"><table><thead><tr><th>实测Rank / Head</th><th>原Top100</th><th>原Rank</th><th>S00</th><th>S09</th><th>S19</th><th>S29</th><th>S39</th><th>All-Noise Mean</th><th>95% CI</th><th>有效Case</th></tr></thead><tbody id="heads"></tbody></table></div></main><script>
const f=v=>v==null?'—':Number(v).toFixed(3);let DATA=null,FILTER='all';function meanTable(rows){return `<div class="scroll"><table><thead><tr><th>Noise</th><th>实测Top30</th><th>实测Top50</th><th>实测Top100</th><th>原Top30</th><th>原Top50</th><th>原Top100</th><th>All720</th></tr></thead><tbody>${rows.map(r=>`<tr><td>S${String(r.step).padStart(2,'0')}</td>${['empirical_top30','empirical_top50','empirical_top100','original_top30','original_top50','original_top100','all720'].map(k=>`<td>${f(r[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}function renderHeads(){if(!DATA)return;let rows=DATA.per_head;if(FILTER==='selected')rows=rows.filter(r=>r.selected_top100);if(FILTER==='missed')rows=rows.filter(r=>r.empirical_rank<=100&&!r.selected_top100);document.getElementById('heads').innerHTML=rows.map(r=>`<tr class="${r.selected_top100?'selected':''}"><td>#${r.empirical_rank} · B${String(r.block).padStart(2,'0')}H${String(r.head).padStart(2,'0')}</td><td class="${r.selected_top100?'yes':''}">${r.selected_top100?'YES':'—'}</td><td>${r.selected_rank??'—'}</td>${[0,9,19,29,39].map(s=>`<td>${f(r[`s${String(s).padStart(2,'0')}`])}</td>`).join('')}<td>${f(r.all_noise_mean)}</td><td>[${f(r.all_noise_ci95_low)}, ${f(r.all_noise_ci95_high)}]</td><td>${r.valid_cases}</td></tr>`).join('')}function render(d){DATA=d;const best=d.per_head[0]||{};document.getElementById('cards').innerHTML=`<article class="card"><span>完整Cases</span><b>${d.completed_cases}/100</b></article><article class="card"><span>Seed Runs</span><b>${d.completed_seed_runs}/5000</b></article><article class="card best"><span>当前Best Head</span><b>B${String(best.block??0).padStart(2,'0')}H${String(best.head??0).padStart(2,'0')}</b></article><article class="card"><span>原Top100 ∩ 实测Top100</span><b>${d.overlap.top100}/100</b></article><article class="card"><span>原Top100 Mean</span><b>${f(d.selected_mean)}</b></article><article class="card"><span>其余620 Mean</span><b>${f(d.unselected_mean)}</b></article>`;document.getElementById('overlap').innerHTML=`Top30: <b>${d.overlap.top30}/30</b><br>Top50: <b>${d.overlap.top50}/50</b><br>Top100: <b>${d.overlap.top100}/100</b>`;document.getElementById('seeds').textContent=d.seeds.join(', ');document.getElementById('means').innerHTML=meanTable(d.mean_head);document.getElementById('charts').innerHTML=d.charts?`<img src="${d.charts.curve}?v=${Date.now()}"><img src="${d.charts.heatmap}?v=${Date.now()}">`:'';renderHeads()}async function load(){try{const s=await fetch('metrics720_seed50/status.json?'+Date.now()).then(r=>r.json());document.getElementById('status').textContent=`${s.state} · ${s.message}`;const r=await fetch('metrics720_seed50/summary.json?'+Date.now());if(r.ok)render(await r.json())}catch(err){document.getElementById('status').textContent=err}}document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');FILTER=b.dataset.filter;renderHeads()});document.getElementById('refresh').onclick=load;load();
</script></body></html>'''
    (output_root / "metrics-all720-seed50.html").write_text(page, encoding="utf-8")


def metrics_root(args) -> Path:
    return args.output_root / "metrics720_seed50"


def load_seeds(root: Path) -> list[int]:
    return [int(seed) for seed in json.loads((root / "seed_manifest.json").read_text(encoding="utf-8"))["seeds"]]


def prepare(args) -> None:
    root = metrics_root(args)
    (root / "cases").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    manifest = root / "seed_manifest.json"
    if not manifest.is_file():
        seeds = random.Random(SEED_SAMPLE_STATE).sample(range(100001), SEED_COUNT)
        base.atomic_json(manifest, {
            "sample_state": SEED_SAMPLE_STATE,
            "range_inclusive": [0, 100000],
            "shared_across_cases": True,
            "count": SEED_COUNT,
            "seeds": seeds,
        })
    write_page(args.output_root)
    base.atomic_json(root / "status.json", {
        "state": "prepared", "message": "100 cases × 50 shared random seeds × 720 heads prepared",
    })


def case_seed_rows(seed_root: Path) -> list[dict]:
    return prior.load_saved_rows(seed_root)


def aggregate_case(case_root: Path, case: dict, targets: list[dict], seeds: list[int]) -> None:
    grouped = defaultdict(list)
    for seed in seeds:
        for row in case_seed_rows(case_root / f"seed_{seed:06d}"):
            grouped[(row["step"], row["block"], row["head"])].append(row["pck32"])
    rows = []
    for target in targets:
        for step in STEPS:
            values = grouped[(step, target["block"], target["head"])]
            rows.append({
                "case_key": case["case_key"], "family": case.get("family", "PyBullet"),
                "step": step, "rank": target["rank"], "block": target["block"],
                "head": target["head"],
                "mean_pck32": float(np.mean(values)) if values else None,
                "seed_std_pck32": float(np.std(values)) if values else None,
                "valid_seeds": len(values),
            })
    prior.write_csv(case_root / "case_head_noise_mean.csv", rows, [
        "case_key", "family", "step", "rank", "block", "head",
        "mean_pck32", "seed_std_pck32", "valid_seeds",
    ])
    (case_root / "complete").write_text("complete\n", encoding="utf-8")


def read_case_summaries(root: Path) -> tuple[list[dict], int]:
    rows = []
    markers = sorted((root / "cases").glob("*/complete"))
    for marker in markers:
        path = marker.parent / "case_head_noise_mean.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row["mean_pck32"]:
                    continue
                rows.append({
                    "case_key": row["case_key"], "step": int(row["step"]),
                    "block": int(row["block"]), "head": int(row["head"]),
                    "mean_pck32": float(row["mean_pck32"]),
                    "seed_std_pck32": float(row["seed_std_pck32"]),
                    "valid_seeds": int(row["valid_seeds"]),
                })
    return rows, len(markers)


def safe_mean(values) -> float | None:
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def mean_ci(values) -> tuple[float | None, float | None, float | None, float | None]:
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(values):
        return None, None, None, None
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = 1.96 * std / np.sqrt(len(values))
    return mean, std, mean - half, mean + half


def aggregate_global(args, final: bool = False) -> None:
    root = metrics_root(args)
    targets = allh.make_targets(args.selection, args.ranking)
    case_rows, completed_cases = read_case_summaries(root)
    grouped = defaultdict(list)
    seed_std_grouped = defaultdict(list)
    by_case_head = defaultdict(list)
    for row in case_rows:
        key = (row["step"], row["block"], row["head"])
        grouped[key].append(row["mean_pck32"])
        seed_std_grouped[key].append(row["seed_std_pck32"])
        by_case_head[(row["case_key"], row["block"], row["head"])].append(row["mean_pck32"])
    per_head = []
    per_noise = []
    for target in targets:
        result = {**target, "valid_cases_by_step": {}}
        for step in STEPS:
            key = (step, target["block"], target["head"])
            mean, std, low, high = mean_ci(grouped[key])
            label = f"s{step:02d}"
            result[label] = mean
            result[f"{label}_std"] = std
            result[f"{label}_ci95_low"] = low
            result[f"{label}_ci95_high"] = high
            result["valid_cases_by_step"][label] = len(grouped[key])
            per_noise.append({
                "step": step, "block": target["block"], "head": target["head"],
                "selected_top100": target["selected_top100"], "selected_rank": target["selected_rank"],
                "mean_pck32": mean, "case_std_pck32": std, "ci95_low": low, "ci95_high": high,
                "mean_within_case_seed_std": safe_mean(seed_std_grouped[key]),
                "valid_cases": len(grouped[key]),
            })
        case_all_noise = [
            safe_mean(values) for (case_key, block, head), values in by_case_head.items()
            if block == target["block"] and head == target["head"]
        ]
        mean, std, low, high = mean_ci(case_all_noise)
        result["all_noise_mean"] = mean
        result["all_noise_std"] = std
        result["all_noise_ci95_low"] = low
        result["all_noise_ci95_high"] = high
        result["valid_cases"] = len([value for value in case_all_noise if value is not None])
        per_head.append(result)
    per_head.sort(key=lambda row: (row["all_noise_mean"] is None, -(row["all_noise_mean"] or -1.0)))
    for rank, row in enumerate(per_head, start=1):
        row["empirical_rank"] = rank
    for step in STEPS:
        ranked = sorted(
            [row for row in per_noise if row["step"] == step],
            key=lambda row: (row["mean_pck32"] is None, -(row["mean_pck32"] or -1.0)),
        )
        for rank, row in enumerate(ranked, start=1):
            row["noise_rank"] = rank
    selected_set = {(row["block"], row["head"]) for row in per_head if row["selected_top100"]}
    overlap = {}
    for count in (30, 50, 100):
        measured = {(row["block"], row["head"]) for row in per_head[:count]}
        overlap[f"top{count}"] = len(measured & selected_set)
    mean_head = []
    for step in STEPS:
        key = f"s{step:02d}"
        row = {"step": step}
        for count in (30, 50, 100):
            row[f"empirical_top{count}"] = safe_mean(item[key] for item in per_head[:count])
            row[f"original_top{count}"] = safe_mean(
                item[key] for item in per_head
                if item["selected_rank"] is not None and item["selected_rank"] <= count
            )
        row["all720"] = safe_mean(item[key] for item in per_head)
        mean_head.append(row)
    completed_seed_runs = sum(1 for _ in (root / "cases").glob("*/seed_*/complete"))
    charts = None
    if final and completed_cases:
        x = np.arange(1, TOTAL_HEADS + 1)
        y = np.asarray([row["all_noise_mean"] for row in per_head], dtype=float)
        selected = np.asarray([row["selected_top100"] for row in per_head], dtype=bool)
        figure, axis = plt.subplots(figsize=(10.5, 5.2))
        axis.plot(x, y, color="#176b5c", linewidth=1.8, label="All720 measured rank")
        axis.scatter(x[selected], y[selected], s=18, color="#ad452f", alpha=.8, label="Original fixed Top100")
        axis.axvline(100, color="#bc812c", linestyle="--", linewidth=1.3)
        axis.set(xlabel="Measured all-noise rank", ylabel="100-case × 50-seed Mean Head PCK@32 (%)", title="All720 seed-robust ranking")
        axis.grid(alpha=.22); axis.legend(); figure.tight_layout()
        curve = root / "all720_seed50_rank_curve.png"; figure.savefig(curve, dpi=180); plt.close(figure)
        matrix = np.asarray([[row[f"s{step:02d}"] for step in STEPS] for row in per_head], dtype=float)
        figure, axis = plt.subplots(figsize=(8.6, 12.5))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(STEPS)), [f"S{step:02d}" for step in STEPS])
        axis.set(xlabel="Noise level", ylabel="Measured all-noise rank", title="All720 PCK across 50 seeds")
        figure.colorbar(image, ax=axis, label="PCK@32 (%)"); figure.tight_layout()
        heatmap = root / "all720_seed50_heatmap.png"; figure.savefig(heatmap, dpi=180); plt.close(figure)
        charts = {"curve": str(curve.relative_to(args.output_root)), "heatmap": str(heatmap.relative_to(args.output_root))}
    prior.write_csv(root / "all720_head_ranking.csv", per_head, [
        "empirical_rank", "rank", "block", "head", "selected_top100", "selected_rank", "ranking_pck32",
        "s00", "s00_std", "s00_ci95_low", "s00_ci95_high",
        "s09", "s09_std", "s09_ci95_low", "s09_ci95_high",
        "s19", "s19_std", "s19_ci95_low", "s19_ci95_high",
        "s29", "s29_std", "s29_ci95_low", "s29_ci95_high",
        "s39", "s39_std", "s39_ci95_low", "s39_ci95_high",
        "all_noise_mean", "all_noise_std", "all_noise_ci95_low", "all_noise_ci95_high",
        "valid_cases", "valid_cases_by_step",
    ])
    prior.write_csv(root / "all720_per_noise_ranking.csv", per_noise, [
        "step", "noise_rank", "block", "head", "selected_top100", "selected_rank",
        "mean_pck32", "case_std_pck32", "ci95_low", "ci95_high",
        "mean_within_case_seed_std", "valid_cases",
    ])
    prior.write_csv(root / "mean_head_pck.csv", mean_head, [
        "step", "empirical_top30", "empirical_top50", "empirical_top100",
        "original_top30", "original_top50", "original_top100", "all720",
    ])
    seeds = load_seeds(root)
    base.atomic_json(root / "summary.json", {
        "completed_cases": completed_cases, "expected_cases": 100,
        "completed_seed_runs": completed_seed_runs, "expected_seed_runs": 5000,
        "seeds": seeds, "steps": list(STEPS), "overlap": overlap,
        "selected_mean": safe_mean(row["all_noise_mean"] for row in per_head if row["selected_top100"]),
        "unselected_mean": safe_mean(row["all_noise_mean"] for row in per_head if not row["selected_top100"]),
        "mean_head": mean_head, "per_head": per_head,
        "missed_top100": [row for row in per_head[:100] if not row["selected_top100"]],
        "selected_outside_empirical_top100": [row for row in per_head[100:] if row["selected_top100"]],
        "charts": charts,
    })


def worker(args) -> None:
    root = metrics_root(args)
    seeds = load_seeds(root)
    targets = allh.make_targets(args.selection, args.ranking)
    _, pipe, _ = base.build_training_baseline(args.baseline_config, args.device)
    cotracker = gt.load_cotracker(args.device)
    cases = base.load_case_manifests(args.output_root / "metrics100" / "dataset")
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    completed_runs = sum(1 for _ in (root / "cases").glob("*/seed_*/complete"))
    for case_index, case in enumerate(cases, start=1):
        case_key, case_base = case["case_key"], case["base"]
        case_root = root / "cases" / case_key
        if (case_root / "complete").is_file():
            continue
        regions, points, _ = base.load_regions(args.cache_root / case_key)
        frames = gt.load_video_prefix(Path(case_base["source_video"]), 49, HEIGHT, WIDTH, "cache")
        context = gt.load_video_prefix(Path(case_base["video"]), 8, HEIGHT, WIDTH, "cache")
        latents = gt.encode_gt_video(pipe, frames, "whole_video")
        frame_array = np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames])
        tracks, visibility = gt.run_cotracker(cotracker, frame_array, points, 4, args.device)
        for seed_index, seed in enumerate(seeds, start=1):
            seed_root = case_root / f"seed_{seed:06d}"
            if (seed_root / "complete").is_file():
                continue
            base.atomic_json(root / "status.json", {
                "state": "computing", "gpu": "GPU5",
                "message": f"case {case_index}/100 · seed {seed_index}/50 ({seed}) · completed {completed_runs}/5000",
            })
            shared, positive = gt.prepare_conditioning(
                pipe, prompt=case_base["caption"], context_video=context,
                height=HEIGHT, width=WIDTH, num_frames=49, sampling_steps=40,
                sigma_shift=5.0, cfg_scale=5.0, seed=seed,
            )
            prefix = gt.validate_geometry(argparse.Namespace(height=HEIGHT, width=WIDTH), latents, shared, 49)
            capture = base.TopHeadCapture(pipe, targets, points, STEPS)
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
            expected = len(STEPS) * TOTAL_HEADS
            if len(capture.records) != expected:
                raise RuntimeError(f"Captured {len(capture.records)}/{expected} records for {case_key}, seed {seed}")
            rows = []
            for step in STEPS:
                for target in targets:
                    pck32, valid_objects, comparisons = prior.case_pck(
                        capture.records[(step, target["block"], target["head"])],
                        regions, tracks, visibility,
                    )
                    if pck32 is None:
                        continue
                    rows.append({
                        "case_key": case_key, "family": case.get("family", "PyBullet"),
                        "step": step, "rank": target["rank"], "block": target["block"],
                        "head": target["head"], "pck32": pck32,
                        "valid_objects": valid_objects, "comparisons": comparisons,
                    })
            prior.write_csv(seed_root / "pck_rows.csv", rows, [
                "case_key", "family", "step", "rank", "block", "head", "pck32",
                "valid_objects", "comparisons",
            ])
            (seed_root / "complete").write_text("complete\n", encoding="utf-8")
            completed_runs += 1
            del capture, shared, positive
            torch.cuda.empty_cache()
        aggregate_case(case_root, case, targets, seeds)
        aggregate_global(args, final=False)
        del latents
        torch.cuda.empty_cache()
    aggregate_global(args, final=True)
    base.atomic_json(root / "status.json", {
        "state": "complete", "gpu": "GPU5",
        "message": "100/100 cases · 5000/5000 seed runs · All720 ranking complete",
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
    parser.add_argument("--final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "aggregate":
        aggregate_global(args, final=args.final)
    elif args.cache_root is None or args.baseline_config is None:
        raise ValueError("worker requires --cache-root and --baseline-config")
    else:
        worker(args)


if __name__ == "__main__":
    main()
