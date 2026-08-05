#!/usr/bin/env python3
"""Compute per-head case-macro PCK@32 for fixed Top100 on 100 GT cases."""

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
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFTRACK = Path("/home/gaoya/Code_Video/DiffTrack-main")
for path in (HERE, CODE_ROOT, DIFFTRACK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import capture_training_object_query_top30_step500 as base
from AAA_my_test import analyze_wan_gt_toy_worker as gt


STEPS = base.STEPS
HEIGHT = base.HEIGHT
WIDTH = base.WIDTH
ANCHORS = np.arange(base.LATENT_FRAMES, dtype=np.int64) * 4


def write_metrics_page(root: Path) -> None:
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>100-Case Top100 Mean Head PCK</title><style>
:root{--paper:#eee7d8;--ink:#17251f;--card:#fffdf8;--line:#bdb19c;--green:#176b5c;--rust:#ad452f;--gold:#bc812c}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99c5550,transparent 34rem),radial-gradient(circle at 98% 2%,#4c947653,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:10;padding:16px 24px;background:#eee7d8ef;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:3px 0;font-size:clamp(27px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button{padding:8px 11px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace;color:#58665f}main{width:min(1900px,calc(100% - 18px));margin:auto;padding:18px 0 70px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}.card,.panel{padding:12px;border:1px solid var(--line);border-radius:13px;background:var(--card)}.card b{display:block;font-size:28px;color:var(--green)}.card.best{border-top:6px solid var(--gold)}.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:13px 0}.charts img{display:block;width:100%;border:1px solid var(--line);background:#fff}.scroll{overflow:auto;border:1px solid var(--line);border-radius:13px;background:var(--card)}table{border-collapse:collapse;min-width:1150px;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;border:1px solid #d8cfbf;text-align:right}th:first-child,td:first-child{text-align:left}thead th{position:sticky;top:0;background:#19362d;color:#fff}.best-cell{background:#dcefe5;color:#125548;font-weight:900}.pending{padding:60px;border:1px dashed var(--line);border-radius:12px;background:var(--card)}@media(max-width:900px){header{position:static}.charts{grid-template-columns:1fr}}
</style></head><body><header><a href="index.html">返回5-Case Overlay</a> · <a href="http://localhost:8855/">总入口</a><h1>100-Case · Fixed Top100 Mean Head PCK</h1><p>Wan2.2 + OpenVid LoRA baseline · 无Top100训练模块 · 每Head先跨case平均，再计算Top30/50/100均值</p><div class="tools"><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="cards" class="cards"></section><section id="charts" class="charts"></section><section class="panel"><h2>Top30 / 50 / 100 Mean Head PCK@32</h2><div id="means"></div></section><h2>每个Head跨所有已完成Case的平均PCK@32</h2><div class="scroll"><table><thead><tr><th>Rank / Head</th><th>S00</th><th>S09</th><th>S19</th><th>S29</th><th>S39</th><th>All-Noise Mean</th><th>有效Case</th></tr></thead><tbody id="heads"></tbody></table></div></main><script>
const f=v=>v==null?'—':Number(v).toFixed(3),e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function table(rows){return `<div class="scroll"><table><thead><tr><th>Noise</th><th>Top30</th><th>Top50</th><th>Top100</th></tr></thead><tbody>${rows.map(r=>`<tr><td>S${String(r.step).padStart(2,'0')}</td>${[30,50,100].map(n=>`<td>${f(r[`top${n}`])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}function render(d){document.getElementById('cards').innerHTML=`<article class="card"><span>Cases</span><b>${d.completed_cases}/100</b></article><article class="card"><span>Head × Noise rows</span><b>${d.per_head.length}/500</b></article><article class="card"><span>Protocol</span><b>PCK@32</b></article>`;document.getElementById('means').innerHTML=table(d.mean_head);document.getElementById('charts').innerHTML=d.charts?`<img src="${d.charts.curve}?v=${Date.now()}"><img src="${d.charts.heatmap}?v=${Date.now()}">`:'';const best={};for(const s of [0,9,19,29,39])best[s]=Math.max(...d.per_head.map(r=>r[`s${String(s).padStart(2,'0')}`]).filter(Number.isFinite));document.getElementById('heads').innerHTML=d.per_head.map(r=>`<tr><td>#${r.rank} · B${String(r.block).padStart(2,'0')}H${String(r.head).padStart(2,'0')}</td>${[0,9,19,29,39].map(s=>{const v=r[`s${String(s).padStart(2,'0')}`];return `<td class="${v===best[s]?'best-cell':''}">${f(v)}</td>`}).join('')}<td>${f(r.all_noise_mean)}</td><td>${Math.min(...Object.values(r.valid_cases))}</td></tr>`).join('')}async function load(){try{const s=await fetch('metrics_status.json?'+Date.now()).then(r=>r.json());document.getElementById('status').textContent=`${s.state} · ${s.message}`;const r=await fetch('metrics100/summary.json?'+Date.now());if(r.ok)render(await r.json());else document.getElementById('heads').innerHTML='<tr><td colspan="8">等待首个case完成</td></tr>'}catch(err){document.getElementById('status').textContent=err}}document.getElementById('refresh').onclick=load;load();
</script></body></html>'''
    (root / "metrics.html").write_text(page, encoding="utf-8")


def prepare(args) -> None:
    rows = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    selected = []
    for family in ("F1", "F2", "F3", "F4", "F5"):
        family_rows = [
            row for row in rows
            if row.get("family_key") == family
            and Path(row["video"]).is_file()
            and row.get("caption")
            and row.get("object_phrases")
        ][:20]
        if len(family_rows) != 20:
            raise RuntimeError(f"Expected 20 valid {family} cases, found {len(family_rows)}")
        selected.extend(family_rows)
    dataset = args.output_root / "metrics100" / "dataset"
    for index, row in enumerate(selected):
        case_key = f"pck100_{index:03d}_{row['case_id']}"
        case_dir = dataset / "cases" / f"case_{case_key}"
        case_dir.mkdir(parents=True, exist_ok=True)
        base.atomic_json(case_dir / "case_manifest.json", {
            "case_key": case_key,
            "family": row["family_key"],
            "object_count": len(row["object_phrases"]),
            "base": {
                "video": row["video"], "source_video": row["video"],
                "caption": row["caption"], "object_phrases": row["object_phrases"],
            },
        })
    base.atomic_json(dataset / "manifest.json", {
        "source": str(args.training_manifest), "case_count": 100,
        "family_counts": {family: 20 for family in ("F1", "F2", "F3", "F4", "F5")},
    })
    base.write_page(args.output_root)
    write_metrics_page(args.output_root)
    base.atomic_json(args.output_root / "metrics_status.json", {
        "state": "prepared", "message": "100 balanced training cases selected",
    })


def top100(selection: Path, ranking: Path) -> list[dict]:
    selected = json.loads(selection.read_text(encoding="utf-8"))["targets"][:100]
    ranked = json.loads(ranking.read_text(encoding="utf-8"))["top100_step_00"]
    scores = {(int(row["block"]), int(row["head"])): float(row["ranking_score"]) for row in ranked}
    return [
        {"rank": index + 1, "block": int(row["block"]), "head": int(row["head"]),
         "ranking_pck32": scores[(int(row["block"]), int(row["head"]))]}
        for index, row in enumerate(selected)
    ]


def case_pck(probabilities, regions, tracks, visibility) -> tuple[float | None, int, int]:
    point_count, time, height, width = probabilities.shape
    flat = probabilities.reshape(point_count, time, height * width)
    best = flat.argmax(axis=-1)
    predictions = np.stack(
        (((best % width) + 0.5) * WIDTH / width, ((best // width) + 0.5) * HEIGHT / height),
        axis=-1,
    ).transpose(1, 0, 2)
    gt_tracks = tracks[ANCHORS]
    gt_visibility = visibility[ANCHORS]
    object_values, comparisons = [], 0
    for region in regions:
        start, end = region["start"], region["end"]
        valid = gt_visibility[:, start:end].copy()
        valid &= gt_visibility[1, start:end][None]
        valid[:2] = False
        if not valid.any():
            continue
        errors = np.linalg.norm(predictions[:, start:end] - gt_tracks[:, start:end], axis=-1)
        values = errors[valid]
        object_values.append(float((values <= 32.0).mean() * 100.0))
        comparisons += int(values.size)
    return (
        float(np.mean(object_values)) if object_values else None,
        len(object_values),
        comparisons,
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_saved_rows(case_root: Path) -> list[dict]:
    path = case_root / "pck_rows.csv"
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "case_key": row["case_key"], "family": row["family"],
                "step": int(row["step"]), "rank": int(row["rank"]),
                "block": int(row["block"]), "head": int(row["head"]),
                "pck32": float(row["pck32"]), "valid_objects": int(row["valid_objects"]),
                "comparisons": int(row["comparisons"]),
            })
    return rows


def aggregate(args, all_rows: list[dict], targets: list[dict], complete: int, final: bool) -> None:
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["step"], row["rank"])].append(row["pck32"])
    per_step = []
    for step in STEPS:
        for target in targets:
            values = grouped[(step, target["rank"])]
            per_step.append({
                "step": step, **target,
                "mean_pck32": float(np.mean(values)) if values else None,
                "std_pck32": float(np.std(values)) if values else None,
                "valid_cases": len(values),
            })
    per_head = []
    for target in targets:
        row = {**target, "valid_cases": {}}
        noise_values = []
        for step in STEPS:
            match = next(item for item in per_step if item["step"] == step and item["rank"] == target["rank"])
            key = f"s{step:02d}"
            row[key] = match["mean_pck32"]
            row["valid_cases"][key] = match["valid_cases"]
            if match["mean_pck32"] is not None:
                noise_values.append(match["mean_pck32"])
        row["all_noise_mean"] = float(np.mean(noise_values)) if noise_values else None
        per_head.append(row)
    mean_head = []
    for step in STEPS:
        row = {"step": step}
        for count in (30, 50, 100):
            values = [item[f"s{step:02d}"] for item in per_head[:count] if item[f"s{step:02d}"] is not None]
            row[f"top{count}"] = float(np.mean(values)) if values else None
        mean_head.append(row)
    metrics_root = args.output_root / "metrics100"
    if final:
        write_csv(metrics_root / "raw_case_head_pck.csv", all_rows, [
            "case_key", "family", "step", "rank", "block", "head", "pck32",
            "valid_objects", "comparisons",
        ])
        write_csv(metrics_root / "per_head_noise_pck.csv", per_step, [
            "step", "rank", "block", "head", "ranking_pck32", "mean_pck32",
            "std_pck32", "valid_cases",
        ])
        write_csv(metrics_root / "mean_head_pck.csv", mean_head, ["step", "top30", "top50", "top100"])
        steps = [row["step"] for row in mean_head]
        figure, axis = plt.subplots(figsize=(8.4, 4.8))
        for count, color in ((30, "#ad452f"), (50, "#bc812c"), (100, "#176b5c")):
            axis.plot(steps, [row[f"top{count}"] for row in mean_head], marker="o", linewidth=2.2, label=f"Top{count}", color=color)
        axis.set(xlabel="Denoising step", ylabel="Case-macro Mean Head PCK@32 (%)", title="Training baseline: Mean Head PCK across 100 cases")
        axis.grid(alpha=.25); axis.legend(); figure.tight_layout()
        curve = metrics_root / "mean_head_pck_curve.png"; figure.savefig(curve, dpi=180); plt.close(figure)
        matrix = np.asarray([[row[f"s{step:02d}"] for step in STEPS] for row in per_head], dtype=float)
        figure, axis = plt.subplots(figsize=(8.4, 12))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        axis.set_xticks(range(len(STEPS)), [f"S{step:02d}" for step in STEPS]); axis.set_ylabel("Fixed PCK rank (Top100)"); axis.set_xlabel("Noise level")
        figure.colorbar(image, ax=axis, label="PCK@32 (%)"); figure.tight_layout()
        heatmap = metrics_root / "noise_head_pck_heatmap.png"; figure.savefig(heatmap, dpi=180); plt.close(figure)
        charts = {"curve": str(curve.relative_to(args.output_root)), "heatmap": str(heatmap.relative_to(args.output_root))}
    else:
        charts = None
    base.atomic_json(metrics_root / "summary.json", {
        "completed_cases": complete, "expected_cases": 100,
        "model": "Wan2.2 + OpenVid LoRA training baseline; no Top100 modules",
        "steps": list(STEPS), "mean_head": mean_head, "per_head": per_head,
        "charts": charts,
    })


def compute(args) -> None:
    targets = top100(args.selection, args.ranking)
    _, pipe, _ = base.build_training_baseline(args.baseline_config, args.device)
    cotracker = gt.load_cotracker(args.device)
    cases = base.load_case_manifests(args.output_root / "metrics100" / "dataset")
    all_rows = []
    for index, case in enumerate(cases, start=1):
        case_key, case_base = case["case_key"], case["base"]
        case_root = args.output_root / "metrics100" / "cases" / case_key
        saved = load_saved_rows(case_root)
        if saved:
            all_rows.extend(saved)
            aggregate(args, all_rows, targets, index, False)
            continue
        base.atomic_json(args.output_root / "metrics_status.json", {
            "state": "computing", "message": f"case {index}/100: {case_key}",
        })
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
        capture = base.TopHeadCapture(pipe, targets, points, STEPS)
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        capture.install()
        try:
            with torch.inference_mode():
                for step in STEPS:
                    timestep = pipe.scheduler.timesteps[step].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                    noised = latents.clone()
                    noised[:, :, prefix:] = pipe.scheduler.add_noise(latents[:, :, prefix:], shared["noise"][:, :, prefix:], timestep)
                    noised[:, :, :prefix] = shared["clean_prefix_latents"]
                    shared["latents"] = noised
                    pipe.model_fn(**models, **shared, **positive, timestep=timestep)
        finally:
            capture.remove()
        if len(capture.records) != len(STEPS) * len(targets):
            raise RuntimeError(f"Captured {len(capture.records)}/500 records for {case_key}")
        frame_array = np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames])
        tracks, visibility = gt.run_cotracker(cotracker, frame_array, points, 4, args.device)
        case_rows = []
        for step in STEPS:
            for target in targets:
                pck32, valid_objects, comparisons = case_pck(
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
        write_csv(case_root / "pck_rows.csv", case_rows, [
            "case_key", "family", "step", "rank", "block", "head", "pck32",
            "valid_objects", "comparisons",
        ])
        all_rows.extend(case_rows)
        aggregate(args, all_rows, targets, index, False)
        del capture, latents, shared, positive
        torch.cuda.empty_cache()
    aggregate(args, all_rows, targets, len(cases), True)
    base.atomic_json(args.output_root / "metrics_status.json", {
        "state": "complete", "message": "100/100 cases; Top100 per-head PCK complete",
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "compute"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--selection", type=Path, default=base.DEFAULT_SELECTION)
    parser.add_argument("--ranking", type=Path, default=base.DEFAULT_RANKING)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args(); args.output_root = args.output_root.resolve(); args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare": prepare(args)
    elif args.cache_root is None or args.baseline_config is None: raise ValueError("compute requires cache and baseline config")
    else: compute(args)


if __name__ == "__main__":
    main()
