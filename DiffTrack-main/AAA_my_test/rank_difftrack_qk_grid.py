#!/usr/bin/env python3
"""Rank DiffTrack-compatible layer/step combinations at multiple PCK radii."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from center_query_utils import DEFAULT_CACHE, select_center_queries


MODELS = ("gt", "stage1b", "lora", "baseline")
THRESHOLDS = (4, 8, 16, 32)
EXPECTED_IMPLEMENTATION = "utils.matching.corr_to_matches+torch.grid_sample"
DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk_49f_difftrack_grid"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def scope_indices(manifest: dict, centers: dict[str, dict]) -> dict[str, list[int]]:
    regions = manifest["query_regions"]
    object_regions = [r for r in regions if r["region_type"] == "object"]
    background_regions = [r for r in regions if r["region_type"] == "background"]
    object_centers = [int(centers[r["region_name"]]["global_index"]) for r in object_regions]
    background_centers = [
        int(centers[r["region_name"]]["global_index"]) for r in background_regions
    ]
    object_queries = [
        index
        for region in object_regions
        for index in range(int(region["point_start"]), int(region["point_end"]))
    ]
    background_queries = [
        index
        for region in background_regions
        for index in range(int(region["point_start"]), int(region["point_end"]))
    ]
    return {
        "object_centers": object_centers,
        "all_centers": object_centers + background_centers,
        "background_centers": background_centers,
        "object_queries": object_queries,
        "all_queries": object_queries + background_queries,
        "background_queries": background_queries,
    }


def distances_for(
    predictions: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    anchors: np.ndarray,
    query_latent: int,
    clean_prefix: int,
    point_indices: list[int],
) -> np.ndarray:
    if not point_indices:
        return np.empty((0,), dtype=np.float32)
    predicted = predictions[:, point_indices]
    target = tracks[anchors][:, point_indices]
    visible = visibility[anchors][:, point_indices].copy()
    visible &= visibility[int(anchors[query_latent]), point_indices][None]
    valid = visible & np.isfinite(predicted).all(axis=-1)
    valid[:clean_prefix] = False
    return np.linalg.norm(predicted - target, axis=-1)[valid]


def collect_rows(root: Path, cache_root: Path) -> tuple[list[dict], list[int], list[int]]:
    rows = []
    observed_layers: set[int] = set()
    observed_steps: set[int] = set()
    for model in MODELS:
        for case_dir in sorted((root / model / "cases").glob("case_*")):
            if not (case_dir / "complete.json").is_file():
                continue
            manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("matching_mode") != "difftrack":
                raise RuntimeError(f"non-DiffTrack matching mode: {case_dir}")
            if manifest.get("matching_implementation") != EXPECTED_IMPLEMENTATION:
                raise RuntimeError(f"unexpected matching implementation: {case_dir}")
            layers = [int(value) for value in manifest["layers"]]
            steps = [int(value) for value in manifest["step_indices"]]
            observed_layers.update(layers)
            observed_steps.update(steps)
            centers = select_center_queries(cache_root, case_dir.name, manifest["query_regions"])
            scopes = scope_indices(manifest, centers)
            cotracker = np.load(case_dir / "cotracker_pseudo_gt.npz")
            tracks = cotracker["tracks"]
            visibility = cotracker["visibility"].astype(bool)
            anchors = np.asarray(manifest["latent_anchor_pixel_frames"], dtype=np.int64)
            predictions = np.load(case_dir / "predicted_tracks.npz")
            for layer in layers:
                for step in steps:
                    key = f"qk_layer{layer:02d}_step{step:03d}_predictions"
                    if key not in predictions:
                        raise RuntimeError(f"missing {key}: {case_dir}")
                    for scope, indices in scopes.items():
                        distances = distances_for(
                            predictions[key],
                            tracks,
                            visibility,
                            anchors,
                            int(manifest["query_latent_index"]),
                            int(manifest["clean_prefix_latents"]),
                            indices,
                        )
                        row = {
                            "model": model,
                            "case_key": case_dir.name,
                            "scope": scope,
                            "layer": layer,
                            "step_index": step,
                            "comparisons": int(distances.size),
                            "mean_error_px": float(distances.mean()) if distances.size else None,
                        }
                        for threshold in THRESHOLDS:
                            row[f"hits_{threshold}"] = int((distances <= threshold).sum())
                            row[f"pck{threshold}"] = (
                                float(100 * (distances <= threshold).mean())
                                if distances.size
                                else None
                            )
                        rows.append(row)
    if not rows:
        raise RuntimeError(f"no completed grid results under {root}")
    return rows, sorted(observed_layers), sorted(observed_steps)


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["scope"], row["layer"], row["step_index"])].append(row)
    summary = []
    for (model, scope, layer, step), case_rows in groups.items():
        total = sum(row["comparisons"] for row in case_rows)
        item = {
            "model": model,
            "scope": scope,
            "layer": layer,
            "step_index": step,
            "cases": sum(row["comparisons"] > 0 for row in case_rows),
            "comparisons": total,
            "mean_error_px": (
                sum(row["mean_error_px"] * row["comparisons"] for row in case_rows if row["comparisons"])
                / total
                if total
                else None
            ),
        }
        for threshold in THRESHOLDS:
            item[f"pck{threshold}"] = (
                100 * sum(row[f"hits_{threshold}"] for row in case_rows) / total
                if total
                else None
            )
            valid = [row[f"pck{threshold}"] for row in case_rows if row[f"pck{threshold}"] is not None]
            item[f"macro_pck{threshold}"] = float(np.mean(valid)) if valid else None
        item["mean_pck"] = float(np.mean([item[f"pck{x}"] for x in THRESHOLDS]))
        summary.append(item)
    return sorted(summary, key=lambda row: (row["model"], row["scope"], row["layer"], row["step_index"]))


def best_rows(summary: list[dict]) -> list[dict]:
    output = []
    for model in MODELS:
        for scope in ("object_centers", "all_centers", "object_queries", "all_queries"):
            candidates = [row for row in summary if row["model"] == model and row["scope"] == scope]
            for metric in (*[f"pck{x}" for x in THRESHOLDS], "mean_pck"):
                best = max(
                    candidates,
                    key=lambda row: (row[metric], -row["mean_error_px"], -row["layer"], -row["step_index"]),
                )
                output.append({"metric": metric, **best})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, best: list[dict], layers: list[int], steps: list[int]) -> None:
    lines = [
        "# DiffTrack-compatible Q/K layer × step 搜索",
        "",
        f"搜索 layers={layers}、steps={steps}。匹配严格使用双向 Q/K、逐方向空间 softmax、多头平均、`corr_to_matches` 和 `grid_sample`。主结果为物体中心点 pooled PCK；当前只有 4 个 case，排名属于探索性结果。",
        "",
        "## 物体中心点最佳组合",
        "",
        "| model | metric | layer/step | score | macro-case | comparisons | mean error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for metric in (*[f"pck{x}" for x in THRESHOLDS], "mean_pck"):
            row = next(
                item
                for item in best
                if item["model"] == model
                and item["scope"] == "object_centers"
                and item["metric"] == metric
            )
            macro = "-" if metric == "mean_pck" else f"{row['macro_' + metric]:.1f}%"
            lines.append(
                f"| {model} | {metric} | L{row['layer']}/S{row['step_index']} | "
                f"{row[metric]:.1f}% | {macro} | {row['comparisons']} | {row['mean_error_px']:.1f}px |"
            )
    lines.extend(
        [
            "",
            "## 全区域中心点最佳组合",
            "",
            "| model | PCK@4 | PCK@8 | PCK@16 | PCK@32 | mean-PCK |",
            "|---|---|---|---|---|---|",
        ]
    )
    for model in MODELS:
        values = []
        for metric in (*[f"pck{x}" for x in THRESHOLDS], "mean_pck"):
            row = next(
                item
                for item in best
                if item["model"] == model
                and item["scope"] == "all_centers"
                and item["metric"] == metric
            )
            values.append(f"L{row['layer']}/S{row['step_index']} ({row[metric]:.1f}%)")
        lines.append(f"| {model} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DiffTrack Q/K grid search</title><style>
:root{--paper:#eee7d6;--ink:#18211d;--card:#fffaf0;--line:#b8ae99;--rust:#b64a30;--green:#176654;--muted:#66716b}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 3% 0,#d4714838,transparent 34rem),radial-gradient(circle at 97% 5%,#4c94793d,transparent 34rem),var(--paper);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1450px,calc(100% - 26px));margin:auto;padding:30px 0 60px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}h1{font-size:clamp(42px,6vw,80px);line-height:.92;letter-spacing:-.045em;margin:4px 0 13px}.eyebrow{color:var(--rust);font-weight:900;letter-spacing:.15em;font-size:12px}.lead{max-width:980px;color:var(--muted);line-height:1.6}.controls{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:22px 0}label{font-size:11px;font-weight:900;letter-spacing:.09em;text-transform:uppercase}select{display:block;width:100%;margin-top:5px;padding:10px;border:1px solid var(--ink);background:var(--card);font-weight:800}.hero,.matrix{background:var(--card);border:1px solid var(--line);padding:16px}.hero strong{font:700 42px Georgia;color:var(--green)}.hero span{display:block;color:var(--muted);margin-top:5px}.matrix{margin-top:12px;overflow:auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid var(--line);padding:10px;text-align:right}th:first-child{text-align:left}.note{margin-top:14px;color:var(--muted);line-height:1.55}@media(max-width:720px){.controls{grid-template-columns:1fr}}
</style></head><body><main><div class="eyebrow">DIFFTRACK-COMPATIBLE · 6 LAYERS × 5 DENOISING STEPS</div><h1>Where correspondence<br>actually concentrates</h1><p class="lead">严格使用双向 Q/K、空间 softmax、多头平均、corr_to_matches 与 grid_sample。切换模型、采样范围和 PCK 半径查看 layer×step 矩阵。</p><section class="controls"><label>Model<select id="model"></select></label><label>Scope<select id="scope"></select></label><label>Metric<select id="metric"></select></label></section><section class="hero" id="hero"></section><section class="matrix" id="matrix"></section><p class="note">主分析建议使用 object_centers；all_centers 会包含容易匹配的静态背景。仅 4 个 case，最优组合属于探索性结果，不等于已获得统计显著性。</p></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const d=JSON.parse(document.getElementById('payload').textContent),M=document.getElementById('model'),S=document.getElementById('scope'),K=document.getElementById('metric');M.innerHTML=d.models.map(x=>`<option>${x}</option>`).join('');S.innerHTML=d.scopes.map(x=>`<option>${x}</option>`).join('');K.innerHTML=d.metrics.map(x=>`<option>${x}</option>`).join('');S.value='object_centers';K.value='pck32';function render(){const rows=d.summary.filter(x=>x.model===M.value&&x.scope===S.value),metric=K.value,best=rows.reduce((a,b)=>b[metric]>a[metric]?b:a),head=`<table><tr><th>Layer</th>${d.steps.map(x=>`<th>S${x}</th>`).join('')}</tr>`,body=d.layers.map(l=>`<tr><th>L${l}</th>${d.steps.map(s=>{const r=rows.find(x=>x.layer===l&&x.step_index===s),v=r[metric],a=Math.max(0,Math.min(1,v/100));return `<td style="background:rgba(23,102,84,${.05+.72*a})">${v.toFixed(1)}%</td>`}).join('')}</tr>`).join('');document.getElementById('matrix').innerHTML=head+body+'</table>';document.getElementById('hero').innerHTML=`<strong>L${best.layer}/S${best.step_index} · ${best[metric].toFixed(1)}%</strong><span>${M.value} · ${S.value} · ${metric} · ${best.comparisons} comparisons · mean error ${best.mean_error_px.toFixed(1)}px</span>`}for(const e of [M,S,K])e.addEventListener('change',render);render();
</script></body></html>'''


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    output = (args.output_dir or (root / "grid_ranking")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, layers, steps = collect_rows(root, args.cache_root.resolve())
    summary = aggregate(rows)
    best = best_rows(summary)
    write_csv(output / "per_case_metrics.csv", rows)
    write_csv(output / "grid_summary.csv", summary)
    (output / "ranking.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output / "RESULTS.md", best, layers, steps)
    payload = json.dumps(
        {
            "models": list(MODELS),
            "scopes": sorted({row["scope"] for row in summary}),
            "metrics": [*[f"pck{x}" for x in THRESHOLDS], "mean_pck"],
            "layers": layers,
            "steps": steps,
            "summary": summary,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    (output / "index.html").write_text(
        HTML.replace("__PAYLOAD__", payload), encoding="utf-8"
    )
    print(f"Ranked {len(summary)} model/scope/layer/step rows in {output}")


if __name__ == "__main__":
    main()
