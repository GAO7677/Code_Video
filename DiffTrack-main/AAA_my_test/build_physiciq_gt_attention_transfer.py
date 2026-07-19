#!/usr/bin/env python3
"""Render GT-Q/K trajectories and heatmaps over three generated-video families."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from AAA_my_test import analyze_stage1b_kubric_generation as probe
from AAA_my_test.sam2_region_query_utils import load_region_cache


DEFAULT_GT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_gt_attention_transfer_l23_s39"
)
DEFAULT_GENERATED_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk"
)
MODELS = (
    ("stage1b", "Stage1b step-004000", "stretch"),
    ("lora", "LoRA step-000500", "cover_crop"),
    ("baseline", "Wan2.2 baseline", "cover_crop"),
)
GT_VARIANTS = {
    ("framewise", "stretch"): "gt_framewise_stretch",
    ("framewise", "cover_crop"): "gt_framewise_cover_crop",
    ("whole", "stretch"): "gt_whole_stretch",
    ("whole", "cover_crop"): "gt_whole_cover_crop",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/gaoya/agent-data/cache/physiciq_selected_sam2_regions"),
    )
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def read_video(path: Path, frame_count: int = 25) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < frame_count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != frame_count:
        raise RuntimeError(f"expected {frame_count} frames in {path}, got {len(frames)}")
    return np.stack(frames)


def load_record(case_dir: Path) -> tuple[probe.MatchRecord, dict]:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    predictions = np.load(case_dir / "predicted_tracks.npz")[
        "qk_layer23_step039_predictions"
    ]
    probabilities = np.load(case_dir / "attention_probabilities.npz")[
        "qk_layer23_step039_probabilities"
    ]
    record = probe.MatchRecord(
        method="GT Q/K",
        layer=23,
        step_index=39,
        timestep=float(manifest["scheduler_timesteps"][39]),
        sigma=float(manifest["scheduler_sigmas"][39]),
        grid=tuple(int(value) for value in manifest["token_grid"]),
        clean_prefix_latents=int(manifest["clean_prefix_latents"]),
        query_latent_index=int(manifest["query_latent_index"]),
        predictions=predictions,
        probabilities=probabilities,
    )
    return record, manifest


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(link)
    os.symlink(target.resolve(), link, target_is_directory=True)


def trajectory_stats(predictions: np.ndarray) -> dict[str, float]:
    valid = predictions[np.isfinite(predictions).all(axis=-1).all(axis=-1)]
    if len(valid) < 2:
        return {"mean_jump_px": float("nan"), "mean_acceleration_px": float("nan")}
    velocity = np.diff(valid, axis=0)
    jumps = np.linalg.norm(velocity, axis=-1)
    acceleration = np.linalg.norm(np.diff(velocity, axis=0), axis=-1)
    return {
        "mean_jump_px": float(jumps.mean()),
        "mean_acceleration_px": float(acceleration.mean()) if acceleration.size else 0.0,
    }


def build_assets(args: argparse.Namespace) -> list[dict]:
    output = args.gt_root / "dashboard"
    output.mkdir(parents=True, exist_ok=True)
    for model, _, _ in MODELS:
        replace_symlink(output / f"original_{model}", args.generated_root / model)
    for directory in set(GT_VARIANTS.values()):
        replace_symlink(output / directory, args.gt_root / directory)
    replace_symlink(output / "transferred", args.gt_root / "transferred")
    comparisons = args.gt_root / "comparisons"
    if comparisons.exists():
        replace_symlink(output / "comparisons", comparisons)

    stage_cases = args.generated_root / "stage1b" / "cases"
    cases = []
    for generated_case in sorted(stage_cases.iterdir()):
        if not (generated_case / "complete.json").is_file():
            continue
        case_key = generated_case.name
        cache = load_region_cache(args.cache_root, case_key)
        region_names = [region.region_name for region in cache.regions]
        transfer_stats: dict[str, dict] = {}
        for mode in ("framewise", "whole"):
            transfer_stats[mode] = {}
            for model, _, coordinate_mode in MODELS:
                variant = GT_VARIANTS[(mode, coordinate_mode)]
                gt_case = args.gt_root / variant / "cases" / case_key
                record, manifest = load_record(gt_case)
                target_frames = read_video(
                    args.generated_root / model / "cases" / case_key / "generated.mp4"
                )
                anchors = np.asarray(manifest["latent_anchor_pixel_frames"], dtype=np.int64)
                model_output = args.gt_root / "transferred" / mode / model / "cases" / case_key
                for region in cache.regions:
                    sliced = probe.slice_match_record(record, region.point_start, region.point_end)
                    region_output = model_output / "regions" / region.region_name
                    region_output.mkdir(parents=True, exist_ok=True)
                    probe.draw_track_video(
                        target_frames,
                        anchors,
                        sliced,
                        None,
                        None,
                        region_output / "tracks_gt_qk_L23_S039.mp4",
                        int(args.fps),
                    )
                    probe.save_heatmap_montage(
                        target_frames,
                        anchors,
                        sliced,
                        0,
                        region_output / "heatmap_gt_qk_L23_S039.png",
                    )
                    transfer_stats[mode].setdefault(model, {})[region.region_name] = trajectory_stats(
                        sliced.predictions
                    )
        gt_manifest = json.loads(
            (
                args.gt_root
                / GT_VARIANTS[("framewise", "stretch")]
                / "cases"
                / case_key
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        cases.append(
            {
                "case_key": case_key,
                "label": Path(gt_manifest["case_manifest"]).parent.name,
                "regions": region_names,
                "stats": transfer_stats,
            }
        )
    return cases


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GT attention transfer</title><style>
:root{--paper:#eee9dc;--ink:#17211e;--card:#fffdf8;--line:#b8b09f;--rust:#b64a31;--teal:#176654;--muted:#65706b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 4% 0,#d7764b38,transparent 36rem),radial-gradient(circle at 96% 4%,#4b9a8038,transparent 36rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1820px,calc(100% - 28px));margin:auto;padding:28px 0 60px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif;margin:0}h1{font-size:clamp(40px,5vw,76px);line-height:.93;letter-spacing:-.04em}.eyebrow{color:var(--rust);font-size:12px;font-weight:900;letter-spacing:.15em}.lead{max-width:1120px;color:var(--muted);line-height:1.6}.controls{display:grid;grid-template-columns:2fr 1fr 1.2fr 1fr;gap:10px;margin:20px 0}label{font-size:11px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}select{display:block;width:100%;margin-top:5px;padding:10px;border:1px solid var(--ink);background:var(--card);font-weight:800}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.card{background:#111815;color:white;padding:10px;min-width:0;border-radius:3px 18px 3px 3px}.card h2{font-size:18px;min-height:44px}.media{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#020403}.heatmap{aspect-ratio:auto;min-height:220px}.meta{margin:8px 2px 2px;color:#bdc9c3;font-size:11px;line-height:1.5}.note{margin-top:14px;padding:13px;background:var(--card);border:1px solid var(--line);line-height:1.55}.warn{color:#8b321f;font-weight:800}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,1fr)}.controls{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.grid,.controls{grid-template-columns:1fr}}
</style></head><body><main><div class="eyebrow">WAN2.2 · LAYER 23 · STEP 39 · CONTROLLED ATTENTION TRANSFER</div><h1>Whose attention<br>owns the track?</h1><p class="lead">同一语义 query、同一 token 网格和同一绘制逻辑。Own Q/K 显示各生成模型自身的匹配；GT frame-wise 和 GT whole-video 则把 GT 前向得到的 Q/K attention 原样移植到对应生成视频底图。</p><section class="controls"><label>Case<select id="case"></select></label><label>Region<select id="region"></select></label><label>Attention source<select id="source"><option value="own">Own Q/K</option><option value="framewise">GT frame-wise VAE</option><option value="whole">GT whole-video VAE</option></select></label><label>View<select id="view"><option value="tracks">Track overlay</option><option value="heatmap">Attention heatmap</option></select></label></section><div class="grid" id="grid"></div><div class="note"><span class="warn">解释边界：</span>GT attention 移植只改变轨迹/热图来源，不改变生成视频，也不代表生成模型内部 attention 得到改善。Stage1b 使用 stretch 坐标版 GT；LoRA 与 baseline 使用 cover-crop 坐标版 GT。整段 VAE 是 Wan 因果时序压缩得到 7 个 latent；逐帧版独立编码 pixel anchors [0,4,8,12,16,20,24] 得到 7 个 latent。</div></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),caseEl=document.getElementById('case'),regionEl=document.getElementById('region'),sourceEl=document.getElementById('source'),viewEl=document.getElementById('view'),grid=document.getElementById('grid');const models=[['gt','GT source'],['stage1b','Stage1b step-004000'],['lora','LoRA step-000500'],['baseline','Wan2.2 baseline']];for(const c of data.cases){const o=document.createElement('option');o.value=c.case_key;o.textContent=c.label;caseEl.append(o)}function current(){return data.cases.find(c=>c.case_key===caseEl.value)}function setRegions(){regionEl.innerHTML=current().regions.map(r=>`<option value="${r}">${r}</option>`).join('')}function path(model,source,view,c,r){const kind=view==='tracks'?'tracks_qk_L23_S039.mp4':'heatmap_qk_L23_S039.png';if(model==='gt'){const variant=source==='whole'?'gt_whole_stretch':'gt_framewise_stretch';return `${variant}/cases/${c}/regions/${r}/${kind}`}if(view==='tracks')return `comparisons/${model}/cases/${c}/regions/${r}/tracks_${source}_vs_cotracker.mp4`;if(source==='own')return `original_${model}/cases/${c}/regions/${r}/${kind}`;return `transferred/${source}/${model}/cases/${c}/regions/${r}/heatmap_gt_qk_L23_S039.png`}function render(){const c=current().case_key,r=regionEl.value,s=sourceEl.value==='own'?'own':sourceEl.value,v=viewEl.value;grid.innerHTML=models.map(([name,label])=>{const effective=name==='gt'&&s==='own'?'framewise':s;const src=path(name,effective,v,c,r);const media=v==='tracks'?`<video class="media" controls muted loop playsinline preload="metadata" src="${src}"></video>`:`<img class="media heatmap" src="${src}">`;const note=name==='gt'?(effective==='whole'?'whole-video VAE Q/K':'frame-wise anchor VAE Q/K'):(effective==='own'?'model-native Q/K vs generated-video CoTracker':`GT ${effective} Q/K vs generated-video CoTracker`);return `<article class="card"><h2>${label}</h2>${media}<div class="meta">${note}</div></article>`}).join('')}caseEl.addEventListener('change',()=>{setRegions();render()});for(const el of [regionEl,sourceEl,viewEl])el.addEventListener('change',render);setRegions();render();
</script></body></html>'''


def main() -> None:
    args = parse_args()
    cases = build_assets(args)
    payload = json.dumps({"cases": cases}, ensure_ascii=False).replace("</", "<\\/")
    output = args.gt_root / "dashboard" / "index.html"
    output.write_text(HTML.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(f"Built {output} with {len(cases)} cases")


if __name__ == "__main__":
    main()
