#!/usr/bin/env python3
"""Build a synchronized Stage1b/LoRA/base-Wan token trajectory dashboard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


OUTPUTS = Path("/data/gaoya/agent-data/outputs")
DEFAULT_OUTPUT = OUTPUTS / "three_model_denoising_trajectory_comparison"
MODEL_SPECS = (
    (
        "stage1b",
        "Stage1b step-004000",
        OUTPUTS / "stage1b_kubric_step004000_sam2_regions_steps40",
    ),
    (
        "lora",
        "LoRA step-000500",
        OUTPUTS / "wan_openvid_0613pybullet_lora_step000500_sam2_regions_steps40",
    ),
    (
        "baseline",
        "Wan2.2-TI2V-5B baseline",
        OUTPUTS / "wan22_ti2v_5b_baseline_sam2_regions_steps40",
    ),
)
TARGET_REGIONS = ("object_A", "object_B", "background")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1b-root", type=Path, default=MODEL_SPECS[0][2])
    parser.add_argument("--lora-root", type=Path, default=MODEL_SPECS[1][2])
    parser.add_argument("--baseline-root", type=Path, default=MODEL_SPECS[2][2])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def completed_cases(root: Path) -> dict[str, Path]:
    case_root = root / "cases"
    return {
        path.name: path
        for path in sorted(case_root.glob("case_*"))
        if all(
            (path / name).is_file()
            for name in (
                "complete.json",
                "manifest.json",
                "generated.mp4",
                "predicted_tracks.npz",
                "cotracker_pseudo_gt.npz",
            )
        )
    }


def link_assets(output_dir: Path, name: str, target: Path) -> None:
    link = output_dir / name
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"asset path exists and is not a symlink: {link}")
    os.symlink(target.resolve(), link, target_is_directory=True)


def finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def trajectory_metrics(
    predictions: np.ndarray,
    gt_tracks: np.ndarray,
    visibility: np.ndarray,
    anchors: np.ndarray,
    point_start: int,
    point_end: int,
    query_latent_index: int,
    clean_prefix_latents: int,
) -> dict:
    pred = predictions[:, point_start:point_end].astype(np.float32)
    gt = gt_tracks[anchors, point_start:point_end].astype(np.float32)
    visible = visibility[anchors, point_start:point_end].astype(bool)
    valid = visible & visible[query_latent_index : query_latent_index + 1]
    valid[:clean_prefix_latents] = False
    error = np.linalg.norm(pred - gt, axis=-1)
    values = error[valid]

    motion_errors = []
    pred_steps = []
    gt_steps = []
    for time_index in range(max(clean_prefix_latents, 1), len(anchors)):
        pair_valid = valid[time_index] & visible[time_index - 1]
        if not pair_valid.any():
            continue
        pred_delta = pred[time_index] - pred[time_index - 1]
        gt_delta = gt[time_index] - gt[time_index - 1]
        motion_errors.extend(np.linalg.norm(pred_delta[pair_valid] - gt_delta[pair_valid], axis=-1))
        pred_steps.extend(np.linalg.norm(pred_delta[pair_valid], axis=-1))
        gt_steps.extend(np.linalg.norm(gt_delta[pair_valid], axis=-1))

    motion_errors = np.asarray(motion_errors, dtype=np.float32)
    pred_steps = np.asarray(pred_steps, dtype=np.float32)
    gt_steps = np.asarray(gt_steps, dtype=np.float32)
    return {
        "comparisons": int(values.size),
        "mean_error_px": finite_float(values.mean()) if values.size else None,
        "median_error_px": finite_float(np.median(values)) if values.size else None,
        "pck32": finite_float((values <= 32).mean() * 100) if values.size else None,
        "motion_error_px": finite_float(motion_errors.mean()) if motion_errors.size else None,
        "mean_pred_step_px": finite_float(pred_steps.mean()) if pred_steps.size else None,
        "mean_gt_step_px": finite_float(gt_steps.mean()) if gt_steps.size else None,
        "jump_rate_64": finite_float((pred_steps > 64).mean() * 100) if pred_steps.size else None,
    }


def load_model_case(
    model_name: str,
    model_label: str,
    case_dir: Path,
    regions: list[dict],
) -> dict:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    tracks_archive = np.load(case_dir / "predicted_tracks.npz")
    gt_archive = np.load(case_dir / "cotracker_pseudo_gt.npz")
    anchors = gt_archive["latent_anchor_frames"].astype(np.int64)
    gt_tracks = gt_archive["tracks"].astype(np.float32)
    visibility = gt_archive["visibility"].astype(bool)
    predictions = {}
    metrics = {}
    for method in ("qk", "hidden"):
        for layer in manifest["layers"]:
            for step in manifest["step_indices"]:
                archive_key = f"{method}_layer{int(layer):02d}_step{int(step):03d}_predictions"
                if archive_key not in tracks_archive:
                    raise KeyError(f"{case_dir}: missing {archive_key}")
                values = tracks_archive[archive_key].astype(np.float32)
                key = f"{method}/L{int(layer):02d}/S{int(step):03d}"
                predictions[key] = {}
                metrics[key] = {}
                for region in regions:
                    start = int(region["point_start"])
                    end = int(region["point_end"])
                    region_name = str(region["region_name"])
                    predictions[key][region_name] = values[:, start:end].round(3).tolist()
                    metrics[key][region_name] = trajectory_metrics(
                        values,
                        gt_tracks,
                        visibility,
                        anchors,
                        start,
                        end,
                        int(manifest["query_latent_index"]),
                        int(manifest["clean_prefix_latents"]),
                    )
    return {
        "name": model_name,
        "label": model_label,
        "video": f"{model_name}/cases/{case_dir.name}/generated.mp4",
        "poster": f"{model_name}/cases/{case_dir.name}/query_points.png",
        "manifest": f"{model_name}/cases/{case_dir.name}/manifest.json",
        "anchors": anchors.tolist(),
        "query_latent_index": int(manifest["query_latent_index"]),
        "clean_prefix_latents": int(manifest["clean_prefix_latents"]),
        "gt_tracks": {
            str(region["region_name"]): gt_tracks[
                :, int(region["point_start"]) : int(region["point_end"])
            ].round(3).tolist()
            for region in regions
        },
        "visibility": {
            str(region["region_name"]): visibility[
                :, int(region["point_start"]) : int(region["point_end"])
            ].astype(np.uint8).tolist()
            for region in regions
        },
        "predictions": predictions,
        "metrics": metrics,
    }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-model token trajectories</title><style>
:root{--paper:#eee9dc;--ink:#18211e;--card:#fffdf7;--line:#bfb8a7;--red:#ba412c;--green:#176654;--muted:#68716c}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 5% 0,#d9784c33,transparent 34rem),radial-gradient(circle at 95% 5%,#4e9c8433,transparent 34rem),var(--paper);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1760px,calc(100% - 28px));margin:auto;padding:30px 0 64px}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif;margin:0}h1{font-size:clamp(38px,5vw,74px);line-height:.94;letter-spacing:-.045em}.eyebrow{color:var(--red);font-weight:900;font-size:12px;letter-spacing:.16em;text-transform:uppercase}.lead{max-width:1050px;color:var(--muted);line-height:1.6}.controls{display:grid;grid-template-columns:1.4fr 1fr 1fr .7fr .7fr;gap:10px;margin:22px 0 10px}.transport{display:grid;grid-template-columns:auto auto 1fr auto;align-items:center;gap:10px;background:var(--card);border:1px solid var(--line);padding:10px;margin-bottom:14px}button,select{border:1px solid var(--ink);background:var(--card);padding:10px;font-weight:800;width:100%}button{width:auto;cursor:pointer}label{font-size:11px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}label select{display:block;margin-top:5px}.models{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.model{background:#111815;color:#fff;padding:10px;border-radius:4px 20px 4px 4px;min-width:0}.model h2{font-size:20px;margin:2px 3px 9px}.stage{position:relative;aspect-ratio:7/4;background:#050806;overflow:hidden}.stage video{display:block;width:100%;height:100%;object-fit:contain}.stage canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.video-meta{display:flex;justify-content:space-between;gap:8px;align-items:center;margin:7px 2px;font-size:11px}.video-meta a{color:#9fd8c5}.video-state{color:#cbd3ce}.video-state.error{color:#ff9b83}.legend{font-size:11px;color:#cbd3ce;margin:8px 2px}.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}.stat{background:#fff;color:var(--ink);padding:9px}.stat span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}.stat b{font:700 23px/1.1 Georgia}.matrix-wrap{margin-top:14px;background:var(--card);border:1px solid var(--line);padding:14px}.matrices{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.matrix{overflow:auto}.matrix h3{margin-bottom:7px}.matrix table{border-collapse:collapse;width:100%}th,td{border:1px solid var(--line);padding:6px;text-align:right;font-size:11px}th:first-child{text-align:left}.note{color:var(--muted);font-size:12px;line-height:1.55}@media(max-width:1050px){.controls,.models,.matrices{grid-template-columns:1fr}.transport{grid-template-columns:auto auto 1fr}.transport output{grid-column:1/-1}}</style></head><body><main><header><div class="eyebrow">frame 4 query · anchors [0,4,8,12,16,20,24] · 40 denoising steps</div><h1>Token Trajectory<br>Cross-examination</h1><p class="lead">三个模型使用同一 source case、同一帧 4 SAM2 query 点和相同匹配规则。每个目标 latent frame 独立取最大相关 token，再连接成轨迹。实线圆点为各自生成视频上的 CoTracker 伪 GT，虚线方点为 Q/K 或 hidden argmax。</p></header><section class="controls"><label>Source case<select id="case"></select></label><label>Region<select id="region"></select></label><label>Feature<select id="method"><option value="qk">Q/K attention</option><option value="hidden">Hidden feature</option></select></label><label>Layer<select id="layer"></select></label><label>Step<select id="step"></select></label></section><section class="transport"><button id="play">Play</button><button id="pause">Pause</button><input id="frame" type="range" min="0" max="24" step="1" value="4"><output id="frame-label">frame 4</output></section><section class="models" id="models"></section><section class="matrix-wrap"><h2>Layer × denoising-step PCK@32</h2><p class="note">仅统计 query 在 frame 4 可见且目标 anchor 可见的未来点。Motion error 比较相邻 anchor 的预测位移向量与 CoTracker 位移向量；Jump@64 是预测相邻 anchor 位移超过 64 px 的比例。</p><div class="matrices" id="matrices"></div></section></main><script id="index-data" type="application/json">__INDEX__</script><script>
const index=JSON.parse(document.getElementById('index-data').textContent);let payload=null,timer=null;const $=id=>document.getElementById(id),caseEl=$('case'),regionEl=$('region'),methodEl=$('method'),layerEl=$('layer'),stepEl=$('step'),frameEl=$('frame');const fmt=(v,d=1)=>v==null?'NA':Number(v).toFixed(d);caseEl.innerHTML=index.cases.map(c=>`<option>${c}</option>`).join('');layerEl.innerHTML=index.layers.map(v=>`<option value="${v}">L${v}</option>`).join('');stepEl.innerHTML=index.steps.map(v=>`<option value="${v}">S${v}</option>`).join('');layerEl.value=17;stepEl.value=39;
function key(){return `${methodEl.value}/L${String(layerEl.value).padStart(2,'0')}/S${String(stepEl.value).padStart(3,'0')}`}function colors(n){return Array.from({length:n},(_,i)=>`hsl(${Math.round(i*330/Math.max(n,1))} 82% 61%)`)}
function loadCase(){payload=index.case_data[caseEl.value];regionEl.innerHTML=payload.regions.map(r=>`<option value="${r.region_name}">${r.region_name}${r.region_phrase?' · '+r.region_phrase:''}</option>`).join('');buildModels();render()}
function buildModels(){$('models').innerHTML=payload.models.map((m,i)=>`<article class="model"><h2>${m.label}</h2><div class="stage"><video id="video-${i}" controls preload="auto" muted playsinline poster="${m.poster}"><source src="${m.video}" type="video/mp4">Your browser cannot play H.264 MP4.</video><canvas id="canvas-${i}" width="896" height="512"></canvas></div><div class="video-meta"><span id="video-state-${i}" class="video-state">loading video...</span><a href="${m.video}" target="_blank">open MP4</a></div><div class="legend">same color = same query ID · circle/solid = CoTracker · square/dashed = token match</div><div class="stats" id="stats-${i}"></div></article>`).join('');payload.models.forEach((_,i)=>{const v=$('video-'+i),state=$('video-state-'+i);v.addEventListener('loadedmetadata',()=>{state.textContent='metadata loaded';seekVideos()});v.addEventListener('loadeddata',()=>{state.textContent='video ready';seekVideos()});v.addEventListener('canplay',()=>{state.textContent='video ready'});v.addEventListener('error',()=>{state.textContent=`video error ${v.error?.code||''}`;state.classList.add('error')});v.load()})}
function drawModel(m,i){const region=regionEl.value,pred=m.predictions[key()][region],gt=m.gt_tracks[region],vis=m.visibility[region],anchors=m.anchors,frame=Number(frameEl.value),canvas=$('canvas-'+i),ctx=canvas.getContext('2d'),cs=colors(pred[0].length);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=2;for(let p=0;p<cs.length;p++){ctx.strokeStyle=cs[p];ctx.fillStyle=cs[p];ctx.globalAlpha=.7;ctx.setLineDash([]);ctx.beginPath();let started=false;for(let t=4;t<=frame&&t<gt.length;t++){if(!vis[t][p])continue;const [x,y]=gt[t][p];if(!started){ctx.moveTo(x,y);started=true}else ctx.lineTo(x,y)}ctx.stroke();if(frame<gt.length&&vis[frame][p]){const [x,y]=gt[frame][p];ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;ctx.setLineDash([7,5]);ctx.beginPath();started=false;for(let t=m.query_latent_index;t<anchors.length;t++){if(anchors[t]>frame)continue;const [x,y]=pred[t][p];if(!started){ctx.moveTo(x,y);started=true}else ctx.lineTo(x,y)}ctx.stroke();ctx.setLineDash([]);for(let t=m.query_latent_index;t<anchors.length;t++){if(anchors[t]>frame)continue;const [x,y]=pred[t][p];ctx.strokeRect(x-4,y-4,8,8)}}}
function stat(label,v,s=''){return `<div class="stat"><span>${label}</span><b>${fmt(v)}${s}</b></div>`}function matrix(m){const region=regionEl.value,method=methodEl.value;let h='<table><tr><th>Layer</th>'+index.steps.map(s=>`<th>S${s}</th>`).join('')+'</tr>';for(const l of index.layers){h+=`<tr><th>L${l}</th>`+index.steps.map(s=>{const k=`${method}/L${String(l).padStart(2,'0')}/S${String(s).padStart(3,'0')}`,v=m.metrics[k][region].pck32,a=Math.max(0,Math.min(1,(v||0)/100));return `<td style="background:rgba(23,102,84,${.06+.7*a})">${fmt(v)}%</td>`}).join('')+'</tr>'}return h+'</table>'}
function render(){if(!payload)return;const frame=Number(frameEl.value);$('frame-label').textContent=`frame ${frame}`;payload.models.forEach((m,i)=>{drawModel(m,i);const x=m.metrics[key()][regionEl.value];$('stats-'+i).innerHTML=stat('PCK@32',x.pck32,'%')+stat('Mean error',x.mean_error_px,' px')+stat('Motion error',x.motion_error_px,' px')+stat('Jump@64',x.jump_rate_64,'%')});$('matrices').innerHTML=payload.models.map(m=>`<article class="matrix"><h3>${m.label}</h3>${matrix(m)}</article>`).join('');seekVideos()}
function seekVideos(){if(!payload)return;payload.models.forEach((_,i)=>{const v=$('video-'+i);if(v&&v.readyState>=1&&Number.isFinite(v.duration))v.currentTime=Math.min(v.duration-.001,Number(frameEl.value)/24*v.duration)})}function play(){clearInterval(timer);timer=setInterval(()=>{let f=Number(frameEl.value)+1;if(f>24){clearInterval(timer);return}frameEl.value=f;render()},1000/6)}function pause(){clearInterval(timer)}caseEl.onchange=loadCase;[regionEl,methodEl,layerEl,stepEl].forEach(e=>e.onchange=render);frameEl.oninput=render;$('play').onclick=play;$('pause').onclick=pause;loadCase();
</script></body></html>'''


def main() -> None:
    args = parse_args()
    specs = (
        ("stage1b", "Stage1b step-004000", args.stage1b_root.resolve()),
        ("lora", "LoRA step-000500", args.lora_root.resolve()),
        ("baseline", "Wan2.2-TI2V-5B baseline", args.baseline_root.resolve()),
    )
    case_maps = {name: completed_cases(root) for name, _, root in specs}
    common = sorted(set.intersection(*(set(case_map) for case_map in case_maps.values())))
    if not common:
        counts = {name: len(case_map) for name, case_map in case_maps.items()}
        raise RuntimeError(f"no complete case shared by all models: {counts}")

    output = args.output_dir.resolve()
    (output / "cases").mkdir(parents=True, exist_ok=True)
    for name, _, root in specs:
        link_assets(output, name, root)

    reference_manifest = json.loads(
        (case_maps["stage1b"][common[0]] / "manifest.json").read_text(encoding="utf-8")
    )
    layers = [int(value) for value in reference_manifest["layers"]]
    steps = [int(value) for value in reference_manifest["step_indices"]]
    case_data = {}
    for case_key in common:
        manifests = {
            name: json.loads((case_maps[name][case_key] / "manifest.json").read_text(encoding="utf-8"))
            for name, _, _ in specs
        }
        reference = manifests["stage1b"]
        if int(reference["query_pixel_frame"]) != 4:
            raise RuntimeError(f"{case_key}: expected query frame 4")
        all_regions = reference["query_regions"]
        regions = [
            region for region in all_regions if region["region_name"] in TARGET_REGIONS
        ]
        for region in regions:
            point_count = int(region["point_end"]) - int(region["point_start"])
            if point_count != 8:
                raise RuntimeError(
                    f"{case_key}/{region['region_name']}: expected 8 query points, got {point_count}"
                )
        reference_points = np.asarray(reference["query_points"], dtype=np.float32)
        for name, manifest in manifests.items():
            if manifest["layers"] != layers or manifest["step_indices"] != steps:
                raise RuntimeError(f"{case_key}/{name}: layer or step mismatch")
            if manifest["latent_anchor_pixel_frames"] != [0, 4, 8, 12, 16, 20, 24]:
                raise RuntimeError(f"{case_key}/{name}: anchor mismatch")
            if manifest["query_regions"] != all_regions:
                raise RuntimeError(f"{case_key}/{name}: region mismatch")
            if not np.array_equal(
                np.asarray(manifest["query_points"], dtype=np.float32), reference_points
            ):
                raise RuntimeError(f"{case_key}/{name}: query points are not identical")
        case_payload = {
            "case_key": case_key,
            "prompt": reference["prompt"],
            "query_frame": 4,
            "query_points": reference_points.tolist(),
            "regions": regions,
            "models": [
                load_model_case(name, label, case_maps[name][case_key], regions)
                for name, label, _ in specs
            ],
        }
        case_data[case_key] = case_payload
        (output / "cases" / f"{case_key}.json").write_text(
            json.dumps(case_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    index_payload = {"cases": common, "layers": layers, "steps": steps}
    page_payload = {**index_payload, "case_data": case_data}
    serialized = json.dumps(page_payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    (output / "index.html").write_text(
        HTML.replace("__INDEX__", serialized), encoding="utf-8"
    )
    (output / "index_data.json").write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"built {len(common)} cases: {output / 'index.html'}")


if __name__ == "__main__":
    main()
