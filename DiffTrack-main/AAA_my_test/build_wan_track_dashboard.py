#!/usr/bin/env python3
"""Render Wan Q/K versus CoTracker trajectories and build a local dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from AAA_my_test.wan_motion_utils import OUTPUT_ROOT, read_video


PRIMARY = {"layer": 17, "step": 49, "label": "L17 / S49", "color": (31, 126, 230)}
SECONDARY = {"layer": 17, "step": 36, "label": "L17 / S36", "color": (230, 142, 36)}
PANEL_WIDTH = 480
PANEL_HEIGHT = 264
TRACE_LENGTH = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "visualization")
    parser.add_argument("--fps", type=float, default=7.5)
    parser.add_argument("--case-key")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def point_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(count):
        hue = int(round(179 * index / max(count, 1)))
        rgb = cv2.cvtColor(np.uint8([[[hue, 205, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(tuple(int(value) for value in rgb))
    return colors


def scale_points(points: np.ndarray) -> np.ndarray:
    scaled = points.copy()
    scaled[..., 0] *= PANEL_WIDTH / 1280
    scaled[..., 1] *= PANEL_HEIGHT / 704
    return scaled


def label_box(image: np.ndarray, title: str, subtitle: str) -> None:
    cv2.rectangle(image, (0, 0), (PANEL_WIDTH, 49), (15, 20, 19), -1)
    cv2.putText(image, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (248, 246, 238), 1, cv2.LINE_AA)
    cv2.putText(image, subtitle, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (188, 199, 194), 1, cv2.LINE_AA)


def draw_tracks(
    frame: np.ndarray,
    gt: np.ndarray,
    visibility: np.ndarray,
    frame_index: int,
    title: str,
    subtitle: str,
    predicted: np.ndarray | None = None,
) -> np.ndarray:
    canvas = cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    gt_scaled = scale_points(gt)
    predicted_scaled = scale_points(predicted) if predicted is not None else None
    colors = point_colors(gt.shape[1])
    trace_start = max(0, frame_index - TRACE_LENGTH)
    for point_index, color in enumerate(colors):
        for time_index in range(trace_start + 1, frame_index + 1):
            if visibility[time_index - 1, point_index] and visibility[time_index, point_index]:
                p0 = tuple(np.rint(gt_scaled[time_index - 1, point_index]).astype(int))
                p1 = tuple(np.rint(gt_scaled[time_index, point_index]).astype(int))
                cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
            if predicted_scaled is not None:
                q0 = tuple(np.rint(predicted_scaled[time_index - 1, point_index]).astype(int))
                q1 = tuple(np.rint(predicted_scaled[time_index, point_index]).astype(int))
                cv2.line(canvas, q0, q1, color, 1, cv2.LINE_AA)
        if predicted_scaled is not None:
            predicted_point = tuple(np.rint(predicted_scaled[frame_index, point_index]).astype(int))
            cv2.rectangle(
                canvas,
                (predicted_point[0] - 3, predicted_point[1] - 3),
                (predicted_point[0] + 3, predicted_point[1] + 3),
                color,
                1,
                cv2.LINE_AA,
            )
        if not visibility[frame_index, point_index]:
            continue
        gt_point = tuple(np.rint(gt_scaled[frame_index, point_index]).astype(int))
        if predicted_scaled is not None:
            predicted_point = tuple(np.rint(predicted_scaled[frame_index, point_index]).astype(int))
            cv2.line(canvas, gt_point, predicted_point, (238, 238, 230), 1, cv2.LINE_AA)
        cv2.circle(canvas, gt_point, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, gt_point, 4, (8, 12, 11), 1, cv2.LINE_AA)
    label_box(canvas, title, subtitle)
    return canvas


def safe_region_name(region_name: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in region_name)


def metrics_for(result_dir: Path, layer: int, step: int, region_name: str) -> dict:
    payload = json.loads((result_dir / f"step_{step:02d}.json").read_text())
    match = [row for row in payload["rows"] if int(row["layer"]) == layer and row["region_name"] == region_name]
    if len(match) != 1:
        raise ValueError(f"Expected one metric row for {result_dir.name}/{region_name}/L{layer}/S{step}")
    return match[0]


def render_region_video(
    output_path: Path,
    frames: np.ndarray,
    anchor_frames: np.ndarray,
    gt: np.ndarray,
    visibility: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    primary_metrics: dict,
    secondary_metrics: dict,
    region_name: str,
    fps: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=7, macro_block_size=None) as writer:
        for anchor_index, pixel_frame in enumerate(anchor_frames):
            frame = frames[int(pixel_frame)]
            frame_label = f"latent {anchor_index:02d} | pixel frame {int(pixel_frame):02d}"
            gt_panel = draw_tracks(frame, gt, visibility, anchor_index, "CoTracker pseudo-GT", frame_label)
            primary_panel = draw_tracks(
                frame,
                gt,
                visibility,
                anchor_index,
                "Wan Q/K | L17 / step 49",
                f"circle=GT | square=Q/K | PCK32 {primary_metrics['pck32']:.1f}%",
                primary,
            )
            secondary_panel = draw_tracks(
                frame,
                gt,
                visibility,
                anchor_index,
                "Wan Q/K | L17 / step 36",
                f"circle=GT | square=Q/K | PCK32 {secondary_metrics['pck32']:.1f}%",
                secondary,
            )
            triptych = np.concatenate((gt_panel, primary_panel, secondary_panel), axis=1)
            cv2.putText(
                triptych,
                region_name,
                (12, PANEL_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.append_data(triptych)


def family_for(case_key: str) -> str:
    case_id = int(case_key.split("_")[1])
    return "F1" if case_id <= 15 else "F2" if case_id <= 35 else "F3"


def result_directories(root: Path) -> dict[str, Path]:
    return {path.parent.name: path.parent for path in root.glob("batch_base/worker_*/*/complete.json")}


def build_dashboard(output_dir: Path, cases: list[dict]) -> None:
    payload = json.dumps({"cases": cases}, ensure_ascii=False).replace("</", "<\\/")
    html = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan Motion Atlas · Q/K Tracking</title>
<style>
:root{--ink:#17201e;--muted:#65706c;--paper:#f2eee3;--panel:#fffdf7;--line:#d7cfbe;--red:#e04f32;--green:#117d65;--blue:#1f70b7;--gold:#c98a24;--shadow:0 18px 55px rgba(35,43,39,.12)}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 6% 0%,rgba(224,79,50,.14),transparent 31rem),radial-gradient(circle at 93% 18%,rgba(17,125,101,.13),transparent 32rem),linear-gradient(135deg,#f7f2e7,#e9e3d6);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif;min-height:100vh}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.19;background-image:linear-gradient(rgba(23,32,30,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(23,32,30,.08) 1px,transparent 1px);background-size:36px 36px;mask-image:linear-gradient(#000,transparent 72%)}
main{position:relative;width:min(1540px,calc(100% - 34px));margin:auto;padding:34px 0 70px}.hero{display:grid;grid-template-columns:1fr auto;gap:36px;align-items:end;padding:14px 0 28px}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.19em;text-transform:uppercase;color:var(--red)}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif;margin:0}h1{font-size:clamp(46px,6vw,88px);line-height:.92;letter-spacing:-.055em;margin-top:10px}.hero p{color:var(--muted);max-width:850px;line-height:1.7;margin:18px 0 0}.stamp{padding:18px 22px;border-left:1px solid var(--line);display:grid;gap:9px;font-size:13px}.stamp b{color:var(--red)}
.summary{display:grid;grid-template-columns:1.4fr repeat(3,1fr);gap:14px;margin:12px 0 24px}.card{background:rgba(255,253,247,.91);border:1px solid var(--line);border-radius:4px 20px 4px 4px;padding:19px 21px;box-shadow:var(--shadow)}.verdict{background:#1c2825;color:#fffaf0}.verdict h2{font-size:25px}.verdict p{color:#c9d2ce;line-height:1.55;margin:8px 0 0;font-size:13px}.metric span{display:block;color:var(--muted);font-size:11px;letter-spacing:.07em;text-transform:uppercase}.metric strong{display:block;font:600 37px/1 Georgia,serif;margin:10px 0 5px}.metric small{color:var(--muted)}
.toolbar{display:grid;grid-template-columns:1fr auto auto;gap:12px;align-items:end;margin:30px 0 14px}.field label{display:block;color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin:0 0 6px}.field select{width:100%;min-width:180px;border:1px solid var(--ink);background:var(--panel);padding:11px 38px 11px 12px;border-radius:3px;font:700 13px "Trebuchet MS",sans-serif;color:var(--ink)}
.viewer{background:rgba(255,253,247,.94);border:1px solid var(--line);border-radius:4px 28px 4px 4px;padding:18px;box-shadow:var(--shadow)}.viewer-head{display:flex;justify-content:space-between;gap:24px;align-items:start;padding:4px 5px 16px}.viewer h2{font-size:clamp(25px,3vw,39px);letter-spacing:-.03em}.caption{color:var(--muted);line-height:1.55;margin:6px 0 0;max-width:900px;font-size:13px}.badge{white-space:nowrap;background:var(--green);color:#fff;border-radius:999px;padding:8px 12px;font-size:11px;font-weight:800}.video-wrap{background:#111816;border-radius:3px 18px 3px 3px;overflow:hidden}.video-wrap video{display:block;width:100%;aspect-ratio:1440/264;background:#111816}.legend{display:flex;gap:18px;flex-wrap:wrap;color:#d3dad6;padding:10px 14px;background:#18211f;font-size:12px}.legend b{color:white}.legend .circle:before{content:"●";color:#ef694d}.legend .square:before{content:"□";color:#4da3ea;font-weight:900}.legend .line:before{content:"━";color:#eeeeea}
.region-metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-top:14px}.config{border-top:5px solid var(--blue)}.config.secondary{border-color:var(--gold)}.config h3{font-size:22px}.metric-row{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin-top:15px}.mini{border-top:1px solid var(--line);padding-top:9px}.mini b{display:block;font-size:16px}.mini span{color:var(--muted);font-size:10px}.aggregate{display:grid;grid-template-columns:1.25fr .75fr;gap:14px;margin-top:14px}.heat img{width:100%;display:block}.finding h3{font-size:24px}.finding p{line-height:1.68;color:#47514e;font-size:14px}.finding code{background:#ece6d9;padding:2px 5px;border-radius:3px}.foot{margin-top:34px;padding-top:15px;border-top:1px solid var(--line);display:flex;justify-content:space-between;color:var(--muted);font-size:11px}
@media(max-width:950px){main{width:min(100% - 18px,760px);padding-top:18px}.hero,.summary,.aggregate,.region-metrics{grid-template-columns:1fr}.stamp{border-left:0;border-top:1px solid var(--line)}.toolbar{grid-template-columns:1fr}.viewer-head{display:block}.badge{display:inline-block;margin-top:10px}.metric-row{grid-template-columns:repeat(2,1fr)}.video-wrap{overflow-x:auto}.video-wrap video{width:1440px;max-width:none}.foot{display:block;line-height:1.7}}
</style></head>
<body><main>
<header class="hero"><div><div class="eyebrow">Wan2.2-TI2V-5B · Motion correspondence atlas</div><h1>对象去哪了？</h1><p>50 个物理运动 case，逐对象查看 CoTracker 与 Wan self-attention Q/K 的跨时间块对应。视频严格显示 13 个原生 VAE latent 锚点，不做 13→49 时间插值。</p></div><aside class="stamp"><span>模型 <b>Wan2.2-TI2V-5B</b></span><span>主配置 <b>Layer 17 · Step 49</b></span><span>对照 <b>Layer 17 · Step 36</b></span><span>空间 token <b>32 × 32 px</b></span></aside></header>
<section class="summary"><article class="card verdict"><h2>Layer 17 是 motion correspondence 集中层</h2><p>低噪声 S49 定位最精确；S36 在中噪声下仍保持稳定。下面可逐 case 检查高速运动、碰撞、链式传递和静态背景。</p></article><article class="card metric"><span>Moving PCK@32</span><strong>90.13%</strong><small>L17 / S49 · 48 cases</small></article><article class="card metric"><span>Mean error</span><strong>20.06px</strong><small>静态 baseline 179.18px</small></article><article class="card metric"><span>Direction cosine</span><strong>0.877</strong><small>预测与 GT 运动方向</small></article></section>
<section class="toolbar"><div class="field"><label>Case</label><select id="case-select"></select></div><div class="field"><label>Family</label><select id="family-select"><option value="all">全部 F1/F2/F3</option><option value="F1">F1 · 单物体</option><option value="F2">F2 · 双物体碰撞</option><option value="F3">F3 · 链式碰撞</option></select></div><div class="field"><label>Region</label><select id="region-select"></select></div></section>
<section class="viewer"><div class="viewer-head"><div><h2 id="case-title"></h2><p class="caption" id="case-caption"></p></div><span class="badge" id="region-badge"></span></div><div class="video-wrap"><video id="track-video" controls muted loop playsinline preload="metadata"></video><div class="legend"><span class="circle"> <b>圆点/粗轨迹</b> CoTracker</span><span class="square"> <b>方框/细轨迹</b> Wan Q/K</span><span class="line"> <b>浅色连线</b> 对应误差</span></div></div><div class="region-metrics" id="region-metrics"></div></section>
<section class="aggregate"><article class="card heat"><img src="heatmap.png" alt="Layer step PCK heatmap"></article><article class="card finding"><h3>如何看这些轨迹</h3><p>优先观察运动对象：方框是否沿圆点的真实方向移动、碰撞后是否发生突跳、同一物体内部点集是否扭曲。背景的约 <code>17px</code> 误差接近 32px token 中心量化下限，不应当解释为相机或背景真的运动。</p><p>如果 S49 正确而 S36 失败，说明 correspondence 依赖低噪声外观；两者同时稳定，才是更可靠的 motion 表征。</p></article></section>
<footer class="foot"><span>49 pixel frames → 13 native Wan latent frames</span><span>CoTracker circle · Wan Q/K square · no temporal interpolation</span></footer>
</main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);const caseSelect=document.getElementById('case-select'),familySelect=document.getElementById('family-select'),regionSelect=document.getElementById('region-select'),video=document.getElementById('track-video');
const fmt=(x,d=1)=>x===null||x===undefined?'—':Number(x).toFixed(d);function filtered(){const f=familySelect.value;return data.cases.filter(x=>f==='all'||x.family===f)}
function populateCases(preferred){const list=filtered();caseSelect.innerHTML=list.map(x=>`<option value="${x.sample_key}">${x.case_key.replace('case_','').replaceAll('_',' ')} · ${x.family}</option>`).join('');if(preferred&&list.some(x=>x.sample_key===preferred))caseSelect.value=preferred;renderCase()}
function current(){return data.cases.find(x=>x.sample_key===caseSelect.value)}function renderCase(){const c=current();if(!c)return;regionSelect.innerHTML=c.regions.map((r,i)=>`<option value="${i}">${r.region_name} · ${r.motion_class}</option>`).join('');document.getElementById('case-title').textContent=c.case_key.replaceAll('_',' ');document.getElementById('case-caption').textContent=c.caption;renderRegion()}
function metricCard(title,m,secondary){return `<article class="card config ${secondary?'secondary':''}"><h3>${title}</h3><div class="metric-row"><div class="mini"><b>${fmt(m.pck32)}%</b><span>PCK@32</span></div><div class="mini"><b>${fmt(m.mean_error_px)}px</b><span>Mean error</span></div><div class="mini"><b>${fmt(m.median_error_px)}px</b><span>Median error</span></div><div class="mini"><b>${fmt(m.mean_direction_cosine,3)}</b><span>Direction</span></div><div class="mini"><b>${fmt(m.mean_gt_rank,2)}</b><span>GT token rank</span></div></div></article>`}
function renderRegion(){const c=current(),r=c.regions[Number(regionSelect.value)||0];video.src=`cases/${c.sample_key}/${r.video}`;video.load();document.getElementById('region-badge').textContent=`${r.region_name} · ${r.motion_class}`;document.getElementById('region-metrics').innerHTML=metricCard('Layer 17 · Step 49',r.primary,false)+metricCard('Layer 17 · Step 36',r.secondary,true)}
familySelect.addEventListener('change',()=>populateCases());caseSelect.addEventListener('change',renderCase);regionSelect.addEventListener('change',renderRegion);populateCases('case_019_wheel_hits_block_base');
</script></body></html>'''.replace("__PAYLOAD__", payload)
    (output_dir / "index.html").write_text(html)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dirs = result_directories(args.root)
    metadata_paths = sorted((args.root / "tracks_base").glob("case_*_base.json"))
    if len(metadata_paths) != 50 or len(result_dirs) != 50:
        raise RuntimeError(f"Expected 50 track/result cases, got {len(metadata_paths)} and {len(result_dirs)}")
    if args.case_key:
        metadata_paths = [path for path in metadata_paths if json.loads(path.read_text())["case_key"] == args.case_key]
        if len(metadata_paths) != 1:
            raise KeyError(f"Expected one case for {args.case_key}, found {len(metadata_paths)}")
    cases = []
    for case_index, metadata_path in enumerate(metadata_paths, start=1):
        metadata = json.loads(metadata_path.read_text())
        sample_key = metadata["sample_key"]
        result_dir = result_dirs[sample_key]
        with np.load(metadata_path.with_suffix(".npz")) as loaded:
            query_points = loaded["query_points"]
            anchor_frames = loaded["anchor_frames"]
            gt_tracks = loaded["anchor_tracks"]
            visibility = loaded["anchor_visibility"]
        with np.load(result_dir / "step_49.npz") as loaded:
            primary_predictions = loaded["layer_17_predictions"]
        with np.load(result_dir / "step_36.npz") as loaded:
            secondary_predictions = loaded["layer_17_predictions"]
        video_tensor = read_video(Path(metadata["video"]))
        frames = video_tensor.permute(0, 2, 3, 1).byte().numpy()
        case_regions = []
        for region in metadata["regions"]:
            point_slice = slice(region["point_start"], region["point_end"])
            primary_metrics = metrics_for(result_dir, PRIMARY["layer"], PRIMARY["step"], region["region_name"])
            secondary_metrics = metrics_for(
                result_dir, SECONDARY["layer"], SECONDARY["step"], region["region_name"]
            )
            video_name = f"{safe_region_name(region['region_name'])}.mp4"
            output_path = args.output_dir / "cases" / sample_key / video_name
            if args.overwrite or not output_path.exists():
                render_region_video(
                    output_path,
                    frames,
                    anchor_frames,
                    gt_tracks[:, point_slice],
                    visibility[:, point_slice],
                    primary_predictions[:, point_slice],
                    secondary_predictions[:, point_slice],
                    primary_metrics,
                    secondary_metrics,
                    region["region_name"],
                    args.fps,
                )
            case_regions.append(
                {
                    **region,
                    "video": video_name,
                    "primary": primary_metrics,
                    "secondary": secondary_metrics,
                }
            )
        cases.append(
            {
                "sample_key": sample_key,
                "case_key": metadata["case_key"],
                "family": family_for(metadata["case_key"]),
                "caption": metadata["caption"],
                "regions": case_regions,
                "excluded_regions": metadata.get("excluded_regions", []),
            }
        )
        print(f"[{case_index:02d}/{len(metadata_paths):02d}] {sample_key}: {len(case_regions)} regions", flush=True)
    heatmap_source = args.root / "aggregate_base" / "layer_step_pck32_heatmap.png"
    heatmap_target = args.output_dir / "heatmap.png"
    heatmap_target.write_bytes(heatmap_source.read_bytes())
    build_dashboard(args.output_dir, cases)
    (args.output_dir / "dashboard_manifest.json").write_text(
        json.dumps({"cases": cases, "primary": PRIMARY, "secondary": SECONDARY}, indent=2) + "\n"
    )
    print(f"Saved {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
