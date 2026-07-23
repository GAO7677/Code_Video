#!/usr/bin/env python3
"""Analyze adjacent-frame DINOv3 motion with cross-phase and phase correlation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

from analyze_dinov3_feature_frequency_temporal import extract_dinov3_features  # noqa: E402
from analyze_slot_temporal_similarity_viewer import (  # noqa: E402
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_VIEWER_DIR,
    normalize_rgb_frames,
    read_frame_sequence,
)
from object_centric_bench.model.dinov3_backbone import DINO3ViT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch-frames", type=int, default=16)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def radial_bins(height: int, width: int, count: int = 12) -> tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(yy**2 + xx**2)
    edges = np.linspace(0.0, float(radius.max()) + 1e-8, count + 1)
    ids = np.digitize(radius.reshape(-1), edges) - 1
    ids = np.clip(ids, 0, count - 1).reshape(height, width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return ids, centers


def analyze_cross_phase(features: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    values = features.astype(np.float64)
    frames, channels, height, width = values.shape
    if frames < 2:
        raise ValueError(f"At least two frames are required, got {frames}")

    eps = np.finfo(np.float64).eps
    normalized = values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True),
        eps,
    )
    patch_change = 1.0 - np.sum(normalized[1:] * normalized[:-1], axis=1)
    patch_change = np.clip(patch_change, 0.0, 2.0)

    centered = values - values.mean(axis=(-2, -1), keepdims=True)
    spatial_window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.fft.fft2(centered * spatial_window[None, None])
    cross = spectrum[1:] * np.conjugate(spectrum[:-1])
    cross_magnitude_sum = np.abs(cross).sum(axis=1)
    cross_sum = cross.sum(axis=1)
    consensus_magnitude = np.abs(cross_sum)

    cross_phase = np.fft.fftshift(np.angle(cross_sum), axes=(-2, -1))
    cross_agreement = np.fft.fftshift(
        consensus_magnitude / np.maximum(cross_magnitude_sum, eps),
        axes=(-2, -1),
    )

    phase_only = cross_sum / np.maximum(consensus_magnitude, eps)
    phase_correlation = np.abs(np.fft.ifft2(phase_only))
    phase_correlation = np.fft.fftshift(phase_correlation, axes=(-2, -1))
    phase_correlation /= np.maximum(
        phase_correlation.sum(axis=(-2, -1), keepdims=True),
        eps,
    )

    flat_peak = phase_correlation.reshape(frames - 1, -1).argmax(axis=1)
    peak_y, peak_x = np.unravel_index(flat_peak, (height, width))
    shift_y = peak_y.astype(np.float64) - height // 2
    shift_x = peak_x.astype(np.float64) - width // 2
    shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)
    peak_value = phase_correlation[np.arange(frames - 1), peak_y, peak_x]
    peak_to_mean = peak_value / np.maximum(
        phase_correlation.mean(axis=(-2, -1)),
        eps,
    )

    phase_agreement = consensus_magnitude.sum(axis=(-2, -1)) / np.maximum(
        cross_magnitude_sum.sum(axis=(-2, -1)),
        eps,
    )
    phase_activity = (
        consensus_magnitude * (np.abs(np.angle(cross_sum)) / np.pi)
    ).sum(axis=(-2, -1)) / np.maximum(
        consensus_magnitude.sum(axis=(-2, -1)),
        eps,
    )

    radial_ids, radial_frequency = radial_bins(height, width)
    shifted_consensus = np.fft.fftshift(consensus_magnitude, axes=(-2, -1))
    radial_phase_activity = np.zeros((frames - 1, len(radial_frequency)), dtype=np.float64)
    shifted_phase_activity = np.abs(cross_phase) / np.pi
    for bin_id in range(len(radial_frequency)):
        mask = radial_ids == bin_id
        weights = shifted_consensus[:, mask]
        radial_phase_activity[:, bin_id] = (
            weights * shifted_phase_activity[:, mask]
        ).sum(axis=1) / np.maximum(weights.sum(axis=1), eps)

    patch_change_mean = patch_change.mean(axis=(-2, -1))
    patch_change_p95 = float(np.percentile(patch_change, 95))
    arrays = {
        "patch_change": patch_change.astype(np.float32),
        "patch_change_mean": patch_change_mean.astype(np.float32),
        "cross_phase": cross_phase.astype(np.float32),
        "cross_agreement": cross_agreement.astype(np.float32),
        "phase_correlation": phase_correlation.astype(np.float32),
        "shift_x": shift_x.astype(np.float32),
        "shift_y": shift_y.astype(np.float32),
        "shift_magnitude": shift_magnitude.astype(np.float32),
        "peak_to_mean": peak_to_mean.astype(np.float32),
        "phase_agreement": phase_agreement.astype(np.float32),
        "phase_activity": phase_activity.astype(np.float32),
        "radial_frequency": radial_frequency.astype(np.float32),
        "radial_phase_activity": radial_phase_activity.astype(np.float32),
    }
    summary = {
        "frames": int(frames),
        "transitions": int(frames - 1),
        "channels": int(channels),
        "grid_h": int(height),
        "grid_w": int(width),
        "patch_change_mean": float(patch_change_mean.mean()),
        "patch_change_p95": patch_change_p95,
        "shift_magnitude_mean_patches": float(shift_magnitude.mean()),
        "shift_magnitude_p95_patches": float(np.percentile(shift_magnitude, 95)),
        "nonzero_shift_fraction": float(np.mean(shift_magnitude > 0)),
        "phase_agreement_mean": float(phase_agreement.mean()),
        "phase_activity_mean": float(phase_activity.mean()),
        "phase_correlation_peak_to_mean": float(peak_to_mean.mean()),
    }
    return summary, arrays


def plot_summary(
    arrays: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    destination_frame = np.arange(1, len(arrays["shift_x"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.0), dpi=145)

    axes[0, 0].plot(destination_frame, arrays["shift_x"], label="dx", color="#2563eb")
    axes[0, 0].plot(destination_frame, arrays["shift_y"], label="dy", color="#dc2626")
    axes[0, 0].axhline(0.0, color="#777", linewidth=0.8)
    axes[0, 0].set_title("Phase-correlation shift")
    axes[0, 0].set_ylabel("patches / frame")
    axes[0, 0].legend()

    axes[0, 1].plot(destination_frame, arrays["shift_magnitude"], color="#7c3aed")
    axes[0, 1].set_title("Global shift magnitude")
    axes[0, 1].set_ylabel("patches / frame")

    axes[0, 2].plot(destination_frame, arrays["patch_change_mean"], color="#ea580c")
    axes[0, 2].set_title("DINO patch feature change")
    axes[0, 2].set_ylabel("mean cosine distance")

    axes[1, 0].plot(destination_frame, arrays["phase_agreement"], color="#0891b2")
    axes[1, 0].plot(destination_frame, arrays["phase_activity"], color="#db2777")
    axes[1, 0].set_title("Cross-phase")
    axes[1, 0].set_ylabel("agreement / activity")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend(("channel agreement", "phase activity"))

    axes[1, 1].plot(destination_frame, arrays["peak_to_mean"], color="#16a34a")
    axes[1, 1].set_title("Phase-correlation peak sharpness")
    axes[1, 1].set_ylabel("peak / map mean")

    heatmap = axes[1, 2].imshow(
        arrays["radial_phase_activity"].T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        extent=[
            1,
            len(destination_frame),
            float(arrays["radial_frequency"][0]),
            float(arrays["radial_frequency"][-1]),
        ],
    )
    axes[1, 2].set_title("Framewise cross-phase activity")
    axes[1, 2].set_ylabel("spatial frequency (cycles/patch)")
    fig.colorbar(heatmap, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for axis in axes.flat:
        axis.set_xlabel("destination frame")
        axis.grid(alpha=0.2)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_result(
    output_path: Path,
    summary: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output_path.with_suffix(".npz"), **arrays)
    browser_payload = {
        "summary": summary,
        "arrays": {
            key: np.round(value, 5).tolist()
            for key, value in arrays.items()
            if key
            in {
                "patch_change",
                "patch_change_mean",
                "cross_phase",
                "cross_agreement",
                "phase_correlation",
                "shift_x",
                "shift_y",
                "shift_magnitude",
                "peak_to_mean",
                "phase_agreement",
                "phase_activity",
            }
        },
    }
    output_path.with_name("motion_data.json").write_text(
        json.dumps(browser_payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def stage_link(link_path: Path, source_path: Path, is_directory: bool = False) -> None:
    source_path = source_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() != source_path:
            raise FileExistsError(f"Unexpected link target: {link_path}")
    elif link_path.exists():
        raise FileExistsError(f"Refusing to replace existing artifact: {link_path}")
    else:
        link_path.symlink_to(source_path, target_is_directory=is_directory)


def build_html(metadata: dict) -> str:
    data = json.dumps(metadata, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DINOv3 Framewise Cross-Phase Motion</title>
<style>
body{{margin:0;color:#171717;background:#fff;font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;background:rgba(255,255,255,.97);border-bottom:1px solid #d4d4d4;z-index:4}}
.bar{{max-width:1440px;margin:auto;padding:12px 20px;display:flex;gap:16px;align-items:end;flex-wrap:wrap}}
h1{{font-size:19px;margin:0 auto 1px 0;letter-spacing:0}}
label{{display:grid;gap:4px;color:#666;font-size:12px;font-weight:650}}
select{{min-width:210px;height:36px;border:1px solid #aaa;border-radius:5px;background:#fff;padding:0 10px}}
input[type=range]{{width:min(380px,70vw)}}
main{{max-width:1440px;margin:auto;padding:18px 20px 32px}}
.video-panel{{max-width:640px;margin:0 auto 18px}}
.video-panel h2,.section-title{{font-size:15px;margin:0 0 8px;letter-spacing:0}}
video{{display:block;width:100%;max-height:44vh;background:#111}}
.summary{{display:block;width:100%;height:auto;border:1px solid #d4d4d4;border-radius:6px}}
.frame-head{{display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin:20px 0 8px}}
.frame-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
.panel{{border:1px solid #d4d4d4;border-radius:6px;overflow:hidden;background:#fafafa}}
.panel h3{{font-size:13px;margin:0;padding:9px 10px;border-bottom:1px solid #ddd}}
.visual{{display:block;width:100%;aspect-ratio:1;background:#111;object-fit:contain}}
.panel .value{{min-height:20px;padding:7px 10px;color:#555;font-variant-numeric:tabular-nums}}
table{{width:100%;margin-top:18px;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 10px;border-bottom:1px solid #d4d4d4;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
thead th{{background:#f5f5f4;color:#444;font-size:12px}}
.method{{margin-top:10px;color:#666;font-size:12px}}
@media(max-width:900px){{.frame-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:540px){{.frame-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header><div class="bar">
  <h1>DINOv3 Framewise Cross-Phase Motion</h1>
  <label>Preprocessing<select id="mode"><option value="crop">Center crop</option><option value="padding">Resize + padding</option></select></label>
  <label>Case<select id="case"></select></label>
  <label>Destination frame <input id="frame" type="range" min="1" value="1" step="1"></label>
</div></header>
<main>
  <section class="video-panel">
    <h2>Source video</h2>
    <video id="video" controls playsinline preload="metadata"></video>
  </section>
  <h2 class="section-title">Motion summary</h2>
  <img id="summary" class="summary" alt="Cross-phase motion summary">
  <div class="frame-head"><h2 class="section-title">Framewise analysis</h2><span id="transition"></span></div>
  <section class="frame-grid">
    <article class="panel"><h3>Analyzed frame</h3><img id="source-frame" class="visual" alt="Analyzed frame"><div id="source-value" class="value"></div></article>
    <article class="panel"><h3>DINO patch change overlay</h3><canvas id="change-map" class="visual" width="320" height="320"></canvas><div id="change-value" class="value"></div></article>
    <article class="panel"><h3>Cross-phase angle</h3><canvas id="phase-map" class="visual" width="320" height="320"></canvas><div id="phase-value" class="value"></div></article>
    <article class="panel"><h3>Phase-correlation displacement</h3><canvas id="correlation-map" class="visual" width="320" height="320"></canvas><div id="correlation-value" class="value"></div></article>
  </section>
  <table><thead><tr><th>Frames</th><th>Patch change</th><th>Mean shift</th><th>P95 shift</th><th>Nonzero shift</th><th>Phase agreement</th><th>Phase activity</th><th>Peak sharpness</th></tr></thead><tbody id="metrics"></tbody></table>
  <div class="method">Adjacent frozen DINOv3 features are spatially mean-centered and Hann-windowed. Cross-phase and phase correlation are computed on the 16x16 patch grid; displacement is therefore measured in patch units.</div>
</main>
<script>
const DATA={data};
const mode=document.getElementById('mode');
const caseSelect=document.getElementById('case');
const frameSlider=document.getElementById('frame');
const summaryImage=document.getElementById('summary');
const video=document.getElementById('video');
const sourceFrame=document.getElementById('source-frame');
const transition=document.getElementById('transition');
const metrics=document.getElementById('metrics');
const cache=new Map();
DATA.cases.forEach(item=>{{const option=document.createElement('option');option.value=item.id;option.textContent=item.label;caseSelect.appendChild(option);}});

function padFrame(value){{return String(value).padStart(4,'0');}}
function magma(value){{
  const v=Math.max(0,Math.min(1,value));
  const stops=[[0,[0,0,4]],[.25,[81,18,124]],[.5,[183,55,121]],[.75,[252,137,97]],[1,[252,253,191]]];
  for(let i=1;i<stops.length;i++){{if(v<=stops[i][0]){{const a=stops[i-1],b=stops[i],q=(v-a[0])/(b[0]-a[0]);return a[1].map((x,j)=>Math.round(x+q*(b[1][j]-x)));}}}}
  return stops.at(-1)[1];
}}
function drawScalarMap(canvas,map,peak=null){{
  const ctx=canvas.getContext('2d');const h=map.length,w=map[0].length;const cell=canvas.width/w;
  const maxValue=Math.max(...map.flat(),1e-8);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){{const c=magma(map[y][x]/maxValue);ctx.fillStyle=`rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;ctx.fillRect(x*cell,y*cell,cell+.5,cell+.5);}}
  ctx.strokeStyle='rgba(255,255,255,.7)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(canvas.width/2,0);ctx.lineTo(canvas.width/2,canvas.height);ctx.moveTo(0,canvas.height/2);ctx.lineTo(canvas.width,canvas.height/2);ctx.stroke();
  if(peak){{const px=(peak.x+.5)*cell,py=(peak.y+.5)*cell;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(canvas.width/2,canvas.height/2);ctx.lineTo(px,py);ctx.stroke();ctx.beginPath();ctx.arc(px,py,Math.max(4,cell*.32),0,Math.PI*2);ctx.stroke();}}
}}
function drawPhaseMap(canvas,phase,agreement){{
  const ctx=canvas.getContext('2d');const h=phase.length,w=phase[0].length;const cell=canvas.width/w;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){{const hue=(phase[y][x]+Math.PI)/(2*Math.PI)*360;const light=14+agreement[y][x]*58;ctx.fillStyle=`hsl(${{hue}} 82% ${{light}}%)`;ctx.fillRect(x*cell,y*cell,cell+.5,cell+.5);}}
  ctx.strokeStyle='rgba(255,255,255,.72)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(canvas.width/2,0);ctx.lineTo(canvas.width/2,canvas.height);ctx.moveTo(0,canvas.height/2);ctx.lineTo(canvas.width,canvas.height/2);ctx.stroke();
}}
function drawChangeOverlay(url,map,scale){{
  const canvas=document.getElementById('change-map');const ctx=canvas.getContext('2d');const image=new Image();
  image.onload=()=>{{ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);const h=map.length,w=map[0].length,cw=canvas.width/w,ch=canvas.height/h;for(let y=0;y<h;y++)for(let x=0;x<w;x++){{const v=Math.max(0,Math.min(1,map[y][x]/Math.max(scale,1e-6)));if(v<.08)continue;const c=magma(v);ctx.fillStyle=`rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{.16+.62*v}})`;ctx.fillRect(x*cw,y*ch,cw+.5,ch+.5);}}}};
  image.src=url;
}}
async function loadEntry(){{
  const entry=DATA.entries[caseSelect.value][mode.value];const key=entry.data;
  if(!cache.has(key))cache.set(key,await fetch(key).then(response=>{{if(!response.ok)throw new Error(`HTTP ${{response.status}}`);return response.json();}}));
  return [entry,cache.get(key)];
}}
async function renderEntry(){{
  const [entry,payload]=await loadEntry();const m=payload.summary;
  summaryImage.src=entry.chart;
  if(video.dataset.src!==entry.video){{video.dataset.src=entry.video;video.src=entry.video;video.load();}}
  frameSlider.max=m.frames-1;frameSlider.value=Math.min(Number(frameSlider.value)||1,m.frames-1);
  metrics.innerHTML=`<tr><td>${{m.frames}}</td><td>${{m.patch_change_mean.toFixed(4)}}</td><td>${{m.shift_magnitude_mean_patches.toFixed(3)}}</td><td>${{m.shift_magnitude_p95_patches.toFixed(3)}}</td><td>${{(100*m.nonzero_shift_fraction).toFixed(1)}}%</td><td>${{m.phase_agreement_mean.toFixed(3)}}</td><td>${{m.phase_activity_mean.toFixed(3)}}</td><td>${{m.phase_correlation_peak_to_mean.toFixed(2)}}</td></tr>`;
  renderFrame(entry,payload);
}}
function renderFrame(entry,payload){{
  const frame=Math.max(1,Number(frameSlider.value));const index=frame-1;const a=payload.arrays;const m=payload.summary;
  const frameUrl=entry.frame_pattern.replace('{{frame}}',padFrame(frame));
  sourceFrame.src=frameUrl;transition.textContent=`frame ${{frame-1}} → ${{frame}}`;
  document.getElementById('source-value').textContent=`destination frame ${{frame}}`;
  drawChangeOverlay(frameUrl,a.patch_change[index],m.patch_change_p95);
  document.getElementById('change-value').textContent=`mean cosine distance ${{a.patch_change_mean[index].toFixed(4)}}`;
  drawPhaseMap(document.getElementById('phase-map'),a.cross_phase[index],a.cross_agreement[index]);
  document.getElementById('phase-value').textContent=`agreement ${{a.phase_agreement[index].toFixed(3)}} · activity ${{a.phase_activity[index].toFixed(3)}}`;
  const corr=a.phase_correlation[index],flat=corr.flat(),peakIndex=flat.indexOf(Math.max(...flat)),w=corr[0].length;
  drawScalarMap(document.getElementById('correlation-map'),corr,{{x:peakIndex%w,y:Math.floor(peakIndex/w)}});
  document.getElementById('correlation-value').textContent=`dx ${{a.shift_x[index].toFixed(1)}} · dy ${{a.shift_y[index].toFixed(1)}} patches · sharpness ${{a.peak_to_mean[index].toFixed(1)}}`;
}}
[mode,caseSelect].forEach(element=>element.addEventListener('change',()=>{{frameSlider.value=1;renderEntry();}}));
frameSlider.addEventListener('input',async()=>{{const [entry,payload]=await loadEntry();renderFrame(entry,payload);}});
renderEntry();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else viewer_dir / "dinov3_framewise_cross_phase"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = json.loads(
        (viewer_dir / "combined_metadata.json").read_text(encoding="utf-8")
    )
    cases = (
        combined["cases"][: args.max_cases]
        if args.max_cases > 0
        else combined["cases"]
    )

    jobs = []
    for case in cases:
        case_id = case["case_id"]
        source_video = Path(case["crop"]["source_key"])
        video_link = output_dir / "videos" / f"{case_id}{source_video.suffix or '.mp4'}"
        stage_link(video_link, source_video)
        for mode, source_key in (("crop", "crop_dir"), ("padding", "padding_dir")):
            frame_root = outputs_root / combined[source_key] / "cases" / case_id / "original"
            frame_link = output_dir / "frames" / case_id / mode
            stage_link(frame_link, frame_root, is_directory=True)
            output_path = output_dir / "cases" / case_id / mode / "motion_summary.png"
            jobs.append((case, mode, frame_root, output_path, video_link, frame_link))

    needs_backbone = args.force or any(
        not (
            output_path.is_file()
            and output_path.with_suffix(".json").is_file()
            and output_path.with_suffix(".npz").is_file()
            and output_path.with_name("motion_data.json").is_file()
        )
        for _, _, _, output_path, _, _ in jobs
    )
    backbone = None
    device = torch.device(args.device)
    amp_dtype = getattr(torch, args.amp_dtype)
    if needs_backbone:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        backbone = DINO3ViT(rearrange=True, norm_out=False).to(device).eval()

    entries: dict[str, dict] = {}
    for job_index, (case, mode, frame_root, output_path, video_link, frame_link) in enumerate(
        jobs,
        start=1,
    ):
        case_id = case["case_id"]
        entries.setdefault(case_id, {})
        complete = (
            output_path.is_file()
            and output_path.with_suffix(".json").is_file()
            and output_path.with_suffix(".npz").is_file()
            and output_path.with_name("motion_data.json").is_file()
        )
        if complete and not args.force:
            summary = json.loads(
                output_path.with_suffix(".json").read_text(encoding="utf-8")
            )
        else:
            rgb = read_frame_sequence(frame_root, int(case["frames"]))
            video_tensor = normalize_rgb_frames(rgb)
            if backbone is None:
                raise RuntimeError("DINOv3 backbone was not initialized")
            features = extract_dinov3_features(
                backbone,
                video_tensor,
                device,
                amp_dtype,
                args.batch_frames,
            )
            summary, arrays = analyze_cross_phase(features)
            summary.update(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "backbone": "dinov3_vitl16_lvd1689m",
                    "feature_shape": list(features.shape),
                }
            )
            plot_summary(
                arrays,
                output_path,
                f"{case_id} | {mode} | DINOv3 framewise cross-phase",
            )
            save_result(output_path, summary, arrays)

        entries[case_id][mode] = {
            "chart": str(output_path.relative_to(output_dir)),
            "data": str(output_path.with_name("motion_data.json").relative_to(output_dir)),
            "video": str(video_link.relative_to(output_dir)),
            "frame_pattern": str(frame_link.relative_to(output_dir)) + "/{frame}.webp",
            "metrics": summary,
        }
        print(
            f"[case] {job_index}/{len(jobs)} {case_id} {mode}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Adjacent frozen DINOv3 feature grids are spatially mean-centered and "
            "Hann-windowed. Multi-channel cross-power phase provides framewise "
            "phase-correlation displacement on the 16x16 patch grid."
        ),
        "cases": [
            {
                "id": case["case_id"],
                "label": f"{index:02d} | {case['case_id']}",
            }
            for index, case in enumerate(cases, start=1)
        ],
        "entries": entries,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        build_html(metadata),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# DINOv3 framewise cross-phase motion analysis\n\n"
        "For every adjacent frame pair, this analysis computes DINO patch cosine "
        "change, multi-channel spatial cross-phase, and phase correlation. A rigid "
        "translation produces a phase ramp and an off-center phase-correlation peak. "
        "Small independently moving objects can be dominated by static background in "
        "the global displacement estimate, so the patch-change overlay and peak "
        "sharpness must be inspected together.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index": str(output_dir / "index.html"),
                "cases": len(cases),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
