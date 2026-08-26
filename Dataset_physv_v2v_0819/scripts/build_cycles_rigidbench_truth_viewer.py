#!/usr/bin/env python3
"""Build a compact browser viewer for one CYCLES/RigidBench truth case."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np


PALETTE = (
    (74, 214, 190),
    (255, 179, 84),
    (202, 126, 255),
    (92, 176, 255),
    (255, 104, 132),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    for codec in ("mp4v", "avc1"):
        path.unlink(missing_ok=True)
        handle = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if handle.isOpened():
            return handle
        handle.release()
    raise RuntimeError(f"OpenCV cannot create an MP4 writer for {path}")


def label(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 560), 34), (10, 17, 20), -1)
    cv2.putText(
        output,
        text,
        (14, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def mask_overlay(frame: np.ndarray, masks: np.ndarray, centers: np.ndarray) -> np.ndarray:
    output = frame.copy()
    for index, mask in enumerate(masks):
        if not mask.any():
            continue
        color = np.asarray(PALETTE[index % len(PALETTE)], dtype=np.uint8)
        color_layer = np.broadcast_to(color, output.shape)
        output[mask] = (
            output[mask].astype(np.float32) * 0.48 + color_layer[mask].astype(np.float32) * 0.52
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, tuple(int(value) for value in color), 2, cv2.LINE_AA)
        x, y = float(centers[index, 0]), float(centers[index, 1])
        if np.isfinite(x) and np.isfinite(y):
            point = (int(round(x)), int(round(y)))
            cv2.drawMarker(output, point, (40, 50, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.putText(
                output,
                str(index + 1),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 50, 255),
                2,
                cv2.LINE_AA,
            )
    return output


def depth_colormap(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    # Nearer geometry is warmer; invalid/background pixels stay black.
    values = (255.0 * (1.0 - normalized)).astype(np.uint8)
    output = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    output[~(np.isfinite(depth) & (depth > 0.0))] = 0
    return output


def html_page(summary: dict) -> str:
    payload = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    sample_id = html.escape(summary["sample_id"])
    prompt = html.escape(summary.get("prompt", ""))
    object_names = html.escape(", ".join(summary["dynamic_objects"]))
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CYCLES truth bench · {sample_id}</title>
<style>
:root{{--bg:#0b1114;--panel:#121c21;--panel2:#17262b;--line:#294149;--text:#e8f1ed;--muted:#8ea5a0;--mint:#6fe0c5;--amber:#f2b766;--red:#ff6978}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 80% -10%,#19443d 0,#0b1114 38rem);color:var(--text);font-family:"IBM Plex Sans","Noto Sans CJK SC",system-ui,sans-serif;line-height:1.5}}
main{{max-width:1440px;margin:0 auto;padding:34px clamp(18px,4vw,64px) 70px}}
.kicker{{color:var(--mint);font:700 11px/1.2 ui-monospace,SFMono-Regular,monospace;letter-spacing:.18em;text-transform:uppercase}}
h1{{margin:12px 0 8px;max-width:1000px;font:600 clamp(31px,5vw,68px)/.98 "Arial Narrow","Helvetica Neue",sans-serif;letter-spacing:-.04em}}
.dek{{max-width:920px;margin:0;color:var(--muted);font-size:15px}}
.meta{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:30px 0 20px}}
.metric{{padding:15px 16px;background:#101a1ee8;border:1px solid var(--line);border-top:2px solid var(--mint);border-radius:10px}}
.metric span{{display:block;color:var(--muted);font:11px ui-monospace,SFMono-Regular,monospace;text-transform:uppercase;letter-spacing:.08em}}
.metric strong{{display:block;margin-top:5px;color:var(--text);font-size:18px;font-weight:600;overflow-wrap:anywhere}}
.hero{{padding:14px;background:linear-gradient(145deg,#16292e,#0e171b);border:1px solid #345158;border-radius:16px;box-shadow:0 24px 70px #0007}}
.hero-head{{display:flex;justify-content:space-between;gap:18px;align-items:center;margin:1px 4px 12px}}
.hero-head h2{{margin:0;font-size:14px;font-weight:600;letter-spacing:.06em}}.hero-head small{{color:var(--muted);font:11px ui-monospace,SFMono-Regular,monospace}}
video{{display:block;width:100%;background:#050809;border-radius:10px;outline:1px solid #294149}}
.controlbar{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:13px 4px 2px}}
button{{border:1px solid #3a5b62;background:#1b3034;color:var(--text);border-radius:7px;padding:9px 12px;cursor:pointer;font:600 12px ui-monospace,SFMono-Regular,monospace}}
button.primary{{border-color:var(--mint);color:#081311;background:#6fe0c5}}button:hover{{filter:brightness(1.1)}}button:focus-visible{{outline:2px solid var(--amber);outline-offset:2px}}
input[type=range]{{flex:1;min-width:180px;accent-color:var(--mint)}}#frame-label{{min-width:120px;color:var(--amber);font:700 12px ui-monospace,SFMono-Regular,monospace;text-align:right}}
.views{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:18px}}
.view{{min-width:0;padding:10px;background:var(--panel);border:1px solid var(--line);border-radius:12px}}.view h3{{margin:0 0 8px;color:var(--mint);font:700 12px ui-monospace,SFMono-Regular,monospace;letter-spacing:.07em}}.view p{{margin:8px 2px 0;color:var(--muted);font-size:12px}}
.note{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}}.note section{{padding:16px;background:#101a1e;border:1px solid var(--line);border-radius:12px}}.note h2{{margin:0 0 8px;color:var(--amber);font-size:13px}}.note p{{margin:0;color:var(--muted);font-size:13px;overflow-wrap:anywhere}}code{{color:var(--mint);font-family:ui-monospace,SFMono-Regular,monospace}}
@media(max-width:880px){{.meta{{grid-template-columns:repeat(2,minmax(0,1fr))}}.views,.note{{grid-template-columns:1fr}}}}
@media(max-width:500px){{.meta{{grid-template-columns:1fr 1fr}}h1{{font-size:38px}}}}
@media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important}}}}
</style></head>
<body><main>
<div class="kicker">CYCLES / RIGIDBENCH TRUTH BENCH</div>
<h1>像素对齐的真值检查台</h1>
<p class="dek">同一帧同步检查 CYCLES RGB、IndexOB 动态物体 mask、Depth/Z pass。红色十字是轨迹中心投影；颜色填充只用于可视化真值，不是模型预测。</p>
<div class="meta">
  <div class="metric"><span>sample</span><strong>{sample_id}</strong></div>
  <div class="metric"><span>frames / fps</span><strong>{summary["frame_count"]} / {summary["fps"]}</strong></div>
  <div class="metric"><span>resolution</span><strong>{summary["width"]} × {summary["height"]}</strong></div>
  <div class="metric"><span>active actors</span><strong>{object_names}</strong></div>
</div>
<section class="hero"><div class="hero-head"><h2>三路同步总览 · RGB / MASK / DEPTH</h2><small id="time-label">0.000 s</small></div>
  <video id="composite" controls muted playsinline preload="metadata" src="assets/truth_composite.mp4"></video>
  <div class="controlbar"><button class="primary" id="play">播放全部</button><button id="pause">暂停</button><button id="back">上一帧</button><button id="forward">下一帧</button><input id="scrub" type="range" min="0" max="{summary["frame_count"] - 1}" value="0" step="1" aria-label="帧号"><span id="frame-label">frame 000 / {summary["frame_count"] - 1:03d}</span></div>
</section>
<section class="views">
  <article class="view"><h3>01 · RGB CYCLES</h3><video class="sync" controls muted playsinline preload="metadata" src="assets/rgb_cycles.mp4"></video><p>原始 CYCLES 渲染视频，作为像素坐标参考。</p></article>
  <article class="view"><h3>02 · INDEXOB MASK</h3><video class="sync" controls muted playsinline preload="metadata" src="assets/mask_overlay.mp4"></video><p>动态 actor GT mask；红色十字为 `trajectory_pixels.npz` 中心投影。</p></article>
  <article class="view"><h3>03 · DEPTH / Z PASS</h3><video class="sync" controls muted playsinline preload="metadata" src="assets/depth_colormap.mp4"></video><p>同一 CYCLES 相机的 Depth/Z pass 伪彩，近处偏暖，背景/无效值为黑。</p></article>
</section>
<section class="note"><section><h2>Caption</h2><p>{prompt}</p></section><section><h2>Files</h2><p><code>cycles_depth.npz</code> · <code>dynamic_masks.npz</code> · <code>trajectory_pixels.npz</code> · <code>rigidbench/</code></p></section></section>
</main>
<script>
const summary={payload};
const composite=document.querySelector('#composite');
const videos=[composite,...document.querySelectorAll('.sync')];
const scrub=document.querySelector('#scrub');
const frameLabel=document.querySelector('#frame-label');
const timeLabel=document.querySelector('#time-label');
let syncing=false;
function setTime(t){{
  syncing=true;
  videos.forEach(v=>{{if(v.readyState>0 && Math.abs(v.currentTime-t)>0.035)v.currentTime=t;}});
  syncing=false;
  const frame=Math.max(0,Math.min(summary.frame_count-1,Math.round(t*summary.fps)));
  scrub.value=frame; frameLabel.textContent=`frame ${{String(frame).padStart(3,'0')}} / ${{String(summary.frame_count-1).padStart(3,'0')}}`; timeLabel.textContent=`${{(frame/summary.fps).toFixed(3)}} s`;
}}
composite.addEventListener('timeupdate',()=>{{if(!syncing)setTime(composite.currentTime);}});
scrub.addEventListener('input',()=>setTime(Number(scrub.value)/summary.fps));
document.querySelector('#play').addEventListener('click',()=>{{videos.forEach(v=>v.play().catch(()=>{{}}));}});
document.querySelector('#pause').addEventListener('click',()=>videos.forEach(v=>v.pause()));
document.querySelector('#back').addEventListener('click',()=>{{videos.forEach(v=>v.pause());setTime(Math.max(0,composite.currentTime-1/summary.fps));}});
document.querySelector('#forward').addEventListener('click',()=>{{videos.forEach(v=>v.pause());setTime(Math.min(composite.duration||summary.frame_count/summary.fps,composite.currentTime+1/summary.fps));}});
videos.forEach(v=>v.addEventListener('play',()=>{{if(v!==composite){{composite.currentTime=v.currentTime;}}}}));
</script></body></html>'''


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.expanduser().resolve()
    output_dir = (args.output_dir or case_dir / "visualization").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    with np.load(case_dir / "dynamic_masks.npz", allow_pickle=False) as arrays:
        masks = arrays["masks_thw"]
        names = [str(value) for value in arrays["object_names"]]
    with np.load(case_dir / "trajectory_pixels.npz", allow_pickle=False) as arrays:
        centers = arrays["centers_tnc"]
    depth = np.load(case_dir / "cycles_depth.npz", allow_pickle=False)["depth"]
    metadata = json.loads((case_dir / "truth_metadata.json").read_text(encoding="utf-8"))
    adapter_metadata_path = case_dir / "rigidbench" / "metadata.json"
    adapter_metadata = (
        json.loads(adapter_metadata_path.read_text(encoding="utf-8"))
        if adapter_metadata_path.is_file()
        else {}
    )
    source_video = Path(metadata["source_rgb_cycles"])

    capture = cv2.VideoCapture(str(source_video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or metadata.get("fps", 30))
    frame_count = int(masks.shape[1])
    height, width = int(masks.shape[2]), int(masks.shape[3])
    depth_values = depth[np.isfinite(depth) & (depth > 0.0)]
    low, high = np.percentile(depth_values, [1.0, 99.0]).tolist()
    composite_writer = writer(assets / "truth_composite.mp4", width * 3, height, fps)
    mask_writer = writer(assets / "mask_overlay.mp4", width, height, fps)
    depth_writer = writer(assets / "depth_colormap.mp4", width, height, fps)
    frames_written = 0
    try:
        while frames_written < frame_count:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"RGB video ended at frame {frames_written}, expected {frame_count}")
            rgb = label(frame, "RGB · CYCLES", (220, 240, 235))
            mask = label(
                mask_overlay(frame, masks[:, frames_written], centers[frames_written]),
                "MASK · INDEXOB / ACTIVE ACTOR",
                (111, 224, 197),
            )
            depth_frame = label(
                depth_colormap(depth[frames_written], low, high),
                f"DEPTH · Z PASS  {low:.2f}–{high:.2f} m",
                (242, 183, 102),
            )
            composite_writer.write(np.concatenate([rgb, mask, depth_frame], axis=1))
            mask_writer.write(mask)
            depth_writer.write(depth_frame)
            frames_written += 1
    finally:
        capture.release()
        composite_writer.release()
        mask_writer.release()
        depth_writer.release()

    source_link = assets / "rgb_cycles.mp4"
    source_link.unlink(missing_ok=True)
    source_link.symlink_to(source_video)
    summary = {
        "sample_id": str(metadata["sample_id"]),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "dynamic_objects": names,
        "depth_percentile_range_m": [float(low), float(high)],
        "prompt": (
            metadata.get("prompt")
            or adapter_metadata.get("prompt")
            or (adapter_metadata.get("captions") or {}).get("abstract", {}).get("text", "")
        ),
        "source_rgb_cycles": str(source_video),
        "truth_case": str(case_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(html_page(summary), encoding="utf-8")
    print(json.dumps({**summary, "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
