#!/usr/bin/env python3
"""Render fixed-A q=t0/q=t1 overlays for one Wan self-attention head."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


CASE = "0613pybullet_sample_001460_w002"
MODEL = "wan_lora"
GRID = (13, 16, 28)
FRAME_SIZE = (896, 512)
TURBO_STOPS = np.asarray(
    [
        [48, 18, 59],
        [56, 89, 140],
        [31, 158, 137],
        [159, 218, 58],
        [253, 231, 37],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/"
            "wan_dit_two_ball_attention/case001460"
        ),
    )
    parser.add_argument(
        "--gallery-root",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/wan_dit_allblock_head_roles/"
            "case001460_latent_aligned_wan_lora"
        ),
    )
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--head", type=int, default=10)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (FRAME_SIZE[1], FRAME_SIZE[0]):
            raise ValueError(f"Unexpected frame shape {frame.shape}: {path}")
        frames.append(frame)
    capture.release()
    if len(frames) != 49:
        raise ValueError(f"Expected 49 frames, found {len(frames)}: {path}")
    return frames


def video_frame_to_latent(frame_index: int) -> int:
    if frame_index == 0:
        return 0
    return 1 + (frame_index - 1) // 4


def query_frame(query_time: int) -> int:
    return 4 * query_time


def query_rect(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 1]
    columns = coords[:, 2]
    return (
        int(columns.min()) * 32,
        int(rows.min()) * 32,
        (int(columns.max()) + 1) * 32,
        (int(rows.max()) + 1) * 32,
    )


def normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Attention map contains no finite values")
    low = float(finite.min())
    high = float(finite.max())
    scale = high - low
    normalized = (
        np.zeros_like(values, dtype=np.float32)
        if scale <= 0.0
        else np.clip((values.astype(np.float32) - low) / scale, 0.0, 1.0)
    )
    return normalized, low, high


def turbo_rgb(values: np.ndarray) -> np.ndarray:
    position = np.clip(values, 0.0, 1.0) * (len(TURBO_STOPS) - 1)
    lower = np.minimum(position.astype(np.int32), len(TURBO_STOPS) - 2)
    weight = (position - lower)[..., None]
    return np.asarray(
        TURBO_STOPS[lower] * (1.0 - weight) + TURBO_STOPS[lower + 1] * weight,
        dtype=np.uint8,
    )


def overlay(frame: np.ndarray, normalized_map: np.ndarray) -> np.ndarray:
    rgb = turbo_rgb(normalized_map)
    color = np.repeat(np.repeat(rgb, 32, axis=0), 32, axis=1)[..., ::-1]
    return np.asarray(frame * 0.38 + color * 0.62, dtype=np.uint8)


def label_panel(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 62), (18, 18, 18), -1)
    cv2.putText(
        result,
        title,
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        subtitle,
        (14, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (215, 220, 216),
        1,
        cv2.LINE_AA,
    )
    return result


def query_reference(
    frame: np.ndarray,
    coords: np.ndarray,
    query_time: int,
) -> np.ndarray:
    result = frame.copy()
    x0, y0, x1, y1 = query_rect(coords)
    cv2.rectangle(result, (x0, y0), (x1, y1), (77, 223, 255), 4)
    return label_panel(
        result,
        f"Fixed A query q=t{query_time}",
        f"video frame {query_frame(query_time):02d} | {len(coords)} query tokens",
    )


def encode_h264(raw_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        candidate = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
        ffmpeg = str(candidate) if candidate.is_file() else None
    if ffmpeg is None:
        raw_path.replace(output_path)
        return
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    raw_path.unlink()


def render_video(
    *,
    frames: list[np.ndarray],
    maps: np.ndarray,
    coords: np.ndarray,
    query_time: int,
    block: int,
    head: int,
    step: int,
    output_path: Path,
) -> None:
    raw_path = output_path.with_suffix(".raw.mp4")
    raw_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (FRAME_SIZE[0] * 3, FRAME_SIZE[1]),
    )
    reference = query_reference(
        frames[query_frame(query_time)],
        coords,
        query_time,
    )
    for frame_index, frame in enumerate(frames):
        key_time = video_frame_to_latent(frame_index)
        subtitle = (
            f"Block {block:02d} Head {head:02d} | denoise {step} | "
            f"q=t{query_time} -> k=t{key_time}"
        )
        current = label_panel(
            frame,
            f"Current K frame {frame_index:02d}",
            subtitle,
        )
        attention = label_panel(
            overlay(frame, maps[key_time]),
            f"Attention overlay A(q=t{query_time}, k=t{key_time})",
            subtitle,
        )
        writer.write(np.concatenate((reference, current, attention), axis=1))
    writer.release()
    encode_h264(raw_path, output_path)


def render_strip(maps: np.ndarray, query_time: int, output_path: Path) -> None:
    scale = 4
    frame_width = GRID[2] * scale
    top = 28
    output = np.full(
        (top + GRID[1] * scale, GRID[0] * frame_width, 3),
        18,
        dtype=np.uint8,
    )
    for key_time in range(GRID[0]):
        rgb = turbo_rgb(maps[key_time])
        tile = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)[..., ::-1]
        x0 = key_time * frame_width
        output[top:, x0 : x0 + frame_width] = tile
        cv2.putText(
            output,
            f"K=t{key_time}",
            (x0 + 5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (77, 223, 255) if key_time == query_time else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if key_time:
            cv2.line(
                output,
                (x0, top),
                (x0, output.shape[0]),
                (235, 235, 235),
                1,
            )
    cv2.imwrite(str(output_path), output)


def page(manifest: dict[str, object]) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Block 00 Head 10 Fixed A q=t0/t1</title>
<style>
:root{{--bg:#f2f4f1;--panel:#fff;--ink:#202421;--line:#c9cec9;--accent:#08725a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid var(--line);padding:14px 20px;box-shadow:0 2px 8px rgba(20,30,24,.08)}}
h1{{font-size:21px;margin:0 0 6px}}p{{margin:4px 0;line-height:1.4}}a{{color:var(--accent);font-weight:700}}
.controls{{display:flex;gap:18px;align-items:end;flex-wrap:wrap;margin-top:10px}}label{{display:grid;gap:4px;font-weight:700}}
input[type=range]{{width:260px}}.value{{color:var(--accent);font-variant-numeric:tabular-nums}}
main{{padding:14px 18px 30px;max-width:1900px;margin:auto}}section{{padding:13px 0;border-bottom:1px solid var(--line)}}
h2{{font-size:16px;margin:0 0 9px}}.query-grid,.overlay-grid,.video-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
figure{{margin:0;min-width:0}}canvas,img,video{{display:block;width:100%;height:auto;background:#111}}
canvas{{image-rendering:pixelated}}figcaption{{padding:6px 2px;color:#535b56;font-size:12px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:8px}}
.legend{{height:8px;margin-top:5px;background:linear-gradient(90deg,#30123b,#38598c,#1f9e89,#9fda3a,#fde725)}}
@media(max-width:900px){{.query-grid,.overlay-grid,.video-grid{{grid-template-columns:1fr}}main{{padding:10px}}}}
</style>
</head>
<body>
<header>
<h1>Wan+LoRA · Block 00 · Head 10 · Fixed A query t0/t1</h1>
<p>case001460 · positive CFG branch · denoise step 25 · exact softmax over all 5824 K tokens</p>
<a href="../../index.html">返回全部 Block/Head 页面</a>
<div class="controls">
  <label>K latent <span class="value" id="latentValue">t0</span><input id="latent" type="range" min="0" max="12" value="0"></label>
  <label>K frame phase <span class="value" id="phaseValue">唯一帧</span><input id="phase" type="range" min="0" max="0" value="0"></label>
</div>
</header>
<main>
<section><h2>Query anchors</h2><div class="query-grid">
<figure class="panel"><canvas id="query0" width="896" height="512"></canvas><figcaption>Fixed A q=t0 · source frame 0</figcaption></figure>
<figure class="panel"><canvas id="query1" width="896" height="512"></canvas><figcaption>Fixed A q=t1 · source frame 4</figcaption></figure>
</div></section>
<section><h2 id="overlayTitle">Attention overlays</h2><div class="overlay-grid">
<figure class="panel"><canvas id="overlay0" width="896" height="512"></canvas><div class="legend"></div><figcaption>Head 10 · A(q=t0, k=t)</figcaption></figure>
<figure class="panel"><canvas id="overlay1" width="896" height="512"></canvas><div class="legend"></div><figcaption>Head 10 · A(q=t1, k=t)</figcaption></figure>
</div></section>
<section><h2>All K latent slices</h2><div class="overlay-grid">
<figure class="panel"><img src="assets/fixed_A_qt0_strip.png"><div class="legend"></div><figcaption>q=t0, independently normalized over all K tokens</figcaption></figure>
<figure class="panel"><img src="assets/fixed_A_qt1_strip.png"><div class="legend"></div><figcaption>q=t1, independently normalized over all K tokens</figcaption></figure>
</div></section>
<section><h2>49-frame overlay videos</h2><div class="video-grid">
<figure class="panel"><video controls loop muted preload="metadata" src="assets/fixed_A_qt0_overlay.mp4"></video><figcaption>Query reference | current K frame | q=t0 overlay</figcaption></figure>
<figure class="panel"><video controls loop muted preload="metadata" src="assets/fixed_A_qt1_overlay.mp4"></video><figcaption>Query reference | current K frame | q=t1 overlay</figcaption></figure>
</div></section>
</main>
<script>
const META={payload};
const stops=[[48,18,59],[56,89,140],[31,158,137],[159,218,58],[253,231,37]];
function turbo(x){{x=Math.max(0,Math.min(1,x));const p=x*(stops.length-1),i=Math.min(stops.length-2,Math.floor(p)),a=p-i;return stops[i].map((v,j)=>Math.round(v*(1-a)+stops[i+1][j]*a));}}
function frameIndex(t,p){{return t===0?0:1+4*(t-1)+p;}}
function imagePath(index){{return `../../generated_frames/frame_${{String(index).padStart(3,"0")}}.png`;}}
async function loadImage(index){{const image=new Image();image.src=imagePath(index);await image.decode();return image;}}
function drawQuery(canvas,image,coords){{const c=canvas.getContext("2d");c.imageSmoothingEnabled=false;c.drawImage(image,0,0);const ys=coords.map(q=>q[1]),xs=coords.map(q=>q[2]);c.strokeStyle="#ffdf4d";c.lineWidth=5;c.strokeRect(Math.min(...xs)*32,Math.min(...ys)*32,(Math.max(...xs)-Math.min(...xs)+1)*32,(Math.max(...ys)-Math.min(...ys)+1)*32);}}
function drawOverlay(canvas,image,array,keyTime){{const c=canvas.getContext("2d");c.imageSmoothingEnabled=false;c.drawImage(image,0,0);c.globalAlpha=.62;const base=keyTime*448;for(let y=0;y<16;y++)for(let x=0;x<28;x++){{const col=turbo(array[base+y*28+x]);c.fillStyle=`rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;c.fillRect(x*32,y*32,32,32);}}c.globalAlpha=1;}}
const state={{maps:[]}};
async function render(){{const t=+document.getElementById("latent").value,phaseEl=document.getElementById("phase");phaseEl.max=t===0?0:3;if(+phaseEl.value>+phaseEl.max)phaseEl.value=phaseEl.max;const phase=+phaseEl.value,index=frameIndex(t,phase),image=await loadImage(index);document.getElementById("latentValue").textContent=`t${{t}}`;document.getElementById("phaseValue").textContent=t===0?"唯一帧":`${{phase+1}}/4`;document.getElementById("overlayTitle").textContent=`Attention overlays · K=t${{t}} / video frame ${{index}}`;drawOverlay(document.getElementById("overlay0"),image,state.maps[0],t);drawOverlay(document.getElementById("overlay1"),image,state.maps[1],t);}}
Promise.all([
  fetch("assets/fixed_A_qt0.f32").then(r=>r.arrayBuffer()),
  fetch("assets/fixed_A_qt1.f32").then(r=>r.arrayBuffer()),
  loadImage(0),loadImage(4)
]).then(([a,b,q0,q1])=>{{state.maps=[new Float32Array(a),new Float32Array(b)];drawQuery(document.getElementById("query0"),q0,META.query_coords.t0);drawQuery(document.getElementById("query1"),q1,META.query_coords.t1);render();}});
document.getElementById("latent").addEventListener("input",()=>{{document.getElementById("phase").value=0;render();}});
document.getElementById("phase").addEventListener("input",render);
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    capture_root = args.capture_root.expanduser().resolve()
    gallery_root = args.gallery_root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else gallery_root
        / "head_details"
        / f"block{args.block:02d}_head{args.head:02d}_fixedA_t0_t1"
    )
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    matrix_path = (
        capture_root
        / "attention"
        / f"block{args.block:02d}"
        / "matrices"
        / MODEL
        / CASE
        / f"step_{args.step:02d}"
        / f"block{args.block:02d}_two_ball_maps.npz"
    )
    video_path = capture_root / "generated" / f"{CASE}.mp4"
    with np.load(matrix_path) as arrays:
        attention = arrays["attention"].astype(np.float32)
        selected_heads = arrays["selected_heads"].astype(int)
        coords = arrays["track_0_query_coords"].astype(int)
        valid = arrays["valid_query_times"].astype(bool)
        track_names = arrays["track_names"].astype(str).tolist()

    if attention.shape != (2, 24, GRID[0], GRID[0], GRID[1], GRID[2]):
        raise ValueError(f"Unexpected attention shape: {attention.shape}")
    if selected_heads.tolist() != list(range(24)):
        raise ValueError("NPZ does not contain heads 0..23 in order")
    if track_names[0] != "ball_A" or not valid[0, :2].all():
        raise ValueError("ball_A q=t0/t1 is unavailable")

    frames = read_video(video_path)
    manifest: dict[str, object] = {
        "case": CASE,
        "model": MODEL,
        "block": args.block,
        "head": args.head,
        "denoise_step_one_based": args.step,
        "cfg_branch": "positive",
        "attention_source": str(matrix_path),
        "background_video": str(video_path),
        "attention_index": (
            "attention[ball_A=0, head=10, query_time={0,1}, "
            "key_time, key_row, key_column]"
        ),
        "normalization": (
            "independent min-max over each query time's full "
            "13x16x28 attention tensor"
        ),
        "spatial_rendering": "native 16x28 tokens rendered as 32x32 cells",
        "interpolation": "none",
        "query_coords": {},
        "raw_ranges": {},
    }

    for query_time in (0, 1):
        query_coords = coords[coords[:, 0] == query_time]
        if not len(query_coords):
            raise ValueError(f"No ball_A query coordinates at t{query_time}")
        maps, low, high = normalize(attention[0, args.head, query_time])
        stem = assets / f"fixed_A_qt{query_time}"
        maps.astype("<f4").tofile(stem.with_suffix(".f32"))
        render_strip(maps, query_time, stem.with_name(stem.name + "_strip.png"))
        render_video(
            frames=frames,
            maps=maps,
            coords=query_coords,
            query_time=query_time,
            block=args.block,
            head=args.head,
            step=args.step,
            output_path=stem.with_name(stem.name + "_overlay.mp4"),
        )
        manifest["query_coords"][f"t{query_time}"] = query_coords.tolist()
        manifest["raw_ranges"][f"t{query_time}"] = {
            "min": low,
            "max": high,
            "num_query_tokens": len(query_coords),
        }

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(page(manifest), encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
