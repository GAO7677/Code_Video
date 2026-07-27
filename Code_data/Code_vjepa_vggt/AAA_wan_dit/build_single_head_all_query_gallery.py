#!/usr/bin/env python3
"""Build per-query all-key strips for one block/head and two tracked objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--gallery-root", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    return parser.parse_args()


def _page(
    *,
    block: int,
    head: int,
    valid: np.ndarray,
    query_coords: list[list[list[list[int]]]],
) -> str:
    valid_json = json.dumps(valid.astype(bool).tolist())
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Block {block:02d} Head {head:02d} all fixed queries</title>
<style>
:root{{--bg:#f3f4f1;--ink:#202421;--line:#c9cec9;--accent:#0d7155}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:3;padding:12px 18px;background:#fff;border-bottom:1px solid var(--line);box-shadow:0 2px 8px rgba(25,35,29,.08)}}
h1{{font-size:21px;margin:0 0 5px}} p{{margin:4px 0;line-height:1.4}} a{{color:var(--accent);font-weight:700;text-decoration:none}}
main{{padding:12px 16px 30px}} section{{margin-bottom:18px}} h2{{font-size:18px;margin:0 0 7px}}
.row{{display:grid;grid-template-columns:115px 224px minmax(720px,1fr);gap:8px;align-items:start;background:#fff;border:1px solid var(--line);border-radius:5px;padding:7px;margin-bottom:7px}}
.label{{font-weight:700;line-height:1.45}} canvas{{display:block;width:100%;height:auto;background:#111;image-rendering:pixelated}}
.query{{aspect-ratio:7/4}} .strip{{min-width:720px}} .unavailable{{display:grid;place-items:center;height:88px;background:#555;color:#fff;font-weight:700}}
.legend{{height:7px;margin-top:4px;background:linear-gradient(90deg,#30123b,#38598c,#1f9e89,#9fda3a,#fde725)}}
@media(max-width:1000px){{.row{{grid-template-columns:90px 168px minmax(620px,1fr);overflow-x:auto}}}}
</style>
</head>
<body>
<header>
<h1>Block {block:02d} · Head {head:02d} · 每个物体Q时刻对全部K</h1>
<p>每一行固定一个物体 Query：左侧白框是 Q token，右侧依次为 K=t0…t12。每行对全部 <code>13×16×28</code> 响应统一 min-max；无空间或时间插值。</p>
<a href="../../grouped_by_role.html">返回按类别分组页面</a>
</header>
<main>
<section><h2>Ball A · Q=t0…t12</h2><div id="rowsA"></div></section>
<section><h2>Ball B · Q=t3…t12（t0…t2不可用）</h2><div id="rowsB"></div></section>
</main>
<script>
const VALID={valid_json}, TIMES=13, H=16, W=28, FRAME=H*W, HEAD_SIZE=TIMES*TIMES*FRAME;
const queryCoords={json.dumps(query_coords, separators=(",", ":"))};
function turbo(x){{x=Math.max(0,Math.min(1,x));const s=[[48,18,59],[56,89,140],[31,158,137],[159,218,58],[253,231,37]],p=x*(s.length-1),i=Math.min(s.length-2,Math.floor(p)),a=p-i;return s[i].map((v,j)=>Math.round(v*(1-a)+s[i+1][j]*a));}}
function finiteRange(a,offset,count){{let lo=Infinity,hi=-Infinity;for(let i=0;i<count;i++){{const v=a[offset+i];if(!Number.isFinite(v))continue;if(v<lo)lo=v;if(v>hi)hi=v;}}return[lo,hi];}}
async function loadFrame(t){{const image=new Image();image.src=`../../generated_frames/frame_${{String(4*t).padStart(3,"0")}}.png`;await image.decode();return image;}}
function drawQuery(canvas,image,coords){{const c=canvas.getContext("2d");c.imageSmoothingEnabled=false;c.drawImage(image,0,0,224,128);if(!coords.length)return;const ys=coords.map(v=>v[1]),xs=coords.map(v=>v[2]);c.strokeStyle="#fff";c.lineWidth=2;c.strokeRect(Math.min(...xs)*8,Math.min(...ys)*8,(Math.max(...xs)-Math.min(...xs)+1)*8,(Math.max(...ys)-Math.min(...ys)+1)*8);}}
function drawStrip(canvas,array,q){{const c=canvas.getContext("2d"),scale=4,top=24,fw=W*scale,base=q*TIMES*FRAME,[lo,hi]=finiteRange(array,base,TIMES*FRAME);c.imageSmoothingEnabled=false;c.fillStyle="#111";c.fillRect(0,0,1456,92);for(let k=0;k<TIMES;k++){{for(let y=0;y<H;y++)for(let x=0;x<W;x++){{const v=array[base+k*FRAME+y*W+x],n=hi>lo?(v-lo)/(hi-lo):0,col=turbo(n);c.fillStyle=`rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;c.fillRect(k*fw+x*scale,top+y*scale,scale,scale);}}c.fillStyle=k===q?"#ffdf4d":"#fff";c.font=k===q?"bold 13px Arial":"12px Arial";c.fillText(`K=t${{k}}`,k*fw+5,16);if(k>0){{c.strokeStyle="rgba(255,255,255,.8)";c.lineWidth=1;c.beginPath();c.moveTo(k*fw,top);c.lineTo(k*fw,88);c.stroke();}}}}c.strokeStyle="#ffdf4d";c.lineWidth=3;c.strokeRect(q*fw+1,top+1,fw-2,62);}}
async function build(track,array,root){{for(let q=0;q<TIMES;q++){{const row=document.createElement("article");row.className="row";const valid=VALID[track][q];row.innerHTML=`<div class="label">Q=t${{q}}<br>${{valid?"13 K frames":"query unavailable"}}</div><canvas class="query" width="224" height="128"></canvas><div>${{valid?'<canvas class="strip" width="1456" height="92"></canvas><div class="legend"></div>':'<div class="unavailable">该物体在此Q时刻不存在</div>'}}</div>`;root.appendChild(row);const image=await loadFrame(q);drawQuery(row.querySelector(".query"),image,queryCoords[track][q]);if(valid)drawStrip(row.querySelector(".strip"),array,q);}}}}
Promise.all(["attention_A.f32","attention_B.f32"].map(p=>fetch(p).then(r=>r.arrayBuffer()))).then(async buffers=>{{const arrays=buffers.map(b=>new Float32Array(b));await build(0,arrays[0],document.getElementById("rowsA"));await build(1,arrays[1],document.getElementById("rowsB"));}});
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    root = args.capture_root.expanduser().resolve()
    gallery = args.gallery_root.expanduser().resolve()
    summary_path = (
        root
        / "attention"
        / f"block{args.block:02d}"
        / "matrices"
        / "wan_lora"
        / "0613pybullet_sample_001460_w002"
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entry = summary["steps"][0]
    maps_path = summary_path.parent / entry["directory"] / entry["maps_npz"]
    with np.load(maps_path) as arrays:
        attention = arrays["attention"].astype(np.float32)
        valid = arrays["valid_query_times"].astype(bool)
        coords = [
            arrays[f"track_{track}_query_coords"].astype(np.int64)
            for track in range(2)
        ]
    if attention.shape != (2, 24, 13, 13, 16, 28):
        raise ValueError(f"unexpected attention shape: {attention.shape}")
    if not 0 <= args.head < attention.shape[1]:
        raise ValueError(f"head must be in [0,{attention.shape[1] - 1}]")
    query_coords: list[list[list[list[int]]]] = []
    for track in range(2):
        query_coords.append(
            [
                coords[track][coords[track][:, 0] == time].tolist()
                for time in range(13)
            ]
        )
    output = (
        gallery / "head_details" / f"block{args.block:02d}_head{args.head:02d}"
    )
    output.mkdir(parents=True, exist_ok=True)
    for track, label in enumerate(("A", "B")):
        np.ascontiguousarray(
            attention[track, args.head], dtype="<f4"
        ).tofile(output / f"attention_{label}.f32")
    (output / "index.html").write_text(
        _page(
            block=args.block,
            head=args.head,
            valid=valid,
            query_coords=query_coords,
        ),
        encoding="utf-8",
    )
    print(f"[single-head-gallery] wrote {output / 'index.html'}")


if __name__ == "__main__":
    main()
