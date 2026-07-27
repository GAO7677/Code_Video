#!/usr/bin/env python3
"""Build a confidence-sorted PhysRVG all-token head gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_latent_aligned_allblock_gallery import _render_full_matrix_images


CASE = "0613pybullet_sample_001460_w002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _head_records(
    *,
    block: int,
    matrix: np.ndarray,
    exact_self: np.ndarray,
    win_rate: np.ndarray,
) -> list[dict[str, Any]]:
    records = []
    for head in range(24):
        current = matrix[head].astype(np.float64)
        diagonal = np.diag(current)
        same = float(diagonal.mean())
        other = float(
            (current.sum() - diagonal.sum())
            / (current.shape[0] * (current.shape[1] - 1))
        )
        enrichment = same / max(other, 1.0e-30)
        confidence = (same - other) / max(same + other, 1.0e-30)
        records.append(
            {
                "block": block,
                "head": head,
                "is_s": same > other,
                "label": "S_all" if same > other else "non-S",
                "confidence": confidence,
                "same": same,
                "other": other,
                "enrichment": enrichment,
                "win_rate": float(win_rate[head].mean()),
                "exact_self": float(exact_self[head].mean()),
            }
        )
    return records


def _page(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhysRVG all-token head classification</title>
<style>
:root{{--bg:#f2f4f1;--panel:#fff;--ink:#202421;--line:#c8ceca;--accent:#086b52}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:13px 20px;box-shadow:0 2px 8px rgba(20,35,28,.08)}}
h1{{font-size:21px;margin:0 0 5px}}p{{margin:4px 0;line-height:1.4}}main{{padding:14px 18px 30px}}
.controls{{display:flex;flex-wrap:wrap;align-items:end;gap:15px;margin-top:9px}}label{{display:grid;gap:4px;font-weight:700}}
select,input{{min-width:180px}}.value{{color:var(--accent);font-variant-numeric:tabular-nums}}
.reference{{max-width:560px;margin-bottom:14px;background:#fff;border:1px solid var(--line);padding:7px;border-radius:5px}}
.reference canvas{{width:100%;height:auto;display:block}}
.grid{{display:grid;grid-template-columns:1fr;gap:10px}}article{{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:9px}}
h2{{font-size:16px;margin:0 0 7px;display:flex;align-items:center;gap:5px;flex-wrap:wrap}}h3{{font-size:12px;margin:0 0 4px}}
.badge{{border:1px solid #aeb6b1;border-radius:4px;padding:2px 5px;font-size:11px;background:#f1f4f2}}.s{{background:#e2f3e9;border-color:#72ad88}}.n{{background:#eee;border-color:#999}}
.panels{{display:grid;grid-template-columns:minmax(240px,520px) minmax(240px,520px);gap:8px;align-items:start}}
canvas,.matrix{{display:block;width:100%;height:auto;background:#111;image-rendering:pixelated;aspect-ratio:1}}
.legend{{height:7px;margin-top:4px;background:linear-gradient(90deg,#30123b,#38598c,#1f9e89,#9fda3a,#fde725)}}
.placeholder{{height:90px;display:grid;place-items:center;color:#68736d;background:#f7f8f7}}
@media(max-width:720px){{.panels{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>PhyRVG · All-token Head 分类</h1>
<p>全部5824个 query 等权；逐 query 移除 exact-self并重新归一化。平均同帧质量大于平均其他单帧质量时标为 S_all。</p>
<p>去噪第25步，positive/official single call，按有符号置信度 Δ=(same-other)/(same+other) 降序。</p>
<div class="controls">
<label>显示<select id="mode"><option value="s" selected>S_all</option><option value="n">non-S</option><option value="all">全部 Head</option></select></label>
<label>Latent <span class="value" id="timeValue">t2</span><input id="time" type="range" min="0" max="12" value="2"></label>
<label>视频帧 <span class="value" id="phaseValue">4/4</span><input id="phase" type="range" min="0" max="3" value="3"></label>
<strong id="count"></strong></div></header>
<main><section class="reference"><canvas id="frame" width="896" height="512"></canvas></section><div class="grid" id="grid"></div></main>
<script>
const META={payload},cache=new Map(),imageCache=new Map(),visible=new Set();
const modeEl=document.getElementById("mode"),timeEl=document.getElementById("time"),phaseEl=document.getElementById("phase"),grid=document.getElementById("grid");let observer,epoch=0;
async function loadBlock(b){{if(cache.has(b))return cache.get(b);const r=await fetch(`data/block${{String(b).padStart(2,"0")}}_all_token_temporal.f32`);const v=new Float32Array(await r.arrayBuffer());cache.set(b,v);return v;}}
async function loadFrame(i){{if(imageCache.has(i))return imageCache.get(i);const x=new Image();x.src=`generated_frames/frame_${{String(i).padStart(3,"0")}}.png`;await x.decode();imageCache.set(i,x);return x;}}
function frameIndex(t,p){{return t===0?0:1+4*(t-1)+p;}}
function turbo(x){{x=Math.max(0,Math.min(1,x));const s=[[48,18,59],[56,89,140],[31,158,137],[159,218,58],[253,231,37]],p=x*4,i=Math.min(3,Math.floor(p)),a=p-i;return s[i].map((v,j)=>Math.round(v*(1-a)+s[i+1][j]*a));}}
function drawMatrix(c,a,o){{const x=c.getContext("2d");let lo=Infinity,hi=-Infinity;for(let i=0;i<169;i++){{const v=a[o+i];lo=Math.min(lo,v);hi=Math.max(hi,v);}}for(let q=0;q<13;q++)for(let k=0;k<13;k++){{const n=hi>lo?(a[o+q*13+k]-lo)/(hi-lo):0,col=turbo(n);x.fillStyle=`rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;x.fillRect(k*40,q*40,40,40);}}x.strokeStyle="rgba(255,255,255,.35)";x.lineWidth=1;for(let i=0;i<=13;i++){{x.beginPath();x.moveTo(i*40,0);x.lineTo(i*40,520);x.stroke();x.beginPath();x.moveTo(0,i*40);x.lineTo(520,i*40);x.stroke();}}x.strokeStyle="#ffdf4d";x.lineWidth=3;for(let i=0;i<13;i++)x.strokeRect(i*40+2,i*40+2,36,36);}}
function card(r){{const b=String(r.block).padStart(2,"0"),h=String(r.head).padStart(2,"0");return `<article data-block="${{r.block}}" data-head="${{r.head}}"><h2>Block ${{b}} · Head ${{h}} <span class="badge ${{r.is_s?"s":"n"}}">${{r.label}}</span><span class="badge">Δ=${{r.confidence.toFixed(4)}}</span><span class="badge">same=${{r.same.toFixed(4)}} · other=${{r.other.toFixed(4)}} · E=${{r.enrichment.toFixed(2)}}</span><span class="badge">win=${{r.win_rate.toFixed(3)}} · self=${{r.exact_self.toFixed(4)}}</span></h2><div class="placeholder">滚动到此处加载</div></article>`;}}
async function renderCard(a,e){{if(!a.querySelector("canvas")){{const b=String(+a.dataset.block).padStart(2,"0"),h=String(+a.dataset.head).padStart(2,"0");a.querySelector(".placeholder").outerHTML=`<div class="panels"><div><h3>精确 Q-time × K-time · exact-self removed</h3><canvas width="520" height="520"></canvas><div class="legend"></div></div><div><h3>全部5824 Q→K · 512 bins</h3><img class="matrix" loading="lazy" src="full_qk/block${{b}}/head${{h}}.png"></div></div>`;}}const data=await loadBlock(+a.dataset.block);if(e!==epoch||!visible.has(a))return;drawMatrix(a.querySelector("canvas"),data,+a.dataset.head*169);}}
function build(){{if(observer)observer.disconnect();visible.clear();const m=modeEl.value;const rows=META.heads.filter(r=>m==="all"||(m==="s"&&r.is_s)||(m==="n"&&!r.is_s)).sort((a,b)=>b.confidence-a.confidence||a.block-b.block||a.head-b.head);document.getElementById("count").textContent=`${{rows.length}} 个 Head`;grid.innerHTML=rows.map(card).join("");observer=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting){{visible.add(e.target);renderCard(e.target,epoch);}}else visible.delete(e.target);}}),{{rootMargin:"700px 0px"}});grid.querySelectorAll("article").forEach(a=>observer.observe(a));}}
async function frame(){{const t=+timeEl.value;phaseEl.max=t===0?0:3;if(+phaseEl.value>+phaseEl.max)phaseEl.value=phaseEl.max;const p=+phaseEl.value,i=frameIndex(t,p),token=++epoch,img=await loadFrame(i);if(token!==epoch)return;const c=document.getElementById("frame").getContext("2d");c.drawImage(img,0,0);document.getElementById("timeValue").textContent=`t${{t}}`;document.getElementById("phaseValue").textContent=t===0?"frame 0":`${{p+1}}/4`;for(const a of visible)renderCard(a,token);}}
modeEl.addEventListener("change",build);timeEl.addEventListener("input",()=>{{phaseEl.value=+timeEl.value===0?0:3;frame();}});phaseEl.addEventListener("input",frame);build();frame();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    root = args.capture_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    data_dir = output / "data"
    frame_dir = output / "generated_frames"
    full_dir = output / "full_qk"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        path
        for path in (root / "generated").rglob(f"{CASE}.mp4")
        if "_runtime" not in path.parts
    ]
    if len(videos) != 1:
        raise ValueError(f"expected one generated video, found: {videos}")
    video = videos[0]
    capture = cv2.VideoCapture(str(video))
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (512, 896):
            raise ValueError(f"unexpected frame shape: {frame.shape}")
        path = frame_dir / f"frame_{frame_count:03d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(path)
        frame_count += 1
    capture.release()
    if frame_count != 49:
        raise ValueError(f"expected 49 frames, found {frame_count}")

    heads = []
    for block in range(30):
        summary_path = (
            root
            / "attention"
            / f"block{block:02d}"
            / "matrices"
            / "physrvg"
            / CASE
            / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        entry = summary["steps"][0]
        npz_path = summary_path.parent / entry["directory"] / entry[
            "full_matrix_npz"
        ]
        with np.load(npz_path) as arrays:
            key_mass = arrays["key_mass"].astype(np.float32)
            matrix = arrays["time_matrix_no_exact_self"].astype(np.float32)
            exact_self = arrays["exact_self_mass"].astype(np.float32)
            win_rate = arrays["same_frame_win_rate"].astype(np.float32)
        if matrix.shape != (24, 13, 13):
            raise ValueError(f"{npz_path}: {matrix.shape}")
        matrix.astype("<f4").tofile(
            data_dir / f"block{block:02d}_all_token_temporal.f32"
        )
        _render_full_matrix_images(key_mass, block=block, output_dir=full_dir)
        heads.extend(
            _head_records(
                block=block,
                matrix=matrix,
                exact_self=exact_self,
                win_rate=win_rate,
            )
        )

    metadata = {
        "case": CASE,
        "model": "physrvg",
        "denoise_step_one_based": 25,
        "latent_grid": [13, 16, 28],
        "generated_video": str(video),
        "head_count": len(heads),
        "s_all_count": sum(record["is_s"] for record in heads),
        "protocol": {
            "queries": "all 5824 query tokens, equally weighted",
            "exact_self": "removed per query, then renormalized",
            "decision": "same-frame mean > one other-frame mean",
            "confidence": "(same-other)/(same+other)",
        },
        "heads": heads,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(_page(metadata), encoding="utf-8")
    print(f"[physrvg-gallery] wrote {output / 'index.html'}")


if __name__ == "__main__":
    main()
