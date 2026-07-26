#!/usr/bin/env python3
"""Build cross-model overlays for selected common moving-ball query heads."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from overlay_multiblock_ball_query_heads import (
    MODEL_LABELS,
    ROLE_NAMES_ZH,
    _generated_video,
    _matrix_for_step,
    _source_video,
    _write_role_frame_sequence,
)


COMMON_HEADS = (
    ("T", 28, 19, "最强公共轨迹 Head"),
    ("T", 19, 12, "四个去噪步均稳定的公共轨迹 Head"),
    ("T", 6, 2, "轨迹传播，兼有固定位置对齐"),
    ("T", 2, 14, "早期轨迹与时序对应"),
    ("S", 27, 2, "帧内局部空间"),
    ("S", 25, 12, "帧内局部空间"),
    ("S", 29, 7, "帧内局部空间"),
    ("S", 28, 22, "帧内局部空间"),
    ("P", 12, 18, "最强公共固定位置时间对齐"),
    ("P", 28, 12, "固定位置时间对齐"),
    ("P", 8, 8, "固定位置时间对齐"),
    ("C", 3, 8, "最强公共首帧/历史上下文"),
    ("C", 22, 15, "首帧/历史上下文"),
    ("C", 0, 5, "首帧/历史上下文"),
    ("G", 10, 7, "最强公共全局聚合"),
    ("G", 9, 21, "全局聚合"),
    ("G", 7, 18, "全局聚合"),
)

ROLE_DESCRIPTIONS = {
    "T": "对运动球在其他时间位置的响应显著增强。",
    "S": "响应主要集中在 query 帧内及运动球附近的空间结构。",
    "P": "跨时间响应 query 所在的固定屏幕坐标。",
    "C": "偏向首帧、历史帧或物体先前所在位置。",
    "G": "注意力广泛分布于物体、支撑物和背景。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allblock-root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--panel-width", type=int, default=448)
    parser.add_argument("--panel-height", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=49)
    parser.add_argument("--color-percentile", type=float, default=99.5)
    return parser.parse_args()


def load_rows(root: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    rows: dict[tuple[str, int, int], dict[str, str]] = {}
    for model in MODEL_LABELS:
        path = (
            root
            / "analysis_by_model"
            / model
            / "multiblock_head_roles.csv"
        )
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[(model, int(row["block"]), int(row["head"]))] = row
    return rows


def find_generated_video(root: Path, model: str, case: str) -> Path:
    matches = sorted((root / "generated" / model).glob(f"**/{case}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated video for {model}/{case}, got {matches}"
        )
    return matches[0]


def add_navigation(page: Path) -> None:
    if not page.is_file():
        return
    marker = "common_heads.html"
    text = page.read_text(encoding="utf-8")
    if marker in text:
        return
    replacement = (
        "<p><a href='common_heads.html' style='color:#8fc9ff;font-weight:700'>"
        "跨模型公共 Head 对比</a></p>\n</header>"
    )
    if "</header>" not in text:
        raise RuntimeError(f"cannot add navigation to {page}")
    page.write_text(text.replace("</header>", replacement, 1), encoding="utf-8")


def build_page(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    case: str,
    step: int,
    vmax: float,
) -> None:
    sections = []
    for role in ("T", "S", "P", "C", "G"):
        head_rows = []
        role_records = [record for record in records if record["role"] == role]
        for block, head in sorted(
            {(record["block"], record["head"]) for record in role_records}
        ):
            current = [
                record
                for record in role_records
                if record["block"] == block and record["head"] == head
            ]
            cards = []
            for model in MODEL_LABELS:
                record = next(item for item in current if item["model"] == model)
                cards.append(
                    "<figure><a target='_blank' href='"
                    f"{html.escape(record['frame_prefix'])}latent_02.jpg'>"
                    "<img class='latent-frame' data-prefix='"
                    f"{html.escape(record['frame_prefix'])}' src='"
                    f"{html.escape(record['frame_prefix'])}latent_02.jpg'></a>"
                    f"<figcaption>{html.escape(MODEL_LABELS[model])} · "
                    f"{html.escape(record['classification'])} · "
                    f"稳定 {record['consistency']:.0%} · "
                    f"margin {record['margin']:.2f}</figcaption></figure>"
                )
            note = current[0]["note"]
            head_rows.append(
                "<article><h3>"
                f"B{block:02d}-H{head:02d} · {html.escape(note)}</h3>"
                f"<div class='models'>{''.join(cards)}</div></article>"
            )
        sections.append(
            f"<section id='role-{role.lower()}'><h2>{role} · "
            f"{html.escape(ROLE_NAMES_ZH[role])}</h2>"
            f"<p>{html.escape(ROLE_DESCRIPTIONS[role])}</p>"
            f"{''.join(head_rows)}</section>"
        )

    nav = " ".join(
        f"<a href='#role-{role.lower()}'>{role}</a>"
        for role in ("T", "S", "P", "C", "G")
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-model common head overlays</title>
<style>
body{{margin:0;background:#0d0f11;color:#f4f5f6;font:15px/1.45 system-ui,sans-serif}}
header,section{{padding:18px 22px;border-bottom:1px solid #30343a}}
h1,h2,h3{{margin:0 0 8px;letter-spacing:0}}p,figcaption{{color:#b9c0c8}}
nav{{display:flex;gap:18px;flex-wrap:wrap}}nav a{{color:#8fc9ff;font-weight:700}}
.controls{{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:12px;
padding:12px 22px;background:#16191ddd;border-bottom:1px solid #3b4047}}
.controls input{{width:min(620px,65vw)}}.controls button{{width:34px;height:34px;
border:1px solid #535962;background:#23272c;color:#fff;cursor:pointer}}
.controls output{{min-width:190px;color:#f3ca52}}
article{{padding:14px 0 20px;border-top:1px solid #25292e}}
.models{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
figure{{margin:0}}img{{width:100%;display:block;background:#000}}
figcaption{{padding-top:5px}}code{{color:#f3ca52}}
@media(max-width:1100px){{.models{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>{html.escape(case)} · exact moving-ball query overlays</h1>
<h2>跨 Wan+LoRA、Wan+xSSC、PhysRVG 的公共 Head</h2>
<p>去噪步 {step}。每张图左侧为对应模型生成结果，右侧为同一 attention
映射到 source/GT 的坐标参照。绿色框表示 output frame 8 的四个精确运动球
query patches。所有 Head 和模型共用
<code>max(log2(attention/uniform),0)</code> 的 0–{vmax:.2f} 色标。</p>
<p>“公共”表示同一 Block/Head 在三个模型中具有相同主角色、均为明确分类，
且每个模型至少在四个去噪步中的三个保持该角色。这是单 case 的观察证据，
不是普适或因果语义证明。</p>
<nav>{nav} <a href='index.html'>全部代表 Head</a>
<a href='block_view.html'>按 Block 查看</a>
<a href='COMMON_HEADS_CASE001460.md'>方法与结论 MD</a></nav>
</header>
<div class="controls">
<button class="previous" title="上一个 latent 时间步">◀</button>
<input class="latent-slider" type="range" min="0" max="12" step="1" value="2">
<button class="next" title="下一个 latent 时间步">▶</button>
<output class="time-label">latent t=02 · output frames 05–08 · shown 08</output>
</div>
{''.join(sections)}
<script>
const slider=document.querySelector('.latent-slider');
const images=[...document.querySelectorAll('.latent-frame')];
const label=document.querySelector('.time-label');
function show(t){{
 t=Math.max(0,Math.min(12,Number(t)));slider.value=t;
 const key=String(t).padStart(2,'0');
 images.forEach(img=>img.src=img.dataset.prefix+'latent_'+key+'.jpg');
 const a=t===0?0:4*t-3,b=t===0?0:4*t;
 label.textContent='latent t='+key+' · output frames '
  +String(a).padStart(2,'0')+'–'+String(b).padStart(2,'0')
  +' · shown '+String(b).padStart(2,'0');
}}
slider.addEventListener('input',()=>show(slider.value));
document.querySelector('.previous').addEventListener('click',()=>show(+slider.value-1));
document.querySelector('.next').addEventListener('click',()=>show(+slider.value+1));
document.addEventListener('keydown',event=>{{
 if(event.key==='ArrowLeft')show(+slider.value-1);
 if(event.key==='ArrowRight')show(+slider.value+1);
}});
</script></body></html>"""
    (output_dir / "common_heads.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.allblock_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    models = tuple(MODEL_LABELS)

    videos = {
        model: find_generated_video(root, model, args.case) for model in models
    }
    source_video = _source_video(videos["wan_lora"])

    captures: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = {}
    positive = []
    for _, block, head, _ in COMMON_HEADS:
        for model in models:
            key = (block, model)
            if key not in captures:
                captures[key] = _matrix_for_step(
                    root / f"block{block:02d}", model, args.case, args.step
                )
            attention, _, grid = captures[key]
            values = np.maximum(
                np.log2(np.maximum(attention[head] * math.prod(grid), 1.0)),
                0.0,
            )
            positive.append(values[values > 0])
    vmax = max(
        float(np.percentile(np.concatenate(positive), args.color_percentile)),
        1.0,
    )

    records: list[dict[str, Any]] = []
    for role, block, head, note in COMMON_HEADS:
        for model in models:
            row = rows[(model, block, head)]
            attention, query_coords, grid = captures[(block, model)]
            relative_dir = (
                Path("frames")
                / "common_heads"
                / role.lower()
                / f"block{block:02d}_head{head:02d}"
                / model
            )
            selection = {
                "head": head,
                "modal_role": role,
                "target_score": float(row[f"{role.lower()}_score"]),
            }
            _write_role_frame_sequence(
                generated_video=videos[model],
                source_video=source_video,
                output_dir=output_dir / relative_dir,
                block=block,
                model=model,
                role=role,
                attention=attention,
                query_coords=query_coords,
                grid=grid,
                selection=selection,
                vmax=vmax,
                alpha=args.alpha,
                panel_size=(args.panel_width, args.panel_height),
                max_frames=args.max_frames,
            )
            records.append(
                {
                    "role": role,
                    "block": block,
                    "head": head,
                    "note": note,
                    "model": model,
                    "classification": row["classification"],
                    "consistency": float(row["step_role_consistency"]),
                    "margin": float(row["role_margin"]),
                    "frame_prefix": relative_dir.as_posix() + "/",
                }
            )

    manifest = {
        "case": args.case,
        "denoise_step": args.step,
        "shared_vmax": vmax,
        "common_head_definition": (
            "same clear primary role in all three models and role consistency "
            "of at least 3/4 denoising steps per model"
        ),
        "heads": records,
    }
    (output_dir / "common_heads_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    source_doc = Path(__file__).with_name("COMMON_HEADS_CASE001460.md")
    shutil.copy2(source_doc, output_dir / source_doc.name)
    build_page(
        output_dir, records, case=args.case, step=args.step, vmax=vmax
    )
    add_navigation(output_dir / "index.html")
    add_navigation(output_dir / "block_view.html")
    print(f"wrote {output_dir / 'common_heads.html'}")
    print(f"shared vmax: {vmax:.6f}; records: {len(records)}")


if __name__ == "__main__":
    main()
