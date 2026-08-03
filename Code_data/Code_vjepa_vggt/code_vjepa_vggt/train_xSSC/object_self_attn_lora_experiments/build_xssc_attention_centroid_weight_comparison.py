#!/usr/bin/env python3
"""Compare attention-centroid motion for DINOv3 and three official xSSC weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import visualize_xssc_attention_centroid_overlay as overlay


DEFAULT_DINOV3_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_slot_separation_cases_dinov3_latest/"
    "models/dinov3_vitl_movic_transfer_movi_c_transfer15000_b64_acc3_20260721T134713Z_step-050000"
)
DEFAULT_OFFICIAL_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_slot_separation_cases"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_attention_centroid_weight_comparison"
)

METHODS = (
    {
        "id": "dinov3_movic_step050000",
        "label": "DINOv3 ViT-L MOVi-C step-050000",
        "kind": "dinov3",
        "accent": "#15803d",
    },
    {"id": "official_42-0130", "label": "Official DINOv2 42-0130", "kind": "official", "accent": "#2563eb"},
    {"id": "official_43-0091", "label": "Official DINOv2 43-0091", "kind": "official", "accent": "#c2410c"},
    {"id": "official_44-0101", "label": "Official DINOv2 44-0101", "kind": "official", "accent": "#7e22ce"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3-root", type=Path, default=DEFAULT_DINOV3_ROOT)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trail-length", type=int, default=12)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--min-hard-area", type=float, default=0.005)
    return parser.parse_args()


def method_paths(
    method: dict[str, str],
    case_id: str,
    dinov3_root: Path,
    official_root: Path,
) -> tuple[Path, Path, Path]:
    if method["kind"] == "dinov3":
        case_dir = dinov3_root / case_id
        result_dir = case_dir / "xssc"
    else:
        case_dir = official_root / case_id
        result_dir = case_dir / method["id"]
    return (
        result_dir / "slot_separation_arrays.npz",
        result_dir / "summary.json",
        case_dir / "xssc_input_49f.mp4",
    )


def case_ids(dinov3_root: Path, official_root: Path) -> list[str]:
    dino = {item.parent.parent.name for item in dinov3_root.glob("*/xssc/slot_separation_arrays.npz")}
    official = {
        item.parent.parent.name
        for item in official_root.glob("*/official_42-0130/slot_separation_arrays.npz")
    }
    return sorted(dino & official)


def render_method(
    method: dict[str, str],
    case_id: str,
    dinov3_root: Path,
    official_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    arrays_path, summary_path, source_path = method_paths(
        method, case_id, dinov3_root, official_root
    )
    for path in (arrays_path, summary_path, source_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as data:
        attention = data["attention"].astype(np.float32)
    frames, source_fps = overlay.read_video(source_path)
    length = min(len(frames), attention.shape[0])
    frames = frames[:length]
    attention = attention[:length]
    height, width = frames[0].shape[:2]
    centroids, mass, hard_area = overlay.attention_statistics(attention, height, width)
    selected_slots = overlay.choose_slots(summary, "selected", attention.shape[1])
    rendered = overlay.render_overlay(
        frames,
        centroids,
        hard_area,
        selected_slots,
        args.trail_length,
        args.min_hard_area,
        normalized_speed=True,
    )
    method_dir = output_dir / "cases" / case_id / method["id"]
    video_path = method_dir / "attention_centroid_overlay.mp4"
    csv_path = method_dir / "attention_centroids.csv"
    overlay.write_video(video_path, rendered, args.fps or source_fps)
    overlay.write_case_csv(csv_path, centroids, mass, hard_area, selected_slots)
    return {
        "id": method["id"],
        "label": method["label"],
        "accent": method["accent"],
        "frames": length,
        "input_resolution": [width, height],
        "selected_slots": selected_slots,
        "video": str(video_path.relative_to(output_dir)),
        "csv": str(csv_path.relative_to(output_dir)),
    }


def build_html(output_dir: Path, records: list[dict[str, Any]]) -> None:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>xSSC attention 质心权重对比</title>
<style>body{{margin:0;background:#f4f6f8;color:#17202a;font:14px Arial,sans-serif}}header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid #d8dee6;padding:12px 18px}}h1{{font-size:19px;margin:0 0 8px}}.controls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}select,button{{height:32px;border:1px solid #aeb7c2;background:#fff;padding:0 10px;border-radius:4px}}main{{margin:16px auto;max-width:1580px;padding:0 14px}}.note{{line-height:1.55;color:#4b5563;max-width:1180px}}.scroll{{overflow-x:auto;padding-bottom:7px}}.grid{{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:10px;min-width:1170px}}figure{{margin:0;background:#fff;border:1px solid #d8dee6;border-top:4px solid var(--accent,#64748b);padding:7px;border-radius:5px}}video{{display:block;width:100%;aspect-ratio:1/1;background:#111}}figcaption{{padding-top:7px;font-weight:700;font-size:13px;line-height:1.35}}.meta{{font-size:12px;color:#64748b;margin-top:4px}}#details{{margin-top:10px;background:#fff;border:1px solid #d8dee6;padding:9px;border-radius:5px;line-height:1.5}}</style></head><body>
<header><h1>xSSC attention 质心位移 · DINOv3 与官方权重对比</h1><div class="controls"><select id="case"></select><button id="play">播放全部</button><button id="pause">暂停全部</button><button id="replay">从头播放</button></div></header><main><p class="note">同一 case 横向比较。圆点为 soft-attention 质心，箭头为一帧位移，轨迹保留最近 12 帧。速度统一为每帧移动占画面对角线的百分比，因此官方 224×224 与 DINOv3 256×256 可比较。各模型只展示其自身选出的活动 object slots。</p><div class="scroll"><div class="grid" id="grid"></div></div><div id="details"></div></main>
<script>const DATA={payload};const select=document.getElementById('case'),grid=document.getElementById('grid'),details=document.getElementById('details');DATA.forEach(r=>{{const o=document.createElement('option');o.value=r.case_id;o.textContent=r.case_id;select.appendChild(o)}});function videos(){{return [...grid.querySelectorAll('video')]}}function card(label,src,accent,meta){{const f=document.createElement('figure');f.style.setProperty('--accent',accent);f.innerHTML=`<video muted playsinline preload="metadata" src="${{src}}"></video><figcaption>${{label}}</figcaption><div class="meta">${{meta}}</div>`;return f}}function load(){{const r=DATA.find(x=>x.case_id===select.value);grid.replaceChildren(card('原始输入',r.source,'#475569',`${{r.source_frames}} frames`),...r.methods.map(m=>card(m.label,m.video,m.accent,`slots ${{m.selected_slots.map(s=>'S'+s).join(', ')}} · ${{m.input_resolution.join('×')}}`)));details.textContent='注意：DINOv3 MOVi-C 使用 bbox-conditioned 初始化；三个官方权重是 DINOv2 YTVIS 配置，因此这里比较的是最终 attention 轨迹表现，不是只隔离 backbone 的消融。';}}select.onchange=load;document.getElementById('play').onclick=()=>videos().forEach(v=>v.play());document.getElementById('pause').onclick=()=>videos().forEach(v=>v.pause());document.getElementById('replay').onclick=()=>videos().forEach(v=>{{v.currentTime=0;v.play()}});load();</script></body></html>"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dinov3_root = args.dinov3_root.expanduser().resolve()
    official_root = args.official_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case_id in case_ids(dinov3_root, official_root):
        source_path = dinov3_root / case_id / "xssc_input_49f.mp4"
        source_frames, _ = overlay.read_video(source_path)
        source_link = output_dir / "cases" / case_id / "source.mp4"
        source_link.parent.mkdir(parents=True, exist_ok=True)
        if source_link.exists() or source_link.is_symlink():
            source_link.unlink()
        source_link.symlink_to(source_path)
        methods = []
        for method in METHODS:
            result = render_method(
                method, case_id, dinov3_root, official_root, output_dir, args
            )
            methods.append(result)
            print(f"[done] {case_id} | {method['id']} | slots={result['selected_slots']}", flush=True)
        records.append(
            {
                "case_id": case_id,
                "source": str(source_link.relative_to(output_dir)),
                "source_frames": len(source_frames),
                "methods": methods,
            }
        )
    if not records:
        raise RuntimeError("No common completed cases found")
    metadata = {
        "dinov3_root": str(dinov3_root),
        "official_root": str(official_root),
        "speed_unit": "percent of frame diagonal per frame",
        "records": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    build_html(output_dir, records)
    print(f"[complete] {len(records)} cases -> {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
