#!/usr/bin/env python3
"""Encode separate center-point Q/K and CoTracker frame sequences as H.264 videos."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg


MODELS = (
    ("gt", "GT source"),
    ("stage1b", "Stage1b step-004000"),
    ("lora", "LoRA step-000500"),
    ("baseline", "Wan2.2 baseline"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--frame-dashboard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--step-index", type=int, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    return parser.parse_args()


def encode_sequence(source: Path, destination: Path, fps: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "1",
        "-i",
        str(source / "latent_%02d.jpg"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(destination),
    ]
    subprocess.run(command, check=True)


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Center track videos</title><style>
:root{--ink:#18211d;--paper:#e9e2d2;--card:#fffaf0;--rust:#b84b2f;--green:#1d6553;--line:#a99f8e}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(135deg,#c9543430,transparent 38%),radial-gradient(circle at 90% 8%,#42866c38,transparent 34rem),var(--paper);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1500px,calc(100% - 24px));margin:auto;padding:30px 0 60px}.eyebrow{color:var(--rust);font-weight:900;letter-spacing:.14em;font-size:12px}h1{margin:4px 0 12px;font:700 clamp(38px,6vw,78px)/.95 Georgia,"Noto Serif CJK SC",serif;letter-spacing:-.04em}.lead{color:#59645f;max-width:950px;line-height:1.6}.controls{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin:22px 0}label{font-size:11px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}select{display:block;width:100%;padding:11px;margin-top:5px;border:1px solid var(--ink);background:var(--card);font-weight:800}.model{margin:18px 0 28px}.model h2{font:700 25px Georgia;margin:0 0 8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}.track{background:var(--card);border:1px solid var(--line);padding:8px}.track strong{display:block;color:var(--green);margin:2px 2px 8px}.track video{display:block;width:100%;background:#07100d;aspect-ratio:7/4}.note{border-left:5px solid var(--rust);padding:12px 15px;background:var(--card);line-height:1.55}@media(max-width:760px){.controls,.pair{grid-template-columns:1fr}h1{font-size:44px}}
</style></head><body><main><div class="eyebrow">LAYER __LAYER__ · DENOISING STEP __STEP__ · CENTER QUERY ONLY</div><h1>Separate track videos</h1><p class="lead">每个区域只跟踪一个中心代表点。Q/K 方框轨迹与 CoTracker 圆点轨迹编码为两个独立 H.264 视频，不做叠加；每个视频的一帧对应一个 latent anchor。</p><div class="controls"><label>Case<select id="case"></select></label><label>Region<select id="region"></select></label></div><div id="models"></div><div class="note">播放器使用 2 fps 便于逐 latent 检查。GT case 2/4 受原视频长度限制，仅包含 latent 1–7；其余包含 latent 1–12。</div></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),caseEl=document.getElementById('case'),regionEl=document.getElementById('region'),modelsEl=document.getElementById('models');for(const c of data.cases){const o=document.createElement('option');o.value=c.case_key;o.textContent=c.label;caseEl.append(o)}function current(){return data.cases.find(c=>c.case_key===caseEl.value)}function regions(){regionEl.innerHTML=current().regions.map(r=>`<option>${r}</option>`).join('')}function render(){const c=current(),r=regionEl.value;modelsEl.innerHTML=data.models.map(m=>`<section class="model"><h2>${m.label}</h2><div class="pair"><div class="track"><strong>Q/K center only · squares</strong><video controls loop preload="metadata" src="videos/${m.name}/${c.case_key}/${r}/qk.mp4"></video></div><div class="track"><strong>CoTracker center only · circles</strong><video controls loop preload="metadata" src="videos/${m.name}/${c.case_key}/${r}/cotracker.mp4"></video></div></div></section>`).join('')}caseEl.addEventListener('change',()=>{regions();render()});regionEl.addEventListener('change',render);regions();render();
</script></body></html>'''


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    frame_root = args.frame_dashboard.resolve() / "frames"
    output = args.output_dir.resolve()
    case_keys = sorted(
        path.parent.name for path in (result_root / "stage1b" / "cases").glob("*/complete.json")
    )
    cases = []
    for case_key in case_keys:
        manifest = json.loads(
            (result_root / "stage1b" / "cases" / case_key / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        regions = [region["region_name"] for region in manifest["query_regions"]]
        cases.append(
            {
                "case_key": case_key,
                "label": case_key.removeprefix("case_physiciq_"),
                "regions": regions,
            }
        )
        for model, _ in MODELS:
            for region in regions:
                for track in ("qk", "cotracker"):
                    encode_sequence(
                        frame_root / model / case_key / region / track,
                        output / "videos" / model / case_key / region / f"{track}.mp4",
                        args.fps,
                    )

    payload = json.dumps(
        {
            "models": [{"name": name, "label": label} for name, label in MODELS],
            "cases": cases,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(
        HTML.replace("__PAYLOAD__", payload)
        .replace("__LAYER__", str(args.layer))
        .replace("__STEP__", str(args.step_index)),
        encoding="utf-8",
    )
    print(f"Built {len(cases) * len(MODELS) * 3 * 2} videos in {output}")


if __name__ == "__main__":
    main()
