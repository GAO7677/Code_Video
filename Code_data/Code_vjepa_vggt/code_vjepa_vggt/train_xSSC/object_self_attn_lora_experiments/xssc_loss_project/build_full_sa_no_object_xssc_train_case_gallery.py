#!/usr/bin/env python3
"""Build the local step-500 vs step-1000 training-case comparison page."""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "full_sa_no_object_xssc_loss_train_cases"
)
HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
PAGE_NAME = "full-sa-no-object-xssc-train-cases"
RESULT_DIRS = {
    500: "step-000500_steps40_512x896_ctx08_49f",
    1000: "step-001000_steps40_512x896_ctx08_49f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        raise RuntimeError(f"Refusing to replace real directory: {link}")
    link.symlink_to(target)


def media_panel(title: str, rel_path: str | None, status: str) -> str:
    if rel_path:
        body = f'<video controls muted playsinline preload="metadata" src="{escape(rel_path)}"></video>'
    else:
        body = f'<div class="pending"><span>{escape(status)}</span></div>'
    return f'<figure><div class="video-shell">{body}</div><figcaption>{escape(title)}</figcaption></figure>'


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    gpu_id = int(manifest["inference"]["gpu"])
    site = root / "site"
    media = site / "media"
    media.mkdir(parents=True, exist_ok=True)

    cards = []
    complete = {500: 0, 1000: 0}
    for case in manifest["cases"]:
        case_id = case["case_id"]
        case_media = media / case_id
        case_media.mkdir(parents=True, exist_ok=True)
        gt_link = case_media / "gt.mp4"
        ctx_link = case_media / "context.mp4"
        replace_symlink(gt_link, Path(case["gt_video"]).resolve())
        replace_symlink(ctx_link, Path(case["context_video"]).resolve())
        panels = [
            media_panel("条件输入 · 前 8 帧", f"media/{case_id}/context.mp4", ""),
            media_panel("训练 GT · 49 帧", f"media/{case_id}/gt.mp4", ""),
        ]
        for step in (500, 1000):
            result = root / "inference" / RESULT_DIRS[step] / f"{case_id}.mp4"
            rel = None
            if result.is_file() and result.stat().st_size > 0:
                result_link = case_media / f"step-{step:04d}.mp4"
                replace_symlink(result_link, result.resolve())
                rel = f"media/{case_id}/step-{step:04d}.mp4"
                complete[step] += 1
            panels.append(
                media_panel(
                    f"生成结果 · step-{step}",
                    rel,
                    f"step-{step} 正在 GPU{gpu_id} 推理",
                )
            )
        cards.append(
            f'''<section class="case" data-case="{escape(case_id)}" data-source="{escape(case['source'])}">
  <header><div><span class="source source-{escape(case['source'])}">{escape(case['source'])}</span>
  <h2>{escape(case_id)}</h2></div><code>training index {int(case['source_index'])}</code></header>
  <p class="prompt">{escape(case['prompt'])}</p>
  <div class="grid">{''.join(panels)}</div>
  <details><summary>训练样本溯源</summary><p>{escape(case['original_video_path'])}</p></details>
</section>'''
        )

    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full-SA + No-Object + xSSC Loss · 训练集回放</title>
<style>
:root{{--ink:#16283a;--paper:#edf2f5;--panel:#f8fafb;--line:#c9d5dd;--muted:#627486;--blue:#315b7d;--amber:#c7852c;--py:#217a6b;--ku:#6a55a3;--ov:#b75a3c}}
*{{box-sizing:border-box}}html{{background:var(--paper);color:var(--ink);font-family:"Avenir Next","Segoe UI",sans-serif}}body{{margin:0}}
.mast{{background:#dce7ed;border-bottom:1px solid #adbec9;padding:26px max(24px,4vw) 20px;display:grid;grid-template-columns:1fr auto;gap:24px;align-items:end}}
.eyebrow{{font:700 12px/1.2 ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase;color:var(--blue)}}h1{{font:600 clamp(27px,4vw,48px)/1.05 Georgia,serif;max-width:930px;margin:9px 0 12px}}
.mast p{{margin:0;color:var(--muted);max-width:920px;line-height:1.55}}.counter{{display:grid;grid-template-columns:repeat(2,auto);gap:8px}}
.counter div{{background:var(--ink);color:white;padding:12px 15px;min-width:105px}}.counter strong{{display:block;font:600 24px/1 Georgia,serif}}.counter span{{font-size:11px;color:#c9d5dd}}
.toolbar{{position:sticky;top:0;z-index:4;background:rgba(237,242,245,.94);backdrop-filter:blur(12px);padding:12px max(24px,4vw);border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}}
button{{font:600 13px/1.2 inherit;border:1px solid #9b6722;background:var(--amber);color:#fff;padding:10px 12px;cursor:pointer}}button:focus{{outline:3px solid rgba(49,91,125,.28);outline-offset:2px}}
main{{padding:26px max(24px,4vw) 90px}}.case{{display:block;scroll-margin-top:72px}}.case+.case{{margin-top:52px;padding-top:42px;border-top:2px solid #adbec9}}.case header{{display:flex;align-items:end;justify-content:space-between;gap:18px;border-bottom:1px solid var(--line);padding-bottom:12px}}h2{{font:600 22px/1.2 Georgia,serif;margin:7px 0 0}}code{{font-size:11px;color:var(--muted)}}
.source{{display:inline-block;color:white;padding:4px 8px;font:700 10px/1 ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase}}.source-pybullet{{background:var(--py)}}.source-kubric{{background:var(--ku)}}.source-openvid{{background:var(--ov)}}
.prompt{{margin:15px 0 20px;max-width:1200px;color:#425668;line-height:1.55}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);box-shadow:0 8px 24px rgba(38,61,79,.07)}}
.video-shell{{aspect-ratio:896/512;background:#243849;display:grid;place-items:center;overflow:hidden}}video{{width:100%;height:100%;object-fit:contain;background:#172636}}figcaption{{padding:10px 12px;font:700 12px/1.3 ui-monospace,monospace;color:#41566a}}
.pending{{width:100%;height:100%;display:grid;place-items:center;background:repeating-linear-gradient(135deg,#253d50,#253d50 9px,#2d485d 9px,#2d485d 18px);color:#dbe7ee;font:600 12px ui-monospace,monospace}}details{{margin-top:15px;border-top:1px solid var(--line);padding-top:11px;color:var(--muted);font-size:12px}}details p{{overflow-wrap:anywhere}}
.legend{{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}}.legend span{{padding:5px 8px;border:1px solid #b8c6cf;font:600 11px ui-monospace,monospace;background:rgba(255,255,255,.55)}}
@media(max-width:1050px){{.grid{{grid-template-columns:repeat(2,1fr)}}.mast{{grid-template-columns:1fr}}}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}.toolbar{{align-items:stretch;flex-direction:column}}.case header{{align-items:start;flex-direction:column}}}}
</style></head><body>
<header class="mast"><div><div class="eyebrow">training mixture / checkpoint contact sheet</div><h1>Full-SA + No-Object + xSSC Loss<br>训练集双权重回放</h1><p>同一组训练输入、同一推理配置，横向比较 step-500 与 step-1000。512×896 · 49 帧 · context 8 · 40 steps · 30 FPS · GPU{gpu_id}。</p><div class="legend"><span>PyBullet 30%</span><span>Kubric 30%</span><span>OpenVid 40%</span><span>DINOv3 MOVi-C xSSC-50k</span></div></div>
<div class="counter"><div><strong>{complete[500]}/9</strong><span>step-500</span></div><div><strong>{complete[1000]}/9</strong><span>step-1000</span></div></div></header>
<nav class="toolbar"><button id="replay" type="button">全部从头播放</button><button id="pause" type="button">全部暂停</button></nav>
<main>{''.join(cards)}</main>
<script>
function videos(){{return [...document.querySelectorAll('.case video')];}}
document.getElementById('replay').addEventListener('click',()=>videos().forEach(v=>{{v.currentTime=0;v.play();}}));
document.getElementById('pause').addEventListener('click',()=>videos().forEach(v=>v.pause()));
</script></body></html>'''
    (site / "index.html").write_text(html, encoding="utf-8")
    replace_symlink(HUB_ROOT / PAGE_NAME, site)
    print(site / "index.html")


if __name__ == "__main__":
    main()
