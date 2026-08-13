#!/usr/bin/env python3
"""Build the 18-method step-500 contact sheet for nine training cases."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "test5_step500_all_methods_train_cases.json"
HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
PAGE_NAME = "test5-step500-all-methods-train-cases"
RESULT_NAME = "step-000500_steps40_512x896_ctx08_49f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        raise RuntimeError(f"Refusing to replace real directory: {link}")
    link.symlink_to(target)


def result_root(output_root: Path, method: dict) -> Path:
    reuse = method.get("reuse_result_root")
    if reuse:
        return Path(reuse).expanduser().resolve()
    return output_root / "inference" / method["key"] / RESULT_NAME


def video_panel(title: str, rel_path: str | None, color: str, pending: str) -> str:
    if rel_path:
        body = (
            f'<video controls muted playsinline preload="none" '
            f'src="{escape(rel_path)}"></video>'
        )
    else:
        body = f'<div class="pending"><span>{escape(pending)}</span></div>'
    return f'''<figure style="--method-color:{escape(color)}">
  <div class="video-shell">{body}</div><figcaption>{escape(title)}</figcaption>
</figure>'''


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"]).expanduser().resolve()
    cases_manifest = json.loads(
        Path(config["cases_manifest"]).read_text(encoding="utf-8")
    )
    cases = cases_manifest["cases"]
    methods = config["methods"]
    site = output_root / "site"
    media = site / "media"
    media.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "runtime_status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {"state": "queued", "methods": {}}
    )

    total_complete = 0
    method_complete: dict[str, int] = {method["key"]: 0 for method in methods}
    cards: list[str] = []
    jumps: list[str] = []
    for case_number, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        input_payload = json.loads(
            Path(case["input_json"]).read_text(encoding="utf-8")
        )
        prompt = str(input_payload["input_caption"])
        if prompt != str(case["prompt"]):
            raise ValueError(f"Prompt mismatch for {case_id}")
        case_media = media / case_id
        case_media.mkdir(parents=True, exist_ok=True)
        context_link = case_media / "context.mp4"
        gt_link = case_media / "gt.mp4"
        replace_symlink(context_link, Path(case["context_video"]).resolve())
        replace_symlink(gt_link, Path(case["gt_video"]).resolve())
        panels = [
            video_panel(
                "条件输入 · 前 8 帧",
                f"media/{case_id}/context.mp4",
                "#315b7d",
                "",
            ),
            video_panel(
                "训练 GT · 49 帧",
                f"media/{case_id}/gt.mp4",
                "#c7852c",
                "",
            ),
        ]
        for method in methods:
            key = str(method["key"])
            result = result_root(output_root, method) / f"{case_id}.mp4"
            rel_path = None
            if result.is_file() and result.stat().st_size > 0:
                result_link = case_media / f"{key}.mp4"
                replace_symlink(result_link, result.resolve())
                rel_path = f"media/{case_id}/{key}.mp4"
                total_complete += 1
                method_complete[key] += 1
            method_state = status.get("methods", {}).get(key, {}).get(
                "state", "pending"
            )
            pending = {
                "running": f"GPU{config['gpu']} 正在推理",
                "failed": "推理失败，等待续跑",
                "complete": "结果文件待刷新",
            }.get(method_state, f"等待 GPU{config['gpu']}")
            panels.append(
                video_panel(
                    f"{method['label']} · step-500",
                    rel_path,
                    method["color"],
                    pending,
                )
            )
        case_label = f"{case['source']} · {int(case['source_index'])}"
        jumps.append(f'<a href="#{escape(case_id)}">{escape(case_label)}</a>')
        cards.append(
            f'''<section class="case" id="{escape(case_id)}">
  <header class="case-head"><div><span class="case-number">CASE {case_number:02d}</span>
  <span class="source source-{escape(case['source'])}">{escape(case['source'])}</span>
  <h2>{escape(case_id)}</h2></div><code>training index {int(case['source_index'])}</code></header>
  <div class="prompt"><span>Inference prompt</span><p>{escape(prompt)}</p></div>
  <div class="grid">{''.join(panels)}</div>
  <details><summary>训练样本溯源</summary><p>{escape(case['original_video_path'])}</p></details>
</section>'''
        )

    roster = "".join(
        f'''<li style="--method-color:{escape(method['color'])}"><span>{escape(method['label'])}</span>
<strong>{method_complete[method['key']]}/9</strong></li>'''
        for method in methods
    )
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>test5 指标表全部 step-500 · 训练集 9-case</title>
<style>
:root{{--ink:#172b3a;--paper:#edf2f5;--panel:#f9fbfc;--line:#c8d4dc;--muted:#607486;--blue:#315b7d;--amber:#c7852c;--py:#217a6b;--ku:#6a55a3;--ov:#b75a3c}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Segoe UI",sans-serif}}body{{margin:0}}
.mast{{padding:28px max(24px,4vw) 22px;background:#dce7ed;border-bottom:1px solid #aebfc9;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:30px;align-items:end}}
.eyebrow{{font:700 11px/1.2 ui-monospace,monospace;letter-spacing:.16em;color:var(--blue);text-transform:uppercase}}h1{{margin:8px 0 12px;font:600 clamp(28px,4vw,48px)/1.06 Georgia,serif;max-width:980px}}.mast p{{margin:0;max-width:980px;color:var(--muted);line-height:1.55}}
.score{{background:var(--ink);color:white;padding:14px 17px;min-width:155px}}.score strong{{display:block;font:600 30px/1 Georgia,serif}}.score span{{font:11px/1.3 ui-monospace,monospace;color:#c9d5dd}}
.toolbar{{position:sticky;top:0;z-index:5;padding:10px max(24px,4vw);display:flex;align-items:center;gap:9px;background:rgba(237,242,245,.95);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}}button{{padding:9px 12px;border:1px solid #9b6722;background:var(--amber);color:white;font:700 12px inherit;cursor:pointer}}button:focus,a:focus{{outline:3px solid rgba(49,91,125,.3);outline-offset:2px}}.jumps{{display:flex;gap:6px;overflow-x:auto;margin-left:8px}}.jumps a{{white-space:nowrap;padding:7px 9px;border:1px solid var(--line);background:white;color:var(--blue);font:700 10px ui-monospace,monospace;text-decoration:none}}
.roster-wrap{{padding:24px max(24px,4vw) 0}}.roster-wrap h2{{margin:0 0 12px;font:600 19px Georgia,serif}}.roster{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin:0;padding:0;list-style:none}}.roster li{{display:flex;gap:12px;justify-content:space-between;padding:8px 10px;border-left:5px solid var(--method-color);background:rgba(255,255,255,.7);font-size:11px}}.roster strong{{font-family:ui-monospace,monospace;white-space:nowrap}}
main{{padding:28px max(24px,4vw) 90px}}.case{{scroll-margin-top:72px}}.case+.case{{margin-top:58px;padding-top:46px;border-top:2px solid #aebfc9}}.case-head{{display:flex;align-items:end;justify-content:space-between;gap:18px;border-bottom:1px solid var(--line);padding-bottom:12px}}.case-number{{margin-right:8px;font:800 10px ui-monospace,monospace;letter-spacing:.13em;color:var(--blue)}}h2{{margin:7px 0 0;font:600 21px/1.2 Georgia,serif}}code{{font-size:11px;color:var(--muted)}}
.source{{display:inline-block;padding:4px 8px;color:white;font:700 10px ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase}}.source-pybullet{{background:var(--py)}}.source-kubric{{background:var(--ku)}}.source-openvid{{background:var(--ov)}}.prompt{{display:grid;grid-template-columns:130px minmax(0,1fr);gap:12px;margin:14px 0 18px;padding:11px 13px;max-width:1400px;background:#e4ebef;border-left:5px solid var(--blue)}}.prompt span{{padding-top:2px;color:var(--blue);font:800 10px/1.3 ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase}}.prompt p{{margin:0;color:#344c5e;line-height:1.5}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-top:5px solid var(--method-color);box-shadow:0 6px 18px rgba(38,61,79,.06)}}.video-shell{{aspect-ratio:896/512;display:grid;place-items:center;overflow:hidden;background:#213747}}video{{width:100%;height:100%;object-fit:contain;background:#162735}}figcaption{{min-height:48px;padding:9px 10px;font:700 11px/1.35 ui-monospace,monospace;color:#40576a}}.pending{{width:100%;height:100%;display:grid;place-items:center;background:repeating-linear-gradient(135deg,#253d50,#253d50 9px,#2d485d 9px,#2d485d 18px);color:#dce8ef;font:600 11px ui-monospace,monospace}}
details{{margin-top:13px;padding-top:10px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}}details p{{overflow-wrap:anywhere}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.roster{{grid-template-columns:repeat(2,minmax(0,1fr))}}.mast{{grid-template-columns:1fr}}}}@media(max-width:650px){{.grid,.roster{{grid-template-columns:1fr}}.toolbar{{align-items:stretch;flex-wrap:wrap}}.jumps{{width:100%;margin-left:0}}.case-head{{align-items:start;flex-direction:column}}.prompt{{grid-template-columns:1fr;gap:5px}}}}
</style></head><body>
<header class="mast"><div><div class="eyebrow">test5 average metrics / step-500 checkpoint contact sheet</div><h1>全部 step-500 方案<br>训练集 9-case 横向对照</h1><p>指标表中 18 个具有 step-500 的训练方案；相同训练输入、相同 prompt/negative prompt、seed 42、512×896、49 帧、context 8、40 inference steps、30 FPS，统一在 GPU{int(config['gpu'])} 续跑。</p></div><div class="score"><strong>{total_complete}/162</strong><span>已生成方案 × case</span></div></header>
<nav class="toolbar"><button id="replay" type="button">整页从头播放</button><button id="pause" type="button">整页暂停</button><div class="jumps">{''.join(jumps)}</div></nav>
<aside class="roster-wrap"><h2>18 个 step-500 方案</h2><ul class="roster">{roster}</ul></aside>
<main>{''.join(cards)}</main>
<script>
const videos=()=>[...document.querySelectorAll('.case video')];
document.getElementById('replay').addEventListener('click',()=>videos().forEach(v=>{{v.currentTime=0;v.play();}}));
document.getElementById('pause').addEventListener('click',()=>videos().forEach(v=>v.pause()));
</script></body></html>'''
    (site / "index.html").write_text(html, encoding="utf-8")
    replace_symlink(HUB_ROOT / PAGE_NAME, site)
    print(site / "index.html")


if __name__ == "__main__":
    main()
