#!/usr/bin/env python3
"""Build a lightweight input-only gallery for the PhysV V2V test70 manifest.

The gallery intentionally exposes the context8 input videos only.  The video
files are represented by symlinks inside the 8844 static root, so the source
dataset is not copied or modified.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path


LIST_FILE = Path(
    "/data/gaoya/AAA_test_video/physv_v2v_0819/testjsons/"
    "physv_v2v_0819_all_cycles_test70_ctx8_description_no_event_timing.txt"
)
HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
PAGE_DIR = HUB_ROOT / "physv-v2v-0819-test70-description-no-event-timing-input"
MEDIA_DIR = PAGE_DIR / "media"
H264_MEDIA_DIR = PAGE_DIR / "media_h264"
FFMPEG = Path("/home/gaoya/.marscode/ai-chat/binary/1.7.8/modules/ai-agent/ffmpeg")


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for line in LIST_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        json_path = Path(line)
        item = json.loads(json_path.read_text(encoding="utf-8"))
        video = Path(item["input_video"])
        if not video.is_file():
            raise FileNotFoundError(f"missing input video: {video}")
        sample_id = str(item["sample_id"])
        cases.append(
            {
                "case_id": sample_id,
                "family_key": item.get("family_key", "UNGROUPED"),
                "title": item.get("title", sample_id),
                "task_type": item.get("task_type", ""),
                "source_group": item.get("source_group", ""),
                "input_caption": item.get("input_caption", ""),
                "input_caption_abstract": item.get("input_caption_abstract", ""),
                "input_video": str(video),
                "json_path": str(json_path),
                "video_url": f"media_h264/{sample_id}.mp4",
            }
        )
    if len({item["case_id"] for item in cases}) != len(cases):
        raise ValueError("duplicate sample_id in input list")
    return cases


def link_media(cases: list[dict]) -> None:
    """Keep original links for audit and create browser-compatible H.264 files."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    H264_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    for item in cases:
        target = Path(item["input_video"])
        link = MEDIA_DIR / f"{item['case_id']}.mp4"
        if link.is_symlink():
            if link.resolve() == target.resolve():
                pass
            else:
                link.unlink()
                link.symlink_to(target)
        elif link.exists():
            raise FileExistsError(f"refusing to replace non-symlink: {link}")
        else:
            link.symlink_to(target)

        output = H264_MEDIA_DIR / f"{item['case_id']}.mp4"
        if output.is_file() and b"avc1" in output.read_bytes()[:200_000]:
            continue
        if not FFMPEG.is_file():
            raise FileNotFoundError(f"ffmpeg not found: {FFMPEG}")
        temporary = output.with_suffix(".tmp.mp4")
        if temporary.exists():
            temporary.unlink()
        command = [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(target),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        subprocess.run(command, check=True)
        if b"avc1" not in temporary.read_bytes()[:200_000]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"transcode did not produce H.264: {target}")
        temporary.replace(output)


def build_manifest(cases: list[dict]) -> dict:
    return {
        "title": "PhysV V2V 0819 · all-cycles test70 · context8 input audit",
        "list_file": str(LIST_FILE),
        "case_count": len(cases),
        "family_count": len({item["family_key"] for item in cases}),
        "input_definition": "input_video = context8_cycles.mp4; first 8 context frames; not GT full video",
        "browser_video_format": "H.264 avc1 derivative generated from the original mp4v input for browser playback",
        "caption_variant": "description_no_event_timing",
        "cases": cases,
    }


def page_html(manifest: dict) -> str:
    payload = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhysV V2V 0819 · test70 · context8 input</title>
<style>
:root{{--ink:#172126;--muted:#66777c;--line:#d8e1e3;--paper:#f5f7f8;--surface:#fff;--teal:#116466;--amber:#e9c46a;--blue:#315c87}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans SC",Arial,sans-serif}}
header{{background:var(--ink);color:#f7fbfa;padding:25px 22px 23px;border-bottom:4px solid var(--amber)}}
.header-inner,main{{max-width:1680px;margin:0 auto}}
.eyebrow{{font-size:10px;letter-spacing:.18em;color:#9fc4bf;font-weight:900}}
h1{{margin:7px 0 6px;font-size:25px;letter-spacing:-.02em}}
.lede{{max-width:920px;margin:0;color:#c9d7d6;font-size:13px;line-height:1.65}}
.header-stats{{display:flex;gap:9px;flex-wrap:wrap;margin-top:17px}}
.header-stats span{{border:1px solid #52666a;background:#223439;border-radius:4px;padding:6px 9px;font-size:11px;font-weight:800;color:#eef7f5}}
main{{padding:18px 22px 46px}}
.notice{{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;flex-wrap:wrap;background:#fff9e7;border:1px solid #ecd49a;border-left:5px solid var(--amber);padding:13px 15px;border-radius:6px}}
.notice strong{{display:block;margin-bottom:4px}}
.notice p{{margin:0;color:#6f5b26;font-size:12px;line-height:1.55}}
.notice .tag{{white-space:nowrap;color:#6f5b26;font-size:11px;font-weight:900;letter-spacing:.1em}}
.toolbar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:16px 0 12px}}
input,select,button{{font:inherit;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--ink);padding:8px 10px}}
input{{min-width:280px}}button{{cursor:pointer;background:#eaf4f2;border-color:#b5d5d0;color:var(--teal);font-weight:850}}button:hover,button:focus-visible{{outline:2px solid var(--amber);outline-offset:2px}}
.toolbar .spacer{{flex:1}}
.toolbar a{{color:var(--teal);font-size:12px;font-weight:850;text-decoration:none;padding:8px 2px}}
.summary{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 16px}}
.summary .metric{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 13px;min-width:150px}}
.summary b{{display:block;font-size:19px}}
.summary span{{font-size:11px;color:var(--muted)}}
.family-section{{margin-top:24px}}
.family-head{{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;flex-wrap:wrap;border-bottom:2px solid var(--line);padding:0 4px 8px}}
.family-kicker{{display:block;color:var(--teal);font-size:10px;letter-spacing:.16em;font-weight:900}}
.family-head h2{{margin:3px 0 0;font-size:19px}}
.family-count{{color:var(--teal);font-size:11px;font-weight:850;background:#e7f1ef;border-radius:999px;padding:5px 8px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:12px}}
.case-card{{background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:10px;min-width:0;box-shadow:0 3px 12px rgba(23,33,38,.035)}}
.case-card video{{display:block;width:100%;aspect-ratio:896/512;object-fit:contain;background:#101618;border-radius:4px}}
.case-top{{display:flex;justify-content:space-between;gap:8px;align-items:flex-start;margin:9px 0 4px}}
.case-top h3{{font-size:13px;line-height:1.3;margin:0}}
.case-no{{color:var(--teal);font-size:10px;font-weight:900;white-space:nowrap}}
.meta{{color:var(--muted);font-size:10px;line-height:1.45;margin:3px 0}}
.caption{{color:#34474c;font-size:11px;line-height:1.5;margin:8px 0 0;border-top:1px solid #edf0f1;padding-top:7px}}
.caption-label{{color:var(--teal);font-size:9px;font-weight:900;letter-spacing:.12em}}
.frame-strip{{display:grid;grid-template-columns:repeat(8,1fr);gap:3px;margin-top:8px}}
.frame-strip span{{display:grid;place-items:center;background:#e8efef;color:#496067;height:17px;font-size:8px;font-weight:900}}
.empty{{padding:35px 14px;text-align:center;border:1px dashed #cbd6d8;color:var(--muted);background:#fff}}
.back-to-top{{position:fixed;right:20px;bottom:20px;border-radius:999px;background:var(--ink);color:#fff8df;border-color:var(--amber);box-shadow:0 8px 22px rgba(23,33,38,.25)}}
@media(max-width:1250px){{.grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}
@media(max-width:850px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}input{{min-width:220px}}}}
@media(max-width:560px){{header{{padding:20px 13px}}main{{padding-left:13px;padding-right:13px}}h1{{font-size:21px}}.grid{{grid-template-columns:1fr}}input{{width:100%;min-width:0}}.toolbar label{{width:100%}}.toolbar select{{width:100%}}}}
@media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important}}}}
</style>
</head>
<body>
<header><div class="header-inner">
  <div class="eyebrow">INPUT AUDIT / PHYSV V2V 0819</div>
  <h1>all-cycles test70 · context8 输入审阅</h1>
  <p class="lede">先检查将要送入 test_5 / PhysicIQ 类推理的输入视频。每张卡只播放该 case 的 <b>8 帧 context video</b>；不展示 GT future video，也不启动任何模型推理。</p>
  <div class="header-stats"><span>70 CASES</span><span>14 FAMILIES</span><span>CONTEXT 8 FRAMES</span><span>NO EVENT-TIMING DESCRIPTION</span></div>
</div></header>
<main>
  <section class="notice"><div><strong>输入定义</strong><p><code>input_video</code> = <code>context8_cycles.mp4</code>，即前 8 帧上下文；GT 完整视频只作为后续评测 reference，不在此页面播放。</p></div><span class="tag">READ ONLY · MANUAL REFRESH</span></section>
  <div class="toolbar">
    <label><input id="search" type="search" placeholder="搜索 case、标题、caption…"></label>
    <label><select id="family"><option value="">全部 FAMILY</option></select></label>
    <label><select id="sort"><option value="list">按测试集顺序</option><option value="family">按 FAMILY / case</option><option value="title">按标题</option></select></label>
    <button id="clear" type="button">清除筛选</button><button id="refresh" type="button">手动刷新</button>
    <span class="spacer"></span><a href="../">返回 8844 总览</a>
  </div>
  <div class="summary" id="summary"></div>
  <div id="gallery"></div>
</main>
<button class="back-to-top" id="top" type="button">回到顶部</button>
<script id="manifest" type="application/json">{payload}</script>
<script>
const DATA=JSON.parse(document.getElementById('manifest').textContent);
const CASES=DATA.cases;
const esc=(v)=>String(v??'').replace(/[&<>"']/g,ch=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
const familySelect=document.getElementById('family');
[...new Set(CASES.map(x=>x.family_key))].sort().forEach(x=>familySelect.add(new Option(x,x)));
function matches(item, query, family){{
  if(family && item.family_key!==family) return false;
  if(!query) return true;
  const hay=[item.case_id,item.family_key,item.title,item.task_type,item.source_group,item.input_caption,item.input_caption_abstract].join(' ').toLowerCase();
  return hay.includes(query.toLowerCase());
}}
function card(item,index,totalInFamily){{
  const caption=item.input_caption || item.input_caption_abstract || '（该 case 未提供文字 caption）';
  return `<article class="case-card"><video controls preload="metadata" playsinline src="${{encodeURI(item.video_url)}}"></video><div class="case-top"><h3>${{esc(item.title)}}</h3><span class="case-no">${{String(index+1).padStart(2,'0')}} / ${{totalInFamily}}</span></div><p class="meta"><b>${{esc(item.case_id)}}</b> · ${{esc(item.task_type)}} · ${{esc(item.source_group)}}</p><p class="caption"><span class="caption-label">INPUT CAPTION</span><br>${{esc(caption)}}</p><div class="frame-strip" aria-label="8 context frames">${{[1,2,3,4,5,6,7,8].map(x=>`<span>F${{String(x).padStart(2,'0')}}</span>`).join('')}}</div></article>`;
}}
function render(){{
  const q=document.getElementById('search').value.trim();
  const family=familySelect.value;
  const sort=document.getElementById('sort').value;
  let items=CASES.filter(x=>matches(x,q,family));
  if(sort==='family') items.sort((a,b)=>(a.family_key+a.case_id).localeCompare(b.family_key+b.case_id));
  if(sort==='title') items.sort((a,b)=>a.title.localeCompare(b.title));
  const familyCount=new Set(items.map(x=>x.family_key)).size;
  document.getElementById('summary').innerHTML=`<div class="metric"><b>${{items.length}}</b><span>当前可见 cases</span></div><div class="metric"><b>${{familyCount}}</b><span>当前可见 families</span></div><div class="metric"><b>${{DATA.case_count}}</b><span>测试集总 cases</span></div><div class="metric"><b>8</b><span>每个 input video 帧数</span></div>`;
  const groups=new Map();
  items.forEach(x=>{{if(!groups.has(x.family_key))groups.set(x.family_key,[]);groups.get(x.family_key).push(x)}});
  let html=''; let familyNo=0;
  for(const [key,group] of groups){{
    familyNo++;
    html+=`<section class="family-section"><div class="family-head"><div><span class="family-kicker">FAMILY ${{String(familyNo).padStart(2,'0')}}</span><h2>${{esc(key)}}</h2></div><span class="family-count">${{group.length}} cases · context8 inputs</span></div><div class="grid">`;
    html+=group.map((item,i)=>card(item,i,group.length)).join('');
    html+='</div></section>';
  }}
  document.getElementById('gallery').innerHTML=html || '<div class="empty">没有匹配的 case，请清除筛选。</div>';
}}
['search','family','sort'].forEach(id=>document.getElementById(id).addEventListener('input',render));
document.getElementById('clear').addEventListener('click',()=>{{document.getElementById('search').value='';familySelect.value='';document.getElementById('sort').value='list';render()}});
document.getElementById('refresh').addEventListener('click',()=>window.location.reload());
document.getElementById('top').addEventListener('click',()=>window.scrollTo({{top:0,behavior:'smooth'}}));
render();
</script>
</body></html>
'''


def main() -> None:
    cases = load_cases()
    manifest = build_manifest(cases)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    link_media(cases)
    (PAGE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (PAGE_DIR / "index.html").write_text(page_html(manifest), encoding="utf-8")
    family_counts = Counter(item["family_key"] for item in cases)
    print(f"wrote {PAGE_DIR / 'index.html'}")
    print(f"linked {len(cases)} input videos into {MEDIA_DIR}")
    print(f"families: {dict(sorted(family_counts.items()))}")


if __name__ == "__main__":
    main()
