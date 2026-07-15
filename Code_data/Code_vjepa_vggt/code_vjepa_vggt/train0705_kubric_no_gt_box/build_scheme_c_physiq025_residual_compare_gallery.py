#!/usr/bin/env python3
"""Build the Scheme-C step/residual matrix for the PhysicsIQ 025 case."""
from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote


CASE_STEM = "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed"
INPUT_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/"
    f"{CASE_STEM}.json"
)
FORMAL_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/"
    "train_stage1b_scheme_c_entity_caption_physical_fresh_20260714T174707Z"
)
NEW_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "scheme_c_physiq025_step2500_3500_residual_compare_20260715/new_inference"
)
OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/"
    "scheme_c_physiq025_step2500_3500_residual_compare_20260715/gallery"
)
STEPS = ("step-002500", "step-003500")
SCALES = ("1.0", "1.2", "1.3", "1.4", "1.5", "2.0")
EXISTING_SCALES = {"1.0", "1.5", "2.0"}


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def safe_link(target: Path, link: Path) -> None:
    target = target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {link}")
    link.symlink_to(target)


def result_root(step: str, scale: str) -> Path:
    tag = scale.replace(".", "p")
    base = FORMAL_ROOT if scale in EXISTING_SCALES else NEW_ROOT
    return base / step / f"object_residual_{tag}x/results"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    prefix = Path("/data/gaoya")
    try:
        return str(resolved.relative_to(prefix))
    except ValueError:
        return str(resolved)


def build() -> dict:
    case = load_json(INPUT_JSON)
    source = Path(case["source_video"])
    context = Path(case["input_video"])
    safe_link(source, OUTPUT_DIR / "videos/source_reference.mp4")
    safe_link(context, OUTPUT_DIR / "videos/context_8f.mp4")

    rows = []
    for step in STEPS:
        variants = []
        for scale in SCALES:
            root = result_root(step, scale)
            metadata = root / f"{CASE_STEM}.json"
            payload = load_json(metadata)
            video = Path(payload["output_video"])
            tag = scale.replace(".", "p")
            video_link = OUTPUT_DIR / "videos" / f"{step}_residual_{tag}x.mp4"
            metadata_link = OUTPUT_DIR / "videos" / f"{step}_residual_{tag}x.json"
            safe_link(video, video_link)
            safe_link(metadata, metadata_link)
            variants.append(
                {
                    "step": step,
                    "scale": scale,
                    "origin": "existing formal sweep" if scale in EXISTING_SCALES else "new single-case inference",
                    "video": str(video_link.relative_to(OUTPUT_DIR)),
                    "metadata": str(metadata_link.relative_to(OUTPUT_DIR)),
                    "output_path": str(video.resolve()),
                    "display_output_path": display_path(video),
                    "seed": payload.get("seed"),
                    "guidance": payload.get("guidance"),
                    "inference_steps": payload.get("step"),
                    "negative_prompt": payload.get("negative_prompt"),
                    "context_frames": payload.get("effective_context_frames"),
                }
            )
        rows.append({"step": step, "variants": variants})

    manifest = {
        "case": CASE_STEM,
        "input_json": str(INPUT_JSON.resolve()),
        "caption": case["input_caption"],
        "source_video": "videos/source_reference.mp4",
        "source_path": str(source.resolve()),
        "context_video": "videos/context_8f.mp4",
        "context_path": str(context.resolve()),
        "timeline": {
            "fps": 30,
            "output_frames": 49,
            "context_frames": 8,
            "source_frames": 30,
            "alignment": "absolute frame/time from frame 0; shorter references hold their final frame",
        },
        "rows": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "index.html").write_text(render(manifest), encoding="utf-8")
    return manifest


def video_card(variant: dict, leader: bool) -> str:
    leader_attr = " data-sync-leader='true'" if leader else ""
    source_tag = "formal" if variant["origin"].startswith("existing") else "new"
    prompt_label = "null" if variant["negative_prompt"] is None else "configured"
    return (
        f"<figure class='variant {source_tag}'>"
        f"<video controls preload='metadata' src='{quote(variant['video'])}'{leader_attr}></video>"
        "<figcaption>"
        f"<div><strong>Residual {html.escape(variant['scale'])}x</strong>"
        f"<span class='tag'>{html.escape(source_tag)}</span></div>"
        f"<p>{html.escape(variant['origin'])}</p>"
        f"<dl><dt>seed</dt><dd>{variant['seed']}</dd><dt>CFG</dt><dd>{variant['guidance']}</dd>"
        f"<dt>steps</dt><dd>{variant['inference_steps']}</dd><dt>negative</dt><dd>{prompt_label}</dd></dl>"
        f"<code>{html.escape(variant['display_output_path'])}</code>"
        f"<a href='{quote(variant['metadata'])}' target='_blank'>metadata JSON</a>"
        "</figcaption></figure>"
    )


def render(manifest: dict) -> str:
    rows = []
    for row in manifest["rows"]:
        cards = "".join(
            video_card(variant, row["step"] == "step-002500" and variant["scale"] == "1.0")
            for variant in row["variants"]
        )
        rows.append(
            f"<section><header><h2>{html.escape(row['step'])}</h2>"
            "<span>Scheme-C entity-caption physical checkpoint</span></header>"
            f"<div class='matrix'>{cards}</div></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scheme-C PhysicsIQ 025 Residual Matrix</title><style>
:root{{--ink:#18201c;--muted:#68716c;--paper:#f3f4f1;--panel:#fff;--line:#c9cfcb;--green:#236747;--red:#9f342d;--blue:#315f7a;--amber:#9b641f}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px "IBM Plex Sans","Noto Sans",sans-serif;letter-spacing:0}}main{{max-width:1940px;margin:auto;padding:0 24px 70px}}
.top{{position:sticky;top:0;z-index:5;padding:16px 0 13px;background:rgba(243,244,241,.97);border-bottom:1px solid var(--line)}}h1{{font:700 26px "IBM Plex Serif","Noto Serif",serif;margin:0 0 5px}}.case{{margin:0 0 5px;font:600 14px ui-monospace,monospace;overflow-wrap:anywhere}}.caption{{max-width:1200px;margin:0 0 12px;color:var(--muted);line-height:1.45}}
.timeline{{display:grid;grid-template-columns:34px 34px 34px minmax(240px,720px) 150px auto;gap:6px;align-items:center}}button{{width:34px;height:32px;border:1px solid #89938d;background:#fff;color:var(--ink);font:inherit;cursor:pointer}}button:hover{{border-color:var(--red);color:var(--red)}}input[type=range]{{width:100%;accent-color:var(--red)}}output{{font:12px ui-monospace,monospace}}.phase{{font-size:12px;color:var(--muted)}}
.references{{display:grid;grid-template-columns:repeat(2,minmax(0,560px));gap:14px;padding:22px 0}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);min-width:0}}.reference{{border-top:4px solid var(--green)}}video{{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#101210}}figcaption{{display:grid;gap:6px;padding:9px}}figcaption>div{{display:flex;justify-content:space-between;gap:8px}}figcaption p{{margin:0;color:var(--muted);font-size:12px}}code{{font:10px ui-monospace,monospace;overflow-wrap:anywhere;word-break:break-word;color:#45504a}}a{{color:var(--blue);font-size:12px;width:max-content}}.tag{{padding:2px 5px;border:1px solid currentColor;font:700 10px ui-monospace,monospace;text-transform:uppercase}}
section{{padding:22px 0 30px;border-top:1px solid var(--line);overflow-x:auto}}section>header{{display:flex;gap:12px;align-items:baseline;margin-bottom:10px}}h2{{margin:0;font-size:19px}}section>header span{{color:var(--muted)}}.matrix{{display:grid;grid-template-columns:repeat(6,minmax(245px,1fr));gap:12px;min-width:1510px}}.variant.formal{{border-top:4px solid var(--blue)}}.variant.new{{border-top:4px solid var(--amber)}}dl{{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:3px 7px;margin:0;padding:6px 0;border-top:1px solid #e3e6e4;border-bottom:1px solid #e3e6e4;font:11px ui-monospace,monospace}}dt{{color:var(--muted)}}dd{{margin:0;text-align:right}}
.legend{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:8px;color:var(--muted);font-size:12px}}.swatch{{display:inline-block;width:16px;height:4px;margin-right:5px;vertical-align:middle}}.swatch.formal{{background:var(--blue)}}.swatch.new{{background:var(--amber)}}
@media(max-width:760px){{main{{padding:0 12px 50px}}.references{{grid-template-columns:1fr}}.timeline{{grid-template-columns:34px 34px 34px minmax(100px,1fr) 115px}}.phase{{grid-column:4/6}}h1{{font-size:22px}}}}
</style></head><body><main><div class="top"><h1>Scheme-C Residual Scale Matrix</h1>
<p class="case">{html.escape(manifest['case'])}</p><p class="caption">{html.escape(manifest['caption'])}</p>
<div class="timeline"><button data-action="play" title="Play all at aligned time">&#9654;</button><button data-action="pause" title="Pause all">II</button><button data-action="reset" title="Reset">&#8634;</button><input id="timeline" type="range" min="0" max="48" value="0" step="1"><output id="position">frame 00 / 0.000 s</output><span class="phase">context frame 0-7</span></div>
<div class="legend"><span><i class="swatch formal"></i>existing formal sweep</span><span><i class="swatch new"></i>new single-case inference</span><span>30 FPS absolute-time alignment; source/context hold on their last available frame</span></div></div>
<div class="references"><figure class="reference"><video controls preload="metadata" src="{quote(manifest['source_video'])}"></video><figcaption><div><strong>Source / reference</strong><span class="tag">30 frames</span></div><p>Ground-truth clip from frame 0.</p><code>{html.escape(display_path(Path(manifest['source_path'])))}</code></figcaption></figure>
<figure class="reference"><video controls preload="metadata" src="{quote(manifest['context_video'])}"></video><figcaption><div><strong>Actual context input</strong><span class="tag">8 frames</span></div><p>Frames 0-7 supplied to Stage1a/Stage1b.</p><code>{html.escape(display_path(Path(manifest['context_path'])))}</code></figcaption></figure></div>
{''.join(rows)}</main><script>
const fps=30,maxFrame=48,slider=document.querySelector('#timeline'),position=document.querySelector('#position'),phase=document.querySelector('.phase');
const videos=[...document.querySelectorAll('video')],leader=document.querySelector('[data-sync-leader]');let timer=null,syncing=false;
const ready=v=>Number.isFinite(v.duration)&&v.duration>0;const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function seekVideo(v,frame){{if(!ready(v))return;const maxTime=Math.max(0,v.duration-1/fps);v.currentTime=Math.min(frame/fps,maxTime)}}
function renderPosition(frame){{position.value=`frame ${{String(frame).padStart(2,'0')}} / ${{(frame/fps).toFixed(3)}} s`;phase.textContent=frame<8?'context frame 0-7':'generated future frame 8-48'}}
function seekAll(frame){{syncing=true;videos.forEach(v=>seekVideo(v,frame));slider.value=frame;renderPosition(frame);setTimeout(()=>syncing=false,0)}}
videos.forEach(v=>v.addEventListener('loadedmetadata',()=>seekVideo(v,Number(slider.value))));slider.addEventListener('input',()=>seekAll(Number(slider.value)));
leader.addEventListener('timeupdate',()=>{{if(syncing)return;const frame=clamp(Math.round(leader.currentTime*fps),0,maxFrame);slider.value=frame;renderPosition(frame)}});
document.querySelector('[data-action=play]').onclick=()=>{{if(Number(slider.value)>=maxFrame)seekAll(0);videos.forEach(v=>{{if(ready(v)&&v.currentTime<v.duration-1/fps)v.play().catch(()=>{{}})}});if(timer)clearInterval(timer);timer=setInterval(()=>{{const frame=clamp(Math.round(leader.currentTime*fps),0,maxFrame);videos.forEach(v=>{{if(v!==leader&&ready(v)&&v.currentTime<v.duration-1/fps&&Math.abs(v.currentTime-frame/fps)>.08)seekVideo(v,frame)}})}},200)}};
document.querySelector('[data-action=pause]').onclick=()=>{{videos.forEach(v=>v.pause());if(timer)clearInterval(timer);timer=null}};
document.querySelector('[data-action=reset]').onclick=()=>{{videos.forEach(v=>v.pause());if(timer)clearInterval(timer);timer=null;seekAll(0)}};
leader.addEventListener('ended',()=>{{videos.forEach(v=>v.pause());if(timer)clearInterval(timer);timer=null;seekAll(maxFrame)}});
</script></body></html>"""


if __name__ == "__main__":
    result = build()
    print(
        f"gallery={OUTPUT_DIR / 'index.html'} rows={len(result['rows'])} "
        f"variants={sum(len(row['variants']) for row in result['rows'])}"
    )
