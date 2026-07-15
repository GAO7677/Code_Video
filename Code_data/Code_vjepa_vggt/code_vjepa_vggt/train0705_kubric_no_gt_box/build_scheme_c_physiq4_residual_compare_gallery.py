#!/usr/bin/env python3
"""Build a four-case Scheme-C checkpoint/residual comparison gallery."""
from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

import cv2


INPUT_DIR = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons")
CASE_STEMS = (
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed",
    "physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px",
)
FORMAL_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/"
    "train_stage1b_scheme_c_entity_caption_physical_fresh_20260714T174707Z"
)
SINGLE_CASE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "scheme_c_physiq025_step2500_3500_residual_compare_20260715/new_inference"
)
MULTI_CASE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "scheme_c_physiq4_step2500_3500_residual_compare_20260715/new_inference"
)
OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/"
    "scheme_c_physiq4_step2500_3500_residual_compare_20260715/gallery"
)
STEPS = ("step-002500", "step-003500")
SCALES = ("1.0", "1.2", "1.3", "1.4", "1.5", "2.0")
FORMAL_SCALES = {"1.0", "1.5", "2.0"}
SINGLE_CASE_STEM = "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed"


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


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


def display_path(path: Path) -> str:
    resolved = path.resolve()
    prefix = Path("/data/gaoya")
    try:
        return str(resolved.relative_to(prefix))
    except ValueError:
        return str(resolved)


def probe_video(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = capture.read()
    capture.release()
    if not ok or frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"could not decode video metadata: {path}")
    return {"frames": frames, "fps": fps, "width": width, "height": height}


def variant_root(stem: str, step: str, scale: str) -> tuple[Path, str]:
    tag = scale.replace(".", "p")
    suffix = Path(step) / f"object_residual_{tag}x/results"
    if scale in FORMAL_SCALES:
        return FORMAL_ROOT / suffix, "formal sweep"
    if stem == SINGLE_CASE_STEM:
        return SINGLE_CASE_ROOT / suffix, "intermediate sweep"
    return MULTI_CASE_ROOT / suffix, "intermediate sweep"


def build() -> dict:
    records = []
    for stem in CASE_STEMS:
        input_json = INPUT_DIR / f"{stem}.json"
        case = load_json(input_json)
        case_dir = OUTPUT_DIR / "videos" / stem
        source = Path(case["source_video"])
        context = Path(case["input_video"])
        source_probe = probe_video(source)
        context_probe = probe_video(context)
        safe_link(source, case_dir / "source_reference.mp4")
        safe_link(context, case_dir / "context_8f.mp4")
        rows = []
        for step in STEPS:
            variants = []
            for scale in SCALES:
                root, origin = variant_root(stem, step, scale)
                metadata = root / f"{stem}.json"
                payload = load_json(metadata)
                output_video = Path(payload["output_video"])
                tag = scale.replace(".", "p")
                video_link = case_dir / f"{step}_residual_{tag}x.mp4"
                metadata_link = case_dir / f"{step}_residual_{tag}x.json"
                safe_link(output_video, video_link)
                safe_link(metadata, metadata_link)
                variants.append(
                    {
                        "step": step,
                        "scale": scale,
                        "origin": origin,
                        "video": str(video_link.relative_to(OUTPUT_DIR)),
                        "metadata": str(metadata_link.relative_to(OUTPUT_DIR)),
                        "output_path": str(output_video.resolve()),
                        "display_output_path": display_path(output_video),
                        "seed": payload.get("seed"),
                        "guidance": payload.get("guidance"),
                        "inference_steps": payload.get("step"),
                        "negative_prompt": payload.get("negative_prompt"),
                    }
                )
            rows.append({"step": step, "variants": variants})
        records.append(
            {
                "case": stem,
                "input_json": str(input_json.resolve()),
                "caption": case["input_caption"],
                "source_video": str((case_dir / "source_reference.mp4").relative_to(OUTPUT_DIR)),
                "source_path": str(source.resolve()),
                "source_probe": source_probe,
                "context_video": str((case_dir / "context_8f.mp4").relative_to(OUTPUT_DIR)),
                "context_path": str(context.resolve()),
                "context_probe": context_probe,
                "rows": rows,
            }
        )

    manifest = {
        "title": "Scheme-C four-case residual comparison",
        "parameters": {
            "checkpoints": list(STEPS),
            "residual_scales": list(SCALES),
            "fps": 30,
            "output_frames": 49,
            "context_frames": 8,
            "resolution": "896x512",
            "inference_steps": 40,
            "cfg_scale": 5.0,
            "seed": 42,
            "negative_prompt": None,
        },
        "num_cases": len(records),
        "num_generated_videos": sum(
            len(row["variants"]) for record in records for row in record["rows"]
        ),
        "cases": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "index.html").write_text(render(manifest), encoding="utf-8")
    return manifest


def card(variant: dict, leader: bool) -> str:
    leader_attr = " data-sync-leader='true'" if leader else ""
    family = "formal" if variant["origin"] == "formal sweep" else "new"
    negative = "null" if variant["negative_prompt"] is None else "configured"
    return (
        f"<figure class='{family}'><video controls preload='metadata' data-fps='30' data-frames='49' "
        f"src='{quote(variant['video'])}'{leader_attr}></video><figcaption>"
        f"<div><strong>Residual {html.escape(variant['scale'])}x</strong>"
        f"<span class='tag'>{html.escape(variant['origin'])}</span></div>"
        f"<dl><dt>seed</dt><dd>{variant['seed']}</dd><dt>CFG</dt><dd>{variant['guidance']}</dd>"
        f"<dt>steps</dt><dd>{variant['inference_steps']}</dd><dt>negative</dt><dd>{negative}</dd></dl>"
        f"<code>{html.escape(variant['display_output_path'])}</code>"
        f"<a href='{quote(variant['metadata'])}' target='_blank'>metadata JSON</a>"
        "</figcaption></figure>"
    )


def render(manifest: dict) -> str:
    nav = "".join(
        f"<a href='#case-{index}'>{index + 1}. {html.escape(record['case'])}</a>"
        for index, record in enumerate(manifest["cases"])
    )
    sections = []
    for index, record in enumerate(manifest["cases"]):
        references = (
            f"<div class='references'><figure class='reference'><video controls preload='metadata' "
            f"data-fps='{record['source_probe']['fps']}' data-frames='{record['source_probe']['frames']}' src='{quote(record['source_video'])}'></video>"
            f"<figcaption><div><strong>Source / reference</strong><span class='tag'>{record['source_probe']['frames']}f @ {record['source_probe']['fps']:g} FPS</span></div><code>{html.escape(display_path(Path(record['source_path'])))}</code></figcaption></figure>"
            f"<figure class='reference'><video controls preload='metadata' data-fps='{record['context_probe']['fps']}' "
            f"data-frames='{record['context_probe']['frames']}' src='{quote(record['context_video'])}'></video>"
            f"<figcaption><div><strong>Actual context input</strong><span class='tag'>{record['context_probe']['frames']}f @ {record['context_probe']['fps']:g} FPS</span></div><code>{html.escape(display_path(Path(record['context_path'])))}</code></figcaption></figure></div>"
        )
        matrices = []
        for row in record["rows"]:
            cards = "".join(
                card(variant, row["step"] == "step-002500" and variant["scale"] == "1.0")
                for variant in row["variants"]
            )
            matrices.append(
                f"<div class='checkpoint'><header><h3>{html.escape(row['step'])}</h3>"
                "<span>Scheme-C entity-caption physical</span></header>"
                f"<div class='matrix'>{cards}</div></div>"
            )
        sections.append(
            f"<section id='case-{index}' data-case><header class='case-head'><span>CASE {index + 1:02d}</span>"
            f"<h2>{html.escape(record['case'])}</h2></header><p class='caption'>{html.escape(record['caption'])}</p>"
            "<div class='timeline'><button data-action='play' title='Play synchronized'>&#9654;</button>"
            "<button data-action='pause' title='Pause all'>II</button><button data-action='reset' title='Reset'>&#8634;</button>"
            "<input data-timeline type='range' min='0' max='48' value='0' step='1'>"
            "<output>frame 00 / 0.000 s</output><span data-phase>context frame 0-7</span></div>"
            f"{references}{''.join(matrices)}</section>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scheme-C Four-Case Residual Comparison</title><style>
:root{{--ink:#18201c;--muted:#68716c;--paper:#f3f4f1;--panel:#fff;--line:#c9cfcb;--green:#236747;--red:#9f342d;--blue:#315f7a;--amber:#9b641f}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px "IBM Plex Sans","Noto Sans",sans-serif;letter-spacing:0}}main{{max-width:1940px;margin:auto;padding:0 24px 70px}}
.top{{position:sticky;top:0;z-index:6;padding:15px 0 12px;background:rgba(243,244,241,.97);border-bottom:1px solid var(--line)}}h1{{font:700 26px "IBM Plex Serif","Noto Serif",serif;margin:0 0 5px}}.summary{{margin:0 0 9px;color:var(--muted)}}nav{{display:flex;gap:7px;overflow-x:auto;padding-bottom:2px}}nav a{{flex:none;padding:6px 8px;border:1px solid var(--line);background:#fff;color:var(--blue);font-size:11px;text-decoration:none;max-width:390px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
section{{padding:28px 0 38px;border-bottom:2px solid var(--line);scroll-margin-top:105px}}.case-head{{display:flex;gap:11px;align-items:baseline}}.case-head span{{color:var(--red);font:700 12px ui-monospace,monospace}}h2{{margin:0;font-size:20px;overflow-wrap:anywhere}}.caption{{max-width:1250px;margin:7px 0 12px;color:var(--muted);line-height:1.45}}
.timeline{{display:grid;grid-template-columns:34px 34px 34px minmax(220px,700px) 150px auto;gap:6px;align-items:center;margin-bottom:14px}}button{{width:34px;height:32px;border:1px solid #89938d;background:#fff;color:var(--ink);cursor:pointer}}input[type=range]{{width:100%;accent-color:var(--red)}}output{{font:12px ui-monospace,monospace}}[data-phase]{{font-size:12px;color:var(--muted)}}
.references{{display:grid;grid-template-columns:repeat(2,minmax(0,540px));gap:12px;margin-bottom:18px}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);min-width:0}}figure.reference{{border-top:4px solid var(--green)}}figure.formal{{border-top:4px solid var(--blue)}}figure.new{{border-top:4px solid var(--amber)}}video{{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#101210}}figcaption{{display:grid;gap:6px;padding:9px}}figcaption>div{{display:flex;justify-content:space-between;gap:8px}}.tag{{padding:2px 5px;border:1px solid currentColor;font:700 9px ui-monospace,monospace;text-transform:uppercase;white-space:nowrap}}code{{font:10px ui-monospace,monospace;color:#45504a;overflow-wrap:anywhere;word-break:break-word}}a{{color:var(--blue);font-size:12px;width:max-content}}dl{{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:3px 7px;margin:0;padding:6px 0;border-top:1px solid #e3e6e4;border-bottom:1px solid #e3e6e4;font:11px ui-monospace,monospace}}dt{{color:var(--muted)}}dd{{margin:0;text-align:right}}
.checkpoint{{overflow-x:auto;margin-top:16px}}.checkpoint>header{{display:flex;gap:10px;align-items:baseline;margin-bottom:8px}}h3{{margin:0;font-size:17px}}.checkpoint>header span{{color:var(--muted)}}.matrix{{display:grid;grid-template-columns:repeat(6,minmax(245px,1fr));gap:12px;min-width:1510px}}
@media(max-width:760px){{main{{padding:0 12px 50px}}.references{{grid-template-columns:1fr}}.timeline{{grid-template-columns:34px 34px 34px minmax(100px,1fr) 115px}}[data-phase]{{grid-column:4/6}}h1{{font-size:22px}}section{{scroll-margin-top:130px}}}}
</style></head><body><main><div class="top"><h1>Scheme-C Four-Case Residual Matrix</h1><p class="summary">4 cases / 2 checkpoints / 6 residual scales / 48 generated outputs. Parameters: 49 frames, context 8, output 30 FPS, 896x512, 40 steps, CFG 5.0, seed 42, null negative prompt. Playback is aligned by frame index across native source FPS.</p><nav>{nav}</nav></div>{''.join(sections)}</main><script>
const fps=30,maxFrame=48,clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
document.querySelectorAll('[data-case]').forEach(section=>{{const videos=[...section.querySelectorAll('video')],leader=section.querySelector('[data-sync-leader]'),slider=section.querySelector('[data-timeline]'),output=section.querySelector('output'),phase=section.querySelector('[data-phase]');let timer=null,syncing=false;
const ready=v=>Number.isFinite(v.duration)&&v.duration>0;const nativeFps=v=>Number(v.dataset.fps)||fps;const nativeFrames=v=>Number(v.dataset.frames)||Math.round(v.duration*nativeFps(v));const videoFrame=v=>Math.round(v.currentTime*nativeFps(v));const seekVideo=(v,f)=>{{if(ready(v))v.currentTime=Math.min(f,nativeFrames(v)-1)/nativeFps(v)}};const render=f=>{{output.value=`frame ${{String(f).padStart(2,'0')}} / ${{(f/fps).toFixed(3)}} output s`;phase.textContent=f<8?'context frame 0-7':'generated future frame 8-48'}};const seekAll=f=>{{syncing=true;videos.forEach(v=>seekVideo(v,f));slider.value=f;render(f);setTimeout(()=>syncing=false,0)}};
videos.forEach(v=>v.addEventListener('loadedmetadata',()=>seekVideo(v,Number(slider.value))));slider.addEventListener('input',()=>seekAll(Number(slider.value)));leader.addEventListener('timeupdate',()=>{{if(!syncing){{const f=clamp(Math.round(leader.currentTime*fps),0,maxFrame);slider.value=f;render(f)}}}});
section.querySelector('[data-action=play]').onclick=()=>{{if(Number(slider.value)>=maxFrame)seekAll(0);videos.forEach(v=>{{v.playbackRate=clamp(fps/nativeFps(v),.25,4);if(ready(v)&&videoFrame(v)<nativeFrames(v)-1)v.play().catch(()=>{{}})}});if(timer)clearInterval(timer);timer=setInterval(()=>{{const f=clamp(Math.round(leader.currentTime*fps),0,maxFrame);videos.forEach(v=>{{if(v!==leader&&ready(v)&&videoFrame(v)<nativeFrames(v)-1&&Math.abs(videoFrame(v)-f)>2)seekVideo(v,f)}})}},200)}};
section.querySelector('[data-action=pause]').onclick=()=>{{videos.forEach(v=>v.pause());if(timer)clearInterval(timer);timer=null}};section.querySelector('[data-action=reset]').onclick=()=>{{videos.forEach(v=>v.pause());if(timer)clearInterval(timer);timer=null;seekAll(0)}};leader.addEventListener('ended',()=>{{videos.forEach(v=>v.pause());if(timer)clearInterval(timer);timer=null;seekAll(maxFrame)}});}});
</script></body></html>"""


if __name__ == "__main__":
    result = build()
    print(
        f"gallery={OUTPUT_DIR / 'index.html'} cases={result['num_cases']} "
        f"generated={result['num_generated_videos']}"
    )
