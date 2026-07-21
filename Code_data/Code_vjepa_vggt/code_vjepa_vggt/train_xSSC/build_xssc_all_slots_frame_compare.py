#!/usr/bin/env python3
"""Build a frame-by-frame All-slots comparison page from cached xSSC labels."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image


PALETTE = np.asarray(
    [
        [239, 68, 68],
        [59, 130, 246],
        [34, 197, 94],
        [250, 204, 21],
        [168, 85, 247],
        [6, 182, 212],
        [249, 115, 22],
    ],
    dtype=np.uint8,
)
DEFAULT_SOURCE = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_test5_source_slot_compare_5weights_ctx8_full"
)
DEFAULT_CACHE = Path(
    "/data/gaoya/agent-data/cache/"
    "xssc_test5_source_slot_compare_5weights_ctx8_full"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_test5_source_all_slots_frame_compare_5weights"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quality", type=int, default=88)
    return parser.parse_args()


def aligned_labels(raw_labels, raw_to_aligned):
    output = np.empty_like(raw_labels)
    for raw_id, aligned_id in raw_to_aligned.items():
        output[raw_labels == int(raw_id)] = int(aligned_id)
    return output


def all_slots_overlay(frame, labels):
    labels_full = labels.repeat(16, axis=0).repeat(16, axis=1)
    colors = PALETTE[labels_full]
    output = frame.astype(np.float32) * 0.43 + colors.astype(np.float32) * 0.57
    output = output.round().clip(0, 255).astype(np.uint8)
    for position in range(16, 256, 16):
        output[position, :, :] = (output[position, :, :].astype(np.float32) * 0.72).astype(np.uint8)
        output[:, position, :] = (output[:, position, :].astype(np.float32) * 0.72).astype(np.uint8)
    return output


def save_webp(path, frame, quality):
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path, format="WEBP", quality=quality, method=4)


def render_case(case, source_dir, cache_dir, output_dir, workers, quality):
    case_id = case["id"]
    rgb = np.load(cache_dir / "inputs" / case_id / "rgb.npy")
    case_dir = output_dir / "cases" / case_id
    jobs = []
    for frame_id, frame in enumerate(rgb):
        jobs.append((case_dir / "original" / f"{frame_id:04d}.webp", frame))

    models = []
    for model in case["models"]:
        model_id = model["id"]
        with np.load(source_dir / "labels" / model_id / f"{case_id}.npz") as payload:
            raw = payload["labels"]
        labels = aligned_labels(raw, model["alignment"]["raw_to_aligned"])
        if len(labels) != len(rgb):
            raise RuntimeError(f"frame mismatch for {case_id}/{model_id}: {len(labels)} != {len(rgb)}")
        for frame_id, frame in enumerate(rgb):
            jobs.append(
                (
                    case_dir / model_id / f"{frame_id:04d}.webp",
                    all_slots_overlay(frame, labels[frame_id]),
                )
            )
        models.append(
            {
                "id": model_id,
                "label": model["label"],
                "architecture": model["architecture"],
                "mean_matched_iou": model["alignment"]["mean_matched_iou"],
                "frame_pattern": f"cases/{case_id}/{model_id}/{{frame}}.webp",
            }
        )

    def execute(job):
        save_webp(job[0], job[1], quality)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(execute, jobs))
    return {
        "id": case_id,
        "json": case["json"],
        "source_video": case["source_video"],
        "frames": case["frames"],
        "fps": case["fps"],
        "ctx_frames": case["ctx_frames"],
        "original_pattern": f"cases/{case_id}/original/{{frame}}.webp",
        "models": models,
    }


def build_html(metadata):
    data = json.dumps(metadata, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>xSSC All-slots frame comparison</title>
<style>
*{{box-sizing:border-box}}:root{{color-scheme:dark}}body{{margin:0;background:#101214;color:#f3f4f6;font:14px system-ui,sans-serif;letter-spacing:0}}header{{position:sticky;top:0;z-index:4;background:rgba(16,18,20,.98);border-bottom:1px solid #353a40}}.bar{{max-width:1900px;margin:auto;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}h1{{font-size:18px;margin:0 auto 0 0}}button,select,input{{font:inherit}}select,.icon{{height:34px;border:1px solid #4b525a;border-radius:5px;background:#202429;color:#f5f6f7}}select{{padding:0 28px 0 9px}}.icon{{width:34px;cursor:pointer}}.icon:hover{{background:#2b3036}}#frameSlider{{min-width:220px;flex:0 1 420px;accent-color:#38bdf8}}.mode{{display:flex;border:1px solid #4b525a;border-radius:5px;overflow:hidden}}.mode button{{height:32px;border:0;border-right:1px solid #4b525a;background:#202429;color:#c4c9cf;padding:0 11px;cursor:pointer}}.mode button:last-child{{border-right:0}}.mode button.active{{background:#0369a1;color:#fff}}#counter{{min-width:105px;color:#c4c9cf;font-variant-numeric:tabular-nums}}main{{max-width:1900px;margin:auto;padding:16px}}.meta{{display:flex;gap:16px;padding-bottom:14px;color:#aeb5bd;white-space:nowrap;overflow:auto}}.compare{{display:grid;grid-template-columns:repeat(6,minmax(230px,1fr));gap:12px;min-width:1450px}}.wrap{{overflow:auto}}figure{{margin:0;min-width:0}}img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#050607;border:1px solid #34383d;border-radius:4px}}figcaption{{padding:7px 2px 0;min-height:42px;color:#c4c9cf}}figcaption strong{{display:block;color:#f3f4f6;margin-bottom:2px}}.metric{{color:#7dd3fc;font-size:12px}}.legend{{display:flex;gap:9px;flex-wrap:wrap;padding-top:18px;color:#aeb5bd}}.legend span{{display:inline-flex;align-items:center;gap:5px}}.swatch{{width:12px;height:12px;border-radius:2px}}@media(max-width:760px){{h1{{width:100%}}main{{padding:11px}}}}
</style>
</head>
<body>
<header><div class="bar"><h1>xSSC All-slots frame comparison</h1><select id="caseSelect" aria-label="Case"></select><div class="mode"><button id="ctxMode" class="active">ctx8</button><button id="fullMode">full</button></div><button id="previous" class="icon" title="Previous frame" aria-label="Previous frame">&#8249;</button><button id="next" class="icon" title="Next frame" aria-label="Next frame">&#8250;</button><input id="frameSlider" type="range" min="0" max="7" step="1" value="0" aria-label="Frame"><span id="counter"></span></div></header>
<main><div id="meta" class="meta"></div><div class="wrap"><section id="compare" class="compare"></section></div><div id="legend" class="legend"></div></main>
<script>
const DATA={data};const caseSelect=document.getElementById('caseSelect');const slider=document.getElementById('frameSlider');const counter=document.getElementById('counter');const compare=document.getElementById('compare');const meta=document.getElementById('meta');const ctxMode=document.getElementById('ctxMode');const fullMode=document.getElementById('fullMode');let mode='ctx8';let frame=0;
function pattern(path,id){{return path.replace('{{frame}}',String(id).padStart(4,'0'))}}function item(){{return DATA.cases[Number(caseSelect.value)]}}function maxFrame(){{const c=item();return (mode==='ctx8'?c.ctx_frames:c.frames)-1}}function panels(c){{return [{{label:'Effective input',sub:'model-visible crop',pattern:c.original_pattern}},...c.models.map(m=>({{label:m.label,sub:`${{m.architecture}} | mean IoU ${{m.mean_matched_iou.toFixed(3)}}`,pattern:m.frame_pattern}}))]}}
function updateImages(){{const c=item();frame=Math.max(0,Math.min(frame,maxFrame()));slider.value=String(frame);counter.textContent=`frame ${{frame+1}} / ${{maxFrame()+1}} | ${{(frame/c.fps).toFixed(3)}} s`;const entries=panels(c);const figures=Array.from(compare.querySelectorAll('figure'));entries.forEach((entry,index)=>{{figures[index].querySelector('img').src=pattern(entry.pattern,frame);figures[index].querySelector('img').alt=`${{entry.label}} frame ${{frame}}`}})}}
function renderCase(){{const c=item();frame=0;slider.max=String(maxFrame());meta.innerHTML=`<span>${{c.id}}</span><span>${{c.frames}} frames</span><span>${{c.fps.toFixed(2)}} fps</span><span>${{c.source_video}}</span>`;compare.innerHTML=panels(c).map(entry=>`<figure><img decoding="async"><figcaption><strong>${{entry.label}}</strong><span class="metric">${{entry.sub}}</span></figcaption></figure>`).join('');updateImages()}}function setMode(next){{mode=next;ctxMode.classList.toggle('active',mode==='ctx8');fullMode.classList.toggle('active',mode==='full');frame=Math.min(frame,maxFrame());slider.max=String(maxFrame());updateImages()}}function step(delta){{frame=Math.max(0,Math.min(maxFrame(),frame+delta));updateImages()}}
DATA.cases.forEach((c,index)=>{{const option=document.createElement('option');option.value=String(index);option.textContent=`${{String(index+1).padStart(2,'0')}} | ${{c.id}}`;caseSelect.appendChild(option)}});document.getElementById('legend').innerHTML=DATA.palette.map((color,index)=>`<span><i class="swatch" style="background:rgb(${{color.join(',')}})"></i>slot ${{index}}</span>`).join('');caseSelect.addEventListener('change',renderCase);slider.addEventListener('input',()=>{{frame=Number(slider.value);updateImages()}});document.getElementById('previous').addEventListener('click',()=>step(-1));document.getElementById('next').addEventListener('click',()=>step(1));ctxMode.addEventListener('click',()=>setMode('ctx8'));fullMode.addEventListener('click',()=>setMode('full'));document.addEventListener('keydown',event=>{{if(event.key==='ArrowLeft')step(-1);if(event.key==='ArrowRight')step(1)}});renderCase();
</script>
</body>
</html>"""


def main():
    args = parse_args()
    source_dir = args.source_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = json.loads((source_dir / "metadata.json").read_text())
    cases = []
    for position, case in enumerate(source_metadata["cases"], start=1):
        rendered = render_case(
            case,
            source_dir,
            cache_dir,
            output_dir,
            args.workers,
            args.quality,
        )
        cases.append(rendered)
        print(f"[frames] {position}/{len(source_metadata['cases'])} {case['id']}", flush=True)
    metadata = {
        "title": "xSSC All-slots frame comparison",
        "source_metadata": str(source_dir / "metadata.json"),
        "unique_cases": len(cases),
        "models": source_metadata["models"],
        "anchor_model": source_metadata["anchor_model"],
        "alignment": source_metadata["alignment"],
        "ctx_frames": source_metadata["ctx_frames"],
        "palette": PALETTE.tolist(),
        "cases": cases,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "index.html").write_text(build_html(metadata))
    image_count = sum(case["frames"] * 6 for case in cases)
    print(json.dumps({"output_dir": str(output_dir), "cases": len(cases), "images": image_count}, indent=2))


if __name__ == "__main__":
    main()
