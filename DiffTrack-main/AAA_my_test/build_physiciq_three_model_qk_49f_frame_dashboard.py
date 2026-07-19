#!/usr/bin/env python3
"""Render separate Q/K and CoTracker latent-timeline frames for 49-frame runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from center_query_utils import DEFAULT_CACHE, select_center_queries


DEFAULT_RESULT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk_49f"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk_49f_dashboard"
)
MODELS = (
    ("gt", "GT source"),
    ("stage1b", "Stage1b step-004000"),
    ("lora", "LoRA step-000500"),
    ("baseline", "Wan2.2 baseline"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def point_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(count):
        hue = int(round(179 * index / max(count, 1)))
        bgr = cv2.cvtColor(np.uint8([[[hue, 210, 245]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(value) for value in bgr))
    return colors


def read_video(path: Path, expected_frames: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != expected_frames:
        raise RuntimeError(f"expected {expected_frames} frames in {path}, got {len(frames)}")
    return np.stack(frames)


def draw_label(frame: np.ndarray, text: str) -> None:
    cv2.putText(frame, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 0), 3)
    cv2.putText(frame, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1)


def render_qk_frame(
    frame: np.ndarray, predictions: np.ndarray, latent: int, label: str
) -> np.ndarray:
    canvas = frame.copy()
    colors = point_colors(predictions.shape[1])
    for point_index, color in enumerate(colors):
        start = max(1, latent - 6)
        for previous in range(start, latent):
            pair = predictions[previous : previous + 2, point_index]
            if np.isfinite(pair).all():
                p0, p1 = (tuple(np.rint(point).astype(int)) for point in pair)
                cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
        if np.isfinite(predictions[latent, point_index]).all():
            point = tuple(np.rint(predictions[latent, point_index]).astype(int))
            cv2.rectangle(
                canvas, (point[0] - 5, point[1] - 5), (point[0] + 5, point[1] + 5), color, 2
            )
    draw_label(canvas, f"{label} | Q/K only | latent {latent} | pixel {latent * 4}")
    return canvas


def render_cotracker_frame(
    frame: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    anchors: np.ndarray,
    latent: int,
    label: str,
) -> np.ndarray:
    canvas = frame.copy()
    colors = point_colors(tracks.shape[1])
    for point_index, color in enumerate(colors):
        start = max(1, latent - 6)
        for previous in range(start, latent):
            f0, f1 = int(anchors[previous]), int(anchors[previous + 1])
            if visibility[f0, point_index] and visibility[f1, point_index]:
                p0 = tuple(np.rint(tracks[f0, point_index]).astype(int))
                p1 = tuple(np.rint(tracks[f1, point_index]).astype(int))
                cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
        pixel_frame = int(anchors[latent])
        if visibility[pixel_frame, point_index]:
            point = tuple(np.rint(tracks[pixel_frame, point_index]).astype(int))
            cv2.circle(canvas, point, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 3, color, -1, cv2.LINE_AA)
    draw_label(canvas, f"{label} | CoTracker only | latent {latent} | pixel {latent * 4}")
    return canvas


def center_metric(
    predictions: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    anchors: np.ndarray,
    point_index: int,
    query_latent: int,
    clean_prefix: int,
) -> dict:
    predicted = predictions[:, point_index]
    target = tracks[anchors, point_index]
    valid = visibility[anchors, point_index].copy()
    valid &= visibility[int(anchors[query_latent]), point_index]
    valid &= np.isfinite(predicted).all(axis=-1)
    valid[:clean_prefix] = False
    distances = np.linalg.norm(predicted - target, axis=-1)[valid]
    if not distances.size:
        return {"pck32": None, "mean_error_px": None, "comparisons": 0}
    return {
        "pck32": float(100.0 * (distances <= 32).mean()),
        "mean_error_px": float(distances.mean()),
        "comparisons": int(distances.size),
    }


def render_case(
    result_root: Path,
    output: Path,
    model: str,
    label: str,
    case_key: str,
    center_queries: dict[str, dict],
) -> dict:
    case_dir = result_root / model / "cases" / case_key
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    frame_count = int(manifest.get("generated_pixel_frames") or manifest["gt_pixel_frames"])
    video_name = "gt.mp4" if model == "gt" else "generated.mp4"
    frames = read_video(case_dir / video_name, frame_count)
    predictions = np.load(case_dir / "predicted_tracks.npz")["qk_layer23_step039_predictions"]
    cotracker = np.load(case_dir / "cotracker_pseudo_gt.npz")
    tracks = cotracker["tracks"]
    visibility = cotracker["visibility"].astype(bool)
    anchors = np.asarray(manifest["latent_anchor_pixel_frames"], dtype=np.int64)
    query_latent = int(manifest["query_latent_index"])
    clean_prefix = int(manifest["clean_prefix_latents"])
    regions = manifest["query_regions"]
    metrics = {}
    for region in regions:
        region_name = region["region_name"]
        point_index = int(center_queries[region_name]["global_index"])
        point_slice = slice(point_index, point_index + 1)
        qk = predictions[:, point_slice]
        cot_tracks = tracks[:, point_slice]
        cot_visibility = visibility[:, point_slice]
        metrics[region_name] = center_metric(
            predictions,
            tracks,
            visibility,
            anchors,
            point_index,
            query_latent,
            clean_prefix,
        )
        for latent in range(1, len(anchors)):
            pixel_frame = int(anchors[latent])
            qk_path = output / "frames" / model / case_key / region_name / "qk" / f"latent_{latent:02d}.jpg"
            cot_path = output / "frames" / model / case_key / region_name / "cotracker" / f"latent_{latent:02d}.jpg"
            qk_path.parent.mkdir(parents=True, exist_ok=True)
            cot_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(qk_path),
                render_qk_frame(frames[pixel_frame], qk, latent, label),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            cv2.imwrite(
                str(cot_path),
                render_cotracker_frame(
                    frames[pixel_frame], cot_tracks, cot_visibility, anchors, latent, label
                ),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
    return {
        "available_latents": len(anchors) - 1,
        "pixel_frames": frame_count,
        "regions": [region["region_name"] for region in regions],
        "metrics": metrics,
        "center_queries": center_queries,
    }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>49-frame Q/K and CoTracker</title><style>
:root{--paper:#eee9dc;--ink:#17211e;--card:#fffdf8;--line:#b8b09f;--rust:#b64a31;--teal:#176654;--muted:#65706b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 4% 0,#d7764b38,transparent 36rem),radial-gradient(circle at 96% 4%,#4b9a8038,transparent 36rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1960px,calc(100% - 24px));margin:auto;padding:26px 0 60px}h1{font-family:Georgia,"Noto Serif CJK SC",serif;font-size:clamp(38px,5vw,72px);line-height:.94;letter-spacing:-.04em;margin:0}.eyebrow{color:var(--rust);font-size:12px;font-weight:900;letter-spacing:.15em}.lead{max-width:1120px;color:var(--muted);line-height:1.6}.controls{display:grid;grid-template-columns:2fr 1fr;gap:10px;margin:20px 0}label{font-size:11px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}select{display:block;width:100%;margin-top:5px;padding:10px;border:1px solid var(--ink);background:var(--card);font-weight:800}.board{overflow-x:auto;border:1px solid var(--line);background:#d8d2c4;padding:8px}.timeline{min-width:3050px}.head,.track-row{display:grid;grid-template-columns:220px repeat(12,minmax(225px,1fr));gap:7px;margin-bottom:7px}.corner,.time,.row-label,.tile,.missing{background:var(--card);border:1px solid var(--line)}.corner,.time{padding:9px;font-weight:900}.time{text-align:center}.time small{display:block;color:var(--muted);margin-top:3px}.row-label{display:flex;flex-direction:column;justify-content:center;padding:12px}.row-label strong{font:700 17px/1.2 Georgia}.row-label span{color:var(--teal);font-weight:900;margin-top:5px}.row-label small{color:var(--muted);margin-top:6px}.tile{margin:0;background:#07100d}.tile img{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;cursor:zoom-in}.missing{display:flex;align-items:center;justify-content:center;color:var(--muted);font-weight:800;min-height:126px}.note{margin-top:14px;padding:13px;background:var(--card);border:1px solid var(--line);line-height:1.55}@media(max-width:720px){.controls{grid-template-columns:1fr}.head,.track-row{grid-template-columns:155px repeat(12,235px)}.timeline{min-width:max-content}}
</style></head><body><main><div class="eyebrow">49 PIXEL FRAMES · 13 WAN LATENTS · LAYER 23 · STEP 39</div><h1>Center-point Q/K and CoTracker<br>on separate timelines</h1><p class="lead">每个区域只跟踪一个中心代表点：物体取现有 query 中离 SAM2 mask 质心最近的点，背景取离画面中心最近的有效背景 query。Q/K 行只显示方框，CoTracker 行只显示圆点；列对应 latent 1–12，即 pixel frames 4–48。</p><section class="controls"><label>Case<select id="case"></select></label><label>Region<select id="region"></select></label></section><div class="board"><div class="timeline" id="timeline"></div></div><div class="note">GT case 1/3 有完整 49 帧；GT case 2/4 原视频只有 30 帧，因此只使用前 29 帧并展示到 latent 7，后续列标记为不可用。三个生成模型均为完整 49 帧。轨迹线只保留同一个中心点最近 6 个 latent 的历史位置。</div></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),caseEl=document.getElementById('case'),regionEl=document.getElementById('region'),timeline=document.getElementById('timeline');for(const c of data.cases){const o=document.createElement('option');o.value=c.case_key;o.textContent=c.label;caseEl.append(o)}function current(){return data.cases.find(c=>c.case_key===caseEl.value)}function setRegions(){regionEl.innerHTML=current().regions.map(r=>`<option value="${r}">${r}</option>`).join('')}const fmt=v=>v==null?'NA':Number(v).toFixed(1);function render(){const item=current(),region=regionEl.value,latents=Array.from({length:12},(_,i)=>i+1);const head=`<div class="head"><div class="corner">Model / track</div>${latents.map(l=>`<div class="time">latent ${l}<small>pixel ${l*4}</small></div>`).join('')}</div>`;const rows=[];for(const model of data.models){const info=item.models[model.name],metric=info.metrics[region]||{};for(const track of ['qk','cotracker']){const name=track==='qk'?'Q/K center only':'CoTracker center only',stat=track==='qk'?`center within CoTracker 32px ${fmt(metric.pck32)}% · error ${fmt(metric.mean_error_px)} px`:`${info.pixel_frames} source frames`;const cells=latents.map(l=>l<=info.available_latents?`<figure class="tile"><a target="_blank" href="frames/${model.name}/${item.case_key}/${region}/${track}/latent_${String(l).padStart(2,'0')}.jpg"><img loading="lazy" src="frames/${model.name}/${item.case_key}/${region}/${track}/latent_${String(l).padStart(2,'0')}.jpg"></a></figure>`:`<div class="missing">unavailable</div>`).join('');rows.push(`<div class="track-row"><div class="row-label"><strong>${model.label}</strong><span>${name}</span><small>${stat}</small></div>${cells}</div>`)}}timeline.innerHTML=head+rows.join('')}caseEl.addEventListener('change',()=>{setRegions();render()});regionEl.addEventListener('change',render);setRegions();render();
</script></body></html>'''


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    output = args.output_dir.resolve()
    cache_root = args.cache_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_keys = sorted(
        path.parent.name for path in (result_root / "stage1b" / "cases").glob("*/complete.json")
    )
    cases = []
    for case_key in case_keys:
        reference_manifest = json.loads(
            (result_root / "stage1b" / "cases" / case_key / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        center_queries = select_center_queries(
            cache_root, case_key, reference_manifest["query_regions"]
        )
        model_data = {}
        for model, label in MODELS:
            complete = result_root / model / "cases" / case_key / "complete.json"
            if not complete.is_file():
                raise RuntimeError(f"missing completed result: {complete}")
            model_data[model] = render_case(
                result_root, output, model, label, case_key, center_queries
            )
        cases.append(
            {
                "case_key": case_key,
                "label": case_key.removeprefix("case_physiciq_"),
                "regions": model_data["stage1b"]["regions"],
                "center_queries": center_queries,
                "models": model_data,
            }
        )
    payload = json.dumps(
        {
            "models": [{"name": name, "label": label} for name, label in MODELS],
            "cases": cases,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    (output / "index.html").write_text(
        HTML.replace("__PAYLOAD__", payload), encoding="utf-8"
    )
    print(f"Built {output / 'index.html'} with {len(cases)} cases")


if __name__ == "__main__":
    main()
