#!/usr/bin/env python3
"""Build a crop-vs-padding viewer with frame slider rows and video rows.

This script consumes existing xSSC all-slot overlay frame outputs. It does not
run xSSC inference. It can also encode each panel's webp frame sequence into an
mp4 video used by the video rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Iterable


DEFAULT_VIEWER_DIR = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_slot_overlay_test5_crop_padding_compare_plus24000"
)
DEFAULT_OUTPUTS_ROOT = Path("/data/gaoya/agent-data/outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--combined-metadata", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--force-videos", action="store_true")
    parser.add_argument("--skip-video-encode", action="store_true")
    parser.add_argument("--crf", type=int, default=19)
    return parser.parse_args()


def load_metadata(viewer_dir: Path, metadata_path: Path | None) -> dict:
    path = metadata_path or (viewer_dir / "combined_metadata.json")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if "crop_dir" not in metadata or "padding_dir" not in metadata:
        raise ValueError(f"{path} is not a crop/padding combined metadata file")
    return metadata


def iter_labels(case: dict, mode: str) -> Iterable[str]:
    yield "original"
    for model in case[mode]["models"]:
        yield model["label"]


def frame_dir(outputs_root: Path, source_dir: str, case_id: str, label: str) -> Path:
    return outputs_root / source_dir / "cases" / case_id / label


def video_path(viewer_dir: Path, case_id: str, mode: str, label: str) -> Path:
    return viewer_dir / "videos" / case_id / mode / f"{label}.mp4"


def get_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def encode_with_ffmpeg(
    ffmpeg: str,
    frames: Path,
    frame_count: int,
    output: Path,
    fps: float,
    crf: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-start_number",
        "0",
        "-i",
        str(frames / "%04d.webp"),
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def encode_with_cv2(frames: Path, frame_count: int, output: Path, fps: float) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    output.parent.mkdir(parents=True, exist_ok=True)
    first = Image.open(frames / "0000.webp").convert("RGB")
    width, height = first.size
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output}")
    try:
        for frame_id in range(frame_count):
            image = Image.open(frames / f"{frame_id:04d}.webp").convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BICUBIC)
            writer.write(np.asarray(image)[:, :, ::-1])
    finally:
        writer.release()


def ensure_video(
    source_frames: Path,
    frame_count: int,
    output: Path,
    fps: float,
    crf: int,
    force: bool,
    ffmpeg: str | None,
) -> str:
    if output.is_file() and output.stat().st_size > 1024 and not force:
        return "skip"
    if not (source_frames / "0000.webp").is_file():
        raise FileNotFoundError(source_frames / "0000.webp")
    if not (source_frames / f"{frame_count - 1:04d}.webp").is_file():
        raise FileNotFoundError(source_frames / f"{frame_count - 1:04d}.webp")
    if ffmpeg:
        try:
            encode_with_ffmpeg(ffmpeg, source_frames, frame_count, output, fps, crf)
            return "ffmpeg"
        except subprocess.CalledProcessError:
            if output.exists():
                output.unlink()
    encode_with_cv2(source_frames, frame_count, output, fps)
    return "cv2"


def ensure_videos(
    metadata: dict,
    outputs_root: Path,
    viewer_dir: Path,
    fps: float,
    crf: int,
    force: bool,
) -> dict:
    ffmpeg = get_ffmpeg()
    stats = {"ffmpeg": 0, "cv2": 0, "skip": 0}
    for case_index, case in enumerate(metadata["cases"], start=1):
        case_id = case["case_id"]
        frames = int(case["frames"])
        for mode, source_key in (("crop", "crop_dir"), ("padding", "padding_dir")):
            for label in iter_labels(case, mode):
                status = ensure_video(
                    frame_dir(outputs_root, metadata[source_key], case_id, label),
                    frames,
                    video_path(viewer_dir, case_id, mode, label),
                    fps,
                    crf,
                    force,
                    ffmpeg,
                )
                stats[status] += 1
        print(f"[video] {case_index}/{len(metadata['cases'])} {case_id}", flush=True)
    return stats


def build_html(metadata: dict, fps: float) -> str:
    data = json.dumps(metadata, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>xSSC Slot Overlay Frame and Video Viewer</title>
<style>
*{{box-sizing:border-box}}
:root{{color-scheme:dark}}
body{{margin:0;background:#101214;color:#f4f5f6;font:14px Arial,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;background:#171a1d;border-bottom:1px solid #2d3338}}
.bar{{max-width:2400px;margin:auto;padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
h1{{font-size:18px;margin:0 auto 0 0;font-weight:700}}
select,input,button{{height:34px;border:1px solid #4b525a;border-radius:5px;background:#202429;color:#f5f6f7;font:inherit}}
select{{padding:0 28px 0 9px;max-width:min(900px,56vw)}}
button{{min-width:72px;padding:0 10px;cursor:pointer}}
#frameSlider{{min-width:220px;flex:0 1 440px;accent-color:#38bdf8}}
#counter,#videoStatus{{min-width:92px;color:#c4c9cf}}
main{{max-width:2400px;margin:auto;padding:16px}}
.caseMeta{{display:flex;gap:16px;padding-bottom:14px;color:#aeb5bd;white-space:nowrap;overflow:auto}}
.mode{{margin:0 0 24px}}
.modeTitle{{display:flex;align-items:baseline;gap:10px;margin:0 0 10px}}
.modeTitle h2{{font-size:17px;margin:0}}
.modeTitle span{{color:#aeb5bd}}
.rowTitle{{margin:12px 0 8px;font-weight:700;color:#d8dde2}}
.scroller{{overflow-x:auto;border:1px solid #2b3137;background:#15181b;padding:10px}}
.grid{{display:grid;grid-template-columns:repeat(var(--panel-count,6),minmax(230px,1fr));gap:12px;min-width:calc(var(--panel-count,6) * 242px)}}
.similarityGrid{{display:grid;grid-template-columns:repeat(var(--panel-count,6),minmax(520px,1fr));gap:12px;min-width:calc(var(--panel-count,6) * 532px)}}
figure{{margin:0;background:#1c2024;border:1px solid #313840;border-radius:6px;overflow:hidden}}
figcaption{{padding:8px 10px;color:#c4c9cf;min-height:52px}}
strong{{display:block;color:#f3f4f6;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.metric{{color:#7dd3fc;font-size:12px}}
img,video{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#050607;border-bottom:1px solid #313840}}
img.similarity{{aspect-ratio:auto}}
</style>
</head>
<body>
<header>
  <div class="bar">
    <h1>xSSC Slot Overlay Frame and Video Viewer</h1>
    <select id="caseSelect"></select>
    <button id="prevFrame">Prev</button>
    <button id="nextFrame">Next</button>
    <input id="frameSlider" type="range" min="0" max="0" step="1" value="0">
    <span id="counter"></span>
    <button id="videoPlay">Video Play</button>
    <button id="videoRestart">Restart</button>
    <span id="videoStatus">paused</span>
  </div>
</header>
<main>
  <div id="caseMeta" class="caseMeta"></div>
  <section class="mode">
    <div class="modeTitle"><h2>Crop</h2><span>center crop preprocessing</span></div>
    <div class="rowTitle">Frame Slider</div>
    <div class="scroller"><div id="cropFrameGrid" class="grid"></div></div>
    <div class="rowTitle">Video</div>
    <div class="scroller"><div id="cropVideoGrid" class="grid"></div></div>
    <div class="rowTitle">Slot Embedding Temporal Similarity</div>
    <div class="scroller"><div id="cropSimilarityGrid" class="similarityGrid"></div></div>
    <div class="rowTitle">Slot Embedding Frequency Similarity</div>
    <div class="scroller"><div id="cropFrequencyGrid" class="similarityGrid"></div></div>
  </section>
  <section class="mode">
    <div class="modeTitle"><h2>Resize + Padding</h2><span>aspect-ratio preserving preprocessing</span></div>
    <div class="rowTitle">Frame Slider</div>
    <div class="scroller"><div id="paddingFrameGrid" class="grid"></div></div>
    <div class="rowTitle">Video</div>
    <div class="scroller"><div id="paddingVideoGrid" class="grid"></div></div>
    <div class="rowTitle">Slot Embedding Temporal Similarity</div>
    <div class="scroller"><div id="paddingSimilarityGrid" class="similarityGrid"></div></div>
    <div class="rowTitle">Slot Embedding Frequency Similarity</div>
    <div class="scroller"><div id="paddingFrequencyGrid" class="similarityGrid"></div></div>
  </section>
</main>
<script>
const DATA={data};
const FPS={fps:g};
const caseSelect=document.getElementById('caseSelect');
const slider=document.getElementById('frameSlider');
const counter=document.getElementById('counter');
const caseMeta=document.getElementById('caseMeta');
const videoPlay=document.getElementById('videoPlay');
const videoRestart=document.getElementById('videoRestart');
const videoStatus=document.getElementById('videoStatus');
let caseIndex=0;
let frameIndex=0;
let videosPlaying=false;

function currentCase(){{return DATA.cases[caseIndex];}}
function frameToken(i){{return String(i).padStart(4,'0');}}
function frameUrl(mode,item,frame){{const root=mode==='crop'?DATA.crop_dir:DATA.padding_dir;return `../${{root}}/${{item.pattern.replace('{{frame}}',frameToken(frame))}}`;}}
function videoUrl(caseId,mode,label){{return `videos/${{caseId}}/${{mode}}/${{label}}.mp4`;}}
function allVideos(){{return Array.from(document.querySelectorAll('video'));}}
function setText(el,text){{el.textContent=text;}}
function makeCaption(label,detail){{const cap=document.createElement('figcaption');const strong=document.createElement('strong');strong.textContent=label;const span=document.createElement('span');span.className='metric';span.textContent=detail;cap.appendChild(strong);cap.appendChild(span);return cap;}}
function makeFramePanel(mode,item){{const fig=document.createElement('figure');const img=document.createElement('img');img.dataset.pattern=item.pattern;img.dataset.mode=mode;img.alt=item.label;fig.appendChild(img);fig.appendChild(makeCaption(item.label,item.detail));return fig;}}
function makeVideoPanel(mode,item,caseId){{const fig=document.createElement('figure');const video=document.createElement('video');video.src=videoUrl(caseId,mode,item.label);video.muted=true;video.playsInline=true;video.preload='metadata';video.dataset.label=item.label;fig.appendChild(video);fig.appendChild(makeCaption(item.label,item.detail));return fig;}}
function makeSimilarityPanel(item){{const fig=document.createElement('figure');const img=document.createElement('img');img.src=item.chart;img.className='similarity';img.alt=`${{item.label}} temporal slot similarity`;fig.appendChild(img);fig.appendChild(makeCaption(item.label,item.detail));return fig;}}
function makeFrequencyPanel(item){{const fig=document.createElement('figure');const img=document.createElement('img');img.src=item.frequencyChart;img.className='similarity';img.alt=`${{item.label}} frequency slot similarity`;fig.appendChild(img);fig.appendChild(makeCaption(item.label,item.frequencyDetail));return fig;}}
function panelItems(c,mode){{return [{{label:'original',detail:'input frames',pattern:c[mode].original_pattern || c[mode].originalPattern || c[mode].original_pattern}}].concat(c[mode].models.map(m=>({{label:m.label,detail:`${{m.slots}} slots | ${{m.condition}}`,pattern:m.frame_pattern}})));}}
function similarityItems(c,mode){{const root=DATA.temporal_similarity?.cases?.[c.case_id]?.[mode] || [];return root.map(item=>({{label:item.label,chart:item.chart,detail:`fixed adj ${{item.metrics.adjacent_fixed_mean.toFixed(4)}} | matched ${{item.metrics.adjacent_matched_mean.toFixed(4)}} | ID ${{(item.metrics.adjacent_identity_rate*100).toFixed(1)}}%`,frequencyChart:item.frequency_chart,frequencyDetail:`amplitude ${{item.frequency_metrics.amplitude_similarity_offdiag_mean.toFixed(4)}} | phase ${{item.frequency_metrics.phase_coherence_offdiag_mean.toFixed(4)}} | centroid ${{item.frequency_metrics.spectral_centroid_cycles_per_frame.toFixed(4)}} cyc/frame`}}));}}
function fillGrid(grid,items,makePanel){{grid.style.setProperty('--panel-count',String(items.length));grid.replaceChildren(...items.map(makePanel));}}
function updateFrames(){{const c=currentCase();frameIndex=Math.max(0,Math.min(frameIndex,c.frames-1));slider.value=String(frameIndex);counter.textContent=`${{frameIndex+1}} / ${{c.frames}}`;for(const img of document.querySelectorAll('img[data-pattern]')){{img.src=frameUrl(img.dataset.mode,{{pattern:img.dataset.pattern}},frameIndex);}}}}
function pauseVideos(){{for(const v of allVideos())v.pause();videosPlaying=false;videoPlay.textContent='Video Play';videoStatus.textContent='paused';}}
async function playVideos(){{const videos=allVideos();if(videos.length===0)return;let t=videos[0].ended?0:videos[0].currentTime;for(const v of videos){{if(v.ended)v.currentTime=0;else if(Math.abs(v.currentTime-t)>0.08)v.currentTime=t;}}videosPlaying=true;videoPlay.textContent='Pause';videoStatus.textContent='playing';await Promise.allSettled(videos.map(v=>v.play()));}}
function restartVideos(){{for(const v of allVideos())v.currentTime=0;if(videosPlaying)playVideos();}}
function render(){{pauseVideos();const c=currentCase();frameIndex=0;slider.max=String(c.frames-1);setText(caseMeta,`${{caseIndex+1}} / ${{DATA.cases.length}}   ${{c.case_id}}   ${{c.frames}} frames`);for(const mode of ['crop','padding']){{const items=panelItems(c,mode);fillGrid(document.getElementById(`${{mode}}FrameGrid`),items,item=>makeFramePanel(mode,item));fillGrid(document.getElementById(`${{mode}}VideoGrid`),items,item=>makeVideoPanel(mode,item,c.case_id));const similarity=similarityItems(c,mode);fillGrid(document.getElementById(`${{mode}}SimilarityGrid`),similarity,makeSimilarityPanel);fillGrid(document.getElementById(`${{mode}}FrequencyGrid`),similarity,makeFrequencyPanel);}}for(const v of allVideos()){{v.addEventListener('ended',()=>{{if(allVideos().every(x=>x.paused||x.ended))pauseVideos();}});}}updateFrames();}}
DATA.cases.forEach((c,i)=>{{const option=document.createElement('option');option.value=String(i);option.textContent=`${{String(i+1).padStart(2,'0')}} | ${{c.case_id}}`;caseSelect.appendChild(option);}});
caseSelect.addEventListener('change',()=>{{caseIndex=Number(caseSelect.value);render();}});
slider.addEventListener('input',()=>{{frameIndex=Number(slider.value);updateFrames();}});
document.getElementById('prevFrame').addEventListener('click',()=>{{frameIndex-=1;updateFrames();}});
document.getElementById('nextFrame').addEventListener('click',()=>{{frameIndex+=1;updateFrames();}});
videoPlay.addEventListener('click',()=>{{videosPlaying?pauseVideos():playVideos();}});
videoRestart.addEventListener('click',restartVideos);
document.addEventListener('keydown',event=>{{if(event.key==='ArrowLeft'){{frameIndex-=1;updateFrames();}}if(event.key==='ArrowRight'){{frameIndex+=1;updateFrames();}}if(event.key===' '){{event.preventDefault();videosPlaying?pauseVideos():playVideos();}}}});
render();
</script>
</body>
</html>
"""


def write_readme(
    viewer_dir: Path, stats: dict, fps: float, metadata: dict
) -> None:
    temporal = metadata.get("temporal_similarity")
    temporal_text = ""
    if temporal:
        temporal_text = (
            "\n## Temporal slot similarity\n\n"
            f"{temporal['method']}\n\n"
            "Each model panel includes fixed-ID adjacent similarity, "
            "Hungarian-matched adjacent similarity, drift from frame 0, "
            "and an all-frame fixed-ID similarity matrix.\n\n"
            f"{temporal.get('frequency_method', '')}\n\n"
            "Frequency panels include per-slot amplitude spectra, "
            "slot-to-slot amplitude-spectrum cosine similarity, "
            "amplitude-weighted phase coherence, and global dynamic power.\n\n"
            "Checkpoints:\n\n"
            + "".join(
                f"- `{item['label']}`: `{item['checkpoint']}`\n"
                for item in temporal["models"]
            )
            + "\n"
        )
    text = (
        "# xSSC crop/padding frame and video viewer\n\n"
        "This viewer is built from existing all-slot overlay webp frames. It shows a manual frame-slider row and a separate mp4 video row for each preprocessing mode.\n\n"
        f"- Video fps: {fps:g}\n"
        f"- Video encode stats: {stats}\n\n"
        f"{temporal_text}"
        "Serve the outputs root with:\n\n"
        "```bash\n"
        "cd /data/gaoya/agent-data/outputs && python3 -m http.server 8897 --bind 0.0.0.0\n"
        "```\n"
    )
    (viewer_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    viewer_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(viewer_dir, args.combined_metadata)
    stats = {"ffmpeg": 0, "cv2": 0, "skip": 0}
    if not args.skip_video_encode:
        stats = ensure_videos(
            metadata,
            outputs_root,
            viewer_dir,
            args.fps,
            args.crf,
            args.force_videos,
        )
    (viewer_dir / "index.html").write_text(build_html(metadata, args.fps), encoding="utf-8")
    write_readme(viewer_dir, stats, args.fps, metadata)
    print(
        json.dumps(
            {
                "viewer_dir": str(viewer_dir),
                "index_html": str(viewer_dir / "index.html"),
                "cases": len(metadata["cases"]),
                "video_stats": stats,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
