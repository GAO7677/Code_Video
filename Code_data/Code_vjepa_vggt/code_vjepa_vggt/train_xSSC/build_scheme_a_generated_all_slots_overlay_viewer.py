#!/usr/bin/env python3
"""Overlay frozen xSSC all-slot assignments on Scheme A generated videos."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from code_vjepa_vggt.train_xSSC import train_xssc_allframe_oracle_slots as train


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

DEFAULT_GENERATED_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5/"
    "train_xssc_allframe_oracle_slots"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/AAA_xSSC/"
    "scheme_a_generated_all_slots_overlay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--step", action="append", default=[])
    parser.add_argument("--case-substring", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--max-frames", type=int, default=49)
    parser.add_argument("--alpha", type=float, default=0.57)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quality", type=int, default=88)
    return parser.parse_args()


def _safe_id(path: Path, generated_root: Path) -> str:
    rel = path.relative_to(generated_root).with_suffix("")
    return "__".join(rel.parts)


def _load_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _discover_videos(args: argparse.Namespace) -> list[Path]:
    generated_root = args.generated_root.expanduser().resolve()
    videos = sorted(generated_root.glob("step-*/step-*/*.mp4"))
    if args.step:
        allowed = set(args.step)
        videos = [p for p in videos if p.parent.name in allowed or p.parent.parent.name in allowed]
    substrings = [str(v) for v in args.case_substring if str(v)]
    if substrings:
        videos = [p for p in videos if any(token in p.stem for token in substrings)]
    if int(args.limit) > 0:
        videos = videos[: int(args.limit)]
    return videos


def _build_minimal_model(device: torch.device):
    model = SimpleNamespace()
    xssc, slot_dim, num_slots = train._load_xssc_model(
        xssc_root=train.DEFAULT_XSSC_ROOT,
        config_path=train.DEFAULT_XSSC_CONFIG,
        checkpoint_path=train.DEFAULT_XSSC_CHECKPOINT,
        device=device,
    )
    model.xssc = xssc
    model.xssc_slot_dim = slot_dim
    model.xssc_num_slots = num_slots
    model.xssc_input_size = 256
    return model


def _preprocess_generated_frames(frames: np.ndarray, *, height: int, width: int) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected generated video [T,H,W,3], got {frames.shape}")
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    if (int(frames.shape[1]), int(frames.shape[2])) != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    tensor = tensor / 255.0 * 2.0 - 1.0
    return tensor.permute(1, 0, 2, 3).contiguous()


@torch.no_grad()
def _extract_attention(model, video_cthw: torch.Tensor) -> torch.Tensor:
    xssc_video = train.XSSCAllFrameOracleSlotsWanModule._preprocess_xssc(model, video_cthw.unsqueeze(0))
    model.xssc.eval()
    batch, time_steps, _, _, _ = xssc_video.shape
    flat_video = xssc_video.flatten(0, 1)
    autocast_enabled = flat_video.device.type == "cuda"
    with torch.autocast(
        device_type=flat_video.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        feature = model.xssc.encode_backbone(flat_video).detach()
        _, _, feature_h, feature_w = feature.shape
        encoded = feature.permute(0, 2, 3, 1)
        encoded = model.xssc.encode_posit_embed(encoded).flatten(1, 2)
        encoded = model.xssc.encode_project(encoded)
        encoded = encoded.view(batch, time_steps, encoded.shape[1], encoded.shape[2])
        slots = None
        attentions = []
        for frame_id in range(time_steps):
            if frame_id == 0:
                query = model.xssc.initializ(batch)
            else:
                query = model.xssc.transit(slots, encoded[:, : frame_id + 1])
            num_iter = None if frame_id == 0 else 1
            current_slots, current_attention = model.xssc.aggregat(
                encoded[:, frame_id], query, num_iter=num_iter
            )
            slots = current_slots[:, None] if slots is None else torch.cat((slots, current_slots[:, None]), dim=1)
            attentions.append(current_attention.view(batch, model.xssc_num_slots, feature_h, feature_w))
    return torch.stack(attentions, dim=1)[0].float().cpu()


def _all_slots_overlay(frame: np.ndarray, labels_hw: np.ndarray, *, alpha: float) -> np.ndarray:
    h, w = frame.shape[:2]
    crop = min(h, w)
    top = (h - crop) // 2
    left = (w - crop) // 2
    labels_full = np.asarray(labels_hw, dtype=np.int64).repeat(crop // labels_hw.shape[0], axis=0).repeat(
        crop // labels_hw.shape[1], axis=1
    )
    if labels_full.shape != (crop, crop):
        labels_full = np.asarray(
            Image.fromarray(labels_hw.astype(np.uint8)).resize((crop, crop), Image.Resampling.NEAREST),
            dtype=np.int64,
        )
    colors = PALETTE[labels_full % len(PALETTE)]
    output = frame.copy().astype(np.float32)
    square = output[top : top + crop, left : left + crop]
    square[:] = square * (1.0 - alpha) + colors.astype(np.float32) * alpha
    for position in range(16, crop, max(16, crop // 16)):
        square[position : position + 1, :, :] *= 0.72
        square[:, position : position + 1, :] *= 0.72
    return output.round().clip(0, 255).astype(np.uint8)


def _save_webp(path: Path, image: np.ndarray, quality: int) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="WEBP", quality=quality, method=4)


def render_video(
    *,
    model,
    video_path: Path,
    generated_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    case_id = _safe_id(video_path, generated_root)
    raw = iio.imread(video_path).astype(np.uint8)
    frames = raw[: int(args.max_frames)]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{video_path} decoded to {frames.shape}")
    video_cthw = _preprocess_generated_frames(frames, height=int(args.height), width=int(args.width)).to(
        device=torch.device(args.device),
        dtype=torch.bfloat16,
    )
    attention = _extract_attention(model, video_cthw)
    labels = attention.argmax(dim=1).numpy()
    case_dir = output_dir / "cases" / case_id
    jobs = []
    for frame_id, frame in enumerate(frames):
        jobs.append((case_dir / "generated" / f"{frame_id:04d}.webp", frame))
        jobs.append(
            (
                case_dir / "all_slots_overlay" / f"{frame_id:04d}.webp",
                _all_slots_overlay(frame, labels[frame_id], alpha=float(args.alpha)),
            )
        )
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        list(executor.map(lambda job: _save_webp(job[0], job[1], int(args.quality)), jobs))
    meta = _load_json(video_path.with_suffix(".json"))
    return {
        "id": case_id,
        "step": video_path.parent.name,
        "stem": video_path.stem,
        "video": str(video_path),
        "json": str(video_path.with_suffix(".json")) if video_path.with_suffix(".json").is_file() else None,
        "input_json": meta.get("input_json"),
        "source_video": meta.get("source_video"),
        "frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "attention_shape": list(attention.shape),
        "generated_pattern": f"cases/{case_id}/generated/{{frame}}.webp",
        "overlay_pattern": f"cases/{case_id}/all_slots_overlay/{{frame}}.webp",
    }


def build_html(metadata: dict[str, object]) -> str:
    data = json.dumps(metadata, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scheme A generated xSSC all-slots overlay</title>
<style>
*{{box-sizing:border-box}}:root{{color-scheme:dark}}body{{margin:0;background:#101214;color:#f3f4f6;font:14px system-ui,sans-serif;letter-spacing:0}}header{{position:sticky;top:0;z-index:5;background:rgba(16,18,20,.98);border-bottom:1px solid #353a40}}.bar{{max-width:1500px;margin:auto;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}h1{{font-size:18px;margin:0 auto 0 0}}button,select,input{{font:inherit}}select,.icon{{height:34px;border:1px solid #4b525a;border-radius:5px;background:#202429;color:#f5f6f7}}select{{padding:0 28px 0 9px;max-width:min(760px,100%)}}.icon{{width:34px;cursor:pointer}}.icon:hover{{background:#2b3036}}#frameSlider{{min-width:220px;flex:0 1 420px;accent-color:#38bdf8}}#counter{{min-width:120px;color:#c4c9cf;font-variant-numeric:tabular-nums}}main{{max-width:1500px;margin:auto;padding:16px}}.meta{{display:grid;gap:4px;padding-bottom:14px;color:#aeb5bd;overflow-wrap:anywhere}}.compare{{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:14px}}figure{{margin:0;min-width:0}}img{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#050607;border:1px solid #34383d;border-radius:4px}}figcaption{{padding:7px 2px 0;color:#c4c9cf}}figcaption strong{{display:block;color:#f3f4f6;margin-bottom:2px}}.legend{{display:flex;gap:9px;flex-wrap:wrap;padding-top:18px;color:#aeb5bd}}.legend span{{display:inline-flex;align-items:center;gap:5px}}.swatch{{width:12px;height:12px;border-radius:2px}}@media(max-width:860px){{.compare{{grid-template-columns:1fr}}main{{padding:11px}}h1{{width:100%}}}}
</style>
</head>
<body>
<header><div class="bar"><h1>Scheme A generated xSSC all-slots overlay</h1><select id="caseSelect"></select><button id="previous" class="icon" title="Previous frame">&#8249;</button><button id="next" class="icon" title="Next frame">&#8250;</button><input id="frameSlider" type="range" min="0" max="48" step="1" value="0"><span id="counter"></span></div></header>
<main><div id="meta" class="meta"></div><section class="compare"><figure><img id="generated"><figcaption><strong>generated video</strong><span>raw output frame</span></figcaption></figure><figure><img id="overlay"><figcaption><strong>all-slots overlay</strong><span>frozen xSSC argmax slot assignment on generated frame</span></figcaption></figure></section><div id="legend" class="legend"></div></main>
<script>
const DATA={data};const select=document.getElementById('caseSelect');const slider=document.getElementById('frameSlider');const counter=document.getElementById('counter');const generated=document.getElementById('generated');const overlay=document.getElementById('overlay');const meta=document.getElementById('meta');let frame=0;
function item(){{return DATA.cases[Number(select.value)]}}function pattern(path,id){{return path.replace('{{frame}}',String(id).padStart(4,'0'))}}function update(){{const c=item();frame=Math.max(0,Math.min(frame,c.frames-1));slider.max=String(c.frames-1);slider.value=String(frame);counter.textContent=`frame ${{frame+1}} / ${{c.frames}}`;generated.src=pattern(c.generated_pattern,frame);overlay.src=pattern(c.overlay_pattern,frame);meta.innerHTML=`<div><strong>${{c.step}}</strong> | <code>${{c.stem}}</code></div><div>generated: <code>${{c.video}}</code></div><div>source: <code>${{c.source_video||''}}</code></div><div>attention: <code>${{JSON.stringify(c.attention_shape)}}</code></div>`}}function changeCase(){{frame=0;update()}}function step(delta){{frame+=delta;update()}}
DATA.cases.forEach((c,i)=>{{const option=document.createElement('option');option.value=String(i);option.textContent=`${{String(i+1).padStart(2,'0')}} | ${{c.step}} | ${{c.stem}}`;select.appendChild(option)}});document.getElementById('legend').innerHTML=DATA.palette.map((color,index)=>`<span><i class="swatch" style="background:rgb(${{color.join(',')}})"></i>slot ${{index}}</span>`).join('');select.addEventListener('change',changeCase);slider.addEventListener('input',()=>{{frame=Number(slider.value);update()}});document.getElementById('previous').addEventListener('click',()=>step(-1));document.getElementById('next').addEventListener('click',()=>step(1));document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1)}});update();
</script>
</body></html>"""


def main() -> None:
    args = parse_args()
    generated_root = args.generated_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = _discover_videos(args)
    if not videos:
        raise FileNotFoundError(f"no videos matched under {generated_root}")
    model = _build_minimal_model(torch.device(args.device))
    cases = []
    for index, video_path in enumerate(videos, start=1):
        rendered = render_video(
            model=model,
            video_path=video_path,
            generated_root=generated_root,
            output_dir=output_dir,
            args=args,
        )
        cases.append(rendered)
        print(f"[overlay] {index}/{len(videos)} {video_path}", flush=True)
    metadata = {
        "generated_root": str(generated_root),
        "output_dir": str(output_dir),
        "num_videos": len(cases),
        "palette": PALETTE.tolist(),
        "note": "Slot overlays are frozen xSSC argmax assignments computed on generated video frames.",
        "cases": cases,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(build_html(metadata), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "videos": len(cases), "index": str(output_dir / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
