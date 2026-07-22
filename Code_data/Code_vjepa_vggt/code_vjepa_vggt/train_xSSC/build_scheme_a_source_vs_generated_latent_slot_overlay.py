#!/usr/bin/env python3
"""Compare source-video and generated-video xSSC slot overlays on Wan latent time."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from code_vjepa_vggt.train_xSSC import train_xssc_allframe_oracle_slots as train
from code_vjepa_vggt.train_xSSC.batch_infer_xssc_allframe_oracle_slots import (
    _resolve_source_video,
    preprocess_video_rgb_uint8,
)
from code_vjepa_vggt.utils.video_io import read_video_prefix


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
    "scheme_a_source_vs_generated_latent_slot_overlay"
)
DEFAULT_CASE_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/"
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", type=Path, default=DEFAULT_CASE_JSON)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--step", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--max-frames", type=int, default=49)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.57)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quality", type=int, default=88)
    return parser.parse_args()


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)


def _build_model(device: torch.device):
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


def _video_uint8_to_cthw(frames: np.ndarray, *, height: int, width: int) -> torch.Tensor:
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
    with torch.autocast(
        device_type=flat_video.device.type,
        dtype=torch.bfloat16,
        enabled=flat_video.device.type == "cuda",
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
            query = model.xssc.initializ(batch) if frame_id == 0 else model.xssc.transit(slots, encoded[:, : frame_id + 1])
            current_slots, current_attention = model.xssc.aggregat(
                encoded[:, frame_id],
                query,
                num_iter=None if frame_id == 0 else 1,
            )
            slots = current_slots[:, None] if slots is None else torch.cat((slots, current_slots[:, None]), dim=1)
            attentions.append(current_attention.view(batch, model.xssc_num_slots, feature_h, feature_w))
    return torch.stack(attentions, dim=1)[0].float().cpu()


def _latent_chunks(time_steps: int, stride: int) -> list[tuple[int, int]]:
    chunks = [(0, 1)]
    for start in range(1, time_steps, stride):
        chunks.append((start, min(start + stride, time_steps)))
    return chunks


def _mean_attention_to_labels(attention_tshw: torch.Tensor, chunks: list[tuple[int, int]]) -> list[np.ndarray]:
    labels = []
    for start, end in chunks:
        mean_attention = attention_tshw[start:end].mean(dim=0)
        labels.append(mean_attention.argmax(dim=0).numpy().astype(np.uint8))
    return labels


def _representative_indices(chunks: list[tuple[int, int]], frame_count: int) -> list[int]:
    return [min(frame_count - 1, (start + end - 1) // 2) for start, end in chunks]


def _overlay_all_slots(frame: np.ndarray, labels_hw: np.ndarray, *, alpha: float) -> np.ndarray:
    height, width = frame.shape[:2]
    crop = min(height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    labels_full = np.asarray(
        Image.fromarray(labels_hw).resize((crop, crop), Image.Resampling.NEAREST),
        dtype=np.int64,
    )
    colors = PALETTE[labels_full % len(PALETTE)]
    out = frame.astype(np.float32).copy()
    square = out[top : top + crop, left : left + crop]
    square[:] = square * (1.0 - alpha) + colors.astype(np.float32) * alpha
    for position in range(16, crop, max(16, crop // 16)):
        square[position : position + 1, :, :] *= 0.72
        square[:, position : position + 1, :] *= 0.72
    return out.round().clip(0, 255).astype(np.uint8)


def _save_webp(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="WEBP", quality=quality, method=4)


def _render_latent_strip(
    *,
    model,
    frames: np.ndarray,
    row_name: str,
    case_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    frames = frames[: int(args.max_frames)].astype(np.uint8)
    video_cthw = _video_uint8_to_cthw(frames, height=int(args.height), width=int(args.width)).to(
        device=torch.device(args.device),
        dtype=torch.bfloat16,
    )
    attention = _extract_attention(model, video_cthw)
    chunks = _latent_chunks(int(frames.shape[0]), int(args.stride))
    labels = _mean_attention_to_labels(attention, chunks)
    reps = _representative_indices(chunks, int(frames.shape[0]))
    jobs = []
    frame_items = []
    for latent_id, (label, frame_id, chunk) in enumerate(zip(labels, reps, chunks)):
        original = frames[frame_id]
        overlay = _overlay_all_slots(original, label, alpha=float(args.alpha))
        original_path = case_dir / row_name / "original" / f"latent{latent_id:02d}.webp"
        overlay_path = case_dir / row_name / "overlay" / f"latent{latent_id:02d}.webp"
        jobs.append((original_path, original))
        jobs.append((overlay_path, overlay))
        frame_items.append(
            {
                "latent_id": latent_id,
                "chunk": [int(chunk[0]), int(chunk[1])],
                "representative_frame": int(frame_id),
                "original": original_path.relative_to(case_dir.parent.parent).as_posix(),
                "overlay": overlay_path.relative_to(case_dir.parent.parent).as_posix(),
            }
        )
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        list(executor.map(lambda item: _save_webp(item[0], item[1], int(args.quality)), jobs))
    return {
        "row": row_name,
        "frames": int(frames.shape[0]),
        "attention_shape": list(attention.shape),
        "latent_count": len(chunks),
        "items": frame_items,
    }


def _find_generated_videos(generated_root: Path, case_stem: str, steps: list[str]) -> dict[str, Path]:
    if not steps:
        steps = [path.name for path in sorted(generated_root.glob("step-*"))]
    found = {}
    for step in steps:
        video = generated_root / step / step / f"{case_stem}.mp4"
        if video.is_file():
            found[step] = video
    return found


def build_html(metadata: dict[str, object]) -> str:
    data = json.dumps(metadata, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scheme A source vs generated latent slot overlay</title>
<style>
*{{box-sizing:border-box}}:root{{color-scheme:dark}}body{{margin:0;background:#101214;color:#f3f4f6;font:13px system-ui,sans-serif;letter-spacing:0}}header{{position:sticky;top:0;z-index:4;background:rgba(16,18,20,.98);border-bottom:1px solid #353a40}}.bar{{padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}h1{{font-size:18px;margin:0 auto 0 0}}main{{padding:16px;display:grid;gap:18px}}.meta{{color:#aeb5bd;display:grid;gap:4px;overflow-wrap:anywhere}}.wrap{{overflow:auto;border:1px solid #34383d;border-radius:6px;background:#14181d}}table{{border-collapse:collapse;min-width:7940px;width:max-content;table-layout:fixed}}th,td{{border-right:1px solid #34383d;border-bottom:1px solid #34383d;vertical-align:top}}th{{background:#20242a;padding:9px 10px;text-align:left}}thead th{{position:sticky;top:0;z-index:3}}th.step{{width:1940px}}th.row{{position:sticky;left:0;z-index:3;width:180px;background:#1b2026}}tbody th.row{{z-index:2}}.cell{{padding:10px;display:grid;gap:8px}}.strip{{display:grid;grid-template-columns:repeat(13,142px);gap:7px;align-items:start}}figure{{margin:0;min-width:0}}img{{display:block;width:142px;height:80px;object-fit:contain;background:#050607;border:1px solid #34383d;border-radius:3px}}figcaption{{font-size:10px;color:#bac2cc;padding-top:3px;text-align:center;font-variant-numeric:tabular-nums;line-height:1.25;white-space:normal}}.legend{{display:flex;gap:9px;flex-wrap:wrap;color:#aeb5bd}}.legend span{{display:inline-flex;align-items:center;gap:5px}}.swatch{{width:12px;height:12px;border-radius:2px}}code{{color:#dbeafe}}
</style></head><body>
<header><div class="bar"><h1>Scheme A source vs generated latent slot overlay</h1><span>rows: source/generated · columns: checkpoint step · inner strip: latent time</span></div></header>
<main><div id="meta" class="meta"></div><div class="wrap"><table id="matrix"></table></div><div id="legend" class="legend"></div></main>
<script>
const DATA={data};const matrix=document.getElementById('matrix');document.getElementById('meta').innerHTML=`<div>case: <code>${{DATA.case_json}}</code></div><div>source: <code>${{DATA.source_video}}</code></div><div>${{DATA.note}}</div>`;
function strip(items,key){{return `<div class="strip">${{items.map(it=>`<figure><img src="${{it[key]}}"><figcaption>z${{it.latent_id}} | f${{it.representative_frame}} | [${{it.chunk[0]}},${{it.chunk[1]}})</figcaption></figure>`).join('')}}</div>`}}
let html='<thead><tr><th class="row">video</th>'+DATA.steps.map(s=>`<th class="step">${{s}}</th>`).join('')+'</tr></thead><tbody>';
html+='<tr><th class="row">source video<br><small>actual xSSC condition</small></th>'+DATA.steps.map(s=>`<td><div class="cell">${{strip(DATA.source_by_step[s].items,'overlay')}}</div></td>`).join('')+'</tr>';
html+='<tr><th class="row">generated video<br><small>xSSC rerun on output</small></th>'+DATA.steps.map(s=>`<td><div class="cell">${{DATA.generated_by_step[s]?strip(DATA.generated_by_step[s].items,'overlay'):'missing'}}</div></td>`).join('')+'</tr>';
matrix.innerHTML=html+'</tbody>';document.getElementById('legend').innerHTML=DATA.palette.map((color,index)=>`<span><i class="swatch" style="background:rgb(${{color.join(',')}})"></i>slot ${{index}}</span>`).join('');
</script></body></html>"""


def main() -> None:
    args = parse_args()
    case_json = args.case_json.expanduser().resolve()
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    source_video = Path(_resolve_source_video(payload, case_json)).expanduser().resolve()
    case_stem = case_json.stem
    generated_root = args.generated_root.expanduser().resolve()
    steps = args.step or ["step-000500", "step-001000", "step-001500", "step-002000"]
    generated = _find_generated_videos(generated_root, case_stem, steps)
    if not generated:
        raise FileNotFoundError(f"no generated videos for {case_stem} under {generated_root}")

    output_dir = args.output_dir.expanduser().resolve()
    case_dir = output_dir / "cases" / _safe_stem(case_json)
    case_dir.mkdir(parents=True, exist_ok=True)
    model = _build_model(torch.device(args.device))

    source_frames_raw, frame_indices = read_video_prefix(source_video, int(args.max_frames))
    source_frames = preprocess_video_rgb_uint8(
        source_frames_raw,
        (int(args.height), int(args.width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(args.height), int(args.width)),
    )
    source_frames_uint8 = ((source_frames.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)

    source_once = _render_latent_strip(
        model=model,
        frames=source_frames_uint8,
        row_name="source",
        case_dir=case_dir,
        args=args,
    )
    source_by_step = {step: source_once for step in steps if step in generated}
    generated_by_step = {}
    for step, video_path in generated.items():
        frames = iio.imread(video_path).astype(np.uint8)
        generated_by_step[step] = _render_latent_strip(
            model=model,
            frames=frames,
            row_name=f"generated_{step}",
            case_dir=case_dir,
            args=args,
        ) | {"video": str(video_path)}
        print(f"[overlay] {step} {video_path}", flush=True)

    metadata = {
        "case_json": str(case_json),
        "source_video": str(source_video),
        "source_frame_indices": [int(v) for v in frame_indices.tolist()],
        "generated_root": str(generated_root),
        "steps": [step for step in steps if step in generated],
        "palette": PALETTE.tolist(),
        "note": "Each thumbnail shows xSSC all-slot argmax after mean pooling attention over the corresponding Wan latent chunk.",
        "source_by_step": source_by_step,
        "generated_by_step": generated_by_step,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(build_html(metadata), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "index": str(output_dir / "index.html"), "steps": metadata["steps"]}, indent=2))


if __name__ == "__main__":
    main()
