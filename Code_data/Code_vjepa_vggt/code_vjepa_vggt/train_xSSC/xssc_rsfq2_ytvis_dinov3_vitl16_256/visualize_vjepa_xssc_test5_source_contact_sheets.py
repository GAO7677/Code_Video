#!/usr/bin/env python3
"""Render test_5 source videos as V-JEPA xSSC slot-overlay contact sheets."""

from argparse import ArgumentParser
import html
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
import torch

from infer_vjepa_xssc_video_slot_overlay import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONFIG,
    SLOT_COLORS,
    add_header,
    add_legend,
    checkpoint_load_summary,
    fit_width,
    infer_slot_labels,
    latest_checkpoint,
    load_model,
    make_contact_sheet,
    normalize_frames,
    preprocess_frames,
    render_overlay,
    set_seed,
    upsample_labels,
)


DEFAULT_TEST5 = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/"
    "vjepa_xssc_test5_source_slot_overlay_step10000"
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test5-file", type=Path, default=DEFAULT_TEST5)
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sampled-frames", type=int, default=32)
    parser.add_argument(
        "--resize-mode",
        choices=("center-crop", "padding"),
        default="padding",
    )
    parser.add_argument("--contact-columns", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_unique_cases(test5_file):
    cases = []
    seen_sources = set()
    for line_number, line in enumerate(test5_file.read_text().splitlines(), 1):
        if not line.strip():
            continue
        json_file = Path(line.strip()).resolve()
        payload = json.loads(json_file.read_text())
        source_video = Path(payload["source_video"]).resolve()
        source_key = str(source_video)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        cases.append(
            {
                "line_number": line_number,
                "case_id": json_file.stem,
                "json_file": str(json_file),
                "source_video": str(source_video),
                "caption": str(payload.get("input_caption", "")),
            }
        )
    if not cases:
        raise RuntimeError(f"no source videos found in {test5_file}")
    return cases


def decode_and_sample(path, max_sampled_frames, resize_mode):
    metadata = iio.immeta(path, plugin="pyav")
    processed = []
    source_shape = None
    for frame in iio.imiter(path, plugin="pyav"):
        frame = np.asarray(frame)[..., :3]
        if source_shape is None:
            source_shape = list(frame.shape)
        processed.append(preprocess_frames(frame[None], 256, resize_mode)[0])
    if len(processed) < 2:
        raise ValueError(f"video must contain at least two frames: {path}")
    frames = np.stack(processed).astype(np.uint8)
    if max_sampled_frames > 0 and len(frames) > max_sampled_frames:
        indices = np.rint(
            np.linspace(0, len(frames) - 1, max_sampled_frames)
        ).astype(np.int64)
        indices = np.unique(indices)
    else:
        indices = np.arange(len(frames), dtype=np.int64)
    return frames[indices], indices, len(frames), source_shape, metadata


def render_case(
    case,
    model,
    cfg,
    device,
    output_root,
    max_sampled_frames,
    resize_mode,
    columns,
    alpha,
    quality,
):
    source_video = Path(case["source_video"])
    frames, source_indices, source_count, source_shape, metadata = decode_and_sample(
        source_video, max_sampled_frames, resize_mode
    )
    source_fps = float(metadata.get("fps", 30.0))
    padded = len(frames) % 2
    model_frames = (
        np.concatenate([frames, frames[-1:]], axis=0) if padded else frames
    )
    video = normalize_frames(model_frames)
    labels, attention_shape = infer_slot_labels(
        model, video, device, getattr(torch, cfg.amp_dtype)
    )
    labels = upsample_labels(labels, 256, 256)
    labels_per_frame = np.repeat(labels, 2, axis=0)[: len(frames)]

    panels = []
    for sampled_index, (frame, label, source_index) in enumerate(
        zip(frames, labels_per_frame, source_indices)
    ):
        time_seconds = float(source_index) / source_fps
        source = add_header(
            frame,
            [
                "MODEL INPUT",
                f"source f{int(source_index):03d} · {time_seconds:.2f}s · letterbox 256²",
            ],
            color=(121, 192, 255),
        )
        overlay = add_header(
            render_overlay(frame, label, alpha),
            [
                "xSSC SLOT OVERLAY",
                f"sample {sampled_index:02d} · tubelet {sampled_index // 2:02d}",
            ],
            color=(255, 196, 107),
        )
        width = max(source.shape[1], overlay.shape[1])
        panel = np.concatenate(
            [fit_width(source, width), fit_width(overlay, width)], axis=1
        )
        panels.append(add_legend(panel))

    sheet, rows = make_contact_sheet(panels, columns)
    case_dir = output_root / "cases" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    sheet_file = case_dir / "contact_sheet.webp"
    sheet.save(sheet_file, format="WEBP", quality=quality, method=6)
    return {
        **case,
        "source_frame_count": source_count,
        "source_shape": source_shape,
        "source_fps": source_fps,
        "sampled_frame_count": len(frames),
        "sampled_source_indices": source_indices.tolist(),
        "padded_model_frame_count": len(model_frames),
        "attention_shape": list(attention_shape),
        "sheet": str(sheet_file.relative_to(output_root)),
        "sheet_columns": columns,
        "sheet_rows": rows,
        "sheet_size": list(sheet.size),
    }


def build_html(report):
    payload = json.dumps(report, separators=(",", ":"))
    legend = "".join(
        f'<span><i style="background:rgb({r},{g},{b})"></i>S{slot}</span>'
        for slot, (r, g, b) in enumerate(SLOT_COLORS.tolist())
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA xSSC · test_5 source slot overlay</title><style>
*{{box-sizing:border-box}}:root{{color-scheme:dark}}body{{margin:0;background:#0d1117;color:#e6edf3;font:14px system-ui,sans-serif}}header{{position:sticky;top:0;z-index:3;background:rgba(13,17,23,.97);border-bottom:1px solid #30363d}}.bar{{max-width:2200px;margin:auto;padding:12px 18px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}h1{{font-size:19px;margin:0 auto 0 0}}select,button{{height:36px;border:1px solid #4b535d;border-radius:6px;background:#20262d;color:#f2f5f7;font:inherit}}select{{min-width:min(760px,75vw);padding:0 10px}}button{{width:38px;cursor:pointer}}main{{max-width:2200px;margin:auto;padding:18px}}.meta{{display:grid;grid-template-columns:150px minmax(0,1fr);gap:7px 14px;padding:14px;border:1px solid #30363d;border-radius:8px;background:#161b22;margin-bottom:14px}}.key{{color:#8b949e}}code{{color:#79c0ff;overflow-wrap:anywhere}}.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0;color:#c9d1d9}}.legend span{{display:flex;align-items:center;gap:5px}}.legend i{{width:15px;height:15px;border-radius:3px}}.sheet{{display:block;width:100%;height:auto;background:#050607;border:1px solid #30363d;border-radius:8px;cursor:zoom-in}}.note{{color:#8b949e;line-height:1.5}}
</style></head><body><header><div class="bar"><h1>V-JEPA xSSC · test_5 source video 帧拼接</h1><button id="prev">‹</button><select id="case"></select><button id="next">›</button></div></header><main><section id="meta" class="meta"></section><div class="legend">{legend}</div><a id="sheetLink" target="_blank"><img id="sheet" class="sheet" alt="Input and slot-overlay frames stitched as a contact sheet"></a><p class="note">每格左侧是模型实际输入，右侧是对应的 slot overlay；顺序从左到右、从上到下。每个 source video 均匀抽取最多32帧覆盖完整时间范围。V-JEPA tubelet=2，同一个 tubelet 的预测重复叠加到组成它的两帧。颜色表示 slot index，不是语义类别。</p></main><script>
const DATA={payload};const sel=document.getElementById('case');const meta=document.getElementById('meta');const image=document.getElementById('sheet');const link=document.getElementById('sheetLink');DATA.cases.forEach((c,i)=>{{const o=document.createElement('option');o.value=String(i);o.textContent=`${{String(i+1).padStart(2,'0')}} | ${{c.case_id}} | ${{c.source_frame_count}}f`;sel.appendChild(o)}});function show(){{const c=DATA.cases[Number(sel.value)];image.src=c.sheet;link.href=c.sheet;meta.innerHTML=`<span class="key">Case</span><strong>${{c.case_id}}</strong><span class="key">Source video</span><code>${{c.source_video}}</code><span class="key">Sampling</span><span>${{c.sampled_frame_count}} / ${{c.source_frame_count}} frames · indices [${{c.sampled_source_indices.join(', ')}}]</span><span class="key">Model</span><span>${{DATA.temporal_mode}} · tubelet=${{DATA.tubelet_size}} · 7 slots</span><span class="key">Checkpoint</span><code>${{DATA.checkpoint}}</code><span class="key">Attention shape</span><code>${{JSON.stringify(c.attention_shape)}}</code>`}}sel.addEventListener('change',show);document.getElementById('prev').onclick=()=>{{sel.selectedIndex=(sel.selectedIndex-1+DATA.cases.length)%DATA.cases.length;show()}};document.getElementById('next').onclick=()=>{{sel.selectedIndex=(sel.selectedIndex+1)%DATA.cases.length;show()}};show();
</script></body></html>"""


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.max_sampled_frames != 0 and args.max_sampled_frames < 2:
        raise ValueError("max-sampled-frames must be 0 or at least 2")
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    checkpoint = (
        latest_checkpoint(DEFAULT_CHECKPOINT_DIR)
        if args.checkpoint is None
        else args.checkpoint.resolve()
    )
    test5_file = args.test5_file.resolve()
    config_file = args.cfg_file.resolve()
    for path in (checkpoint, test5_file, config_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg, model, load_report = load_model(config_file, checkpoint, device)

    cases = load_unique_cases(test5_file)
    rendered = []
    for index, case in enumerate(cases, 1):
        rendered.append(
            render_case(
                case,
                model,
                cfg,
                device,
                output_dir,
                args.max_sampled_frames,
                args.resize_mode,
                args.contact_columns,
                args.alpha,
                args.quality,
            )
        )
        print(
            f"[render] {index}/{len(cases)} {case['case_id']} "
            f"frames={rendered[-1]['sampled_frame_count']}",
            flush=True,
        )

    report = {
        "title": "V-JEPA xSSC test_5 source-video slot overlays",
        "test5_file": str(test5_file),
        "unique_source_videos": len(rendered),
        "config_file": str(config_file),
        "checkpoint": str(checkpoint),
        "checkpoint_load": checkpoint_load_summary(load_report),
        "temporal_mode": cfg.temporal_mode,
        "tubelet_size": 2,
        "tubelet_label_policy": cfg.tubelet_label_policy,
        "resize_mode": args.resize_mode,
        "max_sampled_frames": args.max_sampled_frames,
        "slot_colors_rgb": SLOT_COLORS.tolist(),
        "cases": rendered,
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "index.html").write_text(build_html(report))
    (output_dir / "README.md").write_text(
        "# V-JEPA xSSC test_5 source-video slot overlays\n\n"
        f"- Checkpoint: `{checkpoint}`\n"
        f"- Config: `{config_file}`\n"
        f"- Unique source videos: {len(rendered)}\n"
        f"- Resize: `{args.resize_mode}` to 256x256\n"
        f"- Uniform samples per video: at most {args.max_sampled_frames}\n"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index_html": str(output_dir / "index.html"),
                "cases": len(rendered),
                "checkpoint": str(checkpoint),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
