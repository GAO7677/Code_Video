#!/usr/bin/env python3
"""Visualize the latest complete V-JEPA xSSC checkpoint on val/test_5 cases."""

from argparse import ArgumentParser
import html
import json
import os
from pathlib import Path
import random
import re
import sys

# Must be set before the first CUDA/cuBLAS context is created. The script later
# enables deterministic algorithms, matching the training launchers.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, "/home/gaoya/Code_Video/vjepa2-main")

from infer_vjepa_xssc_video_slot_overlay import (  # noqa: E402
    FONT_SMALL,
    SLOT_COLORS,
    checkpoint_load_summary,
    load_model,
    make_contact_sheet,
    normalize_frames,
    render_overlay,
    set_seed,
    slot_boundaries,
    upsample_labels,
)
from object_centric_bench.datum.transform import (  # noqa: E402
    choose_aspect_ratio_bucket,
)
from object_centric_bench.util import Config, build_from_config  # noqa: E402


DEFAULT_MODEL_CONFIG = ROOT / (
    "upstream/config-randsfq/"
    "rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-"
    "resume14000-clip2-bs64.py"
)
DEFAULT_MODEL_CHECKPOINT = Path(
    "/data/gaoya/agent-data/checkpoints/"
    "xssc_vjepa2_1_video_noncausal_ytvis_hq_10f_ar_bs64_"
    "steps20000_clip2_resume14000/"
    "rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-"
    "resume14000-clip2-bs64/42/step-016000.pth"
)
DEFAULT_MOVIC_DATA_CONFIG = ROOT / (
    "upstream/config-randsfq/"
    "rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-10f-slot512-"
    "transfer16000-clip2.py"
)
DEFAULT_TEST5 = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/"
    "vjepa_xssc_latest_complete_step16000_val5_test5"
)
IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument("--movic-data-config", type=Path, default=DEFAULT_MOVIC_DATA_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--test5-file", type=Path, default=DEFAULT_TEST5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ytvis-cases", type=int, default=5)
    parser.add_argument("--movic-cases", type=int, default=5)
    parser.add_argument(
        "--test5-cases",
        type=int,
        default=0,
        help="Unique source videos from test_5; 0 means all unique sources.",
    )
    parser.add_argument(
        "--external-window-frames",
        type=int,
        default=10,
        help="Consecutive frames per test_5 forward; every decoded frame is retained.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--sheet-columns", type=int, default=4)
    return parser.parse_args()


def decode_dataset_rgb(video):
    rgb = video.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return (
        rgb.clamp(0, 255)
        .round()
        .to(torch.uint8)[0]
        .permute(0, 2, 3, 1)
        .numpy()
    )


def segment_to_labels(segment):
    if segment is None:
        return None
    segment = segment[0].detach().cpu()
    if segment.ndim != 4:
        raise ValueError(f"expected segment [T,H,W,N], got {tuple(segment.shape)}")
    present = segment.bool().any(dim=-1)
    # Both YTVIS-HQ and MOVi-C store a one-hot background in channel 0.
    # Preserve it as label 0; channels 1..N are foreground instances.
    labels = segment.float().argmax(dim=-1).to(torch.int64)
    labels[~present] = 0
    return labels.numpy().astype(np.uint8)


def safe_id(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-_")[:120]


def source_key(dataset, index):
    if not hasattr(dataset, "keys"):
        return f"index={index}"
    value = dataset.keys[index]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def build_val_source(config_file, data_dir, source, count, seed):
    cfg = Config.fromfile(config_file)
    cfg.dataset_v.base_dir = data_dir.resolve()
    dataset = build_from_config(cfg.dataset_v)
    collate = build_from_config(cfg.collate_fn_v)
    indices = sorted(random.Random(seed).sample(range(len(dataset)), count))
    return {
        "source": source,
        "config": str(config_file.resolve()),
        "dataset": dataset,
        "collate": collate,
        "indices": indices,
    }


def load_unique_test5(test5_file, limit):
    cases = []
    seen = set()
    lines = [line.strip() for line in test5_file.read_text().splitlines() if line.strip()]
    for json_position, line in enumerate(lines, start=1):
        json_file = Path(line).resolve()
        payload = json.loads(json_file.read_text())
        video = str(Path(payload["source_video"]).resolve())
        if video in seen:
            continue
        seen.add(video)
        stem = safe_id(f"{Path(video).parent.name}_{Path(video).stem}")
        cases.append(
            {
                "source": "test5",
                "case_id": f"test5_{len(cases) + 1:03d}_{stem}",
                "source_key": video,
                "json_file": str(json_file),
                "json_position": json_position,
            }
        )
        if limit > 0 and len(cases) >= limit:
            break
    return cases


def resize_frame(frame, target):
    height, width = target
    return np.asarray(
        Image.fromarray(np.asarray(frame)[..., :3]).resize(
            (width, height), Image.Resampling.BILINEAR
        )
    )


def decode_external_video(path, buckets):
    frames = []
    raw_shape = None
    target = None
    for frame in iio.imiter(path, plugin="pyav"):
        frame = np.asarray(frame)[..., :3]
        if raw_shape is None:
            raw_shape = list(frame.shape)
            target = choose_aspect_ratio_bucket(frame.shape[0], frame.shape[1], buckets)
        frames.append(resize_frame(frame, target))
    if not frames:
        raise RuntimeError(f"no decoded frames: {path}")
    metadata = iio.immeta(path, plugin="pyav")
    return np.stack(frames).astype(np.uint8), raw_shape, list(target), metadata


def normalize_slot_vectors(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(min=1e-12)


def align_chunk_slots(previous, current, labels, slotz):
    similarity = normalize_slot_vectors(previous) @ normalize_slot_vectors(current).T
    rows, columns = linear_sum_assignment(-similarity)
    local_to_global = np.arange(len(current), dtype=np.uint8)
    local_to_global[columns] = rows
    aligned_labels = local_to_global[labels]
    aligned_slotz = np.empty_like(slotz)
    for local_index, global_index in enumerate(local_to_global):
        aligned_slotz[:, global_index] = slotz[:, local_index]
    return aligned_labels, aligned_slotz, local_to_global.tolist()


@torch.inference_mode()
def infer_window(model, frames, device, amp_dtype):
    original_count = len(frames)
    if original_count < 2:
        frames = np.concatenate([frames, frames[-1:]], axis=0)
    if len(frames) % 2:
        frames = np.concatenate([frames, frames[-1:]], axis=0)
    video = normalize_frames(frames)
    with torch.autocast("cuda", dtype=amp_dtype):
        output = model(batch={"video": video.to(device)})
    attention = output["attentd"][0].detach().float().cpu()
    slotz = output["slotz"][0].detach().float().cpu().numpy()
    labels = attention.argmax(dim=1).to(torch.uint8).numpy()
    expected_tubelets = len(frames) // 2
    if len(labels) != expected_tubelets:
        raise RuntimeError(
            f"decoder returned {len(labels)} tubelets for {len(frames)} frames"
        )
    return labels, slotz, list(attention.shape), original_count


def infer_sequence(model, frames, device, amp_dtype, window_frames):
    if window_frames <= 0 or len(frames) <= window_frames:
        labels, slotz, shape, _ = infer_window(model, frames, device, amp_dtype)
        labels_frame = np.repeat(labels, 2, axis=0)[: len(frames)]
        return labels, labels_frame, [shape], []
    if window_frames < 2:
        raise ValueError("external window must contain at least two frames")
    if window_frames % 2:
        raise ValueError("external window must contain an even number of frames")

    tubelet_labels = []
    frame_labels = []
    shapes = []
    alignments = []
    previous_slotz = None
    for start in range(0, len(frames), window_frames):
        chunk = frames[start : start + window_frames]
        labels, slotz, shape, original_count = infer_window(
            model, chunk, device, amp_dtype
        )
        if previous_slotz is not None:
            labels, slotz, mapping = align_chunk_slots(
                previous_slotz, slotz[0], labels, slotz
            )
            alignments.append({"start_frame": start, "local_to_global": mapping})
        previous_slotz = slotz[-1]
        tubelet_labels.append(labels)
        frame_labels.append(np.repeat(labels, 2, axis=0)[:original_count])
        shapes.append(shape)
    return (
        np.concatenate(tubelet_labels, axis=0),
        np.concatenate(frame_labels, axis=0),
        shapes,
        alignments,
    )


def comb2(values):
    values = np.asarray(values, dtype=np.float64)
    return values * (values - 1) / 2


def adjusted_rand_index(gt, pred):
    gt_values, gt_inverse = np.unique(gt, return_inverse=True)
    pred_values, pred_inverse = np.unique(pred, return_inverse=True)
    table = np.zeros((len(gt_values), len(pred_values)), dtype=np.int64)
    np.add.at(table, (gt_inverse, pred_inverse), 1)
    n = table.sum()
    if n < 2:
        return 1.0
    sum_table = comb2(table).sum()
    sum_rows = comb2(table.sum(axis=1)).sum()
    sum_columns = comb2(table.sum(axis=0)).sum()
    pairs = comb2(n)
    expected = sum_rows * sum_columns / pairs
    maximum = (sum_rows + sum_columns) / 2
    denominator = maximum - expected
    return 1.0 if abs(denominator) < 1e-12 else float((sum_table - expected) / denominator)


def gt_metrics(predicted, gt):
    tubelets = min(len(predicted), len(gt))
    ari_values = []
    best_overlap_values = []
    for pred_frame, gt_frame in zip(predicted[:tubelets], gt[:tubelets]):
        pred_full = upsample_labels(pred_frame[None], gt_frame.shape[0], gt_frame.shape[1])[0]
        foreground = gt_frame > 0
        if foreground.sum() >= 2:
            ari_values.append(adjusted_rand_index(gt_frame[foreground], pred_full[foreground]))
        object_ious = []
        for object_id in np.unique(gt_frame[foreground]):
            target = gt_frame == object_id
            best = 0.0
            for slot_id in np.unique(pred_full):
                candidate = pred_full == slot_id
                union = np.logical_or(target, candidate).sum()
                if union:
                    best = max(best, float(np.logical_and(target, candidate).sum() / union))
            object_ious.append(best)
        if object_ious:
            best_overlap_values.append(float(np.mean(object_ious)))
    return {
        "ari_fg_diagnostic": float(np.mean(ari_values)) if ari_values else None,
        "mean_best_overlap_diagnostic": (
            float(np.mean(best_overlap_values)) if best_overlap_values else None
        ),
        "evaluated_tubelets": tubelets,
    }


def render_gt_overlay(frame, labels, alpha):
    result = frame.copy().astype(np.float32)
    foreground = labels > 0
    colors = SLOT_COLORS[(labels.astype(np.int64) - 1) % len(SLOT_COLORS)]
    result[foreground] = (
        result[foreground] * (1 - alpha) + colors[foreground] * alpha
    )
    result = result.round().clip(0, 255).astype(np.uint8)
    boundary = slot_boundaries(labels)
    result[boundary & foreground] = 255
    return result


def save_webp(path, array, quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="WEBP", quality=quality, method=5)


def label_panel(array, label, width=220):
    image = Image.fromarray(array)
    height = max(1, round(image.height * width / image.width))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 25), (14, 19, 25))
    canvas.paste(image, (0, 25))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), label, fill=(220, 229, 238), font=FONT_SMALL)
    return np.asarray(canvas)


def render_assets(case, frames, pred_frame, gt_frame, output_dir, args):
    case_dir = output_dir / "cases" / case["case_id"]
    contact_tiles = []
    has_gt = gt_frame is not None
    for frame_index, (frame, prediction) in enumerate(zip(frames, pred_frame)):
        original_file = case_dir / "original" / f"{frame_index:04d}.webp"
        prediction_file = case_dir / "prediction" / f"{frame_index:04d}.webp"
        prediction_full = upsample_labels(
            prediction[None], frame.shape[0], frame.shape[1]
        )[0]
        predicted_overlay = render_overlay(frame, prediction_full, args.alpha)
        save_webp(original_file, frame, args.quality)
        save_webp(prediction_file, predicted_overlay, args.quality)
        panels = [
            label_panel(frame, f"f{frame_index:03d} · input"),
            label_panel(predicted_overlay, f"f{frame_index:03d} · predicted slots"),
        ]
        if has_gt:
            gt_overlay = render_gt_overlay(frame, gt_frame[frame_index], args.alpha)
            save_webp(case_dir / "gt" / f"{frame_index:04d}.webp", gt_overlay, args.quality)
            panels.append(label_panel(gt_overlay, f"f{frame_index:03d} · GT instances"))
        contact_tiles.append(np.concatenate(panels, axis=1))
    sheet, rows = make_contact_sheet(contact_tiles, args.sheet_columns, gap=8)
    sheet_file = case_dir / "contact_sheet.webp"
    sheet.save(sheet_file, format="WEBP", quality=args.quality, method=6)
    return {
        "original_pattern": f"cases/{case['case_id']}/original/{{frame}}.webp",
        "prediction_pattern": f"cases/{case['case_id']}/prediction/{{frame}}.webp",
        "gt_pattern": f"cases/{case['case_id']}/gt/{{frame}}.webp" if has_gt else None,
        "contact_sheet": f"cases/{case['case_id']}/contact_sheet.webp",
        "contact_sheet_rows": rows,
        "contact_sheet_size": list(sheet.size),
    }


def build_html(report):
    payload = json.dumps(report, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(report["title"])
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
*{{box-sizing:border-box}}:root{{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--ink:#e6edf3;--muted:#8b949e;--cyan:#58c7da;--orange:#e9a23b}}body{{margin:0;background:linear-gradient(180deg,#111923 0,#0d1117 320px);color:var(--ink);font:14px system-ui,sans-serif}}header{{position:sticky;top:0;z-index:5;background:rgba(13,17,23,.97);border-bottom:1px solid var(--line)}}.bar{{max-width:2200px;margin:auto;padding:11px 18px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}}h1{{font-size:18px;margin:0 auto 0 0}}button,select,input{{height:36px;border:1px solid #46515c;border-radius:6px;background:#1d252e;color:var(--ink);font:inherit}}button{{padding:0 12px;cursor:pointer}}select{{padding:0 10px;max-width:min(780px,72vw)}}#slider{{min-width:220px;flex:0 1 380px;accent-color:var(--cyan)}}main{{max-width:2200px;margin:auto;padding:18px}}.notice{{padding:12px 14px;margin-bottom:14px;border-left:4px solid var(--orange);background:#201b13;color:#d7c4a6;line-height:1.5}}.summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}}.stat,.card{{border:1px solid var(--line);background:rgba(22,27,34,.94);border-radius:8px}}.stat{{padding:12px}}.stat b{{display:block;margin-top:4px;color:#fff}}.case-meta{{display:flex;gap:14px;overflow:auto;white-space:nowrap;color:var(--muted);padding:0 2px 12px}}.case-meta strong{{color:var(--ink)}}.timeline{{display:grid;grid-template-columns:repeat(var(--frames),minmax(3px,1fr));gap:2px;height:9px;margin:0 0 14px}}.timeline i{{background:#293440;border-radius:1px}}.timeline i:nth-child(2n){{background:#3c5965}}.timeline i.active{{background:var(--orange)}}.sheet{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#050607}}.inspector{{display:none;grid-template-columns:repeat(var(--panels),minmax(260px,1fr));gap:12px}}.inspector figure{{margin:0}}.inspector img{{display:block;width:100%;height:min(62vh,720px);object-fit:contain;background:#050607;border:1px solid var(--line);border-radius:8px}}figcaption{{padding:7px 2px;color:#aeb8c2}}.foot{{color:var(--muted);line-height:1.55;margin-top:14px}}code{{color:#79c0ff;overflow-wrap:anywhere}}@media(max-width:900px){{.summary{{grid-template-columns:repeat(2,1fr)}}.inspector{{grid-template-columns:1fr}}}}@media(max-width:560px){{.summary{{grid-template-columns:1fr}}select{{max-width:100%}}}}
</style></head><body><header><div class="bar"><h1>{title}</h1><button onclick="location.href='../'">项目总览</button><select id="source"><option value="">全部来源</option><option value="ytvis_hq_val">YTVIS val</option><option value="movi_c_val">MOVi-C val</option><option value="test5">test_5</option></select><button id="prev">‹</button><select id="case"></select><button id="next">›</button><button id="view">逐帧查看</button><input id="slider" type="range" min="0" value="0"><span id="counter"></span></div></header><main><div class="notice">最新完整 checkpoint 是 YTVIS clip2 step-16000；当前 MOVi-C 训练尚未产生新的落盘 checkpoint。本页不会把进程内未保存更新标为最新权重。</div><section class="summary"><div class="stat">Checkpoint<b>step-16000</b></div><div class="stat">验证抽样<b>YTVIS 5 + MOVi-C 5</b></div><div class="stat">test_5<b>20 个唯一 source</b></div><div class="stat">长视频策略<b>连续 10 帧窗口 · 不丢帧</b></div></section><div id="meta" class="case-meta"></div><div id="timeline" class="timeline"></div><a id="sheetLink" target="_blank"><img id="sheet" class="sheet"></a><section id="inspector" class="inspector"></section><p class="foot">页面只展示静态视频帧图像，不使用视频播放器。val 的 GT 是训练数据管线中与 V-JEPA tubelet 对齐的实例 mask；ARI-FG 和 mean-best-overlap 是这 5 例的诊断值，不冒充完整验证集官方指标。test_5 没有 GT。跨窗口颜色用相邻窗口 slot embedding 的 Hungarian 匹配对齐，仅用于视觉连续性。</p></main><script>
const D={payload},source=document.getElementById('source'),sel=document.getElementById('case'),slider=document.getElementById('slider'),counter=document.getElementById('counter'),meta=document.getElementById('meta'),timeline=document.getElementById('timeline'),sheet=document.getElementById('sheet'),sheetLink=document.getElementById('sheetLink'),inspector=document.getElementById('inspector');let frame=0,mode='sheet',visible=[];function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}function pat(p,i){{return p.replace('{{frame}}',String(i).padStart(4,'0'))}}function current(){{return D.cases[Number(sel.value)]}}function fmt(v){{return v===null||v===undefined?'—':Number(v).toFixed(4)}}function fillCases(){{const prior=current()?.case_id;visible=D.cases.map((c,i)=>[c,i]).filter(([c])=>!source.value||c.source===source.value);sel.innerHTML='';visible.forEach(([c,i])=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{String(i+1).padStart(2,'0')}} | ${{c.source}} | ${{c.case_id}}`;if(c.case_id===prior)o.selected=true;sel.appendChild(o)}});render()}}function updateFrame(){{const c=current();frame=Math.max(0,Math.min(frame,c.frames-1));slider.value=frame;counter.textContent=`${{frame+1}} / ${{c.frames}}`;timeline.querySelectorAll('i').forEach((x,i)=>x.classList.toggle('active',i===frame));inspector.querySelectorAll('img').forEach(x=>x.src=pat(x.dataset.pattern,frame))}}function render(){{const c=current();if(!c)return;frame=0;slider.max=c.frames-1;meta.innerHTML=`<strong>${{esc(c.source)}}</strong><span>${{esc(c.source_key)}}</span><span>${{c.frames}} frames</span><span>${{c.tubelets}} tubelets</span><span>ARI-FG ${{fmt(c.metrics.ari_fg_diagnostic)}}</span><span>mBO ${{fmt(c.metrics.mean_best_overlap_diagnostic)}}</span>`;timeline.style.setProperty('--frames',c.frames);timeline.innerHTML=Array.from({{length:c.frames}},()=>'<i></i>').join('');sheet.src=c.assets.contact_sheet;sheetLink.href=c.assets.contact_sheet;const panels=[['模型输入',c.assets.original_pattern],['Predicted slots',c.assets.prediction_pattern]];if(c.assets.gt_pattern)panels.push(['GT instances',c.assets.gt_pattern]);inspector.style.setProperty('--panels',panels.length);inspector.innerHTML=panels.map(([label,p])=>`<figure><img data-pattern="${{p}}"><figcaption>${{label}}</figcaption></figure>`).join('');updateFrame()}}function setMode(next){{mode=next;sheetLink.style.display=mode==='sheet'?'block':'none';inspector.style.display=mode==='frame'?'grid':'none';slider.style.visibility=mode==='frame'?'visible':'hidden';counter.style.visibility=mode==='frame'?'visible':'hidden';document.getElementById('view').textContent=mode==='sheet'?'逐帧查看':'查看拼接图';updateFrame()}}source.onchange=fillCases;sel.onchange=render;slider.oninput=()=>{{frame=Number(slider.value);updateFrame()}};document.getElementById('prev').onclick=()=>{{sel.selectedIndex=(sel.selectedIndex-1+sel.options.length)%sel.options.length;render()}};document.getElementById('next').onclick=()=>{{sel.selectedIndex=(sel.selectedIndex+1)%sel.options.length;render()}};document.getElementById('view').onclick=()=>setMode(mode==='sheet'?'frame':'sheet');document.addEventListener('keydown',e=>{{if(mode==='frame'&&e.key==='ArrowLeft'){{frame--;updateFrame()}}if(mode==='frame'&&e.key==='ArrowRight'){{frame++;updateFrame()}}}});fillCases();setMode('sheet');
</script></body></html>'''


def process_case(case, frames, gt_tubelet, model, cfg, device, args, window_frames):
    predicted_tubelet, predicted_frame, attention_shapes, alignments = infer_sequence(
        model,
        frames,
        device,
        getattr(torch, cfg.amp_dtype),
        window_frames,
    )
    gt_frame = None
    metrics = {
        "ari_fg_diagnostic": None,
        "mean_best_overlap_diagnostic": None,
        "evaluated_tubelets": 0,
    }
    if gt_tubelet is not None:
        gt_frame = np.repeat(gt_tubelet, 2, axis=0)[: len(frames)]
        metrics = gt_metrics(predicted_tubelet, gt_tubelet)
    assets = render_assets(case, frames, predicted_frame, gt_frame, args.output_dir, args)
    return {
        **case,
        "frames": len(frames),
        "tubelets": len(predicted_tubelet),
        "processed_shape": list(frames.shape),
        "attention_shapes": attention_shapes,
        "window_frames": window_frames if window_frames > 0 else len(frames),
        "window_count": len(attention_shapes),
        "slot_alignment": alignments,
        "metrics": metrics,
        "assets": assets,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    cfg, model, load_report = load_model(
        args.model_config.resolve(), args.checkpoint.resolve(), device
    )
    model_metadata = checkpoint_load_summary(load_report)
    if model_metadata["source_optimizer_step"] != 16000:
        raise RuntimeError(f"expected step-16000, got {model_metadata}")

    ytvis = build_val_source(
        args.model_config.resolve(), args.data_dir, "ytvis_hq_val", args.ytvis_cases, args.seed
    )
    movic = build_val_source(
        args.movic_data_config.resolve(), args.data_dir, "movi_c_val", args.movic_cases, args.seed + 7
    )
    rendered = []
    total_cases = args.ytvis_cases + args.movic_cases
    test5_cases = load_unique_test5(args.test5_file.resolve(), args.test5_cases)
    total_cases += len(test5_cases)
    completed = 0

    for source in (ytvis, movic):
        for index in source["indices"]:
            batch = source["collate"]([source["dataset"][index]])
            frames = decode_dataset_rgb(batch["video"])
            gt = segment_to_labels(batch.get("segment"))
            case = {
                "source": source["source"],
                "case_id": f"{source['source']}_{index:06d}",
                "source_key": source_key(source["dataset"], index),
                "dataset_index": index,
                "sampling_seed": args.seed if source is ytvis else args.seed + 7,
                "input_policy": "official validation transform; one complete forward",
            }
            rendered.append(
                process_case(case, frames, gt, model, cfg, device, args, window_frames=0)
            )
            completed += 1
            print(f"[case] {completed}/{total_cases} {case['case_id']}", flush=True)

    buckets = [tuple(values) for values in cfg.aspect_ratio_buckets]
    for case in test5_cases:
        frames, raw_shape, bucket, video_metadata = decode_external_video(
            Path(case["source_key"]), buckets
        )
        case.update(
            {
                "decoded_frame_shape": raw_shape,
                "aspect_ratio_bucket": bucket,
                "source_fps": float(video_metadata.get("fps", 0.0)),
                "input_policy": (
                    f"all consecutive frames; {args.external_window_frames}-frame "
                    "non-overlapping forwards; no frame sampling"
                ),
            }
        )
        rendered.append(
            process_case(
                case,
                frames,
                None,
                model,
                cfg,
                device,
                args,
                window_frames=args.external_window_frames,
            )
        )
        completed += 1
        print(f"[case] {completed}/{total_cases} {case['case_id']}", flush=True)

    report = {
        "title": "V-JEPA xSSC · latest complete checkpoint · val5 + test_5",
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.model_config.resolve()),
        "checkpoint_load": model_metadata,
        "latest_complete_step": 16000,
        "checkpoint_scope": "YTVIS clip2 branch before a complete MOVi-C checkpoint exists",
        "temporal_mode": cfg.temporal_mode,
        "tubelet_size": 2,
        "train_raw_frames": cfg.raw_clip_frames,
        "train_xssc_steps": cfg.xssc_steps,
        "test5_file": str(args.test5_file.resolve()),
        "test5_unique_sources": len(test5_cases),
        "ytvis_indices": ytvis["indices"],
        "movic_indices": movic["indices"],
        "cases": rendered,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "index.html").write_text(build_html(report))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "cases": len(rendered),
                "ytvis_indices": ytvis["indices"],
                "movic_indices": movic["indices"],
                "test5_unique_sources": len(test5_cases),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
