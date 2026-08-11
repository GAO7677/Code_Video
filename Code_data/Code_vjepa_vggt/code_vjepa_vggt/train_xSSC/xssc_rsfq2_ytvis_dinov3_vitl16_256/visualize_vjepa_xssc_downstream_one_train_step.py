#!/usr/bin/env python3
"""Run one downstream train update and render an auditable video report.

This is intentionally a diagnostic, not a replacement training entry point.  It
uses the configured training dataset, frozen V-JEPA xSSC checkpoint, official
recognition model, and official CE + L1 objectives, then updates only the
recognition head once.
"""

from argparse import ArgumentParser
import html
import json
from pathlib import Path
import random
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

from train_ddp_ytvis_hq import (  # noqa: E402
    checkpoint_load_summary,
    load_matching_checkpoint,
)


IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)
PALETTE = np.asarray(
    [
        [0, 0, 0],
        [244, 67, 54],
        [33, 150, 243],
        [76, 175, 80],
        [255, 193, 7],
        [156, 39, 176],
        [0, 188, 212],
        [255, 112, 67],
        [121, 85, 72],
        [63, 81, 181],
        [205, 220, 57],
        [233, 30, 99],
        [0, 150, 136],
        [103, 58, 183],
        [255, 152, 0],
        [96, 125, 139],
    ],
    dtype=np.uint8,
)
YTVIS_CLASSES = [
    "background",
    "person",
    "giant_panda",
    "lizard",
    "parrot",
    "skateboard",
    "sedan",
    "ape",
    "dog",
    "snake",
    "monkey",
    "hand",
    "rabbit",
    "duck",
    "cat",
    "cow",
    "fish",
    "train",
    "horse",
    "turtle",
    "bear",
    "motorbike",
    "giraffe",
    "leopard",
    "fox",
    "deer",
    "owl",
    "surfboard",
    "airplane",
    "truck",
    "zebra",
    "tiger",
    "elephant",
    "snowboard",
    "boat",
    "shark",
    "mouse",
    "frog",
    "eagle",
    "earless_seal",
    "tennis_racket",
]


def class_text(class_id):
    name = YTVIS_CLASSES[class_id] if 0 <= class_id < len(YTVIS_CLASSES) else "unknown"
    return f"{name} (c{class_id})"


def ltrb_status(box):
    box = np.asarray(box, dtype=np.float32)
    finite = bool(np.isfinite(box).all())
    ordered = finite and bool(box[2] > box[0] and box[3] > box[1])
    in_bounds = finite and bool(((0 <= box) & (box <= 1)).all())
    return ordered, in_bounds


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg-file",
        type=Path,
        default=Path(
            "upstream/config-randsfq/"
            "rsfq2_r_recogn-ytvis_hq-vjepa2_1_vitl16_256-video-"
            "slot512-step7000-pilot.py"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/"
            "vjepa_xssc_downstream_one_step_viewer"
        ),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-sample-scan", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--input-fps", type=float, default=6.0)
    parser.add_argument("--output-fps", type=float, default=3.0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_from_root(path):
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def decode_video(video):
    value = video.detach().float().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return (
        value.clamp(0, 255)
        .round()
        .to(torch.uint8)[0]
        .permute(0, 2, 3, 1)
        .numpy()
    )


def metric_means(metrics):
    result = {}
    for key, (value, valid) in metrics.items():
        selected = value[valid]
        if selected.numel() == 0:
            raise RuntimeError(f"metric {key!r} has no valid values")
        result[key] = float(selected.detach().float().mean().item())
    result["total"] = float(sum(result.values()))
    return result


def official_forward(model, loss_fn, batch, amp_dtype):
    output = model(batch=batch)
    metrics = loss_fn(batch=batch, output=output)
    return output, metrics


@torch.inference_mode()
def snapshot(model, loss_fn, batch, amp_dtype):
    model.eval()
    with torch.autocast("cuda", dtype=amp_dtype):
        # Obj-centric slot initialization is stochastic even in eval mode.
        # Replay the exact RNG state for the auxiliary discovery call so its
        # rendered masks correspond to the matches returned by model.forward.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        output, metrics = official_forward(model, loss_fn, batch, amp_dtype)
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        discovery = model.m.discov(batch["video"])
        attention = discovery[model.m.attpd_idx]
        segment_one_hot = model.m.segpd_func(attention)

    matches = []
    prediction_cursor = 0
    for frame_matches in output["rcidx"]:
        frame_records = []
        for match in frame_matches.detach().cpu().tolist():
            slot_index, gt_index = (int(match[0]), int(match[1]))
            logits = output["clspd"][prediction_cursor]
            pred_box = [
                float(value)
                for value in output["boxpd"][prediction_cursor].detach().float().cpu()
            ]
            gt_box = [
                float(value)
                for value in output["boxgt"][prediction_cursor].detach().float().cpu()
            ]
            box_valid, box_in_bounds = ltrb_status(pred_box)
            frame_records.append(
                {
                    "slot": slot_index,
                    "gt_index": gt_index,
                    "pred_class": int(logits.argmax().item()),
                    "gt_class": int(output["clsgt"][prediction_cursor].item()),
                    "confidence": float(logits.float().softmax(0).max().item()),
                    "pred_box_ltrb_raw": pred_box,
                    "pred_box_valid_ltrb": box_valid,
                    "pred_box_in_unit_bounds": box_in_bounds,
                    "gt_box_ltrb": gt_box,
                }
            )
            prediction_cursor += 1
        matches.append(frame_records)
    if prediction_cursor != len(output["clsgt"]):
        raise RuntimeError(
            f"match/output mismatch: {prediction_cursor} vs {len(output['clsgt'])}"
        )
    return {
        "loss": metric_means(metrics),
        "segment": segment_one_hot[0].to(torch.uint8).argmax(dim=-1).cpu().numpy(),
        "matches": matches,
    }


def one_train_step(model, loss_fn, batch, amp_dtype, learning_rate, clip_norm):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=amp_dtype):
        _, metrics = official_forward(model, loss_fn, batch, amp_dtype)
        loss = sum(value[valid].mean() for value, valid in metrics.values())
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, clip_norm)
    if not bool(torch.isfinite(gradient_norm).item()):
        raise RuntimeError(f"non-finite gradient norm: {gradient_norm}")
    optimizer.step()
    return metric_means(metrics), float(gradient_norm.detach().float().item())


def load_font(size=14):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT = load_font(13)
FONT_SMALL = load_font(11)


def add_header(frame, lines, color=(255, 255, 255), height=58):
    canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + height), (15, 18, 22))
    canvas.paste(Image.fromarray(frame), (0, height))
    draw = ImageDraw.Draw(canvas)
    y = 5
    for line_index, line in enumerate(lines):
        draw.text(
            (7, y),
            str(line),
            fill=color if line_index == 0 else (195, 203, 212),
            font=FONT if line_index == 0 else FONT_SMALL,
        )
        y += 19 if line_index == 0 else 15
    return np.asarray(canvas)


def blend_labels(rgb, labels, include_background=False):
    color = PALETTE[labels % len(PALETTE)]
    selected = np.ones(labels.shape, dtype=bool) if include_background else labels > 0
    result = rgb.copy()
    result[selected] = (
        rgb[selected].astype(np.float32) * 0.52
        + color[selected].astype(np.float32) * 0.48
    ).astype(np.uint8)
    return result


def draw_gt(frame, segment, boxes, classes):
    labels = segment.argmax(axis=-1)
    result = blend_labels(frame, labels, include_background=False)
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    height, width = labels.shape
    for object_index, (box, clazz) in enumerate(zip(boxes, classes)):
        clazz = int(clazz)
        left, top, right, bottom = [float(value) for value in box]
        if clazz <= 0 or right <= left or bottom <= top:
            continue
        coords = (
            max(0, min(width - 1, round(left * width))),
            max(0, min(height - 1, round(top * height))),
            max(0, min(width - 1, round(right * width))),
            max(0, min(height - 1, round(bottom * height))),
        )
        color = tuple(int(value) for value in PALETTE[(object_index + 1) % len(PALETTE)])
        draw.rectangle(coords, outline=color, width=3)
        draw.text(
            (coords[0] + 3, max(1, coords[1] + 2)),
            f"GT {class_text(clazz)}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            font=FONT_SMALL,
        )
    return np.asarray(image)


def summarize_matches(records):
    if not records:
        return "no IoU>0.1 slot/GT match"
    return " | ".join(
        f"P:{class_text(record['pred_class'])} {record['confidence']:.2f} "
        f"| GT:{class_text(record['gt_class'])}"
        for record in records
    )


def box_to_pixels(box, width, height):
    ordered, _ = ltrb_status(box)
    if not ordered:
        return None
    left, top, right, bottom = np.clip(np.asarray(box, dtype=np.float32), 0, 1)
    if right <= left or bottom <= top:
        return None
    return (
        max(0, min(width - 1, round(float(left) * width))),
        max(0, min(height - 1, round(float(top) * height))),
        max(0, min(width - 1, round(float(right) * width))),
        max(0, min(height - 1, round(float(bottom) * height))),
    )


def render_output(frame, labels, records, box_color):
    result = blend_labels(frame, labels, include_background=True)
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    height, width = labels.shape
    invalid_row = 4
    for record in records:
        coords = box_to_pixels(record["pred_box_ltrb_raw"], width, height)
        label = (
            f"PRED {class_text(record['pred_class'])} "
            f"p={record['confidence']:.2f}"
        )
        if coords is None:
            draw.text(
                (5, invalid_row),
                f"INVALID BOX · {label}",
                fill=box_color,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
                font=FONT_SMALL,
            )
            invalid_row += 17
            continue
        draw.rectangle(coords, outline=box_color, width=3)
        draw.text(
            (coords[0] + 3, max(1, coords[1] + 2)),
            label,
            fill=box_color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            font=FONT_SMALL,
        )
    return np.asarray(image), summarize_matches(records)


def fit_width(frame, width):
    if frame.shape[1] == width:
        return frame
    image = Image.fromarray(frame)
    height = round(frame.shape[0] * width / frame.shape[1])
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))


def write_h264(path, frames, fps):
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected T,H,W,3 frames, got {frames.shape}")
    height, width = frames.shape[1:3]
    if width % 2 or height % 2:
        raise ValueError(f"H.264 requires even dimensions, got {width}x{height}")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    subprocess.run(command, input=frames.tobytes(), check=True)


def render_page(metadata):
    before = metadata["losses"]["before"]
    update = metadata["losses"]["train_update"]
    after = metadata["losses"]["after"]
    loss_rows = []
    for key in ("ce", "l1", "total"):
        delta = after[key] - before[key]
        loss_rows.append(
            "<tr>"
            f"<td>{html.escape(key.upper())}</td>"
            f"<td>{before[key]:.6f}</td>"
            f"<td>{update[key]:.6f}</td>"
            f"<td>{after[key]:.6f}</td>"
            f"<td class={'good' if delta < 0 else 'bad'}>{delta:+.6f}</td>"
            "</tr>"
        )
    detection_rows = []
    for frame_index, (before_records, after_records) in enumerate(
        zip(metadata["detections"]["before"], metadata["detections"]["after"])
    ):
        if len(before_records) != len(after_records):
            raise RuntimeError("before/after detection counts differ")
        for before_record, after_record in zip(before_records, after_records):
            if (before_record["slot"], before_record["gt_index"]) != (
                after_record["slot"],
                after_record["gt_index"],
            ):
                raise RuntimeError("before/after slot-to-GT matches differ")
            fmt_box = lambda box: "[" + ", ".join(f"{value:.3f}" for value in box) + "]"
            before_status = (
                "valid" if before_record["pred_box_valid_ltrb"] else "invalid"
            )
            after_status = "valid" if after_record["pred_box_valid_ltrb"] else "invalid"
            detection_rows.append(
                "<tr>"
                f"<td>t{frame_index}</td>"
                f"<td>s{before_record['slot']}→gt{before_record['gt_index']}</td>"
                f"<td>{html.escape(class_text(before_record['gt_class']))}</td>"
                f"<td><code>{fmt_box(before_record['gt_box_ltrb'])}</code></td>"
                f"<td>{html.escape(class_text(before_record['pred_class']))}<br>p={before_record['confidence']:.3f}</td>"
                f"<td class={'good' if before_status == 'valid' else 'bad'}>{before_status}<br><code>{fmt_box(before_record['pred_box_ltrb_raw'])}</code></td>"
                f"<td>{html.escape(class_text(after_record['pred_class']))}<br>p={after_record['confidence']:.3f}</td>"
                f"<td class={'good' if after_status == 'valid' else 'bad'}>{after_status}<br><code>{fmt_box(after_record['pred_box_ltrb_raw'])}</code></td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V-JEPA xSSC 单步训练诊断</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:#0d1117; color:#e6edf3; font:14px system-ui,sans-serif; }}
    header {{ position:sticky; top:0; z-index:3; padding:14px 22px; border-bottom:1px solid #30363d; background:rgba(13,17,23,.96); }}
    h1 {{ margin:0 0 4px; font-size:20px; }} h2 {{ font-size:16px; margin:0 0 12px; }}
    .sub {{ color:#8b949e; }} main {{ max-width:1500px; margin:auto; padding:20px; }}
    .toolbar {{ position:fixed; right:20px; bottom:20px; z-index:5; }}
    button {{ border:1px solid #388bfd; border-radius:7px; padding:10px 15px; color:white; background:#1f6feb; cursor:pointer; font-weight:650; }}
    .grid {{ display:grid; grid-template-columns:1fr 2fr; gap:16px; }}
    .card {{ padding:15px; border:1px solid #30363d; border-radius:8px; background:#161b22; min-width:0; }}
    video {{ display:block; width:100%; background:#000; border-radius:5px; }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid #30363d; text-align:right; }}
    th:first-child,td:first-child {{ text-align:left; }} .good {{ color:#3fb950; }} .bad {{ color:#f85149; }}
    .notes {{ margin-top:16px; display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    .wide {{ margin-top:16px; overflow-x:auto; }} .wide td,.wide th {{ white-space:nowrap; }}
    code {{ color:#79c0ff; overflow-wrap:anywhere; }} li {{ margin:7px 0; color:#b1bac4; }}
    @media(max-width:900px) {{ .grid,.notes {{ grid-template-columns:1fr; }} main {{ padding:10px; }} }}
  </style>
</head>
<body>
  <header><h1>V-JEPA xSSC 下游单步训练诊断</h1><div class="sub">非因果 step-7000 · sample {metadata['sample_index']} · 一次 recognition-head update</div></header>
  <main>
    <section class="grid">
      <article class="card"><h2>完整输入（6 帧）</h2><video controls muted loop autoplay playsinline src="input.mp4"></video></article>
      <article class="card"><h2>同步对照：输入 / GT box+标签 / Pred box+识别（更新前后）</h2><video controls muted loop autoplay playsinline src="comparison.mp4"></video></article>
    </section>
    <section class="notes">
      <article class="card"><h2>Loss</h2><table><thead><tr><th>项</th><th>更新前 eval</th><th>训练 forward</th><th>更新后 eval</th><th>after-before</th></tr></thead><tbody>{''.join(loss_rows)}</tbody></table><p class="sub">梯度 L2 norm（clip 前）：{metadata['gradient_norm_before_clip']:.6f}；clip={metadata['gradient_clip_norm']}; LR={metadata['learning_rate']}</p></article>
      <article class="card"><h2>可复现信息</h2><ul>
        <li>训练输入 <code>{metadata['shapes']['video']}</code>，GT <code>{metadata['shapes']['segment']}</code></li>
        <li>V-JEPA tubelet 对齐原始帧索引：<code>{metadata['label_frame_indices']}</code></li>
        <li>严格加载：{metadata['checkpoint_load']['matched_key_count']} keys，source step={metadata['checkpoint_load']['source_optimizer_step']}</li>
        <li>Checkpoint：<code>{html.escape(metadata['source_checkpoint'])}</code></li>
        <li>GT：真实 mask + 归一化 LTRB bbox + 类别标签；Pred：slot mask + raw LTRB bbox + 识别类别/置信度。</li>
        <li>预测框按训练目标的 LTRB 语义解释；画图副本裁剪到 [0,1]，表格始终保留未修改的 raw 值。</li>
      </ul></article>
    </section>
    <section class="card wide"><h2>GT 与识别/框回归明细</h2><table><thead><tr><th>时刻</th><th>匹配</th><th>GT 标签</th><th>GT LTRB</th><th>更新前识别</th><th>更新前 Pred LTRB</th><th>更新后识别</th><th>更新后 Pred LTRB</th></tr></thead><tbody>{''.join(detection_rows)}</tbody></table></section>
  </main>
  <div class="toolbar"><button id="replay">全部重新播放</button></div>
  <script>document.getElementById('replay').onclick=()=>document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play();}});</script>
</body></html>"""


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic requires a CUDA GPU")

    from object_centric_bench.datum import DataLoader
    from object_centric_bench.learn import MetricWrap
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_file = resolve_from_root(args.cfg_file)
    cfg = Config.fromfile(config_file)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    cfg.dataset_t.base_dir = data_dir
    dataset = build_from_config(cfg.dataset_t)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index {args.sample_index} outside [0, {len(dataset)})")
    collate = build_from_config(cfg.collate_fn_t)
    device = torch.device("cuda", 0)

    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    load_report = load_matching_checkpoint(
        model,
        Path(cfg.source_checkpoint),
        exclude_patterns=cfg.transfer_load_exclude,
        allowed_missing_patterns=cfg.transfer_allowed_missing,
        expected_source_variant=cfg.transfer_expected_source_variant,
        expected_source_step=cfg.transfer_expected_source_step,
        prefix_map=cfg.transfer_prefix_map,
    )
    model.freez(cfg.freez, verbose=False)
    model = model.to(device)
    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn_t)).to(device)
    amp_dtype = getattr(torch, cfg.amp_dtype) if isinstance(cfg.amp_dtype, str) else cfg.amp_dtype
    learning_rate = float(cfg.lr if args.learning_rate is None else args.learning_rate)

    before = None
    selected_sample_index = None
    batch_cpu = None
    batch = None
    for scan_offset in range(args.max_sample_scan):
        candidate_index = (args.sample_index + scan_offset) % len(dataset)
        candidate_cpu = collate([dataset[candidate_index]])
        candidate = to_device(candidate_cpu, device)
        try:
            set_seed(args.seed + 100_000)
            candidate_before = snapshot(model, loss_fn, candidate, amp_dtype)
        except RuntimeError as error:
            if "has no valid values" not in str(error):
                raise
            print(
                f"[sample-scan] index={candidate_index} has no IoU>0.1 match; skip",
                flush=True,
            )
            continue
        selected_sample_index = candidate_index
        batch_cpu = candidate_cpu
        batch = candidate
        before = candidate_before
        break
    if before is None:
        raise RuntimeError(
            f"no valid matched training sample found in {args.max_sample_scan} "
            f"samples starting at {args.sample_index}"
        )
    print(
        f"[sample-selected] requested={args.sample_index} "
        f"selected={selected_sample_index}",
        flush=True,
    )
    set_seed(args.seed + 200_000)
    update_loss, gradient_norm = one_train_step(
        model,
        loss_fn,
        batch,
        amp_dtype,
        learning_rate,
        float(cfg.gradient_clip_norm),
    )
    set_seed(args.seed + 100_000)
    after = snapshot(model, loss_fn, batch, amp_dtype)
    slot_masks_identical = bool(np.array_equal(before["segment"], after["segment"]))
    if not slot_masks_identical:
        raise RuntimeError(
            "frozen xSSC slot masks changed under the same snapshot RNG; "
            "before/after comparison is confounded"
        )

    raw_video = decode_video(batch_cpu["video"])
    label_indices = [int(index) for index in cfg.label_frame_indices]
    aligned = raw_video[label_indices]
    gt_segment = batch_cpu["segment"][0].cpu().numpy()
    gt_boxes = batch_cpu["bbox"][0].cpu().numpy()
    gt_classes = batch_cpu["clazz"][0].cpu().numpy()

    input_frames = [
        add_header(
            frame,
            ["INPUT", f"raw frame {frame_id + 1}/{len(raw_video)} · normalized model tensor"],
            color=(121, 192, 255),
        )
        for frame_id, frame in enumerate(raw_video)
    ]
    comparison_frames = []
    for time_index, source_frame_index in enumerate(label_indices):
        source = add_header(
            aligned[time_index],
            ["INPUT (aligned)", f"raw index {source_frame_index} · tubelet {time_index}"],
            color=(121, 192, 255),
        )
        gt = add_header(
            draw_gt(
                aligned[time_index],
                gt_segment[time_index],
                gt_boxes[time_index],
                gt_classes[time_index],
            ),
            ["GT TARGET", "mask + normalized LTRB bbox + class"],
            color=(126, 231, 135),
        )
        before_frame, before_text = render_output(
            aligned[time_index],
            before["segment"][time_index],
            before["matches"][time_index],
            box_color=(255, 196, 107),
        )
        before_panel = add_header(
            before_frame,
            ["OUTPUT BEFORE", before_text],
            color=(255, 196, 107),
        )
        after_frame, after_text = render_output(
            aligned[time_index],
            after["segment"][time_index],
            after["matches"][time_index],
            box_color=(224, 145, 255),
        )
        after_panel = add_header(
            after_frame,
            ["OUTPUT AFTER 1 UPDATE", after_text],
            color=(224, 145, 255),
        )
        width = max(panel.shape[1] for panel in (source, gt, before_panel, after_panel))
        panels = [fit_width(panel, width) for panel in (source, gt, before_panel, after_panel)]
        comparison_frames.append(np.concatenate(panels, axis=1))

    write_h264(output_dir / "input.mp4", input_frames, args.input_fps)
    write_h264(output_dir / "comparison.mp4", comparison_frames, args.output_fps)
    Image.fromarray(comparison_frames[0]).save(output_dir / "poster.png")

    metadata = {
        "config_file": str(config_file),
        "source_checkpoint": str(Path(cfg.source_checkpoint).resolve()),
        "sample_index": selected_sample_index,
        "requested_sample_index": args.sample_index,
        "sample_scan_count": selected_sample_index - args.sample_index + 1,
        "seed": args.seed,
        "snapshot_seed": args.seed + 100_000,
        "train_update_seed": args.seed + 200_000,
        "device": torch.cuda.get_device_name(0),
        "learning_rate": learning_rate,
        "gradient_clip_norm": float(cfg.gradient_clip_norm),
        "gradient_norm_before_clip": gradient_norm,
        "label_frame_indices": label_indices,
        "shapes": {
            key: list(batch_cpu[key].shape)
            for key in ("video", "segment", "bbox", "clazz")
        },
        "losses": {
            "before": before["loss"],
            "train_update": update_loss,
            "after": after["loss"],
        },
        "matches_per_frame": {
            "before": [len(records) for records in before["matches"]],
            "after": [len(records) for records in after["matches"]],
        },
        "detections": {
            "before": before["matches"],
            "after": after["matches"],
        },
        "frozen_slot_masks_identical_before_after": slot_masks_identical,
        "checkpoint_load": checkpoint_load_summary(load_report),
        "interpretation": {
            "gt_overlay": "ground-truth mask, normalized LTRB box, and class",
            "output_overlay": (
                "xSSC slot mask, raw predicted normalized LTRB box, predicted "
                "class, and confidence"
            ),
            "box_output": (
                "raw prediction is retained in metadata/table; only the rendered "
                "copy is clipped to the [0,1] viewport, without coordinate reordering"
            ),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "index.html").write_text(render_page(metadata))
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"[done] {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
