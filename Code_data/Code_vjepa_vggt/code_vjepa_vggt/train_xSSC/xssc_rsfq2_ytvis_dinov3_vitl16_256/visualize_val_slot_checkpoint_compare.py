#!/usr/bin/env python3
"""Compare decoder slot assignments from two xSSC checkpoints on fixed val cases."""

import argparse
import gc
import html
import json
from pathlib import Path
import random
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path(
            "upstream/config-randsfq/"
            "rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-label", default="step-4000")
    parser.add_argument("--target-label", default="step-15000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--case-indices", type=int, nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    return parser.parse_args()


def resolve_from_root(path):
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def set_inference_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("m.encode_backbone.")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch for {checkpoint}: "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    gc.collect()


def decode_rgb(video):
    rgb = video.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return (
        rgb.clamp(0, 255)
        .round()
        .to(torch.uint8)[0]
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )


def key_to_text(key):
    return key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)


def infer_cases(model, dataset, collate_fn, indices, checkpoint, device, amp_dtype, seed):
    load_checkpoint(model, checkpoint)
    model.eval()
    outputs = {}
    with torch.inference_mode():
        for position, index in enumerate(indices, start=1):
            batch = collate_fn([dataset[index]])
            video = batch["video"]
            rgb = decode_rgb(video)
            set_inference_seed(seed + index)
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(batch={"video": video.to(device, non_blocking=True)})
            attention = output["attentd"][0].detach().float().cpu()
            if tuple(attention.shape[1:]) != (7, 16, 16):
                raise RuntimeError(
                    f"expected attentd [T,7,16,16], got {tuple(attention.shape)}"
                )
            labels = attention.argmax(dim=1).to(torch.uint8).numpy()
            outputs[index] = {
                "labels": labels,
                "rgb": rgb,
                "frames": int(rgb.shape[0]),
                "attention_shape": [1, *attention.shape],
            }
            print(
                f"[{checkpoint.name}] case {position}/{len(indices)}: "
                f"index={index}, frames={rgb.shape[0]}",
                flush=True,
            )
            del output, attention, batch, video
    torch.cuda.empty_cache()
    return outputs


def pairwise_iou(anchor_labels, target_labels, num_slots=7):
    if anchor_labels.shape != target_labels.shape:
        raise ValueError(
            f"label shape mismatch: {anchor_labels.shape} != {target_labels.shape}"
        )
    matrix = np.zeros((num_slots, num_slots), dtype=np.float64)
    for anchor_slot in range(num_slots):
        anchor_mask = anchor_labels == anchor_slot
        for target_slot in range(num_slots):
            target_mask = target_labels == target_slot
            union = np.logical_or(anchor_mask, target_mask).sum()
            if union:
                matrix[anchor_slot, target_slot] = (
                    np.logical_and(anchor_mask, target_mask).sum() / union
                )
    return matrix


def align_target_slots(anchor_labels, target_labels, num_slots=7):
    iou = pairwise_iou(anchor_labels, target_labels, num_slots)
    anchor_ids, target_ids = linear_sum_assignment(-iou)
    raw_to_aligned = {
        int(target_id): int(anchor_id)
        for anchor_id, target_id in zip(anchor_ids, target_ids)
    }
    aligned = np.empty_like(target_labels)
    for raw_id, aligned_id in raw_to_aligned.items():
        aligned[target_labels == raw_id] = aligned_id
    matched_iou = {
        int(anchor_id): float(iou[anchor_id, target_id])
        for anchor_id, target_id in zip(anchor_ids, target_ids)
    }
    aligned_to_raw = {aligned_id: raw_id for raw_id, aligned_id in raw_to_aligned.items()}
    return aligned, iou, raw_to_aligned, aligned_to_raw, matched_iou


def occupancy(labels, num_slots=7):
    total = labels.size
    return [float((labels == slot_id).sum() / total) for slot_id in range(num_slots)]


def add_patch_grid(frames, strength=0.28):
    result = frames.copy()
    for position in range(16, frames.shape[1], 16):
        result[:, position, :, :] = (
            result[:, position, :, :].astype(np.float32) * (1.0 - strength)
        ).astype(np.uint8)
    for position in range(16, frames.shape[2], 16):
        result[:, :, position, :] = (
            result[:, :, position, :].astype(np.float32) * (1.0 - strength)
        ).astype(np.uint8)
    return result


def upscale_patch_labels(labels):
    return labels.repeat(16, axis=1).repeat(16, axis=2)


def combined_overlay(rgb, labels):
    labels_full = upscale_patch_labels(labels)
    colors = PALETTE[labels_full]
    overlay = (
        rgb.astype(np.float32) * 0.43 + colors.astype(np.float32) * 0.57
    ).round().clip(0, 255).astype(np.uint8)
    return add_patch_grid(overlay)


def single_slot_overlay(rgb, labels, slot_id):
    labels_full = upscale_patch_labels(labels)
    selected = labels_full == slot_id
    dimmed = (rgb.astype(np.float32) * 0.24).round().astype(np.uint8)
    colored = (
        rgb.astype(np.float32) * 0.36
        + PALETTE[slot_id].astype(np.float32) * 0.64
    ).round().clip(0, 255).astype(np.uint8)
    result = np.where(selected[..., None], colored, dimmed)
    return add_patch_grid(result)


def write_h264(path, frames, fps, ffmpeg):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    command = [
        ffmpeg,
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
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.run(
        command,
        input=np.ascontiguousarray(frames).tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"ffmpeg failed for {path}: {process.stderr.decode(errors='replace')}"
        )


def relative_path(path, root):
    return path.relative_to(root).as_posix()


def export_case_videos(
    output_dir,
    index,
    rgb,
    anchor_labels,
    target_labels,
    anchor_label,
    target_label,
    fps,
    ffmpeg,
):
    case_dir = output_dir / f"case_{index:03d}"
    original_path = case_dir / "original.mp4"
    write_h264(original_path, rgb, fps, ffmpeg)

    checkpoints = []
    for label, labels in (
        (anchor_label, anchor_labels),
        (target_label, target_labels),
    ):
        checkpoint_dir = case_dir / label
        combined_path = checkpoint_dir / "all_slots.mp4"
        write_h264(combined_path, combined_overlay(rgb, labels), fps, ffmpeg)
        slots = []
        slot_occupancy = occupancy(labels)
        for slot_id in range(7):
            slot_path = checkpoint_dir / f"slot_{slot_id}.mp4"
            write_h264(
                slot_path,
                single_slot_overlay(rgb, labels, slot_id),
                fps,
                ffmpeg,
            )
            slots.append(
                {
                    "aligned_slot": slot_id,
                    "occupancy": slot_occupancy[slot_id],
                    "video": relative_path(slot_path, output_dir),
                }
            )
        checkpoints.append(
            {
                "label": label,
                "combined_video": relative_path(combined_path, output_dir),
                "slots": slots,
            }
        )
    return relative_path(original_path, output_dir), checkpoints


def build_html(metadata):
    data_json = json.dumps(metadata, separators=(",", ":"))
    palette_json = json.dumps(PALETTE.tolist())
    title = html.escape(metadata["title"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0;
      background: #101214;
      color: #f4f5f6;
      font: 14px system-ui, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 3;
      border-bottom: 1px solid #34383d;
      background: rgba(16, 18, 20, 0.97);
    }}
    .bar {{
      max-width: 1500px;
      min-height: 58px;
      margin: 0 auto;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    h1 {{ margin: 0 auto 0 0; font-size: 18px; font-weight: 680; }}
    button, select, input {{ font: inherit; }}
    select {{
      height: 34px;
      border: 1px solid #4a5057;
      border-radius: 5px;
      background: #202429;
      color: #f4f5f6;
      padding: 0 30px 0 10px;
    }}
    .icon-button {{
      width: 34px;
      height: 34px;
      border: 1px solid #4a5057;
      border-radius: 5px;
      background: #202429;
      color: #fff;
      cursor: pointer;
    }}
    .icon-button:hover {{ background: #2b3036; }}
    .seek {{ min-width: 170px; flex: 0 1 280px; accent-color: #38bdf8; }}
    .loop {{ display: inline-flex; align-items: center; gap: 6px; color: #c4c9cf; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 18px; }}
    .case-meta {{
      display: flex;
      gap: 18px;
      padding: 0 0 14px;
      color: #abb2ba;
      overflow-x: auto;
      white-space: nowrap;
    }}
    .original {{ max-width: 420px; margin-bottom: 20px; }}
    .original h2, .row-label h2 {{ margin: 0 0 8px; font-size: 14px; }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 1;
      background: #050607;
      border: 1px solid #34383d;
      border-radius: 4px;
    }}
    .compare-head, .compare-row {{
      display: grid;
      grid-template-columns: 118px minmax(280px, 1fr) minmax(280px, 1fr);
      gap: 14px;
      min-width: 730px;
    }}
    .compare-head {{
      position: sticky;
      top: 59px;
      z-index: 2;
      padding: 9px 0;
      border-top: 1px solid #34383d;
      border-bottom: 1px solid #34383d;
      background: rgba(16, 18, 20, 0.96);
      font-weight: 680;
    }}
    .compare-row {{ padding: 14px 0; border-bottom: 1px solid #2b2f34; }}
    .compare-wrap {{ overflow-x: auto; }}
    .row-label {{ padding-top: 2px; }}
    .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      margin-right: 7px;
      border-radius: 2px;
      vertical-align: -1px;
    }}
    figure {{ margin: 0; min-width: 0; }}
    figcaption {{
      min-height: 22px;
      padding-top: 6px;
      color: #aeb5bd;
      font-size: 12px;
    }}
    .metric {{ color: #7dd3fc; }}
    @media (max-width: 760px) {{
      .bar {{ padding: 9px 12px; }}
      h1 {{ width: 100%; }}
      main {{ padding: 12px; }}
      .compare-head {{ top: 104px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <h1>{title}</h1>
      <select id="caseSelect" aria-label="Validation case"></select>
      <button id="restart" class="icon-button" title="Restart" aria-label="Restart">&#8634;</button>
      <button id="play" class="icon-button" title="Play all" aria-label="Play all">&#9654;</button>
      <input id="seek" class="seek" type="range" min="0" max="1" step="0.01" value="0" aria-label="Timeline">
      <select id="speed" aria-label="Playback speed">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
      </select>
      <label class="loop"><input id="loop" type="checkbox" checked>Loop</label>
    </div>
  </header>
  <main id="app"></main>
  <script>
    const DATA = {data_json};
    const PALETTE = {palette_json};
    const app = document.getElementById('app');
    const caseSelect = document.getElementById('caseSelect');
    const playButton = document.getElementById('play');
    const restartButton = document.getElementById('restart');
    const seek = document.getElementById('seek');
    const speed = document.getElementById('speed');
    const loop = document.getElementById('loop');
    let videos = [];
    let master = null;
    let animationFrame = null;

    function videoElement(src) {{
      return `<video muted playsinline preload="metadata" src="${{src}}"></video>`;
    }}

    function renderCase(caseIndex) {{
      if (animationFrame) cancelAnimationFrame(animationFrame);
      const item = DATA.cases[caseIndex];
      const anchor = item.checkpoints[0];
      const target = item.checkpoints[1];
      const rows = [];
      rows.push(`
        <section class="compare-row">
          <div class="row-label"><h2>All slots</h2><span>16 x 16 patches</span></div>
          <figure>${{videoElement(anchor.combined_video)}}<figcaption>Raw slot IDs, anchor colors</figcaption></figure>
          <figure>${{videoElement(target.combined_video)}}<figcaption>Hungarian-aligned colors</figcaption></figure>
        </section>`);
      for (let slotId = 0; slotId < 7; slotId++) {{
        const anchorSlot = anchor.slots[slotId];
        const targetSlot = target.slots[slotId];
        const color = `rgb(${{PALETTE[slotId].join(',')}})`;
        rows.push(`
          <section class="compare-row">
            <div class="row-label"><h2><span class="swatch" style="background:${{color}}"></span>Slot ${{slotId}}</h2></div>
            <figure>${{videoElement(anchorSlot.video)}}<figcaption>raw ${{anchorSlot.raw_slot}} | occupancy <span class="metric">${{(anchorSlot.occupancy * 100).toFixed(1)}}%</span></figcaption></figure>
            <figure>${{videoElement(targetSlot.video)}}<figcaption>raw ${{targetSlot.raw_slot}} | occupancy <span class="metric">${{(targetSlot.occupancy * 100).toFixed(1)}}%</span> | match IoU <span class="metric">${{targetSlot.match_iou.toFixed(3)}}</span></figcaption></figure>
          </section>`);
      }}
      app.innerHTML = `
        <div class="case-meta">
          <span>val index ${{item.index}}</span>
          <span>${{item.frames}} frames</span>
          <span>${{DATA.fps}} fps</span>
          <span>${{item.source_key}}</span>
        </div>
        <section class="original"><h2>Validation input</h2>${{videoElement(item.original_video)}}</section>
        <div class="compare-wrap">
          <div class="compare-head"><span>View</span><span>${{anchor.label}}</span><span>${{target.label}}</span></div>
          ${{rows.join('')}}
        </div>`;
      videos = Array.from(app.querySelectorAll('video'));
      master = videos[0];
      videos.forEach((video) => {{
        video.playbackRate = Number(speed.value);
        video.loop = loop.checked;
      }});
      master.addEventListener('loadedmetadata', () => {{
        seek.max = String(master.duration || item.duration_seconds);
      }}, {{ once: true }});
      master.addEventListener('ended', () => setPlayIcon(false));
      setPlayIcon(false);
      seek.value = '0';
      updateTimeline();
    }}

    function setPlayIcon(playing) {{
      playButton.innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
      playButton.title = playing ? 'Pause all' : 'Play all';
      playButton.setAttribute('aria-label', playButton.title);
    }}

    async function playAll() {{
      videos.forEach((video) => {{
        if (Math.abs(video.currentTime - master.currentTime) > 0.04) video.currentTime = master.currentTime;
      }});
      await Promise.all(videos.map((video) => video.play().catch(() => null)));
      setPlayIcon(true);
    }}

    function pauseAll() {{
      videos.forEach((video) => video.pause());
      setPlayIcon(false);
    }}

    function updateTimeline() {{
      if (master) {{
        seek.value = String(master.currentTime || 0);
        if (!master.paused) {{
          videos.slice(1).forEach((video) => {{
            if (Math.abs(video.currentTime - master.currentTime) > 0.09) video.currentTime = master.currentTime;
          }});
        }}
      }}
      animationFrame = requestAnimationFrame(updateTimeline);
    }}

    DATA.cases.forEach((item, index) => {{
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = `Case ${{String(item.index).padStart(3, '0')}} | ${{item.frames}} frames`;
      caseSelect.appendChild(option);
    }});
    caseSelect.addEventListener('change', () => renderCase(Number(caseSelect.value)));
    playButton.addEventListener('click', () => master.paused ? playAll() : pauseAll());
    restartButton.addEventListener('click', () => {{
      videos.forEach((video) => video.currentTime = 0);
      seek.value = '0';
    }});
    seek.addEventListener('input', () => {{
      const time = Number(seek.value);
      videos.forEach((video) => video.currentTime = time);
    }});
    speed.addEventListener('change', () => {{
      videos.forEach((video) => video.playbackRate = Number(speed.value));
    }});
    loop.addEventListener('change', () => {{
      videos.forEach((video) => video.loop = loop.checked);
    }});
    renderCase(0);
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    amp_dtype = getattr(torch, args.amp_dtype)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_file = resolve_from_root(args.config_file)
    cfg = Config.fromfile(config_file)
    cfg.dataset_v.base_dir = args.data_dir.resolve()
    dataset = build_from_config(cfg.dataset_v)
    collate_fn = build_from_config(cfg.collate_fn_v)

    if args.case_indices:
        indices = sorted(set(args.case_indices))
    else:
        indices = sorted(random.Random(args.seed).sample(range(len(dataset)), args.num_cases))
    if not indices or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError(f"invalid case indices for dataset size {len(dataset)}: {indices}")

    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    model.freez(cfg.freez, verbose=False)
    model = model.to(device).eval()

    checkpoint_specs = [
        (args.anchor_label, args.anchor_checkpoint.resolve()),
        (args.target_label, args.target_checkpoint.resolve()),
    ]
    for _, checkpoint in checkpoint_specs:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    predictions = []
    for _, checkpoint in checkpoint_specs:
        predictions.append(
            infer_cases(
                model,
                dataset,
                collate_fn,
                indices,
                checkpoint,
                device,
                amp_dtype,
                args.seed,
            )
        )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cases = []
    for position, index in enumerate(indices, start=1):
        anchor = predictions[0][index]
        target = predictions[1][index]
        aligned_target, iou, raw_to_aligned, aligned_to_raw, matched_iou = (
            align_target_slots(anchor["labels"], target["labels"])
        )
        original_video, checkpoint_rows = export_case_videos(
            output_dir,
            index,
            anchor["rgb"],
            anchor["labels"],
            aligned_target,
            args.anchor_label,
            args.target_label,
            args.fps,
            ffmpeg,
        )
        for slot in checkpoint_rows[0]["slots"]:
            slot["raw_slot"] = slot["aligned_slot"]
            slot["match_iou"] = 1.0
        for slot in checkpoint_rows[1]["slots"]:
            aligned_id = slot["aligned_slot"]
            slot["raw_slot"] = aligned_to_raw[aligned_id]
            slot["match_iou"] = matched_iou[aligned_id]

        source_key = key_to_text(dataset.keys[index])
        cases.append(
            {
                "position": position,
                "index": index,
                "source_key": source_key,
                "frames": anchor["frames"],
                "duration_seconds": anchor["frames"] / args.fps,
                "original_video": original_video,
                "attention_shape": {
                    args.anchor_label: anchor["attention_shape"],
                    args.target_label: target["attention_shape"],
                },
                "target_raw_to_anchor_slot": {
                    str(key): value for key, value in sorted(raw_to_aligned.items())
                },
                "pairwise_spatiotemporal_iou": iou.tolist(),
                "mean_matched_iou": float(np.mean(list(matched_iou.values()))),
                "checkpoints": checkpoint_rows,
            }
        )
        print(f"[export] case {position}/{len(indices)}: index={index}", flush=True)

    metadata = {
        "title": "xSSC DINOv3 slot comparison",
        "config": str(config_file),
        "dataset": str(args.data_dir.resolve() / cfg.dataset_v.data_file),
        "dataset_size": len(dataset),
        "seed": args.seed,
        "case_indices": indices,
        "fps": args.fps,
        "attention": "decoder final cross-attention (attentd)",
        "assignment": "argmax over 7 slots at native 16x16 patch grid",
        "upsampling": "nearest-neighbor 16x per patch for visualization",
        "alignment": "per-case Hungarian assignment maximizing IoU over T*16*16 patches",
        "checkpoints": [
            {"label": label, "path": str(checkpoint)}
            for label, checkpoint in checkpoint_specs
        ],
        "cases": cases,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "index.html").write_text(build_html(metadata))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "case_indices": indices,
                "videos": len(indices) * 17,
                "index_html": str(output_dir / "index.html"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
