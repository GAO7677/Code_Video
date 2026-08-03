#!/usr/bin/env python3
"""Visualize xSSC DINOv3 feature reconstruction on fixed MOVi-C val cases."""

import argparse
import gc
import html
import json
from pathlib import Path
import random
import subprocess
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)
SLOT_PALETTE = np.asarray(
    [
        [239, 68, 68], [59, 130, 246], [34, 197, 94], [250, 204, 21],
        [168, 85, 247], [6, 182, 212], [249, 115, 22], [236, 72, 153],
        [132, 204, 22], [20, 184, 166], [148, 163, 184],
    ],
    dtype=np.uint8,
)
ERROR_COLORS = np.asarray(
    [[10, 20, 65], [26, 117, 188], [65, 182, 196], [255, 215, 80], [220, 38, 38]],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path(
            "upstream/config-randsfq/"
            "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--val-subset-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument("--case-indices", type=int, nargs="*")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--model-label", default="DINOv3-L xSSC step-050000")
    parser.add_argument("--backbone-label", default="DINOv3")
    parser.add_argument(
        "--movic-tfrecord-adapter",
        action="store_true",
        help="Use the local MOVi-C TFRecord test split with an official LMDB config.",
    )
    return parser.parse_args()


def resolve_from_root(path):
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_checkpoint(model, checkpoint):
    state_dict = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("m.encode_backbone.")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    gc.collect()


def decode_rgb(video):
    rgb = video.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return (
        rgb.clamp(0, 255).round().to(torch.uint8)[0]
        .permute(0, 2, 3, 1).contiguous().numpy()
    )


def joint_pca_rgb(target, recon):
    # One PCA basis and one robust color range make target/reconstruction colors comparable.
    time_steps, channels, height, width = target.shape
    samples = torch.cat([target, recon], dim=0).permute(0, 2, 3, 1).reshape(-1, channels)
    mean = samples.mean(dim=0, keepdim=True)
    centered = samples - mean
    _, _, basis = torch.pca_lowrank(centered, q=3, center=False, niter=4)
    projected = centered @ basis[:, :3]
    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    projected = (projected * 255).round().to(torch.uint8)
    projected = projected.reshape(2 * time_steps, height, width, 3).cpu().numpy()
    return projected[:time_steps], projected[time_steps:]


def error_heatmap(target, recon):
    error = (recon - target).square().mean(dim=1).sqrt()
    scale = torch.quantile(error, 0.99).clamp_min(1e-6)
    values = (error / scale).clamp(0, 1).cpu().numpy()
    positions = values * (len(ERROR_COLORS) - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(ERROR_COLORS) - 1)
    weight = (positions - lower)[..., None]
    colors = ERROR_COLORS[lower] * (1.0 - weight) + ERROR_COLORS[upper] * weight
    return colors.round().astype(np.uint8), float(scale.float().item())


def upscale_patch_view(values, output_size=256):
    scale_h = output_size // values.shape[1]
    scale_w = output_size // values.shape[2]
    return values.repeat(scale_h, axis=1).repeat(scale_w, axis=2)


def slot_overlay(rgb, attention):
    labels = attention.argmax(dim=1).to(torch.long).cpu().numpy()
    labels = upscale_patch_view(labels)
    colors = SLOT_PALETTE[labels % len(SLOT_PALETTE)]
    return (rgb.astype(np.float32) * 0.42 + colors.astype(np.float32) * 0.58).round().astype(np.uint8)


def label_frames(frames, labels, bar_height=32):
    font = ImageFont.load_default()
    output = []
    for frame in frames:
        canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + bar_height), (15, 18, 22))
        canvas.paste(Image.fromarray(frame), (0, bar_height))
        draw = ImageDraw.Draw(canvas)
        panel_width = frame.shape[1] // len(labels)
        for index, label in enumerate(labels):
            draw.text((index * panel_width + 8, 10), label, fill=(238, 241, 245), font=font)
        output.append(np.asarray(canvas))
    return np.stack(output)


def write_h264(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    process = subprocess.run(
        command,
        input=np.ascontiguousarray(frames).tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors="replace"))


def plot_frame_metrics(path, frame_mse, frame_cosine):
    frames = np.arange(len(frame_mse))
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True)
    axes[0].plot(frames, frame_mse, color="#ef4444", linewidth=1.8)
    axes[0].set_ylabel("feature MSE")
    axes[0].grid(alpha=0.25)
    axes[1].plot(frames, frame_cosine, color="#2563eb", linewidth=1.8)
    axes[1].set_ylabel("patch cosine")
    axes[1].set_xlabel("frame")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_html(metadata):
    data = json.dumps(metadata, separators=(",", ":"))
    title = html.escape(metadata["title"])
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#111418;color:#edf0f3;font:14px system-ui,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:2;background:#111418ee;border-bottom:1px solid #353b43}}
.bar,main{{max-width:1460px;margin:auto}}.bar{{min-height:56px;padding:10px 16px;display:flex;align-items:center;gap:10px}}
h1{{font-size:17px;margin:0 auto 0 0}}select,button{{height:34px;border:1px solid #4a525c;border-radius:5px;background:#20252b;color:#f4f5f6;padding:0 10px}}
button{{width:38px;font-size:17px;cursor:pointer}}main{{padding:18px 16px 36px}}video{{display:block;width:100%;background:#050607;border:1px solid #353b43}}
.note{{color:#abb4be;margin:0 0 14px;line-height:1.55}}.metrics{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:1px;background:#353b43;border:1px solid #353b43;margin:14px 0}}
.metric{{background:#191d22;padding:10px}}.metric span{{display:block;color:#96a0ab;font-size:12px;margin-bottom:4px}}.metric strong{{font-size:16px}}
.chart{{display:block;width:min(860px,100%);margin-top:16px;background:white}}@media(max-width:760px){{.metrics{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:14px}}}}
</style></head><body><header><div class="bar"><h1>{title}</h1><select id="case"></select><button id="restart" title="从头播放">&#8634;</button><button id="play" title="播放">&#9654;</button></div></header><main id="app"></main>
<script>
const DATA={data};const app=document.getElementById('app'),sel=document.getElementById('case'),play=document.getElementById('play'),restart=document.getElementById('restart');let video;
function metric(label,value){{return `<div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>`}}
function render(i){{const c=DATA.cases[i];app.innerHTML=`<p class="note">${{DATA.model_label}} · MOVi-C test index ${{c.index}} · ${{c.frames}} frames · ${{c.record}}<br>五列使用同一时刻：输入 RGB、目标 ${{DATA.backbone_label}} 特征 PCA、xSSC 重构特征 PCA、每个 patch 的特征 RMSE、decoder slot assignment。目标和重构共用同一 PCA 基底与颜色范围。</p><video muted playsinline preload="metadata" src="${{c.video}}"></video><div class="metrics">${{metric('MSE',c.metrics.mse.toFixed(4))}}${{metric('RMSE',c.metrics.rmse.toFixed(4))}}${{metric('Normalized RMSE',c.metrics.normalized_rmse.toFixed(4))}}${{metric('Mean patch cosine',c.metrics.mean_patch_cosine.toFixed(4))}}${{metric('Error q99',c.metrics.error_rmse_q99.toFixed(4))}}</div><img class="chart" src="${{c.frame_metrics_plot}}" alt="逐帧重构指标">`;video=app.querySelector('video');play.innerHTML='&#9654;';}}
DATA.cases.forEach((c,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`Case ${{c.index}} · MSE ${{c.metrics.mse.toFixed(2)}}`;sel.appendChild(o)}});sel.onchange=()=>render(Number(sel.value));play.onclick=()=>{{if(video.paused){{video.play();play.innerHTML='&#10074;&#10074;'}}else{{video.pause();play.innerHTML='&#9654;'}}}};restart.onclick=()=>{{video.currentTime=0;video.play();play.innerHTML='&#10074;&#10074;'}};render(0);
</script></body></html>"""


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_file = resolve_from_root(args.config_file)
    cfg = Config.fromfile(config_file)
    if args.movic_tfrecord_adapter:
        from object_centric_bench.datum import MOViTFRecord

        cfg.dataset_v.type = MOViTFRecord
        cfg.dataset_v.data_file = "kubric-movi/movi-c"
        cfg.dataset_v.split = "test"
    cfg.dataset_v.base_dir = args.data_dir.resolve()
    dataset = build_from_config(cfg.dataset_v)
    collate_fn = build_from_config(cfg.collate_fn_v)

    subset = json.loads(args.val_subset_file.read_text())
    val_indices = [int(index) for index in subset["indices"]]
    if args.case_indices:
        indices = args.case_indices
        outside = sorted(set(indices) - set(val_indices))
        if outside:
            raise ValueError(f"indices are outside the fixed val subset: {outside}")
    else:
        indices = sorted(random.Random(args.seed).sample(val_indices, args.num_cases))

    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    model.freez(cfg.freez, verbose=False)
    load_checkpoint(model, args.checkpoint.resolve())
    model = model.to(device).eval()
    amp_dtype = getattr(torch, args.amp_dtype)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for position, index in enumerate(indices, start=1):
        batch = collate_fn([dataset[index]])
        rgb = decode_rgb(batch["video"])
        gpu_batch = {
            "video": batch["video"].to(device, non_blocking=True),
            "bbox": batch["bbox"].to(device, non_blocking=True),
        }
        with torch.autocast("cuda", dtype=amp_dtype):
            output = model(batch=gpu_batch)
        target = output["feature"][0].float()
        recon = output["recon"][0].float()
        error = recon - target
        frame_mse = error.square().mean(dim=(1, 2, 3))
        frame_cosine = F.cosine_similarity(recon, target, dim=1).mean(dim=(1, 2))
        mse = error.square().mean()
        target_rms = target.square().mean().sqrt()

        target_pca, recon_pca = joint_pca_rgb(target, recon)
        error_rgb, error_q99 = error_heatmap(target, recon)
        target_pca = upscale_patch_view(target_pca)
        recon_pca = upscale_patch_view(recon_pca)
        error_rgb = upscale_patch_view(error_rgb)
        slots = slot_overlay(rgb, output["attentd"][0])
        composite = np.concatenate([rgb, target_pca, recon_pca, error_rgb, slots], axis=2)
        composite = label_frames(
            composite,
            [
                "RGB input",
                f"{args.backbone_label} target PCA",
                "xSSC recon PCA",
                "feature RMSE",
                "decoder slots",
            ],
        )

        case_dir = output_dir / f"case_{index:04d}"
        video_path = case_dir / "reconstruction.mp4"
        plot_path = case_dir / "frame_metrics.png"
        write_h264(video_path, composite, args.fps)
        plot_frame_metrics(plot_path, frame_mse.cpu().numpy(), frame_cosine.cpu().numpy())
        record_path, record_offset, record_size = dataset.records[index]
        metrics = {
            "mse": float(mse.item()),
            "rmse": float(mse.sqrt().item()),
            "normalized_rmse": float((mse.sqrt() / target_rms.clamp_min(1e-8)).item()),
            "mean_patch_cosine": float(frame_cosine.mean().item()),
            "error_rmse_q99": error_q99,
            "frame_mse": frame_mse.cpu().tolist(),
            "frame_patch_cosine": frame_cosine.cpu().tolist(),
        }
        cases.append(
            {
                "index": index,
                "frames": int(rgb.shape[0]),
                "record": f"{Path(record_path).name}@{record_offset}+{record_size}",
                "video": video_path.relative_to(output_dir).as_posix(),
                "frame_metrics_plot": plot_path.relative_to(output_dir).as_posix(),
                "metrics": metrics,
                "shapes": {key: list(output[key].shape) for key in ("feature", "slotz", "recon", "attentd")},
            }
        )
        print(
            f"[{position}/{len(indices)}] val index={index} "
            f"mse={metrics['mse']:.6f} cosine={metrics['mean_patch_cosine']:.6f}",
            flush=True,
        )
        del output, target, recon, error, gpu_batch, batch
        torch.cuda.empty_cache()

    metadata = {
        "title": f"xSSC MOVi-C val reconstruction · {args.model_label}",
        "model_label": args.model_label,
        "backbone_label": args.backbone_label,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(config_file),
        "dataset": str((args.data_dir.resolve() / cfg.dataset_v.data_file)),
        "dataset_split": cfg.dataset_v.split,
        "dataset_size": len(dataset),
        "val_subset_file": str(args.val_subset_file.resolve()),
        "val_subset_size": len(val_indices),
        "selection_seed": args.seed,
        "case_indices": indices,
        "amp_dtype": args.amp_dtype,
        "movic_tfrecord_adapter": args.movic_tfrecord_adapter,
        "metric": "MSE between xSSC decoder reconstruction and frozen DINOv3 patch features",
        "cases": cases,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "index.html").write_text(build_html(metadata))
    print(json.dumps({"output_dir": str(output_dir), "index": str(output_dir / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
