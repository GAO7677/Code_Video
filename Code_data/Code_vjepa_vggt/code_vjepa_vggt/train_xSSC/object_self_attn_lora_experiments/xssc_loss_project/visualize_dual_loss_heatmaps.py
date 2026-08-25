#!/usr/bin/env python3
"""Visualize flow-matching and xSSC loss heatmaps over one training video."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import random
import sys

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_DIR.parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parents[2]
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
for _path in (
    PROJECT_DIR,
    EXPERIMENT_ROOT,
    TRAIN_XSSC_ROOT,
    PACKAGE_ROOT,
    DIFFSYNTH_ROOT,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cv2
import torch
import torch.nn.functional as F

import diagnose_xssc_loss as common
import train_xssc_context_slots as dataset_module
import train_xssc_object_self_attn_lora as core
import train_xssc_object_self_attn_lora_xssc_loss as trainer
from code_vjepa_vggt import context_wan_v_newtrain as context_wan


DEFAULT_CONFIG = (
    PROJECT_DIR
    / "configs/full_sa_no_object_xssc_loss_dinov3_movic_step50000.json"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sigma-targets",
        default="0.05,0.20,0.40,0.60,0.80,0.95",
        help="Comma-separated scheduler sigma targets.",
    )
    return parser.parse_args()


def _resolve_scheduler_levels(
    scheduler,
    requested: list[float],
) -> list[dict[str, float | int]]:
    sigmas = scheduler.sigmas.detach().float().cpu().flatten()
    timesteps = scheduler.timesteps.detach().float().cpu().flatten()
    levels = []
    used = set()
    for target in requested:
        index = int(torch.argmin((sigmas - float(target)).abs()).item())
        if index in used:
            continue
        used.add(index)
        levels.append(
            {
                "scheduler_index": index,
                "sigma_target": float(target),
                "sigma": float(sigmas[index].item()),
                "timestep": float(timesteps[index].item()),
            }
        )
    return sorted(levels, key=lambda item: float(item["sigma"]))


def _flow_map_with_context_mask(
    squared_error: torch.Tensor,
    captured_inputs: dict,
    training_weight: torch.Tensor,
) -> torch.Tensor:
    heatmap = squared_error.float().mean(dim=1) * training_weight.float()
    context_indices = context_wan.resolve_context_latent_indices_from_frames(
        raw_frame_indices=captured_inputs.get("context_frame_indices"),
        raw_num_frames=captured_inputs.get("num_frames"),
        latent_length=int(heatmap.shape[1]),
    )
    if context_indices:
        heatmap = heatmap.clone()
        heatmap[:, context_indices] = 0.0
        return heatmap
    prefix_length = context_wan.resolve_num_clean_prefix_latents(
        clean_prefix_latents=captured_inputs.get("clean_prefix_latents"),
        num_clean_prefix_latents=captured_inputs.get(
            "num_clean_prefix_latents"
        ),
    )
    if prefix_length > 0:
        heatmap = heatmap.clone()
        heatmap[:, :prefix_length] = 0.0
    elif "first_frame_latents" in captured_inputs:
        heatmap = heatmap.clone()
        heatmap[:, :1] = 0.0
    return heatmap


def _xssc_attention_projected_map(
    visuals: dict[str, torch.Tensor],
    future_start_frame: int,
) -> torch.Tensor:
    pred_slots = F.normalize(visuals["pred_slots"].float(), dim=-1)
    target_slots = F.normalize(visuals["target_slots"].float(), dim=-1)
    slot_distance = 1.0 - (pred_slots * target_slots).sum(dim=-1)
    valid = visuals["valid_slots"].to(
        device=slot_distance.device,
        dtype=slot_distance.dtype,
    )
    attention = 0.5 * (
        visuals["pred_attention"].float()
        + visuals["target_attention"].float()
    )
    attention = attention.clamp_min(0.0) * valid[:, None, :, None, None]
    attention = attention / attention.sum(dim=2, keepdim=True).clamp_min(1e-8)
    heatmap = (
        attention * slot_distance[:, :, :, None, None]
    ).sum(dim=2)
    heatmap = heatmap.clone()
    heatmap[:, :future_start_frame] = 0.0
    return heatmap


def _global_p99(maps: list[torch.Tensor]) -> float:
    values = torch.cat([item.detach().float().flatten() for item in maps])
    positive = values[values > 0]
    if not positive.numel():
        return 1.0
    return max(float(torch.quantile(positive, 0.99).item()), 1e-12)


def _upsample_heatmap_spatially(
    heatmap: torch.Tensor,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Resize only H/W, preserving the loss's native temporal axis."""
    if heatmap.ndim != 4:
        raise ValueError(f"Expected [B,T,H,W] heatmap, got {tuple(heatmap.shape)}")
    batch, time_steps, source_height, source_width = heatmap.shape
    resized = F.interpolate(
        heatmap.float().reshape(batch * time_steps, 1, source_height, source_width),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, time_steps, height, width)
    return resized[0].cpu().numpy()


def _causal_video_to_latent_indices(
    *,
    latent_steps: int,
    video_frames: int,
    temporal_upscale: int,
    frames_to_trim: int,
) -> np.ndarray:
    """Map decoded video frames to their Tiny-VAE causal latent groups."""
    raw_mapping = np.repeat(
        np.arange(latent_steps, dtype=np.int64),
        temporal_upscale,
    )
    mapping = raw_mapping[frames_to_trim : frames_to_trim + video_frames]
    if mapping.shape[0] != video_frames:
        raise ValueError(
            "Tiny-VAE temporal mapping is too short: "
            f"{mapping.shape[0]} mapped frames for {video_frames} video frames"
        )
    return mapping


def _latent_video_frame_groups(
    video_to_latent: np.ndarray,
    latent_steps: int,
) -> list[list[int]]:
    groups = [
        np.flatnonzero(video_to_latent == latent_id).astype(int).tolist()
        for latent_id in range(latent_steps)
    ]
    if any(not group for group in groups):
        raise ValueError("Every flow latent step must map to at least one video frame")
    return groups


def _annotate_frames(frames: np.ndarray, labels: list[str]) -> np.ndarray:
    if int(frames.shape[0]) != len(labels):
        raise ValueError("Frame and annotation counts do not match")
    output = frames.copy()
    for frame, label in zip(output, labels):
        cv2.putText(
            frame,
            label,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def _overlay_heatmap(
    base: np.ndarray,
    heatmap: np.ndarray,
    scale: float,
    alpha: float = 0.52,
) -> np.ndarray:
    normalized = np.clip(heatmap / max(float(scale), 1e-12), 0.0, 1.0)
    output = np.empty_like(base)
    for frame_id in range(base.shape[0]):
        heat_u8 = np.rint(normalized[frame_id] * 255.0).astype(np.uint8)
        color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mask = normalized[frame_id, :, :, None]
        mixed = (
            base[frame_id].astype(np.float32) * (1.0 - alpha * mask)
            + color_rgb.astype(np.float32) * (alpha * mask)
        )
        output[frame_id] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return output


def _build_page(
    output_dir: Path,
    records: list[dict],
    *,
    flow_scale: float,
    xssc_scale: float,
    sample_metadata: dict,
    timeline_metadata: dict,
) -> None:
    options = "".join(
        f'<option value="{index}">σ={record["sigma"]:.4f} · '
        f't={record["timestep"]:.1f}</option>'
        for index, record in enumerate(records)
    )
    sections = []
    for index, record in enumerate(records):
        level = html.escape(record["folder"])
        sections.append(
            f"""
<section class="level" data-level="{index}" {'hidden' if index else ''}>
  <div class="metrics">
    <span>scheduler index <b>{record['scheduler_index']}</b></span>
    <span>sigma <b>{record['sigma']:.6f}</b></span>
    <span>timestep <b>{record['timestep']:.2f}</b></span>
    <span>main loss <b>{record['loss_main']:.6f}</b></span>
    <span>xSSC loss <b>{record['loss_xssc']:.6f}</b></span>
    <span>valid slots <b>{record['valid_slot_fraction']:.3f}</b></span>
  </div>
  <div class="grid">
    <article><h2>Predicted x0 · Tiny VAE</h2><video controls muted loop playsinline src="{level}/pred_x0.mp4"></video></article>
    <article><h2>Flow 原生时间轴 · 13 latent steps</h2><video controls muted loop playsinline src="{level}/flow_loss_native_13step.mp4"></video></article>
    <article><h2>Flow 因果保持对齐 · 49 video frames</h2><video controls muted loop playsinline src="{level}/flow_loss_causal_49f.mp4"></video></article>
    <article><h2>xSSC 原生时间轴 · 49 video frames</h2><video controls muted loop playsinline src="{level}/xssc_loss_native_49f.mp4"></video></article>
  </div>
</section>"""
        )
    metadata_text = html.escape(json.dumps(sample_metadata, ensure_ascii=False))
    timeline_text = html.escape(json.dumps(timeline_metadata, ensure_ascii=False))
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual-loss sigma sweep</title><style>
:root{{--bg:#0c1017;--panel:#151b25;--line:#2a3444;--text:#eef3fb;--muted:#9ba9bd;--accent:#8bc5ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px}} h1{{margin:0 0 8px}} p{{color:var(--muted)}}
.toolbar{{position:sticky;top:0;z-index:3;background:rgba(12,16,23,.94);padding:14px 0;display:flex;gap:12px;align-items:center}}
select,button{{background:#202a38;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:15px}}
#replay{{position:fixed;right:22px;bottom:22px;z-index:5;background:#1976d2;border-color:#54a6f5}}
.metrics{{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0 16px}} .metrics span{{background:var(--panel);border:1px solid var(--line);padding:8px 11px;border-radius:8px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} article{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}}
h2{{font-size:16px;margin:0 0 10px;color:var(--accent)}} video{{width:100%;display:block;border-radius:7px;background:#000}}
.note{{background:var(--panel);border-left:4px solid var(--accent);padding:12px 14px;border-radius:6px}}
code{{color:#b8dbff;word-break:break-all}} @media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Full-SA + No-Object：Flow / xSSC 双 Loss 多噪声热力图</h1>
<p>同一训练样本、同一 Gaussian noise realization；只改变 scheduler σ。GT 视频固定，所有 flow 图共享 P99={flow_scale:.6g}，所有 xSSC 图共享 P99={xssc_scale:.6g}。</p>
<div class="note">Flow 保留 13 个 DiT latent step，只做空间上采样；49 帧对齐版按 Tiny VAE 因果分组零阶保持，不做时间插值。xSSC 保留实际逐视频帧的 49-step 时间轴。Flow 是逐 latent voxel MSE；xSSC 是 slot cosine distance 经 attention 投影的空间归因，强度不应按像素一一相等。</div>
<div class="toolbar"><label for="level">噪声强度：</label><select id="level">{options}</select><a href="gt.mp4"><button>GT 视频</button></a></div>
{''.join(sections)}
<p>训练样本：<code>{metadata_text}</code></p>
<p>时间轴映射：<code>{timeline_text}</code></p>
</main><button id="replay">同步重新播放</button>
<script>
const select=document.getElementById('level');
const replayButton=document.getElementById('replay');
function waitUntilReady(video){{
  video.loop=false;
  video.removeAttribute('loop');
  if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) return Promise.resolve();
  return new Promise((resolve) => {{
    const done = () => resolve();
    video.addEventListener('canplay', done, {{once:true}});
    video.addEventListener('error', done, {{once:true}});
    video.preload='auto';
    video.load();
  }});
}}
async function replay(){{
  const videos=[...document.querySelectorAll('.level:not([hidden]) video')];
  if (!videos.length) return;
  replayButton.disabled=true;
  videos.forEach((video)=>{{video.pause(); video.currentTime=0;}});
  await Promise.all(videos.map(waitUntilReady));
  await Promise.allSettled(videos.map((video)=>video.play()));
  replayButton.disabled=false;
}}
function showLevel(){{
  document.querySelectorAll('.level').forEach((el,i)=>{{
    el.hidden=i!==Number(select.value);
    if (el.hidden) el.querySelectorAll('video').forEach((video)=>video.pause());
  }});
  replay();
}}
select.addEventListener('change',showLevel);
replayButton.addEventListener('click',replay);
</script></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    requested_sigmas = [
        float(item.strip())
        for item in args.sigma_targets.split(",")
        if item.strip()
    ]
    if not requested_sigmas:
        raise ValueError("At least one sigma target is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_args, _config = common._parse_trainer_args(
        args.config,
        args.width,
        args.height,
        "lowest",
    )
    train_args.experiment_seed = int(args.seed)
    dataset = dataset_module.build_dataset(train_args)
    sample_index = int(args.sample_index) % len(dataset)
    sample = dataset[sample_index]
    accelerator = common._Accelerator(torch.device("cuda:0"))
    model = trainer.build_model(train_args, accelerator)
    model.train()
    levels = _resolve_scheduler_levels(model.pipe.scheduler, requested_sigmas)

    capture: dict[str, object] = {}
    fixed_noise: torch.Tensor | None = None
    active_level: dict[str, float | int] | None = None
    original_flow_loss = core.base.flow_match_context_sft_loss
    original_xssc_loss = model._xssc_feature_loss

    def controlled_flow_loss(pipe, **inputs):
        nonlocal fixed_noise
        if active_level is None:
            raise RuntimeError("No active scheduler level")
        index = int(active_level["scheduler_index"])
        timestep = pipe.scheduler.timesteps[index : index + 1].to(
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        input_latents = inputs["input_latents"]
        if fixed_noise is None:
            generator = torch.Generator(device=input_latents.device)
            generator.manual_seed(int(args.seed) + 1729)
            fixed_noise = torch.randn(
                input_latents.shape,
                generator=generator,
                device=input_latents.device,
                dtype=input_latents.dtype,
            )
        noise = fixed_noise
        training_target = pipe.scheduler.training_target(
            input_latents,
            noise,
            timestep,
        )
        context_indices = context_wan.resolve_context_latent_indices_from_frames(
            raw_frame_indices=inputs.get("context_frame_indices"),
            raw_num_frames=inputs.get("num_frames"),
            latent_length=int(input_latents.shape[2]),
        )
        clean_prefix = inputs.get("clean_prefix_latents")
        prefix_length = context_wan.resolve_num_clean_prefix_latents(
            clean_prefix_latents=clean_prefix,
            num_clean_prefix_latents=inputs.get("num_clean_prefix_latents"),
        )
        if context_indices:
            latents = pipe.scheduler.add_noise(
                input_latents,
                noise,
                timestep,
            )
            latents = context_wan.apply_clean_latents_at_indices(
                latents,
                input_latents,
                context_indices,
            )
        elif prefix_length > 0:
            latents = input_latents.clone()
            latents[:, :, prefix_length:] = pipe.scheduler.add_noise(
                input_latents[:, :, prefix_length:],
                noise[:, :, prefix_length:],
                timestep,
            )
            latents = context_wan.apply_clean_prefix_to_latents(
                latents,
                clean_prefix,
            )
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
            if "first_frame_latents" in inputs:
                latents[:, :, :1] = inputs["first_frame_latents"]
        inputs["latents"] = latents
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        model_output = pipe.model_fn(**models, **inputs, timestep=timestep)
        squared_error = (model_output.float() - training_target.float()).square()

        prediction_for_loss = model_output
        target_for_loss = training_target
        if context_indices:
            prediction_for_loss = context_wan.slice_non_context_latents(
                prediction_for_loss,
                latent_length=int(input_latents.shape[2]),
                context_latent_indices=context_indices,
            )
            target_for_loss = context_wan.slice_non_context_latents(
                target_for_loss,
                latent_length=int(input_latents.shape[2]),
                context_latent_indices=context_indices,
            )
        elif prefix_length > 0:
            prediction_for_loss = prediction_for_loss[:, :, prefix_length:]
            target_for_loss = target_for_loss[:, :, prefix_length:]
        elif "first_frame_latents" in inputs:
            prediction_for_loss = prediction_for_loss[:, :, 1:]
            target_for_loss = target_for_loss[:, :, 1:]
        training_weight = pipe.scheduler.training_weight(timestep)
        loss = F.mse_loss(
            prediction_for_loss.float(),
            target_for_loss.float(),
        ) * training_weight
        capture["main"] = {
            "flow_map": _flow_map_with_context_mask(
                squared_error,
                inputs,
                training_weight,
            ).detach().cpu(),
            "loss_main": float(loss.detach().item()),
        }
        return loss

    def collect_xssc_visuals(pred_x0_latents, target_x0_latents, **_kwargs):
        result = original_xssc_loss(
            pred_x0_latents,
            target_x0_latents,
            return_visuals=True,
        )
        capture["visuals"] = {
            key: value.detach().cpu()
            for key, value in result[2].items()
        }
        return result

    core.base.flow_match_context_sft_loss = controlled_flow_loss
    model._xssc_feature_loss = collect_xssc_visuals
    records = []
    gt_video = None
    try:
        for level_id, level in enumerate(levels):
            active_level = level
            capture.clear()
            with torch.no_grad():
                model(sample)
            main_capture = capture.get("main")
            visuals = capture.get("visuals")
            if not isinstance(main_capture, dict) or not isinstance(visuals, dict):
                raise RuntimeError("Forward did not capture both losses")
            if gt_video is None:
                gt_video = common._to_uint8_video(visuals["target_video"])
                common._write_mp4(output_dir / "gt.mp4", gt_video, args.fps)
            folder = f"level_{level_id:02d}_sigma_{float(level['sigma']):.4f}"
            level_dir = output_dir / folder
            level_dir.mkdir(parents=True)
            pred_video = common._to_uint8_video(visuals["pred_video"])
            common._write_mp4(level_dir / "pred_x0.mp4", pred_video, args.fps)
            xssc_map = _xssc_attention_projected_map(
                visuals,
                int(model.xssc_loss_future_start_frame),
            ).cpu()
            metrics = dict(model.last_train_metrics)
            records.append(
                {
                    **level,
                    "folder": folder,
                    "loss_main": float(main_capture["loss_main"]),
                    "loss_xssc": float(metrics["train/loss_xssc"]),
                    "loss_total": float(metrics["train/loss_total"]),
                    "valid_slot_fraction": float(
                        metrics["train/xssc_valid_slot_fraction"]
                    ),
                    "flow_map": main_capture["flow_map"],
                    "xssc_map": xssc_map,
                }
            )
            del pred_video, visuals
            torch.cuda.empty_cache()
    finally:
        core.base.flow_match_context_sft_loss = original_flow_loss
        model._xssc_feature_loss = original_xssc_loss

    if gt_video is None:
        raise RuntimeError("No GT video was produced")
    if not records:
        raise RuntimeError("No sigma records were produced")

    video_frames = int(gt_video.shape[0])
    flow_native_steps = int(records[0]["flow_map"].shape[1])
    xssc_native_steps = int(records[0]["xssc_map"].shape[1])
    temporal_upscale = int(model._tiny_vae.t_upscale)
    skip_trim = bool(
        model._tiny_vae.is_cogvideox and flow_native_steps % 2 == 0
    )
    frames_to_trim = 0 if skip_trim else int(model._tiny_vae.frames_to_trim)
    video_to_flow_latent = _causal_video_to_latent_indices(
        latent_steps=flow_native_steps,
        video_frames=video_frames,
        temporal_upscale=temporal_upscale,
        frames_to_trim=frames_to_trim,
    )
    flow_frame_groups = _latent_video_frame_groups(
        video_to_flow_latent,
        flow_native_steps,
    )
    flow_representative_frames = [group[-1] for group in flow_frame_groups]
    if xssc_native_steps != video_frames:
        raise ValueError(
            "xSSC heatmap does not use the decoded-video time axis: "
            f"{xssc_native_steps} xSSC steps versus {video_frames} frames"
        )
    flow_native_fps = max(1, int(round(args.fps / temporal_upscale)))
    timeline_metadata = {
        "flow_native_axis": "DiT latent steps",
        "flow_native_steps": flow_native_steps,
        "flow_native_fps": flow_native_fps,
        "flow_native_background_frames": flow_representative_frames,
        "flow_latent_to_video_frame_groups": flow_frame_groups,
        "xssc_native_axis": "Tiny-VAE decoded video frames",
        "xssc_native_steps": xssc_native_steps,
        "video_fps": int(args.fps),
        "tiny_vae_temporal_upscale": temporal_upscale,
        "tiny_vae_frames_to_trim": frames_to_trim,
        "video_frame_to_flow_latent": video_to_flow_latent.tolist(),
        "temporal_interpolation": False,
    }
    flow_scale = _global_p99([record["flow_map"] for record in records])
    xssc_scale = _global_p99([record["xssc_map"] for record in records])
    for record in records:
        level_dir = output_dir / record["folder"]
        flow_native_tensor = record.pop("flow_map")
        xssc_native_tensor = record.pop("xssc_map")
        flow_native_map = _upsample_heatmap_spatially(
            flow_native_tensor,
            height=int(gt_video.shape[1]),
            width=int(gt_video.shape[2]),
        )
        xssc_native_map = _upsample_heatmap_spatially(
            xssc_native_tensor,
            height=int(gt_video.shape[1]),
            width=int(gt_video.shape[2]),
        )
        flow_native_base = gt_video[flow_representative_frames]
        flow_native_labels = []
        for latent_id, group in enumerate(flow_frame_groups):
            frame_text = (
                f"frame {group[0]:02d}"
                if len(group) == 1
                else f"frames {group[0]:02d}-{group[-1]:02d}"
            )
            flow_native_labels.append(
                f"Flow latent {latent_id:02d}/{flow_native_steps - 1:02d} | {frame_text}"
            )
        flow_causal_map = flow_native_map[video_to_flow_latent]
        flow_causal_labels = [
            f"Video frame {frame_id:02d}/{video_frames - 1:02d} | "
            f"Flow latent {latent_id:02d}/{flow_native_steps - 1:02d} (held)"
            for frame_id, latent_id in enumerate(video_to_flow_latent.tolist())
        ]
        xssc_labels = [
            f"xSSC frame {frame_id:02d}/{video_frames - 1:02d}"
            for frame_id in range(video_frames)
        ]
        common._write_mp4(
            level_dir / "flow_loss_native_13step.mp4",
            _annotate_frames(
                _overlay_heatmap(flow_native_base, flow_native_map, flow_scale),
                flow_native_labels,
            ),
            flow_native_fps,
        )
        common._write_mp4(
            level_dir / "flow_loss_causal_49f.mp4",
            _annotate_frames(
                _overlay_heatmap(gt_video, flow_causal_map, flow_scale),
                flow_causal_labels,
            ),
            args.fps,
        )
        common._write_mp4(
            level_dir / "xssc_loss_native_49f.mp4",
            _annotate_frames(
                _overlay_heatmap(gt_video, xssc_native_map, xssc_scale),
                xssc_labels,
            ),
            args.fps,
        )
        np.savez_compressed(
            level_dir / "native_loss_maps.npz",
            flow=flow_native_tensor[0].numpy(),
            xssc=xssc_native_tensor[0].numpy(),
            video_frame_to_flow_latent=video_to_flow_latent,
            flow_representative_video_frames=np.asarray(
                flow_representative_frames,
                dtype=np.int64,
            ),
        )
        record["flow_native_shape"] = list(flow_native_tensor.shape)
        record["xssc_native_shape"] = list(xssc_native_tensor.shape)

    sample_metadata = common._jsonable_sample_metadata(sample)
    metadata = {
        "config": str(args.config.resolve()),
        "sample_index": sample_index,
        "sample": sample_metadata,
        "seed": int(args.seed),
        "same_noise_across_levels": True,
        "resolution": [int(args.height), int(args.width)],
        "frames": int(gt_video.shape[0]),
        "fps": int(args.fps),
        "timeline": timeline_metadata,
        "flow_global_p99": flow_scale,
        "xssc_global_p99": xssc_scale,
        "levels": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _build_page(
        output_dir,
        records,
        flow_scale=flow_scale,
        xssc_scale=xssc_scale,
        sample_metadata=sample_metadata,
        timeline_metadata=timeline_metadata,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
