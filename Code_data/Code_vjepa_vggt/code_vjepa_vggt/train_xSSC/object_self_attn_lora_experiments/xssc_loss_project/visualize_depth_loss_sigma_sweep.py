#!/usr/bin/env python3
"""Visualize a frozen Depth Anything loss over several training noise levels."""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_DIR.parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parents[2]
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
DEPTH_ANYTHING_ROOT = Path("/home/gaoya/MimicBrush-main/depthanything")
for _path in (
    PROJECT_DIR,
    EXPERIMENT_ROOT,
    TRAIN_XSSC_ROOT,
    PACKAGE_ROOT,
    DIFFSYNTH_ROOT,
    DEPTH_ANYTHING_ROOT,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cv2
import torch
import torch.nn.functional as F

from depth_anything.dpt import DepthAnything

import diagnose_xssc_loss as common
import train_xssc_context_slots as dataset_module
import train_xssc_object_self_attn_lora as core
from code_vjepa_vggt import context_wan_v_newtrain as context_wan
from vjepa_loss_project.train_xssc_object_self_attn_lora_vjepa_loss import (
    _load_tiny_vae,
)


DEFAULT_CONFIG = (
    PROJECT_DIR
    / "configs/full_sa_no_object_xssc_loss_dinov3_movic_step50000.json"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/full_sa_no_object_depth_loss_sigma_demo"
)
DEFAULT_DEPTH = Path("/data/gaoya/ckpt/LiheYoung-depth_anything_vitl14")
HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
PAGE_NAME = "full-sa-no-object-depth-loss-demo"
QUANTILE_MAX_SAMPLES = 4_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--depth-checkpoint", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future-start-frame", type=int, default=8)
    parser.add_argument("--depth-input-height", type=int, default=518)
    parser.add_argument("--depth-input-width", type=int, default=910)
    parser.add_argument("--depth-chunk-size", type=int, default=1)
    parser.add_argument(
        "--sigma-targets",
        default="0.05,0.20,0.40,0.60,0.80,0.95",
    )
    return parser.parse_args()


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        raise RuntimeError(f"Refusing to replace real directory: {link}")
    link.symlink_to(target)


def write_pending_page(output_dir: Path) -> None:
    (output_dir / "index.html").write_text(
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth-loss forward demo</title><style>
body{margin:0;background:#e8eef2;color:#173044;font:16px/1.6 system-ui,sans-serif}
main{max-width:900px;margin:15vh auto;padding:30px;background:white;border-left:7px solid #2f718e}
</style><main><h1>Depth-loss 多噪声前向 Demo</h1>
<p>GPU1 正在生成 Wan predicted x0、Depth Anything 相对深度和逐像素 loss overlay。完成后本页自动替换为结果。</p></main></html>""",
        encoding="utf-8",
    )


def scheduler_levels(scheduler, requested: list[float]) -> list[dict]:
    sigmas = scheduler.sigmas.detach().float().cpu().flatten()
    timesteps = scheduler.timesteps.detach().float().cpu().flatten()
    levels: list[dict] = []
    used: set[int] = set()
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
    return sorted(levels, key=lambda item: item["sigma"])


def restore_condition_latents(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    captured_inputs: dict,
) -> torch.Tensor:
    context_indices = context_wan.resolve_context_latent_indices_from_frames(
        raw_frame_indices=captured_inputs.get("context_frame_indices"),
        raw_num_frames=captured_inputs.get("num_frames"),
        latent_length=int(pred_x0.shape[2]),
    )
    if context_indices:
        return context_wan.apply_clean_latents_at_indices(
            pred_x0,
            target_x0,
            context_indices,
        )
    prefix_length = context_wan.resolve_num_clean_prefix_latents(
        clean_prefix_latents=captured_inputs.get("clean_prefix_latents"),
        num_clean_prefix_latents=captured_inputs.get("num_clean_prefix_latents"),
    )
    if prefix_length > 0:
        pred_x0 = pred_x0.clone()
        pred_x0[:, :, :prefix_length] = target_x0[:, :, :prefix_length]
    elif "first_frame_latents" in captured_inputs:
        pred_x0 = pred_x0.clone()
        pred_x0[:, :, :1] = target_x0[:, :, :1]
    return pred_x0


def decode_tiny_vae(
    latents: torch.Tensor,
    tiny_vae,
    tiny_vae_apply,
) -> torch.Tensor:
    latent_ntchw = latents.permute(0, 2, 1, 3, 4).contiguous()
    with torch.autocast(
        device_type=latent_ntchw.device.type,
        dtype=latents.dtype,
        enabled=latent_ntchw.device.type == "cuda",
    ):
        video = tiny_vae_apply(tiny_vae.decoder, latent_ntchw, False, False)
        if tiny_vae.patch_size > 1:
            video = F.pixel_shuffle(video, tiny_vae.patch_size)
    skip_trim = tiny_vae.is_cogvideox and latent_ntchw.shape[1] % 2 == 0
    if not skip_trim:
        video = video[:, tiny_vae.frames_to_trim :]
    return video.clamp(0.0, 1.0)


def load_depth_model(path: Path, device: torch.device) -> DepthAnything:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = DepthAnything(config)
    state = torch.load(
        path / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False).eval()
    return model.to(device=device, dtype=torch.bfloat16)


@torch.no_grad()
def infer_depth(
    model: DepthAnything,
    video: torch.Tensor,
    *,
    device: torch.device,
    input_height: int,
    input_width: int,
    chunk_size: int,
) -> torch.Tensor:
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[2] != 3:
        raise ValueError(f"Expected [1,T,3,H,W], got {tuple(video.shape)}")
    time_steps, out_height, out_width = (
        int(video.shape[1]),
        int(video.shape[3]),
        int(video.shape[4]),
    )
    flat = video[0].float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output: list[torch.Tensor] = []
    for start in range(0, time_steps, chunk_size):
        frames = flat[start : start + chunk_size]
        frames = F.interpolate(
            frames,
            size=(input_height, input_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        frames = ((frames - mean) / std).to(
            device=device,
            dtype=torch.bfloat16,
        )
        depth = model(frames).float().unsqueeze(1)
        depth = F.interpolate(
            depth,
            size=(out_height, out_width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        output.append(depth.cpu())
    return torch.cat(output, dim=0).view(1, time_steps, out_height, out_width)


def robust_relative_depth(depth: torch.Tensor) -> torch.Tensor:
    flat = depth.float().flatten(2)
    median = flat.median(dim=-1).values[:, :, None]
    mad = (flat - median).abs().median(dim=-1).values[:, :, None]
    normalized = (flat - median) / (1.4826 * mad + 1e-6)
    return normalized.clamp(-5.0, 5.0).view_as(depth)


def sampled_quantile(
    values: torch.Tensor,
    quantile: float,
    *,
    max_samples: int = QUANTILE_MAX_SAMPLES,
    positive_only: bool = False,
) -> float:
    flat = values.detach().float().reshape(-1)
    if positive_only:
        flat = flat[flat > 0]
    if flat.numel() == 0:
        return 0.0
    stride = max(1, math.ceil(flat.numel() / max_samples))
    sampled = flat[::stride]
    return float(torch.quantile(sampled, quantile).item())


def sampled_global_quantile(
    maps: list[torch.Tensor],
    quantile: float,
    *,
    max_samples: int = QUANTILE_MAX_SAMPLES,
) -> float:
    if not maps:
        return 0.0
    per_map_budget = max(1, max_samples // len(maps))
    pieces: list[torch.Tensor] = []
    for loss_map in maps:
        positive = loss_map.detach().float().reshape(-1)
        positive = positive[positive > 0]
        if positive.numel() == 0:
            continue
        stride = max(1, math.ceil(positive.numel() / per_map_budget))
        pieces.append(positive[::stride])
    if not pieces:
        return 0.0
    return float(torch.quantile(torch.cat(pieces), quantile).item())


def depth_color(depth: torch.Tensor) -> np.ndarray:
    values = ((depth[0].float().clamp(-3.0, 3.0) + 3.0) / 6.0).numpy()
    frames = []
    for frame in values:
        color = cv2.applyColorMap(
            np.rint(frame * 255.0).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        frames.append(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def loss_color(loss_map: torch.Tensor, scale: float) -> np.ndarray:
    values = np.clip(loss_map[0].float().numpy() / max(scale, 1e-12), 0.0, 1.0)
    frames = []
    for frame in values:
        color = cv2.applyColorMap(
            np.rint(frame * 255.0).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        frames.append(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def overlay_loss(base: np.ndarray, loss_map: torch.Tensor, scale: float) -> np.ndarray:
    values = np.clip(loss_map[0].float().numpy() / max(scale, 1e-12), 0.0, 1.0)
    colors = loss_color(loss_map, scale).astype(np.float32)
    mask = values[..., None]
    mixed = base.astype(np.float32) * (1.0 - 0.58 * mask) + colors * (0.58 * mask)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def annotate(frames: np.ndarray, labels: list[str]) -> np.ndarray:
    output = frames.copy()
    for frame, label in zip(output, labels):
        cv2.putText(frame, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def labeled_panel(
    frames: np.ndarray,
    label: str,
    *,
    width: int = 448,
    height: int = 256,
) -> np.ndarray:
    output = []
    for frame in frames:
        panel = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        panel = panel.copy()
        cv2.rectangle(panel, (0, 0), (width, 34), (20, 35, 47), -1)
        cv2.putText(
            panel,
            label,
            (11, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        output.append(panel)
    return np.stack(output)


def composite_grid(
    *,
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    gt_depth_rgb: np.ndarray,
    pred_depth_rgb: np.ndarray,
    loss_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
) -> np.ndarray:
    panels = [
        labeled_panel(gt_rgb, "GT RGB"),
        labeled_panel(pred_rgb, "Pred x0 RGB"),
        labeled_panel(overlay_rgb, "Depth-loss overlay on Pred x0"),
        labeled_panel(gt_depth_rgb, "GT relative disparity"),
        labeled_panel(pred_depth_rgb, "Pred relative disparity"),
        labeled_panel(loss_rgb, "Depth-loss heatmap"),
    ]
    frame_counts = {int(panel.shape[0]) for panel in panels}
    if len(frame_counts) != 1:
        raise ValueError(f"Composite inputs have unequal frame counts: {frame_counts}")
    top = np.concatenate(panels[:3], axis=2)
    bottom = np.concatenate(panels[3:], axis=2)
    return np.concatenate((top, bottom), axis=1)


def build_page(output_dir: Path, records: list[dict], sample: dict, scale: float) -> None:
    sections: list[str] = []
    for record in records:
        folder = html.escape(record["folder"])
        sections.append(
            f'''<section class="level">
<header><div><span>NOISE LEVEL {record['scheduler_index']:03d}</span><h2>σ={record['sigma']:.4f} · timestep={record['timestep']:.1f}</h2></div>
<div class="metrics"><b>Flow {record['loss_main']:.5f}</b><b>Depth {record['loss_depth']:.5f}</b><b>Depth sampled P95 {record['loss_depth_p95']:.5f}</b><b>Near/Far floor {record['loss_near_far_floor_ratio']:.2f}×</b></div></header>
<div class="grid">
<figure class="composite"><figcaption>逐帧同步拼接：GT RGB ｜ Pred x0 ｜ loss overlay<br>GT relative disparity ｜ Pred relative disparity ｜ loss heatmap</figcaption><video controls muted loop playsinline preload="metadata" src="{folder}/composite_overlay.mp4"></video></figure>
</div></section>'''
        )
    prompt = html.escape(str(sample.get("prompt", sample.get("caption", ""))))
    metadata = html.escape(json.dumps(common._jsonable_sample_metadata(sample), ensure_ascii=False))
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full-SA No-Object depth-loss demo</title><style>
:root{{--paper:#e7eef2;--ink:#183247;--muted:#617789;--line:#b9cad4;--blue:#2f718e;--orange:#d87832;--panel:#f8fbfc}}
*{{box-sizing:border-box}}html{{background:var(--paper);color:var(--ink);font-family:"Avenir Next","Segoe UI",sans-serif}}body{{margin:0}}.mast{{padding:28px max(24px,4vw) 22px;background:#d6e4eb;border-bottom:1px solid #a8bdc8}}.eyebrow{{font:800 11px ui-monospace,monospace;letter-spacing:.15em;color:var(--blue)}}h1{{margin:8px 0 11px;font:600 clamp(28px,4vw,48px)/1.05 Georgia,serif}}.mast p{{max-width:1250px;margin:5px 0;color:var(--muted);line-height:1.55}}.prompt{{margin-top:14px;padding:10px 13px;background:#eef4f6;border-left:5px solid var(--blue);color:#314e60}}main{{padding:28px max(24px,4vw) 100px}}.level{{padding:20px;background:rgba(255,255,255,.48);border:1px solid var(--line);box-shadow:0 8px 24px #294c6112}}.level+.level{{margin-top:28px}}.level header{{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:15px;border-bottom:1px solid var(--line);padding-bottom:11px}}.level header span{{font:800 10px ui-monospace,monospace;color:var(--orange);letter-spacing:.13em}}h2{{font:600 21px Georgia,serif;margin:5px 0 0}}.metrics{{display:flex;gap:7px;flex-wrap:wrap}}.metrics b{{padding:6px 8px;background:var(--ink);color:white;font:700 10px ui-monospace,monospace}}.grid{{display:block}}figure{{margin:0;background:var(--panel);border:1px solid var(--line)}}figcaption{{min-height:43px;padding:9px 10px;font:700 11px/1.35 ui-monospace,monospace;border-left:5px solid var(--blue)}}video{{display:block;width:100%;background:#122432}}.composite video{{aspect-ratio:1344/512;object-fit:contain}}#replay{{position:fixed;right:22px;bottom:22px;z-index:5;border:0;background:var(--orange);color:white;padding:12px 17px;font-weight:800;box-shadow:0 7px 20px #18324755;cursor:pointer}}code{{overflow-wrap:anywhere}}@media(max-width:650px){{.level header{{align-items:start;flex-direction:column}}}}
</style></head><body><header class="mast"><div class="eyebrow">FROZEN DEPTH ANYTHING VIT-L/14 · FORWARD ONLY</div><h1>Full-SA + No-Object<br>Depth Loss 多噪声前向 Demo</h1><p>同一训练样本、同一 Gaussian noise realization，只改变 scheduler σ。Wan2.2 + merged OpenVid LoRA 的 step-0 Full-SA 初始化；TinyVAE 解码 pred x0 与 GT，冻结 Depth Anything 输出逐帧相对 disparity-like depth。它不是米制 z-depth；近处通常具有更大的相对视差响应。</p><p>Loss = median/MAD 尺度与平移归一化后的 Charbonnier error，仅计算未来帧 8–48。所有 loss 图共享 sampled P99={scale:.6g}（最多 {QUANTILE_MAX_SAMPLES:,} 个确定性等距样本）；帧 0–7 黑色表示条件区间，不参与 loss。Near/Far floor 是底部 22% 与画面 y=35%–55% 两个诊断带的平均 loss 比值。</p><div class="prompt"><b>Prompt：</b> {prompt}</div></header><main>{''.join(sections)}<p><code>{metadata}</code></p></main><button id="replay">整页重新播放</button><script>document.getElementById('replay').onclick=()=>document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play().catch(()=>{{}})}});</script></body></html>'''
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    write_pending_page(output_dir)
    replace_symlink(HUB_ROOT / PAGE_NAME, output_dir)
    requested = [float(value) for value in args.sigma_targets.split(",") if value.strip()]
    if not requested:
        raise ValueError("At least one sigma target is required")
    if args.future_start_frame < 0 or args.future_start_frame >= 49:
        raise ValueError("future-start-frame must be in [0, 48]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_args, resolved = common._parse_trainer_args(
        args.config,
        args.width,
        args.height,
        "lowest",
    )
    train_args.experiment_seed = int(args.seed)
    dataset = dataset_module.build_dataset(train_args)
    sample_index = int(args.sample_index) % len(dataset)
    sample = dataset[sample_index]
    device = torch.device("cuda:0")
    accelerator = common._Accelerator(device)
    model = core.build_model(train_args, accelerator)
    model.train()
    levels = scheduler_levels(model.pipe.scheduler, requested)
    tiny_config = resolved["xssc_loss"]
    tiny_vae, tiny_apply = _load_tiny_vae(
        tiny_config["tiny_vae_root"],
        tiny_config["tiny_vae_checkpoint"],
        device,
        model.pipe.torch_dtype,
    )

    capture: dict[str, object] = {}
    fixed_noise: torch.Tensor | None = None
    active_level: dict | None = None
    original_flow_loss = core.base.flow_match_context_sft_loss
    original_task_loss = model.task_to_loss[model.task]

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
        target = pipe.scheduler.training_target(input_latents, noise, timestep)
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
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
            latents = context_wan.apply_clean_latents_at_indices(
                latents, input_latents, context_indices
            )
        elif prefix_length > 0:
            latents = input_latents.clone()
            latents[:, :, prefix_length:] = pipe.scheduler.add_noise(
                input_latents[:, :, prefix_length:],
                noise[:, :, prefix_length:],
                timestep,
            )
            latents = context_wan.apply_clean_prefix_to_latents(latents, clean_prefix)
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
            if "first_frame_latents" in inputs:
                latents[:, :, :1] = inputs["first_frame_latents"]
        inputs["latents"] = latents
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        model_output = pipe.model_fn(**models, **inputs, timestep=timestep)
        prediction_for_loss = model_output
        target_for_loss = target
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
        main_loss = F.mse_loss(
            prediction_for_loss.float(), target_for_loss.float()
        ) * training_weight
        sigma = context_wan._diffsynth_sigma_for_timestep(
            pipe.scheduler, timestep
        ).to(device=latents.device, dtype=latents.dtype)
        while sigma.ndim < latents.ndim:
            sigma = sigma.unsqueeze(-1)
        pred_x0 = restore_condition_latents(
            latents - sigma * model_output,
            input_latents,
            inputs,
        )
        capture["pred_x0"] = pred_x0
        capture["target_x0"] = input_latents
        capture["loss_main"] = float(main_loss.item())
        return main_loss

    core.base.flow_match_context_sft_loss = controlled_flow_loss
    model.task_to_loss[model.task] = (
        lambda pipe, inputs_shared, inputs_posi, inputs_nega: controlled_flow_loss(
            pipe,
            **inputs_shared,
            **inputs_posi,
        )
    )
    records: list[dict] = []
    gt_video: torch.Tensor | None = None
    try:
        for level_id, level in enumerate(levels):
            active_level = level
            capture.clear()
            with torch.no_grad():
                model(sample)
                pred_video = decode_tiny_vae(
                    capture["pred_x0"], tiny_vae, tiny_apply
                ).cpu().to(torch.float16)
                if gt_video is None:
                    gt_video = decode_tiny_vae(
                        capture["target_x0"], tiny_vae, tiny_apply
                    ).cpu().to(torch.float16)
            folder = f"level_{level_id:02d}_sigma_{level['sigma']:.4f}"
            level_dir = output_dir / folder
            level_dir.mkdir(parents=True)
            pred_rgb = common._to_uint8_video(pred_video)
            common._write_mp4(level_dir / "pred_x0.mp4", pred_rgb, args.fps)
            records.append(
                {
                    **level,
                    "folder": folder,
                    "loss_main": float(capture["loss_main"]),
                    "pred_video": pred_video,
                    "pred_rgb": pred_rgb,
                }
            )
            print(
                f"[depth-demo] Wan forward {level_id + 1}/{len(levels)} "
                f"sigma={level['sigma']:.4f}",
                flush=True,
            )
            torch.cuda.empty_cache()
    finally:
        core.base.flow_match_context_sft_loss = original_flow_loss
        model.task_to_loss[model.task] = original_task_loss
    if gt_video is None:
        raise RuntimeError("GT TinyVAE video was not produced")
    gt_rgb = common._to_uint8_video(gt_video)
    common._write_mp4(output_dir / "gt.mp4", gt_rgb, args.fps)

    del model, tiny_vae, tiny_apply, capture, fixed_noise, dataset
    gc.collect()
    torch.cuda.empty_cache()
    depth_model = load_depth_model(args.depth_checkpoint.resolve(), device)
    gt_depth_raw = infer_depth(
        depth_model,
        gt_video,
        device=device,
        input_height=args.depth_input_height,
        input_width=args.depth_input_width,
        chunk_size=args.depth_chunk_size,
    )
    gt_depth = robust_relative_depth(gt_depth_raw)
    gt_depth_rgb = depth_color(gt_depth)
    common._write_mp4(output_dir / "gt_depth.mp4", gt_depth_rgb, args.fps)
    all_loss_maps: list[torch.Tensor] = []
    for index, record in enumerate(records):
        pred_depth_raw = infer_depth(
            depth_model,
            record["pred_video"],
            device=device,
            input_height=args.depth_input_height,
            input_width=args.depth_input_width,
            chunk_size=args.depth_chunk_size,
        )
        pred_depth = robust_relative_depth(pred_depth_raw)
        difference = pred_depth - gt_depth
        loss_map = torch.sqrt(difference.square() + 1e-6) - 1e-3
        loss_map[:, : args.future_start_frame] = 0.0
        future = loss_map[:, args.future_start_frame :]
        record["loss_depth"] = float(future.mean().item())
        record["loss_depth_p95"] = sampled_quantile(future, 0.95)
        height = int(loss_map.shape[-2])
        far_floor = loss_map[:, args.future_start_frame :, int(0.35 * height) : int(0.55 * height)]
        near_floor = loss_map[:, args.future_start_frame :, int(0.78 * height) :]
        record["loss_far_floor"] = float(far_floor.mean().item())
        record["loss_near_floor"] = float(near_floor.mean().item())
        record["loss_near_far_floor_ratio"] = float(
            record["loss_near_floor"] / max(record["loss_far_floor"], 1e-12)
        )
        record["pred_depth"] = pred_depth
        record["loss_map"] = loss_map
        all_loss_maps.append(loss_map)
        print(
            f"[depth-demo] Depth forward {index + 1}/{len(records)} "
            f"sigma={record['sigma']:.4f} loss={record['loss_depth']:.6f}",
            flush=True,
        )
    del depth_model, gt_depth_raw
    gc.collect()
    torch.cuda.empty_cache()

    loss_scale = sampled_global_quantile(all_loss_maps, 0.99)
    loss_scale = max(loss_scale, 1e-12)
    for record in records:
        level_dir = output_dir / record["folder"]
        labels = [
            f"frame {frame:02d}/48 | sigma {record['sigma']:.4f} | "
            + ("context excluded" if frame < args.future_start_frame else "future depth loss")
            for frame in range(49)
        ]
        pred_depth_rgb = depth_color(record.pop("pred_depth"))
        loss_map = record.pop("loss_map")
        loss_rgb = annotate(loss_color(loss_map, loss_scale), labels)
        overlay_rgb = annotate(
            overlay_loss(record["pred_rgb"], loss_map, loss_scale), labels
        )
        common._write_mp4(level_dir / "pred_depth.mp4", pred_depth_rgb, args.fps)
        common._write_mp4(level_dir / "depth_loss_map.mp4", loss_rgb, args.fps)
        common._write_mp4(level_dir / "depth_loss_overlay.mp4", overlay_rgb, args.fps)
        composite_rgb = composite_grid(
            gt_rgb=gt_rgb,
            pred_rgb=record["pred_rgb"],
            gt_depth_rgb=gt_depth_rgb,
            pred_depth_rgb=pred_depth_rgb,
            loss_rgb=loss_rgb,
            overlay_rgb=overlay_rgb,
        )
        common._write_mp4(
            level_dir / "composite_overlay.mp4",
            composite_rgb,
            args.fps,
        )
        np.savez_compressed(
            level_dir / "depth_loss_maps.npz",
            depth_loss=loss_map[0].numpy(),
            future_start_frame=np.asarray(args.future_start_frame),
        )
        record.pop("pred_video")
        record.pop("pred_rgb")

    sample_metadata = common._jsonable_sample_metadata(sample)
    metadata = {
        "config": str(args.config.resolve()),
        "initialization": "Wan2.2 TI2V 5B + merged OpenVid rank-32 LoRA; Full-SA step-0",
        "depth_checkpoint": str(args.depth_checkpoint.resolve()),
        "depth_model": "Depth Anything V1 ViT-L/14, frozen",
        "depth_loss": "per-frame median/MAD normalized Charbonnier",
        "future_frames": [args.future_start_frame, 48],
        "floor_region_analysis": {
            "far_floor_y_fraction": [0.35, 0.55],
            "near_floor_y_fraction": [0.78, 1.0],
        },
        "same_noise_across_levels": True,
        "sample_index": sample_index,
        "sample": sample_metadata,
        "seed": args.seed,
        "resolution": [args.height, args.width],
        "frames": 49,
        "fps": args.fps,
        "loss_global_p99": loss_scale,
        "loss_quantile_sampling": {
            "method": "deterministic_flat_stride",
            "max_samples": QUANTILE_MAX_SAMPLES,
        },
        "levels": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_page(output_dir, records, sample_metadata, loss_scale)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
