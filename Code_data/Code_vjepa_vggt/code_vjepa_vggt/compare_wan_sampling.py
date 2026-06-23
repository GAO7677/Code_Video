from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from code_vjepa_vggt.infer_context_video_wan import (
    _build_cond_context,
    _load_trainable_state_into_model,
    _resolve_launch_device,
    _run_sampling,
    _select_video_from_path,
    _video_bcthw_to_uint8_thwc,
    _write_mp4,
)
from code_vjepa_vggt.models.wan_context_model import WanContextVideoModel
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


def _run_old_sampling(
    *,
    bundle: WanContextVideoModel,
    text_context: torch.Tensor,
    object_context: torch.Tensor,
    context_latents: torch.Tensor,
    total_frames: int,
    num_context_frames: int,
    num_inference_steps: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    from code_vjepa_vggt.infer_context_video_wan import (
        _print_tensor_stats,
        broadcast_latent_mask,
        expand_context_latents_to_full,
        latent_frame_mask,
    )
    from code_vjepa_vggt.training.flow_match import WanFlowMatchScheduler

    assert bundle.dit is not None
    bundle.dit.eval()
    dit_param = next(bundle.dit.parameters())
    dit_dtype = dit_param.dtype
    dit_device = dit_param.device
    context_latents = context_latents.to(device=dit_device, dtype=dit_dtype)
    latent_h = int(context_latents.shape[2])
    latent_w = int(context_latents.shape[3])
    total_lat_t = max(1, (int(total_frames) - 1) // bundle.config.vae_stride[0] + 1)
    latent_clean = torch.zeros(
        int(context_latents.shape[0]),
        total_lat_t,
        latent_h,
        latent_w,
        device=dit_device,
        dtype=dit_dtype,
    )
    copy_t = min(int(context_latents.shape[1]), total_lat_t)
    latent_clean[:, :copy_t] = context_latents[:, :copy_t]
    noise = torch.randn_like(latent_clean)
    scheduler = WanFlowMatchScheduler(num_train_timesteps=int(bundle.config.num_train_timesteps))
    scheduler.set_timesteps(num_inference_steps, training=False)
    sigma_0 = scheduler.sigmas[0].to(device=dit_device, dtype=dit_dtype)
    x_t = (1.0 - sigma_0) * latent_clean + sigma_0 * noise
    context_mask_t, future_mask_t = latent_frame_mask(
        num_video_frames=total_frames,
        num_context_frames=int(num_context_frames),
        vae_stride_t=bundle.config.vae_stride[0],
        device=dit_device,
    )
    context_mask = broadcast_latent_mask(context_mask_t, latent_clean)
    future_mask = broadcast_latent_mask(future_mask_t, latent_clean)
    context_clean_full = expand_context_latents_to_full(context_latents, latent_clean)
    x_t = context_mask * context_clean_full + (1.0 - context_mask) * x_t
    seq_len = x_t.shape[1] * x_t.shape[2] * x_t.shape[3] // (bundle.config.patch_size[1] * bundle.config.patch_size[2])
    text_context = text_context.to(device=dit_device, dtype=dit_dtype)
    object_context = object_context.to(device=dit_device, dtype=dit_dtype)
    for step_idx, sigma in enumerate(scheduler.sigmas):
        timestep = scheduler.timesteps[step_idx].to(device=dit_device, dtype=dit_dtype)
        t_tokens = torch.full((1, seq_len), float(timestep.item()), device=dit_device, dtype=dit_dtype)
        pred = bundle.dit(
            [x_t],
            t=t_tokens,
            context=None,
            text_context=[text_context],
            object_context=[object_context],
            seq_len=seq_len,
            y=None,
        )[0]
        next_sigma = scheduler.sigmas[step_idx + 1] if step_idx + 1 < len(scheduler.sigmas) else torch.tensor(0.0)
        next_sigma = next_sigma.to(device=dit_device, dtype=dit_dtype)
        sigma = sigma.to(device=dit_device, dtype=dit_dtype)
        x_t = x_t + (next_sigma - sigma) * pred
        x_t = context_mask * context_clean_full + (1.0 - context_mask) * x_t
    target = latent_clean
    loss = ((x_t - target) ** 2 * future_mask).mean()
    debug = {"loss": float(loss.item()), "scheduler": type(scheduler).__name__, "mode": "old_manual_euler"}
    return x_t.detach(), debug


def _resolve_input_videos(
    video_path: Path,
    *,
    num_frames: int,
    resolution: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    frames, _ = _select_video_from_path(video_path, num_frames, "prefix")
    video = preprocess_video_rgb_uint8(frames, resolution)
    context_len = min(12, int(video.shape[1]))
    context_video = video[:, :context_len].contiguous().unsqueeze(0)
    return video, context_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--context-video", required=True)
    parser.add_argument("--config", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml")
    parser.add_argument("--output-dir", default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/compare_wan_sampling")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = _resolve_launch_device()
    device_obj = torch.device(device)
    checkpoint_path = Path(args.checkpoint_dir)
    if checkpoint_path.is_file():
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint_state["model"] if isinstance(checkpoint_state, dict) and "model" in checkpoint_state else checkpoint_state
        object_pooler_latent_dim = int(model_state["bundle.object_pooler.latent_proj.weight"].shape[1])
        config["model"]["object_pooler_latent_dim"] = object_pooler_latent_dim

    trainer = ContextVideoTrainer(config, build_optimizer=True, device=device)
    trainer.build_optimizer = False
    _load_trainable_state_into_model(trainer, Path(args.checkpoint_dir))

    video_path = Path(args.context_video)
    video, context_video = _resolve_input_videos(
        video_path,
        num_frames=int(args.num_frames),
        resolution=tuple(config["data"]["resolution"]),
    )
    num_context_frames = torch.tensor([int(context_video.shape[2])], dtype=torch.long, device=device_obj)

    text_context, object_context, context_latents, prep_debug = _build_cond_context(
        trainer=trainer,
        config=config,
        context_video=context_video.to(device_obj),
        captions=[args.prompt],
        num_context_frames=num_context_frames,
        device_obj=device_obj,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"{Path(args.context_video).stem}_source.mp4"
    _write_mp4(source_path, _video_bcthw_to_uint8_thwc(video.unsqueeze(0)), fps=int(args.fps))

    with torch.inference_mode():
        new_pred, new_debug = _run_sampling(
            bundle=trainer.bundle,
            text_context=text_context,
            object_context=object_context,
            context_latents=context_latents,
            total_frames=int(video.shape[1]),
            num_context_frames=int(num_context_frames.item()),
            num_inference_steps=int(args.sampling_steps),
        )
        old_pred, old_debug = _run_old_sampling(
            bundle=trainer.bundle,
            text_context=text_context,
            object_context=object_context,
            context_latents=context_latents,
            total_frames=int(video.shape[1]),
            num_context_frames=int(num_context_frames.item()),
            num_inference_steps=int(args.sampling_steps),
        )

    new_path = output_dir / f"{Path(args.context_video).stem}_new_scheduler.mp4"
    old_path = output_dir / f"{Path(args.context_video).stem}_old_manual.mp4"
    _write_mp4(new_path, _video_bcthw_to_uint8_thwc(new_pred.unsqueeze(0)), fps=int(args.fps))
    _write_mp4(old_path, _video_bcthw_to_uint8_thwc(old_pred.unsqueeze(0)), fps=int(args.fps))

    report = {
        "prompt": args.prompt,
        "checkpoint_dir": str(args.checkpoint_dir),
        "input_video": str(args.context_video),
        "prep_debug": prep_debug,
        "new_debug": new_debug,
        "old_debug": old_debug,
        "outputs": {
            "source": str(source_path),
            "new_scheduler": str(new_path),
            "old_manual": str(old_path),
        },
    }
    with open(output_dir / f"{Path(args.context_video).stem}_compare.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
