from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from code_vjepa_vggt.infer_context_video_wan import (
    _build_cond_context,
    _ensure_browser_video,
    _infer_context_indices,
    _infer_object_pooler_latent_dim,
    _load_trainable_state,
    _load_trainable_state_into_model,
    _resolve_launch_device,
    _run_sampling,
    _select_video_from_path,
    _tensor_stats,
    _write_mp4,
)
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


def _apply_stage2_inference_checkpoint_chain(
    config: dict,
    *,
    stage1_checkpoint: Path,
    stage2_checkpoint: Path,
) -> dict[str, object]:
    stage2_state = _load_trainable_state(stage2_checkpoint)
    object_pooler_latent_dim = _infer_object_pooler_latent_dim(
        stage2_state,
        int(config["model"].get("object_pooler_latent_dim", 16)),
    )
    config["model"]["object_pooler_latent_dim"] = int(object_pooler_latent_dim)
    # Match the stage2 training path: stage1 provides the Wan/LoRA initialization,
    # then the stage2 trainable-state checkpoint restores adapter weights.
    config["model"]["init_wan_lora_from_checkpoint"] = str(stage1_checkpoint)
    return {
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "object_pooler_latent_dim": int(object_pooler_latent_dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference for stage2 adapter checkpoints saved as step_*.pt trainable-state files."
    )
    parser.add_argument(
        "--stage2-checkpoint",
        required=True,
        help="stage2 trainable-state checkpoint, typically step_XXXXXXX.pt",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        required=True,
        help="base checkpoint used to initialize stage2 training, typically a Wan LoRA .safetensors file",
    )
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_stage2_adapters_gpu67.yaml",
        help="stage2 training config used to construct the inference model",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--context-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--context-fraction", type=float, default=None)
    parser.add_argument("--random-context-frames", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    print("[stage2_infer] load config", flush=True)
    config = load_yaml_config(args.config)
    stage1_checkpoint = Path(args.stage1_checkpoint).expanduser().resolve()
    stage2_checkpoint = Path(args.stage2_checkpoint).expanduser().resolve()
    print("[stage2_infer] resolve checkpoint chain", flush=True)
    checkpoint_chain = _apply_stage2_inference_checkpoint_chain(
        config,
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint,
    )

    data_cfg = config["data"]
    default_context_fraction = float(data_cfg.get("context_fraction", 0.5))
    default_random_context_frames = bool(data_cfg.get("random_context_frames", False))
    context_fraction = default_context_fraction if args.context_fraction is None else float(args.context_fraction)
    random_context_frames = (
        default_random_context_frames
        if args.random_context_frames is None
        else bool(args.random_context_frames)
    )

    print("[stage2_infer] resolve device", flush=True)
    device = _resolve_launch_device()
    device_obj = torch.device(device)

    video_path = Path(args.context_video).expanduser().resolve()
    print("[stage2_infer] load input video", flush=True)
    frames, frame_indices = _select_video_from_path(video_path, args.num_frames, args.sampling_mode)
    video = preprocess_video_rgb_uint8(frames, tuple(config["data"]["resolution"]))
    context_indices = _infer_context_indices(
        total_frames=video.shape[1],
        num_context_frames=int(config["data"]["num_context_frames"]),
        context_fraction=context_fraction,
        random_context_frames=random_context_frames,
        seed=int(args.seed),
    )
    context_video = video[:, context_indices].contiguous().unsqueeze(0)
    num_context_frames = torch.tensor([context_video.shape[2]], dtype=torch.long)

    print("[stage2_infer] construct trainer", flush=True)
    trainer = ContextVideoTrainer(config, build_optimizer=True, device=device)
    trainer.build_optimizer = False
    print("[stage2_infer] load stage2 trainable state", flush=True)
    load_info = _load_trainable_state_into_model(trainer, stage2_checkpoint)
    print("[stage2_infer] trainer ready", flush=True)
    if trainer.bundle.dit is not None:
        trainer.bundle.dit.eval()

    print("[stage2_infer] build conditioning context", flush=True)
    fused_context, context_latents, prep_debug = _build_cond_context(
        trainer=trainer,
        config=config,
        context_video=context_video.to(device_obj),
        captions=[args.prompt],
        num_context_frames=num_context_frames,
        device_obj=device_obj,
    )
    print("[stage2_infer] start sampling", flush=True)
    with torch.inference_mode():
        pred, sample_debug = _run_sampling(
            bundle=trainer.bundle,
            fused_context=fused_context,
            context_latents=context_latents,
            total_frames=int(video.shape[1]),
            num_context_frames=int(num_context_frames.item()),
            num_inference_steps=int(args.sampling_steps),
        )
    print("[stage2_infer] sampling done", flush=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "config": str(Path(args.config).expanduser().resolve()),
        "context_video": str(video_path),
        "prompt": str(args.prompt),
        "frame_indices": frame_indices.tolist(),
        "context_indices": context_indices.tolist(),
        "prep_debug": prep_debug,
        "sample_debug": sample_debug,
        "load_info": load_info,
        "checkpoint_chain": checkpoint_chain,
        "inference_settings": {
            "num_frames": int(args.num_frames),
            "sampling_mode": args.sampling_mode,
            "context_fraction": float(context_fraction),
            "random_context_frames": bool(random_context_frames),
            "sampling_steps": int(args.sampling_steps),
            "fps": int(args.fps),
            "seed": int(args.seed),
            "resolution": [int(config["data"]["resolution"][0]), int(config["data"]["resolution"][1])],
            "configured_num_context_frames": int(config["data"]["num_context_frames"]),
            "used_num_context_frames": int(num_context_frames.item()),
        },
        "trainable_tensor_stats": {
            "fused_context": _tensor_stats("fused_context", fused_context),
            "context_latents": _tensor_stats("context_latents", context_latents),
        },
    }

    context_out = context_video[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
    context_out = ((context_out.clip(-1.0, 1.0) + 1.0) * 127.5).round().astype("uint8")
    context_path = output_dir / "context_used.mp4"
    _write_mp4(context_path, context_out, fps=int(args.fps))
    result["context_used_video"] = str(_ensure_browser_video(context_path))

    if trainer.bundle.dit is not None:
        del trainer.bundle.dit
        trainer.bundle.dit = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_raw:
        print("[stage2_infer] decode prediction", flush=True)
        with torch.no_grad():
            decode_input = pred.to(
                next(trainer.bundle.vae.model.parameters()).device
                if hasattr(trainer.bundle.vae, "model")
                else device_obj
            )
            decoded = trainer.bundle.vae.decode([decode_input])
        if isinstance(decoded, list):
            decoded = decoded[0]
        video_out = decoded.detach().cpu()
        video_out = video_out.permute(1, 0, 2, 3).contiguous()
        video_out = ((video_out.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        raw_path = output_dir / "prediction.mp4"
        _write_mp4(raw_path, video_out, fps=int(args.fps))
        result["prediction_video"] = str(_ensure_browser_video(raw_path))
        print("[stage2_infer] saved prediction video", flush=True)

    with open(output_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
