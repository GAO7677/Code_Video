from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from code_vjepa_vggt.infer_context_video_wan import (
    _build_cond_context,
    _ensure_browser_video,
    _infer_context_indices,
    _infer_object_pooler_latent_dim,
    _load_trainable_state,
    _resolve_launch_device,
    _run_sampling,
    _select_video_from_path,
    _tensor_stats,
    _write_mp4,
)
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


def _infer_stage1_object_pooler_latent_dim(state_dict: dict[str, torch.Tensor], default_dim: int) -> int:
    prefixes = ("", "module.", "bundle.", "module.bundle.")
    for prefix in prefixes:
        key = f"{prefix}object_pooler.latent_proj.weight"
        if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) == 2:
            return int(state_dict[key].shape[1])
    return int(default_dim)


def _load_stage1_adapter_state_into_model(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, object]:
    state_dict = _load_trainable_state(checkpoint_path)
    adapter_prefixes = ("object_pooler.", "object_adapter.", "object_aux_heads.", "context_fuser.")
    adapter_state = {
        key: value
        for key, value in state_dict.items()
        if key.startswith(adapter_prefixes)
    }
    if not adapter_state:
        raise RuntimeError(
            f"stage1 adapter checkpoint does not contain object adapter weights: {checkpoint_path}"
        )

    missing_info = model.load_state_dict(adapter_state, strict=False)
    return {
        "loaded_adapter_keys": len(adapter_state),
        "missing_keys": list(missing_info.missing_keys),
        "unexpected_keys": list(missing_info.unexpected_keys),
        "filtered_checkpoint_keys": sorted(checkpoint_names),
    }


def _tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor):
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="stage1 adapter checkpoint step_*.pt")
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_stage1_adapters_gpu67.yaml",
        help="stage1 training config used to construct the inference model",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--context-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--context-fraction", type=float, default=0.5)
    parser.add_argument("--random-context-frames", action="store_true")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config["experiment"]["output_dir"] = str(Path(args.checkpoint).parent)

    checkpoint_state = _load_trainable_state(Path(args.checkpoint))
    object_pooler_latent_dim = _infer_stage1_object_pooler_latent_dim(
        checkpoint_state,
        int(config["model"].get("object_pooler_latent_dim", 16)),
    )
    config["model"]["object_pooler_latent_dim"] = int(object_pooler_latent_dim)

    device = _resolve_launch_device()
    device_obj = torch.device(device)

    video_path = Path(args.context_video)
    frames, frame_indices = _select_video_from_path(video_path, args.num_frames, args.sampling_mode)
    video = preprocess_video_rgb_uint8(frames, tuple(config["data"]["resolution"]))
    context_indices = _infer_context_indices(
        total_frames=video.shape[1],
        num_context_frames=int(config["data"]["num_context_frames"]),
        context_fraction=float(args.context_fraction),
        random_context_frames=bool(args.random_context_frames),
        seed=int(args.seed),
    )
    context_video = video[:, context_indices].contiguous().unsqueeze(0)
    num_context_frames = torch.tensor([context_video.shape[2]], dtype=torch.long)

    trainer = ContextVideoTrainer(config, build_optimizer=True, device=device)
    trainer.build_optimizer = False
    load_info = _load_stage1_adapter_state_into_model(trainer, Path(args.checkpoint))
    if trainer.bundle.dit is not None:
        trainer.bundle.dit.eval()

    text_context, object_context, context_latents, prep_debug = _build_cond_context(
        trainer=trainer,
        config=config,
        context_video=context_video.to(device_obj),
        captions=[args.prompt],
        num_context_frames=num_context_frames,
        device_obj=device_obj,
    )
    with torch.inference_mode():
        pred, sample_debug = _run_sampling(
            bundle=trainer.bundle,
            text_context=text_context,
            object_context=object_context,
            context_latents=context_latents,
            total_frames=int(video.shape[1]),
            num_context_frames=int(num_context_frames.item()),
            num_inference_steps=int(args.sampling_steps),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "context_video": str(args.context_video),
        "prompt": str(args.prompt),
        "frame_indices": frame_indices.tolist(),
        "context_indices": context_indices.tolist(),
        "prep_debug": prep_debug,
        "sample_debug": sample_debug,
        "load_info": load_info,
        "trainable_tensor_stats": {
            "text_context": _tensor_stats("text_context", text_context),
            "object_context": _tensor_stats("object_context", object_context),
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

    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
