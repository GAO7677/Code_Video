from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path

import torch

from code_vjepa_vggt.infer_context_video_wan import (
    _load_context_video,
    _print_tensor_stats,
    _resolve_launch_device,
    _run_sampling,
    _write_mp4,
    _ensure_browser_video,
)
from code_vjepa_vggt.models.wan_context_model import WanContextVideoModel
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


def _build_no_object_branch_context(
    *,
    config: dict[str, object],
    context_video: torch.Tensor,
    captions: list[str],
    device_obj: torch.device,
) -> tuple[WanContextVideoModel, torch.Tensor, torch.Tensor, dict[str, object]]:
    model_cfg = config["model"]
    bundle = WanContextVideoModel(
        ckpt_dir=model_cfg["wan_ckpt_dir"],
        task=model_cfg["wan_task"],
        device=str(device_obj),
        load_dit=True,
        lora_rank=int(model_cfg.get("wan_lora_rank", 0)),
        lora_alpha=int(model_cfg.get("wan_lora_alpha", 0)),
        lora_dropout=float(model_cfg.get("wan_lora_dropout", 0.0)),
        lora_init=str(model_cfg.get("wan_lora_init", "gaussian")),
        reinitialize_object_branch=False,
        disable_object_branch=True,
    )
    bundle.freeze_parts(
        freeze_vae=bool(model_cfg.get("freeze_vae", True)),
        freeze_text_encoder=bool(model_cfg.get("freeze_text_encoder", True)),
        freeze_dit=bool(model_cfg.get("freeze_wan_dit", True)),
        freeze_lora=bool(model_cfg.get("freeze_wan_lora", True)),
    )
    init_lora_path = model_cfg.get("init_wan_lora_from_checkpoint")
    if init_lora_path is not None:
        bundle.load_lora_checkpoint(
            init_lora_path,
            strict=bool(model_cfg.get("init_wan_lora_strict", True)),
            zero_missing=bool(model_cfg.get("init_wan_lora_zero_missing", False)),
        )
    bundle.dit.eval()

    videos = context_video.to(device_obj)
    with torch.no_grad():
        text_context_list = [
            u.to(device_obj) for u in bundle.text_encoder(list(captions), bundle.text_encoder.device)
        ]
        context_latents_list = bundle.vae.encode([u.to(device_obj) for u in videos])

    text_context = text_context_list[0]
    context_latents = context_latents_list[0]
    debug = {
        "mode": "wan_lora_no_object_branch",
        "text_context": [list(t.shape) for t in text_context_list],
        "context_latents": [list(t.shape) for t in context_latents_list],
        "context_video": list(videos.shape),
        "wan_lora_checkpoint": str(init_lora_path) if init_lora_path is not None else None,
        "object_branch_initialized": False,
        "object_branch_present": False,
    }
    _print_tensor_stats("context_latents", context_latents)
    _print_tensor_stats("text_context", text_context)
    return bundle, text_context, context_latents, debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="placeholder checkpoint path for bookkeeping only; object-branch weights are not loaded")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--context-video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    device = _resolve_launch_device()
    device_obj = torch.device(device)
    target_total_frames = int(args.num_frames)
    target_context_frames = int(config["data"]["num_context_frames"])
    if target_total_frames < target_context_frames:
        raise ValueError(
            f"--num-frames must be >= configured num_context_frames={target_context_frames}, got {target_total_frames}"
        )

    video_path = Path(args.context_video)
    frames, frame_indices = _load_context_video(
        video_path=video_path,
        target_context_frames=target_context_frames,
        sampling_mode=args.sampling_mode,
    )
    context_video_single = preprocess_video_rgb_uint8(frames, tuple(config["data"]["resolution"]))
    context_indices = torch.arange(target_context_frames, dtype=torch.long)
    context_video = context_video_single.unsqueeze(0)
    num_context_frames = torch.tensor([context_video.shape[2]], dtype=torch.long)

    bundle, text_context, context_latents, prep_debug = _build_no_object_branch_context(
        config=config,
        context_video=context_video.to(device_obj),
        captions=[args.prompt],
        device_obj=device_obj,
    )
    with torch.inference_mode():
        pred, sample_debug = _run_sampling(
            bundle=bundle,
            text_context=text_context,
            object_context=None,
            context_latents=context_latents,
            total_frames=target_total_frames,
            num_context_frames=int(num_context_frames.item()),
            num_inference_steps=int(args.sampling_steps),
            disable_object_context=True,
        )
    print("sampling finished", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = Path(args.output_video) if args.output_video else None
    should_save_video = bool(args.save_raw or output_video_path is not None)
    result = {
        "checkpoint": str(checkpoint_path),
        "context_video": str(args.context_video),
        "prompt": str(args.prompt),
        "context_frame_indices_from_input": frame_indices.tolist(),
        "context_indices": context_indices.tolist(),
        "target_num_frames": int(target_total_frames),
        "configured_num_context_frames": int(target_context_frames),
        "prep_debug": prep_debug,
        "sample_debug": sample_debug,
        "load_state_missing": {
            "skipped": True,
            "reason": "wan_lora_no_object_branch",
            "missing_keys": [],
            "unexpected_keys": [],
            "model_state_key_count": 0,
            "checkpoint_key_count": 0,
        },
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if should_save_video:
        if bundle.dit is not None:
            del bundle.dit
            bundle.dit = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with torch.no_grad():
            decode_input = pred.to(next(bundle.vae.model.parameters()).device if hasattr(bundle.vae, "model") else device_obj)
            _print_tensor_stats("vae_decode_input", decode_input)
            decoded = bundle.vae.decode([decode_input])
        if isinstance(decoded, list):
            decoded = decoded[0]
        _print_tensor_stats("vae_decoded_output", decoded)
        video_out = decoded.detach().cpu()
        video_out = video_out.permute(1, 0, 2, 3).contiguous()
        video_out = ((video_out.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        raw_path = output_video_path if output_video_path is not None else (output_dir / "prediction.mp4")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _write_mp4(raw_path, video_out, fps=int(args.fps))
        browser_path = _ensure_browser_video(raw_path)
        result["prediction_video_raw"] = str(raw_path)
        result["prediction_video"] = str(browser_path)
        with open(output_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("saved prediction video", flush=True)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
