from __future__ import annotations
"""
Stage1B context-only no-GT-box inference.

Weight loading order (must match training):
  1. Wan DiT base        : config model.wan_ckpt_dir
  2. Wan LoRA            : config model.init_wan_lora_from_checkpoint
  3. Stage1A pooler      : --init-from
  4. Stage1B trainables  : --checkpoint

Unlike the original `wan_stage1b_context_only_v2v.py`, this script does NOT read
dataset GT `context_boxes` from sample npz. It rebuilds pseudo boxes at inference
time using the same viewer-style GroundingDINO + SAM2 provider used by
`train_stage1b_context_only_no_gt_box.py`.
"""

import argparse
import gc
import json
import re
from pathlib import Path

import torch

from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root
from code_vjepa_vggt.infer_context_video_wan import (
    _build_cond_context,
    _ensure_browser_video,
    _infer_object_pooler_latent_dim,
    _load_trainable_state,
    _resolve_launch_device,
    _run_sampling,
    _tensor_stats,
    _write_mp4,
)
from code_vjepa_vggt.infer_v_newtrain_context_video_wan import _load_context_video
from code_vjepa_vggt.object_token_teacher_student.runtime_stage1b_context_only_no_gt_box import (
    ContextOnlyInjectionNoGTBoxTrainer,
)
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


def _normalize_ckpt_method_name(name: str) -> str:
    normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
    return normalized or name


def _build_method_name_from_checkpoint_path(checkpoint_path: Path) -> str:
    stem = checkpoint_path.stem
    parent = checkpoint_path.parent
    if parent.name:
        method_root = _normalize_ckpt_method_name(parent.name)
        return f"{method_root}_{stem}"
    return stem


def _load_trainable_state_into_model(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, object]:
    resolved = checkpoint_path.expanduser().resolve()
    if resolved.is_dir():
        candidate = resolved / "checkpoint.safetensors"
        if candidate.is_file():
            resolved = candidate
        else:
            candidates = sorted(resolved.rglob("checkpoint.safetensors"))
            if candidates:
                resolved = candidates[-1]
    if resolved.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file

        model_state = load_safetensors_file(str(resolved), device="cpu")
        checkpoint_step = -1
    else:
        state = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "model" not in state or not isinstance(state["model"], dict):
            raise RuntimeError(f"unsupported trainable checkpoint format: {resolved}")
        model_state = state["model"]
        checkpoint_step = int(state.get("step", -1))
    loaded = model.load_state_dict(model_state, strict=False)
    return {
        "checkpoint_step": checkpoint_step,
        "loaded_key_count": len(model_state),
        "missing_keys": list(loaded.missing_keys),
        "unexpected_keys": list(loaded.unexpected_keys),
    }


def _checkpoint_chain_info(config: dict, checkpoint_path: Path, init_from: Path) -> dict[str, object]:
    state_dict = _load_trainable_state(checkpoint_path)
    default_latent_dim = int(config["model"].get("object_pooler_latent_dim", 16))
    object_pooler_latent_dim = _infer_object_pooler_latent_dim(state_dict, default_latent_dim)
    if object_pooler_latent_dim == default_latent_dim and init_from.is_file():
        try:
            init_state = _load_trainable_state(init_from)
            object_pooler_latent_dim = _infer_object_pooler_latent_dim(init_state, object_pooler_latent_dim)
        except Exception:
            pass
    slot_embed = state_dict.get("object_adapter.slot_embed.weight")
    if slot_embed is not None and hasattr(slot_embed, "shape") and len(slot_embed.shape) == 2:
        inferred_num_slots = int(slot_embed.shape[0])
        config["model"]["sam2_max_objects"] = inferred_num_slots
        config["model"]["object_num_queries"] = inferred_num_slots
    config["model"]["object_pooler_latent_dim"] = int(object_pooler_latent_dim)
    config["model"]["init_wan_lora_from_checkpoint"] = str(config["model"]["init_wan_lora_from_checkpoint"])
    return {
        "trainable_checkpoint": str(checkpoint_path),
        "stage1a_init_from": str(init_from),
        "wan_root": str(config["model"]["wan_ckpt_dir"]),
        "frozen_wan_lora_init": str(config["model"].get("init_wan_lora_from_checkpoint")),
        "object_pooler_latent_dim": int(object_pooler_latent_dim),
        "inferred_num_slots": int(config["model"].get("sam2_max_objects", -1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run Stage1B context-only no-GT-box checkpoints over an input json list."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trainable checkpoint: supports old step_XXXXXXX.pt, checkpoint.safetensors, or a checkpoint directory",
    )
    parser.add_argument("--init-from", type=Path, required=True, help="Stage1A checkpoint used during training init-from")
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
            "object_token_teacher_student/config_stage1b_context_only_no_gt_box_template.yaml"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _read_list_file(list_path: Path) -> list[Path]:
    items: list[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            items.append(Path(line).expanduser().resolve())
    return items


def _load_input_json(json_path: Path) -> dict[str, object]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"input json must be an object: {json_path}")
    return data


def _ensure_str_field(payload: dict[str, object], key: str, json_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or empty {key!r} in {json_path}")
    return value.strip()


def _resolve_input_video(payload: dict[str, object], json_path: Path) -> str:
    value = payload.get("input_video")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise KeyError(f"missing required field 'input_video' in {json_path}")


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            if not line.endswith("\n"):
                handle.write("\n")


def _build_zero_batch_extras(
    *,
    context_video: torch.Tensor,
    max_objects: int,
    total_frames: int,
) -> dict[str, torch.Tensor]:
    batch_size = int(context_video.shape[0])
    context_frames = int(context_video.shape[2])
    future_frames = max(0, int(total_frames) - int(context_frames))
    device = context_video.device
    dtype = context_video.dtype
    return {
        "future_boxes": torch.zeros((batch_size, future_frames, max_objects, 4), dtype=dtype, device=device),
        "context_states": torch.zeros((batch_size, context_frames, max_objects, 10), dtype=dtype, device=device),
        "future_states": torch.zeros((batch_size, future_frames, max_objects, 10), dtype=dtype, device=device),
        "appearance": torch.zeros((batch_size, max_objects, 16), dtype=dtype, device=device),
        "camera": torch.zeros((batch_size, context_frames, 10), dtype=dtype, device=device),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    init_from = args.init_from.expanduser().resolve()
    input_json_list_path = args.input_json_list_path.expanduser().resolve()
    model_name = str(args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=args.output_root,
        base_output_root="/data/gaoya/agent-data/outputs/stage1b_context_only_no_gt_box_v2v",
        model_name=model_name,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config(args.config.expanduser().resolve())
    checkpoint_chain = _checkpoint_chain_info(config, checkpoint_path, init_from)
    device = _resolve_launch_device()
    config["model"]["grounding_device"] = str(device)
    trainer = ContextOnlyInjectionNoGTBoxTrainer(config, build_optimizer=False, device=device)

    if init_from.is_file():
        stage1a_state = torch.load(init_from, map_location="cpu", weights_only=False)
        if isinstance(stage1a_state, dict) and "model" in stage1a_state:
            trainer.load_state_dict(stage1a_state["model"], strict=False)
    load_info = _load_trainable_state_into_model(trainer, checkpoint_path)
    if trainer.bundle.dit is not None:
        trainer.bundle.dit.eval()

    json_paths = _read_list_file(input_json_list_path)
    if args.limit is not None:
        json_paths = json_paths[: max(0, int(args.limit))]

    step_output_dir = output_root / checkpoint_path.stem
    step_output_dir.mkdir(parents=True, exist_ok=True)
    method_name = _build_method_name_from_checkpoint_path(checkpoint_path)

    batch_manifest = {
        "input_json_list_path": str(input_json_list_path),
        "checkpoint": str(checkpoint_path),
        "init_from": str(init_from),
        "model_name": model_name,
        "num_items": len(json_paths),
        "sampling_steps": int(args.sampling_steps),
        "sampling_mode": str(args.sampling_mode),
        "seed": int(args.seed),
        "box_source": "viewer_grounding_gdino_sam2",
        "depends_on_dataset_boxes": False,
    }
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(batch_manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    for input_json_path in json_paths:
        payload = _load_input_json(input_json_path)
        try:
            input_video = _resolve_input_video(payload, input_json_path)
            input_caption = _ensure_str_field(payload, "input_caption", input_json_path)
        except (KeyError, ValueError) as exc:
            print(f"[skip] {input_json_path.stem}: {exc}")
            continue

        sample_stem = input_json_path.stem
        output_video = step_output_dir / f"{sample_stem}.mp4"
        output_json = step_output_dir / f"{sample_stem}.json"
        output_log = step_output_dir / f"{sample_stem}.log"
        if output_video.exists() and output_json.exists() and not args.force:
            print(f"[skip] {sample_stem}")
            continue

        try:
            frames, frame_indices = _load_context_video(
                video_path=Path(input_video),
                target_context_frames=int(config["data"]["num_context_frames"]),
                sampling_mode=str(args.sampling_mode),
            )
            context_video_single = preprocess_video_rgb_uint8(frames, tuple(config["data"]["resolution"]))
            context_video = context_video_single.unsqueeze(0).to(trainer.device_obj)
            num_context_frames = torch.tensor([context_video.shape[2]], dtype=torch.long, device=trainer.device_obj)
            total_frames = int(config["data"]["num_context_frames"] / float(config["data"].get("context_fraction", 0.5)))
            batch_extra_tensors = _build_zero_batch_extras(
                context_video=context_video,
                max_objects=int(getattr(trainer, "max_objects", config["model"].get("object_num_queries", 8))),
                total_frames=total_frames,
            )
            text_context, object_context, context_latents, prep_debug = _build_cond_context(
                trainer=trainer,
                config=config,
                context_video=context_video,
                captions=[input_caption],
                num_context_frames=num_context_frames,
                device_obj=trainer.device_obj,
                batch_extra_tensors=batch_extra_tensors,
            )
            with torch.inference_mode():
                pred, sample_debug = _run_sampling(
                    bundle=trainer.bundle,
                    text_context=text_context,
                    object_context=object_context,
                    context_latents=context_latents,
                    total_frames=total_frames,
                    num_context_frames=int(num_context_frames.item()),
                    num_inference_steps=int(args.sampling_steps),
                )
        except Exception as exc:
            _write_text_lines(output_log, [f"[error] {exc}"])
            print(f"[error] {sample_stem}: {exc}")
            continue

        result = {
            "method": method_name,
            "checkpoint": str(checkpoint_path),
            "init_from": str(init_from),
            "config": str(args.config.expanduser().resolve()),
            "checkpoint_chain": checkpoint_chain,
            "load_info": load_info,
            "context_video": str(input_video),
            "prompt": str(input_caption),
            "frame_indices": frame_indices.tolist(),
            "context_boxes_debug": {
                "status": "viewer_grounding_loaded",
                "teacher_student_stage1": prep_debug.get("teacher_student_stage1", {}),
                "grounding_samples": prep_debug.get("grounding_samples", []),
            },
            "prep_debug": prep_debug,
            "sample_debug": sample_debug,
            "trainable_tensor_stats": {
                "text_context": _tensor_stats("text_context", text_context),
                "object_context": _tensor_stats("object_context", object_context),
                "context_latents": _tensor_stats("context_latents", context_latents),
            },
        }

        if args.save_raw:
            with torch.no_grad():
                decode_input = pred.to(
                    next(trainer.bundle.vae.model.parameters()).device
                    if hasattr(trainer.bundle.vae, "model")
                    else trainer.device_obj
                )
                decoded = trainer.bundle.vae.decode([decode_input])
            if isinstance(decoded, list):
                decoded = decoded[0]
            video_out = decoded.detach().cpu()
            video_out = video_out.permute(1, 0, 2, 3).contiguous()
            video_out = ((video_out.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            _write_mp4(output_video, video_out, fps=int(args.fps))
            result["prediction_video"] = str(_ensure_browser_video(output_video))

        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        _write_text_lines(
            output_log,
            [
                f"[checkpoint] {checkpoint_path}",
                f"[init_from] {init_from}",
                f"[load_info] {json.dumps(load_info, ensure_ascii=False)}",
                f"[done] {sample_stem}",
            ],
        )
        print(f"[done] {sample_stem}")

    if trainer.bundle.dit is not None:
        del trainer.bundle.dit
        trainer.bundle.dit = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
