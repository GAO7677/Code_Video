from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file

from code_vjepa_vggt.train_v_newtrain import (
    WanTrainingModule,
    build_wan22_ti2v5b_model_paths,
    find_tokenizer_path,
)
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import (
    preprocess_video_rgb_uint8,
    read_video_prefix,
    read_video_uniform,
)


def _resolve_launch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _resolve_checkpoint_file(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path.is_dir():
        candidate = checkpoint_path / "checkpoint.safetensors"
        if candidate.is_file():
            return candidate
        candidates = sorted(checkpoint_path.rglob("checkpoint.safetensors"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")


def _load_trainable_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    resolved = _resolve_checkpoint_file(checkpoint_path)
    if resolved.suffix == ".safetensors":
        return load_safetensors_file(str(resolved), device="cpu")
    state = torch.load(resolved, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
        return state
    raise RuntimeError(f"unsupported checkpoint format: {resolved}")


def _normalize_checkpoint_key(key: str) -> str:
    normalized = str(key)
    prefixes = (
        "module.",
        "pipe.dit.",
        "base_model.model.",
        "dit.base_model.model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def _load_v_newtrain_state_into_model(model: WanTrainingModule, checkpoint_path: Path) -> dict[str, object]:
    state_dict = _load_trainable_state(checkpoint_path)
    if model.enable_object_branch and model.object_pooler is not None:
        latent_key = None
        for candidate in ("object_pooler.latent_proj.weight", "bundle.object_pooler.latent_proj.weight"):
            if candidate in state_dict:
                latent_key = candidate
                break
        if latent_key is not None:
            latent_dim = int(state_dict[latent_key].shape[1])
            model.object_pooler._ensure_latent_proj(latent_dim, torch.device(model.pipe.device))

    model_state = model.state_dict()
    normalized_model_keys = {_normalize_checkpoint_key(key): key for key in model_state.keys()}
    normalized_checkpoint_keys = {_normalize_checkpoint_key(key): key for key in state_dict.keys()}
    overlapping = sorted(set(normalized_model_keys.keys()) & set(normalized_checkpoint_keys.keys()))
    if not overlapping:
        raise RuntimeError("no overlapping keys between current v_newtrain model and checkpoint")

    filtered_state = {}
    skipped_shape_mismatch = []
    for norm_key in overlapping:
        model_key = normalized_model_keys[norm_key]
        ckpt_key = normalized_checkpoint_keys[norm_key]
        model_value = model_state[model_key]
        ckpt_value = state_dict[ckpt_key]
        if tuple(model_value.shape) != tuple(ckpt_value.shape):
            skipped_shape_mismatch.append(
                {
                    "model_key": model_key,
                    "checkpoint_key": ckpt_key,
                    "model_shape": list(model_value.shape),
                    "checkpoint_shape": list(ckpt_value.shape),
                }
            )
            continue
        filtered_state[model_key] = ckpt_value
    missing = model.load_state_dict(filtered_state, strict=False)
    return {
        "loaded_count": len(filtered_state),
        "missing_keys": list(missing.missing_keys),
        "unexpected_keys": list(missing.unexpected_keys),
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }


def _select_video_from_path(video_path: Path, num_frames: int, sampling_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if sampling_mode == "prefix":
        return read_video_prefix(video_path, num_frames)
    return read_video_uniform(video_path, num_frames)


def _load_context_video(
    *,
    video_path: Path,
    target_context_frames: int,
    sampling_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    frames, frame_indices = _select_video_from_path(video_path, target_context_frames, sampling_mode)
    if int(frames.shape[0]) < int(target_context_frames):
        raise RuntimeError(
            f"context video {video_path} only provides {int(frames.shape[0])} frames, "
            f"smaller than required num_context_frames={int(target_context_frames)}"
        )
    if int(frames.shape[0]) > int(target_context_frames):
        frames = frames[:target_context_frames]
        frame_indices = frame_indices[:target_context_frames]
    return frames, frame_indices


def _clone_context_frames_as_pil_list(pipe, context_video_single: torch.Tensor):
    rgb = context_video_single.detach().cpu().permute(1, 2, 3, 0)
    rgb = ((rgb + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    return [pipe.preprocess_image(frame) if False else frame for frame in []]


def _sample_points_from_box(box_xyxy: torch.Tensor, points_per_object: int) -> torch.Tensor:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
    if x1 <= x0 or y1 <= y0:
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        return torch.tensor([[cx, cy]] * points_per_object, dtype=torch.float32)
    cols = max(1, int(np.ceil(np.sqrt(float(points_per_object)))))
    rows = max(1, int(np.ceil(float(points_per_object) / float(cols))))
    xs = torch.linspace(x0 + 0.2 * (x1 - x0), x0 + 0.8 * (x1 - x0), cols)
    ys = torch.linspace(y0 + 0.2 * (y1 - y0), y0 + 0.8 * (y1 - y0), rows)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    return points[:points_per_object].contiguous()


def _build_center_box_query_priors(
    *,
    height: int,
    width: int,
    aux_max_objects: int,
    object_num_queries: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_box = torch.tensor([0.25, 0.25, 0.75, 0.75], dtype=torch.float32)
    grouped_points = []
    valid_mask = []
    for _ in range(int(aux_max_objects)):
        valid_mask.append(1.0)
        points = _sample_points_from_box(base_box, int(object_num_queries))
        points[:, 0] *= float(width)
        points[:, 1] *= float(height)
        grouped_points.append(points)
    grouped = torch.stack(grouped_points, dim=0)
    flat = grouped.view(1, int(aux_max_objects) * int(object_num_queries), 2)
    valid = torch.tensor(valid_mask, dtype=torch.float32).view(1, int(aux_max_objects))
    return flat, valid


def _tensor_video_to_pil_list(context_video_single: torch.Tensor):
    from PIL import Image

    frames = context_video_single.detach().cpu().permute(1, 2, 3, 0)
    frames = ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    return [Image.fromarray(frame) for frame in frames]


def _build_object_context(
    *,
    model: WanTrainingModule,
    context_video_single: torch.Tensor,
) -> tuple[torch.Tensor | None, dict[str, object]]:
    if not model.enable_object_branch:
        return None, {"enabled": False}

    pipe = model.pipe
    device = torch.device(pipe.device)
    context_video = context_video_single.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
    query_points_prior, object_valid_mask = _build_center_box_query_priors(
        height=image_hw[0],
        width=image_hw[1],
        aux_max_objects=model.aux_max_objects,
        object_num_queries=model.object_num_queries,
    )
    query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)

    frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
    cotracker_out = model.cotracker_adapter(
        frames_bthwc_01,
        query_points_prior=query_points_prior,
        query_image_hw=image_hw,
    )
    tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
        cotracker_out.tracks,
        cotracker_out.visibility,
        cotracker_out.confidence,
        max_objects=model.aux_max_objects,
        points_per_object=model.object_num_queries,
    )
    jepa_dtype = next(model.jepa_adapter.parameters()).dtype
    jepa_out = model.jepa_adapter(context_video.to(dtype=jepa_dtype))
    preprocessed_context_video = pipe.preprocess_video(_tensor_video_to_pil_list(context_video_single))
    clean_prefix_latents = pipe.vae.encode(
        preprocessed_context_video,
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ).to(dtype=pipe.torch_dtype, device=device)
    object_out = model.object_pooler(
        jepa_patch_tokens=jepa_out.patch_tokens,
        context_latents=clean_prefix_latents,
        tracks=tracks_grouped,
        visibility=visibility_grouped,
        confidence=confidence_grouped,
        track_image_hw=image_hw,
        object_valid_mask=object_valid_mask,
        box_prior_xyxy=torch.tensor(
            [[0.25, 0.25, 0.75, 0.75]] * model.aux_max_objects,
            dtype=pipe.torch_dtype,
            device=device,
        ).unsqueeze(0),
        frame_valid_mask=None,
    )
    object_context = model.object_adapter(
        object_out.object_latent_tokens,
        object_valid_mask=object_valid_mask,
    )
    debug = {
        "enabled": True,
        "query_points_shape": list(query_points_prior.shape),
        "tracks_shape": list(cotracker_out.tracks.shape),
        "object_context_shape": list(object_context.shape),
        "clean_prefix_latents_shape": list(clean_prefix_latents.shape),
    }
    return object_context, debug


def build_model(args) -> WanTrainingModule:
    model_paths = build_wan22_ti2v5b_model_paths(args.wan_root)
    tokenizer_path = find_tokenizer_path(args.wan_root)
    return WanTrainingModule(
        model_paths=model_paths,
        tokenizer_path=tokenizer_path,
        trainable_models=None,
        lora_base_model="dit",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=int(args.lora_rank),
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs="input_image",
        fp8_models=None,
        offload_models=None,
        device=args.device,
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        context_sampling_profile="legacy_prefix",
        min_context_frames=1,
        max_context_ratio=0.5,
        context_reference_frames=49,
        context_reference_prefixes="1,4,8,12,16",
        prefix_context_ratio=0.55,
        first_frame_context_ratio=0.20,
        sparse_context_ratio=0.15,
        random_context_ratio=0.05,
        no_context_ratio=0.05,
        fixed_num_context_frames=int(args.context_frames),
        enable_object_branch=not args.disable_object_branch,
        object_num_queries=int(args.object_num_queries),
        aux_max_objects=int(args.aux_max_objects),
        jepa_ckpt_path=args.jepa_ckpt_path,
        jepa_input_size=int(args.jepa_input_size),
        jepa_patch_size=int(args.jepa_patch_size),
        jepa_tubelet_size=int(args.jepa_tubelet_size),
        cotracker_checkpoint=args.cotracker_checkpoint,
        cotracker_input_h=int(args.cotracker_input_h),
        cotracker_input_w=int(args.cotracker_input_w),
        cotracker_window_len=int(args.cotracker_window_len),
        object_pooler_latent_dim=int(args.object_pooler_latent_dim),
        cond_proj_dim=int(args.cond_proj_dim),
        jepa_window_radius=int(args.jepa_window_radius),
        latent_window_radius=int(args.latent_window_radius),
        lambda_track_aux=0.1,
        lambda_box_aux=0.1,
        lambda_depth_aux=0.0,
        depth_target_state_index=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--wan-root", default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
    parser.add_argument("--context-video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--disable-object-branch", action="store_true")
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--jepa-ckpt-path", default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth")
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--jepa-window-radius", type=int, default=1)
    parser.add_argument("--latent-window-radius", type=int, default=1)
    args = parser.parse_args()

    if args.config:
        config = load_yaml_config(args.config)
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        train_cfg = config.get("training", {})
        resolution = data_cfg.get("resolution")
        if "wan_root" in model_cfg and parser.get_default("wan_root") == args.wan_root:
            args.wan_root = str(model_cfg["wan_root"])
        if "wan_ckpt_dir" in model_cfg and parser.get_default("wan_root") == args.wan_root:
            args.wan_root = str(model_cfg["wan_ckpt_dir"])
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            if parser.get_default("height") == args.height:
                args.height = int(resolution[0])
            if parser.get_default("width") == args.width:
                args.width = int(resolution[1])
        if "height" in data_cfg and parser.get_default("height") == args.height:
            args.height = int(data_cfg["height"])
        if "width" in data_cfg and parser.get_default("width") == args.width:
            args.width = int(data_cfg["width"])
        if "num_frames" in data_cfg and parser.get_default("num_frames") == args.num_frames:
            args.num_frames = int(data_cfg["num_frames"])
        if "fps" in data_cfg and parser.get_default("fps") == args.fps:
            args.fps = int(data_cfg["fps"])
        if "fixed_num_context_frames" in data_cfg and parser.get_default("context_frames") == args.context_frames:
            args.context_frames = int(data_cfg["fixed_num_context_frames"])
        if "num_context_frames" in data_cfg and parser.get_default("context_frames") == args.context_frames:
            args.context_frames = int(data_cfg["num_context_frames"])
        if "lora_rank" in train_cfg and parser.get_default("lora_rank") == args.lora_rank:
            args.lora_rank = int(train_cfg["lora_rank"])
        if "wan_lora_rank" in model_cfg and parser.get_default("lora_rank") == args.lora_rank:
            args.lora_rank = int(model_cfg["wan_lora_rank"])
        if "object_num_queries" in model_cfg and parser.get_default("object_num_queries") == args.object_num_queries:
            args.object_num_queries = int(model_cfg["object_num_queries"])
        if "aux_max_objects" in model_cfg and parser.get_default("aux_max_objects") == args.aux_max_objects:
            args.aux_max_objects = int(model_cfg["aux_max_objects"])
        if "sam2_max_objects" in model_cfg and parser.get_default("aux_max_objects") == args.aux_max_objects:
            args.aux_max_objects = int(model_cfg["sam2_max_objects"])
        if "jepa_ckpt_path" in model_cfg and parser.get_default("jepa_ckpt_path") == args.jepa_ckpt_path:
            args.jepa_ckpt_path = str(model_cfg["jepa_ckpt_path"])
        if "jepa_input_size" in model_cfg and parser.get_default("jepa_input_size") == args.jepa_input_size:
            args.jepa_input_size = int(model_cfg["jepa_input_size"])
        if "jepa_patch_size" in model_cfg and parser.get_default("jepa_patch_size") == args.jepa_patch_size:
            args.jepa_patch_size = int(model_cfg["jepa_patch_size"])
        if "jepa_tubelet_size" in model_cfg and parser.get_default("jepa_tubelet_size") == args.jepa_tubelet_size:
            args.jepa_tubelet_size = int(model_cfg["jepa_tubelet_size"])
        if "cotracker_checkpoint" in model_cfg and parser.get_default("cotracker_checkpoint") == args.cotracker_checkpoint:
            args.cotracker_checkpoint = str(model_cfg["cotracker_checkpoint"])
        cotracker_input_hw = model_cfg.get("cotracker_input_hw")
        if isinstance(cotracker_input_hw, (list, tuple)) and len(cotracker_input_hw) == 2:
            if parser.get_default("cotracker_input_h") == args.cotracker_input_h:
                args.cotracker_input_h = int(cotracker_input_hw[0])
            if parser.get_default("cotracker_input_w") == args.cotracker_input_w:
                args.cotracker_input_w = int(cotracker_input_hw[1])
        if "cotracker_input_h" in model_cfg and parser.get_default("cotracker_input_h") == args.cotracker_input_h:
            args.cotracker_input_h = int(model_cfg["cotracker_input_h"])
        if "cotracker_input_w" in model_cfg and parser.get_default("cotracker_input_w") == args.cotracker_input_w:
            args.cotracker_input_w = int(model_cfg["cotracker_input_w"])
        if "cotracker_window_len" in model_cfg and parser.get_default("cotracker_window_len") == args.cotracker_window_len:
            args.cotracker_window_len = int(model_cfg["cotracker_window_len"])
        if "object_pooler_latent_dim" in model_cfg and parser.get_default("object_pooler_latent_dim") == args.object_pooler_latent_dim:
            args.object_pooler_latent_dim = int(model_cfg["object_pooler_latent_dim"])
        if "cond_proj_dim" in model_cfg and parser.get_default("cond_proj_dim") == args.cond_proj_dim:
            args.cond_proj_dim = int(model_cfg["cond_proj_dim"])
        if "jepa_window_radius" in model_cfg and parser.get_default("jepa_window_radius") == args.jepa_window_radius:
            args.jepa_window_radius = int(model_cfg["jepa_window_radius"])
        if "latent_window_radius" in model_cfg and parser.get_default("latent_window_radius") == args.latent_window_radius:
            args.latent_window_radius = int(model_cfg["latent_window_radius"])

    args.device = _resolve_launch_device()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    model = build_model(args)
    model.to(torch.device(args.device))
    model.eval()
    load_info = _load_v_newtrain_state_into_model(model, Path(args.checkpoint))
    pipe = model.pipe
    pipe.dit.eval()

    frames, frame_indices = _load_context_video(
        video_path=Path(args.context_video),
        target_context_frames=int(args.context_frames),
        sampling_mode=args.sampling_mode,
    )
    context_video_single = preprocess_video_rgb_uint8(frames, (int(args.height), int(args.width)))
    context_pil = _tensor_video_to_pil_list(context_video_single)
    object_context, object_debug = _build_object_context(
        model=model,
        context_video_single=context_video_single,
    )

    from diffsynth.utils.data import save_video

    with torch.no_grad():
        video = pipe(
            prompt=args.prompt,
            negative_prompt="",
            context_video=context_pil,
            seed=int(args.seed),
            tiled=True,
            height=int(args.height),
            width=int(args.width),
            num_frames=int(args.num_frames),
            num_inference_steps=int(args.sampling_steps),
            cfg_scale=float(args.cfg_scale),
            object_context=object_context,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_video), fps=int(args.fps), quality=int(args.quality))

    result = {
        "checkpoint": str(_resolve_checkpoint_file(Path(args.checkpoint))),
        "output_video": str(output_video),
        "context_video": str(args.context_video),
        "prompt": str(args.prompt),
        "frame_indices": frame_indices.tolist(),
        "load_info": load_info,
        "object_debug": object_debug,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
