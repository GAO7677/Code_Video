
'''

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py \
  --checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/pybullet0613_wan_lora_gpu67/step_0000940.pt \
  --prompt "your prompt here" \
  --context-video /path/to/context_8frames.mp4 \
  --num-frames 24 \
  --output-video /tmp/prediction.mp4
'''
from __future__ import annotations

import argparse
import json
import math
import gc
import shutil
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file

from code_vjepa_vggt.models.wan_context_model import WanContextVideoModel
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import broadcast_latent_mask, expand_context_latents_to_full, latent_frame_mask
from code_vjepa_vggt.utils.paths import ensure_upstream_paths
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix, read_video_uniform

ensure_upstream_paths()

from wan.utils import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.adapters.jepa_adapter import JEPAPatchAdapter
from code_vjepa_vggt.adapters.sam2_motion import GroundingDINOTextDetector, SAM2MotionTracker, build_motion_prompt_box
from code_vjepa_vggt.adapters.sam2_motion import build_motion_prompt_boxes
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior


def _build_multi_object_prompt(caption: str) -> str:
    caption_lower = str(caption).lower()
    ordered = ["sphere", "ball", "block", "box", "cube", "cylinder", "capsule"]
    found = []
    for token in ordered:
        if token in caption_lower and token not in found:
            found.append(token)
    if not found:
        return str(caption)
    return " . ".join(found) + " ."


def _resolve_launch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _find_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/wan/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("ffmpeg is required to write H.264 mp4 output")


def _tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def _video_bcthw_to_uint8_thwc(video_bcthw: torch.Tensor) -> np.ndarray:
    video = video_bcthw.detach().cpu().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).to(torch.uint8)
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    return video.permute(1, 2, 3, 0).contiguous().numpy()


def _write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    ffmpeg = _find_ffmpeg()
    tmp_path = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {tmp_path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    import subprocess

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(tmp_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp_path.unlink(missing_ok=True)


def _ensure_browser_video(source_path: Path) -> Path:
    try:
        ffmpeg = _find_ffmpeg()
    except RuntimeError:
        return source_path
    out_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
    if out_path.exists() and out_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns and out_path.stat().st_size > 0:
        return out_path
    import subprocess

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def _draw_box_rgb(image: np.ndarray, box_xyxy_px: np.ndarray, color_rgb: tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy_px.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image, (x0, y0), (x1, y1), color_bgr, 2)
    cv2.putText(image, label, (x0 + 2, max(y0 + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)


def _draw_point_rgb(image: np.ndarray, point_xy: np.ndarray, color_rgb: tuple[int, int, int], label: str, radius: int = 5) -> None:
    x, y = [int(round(v)) for v in point_xy.tolist()]
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.circle(image, (x, y), radius, color_bgr, 2)
    if label:
        cv2.putText(image, label, (x + 6, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA)


def _overlay_mask(frame_hwc: np.ndarray, mask_hw: np.ndarray, color_rgb: tuple[int, int, int], alpha: float = 0.32) -> np.ndarray:
    frame = frame_hwc.astype(np.float32).copy()
    mask = mask_hw > 0
    if np.any(mask):
        color = np.asarray(color_rgb, dtype=np.float32)
        frame[mask] = (1.0 - alpha) * frame[mask] + alpha * color[None, :]
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def _tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, object]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(tensor)}")
    finite = torch.isfinite(tensor)
    if not bool(finite.all().item()):
        bad = int((~finite).sum().item())
        raise RuntimeError(f"{name} contains non-finite values: bad_count={bad}, shape={list(tensor.shape)}")
    tensor_f = tensor.detach().float()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor_f.min().item()) if tensor.numel() > 0 else None,
        "max": float(tensor_f.max().item()) if tensor.numel() > 0 else None,
        "mean": float(tensor_f.mean().item()) if tensor.numel() > 0 else None,
        "std": float(tensor_f.std(unbiased=False).item()) if tensor.numel() > 0 else None,
    }


def _print_tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, object]:
    stats = _tensor_stats(name, tensor)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def _resolve_checkpoint_file(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path.is_dir():
        direct_safetensors = checkpoint_path / "checkpoint.safetensors"
        if direct_safetensors.is_file():
            return direct_safetensors
        nested_safetensors = sorted(checkpoint_path.rglob("checkpoint.safetensors"))
        if nested_safetensors:
            return nested_safetensors[-1]
        candidates = sorted(checkpoint_path.glob("step_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"no step_*.pt found under {checkpoint_path}")
        return max(candidates, key=lambda p: p.stat().st_mtime_ns)
    if checkpoint_path.suffix == ".pt":
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
    raise FileNotFoundError(f"checkpoint path not found: {checkpoint_path}")


def _load_trainable_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    latest = _resolve_checkpoint_file(checkpoint_path)
    if latest.suffix == ".safetensors":
        state = load_safetensors_file(str(latest), device="cpu")
        return state
    state = torch.load(latest, map_location="cpu", weights_only=False)
    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if isinstance(state, dict):
        return state
    raise RuntimeError(f"unsupported checkpoint format in {latest}")


def _infer_object_pooler_latent_dim(state_dict: dict[str, torch.Tensor], default_dim: int) -> int:
    candidate_keys = [
        "object_pooler.latent_proj.weight",
        "bundle.object_pooler.latent_proj.weight",
    ]
    for key in candidate_keys:
        if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) == 2:
            return int(state_dict[key].shape[1])
    return int(default_dim)


def _load_trainable_state_into_model(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, object]:
    state_dict = _load_trainable_state(checkpoint_path)
    def _normalize_key(key: str) -> str:
        prefixes = ("module.", "bundle.", "bundle.dit.")
        normalized = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
        return normalized

    export_fn = getattr(model, "export_trainable_state_dict", None)
    if export_fn is None or not callable(export_fn):
        raise AttributeError("model must implement export_trainable_state_dict() for checkpoint validation")
    model_state_dict = export_fn()
    if not isinstance(model_state_dict, dict):
        raise TypeError(f"export_trainable_state_dict() must return a dict, got {type(model_state_dict)}")
    model_state_keys = list(model_state_dict.keys())
    checkpoint_keys = list(state_dict.keys())
    model_by_normalized: dict[str, str] = {}
    for key in model_state_keys:
        model_by_normalized[_normalize_key(key)] = key
    checkpoint_by_normalized: dict[str, str] = {}
    for key in checkpoint_keys:
        checkpoint_by_normalized[_normalize_key(key)] = key

    if hasattr(model, "object_pooler") and hasattr(model.object_pooler, "_ensure_latent_proj"):
        latent_key = "bundle.object_pooler.latent_proj.weight"
        latent_state_key = checkpoint_by_normalized.get(_normalize_key(latent_key))
        if latent_state_key is not None:
            latent_weight = state_dict[latent_state_key]
            if hasattr(latent_weight, "shape") and len(latent_weight.shape) == 2:
                target_latent_dim = int(latent_weight.shape[1])
                model.object_pooler._ensure_latent_proj(target_latent_dim, getattr(model, "device_obj", torch.device("cpu")))

    normalized_model_keys = set(model_by_normalized.keys())
    normalized_checkpoint_keys = set(checkpoint_by_normalized.keys())
    missing_trainable = sorted(normalized_model_keys - normalized_checkpoint_keys)
    unexpected_checkpoint = sorted(normalized_checkpoint_keys - normalized_model_keys)
    if missing_trainable:
        raise RuntimeError(
            "checkpoint does not match current trainable modules; "
            f"missing_trainable_keys={missing_trainable}, "
            f"unexpected_checkpoint_keys={unexpected_checkpoint}"
        )
    filtered_state = {
        model_by_normalized[norm_key]: state_dict[checkpoint_by_normalized[norm_key]]
        for norm_key in normalized_model_keys
    }
    missing = model.load_state_dict(filtered_state, strict=False)
    return {
        "missing_keys": list(missing.missing_keys),
        "unexpected_keys": list(missing.unexpected_keys),
        "model_state_key_count": len(model_state_keys),
        "checkpoint_key_count": len(checkpoint_keys),
    }


def _infer_context_indices(total_frames: int, num_context_frames: int, context_fraction: float, random_context_frames: bool, seed: int, sample_idx: int = 0) -> torch.Tensor:
    max_context_len = max(1, min(total_frames, int(total_frames * context_fraction)))
    if not random_context_frames:
        context_len = min(num_context_frames, max_context_len)
        return torch.arange(context_len, dtype=torch.long)
    if max_context_len <= 1:
        return torch.arange(1, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(seed + sample_idx)
    context_len = int(torch.randint(1, max_context_len + 1, (1,), generator=generator).item())
    return torch.arange(context_len, dtype=torch.long)


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


def _build_query_prior_for_sample(
    *,
    frames_tchw_01: np.ndarray,
    prompt_frame_idx: int,
    caption: str,
    sam2_tracker: SAM2MotionTracker | None,
    text_detector: GroundingDINOTextDetector | None,
    num_queries: int,
    sam2_prior_strategy: str,
    vggt_num_queries: int,
) -> tuple[np.ndarray | None, str, str, dict[str, object]]:
    if sam2_tracker is None:
        raise RuntimeError("SAM2 tracker is required to build query priors")

    if sam2_prior_strategy in {"grounded_text_multi", "text_multi", "grounded_text"}:
        max_objects = 4
        text_prompt = _build_multi_object_prompt(caption)
        detected_boxes = None
        prompt_mode = "caption_gdino_multi"
        detector_error = ""
        if text_detector is not None and text_prompt.strip():
            try:
                detection = text_detector.detect(
                    frames_tchw_01[int(prompt_frame_idx)],
                    text_prompt,
                    guidance_box_xyxy=None,
                )
                if detection.boxes_xyxy.shape[0] > 0:
                    detected_boxes = detection.boxes_xyxy[:max_objects]
                    prompt_mode = detection.prompt_mode
            except Exception as exc:
                detector_error = f"{type(exc).__name__}: {exc}"
        if detected_boxes is None or detected_boxes.shape[0] == 0:
            motion_multi = build_motion_prompt_boxes(frames_tchw_01, max_boxes=max_objects)
            detected_boxes = motion_multi.boxes_xyxy[:max_objects]
            prompt_mode = motion_multi.prompt_mode
            if detected_boxes.shape[0] == 0:
                raise RuntimeError(
                    "failed to build any multi-object query priors from GroundingDINO or motion fallback; "
                    f"prompt_frame_idx={prompt_frame_idx}, text_prompt={text_prompt!r}, detector_error={detector_error}"
                )

        per_object_queries = []
        object_count = min(int(detected_boxes.shape[0]), int(vggt_num_queries))
        detected_boxes = detected_boxes[:object_count]
        base = vggt_num_queries // max(object_count, 1)
        remainder = max(0, vggt_num_queries - base * object_count)
        for obj_idx, box_xyxy in enumerate(detected_boxes):
            sam_out = sam2_tracker.track(
                frames_tchw_01,
                prompt_frame_idx=prompt_frame_idx,
                prompt_box_xyxy=box_xyxy.astype(np.float32),
                caption="",
            )
            alloc = base + (1 if obj_idx < remainder else 0)
            if alloc <= 0:
                continue
            query_points_px, _ = build_vggt_query_prior(
                sam_out.masks_thw,
                sam_out.boxes_t4,
                num_queries=alloc,
            )
            if query_points_px.shape[0] > 0:
                per_object_queries.append(query_points_px)
        if not per_object_queries:
            raise RuntimeError(
                "SAM2 failed to produce any query priors after GroundingDINO detections; "
                f"prompt_frame_idx={prompt_frame_idx}, text_prompt={text_prompt!r}, detector_error={detector_error}"
            )

        query_points = np.concatenate(per_object_queries, axis=0)[:vggt_num_queries].astype(np.float32)
        if query_points.shape[0] < vggt_num_queries:
            extra = query_points[-1:].repeat(vggt_num_queries - query_points.shape[0], axis=0)
            query_points = np.concatenate([query_points, extra], axis=0)
        prior_source = f"grounded_sam_objects{object_count}"
        return query_points.astype(np.float32), prior_source, f"{prompt_mode}_objects{object_count}", {
            "strategy": sam2_prior_strategy,
            "prompt_text": text_prompt,
            "object_count": object_count,
            "prior_source": prior_source,
            "detector_error": detector_error,
        }

    motion_prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)
    sam_out = sam2_tracker.track(
        frames_tchw_01,
        prompt_frame_idx=prompt_frame_idx,
        prompt_box_xyxy=motion_prompt_box_xyxy,
        caption=caption,
    )
    query_points_px, prior_source = build_vggt_query_prior(
        sam_out.masks_thw,
        sam_out.boxes_t4,
        num_queries=vggt_num_queries,
    )
    return query_points_px, prior_source, sam_out.prompt_mode, {
        "strategy": sam2_prior_strategy,
        "prompt_text": sam_out.prompt_text,
        "object_count": 1,
        "prior_source": prior_source,
    }


def _build_cond_context(
    *,
    trainer: torch.nn.Module,
    config: dict[str, object],
    context_video: torch.Tensor,
    captions: list[str],
    num_context_frames: torch.Tensor,
    device_obj: torch.device,
    context_boxes: torch.Tensor | None = None,
    batch_extra_tensors: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    model_cfg = config["model"]
    data_cfg = config["data"]
    bundle = trainer.bundle
    cond_dim = int(model_cfg.get("cond_proj_dim", bundle.config.text_dim if hasattr(bundle.config, "text_dim") else 4096))

    videos = context_video.to(device_obj)
    captions = list(captions)
    num_context_frames = num_context_frames.to(device_obj).long()
    target_context_frames = int(data_cfg["num_context_frames"])
    if videos.shape[2] < target_context_frames:
        pad_t = target_context_frames - videos.shape[2]
        pad = videos[:, :, -1:].expand(-1, -1, pad_t, -1, -1).contiguous()
        videos = torch.cat([videos, pad], dim=2)

    with torch.no_grad():
        batch = {
            "video": videos,
            "context_video": videos,
            "caption": captions,
            "num_context_frames": num_context_frames,
            "video_path": ["inference_input"],
            "frame_indices": torch.arange(videos.shape[2], dtype=torch.long).unsqueeze(0),
        }
        if context_boxes is not None:
            batch["context_boxes"] = context_boxes.to(device_obj)
        if batch_extra_tensors:
            for key, value in batch_extra_tensors.items():
                batch[key] = value.to(device_obj)
        prepared = trainer._prepare_batch(batch)

    debug = prepared["debug"]
    debug["cond_proj_dim"] = cond_dim
    _print_tensor_stats("context_latents", prepared["context_latents"][0])
    _print_tensor_stats("text_context", prepared["text_context"][0])
    _print_tensor_stats("object_context", prepared["object_context"][0])
    return prepared["text_context"][0], prepared["object_context"][0], prepared["context_latents"][0], debug


def _build_wan_lora_only_context(
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
        "mode": "wan_lora_only",
        "text_context": [list(t.shape) for t in text_context_list],
        "context_latents": [list(t.shape) for t in context_latents_list],
        "context_video": list(videos.shape),
        "wan_lora_checkpoint": str(init_lora_path) if init_lora_path is not None else None,
        "object_branch_initialized": False,
    }
    _print_tensor_stats("context_latents", context_latents)
    _print_tensor_stats("text_context", text_context)
    return bundle, text_context, context_latents, debug


def _run_sampling(
    *,
    bundle: WanContextVideoModel,
    text_context: torch.Tensor,
    object_context: torch.Tensor | None,
    context_latents: torch.Tensor,
    total_frames: int,
    num_context_frames: int,
    num_inference_steps: int,
    disable_object_context: bool = False,
) -> tuple[torch.Tensor, dict[str, object]]:
    assert bundle.dit is not None
    bundle.dit.eval()
    dit_param = next(bundle.dit.parameters())
    dit_dtype = dit_param.dtype
    dit_device = dit_param.device
    context_latents = context_latents.to(device=dit_device, dtype=dit_dtype)
    _print_tensor_stats("vae_encoded_context_latents", context_latents)
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
    sampling_shift = float(getattr(bundle.config, "sample_shift", 5.0))
    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=int(bundle.config.num_train_timesteps),
        shift=1,
        use_dynamic_shifting=False,
    )
    sampling_sigmas = get_sampling_sigmas(int(num_inference_steps), sampling_shift)
    timesteps, _ = retrieve_timesteps(
        scheduler,
        device=dit_device,
        sigmas=sampling_sigmas,
    )
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
    _print_tensor_stats("latent_clean_init", latent_clean)
    _print_tensor_stats("x_t_init", x_t)

    seq_len = x_t.shape[1] * x_t.shape[2] * x_t.shape[3] // (bundle.config.patch_size[1] * bundle.config.patch_size[2])
    text_context = text_context.to(device=dit_device, dtype=dit_dtype)
    if object_context is None:
        object_context_input = None
    else:
        object_context = object_context.to(device=dit_device, dtype=dit_dtype)
        object_context_input = None if disable_object_context else [object_context]
    trajectory_stats = []
    for step_idx, timestep in enumerate(timesteps):
        timestep_f = timestep.to(device=dit_device, dtype=dit_dtype)
        t_tokens = torch.full((1, seq_len), float(timestep_f.item()), device=dit_device, dtype=dit_dtype)
        pred = bundle.dit(
            [x_t],
            t=t_tokens,
            context=None,
            text_context=[text_context],
            object_context=object_context_input,
            seq_len=seq_len,
            y=None,
        )[0]
        _print_tensor_stats(f"pred_step_{step_idx}", pred)
        x_t = scheduler.step(
            pred,
            timestep,
            x_t,
            return_dict=False,
        )[0]
        x_t = context_mask * context_clean_full + (1.0 - context_mask) * x_t
        _print_tensor_stats(f"x_t_step_{step_idx}", x_t)
        trajectory_stats.append(
            {
                "step": int(step_idx),
                "sigma": float(scheduler.sigmas[min(step_idx, len(scheduler.sigmas) - 1)].item()),
                "next_sigma": float(scheduler.sigmas[min(step_idx + 1, len(scheduler.sigmas) - 1)].item()),
                "pred_norm": float(pred.norm().item()),
                "latent_norm": float(x_t.norm().item()),
            }
        )

    target = latent_clean
    denom = future_mask.sum().clamp_min(1.0)
    loss = ((x_t - target) ** 2 * future_mask).sum() / denom
    debug = {
        "latent_clean": list(latent_clean.shape),
        "noise": list(noise.shape),
        "x_t": list(x_t.shape),
        "pred": list(pred.shape),
        "target": list(target.shape),
        "future_mask": list(future_mask.shape),
        "loss": float(loss.item()),
        "seq_len": int(seq_len),
        "sampling_shift": float(sampling_shift),
        "scheduler": type(scheduler).__name__,
        "disable_object_context": bool(disable_object_context),
        "trajectory": trajectory_stats[:5],
    }
    return x_t.detach(), debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-dir",
        dest="checkpoint",
        required=True,
        help="checkpoint folder containing step_*.pt, or a direct step_XXXXXXX.pt / .safetensors file",
    )
    parser.add_argument("--prompt", required=True, help="text prompt for the video")
    parser.add_argument("--context-video", required=True, help="path to the context-only input video; the script reads exactly num_context_frames from it")
    parser.add_argument("--config", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml")
    parser.add_argument("--output-dir", default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/infer_context_video_wan")
    parser.add_argument(
        "--output-video",
        default=None,
        help="optional explicit mp4 output path; if provided, decoding and video export are enabled automatically",
    )
    parser.add_argument("--num-frames", type=int, default=24, help="target total generated video length in frames")
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--context-fraction", type=float, default=0.5)
    parser.add_argument("--random-context-frames", action="store_true")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument(
        "--disable-object-context",
        action="store_true",
        help="Disable object-conditioned cross-attention during sampling for ablation/debug.",
    )
    parser.add_argument(
        "--skip-trainable-checkpoint",
        action="store_true",
        help="Do not load step_*.pt trainable weights; keep trainable modules randomly initialized.",
    )
    parser.add_argument(
        "--wan-lora-only",
        action="store_true",
        help="Do not initialize JEPA/CoTracker/VGGT/object branches or load step_*.pt; run inference with Wan backbone + configured LoRA only.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    config["experiment"]["output_dir"] = str(checkpoint_path)
    device = _resolve_launch_device()
    device_obj = torch.device(device)
    target_total_frames = int(args.num_frames)
    target_context_frames = int(config["data"]["num_context_frames"])
    if target_total_frames < target_context_frames:
        raise ValueError(
            f"--num-frames must be >= configured num_context_frames={target_context_frames}, "
            f"got {target_total_frames}"
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

    if args.wan_lora_only:
        if args.skip_trainable_checkpoint:
            print("wan_lora_only enabled; skip_trainable_checkpoint is implied", flush=True)
        bundle, text_context, context_latents, prep_debug = _build_wan_lora_only_context(
            config=config,
            context_video=context_video.to(device_obj),
            captions=[args.prompt],
            device_obj=device_obj,
        )
        object_context = None
        state_info = {
            "skipped": True,
            "reason": "wan_lora_only",
            "missing_keys": [],
            "unexpected_keys": [],
            "model_state_key_count": 0,
            "checkpoint_key_count": 0,
        }
    else:
        trainer = ContextVideoTrainer(config, build_optimizer=True, device=device)
        print("trainer constructed", flush=True)
        if args.skip_trainable_checkpoint:
            state_info = {
                "skipped": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "model_state_key_count": 0,
                "checkpoint_key_count": 0,
            }
            print("skipping trainable checkpoint load; keeping randomly initialized trainable modules", flush=True)
        else:
            state_info = _load_trainable_state_into_model(trainer, checkpoint_path)
            print(f"checkpoint loaded: missing={len(state_info['missing_keys'])} unexpected={len(state_info['unexpected_keys'])}", flush=True)
        if trainer.bundle.dit is not None:
            trainer.bundle.dit.eval()

        bundle = trainer.bundle
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
            bundle=bundle,
            text_context=text_context,
            object_context=object_context,
            context_latents=context_latents,
            total_frames=target_total_frames,
            num_context_frames=int(num_context_frames.item()),
            num_inference_steps=int(args.sampling_steps),
            disable_object_context=bool(args.disable_object_context),
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
        "load_state_missing": state_info,
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
