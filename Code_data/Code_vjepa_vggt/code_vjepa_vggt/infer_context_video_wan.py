from __future__ import annotations

import argparse
import json
import math
import shutil
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from code_vjepa_vggt.models.wan_context_model import WanContextVideoModel
from code_vjepa_vggt.training.flow_match import WanFlowMatchScheduler
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import broadcast_latent_mask, expand_context_latents_to_full, latent_frame_mask
from code_vjepa_vggt.utils.paths import ensure_upstream_paths
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix, read_video_uniform

ensure_upstream_paths()

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.adapters.jepa_adapter import JEPAPatchAdapter
from code_vjepa_vggt.adapters.sam2_motion import GroundingDINOTextDetector, SAM2MotionTracker, build_motion_prompt_box
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


def _tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def _write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _ensure_browser_video(source_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
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


def _load_trainable_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    candidates = sorted(checkpoint_dir.glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no step_*.pt found under {checkpoint_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    state = torch.load(latest, map_location="cpu")
    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if isinstance(state, dict):
        return state
    raise RuntimeError(f"unsupported checkpoint format in {latest}")


def _load_trainable_state_into_model(model: torch.nn.Module, checkpoint_dir: Path) -> dict[str, object]:
    state_dict = _load_trainable_state(checkpoint_dir)
    missing = model.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": list(missing.missing_keys),
        "unexpected_keys": list(missing.unexpected_keys),
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
        return None, "no_sam2", "no_sam2", {"used_fallback": False}

    if sam2_prior_strategy in {"grounded_text_multi", "text_multi", "grounded_text"}:
        max_objects = 4
        text_prompt = _build_multi_object_prompt(caption)
        detected_boxes = None
        prompt_mode = "caption_gdino_multi"
        used_fallback = False
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
                used_fallback = True
                detector_error = f"{type(exc).__name__}: {exc}"
        if detected_boxes is None or detected_boxes.shape[0] == 0:
            used_fallback = True
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
                "used_fallback": used_fallback,
                "prior_source": prior_source,
                "detector_error": detector_error,
            }

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
            return query_points_px, prior_source, "grounded_text_empty_fallback", {
                "strategy": sam2_prior_strategy,
                "prompt_text": text_prompt,
                "object_count": 0,
                "used_fallback": True,
                "prior_source": prior_source,
                "detector_error": detector_error,
            }

        query_points = np.concatenate(per_object_queries, axis=0)[:vggt_num_queries].astype(np.float32)
        if query_points.shape[0] < vggt_num_queries:
            extra = query_points[-1:].repeat(vggt_num_queries - query_points.shape[0], axis=0)
            query_points = np.concatenate([query_points, extra], axis=0)
        prior_source = f"grounded_sam_objects{object_count}"
        return query_points.astype(np.float32), prior_source, f"{prompt_mode}_objects{object_count}", {
            "strategy": sam2_prior_strategy,
            "prompt_text": text_prompt,
            "object_count": object_count,
            "used_fallback": used_fallback,
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
        "used_fallback": bool("fallback" in sam_out.prompt_mode or sam_out.prompt_mode.startswith("proxy_box")),
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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
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
        prepared = trainer._prepare_batch(batch)

    debug = prepared["debug"]
    debug["cond_proj_dim"] = cond_dim
    return prepared["fused_context"][0], prepared["context_latents"][0], debug


def _run_sampling(
    *,
    bundle: WanContextVideoModel,
    fused_context: torch.Tensor,
    context_latents: torch.Tensor,
    total_frames: int,
    num_context_frames: int,
    num_inference_steps: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    assert bundle.dit is not None
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
    scheduler = WanFlowMatchScheduler(num_train_timesteps=bundle.config.num_train_timesteps)
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
    fused_context = fused_context.to(device=dit_device, dtype=dit_dtype)
    trajectory_stats = []
    for step_idx, sigma in enumerate(scheduler.sigmas):
        timestep = scheduler.timesteps[step_idx].to(device=dit_device, dtype=dit_dtype)
        t_tokens = torch.full((1, seq_len), float(timestep.item()), device=dit_device, dtype=dit_dtype)
        pred = bundle.dit([x_t], t=t_tokens, context=[fused_context], seq_len=seq_len, y=None)[0]
        next_sigma = scheduler.sigmas[step_idx + 1] if step_idx + 1 < len(scheduler.sigmas) else torch.tensor(0.0)
        next_sigma = next_sigma.to(device=dit_device, dtype=dit_dtype)
        sigma = sigma.to(device=dit_device, dtype=dit_dtype)
        x_t = x_t + (next_sigma - sigma) * pred
        x_t = context_mask * context_clean_full + (1.0 - context_mask) * x_t
        trajectory_stats.append(
            {
                "step": int(step_idx),
                "sigma": float(sigma.item()),
                "next_sigma": float(next_sigma.item()),
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
        "trajectory": trajectory_stats[:5],
    }
    return x_t.detach(), debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="checkpoint folder containing step_*.pt")
    parser.add_argument("--prompt", required=True, help="text prompt for the video")
    parser.add_argument("--context-video", required=True, help="path to input context video")
    parser.add_argument("--config", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml")
    parser.add_argument("--output-dir", default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/infer_context_video_wan")
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
    config["experiment"]["output_dir"] = str(Path(args.checkpoint_dir))
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

    trainer = ContextVideoTrainer(
        ckpt_dir=str(config["model"]["wan_ckpt_dir"]),
        task=str(config["model"]["wan_task"]),
        device=device,
        build_optimizer=False,
        lora_rank=int(config["model"].get("wan_lora_rank", 0)),
        lora_alpha=int(config["model"].get("wan_lora_alpha", 0)),
        lora_dropout=float(config["model"].get("wan_lora_dropout", 0.0)),
        lora_init=str(config["model"].get("wan_lora_init", "gaussian")),
    )
    state_info = _load_trainable_state_into_model(trainer, Path(args.checkpoint_dir))

    fused_context, context_latents, prep_debug = _build_cond_context(
        trainer=trainer,
        config=config,
        context_video=context_video.to(device_obj),
        captions=[args.prompt],
        num_context_frames=num_context_frames,
        device_obj=device_obj,
    )
    pred, sample_debug = _run_sampling(
        bundle=trainer.bundle,
        fused_context=fused_context,
        context_latents=context_latents,
        total_frames=int(video.shape[1]),
        num_context_frames=int(num_context_frames.item()),
        num_inference_steps=int(args.sampling_steps),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "context_video": str(args.context_video),
        "prompt": str(args.prompt),
        "frame_indices": frame_indices.tolist(),
        "context_indices": context_indices.tolist(),
        "prep_debug": prep_debug,
        "sample_debug": sample_debug,
        "load_state_missing": state_info,
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if args.save_raw:
        with torch.no_grad():
            decoded = trainer.bundle.vae.decode([pred.to(next(trainer.bundle.vae.model.parameters()).device if hasattr(trainer.bundle.vae, "model") else device_obj)])
        if isinstance(decoded, list):
            decoded = decoded[0]
        video_out = decoded.detach().cpu()
        video_out = video_out.permute(1, 0, 2, 3).contiguous()
        video_out = ((video_out.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        raw_path = output_dir / "prediction.mp4"
        _write_mp4(raw_path, video_out, fps=int(args.fps))
        browser_path = _ensure_browser_video(raw_path)
        result["prediction_video"] = str(browser_path)
        with open(output_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
