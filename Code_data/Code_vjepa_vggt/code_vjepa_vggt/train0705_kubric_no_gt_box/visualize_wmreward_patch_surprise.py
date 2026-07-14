#!/usr/bin/env python3
"""Visualize tokenwise WMReward prediction surprise for three aligned videos."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


DEFAULT_INPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "text_noun_attention_x0_every5_step1000_physiq025_20260714/"
    "train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_"
    "init3500_save500_keepall_20260713T090024Z_step-001000_steps40_512x896_ctx08_"
    "49f_defaultnegprompt/physicIQ_025_Solid_Mechanics_0002_perspective-center_"
    "trimmed_text_noun_attention/predicted_x0"
)
DEFAULT_GT = Path(
    "/data/gaoya/AAA_test_video/0623/testdataset/"
    "025_Solid_Mechanics_0002_perspective-center_trimmed/"
    "physicIQ_0002_clip_2p5s_3p5s.mp4"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "wmreward_patch_surprise_30f_physiq025_x0_remaining35_vs01_vs_gt_20260714"
)
WMREWARD_ROOT = Path("/home/gaoya/Code_Video/WMReward-main")
VJEPA_ROOT = Path("/home/gaoya/Code_Video/vjepa2-main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x0-remaining-35", type=Path, default=DEFAULT_INPUT_ROOT / "pred_x0_remaining_35_h264.mp4")
    parser.add_argument("--x0-remaining-01", type=Path, default=DEFAULT_INPUT_ROOT / "pred_x0_remaining_01_h264.mp4")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x0-title-height", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def install_upstream_paths() -> None:
    for path in (WMREWARD_ROOT, VJEPA_ROOT, VJEPA_ROOT / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def install_optional_diffusers_stub() -> None:
    """WMReward imports export_to_video, but scoring never calls it."""
    try:
        import diffusers.utils  # noqa: F401
    except ModuleNotFoundError:
        diffusers_module = types.ModuleType("diffusers")
        utils_module = types.ModuleType("diffusers.utils")

        def unavailable_export(*_args, **_kwargs):
            raise RuntimeError("diffusers.export_to_video is unavailable in this scoring environment")

        utils_module.export_to_video = unavailable_export
        diffusers_module.utils = utils_module
        sys.modules["diffusers"] = diffusers_module
        sys.modules["diffusers.utils"] = utils_module


def sample_video(path: Path, count: int, crop_top: int) -> tuple[np.ndarray, list[int], list[int]]:
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    indices = np.linspace(0, len(reader) - 1, count, dtype=int)
    frames = reader.get_batch(indices).asnumpy()
    source_hw = [int(frames.shape[1]), int(frames.shape[2])]
    if crop_top:
        frames = frames[:, crop_top:]
    return frames, indices.tolist(), source_hw


def prepare_official_input(frames: np.ndarray, img_size: int) -> tuple[torch.Tensor, np.ndarray]:
    # This matches compute_wmreward.load_video_as_tensor before the official
    # ImageNet normalization performed inside compute_vjepa_loss_sliding_window.
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    tensor = F.interpolate(
        tensor.permute(1, 0, 2, 3),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).permute(1, 0, 2, 3)
    visual = tensor.permute(1, 2, 3, 0).clamp(0, 255).byte().numpy()
    return (tensor / 127.5 - 1.0).unsqueeze(0), visual


def compute_patch_surprise(
    video_tensor: torch.Tensor,
    encoder,
    target_encoder,
    predictor,
    *,
    img_size: int,
    window_size: int,
    context_frames: int,
    stride: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    from src.masks.utils import apply_masks
    from utils import build_pt_video_transform, generate_vjepa_masks

    dtype = next(encoder.parameters()).dtype
    video_tensor = video_tensor.to(device=device, dtype=dtype)
    video_255 = (video_tensor + 1.0) * 127.5
    transform = build_pt_video_transform(img_size)
    normalized = transform(video_255[0].permute(1, 0, 2, 3)).unsqueeze(0).to(dtype)

    pieces = normalized.unfold(2, window_size, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
    pieces = rearrange(pieces.flatten(0, 1), "b t c h w -> b c t h w")
    grid = img_size // encoder.patch_size
    tubelet = int(encoder.tubelet_size)
    total_t = video_tensor.shape[2] // tubelet
    surprise_sum = torch.zeros(total_t, grid, grid, device=device, dtype=torch.float32)
    surprise_count = torch.zeros(total_t, grid, grid, device=device, dtype=torch.float32)
    official_chunk_losses = []
    windows = []

    with torch.inference_mode():
        for chunk_id, chunk in enumerate(pieces):
            chunk = chunk.unsqueeze(0)
            masks_enc, masks_pred = generate_vjepa_masks(
                masking_mode="causal",
                batch_size=1,
                img_size=img_size,
                frames_per_clip=window_size,
                encoder=encoder,
                context_frames=context_frames,
                device=device,
                seed=seed + chunk_id,
            )
            h = target_encoder(chunk)
            h = torch.stack([F.layer_norm(item, (item.size(-1),)) for item in h])
            z = encoder(chunk, masks_enc)
            z = predictor(z, masks_enc, masks_pred)
            z = F.layer_norm(z, (z.size(-1),)).to(h.device)
            target = apply_masks(h, masks_pred, concat=False)[0]

            # Preserve the exact scalar expression in the official function.
            official_loss = 1.0 - F.cosine_similarity(z, target, dim=1).mean()
            official_chunk_losses.append(official_loss.float())

            # Localize the same prediction mismatch per target token. The official
            # scalar uses dim=1; tokenwise maps necessarily use embedding dim=-1.
            token_surprise = 1.0 - F.cosine_similarity(z, target, dim=-1)
            token_surprise = token_surprise[0].float()
            future_depth = (window_size - context_frames) // tubelet
            token_surprise = token_surprise.view(future_depth, grid, grid)
            window_start_frame = chunk_id * stride
            global_start_t = (window_start_frame + context_frames) // tubelet
            global_end_t = global_start_t + future_depth
            surprise_sum[global_start_t:global_end_t] += token_surprise
            surprise_count[global_start_t:global_end_t] += 1
            windows.append(
                {
                    "window_start_frame": window_start_frame,
                    "window_end_frame_exclusive": window_start_frame + window_size,
                    "target_frame_range": [window_start_frame + context_frames, window_start_frame + window_size - 1],
                    "official_chunk_surprise": float(official_loss.item()),
                }
            )

    valid = surprise_count > 0
    patch_surprise = torch.full_like(surprise_sum, torch.nan)
    patch_surprise[valid] = surprise_sum[valid] / surprise_count[valid]
    metadata = {
        "official_surprise_mean": float(torch.stack(official_chunk_losses).mean().item()),
        "official_similarity": float(1.0 - torch.stack(official_chunk_losses).mean().item()),
        "windows": windows,
        "grid_t_h_w": [total_t, grid, grid],
        "tubelet_size": tubelet,
    }
    return patch_surprise.cpu().numpy(), metadata


def header(frame: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(frame, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(canvas, text, (9, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def write_h264(path: Path, frames: list[np.ndarray], fps: float = 15.0) -> None:
    height, width = frames[0].shape[:2]
    with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
        intermediate = Path(temp_dir) / "intermediate.mp4"
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open {intermediate}")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        ffmpeg = shutil.which("ffmpeg") or "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(intermediate), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
            check=True,
        )


def render_video(
    name: str,
    visual_input: np.ndarray,
    surprise: np.ndarray,
    *,
    shared_max: float,
    tubelet_size: int,
    context_frames: int,
    windows: list[dict],
    output_dir: Path,
) -> tuple[Path, dict, list[dict]]:
    finite = np.isfinite(surprise)
    flat_index = int(np.nanargmax(surprise))
    max_t, max_y, max_x = np.unravel_index(flat_index, surprise.shape)
    patch_h = visual_input.shape[1] // surprise.shape[1]
    patch_w = visual_input.shape[2] // surprise.shape[2]
    box = [max_x * patch_w, max_y * patch_h, (max_x + 1) * patch_w, (max_y + 1) * patch_h]
    segments = [
        {
            "name": "context",
            "frame_start": 0,
            "frame_end": context_frames - 1,
            "score_status": "not_scored_context",
            "official_window_surprise": None,
            "patch_surprise_mean": None,
        }
    ]
    for window_index, window in enumerate(windows):
        start, end = window["target_frame_range"]
        token_start = start // tubelet_size
        token_end = (end + 1) // tubelet_size
        segments.append(
            {
                "name": f"target_window_{window_index}",
                "frame_start": start,
                "frame_end": end,
                "score_status": "scored_target",
                "official_window_surprise": window["official_chunk_surprise"],
                "patch_surprise_mean": float(np.nanmean(surprise[token_start:token_end])),
            }
        )
    last_scored_frame = max(segment["frame_end"] for segment in segments)
    if last_scored_frame + 1 < len(visual_input):
        segments.append(
            {
                "name": "unscored_tail",
                "frame_start": last_scored_frame + 1,
                "frame_end": len(visual_input) - 1,
                "score_status": "not_scored_incomplete_window",
                "official_window_surprise": None,
                "patch_surprise_mean": None,
            }
        )

    def segment_for_frame(frame_index: int) -> dict:
        return next(
            segment
            for segment in segments
            if segment["frame_start"] <= frame_index <= segment["frame_end"]
        )

    frames = []
    for frame_index, original in enumerate(visual_input):
        token_t = min(frame_index // tubelet_size, surprise.shape[0] - 1)
        patch_map = surprise[token_t]
        if np.isfinite(patch_map).any():
            normalized = np.clip(np.nan_to_num(patch_map) / shared_max, 0.0, 1.0)
            heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            heat = cv2.cvtColor(
                cv2.resize(heat, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_BGR2RGB,
            )
            overlay = cv2.addWeighted(original, 0.55, heat, 0.45, 0)
            status = f"t{token_t:02d} | mean {np.nanmean(patch_map):.4f}"
        else:
            heat = np.full_like(original, 225)
            overlay = original.copy()
            status = "context/unscored"
        if token_t == max_t:
            for panel in (heat, overlay):
                cv2.rectangle(panel, (box[0], box[1]), (box[2] - 1, box[3] - 1), (255, 255, 255), 3)
        panels = [
            header(original, f"{name} | WMReward input | f{frame_index:02d}"),
            header(heat, f"patch surprise | {status}"),
            header(overlay, f"overlay | scale [0,{shared_max:.4f}]"),
        ]
        composite = np.concatenate(panels, axis=1)
        segment = segment_for_frame(frame_index)
        if segment["official_window_surprise"] is None:
            segment_text = (
                f"segment={segment['name']} | frames {segment['frame_start']:02d}-{segment['frame_end']:02d} "
                f"| {segment['score_status']}"
            )
        else:
            segment_text = (
                f"segment={segment['name']} | frames {segment['frame_start']:02d}-{segment['frame_end']:02d} "
                f"| official={segment['official_window_surprise']:.6f} "
                f"| patch_mean={segment['patch_surprise_mean']:.6f}"
            )
        frames.append(header(composite, segment_text))
    path = output_dir / f"{name}_wmreward_patch_surprise_overlay_h264.mp4"
    write_h264(path, frames)
    segment_dir = output_dir / "segments" / name
    segment_dir.mkdir(parents=True, exist_ok=True)
    for segment in segments:
        segment_path = segment_dir / (
            f"{segment['frame_start']:02d}-{segment['frame_end']:02d}_{segment['name']}_h264.mp4"
        )
        write_h264(
            segment_path,
            frames[segment["frame_start"] : segment["frame_end"] + 1],
        )
        segment["video_path"] = str(segment_path)
    maximum = {
        "surprise": float(surprise[max_t, max_y, max_x]),
        "token_t_y_x": [int(max_t), int(max_y), int(max_x)],
        "wmreward_input_pixel_box_xyxy": [int(value) for value in box],
        "sampled_video_frame_indices": list(range(max_t * tubelet_size, (max_t + 1) * tubelet_size)),
        "finite_patch_count": int(finite.sum()),
    }
    return path, maximum, segments


def main() -> None:
    args = parse_args()
    if args.num_frames < args.window_size:
        raise ValueError("num_frames must be at least window_size")
    for path in (args.x0_remaining_35, args.x0_remaining_01, args.ground_truth):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    install_upstream_paths()
    install_optional_diffusers_stub()
    from utils import load_vjepa_model_source

    device = torch.device(args.device)
    cwd = Path.cwd()
    os.chdir(WMREWARD_ROOT)
    try:
        encoder, target_encoder, predictor, img_size = load_vjepa_model_source("vitg384")
    finally:
        os.chdir(cwd)
    encoder = encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    specifications = {
        "x0_remaining_35": (args.x0_remaining_35, args.x0_title_height),
        "x0_remaining_01": (args.x0_remaining_01, args.x0_title_height),
        "ground_truth": (args.ground_truth, 0),
    }
    maps = {}
    visuals = {}
    records = {}
    for name, (path, crop_top) in specifications.items():
        frames, indices, source_hw = sample_video(path, args.num_frames, crop_top)
        tensor, visual = prepare_official_input(frames, img_size)
        patch_map, wmreward = compute_patch_surprise(
            tensor,
            encoder,
            target_encoder,
            predictor,
            img_size=img_size,
            window_size=args.window_size,
            context_frames=args.context_frames,
            stride=args.stride,
            seed=args.seed,
            device=device,
        )
        maps[name] = patch_map
        visuals[name] = visual
        records[name] = {
            "path": str(path),
            "source_hw": source_hw,
            "crop_top": crop_top,
            "sampled_source_frame_indices": indices,
            **wmreward,
        }
        print(f"[computed] {name}: official_surprise={wmreward['official_surprise_mean']:.6f}")

    all_finite = np.concatenate([value[np.isfinite(value)] for value in maps.values()])
    shared_max = max(float(np.quantile(all_finite, 0.99)), 1.0e-8)
    visualizations = {}
    for name in specifications:
        path, maximum, segments = render_video(
            name,
            visuals[name],
            maps[name],
            shared_max=shared_max,
            tubelet_size=records[name]["tubelet_size"],
            context_frames=args.context_frames,
            windows=records[name]["windows"],
            output_dir=args.output_dir,
        )
        maximum["source_frame_indices"] = [
            records[name]["sampled_source_frame_indices"][index]
            for index in maximum["sampled_video_frame_indices"]
        ]
        records[name]["maximum_patch_surprise"] = maximum
        records[name]["segments"] = segments
        visualizations[name] = str(path)

    segment_rows = []
    for name in specifications:
        for segment in records[name]["segments"]:
            segment_rows.append(
                {
                    "video": name,
                    "segment": segment["name"],
                    "frame_start": segment["frame_start"],
                    "frame_end": segment["frame_end"],
                    "score_status": segment["score_status"],
                    "official_window_surprise": segment["official_window_surprise"],
                    "patch_surprise_mean": segment["patch_surprise_mean"],
                    "video_path": segment["video_path"],
                }
            )
    with (args.output_dir / "segment_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(segment_rows[0]))
        writer.writeheader()
        writer.writerows(segment_rows)

    np.savez_compressed(
        args.output_dir / "patch_surprise_maps_fp16.npz",
        **{name: np.nan_to_num(value, nan=-1).astype(np.float16) for name, value in maps.items()},
    )
    result = {
        "method": {
            "reference": str(WMREWARD_ROOT / "utils.py") + ":compute_vjepa_loss_sliding_window",
            "model": "vitg384",
            "checkpoint": "/data/gaoya/ckpt/Sylvest-vjepa2-vit-g/vitg-384.pt",
            "num_frames": args.num_frames,
            "window_size": args.window_size,
            "context_frames": args.context_frames,
            "stride": args.stride,
            "masking_mode": "causal",
            "seed": args.seed,
            "patch_surprise": "1 - cosine(predicted_token, target_token), embedding dim=-1",
            "official_scalar": "unchanged upstream expression using cosine dim=1",
            "shared_visual_scale": [0.0, shared_max],
        },
        "videos": records,
        "visualizations": visualizations,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
