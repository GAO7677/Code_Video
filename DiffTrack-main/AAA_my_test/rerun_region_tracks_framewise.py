#!/usr/bin/env python3
"""Rerun region tracking with DiffTrack's frame-wise VAE protocol."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "diffusers" / "src"))
sys.path.insert(0, str(REPO_ROOT))

import diffusers
from diffusers import CogVideoXTrackPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler
from utils.matching import corr_to_matches

from AAA_my_test.analyze_region_tracks import (
    compute_metrics,
    draw_comparison_video,
    draw_mask_points,
    draw_track_video,
    read_video,
)


DEFAULT_OLD_RESULT = Path(
    "/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks/"
    "case_019_wheel_hits_block_base/layer17_step49"
)
DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise")
DEFAULT_MODEL = Path("/data/gaoya/agent-data/weights/CogVideoX-2b-modelscope")
REGIONS = ("object_a", "object_b", "background")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, default=DEFAULT_OLD_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--matching-timestep", type=int, default=49)
    parser.add_argument("--inverse-step", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--chunk-len", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--trace-length", type=int, default=12)
    return parser.parse_args()


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.manual_seed(seed)


def sample_dense_mapping(correlation: torch.Tensor, points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    h, w = height // 16, width // 16
    queried_coords = points / 16
    margin = width / (64 * 16)
    norm_coords = queried_coords.clone()
    norm_coords[:, 0] = (norm_coords[:, 0] / (w - margin)) * 2 - 1.0
    norm_coords[:, 1] = (norm_coords[:, 1] / (h - margin)) * 2 - 1.0

    x_source, y_source, _, _, _ = corr_to_matches(
        correlation.view(1, h, w, h, w).unsqueeze(1),
        get_maximum=True,
        do_softmax=True,
        device=correlation.device,
    )
    mapping = torch.cat((x_source.unsqueeze(-1), y_source.unsqueeze(-1)), dim=-1)
    mapping = mapping.view(1, h, w, 2).permute(0, 3, 1, 2)
    grid = norm_coords.view(1, len(points), 1, 2)
    track = F.grid_sample(mapping, grid=grid, mode="bilinear", align_corners=True)
    return track[0, :, :, 0].transpose(0, 1) * 16


def extract_framewise_tracks(
    pipe: CogVideoXTrackPipeline,
    video: torch.Tensor,
    points: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict]]:
    _, height, width = video.shape[1:]
    frame_count = video.shape[0]
    target_per_chunk = args.chunk_len - 1
    trajectory = torch.empty((frame_count, len(points), 2), dtype=torch.float32)
    trajectory[0] = points.cpu()
    chunk_records = []
    generator = seed_everything(args.seed)
    params = {
        "trajectory": False,
        "attn_weight": False,
        "query_key": True,
        "head_matching_layer": -1,
        "matching_layer": [args.layer],
    }

    for chunk_index, start in enumerate(range(1, frame_count, target_per_chunk)):
        target_indices = list(range(start, min(start + target_per_chunk, frame_count)))
        chunk_indices = [0, *target_indices]
        chunk_video = video[chunk_indices].unsqueeze(0) / 255.0
        with torch.inference_mode():
            _, queries, keys, text_seq_length = pipe(
                height=height,
                width=width,
                prompt="",
                guidance_scale=6,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                video=chunk_video,
                frame_as_latent=True,
                inverse_step=args.inverse_step,
                matching_timestep=[args.matching_timestep],
                matching_layer=[args.layer],
                add_noise=False,
                output_type="latent",
                return_dict=False,
                params=params,
            )
        if len(queries) != 1 or len(keys) != 1:
            raise RuntimeError(f"Expected one Q/K tensor, got {len(queries)} and {len(keys)}")

        query = queries[0]
        key = keys[0]
        h, w = height // 16, width // 16
        frame_tokens = h * w
        visual_tokens = query.shape[1] - text_seq_length
        if visual_tokens % frame_tokens:
            raise RuntimeError(f"Visual token count {visual_tokens} is not divisible by {frame_tokens}")
        latent_frames = visual_tokens // frame_tokens
        if latent_frames != len(chunk_indices):
            raise RuntimeError(
                f"Frame-wise VAE invariant failed: {len(chunk_indices)} pixel frames -> {latent_frames} latent frames"
            )

        query_frames = rearrange(
            query[:, text_seq_length:],
            "head (frame h w) channel -> head frame (h w) channel",
            frame=latent_frames,
            h=h,
            w=w,
        ).unsqueeze(0)
        key_frames = rearrange(
            key[:, text_seq_length:],
            "head (frame h w) channel -> head frame (h w) channel",
            frame=latent_frames,
            h=h,
            w=w,
        ).unsqueeze(0)

        # This intentionally follows evaluate_tapvid.py, including its sqrt(num_heads) scaling.
        scale = math.sqrt(query_frames.shape[1])
        for local_index, frame_index in enumerate(target_indices, start=1):
            forward = torch.einsum(
                "b h i d, b h j d -> b h i j",
                query_frames[:, :, 0],
                key_frames[:, :, local_index],
            ) / scale
            reverse = torch.einsum(
                "b h i d, b h j d -> b h i j",
                query_frames[:, :, local_index],
                key_frames[:, :, 0],
            ) / scale
            forward = forward.softmax(dim=-1).mean(dim=1)
            reverse = reverse.softmax(dim=-1).mean(dim=1)
            correlation = rearrange(forward, "b (h w) target -> b target h w", h=h, w=w)
            reverse = rearrange(reverse, "b target (h w) -> b target h w", h=h, w=w)
            correlation = (correlation + reverse) / 2
            trajectory[frame_index] = sample_dense_mapping(correlation, points.to(correlation.device), height, width).cpu()
            del forward, reverse, correlation

        chunk_records.append(
            {
                "chunk_index": chunk_index,
                "pixel_frame_indices": chunk_indices,
                "pixel_frame_count": len(chunk_indices),
                "latent_frame_count": latent_frames,
            }
        )
        del query, key, query_frames, key_frames, queries, keys
        torch.cuda.empty_cache()
        print(f"chunk {chunk_index}: pixel frames {chunk_indices} -> {latent_frames} latent frames")

    return trajectory, chunk_records


def main() -> None:
    args = parse_args()
    expected_diffusers = REPO_ROOT / "diffusers" / "src" / "diffusers"
    if Path(diffusers.__file__).resolve().parent != expected_diffusers:
        raise RuntimeError(f"Expected DiffTrack diffusers at {expected_diffusers}, loaded {diffusers.__file__}")
    if args.chunk_len < 2:
        raise ValueError("chunk_len must include frame 0 and at least one target frame")
    if args.inverse_step > args.matching_timestep:
        raise ValueError("inverse_step must not exceed matching_timestep")

    old_manifest = json.loads((args.source_result / "run_manifest.json").read_text())
    video_path = Path(old_manifest["video"])
    video = read_video(video_path, 49, 480, 720)
    frames = video.permute(0, 2, 3, 1).byte().numpy()

    region_data = {}
    all_points = []
    offset = 0
    for region in REGIONS:
        old = np.load(args.source_result / region / "tracks.npz")
        points = old["query_points"].astype(np.float32)
        region_data[region] = {
            "points": points,
            "mask": old["region_mask"].astype(bool),
            "cotracker": old["cotracker_tracks"].astype(np.float32),
            "visibility": old["cotracker_visibility"].astype(bool),
            "slice": slice(offset, offset + len(points)),
        }
        all_points.append(points)
        offset += len(points)
    all_points_tensor = torch.from_numpy(np.concatenate(all_points, axis=0))

    pipe = CogVideoXTrackPipeline.from_pretrained(str(args.model_path), torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device=args.device, dtype=torch.bfloat16)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    trajectory, chunk_records = extract_framewise_tracks(pipe, video, all_points_tensor, args)

    output_root = (
        args.output_dir
        / f"{old_manifest['case_key']}_{old_manifest['sample_type']}"
        / f"layer{args.layer}_step{args.matching_timestep}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    result_manifest = {
        **old_manifest,
        "protocol": "framewise_vae_chunked",
        "source_temporal_vae_result": str(args.source_result),
        "model_pipeline": "CogVideoXTrackPipeline",
        "scheduler_timestep_spacing": "trailing",
        "frame_as_latent": True,
        "temporal_interpolation": False,
        "chunk_len": args.chunk_len,
        "chunk_records": chunk_records,
        "layer": args.layer,
        "inverse_step": args.inverse_step,
        "matching_timestep": args.matching_timestep,
        "regions": {},
    }

    labels = {
        "object_a": "object A: driver_0",
        "object_b": "object B: target_0",
        "background": "background",
    }
    for region, data in region_data.items():
        region_dir = output_root / region
        region_dir.mkdir(parents=True, exist_ok=True)
        qk = trajectory[:, data["slice"]].numpy()
        metrics = compute_metrics(qk, data["cotracker"], data["visibility"])
        np.savez_compressed(
            region_dir / "tracks.npz",
            query_points=data["points"],
            region_mask=data["mask"],
            cotracker_tracks=data["cotracker"],
            cotracker_visibility=data["visibility"],
            qk_tracks=qk,
        )
        (region_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        draw_mask_points(frames[0], data["mask"], data["points"], labels[region], region_dir / "mask_points.png")
        draw_track_video(
            frames, data["cotracker"], region_dir / "cotracker_tracks.mp4", labels[region], "CoTracker",
            args.fps, args.trace_length, data["visibility"],
        )
        draw_track_video(
            frames, qk, region_dir / "qk_tracks.mp4", labels[region], "Q/K frame-wise VAE",
            args.fps, args.trace_length,
        )
        draw_comparison_video(
            frames, data["cotracker"], qk, data["visibility"], region_dir / "overlay_comparison.mp4",
            labels[region], args.fps, args.trace_length,
        )
        result_manifest["regions"][region] = {
            "label": labels[region],
            "valid_mask_pixels": int(data["mask"].sum()),
            "metrics": metrics,
        }
        print(f"{region}: PCK@8={metrics['pck8']:.3f}, mean error={metrics['mean_error_px']:.3f}px")

    (output_root / "run_manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n")
    comparison = {}
    for region in REGIONS:
        old_metrics = json.loads((args.source_result / region / "metrics.json").read_text())
        new_metrics = result_manifest["regions"][region]["metrics"]
        comparison[region] = {
            "temporal_vae_interpolated": {
                "pck8": old_metrics["pck8"],
                "mean_error_px": old_metrics["mean_error_px"],
            },
            "framewise_vae": {
                "pck8": new_metrics["pck8"],
                "mean_error_px": new_metrics["mean_error_px"],
            },
            "pck8_delta": new_metrics["pck8"] - old_metrics["pck8"],
            "mean_error_delta_px": new_metrics["mean_error_px"] - old_metrics["mean_error_px"],
        }
    (output_root / "protocol_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"Frame-wise results saved to {output_root}")


if __name__ == "__main__":
    main()
