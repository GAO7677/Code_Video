#!/usr/bin/env python3
"""Scan Wan2.2-TI2V-5B layers and noise steps for region-level Q/K motion matching."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

from AAA_my_test.wan_motion_utils import (
    NUM_FRAMES,
    OUTPUT_ROOT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TOKEN_STRIDE,
    WAN_CHECKPOINT,
    WAN_ROOT,
    atomic_write_json,
    compute_rigidity_error,
    compute_track_metrics,
    free_space_gib,
    read_video,
)


DEFAULT_LAYERS = [0, 5, 11, 17, 23, 29]
DEFAULT_STEP_INDICES = [0, 12, 24, 36, 49]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-dir", type=Path, default=OUTPUT_ROOT / "tracks_base")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "batch_base" / "worker_0")
    parser.add_argument("--model-path", type=Path, default=WAN_CHECKPOINT)
    parser.add_argument("--wan-root", type=Path, default=WAN_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--step-indices", type=int, nargs="+", default=DEFAULT_STEP_INDICES)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-key")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--convert-model-dtype", action="store_true", default=True)
    return parser.parse_args()


def seed_everything(seed: int, device: str) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


class QKTrackCapture:
    """Convert post-RoPE Q/K into compact tracks while each attention layer is live."""

    def __init__(self, layers: list[int], query_points: np.ndarray, target_tracks: np.ndarray):
        self.layers = set(layers)
        self.query_points = torch.from_numpy(query_points).float()
        self.target_tracks = torch.from_numpy(target_tracks).float()
        self.results: dict[int, dict[str, np.ndarray]] = {}
        self.grid_size: tuple[int, int, int] | None = None

    @staticmethod
    def _point_indices(points: torch.Tensor, height: int, width: int, device: torch.device) -> torch.Tensor:
        points = points.to(device)
        x = torch.floor(points[:, 0] / TOKEN_STRIDE).long().clamp(0, width - 1)
        y = torch.floor(points[:, 1] / TOKEN_STRIDE).long().clamp(0, height - 1)
        return y * width + x

    def consume(self, layer: int, q: torch.Tensor, k: torch.Tensor, grid_sizes: torch.Tensor) -> None:
        if q.shape[0] != 1 or grid_sizes.shape != (1, 3):
            raise ValueError(f"Only batch size one is supported, got q={tuple(q.shape)}, grid={tuple(grid_sizes.shape)}")
        time, height, width = [int(value) for value in grid_sizes[0].tolist()]
        sequence = time * height * width
        if sequence > q.shape[1] or q.shape != k.shape:
            raise ValueError(f"Invalid Q/K geometry: q={tuple(q.shape)}, k={tuple(k.shape)}, grid={(time,height,width)}")
        if self.target_tracks.shape[0] != time:
            raise ValueError(f"CoTracker has {self.target_tracks.shape[0]} anchors but Wan grid has {time} frames")
        self.grid_size = (time, height, width)

        q_frames = q[0, :sequence].view(time, height * width, q.shape[2], q.shape[3])
        k_frames = k[0, :sequence].view(time, height * width, k.shape[2], k.shape[3])
        source_indices = self._point_indices(self.query_points, height, width, q.device)
        source_q = q_frames[0, source_indices].float()
        source_k = k_frames[0, source_indices].float()
        scale = math.sqrt(q.shape[-1])

        point_count = len(source_indices)
        predictions = torch.empty((time, point_count, 2), dtype=torch.float32)
        gt_probability = torch.full((time, point_count), float("nan"), dtype=torch.float32)
        gt_rank = torch.full_like(gt_probability, float("nan"))
        margin = torch.full_like(gt_probability, float("nan"))
        entropy = torch.full_like(gt_probability, float("nan"))
        predictions[0] = self.query_points

        for target_index in range(1, time):
            target_q = q_frames[target_index].float()
            target_k = k_frames[target_index].float()
            forward = torch.einsum("pnd,snd->nps", source_q, target_k) / scale
            role_reverse = torch.einsum("pnd,snd->nps", source_k, target_q) / scale
            probability = 0.5 * (
                forward.softmax(dim=-1).mean(dim=0) + role_reverse.softmax(dim=-1).mean(dim=0)
            )
            best_probability, best_index = probability.max(dim=-1)
            best_y = torch.div(best_index, width, rounding_mode="floor")
            best_x = best_index % width
            predictions[target_index, :, 0] = (best_x.cpu().float() + 0.5) * TOKEN_STRIDE
            predictions[target_index, :, 1] = (best_y.cpu().float() + 0.5) * TOKEN_STRIDE

            gt_indices = self._point_indices(self.target_tracks[target_index], height, width, q.device)
            gt_values = probability.gather(1, gt_indices[:, None]).squeeze(1)
            gt_probability[target_index] = gt_values.cpu()
            gt_rank[target_index] = (probability > gt_values[:, None]).sum(dim=1).cpu().float() + 1
            second = probability.topk(k=2, dim=-1).values[:, 1]
            margin[target_index] = (best_probability - second).cpu()
            entropy[target_index] = (-(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)).cpu()
            del target_q, target_k, forward, role_reverse, probability

        self.results[layer] = {
            "predictions": predictions.numpy(),
            "gt_probability": gt_probability.numpy(),
            "gt_rank": gt_rank.numpy(),
            "margin": margin.numpy(),
            "entropy": entropy.numpy(),
        }


def install_qk_capture(model, capture: QKTrackCapture, layers: list[int]):
    from wan.modules.model import flash_attention, rope_apply

    originals = {}
    for layer in layers:
        attention = model.blocks[layer].self_attn
        originals[layer] = attention.forward

        def captured_forward(self, x, seq_lens, grid_sizes, freqs, *, layer_index=layer):
            batch, sequence = x.shape[:2]
            q = self.norm_q(self.q(x)).view(batch, sequence, self.num_heads, self.head_dim)
            k = self.norm_k(self.k(x)).view(batch, sequence, self.num_heads, self.head_dim)
            v = self.v(x).view(batch, sequence, self.num_heads, self.head_dim)
            q_rope = rope_apply(q, grid_sizes, freqs)
            k_rope = rope_apply(k, grid_sizes, freqs)
            capture.consume(layer_index, q_rope, k_rope, grid_sizes)
            output = flash_attention(
                q=q_rope,
                k=k_rope,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
            )
            return self.o(output.flatten(2))

        attention.forward = types.MethodType(captured_forward, attention)
    return originals


def restore_qk_capture(model, originals: dict[int, object]) -> None:
    for layer, forward in originals.items():
        model.blocks[layer].self_attn.forward = forward


def encode_prompt(pipeline, prompt: str, device: torch.device) -> list[torch.Tensor]:
    context = pipeline.text_encoder([prompt], torch.device("cpu"))
    return [value.to(device) for value in context]


def load_pipeline(args: argparse.Namespace):
    sys.path.insert(0, str(args.wan_root))
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()

    if not getattr(WanModel, "_motion_probe_low_cpu_patch", False):
        original = WanModel.from_pretrained.__func__

        def patched(cls, name_or_path, *positional, **kwargs):
            kwargs.setdefault("low_cpu_mem_usage", False)
            return original(cls, name_or_path, *positional, **kwargs)

        WanModel.from_pretrained = classmethod(patched)
        WanModel._motion_probe_low_cpu_patch = True

    config = WAN_CONFIGS["ti2v-5B"]
    pipeline = wan.WanTI2V(
        config=config,
        checkpoint_dir=str(args.model_path),
        device_id=int(args.device.split(":")[-1]),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )
    pipeline.model.to(args.device)
    return pipeline


def build_scheduler(pipeline, args: argparse.Namespace):
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=pipeline.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(args.num_inference_steps, device=args.device, shift=args.shift)
    invalid = [index for index in args.step_indices if not 0 <= index < len(scheduler.timesteps)]
    if invalid:
        raise ValueError(f"Invalid scheduler step indices: {invalid}")
    return scheduler


def summarize_layer(
    layer_result: dict[str, np.ndarray],
    track_data: dict[str, np.ndarray],
    metadata: dict,
) -> list[dict]:
    rows = []
    predicted = layer_result["predictions"]
    target = track_data["anchor_tracks"]
    visibility = track_data["anchor_visibility"]
    query = track_data["query_points"]
    for region in metadata["regions"]:
        point_slice = slice(region["point_start"], region["point_end"])
        metrics = compute_track_metrics(
            predicted[:, point_slice],
            target[:, point_slice],
            visibility[:, point_slice],
            query[point_slice],
        )
        metrics["rigidity_error_px"] = compute_rigidity_error(
            predicted[:, point_slice], target[:, point_slice], visibility[:, point_slice]
        )
        valid = visibility[:, point_slice].copy()
        valid[0] = False
        for name in ("gt_probability", "gt_rank", "margin", "entropy"):
            values = layer_result[name][:, point_slice][valid]
            metrics[f"mean_{name}"] = float(np.nanmean(values)) if values.size else None
        rows.append({**region, **metrics})
    return rows


def run_sample(pipeline, scheduler, sample_metadata_path: Path, args: argparse.Namespace) -> None:
    metadata = json.loads(sample_metadata_path.read_text())
    sample_key = metadata["sample_key"]
    sample_output = args.output_dir / sample_key
    complete_path = sample_output / "complete.json"
    if complete_path.exists() and not args.overwrite:
        print(f"Skip completed {sample_key}", flush=True)
        return
    if args.overwrite and sample_output.exists():
        for path in sample_output.glob("step_*.npz"):
            path.unlink()
        for path in sample_output.glob("step_*.json"):
            path.unlink()
        complete_path.unlink(missing_ok=True)
    sample_output.mkdir(parents=True, exist_ok=True)
    if free_space_gib(sample_output) < args.min_free_gib:
        raise RuntimeError(f"Free space below {args.min_free_gib:.1f} GiB before {sample_key}")

    with np.load(args.tracks_dir / f"{sample_key}.npz") as loaded:
        track_data = {key: loaded[key] for key in loaded.files}
    video = read_video(Path(metadata["video"]), NUM_FRAMES)
    frames = video.permute(1, 0, 2, 3).div(127.5).sub(1.0).to(args.device)
    with torch.inference_mode():
        latent = pipeline.vae.encode([frames])[0]
    if latent.shape[1] != len(track_data["anchor_frames"]):
        raise RuntimeError(f"Native temporal invariant failed: {NUM_FRAMES} frames -> latent {tuple(latent.shape)}")
    expected_grid = (latent.shape[1], latent.shape[2] // 2, latent.shape[3] // 2)
    if expected_grid[1:] != (TARGET_HEIGHT // TOKEN_STRIDE, TARGET_WIDTH // TOKEN_STRIDE):
        raise RuntimeError(f"Unexpected Wan token grid {expected_grid}")

    context = encode_prompt(pipeline, metadata.get("caption", ""), torch.device(args.device))
    generator = seed_everything(args.seed, args.device)
    noise = torch.randn(latent.shape, dtype=torch.float32, device=args.device, generator=generator)
    seq_len = math.prod(expected_grid)
    all_rows = []

    for step_index in args.step_indices:
        step_npz = sample_output / f"step_{step_index:02d}.npz"
        step_json = sample_output / f"step_{step_index:02d}.json"
        if step_npz.exists() and step_json.exists() and not args.overwrite:
            all_rows.extend(json.loads(step_json.read_text())["rows"])
            print(f"{sample_key}: reuse step {step_index}", flush=True)
            continue

        timestep = int(scheduler.timesteps[step_index].item())
        sigma = float(scheduler.sigmas[step_index].item())
        noisy = (1.0 - sigma) * latent + sigma * noise
        noisy[:, 0] = latent[:, 0]
        token_timestep = torch.full((1, *expected_grid), float(timestep), device=args.device)
        token_timestep[:, 0] = 0.0
        token_timestep = token_timestep.flatten(1)

        capture = QKTrackCapture(args.layers, track_data["query_points"], track_data["anchor_tracks"])
        originals = install_qk_capture(pipeline.model, capture, args.layers)
        try:
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pipeline.model([noisy], t=token_timestep, context=context, seq_len=seq_len)
        finally:
            restore_qk_capture(pipeline.model, originals)
        if set(capture.results) != set(args.layers):
            raise RuntimeError(f"Captured layers {sorted(capture.results)}, expected {sorted(args.layers)}")

        arrays = {
            "layers": np.asarray(args.layers, dtype=np.int16),
            "timestep": np.asarray(timestep, dtype=np.int32),
            "sigma": np.asarray(sigma, dtype=np.float32),
        }
        rows = []
        for layer in args.layers:
            result = capture.results[layer]
            for name, values in result.items():
                arrays[f"layer_{layer:02d}_{name}"] = values
            for row in summarize_layer(result, track_data, metadata):
                rows.append(
                    {
                        "sample_key": sample_key,
                        "case_key": metadata["case_key"],
                        "sample_type": metadata["sample_type"],
                        "step_index": step_index,
                        "timestep": timestep,
                        "sigma": sigma,
                        "layer": layer,
                        **row,
                    }
                )
        temporary = step_npz.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(step_npz)
        atomic_write_json(step_json, {"rows": rows})
        all_rows.extend(rows)
        del capture, arrays, noisy, token_timestep
        torch.cuda.empty_cache()
        print(f"{sample_key}: step {step_index} t={timestep} sigma={sigma:.4f}", flush=True)

    atomic_write_json(sample_output / "metrics.json", {"rows": all_rows})
    atomic_write_json(
        complete_path,
        {
            "sample_key": sample_key,
            "protocol": "wan22_native_temporal_vae_qk_v1",
            "model": str(args.model_path),
            "video": metadata["video"],
            "prompt": metadata.get("caption", ""),
            "seed": args.seed,
            "height": TARGET_HEIGHT,
            "width": TARGET_WIDTH,
            "pixel_frames": NUM_FRAMES,
            "latent_frames": int(latent.shape[1]),
            "token_grid": list(expected_grid),
            "layers": args.layers,
            "step_indices": args.step_indices,
            "scheduler_timesteps": [int(scheduler.timesteps[index]) for index in args.step_indices],
            "scheduler_sigmas": [float(scheduler.sigmas[index]) for index in args.step_indices],
            "shift": args.shift,
            "matching": "mean of target-softmax(Q_query*K_target) and target-softmax(K_query*Q_target)",
            "scale": "sqrt(head_dim)=sqrt(128)",
        },
    )
    del video, frames, latent, noise, context, track_data
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if len(set(args.layers)) != len(args.layers) or not all(0 <= layer < 30 for layer in args.layers):
        raise ValueError(f"Invalid or duplicate layers: {args.layers}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_paths = sorted(args.tracks_dir.glob("*.json"))
    metadata_paths = [path for path in metadata_paths if not path.name.startswith("manifest_")]
    if args.sample_key:
        metadata_paths = [path for path in metadata_paths if path.stem == args.sample_key]
    else:
        metadata_paths = metadata_paths[args.start : args.end]
    if not metadata_paths:
        raise ValueError("No prepared track metadata selected")

    pipeline = load_pipeline(args)
    scheduler = build_scheduler(pipeline, args)
    for index, metadata_path in enumerate(metadata_paths, start=1):
        run_sample(pipeline, scheduler, metadata_path, args)
        print(f"Completed [{index}/{len(metadata_paths)}] {metadata_path.stem}", flush=True)


if __name__ == "__main__":
    main()
