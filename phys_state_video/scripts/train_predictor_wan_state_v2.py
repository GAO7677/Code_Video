from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.dataset import NpzPredictorFullDataset, collate_predictor_full_episodes
from phys_state_video.predictor_wan_state_v2 import (
    WanStateLatentPredictorV2,
    WanStateLatentPredictorV2Config,
    resample_temporal_states,
    wan_state_predictor_v2_loss,
)
from phys_state_video.checkpoint_io import load_torch_checkpoint
from phys_state_video.utils import require_torch
from phys_state_video.wan_adapter_training import load_frozen_state_adapter_encoder
from phys_state_video.wan_bridge import WanLatentExtractor
from phys_state_video.wan_state_v2_helpers import (
    build_state_condition_payload_from_condition_maps,
    filter_state_condition_payload_for_adapter,
    WanPromptContextEncoder,
    compute_future_latent_steps,
    compute_latent_step_count,
    resample_camera_to_latent_steps,
)

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the latent-time Wan state predictor v2.")
    parser.add_argument("--data", required=True, help="Directory containing episode .npz files.")
    parser.add_argument("--val-data", default=None, help="Optional validation directory containing episode .npz files.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs-context", type=int, default=1)
    parser.add_argument("--epochs-future", type=int, default=1)
    parser.add_argument("--epochs-joint", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-source", choices=["wan"], default="wan")
    parser.add_argument("--wan-ckpt-dir", default=None)
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--wan-task", default="i2v-A14B")
    parser.add_argument("--latent-smooth-scale", type=float, default=0.05)
    parser.add_argument("--teacher-predictor", default=None, help="Optional frozen teacher predictor checkpoint for adapter-space alignment.")
    parser.add_argument("--adapter-align-ckpt", default=None, help="Optional trained Wan state-adapter checkpoint used to compute adapter-space alignment.")
    parser.add_argument("--adapter-align-scale", type=float, default=0.1)
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume or finetune from.")
    parser.add_argument("--boundary-continuity-scale", type=float, default=0.0, help="Scale for boundary continuity delta loss between context tail and future head.")
    parser.add_argument("--boundary-head-scale", type=float, default=0.0, help="Scale for direct future-head anchor loss on the first future latent step.")
    parser.add_argument("--boundary-rollout-scale", type=float, default=0.0, help="Scale for weighted short-horizon rollout supervision near the context/future boundary.")
    parser.add_argument("--boundary-rollout-steps", type=int, default=3, help="Number of early future latent steps used by boundary rollout supervision.")
    parser.add_argument("--boundary-rollout-decay", type=float, default=0.5, help="Geometric decay for boundary rollout supervision weights.")
    parser.add_argument("--boundary-curvature-scale", type=float, default=0.0, help="Scale for cross-boundary second-difference smoothness supervision.")
    parser.add_argument("--max-context-frames", type=int, default=12, help="Maximum context frame count used by dynamic training-time slicing.")
    parser.add_argument("--min-context-frames", type=int, default=4, help="Minimum context frame count used by dynamic training-time slicing.")
    parser.add_argument("--min-future-frames", type=int, default=8, help="Minimum future frame count preserved after dynamic context slicing.")
    parser.add_argument("--context-ratio-min", type=float, default=0.25, help="Minimum context ratio relative to the sampled full clip length.")
    parser.add_argument("--context-ratio-max", type=float, default=0.5, help="Maximum context ratio relative to the sampled full clip length.")
    return parser.parse_args()


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return int(torch.distributed.get_rank())


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return int(torch.distributed.get_world_size())


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(device: str) -> tuple[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return device, 0
    if not str(device).startswith("cuda"):
        raise ValueError(f"multi-GPU training requires a CUDA device string, got {device!r}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return f"cuda:{local_rank}", local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def distributed_barrier() -> None:
    if is_distributed():
        torch.distributed.barrier()


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def reduce_scalar_dict(metrics: dict[str, float], device: str) -> dict[str, float]:
    if not is_distributed():
        return metrics
    reduced: dict[str, float] = {}
    for key, value in metrics.items():
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        reduced[key] = float((tensor / get_world_size()).item())
    return reduced


def default_best_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.best")


def stage_best_output(output_path: Path, stage_name: str) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.{stage_name}.best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.{stage_name}.best")


def build_latent_extractor(args):
    if args.wan_ckpt_dir is None:
        raise ValueError("--wan-ckpt-dir is required because wan_state_v2 now always uses Wan VAE latents")
    return WanLatentExtractor(
        ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        task=args.wan_task,
        device=args.device,
    )


def build_prompt_context_encoder(args):
    if args.wan_ckpt_dir is None:
        raise ValueError("--wan-ckpt-dir is required because wan_state_v2 now uses frozen Wan T5 prompt context")
    return WanPromptContextEncoder(
        ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        task=args.wan_task,
        device=args.device,
    )


def load_teacher_predictor(checkpoint_path: str, device: str) -> WanStateLatentPredictorV2:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    if checkpoint.get("predictor_version") != "wan_state_v2_latent_time":
        raise ValueError(
            f"teacher predictor must be a wan_state_v2_latent_time checkpoint, got {checkpoint.get('predictor_version')!r}"
        )
    model = WanStateLatentPredictorV2(WanStateLatentPredictorV2Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.requires_grad_(False)
    return model


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def resolve_context_bounds(total_steps: int, args) -> tuple[int, int]:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    max_context = total_steps - int(args.min_future_frames)
    if args.max_context_frames is not None and args.max_context_frames > 0:
        max_context = min(max_context, int(args.max_context_frames))
    min_context = max(1, int(args.min_context_frames))
    if max_context < min_context:
        raise ValueError(
            f"invalid dynamic slicing bounds for total_steps={total_steps}: "
            f"min_context={min_context}, max_context={max_context}, min_future={args.min_future_frames}"
        )
    return min_context, max_context


def choose_context_steps(total_steps: int, args, *, is_train: bool, legacy_context_steps: int) -> int:
    min_context, max_context = resolve_context_bounds(total_steps, args)
    if is_train:
        ratio = random.uniform(float(args.context_ratio_min), float(args.context_ratio_max))
        sampled = int(round(total_steps * ratio))
        return _clamp_int(sampled, min_context, max_context)
    if legacy_context_steps > 0:
        return _clamp_int(int(legacy_context_steps), min_context, max_context)
    midpoint_ratio = 0.5 * (float(args.context_ratio_min) + float(args.context_ratio_max))
    sampled = int(round(total_steps * midpoint_ratio))
    return _clamp_int(sampled, min_context, max_context)


def slice_dynamic_context_batch(batch, args, *, is_train: bool) -> dict[str, torch.Tensor | int]:
    full_frames = batch["full_frames"]
    full_states = batch["full_states"]
    full_camera = batch["camera"]
    full_lengths = batch["full_lengths"]
    legacy_context_steps = batch["legacy_context_steps"]

    batch_total_steps = int(full_lengths.min().item())
    context_steps = choose_context_steps(
        batch_total_steps,
        args,
        is_train=is_train,
        legacy_context_steps=int(legacy_context_steps.min().item()),
    )
    future_steps = batch_total_steps - context_steps
    if future_steps < int(args.min_future_frames):
        raise ValueError(
            f"future_steps={future_steps} violates min_future_frames={args.min_future_frames} for "
            f"batch_total_steps={batch_total_steps}, context_steps={context_steps}"
        )

    context_frames_list = []
    context_states_list = []
    future_states_list = []
    context_camera_list = []
    for batch_index in range(int(full_frames.shape[0])):
        sample_total = int(full_lengths[batch_index].item())
        if sample_total < batch_total_steps:
            raise ValueError(
                f"sample_total={sample_total} smaller than chosen batch_total_steps={batch_total_steps}"
            )
        max_start = sample_total - batch_total_steps
        start = random.randint(0, max_start) if is_train and max_start > 0 else 0
        end = start + batch_total_steps
        sample_frames = full_frames[batch_index, start:end]
        sample_states = full_states[batch_index, start:end]
        sample_camera = full_camera[batch_index, start:end]
        context_frames_list.append(sample_frames[:context_steps])
        context_states_list.append(sample_states[:context_steps])
        future_states_list.append(sample_states[context_steps:])
        context_camera_list.append(sample_camera[:context_steps])

    return {
        "context_frames": torch.stack(context_frames_list, dim=0),
        "context_states": torch.stack(context_states_list, dim=0),
        "future_states": torch.stack(future_states_list, dim=0),
        "camera": torch.stack(context_camera_list, dim=0),
        "context_steps": context_steps,
        "future_steps": future_steps,
        "window_steps": batch_total_steps,
    }


def infer_model_config(dataset, latent_extractor, prompt_context_encoder, args) -> WanStateLatentPredictorV2Config:
    first_sample = dataset[0]
    first_total_steps = int(first_sample.full_frames.shape[0])
    first_min_context, first_max_context = resolve_context_bounds(first_total_steps, args)
    sample_frames = torch.from_numpy(first_sample.full_frames[:first_max_context][None]).to(latent_extractor.device)
    with torch.no_grad():
        context_latents = latent_extractor.encode_context_frames_raw(sample_frames)

    max_context_latent_steps = int(context_latents.shape[1])
    max_future_latent_steps = compute_future_latent_steps(
        context_steps=first_min_context,
        future_steps=first_total_steps - first_min_context,
        temporal_stride=latent_extractor.temporal_stride,
    )
    max_objects = int(first_sample.full_states.shape[1])

    for index in range(1, len(dataset)):
        sample = dataset[index]
        total_steps = int(sample.full_frames.shape[0])
        min_context_steps, max_context_steps = resolve_context_bounds(total_steps, args)
        context_latent_steps = compute_latent_step_count(
            frame_steps=max_context_steps,
            temporal_stride=latent_extractor.temporal_stride,
        )
        future_latent_steps = compute_future_latent_steps(
            context_steps=min_context_steps,
            future_steps=total_steps - min_context_steps,
            temporal_stride=latent_extractor.temporal_stride,
        )
        max_context_latent_steps = max(max_context_latent_steps, int(context_latent_steps))
        max_future_latent_steps = max(max_future_latent_steps, int(future_latent_steps))
        max_objects = max(max_objects, int(sample.full_states.shape[1]))

    return WanStateLatentPredictorV2Config(
        latent_channels=context_latents.shape[2],
        camera_dim=first_sample.camera.shape[-1],
        prompt_context_dim=prompt_context_encoder.context_dim,
        max_context_latent_steps=max_context_latent_steps,
        max_future_latent_steps=max_future_latent_steps,
        max_prompt_tokens=prompt_context_encoder.max_text_len,
        max_objects=max_objects,
    )


def configure_stage(model: WanStateLatentPredictorV2, train_stage: str) -> None:
    model.requires_grad_(True)
    if train_stage == "context_only":
        model.unfreeze_state_heads()
    elif train_stage == "future_only":
        model.freeze_state_heads()
    elif train_stage == "joint_finetune":
        model.unfreeze_state_heads()
    else:
        raise ValueError(f"unsupported train_stage={train_stage}")


def encode_adapter_payload_branches(adapter_encoder, payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    filtered_payload = filter_state_condition_payload_for_adapter(payload, adapter_encoder)
    encoded = adapter_encoder(filtered_payload)
    splits: dict[str, torch.Tensor] = {}
    offset = 0
    if filtered_payload.get("memory_tokens") is not None and getattr(adapter_encoder, "memory_token_encoder", None) is not None:
        mem_len = int(filtered_payload["memory_tokens"].shape[1])
        splits["memory_context"] = encoded[:, offset : offset + mem_len]
        offset += mem_len
    if filtered_payload.get("condition_maps") is not None and getattr(adapter_encoder, "map_token_encoder", None) is not None:
        maps = filtered_payload["condition_maps"]
        map_len = int(maps.shape[1] * maps.shape[3] * maps.shape[4])
        splits["map_context"] = encoded[:, offset : offset + map_len]
        offset += map_len
    splits["state_context"] = encoded
    return splits


def cosine_alignment_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_flat = predicted.flatten(1)
    target_flat = target.flatten(1)
    cosine = torch.nn.functional.cosine_similarity(predicted_flat, target_flat, dim=1, eps=1e-6)
    return 1.0 - cosine.mean()


def compute_adapter_alignment_terms(
    predicted_payload: dict[str, torch.Tensor],
    reference_payload: dict[str, torch.Tensor],
    adapter_encoder,
) -> dict[str, torch.Tensor]:
    predicted_branches = encode_adapter_payload_branches(adapter_encoder, predicted_payload)
    with torch.no_grad():
        reference_branches = encode_adapter_payload_branches(adapter_encoder, reference_payload)
    losses: dict[str, torch.Tensor] = {}
    zero = predicted_branches["state_context"].new_zeros(())
    losses["adapter_align_map"] = zero
    losses["adapter_align_memory"] = zero
    if predicted_branches.get("map_context") is not None and reference_branches.get("map_context") is not None:
        losses["adapter_align_map"] = torch.mean(
            (predicted_branches["map_context"] - reference_branches["map_context"]) ** 2
        )
    if predicted_branches.get("memory_context") is not None and reference_branches.get("memory_context") is not None:
        losses["adapter_align_memory"] = torch.mean(
            (predicted_branches["memory_context"] - reference_branches["memory_context"]) ** 2
        )
    losses["adapter_align_cosine"] = cosine_alignment_loss(
        predicted_branches["state_context"],
        reference_branches["state_context"],
    )
    losses["adapter_align"] = (
        losses["adapter_align_map"]
        + losses["adapter_align_memory"]
        + 0.1 * losses["adapter_align_cosine"]
    )
    return losses


def save_checkpoint(
    output_path: Path,
    model: WanStateLatentPredictorV2,
    config: WanStateLatentPredictorV2Config,
    args,
    history,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "model": unwrap_model(model).state_dict(),
            "history": history,
            "predictor_version": "wan_state_v2_latent_time",
            "latent_source": "wan",
            "wan_ckpt_dir": args.wan_ckpt_dir,
            "wan_repo_root": args.wan_repo_root,
            "wan_task": args.wan_task,
            "val_data": args.val_data,
            "temporal_stride": getattr(args, "temporal_stride", None),
            "teacher_predictor": args.teacher_predictor,
            "adapter_align_ckpt": args.adapter_align_ckpt,
            "adapter_align_scale": args.adapter_align_scale,
            "resume": args.resume,
            "boundary_continuity_scale": args.boundary_continuity_scale,
            "boundary_head_scale": args.boundary_head_scale,
            "boundary_rollout_scale": args.boundary_rollout_scale,
            "boundary_rollout_steps": args.boundary_rollout_steps,
            "boundary_rollout_decay": args.boundary_rollout_decay,
            "boundary_curvature_scale": args.boundary_curvature_scale,
            "dynamic_context": {
                "max_context_frames": args.max_context_frames,
                "min_context_frames": args.min_context_frames,
                "min_future_frames": args.min_future_frames,
                "context_ratio_min": args.context_ratio_min,
                "context_ratio_max": args.context_ratio_max,
            },
            "world_size": get_world_size(),
            "train_schedule": {
                "epochs_context": args.epochs_context,
                "epochs_future": args.epochs_future,
                "epochs_joint": args.epochs_joint,
            },
        },
        output_path,
    )


def run_epoch(
    model,
    latent_extractor,
    prompt_context_encoder,
    loader,
    optimizer,
    device,
    train_stage: str,
    latent_smooth_scale: float,
    args,
    teacher_predictor=None,
    adapter_encoder=None,
    adapter_align_scale: float = 0.0,
):
    is_train = optimizer is not None
    model.train(mode=is_train)
    running = {
        "loss": 0.0,
        "context_loss": 0.0,
        "context_geom": 0.0,
        "context_motion": 0.0,
        "context_vis": 0.0,
        "future_loss": 0.0,
        "future_geom": 0.0,
        "future_motion": 0.0,
        "future_vis": 0.0,
        "latent_smooth": 0.0,
        "adapter_align": 0.0,
        "adapter_align_map": 0.0,
        "adapter_align_memory": 0.0,
        "adapter_align_cosine": 0.0,
        "boundary_head": 0.0,
        "boundary_head_geom": 0.0,
        "boundary_head_motion": 0.0,
        "boundary_rollout": 0.0,
        "boundary_rollout_geom": 0.0,
        "boundary_rollout_motion": 0.0,
        "boundary_continuity": 0.0,
        "boundary_geom": 0.0,
        "boundary_motion": 0.0,
        "boundary_curvature": 0.0,
        "boundary_curvature_geom": 0.0,
        "boundary_curvature_motion": 0.0,
        "window_frames": 0.0,
        "context_frames": 0.0,
        "future_frames": 0.0,
    }
    for batch in loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            sliced = slice_dynamic_context_batch(batch, args, is_train=is_train)
            context_frames = sliced["context_frames"].to(device)
            camera = sliced["camera"].to(device)
            context_states = sliced["context_states"].to(device)
            future_states = sliced["future_states"].to(device)

            with torch.no_grad():
                context_latents = latent_extractor.encode_context_frames_raw(context_frames)
            context_latent_steps = context_latents.shape[1]
            future_latent_steps = compute_future_latent_steps(
                context_steps=context_frames.shape[1],
                future_steps=future_states.shape[1],
                temporal_stride=latent_extractor.temporal_stride,
            )
            camera_latent = resample_camera_to_latent_steps(camera, context_latent_steps)
            context_target = resample_temporal_states(context_states, context_latent_steps)
            future_target = resample_temporal_states(future_states, future_latent_steps)
            with torch.no_grad():
                prompt_context, prompt_mask = prompt_context_encoder.encode_prompts(list(batch["prompts"]))

            outputs = model(
                context_latents=context_latents,
                camera=camera_latent,
                prompt_context=prompt_context.to(device),
                prompt_mask=prompt_mask.to(device),
                future_latent_steps=future_latent_steps,
                num_objects=context_states.shape[2],
            )
            losses = wan_state_predictor_v2_loss(
                outputs=outputs,
                context_target=context_target,
                future_target=future_target,
                train_stage=train_stage,
                latent_smooth_scale=latent_smooth_scale,
                boundary_continuity_scale=args.boundary_continuity_scale,
                boundary_head_scale=args.boundary_head_scale,
                boundary_rollout_scale=args.boundary_rollout_scale,
                boundary_rollout_steps=args.boundary_rollout_steps,
                boundary_rollout_decay=args.boundary_rollout_decay,
                boundary_curvature_scale=args.boundary_curvature_scale,
            )
            if adapter_encoder is not None and adapter_align_scale > 0.0 and train_stage != "context_only":
                predicted_payload = build_state_condition_payload_from_condition_maps(
                    outputs["condition_maps"],
                    memory_tokens=outputs["memory_tokens"],
                    include_condition_maps=True,
                )
                predicted_payload = filter_state_condition_payload_for_adapter(predicted_payload, adapter_encoder)
                with torch.no_grad():
                    if teacher_predictor is None:
                        raise ValueError(
                            "adapter-space supervision requires --teacher-predictor when adapter_align is enabled"
                        )
                    teacher_outputs = teacher_predictor(
                        context_latents=context_latents,
                        camera=camera_latent,
                        prompt_context=prompt_context.to(device),
                        prompt_mask=prompt_mask.to(device),
                        future_latent_steps=future_latent_steps,
                        num_objects=context_states.shape[2],
                    )
                    reference_payload = build_state_condition_payload_from_condition_maps(
                        teacher_outputs["condition_maps"],
                        memory_tokens=teacher_outputs["memory_tokens"],
                        include_condition_maps=True,
                    )
                    reference_payload = filter_state_condition_payload_for_adapter(reference_payload, adapter_encoder)
                adapter_terms = compute_adapter_alignment_terms(
                    predicted_payload=predicted_payload,
                    reference_payload=reference_payload,
                    adapter_encoder=adapter_encoder,
                )
            else:
                zero = losses["loss"].new_zeros(())
                adapter_terms = {
                    "adapter_align": zero,
                    "adapter_align_map": zero,
                    "adapter_align_memory": zero,
                    "adapter_align_cosine": zero,
                }
            losses.update(adapter_terms)
            losses["loss"] = losses["loss"] + adapter_align_scale * losses["adapter_align"]
        if is_train:
            losses["loss"].backward()
            optimizer.step()
        for key in running:
            if key in losses:
                running[key] += float(losses[key].detach().cpu())
        running["window_frames"] += float(sliced["window_steps"])
        running["context_frames"] += float(sliced["context_steps"])
        running["future_frames"] += float(sliced["future_steps"])
    denom = max(len(loader), 1)
    metrics = {key: value / denom for key, value in running.items()}
    return reduce_scalar_dict(metrics, device=device)


def main():
    local_rank = 0
    args = parse_args()
    try:
        args.device, local_rank = setup_distributed(args.device)
        if args.adapter_align_scale > 0.0 and (args.teacher_predictor is None or args.adapter_align_ckpt is None):
            raise ValueError(
                "--teacher-predictor and --adapter-align-ckpt are required when --adapter-align-scale > 0"
            )
        if args.context_ratio_min <= 0.0 or args.context_ratio_max <= 0.0:
            raise ValueError("context ratios must be positive")
        if args.context_ratio_min > args.context_ratio_max:
            raise ValueError("context-ratio-min must be <= context-ratio-max")
        dataset = NpzPredictorFullDataset(args.data)
        val_dataset = NpzPredictorFullDataset(args.val_data) if args.val_data else None
        train_sampler = None
        val_sampler = None
        if is_distributed():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
            )
            if val_dataset is not None:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    shuffle=False,
                )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_predictor_full_episodes,
            num_workers=args.num_workers,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_predictor_full_episodes,
                num_workers=args.num_workers,
            )

        latent_extractor = build_latent_extractor(args)
        prompt_context_encoder = build_prompt_context_encoder(args)
        args.temporal_stride = latent_extractor.temporal_stride
        config = infer_model_config(dataset, latent_extractor, prompt_context_encoder, args)
        model = WanStateLatentPredictorV2(config).to(args.device)
        if args.resume is not None:
            resume_ckpt = load_torch_checkpoint(args.resume, map_location="cpu")
            resume_config = resume_ckpt.get("config")
            if resume_config is not None and resume_config != asdict(config):
                if is_main_process():
                    print(
                        json.dumps(
                            {
                                "warning": "resume_config_differs_from_current_config",
                                "resume": args.resume,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            model.load_state_dict(resume_ckpt["model"], strict=True)
        if is_distributed():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        teacher_predictor = None
        adapter_encoder = None
        if args.teacher_predictor is not None:
            teacher_predictor = load_teacher_predictor(args.teacher_predictor, args.device)
        if args.adapter_align_ckpt is not None:
            adapter_encoder = load_frozen_state_adapter_encoder(
                args.adapter_align_ckpt,
                wan_repo_root=args.wan_repo_root,
                device=args.device,
            )

        schedule = [
            ("context_only", args.epochs_context),
            ("future_only", args.epochs_future),
            ("joint_finetune", args.epochs_joint),
        ]
        history = []
        output = Path(args.output)
        global_epoch = 0
        for stage_name, num_epochs in schedule:
            if num_epochs <= 0:
                continue
            configure_stage(unwrap_model(model), stage_name)
            optimizer = torch.optim.AdamW(
                [param for param in model.parameters() if param.requires_grad],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            best_metric = float("inf")
            best_path = stage_best_output(output, stage_name)
            for epoch in range(num_epochs):
                global_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(global_epoch)
                train_metrics = run_epoch(
                    model=model,
                    latent_extractor=latent_extractor,
                    prompt_context_encoder=prompt_context_encoder,
                    loader=loader,
                    optimizer=optimizer,
                    device=args.device,
                    train_stage=stage_name,
                    latent_smooth_scale=args.latent_smooth_scale,
                    args=args,
                    teacher_predictor=teacher_predictor,
                    adapter_encoder=adapter_encoder,
                    adapter_align_scale=args.adapter_align_scale,
                )
                val_metrics = None
                if val_loader is not None:
                    val_metrics = run_epoch(
                        model=model,
                        latent_extractor=latent_extractor,
                        prompt_context_encoder=prompt_context_encoder,
                        loader=val_loader,
                        optimizer=None,
                        device=args.device,
                        train_stage=stage_name,
                        latent_smooth_scale=args.latent_smooth_scale,
                        args=args,
                        teacher_predictor=teacher_predictor,
                        adapter_encoder=adapter_encoder,
                        adapter_align_scale=args.adapter_align_scale,
                    )
                record = {
                    "stage": stage_name,
                    "epoch": epoch + 1,
                    "global_epoch": global_epoch,
                    "train_metrics": train_metrics,
                }
                if val_metrics is not None:
                    record["val_metrics"] = val_metrics
                history.append(record)
                if is_main_process():
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                selection_metrics = val_metrics if val_metrics is not None else train_metrics
                selection_loss = float(selection_metrics["loss"])
                if selection_loss <= best_metric:
                    best_metric = selection_loss
                    if is_main_process():
                        save_checkpoint(best_path, model, config, args, history)
            distributed_barrier()
            best_checkpoint = load_torch_checkpoint(str(best_path), map_location="cpu")
            unwrap_model(model).load_state_dict(best_checkpoint["model"])
            distributed_barrier()
            if is_main_process():
                print(
                    json.dumps(
                        {
                            "stage": stage_name,
                            "stage_best_checkpoint": str(best_path),
                            "stage_best_loss": best_metric,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if is_main_process():
            save_checkpoint(output, model, config, args, history)
            save_checkpoint(default_best_output(output), model, config, args, history)
            print(f"saved Wan state predictor v2 checkpoint to {output}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
