from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.utils import require_torch
from phys_state_video.checkpoint_io import load_torch_checkpoint
from phys_state_video.wan_runtime import ensure_wan_importable
from phys_state_video.wan_state_condition_bundles import (
    discover_state_condition_bundles,
    is_ti2v_state_adapter_checkpoint,
    load_episode_npz,
    load_state_condition_npz,
)
from phys_state_video.wan_adapter_training import (
    LocalWanFlowMatchScheduler,
    align_wan_frame_num,
    build_first_frame_mask,
    build_ti2v_timestep_tensor,
    build_ti2v_training_video,
    compute_ti2v_seq_len,
    normalize_video_range,
    resize_and_center_crop_frames,
    select_ti2v_state_adapter_parameters,
    serialize_ti2v_state_adapter_checkpoint,
)
from phys_state_video.wan_state_v2_helpers import (
    build_state_condition_payload_from_condition_maps,
    filter_state_condition_payload_for_adapter,
    resample_condition_maps_to_steps,
)

torch = require_torch()
F = torch.nn.functional
dist = torch.distributed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Wan TI2V state adapter locally from exported phys_state_video state-condition bundles."
    )
    parser.add_argument("--state-condition-root", required=True, help="Directory containing exported state-condition bundles.")
    parser.add_argument("--wan-ckpt-dir", required=True, help="Wan checkpoint directory.")
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--task", default="ti2v-5B", choices=["ti2v-5B"])
    parser.add_argument("--size", default="704*1280")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=0, help="If > 0, stop after this many optimizer steps.")
    parser.add_argument("--frame-num", type=int, default=0, help="If 0, derive per-sample from first-frame + future frames and align to 4n+1.")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--state-scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="If > 0, only use the first N bundles.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume", default=None, help="Optional existing TI2V state-adapter checkpoint to resume from.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Number of local micro-steps to accumulate before each optimizer step.")
    parser.add_argument("--save-every-steps", type=int, default=0, help="If > 0, save a checkpoint every N optimizer steps on rank 0.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(device: str) -> tuple[str, int]:
    device = str(device)
    if not device.startswith("cuda"):
        raise ValueError(f"this trainer currently requires a CUDA device string, got {device!r}")
    if ":" in device:
        return device, int(device.split(":")[1])
    return device, 0


def distributed_context() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return {
        "enabled": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
    }


def init_distributed_if_needed() -> dict[str, int | bool]:
    ctx = distributed_context()
    if ctx["enabled"] and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return ctx


def default_best_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.best")


def broadcast_object(obj, *, src: int = 0):
    if not dist.is_initialized():
        return obj
    payload = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def shuffled_epoch_indices(num_records: int, *, epoch: int, seed: int) -> list[int]:
    if num_records <= 0:
        raise ValueError("num_records must be positive")
    indices = list(range(num_records))
    random.Random(seed + epoch).shuffle(indices)
    return indices


def build_rank_epoch_indices(
    shuffled_indices: list[int],
    *,
    world_size: int,
    rank: int,
) -> list[int]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not shuffled_indices:
        raise ValueError("shuffled_indices must not be empty")
    steps_per_rank = int(math.ceil(len(shuffled_indices) / world_size))
    rank_indices: list[int] = []
    for local_step in range(steps_per_rank):
        global_slot = local_step * world_size + rank
        rank_indices.append(shuffled_indices[global_slot % len(shuffled_indices)])
    return rank_indices


def average_gradients(trainable_params) -> None:
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        return
    for _, param in trainable_params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def reduce_scalar(value: float, *, device) -> float:
    if not dist.is_initialized():
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return float(tensor.item())


def prepare_training_sample(record, *, output_size: tuple[int, int], frame_num: int) -> dict[str, object]:
    episode = load_episode_npz(record.episode_path)
    state_condition = load_state_condition_npz(record.state_condition_path)

    context_frames = episode["context_frames"]
    future_frames = episode["future_frames"]
    training_video = build_ti2v_training_video(
        context_frames=context_frames,
        future_frames=future_frames,
        frame_num=frame_num,
    )

    out_h, out_w = int(output_size[0]), int(output_size[1])
    resized_video = resize_and_center_crop_frames(training_video.float(), out_h=out_h, out_w=out_w)
    normalized_video = normalize_video_range(resized_video)
    first_frame = normalized_video[:1]

    return {
        "sample_id": record.sample_id,
        "prompt": record.prompt,
        "state_condition": state_condition,
        "input_video": normalized_video,
        "first_frame": first_frame,
        "episode_path": str(record.episode_path),
        "bundle_dir": str(record.bundle_dir),
        "training_frame_num": int(normalized_video.shape[0]),
    }


def build_ti2v_state_condition_payload(
    raw_state_condition: dict[str, object],
    *,
    device,
    target_steps: int,
) -> dict[str, torch.Tensor]:
    if "condition_maps" not in raw_state_condition:
        raise ValueError(
            "TI2V adapter training now expects exported predictor condition_maps in each bundle; "
            "state_condition.npz is missing 'condition_maps'."
        )
    condition_maps = torch.from_numpy(raw_state_condition["condition_maps"]).to(device=device, dtype=torch.float32)
    condition_maps = resample_condition_maps_to_steps(condition_maps, target_steps=target_steps)
    memory_tokens = None
    if "memory_tokens" in raw_state_condition:
        memory_tokens = torch.from_numpy(raw_state_condition["memory_tokens"]).to(device=device, dtype=torch.float32)
        if memory_tokens.ndim == 2:
            memory_tokens = memory_tokens.unsqueeze(0)
    return build_state_condition_payload_from_condition_maps(
        condition_maps,
        memory_tokens=memory_tokens,
        include_condition_maps=True,
    )


def run_step(
    *,
    pipeline,
    scheduler: LocalWanFlowMatchScheduler,
    sample: dict[str, object],
    optimizer,
    state_scale: float,
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    device = pipeline.device
    model_dtype = pipeline.model.patch_embedding.weight.dtype

    input_video = sample["input_video"].to(device=device, dtype=torch.float32)
    first_frame = sample["first_frame"].to(device=device, dtype=torch.float32)
    raw_state_condition = sample["state_condition"]

    t5_model = getattr(pipeline.text_encoder, "model", None)
    vae_module = getattr(pipeline.vae, "model", None)
    with torch.no_grad():
        if t5_model is not None:
            t5_model.to(device)
        if vae_module is not None:
            vae_module.to(device)
        context = pipeline.text_encoder([sample["prompt"]], device)
        input_latents = pipeline.vae.encode([input_video.permute(1, 0, 2, 3).contiguous()])[0]
        first_frame_latents = pipeline.vae.encode([first_frame.permute(1, 0, 2, 3).contiguous()])[0]
        if t5_model is not None:
            t5_model.to("cpu")
        if vae_module is not None:
            vae_module.to("cpu")

    del input_video, first_frame
    torch.cuda.empty_cache()

    latent_mask = build_first_frame_mask(input_latents)
    timestep = scheduler.sample_timestep(device=device, dtype=torch.float32)
    noise = torch.randn_like(input_latents)
    noised_latents = scheduler.add_noise(input_latents, noise, timestep)
    noised_latents = (1.0 - latent_mask) * first_frame_latents + latent_mask * noised_latents
    training_target = scheduler.training_target(input_latents, noise, timestep)
    latent_steps = float(input_latents.shape[1])
    target_condition_steps = max(int(input_latents.shape[1] - 1), 1)
    state_condition_payload = build_ti2v_state_condition_payload(
        raw_state_condition,
        device=device,
        target_steps=target_condition_steps,
    )

    seq_len = compute_ti2v_seq_len(input_latents, patch_size=tuple(pipeline.patch_size[1:]))
    seq_len = int(math.ceil(seq_len / pipeline.sp_size)) * pipeline.sp_size
    timestep_tokens = build_ti2v_timestep_tensor(latent_mask, timestep=timestep, seq_len=seq_len)
    adapter_payload = filter_state_condition_payload_for_adapter(state_condition_payload, pipeline.state_adapter)
    state_context = pipeline._build_state_context(adapter_payload, offload_model=False)
    del state_condition_payload, adapter_payload, input_latents, noise, first_frame_latents, latent_mask
    torch.cuda.empty_cache()

    with torch.amp.autocast("cuda", dtype=pipeline.param_dtype):
        noise_pred = pipeline.model(
            [noised_latents.to(dtype=model_dtype)],
            t=timestep_tokens.to(device=device, dtype=torch.float32),
            context=[context[0]],
            seq_len=seq_len,
            state_context=state_context,
            state_scale=state_scale,
        )[0]
        loss = F.mse_loss(noise_pred[:, 1:].float(), training_target[:, 1:].float())
        loss = loss * scheduler.training_weight(timestep)

    (loss / max(int(grad_accum_steps), 1)).backward()
    del context, state_context, timestep_tokens, noised_latents, training_target, noise_pred
    torch.cuda.empty_cache()

    return {
        "loss": float(loss.detach().cpu()),
        "training_frame_num": float(sample["training_frame_num"]),
        "latent_steps": latent_steps,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Local Wan state adapter training requires CUDA, but torch.cuda.is_available() is False in the active "
            "environment. This workspace currently has torch 2.11.0+cu130 in the `wan` env while the machine driver "
            "stack is CUDA 12.8 / driver 570.124.06, so real Wan training remains environment-blocked."
        )

    ctx = init_distributed_if_needed()
    device, device_id = resolve_device(args.device)
    if ctx["enabled"]:
        device_id = int(ctx["local_rank"])
        device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)
    ensure_wan_importable(args.wan_repo_root)

    from wan_.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
    from wan_.textimage2video import WanTI2V

    if args.task not in WAN_CONFIGS:
        raise ValueError(f"unsupported Wan task: {args.task}")
    if args.size not in SUPPORTED_SIZES[args.task]:
        raise ValueError(f"unsupported size {args.size!r} for task {args.task}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_records = discover_state_condition_bundles(args.state_condition_root, limit=args.limit)
    output_size = SIZE_CONFIGS[args.size]

    pipeline = WanTI2V(
        config=WAN_CONFIGS[args.task],
        checkpoint_dir=str(args.wan_ckpt_dir),
        device_id=device_id,
        rank=int(ctx["rank"]),
        init_on_cpu=False,
    )

    first_sample = prepare_training_sample(
        bundle_records[0],
        output_size=output_size,
        frame_num=args.frame_num,
    )
    first_condition_payload = build_ti2v_state_condition_payload(
        first_sample["state_condition"],
        device=pipeline.device,
        target_steps=max(int((first_sample["training_frame_num"] - 1) // pipeline.vae_stride[0]), 1),
    )
    pipeline._build_state_context(
        first_condition_payload,
        offload_model=False,
    )
    if args.resume is not None:
        state_bundle = load_torch_checkpoint(args.resume, map_location="cpu")
        if not is_ti2v_state_adapter_checkpoint(state_bundle):
            raise ValueError(f"resume checkpoint is not a TI2V state-adapter checkpoint: {args.resume}")
        pipeline.load_state_adapter(args.resume, state_condition=first_condition_payload)

    trainable_params = select_ti2v_state_adapter_parameters(pipeline)
    optimizer = torch.optim.AdamW(
        [param for _, param in trainable_params],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = LocalWanFlowMatchScheduler(
        num_train_timesteps=int(pipeline.num_train_timesteps),
        shift=args.shift,
    )

    history: list[dict[str, object]] = []
    global_step = 0
    best_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)

    world_size = int(ctx["world_size"])
    rank = int(ctx["rank"])
    is_rank0 = rank == 0
    stop_training = False
    pending_metrics: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        epoch_indices = build_rank_epoch_indices(
            shuffled_epoch_indices(len(bundle_records), epoch=epoch, seed=args.seed),
            world_size=world_size,
            rank=rank,
        )
        epoch_step_losses: list[float] = []
        for local_step, record_index in enumerate(epoch_indices):
            record = bundle_records[record_index]
            sample = prepare_training_sample(
                record,
                output_size=output_size,
                frame_num=args.frame_num,
            )
            metrics = run_step(
                pipeline=pipeline,
                scheduler=scheduler,
                sample=sample,
                optimizer=optimizer,
                state_scale=args.state_scale,
                grad_accum_steps=args.grad_accum_steps,
            )
            pending_metrics.append(metrics)
            should_step = ((local_step + 1) % max(args.grad_accum_steps, 1) == 0) or (local_step + 1 == len(epoch_indices))
            if not should_step:
                continue

            average_gradients(trainable_params)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            mean_loss = sum(item["loss"] for item in pending_metrics) / max(len(pending_metrics), 1)
            mean_frame_num = sum(item["training_frame_num"] for item in pending_metrics) / max(len(pending_metrics), 1)
            mean_latent_steps = sum(item["latent_steps"] for item in pending_metrics) / max(len(pending_metrics), 1)
            reduced_loss = reduce_scalar(mean_loss, device=pipeline.device)
            reduced_frame_num = reduce_scalar(mean_frame_num, device=pipeline.device)
            reduced_latent_steps = reduce_scalar(mean_latent_steps, device=pipeline.device)
            epoch_step_losses.append(reduced_loss)
            history_record = {
                "epoch": epoch + 1,
                "step": global_step,
                "sample_id": record.sample_id,
                "metrics": {
                    "loss": reduced_loss,
                    "training_frame_num": reduced_frame_num,
                    "latent_steps": reduced_latent_steps,
                },
            }
            if is_rank0:
                history.append(history_record)
                if global_step % max(args.log_every, 1) == 0:
                    print(json.dumps(history_record, ensure_ascii=False))

            pending_metrics = []
            if is_rank0 and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                current_payload = serialize_ti2v_state_adapter_checkpoint(
                    pipeline.export_state_adapter(),
                    meta={
                        "task": args.task,
                        "size": args.size,
                        "wan_ckpt_dir": args.wan_ckpt_dir,
                        "state_condition_root": args.state_condition_root,
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "sample_id": record.sample_id,
                        "loss": reduced_loss,
                        "frame_num_arg": int(args.frame_num),
                        "aligned_frame_num": align_wan_frame_num(int(round(reduced_frame_num))),
                        "history_tail": history[-8:],
                        "world_size": world_size,
                        "grad_accum_steps": int(args.grad_accum_steps),
                    },
                )
                torch.save(current_payload, output_path)

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break
        epoch_loss = reduce_scalar(
            sum(epoch_step_losses) / max(len(epoch_step_losses), 1) if epoch_step_losses else float("inf"),
            device=pipeline.device,
        )
        if is_rank0:
            current_payload = serialize_ti2v_state_adapter_checkpoint(
                pipeline.export_state_adapter(),
                meta={
                    "task": args.task,
                    "size": args.size,
                    "wan_ckpt_dir": args.wan_ckpt_dir,
                    "state_condition_root": args.state_condition_root,
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "epoch_loss": epoch_loss,
                    "frame_num_arg": int(args.frame_num),
                    "history_tail": history[-8:],
                    "world_size": world_size,
                    "grad_accum_steps": int(args.grad_accum_steps),
                },
            )
            torch.save(current_payload, output_path)
            if epoch_loss <= best_loss:
                best_loss = epoch_loss
                torch.save(current_payload, default_best_output(output_path))
        if stop_training:
            break

    summary = {
        "status": "finished",
        "output": str(output_path),
        "best_output": str(default_best_output(output_path)),
        "num_bundles": len(bundle_records),
        "epochs": args.epochs,
        "global_steps": global_step,
        "best_loss": best_loss,
        "trainable_param_count": int(sum(param.numel() for _, param in trainable_params)),
        "trainable_param_names": [name for name, _ in trainable_params[:12]],
        "task": args.task,
        "size": args.size,
        "device": device,
        "frame_num_arg": int(args.frame_num),
        "world_size": world_size,
        "grad_accum_steps": int(args.grad_accum_steps),
    }
    if is_rank0:
        output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
