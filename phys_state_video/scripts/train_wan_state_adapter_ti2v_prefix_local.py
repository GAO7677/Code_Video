from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.checkpoint_io import load_torch_checkpoint
from phys_state_video.utils import require_torch
from phys_state_video.wan_adapter_training import (
    LocalWanFlowMatchScheduler,
    align_wan_frame_num,
    apply_clean_prefix_to_latents,
    build_prefix_latent_mask,
    build_prefix_timestep_tensor,
    build_prefix_training_video,
    preprocess_ti2v_prefix_frames,
    resample_condition_maps_to_steps,
    select_ti2v_state_adapter_parameters,
    serialize_ti2v_state_adapter_checkpoint,
)
from phys_state_video.wan_bridge import _build_prefix_condition_mask
from phys_state_video.wan_runtime import ensure_wan_importable
from phys_state_video.wan_state_condition_bundles import (
    discover_state_condition_bundles,
    is_ti2v_state_adapter_checkpoint,
    load_episode_npz,
    load_state_condition_npz,
)
from phys_state_video.wan_state_v2_helpers import (
    build_future_step_loss_mask,
    build_state_condition_payload_from_condition_maps,
    compute_future_latent_steps,
    filter_state_condition_payload_for_adapter,
)

torch = require_torch()
F = torch.nn.functional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Wan TI2V state adapter locally with full clean-prefix infill semantics."
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
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--frame-num", type=int, default=0, help="If 0, derive per-sample from context+future and align to 4n+1.")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--state-scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def resolve_device(device: str) -> tuple[str, int]:
    device = str(device)
    if not device.startswith("cuda"):
        raise ValueError(f"this trainer currently requires a CUDA device string, got {device!r}")
    if ":" in device:
        return device, int(device.split(":")[1])
    return device, 0


def default_best_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.best")


def prepare_training_sample(record, *, frame_num: int) -> dict[str, object]:
    episode = load_episode_npz(record.episode_path)
    state_condition = load_state_condition_npz(record.state_condition_path)
    context_frame_num = int(np.asarray(episode["context_frames"]).shape[0])
    future_frame_num = int(np.asarray(episode["future_frames"]).shape[0])
    training_video = build_prefix_training_video(
        context_frames=episode["context_frames"],
        future_frames=episode["future_frames"],
        frame_num=frame_num,
    )
    return {
        "sample_id": record.sample_id,
        "prompt": record.prompt,
        "state_condition": state_condition,
        "training_video": training_video,
        "context_frames": torch.from_numpy(np.asarray(episode["context_frames"])).float(),
        "context_frame_num": context_frame_num,
        "future_frame_num": future_frame_num,
        "episode_path": str(record.episode_path),
        "bundle_dir": str(record.bundle_dir),
        "training_frame_num": int(training_video.shape[0]),
    }


def build_ti2v_prefix_state_condition_payload(
    raw_state_condition: dict[str, object],
    *,
    device,
    target_steps: int,
) -> dict[str, torch.Tensor]:
    if "condition_maps" not in raw_state_condition:
        raise ValueError(
            "TI2V clean-prefix adapter training expects exported predictor condition_maps in each bundle; "
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
    output_size: tuple[int, int],
    state_scale: float,
) -> dict[str, float]:
    device = pipeline.device
    model_dtype = pipeline.model.patch_embedding.weight.dtype
    full_video = sample["training_video"].to(device=device, dtype=torch.float32)
    context_frames = sample["context_frames"].to(device=device, dtype=torch.float32)
    out_h, out_w = int(output_size[0]), int(output_size[1])
    full_video = preprocess_ti2v_prefix_frames(full_video, out_h=out_h, out_w=out_w)
    resized_context = preprocess_ti2v_prefix_frames(context_frames, out_h=out_h, out_w=out_w)

    optimizer.zero_grad(set_to_none=True)

    t5_model = getattr(pipeline.text_encoder, "model", None)
    vae_module = getattr(pipeline.vae, "model", None)
    with torch.no_grad():
        if t5_model is not None:
            t5_model.to(device)
        if vae_module is not None:
            vae_module.to(device)
        text_context = pipeline.text_encoder([sample["prompt"]], device)[0]
        full_latents = pipeline.vae.encode([full_video.permute(1, 0, 2, 3).contiguous()])[0]
        first_frame_video = torch.zeros(
            3,
            int(full_video.shape[0]),
            out_h,
            out_w,
            dtype=full_video.dtype,
            device=device,
        )
        first_frame_video[:, :1] = full_video[:1].permute(1, 0, 2, 3)
        first_frame_latents = pipeline.vae.encode([first_frame_video])[0]
        if t5_model is not None:
            t5_model.to("cpu")
        if vae_module is not None:
            vae_module.to("cpu")

    prefix_len = 1 + max(int(resized_context.shape[0]) - 1, 0) // int(pipeline.vae_stride[0])
    if prefix_len >= full_latents.shape[1]:
        raise ValueError(
            f"context prefix covers all latent steps for sample {sample['sample_id']}: "
            f"prefix_len={prefix_len}, latent_steps={full_latents.shape[1]}"
        )
    clean_prefix_latents = full_latents[:, :prefix_len].contiguous()
    future_latent_steps = int(full_latents.shape[1] - prefix_len)
    valid_future_latent_steps = compute_future_latent_steps(
        context_steps=int(sample["context_frame_num"]),
        future_steps=int(sample["future_frame_num"]),
        temporal_stride=int(pipeline.vae_stride[0]),
    )
    if valid_future_latent_steps > future_latent_steps:
        raise ValueError(
            f"valid_future_latent_steps exceeds padded future_latent_steps for sample {sample['sample_id']}: "
            f"{valid_future_latent_steps=} vs {future_latent_steps=}"
        )
    state_condition_payload = build_ti2v_prefix_state_condition_payload(
        sample["state_condition"],
        device=device,
        target_steps=future_latent_steps,
    )

    latent_mask = build_prefix_latent_mask(full_latents, prefix_len=prefix_len)
    timestep = scheduler.sample_timestep(device=device, dtype=torch.float32)
    noise = torch.randn_like(full_latents)
    noised_latents = scheduler.add_noise(full_latents, noise, timestep)
    noised_latents = apply_clean_prefix_to_latents(noised_latents, clean_prefix_latents)
    training_target = scheduler.training_target(full_latents, noise, timestep)

    i2v_mask = _build_prefix_condition_mask(
        total_frames=int(full_video.shape[0]),
        context_steps=1,
        lat_h=first_frame_latents.shape[2],
        lat_w=first_frame_latents.shape[3],
        device=device,
    ).to(first_frame_latents.dtype)
    y = torch.concat([i2v_mask, first_frame_latents], dim=0)

    seq_len = full_latents.shape[1] * full_latents.shape[2] * full_latents.shape[3] // (
        pipeline.patch_size[1] * pipeline.patch_size[2]
    )
    seq_len = int(math.ceil(seq_len / pipeline.sp_size)) * pipeline.sp_size
    timestep_tokens = build_prefix_timestep_tensor(latent_mask, timestep=timestep, seq_len=seq_len)
    adapter_payload = filter_state_condition_payload_for_adapter(state_condition_payload, pipeline.state_adapter)
    state_context = pipeline._build_state_context(adapter_payload, offload_model=False)

    with torch.amp.autocast("cuda", dtype=pipeline.param_dtype):
        noise_pred = pipeline.model(
            [noised_latents.to(dtype=model_dtype)],
            t=timestep_tokens.to(device=device, dtype=torch.float32),
            context=[text_context],
            seq_len=seq_len,
            y=[y],
            state_context=state_context,
            state_scale=state_scale,
        )[0]
        future_noise_pred = noise_pred[:, prefix_len:].float()
        future_target = training_target[:, prefix_len:].float()
        future_loss_mask = build_future_step_loss_mask(
            future_latent_steps=future_latent_steps,
            valid_future_latent_steps=valid_future_latent_steps,
            device=future_noise_pred.device,
            dtype=future_noise_pred.dtype,
        )
        squared_error = (future_noise_pred - future_target) ** 2
        masked_error = squared_error * future_loss_mask
        denom = future_loss_mask.sum() * future_noise_pred.shape[0] * future_noise_pred.shape[2] * future_noise_pred.shape[3]
        loss = masked_error.sum() / denom.clamp_min(1.0)
        loss = loss * scheduler.training_weight(timestep)

    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()
    return {
        "loss": float(loss.detach().cpu()),
        "training_frame_num": float(full_video.shape[0]),
        "prefix_latent_steps": float(prefix_len),
        "future_latent_steps": float(future_latent_steps),
        "valid_future_latent_steps": float(valid_future_latent_steps),
        "padded_future_latent_steps": float(future_latent_steps - valid_future_latent_steps),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Local Wan TI2V clean-prefix state adapter training requires CUDA, but torch.cuda.is_available() is False in the active environment."
        )

    device, device_id = resolve_device(args.device)
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
        rank=0,
        init_on_cpu=False,
    )

    first_sample = prepare_training_sample(bundle_records[0], frame_num=args.frame_num)
    first_condition_payload = build_ti2v_prefix_state_condition_payload(
        first_sample["state_condition"],
        device=pipeline.device,
        target_steps=max(int((first_sample["training_frame_num"] - 1) // pipeline.vae_stride[0]), 1),
    )
    pipeline._build_state_context(first_condition_payload, offload_model=False)
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
    best_payload = None

    for epoch in range(args.epochs):
        for record in bundle_records:
            sample = prepare_training_sample(record, frame_num=args.frame_num)
            metrics = run_step(
                pipeline=pipeline,
                scheduler=scheduler,
                sample=sample,
                optimizer=optimizer,
                output_size=output_size,
                state_scale=args.state_scale,
            )
            global_step += 1
            history_record = {
                "epoch": epoch + 1,
                "step": global_step,
                "sample_id": record.sample_id,
                "metrics": metrics,
            }
            history.append(history_record)
            if global_step % max(args.log_every, 1) == 0:
                print(json.dumps(history_record, ensure_ascii=False))

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
                    "loss": metrics["loss"],
                    "frame_num_arg": int(args.frame_num),
                    "aligned_frame_num": align_wan_frame_num(int(sample["training_frame_num"])),
                    "history_tail": history[-8:],
                    "training_semantics": "ti2v_clean_prefix_context_plus_future_infill",
                },
            )
            torch.save(current_payload, output_path)
            if metrics["loss"] <= best_loss:
                best_loss = metrics["loss"]
                best_payload = current_payload
                torch.save(best_payload, default_best_output(output_path))

            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
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
        "training_semantics": "ti2v_clean_prefix_context_plus_future_infill",
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
