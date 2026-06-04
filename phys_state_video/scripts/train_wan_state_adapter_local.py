from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.utils import require_torch
from phys_state_video.wan_adapter_training import (
    LocalWanFlowMatchScheduler,
    align_wan_frame_num,
    build_first_frame_mask,
    build_ti2v_timestep_tensor,
    build_ti2v_training_video,
    compute_ti2v_seq_len,
    discover_state_condition_bundles,
    is_ti2v_state_adapter_checkpoint,
    load_episode_npz,
    load_state_condition_npz,
    normalize_video_range,
    resize_and_center_crop_frames,
    select_ti2v_state_adapter_parameters,
    serialize_ti2v_state_adapter_checkpoint,
)
from phys_state_video.wan_bridge import _ensure_wan_importable

torch = require_torch()
F = torch.nn.functional


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


def load_checkpoint(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def run_step(
    *,
    pipeline,
    scheduler: LocalWanFlowMatchScheduler,
    sample: dict[str, object],
    optimizer,
    state_scale: float,
) -> dict[str, float]:
    device = pipeline.device
    model_dtype = pipeline.model.patch_embedding.weight.dtype

    input_video = sample["input_video"].to(device=device, dtype=torch.float32)
    first_frame = sample["first_frame"].to(device=device, dtype=torch.float32)
    state_condition = {
        key: torch.from_numpy(value).to(device=device, dtype=torch.float32)
        for key, value in sample["state_condition"].items()
    }

    optimizer.zero_grad(set_to_none=True)

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

    seq_len = compute_ti2v_seq_len(input_latents, patch_size=tuple(pipeline.patch_size[1:]))
    seq_len = int(math.ceil(seq_len / pipeline.sp_size)) * pipeline.sp_size
    timestep_tokens = build_ti2v_timestep_tensor(latent_mask, timestep=timestep, seq_len=seq_len)
    state_context = pipeline._build_state_context(state_condition, offload_model=False)
    del state_condition, input_latents, noise, first_frame_latents, latent_mask
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

    loss.backward()
    del context, state_context, timestep_tokens, noised_latents, training_target, noise_pred
    torch.cuda.empty_cache()
    optimizer.step()

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

    device, device_id = resolve_device(args.device)
    _ensure_wan_importable(args.wan_repo_root)

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

    first_sample = prepare_training_sample(
        bundle_records[0],
        output_size=output_size,
        frame_num=args.frame_num,
    )
    pipeline._build_state_context(
        {
            key: torch.from_numpy(value).to(device=pipeline.device, dtype=torch.float32)
            for key, value in first_sample["state_condition"].items()
        },
        offload_model=False,
    )
    if args.resume is not None:
        state_bundle = load_checkpoint(args.resume, map_location="cpu")
        if not is_ti2v_state_adapter_checkpoint(state_bundle):
            raise ValueError(f"resume checkpoint is not a TI2V state-adapter checkpoint: {args.resume}")
        pipeline.load_state_adapter(args.resume, state_condition=first_sample["state_condition"])

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
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
