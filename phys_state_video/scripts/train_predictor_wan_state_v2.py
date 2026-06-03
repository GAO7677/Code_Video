from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.dataset import NpzPredictorDataset, collate_predictor_episodes
from phys_state_video.predictor_wan_state_v2 import (
    WanStateLatentPredictorV2,
    WanStateLatentPredictorV2Config,
    resample_temporal_states,
    wan_state_predictor_v2_loss,
)
from phys_state_video.utils import require_torch
from phys_state_video.wan_bridge import WanLatentExtractor
from phys_state_video.wan_state_v2_helpers import (
    MockLatentExtractor,
    compute_future_latent_steps,
    resample_camera_to_latent_steps,
)

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the latent-time Wan state predictor v2.")
    parser.add_argument("--data", required=True, help="Directory containing episode .npz files.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs-context", type=int, default=1)
    parser.add_argument("--epochs-future", type=int, default=1)
    parser.add_argument("--epochs-joint", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-source", choices=["mock", "wan"], default="mock")
    parser.add_argument("--wan-ckpt-dir", default=None)
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--wan-task", default="ti2v-5B")
    parser.add_argument("--mock-latent-channels", type=int, default=16)
    parser.add_argument("--mock-latent-height", type=int, default=8)
    parser.add_argument("--mock-latent-width", type=int, default=8)
    parser.add_argument("--latent-smooth-scale", type=float, default=0.05)
    return parser.parse_args()


def default_best_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.best")


def build_latent_extractor(args):
    if args.latent_source == "mock":
        return MockLatentExtractor(
            latent_channels=args.mock_latent_channels,
            latent_height=args.mock_latent_height,
            latent_width=args.mock_latent_width,
            device=args.device,
        )
    if args.wan_ckpt_dir is None:
        raise ValueError("--wan-ckpt-dir is required when --latent-source=wan")
    return WanLatentExtractor(
        ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        task=args.wan_task,
        device=args.device,
    )


def infer_model_config(sample, latent_extractor) -> WanStateLatentPredictorV2Config:
    sample_frames = torch.from_numpy(sample.context_frames[None]).to(latent_extractor.device)
    with torch.no_grad():
        context_latents = latent_extractor.encode_context_frames_raw(sample_frames)
    future_latent_steps = compute_future_latent_steps(
        context_steps=sample.context_frames.shape[0],
        future_steps=sample.future_states.shape[0],
        temporal_stride=latent_extractor.temporal_stride,
    )
    return WanStateLatentPredictorV2Config(
        latent_channels=context_latents.shape[2],
        camera_dim=sample.camera.shape[-1],
        max_context_latent_steps=context_latents.shape[1],
        max_future_latent_steps=future_latent_steps,
        max_objects=sample.context_states.shape[1],
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
            "model": model.state_dict(),
            "history": history,
            "predictor_version": "wan_state_v2_latent_time",
            "latent_source": args.latent_source,
            "mock_latent_channels": getattr(args, "mock_latent_channels", None),
            "mock_latent_height": getattr(args, "mock_latent_height", None),
            "mock_latent_width": getattr(args, "mock_latent_width", None),
            "wan_ckpt_dir": args.wan_ckpt_dir,
            "wan_repo_root": args.wan_repo_root,
            "wan_task": args.wan_task,
            "temporal_stride": getattr(args, "temporal_stride", None),
            "train_schedule": {
                "epochs_context": args.epochs_context,
                "epochs_future": args.epochs_future,
                "epochs_joint": args.epochs_joint,
            },
        },
        output_path,
    )


def run_epoch(model, latent_extractor, loader, optimizer, device, train_stage: str, latent_smooth_scale: float):
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
    }
    for batch in loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        context_frames = batch["context_frames"].to(device)
        camera = batch["camera"].to(device)
        context_states = batch["context_states"].to(device)
        future_states = batch["future_states"].to(device)

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

        outputs = model(
            context_latents=context_latents,
            camera=camera_latent,
            prompt_token_ids=batch["prompt_token_ids"].to(device),
            prompt_token_mask=batch["prompt_token_mask"].to(device),
            future_latent_steps=future_latent_steps,
            num_objects=context_states.shape[2],
        )
        losses = wan_state_predictor_v2_loss(
            outputs=outputs,
            context_target=context_target,
            future_target=future_target,
            train_stage=train_stage,
            latent_smooth_scale=latent_smooth_scale,
        )
        if is_train:
            losses["loss"].backward()
            optimizer.step()
        for key in running:
            running[key] += float(losses[key].detach().cpu())
    denom = max(len(loader), 1)
    return {key: value / denom for key, value in running.items()}


def main():
    args = parse_args()
    dataset = NpzPredictorDataset(args.data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_predictor_episodes,
        num_workers=args.num_workers,
    )

    latent_extractor = build_latent_extractor(args)
    args.temporal_stride = latent_extractor.temporal_stride
    config = infer_model_config(dataset[0], latent_extractor)
    model = WanStateLatentPredictorV2(config).to(args.device)

    schedule = [
        ("context_only", args.epochs_context),
        ("future_only", args.epochs_future),
        ("joint_finetune", args.epochs_joint),
    ]
    history = []
    for stage_name, num_epochs in schedule:
        if num_epochs <= 0:
            continue
        configure_stage(model, stage_name)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        for epoch in range(num_epochs):
            metrics = run_epoch(
                model=model,
                latent_extractor=latent_extractor,
                loader=loader,
                optimizer=optimizer,
                device=args.device,
                train_stage=stage_name,
                latent_smooth_scale=args.latent_smooth_scale,
            )
            record = {"stage": stage_name, "epoch": epoch + 1, "metrics": metrics}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))

    output = Path(args.output)
    save_checkpoint(output, model, config, args, history)
    save_checkpoint(default_best_output(output), model, config, args, history)
    print(f"saved Wan state predictor v2 checkpoint to {output}")


if __name__ == "__main__":
    main()
