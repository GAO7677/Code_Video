from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.adapter import TinyVideoBackbone, adapter_loss
from phys_state_video.conditioning import build_condition_bundle
from phys_state_video.config import AdapterConfig, ConditioningConfig, PredictorConfig
from phys_state_video.dataset import NpzEpisodeDataset, collate_episodes
from phys_state_video.experiment import apply_condition_mode
from phys_state_video.predictor import FutureStatePredictor
from phys_state_video.utils import require_torch

torch = require_torch()

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None


DEFAULT_STATE_LOSS_WEIGHTS = [1.0, 1.0, 0.05, 1.0, 0.25, 0.25, 0.05, 1.0, 0.25, 0.05]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the state-conditioned video adapter.")
    parser.add_argument("--data", required=True, help="Directory containing episode .npz files.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--condition-mode",
        default="state",
        choices=["state", "maps_only", "memory_only", "latent_only", "none"],
        help="Which state condition channels are exposed during training.",
    )
    parser.add_argument("--val-data",
                        default=None,
                        help="Optional validation episode directory.")
    parser.add_argument(
        "--predictor-checkpoint",
        default=None,
        help="Optional frozen predictor checkpoint used to provide future latent tokens during adapter training.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated CUDA device ids for DataParallel, for example '0,1,2,3'.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint to resume model weights and append history from.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--state-loss-weights",
        default=",".join(str(value) for value in DEFAULT_STATE_LOSS_WEIGHTS),
        help="Comma-separated per-dimension weights for the 10D state auxiliary loss.",
    )
    parser.add_argument(
        "--state-loss-scale",
        type=float,
        default=0.1,
        help="Global multiplier applied to the weighted state auxiliary loss.",
    )
    parser.add_argument(
        "--best-output",
        default=None,
        help="Optional best-checkpoint path. Defaults to '<output stem>.best<suffix>'.",
    )
    parser.add_argument(
        "--spatial-loss-scale",
        type=float,
        default=0.5,
        help="Global multiplier for the spatial auxiliary loss on center/bbox maps.",
    )
    parser.add_argument(
        "--spatial-foreground-weight",
        type=float,
        default=4.0,
        help="Extra weight assigned to foreground pixels in the spatial auxiliary loss.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If > 0, also save epoch snapshots every N epochs next to the main checkpoint.",
    )
    return parser.parse_args()


def parse_state_loss_weights(value: str) -> list[float]:
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != 10:
        raise ValueError(f"--state-loss-weights must contain 10 values, got {len(weights)}")
    return weights


def default_best_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.best")


def epoch_snapshot_path(output_path: Path, epoch_index: int) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.epoch{epoch_index:03d}{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.epoch{epoch_index:03d}")


def infer_best_from_history(history: list[dict]) -> tuple[int | None, float | None]:
    best_epoch = None
    best_metric = None
    for record in history:
        monitor = record.get("val", record["train"])["loss"]
        if best_metric is None or monitor < best_metric:
            best_metric = monitor
            best_epoch = int(record["epoch"])
    return best_epoch, best_metric


def same_loss_setup(resume_ckpt: dict, args, state_loss_weights: list[float]) -> bool:
    return (
        resume_ckpt.get("condition_mode", "state") == args.condition_mode
        and list(resume_ckpt.get("state_loss_weights", state_loss_weights)) == list(state_loss_weights)
        and float(resume_ckpt.get("state_loss_scale", args.state_loss_scale)) == float(args.state_loss_scale)
        and float(resume_ckpt.get("spatial_loss_scale", 0.0)) == float(args.spatial_loss_scale)
        and float(resume_ckpt.get("spatial_foreground_weight", args.spatial_foreground_weight))
        == float(args.spatial_foreground_weight)
    )


def save_checkpoint(
    output_path: Path,
    model,
    adapter_cfg: AdapterConfig,
    cond_cfg: ConditioningConfig,
    args,
    gpu_ids,
    history,
    state_loss_weights: list[float],
    best_epoch: int | None,
    best_metric: float | None,
    predictor_checkpoint: str | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "config": asdict(adapter_cfg),
            "conditioning": asdict(cond_cfg),
            "condition_mode": args.condition_mode,
            "gpu_ids": gpu_ids,
            "history": history,
            "model": model_state,
            "state_loss_weights": state_loss_weights,
            "state_loss_scale": args.state_loss_scale,
            "spatial_loss_scale": args.spatial_loss_scale,
            "spatial_foreground_weight": args.spatial_foreground_weight,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "predictor_checkpoint": predictor_checkpoint,
        },
        output_path,
    )


def load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def load_model_state(module, state_dict, checkpoint_label: str) -> None:
    try:
        module.load_state_dict(state_dict)
    except RuntimeError as exc:
        message = str(exc)
        key_mismatch = "Missing key(s) in state_dict" in message or "Unexpected key(s) in state_dict" in message
        if not key_mismatch:
            raise
        incompatible = module.load_state_dict(state_dict, strict=False)
        print(
            f"loaded {checkpoint_label} with non-strict state dict; "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )


def run_epoch(
    model,
    predictor_model,
    loader,
    optimizer,
    device,
    cond_cfg,
    condition_mode,
    state_loss_weights,
    state_loss_scale,
    spatial_loss_scale,
    spatial_foreground_weight,
):
    running = {"loss": 0.0, "recon": 0.0, "state_aux": 0.0, "spatial_aux": 0.0}
    is_train = optimizer is not None
    model.train(mode=is_train)
    if predictor_model is not None:
        predictor_model.eval()
    for batch in loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        future_states = batch["future_states"].to(device)
        future_boxes = batch["future_boxes"].to(device)
        appearance = batch["appearance"].to(device)
        future_latent_tokens = None
        if predictor_model is not None:
            with torch.no_grad():
                predictor_outputs = predictor_model(
                    batch["context_states"].to(device),
                    appearance,
                    batch["camera"].to(device),
                    prompt_token_ids=batch["prompt_token_ids"].to(device),
                    prompt_token_mask=batch["prompt_token_mask"].to(device),
                    future_steps=future_states.shape[1],
                )
            future_latent_tokens = predictor_outputs["latents"]
        target_bundle = build_condition_bundle(
            future_states,
            future_boxes,
            appearance,
            cond_cfg,
        )
        bundle = apply_condition_mode(target_bundle, condition_mode)
        outputs = model(batch["context_frames"].to(device), bundle.maps,
                        bundle.memory_tokens,
                        future_latent_tokens=future_latent_tokens,
                        context_states=batch["context_states"].to(device),
                        prompt_token_ids=batch["prompt_token_ids"].to(device),
                        prompt_token_mask=batch["prompt_token_mask"].to(device))
        target_spatial_maps = target_bundle.maps[:, :, 0:2]
        losses = adapter_loss(
            outputs["frames"],
            batch["future_frames"].to(device),
            outputs["state_logits"],
            future_states,
            state_loss_weights=state_loss_weights,
            state_loss_scale=state_loss_scale,
            predicted_spatial_logits=outputs["spatial_logits"],
            target_spatial_maps=target_spatial_maps,
            spatial_loss_scale=spatial_loss_scale,
            spatial_foreground_weight=spatial_foreground_weight,
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
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(item) for item in args.gpu_ids.split(",") if item.strip()]
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    state_loss_weights = parse_state_loss_weights(args.state_loss_weights)
    dataset = NpzEpisodeDataset(args.data)
    pin_memory = not args.no_pin_memory
    persistent_workers = (not args.no_persistent_workers and args.num_workers > 0)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_episodes,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, **loader_kwargs)
    val_loader = None
    if args.val_data is not None:
        val_dataset = NpzEpisodeDataset(args.val_data)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )
    sample = dataset[0]
    cond_cfg = ConditioningConfig(
        frame_height=sample.context_frames.shape[-2],
        frame_width=sample.context_frames.shape[-1],
    )
    adapter_cfg = AdapterConfig(freeze_backbone=args.freeze_backbone, future_steps=sample.future_frames.shape[0])
    base_model = TinyVideoBackbone(adapter_cfg).to(args.device)

    init_target_bundle = build_condition_bundle(
        torch.from_numpy(sample.future_states[None]).to(args.device),
        torch.from_numpy(sample.future_boxes[None]).to(args.device),
        torch.from_numpy(sample.appearance[None]).to(args.device),
        cond_cfg,
    )
    init_bundle = apply_condition_mode(init_target_bundle, args.condition_mode)
    with torch.no_grad():
        base_model(
            torch.from_numpy(sample.context_frames[None]).to(args.device),
            init_bundle.maps,
            init_bundle.memory_tokens,
            context_states=torch.from_numpy(sample.context_states[None]).to(args.device),
            prompts=[sample.prompt],
        )

    if gpu_ids and args.device.startswith("cuda") and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(base_model, device_ids=gpu_ids)
    else:
        model = base_model

    history = []
    best_epoch = None
    best_metric = None
    if args.resume is not None:
        # Load checkpoints on CPU first to avoid device-specific restore issues
        # when resuming under a different visible CUDA device mapping.
        resume_ckpt = load_checkpoint(args.resume, map_location="cpu")
        load_model_state(base_model, resume_ckpt["model"], args.resume)
        history.extend(resume_ckpt.get("history", []))
        if same_loss_setup(resume_ckpt, args, state_loss_weights):
            best_epoch = resume_ckpt.get("best_epoch")
            best_metric = resume_ckpt.get("best_metric")
            if best_metric is None and history:
                best_epoch, best_metric = infer_best_from_history(history)
        else:
            print("resume checkpoint uses a different loss setup; resetting best metric tracking for this run")
            best_epoch = None
            best_metric = None
        print(f"resumed weights from {args.resume}")

    predictor_model = None
    if args.predictor_checkpoint is not None:
        predictor_ckpt = load_checkpoint(args.predictor_checkpoint, map_location="cpu")
        predictor_cfg = PredictorConfig(**predictor_ckpt["config"])
        predictor_model = FutureStatePredictor(predictor_cfg).to(args.device)
        load_model_state(predictor_model, predictor_ckpt["model"], args.predictor_checkpoint)
        predictor_model.eval()
        for param in predictor_model.parameters():
            param.requires_grad = False
        print(f"loaded frozen predictor from {args.predictor_checkpoint}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    wandb_run = None
    if args.wandb_mode != "disabled":
        if wandb is None:
            raise RuntimeError("wandb is not installed in the active environment.")
        wandb_run = wandb.init(
            project=args.wandb_project or "phys-state-video",
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            config={
                "data": args.data,
                "val_data": args.val_data,
                "output": args.output,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": args.device,
                "freeze_backbone": args.freeze_backbone,
                "condition_mode": args.condition_mode,
                "gpu_ids": gpu_ids,
                "resume": args.resume,
                "predictor_checkpoint": args.predictor_checkpoint,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "train_episodes": len(dataset),
                "val_episodes": len(val_loader.dataset) if val_loader is not None else 0,
                "state_loss_weights": state_loss_weights,
                "state_loss_scale": args.state_loss_scale,
                "spatial_loss_scale": args.spatial_loss_scale,
                "spatial_foreground_weight": args.spatial_foreground_weight,
            },
        )

    output = Path(args.output)
    best_output = Path(args.best_output) if args.best_output is not None else default_best_output(output)
    state_loss_weight_tensor = torch.tensor(state_loss_weights, dtype=torch.float32, device=args.device)
    start_epoch = len(history)
    for epoch in range(args.epochs):
        epoch_index = start_epoch + epoch + 1
        train_metrics = run_epoch(
            model,
            predictor_model,
            loader,
            optimizer,
            args.device,
            cond_cfg,
            args.condition_mode,
            state_loss_weight_tensor,
            args.state_loss_scale,
            args.spatial_loss_scale,
            args.spatial_foreground_weight,
        )
        record = {"epoch": epoch_index, "train": train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    predictor_model,
                    val_loader,
                    None,
                    args.device,
                    cond_cfg,
                    args.condition_mode,
                    state_loss_weight_tensor,
                    args.state_loss_scale,
                    args.spatial_loss_scale,
                    args.spatial_foreground_weight,
                )
            record["val"] = val_metrics
            print(f"epoch={epoch_index} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f}")
        else:
            print(f"epoch={epoch_index} train_loss={train_metrics['loss']:.6f}")
        if wandb_run is not None:
            log_payload = {f"train/{key}": value for key, value in train_metrics.items()}
            if "val" in record:
                log_payload.update({f"val/{key}": value for key, value in record["val"].items()})
            log_payload["epoch"] = epoch_index
            if best_metric is not None:
                log_payload["best/metric"] = best_metric
            if best_epoch is not None:
                log_payload["best/epoch"] = best_epoch
            wandb.log(log_payload, step=epoch_index)
        history.append(record)
        monitor = record["val"]["loss"] if "val" in record else record["train"]["loss"]
        if best_metric is None or monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch_index
            save_checkpoint(
                best_output,
                model,
                adapter_cfg,
                cond_cfg,
                args,
                gpu_ids,
                history,
                state_loss_weights,
                best_epoch,
                best_metric,
                args.predictor_checkpoint,
            )
            print(f"saved best checkpoint to {best_output} (epoch={best_epoch}, metric={best_metric:.6f})")
        if args.save_every > 0 and epoch_index % args.save_every == 0:
            snapshot = epoch_snapshot_path(output, epoch_index)
            save_checkpoint(
                snapshot,
                model,
                adapter_cfg,
                cond_cfg,
                args,
                gpu_ids,
                history,
                state_loss_weights,
                best_epoch,
                best_metric,
                args.predictor_checkpoint,
            )
            print(f"saved snapshot checkpoint to {snapshot}")

    save_checkpoint(
        output,
        model,
        adapter_cfg,
        cond_cfg,
        args,
        gpu_ids,
        history,
        state_loss_weights,
        best_epoch,
        best_metric,
        args.predictor_checkpoint,
    )
    print(f"saved adapter checkpoint to {output}")
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
