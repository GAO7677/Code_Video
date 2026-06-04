from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.dataset import NpzPredictorDataset, collate_predictor_episodes
from phys_state_video.checkpoint_io import load_torch_checkpoint
from phys_state_video.predictor_visual_v3 import (
    VisualContextLatentPredictorV3,
    VisualLatentPredictorConfig,
    predictor_visual_v3_loss,
)
from phys_state_video.utils import require_torch

torch = require_torch()

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train the visual-context latent predictor v3.")
    parser.add_argument("--data", required=True, help="Directory containing episode .npz files.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-data", default=None, help="Optional validation episode directory.")
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--best-output", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--kl-scale", type=float, default=1e-4)
    return parser.parse_args()


def default_best_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.best")


def infer_best_from_history(history: list[dict]) -> tuple[int | None, float | None]:
    best_epoch = None
    best_metric = None
    for record in history:
        monitor = record.get("val", record["train"])["loss"]
        if best_metric is None or monitor < best_metric:
            best_metric = monitor
            best_epoch = int(record["epoch"])
    return best_epoch, best_metric


def epoch_snapshot_path(output_path: Path, epoch_index: int) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.epoch{epoch_index:03d}{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.epoch{epoch_index:03d}")


def save_checkpoint(output_path: Path, model, config: VisualLatentPredictorConfig, args, gpu_ids, history, best_epoch, best_metric):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "config": asdict(config),
            "gpu_ids": gpu_ids,
            "history": history,
            "model": model_state,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "kl_scale": args.kl_scale,
            "predictor_version": "visual_v3",
        },
        output_path,
    )


def run_epoch(model, loader, optimizer, device, kl_scale: float):
    running = {
        "loss": 0.0,
        "mse": 0.0,
        "visibility": 0.0,
        "existence": 0.0,
        "smoothness": 0.0,
        "scale_depth": 0.0,
        "motion_aux": 0.0,
        "velocity_align": 0.0,
        "latent_smooth": 0.0,
        "kl": 0.0,
    }
    is_train = optimizer is not None
    model.train(mode=is_train)
    for batch in loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        num_objects = batch["context_states"].shape[2]
        outputs = model(
            batch["context_frames"].to(device),
            prompt_token_ids=batch["prompt_token_ids"].to(device),
            prompt_token_mask=batch["prompt_token_mask"].to(device),
            future_steps=batch["future_states"].shape[1],
            num_objects=num_objects,
        )
        losses = predictor_visual_v3_loss(outputs, batch["future_states"].to(device), kl_scale=kl_scale)
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

    dataset = NpzPredictorDataset(args.data)
    pin_memory = not args.no_pin_memory
    persistent_workers = (not args.no_persistent_workers and args.num_workers > 0)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_predictor_episodes,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, **loader_kwargs)

    val_loader = None
    if args.val_data is not None:
        val_dataset = NpzPredictorDataset(args.val_data)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    sample = dataset[0]
    config = VisualLatentPredictorConfig(
        context_channels=sample.context_frames.shape[1],
        frame_height=sample.context_frames.shape[-2],
        frame_width=sample.context_frames.shape[-1],
        future_steps=sample.future_states.shape[0],
        max_objects=sample.context_states.shape[1],
    )
    base_model = VisualContextLatentPredictorV3(config).to(args.device)

    if gpu_ids and args.device.startswith("cuda") and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(base_model, device_ids=gpu_ids)
    else:
        model = base_model

    history = []
    best_epoch = None
    best_metric = None
    if args.resume is not None:
        resume_ckpt = load_torch_checkpoint(args.resume, map_location="cpu")
        base_model.load_state_dict(resume_ckpt["model"])
        history.extend(resume_ckpt.get("history", []))
        best_epoch = resume_ckpt.get("best_epoch")
        best_metric = resume_ckpt.get("best_metric")
        if best_metric is None and history:
            best_epoch, best_metric = infer_best_from_history(history)
        print(f"resumed weights from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
                "gpu_ids": gpu_ids,
                "resume": args.resume,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "train_episodes": len(dataset),
                "val_episodes": len(val_loader.dataset) if val_loader is not None else 0,
                "predictor_config": asdict(config),
                "kl_scale": args.kl_scale,
                "predictor_version": "visual_v3",
            },
        )

    output = Path(args.output)
    best_output = Path(args.best_output) if args.best_output is not None else default_best_output(output)
    start_epoch = len(history)
    for epoch in range(args.epochs):
        epoch_index = start_epoch + epoch + 1
        train_metrics = run_epoch(model, loader, optimizer, args.device, args.kl_scale)
        record = {"epoch": epoch_index, "train": train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, None, args.device, args.kl_scale)
            record["val"] = val_metrics
            print(f"epoch={epoch_index} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f}")
        else:
            print(f"epoch={epoch_index} train_loss={train_metrics['loss']:.6f}")
        if wandb_run is not None:
            log_payload = {f"train/{key}": value for key, value in train_metrics.items()}
            if "val" in record:
                log_payload.update({f"val/{key}": value for key, value in record["val"].items()})
            log_payload["epoch"] = epoch_index
            wandb.log(log_payload, step=epoch_index)
        history.append(record)
        monitor = record["val"]["loss"] if "val" in record else record["train"]["loss"]
        if best_metric is None or monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch_index
            save_checkpoint(best_output, model, config, args, gpu_ids, history, best_epoch, best_metric)
            print(f"saved best checkpoint to {best_output} (epoch={best_epoch}, metric={best_metric:.6f})")
        if args.save_every > 0 and epoch_index % args.save_every == 0:
            snapshot = epoch_snapshot_path(output, epoch_index)
            save_checkpoint(snapshot, model, config, args, gpu_ids, history, best_epoch, best_metric)
            print(f"saved snapshot checkpoint to {snapshot}")

    save_checkpoint(output, model, config, args, gpu_ids, history, best_epoch, best_metric)
    print(f"saved visual-context predictor checkpoint to {output}")
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
