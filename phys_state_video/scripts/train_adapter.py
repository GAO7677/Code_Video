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
from phys_state_video.config import AdapterConfig, ConditioningConfig
from phys_state_video.dataset import NpzEpisodeDataset, collate_episodes
from phys_state_video.experiment import apply_condition_mode
from phys_state_video.utils import require_torch

torch = require_torch()

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None


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
        choices=["state", "maps_only", "memory_only", "none"],
        help="Which state condition channels are exposed during training.",
    )
    parser.add_argument("--val-data",
                        default=None,
                        help="Optional validation episode directory.")
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
    return parser.parse_args()


def run_epoch(model, loader, optimizer, device, cond_cfg, condition_mode):
    running = {"loss": 0.0, "recon": 0.0, "state_aux": 0.0}
    is_train = optimizer is not None
    model.train(mode=is_train)
    for batch in loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        future_states = batch["future_states"].to(device)
        future_boxes = batch["future_boxes"].to(device)
        appearance = batch["appearance"].to(device)
        bundle = build_condition_bundle(future_states, future_boxes, appearance,
                                        cond_cfg)
        bundle = apply_condition_mode(bundle, condition_mode)
        outputs = model(batch["context_frames"].to(device), bundle.maps,
                        bundle.memory_tokens)
        losses = adapter_loss(
            outputs["frames"],
            batch["future_frames"].to(device),
            outputs["state_logits"],
            future_states,
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

    init_bundle = build_condition_bundle(
        torch.from_numpy(sample.future_states[None]).to(args.device),
        torch.from_numpy(sample.future_boxes[None]).to(args.device),
        torch.from_numpy(sample.appearance[None]).to(args.device),
        cond_cfg,
    )
    init_bundle = apply_condition_mode(init_bundle, args.condition_mode)
    with torch.no_grad():
        base_model(
            torch.from_numpy(sample.context_frames[None]).to(args.device),
            init_bundle.maps,
            init_bundle.memory_tokens,
        )

    if gpu_ids and args.device.startswith("cuda") and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(base_model, device_ids=gpu_ids)
    else:
        model = base_model

    history = []
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=args.device)
        base_model.load_state_dict(resume_ckpt["model"])
        history.extend(resume_ckpt.get("history", []))
        print(f"resumed weights from {args.resume}")

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
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "train_episodes": len(dataset),
                "val_episodes": len(val_loader.dataset) if val_loader is not None else 0,
            },
        )

    start_epoch = len(history)
    for epoch in range(args.epochs):
        epoch_index = start_epoch + epoch + 1
        train_metrics = run_epoch(model, loader, optimizer, args.device, cond_cfg, args.condition_mode)
        record = {"epoch": epoch_index, "train": train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, None, args.device, cond_cfg, args.condition_mode)
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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.module.state_dict() if isinstance(
        model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        "config": asdict(adapter_cfg),
        "conditioning": asdict(cond_cfg),
        "condition_mode": args.condition_mode,
        "gpu_ids": gpu_ids,
        "history": history,
        "model": model_state
    }, output)
    print(f"saved adapter checkpoint to {output}")
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
