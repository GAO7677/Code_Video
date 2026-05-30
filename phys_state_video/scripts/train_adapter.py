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
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_episodes)
    val_loader = None
    if args.val_data is not None:
        val_dataset = NpzEpisodeDataset(args.val_data)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_episodes,
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

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, loader, optimizer, args.device,
                                  cond_cfg, args.condition_mode)
        record = {"epoch": epoch + 1, "train": train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, None, args.device,
                                        cond_cfg, args.condition_mode)
            record["val"] = val_metrics
            print(
                f"epoch={epoch + 1} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f}")
        else:
            print(f"epoch={epoch + 1} train_loss={train_metrics['loss']:.6f}")
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


if __name__ == "__main__":
    main()
