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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = NpzEpisodeDataset(args.data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_episodes)
    sample = dataset[0]
    cond_cfg = ConditioningConfig(
        frame_height=sample.context_frames.shape[-2],
        frame_width=sample.context_frames.shape[-1],
    )
    adapter_cfg = AdapterConfig(freeze_backbone=args.freeze_backbone, future_steps=sample.future_frames.shape[0])
    model = TinyVideoBackbone(adapter_cfg).to(args.device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            future_states = batch["future_states"].to(args.device)
            future_boxes = batch["future_boxes"].to(args.device)
            appearance = batch["appearance"].to(args.device)
            bundle = build_condition_bundle(future_states, future_boxes, appearance, cond_cfg)
            outputs = model(batch["context_frames"].to(args.device), bundle.maps, bundle.memory_tokens)
            losses = adapter_loss(
                outputs["frames"],
                batch["future_frames"].to(args.device),
                outputs["state_logits"],
                future_states,
            )
            losses["loss"].backward()
            optimizer.step()
            running += float(losses["loss"].detach().cpu())
        avg = running / max(len(loader), 1)
        print(f"epoch={epoch + 1} loss={avg:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(adapter_cfg), "conditioning": asdict(cond_cfg), "model": model.state_dict()}, output)
    print(f"saved adapter checkpoint to {output}")


if __name__ == "__main__":
    main()
