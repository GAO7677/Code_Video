from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.config import PredictorConfig
from phys_state_video.dataset import NpzEpisodeDataset, collate_episodes
from phys_state_video.predictor import FutureStatePredictor, predictor_loss
from phys_state_video.utils import require_torch

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the future object-state predictor.")
    parser.add_argument("--data", required=True, help="Directory containing episode .npz files.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = NpzEpisodeDataset(args.data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_episodes)
    sample = dataset[0]
    config = PredictorConfig(
        appearance_dim=sample.appearance.shape[-1],
        camera_dim=sample.camera.shape[-1],
        future_steps=sample.future_states.shape[0],
    )
    model = FutureStatePredictor(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["context_states"].to(args.device),
                batch["appearance"].to(args.device),
                batch["camera"].to(args.device),
                batch["prompts"],
            )
            losses = predictor_loss(outputs["states"], batch["future_states"].to(args.device))
            losses["loss"].backward()
            optimizer.step()
            running += float(losses["loss"].detach().cpu())
        avg = running / max(len(loader), 1)
        print(f"epoch={epoch + 1} loss={avg:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "model": model.state_dict()}, output)
    print(f"saved predictor checkpoint to {output}")


if __name__ == "__main__":
    main()
