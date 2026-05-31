from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.adapter import TinyVideoBackbone
from phys_state_video.config import AdapterConfig, ConditioningConfig, PredictorConfig
from phys_state_video.dataset import NpzEpisodeDataset, collate_episodes
from phys_state_video.pipeline import StateConditionedGenerationPipeline
from phys_state_video.predictor import FutureStatePredictor
from phys_state_video.projection import ConfidenceAwareProjector
from phys_state_video.utils import detach_to_cpu_numpy, require_torch

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(description="Run state-conditioned future video generation.")
    parser.add_argument("--episode", required=True, help="Episode .npz file.")
    parser.add_argument("--predictor", required=True, help="Predictor checkpoint.")
    parser.add_argument("--adapter", required=True, help="Adapter checkpoint.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


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


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = NpzEpisodeDataset(args.episode)
    batch = collate_episodes([dataset[0]])

    predictor_ckpt = load_checkpoint(args.predictor, map_location=args.device)
    adapter_ckpt = load_checkpoint(args.adapter, map_location=args.device)
    predictor = FutureStatePredictor(PredictorConfig(**predictor_ckpt["config"])).to(args.device)
    predictor.load_state_dict(predictor_ckpt["model"])
    adapter = TinyVideoBackbone(AdapterConfig(**adapter_ckpt["config"])).to(args.device)
    load_model_state(adapter, adapter_ckpt["model"], args.adapter)
    pipeline = StateConditionedGenerationPipeline(
        predictor=predictor.eval(),
        projector=ConfidenceAwareProjector(),
        video_model=adapter.eval(),
        conditioning_config=ConditioningConfig(**adapter_ckpt["conditioning"]),
    )

    with torch.no_grad():
        outputs = pipeline.generate(
            context_frames=batch["context_frames"].to(args.device),
            context_states=batch["context_states"].to(args.device),
            context_boxes=batch["context_boxes"].to(args.device),
            appearance=batch["appearance"].to(args.device),
            camera=batch["camera"].to(args.device),
            prompts=batch["prompts"],
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "inference_outputs.npz",
        predicted_states=detach_to_cpu_numpy(outputs["predicted_states"]),
        future_boxes=detach_to_cpu_numpy(outputs["future_boxes"]),
        generated_frames=detach_to_cpu_numpy(outputs["generated_frames"]),
        condition_maps=detach_to_cpu_numpy(outputs["condition_maps"]),
        state_logits=detach_to_cpu_numpy(outputs["state_logits"]),
    )
    (output_dir / "meta.json").write_text(json.dumps({"prompt": batch["prompts"][0]}, ensure_ascii=False, indent=2))
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
