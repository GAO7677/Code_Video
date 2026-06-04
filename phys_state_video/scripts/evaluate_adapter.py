from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.adapter import TinyVideoBackbone, adapter_loss
from phys_state_video.checkpoint_io import load_torch_checkpoint
from phys_state_video.conditioning import build_condition_bundle
from phys_state_video.config import AdapterConfig, ConditioningConfig, PredictorConfig
from phys_state_video.dataset import NpzEpisodeDataset, collate_episodes
from phys_state_video.experiment import (apply_condition_mode,
                                         compute_state_metrics,
                                         perturb_condition_bundle)
from phys_state_video.predictor import FutureStatePredictor
from phys_state_video.proxy_state import extract_primary_track
from phys_state_video.utils import detach_to_cpu_numpy, require_torch

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained tiny state-conditioned video adapter.")
    parser.add_argument("--data", required=True, help="Episode directory.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--condition-mode",
        default=None,
        choices=["state", "maps_only", "memory_only", "latent_only", "none"],
        help="Override condition mode used at evaluation time.",
    )
    parser.add_argument(
        "--corruption",
        default="none",
        choices=["none", "perturbed"],
        help="Optional corruption applied after condition construction.",
    )
    parser.add_argument("--output", default=None, help="Optional metrics json.")
    return parser.parse_args()


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

    ckpt = load_torch_checkpoint(args.checkpoint, map_location=args.device)
    condition_mode = args.condition_mode or ckpt.get("condition_mode", "state")
    state_loss_weights = ckpt.get("state_loss_weights")
    state_loss_scale = float(ckpt.get("state_loss_scale", 0.1))
    spatial_loss_scale = float(ckpt.get("spatial_loss_scale", 0.0))
    spatial_foreground_weight = float(ckpt.get("spatial_foreground_weight", 4.0))
    dataset = NpzEpisodeDataset(args.data)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         collate_fn=collate_episodes)
    model = TinyVideoBackbone(AdapterConfig(**ckpt["config"])).to(args.device)
    load_model_state(model, ckpt["model"], args.checkpoint)
    model.eval()
    predictor_model = None
    predictor_checkpoint = ckpt.get("predictor_checkpoint")
    if predictor_checkpoint:
        predictor_ckpt = load_torch_checkpoint(predictor_checkpoint, map_location="cpu")
        predictor_model = FutureStatePredictor(PredictorConfig(**predictor_ckpt["config"])).to(args.device)
        load_model_state(predictor_model, predictor_ckpt["model"], predictor_checkpoint)
        predictor_model.eval()
    cond_cfg = ConditioningConfig(**ckpt["conditioning"])
    state_loss_weight_tensor = None
    if state_loss_weights is not None:
        state_loss_weight_tensor = torch.tensor(state_loss_weights, dtype=torch.float32, device=args.device)

    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "state_aux": 0.0,
        "spatial_aux": 0.0,
        "center_error": 0.0,
        "log_scale_error": 0.0,
        "visibility_error": 0.0,
    }
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch_size = batch["future_frames"].shape[0]
            future_states = batch["future_states"].to(args.device)
            future_boxes = batch["future_boxes"].to(args.device)
            appearance = batch["appearance"].to(args.device)
            future_latent_tokens = None
            if predictor_model is not None:
                predictor_outputs = predictor_model(
                    batch["context_states"].to(args.device),
                    appearance,
                    batch["camera"].to(args.device),
                    prompt_token_ids=batch["prompt_token_ids"].to(args.device),
                    prompt_token_mask=batch["prompt_token_mask"].to(args.device),
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
            if args.corruption == "perturbed":
                bundle = perturb_condition_bundle(bundle)
            outputs = model(batch["context_frames"].to(args.device), bundle.maps,
                            bundle.memory_tokens,
                            future_latent_tokens=future_latent_tokens,
                            context_states=batch["context_states"].to(args.device),
                            prompt_token_ids=batch["prompt_token_ids"].to(args.device),
                            prompt_token_mask=batch["prompt_token_mask"].to(args.device))
            target_spatial_maps = target_bundle.maps[:, :, 0:2]
            losses = adapter_loss(outputs["frames"],
                                  batch["future_frames"].to(args.device),
                                  outputs["state_logits"],
                                  future_states,
                                  state_loss_weights=state_loss_weight_tensor,
                                  state_loss_scale=state_loss_scale,
                                  predicted_spatial_logits=outputs.get("spatial_logits"),
                                  target_spatial_maps=target_spatial_maps,
                                  spatial_loss_scale=spatial_loss_scale,
                                  spatial_foreground_weight=spatial_foreground_weight)
            for key in ("loss", "recon", "state_aux", "spatial_aux"):
                totals[key] += float(losses[key].detach().cpu()) * batch_size

            generated = detach_to_cpu_numpy(outputs["frames"])
            target_states_np = detach_to_cpu_numpy(batch["future_states"])
            for sample_idx in range(generated.shape[0]):
                proxy = extract_primary_track(generated[sample_idx])
                metrics = compute_state_metrics(proxy.states,
                                                target_states_np[sample_idx])
                for key, value in metrics.items():
                    totals[key] += value
                count += 1

    denom = max(count, 1)
    averages = {key: value / denom for key, value in totals.items()}
    result = {
        "samples": count,
        "condition_mode": condition_mode,
        "corruption": args.corruption,
        "state_loss_weights": state_loss_weights,
        "state_loss_scale": state_loss_scale,
        "spatial_loss_scale": spatial_loss_scale,
        "spatial_foreground_weight": spatial_foreground_weight,
        "predictor_checkpoint": predictor_checkpoint,
        "metrics": averages,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                          encoding="utf-8")


if __name__ == "__main__":
    main()
