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

from phys_state_video.dataset import NpzPredictorDataset, collate_predictor_episodes
from phys_state_video.predictor_wan_state_v2 import resample_temporal_states
from phys_state_video.wan_predictor_runtime import (
    build_predictor_latent_extractor,
    build_predictor_prompt_context_encoder,
    load_wan_state_predictor,
)
from phys_state_video.utils import detach_to_cpu_numpy, require_torch
from phys_state_video.wan_state_v2_helpers import (
    compute_future_latent_steps,
    resample_camera_to_latent_steps,
    split_context_future_camera,
)

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(description="Run latent-time Wan state predictor v2 inference.")
    parser.add_argument("--episode", required=True, help="Episode .npz file.")
    parser.add_argument("--predictor", required=True, help="Predictor v2 checkpoint.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wan-ckpt-dir", default=None)
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--wan-task", default="i2v-A14B")
    parser.add_argument(
        "--predictor-wan-task",
        default=None,
        help="Optional Wan task override used only for predictor latent extraction and prompt encoding.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = NpzPredictorDataset(args.episode)
    batch = collate_predictor_episodes([dataset[0]])

    predictor, checkpoint = load_wan_state_predictor(args.predictor, args.device)
    latent_extractor = build_predictor_latent_extractor(
        wan_ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        device=args.device,
        predictor_ckpt=checkpoint,
        default_wan_task=args.wan_task,
        predictor_wan_task=args.predictor_wan_task,
        context="Wan predictor v2 inference",
    )
    prompt_context_encoder = build_predictor_prompt_context_encoder(
        wan_ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        device=args.device,
        predictor_ckpt=checkpoint,
        default_wan_task=args.wan_task,
        predictor_wan_task=args.predictor_wan_task,
        context="Wan predictor v2 inference",
    )

    context_frames = batch["context_frames"].to(args.device)
    context_states = batch["context_states"].to(args.device)
    future_states = batch["future_states"].to(args.device)
    camera = batch["camera"].to(args.device)

    with torch.no_grad():
        context_latents = latent_extractor.encode_context_frames_raw(context_frames)
        context_latent_steps = context_latents.shape[1]
        future_latent_steps = compute_future_latent_steps(
            context_steps=context_frames.shape[1],
            future_steps=future_states.shape[1],
            temporal_stride=latent_extractor.temporal_stride,
        )
        context_camera, future_camera = split_context_future_camera(
            camera,
            context_steps=int(context_frames.shape[1]),
            future_steps=int(future_states.shape[1]),
        )
        camera_latent = resample_camera_to_latent_steps(context_camera, context_latent_steps)
        future_camera_latent = resample_camera_to_latent_steps(future_camera, future_latent_steps)
        context_target = resample_temporal_states(context_states, context_latent_steps)
        future_target = resample_temporal_states(future_states, future_latent_steps)
        prompt_context, prompt_mask = prompt_context_encoder.encode_prompts(list(batch["prompts"]))
        outputs = predictor(
            context_latents=context_latents,
            camera=camera_latent,
            prompt_context=prompt_context.to(args.device),
            prompt_mask=prompt_mask.to(args.device),
            future_latent_steps=future_latent_steps,
            num_objects=context_states.shape[2],
            future_camera=future_camera_latent,
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "wan_state_v2_inference_outputs.npz",
        context_frames=detach_to_cpu_numpy(batch["context_frames"][0]),
        context_latents=detach_to_cpu_numpy(context_latents[0]),
        context_state_targets=detach_to_cpu_numpy(context_target[0]),
        future_state_targets=detach_to_cpu_numpy(future_target[0]),
        prompt_context=detach_to_cpu_numpy(prompt_context[0]),
        prompt_mask=detach_to_cpu_numpy(prompt_mask[0]),
        context_state_maps=detach_to_cpu_numpy(outputs["context_state_maps"][0]),
        future_state_maps=detach_to_cpu_numpy(outputs["future_state_maps"][0]),
        condition_maps=detach_to_cpu_numpy(outputs["condition_maps"][0]),
        memory_tokens=detach_to_cpu_numpy(outputs["memory_tokens"][0]),
        context_object_slots=detach_to_cpu_numpy(outputs["debug_context_object_slots"][0]),
        future_object_slots=detach_to_cpu_numpy(outputs["debug_future_object_slots"][0]),
        projected_future_state_maps=detach_to_cpu_numpy(outputs["debug_projected_future_state_maps"][0]),
        context_state_predictions=detach_to_cpu_numpy(outputs["context_state_predictions"][0]),
        future_state_predictions=detach_to_cpu_numpy(outputs["future_state_predictions"][0]),
    )
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "predictor_version": checkpoint.get("predictor_version", "wan_state_v2_latent_time"),
                "latent_source": checkpoint.get("latent_source", "wan"),
                "prompt": batch["prompts"][0],
                "context_frame_steps": int(context_frames.shape[1]),
                "context_latent_steps": int(context_latent_steps),
                "future_frame_steps": int(future_states.shape[1]),
                "future_latent_steps": int(future_latent_steps),
                "temporal_stride": int(latent_extractor.temporal_stride),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved Wan state predictor v2 inference outputs to {output_dir}")


if __name__ == "__main__":
    main()
