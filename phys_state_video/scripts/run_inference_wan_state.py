from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.dataset import NpzPredictorDataset, collate_predictor_episodes
from phys_state_video.predictor_wan_state import WanStateLatentPredictor, WanStateLatentPredictorConfig
from phys_state_video.utils import detach_to_cpu_numpy, require_torch
from phys_state_video.wan_bridge import WanImageToVideoBackend, WanLatentExtractor

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan prefix-continuation inference with state-latent conditioning.")
    parser.add_argument("--episode", required=True, help="Episode .npz file.")
    parser.add_argument("--predictor", required=True, help="Predictor checkpoint path.")
    parser.add_argument("--wan-ckpt-dir", required=True, help="Wan checkpoint directory.")
    parser.add_argument(
        "--wan-state-adapter-ckpt",
        default=None,
        help="Checkpoint for the Wan state adapter branch. Strongly recommended when using predictor state tokens.",
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--wan-task", default="i2v-A14B")
    parser.add_argument("--wan-size", default="480*832")
    parser.add_argument(
        "--frame-num",
        type=int,
        default=None,
        help="Total output frame count K+T. Defaults to context_steps + future_steps.",
    )
    parser.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--state-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=int, default=6)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def write_mp4(path: Path, frames_tchw: np.ndarray, fps: int) -> None:
    import cv2

    frames = np.clip(frames_tchw, 0.0, 1.0)
    frames = (frames.transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = NpzPredictorDataset(args.episode)
    batch = collate_predictor_episodes([dataset[0]])

    predictor_ckpt = load_checkpoint(args.predictor, map_location=args.device)
    predictor = WanStateLatentPredictor(WanStateLatentPredictorConfig(**predictor_ckpt["config"])).to(args.device)
    predictor.load_state_dict(predictor_ckpt["model"])
    predictor.eval()

    latent_extractor = WanLatentExtractor(
        ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        task=args.wan_task,
        device=args.device,
    )
    if args.wan_state_adapter_ckpt is None:
        warnings.warn(
            "running Wan inference without --wan-state-adapter-ckpt: state tokens will be wired in, "
            "but the Wan state adapter branch may stay near ineffective default initialization.",
            stacklevel=2,
        )
    backend = WanImageToVideoBackend(
        ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        task=args.wan_task,
        device=args.device,
        state_adapter_ckpt=args.wan_state_adapter_ckpt,
    )

    context_frames = batch["context_frames"].to(args.device)
    context_steps = context_frames.shape[1]
    future_steps = batch["future_states"].shape[1]
    frame_num = args.frame_num or (context_steps + future_steps)

    with torch.no_grad():
        context_latents = latent_extractor.encode_context_frames(context_frames)
        outputs = predictor(
            context_latents,
            batch["camera"].to(args.device),
            prompt_token_ids=batch["prompt_token_ids"].to(args.device),
            prompt_token_mask=batch["prompt_token_mask"].to(args.device),
            future_steps=future_steps,
            num_objects=batch["context_states"].shape[2],
        )
        state_tokens = outputs["future_state_latents"]
        state_predictions = outputs["future_state_predictions"]
        generated_video = backend.generate(
            prompt=batch["prompts"][0],
            context_frames=context_frames[0],
            state_tokens=state_tokens[0],
            size=args.wan_size,
            frame_num=frame_num,
            sample_solver=args.sample_solver,
            sampling_steps=args.sampling_steps,
            guide_scale=args.guide_scale,
            shift=args.shift,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            state_scale=args.state_scale,
        )

    if generated_video.ndim != 4:
        raise ValueError(f"Wan backend returned unexpected shape {tuple(generated_video.shape)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_video = ((generated_video + 1.0) * 0.5).clamp(0.0, 1.0)
    generated_future = full_video[:, context_steps:context_steps + future_steps]
    context_np = detach_to_cpu_numpy(batch["context_frames"][0])
    future_gt_np = detach_to_cpu_numpy(batch["future_frames"][0])
    predicted_states_np = detach_to_cpu_numpy(state_predictions[0])
    state_tokens_np = detach_to_cpu_numpy(state_tokens[0])
    full_video_np = detach_to_cpu_numpy(full_video)
    generated_future_np = detach_to_cpu_numpy(generated_future)

    np.savez_compressed(
        output_dir / "wan_inference_outputs.npz",
        context_frames=context_np,
        future_gt_frames=future_gt_np,
        predicted_future_states=predicted_states_np,
        future_state_latents=state_tokens_np,
        generated_full_video=full_video_np,
        generated_future_frames=generated_future_np,
    )

    write_mp4(output_dir / "context.mp4", context_np, args.fps)
    write_mp4(output_dir / "gt_future.mp4", future_gt_np, args.fps)
    write_mp4(output_dir / "wan_full.mp4", full_video_np, args.fps)
    write_mp4(output_dir / "wan_future.mp4", generated_future_np, args.fps)

    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "prompt": batch["prompts"][0],
                "wan_task": args.wan_task,
                "wan_size": args.wan_size,
                "frame_num": frame_num,
                "context_steps": context_steps,
                "future_steps": future_steps,
                "sampling_steps": args.sampling_steps,
                "guide_scale": args.guide_scale,
                "shift": args.shift,
                "state_scale": args.state_scale,
                "seed": args.seed,
                "wan_state_adapter_ckpt": args.wan_state_adapter_ckpt,
                "predictor_version": predictor_ckpt.get("predictor_version", "wan_state_v1"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved Wan inference outputs to {output_dir}")


if __name__ == "__main__":
    main()
