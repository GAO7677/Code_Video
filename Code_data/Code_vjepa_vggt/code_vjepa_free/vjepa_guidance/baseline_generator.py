#!/usr/bin/env python3
"""
Baseline video generator for probe_sweep comparison.
Generates video using the same LoRA model but without any V-JEPA guidance.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from code_vjepa_vggt.train0419_reference import batch_eval_lora as core

log = logging.getLogger(__name__)


def _resolve_lora_path(weights_root: Path) -> Path:
    checkpoint_path = weights_root.expanduser().resolve() / "checkpoint.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline video without V-JEPA guidance")
    parser.add_argument("--weights-root", type=Path, required=True,
                       help="step-* dir containing checkpoint.safetensors")
    parser.add_argument("--input-json", type=Path, required=True,
                       help="Input JSON file for the case")
    parser.add_argument("--context-path", type=Path, required=True,
                       help="Context video path (first 8 frames)")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("/data/gaoya/agent-data/outputs/probe_sweep/baseline"),
                       help="Output directory for baseline results")
    parser.add_argument("--wan-root", type=Path,
                       default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Validate paths
    if not args.weights_root.exists():
        raise FileNotFoundError(f"Weights root not found: {args.weights_root}")
    if not args.input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")
    if not args.context_path.exists():
        raise FileNotFoundError(f"Context path not found: {args.context_path}")

    lora_path = _resolve_lora_path(args.weights_root)

    # Load case metadata
    case = json.loads(args.input_json.read_text(encoding="utf-8"))
    prompt = case.get("prompt", "A physics simulation video")
    sample_id = case.get("sample_id", "baseline_sample")

    # Setup output directory
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{sample_id}_baseline.mp4"

    log.info("Generating baseline video for: %s", sample_id)
    log.info("Output: %s", output_video_path)
    log.info("Using LoRA: %s", lora_path)
    log.info("Context: %s", args.context_path)
    log.info("Prompt: %s", prompt)

    # Build args for core.run_single_case
    core_args = argparse.Namespace(
        wan_root=args.wan_root.expanduser().resolve(),
        weights_root=args.weights_root.expanduser().resolve(),
        lora_path=lora_path,
        context_path=args.context_path.expanduser().resolve(),
        output_video_path=output_video_path,
        prompt=prompt,
        sample_id=sample_id,
        dataset_name="baseline",
        future_gt_path=None,
        full_video_path=None,
        first_frame_path=None,
        context_resize_mode="auto",
        conditioning_mode="context_aware",
        device=args.device,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_frames=args.num_frames,
        context_frames=args.context_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        quality=5,
        negative_prompt=args.negative_prompt,
        overwrite=True,
        no_metadata=False,
        output_root=output_dir,
        runtime_root=output_dir / "runtime",
        model_name="baseline_no_vjepa",
    )

    # Align sizes
    aligned_height, aligned_width = core.align_generation_size(core_args.height, core_args.width)
    if (aligned_height, aligned_width) != (core_args.height, core_args.width):
        log.info("Adjusting size from %dx%d to %dx%d",
                core_args.height, core_args.width, aligned_height, aligned_width)
        core_args.height = aligned_height
        core_args.width = aligned_width

    aligned_num_frames = core.align_generation_num_frames(core_args.num_frames)
    if aligned_num_frames != core_args.num_frames:
        log.info("Adjusting frames from %d to %d", core_args.num_frames, aligned_num_frames)
        core_args.requested_output_frames = core_args.num_frames
        core_args.num_frames = aligned_num_frames

    # Build pipeline
    log.info("Building pipeline...")
    pipe = core.build_pipeline(core_args.wan_root, core_args.device, lora_path)

    # Generate video
    log.info("Starting baseline generation (no V-JEPA guidance)...")
    try:
        output_num_frames = getattr(core_args, 'requested_output_frames', core_args.num_frames)
        video, used_context_frames = core.generate_one_video(
            pipe=pipe,
            context_path=args.context_path,
            first_frame_path=None,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=core_args.height,
            width=core_args.width,
            num_frames=core_args.num_frames,
            fps=args.fps,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            context_frames=args.context_frames,
            output_num_frames=output_num_frames,
            context_resize_mode="crop",
            conditioning_mode="context_aware",
        )

        # Save video
        from diffsynth.utils.data import save_video
        save_video(video, str(output_video_path), fps=args.fps, quality=5)
        log.info("✓ Baseline video saved: %s", output_video_path)

        # Save metadata
        meta_path = output_dir / f"{sample_id}_baseline.json"
        metadata = {
            "sample_id": sample_id,
            "prompt": prompt,
            "video_path": str(output_video_path),
            "context_path": str(args.context_path),
            "lora_path": str(lora_path),
            "generation_params": {
                "height": core_args.height,
                "width": core_args.width,
                "num_frames": core_args.num_frames,
                "num_inference_steps": core_args.num_inference_steps,
                "cfg_scale": core_args.cfg_scale,
                "seed": core_args.seed,
                "used_context_frames": used_context_frames,
            },
            "note": "Pure diffusion generation without V-JEPA guidance"
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        log.info("✓ Metadata saved: %s", meta_path)

    except Exception as e:
        log.error("✗ Failed to generate baseline video: %s", e)
        raise


if __name__ == "__main__":
    main()
