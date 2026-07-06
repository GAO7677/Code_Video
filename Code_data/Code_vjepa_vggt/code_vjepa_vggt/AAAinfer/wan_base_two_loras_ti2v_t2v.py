from __future__ import annotations

# Run command examples:
#
# TI2V + Wan base:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
# CUDA_VISIBLE_DEVICES=2 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_base_two_loras_ti2v_t2v.py \
#   --mode ti2v \
#   --model-preset wan_base \
#   --context-path /path/to/context_video_8f.mp4 \
#   --prompt "A ball drops onto a block on a table. Static shot." \
#   --output-video-path /data/gaoya/agent-data/outputs/wan_base_ti2v_demo.mp4
#
# TI2V + OpenVid LoRA:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
# CUDA_VISIBLE_DEVICES=2 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_base_two_loras_ti2v_t2v.py \
#   --mode ti2v \
#   --model-preset openvid_lora_step10000 \
#   --context-path /path/to/context_video_8f.mp4 \
#   --prompt "A ball drops onto a block on a table. Static shot." \
#   --output-video-path /data/gaoya/agent-data/outputs/openvid_lora_ti2v_demo.mp4
#
# T2V + 0613pybullet LoRA:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
# CUDA_VISIBLE_DEVICES=2 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_base_two_loras_ti2v_t2v.py \
#   --mode t2v \
#   --model-preset openvid_0613pybullet_lora_step000500 \
#   --prompt "A ball drops onto a block on a table. Static shot." \
#   --output-video-path /data/gaoya/agent-data/outputs/0613pybullet_lora_t2v_demo.mp4

import argparse
import json
from pathlib import Path

import torch

from diffsynth import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video

from code_vjepa_vggt.train0419_reference import batch_eval_lora as ti2v_core


DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_OPENVID_LORA = Path(
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/"
    "openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors"
)
DEFAULT_0613PYBULLET_LORA = Path(
    "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
    "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
)
DEFAULT_NEGATIVE_PROMPT = ti2v_core.DEFAULT_NEGATIVE_PROMPT
MODEL_PRESET_TO_LORA = {
    "wan_base": None,
    "openvid_lora_step10000": DEFAULT_OPENVID_LORA,
    "openvid_0613pybullet_lora_step000500": DEFAULT_0613PYBULLET_LORA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified single-case inference entry for Wan2.2 TI2V-5B base model and two LoRA presets, "
            "supporting both TI2V and T2V. TI2V is forced to first-frame image conditioning."
        )
    )
    parser.add_argument("--mode", choices=["ti2v", "t2v"], required=True)
    parser.add_argument(
        "--model-preset",
        choices=list(MODEL_PRESET_TO_LORA.keys()),
        required=True,
        help="wan_base uses no LoRA; the other two presets load fixed LoRA checkpoints.",
    )
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--output-video-path", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--context-path", type=Path, default=None, help="Required when --mode=ti2v.")
    parser.add_argument("--first-frame-path", type=Path, default=None, help="Optional first frame override for TI2V.")
    parser.add_argument(
        "--conditioning-mode",
        choices=["context_aware", "input_image_only"],
        default="input_image_only",
        help="For --mode=ti2v this script enforces input_image_only (first-frame conditioning).",
    )
    parser.add_argument("--context-resize-mode", choices=["auto", "crop", "pad"], default="auto")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=ti2v_core.DEFAULT_SINGLE_CASE_HEIGHT)
    parser.add_argument("--width", type=int, default=ti2v_core.DEFAULT_SINGLE_CASE_WIDTH)
    parser.add_argument("--num-frames", type=int, default=ti2v_core.DEFAULT_SINGLE_CASE_NUM_FRAMES)
    parser.add_argument("--context-frames", type=int, default=ti2v_core.DEFAULT_SINGLE_CASE_CONTEXT_FRAMES)
    parser.add_argument("--fps", type=int, default=ti2v_core.DEFAULT_SINGLE_CASE_FPS)
    parser.add_argument("--num-inference-steps", type=int, default=ti2v_core.DEFAULT_SINGLE_CASE_NUM_INFERENCE_STEPS)
    parser.add_argument("--cfg-scale", type=float, default=ti2v_core.DEFAULT_SINGLE_CASE_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=ti2v_core.DEFAULT_SINGLE_CASE_SEED)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def resolve_lora_path(model_preset: str) -> Path | None:
    value = MODEL_PRESET_TO_LORA[str(model_preset)]
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def build_t2v_pipeline(wan_root: Path, device: str, lora_path: Path | None) -> WanVideoPipeline:
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=ti2v_core.build_model_configs(wan_root),
        tokenizer_config=ModelConfig(path=str(ti2v_core.find_tokenizer_path(wan_root))),
    )
    if lora_path is not None:
        pipe.load_lora(pipe.dit, str(lora_path), alpha=1.0)
    return pipe


def build_ti2v_pipeline(wan_root: Path, device: str, lora_path: Path | None):
    return ti2v_core.build_pipeline(wan_root=wan_root, device=device, lora_path=lora_path)


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "ti2v" and args.context_path is None:
        raise ValueError("--context-path is required when --mode=ti2v.")
    if args.mode == "t2v" and args.context_path is not None:
        raise ValueError("--context-path is only valid when --mode=ti2v.")
    if args.mode == "ti2v" and args.conditioning_mode != "input_image_only":
        raise ValueError("--mode=ti2v is fixed to first-frame conditioning; use --conditioning-mode input_image_only.")
    if args.mode == "t2v" and args.conditioning_mode != "context_aware":
        raise ValueError("--conditioning-mode only affects TI2V and should stay at default for T2V.")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height and width must be divisible by 16, got {(args.height, args.width)}.")
    if args.num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {args.num_frames}.")
    if args.mode == "ti2v" and args.context_frames >= args.num_frames:
        raise ValueError(
            f"context_frames must be smaller than num_frames for TI2V, got {args.context_frames} >= {args.num_frames}."
        )


def write_sidecar_json(
    *,
    output_video_path: Path,
    mode: str,
    model_preset: str,
    wan_root: Path,
    lora_path: Path | None,
    prompt: str,
    negative_prompt: str,
    context_path: Path | None,
    first_frame_path: Path | None,
    conditioning_mode: str,
    height: int,
    width: int,
    num_frames_requested: int,
    num_frames_generated: int,
    context_frames: int,
    fps: int,
    num_inference_steps: int,
    cfg_scale: float,
    seed: int,
) -> None:
    payload = {
        "video_path": str(output_video_path),
        "mode": str(mode),
        "model_preset": str(model_preset),
        "wan_root": str(wan_root),
        "lora_path": str(lora_path) if lora_path is not None else None,
        "prompt": str(prompt),
        "negative_prompt": str(negative_prompt),
        "context_path": str(context_path) if context_path is not None else None,
        "first_frame_path": str(first_frame_path) if first_frame_path is not None else None,
        "conditioning_mode": str(conditioning_mode),
        "height": int(height),
        "width": int(width),
        "num_frames_requested": int(num_frames_requested),
        "num_frames_generated": int(num_frames_generated),
        "context_frames": int(context_frames),
        "fps": int(fps),
        "num_inference_steps": int(num_inference_steps),
        "cfg_scale": float(cfg_scale),
        "seed": int(seed),
    }
    output_video_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_t2v(args: argparse.Namespace, wan_root: Path, lora_path: Path | None, output_video_path: Path) -> None:
    pipe = build_t2v_pipeline(wan_root=wan_root, device=str(args.device), lora_path=lora_path)
    with torch.no_grad():
        video = pipe(
            prompt=str(args.prompt),
            negative_prompt=str(args.negative_prompt),
            seed=int(args.seed),
            tiled=True,
            height=int(args.height),
            width=int(args.width),
            num_frames=int(args.num_frames),
            num_inference_steps=int(args.num_inference_steps),
            cfg_scale=float(args.cfg_scale),
        )
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_video_path), fps=int(args.fps), quality=int(args.quality))
    write_sidecar_json(
        output_video_path=output_video_path,
        mode="t2v",
        model_preset=str(args.model_preset),
        wan_root=wan_root,
        lora_path=lora_path,
        prompt=str(args.prompt),
        negative_prompt=str(args.negative_prompt),
        context_path=None,
        first_frame_path=None,
        conditioning_mode="context_aware",
        height=int(args.height),
        width=int(args.width),
        num_frames_requested=int(args.num_frames),
        num_frames_generated=len(video),
        context_frames=0,
        fps=int(args.fps),
        num_inference_steps=int(args.num_inference_steps),
        cfg_scale=float(args.cfg_scale),
        seed=int(args.seed),
    )


def run_ti2v(args: argparse.Namespace, wan_root: Path, lora_path: Path | None, output_video_path: Path) -> None:
    pipe = build_ti2v_pipeline(wan_root=wan_root, device=str(args.device), lora_path=lora_path)
    requested_frames = int(args.num_frames)
    aligned_frames = ti2v_core.align_generation_num_frames(requested_frames)
    resize_mode = str(args.context_resize_mode)
    if resize_mode == "auto":
        resize_mode = ti2v_core.resolve_context_resize_mode("single_case")

    first_frame_path = None
    if args.first_frame_path is not None:
        first_frame_path = args.first_frame_path.expanduser().resolve()

    video, used_context_frames = ti2v_core.generate_one_video(
        pipe=pipe,
        context_path=args.context_path.expanduser().resolve(),
        first_frame_path=first_frame_path,
        prompt=str(args.prompt),
        negative_prompt=str(args.negative_prompt),
        seed=int(args.seed),
        height=int(args.height),
        width=int(args.width),
        num_frames=aligned_frames,
        fps=int(args.fps),
        cfg_scale=float(args.cfg_scale),
        num_inference_steps=int(args.num_inference_steps),
        context_frames=int(args.context_frames),
        output_num_frames=requested_frames,
        context_resize_mode=resize_mode,
        conditioning_mode="input_image_only",
    )
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_video_path), fps=int(args.fps), quality=int(args.quality))
    write_sidecar_json(
        output_video_path=output_video_path,
        mode="ti2v",
        model_preset=str(args.model_preset),
        wan_root=wan_root,
        lora_path=lora_path,
        prompt=str(args.prompt),
        negative_prompt=str(args.negative_prompt),
        context_path=args.context_path.expanduser().resolve(),
        first_frame_path=first_frame_path,
        conditioning_mode="input_image_only",
        height=int(args.height),
        width=int(args.width),
        num_frames_requested=requested_frames,
        num_frames_generated=len(video),
        context_frames=int(used_context_frames),
        fps=int(args.fps),
        num_inference_steps=int(args.num_inference_steps),
        cfg_scale=float(args.cfg_scale),
        seed=int(args.seed),
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    wan_root = args.wan_root.expanduser().resolve()
    lora_path = resolve_lora_path(str(args.model_preset))
    output_video_path = args.output_video_path.expanduser().resolve()

    assert_exists(wan_root, "Wan root")
    if lora_path is not None:
        assert_exists(lora_path, "LoRA checkpoint")
    if args.context_path is not None:
        args.context_path = args.context_path.expanduser().resolve()
        assert_exists(args.context_path, "Context path")
    if args.first_frame_path is not None:
        args.first_frame_path = args.first_frame_path.expanduser().resolve()
        assert_exists(args.first_frame_path, "First frame path")
    if output_video_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_video_path}")

    aligned_height, aligned_width = ti2v_core.align_generation_size(int(args.height), int(args.width))
    if (aligned_height, aligned_width) != (int(args.height), int(args.width)):
        print(
            "[size_align] Adjusting generation size from "
            f"{args.height}x{args.width} to {aligned_height}x{aligned_width}."
        )
        args.height = aligned_height
        args.width = aligned_width

    print(f"[run] mode={args.mode}")
    print(f"[run] model_preset={args.model_preset}")
    print(f"[run] wan_root={wan_root}")
    print(f"[run] lora_path={lora_path}")
    print(f"[run] output_video_path={output_video_path}")
    print(f"[run] device={args.device}")
    print(f"[run] size={args.width}x{args.height} frames={args.num_frames} fps={args.fps}")

    if args.mode == "t2v":
        run_t2v(args=args, wan_root=wan_root, lora_path=lora_path, output_video_path=output_video_path)
    else:
        run_ti2v(args=args, wan_root=wan_root, lora_path=lora_path, output_video_path=output_video_path)

    print(f"[done] saved_video={output_video_path}")
    print(f"[done] saved_json={output_video_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
