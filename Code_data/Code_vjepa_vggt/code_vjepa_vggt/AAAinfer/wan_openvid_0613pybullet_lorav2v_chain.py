from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root, resolve_runtime_root
from code_vjepa_vggt.infer_v_newtrain_chain_rollout import (
    _read_video_frames,
    _read_video_tail,
    _save_json,
    _save_rgb_frames_video,
)
from code_vjepa_vggt.train0419_reference import batch_eval_lora as core
from diffsynth.utils.data import save_video

"""
Run command example:

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
CUDA_VISIBLE_DEVICES=3 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_openvid_0613pybullet_lorav2v_chain.py \
  --weights-root /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name wan_openvid_0613pybullet_lorav2v_step000500_chain

Default output roots:
- /data/gaoya/AAA_test_video/0623/test/v2v/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_chain
- /data/gaoya/AAA_test_video/0623/test/v2v/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_chain_runtime
"""


def _resolve_lora_path(weights_root: Path) -> Path:
    checkpoint_path = weights_root.expanduser().resolve() / "checkpoint.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found under weights-root: {checkpoint_path}")
    return checkpoint_path


def _pil_frames_to_rgb_uint8(frames: list[Image.Image]) -> np.ndarray:
    if not frames:
        raise ValueError("expected at least one PIL frame")
    return np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames], axis=0)


def _rgb_uint8_to_pil_frames(frames_rgb: np.ndarray) -> list[Image.Image]:
    if frames_rgb.ndim != 4 or int(frames_rgb.shape[-1]) != 3:
        raise ValueError(f"expected frames_rgb shape [T,H,W,3], got {list(frames_rgb.shape)}")
    return [Image.fromarray(frame, mode="RGB") for frame in frames_rgb]


def _infer_segment(
    *,
    pipe,
    prompt: str,
    negative_prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    output_num_frames: int,
    num_inference_steps: int,
    cfg_scale: float,
    context_frames: list[Image.Image],
    conditioning_mode: str,
):
    if not context_frames:
        raise ValueError("context_frames must be non-empty")
    core.seed_everything(seed)
    generation_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "seed": int(seed),
        "cfg_scale": float(cfg_scale),
        "num_inference_steps": int(num_inference_steps),
        "tiled": True,
    }
    if conditioning_mode == "input_image_only":
        generation_kwargs["input_image"] = context_frames[0]
    else:
        generation_kwargs["input_image"] = context_frames[0]
        generation_kwargs["context_video"] = context_frames

    with torch.no_grad():
        video = pipe(**generation_kwargs)
    keep = min(int(output_num_frames), len(video))
    if len(video) < keep:
        raise ValueError(f"Generated only {len(video)} frames, need at least {keep}.")
    return video[:keep]


def _run_chain_case(
    *,
    args: argparse.Namespace,
    pipe,
    row: dict[str, object],
    output_dir: Path,
    method_name: str,
) -> dict[str, object]:
    context_path = Path(str(row["context_path"])).expanduser().resolve()
    core.assert_exists(context_path, "Context video")
    sample_stem = str(row["sample_id"])
    prompt_text = str(row["caption"])

    initial_context_pil = core.load_context_frames(
        context_path=context_path,
        context_frames=int(args.context_frames),
        height=int(args.height),
        width=int(args.width),
        resize_mode=str(row.get("context_resize_mode", "crop")),
    )
    initial_context_rgb = _pil_frames_to_rgb_uint8(initial_context_pil)

    segment1_path = output_dir / f"{sample_stem}__segment1.mp4"
    segment2_path = output_dir / f"{sample_stem}__segment2.mp4"
    merged_path = output_dir / f"{sample_stem}__merged.mp4"
    segment1_context_path = output_dir / f"{sample_stem}__segment1_context.mp4"
    segment2_context_path = output_dir / f"{sample_stem}__segment2_context.mp4"
    result_path = output_dir / f"{sample_stem}.json"

    _save_rgb_frames_video(
        frames_rgb=initial_context_rgb,
        output_path=segment1_context_path,
        fps=int(args.fps),
        quality=int(args.quality),
    )

    segment1_video = _infer_segment(
        pipe=pipe,
        prompt=prompt_text,
        negative_prompt=str(args.negative_prompt),
        seed=int(args.seed),
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        output_num_frames=int(args.requested_output_frames),
        num_inference_steps=int(args.num_inference_steps),
        cfg_scale=float(args.cfg_scale),
        context_frames=initial_context_pil,
        conditioning_mode=str(args.conditioning_mode),
    )
    save_video(segment1_video, str(segment1_path), fps=int(args.fps), quality=int(args.quality))

    chained_context_rgb, chained_frame_indices = _read_video_tail(segment1_path, int(args.context_frames))
    _save_rgb_frames_video(
        frames_rgb=chained_context_rgb,
        output_path=segment2_context_path,
        fps=int(args.fps),
        quality=int(args.quality),
    )
    chained_context_pil = _rgb_uint8_to_pil_frames(chained_context_rgb)

    segment2_video = _infer_segment(
        pipe=pipe,
        prompt=prompt_text,
        negative_prompt=str(args.negative_prompt),
        seed=int(args.seed) + 1,
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        output_num_frames=int(args.requested_output_frames),
        num_inference_steps=int(args.num_inference_steps),
        cfg_scale=float(args.cfg_scale),
        context_frames=chained_context_pil,
        conditioning_mode=str(args.conditioning_mode),
    )
    save_video(segment2_video, str(segment2_path), fps=int(args.fps), quality=int(args.quality))

    segment1_saved_frames = _read_video_frames(segment1_path)
    segment2_saved_frames = _read_video_frames(segment2_path)
    merged_frames = np.concatenate([segment1_saved_frames, segment2_saved_frames], axis=0)
    save_video(merged_frames, str(merged_path), fps=int(args.fps), quality=int(args.quality))

    source_paths = row.get("source_paths", {}) if isinstance(row.get("source_paths"), dict) else {}
    result: dict[str, object] = {
        "input_json": str(row.get("meta_path")),
        "input_caption": prompt_text,
        "output_video": str(merged_path),
        "method": method_name,
        "seed": int(args.seed),
        "step": int(args.num_inference_steps),
        "guidance": float(args.cfg_scale),
        "ckpt": str(args.lora_path),
        "weights_root": str(args.weights_root),
        "context_video": str(context_path),
        "segment1_context_video": str(segment1_context_path),
        "segment1_video": str(segment1_path),
        "segment2_context_video": str(segment2_context_path),
        "segment2_video": str(segment2_path),
        "merged_video": str(merged_path),
        "segment1_seed": int(args.seed),
        "segment2_seed": int(args.seed) + 1,
        "fps": int(args.fps),
        "num_frames": int(args.requested_output_frames),
        "context_frames": int(args.context_frames),
        "conditioning_mode": str(args.conditioning_mode),
        "merged_frames": int(merged_frames.shape[0]),
        "segment1_tail_indices_for_segment2": chained_frame_indices.tolist(),
        "status": "generated",
    }
    source_video = source_paths.get("full_video_path")
    if isinstance(source_video, str) and source_video:
        result["source_video"] = source_video
    input_image = source_paths.get("first_frame_path")
    if isinstance(input_image, str) and input_image:
        result["input_image"] = input_image

    _save_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run chained Wan LoRA inference over an input json list. "
            "This reuses the generated tail-8 from segment1 as the context for segment2 "
            "and saves segment videos, context videos, merged output, and per-case json."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True, help="step-* dir containing checkpoint.safetensors")
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--wan-root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=core.DEFAULT_SINGLE_CASE_HEIGHT)
    parser.add_argument("--width", type=int, default=core.DEFAULT_SINGLE_CASE_WIDTH)
    parser.add_argument("--num-frames", type=int, default=core.DEFAULT_SINGLE_CASE_NUM_FRAMES)
    parser.add_argument("--context-frames", type=int, default=core.DEFAULT_SINGLE_CASE_CONTEXT_FRAMES)
    parser.add_argument("--fps", type=int, default=core.DEFAULT_SINGLE_CASE_FPS)
    parser.add_argument("--num-inference-steps", type=int, default=core.DEFAULT_SINGLE_CASE_NUM_INFERENCE_STEPS)
    parser.add_argument("--cfg-scale", type=float, default=core.DEFAULT_SINGLE_CASE_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=core.DEFAULT_SINGLE_CASE_SEED)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--negative-prompt", type=str, default=core.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--conditioning-mode", choices=["context_aware", "input_image_only"], default="context_aware")
    parser.add_argument("--context-resize-mode", choices=["auto", "crop", "pad"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    model_name = str(cli_args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root="/data/gaoya/AAA_test_video/0623/test/v2v/loramodel",
        model_name=model_name,
    )
    runtime_root = resolve_runtime_root(
        explicit_runtime_root=cli_args.runtime_root,
        base_runtime_root="/data/gaoya/AAA_test_video/0623/test/v2v/loramodel",
        model_name=model_name,
    )

    args = argparse.Namespace(
        wan_root=cli_args.wan_root.expanduser().resolve(),
        output_root=output_root,
        runtime_root=runtime_root,
        weights_root=cli_args.weights_root.expanduser().resolve(),
        lora_path=_resolve_lora_path(cli_args.weights_root),
        input_json_list_path=cli_args.input_json_list_path.expanduser().resolve(),
        meta_list_path=None,
        meta_json_path=None,
        context_path=None,
        output_video_path=None,
        prompt=None,
        sample_id=core.DEFAULT_SINGLE_CASE_SAMPLE_ID,
        dataset_name=core.DEFAULT_SINGLE_CASE_DATASET_NAME,
        future_gt_path=None,
        full_video_path=None,
        first_frame_path=None,
        context_resize_mode=cli_args.context_resize_mode,
        conditioning_mode=cli_args.conditioning_mode,
        device=cli_args.device,
        height=int(cli_args.height),
        width=int(cli_args.width),
        fps=int(cli_args.fps),
        num_frames=int(cli_args.num_frames),
        context_frames=int(cli_args.context_frames),
        num_inference_steps=int(cli_args.num_inference_steps),
        cfg_scale=float(cli_args.cfg_scale),
        seed=int(cli_args.seed),
        quality=int(cli_args.quality),
        model_name=model_name,
        negative_prompt=str(cli_args.negative_prompt),
        overwrite=bool(cli_args.overwrite),
        limit=cli_args.limit,
        no_metadata=False,
        multi_gpu=False,
        num_shards=1,
        shard_id=0,
        worker=False,
    )

    core.assert_exists(args.wan_root, "Wan root")
    core.assert_exists(args.weights_root, "Weights root")
    core.assert_exists(args.lora_path, "LoRA checkpoint")
    core.assert_exists(args.input_json_list_path, "Input json list path")
    core.validate_args(args)

    aligned_height, aligned_width = core.align_generation_size(args.height, args.width)
    if (aligned_height, aligned_width) != (args.height, args.width):
        print(
            "[size_align] Adjusting generation size from "
            f"{args.height}x{args.width} to {aligned_height}x{aligned_width}."
        )
        args.height = aligned_height
        args.width = aligned_width

    args.requested_output_frames = int(args.num_frames)
    aligned_num_frames = core.align_generation_num_frames(args.num_frames)
    if aligned_num_frames != args.num_frames:
        print(
            "[time_align] Adjusting generation length from "
            f"{args.num_frames} to {aligned_num_frames} to satisfy 4n+1, "
            f"while saving only the first {args.requested_output_frames} frames."
        )
        args.num_frames = aligned_num_frames

    output_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    cases = core.collect_cases_from_input_json_list(
        core.load_input_json_paths(args.input_json_list_path),
        args.limit,
    )
    manifest = {
        "input_json_list_path": str(args.input_json_list_path),
        "weights_root": str(args.weights_root),
        "lora_path": str(args.lora_path),
        "model_name": args.model_name,
        "num_items": len(cases),
        "num_inference_steps": int(args.num_inference_steps),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.requested_output_frames),
        "context_frames": int(args.context_frames),
        "conditioning_mode": str(args.conditioning_mode),
        "chain_rollout": True,
    }
    core.write_json(output_root / "batch_manifest.json", manifest)

    method_name = core.build_method_name(args.lora_path)
    pipe = core.build_pipeline(args.wan_root, args.device, args.lora_path)

    entries: list[dict[str, object]] = []
    for row in cases:
        sample_stem = str(row["sample_id"])
        merged_video = output_root / f"{sample_stem}__merged.mp4"
        output_json = output_root / f"{sample_stem}.json"
        if merged_video.exists() and output_json.exists() and not args.overwrite:
            print(f"[skip] {sample_stem}")
            entries.append(
                {
                    "dataset": row["dataset"],
                    "sample_id": sample_stem,
                    "status": "skipped_existing",
                    "output_path": str(merged_video),
                }
            )
            continue

        try:
            _run_chain_case(
                args=args,
                pipe=pipe,
                row=row,
                output_dir=output_root,
                method_name=method_name,
            )
            print(f"[done] {sample_stem}")
            entries.append(
                {
                    "dataset": row["dataset"],
                    "sample_id": sample_stem,
                    "status": "generated",
                    "output_path": str(merged_video),
                }
            )
        except Exception as exc:
            failed_payload = {
                "input_json": str(row.get("meta_path")),
                "input_caption": str(row.get("caption")),
                "output_video": str(merged_video),
                "method": method_name,
                "seed": int(args.seed),
                "step": int(args.num_inference_steps),
                "guidance": float(args.cfg_scale),
                "ckpt": str(args.lora_path),
                "weights_root": str(args.weights_root),
                "status": "failed",
                "error": repr(exc),
            }
            _save_json(output_json, failed_payload)
            print(f"[error] {sample_stem}: {exc}")
            entries.append(
                {
                    "dataset": row["dataset"],
                    "sample_id": sample_stem,
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    metadata_dir = runtime_root / "metadata" / args.model_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    per_case_path = metadata_dir / f"{args.model_name}_chain_per_case.jsonl"
    core.write_jsonl(per_case_path, entries)

    summary_payload = {
        "model_name": args.model_name,
        "weights_root": str(args.weights_root),
        "lora_path": str(args.lora_path),
        "generated_dir": str(output_root),
        "runtime_root": str(runtime_root),
        "metadata_dir": str(metadata_dir),
        "input_json_list_path": str(args.input_json_list_path),
        "summary": core.build_summary(entries),
        "selected_videos": core.find_selected_video_paths(output_root, entries),
    }
    core.write_json(runtime_root / "summary.json", summary_payload)
    print(runtime_root / "summary.json")


if __name__ == "__main__":
    main()
