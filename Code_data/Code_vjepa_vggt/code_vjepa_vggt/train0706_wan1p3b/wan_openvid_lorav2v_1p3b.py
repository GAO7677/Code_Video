from __future__ import annotations

import argparse
from pathlib import Path

from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root, resolve_runtime_root
from code_vjepa_vggt.train0706_wan1p3b import batch_eval_lora as core

"""

通用数据集openvid LoRA模型评估脚本


Run command example:

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/wan_openvid_lorav2v_1p3b.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/openvid_full_ctx81_384x672_lora/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name openvid_full_ctx81_384x672_lora_step001000 \
  --num-inference-steps 40

Default output roots:
- /data/gaoya/AAA_test_video/0623/test/v2v_1p3b/loramodel/<model-name>
- /data/gaoya/AAA_test_video/0623/test/v2v_1p3b/loramodel/<model-name>_runtime
"""


def _resolve_lora_path(weights_root: Path) -> Path:
    checkpoint_path = weights_root.expanduser().resolve() / "checkpoint.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found under weights-root: {checkpoint_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run the fixed OpenVid Wan LoRA checkpoint over an input json list. "
            "This is a thin wrapper around train0419_reference/batch_eval_lora.py."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True, help="step-* dir containing checkpoint.safetensors")
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--wan-root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B"))
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
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def build_core_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    model_name = str(cli_args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root="/data/gaoya/AAA_test_video/0623/test/v2v_1p3b/loramodel",
        model_name=model_name,
    )
    runtime_root = resolve_runtime_root(
        explicit_runtime_root=cli_args.runtime_root,
        base_runtime_root="/data/gaoya/AAA_test_video/0623/test/v2v_1p3b/loramodel",
        model_name=model_name,
    )
    return argparse.Namespace(
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
        multi_gpu=bool(cli_args.multi_gpu),
        num_shards=int(cli_args.num_shards),
        shard_id=int(cli_args.shard_id),
        worker=bool(cli_args.worker),
    )


def main() -> None:
    cli_args = parse_args()
    args = build_core_args(cli_args)

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

    generated_dir = args.output_root
    metadata_dir = args.runtime_root / "metadata" / args.model_name
    summary_json_path = args.runtime_root / "summary.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.runtime_root.mkdir(parents=True, exist_ok=True)

    effective_num_shards = args.num_shards
    if args.multi_gpu and not args.worker:
        effective_num_shards = core.launch_multi_gpu_workers(args, generated_dir, metadata_dir)
    else:
        core.run_generation(args, generated_dir, metadata_dir)

    merged_jsonl = core.merge_shard_jsonl_files(metadata_dir, args.model_name, effective_num_shards)
    summary_entries_path = merged_jsonl
    if summary_entries_path is None:
        summary_entries_path = core.per_case_jsonl_path(
            metadata_dir,
            args.model_name,
            effective_num_shards,
            args.shard_id,
        )
    summary_entries = core.load_jsonl(summary_entries_path)
    eval_csv_path = core.infer_eval_csv_path(args.output_root, args.model_name)
    summary_entries, num_entries_with_metrics = core.augment_entries_with_eval_metrics(
        summary_entries,
        eval_csv_path,
    )
    if summary_entries_path is not None and summary_entries:
        core.write_jsonl(summary_entries_path, summary_entries)
    payload = {
        "model_name": args.model_name,
        "weights_root": str(args.weights_root),
        "lora_path": str(args.lora_path),
        "generated_dir": str(generated_dir),
        "metadata_dir": str(metadata_dir),
        "runtime_root": str(args.runtime_root),
        "input_json_list_path": str(args.input_json_list_path),
        "eval_csv": str(eval_csv_path) if eval_csv_path is not None else None,
        "num_entries_with_metrics": num_entries_with_metrics,
        "summary": core.build_summary(summary_entries),
        "selected_videos": core.find_selected_video_paths(generated_dir, summary_entries),
    }
    core.write_json(summary_json_path, payload)
    print(summary_json_path)


if __name__ == "__main__":
    main()
