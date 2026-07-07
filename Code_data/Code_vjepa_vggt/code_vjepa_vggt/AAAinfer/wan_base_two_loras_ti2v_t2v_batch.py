from __future__ import annotations

# Run command examples:
#
# T2V batch, one model pipe reused for all cases:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
# CUDA_VISIBLE_DEVICES=7 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_base_two_loras_ti2v_t2v_batch.py \
#   --mode t2v \
#   --model-preset wan_base \
#   --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt \
#   --output-root /data/gaoya/AAA_test_video/0623/test/t2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_morpheus_real_world/wan_base
#
# TI2V batch, one model pipe reused for all cases:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
# CUDA_VISIBLE_DEVICES=1 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_base_two_loras_ti2v_t2v_batch.py \
#   --mode ti2v \
#   --model-preset openvid_lora_step10000 \
#   --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
#   --output-root /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_physicIQ/openvid_lora_step10000

import argparse
import json
import traceback
from pathlib import Path

import torch
from diffsynth.utils.data import save_video

from code_vjepa_vggt.AAAinfer import wan_base_two_loras_ti2v_t2v as single_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch entry for Wan2.2 base + two LoRA presets. "
            "Builds the model pipe once per method, then loops over all cases."
        )
    )
    parser.add_argument("--mode", choices=["ti2v", "t2v"], required=True)
    parser.add_argument(
        "--model-preset",
        choices=list(single_case.MODEL_PRESET_TO_LORA.keys()),
        required=True,
    )
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, default=single_case.DEFAULT_WAN_ROOT)
    parser.add_argument("--negative-prompt", type=str, default=single_case.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--context-resize-mode", choices=["auto", "crop", "pad"], default="auto")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=single_case.ti2v_core.DEFAULT_SINGLE_CASE_HEIGHT)
    parser.add_argument("--width", type=int, default=single_case.ti2v_core.DEFAULT_SINGLE_CASE_WIDTH)
    parser.add_argument("--num-frames", type=int, default=single_case.ti2v_core.DEFAULT_SINGLE_CASE_NUM_FRAMES)
    parser.add_argument("--context-frames", type=int, default=single_case.ti2v_core.DEFAULT_SINGLE_CASE_CONTEXT_FRAMES)
    parser.add_argument("--fps", type=int, default=single_case.ti2v_core.DEFAULT_SINGLE_CASE_FPS)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=single_case.ti2v_core.DEFAULT_SINGLE_CASE_NUM_INFERENCE_STEPS,
    )
    parser.add_argument("--cfg-scale", type=float, default=single_case.ti2v_core.DEFAULT_SINGLE_CASE_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=single_case.ti2v_core.DEFAULT_SINGLE_CASE_SEED)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height and width must be divisible by 16, got {(args.height, args.width)}.")
    if args.num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {args.num_frames}.")
    if args.mode == "ti2v" and args.context_frames >= args.num_frames:
        raise ValueError(
            f"context_frames must be smaller than num_frames for TI2V, got {args.context_frames} >= {args.num_frames}."
        )


def first_existing_path(payload: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def iter_json_paths(list_path: Path) -> list[Path]:
    json_paths: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        json_paths.append(Path(line))
    return json_paths


def write_case_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def save_t2v_with_pipe(
    *,
    pipe,
    args: argparse.Namespace,
    wan_root: Path,
    lora_path: Path | None,
    input_json_path: Path,
    prompt: str,
    output_video_path: Path,
) -> None:
    with torch.no_grad():
        video = pipe(
            prompt=str(prompt),
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
    single_case.write_sidecar_json(
        output_video_path=output_video_path,
        case_json_path=input_json_path,
        mode="t2v",
        model_preset=str(args.model_preset),
        method=str(args.model_preset),
        wan_root=wan_root,
        lora_path=lora_path,
        prompt=str(prompt),
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


def save_ti2v_with_pipe(
    *,
    pipe,
    args: argparse.Namespace,
    wan_root: Path,
    lora_path: Path | None,
    input_json_path: Path,
    prompt: str,
    context_path: Path,
    first_frame_path: Path | None,
    output_video_path: Path,
) -> None:
    requested_frames = int(args.num_frames)
    aligned_frames = single_case.ti2v_core.align_generation_num_frames(requested_frames)
    resize_mode = str(args.context_resize_mode)
    if resize_mode == "auto":
        resize_mode = single_case.ti2v_core.resolve_context_resize_mode("single_case")

    video, used_context_frames = single_case.ti2v_core.generate_one_video(
        pipe=pipe,
        context_path=context_path,
        first_frame_path=first_frame_path,
        prompt=str(prompt),
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
    single_case.write_sidecar_json(
        output_video_path=output_video_path,
        case_json_path=input_json_path,
        mode="ti2v",
        model_preset=str(args.model_preset),
        method=str(args.model_preset),
        wan_root=wan_root,
        lora_path=lora_path,
        prompt=str(prompt),
        negative_prompt=str(args.negative_prompt),
        context_path=context_path,
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
    input_json_list_path = args.input_json_list_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    lora_path = single_case.resolve_lora_path(str(args.model_preset))

    single_case.assert_exists(wan_root, "Wan root")
    single_case.assert_exists(input_json_list_path, "Input json list path")
    if lora_path is not None:
        single_case.assert_exists(lora_path, "LoRA checkpoint")

    aligned_height, aligned_width = single_case.ti2v_core.align_generation_size(int(args.height), int(args.width))
    if (aligned_height, aligned_width) != (int(args.height), int(args.width)):
        print(
            "[size_align] Adjusting generation size from "
            f"{args.height}x{args.width} to {aligned_height}x{aligned_width}."
        )
        args.height = aligned_height
        args.width = aligned_width

    json_paths = iter_json_paths(input_json_list_path)
    if not json_paths:
        raise ValueError(f"No json paths found in: {input_json_list_path}")

    print(f"[batch] mode={args.mode}")
    print(f"[batch] model_preset={args.model_preset}")
    print(f"[batch] wan_root={wan_root}")
    print(f"[batch] lora_path={lora_path}")
    print(f"[batch] output_root={output_root}")
    print(f"[batch] device={args.device}")
    print(f"[batch] size={args.width}x{args.height} frames={args.num_frames} fps={args.fps}")
    print(f"[batch] cases={len(json_paths)}")
    print("[batch] building pipe once for this method")

    if args.mode == "t2v":
        pipe = single_case.build_t2v_pipeline(wan_root=wan_root, device=str(args.device), lora_path=lora_path)
    else:
        pipe = single_case.build_ti2v_pipeline(wan_root=wan_root, device=str(args.device), lora_path=lora_path)

    failures: list[str] = []
    for index, json_path in enumerate(json_paths, start=1):
        case_name = json_path.stem
        output_video_path = output_root / f"{case_name}.mp4"
        output_log_path = output_root / f"{case_name}.log"

        if output_video_path.exists() and not args.overwrite:
            print(f"[case:skip] {index}/{len(json_paths)} case={case_name} output_exists=1", flush=True)
            continue

        lines = [
            f"mode={args.mode}",
            f"model_preset={args.model_preset}",
            f"wan_root={wan_root}",
            f"lora_path={lora_path}",
            f"input_json={json_path}",
            f"output_video={output_video_path}",
        ]

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            prompt = str(payload.get("input_caption", "")).strip()
            if not prompt:
                raise ValueError("missing input_caption")
            lines.append(f"prompt={prompt}")
            print(f"[case:start] {index}/{len(json_paths)} case={case_name}", flush=True)

            if args.mode == "t2v":
                save_t2v_with_pipe(
                    pipe=pipe,
                    args=args,
                    wan_root=wan_root,
                    lora_path=lora_path,
                    input_json_path=json_path,
                    prompt=prompt,
                    output_video_path=output_video_path,
                )
            else:
                context_path_value = first_existing_path(
                    payload,
                    ["input_video", "input_video_24f", "input_video_16f", "input_video_4f", "input_video_randomf", "source_video"],
                )
                if context_path_value is None:
                    raise ValueError("missing context video path for ti2v")
                context_path = Path(context_path_value).expanduser().resolve()
                single_case.assert_exists(context_path, "Context path")
                first_frame_value = first_existing_path(payload, ["input_image"])
                first_frame_path = None
                if first_frame_value is not None:
                    first_frame_path = Path(first_frame_value).expanduser().resolve()
                    single_case.assert_exists(first_frame_path, "First frame path")
                lines.append(f"context_path={context_path}")
                lines.append(f"first_frame_path={first_frame_path}")
                save_ti2v_with_pipe(
                    pipe=pipe,
                    args=args,
                    wan_root=wan_root,
                    lora_path=lora_path,
                    input_json_path=json_path,
                    prompt=prompt,
                    context_path=context_path,
                    first_frame_path=first_frame_path,
                    output_video_path=output_video_path,
                )

            lines.append("status=done")
            write_case_log(output_log_path, lines)
            print(f"[case:done] {index}/{len(json_paths)} case={case_name} video={output_video_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{case_name}:{exc}")
            lines.append(f"status=failed")
            lines.append(f"error={exc}")
            lines.append("")
            lines.append(traceback.format_exc().rstrip())
            write_case_log(output_log_path, lines)
            print(f"[case:failed] {index}/{len(json_paths)} case={case_name} error={exc} log={output_log_path}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failures:
        print(f"[batch:failed] mode={args.mode} method={args.model_preset} failures={len(failures)}", flush=True)
        for item in failures:
            print(f"[failure] {item}", flush=True)
        raise SystemExit(1)

    print(f"[batch:done] mode={args.mode} method={args.model_preset} cases={len(json_paths)}", flush=True)


if __name__ == "__main__":
    main()
