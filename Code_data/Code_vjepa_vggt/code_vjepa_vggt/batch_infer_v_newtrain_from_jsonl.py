from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from code_vjepa_vggt.infer_v_newtrain_context_video_wan import (
    _build_object_context,
    _load_context_video,
    _load_v_newtrain_state_into_model,
    _resolve_launch_device,
    _tensor_video_to_pil_list,
    build_model,
)
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8
from diffsynth.utils.data import save_video


def _read_list_file(list_path: Path) -> list[Path]:
    items: list[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            items.append(Path(line).expanduser().resolve())
    return items


def _load_input_json(json_path: Path) -> dict[str, object]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"input json must be an object: {json_path}")
    return data


def _ensure_str_field(payload: dict[str, object], key: str, json_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or empty {key!r} in {json_path}")
    return value.strip()


def _resolve_input_video(payload: dict[str, object], json_path: Path) -> str:
    key = "input_video"
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise KeyError(f"missing required field {key!r} in {json_path}")


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            if not line.endswith("\n"):
                handle.write("\n")


def _apply_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser, config: dict[str, object]) -> None:
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    train_cfg = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}

    resolution = data_cfg.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        if args.height == parser.get_default("height"):
            args.height = int(resolution[0])
        if args.width == parser.get_default("width"):
            args.width = int(resolution[1])
    if "height" in data_cfg and args.height == parser.get_default("height"):
        args.height = int(data_cfg["height"])
    if "width" in data_cfg and args.width == parser.get_default("width"):
        args.width = int(data_cfg["width"])
    if "num_frames" in data_cfg and args.num_frames == parser.get_default("num_frames"):
        args.num_frames = int(data_cfg["num_frames"])
    if "fps" in data_cfg and args.fps == parser.get_default("fps"):
        args.fps = int(data_cfg["fps"])
    if "fixed_num_context_frames" in data_cfg and args.context_frames == parser.get_default("context_frames"):
        args.context_frames = int(data_cfg["fixed_num_context_frames"])
    if "num_context_frames" in data_cfg and args.context_frames == parser.get_default("context_frames"):
        args.context_frames = int(data_cfg["num_context_frames"])

    if "wan_root" in model_cfg and args.wan_root == parser.get_default("wan_root"):
        args.wan_root = str(model_cfg["wan_root"])
    if "wan_ckpt_dir" in model_cfg and args.wan_root == parser.get_default("wan_root"):
        args.wan_root = str(model_cfg["wan_ckpt_dir"])
    if "lora_rank" in train_cfg and args.lora_rank == parser.get_default("lora_rank"):
        args.lora_rank = int(train_cfg["lora_rank"])
    if "wan_lora_rank" in model_cfg and args.lora_rank == parser.get_default("lora_rank"):
        args.lora_rank = int(model_cfg["wan_lora_rank"])
    if "object_num_queries" in model_cfg and args.object_num_queries == parser.get_default("object_num_queries"):
        args.object_num_queries = int(model_cfg["object_num_queries"])
    if "aux_max_objects" in model_cfg and args.aux_max_objects == parser.get_default("aux_max_objects"):
        args.aux_max_objects = int(model_cfg["aux_max_objects"])
    if "sam2_max_objects" in model_cfg and args.aux_max_objects == parser.get_default("aux_max_objects"):
        args.aux_max_objects = int(model_cfg["sam2_max_objects"])
    if "jepa_ckpt_path" in model_cfg and args.jepa_ckpt_path == parser.get_default("jepa_ckpt_path"):
        args.jepa_ckpt_path = str(model_cfg["jepa_ckpt_path"])
    if "jepa_input_size" in model_cfg and args.jepa_input_size == parser.get_default("jepa_input_size"):
        args.jepa_input_size = int(model_cfg["jepa_input_size"])
    if "jepa_patch_size" in model_cfg and args.jepa_patch_size == parser.get_default("jepa_patch_size"):
        args.jepa_patch_size = int(model_cfg["jepa_patch_size"])
    if "jepa_tubelet_size" in model_cfg and args.jepa_tubelet_size == parser.get_default("jepa_tubelet_size"):
        args.jepa_tubelet_size = int(model_cfg["jepa_tubelet_size"])
    if "cotracker_checkpoint" in model_cfg and args.cotracker_checkpoint == parser.get_default("cotracker_checkpoint"):
        args.cotracker_checkpoint = str(model_cfg["cotracker_checkpoint"])
    cotracker_input_hw = model_cfg.get("cotracker_input_hw")
    if isinstance(cotracker_input_hw, (list, tuple)) and len(cotracker_input_hw) == 2:
        if args.cotracker_input_h == parser.get_default("cotracker_input_h"):
            args.cotracker_input_h = int(cotracker_input_hw[0])
        if args.cotracker_input_w == parser.get_default("cotracker_input_w"):
            args.cotracker_input_w = int(cotracker_input_hw[1])
    if "cotracker_input_h" in model_cfg and args.cotracker_input_h == parser.get_default("cotracker_input_h"):
        args.cotracker_input_h = int(model_cfg["cotracker_input_h"])
    if "cotracker_input_w" in model_cfg and args.cotracker_input_w == parser.get_default("cotracker_input_w"):
        args.cotracker_input_w = int(model_cfg["cotracker_input_w"])
    if "cotracker_window_len" in model_cfg and args.cotracker_window_len == parser.get_default("cotracker_window_len"):
        args.cotracker_window_len = int(model_cfg["cotracker_window_len"])
    if "object_pooler_latent_dim" in model_cfg and args.object_pooler_latent_dim == parser.get_default("object_pooler_latent_dim"):
        args.object_pooler_latent_dim = int(model_cfg["object_pooler_latent_dim"])
    if "cond_proj_dim" in model_cfg and args.cond_proj_dim == parser.get_default("cond_proj_dim"):
        args.cond_proj_dim = int(model_cfg["cond_proj_dim"])
    if "jepa_window_radius" in model_cfg and args.jepa_window_radius == parser.get_default("jepa_window_radius"):
        args.jepa_window_radius = int(model_cfg["jepa_window_radius"])
    if "latent_window_radius" in model_cfg and args.latent_window_radius == parser.get_default("latent_window_radius"):
        args.latent_window_radius = int(model_cfg["latent_window_radius"])


def _build_model_args(args: argparse.Namespace) -> SimpleNamespace:
    cotracker_input_hw = [int(args.cotracker_input_h), int(args.cotracker_input_w)]
    return SimpleNamespace(
        device=args.device,
        wan_root=str(args.wan_root),
        lora_rank=int(args.lora_rank),
        context_frames=int(args.context_frames),
        disable_object_branch=False,
        object_num_queries=int(args.object_num_queries),
        aux_max_objects=int(args.aux_max_objects),
        jepa_ckpt_path=str(args.jepa_ckpt_path),
        jepa_input_size=int(args.jepa_input_size),
        jepa_patch_size=int(args.jepa_patch_size),
        jepa_tubelet_size=int(args.jepa_tubelet_size),
        cotracker_checkpoint=str(args.cotracker_checkpoint),
        cotracker_input_h=int(cotracker_input_hw[0]),
        cotracker_input_w=int(cotracker_input_hw[1]),
        cotracker_window_len=int(args.cotracker_window_len),
        object_pooler_latent_dim=int(args.object_pooler_latent_dim),
        cond_proj_dim=int(args.cond_proj_dim),
        jepa_window_radius=int(args.jepa_window_radius),
        latent_window_radius=int(args.latent_window_radius),
        # build_model() only consumes the attributes above plus the standard ones.
        wan_lora_rank=int(args.lora_rank),
    )


def _run_single_case_in_process(
    *,
    model,
    checkpoint_dir: Path,
    input_json_path: Path,
    input_video: str,
    input_caption: str,
    output_dir: Path,
    output_video: Path,
    num_frames: int,
    context_frames: int,
    sampling_mode: str,
    sampling_steps: int,
    fps: int,
    seed: int,
    cfg_scale: float,
    height: int,
    width: int,
    quality: int,
) -> tuple[dict[str, object], list[str]]:
    logs: list[str] = []
    logs.append(f"[case] input_json={input_json_path}")
    logs.append(f"[case] input_video={input_video}")
    logs.append(f"[case] input_caption={input_caption}")
    logs.append(f"[case] checkpoint_dir={checkpoint_dir}")

    frames, frame_indices = _load_context_video(
        video_path=Path(input_video),
        target_context_frames=int(context_frames),
        sampling_mode=sampling_mode,
    )
    context_video_single = preprocess_video_rgb_uint8(frames, (int(height), int(width)))
    context_pil = _tensor_video_to_pil_list(context_video_single)
    object_context, object_debug = _build_object_context(
        model=model,
        context_video_single=context_video_single,
    )

    pipe = model.pipe
    pipe.dit.eval()
    with torch.no_grad():
        video = pipe(
            prompt=input_caption,
            negative_prompt="",
            context_video=context_pil,
            seed=int(seed),
            tiled=True,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(sampling_steps),
            cfg_scale=float(cfg_scale),
            object_context=object_context,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_video), fps=int(fps), quality=int(quality))

    result = {
        "input_json": str(input_json_path),
        "input_video": str(input_video),
        "input_caption": str(input_caption),
        "output_video": str(output_video),
        "seed": int(seed),
        "step": int(sampling_steps),
        "guidance": float(cfg_scale),
        "ckpt": str(checkpoint_dir),
        "frame_indices": frame_indices.tolist(),
        "object_debug": object_debug,
    }
    return result, logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run v_newtrain inference from a list of input json files.")
    parser.add_argument("--input-list", required=True, help="text file containing one input json path per line")
    parser.add_argument("--checkpoint-root", required=True, help="root dir containing step-xxxx checkpoint folders")
    parser.add_argument("--steps", nargs="+", required=True, help="checkpoint step names such as step-000400")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wan-root", default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--disable-object-branch", action="store_true")
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--jepa-ckpt-path", default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth")
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--jepa-window-radius", type=int, default=1)
    parser.add_argument("--latent-window-radius", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_list = Path(args.input_list).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    config = load_yaml_config(config_path)
    _apply_config_defaults(args, parser, config)
    args.device = _resolve_launch_device()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    json_paths = _read_list_file(input_list)
    output_root.mkdir(parents=True, exist_ok=True)

    run_manifest: dict[str, object] = {
        "input_list": str(input_list),
        "checkpoint_root": str(checkpoint_root),
        "steps": [str(step) for step in args.steps],
        "num_items": len(json_paths),
        "sampling_steps": int(args.sampling_steps),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "context_frames": int(args.context_frames),
        "input_field": "input_video",
        "single_process": True,
    }
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, ensure_ascii=False)

    model_args = _build_model_args(args)

    for step_name in args.steps:
        checkpoint_dir = checkpoint_root / str(step_name)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

        step_output_dir = output_root / str(step_name)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        step_result_json = step_output_dir / "result.json"
        if step_result_json.exists():
            step_result_json.unlink()

        model = build_model(model_args)
        model.to(torch.device(args.device))
        model.eval()
        load_info = _load_v_newtrain_state_into_model(model, checkpoint_dir)
        model.pipe.dit.eval()

        step_log_lines = [
            f"[checkpoint] {checkpoint_dir}",
            f"[load_info] {json.dumps(load_info, ensure_ascii=False)}",
        ]

        for input_json_path in json_paths:
            payload = _load_input_json(input_json_path)
            try:
                input_video = _resolve_input_video(payload, input_json_path)
            except KeyError as exc:
                print(f"[skip] {step_name} {input_json_path.stem}: {exc}")
                continue

            try:
                input_caption = _ensure_str_field(payload, "input_caption", input_json_path)
            except ValueError as exc:
                print(f"[skip] {step_name} {input_json_path.stem}: {exc}")
                continue

            sample_stem = input_json_path.stem
            output_video = step_output_dir / f"{sample_stem}.mp4"
            output_json = step_output_dir / f"{sample_stem}.json"
            output_log = step_output_dir / f"{sample_stem}.log"

            if output_video.exists() and output_json.exists() and not args.force:
                print(f"[skip] {step_name} {sample_stem}")
                continue

            try:
                result, case_logs = _run_single_case_in_process(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    input_json_path=input_json_path,
                    input_video=input_video,
                    input_caption=input_caption,
                    output_dir=step_output_dir,
                    output_video=output_video,
                    num_frames=int(args.num_frames),
                    context_frames=int(args.context_frames),
                    sampling_mode=str(args.sampling_mode),
                    sampling_steps=int(args.sampling_steps),
                    fps=int(args.fps),
                    seed=int(args.seed),
                    cfg_scale=float(args.cfg_scale),
                    height=int(args.height),
                    width=int(args.width),
                    quality=int(args.quality),
                )
            except Exception as exc:
                error_lines = step_log_lines + [f"[error] {sample_stem}: {exc}"]
                _write_text_lines(output_log, error_lines)
                print(f"[error] {step_name} {sample_stem}: {exc}")
                continue

            success_lines = step_log_lines + case_logs + [f"[done] {step_name} {sample_stem}"]
            _write_text_lines(output_log, success_lines)

            with output_json.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

            print(f"[done] {step_name} {sample_stem}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
