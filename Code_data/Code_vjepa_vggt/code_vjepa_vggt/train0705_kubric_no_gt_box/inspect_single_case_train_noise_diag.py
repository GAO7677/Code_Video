from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu

PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
for _path in (str(PROJECT_ROOT), str(DIFFSYNTH_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.context_wan_v_newtrain import (
    _tensor_numeric_stats,
    apply_clean_latents_at_indices,
    flow_match_context_sft_loss,
    resolve_context_latent_indices_from_frames,
    slice_non_context_latents,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_train_forward_aux_overlay as inspectmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as trainmod,
)
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


DEFAULT_INPUT_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/"
    "physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json"
)
DEFAULT_STAGE2_CHECKPOINTS = (
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_kubric0708/checkpoints/step-003500,"
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_kubric0708/checkpoints/step-004500,"
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_kubric0708/checkpoints/step-005000"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/train_forward_noise_diag_20260710/"
    "physicIQ_026_step003500_004500_005000"
)


def _parse_csv_str(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_csv_int(raw: str | None) -> list[int]:
    return [int(part) for part in _parse_csv_str(raw)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run train-style single-step noise->denoise diagnostics on one arbitrary video case "
            "for multiple stage1b checkpoints."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--source-video", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--stage2-checkpoints", default=DEFAULT_STAGE2_CHECKPOINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--timestep-indices", default="50,250,500,750,950")
    parser.add_argument("--noise-seeds", default="0,1,2")
    parser.add_argument("--decode-noise-seeds", default="0")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wan-root", default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
    parser.add_argument("--diffsynth-root", default=str(DIFFSYNTH_ROOT))
    parser.add_argument(
        "--lora-checkpoint",
        default=(
            "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
            "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
        ),
    )
    parser.add_argument(
        "--stage1a-init-from",
        default=(
            "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
            "pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt"
        ),
    )
    parser.add_argument("--grounding-device", default=None)
    parser.add_argument("--aux-device", default=None)
    return parser.parse_args()


def _build_train_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    parser = trainmod.build_parser()
    parsed = parser.parse_args(
        [
            "--diffsynth_root",
            str(cli_args.diffsynth_root),
            "--wan_root",
            str(cli_args.wan_root),
            "--dataset_type",
            "kubric_no_gt_box",
            "--kubric_root",
            "/data/gaoya/dataset/nnsriram97-phyco_kubric",
            "--kubric_split",
            "train",
            "--height",
            str(int(cli_args.height)),
            "--width",
            str(int(cli_args.width)),
            "--num_frames",
            str(int(cli_args.num_frames)),
            "--fixed_num_context_frames",
            str(int(cli_args.context_frames)),
            "--ctx_max_length",
            str(max(int(cli_args.context_frames) - 1, 0)),
            "--min_context_frames",
            str(max(int(cli_args.context_frames) - 1, 0)),
            "--max_context_ratio",
            "1.0",
            "--no_context_ratio",
            "0.0",
            "--lora_checkpoint",
            str(cli_args.lora_checkpoint),
            "--stage1a_init_from",
            str(cli_args.stage1a_init_from),
            "--extra_inputs",
            "input_image",
            "--enable_object_branch",
            "--freeze_non_object_trainables",
            "--train_object_adapter",
            "--train_object_dit_branch",
            "--object_num_queries",
            "8",
            "--aux_max_objects",
            "4",
            "--jepa_ckpt_path",
            "/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth",
            "--jepa_input_size",
            "384",
            "--jepa_patch_size",
            "16",
            "--jepa_tubelet_size",
            "2",
            "--cotracker_checkpoint",
            "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
            "--cotracker_input_h",
            "384",
            "--cotracker_input_w",
            "512",
            "--cotracker_window_len",
            "60",
            "--vggt_model_path",
            "/data/gaoya/ckpt/facebook-VGGT-1B",
            "--vggt_input_h",
            "420",
            "--vggt_input_w",
            "728",
            "--object_pooler_latent_dim",
            "16",
            "--cond_proj_dim",
            "4096",
            "--jepa_window_radius",
            "1",
            "--latent_window_radius",
            "1",
            "--object_gate_init",
            "0.1",
            "--lambda_main",
            "1.0",
            "--lambda_track_aux",
            "0.0",
            "--lambda_box_aux",
            "0.0",
            "--lambda_depth_aux",
            "0.0",
            "--report_to",
            "none",
            "--grounding_proposal_source",
            "gdino_only",
            "--grounding_motion_score_ratio",
            "0.15",
            "--grounding_text_prompt",
            "box . cube . block . cylinder . capsule . sphere . ball .",
            "--grounding_disable_caption_terms",
            "--grounding_gdino_box_threshold",
            "0.20",
            "--grounding_gdino_text_threshold",
            "0.15",
            "--grounding_prompt_frame_mode",
            "first",
            "--grounding_track_dedupe_iou_threshold",
            "0.75",
            "--grounding_container_suppress_ratio_threshold",
            "0.95",
            "--grounding_container_suppress_min_contained",
            "2",
            "--grounding_container_suppress_min_area_ratio",
            "1.5",
            "--grounding_container_suppress_small_iou_threshold",
            "0.7",
            "--sam2_segment_len",
            "8",
        ]
    )
    parsed = trainmod.tvn.prepare_args(parsed)
    parsed.device = str(cli_args.device)
    parsed.grounding_device = cli_args.grounding_device
    parsed.aux_device = cli_args.aux_device
    parsed.vggt_cache_root = None
    parsed.initialize_model_on_cpu = False
    return parsed


def _load_case_sample(cli_args: argparse.Namespace) -> tuple[dict[str, Any], Path, str]:
    payload = core._load_input_json(cli_args.input_json)
    source_video = (
        cli_args.source_video.expanduser().resolve()
        if cli_args.source_video is not None
        else Path(str(payload["source_video"])).expanduser().resolve()
    )
    prompt = str(cli_args.prompt or core._ensure_str_field(payload, "input_caption", cli_args.input_json))

    vr = VideoReader(str(source_video), ctx=cpu(0))
    frame_count = len(vr)
    if frame_count < int(cli_args.num_frames):
        raise ValueError(
            f"source video has only {frame_count} frames, smaller than requested {int(cli_args.num_frames)}"
        )
    frame_indices = np.arange(int(cli_args.num_frames), dtype=np.int64)
    frames = vr.get_batch(frame_indices).asnumpy()
    video = preprocess_video_rgb_uint8(
        frames,
        (int(cli_args.height), int(cli_args.width)),
        value_range="minus_one_to_one",
    )
    context_indices = torch.arange(int(cli_args.context_frames), dtype=torch.long)
    sample = {
        "video": video,
        "context_video": video[:, context_indices].contiguous(),
        "caption": prompt,
        "video_path": str(source_video),
        "frame_indices": torch.arange(int(cli_args.num_frames), dtype=torch.long),
        "context_frame_indices": context_indices.clone(),
        "num_context_frames": int(cli_args.context_frames),
        "metadata": {
            "sample_key": f"single_case::{source_video.stem}",
            "source_video_path": str(source_video),
            "source_frame_count": int(frame_count),
            "sampled_frame_indices": frame_indices.tolist(),
            "sampling_strategy": "prefix",
        },
        "sampled_ctx_last_index": int(max(int(cli_args.context_frames) - 1, -1)),
        "sampled_ctx_num_frames": int(cli_args.context_frames),
        "ctx_max_length": int(max(int(cli_args.context_frames) - 1, 0)),
    }
    return sample, source_video, prompt


def _move_optional_module(module: torch.nn.Module | None, device: torch.device) -> None:
    if module is not None:
        module.to(device)


def _load_model_for_checkpoint(
    train_args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
) -> trainmod.ContextOnlyNoGTBoxWanModule:
    accelerator = SimpleNamespace(device=torch.device(str(train_args.device)))
    model = trainmod.build_model(train_args, accelerator)
    target_device = torch.device(model.pipe.device)
    _move_optional_module(model.object_pooler, target_device)
    _move_optional_module(model.object_aux_heads, target_device)
    _move_optional_module(model.object_adapter, target_device)
    _move_optional_module(model.vggt_adapter, target_device)
    if train_args.stage1a_init_from is not None:
        trainmod.tvn._load_filtered_checkpoint_into_model(
            model,
            train_args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
    checkpoint_file = str(trainmod.tvn._resolve_checkpoint_file(str(checkpoint_dir)))
    trainmod.tvn._load_filtered_checkpoint_into_model(
        model,
        checkpoint_file,
        include_prefixes=("object_adapter.",),
        include_substrings=(
            "object_embedding",
            ".object_cross_attn.",
            ".object_gate",
            ".norm4.",
        ),
    )
    torch.nn.Module.train(model, False)
    return model


def _collect_object_branch_trace_summary(trace_layers: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not trace_layers:
        return {
            "num_layers": 0,
            "max_gated_to_x_ratio_l2": 0.0,
            "mean_gated_to_x_ratio_l2": 0.0,
            "max_ratio_block_id": None,
        }
    ratios = [float(layer.get("gated_to_x_ratio_l2", 0.0)) for layer in trace_layers]
    max_idx = int(np.argmax(ratios))
    return {
        "num_layers": len(trace_layers),
        "max_gated_to_x_ratio_l2": float(ratios[max_idx]),
        "mean_gated_to_x_ratio_l2": float(sum(ratios) / max(len(ratios), 1)),
        "max_ratio_block_id": int(trace_layers[max_idx].get("block_id", -1)),
        "max_ratio_layer": trace_layers[max_idx],
    }


def _decode_latents_to_video(
    pipe,
    latents: torch.Tensor,
) -> torch.Tensor:
    pipe.load_models_to_device(["vae"])
    return pipe.vae.decode(
        latents,
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )


def _run_single_trial(
    model: trainmod.ContextOnlyNoGTBoxWanModule,
    *,
    sample: dict[str, Any],
    inputs_shared: dict[str, Any],
    inputs_posi: dict[str, Any],
    object_context: torch.Tensor,
    timestep_index: int,
    noise_seed: int,
    decode_video: bool,
    output_dir: Path,
    fps: int,
) -> dict[str, Any]:
    pipe = model.pipe
    input_latents = inputs_shared["input_latents"]
    latent_length = int(input_latents.shape[2])
    context_latent_indices = resolve_context_latent_indices_from_frames(
        raw_frame_indices=inputs_shared.get("context_frame_indices"),
        raw_num_frames=inputs_shared.get("num_frames"),
        latent_length=latent_length,
    )
    if not (0 <= int(timestep_index) < len(pipe.scheduler.timesteps)):
        raise ValueError(f"timestep_index {timestep_index} out of range for {len(pipe.scheduler.timesteps)} timesteps")
    timestep = pipe.scheduler.timesteps[int(timestep_index) : int(timestep_index) + 1].to(
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    generator = torch.Generator(device=input_latents.device)
    generator.manual_seed(int(noise_seed))
    noise = torch.randn(
        tuple(input_latents.shape),
        generator=generator,
        device=input_latents.device,
        dtype=input_latents.dtype,
    )
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)
    latents_noisy = pipe.scheduler.add_noise(input_latents, noise, timestep)
    if context_latent_indices:
        latents_noisy = apply_clean_latents_at_indices(
            latents_noisy,
            input_latents,
            context_latent_indices,
        )

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    active_dit = getattr(pipe, "dit", None)
    if active_dit is not None and hasattr(active_dit, "_object_branch_trace_collect"):
        active_dit._object_branch_trace_collect = True
        active_dit._object_branch_trace_buffer = []
    try:
        model_inputs = dict(inputs_shared)
        model_inputs["latents"] = latents_noisy
        noise_pred = pipe.model_fn(
            **models,
            **model_inputs,
            **inputs_posi,
            object_context=object_context,
            timestep=timestep,
        )
        trace_layers = None
        if active_dit is not None and hasattr(active_dit, "_object_branch_trace_buffer"):
            trace_layers = getattr(active_dit, "_object_branch_trace_buffer", None)
    finally:
        if active_dit is not None and hasattr(active_dit, "_object_branch_trace_collect"):
            active_dit._object_branch_trace_collect = False
            active_dit._object_branch_trace_buffer = None

    if context_latent_indices:
        noise_pred_future = slice_non_context_latents(
            noise_pred,
            latent_length=latent_length,
            context_latent_indices=context_latent_indices,
        )
        training_target_future = slice_non_context_latents(
            training_target,
            latent_length=latent_length,
            context_latent_indices=context_latent_indices,
        )
    else:
        noise_pred_future = noise_pred
        training_target_future = training_target

    loss_raw = torch.nn.functional.mse_loss(
        noise_pred_future.float(),
        training_target_future.float(),
    )
    loss_weighted = loss_raw * pipe.scheduler.training_weight(timestep)
    x0_hat = pipe.scheduler.step(
        noise_pred,
        timestep,
        latents_noisy,
        to_final=True,
    )
    if context_latent_indices:
        x0_hat = apply_clean_latents_at_indices(
            x0_hat,
            input_latents,
            context_latent_indices,
        )
        x0_hat_future = slice_non_context_latents(
            x0_hat,
            latent_length=latent_length,
            context_latent_indices=context_latent_indices,
        )
        clean_future = slice_non_context_latents(
            input_latents,
            latent_length=latent_length,
            context_latent_indices=context_latent_indices,
        )
    else:
        x0_hat_future = x0_hat
        clean_future = input_latents
    x0_latent_mse_future = torch.nn.functional.mse_loss(
        x0_hat_future.float(),
        clean_future.float(),
    )

    trial_result: dict[str, Any] = {
        "timestep_index": int(timestep_index),
        "timestep_value": float(timestep.detach().float().cpu().item()),
        "noise_seed": int(noise_seed),
        "loss_raw_mse": float(loss_raw.detach().item()),
        "loss_weighted": float(loss_weighted.detach().item()),
        "x0_latent_mse_future": float(x0_latent_mse_future.detach().item()),
        "latents_noisy_stats": _tensor_numeric_stats(latents_noisy),
        "noise_pred_stats": _tensor_numeric_stats(noise_pred),
        "x0_hat_stats": _tensor_numeric_stats(x0_hat),
        "object_branch_trace_summary": _collect_object_branch_trace_summary(trace_layers),
    }

    if decode_video:
        decoded_x0 = _decode_latents_to_video(pipe, x0_hat)
        decoded_gt = _decode_latents_to_video(pipe, input_latents)
        decoded_future_mse = torch.nn.functional.mse_loss(
            decoded_x0.float(),
            decoded_gt.float(),
        )
        video_name = f"trial_t{int(timestep_index):04d}_seed{int(noise_seed):03d}_x0hat.mp4"
        gt_name = f"trial_t{int(timestep_index):04d}_seed{int(noise_seed):03d}_gtdecode.mp4"
        inspectmod._write_tensor_video(output_dir / video_name, decoded_x0[0].detach().cpu(), fps=int(fps))
        inspectmod._write_tensor_video(output_dir / gt_name, decoded_gt[0].detach().cpu(), fps=int(fps))
        trial_result["decoded_video"] = video_name
        trial_result["decoded_gt_video"] = gt_name
        trial_result["decoded_video_mse_all"] = float(decoded_future_mse.detach().item())
    return trial_result


def _summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {}
    keys = [
        "loss_raw_mse",
        "loss_weighted",
        "x0_latent_mse_future",
    ]
    summary: dict[str, Any] = {"num_trials": len(trials)}
    for key in keys:
        values = [float(item[key]) for item in trials]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    max_ratios = [
        float(item["object_branch_trace_summary"]["max_gated_to_x_ratio_l2"])
        for item in trials
    ]
    summary["object_branch_trace_summary"] = {
        "max_ratio_mean": float(np.mean(max_ratios)),
        "max_ratio_std": float(np.std(max_ratios)),
        "max_ratio_max": float(np.max(max_ratios)),
    }
    return summary


def _inspect_checkpoint(
    train_args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    sample: dict[str, Any],
    output_dir: Path,
    timestep_indices: list[int],
    noise_seeds: list[int],
    decode_noise_seeds: set[int],
    fps: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model_for_checkpoint(train_args, checkpoint_dir=checkpoint_dir)
    try:
        inputs_shared, inputs_posi, _inputs_nega = inspectmod._prepare_forward_inputs(model, sample)
        forward_debug = inspectmod._run_forward_debug(
            model,
            sample,
            inputs_shared=inputs_shared,
            inputs_posi=inputs_posi,
        )
        object_context = forward_debug["object_context"]
        checkpoint_report: dict[str, Any] = {
            "checkpoint": str(checkpoint_dir),
            "forward_metrics": forward_debug["metrics"],
            "object_context_stats": _tensor_numeric_stats(object_context),
            "object_valid_mask": forward_debug["object_valid_mask"][0].detach().float().cpu().tolist(),
            "trials": [],
        }
        for timestep_index in timestep_indices:
            for noise_seed in noise_seeds:
                trial = _run_single_trial(
                    model,
                    sample=sample,
                    inputs_shared=inputs_shared,
                    inputs_posi=inputs_posi,
                    object_context=object_context,
                    timestep_index=int(timestep_index),
                    noise_seed=int(noise_seed),
                    decode_video=int(noise_seed) in decode_noise_seeds,
                    output_dir=output_dir,
                    fps=int(fps),
                )
                checkpoint_report["trials"].append(trial)
        checkpoint_report["trial_summary"] = _summarize_trials(checkpoint_report["trials"])
        return checkpoint_report
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    cli_args = parse_args()
    random.seed(int(cli_args.seed))
    np.random.seed(int(cli_args.seed))
    torch.manual_seed(int(cli_args.seed))

    train_args = _build_train_args(cli_args)
    sample, source_video, prompt = _load_case_sample(cli_args)
    output_dir = cli_args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestep_indices = _parse_csv_int(cli_args.timestep_indices)
    noise_seeds = _parse_csv_int(cli_args.noise_seeds)
    decode_noise_seeds = set(_parse_csv_int(cli_args.decode_noise_seeds))
    checkpoint_dirs = [Path(item).expanduser().resolve() for item in _parse_csv_str(cli_args.stage2_checkpoints)]

    reports = []
    for checkpoint_dir in checkpoint_dirs:
        label = checkpoint_dir.name
        report = _inspect_checkpoint(
            train_args,
            checkpoint_dir=checkpoint_dir,
            sample=copy.deepcopy(sample),
            output_dir=output_dir / label,
            timestep_indices=timestep_indices,
            noise_seeds=noise_seeds,
            decode_noise_seeds=decode_noise_seeds,
            fps=int(cli_args.fps),
        )
        report["label"] = label
        reports.append(report)

    summary = {
        "input_json": str(cli_args.input_json),
        "source_video": str(source_video),
        "prompt": prompt,
        "num_frames": int(cli_args.num_frames),
        "context_frames": int(cli_args.context_frames),
        "timestep_indices": timestep_indices,
        "noise_seeds": noise_seeds,
        "decode_noise_seeds": sorted(int(v) for v in decode_noise_seeds),
        "reports": reports,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
