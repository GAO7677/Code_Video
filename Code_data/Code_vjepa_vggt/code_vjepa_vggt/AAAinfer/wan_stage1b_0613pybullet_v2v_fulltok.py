from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root
from code_vjepa_vggt.infer_v_newtrain_context_video_wan import (
    _build_center_box_query_priors,
    _load_v_newtrain_state_into_model,
    _load_context_video,
    _select_video_from_path,
    _tensor_video_to_pil_list,
)
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8
from diffsynth.utils.data import save_video

DEFAULT_LORA_CKPT = Path(
    "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
    "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
)

"""
Full-token Stage1B inference: same as wan_stage1b_0613pybullet_v2v.py but constructs
object_context for ALL latent time steps (context + future), giving the DiT cross-attn
access to a full-length token sequence — matching the token layout used during stage2
training.

Future latent time steps have no real object information at inference time; they are
initialized by repeating the last context-latent object token.

Shape comparison:
  context-only (wan_stage1b_0613pybullet_v2v.py):  [1, 2×4=8,  4096]
  full-token   (this script):                       [1, 6×4=24, 4096]

完整运行命令：

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=0 \
python3 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_stage1b_0613pybullet_v2v_fulltok.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_oracle_cross_attn/step_0001000.pt \
  --stage1a-weights /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0629_stage1b_old8000_fulltok

--source-video-field defaults to "source_video" (the full video in the JSON).
--lora-ckpt has a default matching the lora used during stage1b training and can be omitted.
"""

DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_CONFIG = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "object_token_teacher_student/config_stage1b_oracle_cross_attn_template.yaml"
)


def _normalize_ckpt_method_name(name: str) -> str:
    normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
    return normalized or name


def _resolve_stage1b_ckpt(weights_root: Path) -> Path:
    if weights_root.is_file():
        return weights_root
    if weights_root.is_dir():
        candidates = sorted(weights_root.glob("step_*.pt"))
        if candidates:
            return candidates[-1]
        sft = weights_root / "checkpoint.safetensors"
        if sft.is_file():
            return weights_root
    raise FileNotFoundError(f"no valid checkpoint found at: {weights_root}")


def _build_method_name(weights_root: Path) -> str:
    ckpt_file = _resolve_stage1b_ckpt(weights_root)
    step_name = ckpt_file.stem
    parent = ckpt_file.parent
    if parent.name:
        method_root = _normalize_ckpt_method_name(parent.name)
        return f"{method_root}_{step_name}"
    return step_name


def _build_full_object_context(
    *,
    model,
    context_video_single: torch.Tensor,
    num_total_latent_frames: int,
) -> tuple[torch.Tensor | None, dict]:
    """Build object_context covering all latent time steps.

    Context latent frames are computed from the real context video via pooler.
    Future latent frames are filled by repeating the last context latent token,
    since no future video is available at inference time.

    Returns object_context [1, num_total_latent_frames * N_obj, D_cond].
    """
    if not model.enable_object_branch:
        return None, {"enabled": False}

    pipe = model.pipe
    device = torch.device(pipe.device)
    context_video = context_video_single.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))

    query_points_prior, object_valid_mask = _build_center_box_query_priors(
        height=image_hw[0],
        width=image_hw[1],
        aux_max_objects=model.aux_max_objects,
        object_num_queries=model.object_num_queries,
    )
    query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)

    frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
    cotracker_out = model.cotracker_adapter(
        frames_bthwc_01,
        query_points_prior=query_points_prior,
        query_image_hw=image_hw,
    )
    tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
        cotracker_out.tracks,
        cotracker_out.visibility,
        cotracker_out.confidence,
        max_objects=model.aux_max_objects,
        points_per_object=model.object_num_queries,
    )
    jepa_dtype = next(model.jepa_adapter.parameters()).dtype
    jepa_out = model.jepa_adapter(context_video.to(dtype=jepa_dtype))

    preprocessed_context = pipe.preprocess_video(_tensor_video_to_pil_list(context_video_single))
    clean_prefix_latents = pipe.vae.encode(
        preprocessed_context,
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ).to(dtype=pipe.torch_dtype, device=device)

    object_out = model.object_pooler(
        jepa_patch_tokens=jepa_out.patch_tokens,
        context_latents=clean_prefix_latents,
        tracks=tracks_grouped,
        visibility=visibility_grouped,
        confidence=confidence_grouped,
        track_image_hw=image_hw,
        object_valid_mask=object_valid_mask,
        box_prior_xyxy=torch.tensor(
            [[0.25, 0.25, 0.75, 0.75]] * model.aux_max_objects,
            dtype=pipe.torch_dtype,
            device=device,
        ).unsqueeze(0),
        frame_valid_mask=None,
    )
    # object_latent_tokens: [1, T_ctx_lat, N_obj, D_pooler]
    ctx_tokens = object_out.object_latent_tokens
    T_ctx_lat = int(ctx_tokens.shape[1])
    T_future = int(num_total_latent_frames) - T_ctx_lat

    if T_future > 0:
        last_token = ctx_tokens[:, -1:, :, :]              # [1, 1, N_obj, D_pooler]
        future_tokens = last_token.expand(-1, T_future, -1, -1).clone()
        full_tokens = torch.cat([ctx_tokens, future_tokens], dim=1)
    else:
        full_tokens = ctx_tokens
    # full_tokens: [1, num_total_latent_frames, N_obj, D_pooler]

    object_context = model.object_adapter(full_tokens, object_valid_mask=object_valid_mask)
    # object_context: [1, num_total_latent_frames * N_obj, D_cond]

    debug = {
        "enabled": True,
        "T_ctx_lat": T_ctx_lat,
        "T_future": T_future,
        "num_total_latent_frames": num_total_latent_frames,
        "object_context_shape": list(object_context.shape),
        "query_points_shape": list(query_points_prior.shape),
        "tracks_shape": list(cotracker_out.tracks.shape),
    }
    return object_context, debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-token Stage1B inference: object_context covers all latent frames "
            "(context + future filled with last context token repeat)."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    # VAE temporal downsampling factor: 24 frames → 24/4 = 6 latent frames
    parser.add_argument("--vae-temporal-stride", type=int, default=4,
                        help="VAE temporal downsampling factor (default: 4)")
    parser.add_argument(
        "--source-video-field", type=str, default="source_video",
        help="JSON field name for the full source video (default: 'source_video'). "
             "Falls back to input_video if the field is absent.",
    )
    parser.add_argument(
        "--lora-ckpt", type=Path, default=DEFAULT_LORA_CKPT,
        help="LoRA safetensors file (frozen during stage1b). Defaults to stage1b training lora.",
    )
    parser.add_argument(
        "--stage1a-weights", type=Path, default=None, required=True,
        help=".pt file from stage1a containing object_pooler weights.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    weights_root = cli_args.weights_root.expanduser().resolve()
    input_json_list_path = cli_args.input_json_list_path.expanduser().resolve()
    model_name = str(cli_args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root="/data/gaoya/AAA_test_video/0623/test/v2v",
        model_name=model_name,
    )

    config_path = cli_args.config.expanduser().resolve()
    config = load_yaml_config(config_path)
    defaults_parser = argparse.ArgumentParser()
    defaults_parser.add_argument("--height", type=int, default=512)
    defaults_parser.add_argument("--width", type=int, default=896)
    defaults_parser.add_argument("--num-frames", type=int, default=24)
    defaults_parser.add_argument("--fps", type=int, default=30)
    defaults_parser.add_argument("--context-frames", type=int, default=8)
    defaults_parser.add_argument("--wan-root", default=str(DEFAULT_WAN_ROOT))
    defaults_parser.add_argument("--lora-rank", type=int, default=32)
    defaults_parser.add_argument("--object-num-queries", type=int, default=8)
    defaults_parser.add_argument("--aux-max-objects", type=int, default=4)
    defaults_parser.add_argument("--jepa-ckpt-path", default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth")
    defaults_parser.add_argument("--jepa-input-size", type=int, default=384)
    defaults_parser.add_argument("--jepa-patch-size", type=int, default=16)
    defaults_parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    defaults_parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    defaults_parser.add_argument("--cotracker-input-h", type=int, default=384)
    defaults_parser.add_argument("--cotracker-input-w", type=int, default=512)
    defaults_parser.add_argument("--cotracker_window_len", type=int, default=60)
    defaults_parser.add_argument("--cotracker-window-len", type=int, default=60)
    defaults_parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    defaults_parser.add_argument("--cond-proj-dim", type=int, default=4096)
    defaults_parser.add_argument("--jepa-window-radius", type=int, default=1)
    defaults_parser.add_argument("--latent-window-radius", type=int, default=1)
    core._apply_config_defaults(cli_args, defaults_parser, config)

    cli_args.device = core._resolve_launch_device()
    torch.manual_seed(int(cli_args.seed))
    np.random.seed(int(cli_args.seed))

    json_paths = core._read_list_file(input_json_list_path)
    if cli_args.limit is not None:
        json_paths = json_paths[: max(0, int(cli_args.limit))]

    output_root.mkdir(parents=True, exist_ok=True)

    ckpt_file = _resolve_stage1b_ckpt(weights_root)
    step_label = ckpt_file.stem
    method_name = _build_method_name(weights_root)
    lora_ckpt = cli_args.lora_ckpt.expanduser().resolve()
    stage1a_ckpt = cli_args.stage1a_weights.expanduser().resolve()

    num_total_latent_frames = int(cli_args.num_frames) // int(cli_args.vae_temporal_stride)

    manifest = {
        "input_json_list_path": str(input_json_list_path),
        "weights_root": str(weights_root),
        "ckpt_file": str(ckpt_file),
        "lora_ckpt": str(lora_ckpt),
        "stage1a_weights": str(stage1a_ckpt),
        "num_items": len(json_paths),
        "num_inference_steps": int(cli_args.num_inference_steps),
        "cfg_scale": float(cli_args.cfg_scale),
        "seed": int(cli_args.seed),
        "height": int(cli_args.height),
        "width": int(cli_args.width),
        "num_frames": int(cli_args.num_frames),
        "context_frames": int(cli_args.context_frames),
        "num_total_latent_frames": num_total_latent_frames,
        "sampling_mode": str(cli_args.sampling_mode),
        "object_token_mode": "full_repeat_last_ctx",
    }
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    model_args = core._build_model_args(cli_args)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file}")

    step_output_dir = output_root / step_label
    step_output_dir.mkdir(parents=True, exist_ok=True)

    model = core.build_model(model_args)
    model.to(torch.device(cli_args.device))
    model.eval()

    if not lora_ckpt.is_file():
        raise FileNotFoundError(f"--lora-ckpt not found: {lora_ckpt}")
    lora_info = _load_v_newtrain_state_into_model(model, lora_ckpt)
    print(f"[lora] loaded={lora_info['loaded_count']} missing={len(lora_info['missing_keys'])}")

    if not stage1a_ckpt.is_file():
        raise FileNotFoundError(f"--stage1a-weights not found: {stage1a_ckpt}")
    pooler_info = _load_v_newtrain_state_into_model(model, stage1a_ckpt)
    print(f"[stage1a pooler] loaded={pooler_info['loaded_count']} missing={len(pooler_info['missing_keys'])}")

    load_info = _load_v_newtrain_state_into_model(model, ckpt_file)
    print(f"[stage1b] loaded={load_info['loaded_count']} "
          f"missing={len(load_info['missing_keys'])} "
          f"shape_mismatch={len(load_info.get('skipped_shape_mismatch', []))}")

    model.pipe.dit.eval()

    step_success = 0
    step_failed = 0
    step_skipped = 0
    step_entries: list[dict] = []
    step_log_lines = [
        f"[checkpoint] {ckpt_file}",
        f"[lora_ckpt] {lora_ckpt}",
        f"[stage1a_weights] {stage1a_ckpt}",
        f"[load_info/stage1b] {json.dumps(load_info, ensure_ascii=False)}",
        f"[num_total_latent_frames] {num_total_latent_frames}",
    ]

    for input_json_path in json_paths:
        payload = core._load_input_json(input_json_path)
        try:
            input_video = core._resolve_input_video(payload, input_json_path)
            input_caption = core._ensure_str_field(payload, "input_caption", input_json_path)
        except (KeyError, ValueError) as exc:
            print(f"[skip] {step_label} {input_json_path.stem}: {exc}")
            step_skipped += 1
            continue

        sample_stem = input_json_path.stem
        output_video = step_output_dir / f"{sample_stem}.mp4"
        output_json = step_output_dir / f"{sample_stem}.json"
        output_log = step_output_dir / f"{sample_stem}.log"

        if output_video.exists() and output_json.exists() and not (cli_args.force or cli_args.overwrite):
            print(f"[skip] {step_label} {sample_stem}")
            step_skipped += 1
            continue

        try:
            # context video (for CoTracker / JEPA / pooler)
            ctx_frames, ctx_frame_indices = _load_context_video(
                video_path=Path(input_video),
                target_context_frames=int(cli_args.context_frames),
                sampling_mode=str(cli_args.sampling_mode),
            )
            context_video_single = preprocess_video_rgb_uint8(
                ctx_frames, (int(cli_args.height), int(cli_args.width))
            )
            context_pil = _tensor_video_to_pil_list(context_video_single)

            # build full-length object_context
            with torch.no_grad():
                object_context, object_debug = _build_full_object_context(
                    model=model,
                    context_video_single=context_video_single,
                    num_total_latent_frames=num_total_latent_frames,
                )
            print(f"[object_context] shape={object_debug.get('object_context_shape')} "
                  f"T_ctx={object_debug.get('T_ctx_lat')} T_future={object_debug.get('T_future')}")

            pipe = model.pipe
            pipe.dit.eval()
            with torch.no_grad():
                video = pipe(
                    prompt=input_caption,
                    negative_prompt="",
                    context_video=context_pil,
                    seed=int(cli_args.seed),
                    tiled=True,
                    height=int(cli_args.height),
                    width=int(cli_args.width),
                    num_frames=int(cli_args.num_frames),
                    num_inference_steps=int(cli_args.num_inference_steps),
                    cfg_scale=float(cli_args.cfg_scale),
                    object_context=object_context,
                )

            step_output_dir.mkdir(parents=True, exist_ok=True)
            save_video(video, str(output_video), fps=int(cli_args.fps), quality=int(cli_args.quality))

            result = {
                "input_json": str(input_json_path),
                "input_video": str(input_video),
                "input_caption": str(input_caption),
                "output_video": str(output_video),
                "seed": int(cli_args.seed),
                "step": int(cli_args.num_inference_steps),
                "guidance": float(cli_args.cfg_scale),
                "ckpt": str(ckpt_file),
                "frame_indices": ctx_frame_indices.tolist(),
                "object_debug": object_debug,
                "method": method_name,
            }
            case_logs = [
                f"[case] input_json={input_json_path}",
                f"[case] object_context_shape={object_debug.get('object_context_shape')}",
            ]

        except Exception as exc:
            import traceback
            error_lines = step_log_lines + [f"[error] {sample_stem}: {exc}", traceback.format_exc()]
            core._write_text_lines(output_log, error_lines)
            print(f"[error] {step_label} {sample_stem}: {exc}")
            step_failed += 1
            continue

        success_lines = step_log_lines + case_logs + [f"[done] {step_label} {sample_stem}"]
        core._write_text_lines(output_log, success_lines)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        step_entries.append(result)
        step_success += 1
        print(f"[done] {step_label} {sample_stem}")

    step_summary = {
        "step": step_label,
        "ckpt_file": str(ckpt_file),
        "output_dir": str(step_output_dir),
        "load_info": load_info,
        "num_success": step_success,
        "num_failed": step_failed,
        "num_skipped": step_skipped,
        "num_total_requested": len(json_paths),
        "entries": step_entries,
    }
    with (step_output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(step_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "input_json_list_path": str(input_json_list_path),
        "weights_root": str(weights_root),
        "output_root": str(output_root),
        "run": step_summary,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
