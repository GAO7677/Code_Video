from __future__ import annotations

"""
Stage1B context-only no-GT-box DiffSynth inference.
Mirrors wan_vnewtrain_0613pybullet_stage2_v2v.py in CLI/output conventions,
adapted for checkpoints produced by run_train_stage1b_context_only_no_gt_box_diffsynth_gpu2367.sh.

加载的权重文件及来源：
  [1] Wan DiT base + frozen LoRA（由 YAML config 中 init_wan_lora_from_checkpoint 指定）
        /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/
          raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
  [2] Stage1A pooler/adapter init（--head-resume-from）
        /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/
          pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt
  [3] Stage1B trainable weights（--weights-root，覆盖 [2] 中的可训练参数）
        /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/
          pybullet0629_teacher_student/stage1b_context_only_no_gt_box_diffsynth/
          checkpoints/step-001000/checkpoint.safetensors

========== 冒烟测试（1条样本，20步） ==========
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_stage1b_context_only_no_gt_box_diffsynth_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_context_only_no_gt_box_diffsynth/checkpoints/step-001000 \
  --head-resume-from /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0629_stage1b_context_only_no_gt_box_diffsynth_step001000 \
  --sampling-steps 20 \
  --cfg-scale 5.0 \
  --seed 42 \
  --limit 1 \
  --force

========== 正式推理（test_5.txt 全量，40步） ==========
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_stage1b_context_only_no_gt_box_diffsynth_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_context_only_no_gt_box_diffsynth/checkpoints/step-001000 \
  --head-resume-from /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0629_stage1b_context_only_no_gt_box_diffsynth_step001000 \
  --sampling-steps 40 \
  --cfg-scale 5.0 \
  --seed 42 \
  --force

输出目录：
  /data/gaoya/AAA_test_video/0623/test/v2v/
    pybullet0629_stage1b_context_only_no_gt_box_diffsynth_step001000/
      batch_manifest.json
      summary.json
      step-001000/
        <sample_stem>.mp4
        <sample_stem>.json
        <sample_stem>.log
        result.json
"""

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root
from code_vjepa_vggt.infer_context_video_wan import (
    _build_cond_context,
    _ensure_browser_video,
    _resolve_launch_device,
    _run_sampling,
    _tensor_stats,
    _write_mp4,
)
from code_vjepa_vggt.infer_v_newtrain_context_video_wan import _load_context_video
from code_vjepa_vggt.object_token_teacher_student.runtime_stage1b_context_only_no_gt_box import (
    ContextOnlyInjectionNoGTBoxTrainer,
)
from code_vjepa_vggt.train_stage1b_context_only_diffsynth import _load_matching_state_into_model
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


DEFAULT_CONFIG = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "object_token_teacher_student/config_stage1b_context_only_no_gt_box_template.yaml"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")


def _normalize_ckpt_method_name(name: str) -> str:
    normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
    return normalized or name


def _build_method_name_from_checkpoint_dir(checkpoint_dir: Path) -> str:
    step_name = checkpoint_dir.name
    checkpoint_parent = checkpoint_dir.parent
    if checkpoint_parent.name == "checkpoints" and checkpoint_parent.parent.name:
        method_root = _normalize_ckpt_method_name(checkpoint_parent.parent.name)
        return f"{method_root}_{step_name}"
    if checkpoint_parent.name:
        method_root = _normalize_ckpt_method_name(checkpoint_parent.name)
        return f"{method_root}_{step_name}"
    return step_name


def _build_zero_batch_extras(
    *,
    context_video: torch.Tensor,
    max_objects: int,
    total_frames: int,
) -> dict[str, torch.Tensor]:
    batch_size = int(context_video.shape[0])
    context_frames = int(context_video.shape[2])
    future_frames = max(0, int(total_frames) - int(context_frames))
    device = context_video.device
    dtype = context_video.dtype
    return {
        "future_boxes": torch.zeros((batch_size, future_frames, max_objects, 4), dtype=dtype, device=device),
        "context_states": torch.zeros((batch_size, context_frames, max_objects, 10), dtype=dtype, device=device),
        "future_states": torch.zeros((batch_size, future_frames, max_objects, 10), dtype=dtype, device=device),
        "appearance": torch.zeros((batch_size, max_objects, 16), dtype=dtype, device=device),
        "camera": torch.zeros((batch_size, context_frames, 10), dtype=dtype, device=device),
    }


def _override_config_from_args(config: dict, args: argparse.Namespace) -> None:
    model_cfg = config["model"]
    data_cfg = config["data"]

    if args.wan_root is not None:
        model_cfg["wan_ckpt_dir"] = str(Path(args.wan_root).expanduser().resolve())
    if args.lora_rank is not None:
        model_cfg["wan_lora_rank"] = int(args.lora_rank)
    if args.lora_alpha is not None:
        model_cfg["wan_lora_alpha"] = int(args.lora_alpha)
    if args.object_num_queries is not None:
        model_cfg["object_num_queries"] = int(args.object_num_queries)
    if args.aux_max_objects is not None:
        model_cfg["sam2_max_objects"] = int(args.aux_max_objects)
    if args.jepa_ckpt_path is not None:
        p = Path(args.jepa_ckpt_path).expanduser().resolve()
        if p.name == "model.pth" and p.parent.name == "original":
            model_cfg["je_pa_ckpt_dir"] = str(p.parent.parent)
        else:
            model_cfg["je_pa_ckpt_dir"] = str(p)
    if args.jepa_input_size is not None:
        model_cfg["jepa_input_size"] = int(args.jepa_input_size)
    if args.jepa_patch_size is not None:
        model_cfg["jepa_patch_size"] = int(args.jepa_patch_size)
    if args.jepa_tubelet_size is not None:
        model_cfg["jepa_tubelet_size"] = int(args.jepa_tubelet_size)
    if args.cotracker_checkpoint is not None:
        model_cfg["cotracker_checkpoint"] = str(Path(args.cotracker_checkpoint).expanduser().resolve())
    if args.cotracker_input_h is not None and args.cotracker_input_w is not None:
        model_cfg["cotracker_input_hw"] = [int(args.cotracker_input_h), int(args.cotracker_input_w)]
    if args.cotracker_window_len is not None:
        model_cfg["cotracker_window_len"] = int(args.cotracker_window_len)
    if args.cond_proj_dim is not None:
        model_cfg["cond_proj_dim"] = int(args.cond_proj_dim)

    # Override data config for inference geometry
    data_cfg["resolution"] = [int(args.height), int(args.width)]
    data_cfg["num_context_frames"] = int(args.context_frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run Stage1B context-only no-GT-box DiffSynth checkpoints over an input json list. "
            "Matches CLI conventions of wan_vnewtrain_0613pybullet_stage2_v2v.py."
        )
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        required=True,
        help="step-* dir containing checkpoint.safetensors (stage1B trainables)",
    )
    parser.add_argument(
        "--head-resume-from",
        type=Path,
        required=True,
        help="Stage1A init checkpoint (.pt) loaded before the stage1B weights",
    )
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0,
                        help="Classifier-free guidance scale (1.0 = no CFG)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument(
        "--jepa-ckpt-path",
        default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth",
    )
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument(
        "--cotracker-checkpoint",
        default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
    )
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_root = args.weights_root.expanduser().resolve()
    head_resume_from = args.head_resume_from.expanduser().resolve()
    input_json_list_path = args.input_json_list_path.expanduser().resolve()
    model_name = str(args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=args.output_root,
        base_output_root="/data/gaoya/AAA_test_video/0623/test/v2v",
        model_name=model_name,
    )

    config = load_yaml_config(args.config.expanduser().resolve())
    _override_config_from_args(config, args)

    device = args.device if args.device is not None else _resolve_launch_device()
    config["model"]["grounding_device"] = str(device)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    trainer = ContextOnlyInjectionNoGTBoxTrainer(config, build_optimizer=False, device=device)

    head_load_info = _load_matching_state_into_model(trainer, head_resume_from)
    print(
        f"[head_resume_from] loaded_count={head_load_info['loaded_count']} "
        f"shape_mismatch={len(head_load_info['skipped_shape_mismatch'])}"
    )

    stage1b_load_info = _load_matching_state_into_model(trainer, weights_root)
    print(
        f"[weights_root] loaded_count={stage1b_load_info['loaded_count']} "
        f"shape_mismatch={len(stage1b_load_info['skipped_shape_mismatch'])}"
    )

    if trainer.bundle.dit is not None:
        trainer.bundle.dit.eval()

    # Pre-encode null text for CFG (done once, reused for all samples)
    cfg_scale = float(args.cfg_scale)
    negative_text_context = None
    if cfg_scale > 1.0:
        with torch.no_grad():
            neg_ctx_list = trainer.bundle.text_encoder(
                [""], trainer.bundle.text_encoder.device
            )
        negative_text_context = neg_ctx_list[0].to(trainer.device_obj)
        print(f"[cfg] scale={cfg_scale}, negative_text_context shape={list(negative_text_context.shape)}")

    json_paths = core._read_list_file(input_json_list_path)
    if args.limit is not None:
        json_paths = json_paths[: max(0, int(args.limit))]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_json_list_path": str(input_json_list_path),
        "weights_root": str(weights_root),
        "head_resume_from": str(head_resume_from),
        "num_items": len(json_paths),
        "num_inference_steps": int(args.sampling_steps),
        "seed": int(args.seed),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "context_frames": int(args.context_frames),
        "sampling_mode": str(args.sampling_mode),
        "cfg_scale": float(cfg_scale),
    }
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    method_name = _build_method_name_from_checkpoint_dir(weights_root)
    step_output_dir = output_root / weights_root.name
    step_output_dir.mkdir(parents=True, exist_ok=True)

    step_success = 0
    step_failed = 0
    step_skipped = 0
    step_entries: list[dict[str, object]] = []
    step_log_lines = [
        f"[checkpoint] {weights_root}",
        f"[head_resume_from] {head_resume_from}",
        f"[stage1b_load_info] {json.dumps(stage1b_load_info, ensure_ascii=False)}",
    ]

    context_fraction = float(config["data"].get("context_fraction", 0.5))
    total_frames = int(round(int(args.context_frames) / context_fraction))
    max_objects = int(getattr(trainer, "max_objects", config["model"].get("sam2_max_objects", 4)))

    for input_json_path in json_paths:
        payload = core._load_input_json(input_json_path)
        try:
            input_video = core._resolve_input_video(payload, input_json_path)
            input_caption = core._ensure_str_field(payload, "input_caption", input_json_path)
        except (KeyError, ValueError) as exc:
            print(f"[skip] {weights_root.name} {input_json_path.stem}: {exc}")
            step_skipped += 1
            continue

        sample_stem = input_json_path.stem
        output_video = step_output_dir / f"{sample_stem}.mp4"
        output_json = step_output_dir / f"{sample_stem}.json"
        output_log = step_output_dir / f"{sample_stem}.log"

        if output_video.exists() and output_json.exists() and not (args.force or args.overwrite):
            print(f"[skip] {weights_root.name} {sample_stem}")
            step_skipped += 1
            continue

        try:
            frames, frame_indices = _load_context_video(
                video_path=Path(input_video),
                target_context_frames=int(args.context_frames),
                sampling_mode=str(args.sampling_mode),
            )
            context_video_single = preprocess_video_rgb_uint8(frames, (int(args.height), int(args.width)))
            context_video = context_video_single.unsqueeze(0).to(trainer.device_obj)
            num_context_frames = torch.tensor(
                [context_video.shape[2]], dtype=torch.long, device=trainer.device_obj
            )

            batch_extra_tensors = _build_zero_batch_extras(
                context_video=context_video,
                max_objects=max_objects,
                total_frames=total_frames,
            )

            text_context, object_context, context_latents, prep_debug = _build_cond_context(
                trainer=trainer,
                config=config,
                context_video=context_video,
                captions=[input_caption],
                num_context_frames=num_context_frames,
                device_obj=trainer.device_obj,
                batch_extra_tensors=batch_extra_tensors,
            )

            with torch.inference_mode():
                pred, sample_debug = _run_sampling(
                    bundle=trainer.bundle,
                    text_context=text_context,
                    object_context=object_context,
                    context_latents=context_latents,
                    total_frames=total_frames,
                    num_context_frames=int(num_context_frames.item()),
                    num_inference_steps=int(args.sampling_steps),
                    cfg_scale=cfg_scale,
                    negative_text_context=negative_text_context,
                )

            with torch.no_grad():
                vae_device = (
                    next(trainer.bundle.vae.model.parameters()).device
                    if hasattr(trainer.bundle.vae, "model")
                    else trainer.device_obj
                )
                decoded = trainer.bundle.vae.decode([pred.to(vae_device)])
            if isinstance(decoded, list):
                decoded = decoded[0]
            video_out = decoded.detach().cpu()
            video_out = video_out.permute(1, 0, 2, 3).contiguous()
            video_out = (
                (video_out.clamp(-1.0, 1.0) + 1.0) * 127.5
            ).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            _write_mp4(output_video, video_out, fps=int(args.fps))
            _ensure_browser_video(output_video)

        except Exception as exc:
            core._write_text_lines(output_log, step_log_lines + [f"[error] {sample_stem}: {exc}"])
            print(f"[error] {weights_root.name} {sample_stem}: {exc}")
            step_failed += 1
            continue

        result = {
            "method": method_name,
            "input_json": str(input_json_path),
            "input_video": str(input_video),
            "input_caption": str(input_caption),
            "output_video": str(output_video),
            "weights_root": str(weights_root),
            "head_resume_from": str(head_resume_from),
            "seed": int(args.seed),
            "sampling_steps": int(args.sampling_steps),
            "frame_indices": frame_indices.tolist(),
            "prep_debug": prep_debug,
            "sample_debug": sample_debug,
            "trainable_tensor_stats": {
                "text_context": _tensor_stats("text_context", text_context),
                "object_context": _tensor_stats("object_context", object_context),
                "context_latents": _tensor_stats("context_latents", context_latents),
            },
        }

        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        core._write_text_lines(
            output_log,
            step_log_lines + [f"[done] {weights_root.name} {sample_stem}"],
        )
        step_entries.append(result)
        step_success += 1
        print(f"[done] {weights_root.name} {sample_stem}")

    step_summary = {
        "step": weights_root.name,
        "checkpoint_dir": str(weights_root),
        "output_dir": str(step_output_dir),
        "stage1b_load_info": stage1b_load_info,
        "num_success": step_success,
        "num_failed": step_failed,
        "num_skipped": step_skipped,
        "num_total_requested": len(json_paths),
        "entries": step_entries,
    }
    with (step_output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(step_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"input_json_list_path": str(input_json_list_path), "weights_root": str(weights_root), "output_root": str(output_root), "run": step_summary}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(output_root / "summary.json")

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
