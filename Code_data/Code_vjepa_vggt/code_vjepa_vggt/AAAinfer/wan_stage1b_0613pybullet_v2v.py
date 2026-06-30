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
from code_vjepa_vggt.infer_v_newtrain_context_video_wan import _load_v_newtrain_state_into_model
from code_vjepa_vggt.utils.config import load_yaml_config

DEFAULT_LORA_CKPT = Path(
    "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
    "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
)

"""
Stage1B inference: loads a teacher-student stage1b .pt checkpoint (trainer format,
contains 'model' key with object_cross_attn / norm4 / object_gate / object_embedding
/ object_adapter weights) and runs video generation the same way as the stage2 script.

Weight loading order:
  1. LoRA weights  (--lora-ckpt)         → DiT q/k/v/o/ffn lora_A/lora_B
  2. object_pooler (--stage1a-weights)   → pooler 63 tensors from stage1a .pt
  3. stage1b ckpt  (--weights-root)      → object_cross_attn / norm4 / object_gate /
                                           object_embedding (overwrites any overlap)

完整运行命令：

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=0 \
python3 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_stage1b_0613pybullet_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_oracle_cross_attn/step_0000500.pt \
  --stage1a-weights /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --lora-ckpt /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0629_stage1b_cross

测试单条（--limit 1）：

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=5 \
python3 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_stage1b_0613pybullet_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_oracle_cross_attn/step_0000500.pt \
  --stage1a-weights /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0629_stage1b_s500 \
  --limit 1

--lora-ckpt 有默认值（DEFAULT_LORA_CKPT），与 stage1b 训练时一致，可省略。
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
    """Accept either a .pt file directly or a directory; in the latter case pick the latest step_*.pt."""
    if weights_root.is_file():
        return weights_root
    if weights_root.is_dir():
        candidates = sorted(weights_root.glob("step_*.pt"))
        if candidates:
            return candidates[-1]
        # fall back to safetensors for compatibility
        sft = weights_root / "checkpoint.safetensors"
        if sft.is_file():
            return weights_root  # let _load_v_newtrain_state handle directory
    raise FileNotFoundError(f"no valid checkpoint found at: {weights_root}")


def _build_method_name(weights_root: Path) -> str:
    ckpt_file = _resolve_stage1b_ckpt(weights_root)
    step_name = ckpt_file.stem  # e.g. "step_0000500"
    parent = ckpt_file.parent
    if parent.name:
        method_root = _normalize_ckpt_method_name(parent.name)
        return f"{method_root}_{step_name}"
    return step_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run a Stage1B teacher-student checkpoint over an input json list. "
            "Accepts trainer-format .pt files (containing 'model' key). "
            "Thin wrapper around batch_infer_v_newtrain_from_jsonl."
        )
    )
    parser.add_argument(
        "--weights-root", type=Path, required=True,
        help="path to a step_*.pt file OR a directory containing step_*.pt files (latest is picked)",
    )
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
    # --- stage1b specific ---
    parser.add_argument(
        "--lora-ckpt", type=Path, default=DEFAULT_LORA_CKPT,
        help="safetensors file with LoRA weights (frozen during stage1b training). "
             "Defaults to the lora ckpt used during stage1b training.",
    )
    parser.add_argument(
        "--stage1a-weights", type=Path, default=None, required=True,
        help=".pt file from stage1a training containing object_pooler weights.",
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
    # build a mini-parser so _apply_config_defaults can compare against defaults
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

    # resolve the actual .pt file to load and derive a step label
    ckpt_file = _resolve_stage1b_ckpt(weights_root)
    step_label = ckpt_file.stem  # e.g. "step_0000500"
    method_name = _build_method_name(weights_root)

    # resolve extra weight paths early so manifest can reference them
    lora_ckpt = cli_args.lora_ckpt.expanduser().resolve()
    stage1a_ckpt = cli_args.stage1a_weights.expanduser().resolve()

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
        "sampling_mode": str(cli_args.sampling_mode),
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

    # Step 1: load LoRA weights (frozen during stage1b training)
    # Use _load_v_newtrain_state_into_model — same function used by stage2 — which handles
    # safetensors format and normalizes key prefixes to match lora_A/lora_B in the model.
    if not lora_ckpt.is_file():
        raise FileNotFoundError(f"--lora-ckpt not found: {lora_ckpt}")
    lora_load_info = _load_v_newtrain_state_into_model(model, lora_ckpt)
    print(f"[lora] loaded from {lora_ckpt.name}: "
          f"loaded={lora_load_info['loaded_count']} "
          f"missing={len(lora_load_info['missing_keys'])} "
          f"shape_mismatch={len(lora_load_info.get('skipped_shape_mismatch', []))}")

    # Step 2: load stage1a object_pooler weights
    if not stage1a_ckpt.is_file():
        raise FileNotFoundError(f"--stage1a-weights not found: {stage1a_ckpt}")
    pooler_load_info = _load_v_newtrain_state_into_model(model, stage1a_ckpt)
    print(f"[stage1a pooler] loaded={pooler_load_info['loaded_count']} "
          f"missing={len(pooler_load_info['missing_keys'])} "
          f"shape_mismatch={len(pooler_load_info.get('skipped_shape_mismatch', []))}")

    # Step 3: load stage1b weights (object_cross_attn / norm4 / object_gate / object_embedding)
    # This overwrites any overlapping keys from the previous loads.
    load_info = _load_v_newtrain_state_into_model(model, ckpt_file)
    print(f"[stage1b] loaded={load_info['loaded_count']} "
          f"missing={len(load_info['missing_keys'])} "
          f"unexpected={len(load_info['unexpected_keys'])} "
          f"shape_mismatch={len(load_info.get('skipped_shape_mismatch', []))}")

    model.pipe.dit.eval()

    step_success = 0
    step_failed = 0
    step_skipped = 0
    step_entries: list[dict[str, object]] = []
    step_log_lines = [
        f"[checkpoint] {ckpt_file}",
        f"[lora_ckpt] {lora_ckpt}",
        f"[stage1a_weights] {stage1a_ckpt}",
        f"[load_info/stage1b] {json.dumps(load_info, ensure_ascii=False)}",
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
            result, case_logs = core._run_single_case_in_process(
                model=model,
                checkpoint_dir=ckpt_file.parent,
                input_json_path=input_json_path,
                input_video=input_video,
                input_caption=input_caption,
                output_dir=step_output_dir,
                output_video=output_video,
                num_frames=int(cli_args.num_frames),
                context_frames=int(cli_args.context_frames),
                sampling_mode=str(cli_args.sampling_mode),
                sampling_steps=int(cli_args.num_inference_steps),
                fps=int(cli_args.fps),
                seed=int(cli_args.seed),
                cfg_scale=float(cli_args.cfg_scale),
                height=int(cli_args.height),
                width=int(cli_args.width),
                quality=int(cli_args.quality),
            )
        except Exception as exc:
            error_lines = step_log_lines + [f"[error] {sample_stem}: {exc}"]
            core._write_text_lines(output_log, error_lines)
            print(f"[error] {step_label} {sample_stem}: {exc}")
            step_failed += 1
            continue

        success_lines = step_log_lines + case_logs + [f"[done] {step_label} {sample_stem}"]
        core._write_text_lines(output_log, success_lines)
        result["method"] = method_name
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
