"""
診断用スクリプト: stage1b diffsynth 権重 → WanTrainingModule.pipe() で推論
目的: 노이즈가 권중 문제인지 _run_sampling 경로 문제인지 구분한다.

stage1b diffsynth checkpoint에는 LoRA / object_pooler가 없으므로
별도 소스에서 로드한다:
  1) LoRA  : /data/gaoya/AAA_test_video/0529/.../step-000500/checkpoint.safetensors
  2) object_pooler : stage1a checkpoint (step_0005000.pt)
  3) object-branch (blocks.*.object_cross_attn, norm4, object_gate):
     stage1b diffsynth checkpoint (step-001000/checkpoint.safetensors)
     bundle.dit.base_model.model.* → base_model.model.* 로 prefix 변환 후 로드

Smoke test (1 sample, 20 steps, GPU 5):
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_stage1b_context_only_no_gt_box_diffsynth_via_pipe_v2v.py \
  --stage1b-ckpt /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_context_only_no_gt_box_diffsynth/checkpoints/step-001000 \
  --stage1a-ckpt /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --lora-ckpt /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0629_stage1b_no_gt_box_via_pipe_step001000 \
  --sampling-steps 20 --cfg-scale 5.0 --seed 42 --limit 1 --force

Formal run (full test_5.txt, 40 steps, GPU 7):
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_stage1b_context_only_no_gt_box_diffsynth_via_pipe_v2v.py \
  --stage1b-ckpt /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_context_only_no_gt_box_diffsynth/checkpoints/step-001000 \
  --stage1a-ckpt /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --lora-ckpt /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name pybullet0629_stage1b_no_gt_box_via_pipe_step001000 \
  --sampling-steps 40 --cfg-scale 5.0 --seed 42 --force
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root
from code_vjepa_vggt.utils.config import load_yaml_config

DEFAULT_CONFIG = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/"
    "train_0624pybullet_freeze_lora_other_modules_gpu67.yaml"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")


# ---------------------------------------------------------------------------
# Custom loader: stage1b diffsynth checkpoint → WanTrainingModule
# ---------------------------------------------------------------------------

def _normalize_key(key: str) -> str:
    """Strip all well-known prefixes, including bundle.dit., to get bare model keys."""
    normalized = str(key)
    prefixes = (
        "module.",
        "bundle.dit.",   # stage1b diffsynth trainer saves under bundle.dit.*
        "pipe.dit.",
        "base_model.model.",
        "dit.base_model.model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                changed = True
    return normalized


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    resolved = path
    if resolved.is_dir():
        candidate = resolved / "checkpoint.safetensors"
        if candidate.is_file():
            resolved = candidate
        else:
            candidates = sorted(resolved.rglob("checkpoint.safetensors"))
            if candidates:
                resolved = candidates[-1]
            else:
                raise FileNotFoundError(f"No checkpoint.safetensors under {path}")
    return load_safetensors_file(str(resolved), device="cpu")


def _load_pt(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _load_stage1b_into_model(
    model,
    *,
    stage1b_ckpt: Path,
    stage1a_ckpt: Path,
    lora_ckpt: Path,
) -> dict[str, object]:
    """Load weights into WanTrainingModule from three sources.

    Load order (later overwrites earlier for overlapping keys):
      1. LoRA weights (from step-000500 checkpoint)
      2. object_pooler weights (from stage1a checkpoint)
      3. object-branch DiT weights (from stage1b diffsynth checkpoint)
    """
    model_state = model.state_dict()
    norm_to_model = {_normalize_key(k): k for k in model_state.keys()}

    merged: dict[str, torch.Tensor] = {}
    skipped_shape: list[dict] = []
    load_counts: dict[str, int] = {}

    def _apply(source_state: dict[str, torch.Tensor], source_name: str) -> int:
        norm_to_ckpt = {_normalize_key(k): k for k in source_state.keys()}
        overlap = set(norm_to_model.keys()) & set(norm_to_ckpt.keys())
        count = 0
        for nk in overlap:
            mk = norm_to_model[nk]
            ck = norm_to_ckpt[nk]
            mv = model_state[mk]
            cv = source_state[ck]
            if tuple(mv.shape) != tuple(cv.shape):
                skipped_shape.append({
                    "source": source_name,
                    "model_key": mk,
                    "ckpt_key": ck,
                    "model_shape": list(mv.shape),
                    "ckpt_shape": list(cv.shape),
                })
                continue
            merged[mk] = cv
            count += 1
        return count

    # 1) LoRA
    lora_state = _load_safetensors(lora_ckpt)
    load_counts["lora"] = _apply(lora_state, "lora")

    # 2) object_pooler (from stage1a .pt)
    pooler_state = _load_pt(stage1a_ckpt)
    load_counts["stage1a_pooler"] = _apply(pooler_state, "stage1a_pooler")

    # 3) object-branch DiT weights (from stage1b diffsynth checkpoint)
    s1b_state = _load_safetensors(stage1b_ckpt)
    load_counts["stage1b_object_branch"] = _apply(s1b_state, "stage1b_object_branch")

    # also load object_adapter keys from stage1b checkpoint (no bundle.dit. prefix)
    load_counts["stage1b_object_branch"] += _apply(
        {k: v for k, v in s1b_state.items() if k.startswith("object_adapter.")},
        "stage1b_object_adapter",
    )

    if not merged:
        raise RuntimeError("No keys loaded — check checkpoint formats")

    missing = model.load_state_dict(merged, strict=False)
    return {
        "load_counts": load_counts,
        "total_loaded": len(merged),
        "missing_keys_count": len(missing.missing_keys),
        "unexpected_keys_count": len(missing.unexpected_keys),
        "skipped_shape_mismatch": skipped_shape,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _normalize_ckpt_method_name(name: str) -> str:
    normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
    return normalized or name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test stage1b diffsynth weights via WanTrainingModule.pipe() "
            "to distinguish weight problems from sampling-path problems."
        )
    )
    parser.add_argument(
        "--stage1b-ckpt", type=Path, required=True,
        help="step-* dir containing stage1b diffsynth checkpoint.safetensors",
    )
    parser.add_argument(
        "--stage1a-ckpt", type=Path, required=True,
        help="stage1a .pt checkpoint (provides object_pooler weights)",
    )
    parser.add_argument(
        "--lora-ckpt", type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
            "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/"
            "step-000500/checkpoint.safetensors"
        ),
        help="LoRA checkpoint (provides LoRA weights)",
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
    parser.add_argument("--sampling-steps", type=int, default=40)
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
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    stage1b_ckpt = cli_args.stage1b_ckpt.expanduser().resolve()
    stage1a_ckpt = cli_args.stage1a_ckpt.expanduser().resolve()
    lora_ckpt = cli_args.lora_ckpt.expanduser().resolve()
    input_json_list_path = cli_args.input_json_list_path.expanduser().resolve()
    model_name = str(cli_args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root="/data/gaoya/AAA_test_video/0623/test/v2v",
        model_name=model_name,
    )

    config_path = cli_args.config.expanduser().resolve()
    config = load_yaml_config(config_path)
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--height", type=int, default=512)
    _parser.add_argument("--width", type=int, default=896)
    _parser.add_argument("--num-frames", type=int, default=24)
    _parser.add_argument("--fps", type=int, default=30)
    _parser.add_argument("--context-frames", type=int, default=8)
    _parser.add_argument("--wan-root", default=str(DEFAULT_WAN_ROOT))
    _parser.add_argument("--lora-rank", type=int, default=32)
    _parser.add_argument("--object-num-queries", type=int, default=8)
    _parser.add_argument("--aux-max-objects", type=int, default=4)
    _parser.add_argument("--jepa-ckpt-path", default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth")
    _parser.add_argument("--jepa-input-size", type=int, default=384)
    _parser.add_argument("--jepa-patch-size", type=int, default=16)
    _parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    _parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    _parser.add_argument("--cotracker-input-h", type=int, default=384)
    _parser.add_argument("--cotracker-input-w", type=int, default=512)
    _parser.add_argument("--cotracker_window_len", type=int, default=60)
    _parser.add_argument("--cotracker-window-len", type=int, default=60)
    _parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    _parser.add_argument("--cond-proj-dim", type=int, default=4096)
    _parser.add_argument("--jepa-window-radius", type=int, default=1)
    _parser.add_argument("--latent-window-radius", type=int, default=1)
    core._apply_config_defaults(cli_args, _parser, config)

    # Override num-inference-steps from --sampling-steps
    cli_args.num_inference_steps = cli_args.sampling_steps

    cli_args.device = core._resolve_launch_device()

    torch.manual_seed(int(cli_args.seed))
    np.random.seed(int(cli_args.seed))

    json_paths = core._read_list_file(input_json_list_path)
    if cli_args.limit is not None:
        json_paths = json_paths[: max(0, int(cli_args.limit))]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_json_list_path": str(input_json_list_path),
        "stage1b_ckpt": str(stage1b_ckpt),
        "stage1a_ckpt": str(stage1a_ckpt),
        "lora_ckpt": str(lora_ckpt),
        "inference_path": "WanTrainingModule.pipe()",
        "num_items": len(json_paths),
        "num_inference_steps": int(cli_args.sampling_steps),
        "cfg_scale": float(cli_args.cfg_scale),
        "seed": int(cli_args.seed),
    }
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    model_args = core._build_model_args(cli_args)

    if not stage1b_ckpt.exists():
        raise FileNotFoundError(f"stage1b-ckpt not found: {stage1b_ckpt}")
    if not stage1a_ckpt.exists():
        raise FileNotFoundError(f"stage1a-ckpt not found: {stage1a_ckpt}")
    if not lora_ckpt.exists():
        raise FileNotFoundError(f"lora-ckpt not found: {lora_ckpt}")

    step_output_dir = output_root / stage1b_ckpt.name
    step_output_dir.mkdir(parents=True, exist_ok=True)

    model = core.build_model(model_args)
    model.to(torch.device(cli_args.device))
    model.eval()

    load_info = _load_stage1b_into_model(
        model,
        stage1b_ckpt=stage1b_ckpt,
        stage1a_ckpt=stage1a_ckpt,
        lora_ckpt=lora_ckpt,
    )
    print(f"[load_info] {json.dumps(load_info, ensure_ascii=False)}")
    model.pipe.dit.eval()

    step_success = 0
    step_failed = 0
    step_skipped = 0
    step_entries: list[dict[str, object]] = []
    step_log_lines = [
        f"[checkpoint] stage1b={stage1b_ckpt}",
        f"[load_info] {json.dumps(load_info, ensure_ascii=False)}",
    ]

    for input_json_path in json_paths:
        payload = core._load_input_json(input_json_path)
        try:
            input_video = core._resolve_input_video(payload, input_json_path)
            input_caption = core._ensure_str_field(payload, "input_caption", input_json_path)
        except (KeyError, ValueError) as exc:
            print(f"[skip] {stage1b_ckpt.name} {input_json_path.stem}: {exc}")
            step_skipped += 1
            continue

        sample_stem = input_json_path.stem
        output_video = step_output_dir / f"{sample_stem}.mp4"
        output_json = step_output_dir / f"{sample_stem}.json"
        output_log = step_output_dir / f"{sample_stem}.log"

        if output_video.exists() and output_json.exists() and not (cli_args.force or cli_args.overwrite):
            print(f"[skip] {stage1b_ckpt.name} {sample_stem}")
            step_skipped += 1
            continue

        try:
            result, case_logs = core._run_single_case_in_process(
                model=model,
                checkpoint_dir=stage1b_ckpt,
                input_json_path=input_json_path,
                input_video=input_video,
                input_caption=input_caption,
                output_dir=step_output_dir,
                output_video=output_video,
                num_frames=int(cli_args.num_frames),
                context_frames=int(cli_args.context_frames),
                sampling_mode=str(cli_args.sampling_mode),
                sampling_steps=int(cli_args.sampling_steps),
                fps=int(cli_args.fps),
                seed=int(cli_args.seed),
                cfg_scale=float(cli_args.cfg_scale),
                height=int(cli_args.height),
                width=int(cli_args.width),
                quality=int(cli_args.quality),
            )
        except Exception as exc:
            import traceback
            error_lines = step_log_lines + [
                f"[error] {sample_stem}: {exc}",
                traceback.format_exc(),
            ]
            core._write_text_lines(output_log, error_lines)
            print(f"[error] {stage1b_ckpt.name} {sample_stem}: {exc}")
            step_failed += 1
            continue

        success_lines = step_log_lines + case_logs + [f"[done] {stage1b_ckpt.name} {sample_stem}"]
        core._write_text_lines(output_log, success_lines)
        result["method"] = model_name
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        step_entries.append(result)
        step_success += 1
        print(f"[done] {stage1b_ckpt.name} {sample_stem}")

    step_summary = {
        "stage1b_ckpt": str(stage1b_ckpt),
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

    summary = {"output_root": str(output_root), "run": step_summary}
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
