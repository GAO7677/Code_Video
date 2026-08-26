#!/usr/bin/env python3
"""Run the latent-mask P0 model with the PhysRVG-72f output path.

This is intentionally a separate runner.  It keeps the latent-mask model and
the shared P0 inference settings, but exports the 189-frame sample and the
prediction-only 120-frame segment from the same in-memory sample.  The
prediction segment is therefore encoded with the same direct
``export_to_video(..., macro_block_size=1)`` path used by PhysRVG-72f-adapted.

The final submission directory is MP4-only.  Per-case metadata and the
manifest are written outside that directory so the official evaluator can
consume it without a cleanup step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import set_seed
from diffusers.utils import export_to_video


REPO_ROOT = Path("/home/gaoya/code_V2V_baselines/PhysRVG-main")
WORKSPACE = Path("/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified")
DEFAULT_INPUT_LIST = WORKSPACE / "inputs/bpp/verified_v2v_bpp_198.txt"
DEFAULT_MODEL_ID = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers")
DEFAULT_DIT = Path(
    "/data/gaoya/agent-data/weights/physrvg-diffusers-d8caf2/dit/"
    "diffusion_pytorch_model.safetensors"
)
DEFAULT_LORA = Path(
    "/data/gaoya/agent-data/checkpoints/physrvg_full_sa_latent_mask/"
    "full-sa-pybullet-physrvg-latent-mask-b2-gacc2-20260818T052732Z/"
    "checkpoints/step-001000"
)
DEFAULT_RUN_NAME = "physrvg-full-sa-latent-mask-step001000-bpp-run_01-72f-aligned"
DEFAULT_RAW_ROOT = WORKSPACE / "raw" / DEFAULT_RUN_NAME
DEFAULT_SUBMISSION_ROOT = WORKSPACE / "generated_videos_5s" / DEFAULT_RUN_NAME
FFPROBE = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffprobe")

EXPECTED_CASES = 198
EXPECTED_CONDITION_FRAMES = 72
EXPECTED_CONDITION_FPS = 24
EXPECTED_RAW_FRAMES = 189
EXPECTED_PREFIX_FRAMES = 69
EXPECTED_SUBMISSION_FRAMES = 120
EXPECTED_FPS = 24
EXPECTED_HEIGHT = 512
EXPECTED_WIDTH = 896
EXPECTED_STEPS = 40
EXPECTED_GUIDANCE = 5.0
EXPECTED_SEED = 42
EXPECTED_NEGATIVE_PROMPT_VERSION = "physrvg-72f-adapted-long-v1"
EXPECTED_NEGATIVE_PROMPT_SHA256 = (
    "ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Full-SA latent-mask with direct in-memory 189->120 frame "
            "encoding aligned to PhysRVG-72f-adapted."
        )
    )
    parser.add_argument("--input-json-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--raw-output-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--submission-output-root",
        type=Path,
        default=DEFAULT_SUBMISSION_ROOT,
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-id", type=Path, default=DEFAULT_MODEL_ID)
    parser.add_argument("--physrvg-dit-checkpoint", type=Path, default=DEFAULT_DIT)
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=EXPECTED_HEIGHT)
    parser.add_argument("--width", type=int, default=EXPECTED_WIDTH)
    parser.add_argument("--condition-frames", type=int, default=EXPECTED_CONDITION_FRAMES)
    parser.add_argument("--condition-fps", type=int, default=EXPECTED_CONDITION_FPS)
    parser.add_argument("--num-frames", type=int, default=EXPECTED_RAW_FRAMES)
    parser.add_argument("--fps", type=int, default=EXPECTED_FPS)
    parser.add_argument("--num-inference-steps", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=EXPECTED_GUIDANCE)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument(
        "--context-mask-mode",
        choices=("dynamic_effective",),
        default="dynamic_effective",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def reject_gpu4(device: str) -> None:
    if device.strip().lower() in {"cuda:4", "cuda:4/"}:
        raise ValueError("GPU 4 is prohibited by the workspace rules")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if any(token.strip() == "4" for token in visible.split(",") if token.strip()):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES includes physical GPU 4; choose another GPU"
        )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def probe_video(path: Path) -> dict[str, Any]:
    payload = json.loads(
        subprocess.check_output(
            [
                str(FFPROBE),
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "frames": int(stream["nb_read_frames"]),
        "fps": float(numerator) / float(denominator),
        "duration": float(payload["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def validate_video(path: Path, expected_frames: int) -> dict[str, Any]:
    info = probe_video(path)
    if info["frames"] != expected_frames:
        raise RuntimeError(
            f"{path} has {info['frames']} frames; expected {expected_frames}"
        )
    if abs(float(info["fps"]) - EXPECTED_FPS) > 1e-6:
        raise RuntimeError(f"{path} has {info['fps']} FPS; expected {EXPECTED_FPS}")
    if info["width"] != EXPECTED_WIDTH or info["height"] != EXPECTED_HEIGHT:
        raise RuntimeError(
            f"{path} has {info['width']}x{info['height']}; "
            f"expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
        )
    expected_duration = expected_frames / EXPECTED_FPS
    if abs(float(info["duration"]) - expected_duration) > 0.001:
        raise RuntimeError(
            f"{path} duration is {info['duration']}; expected {expected_duration}"
        )
    return info


def resolve_case_path(input_list: Path, declared: str) -> Path:
    path = Path(declared).expanduser()
    if path.is_file():
        return path.resolve()
    candidates = (input_list.parent / path, input_list.parent / "jsons" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"input JSON not found: {declared}")


def load_cases(
    input_list: Path,
    limit: int | None,
    shard_index: int,
    shard_count: int,
) -> tuple[list[tuple[int, Path, dict[str, Any]]], list[str]]:
    declared = [
        line.strip()
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(declared) != EXPECTED_CASES:
        raise ValueError(
            f"strict P0 input list must contain {EXPECTED_CASES} cases, found {len(declared)}"
        )
    all_names: list[str] = []
    all_cases: list[tuple[int, Path, dict[str, Any]]] = []
    for index, declared_path in enumerate(declared, start=1):
        case_path = resolve_case_path(input_list, declared_path)
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        if payload.get("prompt_setting") != "bpp" or payload.get("input_mode") != "v2v":
            raise ValueError(f"case is not BPP V2V: {case_path}")
        if payload.get("conditioning_frames") != EXPECTED_CONDITION_FRAMES:
            raise ValueError(f"case does not declare 72 condition frames: {case_path}")
        if float(payload.get("conditioning_fps", -1)) != EXPECTED_CONDITION_FPS:
            raise ValueError(f"case does not declare 24 FPS: {case_path}")
        generated_name = str(payload.get("generated_video_name", ""))
        if not generated_name.startswith(f"{index:04d}_") or not generated_name.endswith(
            ".mp4"
        ):
            raise ValueError(f"official output name/order mismatch: {case_path}")
        if generated_name in all_names:
            raise ValueError(f"duplicate generated video name: {generated_name}")
        source_value = payload.get("input_video") or payload.get("source_video")
        if not isinstance(source_value, str) or not source_value.strip():
            raise ValueError(f"missing input/source video: {case_path}")
        source_path = Path(source_value).expanduser()
        if not source_path.is_file():
            fallback = case_path.parent.parent / "conditioning" / "24FPS" / source_path.name
            source_path = fallback
        if not source_path.is_file():
            raise FileNotFoundError(f"conditioning video not found: {source_value}")
        if not isinstance(payload.get("input_caption"), str) or not payload["input_caption"].strip():
            raise ValueError(f"missing input_caption: {case_path}")
        payload = dict(payload)
        payload["input_video"] = str(source_path.resolve())
        all_names.append(generated_name)
        all_cases.append((index, case_path, payload))

    if limit is not None:
        if not 1 <= limit <= EXPECTED_CASES:
            raise ValueError(f"--limit must be between 1 and {EXPECTED_CASES}")
        all_cases = all_cases[:limit]
    selected = all_cases[shard_index::shard_count]
    return selected, all_names


def import_inference_module(repo_root: Path):
    repo_root = require_dir(repo_root, "PhysRVG repository")
    inference_dir = require_dir(repo_root / "scripts_mytrain" / "inference", "inference directory")
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(inference_dir))
    import infer_full_sa_lora_json_list as inference

    return inference


def build_loader_args(args: argparse.Namespace) -> Namespace:
    # The target P0 run does not enable the optional object-XSSC branch.  These
    # are the only attributes used by the shared model loader in that mode.
    return Namespace(
        model_id=args.model_id,
        physrvg_dit_checkpoint=args.physrvg_dit_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        device=args.device,
        object_xssc_trainable=None,
    )


def export_frames(frames: list[Any], target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp.mp4")
    export_to_video(
        frames,
        str(temporary),
        fps=EXPECTED_FPS,
        macro_block_size=1,
    )
    temporary.replace(target)
    return validate_video(target, len(frames))


def check_submission_directory(
    submission_root: Path,
    selected_names: set[str],
    all_names: set[str],
    require_complete: bool,
) -> None:
    files = [path for path in submission_root.iterdir() if path.is_file()]
    non_mp4 = sorted(
        path.name
        for path in files
        if path.suffix != ".mp4" and not path.name.startswith(".")
    )
    if non_mp4:
        raise RuntimeError(
            f"submission directory must be MP4-only; unexpected files: {non_mp4[:5]}"
        )
    actual = {path.name for path in files if path.suffix == ".mp4"}
    invalid = sorted(actual - all_names)
    missing_selected = sorted(selected_names - actual)
    if invalid or missing_selected:
        raise RuntimeError(
            f"submission set mismatch: invalid={invalid[:5]}, "
            f"missing_selected={missing_selected[:5]}"
        )
    if require_complete and actual != all_names:
        raise RuntimeError(
            f"complete submission requires {len(all_names)} MP4 files, found {len(actual)}"
        )


def main() -> None:
    args = parse_args()
    reject_gpu4(str(args.device))
    if not FFPROBE.is_file():
        raise FileNotFoundError(f"ffprobe not found: {FFPROBE}")
    if (
        args.height,
        args.width,
        args.condition_frames,
        args.condition_fps,
        args.num_frames,
        args.fps,
        args.num_inference_steps,
        args.guidance_scale,
        args.seed,
    ) != (
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
        EXPECTED_CONDITION_FRAMES,
        EXPECTED_CONDITION_FPS,
        EXPECTED_RAW_FRAMES,
        EXPECTED_FPS,
        EXPECTED_STEPS,
        EXPECTED_GUIDANCE,
        EXPECTED_SEED,
    ):
        raise ValueError(
            "this runner is fixed to the Physics-IQ P0 72f protocol: "
            "72@24 condition, 189@24 raw, 512x896, 40 steps, guidance 5, seed 42"
        )
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard-index/shard-count")

    input_list = require_file(args.input_json_list, "input JSON list")
    model_id = require_dir(args.model_id, "base model")
    dit_checkpoint = require_file(args.physrvg_dit_checkpoint, "PhysRVG DiT checkpoint")
    lora_checkpoint = require_dir(args.lora_checkpoint, "LoRA checkpoint")
    raw_root = args.raw_output_root.expanduser().resolve()
    submission_root = args.submission_output_root.expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    submission_root.mkdir(parents=True, exist_ok=True)
    if args.manifest_path is None:
        manifest_path = submission_root.parent / f"{submission_root.name}.manifest.json"
    else:
        manifest_path = args.manifest_path.expanduser().resolve()

    inference = import_inference_module(args.repo_root)
    prompt_hash = sha256_text(str(inference.NEGATIVE_PROMPT))
    if prompt_hash != EXPECTED_NEGATIVE_PROMPT_SHA256:
        raise RuntimeError(
            "shared inference negative prompt does not match "
            f"{EXPECTED_NEGATIVE_PROMPT_VERSION}: {prompt_hash}"
        )

    selected, all_names_list = load_cases(
        input_list,
        args.limit,
        args.shard_index,
        args.shard_count,
    )
    all_names = set(all_names_list)
    selected_names = {payload["generated_video_name"] for _, _, payload in selected}
    # A fresh submission directory is intentionally incomplete here.  The
    # complete-set check belongs after generation, below; per-case resume
    # validation is handled inside the loop.

    loader_args = build_loader_args(args)
    print(
        f"protocol=Physics-IQ Verified P0; cases={len(selected)}; "
        f"condition=72@24; raw=189@24; submission=120@24; "
        f"resolution=512x896; steps=40; guidance=5; seed=42; "
        f"rng=global_seed_per_case; encoding=in_memory_slice_macro_block_1",
        flush=True,
    )
    print(f"model_id={model_id}", flush=True)
    print(f"dit_checkpoint={dit_checkpoint}", flush=True)
    print(f"lora_checkpoint={lora_checkpoint}", flush=True)
    print(f"submission_root={submission_root}", flush=True)
    print(f"raw_root={raw_root}", flush=True)

    pipe, loaded_lora_targets, physrvg_dit, object_system, object_info = inference.load_pipeline(
        loader_args
    )
    if object_system is not None or object_info.get("enabled"):
        raise RuntimeError("the 72f-aligned runner does not support object-XSSC mode")
    case_records: list[dict[str, Any]] = []
    for index, case_json, payload in selected:
        generated_name = str(payload["generated_video_name"])
        raw_path = raw_root / generated_name
        submission_path = submission_root / generated_name
        metadata_path = raw_path.with_suffix(".json")
        if (
            not args.force
            and raw_path.is_file()
            and submission_path.is_file()
        ):
            raw_info = validate_video(raw_path, EXPECTED_RAW_FRAMES)
            submission_info = validate_video(submission_path, EXPECTED_SUBMISSION_FRAMES)
            status = "skipped_valid_existing"
        else:
            condition_path = Path(str(payload["input_video"])).resolve()
            condition_info = probe_video(condition_path)
            if condition_info["frames"] != EXPECTED_CONDITION_FRAMES:
                raise RuntimeError(f"condition frame count mismatch: {condition_path}")
            if abs(float(condition_info["fps"]) - EXPECTED_CONDITION_FPS) > 1e-6:
                raise RuntimeError(f"condition FPS mismatch: {condition_path}")
            condition = inference.load_context_video(
                condition_path,
                EXPECTED_HEIGHT,
                EXPECTED_WIDTH,
                max_frames=EXPECTED_CONDITION_FRAMES,
            )
            if len(condition) != EXPECTED_CONDITION_FRAMES:
                raise RuntimeError(f"decoded condition frame count mismatch: {condition_path}")

            # This is the same per-case global RNG policy as 72f-adapted.
            set_seed(EXPECTED_SEED)
            sample = pipe(
                video=condition,
                device=torch.device(args.device),
                prompt=str(payload["input_caption"]).strip(),
                negative_prompt=inference.NEGATIVE_PROMPT,
                height=EXPECTED_HEIGHT,
                width=EXPECTED_WIDTH,
                num_frames=EXPECTED_RAW_FRAMES,
                num_inference_steps=EXPECTED_STEPS,
                guidance_scale=EXPECTED_GUIDANCE,
                do_cfg=False,
                generator=None,
                dynamic_condition_mask=True,
                object_slots=None,
                object_keep_mask=None,
            )[0][0]
            raw_frames = list(sample)
            if len(raw_frames) != EXPECTED_RAW_FRAMES:
                raise RuntimeError(
                    f"model returned {len(raw_frames)} frames; expected {EXPECTED_RAW_FRAMES}"
                )
            submission_frames = raw_frames[EXPECTED_PREFIX_FRAMES:]
            if len(submission_frames) != EXPECTED_SUBMISSION_FRAMES:
                raise RuntimeError(
                    f"prediction segment has {len(submission_frames)} frames; "
                    f"expected {EXPECTED_SUBMISSION_FRAMES}"
                )

            raw_info = export_frames(raw_frames, raw_path)
            # Crucially, this uses the in-memory slice, not a decode of raw_path.
            submission_info = export_frames(submission_frames, submission_path)
            status = "generated"
            write_json(
                metadata_path,
                {
                    "schema_version": 1,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "case_index": index,
                    "input_json": str(case_json),
                    "input_video": str(condition_path),
                    "input_caption": str(payload["input_caption"]),
                    "model": {
                        "model_id": str(model_id),
                        "dit_checkpoint": str(dit_checkpoint),
                        "lora_checkpoint": str(lora_checkpoint),
                        "loaded_lora_target_modules": loaded_lora_targets,
                        "dit_strict": bool(physrvg_dit["strict"]),
                    },
                    "inference": {
                        "height": EXPECTED_HEIGHT,
                        "width": EXPECTED_WIDTH,
                        "condition_frames": EXPECTED_CONDITION_FRAMES,
                        "condition_fps": EXPECTED_CONDITION_FPS,
                        "num_frames": EXPECTED_RAW_FRAMES,
                        "fps": EXPECTED_FPS,
                        "num_inference_steps": EXPECTED_STEPS,
                        "guidance_scale": EXPECTED_GUIDANCE,
                        "seed": EXPECTED_SEED,
                        "rng_mode": "global_seed_per_case",
                        "global_seed_reset_per_case": True,
                        "context_mask_mode": args.context_mask_mode,
                        "classifier_free_guidance_enabled": False,
                        "negative_prompt_version": EXPECTED_NEGATIVE_PROMPT_VERSION,
                        "negative_prompt_sha256": EXPECTED_NEGATIVE_PROMPT_SHA256,
                    },
                    "encoding": {
                        "mode": "in_memory_slice_then_export_to_video",
                        "raw_frames": EXPECTED_RAW_FRAMES,
                        "prefix_frames_removed": EXPECTED_PREFIX_FRAMES,
                        "submission_frames": EXPECTED_SUBMISSION_FRAMES,
                        "macro_block_size": 1,
                        "intermediate_decode": False,
                    },
                    "outputs": {
                        "raw": str(raw_path),
                        "submission": str(submission_path),
                        "raw_probe": raw_info,
                        "submission_probe": submission_info,
                    },
                },
            )
        case_records.append(
            {
                "index": index,
                "case_json": str(case_json),
                "generated_video_name": generated_name,
                "status": status,
                "raw": str(raw_path),
                "submission": str(submission_path),
                "raw_probe": raw_info,
                "submission_probe": submission_info,
            }
        )
        print(f"[{index:03d}/{EXPECTED_CASES:03d}] {status} {generated_name}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    check_submission_directory(
        submission_root,
        selected_names,
        all_names,
        require_complete=args.shard_count == 1 and args.limit is None,
    )
    write_json(
        manifest_path,
        {
            "protocol": "physics-iq-verified-bpp-v2v-strict",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner": str(Path(__file__).resolve()),
            "input_list": str(input_list),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "num_cases_selected": len(selected),
            "num_cases_total": EXPECTED_CASES,
            "model": {
                "model_id": str(model_id),
                "dit_checkpoint": str(dit_checkpoint),
                "dit_strict": bool(physrvg_dit["strict"]),
                "dit_source_tensors": int(physrvg_dit["source_tensors"]),
                "lora_checkpoint": str(lora_checkpoint),
                "loaded_lora_target_modules": int(loaded_lora_targets),
                "loader": "/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/inference/infer_full_sa_lora_json_list.py",
                "pipeline": "/home/gaoya/code_V2V_baselines/PhysRVG-main/fastvideo/models/wan_v2v/pipeline_wan_v2v.py",
            },
            "condition": {"frames": EXPECTED_CONDITION_FRAMES, "fps": EXPECTED_CONDITION_FPS},
            "raw": {"frames": EXPECTED_RAW_FRAMES, "fps": EXPECTED_FPS},
            "submission": {
                "frames": EXPECTED_SUBMISSION_FRAMES,
                "fps": EXPECTED_FPS,
                "seconds": 5.0,
                "mp4_only": True,
            },
            "inference": {
                "height": EXPECTED_HEIGHT,
                "width": EXPECTED_WIDTH,
                "num_inference_steps": EXPECTED_STEPS,
                "guidance_scale": EXPECTED_GUIDANCE,
                "seed": EXPECTED_SEED,
                "rng_mode": "global_seed_per_case",
                "context_mask_mode": args.context_mask_mode,
                "do_cfg": False,
                "negative_prompt_version": EXPECTED_NEGATIVE_PROMPT_VERSION,
                "negative_prompt_sha256": EXPECTED_NEGATIVE_PROMPT_SHA256,
            },
            "encoding": {
                "mode": "in_memory_slice_then_export_to_video",
                "slice": f"raw[{EXPECTED_PREFIX_FRAMES}:{EXPECTED_RAW_FRAMES}]",
                "macro_block_size": 1,
                "intermediate_decode": False,
            },
            "outputs": {
                "raw_root": str(raw_root),
                "submission_root": str(submission_root),
                "manifest": str(manifest_path),
            },
            "cases": case_records,
        },
    )
    print(f"manifest={manifest_path}", flush=True)
    print(f"submission=PASS {submission_root}", flush=True)


if __name__ == "__main__":
    main()
