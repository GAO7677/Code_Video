#!/usr/bin/env python3
"""Run a PhysRVG LoRA-OFF context-length prefix sweep.

The model is loaded once per process and the same pipeline is reused for all
requested context lengths.  Only the first K decoded context frames change;
all generation settings and input captions remain fixed.
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PHYSRVG_ROOT = Path("/home/gaoya/code_V2V_baselines/PhysRVG-main")
PHYSRVG_SCRIPT_ROOT = PHYSRVG_ROOT / "scripts_mytrain"
if str(PHYSRVG_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(PHYSRVG_SCRIPT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infer_full_sa_lora_json_list import (  # noqa: E402
    ensure_exists,
    load_pipeline,
    read_input_list,
    run_case,
)


DATASETS: dict[str, dict[str, Any]] = {
    "test5": {
        "input_list": Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"),
        "steps": 8,
        "expected_cases": 20,
    },
    "physiciq": {
        "input_list": Path(
            "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
        ),
        "steps": 40,
        "expected_cases": 67,
    },
}

DEFAULT_MODEL_ID = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers")
DEFAULT_DIT_CHECKPOINT = Path(
    "/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physrvg_context_length_sweep"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--input-list", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-id", type=Path, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--physrvg-dit-checkpoint",
        type=Path,
        default=DEFAULT_DIT_CHECKPOINT,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-lengths", default="1,2,4,5,6,8")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_context_lengths(value: str) -> list[int]:
    lengths: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        length = int(item)
        if length < 1:
            raise ValueError(f"context length must be positive: {length}")
        if length not in lengths:
            lengths.append(length)
    if not lengths:
        raise ValueError("--context-lengths cannot be empty")
    return lengths


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_pipeline_args(args: argparse.Namespace) -> argparse.Namespace:
    """Create the subset of the original inference Namespace used by load_pipeline."""
    return Namespace(
        model_id=args.model_id,
        physrvg_dit_checkpoint=args.physrvg_dit_checkpoint,
        lora_checkpoint=None,
        device=args.device,
        object_xssc_trainable=None,
        object_lora_rank=32,
        object_lora_alpha=32.0,
        object_lora_dropout=0.0,
        object_gate_init=0.1,
        object_context_frames=8,
        xssc_root=Path(
            "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
            "code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256"
        ),
        xssc_config=Path(""),
        xssc_checkpoint=Path(""),
        dinov3_root=Path(""),
        dinov3_checkpoint=Path(""),
        sam2_config=Path(""),
        sam2_checkpoint=Path(""),
        xssc_box_cache_dir=Path("/data/gaoya/agent-data/cache/physrvg_context_sweep"),
        xssc_chunk_size=2,
        slot_dedup_threshold=0.94,
        slot_dedup_min_keep=3,
    )


def make_case_args(
    args: argparse.Namespace,
    output_root: Path,
    steps: int,
    context_frames: int,
) -> argparse.Namespace:
    return Namespace(
        input_json_list=None,
        output_root=output_root,
        model_id=args.model_id,
        physrvg_dit_checkpoint=args.physrvg_dit_checkpoint,
        lora_checkpoint=None,
        device=args.device,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        limit=args.limit,
        flat_output=True,
        force=args.force,
        context_frames=context_frames,
        reset_global_seed_per_case=True,
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda:") and args.device == "cuda:4":
        raise SystemExit("GPU4 is prohibited by workspace rules")
    if args.num_frames < 1 or (args.num_frames - 1) % 4 != 0:
        raise ValueError("--num-frames must satisfy 4n+1 for Wan VAE temporal alignment")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    dataset = DATASETS[args.dataset]
    input_list = ensure_exists(
        (args.input_list or dataset["input_list"]).resolve(), "input-list"
    )
    args.model_id = ensure_exists(args.model_id, "model-id")
    args.physrvg_dit_checkpoint = ensure_exists(
        args.physrvg_dit_checkpoint, "physrvg-dit-checkpoint"
    )
    if not args.physrvg_dit_checkpoint.is_file():
        raise FileNotFoundError(
            f"physrvg-dit-checkpoint is not a file: {args.physrvg_dit_checkpoint}"
        )

    cases = read_input_list(input_list)
    if len(cases) != int(dataset["expected_cases"]):
        raise ValueError(
            f"{args.dataset} expects {dataset['expected_cases']} cases, found {len(cases)}"
        )
    if args.limit is not None:
        cases = cases[: args.limit]

    context_lengths = parse_context_lengths(args.context_lengths)
    output_root = args.output_root.expanduser().resolve()
    dataset_root = output_root / args.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    steps = int(dataset["steps"])

    pipeline_args = make_pipeline_args(args)
    pipe, loaded_lora_targets, physrvg_dit, object_system, object_info = load_pipeline(
        pipeline_args
    )
    if loaded_lora_targets != 0 or object_info.get("enabled"):
        raise RuntimeError(
            "Reference sweep unexpectedly enabled LoRA or object branch: "
            f"lora_targets={loaded_lora_targets}, object={object_info}"
        )
    print(
        f"[load] dataset={args.dataset} base={args.model_id} "
        f"dit={physrvg_dit['checkpoint']} strict={physrvg_dit['strict']} "
        f"lora=None object=False device={args.device}",
        flush=True,
    )

    sweep_manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "input_list": str(input_list),
        "case_count": len(cases),
        "total_cases_in_input_list": len(read_input_list(input_list)),
        "context_lengths": context_lengths,
        "model": {
            "load_mode": "wan22_diffusers_base_plus_physrvg_dit_strict_lora_off",
            "model_id": str(args.model_id),
            "dit_checkpoint": str(args.physrvg_dit_checkpoint),
            "dit_strict": bool(physrvg_dit["strict"]),
            "lora_checkpoint": None,
            "loaded_lora_target_modules": loaded_lora_targets,
            "object_xssc": object_info,
        },
        "inference": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "num_inference_steps": steps,
            "guidance_scale": args.guidance_scale,
            "classifier_free_guidance_enabled": False,
            "seed": args.seed,
            "prefix_policy": "first_K_decoded_context_frames",
            "global_seed_reset_per_case": True,
        },
        "lengths": {},
    }

    for context_length in context_lengths:
        length_root = dataset_root / (
            f"ctx{context_length:02d}_steps{steps:02d}_"
            f"{args.height}x{args.width}_49f"
        )
        length_root.mkdir(parents=True, exist_ok=True)
        case_args = make_case_args(args, length_root, steps, context_length)
        counts: dict[str, int] = {}
        results: list[dict[str, Any]] = []
        print(
            f"[length] dataset={args.dataset} ctx={context_length} "
            f"cases={len(cases)} output={length_root}",
            flush=True,
        )
        for global_index, input_json_path in enumerate(cases):
            print(
                f"[case {global_index:02d}] ctx={context_length} "
                f"{input_json_path.name}",
                flush=True,
            )
            try:
                result = run_case(
                    pipe=pipe,
                    args=case_args,
                    input_json_path=input_json_path,
                    global_index=global_index,
                    loaded_lora_targets=loaded_lora_targets,
                    physrvg_dit=physrvg_dit,
                    object_system=object_system,
                    object_info=object_info,
                )
            except Exception as exc:
                result = {
                    "case_index": global_index,
                    "input_json": str(input_json_path),
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            status = str(result.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
            results.append(result)
            print(f"[case {global_index:02d}] ctx={context_length} {status}", flush=True)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        length_payload = {
            "dataset": args.dataset,
            "context_frames": context_length,
            "steps": steps,
            "output_root": str(length_root),
            "case_count": len(cases),
            "counts": counts,
            "results": results,
        }
        write_json(length_root / "summary.json", length_payload)
        sweep_manifest["lengths"][str(context_length)] = length_payload

    write_json(dataset_root / "sweep_manifest.json", sweep_manifest)
    write_json(output_root / "sweep_manifest.json", {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_manifest": str(dataset_root / "sweep_manifest.json"),
    })
    print(f"[done] {dataset_root / 'sweep_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
