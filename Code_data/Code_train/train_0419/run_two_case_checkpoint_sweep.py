#!/usr/bin/env python3
"""Run baseline and checkpoint sweeps for a small set of meta.json cases."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2


TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
BATCH_EVAL_SCRIPT = TRAIN0419_ROOT / "batch_eval_lora.py"
WAN_PYTHON = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
CHECKPOINT_ROOT = Path(
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints"
)
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/two_case_checkpoint_sweep")
DEFAULT_META_PATHS = [
    Path(
        "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/mytest/"
        "genesis_heldout_0008__10005__case007_entry_fast_center/meta.json"
    ),
    Path(
        "/data/gaoya/dataset/physics-iq-benchmark/mytest/0002_perspective-center_trimmed-ball-and-block-fall/meta.json"
    ),
]
STEP_DIR_PATTERN = re.compile(r"step-(\d{6})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a checkpoint sweep on two meta.json cases.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--meta_paths", nargs="*", type=Path, default=DEFAULT_META_PATHS)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--checkpoint_root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--only_case_indices", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--max_future_frames", type=int, default=24)
    parser.add_argument("--max_context_frames", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_stats(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    stats = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return stats


def list_models(checkpoint_root: Path) -> list[dict[str, str | Path | None]]:
    models: list[dict[str, str | Path | None]] = [
        {"model_name": "base-ti2v-5b", "lora_path": None},
    ]
    checkpoint_dirs = []
    for path in checkpoint_root.iterdir():
        if not path.is_dir():
            continue
        match = STEP_DIR_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        lora_path = path / "checkpoint.safetensors"
        if lora_path.is_file():
            checkpoint_dirs.append((int(match.group(1)), path.name, lora_path))
    checkpoint_dirs.sort()
    for _, step_name, lora_path in checkpoint_dirs:
        models.append({"model_name": step_name, "lora_path": lora_path})
    return models


def build_case_config(meta_path: Path, max_future_frames: int, max_context_frames: int) -> dict:
    meta = load_json(meta_path)
    paths = meta["paths"]
    context_path = Path(paths["context_video_path"])
    future_gt_path = Path(paths["future_gt_video_path"])
    full_video_path = Path(paths["full_video_path"])

    context_stats = video_stats(context_path)
    future_stats = video_stats(future_gt_path)
    full_stats = video_stats(full_video_path)

    requested_frames = min(max_future_frames, int(future_stats["frames"]))
    if requested_frames < 1:
        raise ValueError(f"Future GT has no frames: {future_gt_path}")
    context_frames = min(max_context_frames, int(context_stats["frames"]))
    if context_frames < 0:
        raise ValueError(f"Invalid context frame count for: {context_path}")
    if requested_frames <= context_frames:
        requested_frames = min(
            max(max_future_frames, context_frames + 1),
            int(full_stats["frames"]),
        )
    if requested_frames <= context_frames:
        raise ValueError(
            f"Could not satisfy context_frames < num_frames for {meta_path}: "
            f"context_frames={context_frames}, num_frames={requested_frames}"
        )

    return {
        "meta_path": meta_path,
        "sample_id": str(meta.get("sample_id") or meta_path.parent.name),
        "caption": str(meta.get("caption") or ""),
        "context_stats": context_stats,
        "future_stats": future_stats,
        "full_stats": full_stats,
        "num_frames": requested_frames,
        "context_frames": context_frames,
        "fps": int(round(float(future_stats["fps"]) or float(full_stats["fps"]) or 16.0)),
    }


def run_one_case_model(
    *,
    case_cfg: dict,
    model_name: str,
    lora_path: Path | None,
    output_root: Path,
    device: str,
    overwrite: bool,
    seed: int,
    height: int,
    width: int,
    num_inference_steps: int,
    cfg_scale: float,
) -> None:
    model_output_root = output_root / "generated_videos" / model_name
    model_runtime_root = output_root / "runtime" / model_name
    cmd = [
        str(WAN_PYTHON),
        str(BATCH_EVAL_SCRIPT),
        "--model_name",
        model_name,
        "--meta_json_path",
        str(case_cfg["meta_path"]),
        "--output_root",
        str(model_output_root),
        "--runtime_root",
        str(model_runtime_root),
        "--device",
        device,
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_frames",
        str(case_cfg["num_frames"]),
        "--context_frames",
        str(case_cfg["context_frames"]),
        "--fps",
        str(case_cfg["fps"]),
        "--num_inference_steps",
        str(num_inference_steps),
        "--cfg_scale",
        str(cfg_scale),
        "--seed",
        str(seed),
    ]
    if lora_path is not None:
        cmd.extend(["--lora_path", str(lora_path)])
    if overwrite:
        cmd.append("--overwrite")

    print(
        f"[run] device={device} case={case_cfg['sample_id']} model={model_name} "
        f"future_frames={case_cfg['num_frames']} context_frames={case_cfg['context_frames']}"
    )
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    models = list_models(args.checkpoint_root)
    cases = [build_case_config(path, args.max_future_frames, args.max_context_frames) for path in args.meta_paths]
    if args.only_case_indices is not None and len(args.only_case_indices) > 0:
        wanted = set(args.only_case_indices)
        cases = [case for idx, case in enumerate(cases) if idx in wanted]

    manifest = {
        "device": args.device,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "max_future_frames": args.max_future_frames,
        "max_context_frames": args.max_context_frames,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "models": [
            {
                "model_name": str(item["model_name"]),
                "lora_path": str(item["lora_path"]) if item["lora_path"] is not None else None,
            }
            for item in models
        ],
        "cases": [
            {
                "meta_path": str(case["meta_path"]),
                "sample_id": case["sample_id"],
                "num_frames": case["num_frames"],
                "context_frames": case["context_frames"],
                "fps": case["fps"],
                "context_stats": case["context_stats"],
                "future_stats": case["future_stats"],
                "full_stats": case["full_stats"],
            }
            for case in cases
        ],
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for case_cfg in cases:
        for model in models:
            run_one_case_model(
                case_cfg=case_cfg,
                model_name=str(model["model_name"]),
                lora_path=model["lora_path"] if isinstance(model["lora_path"], Path) else None,
                output_root=args.output_root,
                device=args.device,
                overwrite=args.overwrite,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
            )


if __name__ == "__main__":
    main()
