#!/usr/bin/env python3
"""Extract object-aligned xSSC features for the Phase-1 controlled pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
TRAIN_XSSC = HERE.parent
EXPERIMENTS = TRAIN_XSSC / "object_self_attn_lora_experiments"
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_CONFIG = HERE / "experiment_config.json"
DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/outputs/xssc_physics_representation/phase1")
MASK_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_slot_separation_cases_dinov3_latest/"
    "ball_block_parameter_pairs/object_gt_aligned/role_masks"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("prepare", "extract", "all"), default="all")
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--model-names", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def base_scenario(case_id: str) -> str:
    marker = "_v"
    if marker in case_id:
        prefix, suffix = case_id.rsplit(marker, 1)
        if suffix and suffix[0].isdigit():
            return prefix
    return case_id


def normalize_parameters(payload: dict[str, Any]) -> dict[str, float]:
    params = payload.get("parameters", {})
    friction = params.get("lateral_friction", params.get("friction"))
    return {
        "restitution": float(params["restitution"]),
        "friction": float(friction),
        "ball_mass_kg": float(params["ball_mass_kg"]),
        "block_mass_kg": float(params.get("block_mass_kg", 1.5)),
    }


def discover_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    phase = config["phase1"]
    roots = [
        ("physics", Path(phase["physics_video_root"])),
        ("appearance", Path(phase["appearance_video_root"])),
    ]
    cases: list[dict[str, Any]] = []
    for family, root in roots:
        for metadata_path in sorted(root.glob("*.json")):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            case_id = metadata_path.stem
            video = Path(payload["video"])
            if not video.is_file():
                raise FileNotFoundError(video)
            source_scenario = base_scenario(case_id)
            mask_path = MASK_ROOT / f"{source_scenario}.npz"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            cases.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "base_scenario": source_scenario,
                    "appearance_variant": payload.get("appearance_variant", "base_render"),
                    "video": str(video),
                    "metadata": str(metadata_path),
                    "role_masks": str(mask_path),
                    "parameters": normalize_parameters(payload),
                }
            )
    family_counts = {family: sum(case["family"] == family for case in cases) for family in ("physics", "appearance")}
    if family_counts != {"physics": 8, "appearance": 24}:
        raise RuntimeError(f"Unexpected Phase-1 case counts: {family_counts}")
    return cases


def model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    primary = config["primary_model"]
    comparison = {item["name"]: item for item in config["comparison_models"]}
    dinov3_root = TRAIN_XSSC / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
    specs = [
        {
            "name": primary["name"],
            "short_name": "DINOv3 MOVi-C step-044000",
            "family": "dinov3",
            "comparison_role": "primary_model",
            "xssc_root": str(dinov3_root),
            "xssc_config": primary["config"],
            "xssc_checkpoint": primary["checkpoint"],
            "dinov3_root": str(dinov3_root / "third_party/dinov3"),
            "dinov3_checkpoint": "/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors",
        },
        {
            "name": "official_dinov2_movic_42-0035",
            "short_name": "Official DINOv2 MOVi-C 42-0035",
            "family": "dinov2",
            "comparison_role": comparison["official_dinov2_movic_42-0035"]["role"],
            "xssc_root": "/home/gaoya/Code_Video/xSSC-main",
            "xssc_config": "/home/gaoya/Code_Video/xSSC-main/config-randsfq/rsfq2_c-movi_c.py",
            "xssc_checkpoint": comparison["official_dinov2_movic_42-0035"]["checkpoint"],
        },
        {
            "name": "official_dinov2_ytvis_42-0130",
            "short_name": "Official DINOv2 YTVIS 42-0130",
            "family": "dinov2",
            "comparison_role": comparison["official_dinov2_42-0130"]["role"],
            "xssc_root": "/home/gaoya/Code_Video/xSSC-main",
            "xssc_config": "/home/gaoya/Code_Video/xSSC-main/config-randsfq/rsfq2_r-ytvis.py",
            "xssc_checkpoint": comparison["official_dinov2_42-0130"]["checkpoint"],
        },
    ]
    for spec in specs:
        for key in ("xssc_config", "xssc_checkpoint"):
            if not Path(spec[key]).is_file():
                raise FileNotFoundError(spec[key])
    return specs


def role_boxes_from_masks(mask_path: Path, num_slots: int, num_frames: int) -> np.ndarray:
    with np.load(mask_path) as item:
        masks = item["masks"].astype(np.float32)
    if masks.shape != (num_frames, 2, 16, 16):
        raise RuntimeError(f"Unexpected GT mask shape {masks.shape}: {mask_path}")
    boxes = np.zeros((1, num_frames, num_slots, 4), dtype=np.float32)
    for role_id in range(2):
        mask = masks[0, role_id]
        rows, cols = np.where(mask > 1.0e-4)
        if len(rows) == 0:
            raise RuntimeError(f"Empty frame-0 role {role_id}: {mask_path}")
        boxes[0, :, role_id] = np.asarray(
            [cols.min() / 16.0, rows.min() / 16.0, (cols.max() + 1) / 16.0, (rows.max() + 1) / 16.0],
            dtype=np.float32,
        )
    return boxes


def prepare(args: argparse.Namespace, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = discover_cases(config)
    specs = model_specs(config)
    baseline = str(config["phase1"]["baseline_scenario"])
    cases.sort(key=lambda case: (case["case_id"] != baseline, case["family"], case["case_id"]))
    if args.case_limit > 0:
        cases = cases[: args.case_limit]
    requested_models = {item.strip() for item in args.model_names.split(",") if item.strip()}
    if requested_models:
        specs = [spec for spec in specs if spec["name"] in requested_models]
        missing = requested_models - {spec["name"] for spec in specs}
        if missing:
            raise ValueError(f"Unknown model names: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "specs").mkdir(exist_ok=True)
    for spec in specs:
        write_json(args.output_dir / "specs" / f"{spec['name']}.json", spec)
    manifest = {
        "protocol": {
            "frames": 150,
            "preprocess": "preserve aspect ratio, resize to fit 256x256, ImageNet-mean padding",
            "movic_initialization": "frame-0 simulator-GT ball/block boxes repeated on the condition tensor; only t=0 is consumed by initializ",
            "ytvis_initialization": "official NormalShared; no bbox input",
            "role_binding": "Hungarian assignment on full-video mean soft recall against simulator GT masks",
            "gt_limit": "GT is used only for MOVi-C frame-0 initialization, slot quality audit, and evaluation",
        },
        "models": specs,
        "cases": cases,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    return cases, specs


def worker(args: argparse.Namespace) -> None:
    import torch

    for path in (TRAIN_XSSC.parent.parent, TRAIN_XSSC.parent, TRAIN_XSSC, EXPERIMENTS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import analyze_ball_block_xssc_parameter_pairs as base
    import analyze_xssc_dinov3_object_slot_separation_cases as extractor
    import refine_ball_block_object_gt_pairs as gt_eval
    import run_xssc_slot_dedup_weight_compare as loader

    spec = json.loads(args.worker_spec.read_text(encoding="utf-8"))
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, slot_dim, num_slots, initializer = loader.load_xssc_variant(spec, device)
    model_dir = args.output_dir / "features" / spec["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for position, case in enumerate(manifest["cases"], start=1):
        output_path = model_dir / f"{case['case_id']}.npz"
        if output_path.is_file() and not args.force:
            print(f"[{position:02d}/{len(manifest['cases']):02d}] cached {case['case_id']}", flush=True)
            continue
        normalized, _, fps, transform = base.read_letterboxed_video(Path(case["video"]), 256)
        if len(normalized) != 150:
            raise RuntimeError(f"Expected 150 frames: {case['video']}")
        boxes = None
        if initializer == "bbox_mlp":
            boxes_np = role_boxes_from_masks(Path(case["role_masks"]), num_slots, len(normalized))
            boxes = torch.from_numpy(boxes_np)
        slots, attention = extractor.extract_variant_slots(
            model,
            normalized,
            device=device,
            seed=args.seed,
            batch_size=args.batch_size,
            initializer=initializer,
            boxes=boxes,
        )
        slots_np = slots.numpy().astype(np.float32)
        attention_np = attention.numpy().astype(np.float32)
        with np.load(case["role_masks"]) as item:
            role_masks = item["masks"].astype(np.float32)
        assignment = gt_eval.role_slot_assignment(attention_np, role_masks)
        selected = assignment["selected"].astype(np.int16)
        np.savez_compressed(
            output_path,
            slots=slots_np.astype(np.float16),
            attention=attention_np.astype(np.float16),
            selected_slots=selected,
            recall_matrix=assignment["recall_matrix"].astype(np.float32),
            precision_matrix=assignment["precision_matrix"].astype(np.float32),
            f1_matrix=assignment["f1_matrix"].astype(np.float32),
            frame_indices=np.arange(150, dtype=np.int16),
        )
        records.append(
            {
                "case_id": case["case_id"],
                "selected_slots": selected.tolist(),
                "assignment": assignment["details"],
                "preprocess": transform,
                "fps": fps,
            }
        )
        print(f"[{position:02d}/{len(manifest['cases']):02d}] {case['case_id']} selected={selected.tolist()}", flush=True)
        torch.cuda.empty_cache()
    write_json(
        model_dir / "metadata.json",
        {
            "model": {**spec, "initializer": initializer, "slot_dim": slot_dim, "num_slots": num_slots},
            "completed_cases": len(list(model_dir.glob("*.npz"))),
            "new_records": records,
        },
    )


def extract(args: argparse.Namespace, specs: list[dict[str, Any]]) -> None:
    if args.physical_gpu == 4:
        raise ValueError("GPU 4 is forbidden by workspace policy")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TRAIN_XSSC.parent.parent), str(TRAIN_XSSC.parent), str(TRAIN_XSSC), str(EXPERIMENTS), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    for spec in specs:
        spec_path = args.output_dir / "specs" / f"{spec['name']}.json"
        log_path = args.output_dir / "logs" / f"{spec['name']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), str(Path(__file__).resolve()),
            "--stage", "extract", "--worker-spec", str(spec_path),
            "--output-dir", str(args.output_dir), "--device", "cuda:0",
            "--batch-size", str(args.batch_size), "--seed", str(args.seed),
        ]
        if args.force:
            command.append("--force")
        print(f"[model] {spec['short_name']} -> physical GPU {args.physical_gpu}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(command, env=env, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            raise RuntimeError(f"Feature worker failed: {spec['name']}; see {log_path}")


def main() -> None:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.worker_spec is not None:
        worker(args)
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cases, specs = prepare(args, config)
    print(f"[prepare] cases={len(cases)} models={len(specs)} manifest={args.output_dir / 'manifest.json'}")
    if args.stage in ("extract", "all"):
        extract(args, specs)


if __name__ == "__main__":
    main()
