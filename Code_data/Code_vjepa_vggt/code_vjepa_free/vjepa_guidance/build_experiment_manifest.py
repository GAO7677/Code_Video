#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .experiment_presets import MODE_MAP, PILOT_PROMPT_IDS, ROUND1_MODES
except ImportError:
    from experiment_presets import MODE_MAP, PILOT_PROMPT_IDS, ROUND1_MODES


DEFAULT_SOURCE_DIR = Path(
    "/data/gaoya/AAA_test_video/Output_try0526/phygenbench/output/FLUX_1_Kontext/phygenbench"
)
DEFAULT_RESULTS_ROOT = Path("/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest for V-JEPA guidance experiments.")
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--round_name", type=str, default="round1_pilot")
    parser.add_argument("--prompt_ids", type=str, nargs="*", default=PILOT_PROMPT_IDS)
    parser.add_argument("--mode_ids", type=str, nargs="*", default=[mode.mode_id for mode in ROUND1_MODES])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--frame_num", type=int, default=41)
    parser.add_argument("--size", type=str, default="1280*704")
    parser.add_argument("--device_id", type=int, default=5)
    parser.add_argument("--vjepa_device_id", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    parser.add_argument("--vjepa_ckpt", type=Path, default=Path("/data/gaoya/ckpt/VJEPA2/vith.pt"))
    parser.add_argument("--python_bin", type=Path, default=Path("/data/gaoya/miniconda3/envs/wan/bin/python"))
    parser.add_argument("--script_path", type=Path, default=Path(__file__).resolve().parent / "wan_ti2v_vjepa.py")
    parser.add_argument("--output_manifest", type=Path, default=None)
    return parser.parse_args()


def load_case(source_dir: Path, prompt_id: str) -> dict[str, object]:
    meta_path = source_dir / f"{prompt_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing source metadata: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt_id in args.prompt_ids:
        case = load_case(args.source_dir, prompt_id)
        prompt = case["prompt"]
        first_frame = case.get("first_frame") or case.get("image_path")
        if not first_frame:
            raise ValueError(f"Sample {prompt_id} has no first_frame or image_path")
        for seed in args.seeds:
            for mode_id in args.mode_ids:
                mode = MODE_MAP[mode_id]
                exp_id = f"{args.round_name}_{prompt_id}_{mode_id}_seed{seed}"
                output_dir = args.results_root / args.round_name / mode_id / prompt_id
                output_video = output_dir / f"{exp_id}.mp4"
                output_json = output_dir / f"{exp_id}.json"
                rows.append(
                    {
                        "exp_id": exp_id,
                        "round_name": args.round_name,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "source_json": str(args.source_dir / f"{prompt_id}.json"),
                        "source_image": str(first_frame),
                        "main_category": case.get("main_category", ""),
                        "sub_category": case.get("sub_category", ""),
                        "physical_laws": case.get("physical_laws", ""),
                        "seed": seed,
                        "mode_id": mode.mode_id,
                        "mode_description": mode.description,
                        "disable_vjepa_guidance": int(mode.disable_vjepa_guidance),
                        "vjepa_model": mode.vjepa_model,
                        "vjepa_guidance_steps": mode.vjepa_guidance_steps,
                        "vjepa_min_step_percent": mode.vjepa_min_step_percent,
                        "vjepa_max_step_percent": mode.vjepa_max_step_percent,
                        "vjepa_latent_step_size": mode.vjepa_latent_step_size,
                        "vjepa_preview_downsample_factor": mode.vjepa_preview_downsample_factor,
                        "vjepa_preview_frame_stride": mode.vjepa_preview_frame_stride,
                        "vjepa_window_size": mode.vjepa_window_size,
                        "vjepa_context_frames": mode.vjepa_context_frames,
                        "vjepa_stride": mode.vjepa_stride,
                        "vjepa_reduction": mode.vjepa_reduction,
                        "vjepa_grad_norm_mode": mode.vjepa_grad_norm_mode,
                        "vjepa_max_grad_norm": mode.vjepa_max_grad_norm,
                        "sample_steps": args.sample_steps,
                        "sample_solver": args.sample_solver,
                        "sample_shift": args.sample_shift,
                        "sample_guide_scale": args.sample_guide_scale,
                        "frame_num": args.frame_num,
                        "size": args.size,
                        "device_id": args.device_id,
                        "vjepa_device_id": args.vjepa_device_id,
                        "ckpt_dir": str(args.ckpt_dir),
                        "vjepa_ckpt": str(args.vjepa_ckpt),
                        "python_bin": str(args.python_bin),
                        "script_path": str(args.script_path),
                        "output_video": str(output_video),
                        "output_json": str(output_json),
                    }
                )
    return rows


def write_manifest(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    if not rows:
        raise RuntimeError("No rows were generated for the manifest.")
    output_manifest = (
        args.output_manifest
        if args.output_manifest is not None
        else args.results_root / args.round_name / "manifest.csv"
    )
    write_manifest(output_manifest, rows)
    print(output_manifest)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
