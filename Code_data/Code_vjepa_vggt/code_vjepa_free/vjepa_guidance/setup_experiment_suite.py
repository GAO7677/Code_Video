#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .build_experiment_manifest import DEFAULT_RESULTS_ROOT, DEFAULT_SOURCE_DIR, build_rows, write_manifest
    from .experiment_presets import PILOT_PROMPT_IDS, ROUND1_MODES
except ImportError:
    from build_experiment_manifest import DEFAULT_RESULTS_ROOT, DEFAULT_SOURCE_DIR, build_rows, write_manifest
    from experiment_presets import PILOT_PROMPT_IDS, ROUND1_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up the V-JEPA guidance experiment suite directory tree.")
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--round_name", type=str, default="round1_pilot")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--sample_solver", type=str, default="unipc")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = args.results_root / args.round_name
    (suite_root / "manifests").mkdir(parents=True, exist_ok=True)
    (suite_root / "reports").mkdir(parents=True, exist_ok=True)

    namespace = argparse.Namespace(
        source_dir=args.source_dir,
        results_root=args.results_root,
        round_name=args.round_name,
        prompt_ids=PILOT_PROMPT_IDS,
        mode_ids=[mode.mode_id for mode in ROUND1_MODES],
        seeds=args.seeds,
        sample_steps=args.sample_steps,
        sample_solver=args.sample_solver,
        sample_shift=args.sample_shift,
        sample_guide_scale=args.sample_guide_scale,
        frame_num=args.frame_num,
        size=args.size,
        device_id=args.device_id,
        vjepa_device_id=args.vjepa_device_id,
        ckpt_dir=args.ckpt_dir,
        vjepa_ckpt=args.vjepa_ckpt,
        python_bin=args.python_bin,
        script_path=args.script_path,
        output_manifest=None,
    )
    rows = build_rows(namespace)
    manifest_path = suite_root / "manifests" / "manifest.csv"
    write_manifest(manifest_path, rows)

    preset_summary = {
        "round_name": args.round_name,
        "prompt_ids": PILOT_PROMPT_IDS,
        "seeds": args.seeds,
        "modes": [mode.to_dict() for mode in ROUND1_MODES],
        "source_dir": str(args.source_dir),
        "results_root": str(args.results_root),
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
    }
    summary_path = suite_root / "suite_config.json"
    summary_path.write_text(json.dumps(preset_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme_path = suite_root / "README.txt"
    readme_path.write_text(
        "\n".join(
            [
                f"round_name: {args.round_name}",
                f"manifest: {manifest_path}",
                "directory layout:",
                "  manifests/: manifest CSVs",
                "  reports/: aggregate tables or notebooks",
                "  <mode_id>/<prompt_id>/: mp4, log, and same-name json metadata for each case",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
