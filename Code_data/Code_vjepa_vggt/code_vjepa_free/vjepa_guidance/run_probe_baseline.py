#!/usr/bin/env python3
"""
Run a single baseline generation through the VJEPA-wrapper pipeline
(enable_vjepa_guidance=False), for comparison against the pure pipeline baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from probe_energy_persistence import _build_pipeline, _run_condition, WanVJEPAConfig

log = logging.getLogger(__name__)

WEIGHTS_ROOT = Path("/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
                    "pybullet0629_teacher_student/stage1b_context_only_no_gt_box_diffsynth/"
                    "checkpoints/step-000500")
INPUT_JSON   = Path("/data/gaoya/AAA_test_video/0623/testdataset/"
                    "025_Solid_Mechanics_0002_perspective-center_trimmed/"
                    "025_Solid_Mechanics_0002_perspective-center_trimmed.json")
CONTEXT_PATH = Path("/data/gaoya/AAA_test_video/0623/testdataset/"
                    "025_Solid_Mechanics_0002_perspective-center_trimmed/"
                    "source_video/context_video_8f.mp4")
WAN_ROOT     = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
VJEPA_CKPT   = Path("/data/gaoya/ckpt/VJEPA2/vith.pt")
OUTPUT_DIR   = Path("/data/gaoya/agent-data/outputs/probe_sweep/baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--vjepa-device", type=str, default=None)
    p.add_argument("--output-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    lora_path = WEIGHTS_ROOT / "checkpoint.safetensors"
    case = json.loads(INPUT_JSON.read_text())

    vjepa_config = WanVJEPAConfig(
        guidance_steps=1,
        min_step_percent=0.50,
        max_step_percent=0.50,
        latent_step_size=0.02,
        preview_downsample_factor=4,
        preview_frame_stride=2,
        window_size=16,
        context_frames=8,
        stride=4,
    )

    vjepa_device = args.vjepa_device or args.device
    log.info("Building VJEPA pipeline (device=%s)...", args.device)
    pipe = _build_pipeline(
        wan_root=WAN_ROOT,
        device=args.device,
        lora_path=lora_path,
        vjepa_model="vith",
        vjepa_ckpt=VJEPA_CKPT,
        vjepa_device=vjepa_device,
        vjepa_config=vjepa_config,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"probe_baseline_seed{args.seed}.mp4"
    save_path = OUTPUT_DIR / output_name

    log.info("Running baseline (guidance=False) with seed=%d ...", args.seed)
    _run_condition(
        pipe=pipe,
        case=case,
        seed=args.seed,
        num_frames=49,
        height=480,
        width=832,
        num_inference_steps=40,
        cfg_scale=5.0,
        negative_prompt="",
        context_path=CONTEXT_PATH,
        guidance_step_percents=[],   # baseline: no guidance
        vjepa_config=vjepa_config,
        probe_every_n=2,
        condition_label="probe_baseline",
        save_video_path=save_path,
    )

    log.info("Done: %s", save_path)


if __name__ == "__main__":
    main()
