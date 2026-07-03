#!/usr/bin/env python3
"""
Minimal diagnostic for context-anchored guidance (Bug 1 fix verification).

Runs baseline + ONE single-step guided condition (p50) in context_anchored mode
with dense probing (every step), then prints the immediate effect at the guidance
step. The key question: does energy DROP at/just after the guidance step now that
guidance and probe share the exact same anchor?

Cheap: 2 generations (~1-2 min each), no full sweep.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from probe_energy_persistence import (
    _build_pipeline,
    _resolve_lora_path,
    _load_case,
    _run_condition,
    WanVJEPAConfig,
)

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
OUTPUT_DIR   = Path("/data/gaoya/agent-data/outputs/probe_sweep/diag_anchored")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--vjepa-device", type=str, default=None)
    p.add_argument("--timing", type=float, default=0.50, help="guidance step percent")
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--num-inference-steps", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    lora_path = _resolve_lora_path(WEIGHTS_ROOT)
    case = _load_case(INPUT_JSON)

    vjepa_config = WanVJEPAConfig(
        guidance_steps=1,
        min_step_percent=0.50,
        max_step_percent=0.50,
        latent_step_size=args.step_size,
        preview_downsample_factor=4,
        preview_frame_stride=2,
        window_size=16,
        context_frames=8,
        stride=4,
        guidance_mode="context_anchored",
    )

    vjepa_device = args.vjepa_device or args.device
    log.info("Building pipeline (device=%s vjepa_device=%s)...", args.device, vjepa_device)
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

    base_kwargs = dict(
        pipe=pipe,
        case=case,
        seed=args.seed,
        num_frames=args.num_frames,
        height=480,
        width=832,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=5.0,
        negative_prompt="",
        context_path=CONTEXT_PATH,
        vjepa_config=vjepa_config,
        probe_every_n=1,            # DENSE: probe every step
        guidance_mode="context_anchored",
    )

    log.info("=== BASELINE (anchored energy, no guidance) ===")
    _, baseline_records = _run_condition(
        **base_kwargs,
        guidance_step_percents=[],
        condition_label="diag_baseline",
        save_video_path=OUTPUT_DIR / "diag_baseline.mp4",
    )
    (OUTPUT_DIR / "diag_baseline_records.json").write_text(json.dumps(baseline_records, indent=2))

    log.info("=== GUIDED (single step at p%.0f, step_size=%.3f) ===", args.timing * 100, args.step_size)
    # Enable the decisive overshoot-vs-noise line search at the guidance step:
    # re-evaluate anchored energy after stepping -tap*grad for a range of taps.
    # If ANY tap lowers energy below E(0) -> direction is descending (overshoot);
    # if none do -> the gradient direction itself is not a descent direction.
    pipe._line_search_taps = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    _, guided_records = _run_condition(
        **base_kwargs,
        guidance_step_percents=[args.timing],
        latent_step_size=args.step_size,
        condition_label="diag_guided",
        save_video_path=OUTPUT_DIR / "diag_guided.mp4",
    )
    (OUTPUT_DIR / "diag_guided_records.json").write_text(json.dumps(guided_records, indent=2))

    # --- Report immediate effect ---
    base_map = {r["step"]: r["energy"] for r in baseline_records if r.get("energy") is not None}
    guid_steps = sorted(r["step"] for r in guided_records if r.get("was_guidance_step"))
    g_step = guid_steps[0] if guid_steps else -1

    print("\n" + "=" * 62)
    print(f"DIAGNOSTIC: single-step anchored guidance at step {g_step}")
    print("=" * 62)
    print(f"{'step':>5} {'base_E':>9} {'guid_E':>9} {'delta':>9}  {'note'}")
    for r in guided_records:
        s = r["step"]
        ge = r.get("energy")
        be = base_map.get(s)
        if ge is None or be is None:
            continue
        delta = ge - be
        note = ""
        if r.get("was_guidance_step"):
            note = "*GUIDANCE STEP*"
        elif g_step >= 0 and s == g_step + 1:
            note = "<- immediately after"
        print(f"{s:>5} {be:>9.5f} {ge:>9.5f} {delta:>+9.5f}  {note}")

    # Verdict on the immediate post-guidance step
    post = [r["energy"] - base_map[r["step"]]
            for r in guided_records
            if r.get("energy") is not None and r["step"] in base_map and r["step"] > g_step]
    if post:
        n_neg = sum(1 for d in post if d < 0)
        print(f"\nPost-guidance steps: {len(post)}, negative delta (energy dropped): "
              f"{n_neg}/{len(post)} = {n_neg/len(post)*100:.0f}%")
        print(f"Mean post-guidance delta: {sum(post)/len(post):+.6f} "
              f"(<0 means guidance lowered anchored energy vs baseline)")


if __name__ == "__main__":
    main()
