#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


INPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons")
HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
RESULT_ROOT = Path("/data/gaoya/agent-data/outputs/test5_gt_wmreward/inputs")
ALLOWLIST = Path("/data/gaoya/agent-data/outputs/test5_gt_wmreward/input_jsons.txt")
SUMMARY = Path("/data/gaoya/agent-data/outputs/test5_gt_wmreward/summary.json")
WEB_RESULT = HUB_ROOT / "test5-wmreward-range" / "gt_wmreward.json"

CASE_STEMS = [
    "0613pybullet_sample_000301_w000",
    "0613pybullet_sample_001460_w002",
    "0613pybullet_sample_000331_w001",
    "0613pybullet_sample_001455_w000",
    "0613pybullet_sample_000336_w001",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end",
    "physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px",
    "phyco_kubric_ball_drop_soft_v4_2025-09-05_0144a4",
    "phyco_kubric_ball_drop_soft_v4_2025-09-06_46dd58",
    "phyco_kubric_ball_drop_soft_v4_2025-09-06_d1ceac",
    "phyco_kubric_ball_drop_v2_2025-09-04_018559",
    "phyco_kubric_ball_drop_v3_2025-11-06_010b34",
    "phyco_kubric_ball_wall_collision_2025-08-08_00ac15",
    "phyco_kubric_cube_deform_soft_v2_noeff_2025-09-08_fe6f35",
    "phyco_kubric_friction_slide_flat_force_v3_2025-10-07_003c2c",
    "phyco_kubric_friction_slide_flat_v2_2025-10-08_ffb0b4",
    "phyco_kubric_jenga_force_2025-09-29_00276c",
    "phyco_kubric_pool_table_force_2025-09-27_fef01f",
]


def prepare() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ALLOWLIST.parent.mkdir(parents=True, exist_ok=True)
    input_paths: list[str] = []
    for stem in CASE_STEMS:
        input_json = INPUT_ROOT / f"{stem}.json"
        source = json.loads(input_json.read_text(encoding="utf-8"))
        gt_video = HUB_ROOT / "gallery" / "media" / "_source" / stem / "gt_49f_30fps.mp4"
        output_video = RESULT_ROOT / f"{stem}.mp4"
        if output_video.is_symlink() or output_video.exists():
            output_video.unlink()
        output_video.symlink_to(gt_video.resolve())
        payload = {
            "input_json": str(input_json.resolve()),
            "input_video": source["input_video"],
            "source_video": source["source_video"],
            "input_caption": source["input_caption"],
            "output_video": str(output_video),
            "method": "test5_ground_truth_49f_30fps",
        }
        (RESULT_ROOT / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        input_paths.append(str(input_json.resolve()))
    ALLOWLIST.write_text("\n".join(input_paths) + "\n", encoding="utf-8")
    print(f"Prepared {len(input_paths)} GT cases in {RESULT_ROOT}")


def export() -> None:
    cases: dict[str, dict[str, float | str]] = {}
    for stem in CASE_STEMS:
        payload = json.loads((RESULT_ROOT / f"{stem}.json").read_text(encoding="utf-8"))
        metric = payload.get("wmreward")
        if not isinstance(metric, dict) or "surprise" not in metric:
            raise RuntimeError(f"Missing WMReward result for {stem}")
        cases[stem] = {
            "surprise": float(metric["surprise"]),
            "similarity": float(metric["similarity"]),
        }
    output = {
        "protocol": {
            "video": "dashboard GT clip, 49 frames at 30 FPS",
            "model": "vitg384",
            "img_size": 384,
            "window_size": 16,
            "context_frames": 8,
            "stride": 8,
            "metric_direction": "lower surprise is better",
        },
        "cases": cases,
    }
    WEB_RESULT.parent.mkdir(parents=True, exist_ok=True)
    WEB_RESULT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(cases)} GT scores to {WEB_RESULT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "export"))
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else export()


if __name__ == "__main__":
    main()
