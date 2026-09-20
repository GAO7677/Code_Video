"""CPU-only mutation replay for the observation/supervision boundary.

The temporary sample keeps RGB0-RGB7, context geometry, aligned points and the
scene-token cache unchanged.  It mutates only positions/quaternions/velocities
after RGB7 and adds sentinel future/event files.  The test then reconstructs a
batch through the real pilot loader and compares the eight Predictor inputs.
It intentionally does not claim to exercise the RGB/VGGT/SAM2/Utonia front end.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from predictor_pilot import build_batch


INPUTS = (
    "motion", "object_mask", "last_position", "last_velocity", "future_dt",
    "scene_xyz", "scene_features", "scene_mask",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "small_trial_120")
    parser.add_argument("--case", default="train_barrier_0057")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.root.resolve()
    np.random.seed(20260919)
    torch.manual_seed(20260919)

    baseline, baseline_target, baseline_records = build_batch(
        source_root, source_root / "scene_features", [args.case])
    with tempfile.TemporaryDirectory(prefix="future_replay_", dir="/data/gaoya/agent-data/cache") as name:
        temp = Path(name)
        # Copy only the small context directory so its provenance report can be
        # redirected without touching the existing cache tree.  The two large
        # read-only directories are symlinked and never modified.
        (temp / "observed_context").mkdir()
        shutil.copytree(source_root / "observed_context" / args.case,
                        temp / "observed_context" / args.case)
        for directory in ("aligned_scene", "scene_features"):
            os.symlink(source_root / directory, temp / directory, target_is_directory=True)
        sample = temp / "samples" / args.case
        (sample / "raw").mkdir(parents=True)
        source_sample = source_root / "samples" / args.case
        for filename in ("metadata.json", "replay.json"):
            shutil.copy2(source_sample / filename, sample / filename)
        if (source_sample / "raw/observed_trajectories.json").exists():
            shutil.copy2(source_sample / "raw/observed_trajectories.json",
                         sample / "raw/observed_trajectories.json")
        with np.load(source_sample / "raw/states_xyzw.npz", allow_pickle=False) as archive:
            changed = {key: archive[key].copy() for key in archive.files}
        # Preserve all observation frames exactly; mutate only RGB8-RGB48.
        changed["positions"][8:, 0, 0] += 0.37
        changed["positions"][8:, 0, 1] -= 0.11
        changed["positions"][8:, 0, 2] += 0.23
        changed["linear_velocities"][8:, 0] *= -0.5
        changed["quats"][8:, 0, 0] *= -1.0
        np.savez_compressed(sample / "raw/states_xyzw.npz", **changed)
        # These files represent the allowed mutation targets.  The current
        # predictor loader has no code path that opens either one.
        future_rgb = sample / "raw/future_rgb"
        future_rgb.mkdir()
        (future_rgb / "rgb_08.png").write_bytes(b"mutated-future-rgb")
        (sample / "raw/future_events.json").write_text(json.dumps({"collision": [8, 17]}))

        report_path = temp / "observed_context" / args.case / "report.json"
        report = json.loads(report_path.read_text())
        report["sample"] = str(sample)
        report_path = temp / "observed_context_override.json"
        report_path.write_text(json.dumps(report))
        # build_batch reads the report next to context_geometry.npz; this is a
        # temporary copy, so no existing report or cache is ever changed.
        real_report = temp / "observed_context" / args.case / "report.json"
        real_report.unlink()
        real_report.write_text(json.dumps(report))
        mutated, mutated_target, mutated_records = build_batch(
            temp, temp / "scene_features", [args.case])

        input_equal = {}
        input_diff = {}
        for key in INPUTS:
            equal = bool(torch.equal(baseline[key], mutated[key]))
            input_equal[key] = equal
            if baseline[key].dtype == torch.bool:
                max_abs = float((baseline[key] != mutated[key]).any())
                max_rel = max_abs
            else:
                delta = (baseline[key].float() - mutated[key].float()).abs()
                max_abs = float(delta.max()) if delta.numel() else 0.0
                scale = baseline[key].float().abs().clamp_min(1e-12)
                max_rel = float((delta / scale).max()) if delta.numel() else 0.0
            input_diff[key] = {"max_abs": max_abs, "max_rel": max_rel,
                               "atol": 0.0, "rtol": 0.0}
        target_changed = bool(not torch.equal(baseline_target, mutated_target))
        supervision_record_changed = baseline_records[0]["supervision_sha256"] != mutated_records[0]["supervision_sha256"]
        result = {
            "status": "PASS_CPU_OBSERVATION_BOUNDARY",
            "case": args.case,
            "seed": 20260919,
            "temporary_root": str(temp),
            "input_shapes": {key: list(baseline[key].shape) for key in INPUTS},
            "input_equal": input_equal,
            "input_diff": input_diff,
            "all_eight_inputs_equal": all(input_equal.values()),
            "target_shape": list(baseline_target.shape),
            "target_changed": target_changed,
            "supervision_record_changed": supervision_record_changed,
            "future_rgb_path_consumed_by_build_batch": False,
            "future_event_path_consumed_by_build_batch": False,
            "scope": "CPU batch reconstruction only; no RGB/VGGT/SAM2/Utonia rerun",
            "complete_end_to_end": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["all_eight_inputs_equal"] or not result["target_changed"]:
        raise SystemExit("future replacement CPU boundary check failed")


if __name__ == "__main__":
    main()
