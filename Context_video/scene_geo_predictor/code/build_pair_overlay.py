#!/usr/bin/env python3
"""Build an auditable overlay for one independent-history paired probe.

The paired-history samples are physics-only (no source RGB).  This script
prepares a copy for the existing CPU Blender renderer and later composites the
GT/predicted paths over that canonical rerender.  It never modifies a source
sample, cache, checkpoint, or the existing hub page.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parent
VAL = PROJECT / "validation_20260920"
RUN = VAL / "group_holdout_training_1500_v1"
PAIRS = VAL / "paired_history_physics_v1" / "samples"
OLD_CAMERA_SAMPLE = Path("/data/gaoya/AAA_test_video/physv_v2v_0819/samples/scene_door_frame_ball_w046")
OUT = VAL / "paired_pair_overlay_v1"
SEED = 20262001
GROUP = "history_door_g01"
WIDTHS = (0.46, 0.70)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_key(width: float) -> str:
    return f"paired_train_door_g01_w{int(round(width * 100)):02d}"


def sorted_records():
    manifest = read_json(PAIRS.parent / "paired_history_data_manifest.json")
    return sorted(manifest["records"], key=lambda r: (r["group_id"], r["width_m"])), manifest


def pair_row(rows, arm: str, width_a: float, width_b: float, seed: int):
    for row in rows:
        if (row["arm"] == arm and row["group_id"] == GROUP and
                abs(float(row["width_a_m"]) - width_a) < 1e-8 and
                abs(float(row["width_b_m"]) - width_b) < 1e-8 and
                int(row["sampling_seed"]) == seed):
            return row
    raise KeyError((arm, width_a, width_b, seed))


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    render_inputs = OUT / "render_inputs"
    render_inputs.mkdir()
    canonical = read_json(OLD_CAMERA_SAMPLE / "metadata.json")

    records, manifest = sorted_records()
    keys = [sample_key(w) for w in WIDTHS]
    arrays = np.load(RUN / "predictions.npz", allow_pickle=False)
    eval_seeds = arrays["eval_seeds"].astype(int).tolist()
    seed_index = eval_seeds.index(SEED)
    all_index = {r["key"]: i for i, r in enumerate(records)}

    target = arrays["targets"]
    geom = arrays["geometry_dense_resample_eval16"][seed_index]
    motion = arrays["motion_only_eval16"][seed_index]
    payload = {
        "schema": "paired_pair_overlay_payload_v1",
        "source": {
            "run": str(RUN), "manifest": str(PAIRS.parent / "paired_history_data_manifest.json"),
            "predictions": str(RUN / "predictions.npz"), "sample_keys": keys,
            "source_rgb_available": False,
            "render_status": "canonical_rerender_from_physics_states",
        },
        "selection": {"group_id": GROUP, "width_a_m": WIDTHS[0], "width_b_m": WIDTHS[1],
                      "sampling_seed": SEED, "prediction_seed_index": seed_index},
        "aggregate": {"scope": "train/nondegenerate/new_sampling; 4 histories x 3 width pairs x 16 seeds",
                       "pair_count": 144, "D_gt_m": 0.03191, "D_pred_m": 0.00035,
                       "R_delta": 1.00187},
        "arms": {},
    }

    for width, key in zip(WIDTHS, keys):
        sample = PAIRS / key
        with np.load(sample / "raw" / "states_xyzw.npz", allow_pickle=False) as state_npz:
            states = {k: state_npz[k] for k in state_npz.files}
        names = states["object_names"].astype(str).tolist()
        idx = all_index[key]
        sample_out = render_inputs / key
        sample_out.mkdir()
        metadata = read_json(sample / "metadata.json")
        metadata["family_key"] = "SCENE_DOOR_FRAME_BALL"
        metadata["camera"] = canonical["camera"]
        metadata["render_provenance"] = {
            "kind": "canonical_physics_rerender_for_overlay",
            "source_sample": str(sample),
            "camera_source": str(OLD_CAMERA_SAMPLE / "metadata.json"),
            "not_original_rgb": True,
        }
        (sample_out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        trajectories = {"object_names": names, "frame_times_s": states["frame_times"].astype(float).tolist()}
        for i, name in enumerate(names):
            trajectories[f"{name}_positions"] = states["positions"][:, i].astype(float).tolist()
            trajectories[f"{name}_rotations"] = states["quats"][:, i].astype(float).tolist()
        (sample_out / "trajectories.json").write_text(json.dumps(trajectories), encoding="utf-8")

        def path_list(a):
            return a[idx].astype(float).tolist()

        payload["arms"][key] = {
            "sample_key": key, "width_m": width, "sample_dir": str(sample),
            "render_input": str(sample_out), "dynamic_object": names[0],
            "observed_positions": states["positions"][:8, 0].astype(float).tolist(),
            "target_positions": states["positions"][8:49, 0].astype(float).tolist(),
            "geometry_dense_resample": path_list(geom), "motion_only": path_list(motion),
        }

    # Pair-level exact values, plus all non-degenerate train rows for browsing.
    with (RUN / "pair_eval.csv").open(newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    exact = {}
    for arm in ("motion_only", "geometry_dense_resample", "analytic_cv"):
        exact[arm] = pair_row(pair_rows, arm, WIDTHS[0], WIDTHS[1], SEED)
    payload["selection"]["metrics"] = exact
    payload["available_pair_rows"] = [r for r in pair_rows if r["split"] == "train" and r["nondegenerate"] == "True"]
    (OUT / "data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.txt").write_text(
        "This directory is a derived visualization only. The paired-history samples have no original RGB/MP4.\n"
        "The rendered backgrounds are canonical CPU rerenders from physics states and the old door camera metadata.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "keys": keys, "seed_index": seed_index}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare",))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()


if __name__ == "__main__":
    main()
