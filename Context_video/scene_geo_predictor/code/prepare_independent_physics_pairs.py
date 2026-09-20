"""Prepare independent door-history physics-only paired episodes.

This uses the repository's original Bullet replay through replay_audit.  It
does not render RGB or extract any visual feature.  Group assignment and all
geometry variants are fixed before any predictor result is read.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
BANK = PROJECT.parent / "original_pipeline_experiment_20260917"
AUDIT_ROOT = PROJECT.parent / "original_pipeline_audit_20260917"
WIDTHS_M = (0.46, 0.58, 0.70)
GROUPS = 8
SIM_SEED_BASE = 20261200
OUT_DEFAULT = PROJECT / "validation_20260920/paired_history_physics_v1"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def actor_metadata(obj):
    return {
        "object_id": obj.family_key,
        "role": obj.role,
        "dynamic": obj.dynamic,
        "shape": obj.shape,
        "size_m": obj.size,
        "mass_kg": obj.mass,
        "friction": obj.friction,
        "restitution": obj.restitution,
        "initial_position_m": obj.position,
        "initial_linear_velocity_mps": obj.linear_velocity,
        "initial_angular_velocity_radps": obj.angular_velocity,
    }


def make_group_plan():
    rng = np.random.default_rng(20260920)
    plans = []
    for group in range(GROUPS):
        # u[0] is replaced by the three fixed in-range widths.  The other
        # controls define one independent initial state shared by the group.
        u = rng.random(5)
        split = "train" if group < 6 else "dev"
        plans.append({
            "group_id": f"history_door_g{group:02d}",
            "split": split,
            "u_shared": u.tolist(),
            "widths_m": list(WIDTHS_M),
            "simulation_seed": SIM_SEED_BASE + group,
            "geometry_range": "in_train_door_width_range",
        })
    return plans


def load_generator():
    sys.path.insert(0, str(BANK))
    sys.path.insert(0, str(AUDIT_ROOT))
    import experiment
    import replay_audit
    experiment.g.SIM_DURATION_S = 49 / 30
    torch.set_num_threads(2)
    return experiment, replay_audit


def replay_case(experiment, replay_audit, plan, width, index):
    u = np.asarray(plan["u_shared"], dtype=np.float64)
    u[0] = (width - 0.38) / 0.36
    key = f"paired_{plan['split']}_door_g{index:02d}_w{int(round(width*100)):02d}"
    case = experiment.make_case("door", key, u)
    objects, positions, velocities, quats, corrections = replay_audit.replay(
        case, int(plan["simulation_seed"])
    )
    if positions.shape[0] != 49:
        raise ValueError(f"unexpected frame count {key}: {positions.shape}")
    dynamic = [i for i, obj in enumerate(objects) if obj.dynamic]
    if len(dynamic) != 1:
        raise ValueError(f"expected one dynamic object: {key}")
    dynamic_index = dynamic[0]
    static_names = [obj.name for obj in objects if not obj.dynamic]
    metadata = {
        "schema_version": "physics_only_paired_history_v1",
        "sample_id": key,
        "pair_group": plan["group_id"],
        "split": plan["split"],
        "family": "door",
        "opening_width_m": float(width),
        "geometry_range": plan["geometry_range"],
        "simulation": {
            "fps": 30,
            "sim_hz": int(experiment.g.SIM_HZ),
            "frame_count": 49,
            "pre_roll_s": float(case.blueprint.pre_roll_s),
        },
        "actors": {obj.name: actor_metadata(obj) for obj in objects},
        "scenario_spec": case.blueprint.metadata,
        "source": str(replay_audit.SOURCE),
        "rebound_corrections": int(corrections),
        "rgb_generated": False,
        "vggt_sam2_utonia_extracted": False,
        "post_rgb7_external_action_status": "UNKNOWN",
        "generator_note": (
            "Original Bullet replay only; no RGB/render/cache extraction. "
            "Check replay source for any correction event before physics claim."
        ),
    }
    context = np.concatenate([
        positions[:8, dynamic_index].reshape(-1),
        velocities[:8, dynamic_index].reshape(-1),
        quats[:8, dynamic_index].reshape(-1),
    ]).astype(np.float32)
    history_hash = hashlib.sha256(context.tobytes()).hexdigest()
    target = positions[8:49, dynamic_index].astype(np.float32)
    target_hash = hashlib.sha256(target.tobytes()).hexdigest()
    static_hash = json_sha([
        [obj.name, obj.position, obj.orientation_euler_deg, obj.size,
         obj.friction, obj.restitution]
        for obj in objects if not obj.dynamic
    ])
    return {
        "key": key,
        "group_id": plan["group_id"],
        "split": plan["split"],
        "width_m": float(width),
        "simulation_seed": int(plan["simulation_seed"]),
        "initial_u": u.tolist(),
        "dynamic_name": objects[dynamic_index].name,
        "static_names": static_names,
        "history_hash": history_hash,
        "target_hash": target_hash,
        "static_geometry_hash": static_hash,
        "rebound_corrections": int(corrections),
        "metadata": metadata,
        "objects": objects,
        "positions": positions,
        "velocities": velocities,
        "quats": quats,
        "target": target,
        "case": case,
    }


def write_record(output: Path, record):
    dest = output / "samples" / record["key"]
    if dest.exists():
        raise FileExistsError(dest)
    dest.joinpath("raw").mkdir(parents=True)
    metadata = record["metadata"]
    (dest / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    (dest / "blueprint.json").write_text(
        json.dumps(asdict(record["case"].blueprint), indent=2,
                   allow_nan=False) + "\n"
    )
    np.savez_compressed(
        dest / "raw/states_xyzw.npz",
        positions=record["positions"],
        linear_velocities=record["velocities"],
        quats=record["quats"],
        object_names=np.asarray([obj.name for obj in record["objects"]]),
        frame_times=np.arange(49, dtype=np.float32) / 30,
    )
    dynamic_index = [i for i, obj in enumerate(record["objects"])
                     if obj.dynamic][0]
    trajectory = {
        "object_names": [obj.name for obj in record["objects"]],
        "frame_times_s": (np.arange(8, dtype=np.float32) / 30).tolist(),
        "dynamic_name": record["dynamic_name"],
        "observed_positions": record["positions"][:8, dynamic_index].tolist(),
        "observed_velocities": record["velocities"][:8, dynamic_index].tolist(),
        "future_positions": record["target"].tolist(),
    }
    (dest / "raw/trajectories.json").write_text(
        json.dumps(trajectory, indent=2, allow_nan=False) + "\n"
    )
    receipt = {
        "schema": "physics_only_paired_history_receipt_v1",
        "key": record["key"],
        "group_id": record["group_id"],
        "original_bullet_replay": True,
        "rgb_generated": False,
        "future_used_for_geometry": False,
        "output_sha256": {
            name: file_sha(dest / name)
            for name in (
                "metadata.json", "blueprint.json",
                "raw/states_xyzw.npz", "raw/trajectories.json"
            )
        },
    }
    (dest / "replay.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    experiment, replay_audit = load_generator()
    plans = make_group_plan()
    records = []
    for index, plan in enumerate(plans):
        group_records = []
        for width in WIDTHS_M:
            record = replay_case(experiment, replay_audit, plan, width, index)
            write_record(output, record)
            group_records.append(record)
            records.append(record)
            print("PHYSICS_PAIR_READY", record["key"], flush=True)
        context = {r["history_hash"] for r in group_records}
        if len(context) != 1:
            raise AssertionError(f"history mismatch in {plan['group_id']}")
        print("PHYSICS_GROUP_READY", plan["group_id"], flush=True)
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    split_groups = {
        split: sorted(
            group for group, values in groups.items()
            if values[0]["split"] == split
        )
        for split in ("train", "dev")
    }
    pair_rows = []
    for group, values in sorted(groups.items()):
        values.sort(key=lambda row: row["width_m"])
        for left, right in zip(values, values[1:]):
            left_target = left["target"]
            right_target = right["target"]
            delta = float(np.linalg.norm(left_target - right_target, axis=-1).mean())
            pair_rows.append({
                "group_id": group,
                "split": values[0]["split"],
                "width_a_m": left["width_m"],
                "width_b_m": right["width_m"],
                "D_gt_m": delta,
                "history_equal": left["history_hash"] == right["history_hash"],
                "static_geometry_different": left["static_geometry_hash"] != right["static_geometry_hash"],
                "rebound_corrections": [
                    left["rebound_corrections"], right["rebound_corrections"]
                ],
            })
    source_files = [
        BANK / "experiment.py",
        AUDIT_ROOT / "replay_audit.py",
        Path(replay_audit.SOURCE) / "scripts/generate_v2v_context_demos.py",
    ]
    manifest = {
        "schema": "paired_history_data_manifest_v1",
        "status": "EXECUTED",
        "role": "physics_only_preparation_for_future_group_split",
        "groups_total": GROUPS,
        "episodes_total": len(records),
        "episodes_per_group": 3,
        "group_split": split_groups,
        "widths_m": list(WIDTHS_M),
        "records": [
            {key: value for key, value in record.items()
             if key not in {"metadata", "objects", "positions", "velocities",
                            "quats", "target", "case"}}
            for record in records
        ],
        "pair_rows": pair_rows,
        "group_history_bitwise_equal": all(
            row["history_equal"] for row in pair_rows
        ),
        "all_static_geometry_changed": all(
            row["static_geometry_different"] for row in pair_rows
        ),
        "future_equal_negative_pairs": [
            row for row in pair_rows if row["D_gt_m"] <= 1e-8
        ],
        "rgb_generated": False,
        "visual_frontend_extracted": False,
        "physics_generator": {
            "engine": "original Bullet replay_audit",
            "bank_root": str(BANK),
            "audit_root": str(AUDIT_ROOT),
            "cpu_preflight": True,
            "gpu_required": False,
            "no_engine_substitution": True,
            "post_rgb7_external_action_status": "UNKNOWN",
            "rebound_correction_status": (
                "recorded per episode; source protocol still requires review"
            ),
        },
        "source_sha256": {
            str(path): file_sha(path) for path in source_files
        },
        "fixed_before_model_results": True,
        "split_policy": (
            "6 groups train, 2 groups dev; all three geometry variants and "
            "all windows remain in the same group split."
        ),
        "not_a_formal_blind_test": True,
    }
    (output / "paired_history_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "generation_command.txt").write_text(
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python "
        f"{Path(__file__).resolve()} --output {output}\n"
    )
    print(json.dumps({
        "status": manifest["status"],
        "groups": manifest["groups_total"],
        "episodes": manifest["episodes_total"],
        "train_groups": len(split_groups["train"]),
        "dev_groups": len(split_groups["dev"]),
        "negative_pairs": len(manifest["future_equal_negative_pairs"]),
    }, indent=2))


if __name__ == "__main__":
    main()

