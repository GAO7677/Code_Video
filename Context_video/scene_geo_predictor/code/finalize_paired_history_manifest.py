"""Finalize the manifest for an already completed physics-only paired replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parent
BANK = PROJECT.parent / "original_pipeline_experiment_20260917"
AUDIT_ROOT = PROJECT.parent / "original_pipeline_audit_20260917"
WIDTHS_M = (0.46, 0.58, 0.70)
GROUPS = 8
SIM_SEED_BASE = 20261200


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def group_plan():
    rng = np.random.default_rng(20260920)
    plans = []
    for group in range(GROUPS):
        u = rng.random(5)
        plans.append({
            "group_id": f"history_door_g{group:02d}",
            "split": "train" if group < 6 else "dev",
            "u_shared": u.tolist(),
            "simulation_seed": SIM_SEED_BASE + group,
        })
    return plans


def record_from_sample(output, plan, width):
    key = f"paired_{plan['split']}_door_g{int(plan['group_id'][-2:]):02d}_w{int(round(width*100)):02d}"
    root = output / "samples" / key
    meta = json.loads((root / "metadata.json").read_text())
    with np.load(root / "raw/states_xyzw.npz", allow_pickle=False) as data:
        names = data["object_names"].astype(str).tolist()
        positions = data["positions"].astype(np.float32)
        velocities = data["linear_velocities"].astype(np.float32)
        quats = data["quats"].astype(np.float32)
    dynamic = [i for i, name in enumerate(names)
               if meta["actors"][name].get("dynamic")]
    if len(dynamic) != 1:
        raise ValueError(key)
    j = dynamic[0]
    context = np.concatenate([
        positions[:8, j].reshape(-1),
        velocities[:8, j].reshape(-1),
        quats[:8, j].reshape(-1),
    ])
    target = positions[8:49, j]
    static = [
        [name, actor["initial_position_m"], actor["size_m"],
         actor["friction"], actor["restitution"]]
        for name, actor in meta["actors"].items()
        if not actor.get("dynamic")
    ]
    return {
        "key": key,
        "group_id": plan["group_id"],
        "split": plan["split"],
        "width_m": width,
        "simulation_seed": int(plan["simulation_seed"]),
        "initial_u": [float((width-.38)/.36)] + plan["u_shared"][1:],
        "dynamic_name": names[j],
        "history_hash": hashlib.sha256(context.tobytes()).hexdigest(),
        "target_hash": hashlib.sha256(target.tobytes()).hexdigest(),
        "static_geometry_hash": json_sha(static),
        "rebound_corrections": int(meta.get("rebound_corrections", 0)),
        "rgb_generated": bool(meta.get("rgb_generated", False)),
        "vggt_sam2_utonia_extracted": bool(meta.get(
            "vggt_sam2_utonia_extracted", False)),
        "post_rgb7_external_action_status": meta.get(
            "post_rgb7_external_action_status", "UNKNOWN"),
        "output_root": str(root),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    plans = group_plan()
    records = []
    for plan in plans:
        for width in WIDTHS_M:
            records.append(record_from_sample(output, plan, width))
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    split_groups = {
        split: sorted(group for group, rows in groups.items()
                      if rows[0]["split"] == split)
        for split in ("train", "dev")
    }
    pair_rows = []
    for group, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["width_m"])
        for left, right in zip(rows, rows[1:]):
            with np.load(output / "samples" / left["key"] /
                         "raw/states_xyzw.npz", allow_pickle=False) as la, \
                 np.load(output / "samples" / right["key"] /
                         "raw/states_xyzw.npz", allow_pickle=False) as rb:
                li = np.flatnonzero(np.asarray(la["object_names"]).astype(str)
                                    == left["dynamic_name"])[0]
                ri = np.flatnonzero(np.asarray(rb["object_names"]).astype(str)
                                    == right["dynamic_name"])[0]
                d = np.linalg.norm(
                    la["positions"][8:49, li] - rb["positions"][8:49, ri],
                    axis=-1,
                )
            pair_rows.append({
                "group_id": group,
                "split": rows[0]["split"],
                "width_a_m": left["width_m"],
                "width_b_m": right["width_m"],
                "D_gt_m": float(d.mean()),
                "history_equal": left["history_hash"] == right["history_hash"],
                "static_geometry_different": (
                    left["static_geometry_hash"] != right["static_geometry_hash"]
                ),
                "rebound_corrections": [
                    left["rebound_corrections"], right["rebound_corrections"]
                ],
            })
    sys.path.insert(0, str(BANK))
    sys.path.insert(0, str(AUDIT_ROOT))
    import replay_audit
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
        "records": records,
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
        "source_sha256": {str(path): file_sha(path) for path in source_files},
        "fixed_before_model_results": True,
        "split_policy": (
            "6 groups train, 2 groups dev; all geometry variants and "
            "trajectory windows remain in the same group split."
        ),
        "not_a_formal_blind_test": True,
        "finalization_note": (
            "All 24 replay artifacts were completed before this manifest was "
            "written; the prior run stopped only on a missing source-path hash."
        ),
    }
    (output / "paired_history_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": manifest["status"],
        "groups": manifest["groups_total"],
        "episodes": manifest["episodes_total"],
        "train_groups": len(split_groups["train"]),
        "dev_groups": len(split_groups["dev"]),
        "negative_pairs": len(manifest["future_equal_negative_pairs"]),
        "history_equal": manifest["group_history_bitwise_equal"],
    }, indent=2))


if __name__ == "__main__":
    main()

