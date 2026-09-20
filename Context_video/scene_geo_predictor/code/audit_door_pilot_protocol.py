"""Audit the physical protocol metadata for the six-door CPU pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CASES = ["train_door_0712", "train_door_0660", "train_door_0366",
         "train_door_0437", "train_door_0955", "train_door_0891"]


def audit(root: Path, cases=CASES):
    rows, failures = [], []
    for key in cases:
        sample = root / "samples" / key
        metadata = json.loads((sample / "metadata.json").read_text())
        with np.load(sample / "raw/states_xyzw.npz", allow_pickle=False) as states:
            names = states["object_names"].astype(str).tolist()
            dynamic = [i for i, name in enumerate(names) if metadata["actors"][name].get("dynamic")]
            static = [i for i, name in enumerate(names) if not metadata["actors"][name].get("dynamic")]
            static_motion = [float(np.max(np.abs(states["positions"][:, i] - states["positions"][0, i]))) for i in static]
            dynamic_name = names[dynamic[0]] if len(dynamic) == 1 else None
            dynamic_actor = metadata["actors"].get(dynamic_name, {}) if dynamic_name else {}
            initial = {"position_m": dynamic_actor.get("initial_position_m"),
                       "linear_velocity_mps": dynamic_actor.get("initial_linear_velocity_mps"),
                       "angular_velocity_radps": dynamic_actor.get("initial_angular_velocity_radps"),
                       "mass_kg": dynamic_actor.get("mass_kg"),
                       "friction": dynamic_actor.get("friction"),
                       "restitution": dynamic_actor.get("restitution")}
        # There is no action/force/control stream in this replay metadata. It
        # would be unsafe to infer “no external force” from that absence.
        action_keys = sorted(k for k in metadata if any(token in k.lower() for token in ("force", "action", "control", "impulse")))
        row = {"case": key, "split": metadata.get("split"), "family": "door",
               "dynamic_count": len(dynamic), "dynamic_name": dynamic_name,
               "dynamic_shape": dynamic_actor.get("shape"),
               "initial_state_and_material": initial,
               "static_actor_count": len(static), "max_static_motion_m": max(static_motion, default=None),
               "rebound_corrections": int(metadata.get("rebound_corrections", 0)),
               "post_rgb7_external_action": "UNKNOWN",
               "action_force_metadata_keys": action_keys,
               "direct_state_rewrite_after_observation": "UNKNOWN",
               "source_generator_reexecuted_this_round": False}
        rows.append(row)
        if row["dynamic_count"] != 1 or row["dynamic_shape"] != "sphere" or row["max_static_motion_m"] > 1e-6:
            failures.append({"case": key, "reason": "not a single static-scene sphere protocol"})
    return {"status": "EXECUTED", "cases": rows, "failures": failures,
            "autonomous_physics_prediction_claim_allowed": False,
            "reason": "External force/action and post-observation state rewrite are not represented in these metadata files; no re-execution of source generator was authorized.",
            "scope": "door pilot only; barrier rebound correction is out of pilot and retained in its source metadata"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
