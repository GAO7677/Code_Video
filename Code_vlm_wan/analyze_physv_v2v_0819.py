#!/usr/bin/env python3
"""Summarize the exported PhysV V2V dataset without loading video tensors."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    task_counts = collections.Counter(s["task_type"] for s in samples)
    family_counts = collections.Counter(s["family_key"] for s in samples)
    variable_counts = collections.Counter(s["controlled_variable"] for s in samples)
    presence = collections.Counter()
    for sample in samples:
        p = Path(sample["sample_dir"])
        for rel in (
            "videos/rgb.mp4", "videos/masks.mp4", "videos/depth.mp4",
            "videos/trajectory.mp4", "videos/contacts.mp4", "context/context8.mp4",
            "context/context16.mp4", "raw/source_video.mp4", "raw/depth.npz",
            "raw/masks.npz", "raw/instance_ids.npz", "raw/trajectories.npz",
            "raw/states_xyzw.npz", "physics_supervision.npz", "contacts.json",
        ):
            presence[rel] += int((p / rel).is_file())
    first = Path(samples[0]["sample_dir"])
    npz_schema = {}
    try:
        import numpy as np
        for rel in ("raw/depth.npz", "raw/masks.npz", "raw/instance_ids.npz", "raw/trajectories.npz", "raw/states_xyzw.npz", "physics_supervision.npz"):
            z = np.load(first / rel, allow_pickle=False)
            npz_schema[rel] = {key: {"shape": list(z[key].shape), "dtype": str(z[key].dtype)} for key in z.files}
    except Exception as exc:
        npz_schema = {"error": str(exc)}
    result = {
        "dataset": manifest,
        "counts": {
            "task_type": dict(task_counts), "family_key": dict(family_counts),
            "controlled_variable": dict(variable_counts),
            "frame_count": dict(collections.Counter(s["frame_count"] for s in samples)),
            "dynamic_actor_count": dict(collections.Counter(s["dynamic_actor_count"] for s in samples)),
        },
        "ranges": {
            "contact_point_count": [min(s["contact_point_count"] for s in samples), max(s["contact_point_count"] for s in samples)],
            "controlled_value": {v: [min(s["controlled_value"] for s in samples if s["controlled_variable"] == v), max(s["controlled_value"] for s in samples if s["controlled_variable"] == v)] for v in variable_counts},
        },
        "file_presence": dict(presence),
        "first_sample_npz_schema": npz_schema,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"analysis={args.output}")
    print(f"samples={len(samples)} tasks={dict(task_counts)}")


if __name__ == "__main__":
    main()
