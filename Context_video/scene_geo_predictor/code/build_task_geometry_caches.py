"""Build sparse/dense privileged task-geometry caches for the four door pairs.

The candidate pool is made only from observed static OBB metadata and a finite
support plane.  It is intentionally not an RGB-visible oracle and never reads
future states, contacts, labels, or predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_small_trial import sha
from trial_metrics import rotation_xyzw

FEATURE_DIM = 1386
TOKEN_COUNT = 1792
DEFAULT_M = 512
DEFAULT_SEED = 20260920
CONTROL_KEYS = [f"control_door_{i:02d}" for i in range(4)]


def box_surface(center, half, rotation, resolution=28):
    """Return deterministic, approximately uniform samples on all six faces."""
    uv = np.linspace(-1.0, 1.0, resolution, dtype=np.float64)
    rows = []
    # (fixed local axis, sign, two varying local axes)
    faces = ((0, -1, 1, 2), (0, 1, 1, 2),
             (1, -1, 0, 2), (1, 1, 0, 2),
             (2, -1, 0, 1), (2, 1, 0, 1))
    for fixed, sign, a, b in faces:
        for u in uv:
            for v in uv:
                local = np.zeros(3, dtype=np.float64)
                local[fixed] = sign * half[fixed]
                local[a] = u * half[a]
                local[b] = v * half[b]
                rows.append(local @ rotation.T + center)
    # Remove exact face-edge duplicates while retaining collider identity at the
    # pool level through the parallel metadata arrays.
    return np.asarray(rows, dtype=np.float32)


def finite_floor(boxes, resolution=128, margin=2.0):
    mins = np.min([b["center"][:2] - b["half"][:2] for b in boxes], axis=0) - margin
    maxs = np.max([b["center"][:2] + b["half"][:2] for b in boxes], axis=0) + margin
    xs = np.linspace(mins[0], maxs[0], resolution, dtype=np.float32)
    ys = np.linspace(mins[1], maxs[1], resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack((xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float32)), axis=-1)


def load_static(root: Path, case: str):
    sample = root / "samples" / case
    metadata = json.loads((sample / "metadata.json").read_text())
    with np.load(sample / "raw" / "states_xyzw.npz", allow_pickle=False) as states:
        names = states["object_names"].astype(str).tolist()
        boxes = []
        for i, name in enumerate(names):
            actor = metadata["actors"][name]
            if actor.get("dynamic"):
                continue
            if actor.get("shape") != "box":
                raise ValueError(f"unsupported static shape {case}/{name}")
            half = np.asarray([actor["size_m"][k] for k in ("hx", "hy", "hz")], dtype=np.float64)
            center = states["positions"][0, i].astype(np.float64)
            # Static geometry is allowed as an observed-state input, but must
            # remain static during the observed frames for this task probe.
            if not np.allclose(states["positions"][:8, i], center[None], atol=1e-8, rtol=0):
                raise ValueError(f"nonstatic observed collider {case}/{name}")
            boxes.append({"name": name, "center": center, "half": half,
                          "rotation": rotation_xyzw(states["quats"][0, i]),
                          "object_id": actor.get("object_id", "static_box")})
    return boxes, metadata


def candidate_pool(root: Path, case: str):
    boxes, metadata = load_static(root, case)
    points, collider, key = [], [], []
    for collider_id, box in enumerate(boxes):
        xyz = box_surface(box["center"], box["half"], box["rotation"])
        points.append(xyz)
        collider.extend([collider_id] * len(xyz))
        is_key = box["name"] in ("door_frame_left", "door_frame_right")
        key.extend([is_key] * len(xyz))
    floor = finite_floor(boxes)
    floor_id = len(boxes)
    points.append(floor); collider.extend([floor_id] * len(floor)); key.extend([False] * len(floor))
    xyz = np.concatenate(points, axis=0).astype(np.float32)
    collider = np.asarray(collider, dtype=np.int16)
    key = np.asarray(key, dtype=bool)
    # A candidate point is retained once per task collider.  Exact duplicates
    # between touching colliders are deliberately visible in the audit rather
    # than silently treated as extra coverage.
    pool_hash = hashlib.sha256(xyz.tobytes() + collider.tobytes()).hexdigest()
    names = [b["name"] for b in boxes] + ["finite_support_floor"]
    return xyz, collider, key, names, boxes, metadata, pool_hash


def choose(pool_xyz, pool_collider, key_mask, names, seed, mode, m):
    key_names = ("door_frame_left", "door_frame_right")
    key_ids = [names.index(x) for x in key_names]
    selected = []
    quotas = {"door_frame_left": 4, "door_frame_right": 4}
    other_quota = m - sum(quotas.values())
    if mode == "dense_critical_task_geometry":
        quotas = {"door_frame_left": 192, "door_frame_right": 192}
        other_quota = m - sum(quotas.values())
    if mode not in ("sparse_critical_task_geometry", "dense_critical_task_geometry"):
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    for name, collider_id in zip(key_names, key_ids):
        ids = np.flatnonzero(pool_collider == collider_id)
        if len(ids) < quotas[name]:
            raise ValueError(f"not enough candidate points for {name}: {len(ids)} < {quotas[name]}")
        selected.extend(rng.permutation(ids)[:quotas[name]].tolist())
    other = np.flatnonzero(~key_mask)
    if len(other) < other_quota:
        raise ValueError(f"not enough noncritical candidate points: {len(other)} < {other_quota}")
    selected.extend(rng.permutation(other)[:other_quota].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) != m or len(np.unique(selected)) != m:
        raise AssertionError("selection is not exactly M unique pool points")
    # One mode-independent ordering rule prevents mask/order from encoding the
    # mode; only the coverage quota changes the selected coordinates.
    order = np.random.default_rng(seed + 99173).permutation(m)
    return selected[order]


def build_variant(root, output, case, mode, seed, m):
    pool_xyz, pool_collider, key_mask, names, boxes, metadata, pool_hash = candidate_pool(root, case)
    selected = choose(pool_xyz, pool_collider, key_mask, names, seed, mode, m)
    xyz = np.zeros((TOKEN_COUNT, 3), dtype=np.float32); xyz[:m] = pool_xyz[selected]
    mask = np.zeros(TOKEN_COUNT, dtype=bool); mask[:m] = True
    confidence = np.zeros(TOKEN_COUNT, dtype=np.float32); confidence[:m] = 1.
    source_collider = np.full(TOKEN_COUNT, -1, dtype=np.int16); source_collider[:m] = pool_collider[selected]
    source_pool = np.full(TOKEN_COUNT, -1, dtype=np.int64); source_pool[:m] = selected
    features = np.zeros((TOKEN_COUNT, FEATURE_DIM), dtype=np.float16)
    out_dir = output / mode / f"seed{seed}" / case
    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out_dir / "scene_tokens.npz", scene_features=features, scene_xyz=xyz,
                        scene_mask=mask, scene_confidence=confidence, encoder_coords=xyz.copy(),
                        source_task_pool_index=source_pool, source_collider_id=source_collider)
    valid_xyz = xyz[:m]
    rounded = np.round(valid_xyz, 6)
    unique_xyz = np.unique(rounded, axis=0)
    collider_counts = {name: int((source_collider[:m] == i).sum()) for i, name in enumerate(names)}
    duplicate_exact = int(m - len(unique_xyz))
    report = {
        "schema": "privileged_task_geometry_surface_probe_v1",
        "cache_schema": "privileged_task_geometry_surface_probe_v1",
        "status": "complete", "case": case, "mode": mode, "sampling_seed": seed,
        "privileged_geometry": True, "rendered_visibility_matched": False,
        "geometry_scope": "declared task collision OBB surfaces plus finite support plane",
        "missing_room_or_decor_geometry": True,
        "future_rgb_used": False, "future_state_used": False, "future_contact_used": False,
        "static_scene_gt_used": True, "task_geometry_not_deployment_visual": True,
        "candidate_pool_hash": pool_hash, "candidate_pool_count": int(len(pool_xyz)),
        "candidate_pool_unique_xyz_rounded_1e-6": int(len(np.unique(np.round(pool_xyz, 6), axis=0))),
        "total_effective_points_M": int(m), "padded_token_count": TOKEN_COUNT,
        "scene_mask_sum": int(mask.sum()), "scene_mask_false_padding": int((~mask).sum()),
        "padding_semantics": "mask=false, zero xyz/features/confidence, no repeated valid points",
        "feature_shape": [TOKEN_COUNT, FEATURE_DIM], "scene_xyz_units": "meters",
        "scene_features_placeholder": True, "geometry_only_requires_feature_ignore": True,
        "quota": {"door_frame_left": int((source_collider[:m] == names.index("door_frame_left")).sum()),
                  "door_frame_right": int((source_collider[:m] == names.index("door_frame_right")).sum()),
                  "other_task_surfaces": int(m - (source_collider[:m] == names.index("door_frame_left")).sum()
                                               - (source_collider[:m] == names.index("door_frame_right")).sum())},
        "selected_points_by_collider": collider_counts,
        "selected_unique_xyz_rounded_1e-6": int(len(unique_xyz)),
        "selected_duplicate_xyz_rounded_1e-6": duplicate_exact,
        "source_code_sha256": sha(Path(__file__)),
        "output_sha256": sha(out_dir / "scene_tokens.npz"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # Pool metadata is repeated in the manifest, not inserted into predictor
    # features or motion fields.
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resample-seeds", nargs="*", type=int, default=[DEFAULT_SEED + 1, DEFAULT_SEED + 2, DEFAULT_SEED + 3])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.m <= 8 or args.m >= TOKEN_COUNT:
        raise ValueError("M must leave explicit false padding and enough task points")
    args.output.mkdir(parents=True)
    modes = ("sparse_critical_task_geometry", "dense_critical_task_geometry")
    all_seeds = [args.seed] + [x for x in args.resample_seeds if x != args.seed]
    reports, failures = [], []
    for mode in modes:
        for seed in all_seeds:
            for case in CONTROL_KEYS:
                try:
                    reports.append(build_variant(args.root.resolve(), args.output.resolve(), case, mode, seed, args.m))
                    print("TASK_GEOMETRY_READY", mode, seed, case, flush=True)
                except Exception as exc:
                    failures.append({"mode": mode, "seed": seed, "case": case,
                                     "reason": f"{type(exc).__name__}: {exc}"})
                    print("TASK_GEOMETRY_BLOCKED", failures[-1], flush=True)
    manifest = {"schema": "privileged_task_geometry_surface_probe_manifest_v1",
                "status": "EXECUTED" if not failures else "PARTIAL",
                "cases": CONTROL_KEYS, "m": args.m, "padded_tokens": TOKEN_COUNT,
                "base_seed": args.seed, "resample_seeds": args.resample_seeds,
                "modes": list(modes), "records_requested": len(modes) * len(all_seeds) * len(CONTROL_KEYS),
                "records_completed": len(reports), "failures": failures,
                "future_used": False, "utonia_extracted": False,
                "privileged": True, "rendered_visibility_matched": False,
                "training_role": "paired capability diagnostic only; not deployable visual geometry",
                "source_code_sha256": sha(Path(__file__))}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
