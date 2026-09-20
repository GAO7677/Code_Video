"""CPU analytic visible-geometry oracle for a small door pilot.

The output is intentionally privileged and has its own schema.  It is not a
replacement for RGB/VGGT/Utonia caches and is labelled partial when the
rendering metadata does not describe all room geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from align_observed_depth import crop_transform, morph3, ray_aabb_depth, resize_crop, world_rays
from audit_scene_geometry import boxes_for
from prepare_context import load_model_input, pixel_hash
from scene_token_cache import sha256
from trial_metrics import rotation_xyzw

FEATURE_DIM = 1386
TOKEN_COUNT = 1792
DOOR_PILOT_CASES = [
    "train_door_0712", "train_door_0660", "train_door_0366",
    "train_door_0437", "train_door_0955", "train_door_0891",
]


def sphere_depth(origin, rays, center, radius):
    q = origin - np.asarray(center, dtype=np.float64)
    a = np.sum(rays * rays, axis=-1)
    b = np.sum(rays * q, axis=-1)
    c = np.dot(q, q) - float(radius) ** 2
    disc = b * b - a * c
    near = (-b - np.sqrt(np.maximum(disc, 0.0))) / np.maximum(a, 1e-20)
    far = (-b + np.sqrt(np.maximum(disc, 0.0))) / np.maximum(a, 1e-20)
    hit = (disc >= 0) & (near > 0) & np.isfinite(near) & (far >= near)
    return np.where(hit, near, 0.0), hit


def obb_depth(origin, rays, box):
    rotation = np.asarray(box["rotation"], dtype=np.float64)
    local_origin = (origin - box["center"]) @ rotation
    local_rays = rays @ rotation
    return ray_aabb_depth(local_origin, local_rays, np.zeros(3), 2 * box["half"])[0:3:2]


def finite_ground_depth(origin, rays, bounds):
    """Finite z=0 plane; bounds=(xmin,xmax,ymin,ymax)."""
    xmin, xmax, ymin, ymax = map(float, bounds)
    t = np.full(rays.shape[:-1], np.inf, dtype=np.float64)
    np.divide(-origin[2], rays[..., 2], out=t, where=rays[..., 2] < -1e-12)
    p = origin + rays * t[..., None]
    hit = np.isfinite(t) & (t > 0) & (p[..., 0] >= xmin) & (p[..., 0] <= xmax)
    hit &= (p[..., 1] >= ymin) & (p[..., 1] <= ymax)
    return np.where(hit, t, 0.0), hit


def nearest_depth(candidates):
    """Choose the closest positive candidate, retaining source indices."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    stack = np.stack([np.where(hit, depth, np.inf) for depth, hit, _ in candidates])
    index = np.argmin(stack, axis=0)
    depth = np.take_along_axis(stack, index[None], axis=0)[0]
    valid = np.isfinite(depth)
    source = np.full(depth.shape, -1, dtype=np.int16)
    for i, (_, _, source_id) in enumerate(candidates):
        source[index == i] = source_id
    return np.where(valid, depth, 0.0), valid, source


def self_test():
    origin = np.array([0.0, 0.0, 2.0])
    rays = np.array([[[0.0, 0.0, -1.0], [0.2, 0.0, -1.0],
                      [2.0, 0.0, -1.0]]], dtype=np.float64)
    # A rotated box and a farther box prove nearest-source selection.
    angle = .31
    rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    near = {"center": np.array([0., 0., 1.]), "half": np.array([.4, .4, .4]), "rotation": rot}
    far = {"center": np.array([0., 0., .2]), "half": np.array([.4, .4, .4]), "rotation": np.eye(3)}
    n0, h0 = obb_depth(origin, rays, near)
    n1, h1 = obb_depth(origin, rays, far)
    depth, hit, source = nearest_depth([(n0, h0, 3), (n1, h1, 4)])
    assert bool(hit[0, 0]) and int(source[0, 0]) == 3 and np.isclose(depth[0, 0], n0[0, 0])
    assert not bool(hit[0, 2])
    sd, sh = sphere_depth(origin, rays, np.array([0., 0., 1.]), .2)
    assert bool(sh[0, 0]) and np.isclose(sd[0, 0], .8)
    gd, gh = finite_ground_depth(origin, rays, (-.5, .5, -.5, .5))
    assert bool(gh[0, 0]) and np.isclose(gd[0, 0], 2.) and not bool(gh[0, 2])
    # A finite platform gap is not filled by a ground plane.
    gd2, gh2 = finite_ground_depth(origin, rays, (1., 2., -.5, .5))
    assert not bool(gh2[0, 0])
    return {"status": "PASS", "checks": [
        "non-unit camera ray with rotated OBB", "nearest OBB over farther OBB",
        "sphere occlusion intersection", "no-intersection ray", "finite ground/gap"],
        "tolerance_m": 1e-12}


def floor_bounds(boxes, margin=2.0):
    corners = []
    for box in boxes:
        # Bounds are conservative and only describe the generated pilot's
        # declared fixture extent; room props are intentionally not invented.
        local = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) * box["half"]
        corners.append(local @ box["rotation"].T + box["center"])
    points = np.concatenate(corners, axis=0)
    return (float(points[:, 0].min() - margin), float(points[:, 0].max() + margin),
            float(points[:, 1].min() - margin), float(points[:, 1].max() + margin))


def build_case(root: Path, key: str, out_root: Path):
    context = root / "observed_context" / key
    context_report = json.loads((context / "report.json").read_text())
    input_path = Path(context_report["input_json"])
    rgb, times = load_model_input(input_path)
    with np.load(context / "context_geometry.npz", allow_pickle=False) as archive:
        geometry = {k: archive[k] for k in archive.files}
    with np.load(root / "observed_masks" / key / "observed_masks.npz", allow_pickle=False) as archive:
        per_object = archive["per_object"]
        union = archive["union_dynamic"]
    with np.load(root / "scene_features" / key / "scene_tokens.npz", allow_pickle=False) as archive:
        legacy = {k: archive[k] for k in archive.files}
    sample = root / "samples" / key
    metadata = json.loads((sample / "metadata.json").read_text())
    with np.load(sample / "raw/states_xyzw.npz", allow_pickle=False) as states:
        names = states["object_names"].astype(str).tolist()
        dynamic = [i for i, name in enumerate(names) if metadata["actors"][name].get("dynamic") is True]
        static_states = states["positions"][0].astype(float)
        static_quats = states["quats"][0].astype(float)
        dynamic_pos = states["positions"][:8, dynamic[0]].astype(float)
    if len(dynamic) != 1 or metadata["actors"][names[dynamic[0]]]["shape"] != "sphere":
        raise ValueError("oracle pilot requires one dynamic sphere")
    dynamic_actor = metadata["actors"][names[dynamic[0]]]
    radius = float(dynamic_actor["size_m"]["radius"])
    boxes = []
    for i, name in enumerate(names):
        actor = metadata["actors"][name]
        if actor.get("dynamic"):
            continue
        if actor["shape"] != "box":
            raise ValueError(f"unsupported static shape {actor['shape']}")
        boxes.append({"name": name, "center": static_states[i],
                      "half": np.array([actor["size_m"][k] for k in ("hx", "hy", "hz")], dtype=float),
                      "rotation": rotation_xyzw(static_quats[i])})
    transform = crop_transform(rgb.shape[1:3])
    masks = np.stack([np.stack([resize_crop(m, transform, is_mask=True) for m in frame])
                      for frame in per_object])
    union_processed = masks.any(axis=1)
    intrinsic = transform["affine"] @ geometry["camera_K"]
    h, w = transform["processed_hw"]
    origin, rays = world_rays(intrinsic, geometry["camera_world_to_view"], (h, w))
    bounds = floor_bounds(boxes)
    # The current metadata has no explicit rendered room mesh.  Include a
    # finite support floor only for door; for gap, omitting an unspecified floor
    # is safer than filling the gap and is why the output is partial.
    include_ground = metadata["scenario_spec"].get("controlled_variable") == "door_opening_width_m"
    frame_points, frame_valid, frame_source = [], [], []
    for frame in range(8):
        candidates = []
        for i, box in enumerate(boxes):
            near, hit = obb_depth(origin, rays, box)
            candidates.append((near, hit, i))
        if include_ground:
            gd, gh = finite_ground_depth(origin, rays, bounds)
            candidates.append((gd, gh, len(boxes)))
        sd, sh = sphere_depth(origin, rays, dynamic_pos[frame], radius)
        dynamic_id = len(boxes) + (1 if include_ground else 0)
        candidates.append((sd, sh, dynamic_id))
        depth, hit, source = nearest_depth(candidates)
        # The dynamic sphere participates in z-buffer ordering above, then is
        # removed using the observed mask.  This preserves occlusion behind it.
        static_valid = hit & ~morph3(union_processed[frame], dilate=True)
        points = origin + rays * depth[..., None]
        points[~static_valid] = 0
        frame_points.append(points.astype(np.float32)); frame_valid.append(static_valid); frame_source.append(source)
    points = np.stack(frame_points); valid = np.stack(frame_valid); source = np.stack(frame_source)
    # Reuse legacy source pixels wherever possible, but recompute their xyz
    # from the oracle.  No old coordinates or features are copied.
    legacy_ids = legacy["source_flat_indices"][legacy["scene_mask"]].astype(np.int64)
    valid_flat = valid.reshape(-1)
    common = legacy_ids[(legacy_ids >= 0) & (legacy_ids < valid_flat.size) & valid_flat[legacy_ids]]
    all_valid = np.flatnonzero(valid_flat)
    selected = common
    if len(selected) < TOKEN_COUNT:
        extra = all_valid[~np.isin(all_valid, selected, assume_unique=False)]
        selected = np.concatenate([selected, extra])
    selected = selected[:TOKEN_COUNT]
    out = out_root / key
    out.mkdir(parents=True, exist_ok=False)
    n = len(selected)
    xyz = points.reshape(-1, 3)[selected] if n else np.empty((0, 3), np.float32)
    xyz = np.pad(xyz, ((0, TOKEN_COUNT - n), (0, 0)))
    mask = np.zeros(TOKEN_COUNT, dtype=bool); mask[:n] = True
    source_indices = np.full(TOKEN_COUNT, -1, dtype=np.int64); source_indices[:n] = selected
    source_fyx = np.full((TOKEN_COUNT, 3), -1, dtype=np.int64)
    if n:
        source_fyx[:n] = np.stack(np.unravel_index(selected, valid.shape), axis=-1)
    features = np.zeros((TOKEN_COUNT, FEATURE_DIM), dtype=np.float16)
    confidence = np.zeros(TOKEN_COUNT, dtype=np.float32); confidence[:n] = 1.
    np.savez_compressed(out / "scene_tokens.npz", scene_features=features, scene_xyz=xyz,
                        scene_mask=mask, scene_confidence=confidence,
                        encoder_coords=xyz.copy(), source_flat_indices=source_indices,
                        source_frame_y_x=source_fyx,
                        observed_indices=np.arange(8, dtype=np.int64))
    output_hash = sha256(out / "scene_tokens.npz")
    report = {
        "schema": "oracle_partial_visible_geometry_v1", "cache_schema": "oracle_partial_visible_geometry_v1",
        "cache_version": 1, "status": "complete", "case": key,
        "privileged_geometry": True, "static_scene_gt_used": True,
        "geometry_source": "declared static OBBs + finite inferred door support plane + observed dynamic sphere for z-buffer",
        "geometry_scope": "partial_visible_geometry",
        "missing_geometry": ["rendered room/decorative meshes are not present in metadata; occlusion equivalence to RGB is not proven"],
        "observed_indices": list(range(8)), "observed_frame_times_s": times.tolist(),
        "input_json": str(input_path), "input_pixels_sha256": pixel_hash(rgb),
        "context_geometry": str(context / "context_geometry.npz"),
        "context_geometry_sha256": sha256(context / "context_geometry.npz"),
        "mask_path": str(root / "observed_masks" / key / "observed_masks.npz"),
        "mask_sha256": sha256(root / "observed_masks" / key / "observed_masks.npz"),
        "static_actor_count": len(boxes), "static_actor_names": [b["name"] for b in boxes],
        "dynamic_actor": names[dynamic[0]], "sphere_radius_m": radius,
        "finite_ground_included": include_ground, "finite_ground_bounds_m": list(bounds),
        "point_grid": [8, h, w], "valid_fraction_per_frame": valid.mean(axis=(1, 2)).tolist(),
        "source_pixel_reuse": {"legacy_candidate_count": int(len(legacy_ids)),
                               "common_valid_count": int(len(common)), "stored_count": int(n),
                               "sampling_budget": TOKEN_COUNT, "selection": "legacy valid source pixels then deterministic row-major fill"},
        "feature_shape": [TOKEN_COUNT, FEATURE_DIM], "scene_xyz_units": "meters",
        "scene_xyz_coordinate_frame": "observed simulator world frame via context camera RT",
        "feature_coordinate_row_mapping": "zero placeholder features; geometry-only predictor must ignore scene_features",
        "scene_mask_semantics": "valid analytic visible static token, not occupancy/free-space confidence",
        "future_rgb_used": False, "future_state_used": False, "future_contact_used": False,
        "output_sha256": output_hash, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", nargs="*", default=DOOR_PILOT_CASES)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    if args.root is None or args.output is None:
        parser.error("--root and --output are required unless --self-test is used")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    results, failures = [], []
    for key in args.cases:
        try:
            results.append(build_case(args.root.resolve(), key, args.output.resolve()))
            print("ORACLE_READY", key, flush=True)
        except Exception as exc:
            failures.append({"case": key, "reason": f"{type(exc).__name__}: {exc}"})
            print("ORACLE_BLOCKED", key, failures[-1]["reason"], flush=True)
    manifest = {"schema": "oracle_partial_visible_geometry_pilot_v1",
                "status": "EXECUTED" if results else "DATA_BLOCKED", "requested_cases": args.cases,
                "completed_cases": [r["case"] for r in results], "failures": failures,
                "privileged": True, "training_ready": False,
                "geometry_scope": "partial_visible_geometry; not a complete RGB-matched oracle",
                "source_root": str(args.root.resolve()),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2))
    if failures:
        # Partial completion is useful evidence, but callers can see the
        # failure manifest and decide whether to continue the pilot.
        return


if __name__ == "__main__":
    main()
