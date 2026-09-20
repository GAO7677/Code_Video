"""Build observation-only geometry-only caches for old and sphere scales.

This deliberately does not run Utonia.  Both arms use zero scene_features and
the same legacy source pixel ids wherever the scale-only validity permits it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from align_observed_depth import static_points
from alignment_compare import observed_sphere_scale, processed_inputs
from prepare_small_trial import sha

TOKEN_COUNT = 1792
FEATURE_DIM = 1386
SEED = 20260920


def pad_tokens(xyz, ids, frame_yx, mask, confidence=None):
    count = min(len(ids), TOKEN_COUNT)
    out_xyz = np.zeros((TOKEN_COUNT, 3), dtype=np.float32)
    out_ids = np.full(TOKEN_COUNT, -1, dtype=np.int64)
    out_fyx = np.full((TOKEN_COUNT, 3), -1, dtype=np.int64)
    out_mask = np.zeros(TOKEN_COUNT, dtype=bool)
    out_conf = np.zeros(TOKEN_COUNT, dtype=np.float32)
    if count:
        out_xyz[:count] = np.asarray(xyz[:count], dtype=np.float32)
        out_ids[:count] = np.asarray(ids[:count], dtype=np.int64)
        out_fyx[:count] = np.asarray(frame_yx[:count], dtype=np.int64)
        out_mask[:count] = True
        if confidence is not None:
            out_conf[:count] = np.asarray(confidence[:count], dtype=np.float32)
        else:
            out_conf[:count] = 1.0
    return out_xyz, out_ids, out_fyx, out_mask, out_conf


def build_one(root, key, mode, output):
    report, rgb, times, geometry, masks, union, raw, intrinsic, depth = processed_inputs(root, key)
    context_dir = root / "observed_context" / key
    raw_path = root / "raw_probe" / key / "raw_vggt.npz"
    mask_path = root / "observed_masks" / key / "observed_masks.npz"
    geometry_path = context_dir / "context_geometry.npz"
    old_dir = root / "scene_features" / key
    old_token_path = old_dir / "scene_tokens.npz"
    old_alignment_path = root / "aligned_scene" / key / "coarse_static_points.npz"
    old_alignment_report_path = old_alignment_path.parent / "report.json"
    with np.load(old_token_path, allow_pickle=False) as archive:
        old = {k: archive[k] for k in archive.files}
    with np.load(old_alignment_path, allow_pickle=False) as archive:
        aligned = {k: archive[k] for k in archive.files}
    old_ids = old["source_flat_indices"][old["scene_mask"]].astype(np.int64)
    if len(old_ids) == 0 or len(old_ids) > TOKEN_COUNT:
        raise ValueError(f"unexpected legacy token count for {key}: {len(old_ids)}")
    valid_flat = np.ones(depth.shape[0] * depth.shape[1] * depth.shape[2], dtype=bool)
    if mode == "estimated_aabb":
        alignment = json.loads(old_alignment_report_path.read_text())["alignment"]
        scale = float(alignment["scale_midpoint"])
        coords = aligned["world_midpoint"].reshape(-1, 3).astype(np.float32)
        valid = aligned["static_valid"].reshape(-1)
        scale_report = {"method": "existing_aabb_interval_midpoint",
                        "scale_midpoint": scale,
                        "legacy_scale_interval": alignment.get("robust_interval"),
                        "anchor_pixels": alignment.get("anchor_pixels"),
                        "failure_reasons": alignment.get("failure_reasons", [])}
        source_code = Path(__file__).with_name("align_observed_depth.py")
        source_description = "existing observed dynamic AABB robust interval midpoint"
    elif mode == "estimated_sphere":
        sphere_report = observed_sphere_scale(depth, masks[:, 0], geometry, intrinsic)
        scale_report = sphere_report
        if not sphere_report["accepted"]:
            raise ValueError(f"sphere calibration rejected: {sphere_report['failure_reasons']}")
        coords_map, valid_map = static_points(depth, union, intrinsic,
                                               geometry["camera_world_to_view"],
                                               sphere_report["scale_midpoint"])
        coords = coords_map.reshape(-1, 3).astype(np.float32)
        valid = valid_map.reshape(-1)
        scale = float(sphere_report["scale_midpoint"])
        source_code = Path(__file__).with_name("alignment_compare.py")
        source_description = "observed sphere front-surface calibration from RGB0-7 mask/center/radius"
    else:
        raise ValueError(mode)
    kept = old_ids[valid[old_ids]]
    # Scale changes coordinates only, so this should normally preserve every
    # legacy source id.  If not, retain a deterministic valid subset and log it.
    all_valid = np.flatnonzero(valid)
    selected = kept
    if len(selected) < len(old_ids):
        extra = all_valid[~np.isin(all_valid, selected)]
        selected = np.concatenate([selected, extra])[:len(old_ids)]
    if len(selected) == 0:
        raise ValueError("no valid scene tokens after scale")
    frame_yx = np.stack(np.unravel_index(selected, valid_map.shape if mode == "estimated_sphere" else aligned["static_valid"].shape), axis=-1)
    xyz, ids, fyx, scene_mask, confidence = pad_tokens(coords[selected], selected, frame_yx, None)
    out_dir = output / mode / key
    out_dir.mkdir(parents=True, exist_ok=False)
    features = np.zeros((TOKEN_COUNT, FEATURE_DIM), dtype=np.float16)
    np.savez_compressed(out_dir / "scene_tokens.npz", scene_features=features, scene_xyz=xyz,
                        scene_mask=scene_mask, scene_confidence=confidence,
                        encoder_coords=xyz.copy(), source_flat_indices=ids,
                        source_frame_y_x=fyx, scale_interval=np.asarray(
                            scale_report.get("legacy_scale_interval", [scale, scale]) or [scale, scale]),
                        alignment_mode=np.asarray(mode))
    source_hashes = {"raw": sha(raw_path), "mask": sha(mask_path), "context_geometry": sha(geometry_path),
                     "legacy_tokens": sha(old_token_path), "alignment_report": sha(old_alignment_report_path)}
    output_hash = sha(out_dir / "scene_tokens.npz")
    out_report = {
        "schema": "observed_geometry_tokens_v1", "cache_schema": "observed_geometry_tokens_v1",
        "cache_version": 1, "status": "complete", "case": key,
        "alignment_mode": mode, "depth_convention": "camera_z_ray_parameter_v1",
        "observed_indices": list(range(8)), "observed_frame_times_s": times.tolist(),
        "input_pixels_sha256": report["input_pixels_sha256"], "source_hashes": source_hashes,
        "scale": scale_report, "scale_midpoint": scale,
        "source_frame_y_x": "same legacy source ids when valid; deterministic valid fallback only if missing",
        "sampling_seed": SEED, "sampling_budget": TOKEN_COUNT,
        "stored_token_count": int(scene_mask.sum()),
        "feature_shape": [TOKEN_COUNT, FEATURE_DIM], "scene_xyz_units": "meters",
        "scene_xyz_coordinate_frame": "observed simulator world frame via actual camera RT",
        "scene_features_placeholder": True, "geometry_only_requires_feature_ignore": True,
        "geometry_source": source_description, "static_scene_gt_used": False,
        "future_rgb_used": False, "future_state_used": False, "future_contact_used": False,
        "privileged_geometry": False, "privileged_support": False,
        "source_code_sha256": hashlib.sha256(source_code.read_bytes()).hexdigest(),
        "output_sha256": output_hash,
    }
    (out_dir / "report.json").write_text(json.dumps(out_report, indent=2, allow_nan=False) + "\n")
    return out_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["estimated_aabb", "estimated_sphere"])
    parser.add_argument("--keys", nargs="*", default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    records = [r for r in manifest["records"] if r.get("family") == "door" and r.get("split") in {"train", "val"}]
    if args.keys is not None:
        requested = set(args.keys)
        records = [r for r in records if r["key"] in requested]
    results, failures = [], []
    for mode in args.modes:
        for row in records:
            try:
                results.append(build_one(args.root.resolve(), row["key"], mode, args.output.resolve()))
                print("GEOMETRY_CACHE_READY", mode, row["key"], flush=True)
            except Exception as exc:
                failures.append({"mode": mode, "case": row["key"], "split": row.get("split"),
                                 "reason": f"{type(exc).__name__}: {exc}"})
                print("GEOMETRY_CACHE_BLOCKED", mode, row["key"], failures[-1]["reason"], flush=True)
    out = {"schema": "observed_geometry_cache_variants_manifest_v1", "status": "EXECUTED",
           "source_root": str(args.root.resolve()), "records_requested": len(records) * len(args.modes),
           "records_completed": len(results), "failures": failures,
           "modes": args.modes, "train_keys": [r["key"] for r in records if r["split"] == "train"],
           "val_keys": [r["key"] for r in records if r["split"] == "val"],
           "utonia_extracted": False, "future_used": False}
    (args.output / "manifest.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
