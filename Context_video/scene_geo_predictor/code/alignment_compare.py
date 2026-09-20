"""CPU comparison of observation-only scale anchors on the 12 controlled cases.

The static surface is used only as an offline diagnostic reference.  No value
from it is written into a predictor cache or used to select a case.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from align_observed_depth import (calibrate, morph3, ray_aabb_depth,
                                  resize_crop, crop_transform, world_rays)
from audit_scene_geometry import boxes_for, critical_region, interior_labels, reference
from context_geometry import project
from prepare_context import load_model_input, pixel_hash
from prepare_small_trial import sha


def finite_stats(values: np.ndarray, radius: float | None = None) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median_m": None, "p90_m": None, "mean_m": None,
                "median_radius": None, "p90_radius": None}
    out = {"n": int(values.size), "median_m": float(np.median(values)),
           "p90_m": float(np.quantile(values, .9)), "mean_m": float(values.mean())}
    if radius is not None and radius > 0:
        out.update(median_radius=float(np.median(values) / radius),
                   p90_radius=float(np.quantile(values, .9) / radius))
    else:
        out.update(median_radius=None, p90_radius=None)
    return out


def sphere_front_depth(origin, rays, center, radius):
    q = origin - np.asarray(center, dtype=np.float64)
    a = np.sum(rays * rays, axis=-1)
    b = np.sum(rays * q, axis=-1)
    c = np.dot(q, q) - float(radius) ** 2
    disc = b * b - a * c
    safe = np.maximum(a, 1e-20)
    near = (-b - np.sqrt(np.maximum(disc, 0.0))) / safe
    far = (-b + np.sqrt(np.maximum(disc, 0.0))) / safe
    hit = (disc >= 0) & (near > 0) & np.isfinite(near) & (far >= near)
    return np.where(hit, near, 0.0), hit


def observed_sphere_scale(depth, masks, geometry, intrinsic):
    """Return a robust clip-level scale and complete failure accounting."""
    origin, rays = world_rays(intrinsic, geometry["camera_world_to_view"], depth.shape[1:])
    radius = float(geometry["size_m"][0, 0]) / 2.0
    rows, ratios = [], []
    for frame in range(8):
        interior = morph3(morph3(masks[frame], dilate=False), dilate=False)
        near, hit = sphere_front_depth(origin, rays,
                                       geometry["positions_world"][frame, 0], radius)
        finite = np.isfinite(depth[frame]) & (depth[frame] > 0)
        selected = interior & hit & finite
        n = int(selected.sum())
        ratio = near[selected] / depth[frame][selected] if n else np.empty(0)
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        row = {"frame": frame, "interior_pixels": int(interior.sum()),
               "ray_hit_pixels": int((interior & hit).sum()),
               "selected_pixels": int(ratio.size),
               "hit_fraction": float(n / max(1, int(interior.sum()))),
               "median_scale": float(np.median(ratio)) if ratio.size else None,
               "p10_scale": float(np.quantile(ratio, .1)) if ratio.size else None,
               "p90_scale": float(np.quantile(ratio, .9)) if ratio.size else None}
        rows.append(row)
        if ratio.size:
            # A pixel-level 2--98% trim makes a single silhouette edge unable
            # to dominate the clip estimate while retaining the full report.
            lo, hi = np.quantile(ratio, [.02, .98])
            ratios.append(ratio[(ratio >= lo) & (ratio <= hi)])
    all_ratios = np.concatenate(ratios) if ratios else np.empty(0)
    scale = float(np.median(all_ratios)) if all_ratios.size else None
    frame_scales = np.asarray([r["median_scale"] for r in rows if r["median_scale"] is not None])
    reasons = []
    if not all_ratios.size:
        reasons.append("no positive ray-sphere anchors")
    if any(r["selected_pixels"] < 8 for r in rows):
        reasons.append("one or more frames has fewer than eight anchors")
    if frame_scales.size and scale and float(np.ptp(frame_scales) / scale) > .10:
        reasons.append("per-frame scale span exceeds 10 percent")
    if scale is not None:
        for row in rows:
            # The anchor residual is recorded in the same camera-Z parameter
            # as the scale fit; it is not a static-surface accuracy claim.
            row["robust_residual_scale"] = (
                float(row["median_scale"] - scale) if row["median_scale"] is not None else None)
            row["robust_residual_abs_scale"] = (
                abs(row["robust_residual_scale"]) if row["robust_residual_scale"] is not None else None)
    return {"accepted": not reasons, "scale_midpoint": scale, "per_frame": rows,
            "anchor_pixels": int(all_ratios.size),
            "scale_median": scale,
            "scale_p02": float(np.quantile(all_ratios, .02)) if all_ratios.size else None,
            "scale_p98": float(np.quantile(all_ratios, .98)) if all_ratios.size else None,
            "frame_scale_relative_span": float(np.ptp(frame_scales) / scale)
            if frame_scales.size and scale else None,
            "failure_reasons": reasons,
            "method": "median d_sphere / raw VGGT depth on observed RGB0-7 sphere mask",
            "depth_parameter": "camera-Z ray parameter (ray z normalized to one)",
            "future_state_used": False, "static_scene_gt_used": False}


def processed_inputs(root: Path, key: str):
    context = root / "observed_context" / key
    report = json.loads((context / "report.json").read_text())
    input_path = Path(report["input_json"])
    rgb, times = load_model_input(input_path)
    with np.load(context / "context_geometry.npz", allow_pickle=False) as archive:
        geometry = {k: archive[k] for k in archive.files}
    with np.load(root / "observed_masks" / key / "observed_masks.npz", allow_pickle=False) as archive:
        per_object = archive["per_object"]
        union_dynamic = archive["union_dynamic"]
    with np.load(root / "raw_probe" / key / "raw_vggt.npz", allow_pickle=False) as archive:
        raw = {k: archive[k] for k in archive.files}
    transform = crop_transform(rgb.shape[1:3])
    masks = np.stack([np.stack([resize_crop(m, transform, is_mask=True) for m in frame])
                      for frame in per_object])
    union = masks.any(axis=1)
    intrinsic = transform["affine"] @ geometry["camera_K"]
    depth = raw["depth"][..., 0].astype(np.float64)
    if depth.shape[0] != 8 or depth.shape[1:] != masks.shape[2:]:
        raise ValueError("raw depth and processed mask grids disagree")
    return report, rgb, times, geometry, masks, union, raw, intrinsic, depth


def static_metrics(method_points, method_valid, truth, labels, boxes, critical, radius):
    method_points = np.asarray(method_points)
    method_valid = np.asarray(method_valid, dtype=bool)
    static_label = labels >= 0
    # Labels include an infinite reference ground plane, while interior labels
    # for boxes are used only by the legacy audit.  Here we report both.
    all_errors, by_surface = [], []
    for frame in range(method_points.shape[0]):
        for label in range(len(boxes) + 1):
            selected = method_valid[frame] & (labels == label)
            err = np.linalg.norm(method_points[frame] - truth, axis=-1)[selected]
            all_errors.append(err)
            name = boxes[label]["name"] if label < len(boxes) else "ground_z0"
            by_surface.append({"frame": frame, "surface": name,
                               "error_m": finite_stats(err, radius)})
    errors = np.concatenate([x for x in all_errors if x.size]) if any(x.size for x in all_errors) else np.empty(0)
    critical_targets = truth[critical]
    estimate = method_points[method_valid & static_label]
    if critical_targets.size and estimate.size:
        # Bound CPU/memory on the diagnostic, but use a deterministic stride.
        target = critical_targets[::max(1, critical_targets.shape[0] // 20000)]
        points = estimate[::max(1, estimate.shape[0] // 100000)]
        distance = cKDTree(points).query(target, workers=1)[0]
        coverage = {"target_count": int(target.shape[0]), "estimate_count": int(points.shape[0]),
                    "within_2cm": float(np.mean(distance <= .02)),
                    "within_5cm": float(np.mean(distance <= .05)),
                    "within_10cm": float(np.mean(distance <= .10)),
                    "median_distance_m": float(np.median(distance)),
                    "p90_distance_m": float(np.quantile(distance, .9))}
    else:
        coverage = {"target_count": int(critical_targets.shape[0]), "estimate_count": int(estimate.shape[0]),
                    "within_2cm": None, "within_5cm": None, "within_10cm": None,
                    "median_distance_m": None, "p90_distance_m": None}
    # Temporal consistency is measured only on pixels visible in every frame
    # and is an offline static-scene diagnostic, not a deployment gate.
    common = method_valid.all(axis=0) & static_label
    if common.any():
        temporal = np.linalg.norm(method_points[:, common] - np.median(method_points[:, common], axis=0), axis=-1)
    else:
        temporal = np.empty(0)
    return {"surface_error_m": finite_stats(errors, radius),
            "surface_error_relative_to_radius": finite_stats(errors / radius if radius > 0 else errors),
            "per_surface": by_surface, "critical_coverage_oracle_audit": coverage,
            "temporal_consistency_m": finite_stats(temporal, radius),
            "valid_static_pixels": int((method_valid & static_label).sum()),
            "valid_fraction": float(method_valid.mean())}


def locate_aabb_violation(root: Path, key: str, geometry, masks, depth, intrinsic):
    origin, rays = world_rays(intrinsic, geometry["camera_world_to_view"], depth.shape[1:])
    overlap = masks.sum(axis=1) > 1
    best = None
    for frame in range(8):
        interior = morph3(masks[frame, 0], dilate=False) & ~overlap[frame]
        near, far, hit = ray_aabb_depth(origin, rays,
                                         geometry["positions_world"][frame, 0], geometry["size_m"][0])
        selected = interior & hit & np.isfinite(depth[frame]) & (depth[frame] > 0)
        if not selected.any():
            continue
        scale_report = json.loads((root / "aligned_scene" / key / "report.json").read_text())["alignment"]
        scale = float(scale_report.get("scale_midpoint", np.mean(scale_report["robust_interval"])))
        violation = np.maximum(near - scale * depth[frame], 0) + np.maximum(scale * depth[frame] - far, 0)
        ys, xs = np.where(selected)
        idx = int(np.argmax(violation[selected]))
        cand = {"frame": frame, "y": int(ys[idx]), "x": int(xs[idx]),
                "raw_depth": float(depth[frame, ys[idx], xs[idx]]),
                "scale": scale, "near_z": float(near[ys[idx], xs[idx]]),
                "far_z": float(far[ys[idx], xs[idx]]),
                "violation_m": float(violation[ys[idx], xs[idx]]),
                "mask_pixel": True, "mask_source": str(root / "observed_masks" / key / "observed_masks.npz")}
        if best is None or cand["violation_m"] > best["violation_m"]:
            best = cand
    return best


def plot_examples(out: Path, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "BLOCKED", "reason": f"matplotlib: {exc}"}
    import matplotlib.pyplot as plt
    example_dir = out / "alignment_examples"
    example_dir.mkdir(exist_ok=True)
    for row in rows:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
        image = row["rgb7"].transpose(1, 2, 0)
        ax[0].imshow(image); ax[0].set_title(row["key"] + " RGB7"); ax[0].axis("off")
        methods = ["aabb_midpoint_legacy", "observed_sphere_surface", "oracle_scale_diagnostic"]
        available = [m for m in methods if m in row["error_maps"]]
        for i, method in enumerate(available, 1):
            err = row["error_maps"][method]
            im = ax[i].imshow(np.ma.masked_where(~np.isfinite(err), err), cmap="magma", vmin=0, vmax=.5)
            ax[i].set_title(method.replace("_", "\n")); ax[i].axis("off")
            fig.colorbar(im, ax=ax[i], fraction=.046)
        for i in range(len(available) + 1, 3):
            ax[i].axis("off")
        fig.tight_layout(); fig.savefig(example_dir / f"{row['key']}.png", dpi=140); plt.close(fig)
    return {"status": "EXECUTED", "directory": str(example_dir), "count": len(rows)}


def run(root: Path, output: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    records = manifest["records"]
    output.mkdir(parents=True, exist_ok=False)
    rows, flat_rows, failures = [], [], []
    examples = []
    for record in records:
        key, family = record["key"], record["family"]
        try:
            report, rgb, times, geometry, masks, union, raw, intrinsic, depth = processed_inputs(root, key)
            if family not in ("door", "gap", "barrier"):
                raise ValueError("unknown family")
            aligned_report = json.loads((root / "aligned_scene" / key / "report.json").read_text())
            legacy_scale = float(aligned_report["alignment"].get("scale_midpoint",
                np.mean(aligned_report["alignment"]["robust_interval"])))
            origin, rays = world_rays(intrinsic, geometry["camera_world_to_view"], depth.shape[1:])
            boxes = boxes_for(root, key)
            truth, labels, truth_depth = reference(origin, rays, boxes)
            static_label = labels >= 0
            # Same observed mask/dilation and same raw depth for all three arms.
            valid = np.isfinite(depth) & (depth > 0) & ~np.stack([morph3(m, dilate=True) for m in union])
            valid &= np.broadcast_to(np.isfinite(truth_depth) & (truth_depth > 0), valid.shape)
            critical = critical_region(family, truth, labels, boxes, record["value"])
            radius = float(geometry["size_m"][0, 0]) / 2.0
            estimates = {"aabb_midpoint_legacy": legacy_scale}
            sphere_report = {"accepted": False, "failure_reasons": ["not_applicable_for_puck"]}
            if family in ("door", "gap"):
                sphere_report = observed_sphere_scale(depth, masks[:, 0], geometry, intrinsic)
                if sphere_report["accepted"]:
                    estimates["observed_sphere_surface"] = sphere_report["scale_midpoint"]
                else:
                    failures.append({"key": key, "method": "observed_sphere_surface",
                                     "reason": sphere_report["failure_reasons"]})
            # Static true depth is an offline scale-only decomposition.  It is
            # never used to alter input or remove a failed case.
            static_pixels = valid & static_label
            depth_values = np.broadcast_to(truth_depth, depth.shape)[static_pixels]
            raw_values = depth[static_pixels]
            positive = np.isfinite(depth_values) & np.isfinite(raw_values) & (raw_values > 0) & (depth_values > 0)
            # A least-squares fit is dominated by the unbounded z=0 reference
            # plane near the horizon.  Use a robust positive ratio fit instead;
            # the fit is still privileged static truth and remains diagnostic
            # only.  The untrimmed counts are retained in the case report.
            oracle_ratios = depth_values[positive] / raw_values[positive] if positive.any() else np.empty(0)
            if oracle_ratios.size:
                lo, hi = np.quantile(oracle_ratios, [.02, .98])
                oracle_ratios = oracle_ratios[(oracle_ratios >= lo) & (oracle_ratios <= hi)]
            oracle_scale = float(np.median(oracle_ratios)) if oracle_ratios.size else None
            estimates["oracle_scale_diagnostic"] = oracle_scale
            method_results, error_maps = {}, {}
            for method, scale in estimates.items():
                if scale is None or not np.isfinite(scale) or scale <= 0:
                    method_results[method] = {"status": "BLOCKED", "reason": "no positive scale"}
                    continue
                points = origin[None, None, None, :] + rays[None] * (depth * scale)[..., None]
                points = points.astype(np.float64)
                method_results[method] = {"status": "EXECUTED", "scale": float(scale),
                    "scale_relative_to_legacy": float(scale / legacy_scale),
                    "metrics": static_metrics(points, valid, truth, labels, boxes, critical, radius),
                    "scale_source": {"aabb_midpoint_legacy": "existing observed AABB interval midpoint",
                                     "observed_sphere_surface": "RGB0-7 sphere mask + observed center/radius",
                                     "oracle_scale_diagnostic": "robust median static-reference depth ratio after 2-98% trim; offline only"}[method]}
                error_maps[method] = np.where(valid[7] & static_label[7],
                                              np.linalg.norm(points[7] - truth[7], axis=-1), np.nan)
                for surface, item in enumerate(method_results[method]["metrics"]["per_surface"]):
                    flat_rows.append(dict(key=key, family=family, method=method, scale=float(scale),
                                          surface=item["surface"], frame=item["frame"],
                                          n=item["error_m"]["n"], median_error_m=item["error_m"]["median_m"],
                                          p90_error_m=item["error_m"]["p90_m"], radius_m=radius))
            violation = locate_aabb_violation(root, key, geometry, masks, depth, intrinsic)
            case_result = {"key": key, "family": family, "controlled_value": record["value"],
                "source_provenance": "diagnostic_legacy_raw; controlled_scene_eval_12 was already used for development",
                "observed_indices": list(range(8)), "raw_depth_shape": list(depth.shape),
                "processed_grid": list(depth.shape[1:]), "valid_static_pixels": int(static_pixels.sum()),
                "dynamic_exclusion": "same resized SAM2 union mask dilated by one pixel for all methods",
                "radius_m": radius, "sphere_calibration": sphere_report,
                "methods": method_results, "aabb_max_violation_location": violation,
                "critical_region_definition": "predeclared family geometry band from static reference; offline audit only",
                "oracle_scale_note": "robustly fits static truth on this same diagnostic scene after a 2-98% ratio trim; fit/evaluation surfaces are the same offline reference; not deployment quality or predictor input"}
            (output / f"{key}.json").write_text(json.dumps(case_result, indent=2, allow_nan=False) + "\n")
            rows.append(case_result)
            if len(examples) < 3 and family in ("door", "gap", "barrier"):
                examples.append({"key": key, "rgb7": raw["images"][7], "error_maps": error_maps})
        except Exception as exc:
            failures.append({"key": key, "method": "case", "reason": f"{type(exc).__name__}: {exc}"})
    fixed = {"schema": "alignment_comparison_v1", "cases": [r["key"] for r in records],
             "source": "controlled_scene_eval_12", "development_only": True}
    (output / "fixed_sample_list.json").write_text(json.dumps(fixed, indent=2) + "\n")
    (output / "failure_list.json").write_text(json.dumps(failures, indent=2, allow_nan=False) + "\n")
    with (output / "alignment_comparison.csv").open("w", newline="") as stream:
        fields = ["key", "family", "method", "scale", "surface", "frame", "n", "median_error_m", "p90_error_m", "radius_m"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(flat_rows)
    summary = {"status": "EXECUTED" if rows else "DATA_BLOCKED", "cases_requested": len(records),
               "cases_completed": len(rows), "failure_count": len(failures), "results": rows,
               "failures": failures, "provenance": "controlled_scene_eval_12 / raw VGGT + observed SAM2 mask + context camera",
               "methods": ["aabb_midpoint_legacy", "observed_sphere_surface", "oracle_scale_diagnostic"],
               "observation_gate": "finite depth, positive ray hits, mask anchors and scale stability",
               "oracle_audit": "static surface reference and critical coverage are offline only",
               "train_gap_0221": "located separately below; not part of this 12-case controlled set"}
    (output / "alignment_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "train_gap_0221_violation.json").write_text(json.dumps(
        locate_gap_violation(root.parent / "small_trial_120"), indent=2, allow_nan=False) + "\n")
    summary["examples"] = plot_examples(output, examples)
    (output / "alignment_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "cases_requested", "cases_completed", "failure_count", "examples")}, indent=2))


def locate_gap_violation(root: Path):
    key = "train_gap_0221"
    try:
        report, rgb, times, geometry, masks, union, raw, intrinsic, depth = processed_inputs(root, key)
        return {"status": "EXECUTED", "key": key,
                "violation": locate_aabb_violation(root, key, geometry, masks, depth, intrinsic),
                "source": "small_trial_120 raw VGGT + observed mask + context geometry",
                "note": "This location is an AABB anchor diagnostic; it does not prove the static surface is at that pixel."}
    except Exception as exc:
        return {"status": "DATA_BLOCKED", "key": key, "reason": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
