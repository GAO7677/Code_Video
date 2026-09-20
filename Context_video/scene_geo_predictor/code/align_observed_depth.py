"""Coarse RGB8 depth anchoring from observed dynamic AABBs, never static GT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from observed_masks import load_geometry, validate_masks
from prepare_context import load_model_input, pixel_hash
from probe_vggt import FIELDS, validate_predictions


def crop_transform(image_hw):
    """Match VGGT's crop mode with integer pixel centers and PIL resampling."""
    h, w = map(int, image_hw)
    if h < 2 or w < 2:
        raise ValueError("Source image must have both dimensions >= 2")
    new_w = 518
    new_h = int(round(h * new_w / w / 14) * 14)
    if new_h < 1:
        raise ValueError("Unsupported image aspect ratio for VGGT preprocessing")
    top = max(0, (new_h - 518) // 2)
    out_h = min(new_h, 518)
    sx, sy = new_w / w, new_h / h
    affine = np.array([[sx, 0, 0.5 * sx - 0.5],
                       [0, sy, 0.5 * sy - 0.5 - top], [0, 0, 1]], dtype=np.float64)
    return dict(source_hw=[h, w], resized_hw=[new_h, new_w],
                processed_hw=[out_h, new_w], crop_top=top, affine=affine)


def resize_crop(array, transform, *, is_mask):
    """Transform one source-resolution mask or RGB image, without box clipping."""
    if list(array.shape[:2]) != transform["source_hw"]:
        raise ValueError("Image/mask resolution disagrees with source transform")
    if is_mask and (array.ndim != 2 or array.dtype != np.bool_):
        raise ValueError("Expected a boolean mask")
    if not is_mask and (array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8):
        raise ValueError("Expected uint8 RGB")
    source = array.astype(np.uint8) * 255 if is_mask else array
    im = Image.fromarray(source)
    nh, nw = transform["resized_hw"]
    im = im.resize((nw, nh), Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)
    top = transform["crop_top"]
    oh, ow = transform["processed_hw"]
    im = im.crop((0, top, ow, top + oh))
    out = np.asarray(im)
    return out > 0 if is_mask else out


def morph3(mask, *, dilate):
    """One 3x3 iteration; pixels outside the image are background."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Morphology expects a boolean image")
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    result = np.zeros_like(mask) if dilate else np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            if dilate:
                result |= padded[dy:dy+h, dx:dx+w]
            else:
                result &= padded[dy:dy+h, dx:dx+w]
    return result


def world_rays(intrinsic, world_to_view, image_hw):
    """Rays have camera-space Z component one, so ray parameters are Z depth."""
    h, w = image_hw
    y, x = np.mgrid[:h, :w]
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float64)
    camera_rays = pixels @ np.linalg.inv(intrinsic).T
    if not np.isfinite(camera_rays).all() or np.any(np.abs(camera_rays[..., 2]) < 1e-12):
        raise ValueError("Invalid camera rays")
    camera_rays /= camera_rays[..., 2:3]
    rotation, translation = world_to_view[:, :3], world_to_view[:, 3]
    origin = -rotation.T @ translation
    return origin, camera_rays @ rotation


def ray_aabb_depth(origin, rays, center, size):
    """Intersect with a world-axis-aligned conservative dynamic bounding box."""
    origin = np.asarray(origin, dtype=np.float64)
    rays = np.asarray(rays, dtype=np.float64)
    center, size = np.asarray(center, dtype=np.float64), np.asarray(size, dtype=np.float64)
    if origin.shape != (3,) or center.shape != (3,) or size.shape != (3,) or rays.shape[-1] != 3:
        raise ValueError("Invalid AABB/ray shapes")
    if not all(np.isfinite(x).all() for x in (origin, rays, center, size)) or not (size > 0).all():
        raise ValueError("Invalid AABB/ray values")
    near = np.full(rays.shape[:-1], -np.inf)
    far = np.full(rays.shape[:-1], np.inf)
    hit = np.ones(rays.shape[:-1], dtype=bool)
    lower, upper = center - 0.5 * size, center + 0.5 * size
    for axis in range(3):
        direction = rays[..., axis]
        parallel = np.abs(direction) < 1e-12
        if not lower[axis] <= origin[axis] <= upper[axis]:
            hit &= ~parallel
        a = np.zeros_like(direction)
        b = np.zeros_like(direction)
        np.divide(lower[axis] - origin[axis], direction, out=a, where=~parallel)
        np.divide(upper[axis] - origin[axis], direction, out=b, where=~parallel)
        near = np.maximum(near, np.where(parallel, -np.inf, np.minimum(a, b)))
        far = np.minimum(far, np.where(parallel, np.inf, np.maximum(a, b)))
    # Camera-inside and backward-only intersections are not valid scale anchors.
    hit &= np.isfinite(near) & np.isfinite(far) & (near > 0) & (far >= near)
    return np.where(hit, near, 0.0), np.where(hit, far, 0.0), hit


def scale_intervals(lower, upper):
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if (lower.ndim != 1 or lower.size == 0 or lower.shape != upper.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower <= 0) or np.any(upper < lower)):
        raise ValueError("Expected nonempty positive per-pixel scale intervals")
    strict = [float(lower.max()), float(upper.min())]
    robust = [float(np.quantile(lower, 0.98)), float(np.quantile(upper, 0.02))]
    valid = robust[0] <= robust[1]
    mid = float(np.mean(robust)) if valid else None
    coverage = float(np.mean((lower <= robust[0]) & (upper >= robust[1]))) if valid else 0.0
    midpoint_coverage = float(np.mean((lower <= mid) & (upper >= mid))) if valid else 0.0
    return dict(strict_interval=strict, strict_feasible=bool(strict[0] <= strict[1]),
                robust_interval=robust, robust_feasible=bool(valid),
                robust_lower_quantile=0.98, robust_upper_quantile=0.02,
                scale_midpoint=mid, midpoint_policy="arithmetic midpoint, not uniquely estimated truth",
                full_interval_coverage=coverage, midpoint_coverage=midpoint_coverage,
                interval_width=float(robust[1] - robust[0]) if valid else None,
                relative_interval_width=float((robust[1] - robust[0]) / mid) if valid else None,
                anchor_pixels=int(lower.size), scale_shift_fitted=False)


def validate_reviewed_masks(per_object, union_dynamic, accepted, reviewed_masks):
    validate_masks(per_object, union_dynamic, accepted)
    if not reviewed_masks:
        raise ValueError("--reviewed-masks is required after visual inspection")
    if not accepted.all():
        raise ValueError("All observed per-object masks must pass their prompt gates")


def calibrate(depth, per_object, geometry, intrinsic):
    """Return interval diagnostics; bounds only constrain depth, not surface shape."""
    if depth.ndim != 3 or depth.shape[0] != 8 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Expected eight finite positive depth maps")
    if (per_object.ndim != 4 or per_object.shape[0] != 8
            or per_object.shape[2:] != depth.shape[1:] or per_object.dtype != np.bool_):
        raise ValueError("Depth and per-object masks must share the processed grid")
    if per_object.shape[1] != geometry["positions_world"].shape[1]:
        raise ValueError("Mask/geometry object counts differ")
    origin, rays = world_rays(intrinsic, geometry["camera_world_to_view"], depth.shape[1:])
    lower, upper, depths, near_values, far_values, rows = [], [], [], [], [], []
    failures = []
    selected_total, hit_total = 0, 0
    overlaps = per_object.sum(axis=1) > 1
    for t in range(8):
        frame_selected, frame_hits = 0, 0
        object_rows = []
        for j in range(per_object.shape[1]):
            interior = morph3(per_object[t, j], dilate=False) & ~overlaps[t]
            near, far, hit = ray_aabb_depth(origin, rays, geometry["positions_world"][t, j], geometry["size_m"][j])
            selected = interior & hit
            n, m = int(interior.sum()), int(selected.sum())
            frame_selected += n
            frame_hits += m
            object_rows.append(dict(object_index=j, eroded_unambiguous_pixels=n,
                                    ray_hit_pixels=m, ray_hit_fraction=m/n if n else None))
            if m:
                d, a, b = depth[t][selected], near[selected], far[selected]
                lower.append(a/d)
                upper.append(b/d)
                depths.append(d)
                near_values.append(a)
                far_values.append(b)
        fraction = frame_hits/frame_selected if frame_selected else 0.0
        if frame_selected == 0 or frame_hits == 0:
            failures.append(f"RGB{t}: empty usable eroded foreground")
        elif fraction < 0.9:
            failures.append(f"RGB{t}: foreground ray hit fraction below 0.9")
        selected_total += frame_selected
        hit_total += frame_hits
        rows.append(dict(frame=t, eroded_unambiguous_pixels=frame_selected,
                         ray_hit_pixels=frame_hits, ray_hit_fraction=fraction, objects=object_rows))
    rotation, translation = geometry["camera_world_to_view"][:, :3], geometry["camera_world_to_view"][:, 3]
    center_depth = (geometry["positions_world"] @ rotation.T + translation)[..., 2]
    report = dict(per_frame=rows, erode_pixels=1, overlap_pixels_excluded=int(overlaps.sum()),
                  ray_hit_fraction=hit_total/selected_total if selected_total else 0.0,
                  observed_center_camera_depth_min_m=float(center_depth.min()),
                  observed_center_camera_depth_max_m=float(center_depth.max()),
                  observed_center_depth_span_per_object_m=np.ptp(center_depth, axis=0).tolist(),
                  selected_foreground_pixels=selected_total, ray_hit_pixels=hit_total,
                  box_surface_assumed=False, bounding_box_only=True)
    if report["ray_hit_fraction"] < 0.9:
        failures.append("Global foreground ray hit fraction below 0.9")
    if lower:
        intervals = scale_intervals(np.concatenate(lower), np.concatenate(upper))
        report.update(intervals)
        if not intervals["robust_feasible"]:
            failures.append("No common robust positive scale interval")
        elif intervals["full_interval_coverage"] < 0.95:
            failures.append("Final interval coverage below 0.95")
        else:
            d, a, b = map(np.concatenate, (depths, near_values, far_values))
            calibrated = intervals["scale_midpoint"] * d
            violation = np.maximum(a-calibrated, 0) + np.maximum(calibrated-b, 0)
            report["midpoint_aabb_violation_m"] = dict(mean=float(violation.mean()),
                rmse=float(np.sqrt(np.mean(violation**2))),
                p95=float(np.quantile(violation, 0.95)), max=float(violation.max()))
    else:
        report.update(robust_feasible=False, strict_feasible=False, anchor_pixels=0)
    report.update(accepted=not failures, failure_reasons=failures)
    return report


def static_points(depth, union_dynamic, intrinsic, world_to_view, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    if depth.shape != union_dynamic.shape or union_dynamic.dtype != np.bool_:
        raise ValueError("Depth/mask grid mismatch")
    excluded = np.stack([morph3(mask, dilate=True) for mask in union_dynamic])
    valid = np.isfinite(depth) & (depth > 0) & ~excluded
    origin, rays = world_rays(intrinsic, world_to_view, depth.shape[1:])
    points = origin + rays[None] * (np.where(valid, depth, 0) * scale)[..., None]
    points[~valid] = 0
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite reconstructed points")
    return points.astype(np.float32), valid


def preview(rgb, static_valid, path):
    h, w = rgb.shape[1:3]
    canvas = Image.new("RGB", (w, 2*(h+26)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, t in enumerate((0, 7)):
        pixels = rgb[t].copy()
        excluded = ~static_valid[t]
        pixels[excluded] = (0.4*pixels[excluded] + 0.6*np.array([235, 70, 80])).astype(np.uint8)
        y = row*(h+26)
        draw.text((5, y+5), f"RGB{t}: excluded dynamic / coarse alignment only", fill="black")
        canvas.paste(Image.fromarray(pixels), (0, y+26))
    canvas.save(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("input", "geometry", "masks", "raw", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reviewed-masks", action="store_true")
    parser.add_argument("--scale-method", choices=("aabb-interval", "observed-surface"), default="aabb-interval")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    report = dict(schema="observed_aabb_depth_alignment_v1", status="failed",
                  training_ready=False, coarse_alignment_only=True,
                  observed_indices=list(range(8)), static_scene_gt_used=False,
                  future_frames_used=False, gt_masks_used=False,
                  permitted_context_geometry_used=True, reviewed_masks=bool(args.reviewed_masks),
                  confidence_threshold_applied=False, confidence_calibrated=False,
                  depth_source="VGGT depth head, not point head",
                  coordinate_frame="Actual observed camera in context world coordinates",
                  pixel_convention="Integer pixel centers, PIL affine (u+0.5)*scale-0.5",
                  limitations=["AABB intervals do not identify a unique true metric scale.",
                               "Small interval residual is not static geometry accuracy.",
                               "Single positive scale does not correct spatial depth bias or shift.",
                               "Unseen geometry is unknown, not free space or completed surfaces.",
                               "No static-ground-truth registration, precision collision guarantee, or training approval."])
    failure = None
    try:
        if not args.reviewed_masks:
            raise ValueError("Inspect masks first, then explicitly pass --reviewed-masks")
        rgb, times = load_model_input(args.input)
        geometry = load_geometry(args.geometry, times)
        with np.load(args.masks, allow_pickle=False) as archive:
            per_object = archive["per_object"]
            union_dynamic, accepted = archive["union_dynamic"], archive["accepted"]
            mask_times = archive["frame_times"]
        validate_reviewed_masks(per_object, union_dynamic, accepted, args.reviewed_masks)
        if per_object.shape[2:] != rgb.shape[1:3] or per_object.shape[1] != geometry["positions_world"].shape[1]:
            raise ValueError("Mask resolution/object count disagrees with RGB/context geometry")
        if mask_times.shape != (8,) or not np.allclose(mask_times, times, rtol=0, atol=1e-7):
            raise ValueError("Mask observation times differ from RGB")
        with np.load(args.raw, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        if set(raw) != set(FIELDS):
            raise ValueError("Expected exactly the declared raw VGGT fields")
        validate_predictions(raw)
        transform = crop_transform(rgb.shape[1:3])
        processed_rgb = np.stack([resize_crop(im, transform, is_mask=False) for im in rgb])
        expected = processed_rgb.transpose(0, 3, 1, 2).astype(np.float32)/255.0
        if raw["images"].shape != expected.shape or not np.allclose(raw["images"], expected, rtol=0, atol=1e-6):
            raise ValueError("Raw VGGT images do not match this exact observed RGB8 preprocessing")
        masks = np.stack([np.stack([resize_crop(m, transform, is_mask=True) for m in frame]) for frame in per_object])
        processed_union = masks.any(axis=1)
        intrinsic = transform["affine"] @ geometry["camera_K"]
        depth = raw["depth"][..., 0].astype(np.float64)
        if args.scale_method == "observed-surface":
            from observed_surface_scale import estimate_scale
            if masks.shape[1] != 1:
                raise ValueError("Observed-surface calibration currently supports one object")
            alignment = estimate_scale(depth, masks[:, 0], geometry, intrinsic)
            report.update(schema="observed_surface_scale_v1",
                surface_adapter_sha256=sha256(Path(__file__).with_name("observed_surface_scale.py")),
                scale_interval_interpretation=alignment["scale_interval_interpretation"],
                limitations=["Supported sphere/upright-cylinder shape assumption is required.",
                    "A single observed-object scale does not correct spatial depth bias.",
                    "Unseen geometry is unknown; points are not a precise collision SDF."])
        else:
            alignment = calibrate(depth, masks, geometry, intrinsic)
        report.update(alignment=alignment, input_pixels_sha256=pixel_hash(rgb),
                      provenance={key: dict(path=str(getattr(args, key).resolve()), sha256=sha256(getattr(args, key)))
                                  for key in ("input", "geometry", "masks", "raw")},
                      adapter_sha256=sha256(Path(__file__)),
                      transform={key: value.tolist() if isinstance(value, np.ndarray) else value
                                 for key, value in transform.items()},
                      K_processed=intrinsic.tolist(), camera_world_to_view=geometry["camera_world_to_view"].tolist())
        if not alignment["accepted"]:
            report["failure_reasons"] = alignment["failure_reasons"]
        else:
            points, valid = static_points(depth, processed_union, intrinsic,
                                          geometry["camera_world_to_view"], alignment["scale_midpoint"])
            h, w = depth.shape[1:]
            y, x = np.mgrid[:h, :w]
            pixel_h = np.stack([x, y, np.ones_like(x)], axis=-1)
            source_uv = (pixel_h @ np.linalg.inv(transform["affine"]).T)[..., :2].astype(np.float32)
            np.savez_compressed(args.output/"coarse_static_points.npz", world_midpoint=points,
                                static_valid=valid, scale_interval=np.array(alignment.get("robust_interval",
                                    [alignment["scale_midpoint"], alignment["scale_midpoint"]])),
                                K_processed=intrinsic, RT=geometry["camera_world_to_view"],
                                source_uv=source_uv, frame_times=times)
            preview(processed_rgb, valid, args.output/"preview.png")
            report.update(status="coarse_interval_alignment_complete", world_midpoint_shape=list(points.shape),
                          static_valid_fraction_per_frame=valid.mean(axis=(1, 2)).tolist(),
                          dynamic_dilation_pixels=1, static_points_written=True,
                          output_npz_sha256=sha256(args.output/"coarse_static_points.npz"))
    except (ValueError, KeyError, OSError) as exc:
        failure = exc
        report["failure_reasons"] = [f"{type(exc).__name__}: {exc}"]
    report.setdefault("static_points_written", False)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if failure is not None or report["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
