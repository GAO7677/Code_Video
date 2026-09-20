"""NumPy-only, observation-only adapter for coarse visual scene point caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CONTEXT_KEYS = (
    "node", "edge", "mask", "surface", "surface_mask", "history_dt",
    "future_dt", "last_position", "cv",
)
CACHE_KEYS = {
    "world_midpoint", "static_valid", "scale_interval", "K_processed",
    "RT", "source_uv", "frame_times",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_visual_scene(directory, points_per_frame=224):
    """Sample valid row-major pixels uniformly; retain meters and provenance.

    Returns an unbatched scene with 8 * points_per_frame rows. Empty slots have
    zero XYZ, false validity, and -1 source coordinates. This does not certify
    geometric accuracy or turn the coarse cache into precise collision truth.
    """
    if (isinstance(points_per_frame, bool)
            or not isinstance(points_per_frame, (int, np.integer))
            or points_per_frame <= 0):
        raise ValueError("points_per_frame must be a positive integer")
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    expected = {
        "schema": "observed_aabb_depth_alignment_v1",
        "status": "coarse_interval_alignment_complete",
        "training_ready": False, "coarse_alignment_only": True,
        "static_scene_gt_used": False, "future_frames_used": False,
        "gt_masks_used": False, "permitted_context_geometry_used": True,
        "reviewed_masks": True, "static_points_written": True,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if (actual is not value if isinstance(value, bool) else actual != value):
            raise ValueError(f"Unexpected visual cache report field: {key}")
    if report.get("observed_indices") != list(range(8)):
        raise ValueError("Visual cache must declare exactly RGB0-7")
    if report.get("alignment", {}).get("accepted") is not True:
        raise ValueError("Visual cache alignment was not accepted")
    path = directory / "coarse_static_points.npz"
    if report.get("output_npz_sha256") != _sha256(path):
        raise ValueError("Visual cache SHA256 does not match its report")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != CACHE_KEYS:
            raise ValueError("Unexpected coarse scene cache fields")
        data = {key: archive[key] for key in CACHE_KEYS}
    points, valid = data["world_midpoint"], data["static_valid"]
    if (points.ndim != 4 or points.shape[0] != 8 or points.shape[-1] != 3
            or min(points.shape[1:3]) < 1 or not np.issubdtype(points.dtype, np.floating)):
        raise ValueError("world_midpoint must be floating [8,H,W,3]")
    if valid.shape != points.shape[:-1] or valid.dtype != np.bool_:
        raise ValueError("static_valid must be boolean [8,H,W]")
    if not np.isfinite(points[valid]).all():
        raise ValueError("Valid visual points must be finite")
    if report.get("world_midpoint_shape") != list(points.shape):
        raise ValueError("Visual point shape does not match its report")
    h, w = points.shape[1:3]
    expected_shapes = {"scale_interval": (2,), "K_processed": (3, 3),
                       "RT": (3, 4), "source_uv": (h, w, 2), "frame_times": (8,)}
    for key, shape in expected_shapes.items():
        value = data[key]
        if (value.shape != shape or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()):
            raise ValueError(f"Invalid coarse scene field: {key}")
    lo, hi = data["scale_interval"]
    if not 0 < lo <= hi:
        raise ValueError("Expected an ordered positive scale interval")
    if not np.allclose(data["frame_times"], np.arange(8) / 30, rtol=0, atol=1e-7):
        raise ValueError("Expected observed times RGB0-7 at 30 FPS")
    n = 8 * int(points_per_frame)
    xyz = np.zeros((n, 3), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    origin = np.full((n, 3), -1, dtype=np.int64)
    source_uv = np.full((n, 2), -1, dtype=np.float32)
    counts = np.zeros(8, dtype=np.int64)
    candidates = valid.sum(axis=(1, 2), dtype=np.int64)
    for t in range(8):
        available = np.flatnonzero(valid[t].ravel())
        count = min(int(points_per_frame), len(available))
        counts[t] = count
        if not count:
            continue
        selected = available[np.linspace(0, len(available) - 1, count, dtype=np.int64)]
        y, x = np.divmod(selected, w)
        rows = slice(t * points_per_frame, t * points_per_frame + count)
        xyz[rows] = points[t, y, x]
        mask[rows] = True
        origin[rows] = np.column_stack((np.full(count, t), y, x))
        source_uv[rows] = data["source_uv"][y, x]
    if not np.isfinite(xyz).all() or not np.isfinite(source_uv).all():
        raise ValueError("Selected coordinates are not representable as float32")
    return dict(scene_xyz=xyz, scene_mask=mask, source_tyx=origin,
                source_uv=source_uv, per_frame_counts=counts,
                per_frame_candidate_counts=candidates,
                coarse_alignment_only=True, training_ready=False)


def make_model_inputs(context, scene):
    """Whitelist one unbatched example; keep context values in their own type.

    Targets, oracle geometry, and material fields may exist in the caller's
    record but are never forwarded. Existing scene fields are rejected to stop
    accidental reuse of an oracle scene under a visual-experiment name.
    """
    if "scene_xyz" in context or "scene_mask" in context:
        raise ValueError("Context must not contain an existing scene")
    missing = set(CONTEXT_KEYS) - set(context)
    if missing:
        raise ValueError(f"Missing context fields: {sorted(missing)}")
    shapes = {"node": (2, 90), "edge": (2, 2, 54), "mask": (2,),
              "history_dt": (8,), "future_dt": (41,),
              "last_position": (2, 3), "cv": (2, 41, 3)}
    for key, shape in shapes.items():
        if tuple(context[key].shape) != shape:
            raise ValueError(f"Expected unbatched {key} shape {shape}")
    surface = context["surface"]
    if (len(surface.shape) != 3 or surface.shape[0] != 2 or surface.shape[2] != 3
            or tuple(context["surface_mask"].shape) != tuple(surface.shape[:-1])):
        raise ValueError("Expected surface [2,S,3] and surface_mask [2,S]")
    for key in ("mask", "surface_mask"):
        if str(context[key].dtype) not in ("bool", "torch.bool"):
            raise ValueError(f"Expected boolean {key}")
    if scene.get("coarse_alignment_only") is not True or scene.get("training_ready") is not False:
        raise ValueError("Expected the declared coarse visual scene adapter output")
    xyz, mask = scene["scene_xyz"], scene["scene_mask"]
    if (not isinstance(xyz, np.ndarray) or xyz.ndim != 2 or xyz.shape[1] != 3
            or not np.issubdtype(xyz.dtype, np.floating) or not np.isfinite(xyz).all()
            or not isinstance(mask, np.ndarray) or mask.shape != xyz.shape[:1]
            or mask.dtype != np.bool_):
        raise ValueError("Expected finite visual scene_xyz [N,3] and boolean scene_mask [N]")
    return {**{key: context[key] for key in CONTEXT_KEYS}, "scene_xyz": xyz, "scene_mask": mask}
