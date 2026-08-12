#!/usr/bin/env python3
"""Upgrade frozen v1 GT tubes to CoTracker-prompted SAM2 v2 tubes.

This command intentionally reuses the already audited object identities,
CoTracker anchor points, and initial SAM2 masks from a v1 tube.  It isolates the
new behavior under test:

  same-anchor CoTracker points -> independent SAM2 mask (direct)
  nearest other direct anchor -> SAM2 propagation (neighbor candidate)
  final = direct if available, otherwise neighbor candidate

The neighbor candidate is computed even when direct exists for visualization,
but it never replaces a valid direct result.
"""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from AAA_my_test.precompute_toydataset_sam2_regions import build_provider
from AAA_my_test.run_wan_gt_spatiotemporal_correspondence_guidance import (
    DEFAULT_INPUT_LIST,
    DEFAULT_OUTPUT_ROOT,
    HEIGHT,
    PROTOCOL,
    WIDTH,
    atomic_npz,
    atomic_write_json,
    deduplicated_json_paths,
    motion_scores_d0,
    read_source_prefix,
    resize_frames,
    tube_dir,
)


DEFAULT_BASE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--points-per-object", type=int, default=8)
    parser.add_argument("--moving-threshold-d0", type=float, default=0.05)
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _save_upgrade(
    case: str,
    base_root: Path,
    output_root: Path,
    device: str,
    points_per_object: int,
    moving_threshold_d0: float,
    overwrite: bool,
) -> None:
    output = tube_dir(output_root, case)
    required = (output / "tube.npz", output / "manifest.json", output / "complete.json")
    if all(path.is_file() for path in required) and not overwrite:
        print(f"[upgrade] skip {case}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "complete.json").unlink(missing_ok=True)

    base_dir = tube_dir(base_root, case)
    base_tube_path = base_dir / "tube.npz"
    base_manifest_path = base_dir / "manifest.json"
    if not base_tube_path.is_file() or not base_manifest_path.is_file():
        raise FileNotFoundError(f"missing base tube for {case}: {base_dir}")
    base_manifest = _load_json(base_manifest_path)
    source_video = Path(str(base_manifest["source_video"])).expanduser().resolve()
    frames = resize_frames(read_source_prefix(source_video))
    frames_tchw_01 = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    with np.load(base_tube_path, allow_pickle=False) as base:
        anchors = np.asarray(base["anchor_source_frames"], dtype=np.int64)
        legacy_masks = np.asarray(base["masks_othw"], dtype=np.uint8)
        tracks = np.asarray(base["tracks_tn2"], dtype=np.float32)
        visibility = np.asarray(base["visibility_tn"], dtype=bool)
        query_points = np.asarray(base["query_points_n2"], dtype=np.float32)
        names = tuple(str(value) for value in base["region_names"].tolist())
        starts = np.asarray(base["point_starts"], dtype=np.int64)
        ends = np.asarray(base["point_ends"], dtype=np.int64)

    in_bounds = (
        np.isfinite(tracks).all(axis=-1)
        & (tracks[..., 0] >= 0.0)
        & (tracks[..., 0] < float(WIDTH))
        & (tracks[..., 1] >= 0.0)
        & (tracks[..., 1] < float(HEIGHT))
    )
    visibility &= in_bounds
    provider = build_provider(device, points_per_object)
    try:
        prompted = [
            provider.tracker.segment_tracked_point_tube(
                frames_tchw_01,
                anchors,
                tracks[:, int(start) : int(end)],
                visibility[:, int(start) : int(end)],
            )
            for start, end in zip(starts, ends)
        ]
    finally:
        del provider
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    direct = np.stack([row.direct_masks_ahw for row in prompted]).astype(np.uint8)
    neighbor = np.stack([row.neighbor_masks_ahw for row in prompted]).astype(np.uint8)
    final = np.stack([row.final_masks_ahw for row in prompted]).astype(np.uint8)
    prompt_counts = np.stack(
        [row.direct_prompt_counts_a for row in prompted]
    ).astype(np.int16)
    neighbor_sources = np.stack(
        [row.neighbor_source_anchor_a for row in prompted]
    ).astype(np.int16)
    final_sources = np.stack([row.final_source_a for row in prompted]).astype(np.uint8)
    scores = motion_scores_d0(tracks, starts, ends, final)
    moving = scores >= float(moving_threshold_d0)

    atomic_npz(
        output / "tube.npz",
        anchor_source_frames=anchors,
        masks_othw=final,
        direct_masks_othw=direct,
        neighbor_masks_othw=neighbor,
        legacy_masks_othw=legacy_masks,
        direct_prompt_counts_ot=prompt_counts,
        neighbor_source_anchor_ot=neighbor_sources,
        final_mask_source_ot=final_sources,
        tracks_tn2=tracks,
        visibility_tn=visibility.astype(np.uint8),
        query_points_n2=query_points,
        region_names=np.asarray(names),
        point_starts=starts.astype(np.int32),
        point_ends=ends.astype(np.int32),
        moving=moving.astype(np.uint8),
        motion_score_d0=scores,
        pixel_height=np.int32(HEIGHT),
        pixel_width=np.int32(WIDTH),
    )
    base_objects = list(base_manifest.get("objects") or [])
    objects = []
    for object_index, (name, start, end, score, is_moving) in enumerate(
        zip(names, starts, ends, scores, moving)
    ):
        base_object = base_objects[object_index] if object_index < len(base_objects) else {}
        objects.append(
            {
                "name": name,
                "phrase": str(base_object.get("phrase") or name),
                "point_start": int(start),
                "point_end": int(end),
                "motion_score_d0": float(score),
                "moving": bool(is_moving),
                "anchor_visibility_rate": float(visibility[:, int(start) : int(end)].mean()),
                "direct_prompt_counts": prompt_counts[object_index].astype(int).tolist(),
                "direct_mask_frames": np.flatnonzero(
                    direct[object_index].reshape(len(anchors), -1).any(axis=1)
                ).astype(int).tolist(),
                "fallback_candidate_frames": np.flatnonzero(
                    neighbor[object_index].reshape(len(anchors), -1).any(axis=1)
                ).astype(int).tolist(),
                "fallback_applied_frames": np.flatnonzero(
                    final_sources[object_index] == 1
                ).astype(int).tolist(),
                "missing_final_frames": np.flatnonzero(
                    final_sources[object_index] == 2
                ).astype(int).tolist(),
                "neighbor_source_anchor": neighbor_sources[object_index].astype(int).tolist(),
            }
        )
    manifest = {
        "protocol": PROTOCOL,
        "case": case,
        "source_json": base_manifest.get("source_json"),
        "source_video": str(source_video),
        "source_frame_count": int(base_manifest.get("source_frame_count") or len(frames)),
        "source_frame_policy": base_manifest.get("source_frame_policy"),
        "anchor_source_frames": anchors.astype(int).tolist(),
        "latent_anchor_indices": list(range(len(anchors))),
        "source_processing": (
            "reuse audited v1 object identities and CoTracker anchors -> independent "
            "per-anchor SAM2 point prompts; nearest neighboring direct mask is "
            "propagated for audit and applied only when direct is unavailable/empty"
        ),
        "tube_mask_strategy": "cotracker_prompted_sam2",
        "base_tube_root": str(base_root),
        "final_mask_source_codes": {
            "0": "direct CoTracker points prompted SAM2 at the same anchor",
            "1": "neighboring direct mask propagated by SAM2 fallback",
            "2": "missing direct and fallback mask",
        },
        "objects": objects,
        "moving_threshold_d0": float(moving_threshold_d0),
        "uses_gt_instance_masks": False,
        "oracle_information": "future source-video CoTracker trajectories and point-prompted SAM2 masks",
    }
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(
        output / "complete.json",
        {
            "case": case,
            "object_count": len(names),
            "moving_count": int(moving.sum()),
            "direct_frames": int((final_sources == 0).sum()),
            "fallback_frames": int((final_sources == 1).sum()),
            "missing_frames": int((final_sources == 2).sum()),
        },
    )
    (output / "error.txt").unlink(missing_ok=True)
    print(
        f"[upgrade] complete {case}: direct={(final_sources == 0).sum()} "
        f"fallback={(final_sources == 1).sum()} missing={(final_sources == 2).sum()}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.device in {"cuda:4", "4"}:
        raise ValueError("workspace policy forbids physical GPU 4")
    paths = deduplicated_json_paths(args.input_list.expanduser().resolve())
    selected = set(args.case_keys or [])
    known = {path.stem for path in paths}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown case keys: {sorted(unknown)}")
    cases = [path.stem for path in paths if not selected or path.stem in selected]
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for case in cases:
        try:
            _save_upgrade(
                case,
                args.base_root.expanduser().resolve(),
                output_root,
                args.device,
                int(args.points_per_object),
                float(args.moving_threshold_d0),
                bool(args.overwrite),
            )
        except Exception:
            error = traceback.format_exc()
            error_dir = tube_dir(output_root, case)
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "error.txt").write_text(error, encoding="utf-8")
            print(error, flush=True)
            failures.append(case)
            if not args.continue_on_error:
                raise
    if failures:
        raise RuntimeError(f"upgrade failed for {failures}")


if __name__ == "__main__":
    main()
