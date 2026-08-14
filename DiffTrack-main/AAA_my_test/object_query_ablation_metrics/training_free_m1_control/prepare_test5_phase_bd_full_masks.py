#!/usr/bin/env python3
"""Freeze Baseline SAM2 object_A masks for the full-object Phase-B/D control."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAA_my_test.object_query_ablation_metrics.common import (  # noqa: E402
    atomic_json,
    atomic_npz,
    load_video_frames,
    sha256_file,
)
from AAA_my_test.object_query_ablation_metrics.extract_masks import (  # noqa: E402
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
    track_masks,
)
from AAA_my_test.object_query_ablation_metrics.full_mask_signature_regions import (  # noqa: E402
    build_signature_partition,
)
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_m1_direct_scaling_phase_bd import (  # noqa: E402
    full_mask_cache_path,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache  # noqa: E402


EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1"
)
DEFAULT_SPARSE_ROOT = (
    EXPERIMENT_ROOT
    / "training_free_m1_direct_enhancement_v2"
    / "test5_20case_5seed"
)
DEFAULT_OUTPUT_ROOT = (
    EXPERIMENT_ROOT
    / "training_free_m1_direct_enhancement_v2"
    / "test5_20case_5seed_sam2_full_mask"
)
REUSE_ROOTS = (
    EXPERIMENT_ROOT / "stage4_metrics/head_scope_trajectory",
    EXPERIMENT_ROOT / "stage3_metrics/head_scope_trajectory",
    EXPERIMENT_ROOT
    / "training_free_m1_multi_object_search_v1/metrics/head_scope_trajectory",
    Path(
        "/data/gaoya/agent-data/outputs/object_query_ablation_metrics/"
        "head_scope_trajectory"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_SPARSE_ROOT / "test5_phase_bd_manifest.json",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def packed_masks(path: Path, baseline: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            shape = tuple(int(value) for value in arrays["mask_shape"].tolist())
            cached_video = Path(str(arrays["video_path"].item()))
            packed = arrays["masks_packed"]
        if (
            len(shape) != 4
            or shape[0] != 49
            or shape[2:] != (704, 1280)
            or cached_video.resolve() != baseline.resolve()
        ):
            return None
        count = int(np.prod(shape))
        return np.unpackbits(packed)[:count].reshape(shape).astype(bool)
    except (OSError, KeyError, ValueError):
        return None


def reuse_candidates(case: str, seed: int) -> list[Path]:
    relative = (
        Path(case)
        / f"seed_{seed:05d}"
        / "object_survival/masks/baseline.npz"
    )
    return [root / relative for root in REUSE_ROOTS]


def save_object_a_cache(
    path: Path,
    masks: np.ndarray,
    baseline: Path,
    *,
    source: str,
) -> dict:
    masks = np.asarray(masks, dtype=bool)
    if masks.shape != (49, 1, 704, 1280):
        raise RuntimeError(f"unexpected object_A masks: {masks.shape}")
    partition = build_signature_partition(masks, ("object_A",))
    audit = partition.audit()
    atomic_npz(
        path,
        masks_packed=np.packbits(masks.reshape(-1)),
        mask_shape=np.asarray(masks.shape, dtype=np.int32),
        video_path=np.asarray(str(baseline.resolve())),
        video_sha256=np.asarray(sha256_file(baseline)),
        segmenter=np.asarray("SAM2.1 Hiera Large video predictor"),
        prompt_frame=np.int32(0),
        token_membership_rule=np.asarray(audit["membership_rule"]),
        anchor_frames=np.asarray(partition.anchor_frames, dtype=np.int32),
        token_counts_by_time=np.asarray(
            audit["union_token_counts_by_time"], dtype=np.int32
        ),
        source=np.asarray(source),
    )
    return {
        "cache": str(path),
        "source": source,
        "mask_nonempty_rate": float(masks[:, 0].any(axis=(1, 2)).mean()),
        "token_count": int(len(partition.union_rows)),
        "token_counts_by_time": audit["union_token_counts_by_time"],
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    samples = list(manifest.get("samples") or [])
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("manifest contains no samples")

    mask_root = args.output_root / "baseline_sam2_full_masks"
    temporary_root = args.output_root / "sam2_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    predictor = None
    records: list[dict] = []
    try:
        for index, sample in enumerate(samples, start=1):
            case = str(sample["case"])
            seed = int(sample["seed"])
            baseline = Path(str(sample["baseline_video"]))
            if not baseline.is_file():
                raise FileNotFoundError(baseline)
            output = full_mask_cache_path(mask_root, case, seed)
            existing = None if args.overwrite else packed_masks(output, baseline)
            if existing is not None and existing.shape[1] == 1:
                record = save_object_a_cache(
                    output, existing, baseline, source="existing full-mask cache"
                )
                state = "reuse-output"
            else:
                reused = None
                reused_path = None
                for candidate in reuse_candidates(case, seed):
                    value = packed_masks(candidate, baseline)
                    if value is not None:
                        reused, reused_path = value, candidate
                        break
                if reused is not None:
                    record = save_object_a_cache(
                        output,
                        reused[:, :1],
                        baseline,
                        source=f"reused object_A from {reused_path}",
                    )
                    state = "reuse-metrics"
                else:
                    if predictor is None:
                        from sam2.build_sam import build_sam2_video_predictor

                        predictor = build_sam2_video_predictor(
                            SAM2_CONFIG,
                            str(SAM2_CHECKPOINT),
                            device=args.device,
                        )
                        predictor.fill_hole_area = 0
                    cache_dir = Path(str(sample["query_cache_dir"]))
                    cache = load_region_cache(cache_dir.parent, cache_dir.name)
                    region_index = next(
                        index
                        for index, region in enumerate(cache.regions)
                        if region.region_name == "object_A"
                    )
                    region = cache.regions[region_index]
                    point_slice = slice(region.point_start, region.point_end)
                    frames, _fps = load_video_frames(baseline)
                    masks = track_masks(
                        predictor,
                        frames,
                        cache.query_points[point_slice],
                        cache.masks_rhw[region_index : region_index + 1],
                        temporary_root,
                        [slice(0, region.point_end - region.point_start)],
                    ).astype(bool)
                    record = save_object_a_cache(
                        output, masks, baseline, source="new Baseline SAM2 propagation"
                    )
                    state = "segment"
                    del frames, masks
                    gc.collect()
                    torch.cuda.empty_cache()
            record.update({"case": case, "seed": seed})
            records.append(record)
            print(
                f"[{index}/{len(samples)}] {state} {case}/seed_{seed:05d} "
                f"tokens={record['token_count']}",
                flush=True,
            )
    finally:
        if predictor is not None:
            del predictor
            gc.collect()
            torch.cuda.empty_cache()

    atomic_json(
        args.output_root / "full_mask_manifest.json",
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "frozen_baseline_sam2_object_A_to_latent_any_intersection_v1",
            "source_manifest": str(args.manifest_path),
            "sample_count": len(records),
            "mask_root": str(mask_root),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
