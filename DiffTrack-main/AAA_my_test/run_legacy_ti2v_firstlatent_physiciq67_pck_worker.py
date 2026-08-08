#!/usr/bin/env python3
"""Capture all-step/all-block/all-head PCK@32 for PhysicIQ67 legacy Wan2.2."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
for path in (ROOT, CODE_ROOT, COTRACKER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (
    WanTI2VArgs,
    build_wan_ti2v_pipeline,
    ensure_firstframe_image,
    run_single_case,
)
from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import (
    LEGACY_VIDEO_ROOT,
    OUTPUT_ROOT,
    REGION_CACHE_ROOT,
    SEEDS_FILE,
    WAN_ROOT,
    all_tasks,
    read_payload,
    run_dir,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (
    CompactFirstLatentCapture,
    atomic_npz,
    load_cotracker,
    object_queries,
    run_cotracker,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache, region_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_args(seed: int) -> WanTI2VArgs:
    manifest = json.loads((LEGACY_VIDEO_ROOT / "batch_manifest.json").read_text(encoding="utf-8"))
    return WanTI2VArgs(
        input_list=SEEDS_FILE,
        output_root=OUTPUT_ROOT,
        model_name="wan_ti2v_5b_legacy_firstlatent_physiciq67_pck50",
        wan_root=WAN_ROOT,
        backend="legacy",
        size="704*1280",
        frame_num=49,
        fps=30,
        seed=int(seed),
        sample_solver="unipc",
        sampling_steps=40,
        sample_shift=5.0,
        cfg_scale=5.0,
        negative_prompt=str(manifest.get("negative_prompt", "")),
        offload_model=False,
        t5_cpu=False,
        convert_model_dtype=False,
        force=True,
    )


def process_task(pipe, cotracker, case, seed: int, device: str, overwrite: bool) -> None:
    output = run_dir(case.key, seed)
    if (output / "complete.json").is_file() and not overwrite:
        print(f"skip {case.key} seed={seed}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    cache = load_region_cache(REGION_CACHE_ROOT, case.key)
    if int(cache.metadata.get("query_context_frame", -1)) != 0:
        raise RuntimeError(f"{case.key}: expected query frame 0 cache")
    query_points, query_regions = object_queries(cache)
    payload = read_payload(case)
    payload, firstframe = ensure_firstframe_image(case.json_path, payload)
    args = build_args(seed)
    capture = CompactFirstLatentCapture(pipe.pipe, query_points, (704, 1280))
    capture.install()
    try:
        result, logs = run_single_case(
            pipe=pipe,
            args=args,
            input_json_path=case.json_path,
            payload=payload,
            firstframe_path=firstframe,
            output_video=output / "generated.mp4",
        )
    finally:
        capture.remove()

    frames = iio.imread(output / "generated.mp4")
    gt_tracks, gt_visibility = run_cotracker(cotracker, frames, query_points, device)
    if not capture.records:
        raise RuntimeError("no self-attention records captured")
    reference = next(iter(capture.records.values()))
    latent_time = int(reference.shape[1])
    anchors = np.arange(latent_time, dtype=np.int64) * 4
    if int(anchors[-1]) >= len(frames):
        anchors = np.rint(np.linspace(0, len(frames) - 1, latent_time)).astype(np.int64)
    gt = gt_tracks[anchors]
    visibility = gt_visibility[anchors] & gt_visibility[0][None]
    visibility[0] = False

    correct32 = np.zeros((40, 30, 24), dtype=np.int32)
    comparisons = np.zeros_like(correct32)
    error_sum = np.zeros((40, 30, 24), dtype=np.float64)
    per_object_correct = np.zeros((len(query_regions), 40, 30, 24), dtype=np.int32)
    per_object_comparisons = np.zeros_like(per_object_correct)
    for (step, layer), predictions in capture.records.items():
        for region_index, (_, point_slice) in enumerate(query_regions):
            valid = visibility[:, point_slice]
            error = np.linalg.norm(
                predictions[:, :, point_slice] - gt[None, :, point_slice], axis=-1
            )
            finite = np.isfinite(error)
            mask = valid[None] & finite
            count = mask.sum(axis=(1, 2)).astype(np.int32)
            correct = ((error <= 32.0) & mask).sum(axis=(1, 2)).astype(np.int32)
            summed_error = np.where(mask, error, 0.0).sum(axis=(1, 2), dtype=np.float64)
            correct32[step, layer, : len(correct)] += correct
            comparisons[step, layer, : len(count)] += count
            error_sum[step, layer, : len(summed_error)] += summed_error
            per_object_correct[region_index, step, layer, : len(correct)] = correct
            per_object_comparisons[region_index, step, layer, : len(count)] = count

    atomic_npz(
        output / "metrics.npz",
        correct32=correct32,
        comparisons=comparisons,
        error_sum=error_sum,
        per_object_correct32=per_object_correct,
        per_object_comparisons=per_object_comparisons,
    )
    manifest = {
        **result,
        "case_key": case.key,
        "category": case.category,
        "source_json": str(case.json_path),
        "formal_compare_video": str(case.formal_video_path),
        "region_cache": str(REGION_CACHE_ROOT / case.key),
        "analysis_protocol": "first_latent_frame_query_to_future_latent_frames",
        "capture_location": "self_attention_post_rmsnorm_post_3d_rope_pre_flash_attention",
        "cfg_branch": "positive_conditional_first_call_only",
        "query_latent_index": 0,
        "query_pixel_frame": 0,
        "latent_anchor_pixel_frames": anchors.tolist(),
        "query_points": query_points.tolist(),
        "query_regions": [region_metadata(region) for region, _ in query_regions],
        "layers": list(range(30)),
        "step_indices": list(range(40)),
        "heads": 24,
        "pck_threshold_px": 32,
        "matching": "per-target-frame Q-to-K argmax; no head averaging",
        "logs": logs,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "case": case.key,
                "seed": int(seed),
                "captured_combinations": int(len(capture.records) * 24),
                "expected_combinations": 40 * 30 * 24,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    tasks = all_tasks()[args.worker_id :: args.num_workers]
    if not tasks:
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pipe = build_wan_ti2v_pipeline(build_args(tasks[0][1]))
    cotracker = load_cotracker(str(args.device))
    for index, (case, seed) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] start {case.key} seed={seed}", flush=True)
        try:
            process_task(pipe, cotracker, case, seed, str(args.device), bool(args.overwrite))
        except Exception:
            output = run_dir(case.key, seed)
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(tasks)}] complete {case.key} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
