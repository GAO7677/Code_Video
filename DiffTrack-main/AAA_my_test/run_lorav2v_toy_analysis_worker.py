#!/usr/bin/env python3
"""Persistent worker for raw-physics Wan LoRA context-to-future analysis."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch


CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
for path in (TRAIN0419_ROOT, DIFFSYNTH_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import the target wrapper first so its intended DiffSynth implementation is fixed.
from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as target

from AAA_my_test import analyze_stage1b_kubric_generation as probe
from AAA_my_test.sam2_region_query_utils import (
    DEFAULT_CACHE_ROOT,
    load_region_cache,
    region_metadata,
    save_region_query_visualizations,
)


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")
DEFAULT_WEIGHTS_ROOT = Path(
    "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
    "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_openvid_0613pybullet_lorav2v_step000500_analysis"
)
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
COTRACKER_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--analysis-layers", type=int, nargs="+", default=[0, 5, 11, 17, 23, 29])
    parser.add_argument("--analysis-step-indices", type=int, nargs="+", default=None)
    parser.add_argument("--analysis-region-cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--analysis-matching-mode", choices=("q_to_k", "symmetric"), default="q_to_k")
    parser.add_argument("--analysis-hidden-temperature", type=float, default=0.07)
    parser.add_argument("--analysis-no-hidden", action="store_true")
    parser.add_argument("--analysis-no-video", action="store_true")
    parser.add_argument("--analysis-visualize-layer", type=int, default=17)
    parser.add_argument("--analysis-visualize-step-index", type=int, default=None)
    parser.add_argument("--analysis-heatmap-query-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cases(dataset_root: Path, case_keys: list[str] | None) -> list[dict]:
    selected = set(case_keys or [])
    cases = []
    for manifest_path in sorted((dataset_root / "cases").glob("case_*/case_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_key = str(payload["case_key"])
        if selected and case_key not in selected:
            continue
        base = payload["base"]
        video = Path(base["video"])
        caption = str(base.get("caption") or base.get("short_caption") or "").strip()
        if not video.is_file() or not caption:
            raise RuntimeError(f"invalid ToyDataset base entry: {manifest_path}")
        cases.append(
            {
                "case_key": case_key,
                "manifest": str(manifest_path),
                "video": str(video),
                "caption": caption,
            }
        )
    if not cases:
        raise RuntimeError(f"no cases found under {dataset_root}")
    return cases


def load_cotracker(device: str):
    if str(COTRACKER_ROOT) not in sys.path:
        sys.path.insert(0, str(COTRACKER_ROOT))
    from cotracker.predictor import CoTrackerPredictor

    return CoTrackerPredictor(
        checkpoint=str(COTRACKER_CHECKPOINT), offline=True, v2=False, window_len=60
    ).to(device).eval().requires_grad_(False)


def run_cotracker(
    model,
    frames: np.ndarray,
    query_points: np.ndarray,
    query_pixel_frame: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    input_height, input_width = 384, 512
    native_height, native_width = frames.shape[1:3]
    video = torch.from_numpy(frames).float().div(255.0).permute(0, 3, 1, 2)
    video = torch.nn.functional.interpolate(
        video, size=(input_height, input_width), mode="bilinear", align_corners=True
    ).unsqueeze(0).to(device)
    points = torch.from_numpy(query_points).float().to(device)
    points[:, 0] *= input_width / native_width
    points[:, 1] *= input_height / native_height
    frame_ids = torch.full((len(points), 1), float(query_pixel_frame), device=device)
    queries = torch.cat((frame_ids, points), dim=-1).unsqueeze(0)
    with torch.inference_mode():
        tracks, visibility = model(video, queries=queries, backward_tracking=False)
    tracks = tracks[0].float().cpu().numpy()
    tracks[..., 0] *= max(native_width - 1, 1) / max(input_width - 1, 1)
    tracks[..., 1] *= max(native_height - 1, 1) / max(input_height - 1, 1)
    return tracks, visibility[0].float().cpu().numpy() > 0.5


def process_case(args, pipe, cotracker, case: dict, output_dir: Path) -> None:
    probe.seed_everything(int(args.seed))
    context_path = Path(case["video"])
    prompt = str(case["caption"])
    context = target.core.load_context_frames(
        context_path=context_path,
        context_frames=int(args.context_frames),
        height=int(args.height),
        width=int(args.width),
        resize_mode="crop",
    )
    region_cache = load_region_cache(Path(args.analysis_region_cache_root), case["case_key"])
    query_points = region_cache.query_points
    layers = sorted(set(int(value) for value in args.analysis_layers))
    steps = args.analysis_step_indices or probe.evenly_spaced_steps(int(args.sampling_steps))
    steps = sorted(set(int(value) for value in steps))
    aligned_num_frames = target.core.align_generation_num_frames(int(args.num_frames))
    capture = probe.GenerationCapture(
        pipe=pipe,
        layers=layers,
        step_indices=steps,
        query_points=query_points,
        pixel_hw=(int(args.height), int(args.width)),
        matching_mode=str(args.analysis_matching_mode),
        capture_hidden=not bool(args.analysis_no_hidden),
        hidden_temperature=float(args.analysis_hidden_temperature),
    )
    capture.install()
    try:
        with torch.inference_mode():
            video = pipe(
                prompt=prompt,
                negative_prompt="",
                input_image=context[0],
                context_video=context,
                height=int(args.height),
                width=int(args.width),
                num_frames=aligned_num_frames,
                seed=int(args.seed),
                cfg_scale=float(args.cfg_scale),
                num_inference_steps=int(args.sampling_steps),
                tiled=True,
            )
    finally:
        capture.remove()

    records = sorted(capture.records.values(), key=lambda item: (item.method, item.layer, item.step_index))
    expected = len(layers) * len(steps) * (1 if args.analysis_no_hidden else 2)
    if len(records) != expected:
        raise RuntimeError(f"captured {len(records)}/{expected} records")
    reference = records[0]
    generated_frames = probe.tensor_video_to_uint8(video)
    anchors = probe.latent_anchor_frames(reference.grid[0], len(generated_frames))
    query_pixel_frame = int(anchors[reference.query_latent_index])
    cached_query_frame = int(region_cache.metadata["query_context_frame"])
    if query_pixel_frame != cached_query_frame:
        raise RuntimeError(
            f"query frame mismatch: DiT={query_pixel_frame}, SAM2 cache={cached_query_frame}"
        )
    if not args.analysis_no_video:
        probe.save_video(video, str(output_dir / "generated.mp4"), fps=int(args.fps), quality=int(args.quality))
    probe.draw_query_points(generated_frames[query_pixel_frame], query_points, output_dir / "query_points.png")
    query_visual_files = [
        "query_points.png",
        *save_region_query_visualizations(output_dir, region_cache),
    ]
    gt_tracks, gt_visibility = run_cotracker(
        cotracker, generated_frames, query_points, query_pixel_frame, str(args.device)
    )
    np.savez_compressed(
        output_dir / "cotracker_pseudo_gt.npz",
        tracks=gt_tracks,
        visibility=gt_visibility,
        query_points=query_points,
        latent_anchor_frames=anchors,
    )
    rows = []
    for region in region_cache.regions:
        point_slice = slice(region.point_start, region.point_end)
        for record in records:
            sliced_record = probe.slice_match_record(
                record, region.point_start, region.point_end
            )
            row = probe.evaluate_record(
                sliced_record,
                gt_tracks[:, point_slice],
                gt_visibility[:, point_slice],
                anchors,
                (int(args.height), int(args.width)),
            )
            row.update(region_metadata(region))
            rows.append(row)
    probe.save_records(output_dir, records)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    visualize_step = (
        int(args.analysis_visualize_step_index)
        if args.analysis_visualize_step_index is not None
        else steps[-1]
    )
    visual_files = list(dict.fromkeys(query_visual_files))
    if not args.analysis_no_video:
        visual_files.insert(0, "generated.mp4")
    for region in region_cache.regions:
        point_slice = slice(region.point_start, region.point_end)
        for method in ("qk", "hidden"):
            match = capture.records.get(
                (method, int(args.analysis_visualize_layer), visualize_step)
            )
            if match is None:
                continue
            sliced_match = probe.slice_match_record(
                match, region.point_start, region.point_end
            )
            heatmap_name = (
                f"regions/{region.region_name}/heatmap_{method}_"
                f"L{args.analysis_visualize_layer:02d}_S{visualize_step:03d}.png"
            )
            probe.save_heatmap_montage(
                generated_frames, anchors, sliced_match, 0, output_dir / heatmap_name
            )
            visual_files.append(heatmap_name)
            if not args.analysis_no_video:
                track_name = (
                    f"regions/{region.region_name}/tracks_{method}_"
                    f"L{args.analysis_visualize_layer:02d}_S{visualize_step:03d}.mp4"
                )
                probe.draw_track_video(
                    generated_frames,
                    anchors,
                    sliced_match,
                    gt_tracks[:, point_slice],
                    gt_visibility[:, point_slice],
                    output_dir / track_name,
                    int(args.fps),
                )
                visual_files.append(track_name)
    lora_path = target._resolve_lora_path(args.weights_root)
    manifest = {
        "case_key": case["case_key"],
        "case_manifest": case["manifest"],
        "model": "wan_openvid_0613pybullet_lorav2v",
        "conditioning_mode": "context_aware",
        "analysis_protocol": "last_clean_context_latent_to_future_latents",
        "capture_location": "video_self_attention_post_rmsnorm_post_3d_rope_pre_flash_attention",
        "cfg_branch": "positive_conditional_first_call_only",
        "query_mode": "sam2_regions",
        "query_region_cache": str(
            (Path(args.analysis_region_cache_root) / case["case_key"]).resolve()
        ),
        "query_regions": [region_metadata(region) for region in region_cache.regions],
        "matching_mode": str(args.analysis_matching_mode),
        "weights_root": str(args.weights_root.resolve()),
        "checkpoint": str(lora_path),
        "context_video": str(context_path),
        "prompt": prompt,
        "seed": int(args.seed),
        "sampling_steps": int(args.sampling_steps),
        "layers": layers,
        "step_indices": steps,
        "scheduler_timesteps": [float(value) for value in pipe.scheduler.timesteps.detach().float().cpu()],
        "scheduler_sigmas": [float(value) for value in pipe.scheduler.sigmas.detach().float().cpu()],
        "requested_num_frames": int(args.num_frames),
        "generated_pixel_frames": int(len(generated_frames)),
        "context_pixel_frames": len(context),
        "context_source_frame_indices": list(range(len(context))),
        "clean_prefix_latents": int(reference.clean_prefix_latents),
        "token_grid": list(reference.grid),
        "query_latent_index": int(reference.query_latent_index),
        "query_pixel_frame": query_pixel_frame,
        "future_latent_indices": list(range(reference.clean_prefix_latents, reference.grid[0])),
        "latent_anchor_pixel_frames": anchors.tolist(),
        "query_points": query_points.tolist(),
        "height": int(args.height),
        "width": int(args.width),
        "cfg_scale": float(args.cfg_scale),
        "object_branch_enabled": False,
        "files": visual_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    probe.write_report(output_dir, rows, manifest, visual_files)
    (output_dir / "complete.json").write_text(
        json.dumps(
            {
                "case_key": case["case_key"],
                "checkpoint": str(lora_path),
                "record_count": len(records),
                "metric_row_count": len(rows),
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
    cases = load_cases(args.dataset_root.resolve(), args.case_keys)
    assigned = cases[args.worker_id :: args.num_workers]
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lora_path = target._resolve_lora_path(args.weights_root)
    pipe = target.core.build_pipeline(
        Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"), str(args.device), lora_path
    )
    cotracker = load_cotracker(str(args.device))
    summary = {"worker_id": args.worker_id, "assigned_cases": [case["case_key"] for case in assigned], "completed": []}
    summary_path = output_root / f"worker_{args.worker_id:02d}.json"
    for index, case in enumerate(assigned, start=1):
        case_output = output_root / "cases" / case["case_key"]
        if (case_output / "complete.json").exists() and not args.overwrite:
            summary["completed"].append(case["case_key"])
            print(f"[{index}/{len(assigned)}] skip {case['case_key']}", flush=True)
            continue
        case_output.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(assigned)}] start {case['case_key']}", flush=True)
        try:
            process_case(args, pipe, cotracker, case, case_output)
        except Exception:
            (case_output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        summary["completed"].append(case["case_key"])
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(assigned)}] complete {case['case_key']}", flush=True)
    print(f"worker {args.worker_id} complete: {len(assigned)} cases", flush=True)


if __name__ == "__main__":
    main()
