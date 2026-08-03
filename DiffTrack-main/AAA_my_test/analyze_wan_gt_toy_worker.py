#!/usr/bin/env python3
"""Probe ToyDataset GT videos with Wan2.2-TI2V-5B Q/K and hidden states.

The first eight pixel frames are encoded as the clean context prefix.  The full
25-frame GT video is encoded with the same Wan VAE, and only its future latents
are noised independently at the requested scheduler steps before one positive
conditional Transformer forward pass.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import imageio.v2 as imageio
import cv2
import numpy as np
import torch
from PIL import Image


CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
for path in (TRAIN0419_ROOT, DIFFSYNTH_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as target

from AAA_my_test import analyze_stage1b_kubric_generation as probe
from AAA_my_test.run_lorav2v_toy_analysis_worker import (
    load_cotracker,
    map_cache_points_to_cover_crop,
    run_cotracker,
)
from AAA_my_test.sam2_region_query_utils import (
    DEFAULT_CACHE_ROOT,
    load_region_cache,
    region_metadata,
    save_region_query_visualizations,
)


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_5b_gt_real_sam2_regions_steps40"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lora-weights-root", type=Path, default=None)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--analysis-layers", type=int, nargs="+", default=[0, 5, 11, 17, 23, 29])
    parser.add_argument("--analysis-step-indices", type=int, nargs="+", default=None)
    parser.add_argument("--analysis-region-cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--analysis-matching-mode",
        choices=("difftrack", "q_to_k", "symmetric", "headwise"),
        default="difftrack",
    )
    parser.add_argument("--analysis-hidden-temperature", type=float, default=0.07)
    parser.add_argument("--analysis-no-hidden", action="store_true")
    parser.add_argument("--analysis-no-video", action="store_true")
    parser.add_argument("--analysis-visualize-layer", type=int, default=17)
    parser.add_argument("--analysis-visualize-step-index", type=int, default=None)
    parser.add_argument("--analysis-heatmap-query-index", type=int, default=0)
    parser.add_argument(
        "--video-field",
        choices=("video", "source_video"),
        default="video",
        help="Dataset base field used as the GT video.",
    )
    parser.add_argument(
        "--vae-encode-mode",
        choices=("whole_video", "framewise_anchors"),
        default="whole_video",
        help="Encode all 25 frames causally, or independently encode the seven DiT anchor frames.",
    )
    parser.add_argument(
        "--query-coordinate-mode",
        choices=("cache", "cover_crop"),
        default="cache",
        help="Use stretch-cache coordinates or map them into Wan's center cover-crop coordinates.",
    )
    parser.add_argument(
        "--save-attention-probabilities",
        action="store_true",
        help="Save the captured per-query spatial attention probabilities.",
    )
    parser.add_argument(
        "--allow-short-gt",
        action="store_true",
        help="Use the longest available 4n+1 prefix when the GT is shorter than num-frames.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cases(dataset_root: Path, case_keys: list[str] | None, video_field: str) -> list[dict]:
    selected = set(case_keys or [])
    cases = []
    for manifest_path in sorted((dataset_root / "cases").glob("case_*/case_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_key = str(payload["case_key"])
        if selected and case_key not in selected:
            continue
        base = payload["base"]
        video_path = Path(base[video_field])
        context_path = Path(base["video"])
        caption = str(base.get("caption") or base.get("short_caption") or "").strip()
        if not video_path.is_file() or not context_path.is_file() or not caption:
            raise RuntimeError(f"invalid dataset entry: {manifest_path}")
        cases.append(
            {
                "case_key": case_key,
                "manifest": str(manifest_path),
                "video": str(video_path),
                "context_video": str(context_path),
                "caption": caption,
            }
        )
    if not cases:
        raise RuntimeError(f"no cases found under {dataset_root}")
    return cases


def prepare_conditioning(
    pipe,
    *,
    prompt: str,
    context_video: list,
    height: int,
    width: int,
    num_frames: int,
    sampling_steps: int,
    sigma_shift: float,
    cfg_scale: float,
    seed: int,
) -> tuple[dict, dict]:
    """Run standard Wan pipeline units without entering the denoising loop."""
    pipe.scheduler.set_timesteps(
        sampling_steps,
        denoising_strength=1.0,
        shift=sigma_shift,
    )
    inputs_posi = {
        "prompt": prompt,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": sampling_steps,
    }
    inputs_nega = {
        "negative_prompt": "",
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": sampling_steps,
    }
    inputs_shared = {
        "input_image": context_video[0],
        "end_image": None,
        "input_video": None,
        "context_video": context_video,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1.0,
        "seed": seed,
        "rand_device": "cpu",
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "cfg_scale": cfg_scale,
        "cfg_merge": False,
        "sigma_shift": sigma_shift,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
        "wantodance_music_path": None,
        "wantodance_reference_image": None,
        "wantodance_fps": 30.0,
        "wantodance_keyframes": None,
        "wantodance_keyframes_mask": None,
        "framewise_decoding": False,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )
    return inputs_shared, inputs_posi


def encode_gt_video(pipe, frames: list, mode: str) -> torch.Tensor:
    pipe.load_models_to_device(["vae"])
    if mode == "framewise_anchors":
        video = pipe.preprocess_video([frames[index] for index in range(0, len(frames), 4)])
        latents = pipe.vae.encode_framewise(video, device=pipe.device)
    else:
        video = pipe.preprocess_video(frames)
        latents = pipe.vae.encode(
            video,
            device=pipe.device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
    return latents.to(dtype=pipe.torch_dtype, device=pipe.device)


def load_video_prefix(path: Path, count: int, height: int, width: int, mode: str) -> list[Image.Image]:
    if mode == "cover_crop":
        return target.core.load_context_frames(
            context_path=path,
            context_frames=count,
            height=height,
            width=width,
            resize_mode="crop",
        )
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def save_gt_video(frames: list, path: Path, fps: int, quality: int) -> None:
    writer = imageio.get_writer(path, fps=fps, quality=quality)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))
    finally:
        writer.close()


def validate_geometry(args, gt_latents: torch.Tensor, inputs_shared: dict, pixel_frames: int) -> int:
    noise = inputs_shared["noise"]
    clean_prefix = inputs_shared.get("clean_prefix_latents")
    if clean_prefix is None:
        raise RuntimeError("context pipeline did not produce clean_prefix_latents")
    if gt_latents.shape != noise.shape:
        raise RuntimeError(f"GT/noise latent shape mismatch: {gt_latents.shape} vs {noise.shape}")
    prefix_len = int(clean_prefix.shape[2])
    if prefix_len != 2:
        raise RuntimeError(f"expected 8 context frames -> 2 clean latents, got {prefix_len}")
    expected_latents = (int(pixel_frames) - 1) // 4 + 1
    if int(gt_latents.shape[2]) != expected_latents:
        raise RuntimeError(
            f"expected {pixel_frames} GT frames -> {expected_latents} latents, "
            f"got {gt_latents.shape[2]}"
        )
    if (int(args.height), int(args.width)) != (512, 896):
        raise RuntimeError("formal SAM2 cache requires Wan analysis geometry 512x896")
    return prefix_len


def process_case(args, pipe, cotracker, case: dict, output_dir: Path) -> None:
    probe.seed_everything(int(args.seed))
    video_path = Path(case["video"])
    context_path = Path(case["context_video"])
    prompt = str(case["caption"])
    gt_frames = load_video_prefix(
        video_path,
        int(args.num_frames),
        int(args.height),
        int(args.width),
        str(args.query_coordinate_mode),
    )
    if len(gt_frames) != int(args.num_frames):
        if not args.allow_short_gt or len(gt_frames) < 9:
            raise RuntimeError(f"loaded {len(gt_frames)}/{args.num_frames} GT frames")
        valid_length = ((len(gt_frames) - 1) // 4) * 4 + 1
        gt_frames = gt_frames[:valid_length]
    context = load_video_prefix(
        context_path,
        int(args.context_frames),
        int(args.height),
        int(args.width),
        str(args.query_coordinate_mode),
    )
    region_cache = load_region_cache(Path(args.analysis_region_cache_root), case["case_key"])
    query_points = region_cache.query_points
    if args.query_coordinate_mode == "cover_crop":
        query_points = map_cache_points_to_cover_crop(
            query_points,
            context_path,
            region_cache.context_frame_rgb.shape[:2],
            (int(args.height), int(args.width)),
        )
    if region_cache.context_frame_rgb.shape[:2] != (int(args.height), int(args.width)):
        raise RuntimeError(f"SAM2 cache geometry mismatch: {region_cache.context_frame_rgb.shape[:2]}")

    layers = sorted(set(int(value) for value in args.analysis_layers))
    steps = args.analysis_step_indices or [0, 10, 20, 29, 39]
    steps = sorted(set(int(value) for value in steps))
    invalid_steps = [step for step in steps if not 0 <= step < int(args.sampling_steps)]
    if invalid_steps:
        raise ValueError(f"invalid analysis steps: {invalid_steps}")

    gt_latents = encode_gt_video(pipe, gt_frames, str(args.vae_encode_mode))
    inputs_shared, inputs_posi = prepare_conditioning(
        pipe,
        prompt=prompt,
        context_video=context,
        height=int(args.height),
        width=int(args.width),
        num_frames=len(gt_frames),
        sampling_steps=int(args.sampling_steps),
        sigma_shift=float(args.sigma_shift),
        cfg_scale=float(args.cfg_scale),
        seed=int(args.seed),
    )
    prefix_len = validate_geometry(args, gt_latents, inputs_shared, len(gt_frames))
    clean_prefix = inputs_shared["clean_prefix_latents"]
    noise = inputs_shared["noise"]

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
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    capture.install()
    try:
        with torch.inference_mode():
            for step_index in steps:
                timestep = pipe.scheduler.timesteps[step_index].unsqueeze(0).to(
                    dtype=pipe.torch_dtype, device=pipe.device
                )
                noised_gt = gt_latents.clone()
                noised_gt[:, :, prefix_len:] = pipe.scheduler.add_noise(
                    gt_latents[:, :, prefix_len:],
                    noise[:, :, prefix_len:],
                    timestep,
                )
                noised_gt[:, :, :prefix_len] = clean_prefix
                inputs_shared["latents"] = noised_gt
                pipe.model_fn(
                    **models,
                    **inputs_shared,
                    **inputs_posi,
                    timestep=timestep,
                )
    finally:
        capture.remove()

    records = sorted(capture.records.values(), key=lambda item: (item.method, item.layer, item.step_index))
    expected = (
        len(records)
        if str(args.analysis_matching_mode) == "headwise"
        else len(layers) * len(steps) * (1 if args.analysis_no_hidden else 2)
    )
    if len(records) != expected:
        raise RuntimeError(f"captured {len(records)}/{expected} records")
    reference = records[0]
    anchors = probe.latent_anchor_frames(reference.grid[0], len(gt_frames))
    query_pixel_frame = int(anchors[reference.query_latent_index])
    cached_query_frame = int(region_cache.metadata["query_context_frame"])
    if query_pixel_frame != cached_query_frame:
        raise RuntimeError(
            f"query frame mismatch: Wan={query_pixel_frame}, SAM2={cached_query_frame}"
        )

    frame_array = np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in gt_frames])
    if not args.analysis_no_video:
        save_gt_video(gt_frames, output_dir / "gt.mp4", int(args.fps), int(args.quality))
    probe.draw_query_points(frame_array[query_pixel_frame], query_points, output_dir / "query_points.png")
    query_visual_files = [
        "query_points.png",
        *save_region_query_visualizations(output_dir, region_cache),
    ]
    gt_tracks, gt_visibility = run_cotracker(
        cotracker, frame_array, query_points, query_pixel_frame, str(args.device)
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
            sliced_record = probe.slice_match_record(record, region.point_start, region.point_end)
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
    if args.save_attention_probabilities:
        np.savez_compressed(
            output_dir / "attention_probabilities.npz",
            **{
                f"{record.method}_layer{record.layer:02d}_step{record.step_index:03d}_probabilities": record.probabilities
                for record in records
            },
        )
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
        visual_files.insert(0, "gt.mp4")
    for region in region_cache.regions:
        point_slice = slice(region.point_start, region.point_end)
        for method in ("qk", "hidden"):
            match = capture.records.get((method, int(args.analysis_visualize_layer), visualize_step))
            if match is None:
                continue
            sliced_match = probe.slice_match_record(match, region.point_start, region.point_end)
            heatmap_name = (
                f"regions/{region.region_name}/heatmap_{method}_"
                f"L{args.analysis_visualize_layer:02d}_S{visualize_step:03d}.png"
            )
            probe.save_heatmap_montage(
                frame_array, anchors, sliced_match, 0, output_dir / heatmap_name
            )
            visual_files.append(heatmap_name)
            if not args.analysis_no_video:
                track_name = (
                    f"regions/{region.region_name}/tracks_{method}_"
                    f"L{args.analysis_visualize_layer:02d}_S{visualize_step:03d}.mp4"
                )
                probe.draw_track_video(
                    frame_array,
                    anchors,
                    sliced_match,
                    gt_tracks[:, point_slice],
                    gt_visibility[:, point_slice],
                    output_dir / track_name,
                    int(args.fps),
                )
                visual_files.append(track_name)

    lora_path = (
        target._resolve_lora_path(args.lora_weights_root)
        if args.lora_weights_root is not None
        else None
    )
    manifest = {
        "case_key": case["case_key"],
        "case_manifest": case["manifest"],
        "model": "Wan2.2-TI2V-5B",
        "model_variant": "lora" if lora_path is not None else "base",
        "checkpoint": str(lora_path) if lora_path is not None else None,
        "wan_root": str(Path(args.wan_root).resolve()),
        "prompt": prompt,
        "gt_video": str(video_path),
        "context_video": str(context_path),
        "analysis_protocol": "Wan_GT_VAE_clean_prefix_fixed_noise_single_forward",
        "vae_encode_mode": str(args.vae_encode_mode),
        "video_field": str(args.video_field),
        "query_coordinate_mode": str(args.query_coordinate_mode),
        "capture_location": "video_self_attention_post_rmsnorm_post_3d_rope_pre_flash_attention",
        "cfg_branch": "positive_conditional_only",
        "query_mode": "sam2_regions",
        "query_region_cache": str((Path(args.analysis_region_cache_root) / case["case_key"]).resolve()),
        "query_regions": [region_metadata(region) for region in region_cache.regions],
        "matching_mode": str(args.analysis_matching_mode),
        "matching_implementation": (
            probe.DIFFTRACK_MATCHING_IMPLEMENTATION
            if str(args.analysis_matching_mode) == "difftrack"
            else "AAA_my_test.GenerationCapture.direct_token_argmax"
        ),
        "seed": int(args.seed),
        "sampling_steps": int(args.sampling_steps),
        "layers": layers,
        "step_indices": steps,
        "scheduler_timesteps": [float(value) for value in pipe.scheduler.timesteps.detach().float().cpu()],
        "scheduler_sigmas": [float(value) for value in pipe.scheduler.sigmas.detach().float().cpu()],
        "gt_pixel_frames": len(gt_frames),
        "requested_gt_pixel_frames": int(args.num_frames),
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
                "model": manifest["model"],
                "model_variant": manifest["model_variant"],
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
    if int(args.num_frames) % 4 != 1 or int(args.context_frames) != 8:
        raise ValueError("controlled Wan comparison requires 4n+1 GT frames and 8 context frames")
    cases = load_cases(args.dataset_root.resolve(), args.case_keys, str(args.video_field))
    assigned = cases[args.worker_id :: args.num_workers]
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lora_path = (
        target._resolve_lora_path(args.lora_weights_root)
        if args.lora_weights_root is not None
        else None
    )
    pipe = target.core.build_pipeline(args.wan_root.resolve(), str(args.device), lora_path)
    cotracker = load_cotracker(str(args.device))
    summary = {
        "worker_id": args.worker_id,
        "assigned_cases": [case["case_key"] for case in assigned],
        "completed": [],
    }
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
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(assigned)}] complete {case['case_key']}", flush=True)
    print(f"worker {args.worker_id} complete: {len(assigned)} cases", flush=True)


if __name__ == "__main__":
    main()
