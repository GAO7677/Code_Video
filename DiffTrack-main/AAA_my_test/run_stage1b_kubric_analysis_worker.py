#!/usr/bin/env python3
"""Persistent-model worker for ToyDataset Stage1b generation analysis."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from AAA_my_test import analyze_stage1b_kubric_generation as analysis
from AAA_my_test.sam2_region_query_utils import (
    load_region_cache,
    region_metadata,
    save_region_query_visualizations,
)


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")


def build_parser() -> argparse.ArgumentParser:
    parser = analysis.build_parser()
    for action in parser._actions:
        if action.dest in {"context_video", "prompt"}:
            action.required = False
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_cases(dataset_root: Path, case_keys: list[str] | None) -> list[dict]:
    cases = []
    for manifest_path in sorted((dataset_root / "cases").glob("case_*/case_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_key = str(payload["case_key"])
        if case_keys and case_key not in set(case_keys):
            continue
        base = payload["base"]
        video = Path(base["video"])
        caption = str(base.get("caption") or base.get("short_caption") or "").strip()
        if not video.is_file():
            raise FileNotFoundError(f"missing base video for {case_key}: {video}")
        if not caption:
            raise ValueError(f"missing base caption for {case_key}: {manifest_path}")
        cases.append(
            {
                "case_key": case_key,
                "manifest": str(manifest_path),
                "video": str(video),
                "caption": caption,
            }
        )
    if not cases:
        raise RuntimeError(f"no ToyDataset cases found under {dataset_root}")
    return cases


def process_case(
    args: argparse.Namespace,
    model,
    model_args,
    load_info: dict,
    case: dict,
    output_dir: Path,
) -> None:
    analysis.seed_everything(int(args.seed))
    pipe = model.pipe
    context_path = Path(case["video"])
    prompt = str(case["caption"])
    frames, frame_indices = analysis.base._load_context_video(
        video_path=context_path,
        target_context_frames=int(args.context_frames),
    )
    context_video = analysis.base.preprocess_video_rgb_uint8(
        frames, (int(args.height), int(args.width))
    )
    context_pil = analysis.base._tensor_video_to_pil_list(context_video)
    region_cache = load_region_cache(Path(args.analysis_region_cache_root), case["case_key"])
    query_points = region_cache.query_points
    layers = sorted(set(int(layer) for layer in args.analysis_layers))
    step_indices = args.analysis_step_indices or analysis.evenly_spaced_steps(
        int(args.sampling_steps)
    )
    step_indices = sorted(set(int(step) for step in step_indices))

    with torch.inference_mode():
        object_context_raw, object_debug = analysis.kubric_infer._build_object_context(
            model,
            context_video_single=context_video,
            prompt=prompt,
            video_path=str(context_path),
        )
        object_context, ablation_debug = analysis.base._apply_object_context_ablation(
            object_context_raw,
            mode=str(args.object_context_ablation),
            random_seed=args.object_context_random_seed,
            random_scale=float(args.object_context_random_scale),
            scale_factor=float(args.object_context_scale_factor),
            token_norm_max=args.object_context_token_norm_max,
        )
        object_debug["object_context_ablation"] = ablation_debug
        capture = analysis.GenerationCapture(
            pipe=pipe,
            layers=layers,
            step_indices=step_indices,
            query_points=query_points,
            pixel_hw=(int(args.height), int(args.width)),
            matching_mode=str(args.analysis_matching_mode),
            capture_hidden=not bool(args.analysis_no_hidden),
            hidden_temperature=float(args.analysis_hidden_temperature),
        )
        capture.install()
        try:
            pipe_kwargs = dict(
                prompt=prompt,
                negative_prompt="",
                context_video=context_pil,
                seed=int(args.seed),
                tiled=True,
                height=int(args.height),
                width=int(args.width),
                num_frames=int(args.num_frames),
                num_inference_steps=int(args.sampling_steps),
                cfg_scale=float(args.cfg_scale),
            )
            if bool(getattr(model, "enable_object_branch", False)):
                pipe_kwargs["object_context"] = object_context
            video = pipe(**pipe_kwargs)
        finally:
            capture.remove()

    records = sorted(
        capture.records.values(), key=lambda item: (item.method, item.layer, item.step_index)
    )
    expected = len(layers) * len(step_indices) * (1 if args.analysis_no_hidden else 2)
    if len(records) != expected:
        observed = [(record.method, record.layer, record.step_index) for record in records]
        raise RuntimeError(f"captured {len(records)}/{expected} records: {observed}")
    reference_record = records[0]
    generated_frames = analysis.tensor_video_to_uint8(video)
    anchors = analysis.latent_anchor_frames(reference_record.grid[0], len(generated_frames))
    query_pixel_frame = int(anchors[reference_record.query_latent_index])
    cached_query_frame = int(region_cache.metadata["query_context_frame"])
    if query_pixel_frame != cached_query_frame:
        raise RuntimeError(
            f"query frame mismatch: DiT={query_pixel_frame}, SAM2 cache={cached_query_frame}"
        )

    generated_video_name = "generated.mp4"
    if not args.analysis_no_video:
        analysis.save_video(
            video,
            str(output_dir / generated_video_name),
            fps=int(args.fps),
            quality=int(args.quality),
        )
    analysis.draw_query_points(
        generated_frames[query_pixel_frame], query_points, output_dir / "query_points.png"
    )
    query_visual_files = [
        "query_points.png",
        *save_region_query_visualizations(output_dir, region_cache),
    ]

    gt_tracks = None
    gt_visibility = None
    if not args.analysis_no_cotracker:
        gt_tracks, gt_visibility = analysis.run_cotracker(
            model, generated_frames, query_points, query_pixel_frame
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
            sliced_record = analysis.slice_match_record(
                record, region.point_start, region.point_end
            )
            row = analysis.evaluate_record(
                sliced_record,
                None if gt_tracks is None else gt_tracks[:, point_slice],
                None if gt_visibility is None else gt_visibility[:, point_slice],
                anchors,
                (int(args.height), int(args.width)),
            )
            row.update(region_metadata(region))
            rows.append(row)
    analysis.save_records(output_dir, records)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    visualize_layer = (
        int(args.analysis_visualize_layer)
        if args.analysis_visualize_layer is not None
        else layers[len(layers) // 2]
    )
    visualize_step = (
        int(args.analysis_visualize_step_index)
        if args.analysis_visualize_step_index is not None
        else step_indices[-1]
    )
    visual_files = list(dict.fromkeys(query_visual_files))
    if not args.analysis_no_video:
        visual_files.insert(0, generated_video_name)
    for region in region_cache.regions:
        point_slice = slice(region.point_start, region.point_end)
        for method in ("qk", "hidden"):
            match = capture.records.get((method, visualize_layer, visualize_step))
            if match is None:
                continue
            sliced_match = analysis.slice_match_record(
                match, region.point_start, region.point_end
            )
            heatmap_name = (
                f"regions/{region.region_name}/heatmap_{method}_"
                f"L{visualize_layer:02d}_S{visualize_step:03d}.png"
            )
            analysis.save_heatmap_montage(
                generated_frames, anchors, sliced_match, 0, output_dir / heatmap_name
            )
            visual_files.append(heatmap_name)
            if not args.analysis_no_video:
                track_name = (
                    f"regions/{region.region_name}/tracks_{method}_"
                    f"L{visualize_layer:02d}_S{visualize_step:03d}.mp4"
                )
                analysis.draw_track_video(
                    generated_frames,
                    anchors,
                    sliced_match,
                    None if gt_tracks is None else gt_tracks[:, point_slice],
                    None if gt_visibility is None else gt_visibility[:, point_slice],
                    output_dir / track_name,
                    int(args.fps),
                )
                visual_files.append(track_name)

    checkpoint_path = Path(
        analysis.base.tvn._resolve_checkpoint_file(args.checkpoint)
    ).resolve()
    manifest = {
        "case_key": case["case_key"],
        "case_manifest": case["manifest"],
        "analysis_protocol": "last_clean_context_latent_to_future_latents",
        "capture_location": "video_self_attention_post_rmsnorm_post_3d_rope_pre_flash_attention",
        "cfg_branch": "positive_conditional_first_call_only",
        "query_mode": "sam2_regions",
        "query_region_cache": str(
            (Path(args.analysis_region_cache_root) / case["case_key"]).resolve()
        ),
        "query_regions": [region_metadata(region) for region in region_cache.regions],
        "matching_mode": str(args.analysis_matching_mode),
        "matching_implementation": (
            analysis.DIFFTRACK_MATCHING_IMPLEMENTATION
            if str(args.analysis_matching_mode) == "difftrack"
            else "AAA_my_test.GenerationCapture.direct_token_argmax"
        ),
        "checkpoint": str(checkpoint_path),
        "context_video": str(context_path),
        "prompt": prompt,
        "seed": int(args.seed),
        "sampling_steps": int(args.sampling_steps),
        "layers": layers,
        "step_indices": step_indices,
        "scheduler_timesteps": [
            float(value) for value in pipe.scheduler.timesteps.detach().float().cpu()
        ],
        "scheduler_sigmas": [
            float(value) for value in pipe.scheduler.sigmas.detach().float().cpu()
        ]
        if getattr(pipe.scheduler, "sigmas", None) is not None
        else None,
        "requested_num_frames": int(args.num_frames),
        "generated_pixel_frames": int(len(generated_frames)),
        "context_pixel_frames": int(context_video.shape[1]),
        "context_source_frame_indices": frame_indices.tolist(),
        "clean_prefix_latents": int(reference_record.clean_prefix_latents),
        "token_grid": list(reference_record.grid),
        "query_latent_index": int(reference_record.query_latent_index),
        "query_pixel_frame": query_pixel_frame,
        "future_latent_indices": list(
            range(reference_record.clean_prefix_latents, reference_record.grid[0])
        ),
        "latent_anchor_pixel_frames": anchors.tolist(),
        "query_points": query_points.tolist(),
        "height": int(args.height),
        "width": int(args.width),
        "cfg_scale": float(args.cfg_scale),
        "object_branch_enabled": bool(getattr(model, "enable_object_branch", False)),
        "object_debug": object_debug,
        "model_args": {
            "height": int(model_args.height),
            "width": int(model_args.width),
            "num_frames": int(model_args.num_frames),
            "fixed_num_context_frames": int(model_args.fixed_num_context_frames),
        },
        "load_info": analysis.base._summarize_load_info(load_info),
        "files": visual_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    analysis.write_report(output_dir, rows, manifest, visual_files)
    (output_dir / "complete.json").write_text(
        json.dumps(
            {
                "case_key": case["case_key"],
                "checkpoint": str(checkpoint_path),
                "record_count": len(records),
                "metric_row_count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    if args.enable_vjepa_guidance:
        raise ValueError("V-JEPA inference guidance is disabled for correspondence analysis")
    args.device = args.analysis_device or "cuda:0"
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.dataset_root.expanduser().resolve(), args.case_keys)
    assigned = [case for index, case in enumerate(cases) if index % args.num_workers == args.worker_id]
    if not assigned:
        raise RuntimeError(f"worker {args.worker_id} has no assigned cases")
    analysis.kubric_infer.base.t0705 = analysis.trainmod
    analysis.kubric_infer.base._build_object_context = analysis.kubric_infer._build_object_context
    analysis.kubric_infer.base._build_model_args = analysis.kubric_infer._build_model_args
    model, model_args, load_info = analysis.base._build_runtime_model(args)
    model.pipe.dit.eval()

    worker_summary = {
        "worker_id": int(args.worker_id),
        "num_workers": int(args.num_workers),
        "device": str(args.device),
        "assigned_cases": [case["case_key"] for case in assigned],
        "completed": [],
    }
    summary_path = output_root / f"worker_{args.worker_id:02d}.json"
    for position, case in enumerate(assigned, start=1):
        case_output = output_root / "cases" / case["case_key"]
        complete_path = case_output / "complete.json"
        if complete_path.exists() and not args.overwrite:
            print(f"[{position}/{len(assigned)}] skip {case['case_key']}", flush=True)
            worker_summary["completed"].append(case["case_key"])
            continue
        case_output.mkdir(parents=True, exist_ok=True)
        print(f"[{position}/{len(assigned)}] start {case['case_key']}", flush=True)
        try:
            process_case(args, model, model_args, load_info, case, case_output)
        except Exception:
            (case_output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            summary_path.write_text(
                json.dumps(worker_summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise
        worker_summary["completed"].append(case["case_key"])
        summary_path.write_text(
            json.dumps(worker_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{position}/{len(assigned)}] complete {case['case_key']}", flush=True)
    print(f"worker {args.worker_id} complete: {len(assigned)} cases", flush=True)


if __name__ == "__main__":
    main()
