#!/usr/bin/env python3
"""Run DiffTrack real-video probing on ordinary MP4 videos with CoTracker tracks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "diffusers" / "src"))
sys.path.insert(0, str(REPO_ROOT))

import diffusers
from diffusers import CogVideoXPipeline
from utils.confidence_attention_score import ConfidenceAttentionScore
from AAA_my_test.sam2_region_query_utils import (
    RegionQueryCache,
    load_region_cache,
    save_region_query_visualizations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["cogvideox_t2v_2b", "cogvideox_t2v_5b"], default="cogvideox_t2v_2b")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--inverse-steps", nargs="+", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matching-accuracy", action="store_true")
    parser.add_argument("--conf-attn-score", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visualize-layer", type=int, default=17)
    parser.add_argument("--visualize-step", type=int, default=39)
    return parser.parse_args()


def seed_everything(seed: int, device: str) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def read_video(
    path: Path,
    num_frames: int,
    height: int,
    width: int,
    source_start_frame: int,
) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_index >= source_start_frame:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"{path} has no frames at or after {source_start_frame}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(256, 256), mode="bilinear", align_corners=False)
    return F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)


class RegionMatchingEvaluator:
    """DiffTrack evaluator with independent object/background metrics."""

    metric_names = ("mean_error_px", "pck8", "pck16", "pck32")

    def __init__(
        self,
        timestep_num: int,
        layer_num: int,
        gt_tracks: torch.Tensor,
        gt_visibility: torch.Tensor,
        regions: list[dict],
        visualize_layer: int,
        visualize_step: int,
    ) -> None:
        self.pck = torch.zeros([layer_num, timestep_num])
        self.gt_tracks = gt_tracks
        self.gt_visibility = gt_visibility.bool()
        self.regions = regions
        self.values = {
            name: np.full((len(regions), layer_num, timestep_num), np.nan, dtype=np.float32)
            for name in self.metric_names
        }
        self.comparisons = np.zeros((len(regions), layer_num, timestep_num), dtype=np.int64)
        self.visualize_layer = int(visualize_layer)
        self.visualize_step = int(visualize_step)
        self.visualized_tracks: np.ndarray | None = None

    def update(self, pred_tracks: torch.Tensor, layer: int, timestep_idx: int) -> None:
        error = torch.linalg.norm(pred_tracks - self.gt_tracks, dim=-1)
        valid_all = self.gt_visibility.clone()
        valid_all[:, 0] = False
        values = error[valid_all]
        self.pck[layer, timestep_idx] = (
            (values < 8).float().mean() * 100 if values.numel() else 0.0
        )
        for region_index, region in enumerate(self.regions):
            point_slice = slice(int(region["point_start"]), int(region["point_end"]))
            valid = valid_all[:, :, point_slice]
            region_error = error[:, :, point_slice][valid]
            self.comparisons[region_index, layer, timestep_idx] = int(region_error.numel())
            if not region_error.numel():
                continue
            self.values["mean_error_px"][region_index, layer, timestep_idx] = float(
                region_error.mean().item()
            )
            for threshold in (8, 16, 32):
                self.values[f"pck{threshold}"][region_index, layer, timestep_idx] = float(
                    (region_error <= threshold).float().mean().item() * 100
                )
        if layer == self.visualize_layer and timestep_idx == self.visualize_step:
            self.visualized_tracks = pred_tracks[0].detach().float().cpu().numpy()

    def metric_rows(self, descriptor: str, steps: list[int]) -> list[dict]:
        rows = []
        for region_index, region in enumerate(self.regions):
            for layer in range(self.pck.shape[0]):
                for step in steps:
                    comparisons = int(self.comparisons[region_index, layer, step])
                    row = {
                        "method": descriptor,
                        "layer": layer,
                        "step_index": step,
                        "comparisons": comparisons,
                        **{key: region.get(key) for key in (
                            "region_name",
                            "region_type",
                            "region_phrase",
                            "region_slot",
                            "point_start",
                            "point_end",
                        )},
                    }
                    for metric in self.metric_names:
                        value = self.values[metric][region_index, layer, step]
                        row[metric] = None if np.isnan(value) else float(value)
                    rows.append(row)
        return rows


def load_pipeline(args: argparse.Namespace) -> CogVideoXPipeline:
    model_id = args.model_path or ("THUDM/CogVideoX-2b" if args.model.endswith("2b") else "THUDM/CogVideoX-5b")
    pipe = CogVideoXPipeline.from_pretrained(
        str(model_id),
        torch_dtype=torch.bfloat16,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    ).to(args.device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def write_matrix(path: Path, matrix: torch.Tensor, measured_steps: list[int]) -> None:
    with path.open("w") as handle:
        handle.write("inverse_steps: " + ", ".join(map(str, measured_steps)) + "\n")
        for layer, row in enumerate(matrix):
            values = ", ".join(f"{row[step].item():.4f}" for step in measured_steps)
            handle.write(f"Layer {layer}: {values}\n")


def best_pck_rows(sample_id: str, descriptor: str, matrix: torch.Tensor, steps: list[int]) -> list[dict]:
    rows = []
    for step in steps:
        values = matrix[:, step]
        best_value, best_layer = values.max(dim=0)
        rows.append(
            {
                "sample_id": sample_id,
                "descriptor": descriptor,
                "inverse_step": step,
                "best_layer": int(best_layer),
                "best_pck8": float(best_value),
                "mean_layer_pck8": float(values.mean()),
            }
        )
    return rows


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def save_checkpoint(
    save_dir: Path,
    completed_steps: list[int],
    qk_evaluator: RegionMatchingEvaluator | None,
    feat_evaluator: RegionMatchingEvaluator | None,
    score: ConfidenceAttentionScore | None,
) -> None:
    state_path = save_dir / "step_state.npz"
    temporary_state = state_path.with_suffix(".npz.tmp")
    arrays = {"completed_steps": np.asarray(completed_steps, dtype=np.int64)}
    if qk_evaluator is not None and feat_evaluator is not None:
        arrays["qk_pck"] = qk_evaluator.pck.cpu().numpy()
        arrays["feature_pck"] = feat_evaluator.pck.cpu().numpy()
        arrays["qk_comparisons"] = qk_evaluator.comparisons
        arrays["feature_comparisons"] = feat_evaluator.comparisons
        for metric in RegionMatchingEvaluator.metric_names:
            arrays[f"qk_{metric}"] = qk_evaluator.values[metric]
            arrays[f"feature_{metric}"] = feat_evaluator.values[metric]
        if qk_evaluator.visualized_tracks is not None:
            arrays["qk_visualized_tracks"] = qk_evaluator.visualized_tracks
        if feat_evaluator.visualized_tracks is not None:
            arrays["feature_visualized_tracks"] = feat_evaluator.visualized_tracks
    with temporary_state.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary_state, state_path)

    if score is not None:
        for filename, dataframe in (
            ("confidence_score.csv", score.attention_max_df),
            ("attention_score.csv", score.attention_sum_df),
        ):
            path = save_dir / filename
            temporary = path.with_suffix(path.suffix + ".tmp")
            dataframe.to_csv(temporary)
            os.replace(temporary, path)


def restore_checkpoint(
    save_dir: Path,
    qk_evaluator: RegionMatchingEvaluator | None,
    feat_evaluator: RegionMatchingEvaluator | None,
    score: ConfidenceAttentionScore | None,
) -> list[int]:
    state_path = save_dir / "step_state.npz"
    if not state_path.exists():
        return []
    state = np.load(state_path)
    if qk_evaluator is not None and feat_evaluator is not None:
        qk_evaluator.pck.copy_(torch.from_numpy(state["qk_pck"]))
        feat_evaluator.pck.copy_(torch.from_numpy(state["feature_pck"]))
        if "qk_comparisons" in state:
            qk_evaluator.comparisons[...] = state["qk_comparisons"]
            feat_evaluator.comparisons[...] = state["feature_comparisons"]
            for metric in RegionMatchingEvaluator.metric_names:
                qk_evaluator.values[metric][...] = state[f"qk_{metric}"]
                feat_evaluator.values[metric][...] = state[f"feature_{metric}"]
            if "qk_visualized_tracks" in state:
                qk_evaluator.visualized_tracks = state["qk_visualized_tracks"].copy()
            if "feature_visualized_tracks" in state:
                feat_evaluator.visualized_tracks = state["feature_visualized_tracks"].copy()
    if score is not None:
        for filename, attribute in (
            ("confidence_score.csv", "attention_max_df"),
            ("attention_score.csv", "attention_sum_df"),
        ):
            path = save_dir / filename
            if path.exists():
                setattr(score, attribute, pd.read_csv(path, index_col=[0, 1]))
    return [int(step) for step in state["completed_steps"]]


def resize_region_cache(cache: RegionQueryCache, height: int, width: int) -> RegionQueryCache:
    source_h, source_w = cache.context_frame_rgb.shape[:2]
    points = cache.query_points.copy()
    points[:, 0] *= width / source_w
    points[:, 1] *= height / source_h
    masks = np.stack(
        [
            cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            for mask in cache.masks_rhw
        ]
    ).astype(np.uint8)
    frame = cv2.resize(cache.context_frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    metadata = dict(cache.metadata)
    metadata.update({"height": height, "width": width, "resize_mode": "DiffTrack_256_then_480x720"})
    return RegionQueryCache(cache.case_key, points, masks, cache.regions, frame, metadata)


def save_input_video(video_tchw: torch.Tensor, path: Path, fps: int = 30) -> None:
    writer = imageio.get_writer(path, fps=fps, quality=6)
    for frame in video_tchw:
        writer.append_data(frame.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy())
    writer.close()


def point_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(count):
        hue = int(round(179 * index / max(count, 1)))
        bgr = cv2.cvtColor(np.uint8([[[hue, 220, 250]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(value) for value in bgr))
    return colors


def draw_region_track_video(
    video_tchw: torch.Tensor,
    predicted: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    region: dict,
    descriptor: str,
    output_path: Path,
    fps: int = 30,
) -> None:
    point_slice = slice(int(region["point_start"]), int(region["point_end"]))
    pred = predicted[:, point_slice]
    gt = target[:, point_slice]
    visible = visibility[:, point_slice].astype(bool)
    frames = video_tchw.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()
    colors = point_colors(pred.shape[1])
    writer = imageio.get_writer(output_path, fps=fps, quality=6)
    for frame_index, frame in enumerate(frames):
        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for point_index, color in enumerate(colors):
            for previous in range(1, frame_index + 1):
                if not np.isfinite(pred[previous - 1 : previous + 1, point_index]).all():
                    continue
                p0 = tuple(np.rint(pred[previous - 1, point_index]).astype(int))
                p1 = tuple(np.rint(pred[previous, point_index]).astype(int))
                cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
            if np.isfinite(pred[frame_index, point_index]).all():
                point = tuple(np.rint(pred[frame_index, point_index]).astype(int))
                cv2.rectangle(canvas, (point[0] - 4, point[1] - 4), (point[0] + 4, point[1] + 4), color, 2)
            if visible[frame_index, point_index]:
                point = tuple(np.rint(gt[frame_index, point_index]).astype(int))
                cv2.circle(canvas, point, 4, color, -1, cv2.LINE_AA)
        label = f"GT CogVideoX {descriptor} L17 S39 | circle=CoTracker square=match"
        cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    writer.close()


def main() -> None:
    args = parse_args()
    expected_diffusers = REPO_ROOT / "diffusers" / "src" / "diffusers"
    actual_diffusers = Path(diffusers.__file__).resolve().parent
    if actual_diffusers != expected_diffusers:
        raise RuntimeError(f"Expected DiffTrack diffusers at {expected_diffusers}, loaded {actual_diffusers}")
    if not args.matching_accuracy and not args.conf_attn_score:
        raise ValueError("Enable --matching-accuracy and/or --conf-attn-score")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.track_dir / "tracks_manifest.json").read_text())
    samples = manifest["samples"][args.start : args.end]
    if not samples:
        raise ValueError("The selected sample range is empty")
    if (args.height, args.width, args.num_frames) != (
        manifest["height"], manifest["width"], manifest["num_frames"]
    ):
        raise ValueError("Analysis geometry must match the prepared CoTracker tracks")

    steps = args.inverse_steps or [0, 10, 20, 29, 39]
    invalid_steps = [step for step in steps if not 0 <= step < args.num_inference_steps]
    if invalid_steps:
        raise ValueError(f"Invalid inverse steps: {invalid_steps}")

    pipe = load_pipeline(args)
    layer_count = pipe.transformer.config.num_layers
    summary_rows = []

    for sample in samples:
        sample_id = sample["sample_id"]
        save_dir = args.output_dir / sample_id
        complete_marker = save_dir / "complete.json"
        if complete_marker.exists() and not args.overwrite:
            print(f"Skip completed {sample_id}")
            continue
        save_dir.mkdir(parents=True, exist_ok=True)

        video_path = args.dataset_root / sample["canonical_video"]
        source_start_frame = int(manifest.get("source_start_frame", 0))
        video_single = read_video(
            video_path,
            args.num_frames,
            args.height,
            args.width,
            source_start_frame,
        )
        video = video_single.unsqueeze(0)
        track_data = np.load(args.track_dir / sample["tracks"])
        gt_tracks = torch.from_numpy(track_data["tracks"]).unsqueeze(0).to(args.device)
        gt_visibility = torch.from_numpy(track_data["visibility"]).unsqueeze(0).to(args.device)
        query_coords = torch.from_numpy(track_data["queries"][:, 1:]).unsqueeze(0).to(args.device)
        regions = list(sample.get("regions") or [])
        if not regions:
            raise ValueError(f"{sample_id}: region metadata missing from tracks manifest")
        cache = resize_region_cache(
            load_region_cache(Path(manifest["region_cache_root"]), sample_id),
            args.height,
            args.width,
        )
        query_visual_files = save_region_query_visualizations(save_dir, cache)
        save_input_video(video_single, save_dir / "gt_shifted.mp4")

        params = {
            "trajectory": args.matching_accuracy,
            "attn_weight": args.conf_attn_score,
            "query_key": False,
            "feature": False,
            "video_mode": "bg",
            "matching_layer": [],
            "query_coords": query_coords,
        }
        qk_evaluator = (
            RegionMatchingEvaluator(
                args.num_inference_steps,
                layer_count,
                gt_tracks,
                gt_visibility,
                regions,
                args.visualize_layer,
                args.visualize_step,
            )
            if args.matching_accuracy
            else None
        )
        feat_evaluator = (
            RegionMatchingEvaluator(
                args.num_inference_steps,
                layer_count,
                gt_tracks,
                gt_visibility,
                regions,
                args.visualize_layer,
                args.visualize_step,
            )
            if args.matching_accuracy
            else None
        )
        score = (
            ConfidenceAttentionScore(
                num_inference_steps=args.num_inference_steps,
                num_layers=layer_count,
                visibility=gt_visibility,
                model=args.model,
            )
            if args.conf_attn_score
            else None
        )

        completed_steps = [] if args.overwrite else restore_checkpoint(
            save_dir, qk_evaluator, feat_evaluator, score
        )
        if completed_steps:
            print(f"Resume {sample_id}: completed inverse steps {completed_steps}")

        for inverse_step in steps:
            if inverse_step in completed_steps:
                continue
            generator = seed_everything(args.seed, args.device)
            with torch.inference_mode():
                pipe(
                    prompt="",
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=False,
                    generator=generator,
                    conf_attn_score=score,
                    qk_acc_evaluator=qk_evaluator,
                    feat_acc_evaluator=feat_evaluator,
                    vis_timesteps=[int(args.visualize_step)],
                    vis_layers=[int(args.visualize_layer)],
                    output_type="latent",
                    params=params,
                    video=video,
                    inverse_step=inverse_step,
                )
            completed_steps.append(inverse_step)
            completed_steps.sort()
            save_checkpoint(save_dir, completed_steps, qk_evaluator, feat_evaluator, score)
            print(f"{sample_id}: inverse step {inverse_step}/{args.num_inference_steps - 1}")

        rows = []
        if qk_evaluator is not None and feat_evaluator is not None:
            write_matrix(save_dir / "qk_pck8.txt", qk_evaluator.pck, steps)
            write_matrix(save_dir / "feature_pck8.txt", feat_evaluator.pck, steps)
            summary_rows.extend(best_pck_rows(sample_id, "qk", qk_evaluator.pck, steps))
            summary_rows.extend(best_pck_rows(sample_id, "feature", feat_evaluator.pck, steps))
            rows = qk_evaluator.metric_rows("qk", steps)
            rows.extend(feat_evaluator.metric_rows("feature", steps))
            atomic_write_text(
                save_dir / "metrics.json",
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            )
            if qk_evaluator.visualized_tracks is None or feat_evaluator.visualized_tracks is None:
                raise RuntimeError(
                    f"{sample_id}: missing L{args.visualize_layer}/S{args.visualize_step} tracks"
                )
            np.savez_compressed(
                save_dir / "predicted_tracks.npz",
                qk=qk_evaluator.visualized_tracks,
                feature=feat_evaluator.visualized_tracks,
                cotracker=gt_tracks[0].detach().float().cpu().numpy(),
                visibility=gt_visibility[0].detach().bool().cpu().numpy(),
            )
            target_np = gt_tracks[0].detach().float().cpu().numpy()
            visibility_np = gt_visibility[0].detach().bool().cpu().numpy()
            for region in regions:
                region_dir = save_dir / "regions" / region["region_name"]
                region_dir.mkdir(parents=True, exist_ok=True)
                draw_region_track_video(
                    video_single,
                    qk_evaluator.visualized_tracks,
                    target_np,
                    visibility_np,
                    region,
                    "Q/K",
                    region_dir / f"tracks_qk_L{args.visualize_layer:02d}_S{args.visualize_step:03d}.mp4",
                )
                draw_region_track_video(
                    video_single,
                    feat_evaluator.visualized_tracks,
                    target_np,
                    visibility_np,
                    region,
                    "feature",
                    region_dir / f"tracks_feature_L{args.visualize_layer:02d}_S{args.visualize_step:03d}.mp4",
                )
        if score is not None:
            score.attention_max_df.to_csv(save_dir / "confidence_score.csv")
            score.attention_sum_df.to_csv(save_dir / "attention_score.csv")

        case_manifest = json.loads(Path(sample["case_manifest"]).read_text(encoding="utf-8"))
        run_manifest = {
            "case_key": sample_id,
            "case_manifest": sample["case_manifest"],
            "model": args.model,
            "model_path": str(args.model_path),
            "prompt": case_manifest["base"]["caption"],
            "context_video": str(video_path),
            "analysis_protocol": "CogVideoX_DiffTrack_GT_inversion_source_frame4_to_future",
            "source_start_frame": source_start_frame,
            "query_pixel_frame": 0,
            "query_source_pixel_frame": source_start_frame,
            "query_mode": "sam2_regions",
            "query_regions": regions,
            "layers": list(range(layer_count)),
            "step_indices": steps,
            "sampling_steps": int(args.num_inference_steps),
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "seed": int(args.seed),
            "files": ["gt_shifted.mp4", *query_visual_files],
        }
        atomic_write_text(
            save_dir / "manifest.json",
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write_text(
            complete_marker,
            json.dumps(
                {
                    "sample_id": sample_id,
                    "video": str(video_path),
                    "inverse_steps": steps,
                    "model": args.model,
                    "seed": args.seed,
                    "metric_row_count": len(rows) if qk_evaluator is not None else 0,
                },
                indent=2,
            )
            + "\n",
        )

    if summary_rows:
        summary_path = args.output_dir / f"summary_{args.start}_{args.end or 'end'}.csv"
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
