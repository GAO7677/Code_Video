#!/usr/bin/env python3
"""Score matched motion-region inputs with regional WMReward or ROI VLM metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_ROOT = Path("/data/gaoya/agent-data/outputs/sam2_region_motion_roi_scores")
PHYSV_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
MOTION_ANALYSIS_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "train0705_kubric_no_gt_box"
)
MODEL_ORDER = ("stage1b", "lora", "gt")

for path in (PHYSV_ROOT, MOTION_ANALYSIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric", required=True, choices=["wmreward_region", "videophy2_roi", "cosmos_roi"]
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def load_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = set(args.models)
    jobs = []
    case_dirs = sorted((args.root / "cases").glob("case_*"))
    if args.case_limit is not None:
        case_dirs = case_dirs[: args.case_limit]
    for case_dir in case_dirs:
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        for model_name in MODEL_ORDER:
            if model_name not in selected:
                continue
            jobs.append(
                {
                    "case_dir": case_dir,
                    "case_key": metadata["case_key"],
                    "prompt": metadata["prompt"],
                    "model": model_name,
                    "metadata": metadata,
                }
            )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    return [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_index]


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def read_rgb_frames(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    return np.stack(frames)


def token_region_mask(frame_masks: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    time_tokens, grid_h, grid_w = target_shape
    output = np.zeros(target_shape, dtype=bool)
    for token_t in range(time_tokens):
        frame_ids = [index for index in (2 * token_t, 2 * token_t + 1) if index < len(frame_masks)]
        if not frame_ids:
            continue
        frame_union = np.maximum.reduce(frame_masks[frame_ids]).astype(np.float32)
        coverage = cv2.resize(frame_union, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        output[token_t] = coverage >= 0.10
    return output


class RegionalWMRewardRunner:
    def __init__(self) -> None:
        import torch

        from visualize_wmreward_patch_surprise import (
            WMREWARD_ROOT,
            compute_patch_surprise,
            install_optional_diffusers_stub,
            install_upstream_paths,
            prepare_official_input,
        )

        self.torch = torch
        self.compute_patch_surprise = compute_patch_surprise
        self.prepare_official_input = prepare_official_input
        install_upstream_paths()
        install_optional_diffusers_stub()
        from utils import load_vjepa_model_source

        original_torch_load = torch.load

        def trusted_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = trusted_load
        cwd = Path.cwd()
        os.chdir(WMREWARD_ROOT)
        try:
            encoder, target_encoder, predictor, img_size = load_vjepa_model_source("vitg384")
        finally:
            os.chdir(cwd)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device).eval()
        self.target_encoder = target_encoder.to(self.device).eval()
        self.predictor = predictor.to(self.device).eval()
        self.img_size = int(img_size)

    def score(self, job: dict[str, Any]) -> dict[str, Any]:
        video_path = job["case_dir"] / job["model"] / "wm_input_full25.mp4"
        frames = read_rgb_frames(video_path)
        video_tensor, _ = self.prepare_official_input(frames, self.img_size)
        surprise, info = self.compute_patch_surprise(
            video_tensor,
            self.encoder,
            self.target_encoder,
            self.predictor,
            img_size=self.img_size,
            window_size=16,
            context_frames=8,
            stride=8,
            seed=42,
            device=self.device,
        )
        frame_masks = np.load(job["case_dir"] / "motion_masks.npz")["shared_motion_frames"]
        region = token_region_mask(frame_masks, surprise.shape)
        valid = np.isfinite(surprise)
        motion = valid & region
        static = valid & ~region
        if not motion.any():
            raise RuntimeError("Shared motion region has no valid scored V-JEPA tokens")
        motion_score = float(surprise[motion].mean())
        static_score = float(surprise[static].mean()) if static.any() else None
        global_score = float(surprise[valid].mean())
        return {
            "metric_name": "WMReward-style regional token surprise",
            "input_video": str(video_path),
            "motion_region_surprise": motion_score,
            "static_region_surprise": static_score,
            "motion_minus_static": motion_score - static_score if static_score is not None else None,
            "global_patch_surprise": global_score,
            "official_window_surprise": info["official_surprise_mean"],
            "motion_token_count": int(motion.sum()),
            "static_token_count": int(static.sum()),
            "motion_token_ratio": float(motion.sum() / valid.sum()),
            "patch_grid_t_h_w": list(surprise.shape),
            "mask_patch_overlap_threshold": 0.10,
        }


def create_runner(metric: str) -> Any:
    if metric == "wmreward_region":
        return RegionalWMRewardRunner()
    if metric == "videophy2_roi":
        from physv_eval.videophy2_auto import VideoPhy2Runner

        return VideoPhy2Runner(device="cuda")
    from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner

    return OfficialCosmosReason1Runner()


def score_job(metric: str, runner: Any, job: dict[str, Any]) -> dict[str, Any]:
    if metric == "wmreward_region":
        return runner.score(job)
    video = job["case_dir"] / job["model"] / "motion_roi_input.mp4"
    if metric == "videophy2_roi":
        return {
            "input_video": str(video),
            "sa": runner.score_video(video, task="sa", caption=job["prompt"]),
            "pc": runner.score_video(video, task="pc"),
        }
    result = runner.score(video)
    result["input_video"] = str(video)
    return result


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    jobs = load_jobs(args)
    pending = []
    for job in jobs:
        output = job["case_dir"] / job["model"] / "scores" / f"{args.metric}.json"
        if args.force or not complete(output):
            pending.append((job, output))
    print(
        f"metric={args.metric} shard={args.shard_index}/{args.num_shards} "
        f"selected={len(jobs)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return
    random.seed(42)
    np.random.seed(42)
    runner = create_runner(args.metric)
    failures = 0
    for index, (job, output) in enumerate(pending, start=1):
        started = time.time()
        record = {
            "status": "ok",
            "metric": args.metric,
            "model": job["model"],
            "case_key": job["case_key"],
            "prompt": job["prompt"],
        }
        try:
            record["result"] = score_job(args.metric, runner, job)
        except Exception as exc:  # Keep successful cases resumable after one failure.
            failures += 1
            record.update(
                status="error", error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        record["elapsed_seconds"] = time.time() - started
        atomic_json(output, record)
        print(
            f"[{index}/{len(pending)}] {record['status']} "
            f"{job['model']}/{job['case_key']} {record['elapsed_seconds']:.2f}s",
            flush=True,
        )
    if failures:
        raise SystemExit(f"{failures} case(s) failed")


if __name__ == "__main__":
    main()
