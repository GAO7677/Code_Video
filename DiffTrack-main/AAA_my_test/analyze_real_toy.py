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
from utils.evaluation import MatchingEvaluator


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


def read_video(path: Path, num_frames: int, height: int, width: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != num_frames:
        raise ValueError(f"{path} contains {len(frames)} frames, expected at least {num_frames}")
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(256, 256), mode="bilinear", align_corners=False)
    return F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)


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
    qk_evaluator: MatchingEvaluator | None,
    feat_evaluator: MatchingEvaluator | None,
    score: ConfidenceAttentionScore | None,
) -> None:
    state_path = save_dir / "step_state.npz"
    temporary_state = state_path.with_suffix(".npz.tmp")
    arrays = {"completed_steps": np.asarray(completed_steps, dtype=np.int64)}
    if qk_evaluator is not None and feat_evaluator is not None:
        arrays["qk_pck"] = qk_evaluator.pck.cpu().numpy()
        arrays["feature_pck"] = feat_evaluator.pck.cpu().numpy()
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
    qk_evaluator: MatchingEvaluator | None,
    feat_evaluator: MatchingEvaluator | None,
    score: ConfidenceAttentionScore | None,
) -> list[int]:
    state_path = save_dir / "step_state.npz"
    if not state_path.exists():
        return []
    state = np.load(state_path)
    if qk_evaluator is not None and feat_evaluator is not None:
        qk_evaluator.pck.copy_(torch.from_numpy(state["qk_pck"]))
        feat_evaluator.pck.copy_(torch.from_numpy(state["feature_pck"]))
    if score is not None:
        for filename, attribute in (
            ("confidence_score.csv", "attention_max_df"),
            ("attention_score.csv", "attention_sum_df"),
        ):
            path = save_dir / filename
            if path.exists():
                setattr(score, attribute, pd.read_csv(path, index_col=[0, 1]))
    return [int(step) for step in state["completed_steps"]]


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

    steps = args.inverse_steps or list(range(args.num_inference_steps))
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
        video = read_video(video_path, args.num_frames, args.height, args.width).unsqueeze(0)
        track_data = np.load(args.track_dir / sample["tracks"])
        gt_tracks = torch.from_numpy(track_data["tracks"]).unsqueeze(0).to(args.device)
        gt_visibility = torch.from_numpy(track_data["visibility"]).unsqueeze(0).to(args.device)
        query_coords = torch.from_numpy(track_data["queries"][:, 1:]).unsqueeze(0).to(args.device)

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
            MatchingEvaluator(args.num_inference_steps, layer_count, gt_tracks, gt_visibility)
            if args.matching_accuracy
            else None
        )
        feat_evaluator = (
            MatchingEvaluator(args.num_inference_steps, layer_count, gt_tracks, gt_visibility)
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
                    vis_timesteps=[steps[-1]],
                    vis_layers=[layer_count // 2],
                    output_type="latent",
                    params=params,
                    video=video,
                    inverse_step=inverse_step,
                )
            completed_steps.append(inverse_step)
            completed_steps.sort()
            save_checkpoint(save_dir, completed_steps, qk_evaluator, feat_evaluator, score)
            print(f"{sample_id}: inverse step {inverse_step}/{args.num_inference_steps - 1}")

        if qk_evaluator is not None and feat_evaluator is not None:
            write_matrix(save_dir / "qk_pck8.txt", qk_evaluator.pck, steps)
            write_matrix(save_dir / "feature_pck8.txt", feat_evaluator.pck, steps)
            summary_rows.extend(best_pck_rows(sample_id, "qk", qk_evaluator.pck, steps))
            summary_rows.extend(best_pck_rows(sample_id, "feature", feat_evaluator.pck, steps))
        if score is not None:
            score.attention_max_df.to_csv(save_dir / "confidence_score.csv")
            score.attention_sum_df.to_csv(save_dir / "attention_score.csv")

        atomic_write_text(
            complete_marker,
            json.dumps(
                {
                    "sample_id": sample_id,
                    "video": str(video_path),
                    "inverse_steps": steps,
                    "model": args.model,
                    "seed": args.seed,
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
