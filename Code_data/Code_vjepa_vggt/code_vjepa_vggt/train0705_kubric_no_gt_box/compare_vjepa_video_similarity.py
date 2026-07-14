#!/usr/bin/env python3
"""Encode videos with the training V-JEPA2 adapter and compare their features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from code_vjepa_vggt.adapters.jepa_adapter import JEPAPatchAdapter
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_uniform


DEFAULT_X0_REMAINING_35 = (
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "text_noun_attention_x0_every5_step1000_physiq025_20260714/"
    "train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_"
    "init3500_save500_keepall_20260713T090024Z_step-001000_steps40_512x896_ctx08_"
    "49f_defaultnegprompt/physicIQ_025_Solid_Mechanics_0002_perspective-center_"
    "trimmed_text_noun_attention/predicted_x0/pred_x0_remaining_35_h264.mp4"
)
DEFAULT_X0_REMAINING_01 = DEFAULT_X0_REMAINING_35.replace(
    "remaining_35", "remaining_01"
)
DEFAULT_GT = (
    "/data/gaoya/AAA_test_video/0623/testdataset/"
    "025_Solid_Mechanics_0002_perspective-center_trimmed/"
    "physicIQ_0002_clip_2p5s_3p5s.mp4"
)
DEFAULT_CKPT = (
    "/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth"
)
DEFAULT_OUTPUT_DIR = (
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "vjepa_similarity_30f_patch_overlay_physiq025_x0_remaining35_vs01_vs_gt_20260714"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x0-remaining-35", default=DEFAULT_X0_REMAINING_35)
    parser.add_argument("--x0-remaining-01", default=DEFAULT_X0_REMAINING_01)
    parser.add_argument("--ground-truth", default=DEFAULT_GT)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--x0-title-height",
        type=int,
        default=60,
        help="Pixels to remove from the top of the two labeled x0 videos.",
    )
    return parser.parse_args()


def load_video(
    path: Path,
    num_frames: int,
    crop_top: int,
    crop_size: int,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    frames, indices = read_video_uniform(path, num_frames=num_frames)
    source_hw = [int(frames.shape[1]), int(frames.shape[2])]
    if crop_top:
        if not 0 <= crop_top < frames.shape[1]:
            raise ValueError(f"invalid crop_top={crop_top} for {path} with height={frames.shape[1]}")
        frames = frames[:, crop_top:, :, :]
    video = preprocess_video_rgb_uint8(
        frames,
        out_hw=(crop_size, crop_size),
        value_range="minus_one_to_one",
        resize_mode="stretch",
    )
    metadata = {
        "path": str(path),
        "source_hw": source_hw,
        "crop_top": int(crop_top),
        "content_hw": [int(frames.shape[1]), int(frames.shape[2])],
        "sampled_frame_indices": indices.tolist(),
    }
    actual_input = (
        video.permute(1, 2, 3, 0).add(1.0).mul(127.5).clamp(0, 255).byte().numpy()
    )
    return video, actual_input, metadata


def encode(
    adapter: JEPAPatchAdapter,
    video_cthw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    dtype = next(adapter.parameters()).dtype
    with torch.inference_mode():
        output = adapter(video_cthw.unsqueeze(0).to(device=device, dtype=dtype))
    return output.patch_tokens[0].float().cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def temporal_align(features: torch.Tensor, target_t: int) -> torch.Tensor:
    if features.shape[0] == target_t:
        return features
    return F.interpolate(
        features.transpose(0, 1).unsqueeze(0),
        size=target_t,
        mode="linear",
        align_corners=False,
    )[0].transpose(0, 1)


def compare_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    global_a = a.mean(dim=(0, 1, 2))
    global_b = b.mean(dim=(0, 1, 2))

    temporal_a = a.mean(dim=(1, 2))
    temporal_b = b.mean(dim=(1, 2))
    target_t = max(temporal_a.shape[0], temporal_b.shape[0])
    temporal_a = temporal_align(temporal_a, target_t)
    temporal_b = temporal_align(temporal_b, target_t)
    temporal_cosines = F.cosine_similarity(temporal_a, temporal_b, dim=-1)

    if a.shape != b.shape:
        raise ValueError(f"token grids differ after common preprocessing: {a.shape} vs {b.shape}")
    patch_a = a.flatten(1, 2)
    patch_b = b.flatten(1, 2)
    patch_cosines = F.cosine_similarity(patch_a, patch_b, dim=-1).view(
        a.shape[0], a.shape[1], a.shape[2]
    )

    return {
        "global_pooled_cosine": cosine(global_a, global_b),
        "temporal_pooled_cosine_mean": float(temporal_cosines.mean().item()),
        "temporal_pooled_cosine_min": float(temporal_cosines.min().item()),
        "same_position_patch_cosine_mean": float(patch_cosines.mean().item()),
        "same_position_patch_cosine_min": float(patch_cosines.min().item()),
    }, patch_cosines


def add_header(frame: np.ndarray, text: str, height: int = 44) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        frame, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas,
        text,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_h264(path: Path, frames_rgb: list[np.ndarray], fps: float = 15.0) -> None:
    if not frames_rgb:
        raise ValueError(f"no frames provided for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp_dir:
        intermediate = Path(tmp_dir) / "intermediate.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {intermediate}")
        for frame in frames_rgb:
            if frame.shape[:2] != (height, width):
                raise ValueError("visualization frame sizes are inconsistent")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        ffmpeg = shutil.which("ffmpeg") or "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(intermediate),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(path),
            ],
            check=True,
        )


def save_actual_input_triptych(
    output_dir: Path,
    visual_inputs: dict[str, np.ndarray],
) -> Path:
    names = list(visual_inputs)
    frames = []
    for frame_index in range(next(iter(visual_inputs.values())).shape[0]):
        panels = [
            add_header(visual_inputs[name][frame_index], f"V-JEPA input | {name} | frame {frame_index:02d}")
            for name in names
        ]
        frames.append(np.concatenate(panels, axis=1))
    path = output_dir / "actual_vjepa_inputs_triptych_h264.mp4"
    write_h264(path, frames)
    return path


def visualize_pair_similarity(
    output_dir: Path,
    name_a: str,
    name_b: str,
    input_a: np.ndarray,
    input_b: np.ndarray,
    patch_cosines: torch.Tensor,
    tubelet_size: int,
) -> tuple[Path, dict]:
    cosine_np = patch_cosines.numpy()
    flat_index = int(np.argmin(cosine_np))
    token_t, token_y, token_x = np.unravel_index(flat_index, cosine_np.shape)
    patch_h = input_a.shape[1] // cosine_np.shape[1]
    patch_w = input_a.shape[2] // cosine_np.shape[2]
    x0, x1 = token_x * patch_w, (token_x + 1) * patch_w
    y0, y1 = token_y * patch_h, (token_y + 1) * patch_h

    frames = []
    for frame_index in range(input_a.shape[0]):
        local_t = min(frame_index // tubelet_size, cosine_np.shape[0] - 1)
        similarity = cosine_np[local_t]
        heat_u8 = np.clip(similarity * 255.0, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(
            cv2.resize(
                heat_u8,
                (input_a.shape[2], input_a.shape[1]),
                interpolation=cv2.INTER_NEAREST,
            ),
            cv2.COLORMAP_TURBO,
        )
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay_a = cv2.addWeighted(input_a[frame_index], 0.55, heat, 0.45, 0)
        overlay_b = cv2.addWeighted(input_b[frame_index], 0.55, heat, 0.45, 0)
        if local_t == token_t:
            for panel in (overlay_a, heat, overlay_b):
                cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 3)
        panels = [
            add_header(overlay_a, f"{name_a} | overlay | f{frame_index:02d}"),
            add_header(heat, f"cos [0,1] | t{local_t:02d} | min {similarity.min():.4f}"),
            add_header(overlay_b, f"{name_b} | overlay | f{frame_index:02d}"),
        ]
        frames.append(np.concatenate(panels, axis=1))

    path = output_dir / f"patch_similarity_overlay__{name_a}__vs__{name_b}_h264.mp4"
    write_h264(path, frames)
    minimum = {
        "cosine": float(cosine_np[token_t, token_y, token_x]),
        "token_t_y_x": [int(token_t), int(token_y), int(token_x)],
        "vjepa_input_pixel_box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "sampled_video_frame_indices": list(
            range(token_t * tubelet_size, min((token_t + 1) * tubelet_size, input_a.shape[0]))
        ),
    }
    return path, minimum


def main() -> None:
    args = parse_args()
    if args.num_frames <= 0 or args.num_frames % 2:
        raise ValueError("--num-frames must be a positive multiple of the V-JEPA tubelet size 2")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "x0_remaining_35": (Path(args.x0_remaining_35), args.x0_title_height),
        "x0_remaining_01": (Path(args.x0_remaining_01), args.x0_title_height),
        "ground_truth": (Path(args.ground_truth), 0),
    }
    for path, _ in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    adapter = JEPAPatchAdapter(
        ckpt_path=args.checkpoint,
        device=str(device),
        crop_size=args.crop_size,
        num_frames=args.num_frames,
        patch_size=16,
        tubelet_size=2,
        trainable=False,
    )

    features: dict[str, torch.Tensor] = {}
    visual_inputs: dict[str, np.ndarray] = {}
    input_metadata: dict[str, dict] = {}
    for name, (path, crop_top) in inputs.items():
        video, actual_input, metadata = load_video(
            path, args.num_frames, crop_top, args.crop_size
        )
        features[name] = encode(adapter, video, device)
        visual_inputs[name] = actual_input
        input_metadata[name] = metadata
        print(f"[encoded] {name}: video={tuple(video.shape)} tokens={tuple(features[name].shape)}")

    input_triptych = save_actual_input_triptych(output_dir, visual_inputs)
    rows = []
    minimum_locations = {}
    overlay_paths = {}
    for name_a, name_b in combinations(features, 2):
        metrics, patch_cosines = compare_pair(features[name_a], features[name_b])
        row = {"video_a": name_a, "video_b": name_b, **metrics}
        rows.append(row)
        pair_key = f"{name_a}__vs__{name_b}"
        overlay_path, minimum = visualize_pair_similarity(
            output_dir,
            name_a,
            name_b,
            visual_inputs[name_a],
            visual_inputs[name_b],
            patch_cosines,
            tubelet_size=2,
        )
        overlay_paths[pair_key] = str(overlay_path)
        minimum["source_frame_indices_a"] = [
            input_metadata[name_a]["sampled_frame_indices"][index]
            for index in minimum["sampled_video_frame_indices"]
        ]
        minimum["source_frame_indices_b"] = [
            input_metadata[name_b]["sampled_frame_indices"][index]
            for index in minimum["sampled_video_frame_indices"]
        ]
        minimum_locations[pair_key] = minimum
        print(f"[similarity] {name_a} vs {name_b}: {metrics}")

    csv_path = output_dir / "pairwise_similarity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        output_dir / "vjepa_features_fp16.npz",
        **{name: value.numpy().astype(np.float16) for name, value in features.items()},
    )
    report = {
        "model": {
            "checkpoint": args.checkpoint,
            "adapter": "JEPAPatchAdapter",
            "num_frames": args.num_frames,
            "crop_size": args.crop_size,
            "patch_size": 16,
            "tubelet_size": 2,
            "value_range": "[-1,1]",
            "spatial_preprocess": "stretch to square, matching the training adapter",
            "seed": args.seed,
        },
        "inputs": input_metadata,
        "feature_shapes": {name: list(value.shape) for name, value in features.items()},
        "pairwise_similarity": rows,
        "minimum_patch_locations": minimum_locations,
        "visualizations": {
            "actual_vjepa_inputs_triptych": str(input_triptych),
            "pairwise_patch_similarity_overlays": overlay_paths,
            "color_scale": "fixed cosine [0,1], OpenCV TURBO; blue=low, red=high",
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
