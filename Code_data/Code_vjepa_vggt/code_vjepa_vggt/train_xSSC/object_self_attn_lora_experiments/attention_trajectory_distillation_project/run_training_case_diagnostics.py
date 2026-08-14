#!/usr/bin/env python3
"""Run auditable SAM2, x0, and Frozen Motion Probe training-case diagnostics."""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFTRACK_ROOT = Path("/home/gaoya/Code_Video/DiffTrack-main")
GROUNDED_SAM2_ROOT = Path("/home/gaoya/Grounded-SAM-2-main")
for _path in (EXPERIMENT_ROOT, CODE_ROOT, DIFFTRACK_ROOT, GROUNDED_SAM2_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

from code_vjepa_vggt import context_wan_v_newtrain as context_wan
from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import (
    PyBullet0713NoGTBoxDataset,
)
from code_vjepa_vggt.train0419_reference.batch_eval_lora import build_pipeline
from code_vjepa_vggt.train_xSSC.visualize_movi_c_sam2_amg import (
    draw_selected_boxes,
    overlay_masks,
    resolve_sam2_config_name,
    select_xssc_candidates,
)
from AAA_my_test import analyze_wan_gt_toy_worker as wan_tools
from AAA_my_test.precompute_toydataset_sam2_regions import (
    build_provider,
    detect_and_track_objects,
)

import train_xssc_object_self_attn_lora as train_core
from attention_trajectory_distillation_project.frozen_motion_probe import (
    blend_with_fixed_probe_noise,
    heatmap_soft_argmax_trajectory,
    query_rows_from_mask,
    student_teacher_heatmap_kl,
    trajectory_huber_loss,
)
from attention_trajectory_distillation_project.train_xssc_object_self_attn_lora_frozen_motion_probe import (
    FrozenMotionProbeWanModule,
    _probe_grid,
)


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/frozen_motion_probe_training_diagnostics"
)
DEFAULT_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/frozen_motion_probe_training_diagnostics"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_SAM2_CONFIG = Path(
    "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml"
)
DEFAULT_SAM2_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt"
)
DEFAULT_HEAD_CONFIG = EXPERIMENT_ROOT / "configs/physiciq67_pck32_s039_latest3350_top100_heads.json"
DEFAULT_HEAD_SUBSET = "T_physiciq67_pck32_s039_latest3350_top100"
DEFAULT_HEAD_SUBTYPE = "physiciq67_pck32_s039_latest3350"
DEFAULT_FAMILIES = ("F1", "F2", "F3")
DEFAULT_SWEEP_TRAINING_TIMESTEPS = (100.0, 300.0, 500.0, 700.0, 900.0)
DEFAULT_SWEEP_PROBE_NOISE_LEVELS = (0.1, 0.2)
DEFAULT_SWEEP_PROBE_TIMESTEPS = (100.0, 200.0)
PALETTE_RGB = np.asarray(
    [
        [230, 57, 70],
        [28, 126, 214],
        [25, 164, 99],
        [235, 153, 33],
        [144, 83, 198],
        [13, 154, 166],
        [220, 83, 32],
        [194, 63, 128],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SAM2 masks, run baseline x0/Frozen Motion Probe forwards, "
            "and render a diagnostic report."
        )
    )
    parser.add_argument(
        "mode",
        choices=("prepare", "forward", "render", "sweep-forward", "sweep-render"),
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--sam2-config", type=Path, default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--head-config", type=Path, default=DEFAULT_HEAD_CONFIG)
    parser.add_argument("--head-subset", default=DEFAULT_HEAD_SUBSET)
    parser.add_argument("--head-subtype", default=DEFAULT_HEAD_SUBTYPE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--query-pixel-frame", type=int, default=4)
    parser.add_argument("--query-latent-frame", type=int, default=1)
    parser.add_argument("--training-timestep", type=float, default=500.0)
    parser.add_argument("--probe-timestep", type=float, default=500.0)
    parser.add_argument("--probe-noise-level", type=float, default=0.5)
    parser.add_argument(
        "--sweep-training-timesteps",
        type=float,
        nargs="+",
        default=list(DEFAULT_SWEEP_TRAINING_TIMESTEPS),
    )
    parser.add_argument(
        "--sweep-probe-noise-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_SWEEP_PROBE_NOISE_LEVELS),
    )
    parser.add_argument(
        "--sweep-probe-timesteps",
        type=float,
        nargs="+",
        default=list(DEFAULT_SWEEP_PROBE_TIMESTEPS),
    )
    parser.add_argument("--heatmap-weight", type=float, default=0.1)
    parser.add_argument("--trajectory-weight", type=float, default=0.1)
    parser.add_argument("--trajectory-huber-delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--heatmap-fps", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(value).all().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "abs_mean": float(value.abs().mean().item()),
    }


def check_common_args(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda:4"):
        raise ValueError("GPU 4 is prohibited by workspace rules")
    if int(args.num_frames) != 49 or int(args.context_frames) != 8:
        raise ValueError("This controlled diagnostic requires 49 frames and 8 context frames")
    if int(args.query_pixel_frame) != 4 or int(args.query_latent_frame) != 1:
        raise ValueError("This controlled diagnostic requires F04 / latent-1 query")
    if int(args.height) % 32 or int(args.width) % 32:
        raise ValueError("height and width must be divisible by 32")
    if not 0.0 <= float(args.probe_noise_level) <= 1.0:
        raise ValueError("probe-noise-level must be in [0,1]")
    noise_sweep_config(args)


def sweep_value_code(prefix: str, value: float) -> str:
    scaled = round(float(value) * (100 if prefix == "probe" else 1))
    width = 3 if prefix == "probe" else 4
    return f"{prefix}_{scaled:0{width}d}"


def noise_sweep_config(args: argparse.Namespace) -> dict[str, Any]:
    training_timesteps = tuple(float(value) for value in args.sweep_training_timesteps)
    probe_levels = tuple(float(value) for value in args.sweep_probe_noise_levels)
    probe_timesteps = tuple(float(value) for value in args.sweep_probe_timesteps)
    if not training_timesteps:
        raise ValueError("sweep-training-timesteps cannot be empty")
    if len(set(training_timesteps)) != len(training_timesteps):
        raise ValueError("sweep-training-timesteps must be unique")
    if any(value < 0.0 or value > 1000.0 for value in training_timesteps):
        raise ValueError("sweep-training-timesteps must be in [0,1000]")
    if len(probe_levels) != len(probe_timesteps):
        raise ValueError(
            "sweep-probe-noise-levels and sweep-probe-timesteps must have equal lengths"
        )
    if not probe_levels or any(value < 0.0 or value > 1.0 for value in probe_levels):
        raise ValueError("sweep-probe-noise-levels must be non-empty and in [0,1]")
    if len(set(probe_levels)) != len(probe_levels):
        raise ValueError("sweep-probe-noise-levels must be unique")
    if any(value < 0.0 or value > 1000.0 for value in probe_timesteps):
        raise ValueError("sweep-probe-timesteps must be in [0,1000]")
    probes = tuple(
        {
            "noise_level": level,
            "timestep": timestep,
            "id": sweep_value_code("probe", level),
        }
        for level, timestep in zip(probe_levels, probe_timesteps)
    )
    stages = tuple(
        {"timestep": timestep, "id": sweep_value_code("train", timestep)}
        for timestep in training_timesteps
    )
    return {"training_stages": stages, "probe_settings": probes}


def sample_video_to_uint8(video: torch.Tensor) -> np.ndarray:
    if video.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] video, got {tuple(video.shape)}")
    return (
        video.detach()
        .float()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
        .numpy()
    )


def decoded_video_to_uint8(video: torch.Tensor) -> np.ndarray:
    if video.ndim == 5:
        video = video[0]
    return sample_video_to_uint8(video)


def write_video(path: Path, frames: Iterable[np.ndarray], fps: float) -> None:
    frames = list(frames)
    if not frames:
        raise ValueError(f"cannot write empty video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=float(fps),
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in frames:
            frame = np.asarray(frame, dtype=np.uint8)[:height, :width]
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def add_label(frame: np.ndarray, label: str, *, bar_height: int = 28) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], bar_height), (19, 24, 22), -1)
    cv2.putText(
        output,
        label,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 248, 246),
        1,
        cv2.LINE_AA,
    )
    return output


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def amg_filter_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_selected=11,
        min_area_ratio=0.004,
        max_area_ratio=0.35,
        min_bbox_side=7.0,
        background_area_ratio=0.06,
        background_span_ratio=0.75,
        border_area_ratio=0.025,
        border_occupancy_ratio=0.18,
        opposite_edge_area_ratio=0.04,
        shadow_min_area_ratio=0.03,
        shadow_max_luminance_ratio=0.55,
        shadow_max_chromaticity_distance=0.10,
        shadow_max_gradient_mean=20.0,
        duplicate_iou=0.70,
        duplicate_containment=0.85,
    )


def select_training_cases(
    dataset: PyBullet0713NoGTBoxDataset,
    families: Iterable[str],
) -> list[tuple[int, Any]]:
    wanted = [str(value) for value in families]
    selected: list[tuple[int, Any]] = []
    for family in wanted:
        match = next(
            (
                (index, record)
                for index, record in enumerate(dataset.samples)
                if record.family_key == family
            ),
            None,
        )
        if match is None:
            raise RuntimeError(f"no train-split sample found for family {family}")
        selected.append(match)
    return selected


def expected_grid(args: argparse.Namespace) -> tuple[int, int, int]:
    return (13, int(args.height) // 32, int(args.width) // 32)


def overlay_selected_mask(frame: np.ndarray, mask: np.ndarray, phrase: str) -> np.ndarray:
    output = frame.astype(np.float32).copy()
    inside = np.asarray(mask, dtype=bool)
    color = np.asarray([230, 57, 70], dtype=np.float32)
    output[inside] = 0.46 * output[inside] + 0.54 * color
    contours, _ = cv2.findContours(
        inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bgr = cv2.cvtColor(output.round().clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    return add_label(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), phrase)


def draw_query_grid(
    frame: np.ndarray,
    mask: np.ndarray,
    grid: tuple[int, int, int],
    query_latent_frame: int,
) -> tuple[np.ndarray, list[int]]:
    rows = query_rows_from_mask(
        torch.from_numpy(np.asarray(mask, dtype=np.uint8)),
        grid=grid,
        query_latent_frame=int(query_latent_frame),
    )
    output = frame.copy()
    _, grid_h, grid_w = grid
    spatial_offset = int(query_latent_frame) * grid_h * grid_w
    for row in rows.tolist():
        spatial = int(row) - spatial_offset
        y, x = divmod(spatial, grid_w)
        x0 = int(round(x * output.shape[1] / grid_w))
        x1 = int(round((x + 1) * output.shape[1] / grid_w))
        y0 = int(round(y * output.shape[0] / grid_h))
        y1 = int(round((y + 1) * output.shape[0] / grid_h))
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (255, 235, 48), 2)
    return add_label(
        output,
        f"fixed query | latent-{query_latent_frame} | {len(rows)} cells",
    ), rows.tolist()


def candidate_overlay(
    frame: np.ndarray,
    annotation: dict[str, Any],
    index: int,
    iou: float,
    filtered: bool,
) -> np.ndarray:
    mask = np.asarray(annotation["segmentation"], dtype=bool)
    output = frame.astype(np.float32).copy()
    color = PALETTE_RGB[index % len(PALETTE_RGB)].astype(np.float32)
    output[mask] = 0.43 * output[mask] + 0.57 * color
    x, y, w, h = annotation["bbox"]
    bgr = cv2.cvtColor(output.round().clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.rectangle(
        bgr,
        (int(round(x)), int(round(y))),
        (int(round(x + w)), int(round(y + h))),
        tuple(int(v) for v in color[::-1]),
        2,
    )
    label = (
        f"AMG {index:03d} | IoU={iou:.3f} | "
        f"pIoU={float(annotation['predicted_iou']):.3f} | "
        f"stable={float(annotation['stability_score']):.3f} | "
        f"{'filtered-in' if filtered else 'raw-only'}"
    )
    return add_label(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), label)


def run_prepare(args: argparse.Namespace) -> None:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    output_root = args.output_root.resolve()
    cache_root = args.cache_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    dataset = PyBullet0713NoGTBoxDataset(
        root=args.dataset_root.resolve(),
        split="train",
        resolution=(int(args.height), int(args.width)),
        num_frames=int(args.num_frames),
        num_context_frames=int(args.context_frames),
        sampling_strategy="prefix",
    )
    selected = select_training_cases(dataset, args.families)
    cases: list[dict[str, Any]] = []

    provider = build_provider(str(args.device), points_per_region=8)
    grounded: dict[str, dict[str, Any]] = {}
    for position, (dataset_index, record) in enumerate(selected, start=1):
        sample = dataset[dataset_index]
        metadata = sample["metadata"]
        frames = sample_video_to_uint8(sample["video"])
        phrases = [
            str(value).strip()
            for value in metadata.get("object_phrases", [])
            if str(value).strip()
        ]
        dynamic_phrases = [
            str(value).strip()
            for value in metadata.get("dynamic_object_phrases", [])
            if str(value).strip()
        ]
        target_phrase = dynamic_phrases[0] if dynamic_phrases else phrases[0]
        frames_tchw_01 = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        print(
            f"[prepare grounding {position}/{len(selected)}] "
            f"{record.key}: {target_phrase}",
            flush=True,
        )
        tracked = detect_and_track_objects(provider, frames_tchw_01, [target_phrase])
        target_mask = np.asarray(
            tracked.object_tracks[0].masks_thw[int(args.query_pixel_frame)],
            dtype=np.uint8,
        )
        if not target_mask.any():
            raise RuntimeError(f"empty SAM2 F04 target mask for {record.key}")
        grounded[record.key] = {
            "dataset_index": dataset_index,
            "record": record,
            "sample": sample,
            "frames": frames,
            "target_phrase": target_phrase,
            "target_mask": target_mask,
            "grounding_debug": tracked.debug,
        }
    del provider
    gc.collect()
    torch.cuda.empty_cache()

    sam2 = build_sam2(
        resolve_sam2_config_name(args.sam2_config),
        str(args.sam2_checkpoint.resolve()),
        device=str(args.device),
        mode="eval",
    )
    generator = SAM2AutomaticMaskGenerator(sam2)
    filter_args = amg_filter_args()
    for position, (case_key, item) in enumerate(grounded.items(), start=1):
        case_dir = cache_root / "cases" / case_key
        complete = case_dir / "prepare_complete.json"
        if complete.is_file() and not args.overwrite:
            cases.append(json.loads((case_dir / "case.json").read_text(encoding="utf-8")))
            print(f"[prepare AMG {position}/{len(grounded)}] skip {case_key}", flush=True)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        frames = item["frames"]
        frame = frames[int(args.query_pixel_frame)]
        target_mask = item["target_mask"]
        print(f"[prepare AMG {position}/{len(grounded)}] {case_key}", flush=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            annotations = generator.generate(frame)
        annotations = sorted(
            annotations,
            key=lambda row: float(row["predicted_iou"] * row["stability_score"]),
            reverse=True,
        )
        filtered = select_xssc_candidates(
            annotations,
            frame.shape[0] * frame.shape[1],
            filter_args,
            image=frame,
        )
        filtered_ids = {id(row) for row in filtered}
        ious = [mask_iou(row["segmentation"], target_mask) for row in annotations]
        best_index = int(np.argmax(ious))
        source_dir = case_dir / "sam2_amg_candidates"
        source_dir.mkdir(parents=True, exist_ok=True)
        candidate_rows = []
        for index, (annotation, iou) in enumerate(zip(annotations, ious)):
            rendered = candidate_overlay(
                frame,
                annotation,
                index,
                iou,
                id(annotation) in filtered_ids,
            )
            filename = f"candidate_{index:03d}.jpg"
            cv2.imwrite(
                str(source_dir / filename),
                cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 91],
            )
            candidate_rows.append(
                {
                    "index": index,
                    "image": f"sam2_amg_candidates/{filename}",
                    "area": int(annotation["area"]),
                    "area_ratio": float(annotation["area"] / (frame.shape[0] * frame.shape[1])),
                    "bbox": [float(value) for value in annotation["bbox"]],
                    "predicted_iou": float(annotation["predicted_iou"]),
                    "stability_score": float(annotation["stability_score"]),
                    "iou_with_identity_mask": float(iou),
                    "passes_training_filter": id(annotation) in filtered_ids,
                    "best_identity_match": index == best_index,
                }
            )
        candidate_masks = np.stack(
            [np.asarray(row["segmentation"], dtype=np.uint8) for row in annotations]
        )
        np.savez_compressed(
            case_dir / "sam2_masks.npz",
            selected_identity_mask=target_mask,
            amg_candidate_masks=candidate_masks,
            f04_rgb=frame,
        )
        np.savez_compressed(case_dir / "source_frames.npz", frames=frames)
        write_video(case_dir / "source_training_video.mp4", frames, args.fps)
        selected_overlay = overlay_selected_mask(frame, target_mask, item["target_phrase"])
        query_overlay, query_rows = draw_query_grid(
            frame,
            target_mask,
            expected_grid(args),
            int(args.query_latent_frame),
        )
        raw_overlay = overlay_masks(frame, annotations)
        filtered_overlay = draw_selected_boxes(overlay_masks(frame, filtered), filtered)
        for name, image in (
            ("f04.png", add_label(frame, "F04 training frame")),
            ("sam2_identity_mask.png", selected_overlay),
            ("sam2_amg_all_overlay.png", add_label(raw_overlay, f"AMG all candidates | n={len(annotations)}")),
            ("sam2_amg_filtered_overlay.png", add_label(filtered_overlay, f"training AMG filter | n={len(filtered)}")),
            ("fixed_query_grid.png", query_overlay),
        ):
            cv2.imwrite(str(case_dir / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        record = item["record"]
        sample = item["sample"]
        case_payload = {
            "schema_version": 1,
            "case_key": case_key,
            "family": record.family_key,
            "dataset_split": "train",
            "dataset_index": int(item["dataset_index"]),
            "case_id": record.case_id,
            "caption": record.caption,
            "source_video": record.video_path,
            "sampled_frame_indices": sample["metadata"]["sampled_frame_indices"],
            "resolution": [int(args.height), int(args.width)],
            "target_phrase": item["target_phrase"],
            "target_selection": (
                "first dynamic object phrase -> GroundingDINO box on F00 -> "
                "SAM2 video propagation -> fixed F04 mask"
            ),
            "grounding_debug": item["grounding_debug"],
            "sam2_amg": {
                "raw_candidate_count": len(annotations),
                "training_filtered_count": len(filtered),
                "best_identity_match_index": best_index,
                "best_identity_match_iou": float(ious[best_index]),
                "candidates": candidate_rows,
            },
            "query": {
                "pixel_frame": int(args.query_pixel_frame),
                "latent_frame": int(args.query_latent_frame),
                "expected_grid": list(expected_grid(args)),
                "token_count": len(query_rows),
                "token_indices": query_rows,
            },
            "cache_dir": str(case_dir),
        }
        atomic_json(case_dir / "case.json", case_payload)
        atomic_json(complete, {"case_key": case_key, "state": "complete"})
        cases.append(case_payload)
    del generator, sam2
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(
        cache_root / "manifest.json",
        {
            "schema_version": 1,
            "dataset_root": str(args.dataset_root.resolve()),
            "dataset_split": "train",
            "case_count": len(cases),
            "cases": cases,
        },
    )
    atomic_json(output_root / "status.json", {"state": "prepared", "case_count": len(cases)})


def restore_clean_conditioning(
    latent: torch.Tensor,
    target_x0: torch.Tensor,
    inputs: dict[str, Any],
) -> torch.Tensor:
    clean = inputs.get("clean_prefix_latents")
    prefix = context_wan.resolve_num_clean_prefix_latents(
        clean_prefix_latents=clean,
        num_clean_prefix_latents=inputs.get("num_clean_prefix_latents"),
    )
    output = latent.clone()
    if prefix > 0:
        output[:, :, :prefix] = target_x0[:, :, :prefix]
    elif "first_frame_latents" in inputs:
        output[:, :, :1] = target_x0[:, :, :1]
    return output


def run_forward(args: argparse.Namespace) -> None:
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run prepare first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    print("[forward] loading official Wan2.2-TI2V-5B baseline", flush=True)
    pipe = build_pipeline(args.wan_root.resolve(), str(device), None)
    pipe.dit.requires_grad_(False).eval()
    selected_heads, head_metadata = train_core.load_head_selection_config(
        args.head_config,
        expected_subset_id=args.head_subset,
        expected_role="T",
        expected_feature_subtype=args.head_subtype,
        expected_num_heads=100,
        num_blocks=30,
        num_heads=24,
    )
    runner = SimpleNamespace(
        _motion_probe_dit=pipe.dit,
        motion_probe_selected_heads_by_block=selected_heads,
        motion_probe_gradient_checkpointing_offload=True,
    )
    case_summaries = []
    for case_position, case in enumerate(manifest["cases"], start=1):
        case_key = case["case_key"]
        case_cache = Path(case["cache_dir"])
        case_output = output_root / "cases" / case_key
        complete = case_output / "forward_complete.json"
        if complete.is_file() and not args.overwrite:
            case_summaries.append(json.loads((case_output / "metrics.json").read_text()))
            print(f"[forward {case_position}/{len(manifest['cases'])}] skip {case_key}", flush=True)
            continue
        case_output.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats(device)
        print(f"[forward {case_position}/{len(manifest['cases'])}] {case_key}", flush=True)
        source_frames = wan_tools.load_video_prefix(
            Path(case["source_video"]),
            int(args.num_frames),
            int(args.height),
            int(args.width),
            "cache",
        )
        context_frames = source_frames[: int(args.context_frames)]
        target_x0 = wan_tools.encode_gt_video(pipe, source_frames, "whole_video")
        inputs_shared, inputs_positive = wan_tools.prepare_conditioning(
            pipe,
            prompt=case["caption"],
            context_video=context_frames,
            height=int(args.height),
            width=int(args.width),
            num_frames=int(args.num_frames),
            sampling_steps=40,
            sigma_shift=5.0,
            cfg_scale=5.0,
            seed=int(args.seed) + case_position,
        )
        captured_inputs = dict(inputs_shared)
        captured_inputs.update(inputs_positive)
        pipe.scheduler.set_timesteps(1000, training=True)
        timestep = torch.full(
            (1,),
            float(args.training_timestep),
            device=device,
            dtype=pipe.torch_dtype,
        )
        train_generator = torch.Generator(device=device)
        train_generator.manual_seed(int(args.seed) + 100 * case_position)
        training_noise = torch.randn(
            target_x0.shape,
            generator=train_generator,
            device=device,
            dtype=target_x0.dtype,
        )
        latent_xt = pipe.scheduler.add_noise(target_x0, training_noise, timestep)
        latent_xt = restore_clean_conditioning(latent_xt, target_x0, captured_inputs)
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        main_inputs = dict(captured_inputs)
        main_inputs["latents"] = latent_xt
        with torch.no_grad():
            velocity = pipe.model_fn(**models, **main_inputs, timestep=timestep)
        sigma = context_wan._diffsynth_sigma_for_timestep(pipe.scheduler, timestep).to(
            device=device, dtype=target_x0.dtype
        )
        pred_x0 = restore_clean_conditioning(
            latent_xt - sigma * velocity,
            target_x0,
            captured_inputs,
        )
        target_velocity = pipe.scheduler.training_target(target_x0, training_noise, timestep)
        clean_prefix_len = context_wan.resolve_num_clean_prefix_latents(
            clean_prefix_latents=captured_inputs.get("clean_prefix_latents"),
            num_clean_prefix_latents=captured_inputs.get("num_clean_prefix_latents"),
        )
        flow_mse = torch.nn.functional.mse_loss(
            velocity[:, :, clean_prefix_len:].float(),
            target_velocity[:, :, clean_prefix_len:].float(),
        )
        flow_weight = pipe.scheduler.training_weight(timestep).to(flow_mse)
        flow_loss = flow_mse * flow_weight
        grid = _probe_grid(pipe.dit, target_x0)
        with np.load(case_cache / "sam2_masks.npz") as mask_arrays:
            selected_mask = mask_arrays["selected_identity_mask"].astype(np.uint8)
        query_rows = query_rows_from_mask(
            torch.from_numpy(selected_mask),
            grid=grid,
            query_latent_frame=int(args.query_latent_frame),
        )
        probe_generator = torch.Generator(device=device)
        probe_generator.manual_seed(int(args.seed) + 1000 + case_position)
        probe_noise = torch.randn(
            target_x0.shape,
            generator=probe_generator,
            device=device,
            dtype=target_x0.dtype,
        )
        teacher_probe_input = restore_clean_conditioning(
            blend_with_fixed_probe_noise(
                target_x0,
                probe_noise,
                noise_level=float(args.probe_noise_level),
            ),
            target_x0,
            captured_inputs,
        )
        probe_timestep = torch.full(
            (1,),
            float(args.probe_timestep),
            device=device,
            dtype=pipe.torch_dtype,
        )
        with torch.no_grad():
            teacher_heatmap, fixed_query_by_block = FrozenMotionProbeWanModule._run_frozen_probe(
                runner,
                latents=teacher_probe_input,
                timestep=probe_timestep,
                captured_inputs=captured_inputs,
                query_rows=query_rows,
                grid=grid,
                retain_input_gradient=False,
                fixed_query_by_block=None,
            )
            teacher_heatmap = teacher_heatmap.detach()
            fixed_query_by_block = {
                block: value.detach() for block, value in fixed_query_by_block.items()
            }
        velocity_leaf = velocity.detach().requires_grad_(True)
        pred_x0_leaf = restore_clean_conditioning(
            latent_xt.detach() - sigma * velocity_leaf,
            target_x0,
            captured_inputs,
        )
        student_probe_input = restore_clean_conditioning(
            blend_with_fixed_probe_noise(
                pred_x0_leaf,
                probe_noise,
                noise_level=float(args.probe_noise_level),
            ),
            target_x0,
            captured_inputs,
        )
        with torch.enable_grad():
            student_heatmap, _ = FrozenMotionProbeWanModule._run_frozen_probe(
                runner,
                latents=student_probe_input,
                timestep=probe_timestep,
                captured_inputs=captured_inputs,
                query_rows=query_rows,
                grid=grid,
                retain_input_gradient=True,
                fixed_query_by_block=fixed_query_by_block,
            )
            heatmap_loss = student_teacher_heatmap_kl(student_heatmap, teacher_heatmap)
            trajectory_loss, student_trajectory, teacher_trajectory = trajectory_huber_loss(
                student_heatmap,
                teacher_heatmap,
                delta=float(args.trajectory_huber_delta),
            )
            auxiliary_loss = (
                float(args.heatmap_weight) * heatmap_loss
                + float(args.trajectory_weight) * trajectory_loss
            )
            velocity_gradient = torch.autograd.grad(
                auxiliary_loss,
                velocity_leaf,
                retain_graph=False,
                create_graph=False,
            )[0]
        student_probe_input_saved = student_probe_input.detach()
        student_heatmap_saved = student_heatmap.detach()
        trajectory_distance = torch.linalg.vector_norm(
            student_trajectory.detach() - teacher_trajectory.detach(), dim=-1
        ).mean()
        probe_sigma = context_wan._diffsynth_sigma_for_timestep(
            pipe.scheduler, probe_timestep
        )
        torch.save(
            {
                "target_x0": target_x0.detach().cpu().to(torch.float16),
                "training_xt": latent_xt.detach().cpu().to(torch.float16),
                "predicted_velocity": velocity.detach().cpu().to(torch.float16),
                "predicted_x0": pred_x0.detach().cpu().to(torch.float16),
                "teacher_probe_input": teacher_probe_input.detach().cpu().to(torch.float16),
                "student_probe_input": student_probe_input_saved.cpu().to(torch.float16),
            },
            case_output / "latents.pt",
        )
        np.savez_compressed(
            case_output / "probe_outputs.npz",
            teacher_heatmap=teacher_heatmap.cpu().float().numpy(),
            student_heatmap=student_heatmap_saved.cpu().float().numpy(),
            heatmap_difference=(student_heatmap_saved - teacher_heatmap).cpu().float().numpy(),
            teacher_trajectory=teacher_trajectory.detach().cpu().float().numpy(),
            student_trajectory=student_trajectory.detach().cpu().float().numpy(),
            query_rows=query_rows.cpu().numpy(),
            selected_mask=selected_mask,
        )
        metrics = {
            "schema_version": 1,
            "case_key": case_key,
            "family": case["family"],
            "caption": case["caption"],
            "dataset_split": "train",
            "dataset_index": case["dataset_index"],
            "model": "official Wan2.2-TI2V-5B baseline; no loaded LoRA",
            "main_student_equivalence": (
                "step-0 baseline forward; zero-initialized adapter has identical output"
            ),
            "probe_parameter_sharing": (
                "single frozen baseline reused sequentially for diagnostic; weights equal to "
                "the separately loaded training probe"
            ),
            "fixed_query_definition": "Teacher GT-Q detached; Student map uses GT-Q x Student-K",
            "target_phrase": case["target_phrase"],
            "grid": list(grid),
            "query_token_count": int(query_rows.numel()),
            "query_rows": query_rows.tolist(),
            "head_selection": {
                "subset_id": head_metadata["subset_id"],
                "feature_subtype": head_metadata["feature_subtype"],
                "num_heads": 100,
                "num_blocks": len(selected_heads),
                "config_sha256": head_metadata["config_sha256"],
            },
            "flow": {
                "training_timestep": float(timestep.item()),
                "scheduler_sigma": float(sigma.float().item()),
                "clean_prefix_latents": int(clean_prefix_len),
                "raw_v_mse": float(flow_mse.item()),
                "training_weight": float(flow_weight.item()),
                "weighted_loss": float(flow_loss.item()),
            },
            "probe": {
                "timestep": float(probe_timestep.item()),
                "noise_level": float(args.probe_noise_level),
                "scheduler_sigma": float(probe_sigma.float().item()),
                "shared_noise_seed": int(args.seed) + 1000 + case_position,
                "teacher_requires_grad": bool(teacher_heatmap.requires_grad),
                "student_requires_grad": bool(student_heatmap.requires_grad),
                "heatmap_kl_student_teacher": float(heatmap_loss.detach().item()),
                "trajectory_huber": float(trajectory_loss.detach().item()),
                "heatmap_weight": float(args.heatmap_weight),
                "trajectory_weight": float(args.trajectory_weight),
                "weighted_auxiliary_loss": float(auxiliary_loss.detach().item()),
                "trajectory_distance_normalized": float(trajectory_distance.item()),
                "gradient_to_first_pass_v_pred_norm": float(
                    velocity_gradient.detach().float().norm().item()
                ),
                "gradient_to_first_pass_v_pred_abs_mean": float(
                    velocity_gradient.detach().float().abs().mean().item()
                ),
                "probe_trainable_parameters": 0,
            },
            "tensors": {
                "target_x0": tensor_stats(target_x0),
                "training_xt": tensor_stats(latent_xt),
                "predicted_velocity": tensor_stats(velocity),
                "predicted_x0": tensor_stats(pred_x0),
                "teacher_probe_input": tensor_stats(teacher_probe_input),
                "student_probe_input": tensor_stats(student_probe_input_saved),
                "teacher_heatmap": tensor_stats(teacher_heatmap),
                "student_heatmap": tensor_stats(student_heatmap_saved),
            },
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        }
        atomic_json(case_output / "metrics.json", metrics)
        atomic_json(complete, {"case_key": case_key, "state": "complete"})
        case_summaries.append(metrics)
        del (
            target_x0,
            inputs_shared,
            inputs_positive,
            captured_inputs,
            training_noise,
            latent_xt,
            velocity,
            target_velocity,
            pred_x0,
            probe_noise,
            teacher_probe_input,
            teacher_heatmap,
            fixed_query_by_block,
            velocity_leaf,
            pred_x0_leaf,
            student_probe_input,
            student_probe_input_saved,
            student_heatmap,
            student_heatmap_saved,
            velocity_gradient,
        )
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        output_root / "forward_summary.json",
        {
            "schema_version": 1,
            "case_count": len(case_summaries),
            "cases": case_summaries,
        },
    )
    atomic_json(output_root / "status.json", {"state": "forward_complete", "case_count": len(case_summaries)})


def collect_case_sweep_metrics(
    case: dict[str, Any],
    case_sweep: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    stages = []
    probes = []
    comparisons = []
    for stage in config["training_stages"]:
        stage_root = case_sweep / "stages" / stage["id"]
        stages.append(json.loads((stage_root / "metrics.json").read_text(encoding="utf-8")))
    for probe in config["probe_settings"]:
        probe_root = case_sweep / "probes" / probe["id"]
        probes.append(json.loads((probe_root / "metrics.json").read_text(encoding="utf-8")))
    for stage in config["training_stages"]:
        for probe in config["probe_settings"]:
            comparison_root = case_sweep / "comparisons" / stage["id"] / probe["id"]
            comparisons.append(
                json.loads((comparison_root / "metrics.json").read_text(encoding="utf-8"))
            )
    return {
        "schema_version": 2,
        "case_key": case["case_key"],
        "training_noise_protocol": "one shared epsilon_train across all training timesteps",
        "probe_noise_protocol": (
            "one shared epsilon_p across both noise levels and all training timesteps; "
            "Teacher and Student share epsilon_p"
        ),
        "training_stages": stages,
        "probe_settings": probes,
        "comparisons": comparisons,
    }


def run_sweep_forward(args: argparse.Namespace) -> None:
    config = noise_sweep_config(args)
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run prepare first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    print("[sweep-forward] loading official Wan2.2-TI2V-5B baseline", flush=True)
    pipe = build_pipeline(args.wan_root.resolve(), str(device), None)
    pipe.dit.requires_grad_(False).eval()
    selected_heads, head_metadata = train_core.load_head_selection_config(
        args.head_config,
        expected_subset_id=args.head_subset,
        expected_role="T",
        expected_feature_subtype=args.head_subtype,
        expected_num_heads=100,
        num_blocks=30,
        num_heads=24,
    )
    runner = SimpleNamespace(
        _motion_probe_dit=pipe.dit,
        motion_probe_selected_heads_by_block=selected_heads,
        motion_probe_gradient_checkpointing_offload=True,
    )
    completed_cases = []
    for case_position, case in enumerate(manifest["cases"], start=1):
        case_key = case["case_key"]
        case_cache = Path(case["cache_dir"])
        case_output = output_root / "cases" / case_key
        case_sweep = case_output / "noise_sweep"
        complete = case_sweep / "forward_complete.json"
        if complete.is_file() and not args.overwrite:
            completed_cases.append(
                json.loads((case_sweep / "metrics.json").read_text(encoding="utf-8"))
            )
            print(
                f"[sweep-forward {case_position}/{len(manifest['cases'])}] skip {case_key}",
                flush=True,
            )
            continue
        print(
            f"[sweep-forward {case_position}/{len(manifest['cases'])}] {case_key}",
            flush=True,
        )
        case_sweep.mkdir(parents=True, exist_ok=True)
        source_frames = wan_tools.load_video_prefix(
            Path(case["source_video"]),
            int(args.num_frames),
            int(args.height),
            int(args.width),
            "cache",
        )
        context_frames = source_frames[: int(args.context_frames)]
        target_x0 = wan_tools.encode_gt_video(pipe, source_frames, "whole_video")
        inputs_shared, inputs_positive = wan_tools.prepare_conditioning(
            pipe,
            prompt=case["caption"],
            context_video=context_frames,
            height=int(args.height),
            width=int(args.width),
            num_frames=int(args.num_frames),
            sampling_steps=40,
            sigma_shift=5.0,
            cfg_scale=5.0,
            seed=int(args.seed) + case_position,
        )
        captured_inputs = dict(inputs_shared)
        captured_inputs.update(inputs_positive)
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        clean_prefix_len = context_wan.resolve_num_clean_prefix_latents(
            clean_prefix_latents=captured_inputs.get("clean_prefix_latents"),
            num_clean_prefix_latents=captured_inputs.get("num_clean_prefix_latents"),
        )
        grid = _probe_grid(pipe.dit, target_x0)
        with np.load(case_cache / "sam2_masks.npz") as mask_arrays:
            selected_mask = mask_arrays["selected_identity_mask"].astype(np.uint8)
        query_rows = query_rows_from_mask(
            torch.from_numpy(selected_mask),
            grid=grid,
            query_latent_frame=int(args.query_latent_frame),
        )

        train_noise_seed = int(args.seed) + 100 * case_position
        train_generator = torch.Generator(device=device)
        train_generator.manual_seed(train_noise_seed)
        training_noise = torch.randn(
            target_x0.shape,
            generator=train_generator,
            device=device,
            dtype=target_x0.dtype,
        )
        for stage_position, stage in enumerate(config["training_stages"], start=1):
            stage_root = case_sweep / "stages" / stage["id"]
            stage_complete = stage_root / "forward_complete.json"
            if stage_complete.is_file() and not args.overwrite:
                print(
                    f"  [main {stage_position}/{len(config['training_stages'])}] "
                    f"skip t={stage['timestep']:.0f}",
                    flush=True,
                )
                continue
            stage_root.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats(device)
            timestep = torch.full(
                (1,),
                float(stage["timestep"]),
                device=device,
                dtype=pipe.torch_dtype,
            )
            latent_xt = pipe.scheduler.add_noise(target_x0, training_noise, timestep)
            latent_xt = restore_clean_conditioning(latent_xt, target_x0, captured_inputs)
            main_inputs = dict(captured_inputs)
            main_inputs["latents"] = latent_xt
            print(
                f"  [main {stage_position}/{len(config['training_stages'])}] "
                f"t={stage['timestep']:.0f}",
                flush=True,
            )
            with torch.no_grad():
                velocity = pipe.model_fn(**models, **main_inputs, timestep=timestep)
            sigma = context_wan._diffsynth_sigma_for_timestep(pipe.scheduler, timestep).to(
                device=device, dtype=target_x0.dtype
            )
            pred_x0 = restore_clean_conditioning(
                latent_xt - sigma * velocity,
                target_x0,
                captured_inputs,
            )
            target_velocity = pipe.scheduler.training_target(
                target_x0, training_noise, timestep
            )
            flow_mse = torch.nn.functional.mse_loss(
                velocity[:, :, clean_prefix_len:].float(),
                target_velocity[:, :, clean_prefix_len:].float(),
            )
            flow_weight = pipe.scheduler.training_weight(timestep).to(flow_mse)
            flow_loss = flow_mse * flow_weight
            torch.save(
                {
                    "training_xt": latent_xt.detach().cpu(),
                    "predicted_velocity": velocity.detach().cpu(),
                    "predicted_x0": pred_x0.detach().cpu(),
                },
                stage_root / "latents.pt",
            )
            atomic_json(
                stage_root / "metrics.json",
                {
                    "id": stage["id"],
                    "training_timestep": float(timestep.item()),
                    "scheduler_sigma": float(sigma.float().item()),
                    "shared_training_noise_seed": train_noise_seed,
                    "clean_prefix_latents": int(clean_prefix_len),
                    "raw_v_mse": float(flow_mse.item()),
                    "training_weight": float(flow_weight.item()),
                    "weighted_loss": float(flow_loss.item()),
                    "peak_gpu_memory_mib": float(
                        torch.cuda.max_memory_allocated(device) / 2**20
                    ),
                    "tensors": {
                        "training_xt": tensor_stats(latent_xt),
                        "predicted_velocity": tensor_stats(velocity),
                        "predicted_x0": tensor_stats(pred_x0),
                    },
                },
            )
            atomic_json(stage_complete, {"state": "complete", "id": stage["id"]})
            del latent_xt, velocity, pred_x0, target_velocity
            gc.collect()
            torch.cuda.empty_cache()

        probe_noise_seed = int(args.seed) + 1000 + case_position
        probe_generator = torch.Generator(device=device)
        probe_generator.manual_seed(probe_noise_seed)
        probe_noise = torch.randn(
            target_x0.shape,
            generator=probe_generator,
            device=device,
            dtype=target_x0.dtype,
        )
        for probe_position, probe in enumerate(config["probe_settings"], start=1):
            probe_root = case_sweep / "probes" / probe["id"]
            teacher_path = probe_root / "teacher.pt"
            teacher_complete = probe_root / "forward_complete.json"
            teacher_bundle = None
            if teacher_complete.is_file() and not args.overwrite:
                teacher_bundle = torch.load(teacher_path, map_location="cpu", weights_only=True)
                teacher_probe_input = teacher_bundle["teacher_probe_input"].to(device)
                teacher_heatmap = teacher_bundle["teacher_heatmap"].to(device)
                fixed_query_by_block = {
                    int(block): value.to(device)
                    for block, value in teacher_bundle["fixed_query_by_block"].items()
                }
                print(
                    f"  [teacher {probe_position}/{len(config['probe_settings'])}] "
                    f"load noise={probe['noise_level']:.2f}",
                    flush=True,
                )
            else:
                probe_root.mkdir(parents=True, exist_ok=True)
                torch.cuda.reset_peak_memory_stats(device)
                probe_timestep = torch.full(
                    (1,),
                    float(probe["timestep"]),
                    device=device,
                    dtype=pipe.torch_dtype,
                )
                teacher_probe_input = restore_clean_conditioning(
                    blend_with_fixed_probe_noise(
                        target_x0,
                        probe_noise,
                        noise_level=float(probe["noise_level"]),
                    ),
                    target_x0,
                    captured_inputs,
                )
                print(
                    f"  [teacher {probe_position}/{len(config['probe_settings'])}] "
                    f"noise={probe['noise_level']:.2f}, t={probe['timestep']:.0f}",
                    flush=True,
                )
                with torch.no_grad():
                    teacher_heatmap, fixed_query_by_block = (
                        FrozenMotionProbeWanModule._run_frozen_probe(
                            runner,
                            latents=teacher_probe_input,
                            timestep=probe_timestep,
                            captured_inputs=captured_inputs,
                            query_rows=query_rows,
                            grid=grid,
                            retain_input_gradient=False,
                            fixed_query_by_block=None,
                        )
                    )
                    teacher_heatmap = teacher_heatmap.detach()
                    fixed_query_by_block = {
                        block: value.detach() for block, value in fixed_query_by_block.items()
                    }
                probe_sigma = context_wan._diffsynth_sigma_for_timestep(
                    pipe.scheduler, probe_timestep
                )
                torch.save(
                    {
                        "teacher_probe_input": teacher_probe_input.detach().cpu(),
                        "teacher_heatmap": teacher_heatmap.cpu().float(),
                        "fixed_query_by_block": {
                            block: value.cpu()
                            for block, value in fixed_query_by_block.items()
                        },
                    },
                    teacher_path,
                )
                atomic_json(
                    probe_root / "metrics.json",
                    {
                        "id": probe["id"],
                        "noise_level": float(probe["noise_level"]),
                        "timestep": float(probe_timestep.item()),
                        "scheduler_sigma": float(probe_sigma.float().item()),
                        "shared_probe_noise_seed": probe_noise_seed,
                        "teacher_requires_grad": bool(teacher_heatmap.requires_grad),
                        "peak_gpu_memory_mib": float(
                            torch.cuda.max_memory_allocated(device) / 2**20
                        ),
                        "tensors": {
                            "teacher_probe_input": tensor_stats(teacher_probe_input),
                            "teacher_heatmap": tensor_stats(teacher_heatmap),
                        },
                    },
                )
                atomic_json(
                    teacher_complete, {"state": "complete", "id": probe["id"]}
                )
            probe_timestep = torch.full(
                (1,),
                float(probe["timestep"]),
                device=device,
                dtype=pipe.torch_dtype,
            )
            for stage_position, stage in enumerate(config["training_stages"], start=1):
                comparison_root = (
                    case_sweep / "comparisons" / stage["id"] / probe["id"]
                )
                comparison_complete = comparison_root / "forward_complete.json"
                if comparison_complete.is_file() and not args.overwrite:
                    print(
                        f"    [student {stage_position}/{len(config['training_stages'])}] "
                        f"skip t={stage['timestep']:.0f}",
                        flush=True,
                    )
                    continue
                comparison_root.mkdir(parents=True, exist_ok=True)
                stage_root = case_sweep / "stages" / stage["id"]
                stage_bundle = torch.load(
                    stage_root / "latents.pt", map_location="cpu", weights_only=True
                )
                latent_xt = stage_bundle["training_xt"].to(device)
                velocity = stage_bundle["predicted_velocity"].to(device)
                timestep = torch.full(
                    (1,),
                    float(stage["timestep"]),
                    device=device,
                    dtype=pipe.torch_dtype,
                )
                sigma = context_wan._diffsynth_sigma_for_timestep(
                    pipe.scheduler, timestep
                ).to(device=device, dtype=target_x0.dtype)
                velocity_leaf = velocity.detach().requires_grad_(True)
                pred_x0_leaf = restore_clean_conditioning(
                    latent_xt.detach() - sigma * velocity_leaf,
                    target_x0,
                    captured_inputs,
                )
                student_probe_input = restore_clean_conditioning(
                    blend_with_fixed_probe_noise(
                        pred_x0_leaf,
                        probe_noise,
                        noise_level=float(probe["noise_level"]),
                    ),
                    target_x0,
                    captured_inputs,
                )
                torch.cuda.reset_peak_memory_stats(device)
                print(
                    f"    [student {stage_position}/{len(config['training_stages'])}] "
                    f"t={stage['timestep']:.0f}, noise={probe['noise_level']:.2f}",
                    flush=True,
                )
                with torch.enable_grad():
                    student_heatmap, _ = FrozenMotionProbeWanModule._run_frozen_probe(
                        runner,
                        latents=student_probe_input,
                        timestep=probe_timestep,
                        captured_inputs=captured_inputs,
                        query_rows=query_rows,
                        grid=grid,
                        retain_input_gradient=True,
                        fixed_query_by_block=fixed_query_by_block,
                    )
                    heatmap_loss = student_teacher_heatmap_kl(
                        student_heatmap, teacher_heatmap
                    )
                    trajectory_loss, student_trajectory, teacher_trajectory = (
                        trajectory_huber_loss(
                            student_heatmap,
                            teacher_heatmap,
                            delta=float(args.trajectory_huber_delta),
                        )
                    )
                    auxiliary_loss = (
                        float(args.heatmap_weight) * heatmap_loss
                        + float(args.trajectory_weight) * trajectory_loss
                    )
                    velocity_gradient = torch.autograd.grad(
                        auxiliary_loss,
                        velocity_leaf,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                student_heatmap_saved = student_heatmap.detach()
                trajectory_distance = torch.linalg.vector_norm(
                    student_trajectory.detach() - teacher_trajectory.detach(), dim=-1
                ).mean()
                torch.save(
                    {"student_probe_input": student_probe_input.detach().cpu()},
                    comparison_root / "latents.pt",
                )
                np.savez_compressed(
                    comparison_root / "probe_outputs.npz",
                    student_heatmap=student_heatmap_saved.cpu().float().numpy(),
                    heatmap_difference=(student_heatmap_saved - teacher_heatmap)
                    .cpu()
                    .float()
                    .numpy(),
                    teacher_trajectory=teacher_trajectory.detach().cpu().float().numpy(),
                    student_trajectory=student_trajectory.detach().cpu().float().numpy(),
                )
                atomic_json(
                    comparison_root / "metrics.json",
                    {
                        "training_stage_id": stage["id"],
                        "probe_setting_id": probe["id"],
                        "training_timestep": float(stage["timestep"]),
                        "probe_noise_level": float(probe["noise_level"]),
                        "probe_timestep": float(probe["timestep"]),
                        "heatmap_kl_student_teacher": float(heatmap_loss.detach().item()),
                        "trajectory_huber": float(trajectory_loss.detach().item()),
                        "weighted_auxiliary_loss": float(auxiliary_loss.detach().item()),
                        "trajectory_distance_normalized": float(
                            trajectory_distance.item()
                        ),
                        "gradient_to_first_pass_v_pred_norm": float(
                            velocity_gradient.detach().float().norm().item()
                        ),
                        "gradient_to_first_pass_v_pred_abs_mean": float(
                            velocity_gradient.detach().float().abs().mean().item()
                        ),
                        "teacher_requires_grad": bool(teacher_heatmap.requires_grad),
                        "student_requires_grad": bool(student_heatmap.requires_grad),
                        "probe_trainable_parameters": 0,
                        "peak_gpu_memory_mib": float(
                            torch.cuda.max_memory_allocated(device) / 2**20
                        ),
                        "tensors": {
                            "student_probe_input": tensor_stats(student_probe_input),
                            "student_heatmap": tensor_stats(student_heatmap_saved),
                        },
                    },
                )
                atomic_json(
                    comparison_complete,
                    {
                        "state": "complete",
                        "training_stage_id": stage["id"],
                        "probe_setting_id": probe["id"],
                    },
                )
                del (
                    stage_bundle,
                    latent_xt,
                    velocity,
                    velocity_leaf,
                    pred_x0_leaf,
                    student_probe_input,
                    student_heatmap,
                    student_heatmap_saved,
                    velocity_gradient,
                )
                gc.collect()
                torch.cuda.empty_cache()
            del teacher_bundle, teacher_probe_input, teacher_heatmap, fixed_query_by_block
            gc.collect()
            torch.cuda.empty_cache()

        sweep_metrics = collect_case_sweep_metrics(case, case_sweep, config)
        sweep_metrics.update(
            {
                "grid": list(grid),
                "query_token_count": int(query_rows.numel()),
                "head_selection": {
                    "subset_id": head_metadata["subset_id"],
                    "feature_subtype": head_metadata["feature_subtype"],
                    "num_heads": 100,
                    "num_blocks": len(selected_heads),
                    "config_sha256": head_metadata["config_sha256"],
                },
            }
        )
        atomic_json(case_sweep / "metrics.json", sweep_metrics)
        atomic_json(complete, {"state": "complete", "case_key": case_key})
        completed_cases.append(sweep_metrics)
        del (
            source_frames,
            context_frames,
            target_x0,
            inputs_shared,
            inputs_positive,
            captured_inputs,
            training_noise,
            probe_noise,
        )
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        output_root / "noise_sweep_forward_summary.json",
        {
            "schema_version": 2,
            "case_count": len(completed_cases),
            "training_timesteps": [
                stage["timestep"] for stage in config["training_stages"]
            ],
            "probe_settings": list(config["probe_settings"]),
            "cases": completed_cases,
        },
    )
    atomic_json(
        output_root / "noise_sweep_status.json",
        {"state": "forward_complete", "case_count": len(completed_cases)},
    )


def decode_latent_bundle(
    vae_pipe: WanVideoPipeline,
    bundle: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    vae_pipe.load_models_to_device(["vae"])
    vae_dtype = next(vae_pipe.vae.model.parameters()).dtype
    decoded: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name, latent in bundle.items():
            value = latent.to(device=vae_pipe.device, dtype=vae_dtype)
            video = vae_pipe.vae.decode(
                value,
                device=vae_pipe.device,
                tiled=True,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            ).clamp(-1, 1)
            decoded[name] = decoded_video_to_uint8(video.cpu())
    return decoded


def anchor_frame_indices(latent_frames: int, pixel_frames: int = 49) -> list[int]:
    return [min(4 * index, pixel_frames - 1) for index in range(int(latent_frames))]


def heatmap_scale(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float32)
    positive = positive[positive > 0]
    return max(float(np.quantile(positive, 0.995)) if positive.size else 0.0, 1.0e-12)


def colorize_heatmap(values: np.ndarray, size: tuple[int, int], vmax: float) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    resized = cv2.resize(np.asarray(values, dtype=np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    normalized = np.clip(resized / max(float(vmax), 1.0e-12), 0.0, 1.0)
    bgr = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), normalized


def overlay_heatmap(frame: np.ndarray, values: np.ndarray, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    heat, normalized = colorize_heatmap(values, (frame.shape[1], frame.shape[0]), vmax)
    alpha = (0.10 + 0.76 * normalized)[..., None]
    overlay = np.clip(frame * (1.0 - alpha) + heat * alpha, 0, 255).astype(np.uint8)
    return heat, overlay


def signed_difference_image(values: np.ndarray, size: tuple[int, int], vmax: float) -> np.ndarray:
    width, height = size
    resized = cv2.resize(np.asarray(values, dtype=np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    normalized = np.clip(resized / max(float(vmax), 1.0e-12), -1.0, 1.0)
    output = np.full((height, width, 3), 245.0, dtype=np.float32)
    positive = np.clip(normalized, 0.0, 1.0)[..., None]
    negative = np.clip(-normalized, 0.0, 1.0)[..., None]
    output = output * (1.0 - positive) + np.asarray([213, 52, 65]) * positive
    output = output * (1.0 - negative) + np.asarray([32, 101, 184]) * negative
    return output.round().clip(0, 255).astype(np.uint8)


def resize_panel(frame: np.ndarray, width: int = 448, height: int = 256) -> np.ndarray:
    return cv2.resize(np.asarray(frame, dtype=np.uint8), (width, height), interpolation=cv2.INTER_AREA)


def render_heatmap_media(
    case_output: Path,
    decoded: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    heatmap_fps: float,
) -> dict[str, str]:
    teacher = arrays["teacher_heatmap"][0]
    student = arrays["student_heatmap"][0]
    difference = arrays["heatmap_difference"][0]
    anchors = anchor_frame_indices(teacher.shape[0], decoded["target_x0"].shape[0])
    teacher_scale = heatmap_scale(teacher)
    student_scale = heatmap_scale(student)
    shared_scale = max(teacher_scale, student_scale)
    difference_scale = max(float(np.quantile(np.abs(difference), 0.995)), 1.0e-12)
    teacher_pure, teacher_overlay, teacher_triptych = [], [], []
    student_pure, student_overlay, student_triptych = [], [], []
    comparison = []
    contact_panels = []
    for latent_index, pixel_index in enumerate(anchors):
        gt_frame = decoded["target_x0"][pixel_index]
        pred_frame = decoded["predicted_x0"][pixel_index]
        t_heat, t_overlay = overlay_heatmap(gt_frame, teacher[latent_index], shared_scale)
        s_heat, s_overlay = overlay_heatmap(pred_frame, student[latent_index], shared_scale)
        diff_image = signed_difference_image(
            difference[latent_index], (gt_frame.shape[1], gt_frame.shape[0]), difference_scale
        )
        label = f"latent-{latent_index:02d} / F{pixel_index:02d}"
        teacher_pure.append(add_label(t_heat, f"Teacher Top100 | {label}"))
        teacher_overlay.append(add_label(t_overlay, f"Teacher overlay | {label}"))
        student_pure.append(add_label(s_heat, f"Student Top100 | {label}"))
        student_overlay.append(add_label(s_overlay, f"Student overlay | {label}"))
        t_panels = [
            add_label(resize_panel(gt_frame), f"GT x0 | F{pixel_index:02d}"),
            add_label(resize_panel(t_heat), f"Teacher Top100 | L{latent_index:02d}"),
            add_label(resize_panel(t_overlay), "Teacher overlay"),
        ]
        s_panels = [
            add_label(resize_panel(pred_frame), f"x0 pred | F{pixel_index:02d}"),
            add_label(resize_panel(s_heat), f"Student Top100 | L{latent_index:02d}"),
            add_label(resize_panel(s_overlay), "Student overlay"),
        ]
        teacher_triptych.append(np.concatenate(t_panels, axis=1))
        student_triptych.append(np.concatenate(s_panels, axis=1))
        five = np.concatenate(
            [
                add_label(resize_panel(gt_frame), f"GT x0 F{pixel_index:02d}"),
                add_label(resize_panel(pred_frame), f"x0 pred F{pixel_index:02d}"),
                add_label(resize_panel(t_overlay), "Teacher overlay"),
                add_label(resize_panel(s_overlay), "Student overlay"),
                add_label(resize_panel(diff_image), "Student - Teacher"),
            ],
            axis=1,
        )
        comparison.append(five)
        contact_panels.append(five)
    files = {
        "teacher_heatmap": "teacher_top100_heatmap.mp4",
        "teacher_overlay": "teacher_top100_overlay.mp4",
        "teacher_triptych": "teacher_frame_heatmap_overlay.mp4",
        "student_heatmap": "student_top100_heatmap.mp4",
        "student_overlay": "student_top100_overlay.mp4",
        "student_triptych": "student_frame_heatmap_overlay.mp4",
        "comparison": "teacher_student_five_panel.mp4",
        "contact_sheet": "teacher_student_13frame_contact_sheet.jpg",
    }
    for name, frames in (
        (files["teacher_heatmap"], teacher_pure),
        (files["teacher_overlay"], teacher_overlay),
        (files["teacher_triptych"], teacher_triptych),
        (files["student_heatmap"], student_pure),
        (files["student_overlay"], student_overlay),
        (files["student_triptych"], student_triptych),
        (files["comparison"], comparison),
    ):
        write_video(case_output / name, frames, heatmap_fps)
    sheet = make_contact_sheet(contact_panels, columns=1, background=236)
    cv2.imwrite(
        str(case_output / files["contact_sheet"]),
        cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 91],
    )
    return files


def make_contact_sheet(
    frames: list[np.ndarray],
    *,
    columns: int,
    background: int = 242,
) -> np.ndarray:
    if not frames:
        raise ValueError("contact sheet needs at least one frame")
    height = max(frame.shape[0] for frame in frames)
    width = max(frame.shape[1] for frame in frames)
    rows = math.ceil(len(frames) / columns)
    canvas = np.full((rows * height, columns * width, 3), background, dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        canvas[row * height : row * height + frame.shape[0], column * width : column * width + frame.shape[1]] = frame
    return canvas


def trajectory_visualization(
    base_frame: np.ndarray,
    teacher: np.ndarray,
    student: np.ndarray,
) -> np.ndarray:
    output = base_frame.copy()
    height, width = output.shape[:2]
    teacher_points = np.stack(
        [teacher[:, 0] * (width - 1), teacher[:, 1] * (height - 1)], axis=1
    ).round().astype(np.int32)
    student_points = np.stack(
        [student[:, 0] * (width - 1), student[:, 1] * (height - 1)], axis=1
    ).round().astype(np.int32)
    cv2.polylines(output, [teacher_points], False, (31, 204, 116), 4, cv2.LINE_AA)
    cv2.polylines(output, [student_points], False, (239, 73, 69), 4, cv2.LINE_AA)
    for index, (teacher_point, student_point) in enumerate(zip(teacher_points, student_points)):
        cv2.circle(output, tuple(teacher_point), 5, (31, 204, 116), -1, cv2.LINE_AA)
        cv2.circle(output, tuple(student_point), 5, (239, 73, 69), -1, cv2.LINE_AA)
        if index in (0, len(teacher_points) - 1):
            cv2.putText(
                output,
                f"L{index:02d}",
                tuple((teacher_point + np.asarray([6, -6])).tolist()),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
    return add_label(output, "trajectory | Teacher green | Student red")


def media_figure(path: str, title: str, note: str = "") -> str:
    escaped = html.escape(path)
    suffix = Path(path).suffix.lower()
    media = (
        f"<video controls muted loop playsinline preload='metadata' src='{escaped}'></video>"
        if suffix == ".mp4"
        else f"<img loading='lazy' src='{escaped}' alt='{html.escape(title)}'>"
    )
    return (
        f"<figure><figcaption><strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(note)}</span></figcaption>{media}</figure>"
    )


def noise_sweep_html(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return ""
    probes_by_id = {row["id"]: row for row in metrics["probe_settings"]}
    comparisons = {
        (row["training_stage_id"], row["probe_setting_id"]): row
        for row in metrics["comparisons"]
    }
    summary_rows = []
    stage_sections = []
    for stage in metrics["training_stages"]:
        stage_id = stage["id"]
        probe_sections = []
        for probe_id, probe in probes_by_id.items():
            row = comparisons[(stage_id, probe_id)]
            summary_rows.append(
                "<tr>"
                f"<td>{stage['training_timestep']:.0f}</td>"
                f"<td>{stage['scheduler_sigma']:.4f}</td>"
                f"<td>{probe['noise_level']:.2f}</td>"
                f"<td>{probe['timestep']:.0f}</td>"
                f"<td>{row['heatmap_kl_student_teacher']:.6f}</td>"
                f"<td>{row['trajectory_huber']:.6f}</td>"
                f"<td>{row['gradient_to_first_pass_v_pred_norm']:.5f}</td>"
                "</tr>"
            )
            probe_base = f"noise_sweep/probes/{probe_id}"
            comparison_base = f"noise_sweep/comparisons/{stage_id}/{probe_id}"
            probe_sections.append(
                f"""<div class="sweep-probe"><h4>Probe noise {probe['noise_level']:.2f} / timestep {probe['timestep']:.0f}</h4>
<div class="facts"><div class="fact"><span>KL(student || teacher)</span><strong>{row['heatmap_kl_student_teacher']:.6f}</strong></div><div class="fact"><span>Trajectory Huber</span><strong>{row['trajectory_huber']:.6f}</strong></div><div class="fact"><span>Weighted auxiliary</span><strong>{row['weighted_auxiliary_loss']:.6f}</strong></div><div class="fact"><span>Gradient ||dL/dv_pred||</span><strong>{row['gradient_to_first_pass_v_pred_norm']:.5f}</strong></div><div class="fact"><span>Peak allocated</span><strong>{row['peak_gpu_memory_mib']:.0f} MiB</strong></div></div>
<div class="grid">
{media_figure(f'{probe_base}/vae_teacher_probe_input.mp4','GT x0 + shared Probe noise',f"level={probe['noise_level']:.2f}; t={probe['timestep']:.0f}; sigma={probe['scheduler_sigma']:.4f}")}
{media_figure(f'{comparison_base}/vae_student_probe_input.mp4','x0_pred + same Probe noise',f"shared seed={probe['shared_probe_noise_seed']}")}
{media_figure(f'{comparison_base}/teacher_frame_heatmap_overlay.mp4','Teacher: GT frame | Top100 | overlay','corresponding frame concatenation')}
{media_figure(f'{comparison_base}/student_frame_heatmap_overlay.mp4','Student: x0_pred frame | Top100 | overlay','corresponding frame concatenation')}
{media_figure(f'{comparison_base}/teacher_top100_heatmap.mp4','Teacher Top100 aggregate heatmap')}
{media_figure(f'{comparison_base}/student_top100_heatmap.mp4','Student Top100 aggregate heatmap')}
<div class="wide">{media_figure(f'{comparison_base}/teacher_student_five_panel.mp4','GT | x0_pred | Teacher overlay | Student overlay | difference')}</div>
<div class="wide">{media_figure(f'{comparison_base}/teacher_student_13frame_contact_sheet.jpg','13-frame aligned heatmap contact sheet')}</div>
{media_figure(f'{comparison_base}/trajectory_overlay.jpg','Teacher and Student soft-argmax trajectories')}
<figure><figcaption><strong>Trajectory numeric values</strong><span>normalized (x,y), 13 latent frames</span></figcaption><a href="{html.escape(comparison_base)}/trajectory_values.json">Open trajectory_values.json</a></figure>
</div></div>"""
            )
        stage_base = f"noise_sweep/stages/{stage_id}"
        stage_sections.append(
            f"""<div class="sweep-stage"><h3>Training timestep {stage['training_timestep']:.0f} / scheduler sigma {stage['scheduler_sigma']:.4f}</h3>
<div class="facts"><div class="fact"><span>Flow loss</span><strong>{stage['weighted_loss']:.6f}</strong></div><div class="fact"><span>Raw v MSE</span><strong>{stage['raw_v_mse']:.6f}</strong></div><div class="fact"><span>Training noise seed</span><strong>{stage['shared_training_noise_seed']}</strong></div><div class="fact"><span>Peak allocated</span><strong>{stage['peak_gpu_memory_mib']:.0f} MiB</strong></div></div>
<div class="grid">
{media_figure(f'{stage_base}/vae_training_xt.mp4','Training x_t',f"t={stage['training_timestep']:.0f}; sigma={stage['scheduler_sigma']:.4f}")}
{media_figure(f'{stage_base}/vae_predicted_x0.mp4','Baseline x0_pred','x_t - sigma_t * v_pred')}
{media_figure(f'{stage_base}/vae_x0_difference.mp4','Decoded |x0_pred - GT x0| x3')}
{media_figure(f'{stage_base}/x0_anchor_contact_sheet.jpg','13 aligned x0 anchor frames','GT | x_t | x0_pred | difference')}
</div>{''.join(probe_sections)}</div>"""
        )
    return f"""<section id="noise-sweep"><h2>Training-noise stages × low-noise Frozen Probe</h2>
<p class="protocol">Controlled sweep: all five training stages share one <code>epsilon_train</code>. Probe 0.1 and 0.2 share one <code>epsilon_p</code>, and each Teacher/Student pair uses exactly the same corruption. The original t=500 / Probe=0.5 result above is retained.</p>
<div class="table-wrap"><table class="metric-table"><thead><tr><th>Training t</th><th>Training sigma</th><th>Probe noise</th><th>Probe t</th><th>KL</th><th>Trajectory Huber</th><th>||dL/dv_pred||</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>
{''.join(stage_sections)}</section>"""


def case_page(
    case: dict[str, Any],
    metrics: dict[str, Any],
    *,
    index_href: str,
    sweep_metrics: dict[str, Any] | None = None,
) -> str:
    candidates = case["sam2_amg"]["candidates"]
    candidate_html = "".join(
        media_figure(
            f"prep/{row['image']}",
            f"AMG #{row['index']:03d}",
            (
                f"identity IoU {row['iou_with_identity_mask']:.3f}; "
                f"pIoU {row['predicted_iou']:.3f}; stability {row['stability_score']:.3f}; "
                f"{'training-filter in' if row['passes_training_filter'] else 'raw only'}"
            ),
        )
        for row in candidates
    )
    probe = metrics["probe"]
    flow = metrics["flow"]
    sweep_html = noise_sweep_html(sweep_metrics)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(case['case_key'])}</title>
<style>
:root{{--bg:#eef1ed;--paper:#fff;--ink:#18201c;--muted:#5d6962;--line:#b9c2bc;--green:#176c59;--red:#a83e36;--gold:#9a6a16}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Noto Sans CJK SC","Source Han Sans SC",sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:10;background:#eef1edf2;border-bottom:1px solid var(--line);padding:12px 20px;backdrop-filter:blur(10px)}}
header h1{{font-size:22px;margin:2px 0}}header p{{margin:0;color:var(--muted)}}nav{{display:flex;gap:13px;flex-wrap:wrap;margin-top:8px}}a{{color:var(--green);font-weight:700;text-decoration:none}}
main{{width:min(1900px,calc(100% - 24px));margin:auto;padding:18px 0 60px}}section{{border-top:2px solid var(--ink);padding:15px 0 26px}}h2{{font-size:19px;margin:0 0 10px}}h3{{font-size:18px;margin:0 0 10px}}h4{{font-size:16px;margin:0 0 8px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.wide{{grid-column:1/-1}}figure{{margin:0;background:var(--paper);border:1px solid var(--line);padding:8px}}figcaption{{display:flex;justify-content:space-between;gap:12px;margin-bottom:6px}}figcaption span{{font-size:11px;color:var(--muted)}}video,img{{display:block;width:100%;height:auto;background:#101210}}.facts{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));border:1px solid var(--line);background:var(--paper);margin-bottom:12px}}.fact{{padding:10px;border-right:1px solid var(--line)}}.fact:last-child{{border:0}}.fact span{{display:block;color:var(--muted);font-size:11px}}.fact strong{{font-size:16px}}.candidates{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.sweep-stage{{border-top:1px solid var(--line);padding:22px 0 8px}}.sweep-probe{{border-top:1px dashed var(--line);padding:18px 0 8px;margin-top:18px}}.table-wrap{{overflow:auto}}.metric-table{{width:100%;border-collapse:collapse;background:var(--paper);margin:12px 0}}.metric-table th,.metric-table td{{border:1px solid var(--line);padding:8px;text-align:left;white-space:nowrap}}.metric-table th{{background:#dfe5e0}}.protocol{{padding:12px;border-left:5px solid var(--green);background:var(--paper)}}code{{font-size:12px}}@media(max-width:900px){{header{{position:static}}.grid,.candidates{{grid-template-columns:1fr}}.facts{{grid-template-columns:1fr 1fr}}}}
</style></head><body><header><a href="{html.escape(index_href)}">返回总览</a><h1>{html.escape(case['case_key'])}</h1><p>{html.escape(case['caption'])}</p><nav><a href="#sam2">SAM2</a><a href="#x0">x0</a><a href="#probe">Frozen Probe</a>{'<a href="#noise-sweep">Noise sweep</a>' if sweep_metrics else ''}<a href="#candidates">全部 AMG candidates</a></nav></header><main>
<section><div class="facts"><div class="fact"><span>Split / index</span><strong>train / {case['dataset_index']}</strong></div><div class="fact"><span>Grid</span><strong>{' x '.join(map(str,metrics['grid']))}</strong></div><div class="fact"><span>Query cells</span><strong>{metrics['query_token_count']}</strong></div><div class="fact"><span>Top heads</span><strong>100 / {metrics['head_selection']['num_blocks']} blocks</strong></div><div class="fact"><span>Peak allocated</span><strong>{metrics['peak_gpu_memory_mib']:.0f} MiB</strong></div></div></section>
<section id="sam2"><h2>SAM2 target identity and query</h2><div class="grid">
{media_figure('prep/f04.png','F04 training frame')}{media_figure('prep/sam2_identity_mask.png','Selected identity-constrained SAM2 mask',case['target_phrase'])}
{media_figure('prep/sam2_amg_all_overlay.png','Prompt-free SAM2 AMG: all candidates',f"n={case['sam2_amg']['raw_candidate_count']}")}{media_figure('prep/sam2_amg_filtered_overlay.png','Training AMG filter',f"n={case['sam2_amg']['training_filtered_count']}")}
<div class="wide">{media_figure('prep/fixed_query_grid.png','F04 mask mapped to fixed latent-1 query cells',f"{metrics['query_token_count']} flattened rows")}</div></div></section>
<section id="x0"><h2>Main Student step-0 baseline x0 audit</h2><div class="grid">
{media_figure('source_training_video.mp4','Training sample: 49 frames')}{media_figure('vae_gt_x0.mp4','Wan VAE decoded GT x0')}
{media_figure('vae_training_xt.mp4','Training x_t',f"t={flow['training_timestep']:.0f}; sigma={flow['scheduler_sigma']:.4f}")}{media_figure('vae_predicted_x0.mp4','Baseline x0_pred','x_t - sigma_t * v_pred')}
{media_figure('vae_x0_difference.mp4','Decoded |x0_pred - GT x0| x3')}{media_figure('x0_anchor_contact_sheet.jpg','13 aligned x0 anchor frames','F00/F04/.../F48')}</div></section>
<section id="probe"><h2>Shared-noise Frozen Motion Probe</h2><div class="facts"><div class="fact"><span>Flow loss</span><strong>{flow['weighted_loss']:.6f}</strong></div><div class="fact"><span>KL(student || teacher)</span><strong>{probe['heatmap_kl_student_teacher']:.6f}</strong></div><div class="fact"><span>Trajectory Huber</span><strong>{probe['trajectory_huber']:.6f}</strong></div><div class="fact"><span>Weighted auxiliary</span><strong>{probe['weighted_auxiliary_loss']:.6f}</strong></div><div class="fact"><span>Gradient ||dL/dv_pred||</span><strong>{probe['gradient_to_first_pass_v_pred_norm']:.5f}</strong></div></div><div class="grid">
{media_figure('vae_teacher_probe_input.mp4','GT x0 + shared probe noise',f"level={probe['noise_level']:.2f}; t={probe['timestep']:.0f}")}{media_figure('vae_student_probe_input.mp4','x0_pred + same probe noise',f"seed={probe['shared_noise_seed']}")}
{media_figure('teacher_frame_heatmap_overlay.mp4','Teacher: corresponding GT frame | Top100 | overlay')}{media_figure('student_frame_heatmap_overlay.mp4','Student: corresponding x0_pred frame | Top100 | overlay')}
{media_figure('teacher_top100_heatmap.mp4','Teacher Top100 aggregate heatmap')}{media_figure('student_top100_heatmap.mp4','Student Top100 aggregate heatmap')}
<div class="wide">{media_figure('teacher_student_five_panel.mp4','GT | x0_pred | Teacher overlay | Student overlay | difference')}</div>
<div class="wide">{media_figure('teacher_student_13frame_contact_sheet.jpg','13-frame aligned heatmap contact sheet')}</div>
{media_figure('trajectory_overlay.jpg','Teacher and Student soft-argmax trajectories')}<figure><figcaption><strong>Trajectory numeric values</strong><span>normalized (x,y), 13 latent frames</span></figcaption><a href="trajectory_values.json">Open trajectory_values.json</a></figure></div></section>
{sweep_html}
<section id="candidates"><h2>All prompt-free SAM2 AMG candidates</h2><div class="candidates">{candidate_html}</div></section>
</main></body></html>"""


def index_page(cases: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td><a href='cases/{html.escape(row['case_key'])}/index.html'>{html.escape(row['case_key'])}</a></td>"
        f"<td>{html.escape(row['family'])}</td><td>{row['dataset_index']}</td>"
        f"<td>{html.escape(row['target_phrase'])}</td><td>{row['query_token_count']}</td>"
        f"<td>{html.escape(row['noise_sweep'])}</td>"
        f"<td>{row['heatmap_kl']:.6f}</td><td>{row['trajectory_huber']:.6f}</td>"
        f"<td>{row['gradient_norm']:.5f}</td></tr>"
        for row in cases
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frozen Motion Probe training diagnostics</title><style>
:root{{--bg:#edf0ec;--paper:#fff;--ink:#17201b;--muted:#59665f;--line:#b7c1bb;--green:#176c59}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Noto Sans CJK SC","Source Han Sans SC",sans-serif;letter-spacing:0}}header,main{{width:min(1500px,calc(100% - 28px));margin:auto}}header{{padding:24px 0 14px;border-bottom:2px solid var(--ink)}}h1{{font-size:28px;margin:0}}p{{margin:4px 0;color:var(--muted)}}main{{padding:20px 0 60px}}table{{width:100%;border-collapse:collapse;background:var(--paper)}}th,td{{padding:11px;border:1px solid var(--line);text-align:left}}th{{background:#dfe5e0;font-size:12px}}a{{color:var(--green);font-weight:800;text-decoration:none}}.protocol{{margin-top:18px;padding:12px;border-left:5px solid var(--green);background:var(--paper)}}code{{font-size:12px}}@media(max-width:850px){{main{{overflow:auto}}table{{min-width:1050px}}}}
</style></head><body><header><h1>Frozen Motion Probe training-case diagnostics</h1><p>PyBullet train split · official Wan2.2-TI2V-5B baseline · F04/latent-1 fixed query · latest3350 Top100</p></header><main><table><thead><tr><th>Case</th><th>Family</th><th>Train index</th><th>Identity target</th><th>Query cells</th><th>Noise sweep</th><th>Original KL</th><th>Original Trajectory Huber</th><th>Original ||dL/dv_pred||</th></tr></thead><tbody>{rows}</tbody></table><div class="protocol"><strong>Controlled probe:</strong> Teacher and Student share noise, timestep, text and clean TI2V conditioning. Teacher heatmap uses <code>Q_GT @ K_GT</code>; Student uses detached <code>Q_GT @ K_Student</code>. The displayed Student is the step-0 official baseline, which is output-equivalent to the zero-initialized adapter. Case pages retain the original t=500 / Probe=0.5 result and append the five-stage Probe=0.1/0.2 sweep.</div></main></body></html>"""


def report_index_row(
    case: dict[str, Any],
    metrics: dict[str, Any],
    sweep_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_key": case["case_key"],
        "family": case["family"],
        "dataset_index": case["dataset_index"],
        "target_phrase": case["target_phrase"],
        "query_token_count": metrics["query_token_count"],
        "noise_sweep": (
            f"{len(sweep_metrics['training_stages'])} t x "
            f"{len(sweep_metrics['probe_settings'])} probe"
            if sweep_metrics
            else "pending"
        ),
        "heatmap_kl": metrics["probe"]["heatmap_kl_student_teacher"],
        "trajectory_huber": metrics["probe"]["trajectory_huber"],
        "gradient_norm": metrics["probe"]["gradient_to_first_pass_v_pred_norm"],
    }


def run_render(args: argparse.Namespace) -> None:
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    vae_pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=torch.device(args.device),
        model_configs=[ModelConfig(str(args.wan_root.resolve() / "Wan2.2_VAE.pth"))],
        tokenizer_config=None,
        redirect_common_files=False,
    )
    index_rows = []
    for position, case in enumerate(manifest["cases"], start=1):
        case_key = case["case_key"]
        source_cache = Path(case["cache_dir"])
        case_output = output_root / "cases" / case_key
        metrics = json.loads((case_output / "metrics.json").read_text(encoding="utf-8"))
        sweep_metrics_path = case_output / "noise_sweep" / "metrics.json"
        sweep_metrics = (
            json.loads(sweep_metrics_path.read_text(encoding="utf-8"))
            if sweep_metrics_path.is_file()
            else None
        )
        complete = case_output / "render_complete.json"
        print(f"[render {position}/{len(manifest['cases'])}] {case_key}", flush=True)
        index_href = "../" * (len(Path(case_key).parts) + 1) + "index.html"
        if complete.is_file() and not args.overwrite:
            (case_output / "index.html").write_text(
                case_page(
                    case,
                    metrics,
                    index_href=index_href,
                    sweep_metrics=sweep_metrics,
                ),
                encoding="utf-8",
            )
            index_rows.append(report_index_row(case, metrics, sweep_metrics))
            continue
        bundle = torch.load(case_output / "latents.pt", map_location="cpu", weights_only=True)
        decoded = decode_latent_bundle(
            vae_pipe,
            {
                "target_x0": bundle["target_x0"],
                "training_xt": bundle["training_xt"],
                "predicted_x0": bundle["predicted_x0"],
                "teacher_probe_input": bundle["teacher_probe_input"],
                "student_probe_input": bundle["student_probe_input"],
            },
        )
        with np.load(source_cache / "source_frames.npz") as source_arrays:
            source_frames = source_arrays["frames"].astype(np.uint8)
        media_videos = {
            "source_training_video.mp4": source_frames,
            "vae_gt_x0.mp4": decoded["target_x0"],
            "vae_training_xt.mp4": decoded["training_xt"],
            "vae_predicted_x0.mp4": decoded["predicted_x0"],
            "vae_teacher_probe_input.mp4": decoded["teacher_probe_input"],
            "vae_student_probe_input.mp4": decoded["student_probe_input"],
        }
        for filename, frames in media_videos.items():
            write_video(case_output / filename, frames, args.fps)
        difference = np.clip(
            np.abs(decoded["predicted_x0"].astype(np.float32) - decoded["target_x0"].astype(np.float32)) * 3.0,
            0,
            255,
        ).astype(np.uint8)
        write_video(case_output / "vae_x0_difference.mp4", difference, args.fps)
        anchor_indices = anchor_frame_indices(13, len(source_frames))
        x0_panels = []
        for latent_index, pixel_index in enumerate(anchor_indices):
            x0_panels.append(
                np.concatenate(
                    [
                        add_label(resize_panel(source_frames[pixel_index]), f"source F{pixel_index:02d}"),
                        add_label(resize_panel(decoded["target_x0"][pixel_index]), f"VAE GT L{latent_index:02d}"),
                        add_label(resize_panel(decoded["predicted_x0"][pixel_index]), "x0 pred"),
                        add_label(resize_panel(difference[pixel_index]), "abs diff x3"),
                    ],
                    axis=1,
                )
            )
        x0_sheet = make_contact_sheet(x0_panels, columns=1)
        cv2.imwrite(
            str(case_output / "x0_anchor_contact_sheet.jpg"),
            cv2.cvtColor(x0_sheet, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 91],
        )
        with np.load(case_output / "probe_outputs.npz") as probe_file:
            probe_arrays = {key: probe_file[key] for key in probe_file.files}
        render_heatmap_media(
            case_output,
            decoded,
            probe_arrays,
            float(args.heatmap_fps),
        )
        teacher_trajectory = probe_arrays["teacher_trajectory"][0]
        student_trajectory = probe_arrays["student_trajectory"][0]
        trajectory_image = trajectory_visualization(
            source_frames[int(args.query_pixel_frame)],
            teacher_trajectory,
            student_trajectory,
        )
        cv2.imwrite(
            str(case_output / "trajectory_overlay.jpg"),
            cv2.cvtColor(trajectory_image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 93],
        )
        atomic_json(
            case_output / "trajectory_values.json",
            {
                "coordinate_order": ["x", "y"],
                "coordinate_range": [0.0, 1.0],
                "teacher": teacher_trajectory,
                "student": student_trajectory,
            },
        )
        prep_link = case_output / "prep"
        if prep_link.is_symlink() or prep_link.exists():
            if prep_link.is_symlink() and prep_link.resolve() == source_cache.resolve():
                pass
            else:
                raise RuntimeError(f"unexpected existing prep path: {prep_link}")
        else:
            prep_link.symlink_to(source_cache, target_is_directory=True)
        (case_output / "index.html").write_text(
            case_page(
                case,
                metrics,
                index_href=index_href,
                sweep_metrics=sweep_metrics,
            ),
            encoding="utf-8",
        )
        atomic_json(complete, {"case_key": case_key, "state": "complete"})
        index_rows.append(report_index_row(case, metrics, sweep_metrics))
        del bundle, decoded
        gc.collect()
        torch.cuda.empty_cache()
    (output_root / "index.html").write_text(index_page(index_rows), encoding="utf-8")
    atomic_json(output_root / "report_manifest.json", {"schema_version": 1, "cases": index_rows})
    atomic_json(output_root / "status.json", {"state": "complete", "case_count": len(index_rows)})


def run_sweep_render(args: argparse.Namespace) -> None:
    config = noise_sweep_config(args)
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    vae_pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=torch.device(args.device),
        model_configs=[ModelConfig(str(args.wan_root.resolve() / "Wan2.2_VAE.pth"))],
        tokenizer_config=None,
        redirect_common_files=False,
    )
    index_rows = []
    for position, case in enumerate(manifest["cases"], start=1):
        case_key = case["case_key"]
        case_output = output_root / "cases" / case_key
        case_sweep = case_output / "noise_sweep"
        sweep_metrics_path = case_sweep / "metrics.json"
        if not sweep_metrics_path.is_file():
            raise FileNotFoundError(f"run sweep-forward first: {sweep_metrics_path}")
        metrics = json.loads((case_output / "metrics.json").read_text(encoding="utf-8"))
        sweep_metrics = json.loads(sweep_metrics_path.read_text(encoding="utf-8"))
        complete = case_sweep / "render_complete.json"
        index_href = "../" * (len(Path(case_key).parts) + 1) + "index.html"
        print(f"[sweep-render {position}/{len(manifest['cases'])}] {case_key}", flush=True)
        if complete.is_file() and not args.overwrite:
            (case_output / "index.html").write_text(
                case_page(
                    case,
                    metrics,
                    index_href=index_href,
                    sweep_metrics=sweep_metrics,
                ),
                encoding="utf-8",
            )
            index_rows.append(report_index_row(case, metrics, sweep_metrics))
            continue

        with np.load(Path(case["cache_dir"]) / "source_frames.npz") as source_arrays:
            source_frames = source_arrays["frames"].astype(np.uint8)
        base_bundle = torch.load(
            case_output / "latents.pt", map_location="cpu", weights_only=True
        )
        target_decoded = decode_latent_bundle(
            vae_pipe, {"target_x0": base_bundle["target_x0"]}
        )["target_x0"]
        del base_bundle

        teacher_heatmaps: dict[str, np.ndarray] = {}
        teacher_decoded: dict[str, np.ndarray] = {}
        for probe in config["probe_settings"]:
            probe_root = case_sweep / "probes" / probe["id"]
            teacher_bundle = torch.load(
                probe_root / "teacher.pt", map_location="cpu", weights_only=True
            )
            teacher_heatmaps[probe["id"]] = (
                teacher_bundle["teacher_heatmap"].float().numpy()
            )
            teacher_decoded[probe["id"]] = decode_latent_bundle(
                vae_pipe,
                {"teacher_probe_input": teacher_bundle["teacher_probe_input"]},
            )["teacher_probe_input"]
            write_video(
                probe_root / "vae_teacher_probe_input.mp4",
                teacher_decoded[probe["id"]],
                float(args.fps),
            )
            atomic_json(
                probe_root / "render_complete.json",
                {"state": "complete", "id": probe["id"]},
            )
            del teacher_bundle

        for stage_position, stage in enumerate(config["training_stages"], start=1):
            print(
                f"  [stage {stage_position}/{len(config['training_stages'])}] "
                f"t={stage['timestep']:.0f}",
                flush=True,
            )
            stage_root = case_sweep / "stages" / stage["id"]
            stage_bundle = torch.load(
                stage_root / "latents.pt", map_location="cpu", weights_only=True
            )
            stage_decoded = decode_latent_bundle(
                vae_pipe,
                {
                    "training_xt": stage_bundle["training_xt"],
                    "predicted_x0": stage_bundle["predicted_x0"],
                },
            )
            write_video(
                stage_root / "vae_training_xt.mp4",
                stage_decoded["training_xt"],
                float(args.fps),
            )
            write_video(
                stage_root / "vae_predicted_x0.mp4",
                stage_decoded["predicted_x0"],
                float(args.fps),
            )
            difference = np.clip(
                np.abs(
                    stage_decoded["predicted_x0"].astype(np.float32)
                    - target_decoded.astype(np.float32)
                )
                * 3.0,
                0,
                255,
            ).astype(np.uint8)
            write_video(
                stage_root / "vae_x0_difference.mp4", difference, float(args.fps)
            )
            anchors = anchor_frame_indices(13, len(source_frames))
            x0_panels = []
            for latent_index, pixel_index in enumerate(anchors):
                x0_panels.append(
                    np.concatenate(
                        [
                            add_label(
                                resize_panel(target_decoded[pixel_index]),
                                f"VAE GT L{latent_index:02d}",
                            ),
                            add_label(
                                resize_panel(stage_decoded["training_xt"][pixel_index]),
                                f"x_t F{pixel_index:02d}",
                            ),
                            add_label(
                                resize_panel(stage_decoded["predicted_x0"][pixel_index]),
                                "x0 pred",
                            ),
                            add_label(resize_panel(difference[pixel_index]), "abs diff x3"),
                        ],
                        axis=1,
                    )
                )
            x0_sheet = make_contact_sheet(x0_panels, columns=1)
            cv2.imwrite(
                str(stage_root / "x0_anchor_contact_sheet.jpg"),
                cv2.cvtColor(x0_sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 91],
            )
            atomic_json(
                stage_root / "render_complete.json",
                {"state": "complete", "id": stage["id"]},
            )

            for probe in config["probe_settings"]:
                comparison_root = (
                    case_sweep / "comparisons" / stage["id"] / probe["id"]
                )
                comparison_bundle = torch.load(
                    comparison_root / "latents.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                student_decoded = decode_latent_bundle(
                    vae_pipe,
                    {
                        "student_probe_input": comparison_bundle[
                            "student_probe_input"
                        ]
                    },
                )["student_probe_input"]
                write_video(
                    comparison_root / "vae_student_probe_input.mp4",
                    student_decoded,
                    float(args.fps),
                )
                with np.load(comparison_root / "probe_outputs.npz") as probe_file:
                    probe_arrays = {key: probe_file[key] for key in probe_file.files}
                probe_arrays["teacher_heatmap"] = teacher_heatmaps[probe["id"]]
                render_heatmap_media(
                    comparison_root,
                    {
                        "target_x0": target_decoded,
                        "predicted_x0": stage_decoded["predicted_x0"],
                    },
                    probe_arrays,
                    float(args.heatmap_fps),
                )
                teacher_trajectory = probe_arrays["teacher_trajectory"][0]
                student_trajectory = probe_arrays["student_trajectory"][0]
                trajectory_image = trajectory_visualization(
                    source_frames[int(args.query_pixel_frame)],
                    teacher_trajectory,
                    student_trajectory,
                )
                cv2.imwrite(
                    str(comparison_root / "trajectory_overlay.jpg"),
                    cv2.cvtColor(trajectory_image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 93],
                )
                atomic_json(
                    comparison_root / "trajectory_values.json",
                    {
                        "coordinate_order": ["x", "y"],
                        "coordinate_range": [0.0, 1.0],
                        "teacher": teacher_trajectory,
                        "student": student_trajectory,
                    },
                )
                atomic_json(
                    comparison_root / "render_complete.json",
                    {
                        "state": "complete",
                        "training_stage_id": stage["id"],
                        "probe_setting_id": probe["id"],
                    },
                )
                del comparison_bundle, student_decoded, probe_arrays
                gc.collect()
                torch.cuda.empty_cache()
            del stage_bundle, stage_decoded, difference, x0_panels, x0_sheet
            gc.collect()
            torch.cuda.empty_cache()

        (case_output / "index.html").write_text(
            case_page(
                case,
                metrics,
                index_href=index_href,
                sweep_metrics=sweep_metrics,
            ),
            encoding="utf-8",
        )
        atomic_json(complete, {"state": "complete", "case_key": case_key})
        index_rows.append(report_index_row(case, metrics, sweep_metrics))
        del source_frames, target_decoded, teacher_heatmaps, teacher_decoded
        gc.collect()
        torch.cuda.empty_cache()
    (output_root / "index.html").write_text(index_page(index_rows), encoding="utf-8")
    atomic_json(
        output_root / "report_manifest.json",
        {"schema_version": 2, "cases": index_rows},
    )
    atomic_json(
        output_root / "noise_sweep_status.json",
        {"state": "complete", "case_count": len(index_rows)},
    )


def main() -> None:
    args = parse_args()
    check_common_args(args)
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_root = args.cache_root.expanduser().resolve()
    if args.mode == "prepare":
        run_prepare(args)
    elif args.mode == "forward":
        run_forward(args)
    elif args.mode == "render":
        run_render(args)
    elif args.mode == "sweep-forward":
        run_sweep_forward(args)
    else:
        run_sweep_render(args)


if __name__ == "__main__":
    main()
