#!/usr/bin/env python3
"""Controlled xSSC slot-sensitivity comparison on the 0526dp ball/block videos.

The comparison deliberately keeps video preprocessing and first-frame box
conditioning identical between the official DINOv2 model and the DINOv3 model.
Embeddings with different channel dimensions are never compared directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import torch


ROOT = Path(__file__).resolve().parent
TRAIN_XSSC_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, str(TRAIN_XSSC_ROOT))
sys.path.insert(0, "/home/gaoya/Grounded-SAM-2-main")

from build_movi_c_amg_gtbox_xssc_slider import masks_to_boxes  # noqa: E402
from visualize_movi_c_sam2_amg import (  # noqa: E402
    DEFAULT_CHECKPOINT as DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_CONFIG as DEFAULT_SAM2_CONFIG,
    resolve_sam2_config_name,
    select_xssc_candidates,
)


DEFAULT_DATA_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/xssc_0526dp_slot_sensitivity"
)
DEFAULT_OFFICIAL_CONFIG = (
    ROOT / "upstream/config-randsfq/rsfq2_c-movi_c.py"
)
DEFAULT_OFFICIAL_CHECKPOINT = Path(
    "/data/gaoya/agent-data/weights/xssc_official_archive_rsfq2/"
    "rsfq2_c-movi_c/42-0035.pth"
)
DEFAULT_DINOV3_CONFIG = (
    ROOT
    / "upstream/config-randsfq/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
)
DEFAULT_DINOV3_CHECKPOINT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/"
    "dinov3_xSSC/restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/"
    "42/step-036000.pth"
)
REFERENCE_CASE = "jepa_sensitivity/mass_100"
NUM_SLOTS = 11
GRID_SIZE = 16
IMAGE_SIZE = 256
IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(
    1, 1, 3, 1, 1
)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(
    1, 1, 3, 1, 1
)
SLOT_COLORS = np.asarray(
    [
        [239, 68, 68],
        [59, 130, 246],
        [34, 197, 94],
        [250, 204, 21],
        [168, 85, 247],
        [6, 182, 212],
        [249, 115, 22],
        [236, 72, 153],
        [132, 204, 22],
        [20, 184, 166],
        [148, 163, 184],
    ],
    dtype=np.uint8,
)
MODEL_SPECS = {
    "official_dinov2": {
        "label": "Official DINOv2 MOVi-C 42-0035",
        "config": DEFAULT_OFFICIAL_CONFIG,
        "checkpoint": DEFAULT_OFFICIAL_CHECKPOINT,
    },
    "dinov3_step036000": {
        "label": "DINOv3 MOVi-C step-036000",
        "config": DEFAULT_DINOV3_CONFIG,
        "checkpoint": DEFAULT_DINOV3_CHECKPOINT,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("boxes", "extract", "report", "all"), default="all"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 uses all frames; a positive value samples that many uniformly.",
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sam2-config", type=Path, default=DEFAULT_SAM2_CONFIG)
    parser.add_argument(
        "--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT
    )
    parser.add_argument("--max-selected", type=int, default=11)
    parser.add_argument("--min-area-ratio", type=float, default=0.004)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--min-bbox-side", type=float, default=7.0)
    parser.add_argument("--background-area-ratio", type=float, default=0.06)
    parser.add_argument("--background-span-ratio", type=float, default=0.75)
    parser.add_argument("--border-area-ratio", type=float, default=0.025)
    parser.add_argument("--border-occupancy-ratio", type=float, default=0.18)
    parser.add_argument("--opposite-edge-area-ratio", type=float, default=0.04)
    parser.add_argument("--shadow-min-area-ratio", type=float, default=0.03)
    parser.add_argument("--shadow-max-luminance-ratio", type=float, default=0.55)
    parser.add_argument(
        "--shadow-max-chromaticity-distance", type=float, default=0.10
    )
    parser.add_argument("--shadow-max-gradient-mean", type=float, default=20.0)
    parser.add_argument("--duplicate-iou", type=float, default=0.70)
    parser.add_argument("--duplicate-containment", type=float, default=0.85)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def case_key(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root).with_suffix("").as_posix()


def safe_key(key: str) -> str:
    return key.replace("/", "__")


def discover_cases(data_root: Path, max_cases: int = 0) -> list[dict[str, Any]]:
    roots = [
        data_root / "jepa_sensitivity",
        data_root / "ball_block",
        data_root / "ball_block_appearance",
    ]
    cases = []
    for root in roots:
        for video in sorted(root.glob("*.mp4")):
            key = case_key(video, data_root)
            cases.append(
                {
                    "key": key,
                    "stem": video.stem,
                    "source": root.name,
                    "video": video,
                    "json": video.with_suffix(".json"),
                    "md5": file_md5(video),
                }
            )
    if max_cases > 0:
        keep = {REFERENCE_CASE}
        keep.update(item["key"] for item in cases[:max_cases])
        cases = [item for item in cases if item["key"] in keep]
    if not cases:
        raise FileNotFoundError(f"No MP4 files found below {data_root}")
    return cases


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_indices(frame_count: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or max_frames >= frame_count:
        return np.arange(frame_count, dtype=np.int64)
    return np.unique(
        np.linspace(0, frame_count - 1, max_frames).round().astype(np.int64)
    )


def decode_video(path: Path, max_frames: int = 0) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    raw = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not raw:
        raise RuntimeError(f"Could not decode {path}")
    indices = frame_indices(len(raw), max_frames)
    frames = np.stack([raw[index] for index in indices])
    height, width = frames.shape[1:3]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    frames = frames[:, y0 : y0 + side, x0 : x0 + side]
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                )
            )
            for frame in frames
        ]
    ).astype(np.uint8)
    return resized, indices


def normalize_video(frames: np.ndarray) -> torch.Tensor:
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()[None]
    return (video - IMAGENET_MEAN) / IMAGENET_STD


def build_sam2_generator(args: argparse.Namespace, device: torch.device):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(
        resolve_sam2_config_name(args.sam2_config),
        str(args.sam2_checkpoint.resolve()),
        device=str(device),
        mode="eval",
    )
    return SAM2AutomaticMaskGenerator(model)


def annotation_box_xyxy(annotation: dict[str, Any]) -> np.ndarray:
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def choose_canonical_objects(
    selected: list[dict[str, Any]], image_shape: tuple[int, int]
) -> list[dict[str, Any]]:
    """Choose the ball and block without using either xSSC model.

    The target locations only disambiguate duplicate/subpart AMG proposals in
    this fixed-camera dataset. They are not inferred separately per case.
    """
    height, width = image_shape
    targets = {
        "ball": np.asarray([0.14, 0.57], dtype=np.float32),
        "block": np.asarray([0.61, 0.60], dtype=np.float32),
    }
    candidates = []
    for annotation in selected:
        box = annotation_box_xyxy(annotation)
        center = np.asarray(
            [(box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)]
        )
        area = float(annotation["area"]) / float(height * width)
        if 0.003 <= area <= 0.16:
            candidates.append((annotation, center, area))
    if len(candidates) < 2:
        raise RuntimeError(
            f"AMG filtering left only {len(candidates)} object candidates"
        )
    costs = np.asarray(
        [
            [
                np.linalg.norm(center - targets[name])
                + 0.05 * abs(math.log(max(area, 1.0e-6) / 0.035))
                for annotation, center, area in candidates
            ]
            for name in ("ball", "block")
        ]
    )
    rows, cols = linear_sum_assignment(costs)
    chosen = [None, None]
    for row, col in zip(rows, cols):
        if costs[row, col] > 0.35:
            raise RuntimeError(
                f"AMG candidate for {('ball', 'block')[row]} is implausible: "
                f"cost={costs[row, col]:.3f}"
            )
        chosen[row] = candidates[col][0]
    return chosen


def draw_box_preview(
    image: np.ndarray, objects: list[dict[str, Any]], output: Path
) -> None:
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for name, annotation, color in zip(
        ("ball", "block"), objects, ("#ef4444", "#3b82f6")
    ):
        box = annotation_box_xyxy(annotation).tolist()
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 3, max(1, box[1] - 14)), name, fill=color)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_canonical_boxes(
    args: argparse.Namespace, cases: list[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    output = args.output_dir / "canonical_boxes.json"
    if output.is_file() and not args.force:
        return json.loads(output.read_text())
    reference = next(item for item in cases if item["key"] == REFERENCE_CASE)
    frames, _ = decode_video(reference["video"], max_frames=1)
    generator = build_sam2_generator(args, device)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        raw = generator.generate(frames[0])
    selected = select_xssc_candidates(
        raw,
        frames.shape[1] * frames.shape[2],
        args,
        image=frames[0],
    )
    objects = choose_canonical_objects(selected, frames.shape[1:3])
    masks = np.stack(
        [annotation["segmentation"].astype(bool) for annotation in objects]
    )
    boxes = masks_to_boxes(masks, NUM_SLOTS, 1)[0, 0].tolist()
    payload = {
        "reference_case": REFERENCE_CASE,
        "preprocessing": "center crop square, bilinear resize to 256x256",
        "slot_order": ["ball", "block"] + [f"unused_{i}" for i in range(2, 11)],
        "boxes_xyxy_normalized": boxes,
        "raw_amg_count": len(raw),
        "filtered_amg_count": len(selected),
        "sam2_config": str(args.sam2_config.resolve()),
        "sam2_checkpoint": str(args.sam2_checkpoint.resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    draw_box_preview(
        frames[0], objects, args.output_dir / "assets/canonical_boxes.png"
    )
    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def boxes_for_frames(box_metadata: dict[str, Any], count: int) -> torch.Tensor:
    base = np.asarray(
        box_metadata["boxes_xyxy_normalized"], dtype=np.float32
    )
    return torch.from_numpy(np.tile(base[None, None], (1, count, 1, 1)))


def load_checkpoint(model, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("m.encode_backbone.")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint}: "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )


def build_model(spec: dict[str, Any], device: torch.device):
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(spec["config"])
    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    model.freez(cfg.freez, verbose=False)
    load_checkpoint(model, Path(spec["checkpoint"]))
    return model.to(device).eval()


def semantic_slot_ids(
    attention: np.ndarray, boxes: torch.Tensor
) -> tuple[int, int]:
    """Match decoder slots to ball/block boxes using frame-0 attention."""
    frame_attention = attention[0]
    scores = np.zeros((2, frame_attention.shape[0]), dtype=np.float32)
    for object_id in range(2):
        x1, y1, x2, y2 = boxes[0, 0, object_id].numpy()
        gx1 = int(np.floor(x1 * GRID_SIZE))
        gy1 = int(np.floor(y1 * GRID_SIZE))
        gx2 = max(gx1 + 1, int(np.ceil(x2 * GRID_SIZE)))
        gy2 = max(gy1 + 1, int(np.ceil(y2 * GRID_SIZE)))
        region = frame_attention[
            :, max(0, gy1) : min(GRID_SIZE, gy2), max(0, gx1) : min(GRID_SIZE, gx2)
        ]
        scores[object_id] = region.mean(axis=(1, 2))
    rows, cols = linear_sum_assignment(-scores)
    result = [-1, -1]
    for row, col in zip(rows, cols):
        result[row] = int(col)
    return result[0], result[1]


def infer_case(
    model,
    frames: np.ndarray,
    boxes: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    video = normalize_video(frames)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        output = model(
            batch={
                "video": video.to(device, non_blocking=True),
                "bbox": boxes.to(device, non_blocking=True),
            }
        )
    slots = output["slotz"][0].detach().float().cpu().numpy()
    attention = output["attentd"][0].detach().float().cpu().numpy()
    if slots.ndim != 3 or attention.shape[1:] != (
        NUM_SLOTS,
        GRID_SIZE,
        GRID_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected output: slotz={slots.shape}, attentd={attention.shape}"
        )
    ball_slot, block_slot = semantic_slot_ids(attention, boxes)
    return {
        "slotz": slots.astype(np.float32),
        "labels": attention.argmax(axis=1).astype(np.uint8),
        "semantic_slots": np.asarray([ball_slot, block_slot], dtype=np.int64),
    }


def extract_all(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    box_metadata: dict[str, Any],
    device: torch.device,
) -> None:
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.amp_dtype]
    for model_name, spec in MODEL_SPECS.items():
        print(f"[model] loading {model_name}: {spec['checkpoint']}", flush=True)
        model = build_model(spec, device)
        for position, case in enumerate(cases, start=1):
            destination = (
                args.output_dir
                / "embeddings"
                / model_name
                / f"{safe_key(case['key'])}.npz"
            )
            if destination.is_file() and not args.force:
                print(
                    f"[{model_name}] {position}/{len(cases)} cached {case['key']}",
                    flush=True,
                )
                continue
            frames, indices = decode_video(case["video"], args.max_frames)
            boxes = boxes_for_frames(box_metadata, len(frames))
            result = infer_case(model, frames, boxes, device, amp_dtype)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                slotz=result["slotz"],
                labels=result["labels"],
                semantic_slots=result["semantic_slots"],
                frame_indices=indices,
                source_video=str(case["video"]),
                md5=case["md5"],
            )
            print(
                f"[{model_name}] {position}/{len(cases)} {case['key']} "
                f"slotz={result['slotz'].shape} ids={result['semantic_slots'].tolist()}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def l2_normalize(array: np.ndarray) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def cosine_distance_curve(
    reference: np.ndarray, current: np.ndarray
) -> np.ndarray:
    count = min(len(reference), len(current))
    a = l2_normalize(reference[:count])
    b = l2_normalize(current[:count])
    return 1.0 - np.sum(a * b, axis=-1)


def all_slot_hungarian_curve(
    reference: np.ndarray, current: np.ndarray
) -> np.ndarray:
    count = min(len(reference), len(current))
    output = []
    for frame_id in range(count):
        a = l2_normalize(reference[frame_id])
        b = l2_normalize(current[frame_id])
        distance = 1.0 - a @ b.T
        rows, cols = linear_sum_assignment(distance)
        output.append(float(distance[rows, cols].mean()))
    return np.asarray(output, dtype=np.float32)


def motion_signal(video: Path, max_frames: int) -> np.ndarray:
    frames, _ = decode_video(video, max_frames)
    gray = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    ).astype(np.float32)
    signal = np.zeros(len(gray), dtype=np.float32)
    signal[1:] = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
    scale = np.std(signal)
    return (signal - np.mean(signal)) / max(float(scale), 1e-6)


def dtw_path(reference: np.ndarray, current: np.ndarray) -> list[tuple[int, int]]:
    n, m = len(reference), len(current)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    parent = np.zeros((n + 1, m + 1), dtype=np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            move = int(np.argmin(choices))
            cost[i, j] = abs(float(reference[i - 1] - current[j - 1])) + choices[move]
            parent[i, j] = move
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = parent[i, j]
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return path[::-1]


def aligned_cosine_distance(
    reference: np.ndarray,
    current: np.ndarray,
    path: list[tuple[int, int]],
) -> float:
    a = l2_normalize(reference)
    b = l2_normalize(current)
    values = [1.0 - float(np.dot(a[i], b[j])) for i, j in path]
    return float(np.mean(values))


def group_definitions(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {item["key"] for item in cases}

    def available(items: list[str]) -> list[str]:
        return [item for item in items if item in keys]

    groups = [
        {
            "name": "JEPA ball mass",
            "baseline": "jepa_sensitivity/mass_100",
            "cases": available(
                [
                    "jepa_sensitivity/mass_001",
                    "jepa_sensitivity/mass_005",
                    "jepa_sensitivity/mass_010",
                    "jepa_sensitivity/mass_100",
                    "jepa_sensitivity/mass_500",
                    "jepa_sensitivity/mass_2000",
                    "jepa_sensitivity/mass_9999",
                ]
            ),
        },
        {
            "name": "JEPA block mass",
            "baseline": "jepa_sensitivity/mass_100",
            "cases": available(
                [
                    "jepa_sensitivity/blk_005",
                    "jepa_sensitivity/mass_100",
                    "jepa_sensitivity/blk_500",
                    "jepa_sensitivity/blk_2000",
                ]
            ),
        },
        {
            "name": "JEPA gravity",
            "baseline": "jepa_sensitivity/grav_098",
            "cases": available(
                [
                    "jepa_sensitivity/grav_050",
                    "jepa_sensitivity/grav_098",
                    "jepa_sensitivity/grav_200",
                ]
            ),
        },
        {
            "name": "JEPA initial velocity",
            "baseline": "jepa_sensitivity/vel_035",
            "cases": available(
                [
                    "jepa_sensitivity/vel_005",
                    "jepa_sensitivity/vel_015",
                    "jepa_sensitivity/vel_035",
                    "jepa_sensitivity/vel_070",
                    "jepa_sensitivity/vel_140",
                ]
            ),
        },
        {
            "name": "Ball-block restitution",
            "baseline": "ball_block/e07_mu05_m1",
            "cases": available(
                [
                    "ball_block/e03_mu05_m1",
                    "ball_block/e05_mu05_m1",
                    "ball_block/e07_mu05_m1",
                    "ball_block/e09_mu05_m1",
                ]
            ),
        },
        {
            "name": "Ball-block friction",
            "baseline": "ball_block/e07_mu05_m1",
            "cases": available(
                [
                    "ball_block/e07_mu01_m1",
                    "ball_block/e07_mu05_m1",
                    "ball_block/e07_mu10_m1",
                ]
            ),
        },
        {
            "name": "Ball-block mass",
            "baseline": "ball_block/e07_mu05_m1",
            "cases": available(
                [
                    "ball_block/e07_mu05_m01",
                    "ball_block/e07_mu05_m1",
                    "ball_block/e07_mu05_m5",
                ]
            ),
        },
    ]
    appearance = {}
    for case in cases:
        if case["source"] != "ball_block_appearance":
            continue
        base = re.sub(r"_v[123]_(default|dark_blue|warm_bright)$", "", case["stem"])
        appearance.setdefault(base, []).append(case["key"])
    for base, items in sorted(appearance.items()):
        baseline = f"ball_block_appearance/{base}_v1_default"
        groups.append(
            {
                "name": f"Appearance {base}",
                "baseline": baseline,
                "cases": sorted(items),
                "family": "appearance",
            }
        )
    return [
        group
        for group in groups
        if group["baseline"] in keys and len(group["cases"]) >= 2
    ]


def load_embedding(
    output_dir: Path, model_name: str, key: str
) -> dict[str, np.ndarray]:
    path = output_dir / "embeddings" / model_name / f"{safe_key(key)}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as item:
        return {name: item[name] for name in item.files}


def descriptor(item: dict[str, np.ndarray]) -> np.ndarray:
    slots = item["slotz"]
    ids = item["semantic_slots"].astype(int)
    trajectories = [l2_normalize(slots[:, slot_id]) for slot_id in ids]
    return np.concatenate([track.mean(axis=0) for track in trajectories])


def cosine_rdm(features: np.ndarray) -> np.ndarray:
    features = l2_normalize(features)
    return 1.0 - features @ features.T


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(
        y.T @ y, ord="fro"
    )
    return float(cross / max(float(denom), 1e-12))


def overlay_labels(
    frame: np.ndarray, labels: np.ndarray, semantic_slots: np.ndarray
) -> np.ndarray:
    semantic_slots = semantic_slots.astype(int)
    remap = np.full(NUM_SLOTS, 10, dtype=np.uint8)
    remap[semantic_slots[0]] = 0
    remap[semantic_slots[1]] = 1
    next_color = 2
    for slot_id in range(NUM_SLOTS):
        if slot_id not in semantic_slots:
            remap[slot_id] = next_color
            next_color = min(next_color + 1, 10)
    semantic_labels = remap[labels]
    full = semantic_labels.repeat(IMAGE_SIZE // GRID_SIZE, axis=0).repeat(
        IMAGE_SIZE // GRID_SIZE, axis=1
    )
    colors = SLOT_COLORS[full % len(SLOT_COLORS)]
    return np.clip(
        frame.astype(np.float32) * 0.48 + colors.astype(np.float32) * 0.52,
        0,
        255,
    ).astype(np.uint8)


def save_case_contact_sheet(
    case: dict[str, Any],
    outputs: dict[str, dict[str, np.ndarray]],
    output: Path,
    max_frames: int,
) -> None:
    frames, _ = decode_video(case["video"], max_frames)
    positions = np.linspace(0, len(frames) - 1, 3).round().astype(int)
    canvas = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE * 3), "#111827")
    names = ["Input", "Official DINOv2", "DINOv3 step-036000"]
    draw = ImageDraw.Draw(canvas)
    for row, position in enumerate(positions):
        images = [frames[position]]
        images.extend(
            overlay_labels(
                frames[position],
                outputs[name]["labels"][position],
                outputs[name]["semantic_slots"],
            )
            for name in MODEL_SPECS
        )
        for column, image in enumerate(images):
            canvas.paste(Image.fromarray(image), (column * IMAGE_SIZE, row * IMAGE_SIZE))
            if row == 0:
                draw.rectangle(
                    (
                        column * IMAGE_SIZE,
                        0,
                        (column + 1) * IMAGE_SIZE,
                        22,
                    ),
                    fill="#111827",
                )
                draw.text(
                    (column * IMAGE_SIZE + 6, 5), names[column], fill="#f8fafc"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def save_case_curve(
    case_key_value: str,
    baseline_key: str,
    model_metrics: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for axis, object_name in zip(axes, ("ball", "block")):
        for model_name, color in zip(MODEL_SPECS, ("#2563eb", "#dc2626")):
            values = model_metrics[model_name][f"{object_name}_curve"]
            axis.plot(values, color=color, linewidth=1.6, label=MODEL_SPECS[model_name]["label"])
        axis.set_title(f"{object_name.capitalize()} slot")
        axis.set_xlabel("Sampled frame")
        axis.set_ylabel("Cosine distance to baseline")
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{case_key_value} vs {baseline_key}", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_rdm_figure(
    keys: list[str], rdms: dict[str, np.ndarray], output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    vmax = max(float(matrix.max()) for matrix in rdms.values())
    for axis, (model_name, matrix) in zip(axes, rdms.items()):
        image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=max(vmax, 1e-6))
        axis.set_title(MODEL_SPECS[model_name]["label"])
        axis.set_xticks(range(len(keys)), [Path(key).name for key in keys], rotation=90, fontsize=6)
        axis.set_yticks(range(len(keys)), [Path(key).name for key in keys], fontsize=6)
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle("Case-level object-slot representational distance")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def nearest_baseline_group(
    key: str, groups: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for group in groups:
        if key in group["cases"]:
            return group
    return None


def compute_report(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    groups = group_definitions(cases)
    case_by_key = {item["key"]: item for item in cases}
    assets = args.output_dir / "assets"
    metric_rows: list[dict[str, Any]] = []
    case_payload: dict[str, Any] = {}
    motion_cache: dict[str, np.ndarray] = {}

    def motion(key: str) -> np.ndarray:
        if key not in motion_cache:
            motion_cache[key] = motion_signal(
                case_by_key[key]["video"], args.max_frames
            )
        return motion_cache[key]

    for case in cases:
        key = case["key"]
        media = args.output_dir / "media" / f"{safe_key(key)}.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        if not media.exists():
            media.symlink_to(case["video"].resolve())
        group = nearest_baseline_group(key, groups)
        baseline_key = group["baseline"] if group else key
        model_metrics = {}
        outputs = {}
        path = dtw_path(motion(baseline_key), motion(key))
        for model_name in MODEL_SPECS:
            current = load_embedding(args.output_dir, model_name, key)
            baseline = load_embedding(args.output_dir, model_name, baseline_key)
            outputs[model_name] = current
            current_ids = current["semantic_slots"].astype(int)
            baseline_ids = baseline["semantic_slots"].astype(int)
            ball_curve = cosine_distance_curve(
                baseline["slotz"][:, baseline_ids[0]],
                current["slotz"][:, current_ids[0]],
            )
            block_curve = cosine_distance_curve(
                baseline["slotz"][:, baseline_ids[1]],
                current["slotz"][:, current_ids[1]],
            )
            all_curve = all_slot_hungarian_curve(
                baseline["slotz"], current["slotz"]
            )
            ball_dtw = aligned_cosine_distance(
                baseline["slotz"][:, baseline_ids[0]],
                current["slotz"][:, current_ids[0]],
                path,
            )
            block_dtw = aligned_cosine_distance(
                baseline["slotz"][:, baseline_ids[1]],
                current["slotz"][:, current_ids[1]],
                path,
            )
            model_metrics[model_name] = {
                "ball_curve": ball_curve,
                "block_curve": block_curve,
                "all_curve": all_curve,
                "ball_raw_mean": float(ball_curve.mean()),
                "block_raw_mean": float(block_curve.mean()),
                "all_raw_mean": float(all_curve.mean()),
                "ball_dtw_mean": ball_dtw,
                "block_dtw_mean": block_dtw,
                "semantic_slots": current_ids.tolist(),
                "slot_dim": int(current["slotz"].shape[-1]),
            }
            metric_rows.append(
                {
                    "case": key,
                    "group": group["name"] if group else "ungrouped",
                    "baseline": baseline_key,
                    "model": model_name,
                    "ball_raw_cosine_distance": float(ball_curve.mean()),
                    "block_raw_cosine_distance": float(block_curve.mean()),
                    "all_slot_hungarian_distance": float(all_curve.mean()),
                    "ball_motion_dtw_distance": ball_dtw,
                    "block_motion_dtw_distance": block_dtw,
                    "ball_slot_id": int(current_ids[0]),
                    "block_slot_id": int(current_ids[1]),
                }
            )
        contact = assets / "cases" / f"{safe_key(key)}_contact.jpg"
        curve = assets / "cases" / f"{safe_key(key)}_curve.png"
        save_case_contact_sheet(case, outputs, contact, args.max_frames)
        save_case_curve(key, baseline_key, model_metrics, curve)
        case_payload[key] = {
            "source_video": media.relative_to(args.output_dir).as_posix(),
            "baseline": baseline_key,
            "group": group["name"] if group else "ungrouped",
            "contact": contact.relative_to(args.output_dir).as_posix(),
            "curve": curve.relative_to(args.output_dir).as_posix(),
            "metrics": {
                model_name: {
                    metric: value
                    for metric, value in values.items()
                    if not isinstance(value, np.ndarray)
                }
                for model_name, values in model_metrics.items()
            },
        }

    strict_excluded = {
        "jepa_sensitivity/nomiss",
        "jepa_sensitivity/rev_035",
    }
    common_keys = sorted(set(case_by_key) - strict_excluded)
    features = {
        model_name: np.stack(
            [
                descriptor(load_embedding(args.output_dir, model_name, key))
                for key in common_keys
            ]
        )
        for model_name in MODEL_SPECS
    }
    rdms = {name: cosine_rdm(value) for name, value in features.items()}
    upper = np.triu_indices(len(common_keys), k=1)
    rdm_spearman = float(
        spearmanr(
            rdms["official_dinov2"][upper],
            rdms["dinov3_step036000"][upper],
        ).statistic
    )
    observations = {}
    for model_name in MODEL_SPECS:
        rows = []
        for key in common_keys:
            item = load_embedding(args.output_dir, model_name, key)
            for slot_id in item["semantic_slots"].astype(int):
                rows.append(l2_normalize(item["slotz"][:, slot_id]))
        observations[model_name] = np.concatenate(rows, axis=0)
    cka = linear_cka(
        observations["official_dinov2"], observations["dinov3_step036000"]
    )
    save_rdm_figure(common_keys, rdms, assets / "summary_rdm.png")

    duplicates = {}
    md5_groups = {}
    for case in cases:
        md5_groups.setdefault(case["md5"], []).append(case["key"])
    for md5, keys in md5_groups.items():
        if len(keys) < 2:
            continue
        duplicates[md5] = {"cases": keys, "models": {}}
        for model_name in MODEL_SPECS:
            base = load_embedding(args.output_dir, model_name, keys[0])
            base_ids = base["semantic_slots"].astype(int)
            values = []
            for key in keys[1:]:
                item = load_embedding(args.output_dir, model_name, key)
                ids = item["semantic_slots"].astype(int)
                values.append(
                    float(
                        np.mean(
                            [
                                cosine_distance_curve(
                                    base["slotz"][:, base_ids[index]],
                                    item["slotz"][:, ids[index]],
                                ).mean()
                                for index in range(2)
                            ]
                        )
                    )
                )
            duplicates[md5]["models"][model_name] = values

    group_summary = []
    for group in groups:
        rows = [
            row
            for row in metric_rows
            if row["group"] == group["name"] and row["case"] != group["baseline"]
        ]
        for model_name in MODEL_SPECS:
            selected = [row for row in rows if row["model"] == model_name]
            if not selected:
                continue
            group_summary.append(
                {
                    "group": group["name"],
                    "family": group.get("family", "physics"),
                    "model": model_name,
                    "n_nonbaseline": len(selected),
                    "ball_raw_mean": float(
                        np.mean([row["ball_raw_cosine_distance"] for row in selected])
                    ),
                    "block_raw_mean": float(
                        np.mean([row["block_raw_cosine_distance"] for row in selected])
                    ),
                    "ball_dtw_mean": float(
                        np.mean([row["ball_motion_dtw_distance"] for row in selected])
                    ),
                    "block_dtw_mean": float(
                        np.mean([row["block_motion_dtw_distance"] for row in selected])
                    ),
                }
            )

    summary = {
        "models": {
            name: {
                "label": spec["label"],
                "config": str(Path(spec["config"]).resolve()),
                "checkpoint": str(Path(spec["checkpoint"]).resolve()),
                "slot_dim": int(features[name].shape[1] // 2),
            }
            for name, spec in MODEL_SPECS.items()
        },
        "control": {
            "preprocessing": "same center crop and 256x256 bilinear resize",
            "frames": "all frames" if args.max_frames <= 0 else args.max_frames,
            "boxes": "same two canonical first-frame AMG boxes for every case/model",
            "direct_cross_model_cosine": False,
        },
        "case_count": len(cases),
        "strict_case_count": len(common_keys),
        "strict_excluded_cases": sorted(strict_excluded & set(case_by_key)),
        "rdm_spearman": rdm_spearman,
        "linear_cka": cka,
        "exact_duplicate_sanity": duplicates,
        "group_summary": group_summary,
        "cases": case_payload,
    }
    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "group_summary.csv", group_summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_report_html(args.output_dir, summary)
    write_readme(args.output_dir, summary)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report_html(output_dir: Path, summary: dict[str, Any]) -> None:
    cases_json = json.dumps(summary["cases"], ensure_ascii=True)
    first_case = next(iter(summary["cases"]))
    group_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['group'])}</td>"
        f"<td>{html.escape(row['model'])}</td>"
        f"<td>{row['ball_raw_mean']:.5f}</td>"
        f"<td>{row['block_raw_mean']:.5f}</td>"
        f"<td>{row['ball_dtw_mean']:.5f}</td>"
        f"<td>{row['block_dtw_mean']:.5f}</td>"
        "</tr>"
        for row in summary["group_summary"]
    )
    options = "\n".join(
        f"<option value='{html.escape(key)}'>{html.escape(key)}</option>"
        for key in summary["cases"]
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC 0526dp slot sensitivity</title>
<style>
:root{{--bg:#f5f7fa;--ink:#17202a;--muted:#65717e;--line:#ccd3db;--blue:#2563eb;--red:#dc2626}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{background:#17202a;color:#fff;padding:18px 24px}}header h1{{font-size:22px;margin:0 0 5px}}header p{{margin:0;color:#cbd5e1}}
main{{max-width:1500px;margin:auto;padding:20px 24px 40px}}h2{{font-size:17px;margin:25px 0 10px}}
.metrics{{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));border:1px solid var(--line);background:#fff}}
.metric{{padding:14px 16px;border-right:1px solid var(--line)}}.metric:last-child{{border:0}}.metric b{{display:block;font-size:22px}}
.metric span,.note{{color:var(--muted)}}select{{width:100%;max-width:760px;padding:8px;border:1px solid #98a3af;background:white}}
.viewer{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}}figure{{margin:0}}figure img{{display:block;width:100%;border:1px solid var(--line);background:#fff}}
figcaption{{padding-top:6px;color:var(--muted)}}video{{width:100%;max-height:540px;background:#111}}
table{{width:100%;border-collapse:collapse;background:#fff;font-variant-numeric:tabular-nums}}th,td{{padding:7px 9px;border:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}
.summary-image{{width:100%;background:#fff;border:1px solid var(--line)}}code{{font-size:12px}}@media(max-width:850px){{.metrics,.viewer{{grid-template-columns:1fr}}.metric{{border-right:0;border-bottom:1px solid var(--line)}}}}
</style>
</head>
<body>
<header><h1>xSSC slot sensitivity: controlled DINOv2 vs DINOv3</h1><p>Same source frames, center crop, 256 resolution, canonical boxes and evaluation metrics.</p></header>
<main>
<div class="metrics">
 <div class="metric"><span>Cases shown / strict metrics</span><b>{summary['case_count']} / {summary['strict_case_count']}</b></div>
 <div class="metric"><span>Cross-model RDM Spearman, higher is more similar</span><b>{summary['rdm_spearman']:.4f}</b></div>
 <div class="metric"><span>Cross-model linear CKA, higher is more similar</span><b>{summary['linear_cka']:.4f}</b></div>
</div>
<h2>Case inspection</h2>
<select id="case-select">{options}</select>
<p id="case-note" class="note"></p>
<video id="source-video" controls preload="metadata"></video>
<div class="viewer">
 <figure><img id="contact" alt="slot overlays"><figcaption>Three sampled frames: input, official DINOv2, DINOv3.</figcaption></figure>
 <figure><img id="curve" alt="slot distance curves"><figcaption>Per-frame cosine distance to the controlled group baseline. Lower means more similar.</figcaption></figure>
</div>
<h2>Representational distance matrices</h2>
<img class="summary-image" src="assets/summary_rdm.png" alt="RDM comparison">
<p class="note">Each cell is a within-model cosine distance between case-level ball+block descriptors. Darker means more similar. Only matrix geometry is compared across the 256-D and 512-D spaces.</p>
<h2>Group means</h2>
<table><thead><tr><th>Group</th><th>Model</th><th>Ball raw</th><th>Block raw</th><th>Ball motion-DTW</th><th>Block motion-DTW</th></tr></thead><tbody>{group_rows}</tbody></table>
</main>
<script>
const cases={cases_json};
const select=document.getElementById("case-select");
const video=document.getElementById("source-video");
const contact=document.getElementById("contact");
const curve=document.getElementById("curve");
const note=document.getElementById("case-note");
function show(key){{
 const item=cases[key];
 video.src=item.source_video;
 contact.src=item.contact;
 curve.src=item.curve;
 note.textContent=`Group: ${{item.group}} | baseline: ${{item.baseline}}`;
}}
select.value={json.dumps(first_case)};select.addEventListener("change",()=>show(select.value));show(select.value);
</script>
</body></html>
"""
    (output_dir / "index.html").write_text(page)


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# xSSC 0526dp controlled slot comparison

Models:
- Official DINOv2 MOVi-C `42-0035.pth` (slot dimension 256)
- DINOv3 MOVi-C `step-036000.pth` (slot dimension 512)

Controls:
- The exact same decoded frame indices are used for both models.
- Every frame is center-cropped to a square and resized to 256x256.
- Prompt-free SAM2 AMG runs once on `{REFERENCE_CASE}` frame 0.
- The same ball/block boxes are repeated over time and reused for every case/model.
- Both models run in eval/inference mode with bbox order `[ball, block, unused...]`.
- Within-model cosine distance measures sensitivity. Cross-model comparison uses
  RDM Spearman and linear CKA; direct 256-D to 512-D cosine is intentionally absent.

Alignment:
- Raw metrics compare equal frame indices.
- Motion-DTW aligns scalar pixel-motion curves, then compares the corresponding
  object slots. It is secondary to raw alignment and does not alter extraction.

Summary:
- Cases shown: {summary['case_count']}
- Strict metric cases: {summary['strict_case_count']}
- Excluded from strict metrics because their initial geometry does not match the
  canonical boxes: {", ".join(summary['strict_excluded_cases'])}
- RDM Spearman: {summary['rdm_spearman']:.6f}
- Linear CKA: {summary['linear_cka']:.6f}

Files:
- `index.html`: interactive report
- `canonical_boxes.json`: exact shared conditioning boxes and SAM2 provenance
- `embeddings/`: cached slot tensors, attention labels, frame indices
- `metrics.csv`: case/model metrics
- `group_summary.csv`: family-level means
- `summary.json`: machine-readable report
"""
    (output_dir / "README.md").write_text(text)


def validate_inputs(args: argparse.Namespace) -> None:
    paths = [args.data_root]
    if args.stage in ("boxes", "all"):
        paths.extend([args.sam2_config, args.sam2_checkpoint])
    if args.stage in ("extract", "all"):
        for spec in MODEL_SPECS.values():
            paths.extend([Path(spec["config"]), Path(spec["checkpoint"])])
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(args.data_root, args.max_cases)
    device = torch.device(args.device)
    box_path = args.output_dir / "canonical_boxes.json"
    box_metadata = None
    if args.stage in ("boxes", "all"):
        box_metadata = make_canonical_boxes(args, cases, device)
        print(f"[boxes] {box_path}", flush=True)
    if args.stage in ("extract", "all"):
        if box_metadata is None:
            if not box_path.is_file():
                raise FileNotFoundError(
                    f"{box_path} does not exist; run --stage boxes first"
                )
            box_metadata = json.loads(box_path.read_text())
        extract_all(args, cases, box_metadata, device)
    if args.stage in ("report", "all"):
        summary = compute_report(args, cases)
        print(
            f"[report] cases={summary['case_count']} "
            f"RDM rho={summary['rdm_spearman']:.5f} "
            f"CKA={summary['linear_cka']:.5f}",
            flush=True,
        )
        print(f"[report] {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
