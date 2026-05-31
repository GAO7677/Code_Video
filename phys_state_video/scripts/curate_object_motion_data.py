from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.proxy_state import extract_primary_track
from phys_state_video.schemas import StateIndex


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_TOKEN = "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"

HUMAN_RE = re.compile(
    r"\b("
    r"man|men|woman|women|person|people|boy|girl|child|children|adult|male|female|"
    r"speaker|chef|dancer|singer|worker|player|rider|passenger|pedestrian|"
    r"human|face|hand|hands|arm|arms|leg|legs|head|mouth|eye|eyes|"
    r"student|students|teacher|teachers|cook|cooks|performer|performers|"
    r"mechanic|mechanics|serviceman|servicewoman|technician|technicians|"
    r"interview|presenter|host|audience|crowd|couple|family|baby|babies"
    r")\b",
    re.IGNORECASE,
)
ANIMATED_RE = re.compile(
    r"\b("
    r"animated|animation|cartoon|anime|cgi|3d animated|rendered|illustration|"
    r"pixel art|gameplay|video game|toy animation"
    r")\b",
    re.IGNORECASE,
)
MOTION_RE = re.compile(
    r"\b("
    r"move|moving|moves|driving|drive|drives|racing|race|races|rolling|roll|rolls|"
    r"spinning|spin|spins|rotating|rotate|rotates|turning|turn|turns|"
    r"sliding|slide|slides|flying|fly|flies|landing|takeoff|sailing|sail|sails|"
    r"floating|float|floats|swimming|swim|swims|jumping|jump|jumps|"
    r"running|run|runs|falling|fall|falls|pouring|pour|pours|flowing|flow|flows|"
    r"drifting|drift|drifts|crashing|crash|crashes|bouncing|bounce|bounces|"
    r"colliding|collide|collides|tracking|follows|following|approaching|leaving"
    r")\b",
    re.IGNORECASE,
)
EXPLICIT_MOTION_RE = re.compile(
    r"\b("
    r"in motion|moving|moves|speeding|rolling|spinning|rotating|turning|sliding|"
    r"flying|landing|takeoff|sailing|gliding|floating|swimming|jumping|"
    r"falling|drifting|crashing|bouncing|colliding|approaching|leaving|"
    r"driving (down|on|along|through|towards|toward|across|up|off|into|past)|"
    r"is driving|are driving"
    r")\b",
    re.IGNORECASE,
)
STATIC_CONTEXT_RE = re.compile(
    r"\b("
    r"parked|driver's seat|drivers seat|interior of a car|interior of the car|"
    r"car interior|dashboard|steering wheel|showroom|car show|engine cover|"
    r"engine\b|split-screen comparison|comparison of two cars|design and features|"
    r"showcasing its design and features|showcasing the engine|appears to be parked|"
    r"stationary|repair shop|garage|car hood|transmission|cardan|wheel removed|"
    r"disk break|disc brake|axel|axle|inside the crowded train|inside the train|"
    r"train interior"
    r")\b",
    re.IGNORECASE,
)
DISALLOWED_CONTEXT_RE = re.compile(
    r"\b("
    r"driver's seat|drivers seat|interior of a car|interior of the car|car interior|"
    r"dashboard|steering wheel|showroom|car show|engine cover|split-screen comparison|"
    r"comparison of two cars|repair shop|garage|car hood|transmission|cardan|"
    r"wheel removed|disk break|disc brake|axel|axle|inside the crowded train|"
    r"inside the train|train interior"
    r")\b",
    re.IGNORECASE,
)
LEADING_VIDEO_RE = re.compile(
    r"^(the video (shows|captures|features|depicts)|in the video|this video)\s*,?\s*",
    re.IGNORECASE,
)

OBJECT_GROUPS: dict[str, tuple[str, ...]] = {
    "driving": (
        "car", "cars", "truck", "trucks", "bus", "buses", "train", "trains",
        "motorcycle", "motorcycles", "bike", "bikes", "bicycle", "bicycles",
        "scooter", "scooters", "tram", "vehicle", "vehicles", "engine",
        "dashboard", "steering wheel", "race track",
    ),
    "maritime": (
        "boat", "boats", "ship", "ships", "yacht", "yachts", "marina",
        "harbor", "harbour", "sailing", "sailboat", "sailboats", "vessel",
        "vessels", "watercraft", "kayak", "canoe", "ferry",
    ),
    "aviation": (
        "airplane", "airplanes", "plane", "planes", "helicopter", "helicopters",
        "jet", "jets", "aircraft",
    ),
    "animals": (
        "dog", "dogs", "cat", "cats", "bird", "birds", "horse", "horses",
        "fish", "shark", "whale", "duck", "ducks", "donkey", "animal", "animals",
    ),
    "industrial": (
        "robot", "robots", "machine", "machines", "forklift", "crane",
        "excavator", "bulldozer", "loader", "conveyor", "assembly line",
    ),
    "object_interaction": (
        "ball", "balls", "box", "boxes", "container", "containers", "package",
        "packages", "door", "wheels", "wheel", "domino", "marble", "toy",
        "toys", "cookie", "cookies", "glass", "bottle", "bottles",
    ),
    "natural_motion": (
        "rain", "snow", "smoke", "fire", "cloud", "clouds", "forest", "tree",
        "trees", "waterfall", "waterfalls", "wave", "waves",
    ),
}
OBJECT_GROUP_PATTERNS = {
    name: re.compile(
        "|".join(rf"\b{re.escape(token)}\b" for token in tokens),
        re.IGNORECASE,
    )
    for name, tokens in OBJECT_GROUPS.items()
}
PREFERRED_CATEGORIES = {
    "driving",
    "maritime",
    "aviation",
    "industrial",
    "object_interaction",
}
SCENIC_RE = re.compile(
    r"\b("
    r"landscape|scenery|forest scene|serene scene|tranquil|peaceful|lush greenery|"
    r"foliage|canopy|sunlight filters|hidden trail|waterfall|mist|mountain range|"
    r"beautiful nature|picturesque|panoramic|time-lapse|timelapse"
    r")\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class CaptionDecision:
    keep: bool
    reason: str
    categories: list[str]
    preferred_categories: list[str]
    has_human: bool
    has_motion: bool
    explicit_motion: bool
    animated: bool
    scenic: bool
    static_context: bool
    disallowed_context: bool
    score: float
    recaption: str


@dataclass(slots=True)
class ResizeMetadata:
    mode: str
    target_height: int
    target_width: int
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    scale_y: float
    scale_x: float
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


class OpenVidSource:
    def __init__(self, roots: Sequence[Path]):
        self.files: list[dict[str, object]] = []
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.parquet")):
                pf = pq.ParquetFile(path)
                self.files.append(
                    {
                        "root": str(root),
                        "path": str(path),
                        "rows": int(pf.metadata.num_rows),
                    }
                )
        if not self.files:
            raise FileNotFoundError("no OpenVid parquet files found")

    def iter_candidates(self, seed: int) -> Iterable[tuple[str, int, dict, bytes]]:
        file_order = list(self.files)
        rng = random.Random(seed)
        rng.shuffle(file_order)
        for file_info in file_order:
            parquet_path = str(file_info["path"])
            pf = pq.ParquetFile(parquet_path)
            row_ids = list(range(int(file_info["rows"])))
            rng.shuffle(row_ids)
            for row_id in row_ids:
                row = pf.read_row_group(row_id, columns=["info", "raw_video"])
                info = torch.load(io.BytesIO(row.column("info")[0].as_py()), map_location="cpu", weights_only=False)
                raw_video = row.column("raw_video")[0].as_py()
                yield parquet_path, row_id, info, raw_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate non-human object-motion datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter-episodes", help="Filter an existing episode dataset by prompt.")
    filter_parser.add_argument("--input-root", type=Path, required=True)
    filter_parser.add_argument("--output-root", type=Path, required=True)
    filter_parser.add_argument("--symlink", action="store_true")
    filter_parser.add_argument("--min-score", type=float, default=1.5)

    openvid_parser = subparsers.add_parser("build-openvid", help="Build curated episodes directly from local OpenVid parquet roots.")
    openvid_parser.add_argument(
        "--parquet-root",
        type=Path,
        action="append",
        required=True,
        help="One or more parquet directories, e.g. train_subset_0530/train",
    )
    openvid_parser.add_argument("--output-root", type=Path, required=True)
    openvid_parser.add_argument("--height", type=int, default=96)
    openvid_parser.add_argument("--width", type=int, default=96)
    openvid_parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="stretch",
        help="Frame resize policy before proxy-state extraction.",
    )
    openvid_parser.add_argument("--context-frames", type=int, default=8)
    openvid_parser.add_argument("--future-frames", type=int, default=16)
    openvid_parser.add_argument("--train-count", type=int, default=512)
    openvid_parser.add_argument("--val-count", type=int, default=64)
    openvid_parser.add_argument("--seed", type=int, default=20260531)
    openvid_parser.add_argument("--min-aesthetic", type=float, default=4.8)
    openvid_parser.add_argument("--min-motion", type=float, default=1.4)
    openvid_parser.add_argument("--min-visible-fraction", type=float, default=0.65)
    openvid_parser.add_argument("--min-score", type=float, default=2.0)
    openvid_parser.add_argument("--min-track-displacement", type=float, default=0.05)
    openvid_parser.add_argument("--min-track-path", type=float, default=0.12)
    openvid_parser.add_argument("--min-track-scale-span", type=float, default=0.10)

    webvid_parser = subparsers.add_parser("download-webvid", help="Download a small direct-link WebVid object-motion subset and build episodes.")
    webvid_parser.add_argument("--output-root", type=Path, required=True)
    webvid_parser.add_argument("--metadata-root", type=Path, required=True)
    webvid_parser.add_argument("--height", type=int, default=96)
    webvid_parser.add_argument("--width", type=int, default=96)
    webvid_parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="stretch",
        help="Frame resize policy before proxy-state extraction.",
    )
    webvid_parser.add_argument("--context-frames", type=int, default=8)
    webvid_parser.add_argument("--future-frames", type=int, default=16)
    webvid_parser.add_argument("--train-count", type=int, default=96)
    webvid_parser.add_argument("--val-count", type=int, default=16)
    webvid_parser.add_argument("--seed", type=int, default=20260531)
    webvid_parser.add_argument("--partitions", type=int, default=12)
    webvid_parser.add_argument("--min-duration", type=float, default=5.0)
    webvid_parser.add_argument("--max-duration", type=float, default=20.0)
    webvid_parser.add_argument("--min-visible-fraction", type=float, default=0.60)
    webvid_parser.add_argument("--min-score", type=float, default=2.0)
    webvid_parser.add_argument("--min-track-displacement", type=float, default=0.05)
    webvid_parser.add_argument("--min-track-path", type=float, default=0.12)
    webvid_parser.add_argument("--min-track-scale-span", type=float, default=0.10)
    webvid_parser.add_argument("--hf-endpoint", type=str, default=DEFAULT_HF_ENDPOINT)
    webvid_parser.add_argument("--hf-token", type=str, default=DEFAULT_HF_TOKEN)

    panda_parser = subparsers.add_parser("build-panda-candidates", help="Download Panda metadata and write filtered candidate manifests.")
    panda_parser.add_argument("--output-root", type=Path, required=True)
    panda_parser.add_argument("--train-files", type=int, default=2)
    panda_parser.add_argument("--max-candidates", type=int, default=400)
    panda_parser.add_argument("--max-rows-per-file", type=int, default=250000)
    panda_parser.add_argument("--seed", type=int, default=20260531)
    panda_parser.add_argument("--hf-endpoint", type=str, default=DEFAULT_HF_ENDPOINT)
    panda_parser.add_argument("--hf-token", type=str, default=DEFAULT_HF_TOKEN)

    merge_parser = subparsers.add_parser("merge-episodes", help="Merge multiple episode roots into one dataset via symlinks.")
    merge_parser.add_argument("--input-root", type=Path, action="append", required=True)
    merge_parser.add_argument("--output-root", type=Path, required=True)

    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def parse_iso8601_duration(text: str) -> float:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", text.strip())
    if not match:
        return 0.0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return float(hours * 3600 + minutes * 60 + seconds)


def detect_categories(text: str) -> list[str]:
    hits: list[str] = []
    for name, pattern in OBJECT_GROUP_PATTERNS.items():
        if pattern.search(text):
            hits.append(name)
    return hits


def recaption_object_motion(text: str, categories: Sequence[str]) -> str:
    cleaned = clean_text(text)
    cleaned = LEADING_VIDEO_RE.sub("", cleaned)
    cleaned = cleaned.strip(" .")
    if not cleaned:
        cleaned = "a non-human object-focused scene"
    cleaned = cleaned[0].lower() + cleaned[1:] if cleaned else cleaned
    category_text = ", ".join(categories) if categories else "object motion"
    prefix = f"A real-world clip focused on {category_text}. "
    suffix = " Motion is the main signal in the scene."
    return prefix + cleaned + "." + suffix


def classify_caption(
    caption: str,
    *,
    motion_score: float | None = None,
    aesthetic_score: float | None = None,
    allow_humans: bool = False,
) -> CaptionDecision:
    cleaned = clean_text(caption)
    categories = detect_categories(cleaned)
    preferred_categories = [name for name in categories if name in PREFERRED_CATEGORIES]
    has_human = bool(HUMAN_RE.search(cleaned))
    animated = bool(ANIMATED_RE.search(cleaned))
    has_motion = bool(MOTION_RE.search(cleaned))
    explicit_motion = bool(EXPLICIT_MOTION_RE.search(cleaned))
    scenic = bool(SCENIC_RE.search(cleaned))
    static_context = bool(STATIC_CONTEXT_RE.search(cleaned))
    disallowed_context = bool(DISALLOWED_CONTEXT_RE.search(cleaned))

    score = 0.0
    if preferred_categories:
        score += 1.9 + 0.45 * len(preferred_categories)
    elif categories:
        score += 0.4
    if explicit_motion:
        score += 1.25
    elif has_motion:
        score += 0.25
    if motion_score is not None:
        if motion_score >= 2.2:
            score += 1.2
        elif motion_score >= 1.6:
            score += 0.6
        elif motion_score >= 1.2:
            score += 0.2
    if aesthetic_score is not None:
        if aesthetic_score >= 5.6:
            score += 0.6
        elif aesthetic_score >= 4.8:
            score += 0.3

    if has_human and not allow_humans:
        score -= 3.0
    if animated:
        score -= 1.5
    if scenic and not preferred_categories:
        score -= 1.0
    if static_context and not explicit_motion:
        score -= 1.25
    if disallowed_context:
        score -= 2.0

    reason = "accepted"
    keep = True
    if not categories:
        keep = False
        reason = "no_object_category"
    elif not preferred_categories:
        keep = False
        reason = "weak_target_category"
    elif has_human and not allow_humans:
        keep = False
        reason = "human_centric"
    elif animated:
        keep = False
        reason = "animated"
    elif scenic and not preferred_categories:
        keep = False
        reason = "scenic_background"
    elif disallowed_context:
        keep = False
        reason = "disallowed_context"
    elif static_context and not explicit_motion:
        keep = False
        reason = "static_showcase"
    elif not explicit_motion and (motion_score or 0.0) < 1.4:
        keep = False
        reason = "too_static"

    return CaptionDecision(
        keep=keep,
        reason=reason,
        categories=categories,
        preferred_categories=preferred_categories,
        has_human=has_human,
        has_motion=has_motion,
        explicit_motion=explicit_motion,
        animated=animated,
        scenic=scenic,
        static_context=static_context,
        disallowed_context=disallowed_context,
        score=float(score),
        recaption=recaption_object_motion(cleaned, categories),
    )


def summarize_track_motion(track) -> dict[str, float]:
    visibility = track.states[:, 0, StateIndex.VISIBILITY] > 0.5
    boxes = track.boxes[:, 0, :]
    visible_boxes = boxes[visibility]
    if visible_boxes.shape[0] < 2:
        return {
            "path_length": 0.0,
            "net_displacement": 0.0,
            "scale_span": 0.0,
        }

    centers = 0.5 * (visible_boxes[:, 0:2] + visible_boxes[:, 2:4])
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    widths = np.clip(visible_boxes[:, 2] - visible_boxes[:, 0], 1e-6, 1.0)
    heights = np.clip(visible_boxes[:, 3] - visible_boxes[:, 1], 1e-6, 1.0)
    log_areas = np.log(widths * heights)
    return {
        "path_length": float(np.sum(steps)),
        "net_displacement": float(np.linalg.norm(centers[-1] - centers[0])),
        "scale_span": float(np.max(log_areas) - np.min(log_areas)),
    }


def resize_frames_uint8(
    frames: Sequence[np.ndarray],
    height: int,
    width: int,
    *,
    resize_mode: str = "stretch",
) -> tuple[np.ndarray, ResizeMetadata]:
    if not frames:
        raise ValueError("expected at least one frame")
    first_frame = frames[0]
    if first_frame.ndim != 3:
        raise ValueError(f"expected HWC frame, got shape {first_frame.shape}")
    original_height, original_width = first_frame.shape[:2]

    resized: list[np.ndarray] = []
    resize_meta: ResizeMetadata | None = None
    for frame in frames:
        if frame.ndim != 3:
            raise ValueError(f"expected HWC frame, got shape {frame.shape}")
        if frame.shape[:2] != (original_height, original_width):
            raise ValueError("all frames in a clip must share the same spatial size")

        if resize_mode == "stretch":
            resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if resize_meta is None:
                resize_meta = ResizeMetadata(
                    mode="stretch",
                    target_height=height,
                    target_width=width,
                    original_height=original_height,
                    original_width=original_width,
                    resized_height=height,
                    resized_width=width,
                    scale_y=float(height) / float(original_height),
                    scale_x=float(width) / float(original_width),
                    pad_top=0,
                    pad_bottom=0,
                    pad_left=0,
                    pad_right=0,
                )
        elif resize_mode == "letterbox":
            scale = min(float(width) / float(original_width), float(height) / float(original_height))
            resized_width = max(1, int(round(original_width * scale)))
            resized_height = max(1, int(round(original_height * scale)))
            content = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((height, width, frame.shape[2]), dtype=np.uint8)
            pad_top = (height - resized_height) // 2
            pad_left = (width - resized_width) // 2
            pad_bottom = height - resized_height - pad_top
            pad_right = width - resized_width - pad_left
            canvas[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = content
            resized_frame = canvas
            if resize_meta is None:
                resize_meta = ResizeMetadata(
                    mode="letterbox",
                    target_height=height,
                    target_width=width,
                    original_height=original_height,
                    original_width=original_width,
                    resized_height=resized_height,
                    resized_width=resized_width,
                    scale_y=float(resized_height) / float(original_height),
                    scale_x=float(resized_width) / float(original_width),
                    pad_top=pad_top,
                    pad_bottom=pad_bottom,
                    pad_left=pad_left,
                    pad_right=pad_right,
                )
        else:
            raise ValueError(f"unsupported resize_mode: {resize_mode}")
        resized.append(np.transpose(resized_frame.astype(np.float32) / 255.0, (2, 0, 1)))
    assert resize_meta is not None
    return np.stack(resized, axis=0), resize_meta


def write_episode(
    output_root: Path,
    split: str,
    sample_id: str,
    frames_chw: np.ndarray,
    prompt: str,
    metadata: dict[str, object],
    resize_meta: ResizeMetadata | None,
    context_frames: int,
    future_frames: int,
    min_visible_fraction: float,
    min_track_displacement: float,
    min_track_path: float,
    min_track_scale_span: float,
) -> tuple[dict[str, object] | None, str | None]:
    if frames_chw.shape[0] < context_frames + future_frames:
        return None, "too_few_frames"
    clip = frames_chw[: context_frames + future_frames]
    track = extract_primary_track(clip)
    if track.visible_fraction < min_visible_fraction:
        return None, "low_visible_fraction"
    motion_summary = summarize_track_motion(track)
    if (
        motion_summary["net_displacement"] < min_track_displacement
        and motion_summary["path_length"] < min_track_path
        and motion_summary["scale_span"] < min_track_scale_span
    ):
        return None, "low_track_motion"

    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    episode_path = split_dir / f"{sample_id}.npz"
    np.savez_compressed(
        episode_path,
        context_frames=track.frames[:context_frames].astype(np.float32),
        future_frames=track.frames[context_frames:context_frames + future_frames].astype(np.float32),
        context_states=track.states[:context_frames].astype(np.float32),
        future_states=track.states[context_frames:context_frames + future_frames].astype(np.float32),
        context_boxes=track.boxes[:context_frames].astype(np.float32),
        future_boxes=track.boxes[context_frames:context_frames + future_frames].astype(np.float32),
        appearance=track.appearance.astype(np.float32),
        camera=np.zeros((context_frames, 8), dtype=np.float32),
    )
    payload = {
        "prompt": prompt,
        "source": metadata,
        "track_motion": motion_summary,
    }
    if resize_meta is not None:
        payload["resize"] = asdict(resize_meta)
    episode_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "sample_id": sample_id,
        "split": split,
        "visible_fraction": track.visible_fraction,
        "track_motion": motion_summary,
        "prompt": prompt,
        "source": metadata,
        "resize": asdict(resize_meta) if resize_meta is not None else None,
    }, None


def decode_video_bytes(raw_video: bytes, *, target_frames: int, sample_key: str) -> tuple[list[np.ndarray], int, int, int]:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(raw_video)
            temp_path = temp_file.name
        reader = imageio.get_reader(temp_path)
        try:
            total_frames = reader.count_frames()
            if total_frames < target_frames:
                raise ValueError(f"{sample_key}: only {total_frames} frames")
            meta = reader.get_meta_data()
            fps = int(round(float(meta.get("fps", 24)))) if meta.get("fps") else 24
            max_start = total_frames - target_frames
            start_frame = stable_hash(sample_key) % (max_start + 1)
            frames = [
                np.asarray(reader.get_data(frame_id), dtype=np.uint8)
                for frame_id in range(start_frame, start_frame + target_frames)
            ]
            return frames, max(1, fps), start_frame, total_frames
        finally:
            reader.close()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def stable_hash(text: str) -> int:
    import hashlib

    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def optional_positive_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        dst.symlink_to(src)
    else:
        dst.write_bytes(src.read_bytes())


def filter_episodes(args: argparse.Namespace) -> None:
    summary: dict[str, object] = {"splits": {}, "examples": []}
    for split in ("train", "val"):
        input_dir = args.input_root / split
        if not input_dir.exists():
            continue
        kept = 0
        skipped = 0
        reasons: dict[str, int] = {}
        output_dir = args.output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for json_path in sorted(input_dir.glob("*.json")):
            npz_path = json_path.with_suffix(".npz")
            if not npz_path.exists():
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            prompt = str(payload.get("prompt") or "")
            decision = classify_caption(prompt)
            if not decision.keep or decision.score < args.min_score:
                skipped += 1
                reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
                continue
            kept += 1
            if len(summary["examples"]) < 20:
                summary["examples"].append(
                    {
                        "split": split,
                        "sample": json_path.stem,
                        "prompt": prompt,
                        "recaption": decision.recaption,
                        "categories": decision.categories,
                        "score": decision.score,
                    }
                )
            dst_npz = output_dir / npz_path.name
            dst_json = output_dir / json_path.name
            link_or_copy(npz_path, dst_npz, args.symlink)
            new_payload = payload | {
                "prompt": decision.recaption,
                "curation": asdict(decision),
            }
            dst_json.write_text(json.dumps(new_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["splits"][split] = {
            "kept": kept,
            "skipped": skipped,
            "reasons": reasons,
        }
    (args.output_root / "curation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_openvid(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    total_target = args.train_count + args.val_count
    source = OpenVidSource(args.parquet_root)
    seen_source_names: set[str] = set()
    records: list[dict[str, object]] = []
    reject_counts: dict[str, int] = {}

    for parquet_path, row_id, info, raw_video in source.iter_candidates(args.seed):
        if len(records) >= total_target:
            break
        caption = clean_text(str(info.get("caption") or ""))
        source_video_name = str(info.get("source_video_name") or "")
        if source_video_name and source_video_name in seen_source_names:
            reject_counts["duplicate_source_video"] = reject_counts.get("duplicate_source_video", 0) + 1
            continue

        motion_score = optional_positive_float(info.get("motion_score"))
        aesthetic_score = optional_positive_float(info.get("aesthetic_score"))
        temporal_score = optional_positive_float(info.get("temporal_consistency_score"))
        decision = classify_caption(
            caption,
            motion_score=motion_score,
            aesthetic_score=aesthetic_score,
        )
        if not decision.keep or decision.score < args.min_score:
            reject_counts[decision.reason] = reject_counts.get(decision.reason, 0) + 1
            continue
        if aesthetic_score is not None and aesthetic_score < args.min_aesthetic:
            reject_counts["low_aesthetic"] = reject_counts.get("low_aesthetic", 0) + 1
            continue
        if motion_score is not None and motion_score < args.min_motion:
            reject_counts["low_motion"] = reject_counts.get("low_motion", 0) + 1
            continue
        if temporal_score is not None and temporal_score < 0.995:
            reject_counts["low_temporal_consistency"] = reject_counts.get("low_temporal_consistency", 0) + 1
            continue

        sample_id = f"openvid_obj_{len(records):05d}__{Path(parquet_path).stem}__row{row_id:03d}"
        try:
            frames_hwc, fps, start_frame, total_frames = decode_video_bytes(
                raw_video,
                target_frames=args.context_frames + args.future_frames,
                sample_key=sample_id,
            )
        except Exception:
            reject_counts["decode_error"] = reject_counts.get("decode_error", 0) + 1
            continue

        frames_chw, resize_meta = resize_frames_uint8(
            frames_hwc,
            args.height,
            args.width,
            resize_mode=args.resize_mode,
        )
        split = "train" if len(records) < args.train_count else "val"
        metadata = {
            "dataset": "OpenVidHD",
            "source_type": "local_parquet",
            "caption": caption,
            "recaption": decision.recaption,
            "categories": decision.categories,
            "aesthetic_score": aesthetic_score,
            "motion_score": motion_score,
            "temporal_consistency_score": temporal_score,
            "camera_motion": str(info.get("camera_motion") or ""),
            "fps": fps,
            "raw_total_frames": total_frames,
            "clip_start_frame": start_frame,
            "source_video_name": source_video_name,
            "parquet_path": parquet_path,
            "parquet_row_id": row_id,
            "resize_mode": args.resize_mode,
        }
        record, reject_reason = write_episode(
            output_root=output_root,
            split=split,
            sample_id=sample_id,
            frames_chw=frames_chw,
            prompt=decision.recaption,
            metadata=metadata,
            resize_meta=resize_meta,
            context_frames=args.context_frames,
            future_frames=args.future_frames,
            min_visible_fraction=args.min_visible_fraction,
            min_track_displacement=args.min_track_displacement,
            min_track_path=args.min_track_path,
            min_track_scale_span=args.min_track_scale_span,
        )
        if record is None:
            key = reject_reason or "episode_write_rejected"
            reject_counts[key] = reject_counts.get(key, 0) + 1
            continue

        seen_source_names.add(source_video_name)
        records.append(record)
        if len(records) % 50 == 0:
            print(f"accepted {len(records)} / {total_target}")

    summary = {
        "output_root": str(output_root),
        "height": args.height,
        "width": args.width,
        "resize_mode": args.resize_mode,
        "train_count": sum(1 for r in records if r["split"] == "train"),
        "val_count": sum(1 for r in records if r["split"] == "val"),
        "target_train_count": args.train_count,
        "target_val_count": args.val_count,
        "reject_counts": reject_counts,
        "examples": records[:20],
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def iter_webvid_rows(metadata_root: Path, partitions: int, hf_endpoint: str, hf_token: str) -> Iterable[dict[str, str]]:
    metadata_root.mkdir(parents=True, exist_ok=True)
    for part_id in range(partitions):
        filename = f"data/train/partitions/{part_id:04d}.csv"
        csv_path = hf_hub_download(
            repo_id="TempoFunk/webvid-10M",
            repo_type="dataset",
            filename=filename,
            local_dir=str(metadata_root),
            endpoint=hf_endpoint,
            token=hf_token,
        )
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row


def http_session_no_proxy() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def download_webvid(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    total_target = args.train_count + args.val_count
    rng = random.Random(args.seed)
    rows = list(iter_webvid_rows(args.metadata_root, args.partitions, args.hf_endpoint, args.hf_token))
    rng.shuffle(rows)
    session = http_session_no_proxy()
    records: list[dict[str, object]] = []
    reject_counts: dict[str, int] = {}
    seen_urls: set[str] = set()

    for row in rows:
        if len(records) >= total_target:
            break
        name = clean_text(row.get("name", ""))
        url = clean_text(row.get("contentUrl", ""))
        if not url or url in seen_urls:
            continue
        duration = parse_iso8601_duration(clean_text(row.get("duration", "")))
        if duration < args.min_duration or duration > args.max_duration:
            reject_counts["duration_out_of_range"] = reject_counts.get("duration_out_of_range", 0) + 1
            continue

        decision = classify_caption(name, motion_score=1.6, aesthetic_score=5.0)
        if not decision.keep or decision.score < args.min_score:
            reject_counts[decision.reason] = reject_counts.get(decision.reason, 0) + 1
            continue

        sample_id = f"webvid_obj_{len(records):05d}__{row.get('videoid','unknown')}"
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            frames_hwc, fps, start_frame, total_frames = decode_video_bytes(
                response.content,
                target_frames=args.context_frames + args.future_frames,
                sample_key=sample_id,
            )
        except Exception:
            reject_counts["download_or_decode_error"] = reject_counts.get("download_or_decode_error", 0) + 1
            continue

        frames_chw, resize_meta = resize_frames_uint8(
            frames_hwc,
            args.height,
            args.width,
            resize_mode=args.resize_mode,
        )
        split = "train" if len(records) < args.train_count else "val"
        metadata = {
            "dataset": "WebVid10M",
            "source_type": "direct_mp4",
            "caption": name,
            "recaption": decision.recaption,
            "categories": decision.categories,
            "content_url": url,
            "duration_seconds": duration,
            "fps": fps,
            "raw_total_frames": total_frames,
            "clip_start_frame": start_frame,
            "page_dir": row.get("page_dir", ""),
            "resize_mode": args.resize_mode,
        }
        record, reject_reason = write_episode(
            output_root=output_root,
            split=split,
            sample_id=sample_id,
            frames_chw=frames_chw,
            prompt=decision.recaption,
            metadata=metadata,
            resize_meta=resize_meta,
            context_frames=args.context_frames,
            future_frames=args.future_frames,
            min_visible_fraction=args.min_visible_fraction,
            min_track_displacement=args.min_track_displacement,
            min_track_path=args.min_track_path,
            min_track_scale_span=args.min_track_scale_span,
        )
        if record is None:
            key = reject_reason or "episode_write_rejected"
            reject_counts[key] = reject_counts.get(key, 0) + 1
            continue

        seen_urls.add(url)
        records.append(record)
        if len(records) % 20 == 0:
            print(f"accepted {len(records)} / {total_target}")
        time.sleep(0.15)

    summary = {
        "output_root": str(output_root),
        "height": args.height,
        "width": args.width,
        "resize_mode": args.resize_mode,
        "train_count": sum(1 for r in records if r["split"] == "train"),
        "val_count": sum(1 for r in records if r["split"] == "val"),
        "target_train_count": args.train_count,
        "target_val_count": args.val_count,
        "reject_counts": reject_counts,
        "examples": records[:20],
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_panda_candidates(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    file_names = ["data/val-00000-of-00001.parquet"]
    for idx in range(args.train_files):
        file_names.append(f"data/train-{idx:05d}-of-00017.parquet")

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    matched_rows = 0
    for file_name in file_names:
        parquet_path = hf_hub_download(
            repo_id="SUSTech/panda-70m",
            repo_type="dataset",
            filename=file_name,
            local_dir=str(metadata_dir),
            endpoint=args.hf_endpoint,
            token=args.hf_token,
        )
        pf = pq.ParquetFile(parquet_path)
        scanned_rows = 0
        for batch in pf.iter_batches(
            columns=["videoID", "url", "timestamp", "caption", "matching_score"],
            batch_size=4096,
        ):
            payload = batch.to_pydict()
            row_count = len(payload["videoID"])
            scanned_rows += row_count
            for idx in range(row_count):
                caption = clean_text(str(payload["caption"][idx] or ""))
                decision = classify_caption(caption, motion_score=1.5, aesthetic_score=5.0)
                if not decision.keep or decision.score < 2.0:
                    continue
                row = {
                    "videoID": str(payload["videoID"][idx]),
                    "url": str(payload["url"][idx]),
                    "timestamp": str(payload["timestamp"][idx]),
                    "caption": caption,
                    "recaption": decision.recaption,
                    "categories": decision.categories,
                    "matching_score": str(payload["matching_score"][idx]),
                    "source_file": file_name,
                }
                matched_rows += 1
                if len(rows) < args.max_candidates:
                    rows.append(row)
                else:
                    replace_idx = rng.randrange(matched_rows)
                    if replace_idx < args.max_candidates:
                        rows[replace_idx] = row
            if scanned_rows >= args.max_rows_per_file:
                break
    manifest_path = output_root / "panda_object_motion_candidates.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "candidate_count": len(rows),
        "metadata_files": file_names,
        "max_rows_per_file": args.max_rows_per_file,
        "matched_rows_seen": matched_rows,
        "manifest_path": str(manifest_path),
        "note": "Panda-70M exposes YouTube URLs in metadata. This step prepares filtered candidates only.",
        "examples": rows[:20],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def merge_episodes(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"sources": [str(path) for path in args.input_root], "splits": {}}
    for split in ("train", "val"):
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for root in args.input_root:
            input_dir = root / split
            if not input_dir.exists():
                continue
            for npz_path in sorted(input_dir.glob("*.npz")):
                json_path = npz_path.with_suffix(".json")
                dst_npz = split_dir / npz_path.name
                dst_json = split_dir / json_path.name
                if dst_npz.exists() or dst_npz.is_symlink():
                    continue
                dst_npz.symlink_to(npz_path)
                if json_path.exists():
                    dst_json.symlink_to(json_path)
                count += 1
        summary["splits"][split] = {"count": count}
    (output_root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "filter-episodes":
        filter_episodes(args)
    elif args.command == "build-openvid":
        build_openvid(args)
    elif args.command == "download-webvid":
        download_webvid(args)
    elif args.command == "build-panda-candidates":
        build_panda_candidates(args)
    elif args.command == "merge-episodes":
        merge_episodes(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
