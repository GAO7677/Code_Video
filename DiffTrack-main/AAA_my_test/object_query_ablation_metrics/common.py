from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_CASE = "0613pybullet_sample_001460_w002"
CASE = os.environ.get("OBJECT_QUERY_ABLATION_CASE", DEFAULT_CASE)
SEED = int(os.environ.get("OBJECT_QUERY_ABLATION_SEED", "47326"))
if SEED < 0:
    raise ValueError("OBJECT_QUERY_ABLATION_SEED must be non-negative")
FRAME_COUNT = 49
HEIGHT = 704
WIDTH = 1280

DEFAULT_RESULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1/"
    f"{CASE}/seed_{SEED:05d}"
)
TUBE_CASE_ROOT = Path(
    os.environ.get("OBJECT_QUERY_ABLATION_RESULT_DIR", str(DEFAULT_RESULT_ROOT))
).expanduser().resolve()
INVENTORY_PATH = Path(
    os.environ.get(
        "OBJECT_QUERY_ABLATION_INVENTORY",
        str(TUBE_CASE_ROOT / "video_similarity_top100.json"),
    )
).expanduser().resolve()
RAFT_ROOT = TUBE_CASE_ROOT / "raft_motion_top100_v1"
BASELINE_TRACKS = TUBE_CASE_ROOT / "frozen_baseline_tracks/tracks.npz"
DEFAULT_REGION_CACHE = Path(
    "/data/gaoya/agent-data/cache/"
    "wan22_ti2v_legacy_firstlatent_regions_704x1280"
) / CASE
REGION_CACHE = Path(
    os.environ.get("OBJECT_QUERY_ABLATION_REGION_CACHE", str(DEFAULT_REGION_CACHE))
).expanduser().resolve()
DEFAULT_SOURCE_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/"
    "industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_001460"
)
SOURCE_ROOT = Path(
    os.environ.get("OBJECT_QUERY_ABLATION_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT))
).expanduser().resolve()
SOURCE_VIDEO = SOURCE_ROOT / "source_video.mp4"
SOURCE_STATES = SOURCE_ROOT / "states.npz"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_ablation_metrics/"
    f"{CASE}/seed_{SEED:05d}"
)
OUTPUT_ROOT = Path(
    os.environ.get("OBJECT_QUERY_ABLATION_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))
).expanduser().resolve()

OBJECTS = ("object_A", "object_B")
OBJECT_LABELS = {"object_A": "sphere / drop_ball", "object_B": "box / support_platform"}
MODE_IDS = {
    "self_only": "M1",
    "incoming_only": "M2",
    "outgoing_only": "M3",
    "query_row": "M4",
    "key_value_column": "M5",
    "cross_boundary": "M6",
    "row_and_column": "M7",
    "literal_kv_zero": "C1",
}


def safe_id(video_id: str) -> str:
    return video_id.replace(":", "__")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(include_source: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if payload.get("case") != CASE or int(payload.get("seed", -1)) != SEED:
        raise RuntimeError(f"inventory does not match {CASE} seed={SEED}")
    videos = [dict(row) for row in payload["videos"]]
    if int(payload.get("video_count", -1)) != len(videos):
        raise RuntimeError("inventory video_count does not match videos")
    if len(videos) < 2 or videos[0].get("id") != "baseline":
        raise RuntimeError("expected baseline plus at least one ablation video")
    ids = [str(row.get("id") or "") for row in videos]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("inventory video IDs must be nonempty and unique")
    for row in videos:
        path = Path(row["path"])
        if not path.is_file():
            raise RuntimeError(f"missing video: {path}")
    if include_source:
        videos.append(
            {
                "id": "source_gt_video",
                "protocol": "source",
                "target_scope": None,
                "region": None,
                "mask_mode": None,
                "path": str(SOURCE_VIDEO),
            }
        )
    return videos


def load_video_frames(path: Path, frame_count: int = FRAME_COUNT) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = []
    while len(frames) < frame_count:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (HEIGHT, WIDTH):
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != frame_count:
        raise RuntimeError(f"expected {frame_count} frames, got {len(frames)}: {path}")
    return np.stack(frames), fps


def load_query_data() -> tuple[np.ndarray, dict[str, slice], np.ndarray]:
    with np.load(REGION_CACHE / "regions.npz", allow_pickle=False) as arrays:
        points = arrays["query_points"].astype(np.float32)[:16]
        masks = arrays["masks_rhw"].astype(bool)[:2]
    return points, {"object_A": slice(0, 8), "object_B": slice(8, 16)}, masks


def video_manifest(video: dict[str, Any]) -> dict[str, Any]:
    path = Path(video["path"]).parent / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
