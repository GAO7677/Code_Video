from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from code_vjepa_vggt.data.phyco_dataset import _build_object_state_arrays
from phys_state_video.proxy_state import extract_primary_track, read_video_frames
from phys_state_video.schemas import StateIndex


@dataclass(slots=True)
class EpisodeCase:
    family: str
    sample_id: str
    raw_sample_dir: Path
    episode_npz_path: Path
    episode_json_path: Path
    window_index: int
    window_start: int
    frame_stride: int

    @property
    def case_id(self) -> str:
        return f"{self.family}__{self.sample_id}__w{self.window_index:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PhyCo-style proxy logic on 0613pybullet.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500"),
    )
    parser.add_argument(
        "--episode-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--families",
        default="F1_single_object,F2_two_object,F3_chain_reaction,F4_occlusion,F5_drop_support",
    )
    parser.add_argument("--cases-per-family", type=int, default=1)
    parser.add_argument(
        "--modes",
        default="gt_boxes_proxy,video_primary_proxy",
        help="Comma-separated subset of gt_boxes_proxy,video_primary_proxy",
    )
    parser.add_argument(
        "--video-primary-families",
        default="F1_single_object,F5_drop_support",
        help="Families eligible for video_primary_proxy mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/phyco_proxy_eval_0613_small"),
    )
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def load_episode_meta(meta_path: Path) -> dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def select_sample_dirs(raw_split_root: Path, families: list[str], cases_per_family: int) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = {}
    for family in families:
        family_root = raw_split_root / family
        sample_dirs = sorted(path for path in family_root.iterdir() if path.is_dir()) if family_root.is_dir() else []
        if not sample_dirs:
            selected[family] = []
            continue
        if len(sample_dirs) <= cases_per_family:
            selected[family] = sample_dirs
            continue
        stride = max(len(sample_dirs) // cases_per_family, 1)
        picks = []
        idx = 0
        while len(picks) < cases_per_family and idx < len(sample_dirs):
            picks.append(sample_dirs[idx])
            idx += stride
        while len(picks) < cases_per_family:
            picks.append(sample_dirs[len(picks)])
        selected[family] = picks[:cases_per_family]
    return selected


def collect_episode_cases(episode_split_root: Path, family_to_samples: dict[str, list[Path]]) -> list[EpisodeCase]:
    cases: list[EpisodeCase] = []
    for family, sample_dirs in family_to_samples.items():
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            meta_paths = sorted(episode_split_root.glob(f"{sample_id}_w*.json"))
            for meta_path in meta_paths:
                meta = load_episode_meta(meta_path)
                npz_path = meta_path.with_suffix(".npz")
                cases.append(
                    EpisodeCase(
                        family=family,
                        sample_id=sample_id,
                        raw_sample_dir=sample_dir,
                        episode_npz_path=npz_path,
                        episode_json_path=meta_path,
                        window_index=int(meta["window_index"]),
                        window_start=int(meta["window_start"]),
                        frame_stride=int(meta["frame_stride"]),
                    )
                )
    return sorted(cases, key=lambda item: (item.family, item.sample_id, item.window_index))


def load_episode_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    payload = np.load(npz_path)
    return {key: payload[key].astype(np.float32) for key in payload.files}


def gt_boxes_proxy_from_episode(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    full_boxes = payload["full_boxes"].astype(np.float32)
    full_states = payload["full_states"].astype(np.float32)
    full_frames = payload["full_frames"].astype(np.float32)
    _, _, height, width = full_frames.shape
    num_frames, max_objects, _ = full_boxes.shape
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    boxes_xyxy = full_boxes * scale[None, None, :]
    widths = np.clip(boxes_xyxy[..., 2] - boxes_xyxy[..., 0], 0.0, None)
    heights = np.clip(boxes_xyxy[..., 3] - boxes_xyxy[..., 1], 0.0, None)
    areas = widths * heights
    existence_any = full_states[..., StateIndex.EXISTENCE].sum(axis=0)
    selected_indices = [idx for idx in range(max_objects) if float(existence_any[idx]) > 0.0]
    proxy_states, proxy_boxes = _build_object_state_arrays(
        boxes_xyxy,
        areas,
        image_hw=(height, width),
        selected_indices=selected_indices,
        max_objects=max_objects,
    )
    return proxy_states.astype(np.float32), proxy_boxes.astype(np.float32)


def video_primary_proxy_from_raw(case: EpisodeCase, target_frames: int = 24) -> tuple[np.ndarray, np.ndarray]:
    video_path = case.raw_sample_dir / "video.mp4"
    raw_frames = read_video_frames(video_path)
    sampled = raw_frames[:: case.frame_stride]
    clip = sampled[case.window_start : case.window_start + target_frames]
    if clip.shape[0] != target_frames:
        raise RuntimeError(
            f"{case.case_id} expected {target_frames} frames after stride/slice, got {clip.shape[0]}"
        )
    track = extract_primary_track(clip)
    return track.states.astype(np.float32), track.boxes.astype(np.float32)


def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(inter_x1 - inter_x0, 0.0)
    inter_h = max(inter_y1 - inter_y0, 0.0)
    inter = inter_w * inter_h
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = max(area_a + area_b - inter, 1.0e-6)
    return float(inter / union)


def compute_state_metrics(
    pred_states: np.ndarray,
    pred_boxes: np.ndarray,
    gt_states: np.ndarray,
    gt_boxes: np.ndarray,
    *,
    compare_slot_count: int,
) -> dict[str, float | int | None]:
    if compare_slot_count <= 0:
        return {
            "compare_slot_count": 0,
            "visible_points": 0,
            "box_iou_mean": None,
            "center_l2_norm_mean": None,
            "center_l2_px_mean": None,
            "depth_mae": None,
            "log_scale_mae": None,
            "vel_l2_mean": None,
            "visibility_f1": None,
            "existence_acc": None,
            "confidence_mae": None,
        }

    pred_states = pred_states[:, :compare_slot_count]
    pred_boxes = pred_boxes[:, :compare_slot_count]
    gt_states = gt_states[:, :compare_slot_count]
    gt_boxes = gt_boxes[:, :compare_slot_count]

    visible_mask = gt_states[..., StateIndex.VISIBILITY] > 0.5
    existence_mask = gt_states[..., StateIndex.EXISTENCE] > 0.5
    image_hw = np.array([256.0, 144.0], dtype=np.float32)

    center_delta = pred_states[..., StateIndex.CENTER_X : StateIndex.CENTER_Y + 1] - gt_states[
        ..., StateIndex.CENTER_X : StateIndex.CENTER_Y + 1
    ]
    center_l2_norm = np.linalg.norm(center_delta, axis=-1)
    center_l2_px = np.linalg.norm(center_delta * image_hw[None, None, :], axis=-1)
    depth_abs = np.abs(pred_states[..., StateIndex.DEPTH] - gt_states[..., StateIndex.DEPTH])
    log_scale_abs = np.abs(pred_states[..., StateIndex.LOG_SCALE] - gt_states[..., StateIndex.LOG_SCALE])
    vel_l2 = np.linalg.norm(
        pred_states[..., StateIndex.VEL_X : StateIndex.VEL_Y + 1] - gt_states[..., StateIndex.VEL_X : StateIndex.VEL_Y + 1],
        axis=-1,
    )
    confidence_abs = np.abs(pred_states[..., StateIndex.CONFIDENCE] - gt_states[..., StateIndex.CONFIDENCE])
    existence_acc = (
        (pred_states[..., StateIndex.EXISTENCE] > 0.5) == existence_mask
    ).astype(np.float32)

    ious: list[float] = []
    for frame_idx in range(gt_boxes.shape[0]):
        for obj_idx in range(gt_boxes.shape[1]):
            if not visible_mask[frame_idx, obj_idx]:
                continue
            ious.append(_box_iou_xyxy(pred_boxes[frame_idx, obj_idx], gt_boxes[frame_idx, obj_idx]))

    pred_visible = pred_states[..., StateIndex.VISIBILITY] > 0.5
    gt_visible = visible_mask
    tp = int(np.logical_and(pred_visible, gt_visible).sum())
    fp = int(np.logical_and(pred_visible, np.logical_not(gt_visible)).sum())
    fn = int(np.logical_and(np.logical_not(pred_visible), gt_visible).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    visibility_f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-8)

    visible_points = int(visible_mask.sum())
    return {
        "compare_slot_count": int(compare_slot_count),
        "visible_points": visible_points,
        "box_iou_mean": float(np.mean(ious)) if ious else None,
        "center_l2_norm_mean": float(center_l2_norm[visible_mask].mean()) if visible_points else None,
        "center_l2_px_mean": float(center_l2_px[visible_mask].mean()) if visible_points else None,
        "depth_mae": float(depth_abs[visible_mask].mean()) if visible_points else None,
        "log_scale_mae": float(log_scale_abs[visible_mask].mean()) if visible_points else None,
        "vel_l2_mean": float(vel_l2[visible_mask].mean()) if visible_points else None,
        "visibility_f1": float(visibility_f1),
        "existence_acc": float(existence_acc.mean()),
        "confidence_mae": float(confidence_abs[existence_mask].mean()) if int(existence_mask.sum()) else None,
    }


def draw_overlay_frame(
    frame_chw: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    gt_states: np.ndarray,
    pred_states: np.ndarray,
) -> np.ndarray:
    image = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    canvas = cv2.cvtColor((image * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]

    def to_px(box: np.ndarray) -> tuple[int, int, int, int]:
        x0 = int(round(float(box[0]) * width))
        y0 = int(round(float(box[1]) * height))
        x1 = int(round(float(box[2]) * width))
        y1 = int(round(float(box[3]) * height))
        return x0, y0, x1, y1

    for obj_idx in range(gt_boxes.shape[0]):
        if float(gt_states[obj_idx, StateIndex.VISIBILITY]) > 0.5:
            x0, y0, x1, y1 = to_px(gt_boxes[obj_idx])
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)
        if float(pred_states[obj_idx, StateIndex.VISIBILITY]) > 0.5:
            x0, y0, x1, y1 = to_px(pred_boxes[obj_idx])
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def write_mp4(path: Path, frames_bgr: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_bgr.shape[1]), int(frames_bgr.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()


def overlay_video(
    future_frames: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    gt_states: np.ndarray,
    pred_states: np.ndarray,
) -> np.ndarray:
    frames = []
    for frame_idx in range(future_frames.shape[0]):
        frames.append(
            draw_overlay_frame(
                future_frames[frame_idx],
                gt_boxes[frame_idx],
                pred_boxes[frame_idx],
                gt_states[frame_idx],
                pred_states[frame_idx],
            )
        )
    return np.stack(frames, axis=0)


def aggregate_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "box_iou_mean",
        "center_l2_norm_mean",
        "center_l2_px_mean",
        "depth_mae",
        "log_scale_mae",
        "vel_l2_mean",
        "visibility_f1",
        "existence_acc",
        "confidence_mae",
    ]
    summary: dict[str, Any] = {
        "count": len(entries),
    }
    for key in numeric_keys:
        values = [float(item[key]) for item in entries if item.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def main() -> None:
    args = parse_args()
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    video_primary_families = {item.strip() for item in args.video_primary_families.split(",") if item.strip()}

    raw_split_root = args.raw_root / args.split
    episode_split_root = args.episode_root / args.split
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    selected = select_sample_dirs(raw_split_root, families, int(args.cases_per_family))
    cases = collect_episode_cases(episode_split_root, selected)

    sample_manifest = {
        "raw_root": str(args.raw_root),
        "episode_root": str(args.episode_root),
        "split": args.split,
        "families": families,
        "cases_per_family": int(args.cases_per_family),
        "modes": modes,
        "video_primary_families": sorted(video_primary_families),
        "cases": [
            {
                "case_id": case.case_id,
                "family": case.family,
                "sample_id": case.sample_id,
                "raw_sample_dir": str(case.raw_sample_dir),
                "episode_npz_path": str(case.episode_npz_path),
                "episode_json_path": str(case.episode_json_path),
                "window_index": int(case.window_index),
                "window_start": int(case.window_start),
                "frame_stride": int(case.frame_stride),
            }
            for case in cases
        ],
    }
    (args.output_dir / "sample_manifest.json").write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_case_results: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in cases:
        payload = load_episode_arrays(case.episode_npz_path)
        gt_states = payload["full_states"]
        gt_boxes = payload["full_boxes"]
        future_frames = payload["future_frames"]
        gt_future_states = payload["future_states"]
        gt_future_boxes = payload["future_boxes"]

        if "gt_boxes_proxy" in modes:
            proxy_states, proxy_boxes = gt_boxes_proxy_from_episode(payload)
            metrics = compute_state_metrics(
                proxy_states,
                proxy_boxes,
                gt_states,
                gt_boxes,
                compare_slot_count=int(gt_states.shape[1]),
            )
            result = {
                "mode": "gt_boxes_proxy",
                "case_id": case.case_id,
                "family": case.family,
                "sample_id": case.sample_id,
                "window_index": int(case.window_index),
                **metrics,
            }
            per_case_results.append(result)
            grouped[f"gt_boxes_proxy::{case.family}"].append(result)
            grouped["gt_boxes_proxy::overall"].append(result)

        if "video_primary_proxy" in modes and case.family in video_primary_families:
            pred_states, pred_boxes = video_primary_proxy_from_raw(case, target_frames=int(gt_states.shape[0]))
            compare_slot_count = 1
            metrics = compute_state_metrics(
                pred_states,
                pred_boxes,
                gt_states,
                gt_boxes,
                compare_slot_count=compare_slot_count,
            )
            result = {
                "mode": "video_primary_proxy",
                "case_id": case.case_id,
                "family": case.family,
                "sample_id": case.sample_id,
                "window_index": int(case.window_index),
                **metrics,
            }
            per_case_results.append(result)
            grouped[f"video_primary_proxy::{case.family}"].append(result)
            grouped["video_primary_proxy::overall"].append(result)

            pred_future_states = pred_states[int(payload["context_states"].shape[0]) :]
            pred_future_boxes = pred_boxes[int(payload["context_boxes"].shape[0]) :]
            overlay_frames = overlay_video(
                future_frames,
                gt_future_boxes[:, :1],
                pred_future_boxes[:, :1],
                gt_future_states[:, :1],
                pred_future_states[:, :1],
            )
            write_mp4(overlay_dir / f"{case.case_id}__video_primary_proxy.mp4", overlay_frames, fps=int(args.fps))

    metrics_path = args.output_dir / "metrics_per_case.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for item in per_case_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "manifest_path": str(args.output_dir / "sample_manifest.json"),
        "metrics_per_case_path": str(metrics_path),
        "overlay_dir": str(overlay_dir),
        "grouped": {key: aggregate_metrics(value) for key, value in sorted(grouped.items())},
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
