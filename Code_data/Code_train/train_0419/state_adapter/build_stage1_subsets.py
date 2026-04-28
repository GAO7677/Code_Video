#!/usr/bin/env python3
"""该脚本用于从 /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases 提取 Stage-1A/1B 状态预测子集；输入为样本目录及物理标注，输出为 /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/stage1_subsets_v1 下的窗口数据和 summary.json。"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


STAGE1A_FUTURE_CANDIDATES = [8, 12, 16]
STAGE1A_CONTEXT_LEN = 8
STAGE1A_SAFETY_MARGIN = 2
WINDOW_STRIDE = 4

STAGE1B_CONTEXT_LEN = 8
STAGE1B_FUTURE_CANDIDATES = [8, 12, 16, 24, 41]
STAGE1B_SAFETY_MARGIN = 2

CONTACT_PHASE_IDS = {1, 2, 3}


@dataclass(frozen=True)
class WindowCandidate:
    start_index: int
    context_len: int
    future_len: int
    x_idx: np.ndarray
    y_idx: np.ndarray
    future_main_visibility_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-1A/1B state predictor subsets from synthetic rigid data.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/stage1_subsets_v1"),
    )
    parser.add_argument("--max_source_samples", type=int, default=0, help="0 means scan all source samples.")
    parser.add_argument("--max_windows_per_subset", type=int, default=30, help="Stop after enough accepted windows per subset.")
    parser.add_argument("--sample_filter", type=str, default="")
    parser.add_argument(
        "--count_buckets",
        type=str,
        default="count_01",
        help="Comma-separated object-count buckets to include, e.g. count_01,count_02. Default only keeps count_01.",
    )
    parser.add_argument(
        "--future_main_visibility_threshold",
        type=float,
        default=0.5,
        help="Require the main object to be visible for at least this ratio of future frames.",
    )
    return parser.parse_args()


def find_samples(dataset_root: Path, sample_filter: str, count_buckets: Sequence[str]) -> List[Path]:
    samples = sorted(path.parent for path in dataset_root.rglob("metadata.json"))
    samples = [p for p in samples if (p / "physics" / "anchor_targets.npz").exists()]
    wanted_buckets = {bucket.strip() for bucket in count_buckets if bucket.strip()}
    if wanted_buckets:
        samples = [p for p in samples if any(bucket in p.parts for bucket in wanted_buckets)]
    if sample_filter:
        samples = [p for p in samples if sample_filter in str(p)]
    return samples


def load_raw_state(sample_dir: Path, fps: float) -> Dict[str, np.ndarray]:
    data = np.load(sample_dir / "physics" / "anchor_targets.npz")
    object_ids = data["object_ids"].astype(np.int32)
    seg_ids = data["seg_ids"].astype(np.int32)
    com_uv = data["com_uv"].astype(np.float32)
    center_depth = data["center_depth"].astype(np.float32)
    bbox_xyxy = data["bbox_xyxy"].astype(np.float32)
    visibility_mask = data["visibility_mask"].astype(np.uint8)

    dt = np.float32(1.0 / float(fps))
    x1 = bbox_xyxy[..., 0]
    y1 = bbox_xyxy[..., 1]
    x2 = bbox_xyxy[..., 2]
    y2 = bbox_xyxy[..., 3]
    width = np.maximum(0.0, x2 - x1).astype(np.float32)
    height = np.maximum(0.0, y2 - y1).astype(np.float32)

    u = com_uv[..., 0]
    v = com_uv[..., 1]
    d = center_depth
    du = np.zeros_like(u, dtype=np.float32)
    dv = np.zeros_like(v, dtype=np.float32)
    dd = np.zeros_like(d, dtype=np.float32)
    if u.shape[0] > 1:
        du[1:] = (u[1:] - u[:-1]) / dt
        dv[1:] = (v[1:] - v[:-1]) / dt
        dd[1:] = (d[1:] - d[:-1]) / dt

    vis = visibility_mask.astype(np.float32)
    state_raw = np.stack([u, v, d, width, height, du, dv, dd, vis], axis=-1).astype(np.float32)
    return {
        "object_ids": object_ids,
        "seg_ids": seg_ids,
        "visibility_mask": visibility_mask,
        "state_raw": state_raw,
        "dt": np.asarray(dt, dtype=np.float32),
    }


def normalize_state(
    state_raw: np.ndarray,
    width: float,
    height: float,
    depth_near: float,
    depth_far: float,
) -> np.ndarray:
    state = state_raw.copy().astype(np.float32)
    depth_range = max(float(depth_far) - float(depth_near), 1e-6)

    state[..., 0] /= float(width)
    state[..., 1] /= float(height)
    state[..., 3] /= float(width)
    state[..., 4] /= float(height)
    state[..., 5] /= float(width)
    state[..., 6] /= float(height)
    state[..., 2] = (state[..., 2] - float(depth_near)) / depth_range
    state[..., 7] /= depth_range
    state[..., 8] = np.clip(state[..., 8], 0.0, 1.0)

    invisible = state[..., 8] < 0.5
    state[..., 0:8][invisible] = 0.0
    return state


def load_contact_annotations(sample_dir: Path, num_objects: int, T: int) -> Dict[str, object]:
    frame_phase = np.load(sample_dir / "physics" / "frame_phase.npy").astype(np.int32)
    contact_graph = np.load(sample_dir / "physics" / "contact_graph.npy").astype(np.uint8)
    event_windows = json.loads((sample_dir / "physics" / "event_windows.json").read_text(encoding="utf-8"))

    env_contact = np.zeros((T, num_objects), dtype=np.uint8)
    object_contact_events = []
    all_contact_start_frames = []
    env_contact_start_frames = []
    object_contact_start_frames = []

    for event in event_windows:
        sf = int(event.get("start_frame", -1))
        ef = int(event.get("end_frame", sf))
        participants = [int(x) for x in event.get("participants", [])]
        object_indices = [int(x) for x in event.get("object_indices", [])]
        if sf >= 0:
            all_contact_start_frames.append(sf)
        if any(pid < 0 for pid in participants):
            env_contact_start_frames.append(sf)
            obj_idx = object_indices[0] if object_indices else -1
            if 0 <= obj_idx < num_objects:
                env_contact[max(sf, 0) : min(ef, T - 1) + 1, obj_idx] = 1
        elif sf >= 0:
            object_contact_start_frames.append(sf)
            object_contact_events.append(
                {
                    "start_frame": sf,
                    "end_frame": ef,
                    "participants": participants,
                    "object_indices": object_indices,
                    "window_type": str(event.get("window_type", "")),
                }
            )

    return {
        "frame_phase": frame_phase,
        "contact_graph": contact_graph,
        "event_windows": event_windows,
        "env_contact": env_contact,
        "all_contact_start_frames": all_contact_start_frames,
        "env_contact_start_frames": env_contact_start_frames,
        "object_contact_start_frames": object_contact_start_frames,
        "object_contact_events": object_contact_events,
    }


def first_contact_frame_stage1a(ann: Dict[str, object], T: int) -> Optional[int]:
    frame_phase = ann["frame_phase"]
    contact_graph = ann["contact_graph"]
    candidates: List[int] = []

    phase_hit = np.where(np.isin(frame_phase, list(CONTACT_PHASE_IDS)))[0]
    if phase_hit.size > 0:
        candidates.append(int(phase_hit[0]))

    graph_idx = np.argwhere(np.asarray(contact_graph).sum(axis=(1, 2)) > 0)
    if graph_idx.size > 0:
        candidates.append(int(graph_idx[0, 0]))

    candidates.extend(int(x) for x in ann["all_contact_start_frames"] if int(x) >= 0)
    return min(candidates) if candidates else None


def overlaps(frame_start: int, frame_end_exclusive: int, event_start: int, event_end_inclusive: int) -> bool:
    return max(frame_start, event_start) <= min(frame_end_exclusive - 1, event_end_inclusive)


def future_has_object_object_contact(ann: Dict[str, object], future_start: int, future_end: int) -> Dict[str, bool]:
    contact_graph = ann["contact_graph"]
    # contact_graph only stores object-object contact; environment contacts are in event_windows with participant -1.
    graph_hit = bool(np.asarray(contact_graph[future_start:future_end]).sum() > 0)
    event_hit = False
    for event in ann["object_contact_events"]:
        if overlaps(future_start, future_end, int(event["start_frame"]), int(event["end_frame"])):
            event_hit = True
            break
    return {
        "phase_hit": False,
        "graph_hit": graph_hit,
        "event_hit": event_hit,
        "any_hit": graph_hit or event_hit,
    }


def rgb_frame_paths(sample_dir: Path, frame_indices: Sequence[int]) -> List[str]:
    return [str(sample_dir / "rgb" / f"frame_{int(idx):03d}.png") for idx in frame_indices]


def window_has_visible_object_every_frame(visibility_mask: np.ndarray, start: int, end: int) -> bool:
    if end <= start:
        return False
    vis = np.asarray(visibility_mask[start:end])
    if vis.ndim != 2:
        return False
    return bool(np.all(vis.sum(axis=1) > 0))


def resolve_main_object_index(metadata: Dict[str, object], object_ids: np.ndarray) -> int:
    explicit_main_index = metadata.get("main_object_index")
    if explicit_main_index is not None:
        try:
            explicit_main_index = int(explicit_main_index)
        except (TypeError, ValueError):
            explicit_main_index = None
        if explicit_main_index is not None and 0 <= explicit_main_index < int(object_ids.shape[0]):
            return explicit_main_index

    object_id_to_index = {int(obj_id): idx for idx, obj_id in enumerate(object_ids.tolist())}
    objects = metadata.get("objects", [])
    source_main_id = str(metadata.get("object_id", "")).strip()

    def try_match(predicate) -> Optional[int]:
        for obj in objects:
            if not isinstance(obj, dict) or not predicate(obj):
                continue
            obj_key = obj.get("object_id")
            if obj_key is None:
                continue
            idx = object_id_to_index.get(int(obj_key))
            if idx is not None:
                return idx
        return None

    matched = try_match(lambda obj: str(obj.get("source_object_id", "")).strip() == source_main_id and source_main_id != "")
    if matched is not None:
        return matched
    matched = try_match(lambda obj: str(obj.get("role", "")) == "target")
    if matched is not None:
        return matched
    matched = try_match(lambda obj: str(obj.get("dataset_source", "")) == "PhysXNet")
    if matched is not None:
        return matched
    return 0


def future_main_object_visibility_ok(
    visibility_mask: np.ndarray,
    start: int,
    end: int,
    main_object_index: int,
    threshold: float,
) -> Tuple[bool, float]:
    if end <= start:
        return False, 0.0
    vis = np.asarray(visibility_mask[start:end, main_object_index]).astype(np.float32)
    if vis.size == 0:
        return False, 0.0
    ratio = float(vis.mean())
    return bool(vis.sum() > 0 and ratio >= float(threshold)), ratio


def iter_window_candidates(
    *,
    visibility_mask: np.ndarray,
    context_len: int,
    future_candidates: Sequence[int],
    valid_end: int,
    main_object_index: int,
    future_main_visibility_threshold: float,
) -> List[WindowCandidate]:
    candidates: List[WindowCandidate] = []
    if valid_end < context_len + min(int(x) for x in future_candidates):
        return candidates

    for future_len in future_candidates:
        future_len = int(future_len)
        max_start = int(valid_end) - int(context_len) - future_len
        if max_start < 0:
            continue
        for start_index in range(0, max_start + 1, WINDOW_STRIDE):
            c0 = int(start_index)
            c1 = c0 + int(context_len)
            f0 = c1
            f1 = f0 + future_len
            if not window_has_visible_object_every_frame(visibility_mask, c0, c1):
                continue
            future_visible_ok, future_main_visibility_ratio = future_main_object_visibility_ok(
                visibility_mask=visibility_mask,
                start=f0,
                end=f1,
                main_object_index=main_object_index,
                threshold=future_main_visibility_threshold,
            )
            if not future_visible_ok:
                continue
            candidates.append(
                WindowCandidate(
                    start_index=c0,
                    context_len=int(context_len),
                    future_len=future_len,
                    x_idx=np.arange(c0, c1, dtype=np.int32),
                    y_idx=np.arange(f0, f1, dtype=np.int32),
                    future_main_visibility_ratio=float(future_main_visibility_ratio),
                )
            )
    return candidates


def build_window_payload(
    *,
    state_raw: np.ndarray,
    state_norm: np.ndarray,
    object_ids: np.ndarray,
    seg_ids: np.ndarray,
    visibility_mask: np.ndarray,
    dt: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "object_ids": object_ids,
        "seg_ids": seg_ids,
        "visibility_mask": visibility_mask,
        "state_raw": state_raw,
        "state_norm": state_norm,
        "x_state": state_norm[x_idx[0] : x_idx[-1] + 1],
        "y_state": state_norm[y_idx[0] : y_idx[-1] + 1],
        "x_visibility": visibility_mask[x_idx[0] : x_idx[-1] + 1],
        "y_visibility": visibility_mask[y_idx[0] : y_idx[-1] + 1],
        "x_frame_indices": x_idx,
        "y_frame_indices": y_idx,
        "dt": dt,
    }


def build_common_meta(
    *,
    metadata: Dict[str, object],
    sample_dir: Path,
    candidate: WindowCandidate,
    main_object_index: int,
    future_main_visibility_threshold: float,
) -> Dict[str, object]:
    return {
        "source_scene_id": metadata["scene_id"],
        "source_sample_dir": str(sample_dir),
        "start_index": int(candidate.start_index),
        "context_len": int(candidate.context_len),
        "future_len": int(candidate.future_len),
        "main_object_index": int(main_object_index),
        "future_main_visibility_threshold": float(future_main_visibility_threshold),
        "future_main_visibility_ratio": float(candidate.future_main_visibility_ratio),
        "x_frame_paths": rgb_frame_paths(sample_dir, candidate.x_idx),
        "y_frame_paths": rgb_frame_paths(sample_dir, candidate.y_idx),
        "objects": metadata.get("objects", []),
        "resolution": metadata.get("resolution"),
        "camera_intrinsics": metadata.get("camera_intrinsics"),
    }


def export_window(
    subset_root: Path,
    rel_sample: Path,
    window_name: str,
    payload: Dict[str, np.ndarray],
    meta_payload: Dict[str, object],
) -> Path:
    out_dir = subset_root / rel_sample / window_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "state_pair.npz", **payload)
    (out_dir / "pair_meta.json").write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def stage1a_windows(
    sample_dir: Path,
    dataset_root: Path,
    subset_root: Path,
    metadata: Dict[str, object],
    state_raw: np.ndarray,
    state_norm: np.ndarray,
    object_ids: np.ndarray,
    seg_ids: np.ndarray,
    visibility_mask: np.ndarray,
    dt: np.ndarray,
    ann: Dict[str, object],
    future_main_visibility_threshold: float,
) -> List[Dict[str, object]]:
    T = state_norm.shape[0]
    first_contact = first_contact_frame_stage1a(ann, T)
    valid_end = T if first_contact is None else max(0, int(first_contact) - STAGE1A_SAFETY_MARGIN)

    results = []
    rel_sample = sample_dir.relative_to(dataset_root)
    main_object_index = resolve_main_object_index(metadata, object_ids)
    candidates = iter_window_candidates(
        visibility_mask=visibility_mask,
        context_len=STAGE1A_CONTEXT_LEN,
        future_candidates=STAGE1A_FUTURE_CANDIDATES,
        valid_end=valid_end,
        main_object_index=main_object_index,
        future_main_visibility_threshold=future_main_visibility_threshold,
    )
    for candidate in candidates:
        payload = build_window_payload(
            state_raw=state_raw,
            state_norm=state_norm,
            object_ids=object_ids,
            seg_ids=seg_ids,
            visibility_mask=visibility_mask,
            dt=dt,
            x_idx=candidate.x_idx,
            y_idx=candidate.y_idx,
        )
        meta_payload = build_common_meta(
            metadata=metadata,
            sample_dir=sample_dir,
            candidate=candidate,
            main_object_index=main_object_index,
            future_main_visibility_threshold=future_main_visibility_threshold,
        )
        meta_payload.update(
            {
                "first_contact_frame": None if first_contact is None else int(first_contact),
                "valid_end": int(valid_end),
                "is_precontact_strict": True,
                "subset_rule": "future must end before the earliest detected contact frame minus safety margin",
            }
        )
        window_name = (
            f"window_s{candidate.start_index:04d}_ctx{candidate.context_len:02d}_fut{candidate.future_len:02d}"
        )
        out_dir = export_window(subset_root, rel_sample, window_name, payload, meta_payload)
        results.append(
            {
                "out_dir": str(out_dir),
                "start_index": int(candidate.start_index),
                "future_len": int(candidate.future_len),
            }
        )
    return results


def stage1b_windows(
    sample_dir: Path,
    dataset_root: Path,
    subset_root: Path,
    metadata: Dict[str, object],
    state_raw: np.ndarray,
    state_norm: np.ndarray,
    object_ids: np.ndarray,
    seg_ids: np.ndarray,
    visibility_mask: np.ndarray,
    dt: np.ndarray,
    ann: Dict[str, object],
    future_main_visibility_threshold: float,
) -> List[Dict[str, object]]:
    T = state_norm.shape[0]
    min_total_needed = STAGE1B_CONTEXT_LEN + min(STAGE1B_FUTURE_CANDIDATES)
    if T < min_total_needed:
        return []

    rel_sample = sample_dir.relative_to(dataset_root)
    main_object_index = resolve_main_object_index(metadata, object_ids)
    results = []
    candidates = iter_window_candidates(
        visibility_mask=visibility_mask,
        context_len=STAGE1B_CONTEXT_LEN,
        future_candidates=STAGE1B_FUTURE_CANDIDATES,
        valid_end=T,
        main_object_index=main_object_index,
        future_main_visibility_threshold=future_main_visibility_threshold,
    )
    for candidate in candidates:
        obj_contact = future_has_object_object_contact(ann, int(candidate.y_idx[0]), int(candidate.y_idx[-1]) + 1)
        if obj_contact["any_hit"]:
            continue
        payload = build_window_payload(
            state_raw=state_raw,
            state_norm=state_norm,
            object_ids=object_ids,
            seg_ids=seg_ids,
            visibility_mask=visibility_mask,
            dt=dt,
            x_idx=candidate.x_idx,
            y_idx=candidate.y_idx,
        )
        meta_payload = build_common_meta(
            metadata=metadata,
            sample_dir=sample_dir,
            candidate=candidate,
            main_object_index=main_object_index,
            future_main_visibility_threshold=future_main_visibility_threshold,
        )
        meta_payload.update(
            {
                "is_simple_dynamics": True,
                "allows_environment_contact": True,
                "forbid_object_object_contact": True,
                "object_object_contact_filter": obj_contact,
                "subset_rule": "future may contain environment contact but must not overlap any object-object contact",
            }
        )
        window_name = (
            f"window_s{candidate.start_index:04d}_ctx{candidate.context_len:02d}_fut{candidate.future_len:02d}"
        )
        out_dir = export_window(subset_root, rel_sample, window_name, payload, meta_payload)
        results.append(
            {
                "out_dir": str(out_dir),
                "start_index": int(candidate.start_index),
                "future_len": int(candidate.future_len),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    dataset_train_root = args.dataset_root / "train"
    samples = find_samples(dataset_train_root, args.sample_filter, args.count_buckets.split(","))
    if int(args.max_source_samples) > 0:
        samples = samples[: int(args.max_source_samples)]
    if not samples:
        raise RuntimeError(f"No samples found under {dataset_train_root}")

    subset_a_root = args.out_root / "stage1a_precontact_strict"
    subset_b_root = args.out_root / "stage1b_simple_dynamics"
    subset_a_root.mkdir(parents=True, exist_ok=True)
    subset_b_root.mkdir(parents=True, exist_ok=True)

    manifest_a = {"subset": "stage1a_precontact_strict", "accepted": [], "skipped": []}
    manifest_b = {"subset": "stage1b_simple_dynamics", "accepted": [], "skipped": []}
    accepted_a = 0
    accepted_b = 0

    for sample_dir in samples:
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        fps = float(metadata.get("fps", metadata.get("video_fps", 12)))
        if fps <= 0:
            fps = 12.0
        raw = load_raw_state(sample_dir, fps)
        state_raw = raw["state_raw"]
        visibility_mask = raw["visibility_mask"]
        object_ids = raw["object_ids"]
        seg_ids = raw["seg_ids"]
        T = state_raw.shape[0]

        cam = metadata["camera_intrinsics"]
        width, height = map(float, metadata["resolution"])
        state_norm = normalize_state(
            state_raw=state_raw,
            width=width,
            height=height,
            depth_near=float(cam["near"]),
            depth_far=float(cam["far"]),
        )
        ann = load_contact_annotations(sample_dir=sample_dir, num_objects=int(object_ids.shape[0]), T=T)

        if accepted_a < int(args.max_windows_per_subset):
            windows_a = stage1a_windows(
                sample_dir,
                dataset_train_root,
                subset_a_root,
                metadata,
                state_raw,
                state_norm,
                object_ids,
                seg_ids,
                visibility_mask,
                raw["dt"],
                ann,
                args.future_main_visibility_threshold,
            )
            if windows_a:
                for item in windows_a:
                    manifest_a["accepted"].append({"sample_dir": str(sample_dir), **item})
                accepted_a += len(windows_a)
                print(f"STAGE1A_ACCEPT sample={sample_dir} windows={len(windows_a)}")
            else:
                manifest_a["skipped"].append({"sample_dir": str(sample_dir), "reason": "no_valid_precontact_window"})

        if accepted_b < int(args.max_windows_per_subset):
            windows_b = stage1b_windows(
                sample_dir,
                dataset_train_root,
                subset_b_root,
                metadata,
                state_raw,
                state_norm,
                object_ids,
                seg_ids,
                visibility_mask,
                raw["dt"],
                ann,
                args.future_main_visibility_threshold,
            )
            if windows_b:
                for item in windows_b:
                    manifest_b["accepted"].append({"sample_dir": str(sample_dir), **item})
                accepted_b += len(windows_b)
                print(f"STAGE1B_ACCEPT sample={sample_dir} windows={len(windows_b)}")
            else:
                manifest_b["skipped"].append({"sample_dir": str(sample_dir), "reason": "no_valid_simple_dynamics_window"})

        if accepted_a >= int(args.max_windows_per_subset) and accepted_b >= int(args.max_windows_per_subset):
            break

    (subset_a_root / "manifest.json").write_text(json.dumps(manifest_a, ensure_ascii=False, indent=2), encoding="utf-8")
    (subset_b_root / "manifest.json").write_text(json.dumps(manifest_b, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "dataset_root": str(args.dataset_root),
        "out_root": str(args.out_root),
        "stage1a_windows": len(manifest_a["accepted"]),
        "stage1b_windows": len(manifest_b["accepted"]),
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE stage1a={len(manifest_a['accepted'])} stage1b={len(manifest_b['accepted'])}")


if __name__ == "__main__":
    main()
