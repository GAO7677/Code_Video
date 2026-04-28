"""Window-level interaction summaries for oracle-state training windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _normalize_participants(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in values)


def _normalize_object_indices(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in values)


def _compress_env_events(events: Sequence[dict]) -> List[dict]:
    merged: List[dict] = []
    grouped = sorted(
        events,
        key=lambda ev: (
            tuple(int(x) for x in ev.get("participants", [])),
            tuple(int(x) for x in ev.get("object_indices", ev.get("participant_indices", []))),
            str(ev.get("environment_name", "")),
            int(ev.get("start_frame", -1)),
            int(ev.get("end_frame", -1)),
        ),
    )
    for ev in grouped:
        participants = _normalize_participants(ev.get("participants", []))
        object_indices = _normalize_object_indices(ev.get("object_indices", ev.get("participant_indices", [])))
        env_name = str(ev.get("environment_name", "")).strip() or "environment"
        start_frame = int(ev.get("start_frame", -1))
        end_frame = int(ev.get("end_frame", start_frame))
        if merged:
            last = merged[-1]
            same_key = (
                last["participants"] == participants
                and last["object_indices"] == object_indices
                and last["environment_name"] == env_name
            )
            if same_key and start_frame <= int(last["end_frame"]) + 1:
                last["end_frame"] = max(int(last["end_frame"]), end_frame)
                continue
        merged.append(
            {
                "kind": "object_environment",
                "participants": participants,
                "object_indices": object_indices,
                "environment_name": env_name,
                "window_type": "environment_contact",
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        )
    return merged


def _compress_object_object_events(events: Sequence[dict]) -> List[dict]:
    merged: List[dict] = []
    grouped = sorted(
        events,
        key=lambda ev: (
            tuple(int(x) for x in ev.get("participants", [])),
            str(ev.get("window_type", "")),
            int(ev.get("start_frame", -1)),
            int(ev.get("end_frame", -1)),
        ),
    )
    for ev in grouped:
        participants = _normalize_participants(ev.get("participants", []))
        object_indices = _normalize_object_indices(ev.get("object_indices", ev.get("participant_indices", participants)))
        window_type = str(ev.get("window_type", "")).strip() or "object_object_contact"
        start_frame = int(ev.get("start_frame", -1))
        end_frame = int(ev.get("end_frame", start_frame))
        if merged:
            last = merged[-1]
            same_key = (
                last["participants"] == participants
                and last["window_type"] == window_type
            )
            if same_key and start_frame <= int(last["end_frame"]) + 1:
                last["end_frame"] = max(int(last["end_frame"]), end_frame)
                continue
        merged.append(
            {
                "kind": "object_object",
                "participants": participants,
                "object_indices": object_indices,
                "environment_name": "",
                "window_type": window_type,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        )
    return merged


def load_interaction_episodes(source_sample_dir: Path) -> List[dict]:
    event_path = source_sample_dir / "physics" / "event_windows.json"
    if not event_path.exists():
        return []
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    env_events: List[dict] = []
    obj_obj_events: List[dict] = []
    for ev in payload:
        participants = [int(x) for x in ev.get("participants", [])]
        if any(x < 0 for x in participants):
            env_events.append(ev)
        elif len(participants) >= 2:
            obj_obj_events.append(ev)
    return _compress_object_object_events(obj_obj_events) + _compress_env_events(env_events)


def overlaps(frame_start: int, frame_end_exclusive: int, event_start: int, event_end_inclusive: int) -> bool:
    return max(int(frame_start), int(event_start)) <= min(int(frame_end_exclusive) - 1, int(event_end_inclusive))


def collision_count_bucket(count: int) -> str:
    count = int(count)
    if count <= 0:
        return "c0"
    if count == 1:
        return "c1"
    return "c2plus"


def collision_type_bucket(obj_env_count: int, obj_obj_count: int) -> str:
    if int(obj_env_count) <= 0 and int(obj_obj_count) <= 0:
        return "none"
    if int(obj_env_count) > 0 and int(obj_obj_count) <= 0:
        return "env_only"
    if int(obj_env_count) <= 0 and int(obj_obj_count) > 0:
        return "obj_obj_only"
    return "mixed"


def summarize_window_range(
    episodes: Sequence[dict],
    frame_start: int,
    frame_end_exclusive: int,
) -> Dict[str, object]:
    overlapping = [
        episode
        for episode in episodes
        if overlaps(
            frame_start=frame_start,
            frame_end_exclusive=frame_end_exclusive,
            event_start=int(episode["start_frame"]),
            event_end_inclusive=int(episode["end_frame"]),
        )
    ]
    obj_env = [episode for episode in overlapping if str(episode["kind"]) == "object_environment"]
    obj_obj = [episode for episode in overlapping if str(episode["kind"]) == "object_object"]
    subtype_names = sorted(
        {
            (
                f"object_environment:{episode['environment_name']}"
                if str(episode["kind"]) == "object_environment"
                else f"object_object:{episode['window_type']}"
            )
            for episode in overlapping
        }
    )
    return {
        "frame_start": int(frame_start),
        "frame_end_exclusive": int(frame_end_exclusive),
        "collision_episode_count": int(len(overlapping)),
        "object_environment_count": int(len(obj_env)),
        "object_object_count": int(len(obj_obj)),
        "collision_type_bucket": collision_type_bucket(len(obj_env), len(obj_obj)),
        "collision_count_bucket": collision_count_bucket(len(overlapping)),
        "collision_subtypes": subtype_names,
        "episodes": [
            {
                "kind": str(episode["kind"]),
                "participants": [int(x) for x in episode["participants"]],
                "object_indices": [int(x) for x in episode["object_indices"]],
                "environment_name": str(episode["environment_name"]),
                "window_type": str(episode["window_type"]),
                "start_frame": int(episode["start_frame"]),
                "end_frame": int(episode["end_frame"]),
            }
            for episode in overlapping
        ],
    }


def infer_window_interactions(meta: Dict[str, object]) -> Dict[str, object]:
    source_sample_dir = Path(str(meta.get("source_sample_dir", "")))
    start_index = int(meta.get("start_index", 0))
    context_len = int(meta.get("context_len", 0))
    future_len = int(meta.get("future_len", 0))
    object_count = len(meta.get("objects", []))
    if object_count <= 0:
        object_count = int(meta.get("num_objects", 0) or 0)

    full_start = start_index
    future_start = start_index + context_len
    future_end = future_start + future_len
    full_end = future_end
    episodes = load_interaction_episodes(source_sample_dir)

    full_summary = summarize_window_range(episodes, full_start, full_end)
    future_summary = summarize_window_range(episodes, future_start, future_end)
    future_bucket = (
        f"obj{int(object_count)}__{future_summary['collision_count_bucket']}__{future_summary['collision_type_bucket']}"
    )
    return {
        "object_count": int(object_count),
        "full_window": full_summary,
        "future_window": future_summary,
        "future_bucket": future_bucket,
        "source_event_episode_count": int(len(episodes)),
    }
