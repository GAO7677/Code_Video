#!/usr/bin/env python3
"""Fix environment collision records so frame-0 support contact is not treated as collision onset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_env_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    prev_by_key: dict[tuple[int, str], int] = {}
    for item in sorted(records, key=lambda x: (int(x.get("start_frame", x.get("frame_idx", -1))), int(x.get("event_id", 10**9)))):
        participants = item.get("participants", [])
        if len(participants) != 2 or -1 not in participants:
            filtered.append(item)
            continue
        obj_idx = int(item.get("object_indices", [participants[0], -1])[0])
        env_name = str(item.get("environment_name") or "environment")
        start = int(item.get("start_frame", item.get("frame_idx", -1)))
        if start <= 0:
            # frame-0 support contact is not a collision onset
            if start == 0:
                prev_by_key[(obj_idx, env_name)] = 0
            continue
        prev = prev_by_key.get((obj_idx, env_name))
        if prev is not None and start == prev + 1:
            prev_by_key[(obj_idx, env_name)] = start
            continue
        prev_by_key[(obj_idx, env_name)] = start
        filtered.append(item)

    for new_id, item in enumerate(filtered):
        if isinstance(item, dict) and "event_id" in item:
            item["event_id"] = int(new_id)
        if isinstance(item, dict) and "window_id" in item:
            item["window_id"] = int(new_id)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dirs", nargs="+")
    args = parser.parse_args()

    for sample_dir_str in args.sample_dirs:
        sample_dir = Path(sample_dir_str)
        physics_dir = sample_dir / "physics"
        collision_path = physics_dir / "collision_events.json"
        event_windows_path = physics_dir / "event_windows.json"
        collision_events = load_json(collision_path)
        event_windows = load_json(event_windows_path)
        fixed_collision = compact_env_events(collision_events)
        fixed_windows = compact_env_events(event_windows)
        write_json(collision_path, fixed_collision)
        write_json(event_windows_path, fixed_windows)
        print(json.dumps({
            "sample_dir": str(sample_dir),
            "collision_events": len(collision_events),
            "collision_events_fixed": len(fixed_collision),
            "event_windows": len(event_windows),
            "event_windows_fixed": len(fixed_windows),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
