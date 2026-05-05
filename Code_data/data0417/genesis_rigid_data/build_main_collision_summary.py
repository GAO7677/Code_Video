#!/usr/bin/env python3
"""Build a compact main-collision summary and visualization for Genesis rigid samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_meta_path(sample_dir: Path) -> Path:
    for name in ("meta.json", "metadata.json"):
        path = sample_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing meta.json/metadata.json under {sample_dir}")


def object_name(obj: dict[str, Any]) -> str:
    return str(obj.get("name") or obj.get("source_object_id") or f"obj{obj.get('object_id', '?')}")


def candidate_pair_from_roles(meta: dict[str, Any]) -> tuple[int, int] | None:
    objects = meta.get("objects", [])
    target_idx = None
    initiator_idx = None
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        role = str(obj.get("role", ""))
        if role == "target" and target_idx is None:
            target_idx = idx
        if role == "initiator" and initiator_idx is None:
            initiator_idx = idx
    if target_idx is not None and initiator_idx is not None:
        return target_idx, initiator_idx
    return None


def first_object_object_event(event_windows: list[dict[str, Any]]) -> dict[str, Any] | None:
    pair_events = [
        item for item in event_windows
        if isinstance(item, dict)
        and len(item.get("participants", [])) == 2
        and int(item["participants"][0]) >= 0
        and int(item["participants"][1]) >= 0
    ]
    if not pair_events:
        return None
    pair_events.sort(key=lambda x: (int(x.get("start_frame", 10**9)), int(x.get("event_id", 10**9))))
    return pair_events[0]


def contiguous_onsets(frames: list[int]) -> list[int]:
    if not frames:
        return []
    frames = sorted(int(x) for x in frames)
    onsets: list[int] = [frames[0]] if frames[0] > 0 else []
    for prev, cur in zip(frames[:-1], frames[1:]):
        if cur != prev + 1 and cur > 0:
            onsets.append(cur)
    return onsets


def summarize_object_object_onsets(contact_graph: np.ndarray, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    n = int(contact_graph.shape[1])
    for i in range(n):
        for j in range(i + 1, n):
            active = np.where(contact_graph[:, i, j] > 0)[0].astype(int).tolist()
            onsets = contiguous_onsets(active)
            records.append(
                {
                    "pair": [i, j],
                    "names": [
                        object_name(objects[i]) if i < len(objects) and isinstance(objects[i], dict) else f"obj{i}",
                        object_name(objects[j]) if j < len(objects) and isinstance(objects[j], dict) else f"obj{j}",
                    ],
                    "contact_frames": active,
                    "onset_frames": onsets,
                }
            )
    return records


def summarize_environment_onsets(event_windows: list[dict[str, Any]], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[int]] = {}
    for item in event_windows:
        if not isinstance(item, dict):
            continue
        parts = item.get("participants", [])
        if len(parts) != 2 or -1 not in parts:
            continue
        obj_idx = int(parts[0]) if int(parts[0]) >= 0 else int(parts[1])
        env_name = str(item.get("environment_name") or "environment")
        frame = int(item.get("start_frame", item.get("frame_idx", -1)))
        if frame < 0:
            continue
        grouped.setdefault((obj_idx, env_name), []).append(frame)

    records: list[dict[str, Any]] = []
    for (obj_idx, env_name), frames in sorted(grouped.items()):
        records.append(
            {
                "object_index": int(obj_idx),
                "object_name": object_name(objects[obj_idx]) if obj_idx < len(objects) and isinstance(objects[obj_idx], dict) else f"obj{obj_idx}",
                "environment_name": env_name,
                "contact_frames": sorted(int(x) for x in frames),
                "onset_frames": contiguous_onsets(frames),
            }
        )
    return records


def closest_pair_by_distance(rigid: dict[str, np.ndarray], pair: tuple[int, int] | None = None) -> dict[str, Any] | None:
    com_pos = np.asarray(rigid["com_pos"], dtype=np.float32)
    vis = np.asarray(rigid["visibility_mask"], dtype=np.uint8)
    n = int(com_pos.shape[1])
    pairs = [pair] if pair is not None else [(i, j) for i in range(n) for j in range(i + 1, n)]
    best = None
    for i, j in pairs:
        mask = (vis[:, i] > 0) & (vis[:, j] > 0)
        if not np.any(mask):
            continue
        d = np.linalg.norm(com_pos[:, i] - com_pos[:, j], axis=-1)
        valid_idx = np.where(mask)[0]
        best_frame = int(valid_idx[np.argmin(d[valid_idx])])
        best_dist = float(d[best_frame])
        if best is None or best_dist < best["distance"]:
            best = {"pair": [int(i), int(j)], "frame": best_frame, "distance": best_dist}
    return best


def summarize_sample(sample_dir: Path) -> dict[str, Any]:
    meta = load_json(resolve_meta_path(sample_dir))
    physics_dir = sample_dir / "physics"
    rigid_npz = np.load(physics_dir / "rigid_kinematics.npz", allow_pickle=True)
    rigid = {key: np.asarray(rigid_npz[key]) for key in rigid_npz.files}
    contact_graph = np.asarray(np.load(physics_dir / "contact_graph.npy"), dtype=np.uint8)
    contact_impulse = np.asarray(np.load(physics_dir / "contact_impulse.npy"), dtype=np.float32)
    event_windows = load_json(physics_dir / "event_windows.json")

    role_pair = candidate_pair_from_roles(meta)
    role_pair_event = None
    if role_pair is not None:
        for item in event_windows:
            parts = item.get("participants", [])
            if sorted(parts) == sorted(role_pair):
                role_pair_event = item
                break

    first_pair_event = first_object_object_event(event_windows)
    closest_role_pair = closest_pair_by_distance(rigid, role_pair)
    closest_any_pair = closest_pair_by_distance(rigid, None)

    objects = meta.get("objects", [])
    n = int(rigid["object_ids"].shape[0])
    pair_records = summarize_object_object_onsets(contact_graph, objects)
    for item in pair_records:
        i, j = item["pair"]
        item["impulse_max"] = float(contact_impulse[:, i, j].max()) if contact_impulse.size else 0.0

    environment_records = summarize_environment_onsets(event_windows, objects)

    summary = {
        "scene_id": str(meta.get("scene_id", sample_dir.name)),
        "sample_dir": str(sample_dir),
        "num_objects": int(meta.get("num_objects", n)),
        "object_roles": [
            {
                "index": idx,
                "role": str(obj.get("role", "unknown")) if isinstance(obj, dict) else "unknown",
                "name": object_name(obj) if isinstance(obj, dict) else f"obj{idx}",
            }
            for idx, obj in enumerate(objects)
        ],
        "derived_collision_bucket": str(meta.get("collision_profile_bucket", "")),
        "contact_graph_nonzero_frames": np.where(contact_graph.sum(axis=(1, 2)) > 0)[0].astype(int).tolist(),
        "contact_impulse_max": float(contact_impulse.max()) if contact_impulse.size else 0.0,
        "first_object_object_event": first_pair_event,
        "role_pair": list(role_pair) if role_pair is not None else None,
        "role_pair_event": role_pair_event,
        "closest_role_pair": closest_role_pair,
        "closest_any_pair": closest_any_pair,
        "pair_records": pair_records,
        "environment_contact_records": environment_records,
        "first_environment_contact_onset": min(
            (int(frame) for item in environment_records for frame in item["onset_frames"]),
            default=None,
        ),
        "collision_definition": {
            "object_object": "previous frame not in contact and current frame in contact, derived from contact_graph",
            "object_environment": "previous frame not in contact and current frame in contact, derived from event_windows ground/environment records",
        },
        "notes": [
            "contact_graph/contact_impulse capture object-object contact only",
            "event_windows may include object-environment support contacts",
            "environment collision is only counted at contact onset, not every support-contact frame",
            "if contact_graph is empty but role-pair distance becomes small, treat it as likely intended interaction rather than confirmed recorded collision",
        ],
    }
    return summary


def save_summary_plot(sample_dir: Path, summary: dict[str, Any]) -> Path:
    vis_dir = sample_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / "main_collision_summary.png"

    meta = load_json(resolve_meta_path(sample_dir))
    rigid_npz = np.load(sample_dir / "physics" / "rigid_kinematics.npz", allow_pickle=True)
    rigid = {key: np.asarray(rigid_npz[key]) for key in rigid_npz.files}
    contact_graph = np.asarray(np.load(sample_dir / "physics" / "contact_graph.npy"), dtype=np.uint8)

    objects = meta.get("objects", [])
    t = np.arange(rigid["com_pos"].shape[0], dtype=np.int32)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), dpi=160, sharex=True)

    for idx in range(rigid["com_pos"].shape[1]):
        label = object_name(objects[idx]) if idx < len(objects) and isinstance(objects[idx], dict) else f"obj{idx}"
        axes[0].plot(t, rigid["com_uv"][:, idx, 0], label=f"{label} u")
    axes[0].set_title("Object Center U Trajectory")
    axes[0].set_ylabel("u / px")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)

    pair_records = summary.get("pair_records", [])
    axes[1].set_title("Object-Object Collision Onsets")
    axes[1].set_ylabel("pair")
    if pair_records:
        axes[1].set_ylim(-0.5, len(pair_records) - 0.5)
        axes[1].set_yticks(np.arange(len(pair_records)))
        axes[1].set_yticklabels(
            [f"{item['names'][0]} <-> {item['names'][1]}" for item in pair_records],
            fontsize=8,
        )
        for row, item in enumerate(pair_records):
            for frame in item.get("onset_frames", []):
                axes[1].axvline(int(frame), ymin=max(0.0, (row - 0.35) / max(len(pair_records), 1)), ymax=min(1.0, (row + 0.35) / max(len(pair_records), 1)), color="#1864ab", linewidth=2.2)
                axes[1].text(int(frame) + 0.08, row, f"{item['names'][0]}-{item['names'][1]}", fontsize=7, va="center", color="#1864ab")
        if not any(item.get("onset_frames") for item in pair_records):
            axes[1].text(0.5, 0.5, "No recorded object-object collision onset", ha="center", va="center", transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, "No object-object pair", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].grid(alpha=0.15, axis="x")

    env_records = summary.get("environment_contact_records", [])
    axes[2].set_title("Object-Environment Collision Onsets")
    axes[2].set_xlabel("frame index")
    axes[2].set_ylabel("obj-env")
    if env_records:
        axes[2].set_ylim(-0.5, len(env_records) - 0.5)
        axes[2].set_yticks(np.arange(len(env_records)))
        axes[2].set_yticklabels(
            [f"{item['object_name']} -> {item['environment_name']}" for item in env_records],
            fontsize=8,
        )
        for row, item in enumerate(env_records):
            for frame in item.get("onset_frames", []):
                axes[2].axvline(int(frame), ymin=max(0.0, (row - 0.35) / max(len(env_records), 1)), ymax=min(1.0, (row + 0.35) / max(len(env_records), 1)), color="#2b8a3e", linewidth=2.2)
                axes[2].text(int(frame) + 0.08, row, f"{item['object_name']}", fontsize=7, va="center", color="#2b8a3e")
    else:
        axes[2].text(0.5, 0.5, "No environment contact onset", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].grid(alpha=0.15, axis="x")

    for ax in axes:
        if summary.get("closest_role_pair"):
            ax.axvline(int(summary["closest_role_pair"]["frame"]), color="#d9480f", linestyle="--", alpha=0.8)
        if summary.get("first_object_object_event"):
            ax.axvline(int(summary["first_object_object_event"]["start_frame"]), color="#1864ab", linestyle=":", alpha=0.8)
        if summary.get("first_environment_contact_onset") is not None:
            ax.axvline(int(summary["first_environment_contact_onset"]), color="#2b8a3e", linestyle="-.", alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    summary = summarize_sample(sample_dir)
    fig_path = save_summary_plot(sample_dir, summary)
    print(json.dumps({"summary": summary, "summary_plot": str(fig_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
