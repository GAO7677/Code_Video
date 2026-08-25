#!/usr/bin/env python3
"""Prepare an isolated cached-training view for the CYCLES-rendered dataset.

The source dataset manifest is intentionally left untouched.  This utility
creates a list-manifest overlay whose video source is always
``videos/rgb_cycles.mp4`` and materializes the two labels that are consumed by
the auxiliary training script:

* CYCLES-coordinate dynamic masks, resized with the same stretch policy as
  the 512x896 VAE cache;
* collision supervision from the first 49 simulator contact records.

VAE, prompt, latent-mask and Utonia feature tensors are produced by their
dedicated cache builders after this metadata/source preparation step.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3] / "code_V2V_baselines" / "PhysRVG-main"
COLLISION_MODULE_ROOT = REPO_ROOT / "scripts_mytrain" / "train"
if str(COLLISION_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(COLLISION_MODULE_ROOT))

from collision_supervision import save_collision_supervision  # noqa: E402


PHRASES_BY_TASK: dict[str, list[str]] = {
    "table_rolloff": ["a ball"],
    "incline_release": ["a red wooden block"],
    "incline_length_release": ["a red wooden block"],
    "door_frame_clearance_ball": ["a blue rubber ball"],
    "door_frame_clearance": ["a wooden crate"],
    "puck_barrier_collision": ["an ice puck"],
    "bowl_descent": ["a blue rubber ball"],
    "domino_chain": ["a red trigger ball", "a domino"],
    "gap_rolloff": ["a red ball"],
    "obstacle_collision": ["a red ball"],
    "pendulum_cabinet_collision": ["a pendulum bob"],
    "pendulum_swing": ["a pendulum bob"],
    "seesaw_rotation": ["a block", "a seesaw board"],
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _sample_rows(dataset_root: Path) -> list[dict[str, Any]]:
    payload = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    rows = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("the source manifest must contain a non-empty samples list")
    return rows


def _dynamic_phrases(task_type: str) -> list[str]:
    try:
        return list(PHRASES_BY_TASK[task_type])
    except KeyError as exc:
        raise ValueError(f"no Utonia dynamic-object phrase mapping for task_type={task_type}") from exc


def _load_caption(sample_dir: Path) -> str:
    path = sample_dir / "captions" / "caption_abstract.txt"
    caption = path.read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError(f"empty training caption: {path}")
    return caption


def _dynamic_names(sample_dir: Path) -> list[str]:
    with np.load(sample_dir / "raw" / "masks.npz", allow_pickle=False) as arrays:
        names = [str(value) for value in arrays["object_names"]]
        roles = [str(value) for value in arrays["object_roles"]]
    names = [name for name, role in zip(names, roles) if role.startswith("dynamic")]
    if not names:
        raise ValueError(f"no dynamic objects in {sample_dir / 'raw' / 'masks.npz'}")
    return names


def _write_training_overlay(dataset_root: Path, training_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlay: list[dict[str, Any]] = []
    for source_row in rows:
        sample_id = str(source_row["sample_id"])
        sample_dir = (dataset_root / "samples" / sample_id).resolve()
        if not sample_dir.is_dir():
            sample_dir = Path(str(source_row["sample_dir"])).expanduser().resolve()
        video = sample_dir / "videos" / "rgb_cycles.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"missing CYCLES video for {sample_id}: {video}")
        caption = _load_caption(sample_dir)
        phrases = _dynamic_phrases(str(source_row["task_type"]))
        details = [
            {"dynamic": True, "object_noun": phrase, "object_phrase": phrase}
            for phrase in phrases
        ]
        overlay.append(
            {
                "case_id": sample_id,
                "family_key": str(source_row["family_key"]),
                "caption": caption,
                "short_caption": caption,
                "video": str(video),
                "dynamic_object_phrases": phrases,
                "object_phrase_details": details,
                "task_type": str(source_row["task_type"]),
                "source_sample_dir": str(sample_dir),
                "source_video_name": "rgb_cycles.mp4",
                "source_manifest_status": str(source_row.get("status", "unknown")),
            }
        )
    training_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(training_root / "manifest.json", overlay)
    _atomic_json(
        training_root / "dataset_metadata.json",
        {
            "schema_version": "physv_cycles_training_overlay_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset_root": str(dataset_root),
            "sample_count": len(overlay),
            "video_policy": "all rows use samples/<case_id>/videos/rgb_cycles.mp4",
            "caption_policy": "captions/caption_abstract.txt",
            "split_policy": "training code SHA1(family_key/case_id): <0.90 train, <0.95 val, else test",
            "utonia_dynamic_phrases": "task-type mapping in prepare_physv_cycles_training_assets.py",
        },
    )
    (training_root / "README.md").write_text(
        "# physv_v2v_0819 CYCLES training overlay\n\n"
        "This independent list-manifest view uses only `rgb_cycles.mp4` as the\n"
        "conditioning/training video. The original dataset manifest is not\n"
        "modified. Captions come from `caption_abstract.txt`; simulator truth\n"
        "is materialized in the sibling mask and collision cache directories.\n",
        encoding="utf-8",
    )
    return overlay


def _write_mask_source(
    dataset_root: Path,
    aligned_truth_root: Path,
    source_root: Path,
    overlay: list[dict[str, Any]],
) -> dict[str, int]:
    source_root.mkdir(parents=True, exist_ok=True)
    counts = {"written": 0, "resized": 0}
    for row in overlay:
        logical_key = f"{row['family_key']}/{row['case_id']}"
        truth_dir = aligned_truth_root / "cases" / str(row["case_id"])
        truth_metadata = json.loads((truth_dir / "truth_metadata.json").read_text(encoding="utf-8"))
        with np.load(truth_dir / "dynamic_masks.npz", allow_pickle=False) as arrays:
            union = np.asarray(arrays["union_thw"], dtype=np.bool_)
            object_names = [str(value) for value in arrays["object_names"]]
        if union.ndim != 3 or union.shape[0] < 49:
            raise ValueError(f"invalid aligned CYCLES union mask for {logical_key}: {union.shape}")
        source_height, source_width = map(int, union.shape[1:])
        frames = union[:49]
        if (source_height, source_width) != (512, 896):
            frames = np.stack(
                [
                    cv2.resize(frame.astype(np.uint8), (896, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
                    for frame in frames
                ],
                axis=0,
            )
            counts["resized"] += 1
        destination = source_root / "cases" / logical_key / "object_masks.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                masks_othw=frames[None].astype(np.float32),
                object_names=np.asarray(["dynamic_union"]),
                source_object_names=np.asarray(object_names),
            )
        temporary.replace(destination)
        _atomic_json(
            destination.parent / "metadata.json",
            {
                "schema_version": "physv_cycles_mask_source_v1",
                "logical_key": logical_key,
                "source_video": row["video"],
                "source_aligned_truth": str(truth_dir),
                "source_truth_metadata": truth_metadata,
                "source_policy": "union of CYCLES Object Index dynamic masks, first 49 frames",
                "resize_policy": "nearest stretch to 512x896 when source is 640x360",
                "mask_shape": [1, 49, 512, 896],
            },
        )
        counts["written"] += 1
    _atomic_json(
        source_root / "cache_config.json",
        {
            "schema_version": "physv_cycles_mask_source_v1",
            "status": "complete",
            "sample_count": len(overlay),
            "source_aligned_truth_root": str(aligned_truth_root),
            "mask_file": "cases/<family_key>/<case_id>/object_masks.npz",
            "mask_key": "masks_othw",
            "mask_shape": [1, 49, 512, 896],
        },
    )
    return counts


def _write_collision_cache(dataset_root: Path, collision_root: Path, overlay: list[dict[str, Any]]) -> dict[str, int]:
    collision_root.mkdir(parents=True, exist_ok=True)
    event_cases = 0
    for row in overlay:
        logical_key = f"{row['family_key']}/{row['case_id']}"
        sample_dir = Path(str(row["source_sample_dir"]))
        with np.load(sample_dir / "raw" / "masks.npz", allow_pickle=False) as arrays:
            names = [str(value) for value in arrays["object_names"]]
            roles = [str(value) for value in arrays["object_roles"]]
        dynamic_names = {name for name, role in zip(names, roles) if role.startswith("dynamic")}
        contacts = json.loads((sample_dir / "contacts.json").read_text(encoding="utf-8"))
        # contacts.json is sparse for samples with no contact on a frame; the
        # collision builder uses each record's explicit ``frame`` field and
        # supplies the fixed 49-frame output grid itself.
        if not isinstance(contacts, list):
            raise ValueError(f"contacts.json must contain a list: {sample_dir}")
        entry_root = collision_root / "cases" / logical_key
        metadata = save_collision_supervision(
            entry_root,
            logical_key=logical_key,
            dynamic_names=dynamic_names,
            records=contacts,
            build_kwargs={"frame_count": 49, "latent_frames": 13, "latent_stride": 4},
            replay_metadata={
                "source_contacts": str(sample_dir / "contacts.json"),
                "source_video": str(sample_dir / "videos" / "rgb_cycles.mp4"),
                "frame_policy": "first 49 simulator frames aligned with training video window",
            },
        )
        if metadata["events"]:
            event_cases += 1
    index_rows = []
    for row in overlay:
        logical_key = f"{row['family_key']}/{row['case_id']}"
        index_rows.append(
            {
                "logical_key": logical_key,
                "case_dir": f"cases/{logical_key}",
                "arrays_file": f"cases/{logical_key}/collision_supervision.npz",
                "metadata_file": f"cases/{logical_key}/metadata.json",
            }
        )
    _atomic_jsonl(collision_root / "index.jsonl", index_rows)
    _atomic_json(
        collision_root / "cache_config.json",
        {
            "schema_version": "physrvg_collision_supervision_v1",
            "status": "complete",
            "sample_count": len(index_rows),
            "frame_count": 49,
            "latent_frames": 13,
            "latent_stride": 4,
            "weight_formula": "clip(1 + 2 * latent_score, 1, 3)",
            "source_video_policy": "rgb_cycles.mp4; contact truth remains simulator-time aligned",
            "source_dataset_root": str(dataset_root),
            "event_case_count": event_cases,
        },
    )
    return {"written": len(index_rows), "event_cases": event_cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/gaoya/AAA_test_video/physv_v2v_0819"))
    parser.add_argument("--aligned-truth-root", type=Path, default=None)
    parser.add_argument("--training-root", type=Path, default=None)
    parser.add_argument("--mask-source-root", type=Path, default=None)
    parser.add_argument("--collision-root", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    aligned_truth_root = (args.aligned_truth_root or dataset_root / "physv_v2v_0819_cycles_aligned_truth_v1").expanduser().resolve()
    training_root = (args.training_root or dataset_root / "physv_v2v_0819_cycles_train_v1").expanduser().resolve()
    mask_source_root = (args.mask_source_root or dataset_root / "physv_v2v_0819_cycles_mask_source_v1").expanduser().resolve()
    collision_root = (args.collision_root or dataset_root / "physv_v2v_0819_collision_supervision").expanduser().resolve()
    for root in (training_root, mask_source_root, collision_root):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty preparation root: {root}")

    rows = _sample_rows(dataset_root)
    overlay = _write_training_overlay(dataset_root, training_root, rows)
    mask_counts = _write_mask_source(dataset_root, aligned_truth_root, mask_source_root, overlay)
    collision_counts = _write_collision_cache(dataset_root, collision_root, overlay)
    print(
        json.dumps(
            {
                "status": "complete",
                "sample_count": len(overlay),
                "training_root": str(training_root),
                "mask_source_root": str(mask_source_root),
                "collision_root": str(collision_root),
                "mask": mask_counts,
                "collision": collision_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
