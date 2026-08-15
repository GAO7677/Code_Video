"""Compact precomputed PyBullet object-trajectory cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open

from code_vjepa_vggt.data.pybullet_vae_cache import sample_uid


CACHE_SCHEMA_VERSION = 1
TENSOR_KEYS = (
    "query_points",
    "gt_tracks",
    "gt_visibility_probability",
    "gt_confidence_probability",
    "gt_geometric_visibility",
)


class PyBulletTrajectoryCacheError(RuntimeError):
    pass


def trajectory_relative_path(logical_key: str) -> str:
    uid = sample_uid(logical_key)
    return f"trajectories/{uid[:2]}/{uid}.safetensors"


class PyBulletTrajectoryCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        num_frames: int,
        anchor_frame: int,
        points_per_object: int,
        track_height: int,
        track_width: int,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        config_path = self.cache_dir / "cache_config.json"
        if not config_path.is_file():
            raise PyBulletTrajectoryCacheError(
                f"trajectory cache config not found: {config_path}"
            )
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(self.config.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise PyBulletTrajectoryCacheError(
                f"unsupported trajectory cache schema: {self.config.get('schema_version')}"
            )
        if self.config.get("status") != "complete":
            raise PyBulletTrajectoryCacheError(
                f"trajectory cache is not complete: {self.config.get('status')!r}"
            )
        expected = {
            "num_frames": int(num_frames),
            "anchor_frame": int(anchor_frame),
            "points_per_object": int(points_per_object),
            "track_height": int(track_height),
            "track_width": int(track_width),
        }
        actual = {key: self.config.get(key) for key in expected}
        if actual != expected:
            raise PyBulletTrajectoryCacheError(
                f"trajectory cache settings mismatch: expected={expected}, actual={actual}"
            )

    def validate_records(self, records: Iterable[Any]) -> None:
        missing = []
        invalid = []
        for record in records:
            logical_key = str(record.key)
            path = self.cache_dir / trajectory_relative_path(logical_key)
            if not path.is_file():
                missing.append(logical_key)
                continue
            try:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    metadata = handle.metadata() or {}
                    if metadata.get("logical_key") != logical_key:
                        raise ValueError("logical_key mismatch")
                    if metadata.get("sample_uid") != sample_uid(logical_key):
                        raise ValueError("sample_uid mismatch")
                    for key in TENSOR_KEYS:
                        handle.get_slice(key)
            except Exception as exc:  # noqa: BLE001
                invalid.append(f"{logical_key}: {exc}")
        if missing or invalid:
            raise PyBulletTrajectoryCacheError(
                "PyBullet trajectory cache preflight failed: "
                f"missing={len(missing)}, invalid={len(invalid)}, "
                f"missing_examples={missing[:3]}, invalid_examples={invalid[:3]}"
            )

    def load(self, logical_key: str) -> dict[str, Any]:
        path = self.cache_dir / trajectory_relative_path(logical_key)
        if not path.is_file():
            raise PyBulletTrajectoryCacheError(
                f"trajectory cache entry not found: {logical_key}"
            )
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("logical_key") != logical_key:
                raise PyBulletTrajectoryCacheError(
                    f"trajectory logical_key mismatch in {path}"
                )
            tensors = {key: handle.get_tensor(key) for key in TENSOR_KEYS}
        object_count = int(metadata["object_count"])
        points_per_object = int(metadata["points_per_object"])
        expected_track_shape = (
            int(self.config["num_frames"]),
            object_count,
            points_per_object,
        )
        if tuple(tensors["query_points"].shape) != (
            object_count,
            points_per_object,
            2,
        ):
            raise PyBulletTrajectoryCacheError(
                f"invalid query shape for {logical_key}: {tensors['query_points'].shape}"
            )
        if tuple(tensors["gt_tracks"].shape) != (*expected_track_shape, 2):
            raise PyBulletTrajectoryCacheError(
                f"invalid track shape for {logical_key}: {tensors['gt_tracks'].shape}"
            )
        for key in TENSOR_KEYS[2:]:
            if tuple(tensors[key].shape) != expected_track_shape:
                raise PyBulletTrajectoryCacheError(
                    f"invalid {key} shape for {logical_key}: {tensors[key].shape}"
                )
        return {
            **tensors,
            "logical_key": logical_key,
            "object_count": object_count,
            "points_per_object": points_per_object,
            "object_phrases": json.loads(metadata["object_phrases"]),
        }


def _pybullet_records(dataset) -> list[Any]:
    if hasattr(dataset, "samples") and dataset.__class__.__name__ == "PyBullet0713NoGTBoxDataset":
        return list(dataset.samples)
    records = []
    for child in getattr(dataset, "datasets", []):
        records.extend(_pybullet_records(child))
    return records


class TrajectoryCachedDataset(torch.utils.data.Dataset):
    """Attach a validated trajectory-cache record to each PyBullet sample."""

    def __init__(self, dataset, cache: PyBulletTrajectoryCache) -> None:
        self.dataset = dataset
        self.cache = cache
        records = _pybullet_records(dataset)
        if not records:
            raise ValueError("trajectory loss requires a PyBullet0713 dataset")
        cache.validate_records(records)
        self.sample_weights = getattr(dataset, "sample_weights", None)
        self.load_from_cache = False
        self.dataset_stats = {
            "kind": "trajectory_cached",
            "trajectory_cache_dir": str(cache.cache_dir),
            "source": getattr(dataset, "dataset_stats", None),
        }
        print(
            "[pybullet-trajectory-cache] "
            f"cache_dir={cache.cache_dir} selected={len(records)} missing=0 invalid=0",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        metadata = dict(sample.get("metadata", {}))
        logical_key = str(metadata.get("sample_key", ""))
        if not logical_key:
            raise PyBulletTrajectoryCacheError("sample metadata has no sample_key")
        sample["trajectory_cache"] = self.cache.load(logical_key)
        metadata["trajectory_cache"] = {
            "hit": True,
            "cache_dir": str(self.cache.cache_dir),
        }
        sample["metadata"] = metadata
        return sample
