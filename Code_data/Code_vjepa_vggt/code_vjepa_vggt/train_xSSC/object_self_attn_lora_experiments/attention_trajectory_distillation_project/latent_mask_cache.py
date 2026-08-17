"""Validated GroundingDINO + SAM2 mask cache for PyBullet training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


CACHE_SCHEMA_VERSION = 1


class PyBulletLatentMaskCacheError(RuntimeError):
    pass


def mask_case_root(cache_dir: str | Path, logical_key: str) -> Path:
    cases_root = (Path(cache_dir).expanduser().resolve() / "cases").resolve()
    case_root = (cases_root / str(logical_key)).resolve()
    try:
        case_root.relative_to(cases_root)
    except ValueError as exc:
        raise PyBulletLatentMaskCacheError(
            f"unsafe logical key for latent-mask cache: {logical_key!r}"
        ) from exc
    return case_root


class PyBulletLatentMaskCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        num_frames: int,
        anchor_frame: int,
        native_height: int,
        native_width: int,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        config_path = self.cache_dir / "cache_config.json"
        if not config_path.is_file():
            raise PyBulletLatentMaskCacheError(
                f"latent-mask cache config not found: {config_path}"
            )
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(self.config.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise PyBulletLatentMaskCacheError(
                f"unsupported latent-mask cache schema: {self.config.get('schema_version')}"
            )
        if self.config.get("status") != "complete":
            raise PyBulletLatentMaskCacheError(
                f"latent-mask cache is not complete: {self.config.get('status')!r}"
            )
        expected = {
            "num_frames": int(num_frames),
            "anchor_frame": int(anchor_frame),
            "native_height": int(native_height),
            "native_width": int(native_width),
        }
        actual = {key: self.config.get(key) for key in expected}
        if actual != expected:
            raise PyBulletLatentMaskCacheError(
                f"latent-mask cache settings mismatch: expected={expected}, actual={actual}"
            )

    def validate_records(self, records: Iterable[Any]) -> None:
        missing: list[str] = []
        invalid: list[str] = []
        for record in records:
            logical_key = str(record.key)
            case_root = mask_case_root(self.cache_dir, logical_key)
            arrays_path = case_root / "object_masks.npz"
            metadata_path = case_root / "entry.json"
            if not arrays_path.is_file() or not metadata_path.is_file():
                missing.append(logical_key)
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("logical_key") != logical_key:
                    raise ValueError("logical_key mismatch")
                expected_shape = [
                    int(metadata["object_count"]),
                    int(self.config["num_frames"]),
                    int(self.config["native_height"]),
                    int(self.config["native_width"]),
                ]
                if metadata.get("mask_shape") != expected_shape:
                    raise ValueError(
                        f"mask_shape mismatch: {metadata.get('mask_shape')}/{expected_shape}"
                    )
                if int(metadata.get("object_count", 0)) <= 0:
                    raise ValueError("object_count must be positive")
                if float(metadata.get("reverse_recall", 0.0)) < 1.0:
                    raise ValueError("latent reverse-mapping recall is below 1")
            except Exception as exc:  # noqa: BLE001
                invalid.append(f"{logical_key}: {exc}")
        if missing or invalid:
            raise PyBulletLatentMaskCacheError(
                "PyBullet latent-mask cache preflight failed: "
                f"missing={len(missing)}, invalid={len(invalid)}, "
                f"missing_examples={missing[:3]}, invalid_examples={invalid[:3]}"
            )

    def load(self, logical_key: str) -> dict[str, Any]:
        case_root = mask_case_root(self.cache_dir, logical_key)
        arrays_path = case_root / "object_masks.npz"
        metadata_path = case_root / "entry.json"
        if not arrays_path.is_file() or not metadata_path.is_file():
            raise PyBulletLatentMaskCacheError(
                f"latent-mask cache entry not found: {logical_key}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("logical_key") != logical_key:
            raise PyBulletLatentMaskCacheError(
                f"latent-mask logical_key mismatch in {metadata_path}"
            )
        with np.load(arrays_path, allow_pickle=False) as arrays:
            masks = np.asarray(arrays["masks_othw"], dtype=np.uint8)
            frame_indices = np.asarray(arrays["frame_indices"], dtype=np.int64)
        expected_shape = tuple(metadata["mask_shape"])
        if tuple(masks.shape) != expected_shape:
            raise PyBulletLatentMaskCacheError(
                f"invalid mask shape for {logical_key}: {masks.shape}/{expected_shape}"
            )
        return {
            "masks_othw": masks,
            "frame_indices": frame_indices,
            "metadata": metadata,
        }


def _pybullet_records(dataset) -> list[Any]:
    if (
        hasattr(dataset, "samples")
        and dataset.__class__.__name__
        in {"PyBullet0713NoGTBoxDataset", "PyBulletRawNoGTBoxDataset"}
    ):
        return list(dataset.samples)
    records: list[Any] = []
    for child in getattr(dataset, "datasets", []):
        records.extend(_pybullet_records(child))
    return records


class LatentMaskCachedDataset(torch.utils.data.Dataset):
    """Preflight the complete cache while preserving the original sample payload."""

    def __init__(self, dataset, cache: PyBulletLatentMaskCache) -> None:
        self.dataset = dataset
        self.cache = cache
        records = _pybullet_records(dataset)
        if not records:
            raise ValueError("GT latent-mask loss requires a supported PyBullet dataset")
        cache.validate_records(records)
        self.sample_weights = getattr(dataset, "sample_weights", None)
        self.load_from_cache = False
        self.dataset_stats = {
            "kind": "latent_mask_cached",
            "latent_mask_cache_dir": str(cache.cache_dir),
            "source": getattr(dataset, "dataset_stats", None),
        }
        print(
            "[pybullet-latent-mask-cache] "
            f"cache_dir={cache.cache_dir} selected={len(records)} missing=0 invalid=0",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]
