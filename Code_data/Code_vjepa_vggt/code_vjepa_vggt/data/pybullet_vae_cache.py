from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open


CACHE_SCHEMA_VERSION = 1
LATENT_TENSOR_KEY = "vae_latent"


class PyBulletVaeCacheError(RuntimeError):
    pass


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sample_uid(logical_key: str) -> str:
    return sha256_bytes(f"pybullet0713\0{logical_key}".encode("utf-8"))


def encoding_id(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def latent_relative_path(uid: str) -> str:
    return f"latents/{uid[:2]}/{uid}.safetensors"


def torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class CacheIndexEntry:
    logical_key: str
    sample_uid: str
    source_relpath: str
    source_size: int
    source_mtime_ns: int
    source_sha256: str
    sampled_frame_indices: tuple[int, ...]
    latent_file: str
    latent_shape: tuple[int, ...]
    latent_dtype: str

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "CacheIndexEntry":
        return cls(
            logical_key=str(row["logical_key"]),
            sample_uid=str(row["sample_uid"]),
            source_relpath=str(row["source_relpath"]),
            source_size=int(row["source_size"]),
            source_mtime_ns=int(row["source_mtime_ns"]),
            source_sha256=str(row["source_sha256"]),
            sampled_frame_indices=tuple(int(value) for value in row["sampled_frame_indices"]),
            latent_file=str(row["latent_file"]),
            latent_shape=tuple(int(value) for value in row["latent_shape"]),
            latent_dtype=str(row["latent_dtype"]),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "logical_key": self.logical_key,
            "sample_uid": self.sample_uid,
            "source_relpath": self.source_relpath,
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "source_sha256": self.source_sha256,
            "sampled_frame_indices": list(self.sampled_frame_indices),
            "latent_file": self.latent_file,
            "latent_shape": list(self.latent_shape),
            "latent_dtype": self.latent_dtype,
        }


class PyBulletVaeLatentCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        resolution: tuple[int, int],
        num_frames: int,
        sampling_strategy: str,
        vae_checkpoint_path: str | Path | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        config_path = self.cache_dir / "cache_config.json"
        index_path = self.cache_dir / "index.jsonl"
        if not config_path.is_file():
            raise PyBulletVaeCacheError(f"VAE cache config not found: {config_path}")
        if not index_path.is_file():
            raise PyBulletVaeCacheError(f"VAE cache index not found: {index_path}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(self.config.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise PyBulletVaeCacheError(
                f"Unsupported VAE cache schema: {self.config.get('schema_version')}"
            )
        if self.config.get("status") != "complete":
            raise PyBulletVaeCacheError(
                f"VAE cache is not complete: status={self.config.get('status')!r}"
            )
        preprocess = self.config.get("preprocess", {})
        expected = {
            "height": int(resolution[0]),
            "width": int(resolution[1]),
            "num_frames": int(num_frames),
            "sampling_strategy": str(sampling_strategy),
        }
        actual = {key: preprocess.get(key) for key in expected}
        if actual != expected:
            raise PyBulletVaeCacheError(
                f"VAE cache preprocessing mismatch: expected={expected}, actual={actual}"
            )
        if vae_checkpoint_path is not None:
            checkpoint_path = Path(vae_checkpoint_path).expanduser().resolve()
            checkpoint_stat = checkpoint_path.stat()
            cached_vae = self.config.get("vae", {})
            if (
                cached_vae.get("checkpoint_name") != checkpoint_path.name
                or int(cached_vae.get("checkpoint_size", -1)) != checkpoint_stat.st_size
            ):
                raise PyBulletVaeCacheError(
                    "VAE cache checkpoint mismatch: "
                    f"cache={cached_vae.get('checkpoint_name')}/"
                    f"{cached_vae.get('checkpoint_size')}, "
                    f"training={checkpoint_path.name}/{checkpoint_stat.st_size}"
                )

        self.entries: dict[str, CacheIndexEntry] = {}
        seen_uids: set[str] = set()
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                entry = CacheIndexEntry.from_json(json.loads(line))
                if entry.logical_key in self.entries:
                    raise PyBulletVaeCacheError(
                        f"Duplicate logical_key in cache index at line {line_number}: "
                        f"{entry.logical_key}"
                    )
                if entry.sample_uid in seen_uids:
                    raise PyBulletVaeCacheError(
                        f"Duplicate sample_uid in cache index at line {line_number}: "
                        f"{entry.sample_uid}"
                    )
                expected_uid = sample_uid(entry.logical_key)
                if entry.sample_uid != expected_uid:
                    raise PyBulletVaeCacheError(
                        f"Invalid sample_uid for {entry.logical_key}: "
                        f"expected={expected_uid}, actual={entry.sample_uid}"
                    )
                self.entries[entry.logical_key] = entry
                seen_uids.add(entry.sample_uid)

        expected_count = int(self.config.get("num_samples", -1))
        if len(self.entries) != expected_count:
            raise PyBulletVaeCacheError(
                f"VAE cache count mismatch: config={expected_count}, index={len(self.entries)}"
            )
        self.encoding_id = str(self.config["encoding_id"])

    def validate_records(self, records: Iterable[Any], dataset_root: str | Path) -> None:
        root = Path(dataset_root).expanduser().resolve()
        missing: list[str] = []
        invalid: list[str] = []
        record_keys: set[str] = set()
        for record in records:
            logical_key = str(record.key)
            if logical_key in record_keys:
                raise PyBulletVaeCacheError(f"Duplicate dataset logical_key: {logical_key}")
            record_keys.add(logical_key)
            entry = self.entries.get(logical_key)
            if entry is None:
                missing.append(logical_key)
                continue
            video_path = Path(record.video_path).resolve()
            try:
                source_relpath = video_path.relative_to(root).as_posix()
            except ValueError:
                invalid.append(f"{logical_key}: video is outside dataset root")
                continue
            stat = video_path.stat()
            latent_path = self.cache_dir / entry.latent_file
            if (
                source_relpath != entry.source_relpath
                or stat.st_size != entry.source_size
                or stat.st_mtime_ns != entry.source_mtime_ns
                or not latent_path.is_file()
            ):
                invalid.append(logical_key)
        if missing or invalid:
            raise PyBulletVaeCacheError(
                "PyBullet VAE cache preflight failed: "
                f"selected={len(record_keys)}, missing={len(missing)}, invalid={len(invalid)}, "
                f"missing_examples={missing[:3]}, invalid_examples={invalid[:3]}"
            )

    def load(self, logical_key: str) -> torch.Tensor:
        entry = self.entries.get(logical_key)
        if entry is None:
            raise PyBulletVaeCacheError(f"VAE cache entry not found: {logical_key}")
        latent_path = self.cache_dir / entry.latent_file
        try:
            with safe_open(latent_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("logical_key") != logical_key:
                    raise PyBulletVaeCacheError(
                        f"VAE cache logical_key mismatch in {latent_path}"
                    )
                if metadata.get("sample_uid") != entry.sample_uid:
                    raise PyBulletVaeCacheError(
                        f"VAE cache sample_uid mismatch in {latent_path}"
                    )
                if metadata.get("encoding_id") != self.encoding_id:
                    raise PyBulletVaeCacheError(
                        f"VAE cache encoding_id mismatch in {latent_path}"
                    )
                latent = handle.get_tensor(LATENT_TENSOR_KEY)
        except PyBulletVaeCacheError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PyBulletVaeCacheError(f"Failed to load VAE cache file: {latent_path}") from exc
        actual_shape = tuple(int(value) for value in latent.shape)
        actual_dtype = torch_dtype_name(latent.dtype)
        if actual_shape != entry.latent_shape or actual_dtype != entry.latent_dtype:
            raise PyBulletVaeCacheError(
                f"VAE latent metadata mismatch for {logical_key}: "
                f"index={entry.latent_shape}/{entry.latent_dtype}, "
                f"tensor={actual_shape}/{actual_dtype}"
            )
        return latent.contiguous()
