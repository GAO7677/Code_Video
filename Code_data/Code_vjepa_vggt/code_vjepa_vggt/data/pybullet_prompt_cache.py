from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open

from code_vjepa_vggt.data.pybullet_vae_cache import (
    canonical_json,
    sample_uid,
    sha256_bytes,
    sha256_file,
    torch_dtype_name,
)


CACHE_SCHEMA_VERSION = 1
PROMPT_TENSOR_KEY = "prompt_embedding"
ATTENTION_MASK_TENSOR_KEY = "attention_mask"
PROMPT_ROLES = (
    "positive_prompt",
    "negative_prompt",
    "object_phrases",
    "dynamic_object_phrases",
    "static_object_phrases",
)


class PyBulletPromptCacheError(RuntimeError):
    pass


def prompt_sha256(prompt: str) -> str:
    return sha256_bytes(str(prompt).encode("utf-8"))


def embedding_relative_path(prompt_hash: str) -> str:
    return f"embeddings/{prompt_hash[:2]}/{prompt_hash}.safetensors"


def tokenizer_fingerprint(path: str | Path) -> str:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {root}")
    rows = []
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        rows.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return sha256_bytes(canonical_json(rows).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class PromptIndexEntry:
    prompt_sha256: str
    prompt: str
    embedding_file: str
    embedding_shape: tuple[int, ...]
    embedding_dtype: str
    valid_token_count: int

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "PromptIndexEntry":
        return cls(
            prompt_sha256=str(row["prompt_sha256"]),
            prompt=str(row["prompt"]),
            embedding_file=str(row["embedding_file"]),
            embedding_shape=tuple(int(value) for value in row["embedding_shape"]),
            embedding_dtype=str(row["embedding_dtype"]),
            valid_token_count=int(row["valid_token_count"]),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt_sha256": self.prompt_sha256,
            "prompt": self.prompt,
            "embedding_file": self.embedding_file,
            "embedding_shape": list(self.embedding_shape),
            "embedding_dtype": self.embedding_dtype,
            "valid_token_count": self.valid_token_count,
        }


@dataclass(frozen=True, slots=True)
class SamplePromptIndexEntry:
    logical_key: str
    sample_uid: str
    source_relpath: str
    source_size: int
    source_mtime_ns: int
    source_sha256: str
    positive_prompt: str
    negative_prompt: str
    object_phrases: tuple[str, ...]
    dynamic_object_phrases: tuple[str, ...]
    static_object_phrases: tuple[str, ...]

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "SamplePromptIndexEntry":
        return cls(
            logical_key=str(row["logical_key"]),
            sample_uid=str(row["sample_uid"]),
            source_relpath=str(row["source_relpath"]),
            source_size=int(row["source_size"]),
            source_mtime_ns=int(row["source_mtime_ns"]),
            source_sha256=str(row["source_sha256"]),
            positive_prompt=str(row["positive_prompt"]),
            negative_prompt=str(row["negative_prompt"]),
            object_phrases=tuple(str(value) for value in row["object_phrases"]),
            dynamic_object_phrases=tuple(
                str(value) for value in row["dynamic_object_phrases"]
            ),
            static_object_phrases=tuple(
                str(value) for value in row["static_object_phrases"]
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "logical_key": self.logical_key,
            "sample_uid": self.sample_uid,
            "source_relpath": self.source_relpath,
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "source_sha256": self.source_sha256,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "object_phrases": list(self.object_phrases),
            "dynamic_object_phrases": list(self.dynamic_object_phrases),
            "static_object_phrases": list(self.static_object_phrases),
        }


class PyBulletPromptEmbeddingCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        text_encoder_checkpoint_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        config_path = self.cache_dir / "cache_config.json"
        prompt_index_path = self.cache_dir / "prompt_index.jsonl"
        sample_index_path = self.cache_dir / "index.jsonl"
        for path in (config_path, prompt_index_path, sample_index_path):
            if not path.is_file():
                raise PyBulletPromptCacheError(f"Prompt cache file not found: {path}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(self.config.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise PyBulletPromptCacheError(
                f"Unsupported prompt cache schema: {self.config.get('schema_version')}"
            )
        if self.config.get("status") != "complete":
            raise PyBulletPromptCacheError(
                f"Prompt cache is not complete: status={self.config.get('status')!r}"
            )

        prompt_index_text = prompt_index_path.read_text(encoding="utf-8")
        sample_index_text = sample_index_path.read_text(encoding="utf-8")
        expected_prompt_index_hash = self.config.get("prompt_index_sha256")
        expected_sample_index_hash = self.config.get("index_sha256")
        if sha256_bytes(prompt_index_text.encode("utf-8")) != expected_prompt_index_hash:
            raise PyBulletPromptCacheError("Prompt index SHA256 mismatch")
        if sha256_bytes(sample_index_text.encode("utf-8")) != expected_sample_index_hash:
            raise PyBulletPromptCacheError("Sample prompt index SHA256 mismatch")

        encoder = self.config.get("text_encoder", {})
        if text_encoder_checkpoint_path is not None:
            checkpoint_path = Path(text_encoder_checkpoint_path).expanduser().resolve()
            checkpoint_stat = checkpoint_path.stat()
            if (
                encoder.get("checkpoint_name") != checkpoint_path.name
                or int(encoder.get("checkpoint_size", -1)) != checkpoint_stat.st_size
            ):
                raise PyBulletPromptCacheError(
                    "Prompt cache text encoder mismatch: "
                    f"cache={encoder.get('checkpoint_name')}/{encoder.get('checkpoint_size')}, "
                    f"training={checkpoint_path.name}/{checkpoint_stat.st_size}"
                )
        if tokenizer_path is not None:
            actual_fingerprint = tokenizer_fingerprint(tokenizer_path)
            if actual_fingerprint != self.config.get("tokenizer", {}).get("fingerprint"):
                raise PyBulletPromptCacheError("Prompt cache tokenizer fingerprint mismatch")

        self.encoding_id = str(self.config["encoding_id"])
        self.prompts: dict[str, PromptIndexEntry] = {}
        for line_number, line in enumerate(prompt_index_text.splitlines(), start=1):
            if not line.strip():
                continue
            entry = PromptIndexEntry.from_json(json.loads(line))
            if entry.prompt_sha256 in self.prompts:
                raise PyBulletPromptCacheError(
                    f"Duplicate prompt hash at line {line_number}: {entry.prompt_sha256}"
                )
            if prompt_sha256(entry.prompt) != entry.prompt_sha256:
                raise PyBulletPromptCacheError(
                    f"Invalid prompt hash at line {line_number}: {entry.prompt_sha256}"
                )
            self.prompts[entry.prompt_sha256] = entry

        self.entries: dict[str, SamplePromptIndexEntry] = {}
        seen_uids: set[str] = set()
        for line_number, line in enumerate(sample_index_text.splitlines(), start=1):
            if not line.strip():
                continue
            entry = SamplePromptIndexEntry.from_json(json.loads(line))
            if entry.logical_key in self.entries:
                raise PyBulletPromptCacheError(
                    f"Duplicate logical_key at line {line_number}: {entry.logical_key}"
                )
            if entry.sample_uid in seen_uids or entry.sample_uid != sample_uid(entry.logical_key):
                raise PyBulletPromptCacheError(
                    f"Invalid or duplicate sample_uid at line {line_number}: {entry.sample_uid}"
                )
            referenced = [
                entry.positive_prompt,
                entry.negative_prompt,
                *entry.object_phrases,
                *entry.dynamic_object_phrases,
                *entry.static_object_phrases,
            ]
            missing = [value for value in referenced if value not in self.prompts]
            if missing:
                raise PyBulletPromptCacheError(
                    f"Sample index references missing prompt hashes: {entry.logical_key} {missing[:3]}"
                )
            self.entries[entry.logical_key] = entry
            seen_uids.add(entry.sample_uid)

        if len(self.entries) != int(self.config.get("num_samples", -1)):
            raise PyBulletPromptCacheError("Prompt cache sample count mismatch")
        if len(self.prompts) != int(self.config.get("num_unique_prompts", -1)):
            raise PyBulletPromptCacheError("Prompt cache unique prompt count mismatch")

    def validate_records(self, records: Iterable[Any], dataset_root: str | Path) -> None:
        root = Path(dataset_root).expanduser().resolve()
        missing: list[str] = []
        invalid: list[str] = []
        for record in records:
            entry = self.entries.get(str(record.key))
            if entry is None:
                missing.append(str(record.key))
                continue
            manifest_path = Path(record.manifest_path).resolve()
            try:
                source_relpath = manifest_path.relative_to(root).as_posix()
            except ValueError:
                invalid.append(f"{record.key}: manifest outside dataset root")
                continue
            stat = manifest_path.stat()
            positive = self.prompts[entry.positive_prompt]
            negative = self.prompts[entry.negative_prompt]
            if (
                source_relpath != entry.source_relpath
                or stat.st_size != entry.source_size
                or stat.st_mtime_ns != entry.source_mtime_ns
                or prompt_sha256(record.caption) != positive.prompt_sha256
                or prompt_sha256(record.negative_prompt) != negative.prompt_sha256
            ):
                invalid.append(str(record.key))
        if missing or invalid:
            raise PyBulletPromptCacheError(
                "PyBullet prompt cache preflight failed: "
                f"missing={len(missing)}, invalid={len(invalid)}, "
                f"missing_examples={missing[:3]}, invalid_examples={invalid[:3]}"
            )

    def load_prompt_hash(self, prompt_hash: str) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.prompts.get(str(prompt_hash))
        if entry is None:
            raise PyBulletPromptCacheError(f"Prompt hash not found: {prompt_hash}")
        path = self.cache_dir / entry.embedding_file
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if (
                    metadata.get("prompt_sha256") != entry.prompt_sha256
                    or metadata.get("encoding_id") != self.encoding_id
                ):
                    raise PyBulletPromptCacheError(f"Prompt metadata mismatch: {path}")
                embedding = handle.get_tensor(PROMPT_TENSOR_KEY)
                attention_mask = handle.get_tensor(ATTENTION_MASK_TENSOR_KEY)
        except PyBulletPromptCacheError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PyBulletPromptCacheError(f"Failed to load prompt cache file: {path}") from exc
        if (
            tuple(int(value) for value in embedding.shape) != entry.embedding_shape
            or torch_dtype_name(embedding.dtype) != entry.embedding_dtype
            or attention_mask.ndim != 1
            or int(attention_mask.sum().item()) != entry.valid_token_count
        ):
            raise PyBulletPromptCacheError(f"Prompt tensor/index mismatch: {path}")
        return embedding.contiguous(), attention_mask.contiguous()

    def load(self, logical_key: str) -> torch.Tensor:
        entry = self.entries.get(str(logical_key))
        if entry is None:
            raise PyBulletPromptCacheError(f"Prompt cache entry not found: {logical_key}")
        embedding, _ = self.load_prompt_hash(entry.positive_prompt)
        return embedding

