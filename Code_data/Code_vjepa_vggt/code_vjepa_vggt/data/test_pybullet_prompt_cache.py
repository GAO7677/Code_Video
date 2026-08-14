from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file

from code_vjepa_vggt.data.prepare_pybullet_prompt_cache import _collect_prompts
from code_vjepa_vggt.data.pybullet_prompt_cache import (
    ATTENTION_MASK_TENSOR_KEY,
    PROMPT_TENSOR_KEY,
    PyBulletPromptCacheError,
    PyBulletPromptEmbeddingCache,
    embedding_relative_path,
    prompt_sha256,
)
from code_vjepa_vggt.data.pybullet_vae_cache import (
    canonical_json,
    sample_uid,
    sha256_bytes,
)


def _write_cache(root: Path) -> tuple[Path, Path, SimpleNamespace]:
    dataset_root = root / "dataset"
    cache_dir = dataset_root / "prompt_embeddings"
    manifest_path = dataset_root / "cases" / "F1" / "case001" / "case_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "caption": "A wheel rolls.",
        "negative_prompt": "blurred",
        "object_phrases": ["a wheel"],
        "dynamic_object_phrases": ["a wheel"],
        "static_object_phrases": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    record = SimpleNamespace(
        key="F1/case001",
        caption=manifest["caption"],
        negative_prompt=manifest["negative_prompt"],
        manifest_path=str(manifest_path),
    )
    prompts = [manifest["caption"], manifest["negative_prompt"], "a wheel"]
    encoding = "encoding001"
    prompt_rows = []
    for prompt in prompts:
        prompt_hash = prompt_sha256(prompt)
        path = cache_dir / embedding_relative_path(prompt_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = torch.zeros((512,), dtype=torch.int64)
        mask[:3] = 1
        save_file(
            {
                PROMPT_TENSOR_KEY: torch.zeros((512, 4096), dtype=torch.bfloat16),
                ATTENTION_MASK_TENSOR_KEY: mask,
            },
            path,
            metadata={
                "prompt_sha256": prompt_hash,
                "encoding_id": encoding,
            },
        )
        prompt_rows.append(
            {
                "prompt_sha256": prompt_hash,
                "prompt": prompt,
                "embedding_file": path.relative_to(cache_dir).as_posix(),
                "embedding_shape": [512, 4096],
                "embedding_dtype": "bfloat16",
                "valid_token_count": 3,
            }
        )
    prompt_text = "".join(canonical_json(row) + "\n" for row in prompt_rows)
    stat = manifest_path.stat()
    sample_row = {
        "logical_key": record.key,
        "sample_uid": sample_uid(record.key),
        "source_relpath": manifest_path.relative_to(dataset_root).as_posix(),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha256": "manifest001",
        "positive_prompt": prompt_sha256(manifest["caption"]),
        "negative_prompt": prompt_sha256(manifest["negative_prompt"]),
        "object_phrases": [prompt_sha256("a wheel")],
        "dynamic_object_phrases": [prompt_sha256("a wheel")],
        "static_object_phrases": [],
    }
    sample_text = canonical_json(sample_row) + "\n"
    (cache_dir / "prompt_index.jsonl").write_text(prompt_text, encoding="utf-8")
    (cache_dir / "index.jsonl").write_text(sample_text, encoding="utf-8")
    config = {
        "schema_version": 1,
        "status": "complete",
        "encoding_id": encoding,
        "num_samples": 1,
        "num_unique_prompts": 3,
        "prompt_index_sha256": sha256_bytes(prompt_text.encode("utf-8")),
        "index_sha256": sha256_bytes(sample_text.encode("utf-8")),
    }
    (cache_dir / "cache_config.json").write_text(json.dumps(config), encoding="utf-8")
    return dataset_root, cache_dir, record


class PyBulletPromptCacheTests(unittest.TestCase):
    def test_cache_loads_positive_prompt_and_validates_record(self) -> None:
        with self._temporary_directory() as root:
            dataset_root, cache_dir, record = _write_cache(root)
            cache = PyBulletPromptEmbeddingCache(cache_dir)
            cache.validate_records([record], dataset_root)
            embedding = cache.load(record.key)
            self.assertEqual(embedding.shape, (512, 4096))
            self.assertEqual(embedding.dtype, torch.bfloat16)

    def test_cache_rejects_changed_caption(self) -> None:
        with self._temporary_directory() as root:
            dataset_root, cache_dir, record = _write_cache(root)
            record.caption = "A changed caption."
            cache = PyBulletPromptEmbeddingCache(cache_dir)
            with self.assertRaisesRegex(PyBulletPromptCacheError, "preflight failed"):
                cache.validate_records([record], dataset_root)

    def test_collect_prompts_deduplicates_across_roles_and_samples(self) -> None:
        with self._temporary_directory() as root:
            manifest = root / "case_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "object_phrases": ["a wheel", "a wheel"],
                        "dynamic_object_phrases": ["a wheel"],
                        "static_object_phrases": [],
                    }
                ),
                encoding="utf-8",
            )
            records = [
                SimpleNamespace(
                    key=f"F1/case{index}",
                    caption="A wheel rolls.",
                    negative_prompt="blurred",
                    manifest_path=str(manifest),
                )
                for index in range(2)
            ]
            prompts, roles = _collect_prompts(SimpleNamespace(samples=records))
            self.assertEqual(len(prompts), 3)
            self.assertEqual(len(roles), 2)
            self.assertEqual(len(roles["F1/case0"]["object_phrases"]), 1)

    @staticmethod
    def _temporary_directory():
        import tempfile

        class TemporaryPath(tempfile.TemporaryDirectory):
            def __enter__(self) -> Path:
                return Path(super().__enter__())

        return TemporaryPath()


if __name__ == "__main__":
    unittest.main()
