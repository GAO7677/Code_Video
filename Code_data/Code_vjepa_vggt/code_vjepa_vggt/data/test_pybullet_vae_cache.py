from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from code_vjepa_vggt.data.prepare_pybullet_vae_cache import (
    _build_dataset,
    _distributed_context,
    _latent_comparison_metrics,
)
from code_vjepa_vggt.data.pybullet_vae_cache import (
    LATENT_TENSOR_KEY,
    PyBulletVaeCacheError,
    PyBulletVaeLatentCache,
    canonical_json,
    sample_uid,
)


def _write_cache(root: Path, logical_key: str = "F1/case001") -> tuple[Path, Path]:
    dataset_root = root / "dataset"
    cache_dir = dataset_root / "vae_latents"
    video_path = dataset_root / "cases" / "F1" / "case001.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    source_stat = video_path.stat()
    uid = sample_uid(logical_key)
    latent_path = cache_dir / "latents" / uid[:2] / f"{uid}.safetensors"
    latent_path.parent.mkdir(parents=True)
    encoding = "encoding001"
    save_file(
        {LATENT_TENSOR_KEY: torch.zeros((48, 13, 32, 56), dtype=torch.bfloat16)},
        latent_path,
        metadata={
            "logical_key": logical_key,
            "sample_uid": uid,
            "encoding_id": encoding,
        },
    )
    config = {
        "schema_version": 1,
        "status": "complete",
        "num_samples": 1,
        "encoding_id": encoding,
        "preprocess": {
            "height": 512,
            "width": 896,
            "num_frames": 49,
            "sampling_strategy": "prefix",
        },
    }
    (cache_dir / "cache_config.json").write_text(json.dumps(config), encoding="utf-8")
    row = {
        "logical_key": logical_key,
        "sample_uid": uid,
        "source_relpath": "cases/F1/case001.mp4",
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_sha256": "source001",
        "sampled_frame_indices": list(range(49)),
        "latent_file": latent_path.relative_to(cache_dir).as_posix(),
        "latent_shape": [48, 13, 32, 56],
        "latent_dtype": "bfloat16",
    }
    (cache_dir / "index.jsonl").write_text(canonical_json(row) + "\n", encoding="utf-8")
    return dataset_root, cache_dir


class PyBulletVaeCacheTests(unittest.TestCase):
    def test_raw_dataset_factory_combines_train_val_with_stable_prompt_roles(self) -> None:
        with self._temporary_directory() as root:
            for split, sample_id in (("train", "sample_000001"), ("val", "sample_000002")):
                sample_dir = root / split / "F1_single_object" / sample_id
                sample_dir.mkdir(parents=True)
                (sample_dir / "video.mp4").write_bytes(b"video")
                (sample_dir / "meta.json").write_text(
                    json.dumps(
                        {
                            "objects": [
                                {
                                    "shape": "sphere",
                                    "dynamic": True,
                                    "linear_velocity": [1.0, 0.0, 0.0],
                                },
                                {"shape": "box", "dynamic": False},
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            args = argparse.Namespace(
                dataset_format="pybullet0613_raw",
                pybullet_root=str(root),
                height=512,
                width=896,
                num_frames=49,
                num_context_frames=8,
                sampling_strategy="prefix",
            )
            dataset = _build_dataset(args)
            self.assertEqual(len(dataset.samples), 2)
            self.assertEqual(
                dataset.samples[0].key,
                "raw0613/train/F1_single_object/sample_000001",
            )
            self.assertIn("from left to right", dataset.samples[0].caption)
            self.assertEqual(dataset.samples[0].dynamic_object_phrases, ("a ball",))
            self.assertEqual(dataset.samples[0].static_object_phrases, ("a block",))

    def test_online_comparison_uses_bfloat16_numerical_tolerance(self) -> None:
        reference = torch.ones((16,), dtype=torch.float32)
        close = reference + 0.001
        far = reference.clone()
        far[0] += 1.0

        self.assertTrue(_latent_comparison_metrics(close, reference)["within_tolerance"])
        self.assertFalse(_latent_comparison_metrics(far, reference)["within_tolerance"])

    def test_distributed_context_maps_five_workers_per_visible_gpu(self) -> None:
        calls: list[tuple[str, object]] = []
        environment = {"WORLD_SIZE": "15", "RANK": "5", "LOCAL_RANK": "5"}
        with (
            mock.patch.dict("os.environ", environment, clear=False),
            mock.patch("torch.cuda.device_count", return_value=3),
            mock.patch(
                "torch.cuda.set_device",
                side_effect=lambda index: calls.append(("set_device", index)),
            ),
            mock.patch("torch.distributed.is_initialized", return_value=False),
            mock.patch(
                "torch.distributed.init_process_group",
                side_effect=lambda **kwargs: calls.append(("init_process_group", kwargs)),
            ),
        ):
            self.assertEqual(_distributed_context(), (5, 15, 2))

        self.assertEqual(
            calls,
            [
                ("set_device", 2),
                ("init_process_group", {"backend": "gloo"}),
            ],
        )

    def test_cache_loads_tensor_and_validates_selected_record(self) -> None:
        with self.subTest(), self._temporary_directory() as tmp_path:
            dataset_root, cache_dir = _write_cache(tmp_path)
            cache = PyBulletVaeLatentCache(
                cache_dir,
                resolution=(512, 896),
                num_frames=49,
                sampling_strategy="prefix",
            )
            record = SimpleNamespace(
                key="F1/case001",
                video_path=str(dataset_root / "cases" / "F1" / "case001.mp4"),
            )
            cache.validate_records([record], dataset_root)
            latent = cache.load(record.key)
            self.assertEqual(latent.shape, (48, 13, 32, 56))
            self.assertEqual(latent.dtype, torch.bfloat16)

    def test_cache_rejects_preprocess_mismatch(self) -> None:
        with self._temporary_directory() as tmp_path:
            _, cache_dir = _write_cache(tmp_path)
            with self.assertRaisesRegex(PyBulletVaeCacheError, "preprocessing mismatch"):
                PyBulletVaeLatentCache(
                    cache_dir,
                    resolution=(384, 672),
                    num_frames=49,
                    sampling_strategy="prefix",
                )

    def test_cache_rejects_duplicate_logical_keys(self) -> None:
        with self._temporary_directory() as tmp_path:
            _, cache_dir = _write_cache(tmp_path)
            index_path = cache_dir / "index.jsonl"
            row = index_path.read_text(encoding="utf-8")
            index_path.write_text(row + row, encoding="utf-8")
            config_path = cache_dir / "cache_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["num_samples"] = 2
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(PyBulletVaeCacheError, "Duplicate logical_key"):
                PyBulletVaeLatentCache(
                    cache_dir,
                    resolution=(512, 896),
                    num_frames=49,
                    sampling_strategy="prefix",
                )

    @staticmethod
    def _temporary_directory():
        import tempfile

        class TemporaryPath(tempfile.TemporaryDirectory):
            def __enter__(self) -> Path:
                return Path(super().__enter__())

        return TemporaryPath()


if __name__ == "__main__":
    unittest.main()
