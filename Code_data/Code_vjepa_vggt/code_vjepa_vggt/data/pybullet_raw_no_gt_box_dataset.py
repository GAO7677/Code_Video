from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from code_vjepa_vggt.data.pybullet_prompt_cache import (
    PyBulletPromptCacheError,
    PyBulletPromptEmbeddingCache,
)
from code_vjepa_vggt.data.pybullet_vae_cache import (
    PyBulletVaeCacheError,
    PyBulletVaeLatentCache,
)
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, sample_frame_indices


@dataclass(frozen=True, slots=True)
class RawWindow:
    video_path: Path
    meta_path: Path
    split: str
    family_slug: str
    sample_id: str
    window_start: int
    caption: str
    negative_prompt: str
    object_phrases: tuple[str, ...]
    dynamic_object_phrases: tuple[str, ...]
    static_object_phrases: tuple[str, ...]

    @property
    def key(self) -> str:
        base = f"raw0613/{self.split}/{self.family_slug}/{self.sample_id}"
        return base if self.window_start == 0 else f"{base}/window_{self.window_start:06d}"

    @property
    def manifest_path(self) -> str:
        return str(self.meta_path)

    @property
    def prompt_roles(self) -> dict[str, str | list[str]]:
        return {
            "positive_prompt": self.caption,
            "negative_prompt": self.negative_prompt,
            "object_phrases": list(self.object_phrases),
            "dynamic_object_phrases": list(self.dynamic_object_phrases),
            "static_object_phrases": list(self.static_object_phrases),
        }


_SHAPE_PHRASES = {
    "sphere": ("ball", "round rigid object"),
    "puck": ("puck", "flat round rigid object"),
    "box": ("block", "box-shaped rigid object"),
    "cylinder": ("cylinder", "cylindrical rigid object"),
    "capsule": ("capsule", "capsule-shaped rigid object"),
}


def _object_prompt_roles(metadata: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    objects: list[str] = []
    dynamic: list[str] = []
    static: list[str] = []
    for item in metadata.get("objects", []):
        if not isinstance(item, dict):
            continue
        shape = str(item.get("shape", "")).strip().lower()
        noun = _SHAPE_PHRASES.get(shape, (shape, ""))[0]
        if not noun:
            continue
        phrase = f"a {noun}"
        objects.append(phrase)
        (dynamic if bool(item.get("dynamic")) else static).append(phrase)
    return tuple(dict.fromkeys(objects)), tuple(dict.fromkeys(dynamic)), tuple(dict.fromkeys(static))


def _objects_by_shape(metadata: dict[str, Any], shape: str) -> list[dict[str, Any]]:
    return [
        item
        for item in metadata.get("objects", [])
        if isinstance(item, dict) and str(item.get("shape")) == shape
    ]


def _motion_direction(metadata: dict[str, Any]) -> str:
    dynamic = [
        item
        for item in metadata.get("objects", [])
        if isinstance(item, dict) and bool(item.get("dynamic"))
    ]
    if not dynamic:
        return "across the scene"
    velocity = dynamic[0].get("linear_velocity", [0.0, 0.0, 0.0])
    if isinstance(velocity, list) and velocity:
        if float(velocity[0]) > 0.2:
            return "from left to right"
        if float(velocity[0]) < -0.2:
            return "from right to left"
    return "across the scene"


def _english_caption(metadata: dict[str, Any], family_slug: str) -> str:
    direction = _motion_direction(metadata)
    spheres = _objects_by_shape(metadata, "sphere")
    pucks = _objects_by_shape(metadata, "puck")
    boxes = _objects_by_shape(metadata, "box")
    cylinders = _objects_by_shape(metadata, "cylinder")
    if family_slug == "F1_single_object" and spheres:
        return (
            f"A ball, a round rigid object, enters {direction}, falls under gravity, "
            "bounces on the floor, and rolls forward."
        )
    if family_slug == "F2_two_object":
        mover, generic = _SHAPE_PHRASES["puck"] if pucks else _SHAPE_PHRASES["sphere"]
        target, target_generic = (
            _SHAPE_PHRASES["box"] if boxes else ("target", "stationary rigid object")
        )
        return (
            f"A moving {mover}, a {generic}, travels {direction} and collides with a "
            f"{target}, a {target_generic}, transferring momentum and causing deflection."
        )
    if family_slug == "F3_chain_reaction" and spheres and len(boxes) >= 2:
        return (
            f"A moving ball, a round rigid object, travels {direction} and hits the first "
            "block, causing two box-shaped rigid objects to move in sequence in a chain reaction."
        )
    if family_slug == "F4_occlusion" and spheres and cylinders:
        return (
            f"A moving ball, a round rigid object, travels {direction}, passes behind two "
            "stationary cylinders acting as occluders, and continues its motion after reappearing."
        )
    if family_slug == "F5_drop_support" and spheres and boxes:
        return (
            "A ball, a round rigid object, falls under gravity onto a stationary block-shaped "
            "support, then rolls across and leaves the supporting surface."
        )
    shape_terms = [
        _SHAPE_PHRASES[str(item["shape"])][0]
        for item in metadata.get("objects", [])
        if isinstance(item, dict) and str(item.get("shape")) in _SHAPE_PHRASES
    ]
    visible = ", ".join(shape_terms) if shape_terms else "rigid bodies"
    return f"The scene contains {visible}, which move and interact according to rigid-body physics."


class PyBulletRawNoGTBoxDataset(Dataset):
    """Read complete raw PyBullet videos for the no-GT-box Stage1B branch."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: tuple[int, int],
        *,
        num_frames: int = 49,
        num_context_frames: int = 8,
        sampling_strategy: str = "prefix",
        window_starts: tuple[int, ...] = (0,),
        init_scan_limit: int | None = None,
        vae_cache_dir: str | Path | None = None,
        vae_checkpoint_path: str | Path | None = None,
        prompt_cache_dir: str | Path | None = None,
        text_encoder_checkpoint_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split).strip().lower()
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.num_frames = int(num_frames)
        self.num_context_frames = int(num_context_frames)
        self.sampling_strategy = str(sampling_strategy)
        self.window_starts = tuple(dict.fromkeys(int(value) for value in window_starts))
        if self.num_context_frames > self.num_frames:
            raise ValueError("num_context_frames cannot exceed num_frames")
        if self.sampling_strategy not in {"prefix", "uniform"}:
            raise ValueError(f"unsupported sampling_strategy={self.sampling_strategy!r}")
        if not self.window_starts or min(self.window_starts) < 0:
            raise ValueError(f"window_starts must contain non-negative values: {self.window_starts}")
        if self.sampling_strategy == "uniform" and self.window_starts != (0,):
            raise ValueError("window_starts are only supported with prefix sampling")
        if self.split not in {"train", "val", "test", "all"}:
            raise ValueError(f"unsupported split={split!r}")

        split_names = ("train", "val", "test") if self.split == "all" else (self.split,)
        videos = sorted(
            video
            for split_name in split_names
            for video in (self.root / split_name).glob("*/sample_*/video.mp4")
        )
        if init_scan_limit is not None:
            videos = videos[: max(1, int(init_scan_limit))]
        self.samples: list[RawWindow] = []
        for video_path in videos:
            meta_path = video_path.with_name("meta.json")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            family_slug = video_path.parent.parent.name
            prompt_roles = _object_prompt_roles(metadata)
            for window_start in self.window_starts:
                self.samples.append(
                    RawWindow(
                        video_path=video_path,
                        meta_path=meta_path,
                        split=video_path.parent.parent.parent.name,
                        family_slug=family_slug,
                        sample_id=video_path.parent.name,
                        window_start=window_start,
                        caption=_english_caption(metadata, family_slug),
                        negative_prompt="",
                        object_phrases=prompt_roles[0],
                        dynamic_object_phrases=prompt_roles[1],
                        static_object_phrases=prompt_roles[2],
                    )
                )
        if not self.samples:
            raise RuntimeError(f"no */sample_*/video.mp4 found for split={self.split} under {self.root}")
        keys = [record.key for record in self.samples]
        if len(keys) != len(set(keys)):
            raise RuntimeError("raw PyBullet index contains duplicate logical keys")

        self.vae_cache = None
        if vae_cache_dir is not None:
            self.vae_cache = PyBulletVaeLatentCache(
                vae_cache_dir,
                resolution=self.resolution,
                num_frames=self.num_frames,
                sampling_strategy=self.sampling_strategy,
                vae_checkpoint_path=vae_checkpoint_path,
            )
            self.vae_cache.validate_records(self.samples, self.root)

        self.prompt_cache = None
        if prompt_cache_dir is not None:
            self.prompt_cache = PyBulletPromptEmbeddingCache(
                prompt_cache_dir,
                text_encoder_checkpoint_path=text_encoder_checkpoint_path,
                tokenizer_path=tokenizer_path,
            )
            self.prompt_cache.validate_records(self.samples, self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        try:
            return self._load_sample(self.samples[idx])
        except (PyBulletVaeCacheError, PyBulletPromptCacheError):
            raise

    def _load_sample(self, record: RawWindow) -> dict[str, Any]:
        video_path = record.video_path
        metadata = json.loads(record.meta_path.read_text(encoding="utf-8"))
        vr = VideoReader(str(video_path), ctx=cpu(0))
        source_frames = len(vr)
        if source_frames < self.num_frames + record.window_start:
            raise RuntimeError(
                f"{video_path} has {source_frames} frames, requested window "
                f"[{record.window_start}, {record.window_start + self.num_frames})"
            )
        if self.sampling_strategy == "uniform":
            frame_indices_np = sample_frame_indices(source_frames, self.num_frames)
        else:
            frame_indices_np = np.arange(
                record.window_start,
                record.window_start + self.num_frames,
                dtype=np.int64,
            )
        frames = vr.get_batch(frame_indices_np).asnumpy()
        video = preprocess_video_rgb_uint8(
            frames,
            self.resolution,
            value_range="minus_one_to_one",
        )
        context_indices = torch.arange(self.num_context_frames, dtype=torch.long)
        metadata = {
            **metadata,
            "dataset_name": "pybullet0613_raw",
            "sample_key": record.key,
            "source_video_path": str(video_path),
            "source_frame_count": int(source_frames),
            "sampled_frame_indices": frame_indices_np.tolist(),
            "sampling_strategy": self.sampling_strategy,
            "window_start": record.window_start,
            "family_slug": record.family_slug,
            "manifest_path": str(record.meta_path),
            "negative_prompt": record.negative_prompt,
            "object_phrases": list(record.object_phrases),
            "dynamic_object_phrases": list(record.dynamic_object_phrases),
            "static_object_phrases": list(record.static_object_phrases),
        }
        sample = {
            "video": video,
            "context_video": video[:, context_indices].contiguous(),
            "caption": record.caption,
            "video_path": str(video_path),
            "frame_indices": torch.arange(self.num_frames, dtype=torch.long),
            "context_frame_indices": context_indices,
            "num_context_frames": self.num_context_frames,
            "metadata": metadata,
        }
        if self.vae_cache is not None:
            sample["precomputed_input_latents"] = self.vae_cache.load(record.key)
            metadata["vae_cache"] = {
                "hit": True,
                "encoding_id": self.vae_cache.encoding_id,
                "cache_dir": str(self.vae_cache.cache_dir),
            }
        if self.prompt_cache is not None:
            sample["precomputed_prompt_embedding"] = self.prompt_cache.load(record.key)
            metadata["prompt_cache"] = {
                "hit": True,
                "encoding_id": self.prompt_cache.encoding_id,
                "cache_dir": str(self.prompt_cache.cache_dir),
            }
        return sample
