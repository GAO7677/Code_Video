"""Fixed-index multi-source dataset for comparable SAVi Stage 1 experiments."""

from __future__ import annotations

import json
from pathlib import Path

import decord
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Stage1Indexed(Dataset):
    SPLIT_FILES = {
        "train": "train.jsonl",
        "valid": "handoff_monitor.jsonl",
        "val": "handoff_monitor.jsonl",
    }
    KUBRIC_VISUAL_BACKGROUND_TYPES = {"dome"}
    KUBRIC_STATIC_GEOMETRY_TOKENS = {
        "cube_platform",
        "wall",
        "ground",
        "pool_table",
        "support",
        "platform",
        "floor",
    }

    def __init__(
        self,
        index_root,
        dataset_mode,
        split,
        num_frames=10,
        img_size=(216, 384),
        frame_stride=1,
        random_start=True,
        preprocess_mode="resize",
        vjepa_short_side=438,
        vjepa_crop_size=384,
        load_masks=False,
        max_mask_instances=6,
        mask_temporal_stride=2,
        mask_spatial_stride=16,
        max_samples=None,
        **kwargs,
    ):
        del kwargs
        if dataset_mode not in {"pybullet", "kubric", "mixed"}:
            raise ValueError(f"Unsupported dataset_mode={dataset_mode!r}")
        if split not in self.SPLIT_FILES:
            raise ValueError(f"Unsupported split={split!r}")
        self.dataset_mode = dataset_mode
        self.split = split
        self.num_frames = int(num_frames)
        self.img_size = tuple(int(value) for value in img_size)
        self.frame_stride = int(frame_stride)
        self.random_start = bool(random_start) and split == "train"
        if preprocess_mode not in {"resize", "vjepa"}:
            raise ValueError(f"Unsupported preprocess_mode={preprocess_mode!r}")
        self.preprocess_mode = preprocess_mode
        self.vjepa_short_side = int(vjepa_short_side)
        self.vjepa_crop_size = int(vjepa_crop_size)
        self.load_masks = bool(load_masks)
        self.max_mask_instances = int(max_mask_instances)
        self.mask_temporal_stride = int(mask_temporal_stride)
        self.mask_spatial_stride = int(mask_spatial_stride)
        if self.load_masks:
            if self.preprocess_mode != "vjepa":
                raise ValueError("Mask supervision currently requires preprocess_mode='vjepa'")
            if self.max_mask_instances < 1:
                raise ValueError("max_mask_instances must be positive")
            if self.num_frames % self.mask_temporal_stride != 0:
                raise ValueError("num_frames must divide mask_temporal_stride")
            if self.vjepa_crop_size % self.mask_spatial_stride != 0:
                raise ValueError("vjepa_crop_size must divide mask_spatial_stride")
        index_path = Path(index_root).resolve() / dataset_mode / self.SPLIT_FILES[split]
        if not index_path.is_file():
            raise FileNotFoundError(f"Stage 1 index does not exist: {index_path}")
        self.records = [
            json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line
        ]
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        if not self.records:
            raise RuntimeError(f"No records loaded from {index_path}")

    def __len__(self):
        return len(self.records)

    def _empty_mask_targets(self):
        latent_time = self.num_frames // self.mask_temporal_stride
        latent_size = self.vjepa_crop_size // self.mask_spatial_stride
        return {
            "dynamic_instance_masks": torch.zeros(
                latent_time, self.max_mask_instances, latent_size, latent_size
            ),
            "dynamic_instance_valid": torch.zeros(
                self.max_mask_instances, dtype=torch.bool
            ),
            "dynamic_union_mask": torch.zeros(latent_time, 1, latent_size, latent_size),
            "static_geometry_mask": torch.zeros(
                latent_time, 1, latent_size, latent_size
            ),
            "mask_supervision_valid": torch.tensor(False),
            "instance_supervision_valid": torch.tensor(False),
        }

    def _preprocess_segmentation(self, frames):
        source_height, source_width = frames.shape[-2:]
        scale = self.vjepa_short_side / min(source_height, source_width)
        resized_height = max(self.vjepa_crop_size, round(source_height * scale))
        resized_width = max(self.vjepa_crop_size, round(source_width * scale))
        frames = F.interpolate(frames, size=(resized_height, resized_width), mode="nearest")
        top = (resized_height - self.vjepa_crop_size) // 2
        left = (resized_width - self.vjepa_crop_size) // 2
        return frames[
            :,
            :,
            top:top + self.vjepa_crop_size,
            left:left + self.vjepa_crop_size,
        ]

    @staticmethod
    def _nearest_segmentation_ids(segmentation_rgb, available_ids, color_map):
        """Decode H.264 segmentation colors through a compact 5-bit RGB LUT."""
        colors = np.asarray(
            [color_map[str(value)] for value in available_ids], dtype=np.float32
        )
        levels = np.arange(32, dtype=np.float32) * 8.0 + 3.5
        red, green, blue = np.meshgrid(levels, levels, levels, indexing="ij")
        centers = np.stack((red, green, blue), axis=-1).reshape(-1, 3)
        nearest = np.empty((centers.shape[0],), dtype=np.int32)
        for start in range(0, centers.shape[0], 4096):
            chunk = centers[start:start + 4096]
            distances = ((chunk[:, None, :] - colors[None, :, :]) ** 2).sum(axis=-1)
            nearest[start:start + 4096] = distances.argmin(axis=1)
        rgb = segmentation_rgb.astype(np.int32)
        lut_indices = (
            (rgb[..., 0] >> 3) * 1024
            + (rgb[..., 1] >> 3) * 32
            + (rgb[..., 2] >> 3)
        )
        return np.asarray(available_ids, dtype=np.int32)[nearest[lut_indices]]

    def _pool_mask_to_latent(self, masks):
        """Average RGB-frame occupancy over V-JEPA tubelets and patches."""
        # [T,K,H,W] -> [1,K,T,H,W] -> [T_latent,K,H_latent,W_latent]
        values = masks.permute(1, 0, 2, 3).unsqueeze(0).float()
        values = F.avg_pool3d(
            values,
            kernel_size=(
                self.mask_temporal_stride,
                self.mask_spatial_stride,
                self.mask_spatial_stride,
            ),
            stride=(
                self.mask_temporal_stride,
                self.mask_spatial_stride,
                self.mask_spatial_stride,
            ),
        )
        return values[0].permute(1, 0, 2, 3).contiguous()

    def _load_kubric_mask_targets(self, record, frame_ids):
        targets = self._empty_mask_targets()
        video_path = Path(record["video_path"])
        segmentation_path = Path(
            record.get("segmentation_path", video_path.parent / "segmentation.mp4")
        )
        if not segmentation_path.is_file():
            return targets, {"mask_skip_reason": f"missing {segmentation_path}"}
        metadata = json.loads(Path(record["metadata_path"]).read_text(encoding="utf-8"))
        object_data = metadata.get("object_data", {})
        object_types = object_data.get("type", [])
        segmentation_ids = [int(value) for value in object_data.get("segmentation_id", [])]
        color_map = metadata.get("segmentation_color_map", {})
        available_ids = [value for value in segmentation_ids if str(value) in color_map]
        if not available_ids:
            return targets, {"mask_skip_reason": "no valid segmentation colors"}

        reader = decord.VideoReader(
            str(segmentation_path), ctx=decord.cpu(0), num_threads=1
        )
        if int(frame_ids[-1]) >= len(reader):
            return targets, {"mask_skip_reason": "segmentation video is too short"}
        frames = torch.from_numpy(reader.get_batch(frame_ids).asnumpy()).float()
        frames = frames.permute(0, 3, 1, 2)
        frames = self._preprocess_segmentation(frames)
        segmentation_rgb = frames.permute(0, 2, 3, 1).round().byte().numpy()
        decoded_ids = self._nearest_segmentation_ids(
            segmentation_rgb, available_ids, color_map
        )

        id_to_type = {
            segmentation_id: object_type
            for segmentation_id, object_type in zip(segmentation_ids, object_types)
        }
        visual_background_ids = {
            value
            for value in available_ids
            if id_to_type.get(value, "") in self.KUBRIC_VISUAL_BACKGROUND_TYPES
        }
        static_ids = {
            value
            for value in available_ids
            if any(
                token in id_to_type.get(value, "")
                for token in self.KUBRIC_STATIC_GEOMETRY_TOKENS
            )
            and value not in visual_background_ids
        }
        dynamic_ids = [
            value
            for value in available_ids
            if value not in visual_background_ids and value not in static_ids
        ]
        dynamic_union = torch.from_numpy(np.isin(decoded_ids, dynamic_ids)).float()
        static_union = torch.from_numpy(np.isin(decoded_ids, list(static_ids))).float()
        targets["dynamic_union_mask"] = self._pool_mask_to_latent(
            dynamic_union[:, None]
        )
        targets["static_geometry_mask"] = self._pool_mask_to_latent(
            static_union[:, None]
        )
        instance_valid = 0 < len(dynamic_ids) <= self.max_mask_instances
        if instance_valid:
            instances = torch.stack(
                [torch.from_numpy(decoded_ids == value).float() for value in dynamic_ids],
                dim=1,
            )
            pooled = self._pool_mask_to_latent(instances)
            targets["dynamic_instance_masks"][:, :len(dynamic_ids)] = pooled
            targets["dynamic_instance_valid"][:len(dynamic_ids)] = True
        targets["mask_supervision_valid"] = torch.tensor(True)
        targets["instance_supervision_valid"] = torch.tensor(instance_valid)
        details = {
            "mask_source": str(segmentation_path),
            "mask_dynamic_instances": len(dynamic_ids),
            "mask_static_instances": len(static_ids),
            "mask_visual_background_instances": len(visual_background_ids),
            "mask_ignored_ids": [
                value for value in segmentation_ids if value not in available_ids
            ],
        }
        return targets, details

    def _load_mask_targets(self, record, frame_ids):
        if record["source"] == "kubric":
            return self._load_kubric_mask_targets(record, frame_ids)
        targets = self._empty_mask_targets()
        return targets, {
            "mask_skip_reason": "PyBullet index has no precomputed segmentation target"
        }

    def __getitem__(self, index):
        record = self.records[index]
        video_path = Path(record["video_path"])
        reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
        range_start, configured_range_end = record.get("sampling_frame_range", [0, 49])
        range_end = min(int(configured_range_end), len(reader) - 1)
        required_span = 1 + (self.num_frames - 1) * self.frame_stride
        max_start = range_end - required_span + 1
        if max_start < range_start:
            raise ValueError(
                f"Cannot sample {self.num_frames} frames from {video_path} in "
                f"[{range_start}, {range_end}]"
            )
        if self.random_start:
            start = int(torch.randint(int(range_start), max_start + 1, size=(1,)).item())
        else:
            start = (int(range_start) + max_start) // 2
        frame_ids = start + np.arange(self.num_frames) * self.frame_stride
        frames = torch.from_numpy(reader.get_batch(frame_ids).asnumpy()).float()
        frames = frames.permute(0, 3, 1, 2).div_(255.0)
        source_height, source_width = frames.shape[-2:]
        resized_shape = None
        if self.preprocess_mode == "vjepa":
            scale = self.vjepa_short_side / min(source_height, source_width)
            resized_height = max(self.vjepa_crop_size, round(source_height * scale))
            resized_width = max(self.vjepa_crop_size, round(source_width * scale))
            frames = F.interpolate(
                frames,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            top = (resized_height - self.vjepa_crop_size) // 2
            left = (resized_width - self.vjepa_crop_size) // 2
            frames = frames[
                :,
                :,
                top:top + self.vjepa_crop_size,
                left:left + self.vjepa_crop_size,
            ]
            resized_shape = [resized_height, resized_width]
        else:
            frames = F.interpolate(
                frames,
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        metadata = {
            **record,
            "start_frame": start,
            "frame_ids": frame_ids.tolist(),
            "source_resolution_hw": [source_height, source_width],
            "preprocess_mode": self.preprocess_mode,
            "resized_resolution_hw": resized_shape,
            "training_shape": list(frames.shape),
        }
        if self.load_masks:
            mask_targets, mask_details = self._load_mask_targets(record, frame_ids)
            metadata["_mask_targets"] = mask_targets
            metadata.update(mask_details)
        return frames, metadata

    @staticmethod
    def collate_fn(data):
        videos = torch.stack([sample[0] for sample in data], dim=0)
        metadata = []
        mask_targets = []
        for _, raw_metadata in data:
            item_metadata = dict(raw_metadata)
            mask_targets.append(item_metadata.pop("_mask_targets", None))
            metadata.append(item_metadata)
        result = {
            "metadata": metadata,
            "sources": [item["source"] for item in metadata],
        }
        if mask_targets and mask_targets[0] is not None:
            for key in mask_targets[0]:
                result[key] = torch.stack([item[key] for item in mask_targets], dim=0)
        return videos, result
