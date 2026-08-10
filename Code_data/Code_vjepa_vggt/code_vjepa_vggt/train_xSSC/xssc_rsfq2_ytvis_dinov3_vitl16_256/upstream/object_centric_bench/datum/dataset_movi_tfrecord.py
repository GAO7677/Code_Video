"""Direct PyTorch reader for MOVi TFDS TFRecord shards.

This mirrors :class:`MOVi`'s output contract without requiring TensorFlow at
training time. Only RGB frames and instance segmentations are decoded; boxes
are recomputed from the transformed masks, as in the official xSSC reader.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
import re
import struct
import warnings

from einops import rearrange
import cv2
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from ..util_datum import mask_segment_to_bbox_np


_SHARD_PATTERN = re.compile(r"tfrecord-(\d+)-of-(\d+)$")


def _read_varint(data, position):
    value = 0
    shift = 0
    while position < len(data):
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7
        if shift >= 70:
            raise ValueError("Invalid protobuf varint")
    raise ValueError("Truncated protobuf varint")


def _iter_protobuf_fields(data):
    """Yield ``(field_number, wire_type, value)`` without copying payloads."""
    position = 0
    while position < len(data):
        tag, position = _read_varint(data, position)
        field_number, wire_type = tag >> 3, tag & 7
        if wire_type == 0:
            value, position = _read_varint(data, position)
        elif wire_type == 1:
            value = data[position : position + 8]
            position += 8
        elif wire_type == 2:
            size, position = _read_varint(data, position)
            value = data[position : position + size]
            position += size
        elif wire_type == 5:
            value = data[position : position + 4]
            position += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type: {wire_type}")
        if position > len(data):
            raise ValueError("Truncated protobuf field")
        yield field_number, wire_type, value


def _decode_bytes_list(feature):
    for field_number, _, bytes_list in _iter_protobuf_fields(feature):
        if field_number == 1:
            return [
                value
                for number, _, value in _iter_protobuf_fields(bytes_list)
                if number == 1
            ]
    return []


def _decode_int64_list(feature):
    values = []
    for field_number, _, int64_list in _iter_protobuf_fields(feature):
        if field_number != 3:
            continue
        for number, wire_type, value in _iter_protobuf_fields(int64_list):
            if number != 1:
                continue
            if wire_type == 0:
                values.append(value)
            elif wire_type == 2:
                position = 0
                while position < len(value):
                    item, position = _read_varint(value, position)
                    values.append(item)
    return values


def _extract_example_features(serialized, requested):
    root = next(
        (
            value
            for number, _, value in _iter_protobuf_fields(serialized)
            if number == 1
        ),
        None,
    )
    if root is None:
        raise ValueError("TFRecord payload is not a tf.train.Example")

    result = {}
    for number, _, entry in _iter_protobuf_fields(root):
        if number != 1:
            continue
        key = None
        feature = None
        for entry_number, _, value in _iter_protobuf_fields(entry):
            if entry_number == 1:
                key = bytes(value).decode("utf-8")
            elif entry_number == 2:
                feature = value
        if key in requested and feature is not None:
            result[key] = feature

    missing = requested.difference(result)
    if missing:
        raise KeyError(f"Missing MOVi TFRecord features: {sorted(missing)}")
    return result


class MOViTFRecord(ptud.Dataset):
    """Read raw MOVi-C TFDS shards with the xSSC ``MOVi`` sample format.

    Returned fields:
    - ``video``: ``[T, 3, H, W]``, uint8 before ``transform`` and usually
      normalized float32 afterwards.
    - ``segment``: ``[T, H, W, S]`` bool one-hot masks, including background
      when it remains visible after spatial augmentation.
    - ``bbox``: optional ``[T, S_fg, 4]`` float32 normalized LTRB boxes.

    The class is map-style, so it works with xSSC's ``DistributedSampler``.
    ``transform0`` is applied to encoded frame lists and should contain only
    temporal slicing. Spatial transforms belong in ``transform``.
    """

    def __init__(
        self,
        data_file,
        split="train",
        extra_keys=("segment", "bbox"),
        transform0=lambda **sample: sample,
        transform=lambda **sample: sample,
        base_dir=None,
        require_complete=False,
        index_cache_dir=None,
    ):
        if base_dir is not None:
            data_file = Path(base_dir) / data_file
        self.data_dir = self._resolve_version_dir(Path(data_file))
        self.split = "validation" if split == "val" else split
        self.extra_keys = tuple(extra_keys)
        self.transform0 = transform0
        self.transform = transform

        supported = {"segment", "bbox"}
        unknown = set(self.extra_keys).difference(supported)
        if unknown:
            raise ValueError(f"Unsupported extra_keys: {sorted(unknown)}")
        if "bbox" in self.extra_keys and "segment" not in self.extra_keys:
            raise ValueError("bbox requires segment in extra_keys")

        info = json.loads((self.data_dir / "dataset_info.json").read_text())
        split_info = next(
            (item for item in info["splits"] if item["name"] == self.split), None
        )
        if split_info is None:
            names = [item["name"] for item in info["splits"]]
            raise ValueError(f"Unknown split {split!r}; available splits: {names}")

        shard_lengths = [int(value) for value in split_info["shardLengths"]]
        shard_paths = sorted(
            self.data_dir.glob(f"movi_c-{self.split}.tfrecord-*-of-*")
        )
        if not shard_paths:
            raise FileNotFoundError(
                f"No local TFRecord shards for split {self.split!r} in "
                f"{self.data_dir}"
            )

        indexed_shards = []
        local_shards = set()
        for shard_path in shard_paths:
            match = _SHARD_PATTERN.search(shard_path.name)
            if match is None:
                continue
            shard_index, shard_total = map(int, match.groups())
            if shard_total != len(shard_lengths) or shard_index >= shard_total:
                raise ValueError(f"Shard name conflicts with metadata: {shard_path}")
            local_shards.add(shard_index)
            indexed_shards.append(
                (shard_path, shard_index, shard_lengths[shard_index])
            )

        if index_cache_dir is None:
            self.records = self._build_records(indexed_shards)
        else:
            self.records = self._load_or_build_cached_records(
                indexed_shards, Path(index_cache_dir)
            )

        missing_shards = sorted(set(range(len(shard_lengths))) - local_shards)
        if missing_shards:
            message = (
                f"MOVi-C {self.split} is incomplete: found {len(local_shards)}/"
                f"{len(shard_lengths)} shards and {len(self.records)}/"
                f"{sum(shard_lengths)} samples"
            )
            if require_complete:
                raise FileNotFoundError(message)
            warnings.warn(message, stacklevel=2)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        shard_path, payload_offset, payload_size = self.records[index]
        with open(shard_path, "rb") as stream:
            stream.seek(payload_offset)
            serialized = memoryview(stream.read(payload_size))
        if len(serialized) != payload_size:
            raise OSError(f"Truncated TFRecord payload in {shard_path}")

        requested = {"video", "metadata/num_instances"}
        if "segment" in self.extra_keys:
            requested.add("segmentations")
        features = _extract_example_features(serialized, requested)

        video_encoded = _decode_bytes_list(features["video"])
        int_values = _decode_int64_list(features["metadata/num_instances"])
        if len(int_values) != 1:
            raise ValueError("metadata/num_instances must contain one value")
        num_instances = int(int_values[0])

        sample0 = {"video": video_encoded, "s": num_instances}
        if "segment" in self.extra_keys:
            sample0["segment"] = _decode_bytes_list(features["segmentations"])
        sample0 = self.transform0(**sample0)

        video = np.stack([self._decode_rgb(frame) for frame in sample0["video"]])
        sample1 = {"video": pt.from_numpy(video).permute(0, 3, 1, 2)}
        if "segment" in self.extra_keys:
            segment = np.stack(
                [self._decode_segment(frame) for frame in sample0["segment"]]
            )
            sample1["segment"] = pt.from_numpy(segment)

        sample2 = self.transform(**sample1)
        if "segment" not in self.extra_keys:
            return sample2

        segment = sample2["segment"]
        if segment.min() < 0 or segment.max() > num_instances:
            raise ValueError(
                f"Segmentation id outside [0, {num_instances}] in sample {index}"
            )
        masks = ptnf.one_hot(segment.long(), num_instances + 1).bool()
        present = masks.any(dim=(0, 1, 2))
        masks = masks[..., present]
        sample2["segment"] = masks

        if "bbox" in self.extra_keys:
            foreground_start = 1 if present[0] else 0
            flattened = rearrange(
                masks[..., foreground_start:], "t h w s -> h w (t s)"
            )
            boxes = pt.from_numpy(mask_segment_to_bbox_np(flattened.numpy())).float()
            boxes = rearrange(boxes, "(t s) c -> t s c", t=masks.shape[0])
            height, width = masks.shape[1:3]
            boxes[..., 0::2] /= width
            boxes[..., 1::2] /= height
            sample2["bbox"] = boxes

        return sample2

    @staticmethod
    def _resolve_version_dir(data_dir):
        candidates = [
            data_dir,
            data_dir / "1.0.0",
            data_dir / "256x256" / "1.0.0",
        ]
        for candidate in candidates:
            if (candidate / "dataset_info.json").is_file():
                return candidate
        raise FileNotFoundError(
            f"Cannot find MOVi dataset_info.json below {data_dir}"
        )

    @staticmethod
    def _index_shard(shard_path):
        records = []
        file_size = shard_path.stat().st_size
        with shard_path.open("rb") as stream:
            position = 0
            while position < file_size:
                header = stream.read(12)
                if len(header) != 12:
                    raise OSError(f"Truncated TFRecord header in {shard_path}")
                payload_size = struct.unpack("<Q", header[:8])[0]
                payload_offset = position + 12
                next_position = payload_offset + payload_size + 4
                if next_position > file_size:
                    raise OSError(f"Truncated TFRecord payload in {shard_path}")
                records.append((str(shard_path), payload_offset, payload_size))
                stream.seek(payload_size + 4, 1)
                position = next_position
        return records

    @classmethod
    def _index_expected_shard(cls, indexed_shard):
        shard_path, _, expected = indexed_shard
        shard_records = cls._index_shard(shard_path)
        if len(shard_records) != expected:
            raise ValueError(
                f"{shard_path.name} has {len(shard_records)} records; "
                f"expected {expected}"
            )
        return shard_records

    @classmethod
    def _build_records(cls, indexed_shards):
        worker_count = int(os.environ.get("MOVI_INDEX_WORKERS", "8"))
        if worker_count <= 0:
            raise ValueError("MOVI_INDEX_WORKERS must be positive")
        worker_count = min(worker_count, len(indexed_shards))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            shard_record_groups = executor.map(
                cls._index_expected_shard, indexed_shards
            )
            return [
                record
                for shard_records in shard_record_groups
                for record in shard_records
            ]

    def _load_or_build_cached_records(self, indexed_shards, cache_dir):
        """Build the large TFRecord offset table once across all DDP ranks."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_key = hashlib.sha256(str(self.data_dir.resolve()).encode()).hexdigest()[:16]
        cache_file = cache_dir / f"movi_c-{self.split}-{data_key}.index.json"
        lock_file = cache_file.with_suffix(f"{cache_file.suffix}.lock")
        manifest = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "shard_index": shard_index,
                "expected_records": expected,
            }
            for path, shard_index, expected in indexed_shards
        ]
        paths_by_name = {path.name: str(path) for path, _, _ in indexed_shards}

        with lock_file.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if cache_file.is_file():
                cached = json.loads(cache_file.read_text())
                if (
                    cached.get("format") == "movi_tfrecord_index_v1"
                    and cached.get("data_dir") == str(self.data_dir.resolve())
                    and cached.get("split") == self.split
                    and cached.get("shards") == manifest
                ):
                    print(
                        f"[movi-index] cache hit split={self.split} "
                        f"records={len(cached['records'])} file={cache_file}",
                        flush=True,
                    )
                    return [
                        (paths_by_name[name], offset, size)
                        for name, offset, size in cached["records"]
                    ]
                warnings.warn(
                    f"Ignoring stale MOVi index cache: {cache_file}",
                    stacklevel=2,
                )

            print(
                f"[movi-index] building split={self.split} "
                f"shards={len(indexed_shards)} "
                f"workers={os.environ.get('MOVI_INDEX_WORKERS', '8')} "
                f"file={cache_file}",
                flush=True,
            )
            records = self._build_records(indexed_shards)
            payload = {
                "format": "movi_tfrecord_index_v1",
                "data_dir": str(self.data_dir.resolve()),
                "split": self.split,
                "shards": manifest,
                "records": [
                    [Path(path).name, offset, size]
                    for path, offset, size in records
                ],
            }
            temporary_file = cache_file.with_suffix(
                f"{cache_file.suffix}.tmp-{os.getpid()}"
            )
            try:
                temporary_file.write_text(json.dumps(payload, separators=(",", ":")))
                os.replace(temporary_file, cache_file)
            finally:
                temporary_file.unlink(missing_ok=True)
            print(
                f"[movi-index] cache ready split={self.split} "
                f"records={len(records)} file={cache_file}",
                flush=True,
            )
            return records

    @staticmethod
    def _decode_rgb(encoded):
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode MOVi RGB frame")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _decode_segment(encoded):
        image = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if image is None:
            raise ValueError("Failed to decode MOVi segmentation frame")
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D segmentation, got {image.shape}")
        return image
