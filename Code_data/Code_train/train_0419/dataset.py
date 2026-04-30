"""Wan 2.2 TI2V dataset builders for OpenVid, MOVI-D, and Genesis rigid data."""

import io
import hashlib
import json
import os
import random
import struct
import sys
import tempfile
from bisect import bisect_right
from pathlib import Path

import imageio
import torch
from PIL import Image

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
except ImportError:
    descriptor_pb2 = None
    descriptor_pool = None
    message_factory = None


DIFFSYNTH_ROOT = os.environ.get(
    "DIFFSYNTH_ROOT", "/home/gaoya/Code_Video/DiffSynth-Studio-main"
)
if DIFFSYNTH_ROOT and DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadAudio,
    LoadVideo,
    ToAbsolutePath,
)


WAN_SPATIAL_DIVISIBILITY = 32
MOVI_D_TFRECORD_MIN_BYTES = 1_000_000
MOVI_D_TFRECORD_MAX_BYTES = 100_000_000
MOVI_D_SPLIT_ALIASES = {"tests": "test", "testing": "test"}
GENESIS_HELDOUT_POOL_NAME = "benchmark_v1_reserved_seedspace_try1"
GENESIS_HELDOUT_DEFAULT_SEED = 20260421
GENESIS_HELDOUT_DEFAULT_COUNT = 8


def _normalize_optional_path(path):
    if path in (None, "", "none", "None"):
        return None
    return path


def _clean_text(text):
    return " ".join(str(text).strip().split())


def _stable_hash(text):
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def _decode_utf8_list(values):
    return [value.decode("utf-8", errors="replace") for value in values]


def _clean_object_name(name):
    name = str(name).replace("_", " ").replace("-", " ")
    return _clean_text(name)


def _resolve_clip_length(total_frames, requested_frames):
    clip_frames = min(total_frames, requested_frames)
    if clip_frames < 1:
        raise ValueError(f"Video is empty: total_frames={total_frames}.")
    return clip_frames


def _sample_start_index(total_frames, clip_frames):
    max_start = max(total_frames - clip_frames, 0)
    return random.randint(0, max_start) if max_start > 0 else 0


def _decode_video_path(video_path, num_frames, frame_processor, require_min_frames=False):
    reader = imageio.get_reader(video_path)
    try:
        total_frames = reader.count_frames()
        if require_min_frames and total_frames < num_frames:
            raise ValueError(
                f"Video has only {total_frames} frames, fewer than requested {num_frames}."
            )
        clip_frames = _resolve_clip_length(total_frames, num_frames)
        start_frame = _sample_start_index(total_frames, clip_frames)

        frames = []
        for frame_id in range(start_frame, start_frame + clip_frames):
            frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
            frame = frame_processor(frame)
            frames.append(frame)
        return frames
    finally:
        reader.close()


def _decode_video_bytes(raw_video, num_frames, frame_processor, require_min_frames=False):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(raw_video)
            temp_path = temp_file.name
        return _decode_video_path(
            temp_path,
            num_frames,
            frame_processor,
            require_min_frames=require_min_frames,
        )
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def _frame_processor(height, width, max_pixels):
    return ImageCropAndResize(
        height=height,
        width=width,
        max_pixels=max_pixels,
        height_division_factor=WAN_SPATIAL_DIVISIBILITY,
        width_division_factor=WAN_SPATIAL_DIVISIBILITY,
    )


class _ImagePadAndResize:
    def __init__(self, height, width):
        self.height = int(height)
        self.width = int(width)

    def __call__(self, image):
        src_width, src_height = image.size
        if src_width < 1 or src_height < 1:
            raise ValueError(f"Invalid image size: {(src_width, src_height)}")

        scale = min(self.width / src_width, self.height / src_height)
        resized_width = max(1, int(round(src_width * scale)))
        resized_height = max(1, int(round(src_height * scale)))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (self.width, self.height))
        left = (self.width - resized_width) // 2
        top = (self.height - resized_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


def _pad_frame_processor(height, width, max_pixels):
    if height is None or width is None:
        return _frame_processor(height, width, max_pixels)
    return _ImagePadAndResize(height=height, width=width)


def _build_tf_example_message():
    if descriptor_pb2 is None:
        return None

    file_desc = descriptor_pb2.FileDescriptorProto()
    file_desc.name = "tensorflow/core/example/example.proto"
    file_desc.package = "tensorflow"
    file_desc.syntax = "proto3"

    for message_name, field_type in (
        ("BytesList", descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
        ("FloatList", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        ("Int64List", descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
    ):
        message = file_desc.message_type.add()
        message.name = message_name
        field = message.field.add()
        field.name = "value"
        field.number = 1
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        field.type = field_type
        if field_type in (
            descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
            descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
        ):
            field.options.packed = True

    feature = file_desc.message_type.add()
    feature.name = "Feature"
    oneof = feature.oneof_decl.add()
    oneof.name = "kind"
    for field_number, (field_name, type_name) in enumerate(
        (
            ("bytes_list", ".tensorflow.BytesList"),
            ("float_list", ".tensorflow.FloatList"),
            ("int64_list", ".tensorflow.Int64List"),
        ),
        start=1,
    ):
        field = feature.field.add()
        field.name = field_name
        field.number = field_number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        field.type_name = type_name
        field.oneof_index = 0

    features = file_desc.message_type.add()
    features.name = "Features"
    features_entry = features.nested_type.add()
    features_entry.name = "FeatureEntry"
    features_entry.options.map_entry = True

    field = features_entry.field.add()
    field.name = "key"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = features_entry.field.add()
    field.name = "value"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Feature"

    field = features.field.add()
    field.name = "feature"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features.FeatureEntry"

    example = file_desc.message_type.add()
    example.name = "Example"
    field = example.field.add()
    field.name = "features"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_desc)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.Example"))


TF_EXAMPLE = _build_tf_example_message()


class OpenVidParquetDataset(torch.utils.data.Dataset):
    """Read OpenVid parquet shards directly without converting them to loose files."""

    dataset_kind = "openvid_parquet"

    def __init__(
        self,
        dataset_base_path,
        dataset_repeat=1,
        max_pixels=1024 * 1024,
        height=None,
        width=None,
        num_frames=81,
    ):
        if pq is None:
            raise ImportError(
                "pyarrow is required for OpenVid parquet training. "
                "Install it in the training environment first."
            )

        self.dataset_base_path = dataset_base_path
        self.dataset_repeat = dataset_repeat
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.load_from_cache = False
        self.frame_processor = _frame_processor(height, width, max_pixels)

        self.files = self._gather_files(dataset_base_path)
        if not self.files:
            raise FileNotFoundError(
                f"No parquet files found under OpenVid root: {dataset_base_path}"
            )
        self.total_rows = self.files[-1]["cumulative_rows"]
        self._parquet_handles = {}

        resolution = self._infer_resolution_from_filename()
        self.dataset_stats = {
            "name": "OpenVid",
            "kind": self.dataset_kind,
            "path": os.path.abspath(dataset_base_path),
            "num_samples": self.total_rows,
            "effective_num_samples": len(self),
            "resolution": resolution or "variable",
            "raw_frames": "variable",
        }

    @staticmethod
    def _decode_info(blob):
        info = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        if not isinstance(info, dict):
            raise TypeError(f"Expected dict in info blob, got {type(info).__name__}.")
        return info

    def _gather_files(self, root):
        parquet_paths = sorted(
            os.path.join(root, file_name)
            for file_name in os.listdir(root)
            if file_name.endswith(".parquet")
        )

        files = []
        cumulative_rows = 0
        for path in parquet_paths:
            parquet_file = pq.ParquetFile(path)
            rows = parquet_file.metadata.num_rows
            cumulative_rows += rows
            files.append(
                {
                    "path": path,
                    "rows": rows,
                    "cumulative_rows": cumulative_rows,
                }
            )
        return files

    def _infer_resolution_from_filename(self):
        for item in self.files:
            name = Path(item["path"]).stem
            parts = name.split("_")
            for part in parts:
                if "x" not in part:
                    continue
                left, right = part.split("x", 1)
                if left.isdigit() and right.isdigit():
                    return [int(right), int(left)]
        return None

    def _get_parquet_file(self, path):
        parquet_file = self._parquet_handles.get(path)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(path)
            self._parquet_handles[path] = parquet_file
        return parquet_file

    def _map_global_row(self, row_index):
        cumulative = [item["cumulative_rows"] for item in self.files]
        file_index = bisect_right(cumulative, row_index)
        previous_rows = 0 if file_index == 0 else cumulative[file_index - 1]
        local_row_index = row_index - previous_rows
        return self.files[file_index], local_row_index

    def _read_row(self, row_index):
        file_info, local_row_index = self._map_global_row(row_index)
        parquet_file = self._get_parquet_file(file_info["path"])
        row = parquet_file.read_row_group(local_row_index, columns=["info", "raw_video"])
        return (
            self._decode_info(row.column("info")[0].as_py()),
            row.column("raw_video")[0].as_py(),
        )

    def __getitem__(self, index):
        base_index = index % self.total_rows
        for attempt in range(5):
            row_index = (base_index + attempt) % self.total_rows
            try:
                info, raw_video = self._read_row(row_index)
                prompt = _clean_text(info.get("caption", ""))
                if not prompt:
                    raise ValueError("OpenVid sample is missing caption text.")
                video = _decode_video_bytes(
                    raw_video,
                    num_frames=self.num_frames,
                    frame_processor=self.frame_processor,
                    require_min_frames=True,
                )
                return {"video": video, "prompt": prompt}
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(
                        f"Failed to load OpenVid sample at row {row_index}."
                    ) from exc
        raise RuntimeError("Unexpected OpenVid dataset retry fallthrough.")

    def __len__(self):
        return self.total_rows * self.dataset_repeat


class MoviDTFRecordDataset(torch.utils.data.Dataset):
    """Read MOVI-D TFRecord shards directly without TensorFlow."""

    dataset_kind = "movi_d_tfrecord"

    def __init__(
        self,
        dataset_base_path,
        split="train",
        splits=None,
        dataset_repeat=1,
        max_pixels=1024 * 1024,
        height=None,
        width=None,
        num_frames=81,
    ):
        if TF_EXAMPLE is None:
            raise ImportError(
                "protobuf is required for MOVI-D TFRecord parsing. "
                "Install protobuf in the training environment first."
            )

        self.dataset_base_path = os.path.abspath(dataset_base_path)
        self.dataset_repeat = dataset_repeat
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.load_from_cache = False
        self.frame_processor = _pad_frame_processor(height, width, max_pixels)
        self.selected_splits = self._normalize_splits(splits or [split])
        self.record_refs, skipped_shards = self._index_records()
        if not self.record_refs:
            raise FileNotFoundError(
                f"No readable MOVI-D TFRecord records found under {self.dataset_base_path}"
            )

        metadata = self._read_metadata_preview(self.record_refs[0])
        self.dataset_stats = {
            "name": "MOVI-D",
            "kind": self.dataset_kind,
            "path": self.dataset_base_path,
            "split": ",".join(self.selected_splits),
            "num_samples": len(self.record_refs),
            "effective_num_samples": len(self),
            "resolution": [int(metadata["width"]), int(metadata["height"])],
            "raw_frames": int(metadata["num_frames"]),
            "skipped_shards": skipped_shards,
            "resize_mode": "pad_to_canvas",
        }

    @staticmethod
    def _normalize_split_name(split_name):
        return MOVI_D_SPLIT_ALIASES.get(str(split_name).strip(), str(split_name).strip())

    def _normalize_splits(self, splits):
        return [self._normalize_split_name(split) for split in splits]

    def _get_split_shards(self, split_name):
        split_dir = Path(self.dataset_base_path) / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Cannot find MOVI-D split directory: {split_dir}")
        return sorted(split_dir.glob("*.tfrecord-*"))

    @staticmethod
    def _iter_serialized_records(tfrecord_path):
        file_size = tfrecord_path.stat().st_size
        with tfrecord_path.open("rb") as handle:
            while True:
                header = handle.read(12)
                if not header:
                    return
                if len(header) != 12:
                    return
                record_length, _length_crc = struct.unpack("<QI", header)
                remaining = file_size - handle.tell()
                if (
                    record_length < MOVI_D_TFRECORD_MIN_BYTES
                    or record_length > MOVI_D_TFRECORD_MAX_BYTES
                    or record_length + 4 > remaining
                ):
                    return
                payload = handle.read(record_length)
                footer = handle.read(4)
                if len(payload) != record_length or len(footer) != 4:
                    return
                yield payload

    def _index_records(self):
        record_refs = []
        skipped_shards = []
        for split_name in self.selected_splits:
            for shard_path in self._get_split_shards(split_name):
                shard_count = 0
                for record_index, _payload in enumerate(
                    self._iter_serialized_records(shard_path)
                ):
                    record_refs.append(
                        {
                            "split": split_name,
                            "shard_path": str(shard_path),
                            "record_index": record_index,
                        }
                    )
                    shard_count += 1
                if shard_count == 0:
                    skipped_shards.append(str(shard_path))
        return record_refs, skipped_shards

    def _read_serialized_record(self, record_ref):
        shard_path = Path(record_ref["shard_path"])
        target_index = int(record_ref["record_index"])
        for record_index, payload in enumerate(self._iter_serialized_records(shard_path)):
            if record_index == target_index:
                return payload
        raise IndexError(
            f"Cannot read MOVI-D record {target_index} from shard {shard_path}."
        )

    @staticmethod
    def _parse_example(payload):
        example = TF_EXAMPLE()
        example.ParseFromString(payload)
        return example

    def _read_metadata_preview(self, record_ref):
        example = self._parse_example(self._read_serialized_record(record_ref))
        features = example.features.feature
        return {
            "num_frames": int(features["metadata/num_frames"].int64_list.value[0]),
            "height": int(features["metadata/height"].int64_list.value[0]),
            "width": int(features["metadata/width"].int64_list.value[0]),
        }

    @staticmethod
    def _build_prompt(features):
        background = _clean_object_name(
            _decode_utf8_list(features["background"].bytes_list.value)[0]
        )
        asset_ids = [
            _clean_object_name(name)
            for name in _decode_utf8_list(features["instances/asset_id"].bytes_list.value)
        ]
        unique_assets = []
        for asset_name in asset_ids:
            if asset_name and asset_name not in unique_assets:
                unique_assets.append(asset_name)
        preview = unique_assets[:4]
        num_instances = int(features["metadata/num_instances"].int64_list.value[0])
        dynamic = sum(int(x) for x in features["instances/is_dynamic"].int64_list.value)

        if preview:
            object_phrase = ", ".join(preview)
        else:
            object_phrase = "assorted objects"

        prompt = (
            f"A synthetic Kubric scene on a {background} background with "
            f"{num_instances} object(s), including {object_phrase}. "
            f"{dynamic} object(s) are dynamic."
        )
        return _clean_text(prompt)

    def _decode_frames(self, frame_bytes):
        total_frames = len(frame_bytes)
        clip_frames = _resolve_clip_length(total_frames, self.num_frames)
        start_frame = _sample_start_index(total_frames, clip_frames)

        frames = []
        for frame_id in range(start_frame, start_frame + clip_frames):
            frame = Image.open(io.BytesIO(frame_bytes[frame_id])).convert("RGB")
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames

    def __getitem__(self, index):
        base_index = index % len(self.record_refs)
        for attempt in range(5):
            row_index = (base_index + attempt) % len(self.record_refs)
            record_ref = self.record_refs[row_index]
            try:
                example = self._parse_example(self._read_serialized_record(record_ref))
                features = example.features.feature
                video = self._decode_frames(features["video"].bytes_list.value)
                prompt = self._build_prompt(features)
                return {"video": video, "prompt": prompt}
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(
                        "Failed to load MOVI-D sample "
                        f"split={record_ref['split']} shard={record_ref['shard_path']} "
                        f"record={record_ref['record_index']}."
                    ) from exc
        raise RuntimeError("Unexpected MOVI-D dataset retry fallthrough.")

    def __len__(self):
        return len(self.record_refs) * self.dataset_repeat


class GenesisRigidDataset(torch.utils.data.Dataset):
    """Read the local Genesis rigid video dataset."""

    dataset_kind = "genesis_rigid"

    def __init__(
        self,
        dataset_base_path,
        split="train",
        dataset_repeat=1,
        max_pixels=1024 * 1024,
        height=None,
        width=None,
        num_frames=81,
        heldout_seed=GENESIS_HELDOUT_DEFAULT_SEED,
        heldout_count=GENESIS_HELDOUT_DEFAULT_COUNT,
        heldout_ids=None,
    ):
        self.dataset_base_path = os.path.abspath(dataset_base_path)
        self.split = str(split).strip().lower()
        self.dataset_repeat = dataset_repeat
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.load_from_cache = False
        self.frame_processor = _frame_processor(height, width, max_pixels)

        self.samples_root = self._resolve_samples_root(self.dataset_base_path)
        all_entries = self._gather_entries(self.samples_root)
        if not all_entries:
            raise FileNotFoundError(
                f"No Genesis rigid samples found under {self.samples_root}"
            )

        self.heldout_ids = self._select_heldout_ids(
            all_entries,
            heldout_seed=heldout_seed,
            heldout_count=heldout_count,
            heldout_ids=heldout_ids,
        )
        self.entries = self._filter_entries_for_split(all_entries)
        if not self.entries:
            raise RuntimeError(
                f"Genesis rigid split {self.split!r} is empty after held-out filtering."
            )

        frames_set = sorted({int(entry["frames"]) for entry in all_entries if entry["frames"]})
        resolution_set = sorted(
            {
                tuple(entry["resolution"])
                for entry in all_entries
                if isinstance(entry["resolution"], (list, tuple)) and len(entry["resolution"]) == 2
            }
        )
        heldout_sample_count = sum(
            1 for entry in all_entries if entry["object_id"] in self.heldout_ids
        )
        self.dataset_stats = {
            "name": "GenesisRigid",
            "kind": self.dataset_kind,
            "path": self.samples_root,
            "split": self.split,
            "num_samples": len(self.entries),
            "effective_num_samples": len(self),
            "raw_total_samples": len(all_entries),
            "resolution": [list(item) for item in resolution_set] if len(resolution_set) > 1 else list(resolution_set[0]),
            "raw_frames": frames_set if len(frames_set) > 1 else frames_set[0],
            "heldout_seed": int(heldout_seed),
            "heldout_count": int(len(self.heldout_ids)),
            "heldout_ids": list(self.heldout_ids),
            "heldout_sample_count": heldout_sample_count,
        }

    @staticmethod
    def _resolve_samples_root(dataset_base_path):
        base = Path(dataset_base_path)
        if (base / "train" / "rigid").is_dir():
            return str((base / "train" / "rigid").resolve())
        return str(base.resolve())

    @staticmethod
    def _load_json(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def _build_prompt_for_entry(cls, sample_dir, metadata):
        caption_txt = sample_dir / "caption.txt"
        if caption_txt.exists():
            text = _clean_text(caption_txt.read_text(encoding="utf-8"))
            if text:
                return text

        caption_simple = sample_dir / "caption_simple.txt"
        if caption_simple.exists():
            text = _clean_text(caption_simple.read_text(encoding="utf-8"))
            if text:
                return text

        caption_json = sample_dir / "caption.json"
        if caption_json.exists():
            try:
                data = cls._load_json(caption_json)
                text = _clean_text(
                    data.get("caption")
                    or data.get("simple_caption")
                    or data.get("scene_id")
                    or ""
                )
                if text:
                    return text
            except Exception:
                pass

        object_names = []
        for item in metadata.get("objects", []):
            name = item.get("name") or item.get("category") or item.get("source_object_id")
            if name:
                object_names.append(_clean_object_name(name))
        object_names = [name for name in object_names if name]
        object_phrase = ", ".join(object_names[:4]) if object_names else "rigid objects"
        prompt = (
            f"A Genesis rigid scene showing {object_phrase}. "
            f"Motion category: {metadata.get('motion_category', 'unknown')}."
        )
        return _clean_text(prompt)

    @classmethod
    def _gather_entries(cls, samples_root):
        entries = []
        seen_sample_dirs = set()
        for meta_name in ("meta.json", "metadata.json"):
            for metadata_path in Path(samples_root).rglob(meta_name):
                sample_dir = metadata_path.parent
                if sample_dir in seen_sample_dirs:
                    continue
                seen_sample_dirs.add(sample_dir)
                rgb_video_path = sample_dir / "videos" / "rgb.mp4"
                if not rgb_video_path.exists():
                    continue
                try:
                    metadata = cls._load_json(metadata_path)
                except Exception:
                    continue

                entries.append(
                    {
                        "sample_dir": str(sample_dir),
                        "video_path": str(rgb_video_path),
                        "object_id": str(metadata.get("object_id", "")),
                        "frames": metadata.get("frames"),
                        "resolution": metadata.get("resolution"),
                        "scene_composition": metadata.get("scene_composition"),
                        "prompt": cls._build_prompt_for_entry(sample_dir, metadata),
                    }
                )
        entries.sort(key=lambda item: item["sample_dir"])
        return entries

    @staticmethod
    def _select_heldout_ids(all_entries, heldout_seed, heldout_count, heldout_ids=None):
        if heldout_ids:
            return [str(object_id) for object_id in heldout_ids]

        object_ids = sorted(
            {entry["object_id"] for entry in all_entries if entry["object_id"]}
        )
        eligible = [
            object_id
            for object_id in object_ids
            if _stable_hash(object_id + GENESIS_HELDOUT_POOL_NAME) % 5 == 0
        ]
        if not eligible:
            eligible = object_ids
        random.Random(int(heldout_seed)).shuffle(eligible)
        count = max(0, min(int(heldout_count), len(eligible)))
        return eligible[:count]

    def _filter_entries_for_split(self, all_entries):
        if self.split in ("all", "full"):
            return all_entries
        if self.split in ("train", "training"):
            return [
                entry for entry in all_entries if entry["object_id"] not in self.heldout_ids
            ]
        if self.split in ("test", "heldout", "val", "validation"):
            return [
                entry for entry in all_entries if entry["object_id"] in self.heldout_ids
            ]
        raise ValueError(
            "GenesisRigidDataset split must be one of "
            "'train', 'test', 'heldout', 'val', 'validation', or 'all'."
        )

    def __getitem__(self, index):
        base_index = index % len(self.entries)
        for attempt in range(5):
            row_index = (base_index + attempt) % len(self.entries)
            entry = self.entries[row_index]
            try:
                prompt = _clean_text(entry["prompt"])
                if not prompt:
                    raise ValueError("Genesis rigid sample is missing prompt text.")
                video = _decode_video_path(
                    entry["video_path"],
                    num_frames=self.num_frames,
                    frame_processor=self.frame_processor,
                )
                return {"video": video, "prompt": prompt}
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(
                        f"Failed to load Genesis rigid sample: {entry['sample_dir']}"
                    ) from exc
        raise RuntimeError("Unexpected Genesis rigid dataset retry fallthrough.")

    def __len__(self):
        return len(self.entries) * self.dataset_repeat


class MixedVideoDataset(torch.utils.data.Dataset):
    """Concatenate multiple video datasets and keep their summaries together."""

    dataset_kind = "mixed"

    def __init__(self, items):
        self.items = items
        self.datasets = [item["dataset"] for item in items]
        self.dataset_stats = [item["stats"] for item in items]
        self.cumulative_lengths = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_lengths.append(total)
        self.load_from_cache = False

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, index):
        if index < 0:
            raise IndexError("Negative indices are not supported.")
        dataset_index = bisect_right(self.cumulative_lengths, index)
        previous_total = 0 if dataset_index == 0 else self.cumulative_lengths[dataset_index - 1]
        local_index = index - previous_total
        return self.datasets[dataset_index][local_index]


class WanTI2VDataset:
    """Dataset wrapper that can build single-source or mixed-source training sets."""

    def __init__(
        self,
        dataset_base_path,
        dataset_metadata_path=None,
        dataset_repeat=1,
        data_file_keys="image,video",
        max_pixels=1024 * 1024,
        height=None,
        width=None,
        num_frames=81,
        framewise_decoding=False,
    ):
        dataset_metadata_path = _normalize_optional_path(dataset_metadata_path)
        self.dataset_stats = []

        dataset_specs = self._parse_dataset_specs(
            dataset_base_path=dataset_base_path,
            dataset_metadata_path=dataset_metadata_path,
            dataset_repeat=dataset_repeat,
        )
        built_items = [
            self._build_dataset_from_spec(
                spec,
                max_pixels=max_pixels,
                height=height,
                width=width,
                num_frames=num_frames,
                framewise_decoding=framewise_decoding,
                data_file_keys=data_file_keys,
            )
            for spec in dataset_specs
        ]

        if len(built_items) == 1:
            self.dataset = built_items[0]["dataset"]
            self.dataset_stats = [built_items[0]["stats"]]
        else:
            self.dataset = MixedVideoDataset(built_items)
            self.dataset_stats = self.dataset.dataset_stats

        self._print_dataset_summary()

    @staticmethod
    def _contains_parquet(dataset_base_path):
        if not dataset_base_path or not os.path.isdir(dataset_base_path):
            return False
        return any(file_name.endswith(".parquet") for file_name in os.listdir(dataset_base_path))

    @staticmethod
    def _looks_like_movi_d_root(dataset_base_path):
        base = Path(dataset_base_path)
        return (base / "train").is_dir() and any((base / "train").glob("*.tfrecord-*"))

    @staticmethod
    def _looks_like_genesis_rigid_root(dataset_base_path):
        base = Path(dataset_base_path)
        if (base / "train" / "rigid").is_dir():
            return True
        return (base / "single_object_preview").is_dir() or (base / "interaction_pair_plus_dynamic").is_dir()

    @staticmethod
    def _parse_data_file_keys(data_file_keys):
        if isinstance(data_file_keys, str):
            return [key.strip() for key in data_file_keys.split(",") if key.strip()]
        return data_file_keys

    @classmethod
    def _normalize_dataset_spec_item(cls, item, default_metadata_path, default_repeat):
        if isinstance(item, str):
            return {
                "path": item,
                "metadata_path": default_metadata_path,
                "repeat": default_repeat,
            }

        if not isinstance(item, dict):
            raise TypeError(
                f"Dataset spec must be a string path or dict, got {type(item).__name__}."
            )

        spec = dict(item)
        if "path" not in spec:
            raise ValueError(f"Dataset spec is missing required key 'path': {spec}")
        spec.setdefault("metadata_path", default_metadata_path)
        spec.setdefault("repeat", default_repeat)
        return spec

    @classmethod
    def _parse_dataset_specs(
        cls,
        dataset_base_path,
        dataset_metadata_path,
        dataset_repeat,
    ):
        if isinstance(dataset_base_path, (list, tuple)):
            return [
                cls._normalize_dataset_spec_item(item, dataset_metadata_path, dataset_repeat)
                for item in dataset_base_path
            ]

        if isinstance(dataset_base_path, str):
            stripped = dataset_base_path.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                data = json.loads(stripped)
                if isinstance(data, dict) and "datasets" in data:
                    data = data["datasets"]
                if not isinstance(data, list):
                    data = [data]
                return [
                    cls._normalize_dataset_spec_item(item, dataset_metadata_path, dataset_repeat)
                    for item in data
                ]

            if stripped.endswith(".json") and os.path.isfile(stripped):
                with open(stripped, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict) and "datasets" in data:
                    data = data["datasets"]
                if not isinstance(data, list):
                    data = [data]
                return [
                    cls._normalize_dataset_spec_item(item, dataset_metadata_path, dataset_repeat)
                    for item in data
                ]

            if "," in stripped and not os.path.exists(stripped):
                return [
                    cls._normalize_dataset_spec_item(item.strip(), dataset_metadata_path, dataset_repeat)
                    for item in stripped.split(",")
                    if item.strip()
                ]

        return [
            cls._normalize_dataset_spec_item(
                dataset_base_path,
                dataset_metadata_path,
                dataset_repeat,
            )
        ]

    def _infer_dataset_type(self, spec):
        dataset_type = spec.get("type")
        if dataset_type:
            return str(dataset_type).strip().lower()

        dataset_path = str(spec["path"])
        if self._contains_parquet(dataset_path):
            return "openvid"
        if self._looks_like_movi_d_root(dataset_path):
            return "movi_d"
        if self._looks_like_genesis_rigid_root(dataset_path):
            return "genesis_rigid"
        return "unified"

    def _build_dataset_from_spec(
        self,
        spec,
        max_pixels,
        height,
        width,
        num_frames,
        framewise_decoding,
        data_file_keys,
    ):
        dataset_type = self._infer_dataset_type(spec)
        dataset_path = spec["path"]
        dataset_repeat = int(spec.get("repeat", 1))
        metadata_path = _normalize_optional_path(spec.get("metadata_path"))

        if dataset_type == "openvid":
            dataset = OpenVidParquetDataset(
                dataset_base_path=dataset_path,
                dataset_repeat=dataset_repeat,
                max_pixels=max_pixels,
                height=height,
                width=width,
                num_frames=num_frames,
            )
            return {"dataset": dataset, "stats": dataset.dataset_stats}

        if dataset_type == "movi_d":
            splits = spec.get("splits")
            split = spec.get("split", "train")
            dataset = MoviDTFRecordDataset(
                dataset_base_path=dataset_path,
                split=split,
                splits=splits,
                dataset_repeat=dataset_repeat,
                max_pixels=max_pixels,
                height=height,
                width=width,
                num_frames=num_frames,
            )
            return {"dataset": dataset, "stats": dataset.dataset_stats}

        if dataset_type == "genesis_rigid":
            dataset = GenesisRigidDataset(
                dataset_base_path=dataset_path,
                split=spec.get("split", "train"),
                dataset_repeat=dataset_repeat,
                max_pixels=max_pixels,
                height=height,
                width=width,
                num_frames=num_frames,
                heldout_seed=spec.get("heldout_seed", GENESIS_HELDOUT_DEFAULT_SEED),
                heldout_count=spec.get("heldout_count", GENESIS_HELDOUT_DEFAULT_COUNT),
                heldout_ids=spec.get("heldout_ids"),
            )
            return {"dataset": dataset, "stats": dataset.dataset_stats}

        dataset = UnifiedDataset(
            base_path=dataset_path,
            metadata_path=metadata_path,
            repeat=dataset_repeat,
            data_file_keys=self._parse_data_file_keys(data_file_keys),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=dataset_path,
                max_pixels=max_pixels,
                height=height,
                width=width,
                height_division_factor=WAN_SPATIAL_DIVISIBILITY,
                width_division_factor=WAN_SPATIAL_DIVISIBILITY,
                num_frames=num_frames,
                time_division_factor=4 if not framewise_decoding else 1,
                time_division_remainder=1 if not framewise_decoding else 0,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(dataset_path)
                >> LoadVideo(
                    num_frames,
                    4,
                    1,
                    frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
                ),
                "input_audio": ToAbsolutePath(dataset_path) >> LoadAudio(sr=16000),
                "wantodance_music_path": ToAbsolutePath(dataset_path),
            },
        )
        stats = {
            "name": os.path.basename(os.path.abspath(dataset_path)) or "UnifiedDataset",
            "kind": "unified",
            "path": os.path.abspath(dataset_path),
            "num_samples": len(dataset),
            "effective_num_samples": len(dataset),
            "resolution": "unknown",
            "raw_frames": "unknown",
            "metadata_path": metadata_path,
        }
        return {"dataset": dataset, "stats": stats}

    def _print_dataset_summary(self):
        if not self.dataset_stats:
            return

        print("[WanTI2VDataset] Dataset summary:")
        total = 0
        for stats in self.dataset_stats:
            total += int(stats.get("effective_num_samples", stats.get("num_samples", 0)))
            line = (
                f"  - {stats.get('name', stats.get('kind', 'dataset'))}: "
                f"samples={stats.get('num_samples')} "
                f"effective={stats.get('effective_num_samples', stats.get('num_samples'))} "
                f"resolution={stats.get('resolution')} "
                f"frames={stats.get('raw_frames')}"
            )
            if stats.get("split"):
                line += f" split={stats['split']}"
            if stats.get("kind") == "genesis_rigid":
                line += (
                    f" heldout_count={stats.get('heldout_count')} "
                    f"heldout_sample_count={stats.get('heldout_sample_count')}"
                )
            if stats.get("kind") == "movi_d_tfrecord" and stats.get("skipped_shards"):
                line += f" skipped_shards={len(stats['skipped_shards'])}"
            print(line)
        print(f"  Total effective samples: {total}")

    @property
    def load_from_cache(self):
        return getattr(self.dataset, "load_from_cache", False)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
