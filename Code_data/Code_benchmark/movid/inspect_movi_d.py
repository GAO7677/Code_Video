#!/usr/bin/env python3
"""Read and inspect the MOVI-D TFRecord dataset without TensorFlow.

This script targets the local dataset layout:
  /data/gaoya/dataset/kubric_tfds_movi-d/
    train/
    test/

It can:
  1. Read a sample from `train` and/or `test` (`tests` is accepted as an alias).
  2. Parse the serialized `tf.train.Example` payload using protobuf only.
  3. Decode MOVI-D's PNG image fields and reshape the dense/ragged arrays.
  4. Print a readable summary of the dataset schema and sample shapes.

Dependencies:
  - numpy
  - Pillow
  - protobuf

Example:
  /data/gaoya/home_miniconda3/envs/bagel/bin/python inspect_movi_d.py
  /data/gaoya/home_miniconda3/envs/bagel/bin/python inspect_movi_d.py --splits train tests
  /data/gaoya/home_miniconda3/envs/bagel/bin/python inspect_movi_d.py --split train --index 0 --json
"""

from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple


def _import_runtime_deps() -> Tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - import error path
        raise SystemExit(
            "Missing dependency `numpy`. "
            "Please run this script in an environment that has numpy, Pillow, and protobuf."
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - import error path
        raise SystemExit(
            "Missing dependency `Pillow`. "
            "Please run this script in an environment that has numpy, Pillow, and protobuf."
        ) from exc

    try:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    except ImportError as exc:  # pragma: no cover - import error path
        raise SystemExit(
            "Missing dependency `protobuf`. "
            "Please run this script in an environment that has numpy, Pillow, and protobuf."
        ) from exc

    return np, Image, descriptor_pb2, descriptor_pool, message_factory


np, Image, descriptor_pb2, descriptor_pool, message_factory = _import_runtime_deps()


DEFAULT_DATASET_ROOT = Path("/data/gaoya/dataset/kubric_tfds_movi-d")
SPLIT_ALIASES = {
    "tests": "test",
    "testing": "test",
}


def build_tf_example_message():
    """Build a runtime `tf.train.Example` protobuf class without TensorFlow."""

    file_desc = descriptor_pb2.FileDescriptorProto()
    file_desc.name = "tensorflow/core/example/example.proto"
    file_desc.package = "tensorflow"
    file_desc.syntax = "proto3"

    scalar_lists = [
        ("BytesList", descriptor_pb2.FieldDescriptorProto.TYPE_BYTES),
        ("FloatList", descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT),
        ("Int64List", descriptor_pb2.FieldDescriptorProto.TYPE_INT64),
    ]
    for message_name, field_type in scalar_lists:
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
        [
            ("bytes_list", ".tensorflow.BytesList"),
            ("float_list", ".tensorflow.FloatList"),
            ("int64_list", ".tensorflow.Int64List"),
        ],
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


TF_EXAMPLE = build_tf_example_message()


def normalize_split_name(split: str) -> str:
    return SPLIT_ALIASES.get(split, split)


def get_split_dir(dataset_root: Path, split: str) -> Path:
    split = normalize_split_name(split)
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    return split_dir


def get_split_shards(dataset_root: Path, split: str) -> List[Path]:
    split_dir = get_split_dir(dataset_root, split)
    shards = sorted(split_dir.glob("*.tfrecord-*"))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards found under: {split_dir}")
    return shards


def iter_tfrecord_records(tfrecord_path: Path) -> Iterator[bytes]:
    """Yield serialized records from a TFRecord file.

    CRC fields are read and skipped. This is enough for local inspection and parsing.
    """

    with tfrecord_path.open("rb") as handle:
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(
                    f"Incomplete TFRecord header in {tfrecord_path}: expected 12 bytes, got {len(header)}."
                )
            record_length, _length_crc = struct.unpack("<QI", header)
            payload = handle.read(record_length)
            if len(payload) != record_length:
                raise ValueError(
                    f"Incomplete TFRecord payload in {tfrecord_path}: "
                    f"expected {record_length} bytes, got {len(payload)}."
                )
            data_crc = handle.read(4)
            if len(data_crc) != 4:
                raise ValueError(f"Incomplete TFRecord footer in {tfrecord_path}.")
            yield payload


def load_serialized_example(
    dataset_root: Path,
    split: str,
    index: int = 0,
) -> Tuple[bytes, Path, int]:
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")

    remaining = index
    for shard_path in get_split_shards(dataset_root, split):
        for record_in_shard, payload in enumerate(iter_tfrecord_records(shard_path)):
            if remaining == 0:
                return payload, shard_path, record_in_shard
            remaining -= 1

    raise IndexError(
        f"Requested split={normalize_split_name(split)!r} index={index}, "
        "but the dataset ran out of records."
    )


def parse_example(payload: bytes):
    example = TF_EXAMPLE()
    example.ParseFromString(payload)
    return example


def decode_utf8_list(values: Sequence[bytes]) -> List[str]:
    return [value.decode("utf-8", errors="replace") for value in values]


def decode_png_sequence(bytes_values: Sequence[bytes]) -> np.ndarray:
    frames = []
    for encoded in bytes_values:
        image = Image.open(io.BytesIO(encoded))
        frames.append(np.array(image))
    return np.stack(frames, axis=0)


def dequantize_uint(values: np.ndarray, value_range: Sequence[float], max_value: int) -> np.ndarray:
    lo, hi = float(value_range[0]), float(value_range[1])
    values = values.astype(np.float32)
    return values / float(max_value) * (hi - lo) + lo


def reshape_instance_ragged(
    flat_values: np.ndarray,
    row_lengths: np.ndarray,
    trailing_shape: Tuple[int, ...],
) -> List[np.ndarray]:
    reshaped = []
    offset = 0
    for row_length in row_lengths.tolist():
        chunk = flat_values[offset : offset + row_length]
        if trailing_shape:
            chunk = chunk.reshape(row_length, *trailing_shape)
        else:
            chunk = chunk.reshape(row_length)
        reshaped.append(chunk)
        offset += row_length
    if offset != len(flat_values):
        raise ValueError(
            f"Ragged decode mismatch: consumed {offset} values, but flat array has {len(flat_values)} values."
        )
    return reshaped


def feature_kind_and_count(feature) -> Tuple[str, int]:
    kind = feature.WhichOneof("kind") or "unknown"
    if kind == "bytes_list":
        return kind, len(feature.bytes_list.value)
    if kind == "float_list":
        return kind, len(feature.float_list.value)
    if kind == "int64_list":
        return kind, len(feature.int64_list.value)
    return kind, 0


def summarize_raw_features(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    summary = []
    for name in sorted(features):
        feature = features[name]
        kind, count = feature_kind_and_count(feature)
        item: Dict[str, Any] = {
            "name": name,
            "kind": kind,
            "count": count,
        }
        if kind == "bytes_list":
            item["sample_byte_lengths"] = [len(value) for value in feature.bytes_list.value[:3]]
        elif kind == "float_list":
            item["sample_values"] = [float(value) for value in feature.float_list.value[:5]]
        elif kind == "int64_list":
            item["sample_values"] = [int(value) for value in feature.int64_list.value[:10]]
        summary.append(item)
    return summary


def decode_movi_d_example(example) -> Dict[str, Any]:
    features = example.features.feature

    metadata = {
        "video_name": decode_utf8_list(features["metadata/video_name"].bytes_list.value)[0],
        "background": decode_utf8_list(features["background"].bytes_list.value)[0],
        "num_frames": int(features["metadata/num_frames"].int64_list.value[0]),
        "height": int(features["metadata/height"].int64_list.value[0]),
        "width": int(features["metadata/width"].int64_list.value[0]),
        "num_instances": int(features["metadata/num_instances"].int64_list.value[0]),
        "depth_range": [float(x) for x in features["metadata/depth_range"].float_list.value],
        "forward_flow_range": [
            float(x) for x in features["metadata/forward_flow_range"].float_list.value
        ],
        "backward_flow_range": [
            float(x) for x in features["metadata/backward_flow_range"].float_list.value
        ],
    }

    num_frames = metadata["num_frames"]
    height = metadata["height"]
    width = metadata["width"]
    num_instances = metadata["num_instances"]

    video = decode_png_sequence(features["video"].bytes_list.value)
    depth_u16 = decode_png_sequence(features["depth"].bytes_list.value).astype(np.uint16)
    normal_u8 = decode_png_sequence(features["normal"].bytes_list.value).astype(np.uint8)
    object_coordinates_u8 = decode_png_sequence(
        features["object_coordinates"].bytes_list.value
    ).astype(np.uint8)
    segmentations = decode_png_sequence(features["segmentations"].bytes_list.value).astype(np.uint8)

    forward_flow_u16 = np.asarray(
        features["forward_flow"].int64_list.value,
        dtype=np.uint16,
    ).reshape(num_frames, height, width, 2)
    backward_flow_u16 = np.asarray(
        features["backward_flow"].int64_list.value,
        dtype=np.uint16,
    ).reshape(num_frames, height, width, 2)

    depth = dequantize_uint(depth_u16, metadata["depth_range"], max_value=65535)
    forward_flow = dequantize_uint(
        forward_flow_u16, metadata["forward_flow_range"], max_value=65535
    )
    backward_flow = dequantize_uint(
        backward_flow_u16, metadata["backward_flow_range"], max_value=65535
    )

    camera = {
        "positions": np.asarray(features["camera/positions"].float_list.value, dtype=np.float32).reshape(
            num_frames, 3
        ),
        "quaternions": np.asarray(
            features["camera/quaternions"].float_list.value, dtype=np.float32
        ).reshape(num_frames, 4),
        "field_of_view": float(features["camera/field_of_view"].float_list.value[0]),
        "focal_length": float(features["camera/focal_length"].float_list.value[0]),
        "sensor_width": float(features["camera/sensor_width"].float_list.value[0]),
    }

    instance_track_lengths = np.asarray(
        features["instances/bboxes/ragged_row_lengths_0"].int64_list.value,
        dtype=np.int32,
    )
    bbox_frames = np.asarray(
        features["instances/bbox_frames/ragged_flat_values"].int64_list.value,
        dtype=np.int32,
    )
    bbox_frames_ragged = reshape_instance_ragged(bbox_frames, instance_track_lengths, ())
    bbox_values = np.asarray(
        features["instances/bboxes/ragged_flat_values"].float_list.value,
        dtype=np.float32,
    ).reshape(-1, 4)
    bbox_values_ragged = reshape_instance_ragged(bbox_values, instance_track_lengths, (4,))

    instances = {
        "asset_id": decode_utf8_list(features["instances/asset_id"].bytes_list.value),
        "category": np.asarray(features["instances/category"].int64_list.value, dtype=np.int32),
        "is_dynamic": np.asarray(features["instances/is_dynamic"].int64_list.value, dtype=np.int32),
        "mass": np.asarray(features["instances/mass"].float_list.value, dtype=np.float32),
        "friction": np.asarray(features["instances/friction"].float_list.value, dtype=np.float32),
        "restitution": np.asarray(
            features["instances/restitution"].float_list.value, dtype=np.float32
        ),
        "scale": np.asarray(features["instances/scale"].float_list.value, dtype=np.float32),
        "positions": np.asarray(
            features["instances/positions"].float_list.value, dtype=np.float32
        ).reshape(num_instances, num_frames, 3),
        "velocities": np.asarray(
            features["instances/velocities"].float_list.value, dtype=np.float32
        ).reshape(num_instances, num_frames, 3),
        "angular_velocities": np.asarray(
            features["instances/angular_velocities"].float_list.value, dtype=np.float32
        ).reshape(num_instances, num_frames, 3),
        "quaternions": np.asarray(
            features["instances/quaternions"].float_list.value, dtype=np.float32
        ).reshape(num_instances, num_frames, 4),
        "image_positions": np.asarray(
            features["instances/image_positions"].float_list.value, dtype=np.float32
        ).reshape(num_instances, num_frames, 2),
        "visibility": np.asarray(
            features["instances/visibility"].int64_list.value, dtype=np.int32
        ).reshape(num_instances, num_frames),
        "bboxes_3d": np.asarray(
            features["instances/bboxes_3d"].float_list.value, dtype=np.float32
        ).reshape(num_instances, num_frames, 8, 3),
        "bbox_frames": bbox_frames_ragged,
        "bboxes": bbox_values_ragged,
        "bbox_track_lengths": instance_track_lengths,
    }

    collision_count = len(features["events/collisions/frame"].int64_list.value)
    events = {
        "collisions": {
            "frame": np.asarray(
                features["events/collisions/frame"].int64_list.value, dtype=np.int32
            ),
            "force": np.asarray(
                features["events/collisions/force"].float_list.value, dtype=np.float32
            ),
            "position": np.asarray(
                features["events/collisions/position"].float_list.value, dtype=np.float32
            ).reshape(collision_count, 3),
            "image_position": np.asarray(
                features["events/collisions/image_position"].float_list.value,
                dtype=np.float32,
            ).reshape(collision_count, 2),
            "contact_normal": np.asarray(
                features["events/collisions/contact_normal"].float_list.value,
                dtype=np.float32,
            ).reshape(collision_count, 3),
            "instances": np.asarray(
                features["events/collisions/instances"].int64_list.value, dtype=np.int32
            ).reshape(collision_count, 2),
        }
    }

    return {
        "metadata": metadata,
        "camera": camera,
        "render": {
            "video": video,
            "depth_u16": depth_u16,
            "depth": depth,
            "normal_u8": normal_u8,
            "normal_unit": normal_u8.astype(np.float32) / 255.0 * 2.0 - 1.0,
            "object_coordinates_u8": object_coordinates_u8,
            "object_coordinates_unit": object_coordinates_u8.astype(np.float32) / 255.0 * 2.0
            - 1.0,
            "segmentations": segmentations,
            "forward_flow_u16": forward_flow_u16,
            "forward_flow": forward_flow,
            "backward_flow_u16": backward_flow_u16,
            "backward_flow": backward_flow,
        },
        "instances": instances,
        "events": events,
        "raw_feature_summary": summarize_raw_features(features),
    }


def build_decoded_summary(
    split: str,
    index: int,
    shard_path: Path,
    record_in_shard: int,
    decoded: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = decoded["metadata"]
    render = decoded["render"]
    instances = decoded["instances"]
    collisions = decoded["events"]["collisions"]

    return {
        "split": normalize_split_name(split),
        "requested_index": index,
        "shard_path": str(shard_path),
        "record_in_shard": record_in_shard,
        "metadata": metadata,
        "decoded_shapes": {
            "video": list(render["video"].shape),
            "depth_u16": list(render["depth_u16"].shape),
            "normal_u8": list(render["normal_u8"].shape),
            "object_coordinates_u8": list(render["object_coordinates_u8"].shape),
            "segmentations": list(render["segmentations"].shape),
            "forward_flow_u16": list(render["forward_flow_u16"].shape),
            "backward_flow_u16": list(render["backward_flow_u16"].shape),
            "camera/positions": list(decoded["camera"]["positions"].shape),
            "camera/quaternions": list(decoded["camera"]["quaternions"].shape),
            "instances/positions": list(instances["positions"].shape),
            "instances/velocities": list(instances["velocities"].shape),
            "instances/angular_velocities": list(instances["angular_velocities"].shape),
            "instances/quaternions": list(instances["quaternions"].shape),
            "instances/image_positions": list(instances["image_positions"].shape),
            "instances/visibility": list(instances["visibility"].shape),
            "instances/bboxes_3d": list(instances["bboxes_3d"].shape),
            "events/collisions/frame": list(collisions["frame"].shape),
            "events/collisions/position": list(collisions["position"].shape),
            "events/collisions/image_position": list(collisions["image_position"].shape),
            "events/collisions/contact_normal": list(collisions["contact_normal"].shape),
            "events/collisions/instances": list(collisions["instances"].shape),
        },
        "decoded_dtypes": {
            "video": str(render["video"].dtype),
            "depth_u16": str(render["depth_u16"].dtype),
            "depth": str(render["depth"].dtype),
            "normal_u8": str(render["normal_u8"].dtype),
            "normal_unit": str(render["normal_unit"].dtype),
            "object_coordinates_u8": str(render["object_coordinates_u8"].dtype),
            "object_coordinates_unit": str(render["object_coordinates_unit"].dtype),
            "segmentations": str(render["segmentations"].dtype),
            "forward_flow_u16": str(render["forward_flow_u16"].dtype),
            "forward_flow": str(render["forward_flow"].dtype),
            "backward_flow_u16": str(render["backward_flow_u16"].dtype),
            "backward_flow": str(render["backward_flow"].dtype),
        },
        "range_notes": {
            "depth_range": metadata["depth_range"],
            "forward_flow_range": metadata["forward_flow_range"],
            "backward_flow_range": metadata["backward_flow_range"],
            "normal_unit_formula_inferred": "normal_u8 / 255 * 2 - 1",
            "object_coordinates_unit_formula_inferred": "object_coordinates_u8 / 255 * 2 - 1",
            "depth_formula": "depth_u16 / 65535 * (depth_max - depth_min) + depth_min",
            "flow_formula": "flow_u16 / 65535 * (flow_max - flow_min) + flow_min",
        },
        "strings": {
            "background": metadata["background"],
            "video_name": metadata["video_name"],
            "asset_id_preview": instances["asset_id"][:5],
        },
        "instance_track_lengths": instances["bbox_track_lengths"].tolist(),
        "collision_count": int(collisions["frame"].shape[0]),
        "raw_feature_summary": decoded["raw_feature_summary"],
    }


def format_summary_text(summary: Dict[str, Any]) -> str:
    metadata = summary["metadata"]
    lines = []
    lines.append(f"Split: {summary['split']}")
    lines.append(
        f"  Record: global_index={summary['requested_index']} "
        f"shard_record={summary['record_in_shard']} "
        f"path={summary['shard_path']}"
    )
    lines.append(
        "  Metadata: "
        f"video_name={metadata['video_name']} "
        f"background={metadata['background']} "
        f"frames={metadata['num_frames']} "
        f"size={metadata['height']}x{metadata['width']} "
        f"instances={metadata['num_instances']}"
    )
    lines.append("  Decoded arrays:")
    for name, shape in summary["decoded_shapes"].items():
        dtype = summary["decoded_dtypes"].get(name, "n/a")
        lines.append(f"    {name}: shape={tuple(shape)} dtype={dtype}")
    lines.append("  Quantization:")
    for key, value in summary["range_notes"].items():
        lines.append(f"    {key}: {value}")
    lines.append(
        f"  Collision events: count={summary['collision_count']} "
        f"instance_track_lengths={summary['instance_track_lengths']}"
    )
    lines.append(
        "  Asset preview: " + ", ".join(summary["strings"]["asset_id_preview"])
    )
    lines.append("  Raw features:")
    for item in summary["raw_feature_summary"]:
        extra = ""
        if "sample_byte_lengths" in item:
            extra = f" sample_byte_lengths={item['sample_byte_lengths']}"
        elif "sample_values" in item:
            extra = f" sample_values={item['sample_values']}"
        lines.append(
            f"    {item['name']}: kind={item['kind']} count={item['count']}{extra}"
        )
    return "\n".join(lines)


def inspect_split(dataset_root: Path, split: str, index: int) -> Dict[str, Any]:
    payload, shard_path, record_in_shard = load_serialized_example(dataset_root, split, index=index)
    example = parse_example(payload)
    decoded = decode_movi_d_example(example)
    return build_decoded_summary(split, index, shard_path, record_in_shard, decoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the MOVI-D TFRecord dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Inspect a single split, for example `train` or `test` (`tests` is accepted).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Inspect multiple splits. Default is `train test`.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Global example index within a split. Default: 0.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the decoded summary as JSON instead of text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.split is not None and args.splits is not None:
        raise SystemExit("Please use either --split or --splits, not both.")

    splits = args.splits or ([args.split] if args.split is not None else ["train", "test"])
    summaries = [
        inspect_split(args.dataset_root, split=split_name, index=args.index)
        for split_name in splits
    ]

    if args.json:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return

    for idx, summary in enumerate(summaries):
        if idx:
            print()
        print(format_summary_text(summary))


if __name__ == "__main__":
    main()
