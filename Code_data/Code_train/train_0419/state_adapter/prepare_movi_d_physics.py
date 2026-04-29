#!/usr/bin/env python3
"""Convert MOVI-D TFRecord shards into Genesis-like physics samples."""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from PIL import Image
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


MOVI_D_SPLIT_ALIASES = {
    "tests": "test",
    "testing": "test",
    "trains": "train",
    "valid": "validation",
    "val": "validation",
}

MOVI_D_CATEGORY_NAMES = [
    "Action Figures",
    "Bag",
    "Board Games",
    "Bottles and Cans and Cups",
    "Camera",
    "Car Seat",
    "Consumer Goods",
    "Hat",
    "Headphones",
    "Keyboard",
    "Legos",
    "Media Cases",
    "Mouse",
    "None",
    "Shoe",
    "Stuffed Toys",
    "Toys",
]


def build_tf_example_message():
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
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True

    field = entry.field.add()
    field.name = "key"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = entry.field.add()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MOVI-D TFRecords into Genesis-like per-sample physics directories.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/data/gaoya/dataset/kubric_tfds_movi-d"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("/data/gaoya/dataset/kubric_tfds_movi-d/preprocess_v1/movi_d_physics"),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated splits to convert.",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=0,
        help="0 means convert all readable records in the selected split.",
    )
    parser.add_argument(
        "--sample_filter",
        type=str,
        default="",
        help="Only keep samples whose generated sample_id contains this substring.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples that already have metadata.json and anchor_targets.npz.",
    )
    parser.add_argument(
        "--shard_limit",
        type=int,
        default=0,
        help="Optional cap on the number of shards scanned per split.",
    )
    parser.add_argument(
        "--save_dense_modalities",
        action="store_true",
        help=(
            "Also save large dense arrays like metric flow/depth/segmentation. "
            "Default is lean mode that only keeps training-required files."
        ),
    )
    return parser.parse_args()


def normalize_split_name(name: str) -> str:
    split = str(name).strip().lower()
    split = MOVI_D_SPLIT_ALIASES.get(split, split)
    if split not in {"train", "test", "validation"}:
        raise ValueError(f"Unsupported MOVI-D split: {name}")
    return split


def parse_split_list(value: str) -> List[str]:
    return [normalize_split_name(item) for item in value.split(",") if item.strip()]


def iter_serialized_records(tfrecord_path: Path) -> Iterator[bytes]:
    with tfrecord_path.open("rb") as handle:
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                return
            record_length, _length_crc = struct.unpack("<QI", header)
            payload = handle.read(record_length)
            footer = handle.read(4)
            if len(payload) != record_length or len(footer) != 4:
                return
            yield payload


def parse_example(payload: bytes):
    example = TF_EXAMPLE()
    example.ParseFromString(payload)
    return example.features.feature


def decode_text_feature(feature) -> List[str]:
    return [value.decode("utf-8", errors="replace") for value in feature.bytes_list.value]


def decode_png_array(blob: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(blob))
    return np.array(image)


def decode_rgb_frames(blobs: Sequence[bytes]) -> np.ndarray:
    frames = [np.array(Image.open(io.BytesIO(blob)).convert("RGB")) for blob in blobs]
    return np.stack(frames, axis=0).astype(np.uint8)


def decode_image_sequence(blobs: Sequence[bytes]) -> np.ndarray:
    frames = [decode_png_array(blob) for blob in blobs]
    return np.stack(frames, axis=0)


def decode_float_tensor(feature, shape: Sequence[int]) -> np.ndarray:
    return np.asarray(feature.float_list.value, dtype=np.float32).reshape(shape)


def decode_int_tensor(feature, shape: Sequence[int], dtype=np.int64) -> np.ndarray:
    return np.asarray(feature.int64_list.value, dtype=dtype).reshape(shape)


def uint16_to_metric(raw_uint16: np.ndarray, value_range: Sequence[float]) -> np.ndarray:
    minv = float(value_range[0])
    maxv = float(value_range[1])
    if maxv <= minv:
        return np.zeros_like(raw_uint16, dtype=np.float32)
    return raw_uint16.astype(np.float32) / 65535.0 * (maxv - minv) + minv


def ragged_to_frame_boxes(
    bbox_frames_flat: np.ndarray,
    bbox_frames_row_lengths: np.ndarray,
    bboxes_flat: np.ndarray,
    bboxes_row_lengths: np.ndarray,
    num_instances: int,
    num_frames: int,
    width: int,
    height: int,
) -> np.ndarray:
    bbox_xyxy = np.zeros((num_frames, num_instances, 4), dtype=np.float32)
    frame_offset = 0
    box_offset = 0
    for obj_idx in range(num_instances):
        num_visible_frames = int(bbox_frames_row_lengths[obj_idx])
        num_boxes = int(bboxes_row_lengths[obj_idx])
        if num_visible_frames != num_boxes:
            raise ValueError(
                f"bbox ragged mismatch for object {obj_idx}: "
                f"frames={num_visible_frames} boxes={num_boxes}"
            )
        frames = bbox_frames_flat[frame_offset : frame_offset + num_visible_frames].astype(np.int32)
        boxes = bboxes_flat[box_offset : box_offset + num_boxes].astype(np.float32)
        frame_offset += num_visible_frames
        box_offset += num_boxes
        if not len(frames):
            continue
        ymin = boxes[:, 0] * float(height)
        xmin = boxes[:, 1] * float(width)
        ymax = boxes[:, 2] * float(height)
        xmax = boxes[:, 3] * float(width)
        bbox_xyxy[frames, obj_idx, 0] = xmin
        bbox_xyxy[frames, obj_idx, 1] = ymin
        bbox_xyxy[frames, obj_idx, 2] = xmax
        bbox_xyxy[frames, obj_idx, 3] = ymax
    return bbox_xyxy


def boxes_from_segmentations(
    segmentations: np.ndarray,
    num_instances: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_frames, height, width = segmentations.shape
    bbox_xyxy = np.zeros((num_frames, num_instances, 4), dtype=np.float32)
    visibility_mask = np.zeros((num_frames, num_instances), dtype=np.uint8)
    for t in range(num_frames):
        seg = segmentations[t]
        for obj_idx in range(num_instances):
            seg_id = obj_idx + 1
            ys, xs = np.nonzero(seg == seg_id)
            if ys.size == 0:
                continue
            visibility_mask[t, obj_idx] = 1
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max()) + 1
            y2 = int(ys.max()) + 1
            bbox_xyxy[t, obj_idx] = np.asarray([x1, y1, x2, y2], dtype=np.float32)
    return bbox_xyxy, visibility_mask


def center_depth_from_masks(
    depth_metric: np.ndarray,
    segmentations: np.ndarray,
    visibility_pixels: np.ndarray,
    num_instances: int,
) -> np.ndarray:
    num_frames = depth_metric.shape[0]
    center_depth = np.zeros((num_frames, num_instances), dtype=np.float32)
    for obj_idx in range(num_instances):
        seg_id = obj_idx + 1
        last_depth = 0.0
        for t in range(num_frames):
            mask = segmentations[t] == seg_id
            if np.any(mask):
                value = float(depth_metric[t][mask].mean())
                center_depth[t, obj_idx] = value
                last_depth = value
            elif visibility_pixels[t, obj_idx] > 0:
                center_depth[t, obj_idx] = last_depth
            else:
                center_depth[t, obj_idx] = last_depth
    return center_depth


def choose_main_object_index(is_dynamic: np.ndarray, visibility_pixels: np.ndarray) -> int:
    dynamic_indices = np.flatnonzero(is_dynamic.astype(bool))
    if dynamic_indices.size == 1:
        return int(dynamic_indices[0])
    if dynamic_indices.size > 1:
        scores = visibility_pixels[:, dynamic_indices].sum(axis=0)
        return int(dynamic_indices[int(np.argmax(scores))])
    scores = visibility_pixels.sum(axis=0)
    return int(np.argmax(scores)) if scores.size else 0


def compute_state_9d(
    com_uv: np.ndarray,
    center_depth: np.ndarray,
    bbox_xyxy: np.ndarray,
    visibility_pixels: np.ndarray,
    fps: float,
) -> np.ndarray:
    x1 = bbox_xyxy[..., 0]
    y1 = bbox_xyxy[..., 1]
    x2 = bbox_xyxy[..., 2]
    y2 = bbox_xyxy[..., 3]
    width = np.maximum(0.0, x2 - x1).astype(np.float32)
    height = np.maximum(0.0, y2 - y1).astype(np.float32)

    dt = 1.0 / max(float(fps), 1e-6)
    u = com_uv[..., 0].astype(np.float32)
    v = com_uv[..., 1].astype(np.float32)
    d = center_depth.astype(np.float32)
    du = np.zeros_like(u, dtype=np.float32)
    dv = np.zeros_like(v, dtype=np.float32)
    dd = np.zeros_like(d, dtype=np.float32)
    if u.shape[0] > 1:
        du[1:] = (u[1:] - u[:-1]) / dt
        dv[1:] = (v[1:] - v[:-1]) / dt
        dd[1:] = (d[1:] - d[:-1]) / dt
    vis = visibility_pixels.astype(np.float32)
    return np.stack([u, v, d, width, height, du, dv, dd, vis], axis=-1).astype(np.float32)


def build_contact_tensors(
    frames: np.ndarray,
    instances: np.ndarray,
    forces: np.ndarray,
    num_frames: int,
    num_instances: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict], List[dict]]:
    contact_graph = np.zeros((num_frames, num_instances, num_instances), dtype=np.uint8)
    contact_force = np.zeros((num_frames, num_instances, num_instances), dtype=np.float32)
    frame_has_env = np.zeros((num_frames,), dtype=np.uint8)
    raw_events: List[dict] = []
    grouped: Dict[Tuple[int, int, int], dict] = {}

    for event_index in range(int(frames.shape[0])):
        frame = int(frames[event_index])
        if frame < 0 or frame >= num_frames:
            continue
        a = int(instances[event_index, 0])
        b = int(instances[event_index, 1])
        force = float(forces[event_index])
        if a >= 65535 or b >= 65535:
            obj_idx = b if a >= 65535 else a
            if 0 <= obj_idx < num_instances:
                frame_has_env[frame] = 1
                key = (frame, min(obj_idx, -1), max(obj_idx, -1))
                grouped.setdefault(
                    key,
                    {
                        "kind": "object_environment",
                        "participants": [int(obj_idx), -1],
                        "object_indices": [int(obj_idx)],
                        "environment_name": "background_or_floor",
                        "window_type": "environment_contact",
                        "start_frame": frame,
                        "end_frame": frame,
                        "max_force": force,
                        "force_sum": force,
                    },
                )
                grouped[key]["end_frame"] = max(int(grouped[key]["end_frame"]), frame)
                grouped[key]["max_force"] = max(float(grouped[key]["max_force"]), force)
                grouped[key]["force_sum"] = float(grouped[key]["force_sum"]) + force
            raw_events.append(
                {
                    "frame": frame,
                    "instances": [a, b],
                    "force": force,
                    "kind": "object_environment",
                }
            )
            continue

        if not (0 <= a < num_instances and 0 <= b < num_instances):
            continue
        i, j = sorted((a, b))
        contact_graph[frame, i, j] = 1
        contact_graph[frame, j, i] = 1
        contact_force[frame, i, j] += force
        contact_force[frame, j, i] += force
        key = (frame, i, j)
        grouped.setdefault(
            key,
            {
                "kind": "object_object",
                "participants": [i, j],
                "object_indices": [i, j],
                "environment_name": "",
                "window_type": "collision",
                "start_frame": frame,
                "end_frame": frame,
                "max_force": force,
                "force_sum": force,
            },
        )
        grouped[key]["end_frame"] = max(int(grouped[key]["end_frame"]), frame)
        grouped[key]["max_force"] = max(float(grouped[key]["max_force"]), force)
        grouped[key]["force_sum"] = float(grouped[key]["force_sum"]) + force
        raw_events.append(
            {
                "frame": frame,
                "instances": [i, j],
                "force": force,
                "kind": "object_object",
            }
        )

    frame_phase = np.zeros((num_frames,), dtype=np.int32)
    obj_contact_mask = contact_graph.sum(axis=(1, 2)) > 0
    frame_phase[frame_has_env > 0] = 1
    frame_phase[obj_contact_mask] = 2
    frame_phase[(frame_has_env > 0) & obj_contact_mask] = 3

    event_windows: List[dict] = []
    per_pair_events: Dict[Tuple[str, Tuple[int, ...]], List[dict]] = {}
    for event in grouped.values():
        participants = tuple(int(x) for x in event["participants"])
        key = (str(event["kind"]), participants)
        per_pair_events.setdefault(key, []).append(event)

    for events in per_pair_events.values():
        events = sorted(events, key=lambda item: int(item["start_frame"]))
        merged: List[dict] = []
        for event in events:
            if merged:
                last = merged[-1]
                same_track = (
                    str(last["kind"]) == str(event["kind"])
                    and tuple(last["participants"]) == tuple(event["participants"])
                    and str(last["window_type"]) == str(event["window_type"])
                )
                if same_track and int(event["start_frame"]) <= int(last["end_frame"]) + 1:
                    last["end_frame"] = max(int(last["end_frame"]), int(event["end_frame"]))
                    last["max_force"] = max(float(last["max_force"]), float(event["max_force"]))
                    last["force_sum"] = float(last["force_sum"]) + float(event["force_sum"])
                    continue
            merged.append(dict(event))
        event_windows.extend(merged)

    event_windows = sorted(
        event_windows,
        key=lambda item: (int(item["start_frame"]), str(item["kind"]), tuple(item["participants"])),
    )
    return contact_graph, contact_force, frame_phase, raw_events, event_windows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_rgb_frames(rgb_frames: np.ndarray, rgb_dir: Path) -> None:
    ensure_dir(rgb_dir)
    for frame_idx, frame in enumerate(rgb_frames):
        Image.fromarray(frame).save(rgb_dir / f"frame_{frame_idx:03d}.png")


def write_caption_files(sample_dir: Path, prompt: str, objects: Sequence[dict]) -> None:
    payload = {
        "caption": prompt,
        "simple_caption": prompt,
        "objects": [
            {
                "object_id": int(obj["object_id"]),
                "name": str(obj.get("name", "")),
                "category": str(obj.get("category", "")),
            }
            for obj in objects
        ],
    }
    (sample_dir / "caption.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (sample_dir / "caption.txt").write_text(prompt + "\n", encoding="utf-8")
    (sample_dir / "caption_simple.txt").write_text(prompt + "\n", encoding="utf-8")


def build_prompt(background: str, asset_ids: Sequence[str], num_instances: int, dynamic_count: int) -> str:
    preview = []
    for asset_name in asset_ids:
        clean = " ".join(str(asset_name).replace("_", " ").replace("-", " ").split())
        if clean and clean not in preview:
            preview.append(clean)
        if len(preview) >= 4:
            break
    object_phrase = ", ".join(preview) if preview else "assorted objects"
    return (
        f"A synthetic Kubric scene on a {background} background with "
        f"{num_instances} object(s), including {object_phrase}. "
        f"{dynamic_count} object(s) are dynamic."
    )


def convert_record(
    *,
    features,
    split: str,
    shard_path: Path,
    record_index: int,
    out_root: Path,
    skip_existing: bool,
    save_dense_modalities: bool,
) -> dict:
    num_frames = int(features["metadata/num_frames"].int64_list.value[0])
    height = int(features["metadata/height"].int64_list.value[0])
    width = int(features["metadata/width"].int64_list.value[0])
    num_instances = int(features["metadata/num_instances"].int64_list.value[0])
    video_name = decode_text_feature(features["metadata/video_name"])[0]
    background = decode_text_feature(features["background"])[0]
    asset_ids = decode_text_feature(features["instances/asset_id"])
    is_dynamic = np.asarray(features["instances/is_dynamic"].int64_list.value, dtype=np.uint8)
    dynamic_count = int(is_dynamic.sum())
    prompt = build_prompt(background, asset_ids, num_instances, dynamic_count)
    sample_id = f"movi_d_{split}__video_{video_name}"
    count_bucket = f"count_{num_instances:02d}"
    sample_dir = out_root / split / "rigid" / "movi_d" / count_bucket / sample_id
    physics_dir = sample_dir / "physics"
    anchor_path = physics_dir / "anchor_targets.npz"
    metadata_path = sample_dir / "metadata.json"

    if skip_existing and anchor_path.exists() and metadata_path.exists():
        return {
            "sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "skipped_existing": True,
        }

    rgb_frames = decode_rgb_frames(features["video"].bytes_list.value)
    seg_raw = decode_image_sequence(features["segmentations"].bytes_list.value)
    depth_raw = decode_image_sequence(features["depth"].bytes_list.value).astype(np.uint16)
    segmentations = seg_raw.reshape(num_frames, height, width).astype(np.uint8)
    depth_metric = uint16_to_metric(
        depth_raw.reshape(num_frames, height, width),
        np.asarray(features["metadata/depth_range"].float_list.value, dtype=np.float32),
    )
    forward_flow_raw = decode_int_tensor(
        features["forward_flow"],
        (num_frames, height, width, 2),
        dtype=np.uint16,
    )
    backward_flow_raw = decode_int_tensor(
        features["backward_flow"],
        (num_frames, height, width, 2),
        dtype=np.uint16,
    )
    forward_flow_metric = uint16_to_metric(
        forward_flow_raw,
        np.asarray(features["metadata/forward_flow_range"].float_list.value, dtype=np.float32),
    )
    backward_flow_metric = uint16_to_metric(
        backward_flow_raw,
        np.asarray(features["metadata/backward_flow_range"].float_list.value, dtype=np.float32),
    )

    com_uv_norm = decode_float_tensor(features["instances/image_positions"], (num_instances, num_frames, 2))
    com_uv = np.transpose(com_uv_norm, (1, 0, 2)).astype(np.float32)
    com_uv[..., 0] *= float(width)
    com_uv[..., 1] *= float(height)

    visibility_pixels = decode_int_tensor(
        features["instances/visibility"],
        (num_instances, num_frames),
        dtype=np.int32,
    ).transpose(1, 0)
    visibility_ratio = np.clip(visibility_pixels.astype(np.float32) / float(width * height), 0.0, 1.0)
    visibility_mask = (visibility_pixels > 0).astype(np.uint8)

    bbox_frames_flat = np.asarray(
        features["instances/bbox_frames/ragged_flat_values"].int64_list.value,
        dtype=np.int32,
    )
    bbox_frames_row_lengths = np.asarray(
        features["instances/bbox_frames/ragged_row_lengths_0"].int64_list.value,
        dtype=np.int32,
    )
    bboxes_flat = np.asarray(
        features["instances/bboxes/ragged_flat_values"].float_list.value,
        dtype=np.float32,
    ).reshape(-1, 4)
    bboxes_row_lengths = np.asarray(
        features["instances/bboxes/ragged_row_lengths_0"].int64_list.value,
        dtype=np.int32,
    )
    bbox_xyxy_ragged = ragged_to_frame_boxes(
        bbox_frames_flat=bbox_frames_flat,
        bbox_frames_row_lengths=bbox_frames_row_lengths,
        bboxes_flat=bboxes_flat,
        bboxes_row_lengths=bboxes_row_lengths,
        num_instances=num_instances,
        num_frames=num_frames,
        width=width,
        height=height,
    )
    bbox_xyxy_seg, visibility_mask_seg = boxes_from_segmentations(segmentations, num_instances)
    bbox_xyxy = bbox_xyxy_ragged.copy()
    missing_boxes = (bbox_xyxy[..., 2] <= bbox_xyxy[..., 0]) | (bbox_xyxy[..., 3] <= bbox_xyxy[..., 1])
    bbox_xyxy[missing_boxes] = bbox_xyxy_seg[missing_boxes]
    visibility_mask = np.maximum(visibility_mask, visibility_mask_seg).astype(np.uint8)

    center_depth = center_depth_from_masks(
        depth_metric=depth_metric,
        segmentations=segmentations,
        visibility_pixels=visibility_pixels,
        num_instances=num_instances,
    )
    state_9d = compute_state_9d(
        com_uv=com_uv,
        center_depth=center_depth,
        bbox_xyxy=bbox_xyxy,
        visibility_pixels=visibility_ratio,
        fps=12.0,
    )

    collision_frames = np.asarray(features["events/collisions/frame"].int64_list.value, dtype=np.int32)
    collision_instances = np.asarray(
        features["events/collisions/instances"].int64_list.value,
        dtype=np.int32,
    ).reshape(-1, 2)
    collision_forces = np.asarray(features["events/collisions/force"].float_list.value, dtype=np.float32)
    collision_positions = np.asarray(
        features["events/collisions/position"].float_list.value,
        dtype=np.float32,
    ).reshape(-1, 3)
    collision_image_positions = np.asarray(
        features["events/collisions/image_position"].float_list.value,
        dtype=np.float32,
    ).reshape(-1, 2)
    collision_normals = np.asarray(
        features["events/collisions/contact_normal"].float_list.value,
        dtype=np.float32,
    ).reshape(-1, 3)

    contact_graph, contact_force, frame_phase, raw_events, event_windows = build_contact_tensors(
        frames=collision_frames,
        instances=collision_instances,
        forces=collision_forces,
        num_frames=num_frames,
        num_instances=num_instances,
    )
    collision_events = []
    for idx in range(int(collision_frames.shape[0])):
        collision_events.append(
            {
                "frame": int(collision_frames[idx]),
                "instances": [int(collision_instances[idx, 0]), int(collision_instances[idx, 1])],
                "force": float(collision_forces[idx]),
                "position": [float(x) for x in collision_positions[idx].tolist()],
                "image_position": [float(x) for x in collision_image_positions[idx].tolist()],
                "contact_normal": [float(x) for x in collision_normals[idx].tolist()],
                "kind": str(raw_events[idx]["kind"]) if idx < len(raw_events) else "",
            }
        )

    category_ids = np.asarray(features["instances/category"].int64_list.value, dtype=np.int32)
    scale = np.asarray(features["instances/scale"].float_list.value, dtype=np.float32)
    friction = np.asarray(features["instances/friction"].float_list.value, dtype=np.float32)
    restitution = np.asarray(features["instances/restitution"].float_list.value, dtype=np.float32)
    mass = np.asarray(features["instances/mass"].float_list.value, dtype=np.float32)
    objects = []
    for obj_idx in range(num_instances):
        category_id = int(category_ids[obj_idx]) if obj_idx < category_ids.shape[0] else -1
        category_name = (
            MOVI_D_CATEGORY_NAMES[category_id]
            if 0 <= category_id < len(MOVI_D_CATEGORY_NAMES)
            else "Unknown"
        )
        role = "dynamic" if bool(is_dynamic[obj_idx]) else "static"
        objects.append(
            {
                "object_id": int(obj_idx),
                "seg_id": int(obj_idx + 1),
                "role": "primary" if obj_idx == choose_main_object_index(is_dynamic, visibility_pixels) else role,
                "motion_type": role,
                "dataset_source": "MOVI-D",
                "source_object_id": str(asset_ids[obj_idx]),
                "name": str(asset_ids[obj_idx]),
                "category": category_name,
                "is_dynamic": bool(is_dynamic[obj_idx]),
                "scale": float(scale[obj_idx]),
                "friction": float(friction[obj_idx]),
                "restitution": float(restitution[obj_idx]),
                "mass": float(mass[obj_idx]),
            }
        )

    main_object_index = choose_main_object_index(is_dynamic, visibility_pixels)
    if 0 <= main_object_index < len(objects):
        objects[main_object_index]["role"] = "primary"

    focal_length = float(features["camera/focal_length"].float_list.value[0])
    sensor_width = float(features["camera/sensor_width"].float_list.value[0])
    fx = focal_length / sensor_width * float(width) if sensor_width > 0 else float(width)
    fy = fx
    depth_range = np.asarray(features["metadata/depth_range"].float_list.value, dtype=np.float32)
    metadata = {
        "scene_id": sample_id,
        "prompt": prompt,
        "dataset_source": "MOVI-D",
        "dataset_name": "MOVI-D",
        "split": split,
        "family": "movi_d",
        "bucket_label": count_bucket,
        "bucket_key": count_bucket,
        "object_count_bucket": count_bucket,
        "num_objects": int(num_instances),
        "object_id": str(asset_ids[main_object_index]) if asset_ids else "",
        "main_object_index": int(main_object_index),
        "resolution": [int(width), int(height)],
        "fps": 12.0,
        "frames": int(num_frames),
        "background": background,
        "has_contact_graph": True,
        "has_depth_metric": True,
        "has_seg": True,
        "camera_intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(width) / 2.0,
            "cy": float(height) / 2.0,
            "near": float(depth_range[0]),
            "far": float(depth_range[1]),
        },
        "camera": {
            "field_of_view": float(features["camera/field_of_view"].float_list.value[0]),
            "focal_length": focal_length,
            "sensor_width": sensor_width,
        },
        "outputs": {
            "rgb_dir": "rgb",
            "physics_dir": "physics",
        },
        "source_paths": {
            "tfrecord_path": str(shard_path),
            "tfrecord_record_index": int(record_index),
        },
        "objects": objects,
        "collision_count_bucket": (
            "c0"
            if not event_windows
            else ("c1" if len(event_windows) == 1 else "c2plus")
        ),
        "obj_env_event_count": int(sum(1 for item in event_windows if str(item["kind"]) == "object_environment")),
        "obj_obj_event_count": int(sum(1 for item in event_windows if str(item["kind"]) == "object_object")),
    }
    metadata["collision_type_bucket"] = (
        "none"
        if metadata["obj_env_event_count"] == 0 and metadata["obj_obj_event_count"] == 0
        else (
            "env_only"
            if metadata["obj_env_event_count"] > 0 and metadata["obj_obj_event_count"] == 0
            else (
                "obj_obj_only"
                if metadata["obj_env_event_count"] == 0 and metadata["obj_obj_event_count"] > 0
                else "mixed"
            )
        )
    )

    ensure_dir(sample_dir)
    ensure_dir(physics_dir)
    write_rgb_frames(rgb_frames, sample_dir / "rgb")
    write_caption_files(sample_dir, prompt, objects)
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        anchor_path,
        object_ids=np.arange(num_instances, dtype=np.int32),
        seg_ids=np.arange(1, num_instances + 1, dtype=np.int32),
        com_uv=com_uv.astype(np.float32),
        bbox_xyxy=bbox_xyxy.astype(np.float32),
        visibility_mask=visibility_mask.astype(np.uint8),
        center_depth=center_depth.astype(np.float32),
    )
    np.save(physics_dir / "state_9d.npy", state_9d.astype(np.float32))
    np.save(physics_dir / "contact_graph.npy", contact_graph.astype(np.uint8))
    np.save(physics_dir / "contact_force.npy", contact_force.astype(np.float32))
    np.save(physics_dir / "contact_impulse.npy", contact_force.astype(np.float32))
    np.save(physics_dir / "frame_phase.npy", frame_phase.astype(np.int32))
    if save_dense_modalities:
        np.save(physics_dir / "seg.npy", segmentations.astype(np.uint8))
        np.save(physics_dir / "depth_metric.npy", depth_metric.astype(np.float32))
        np.save(physics_dir / "depth_normalized.npy", depth_raw.reshape(num_frames, height, width).astype(np.uint16))
        np.save(physics_dir / "forward_flow_metric.npy", forward_flow_metric.astype(np.float32))
        np.save(physics_dir / "backward_flow_metric.npy", backward_flow_metric.astype(np.float32))
        np.save(physics_dir / "forward_flow_normalized.npy", forward_flow_raw.astype(np.uint16))
        np.save(physics_dir / "backward_flow_normalized.npy", backward_flow_raw.astype(np.uint16))
    (physics_dir / "collision_events.json").write_text(
        json.dumps(collision_events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (physics_dir / "event_windows.json").write_text(
        json.dumps(event_windows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (physics_dir / "properties.json").write_text(
        json.dumps(
            {
                "background": background,
                "video_name": video_name,
                "forward_flow_range": [float(x) for x in features["metadata/forward_flow_range"].float_list.value],
                "backward_flow_range": [float(x) for x in features["metadata/backward_flow_range"].float_list.value],
                "depth_range": [float(x) for x in depth_range.tolist()],
                "num_instances": int(num_instances),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "sample_id": sample_id,
        "sample_dir": str(sample_dir),
        "split": split,
        "num_objects": int(num_instances),
        "main_object_index": int(main_object_index),
        "event_window_count": int(len(event_windows)),
        "skipped_existing": False,
    }


def build_summary(processed: Sequence[dict], split: str) -> dict:
    effective = [item for item in processed if not item.get("skipped_existing")]
    object_hist: Dict[str, int] = {}
    for item in effective:
        key = str(item.get("num_objects", ""))
        object_hist[key] = int(object_hist.get(key, 0)) + 1
    return {
        "split": split,
        "processed_samples": int(len(effective)),
        "skipped_existing": int(sum(1 for item in processed if item.get("skipped_existing"))),
        "object_histogram": dict(sorted(object_hist.items(), key=lambda item: int(item[0]))),
    }


def iter_split_records(split_dir: Path, shard_limit: int) -> Iterator[Tuple[Path, int, bytes]]:
    shard_paths = sorted(split_dir.glob("*.tfrecord-*"))
    if int(shard_limit) > 0:
        shard_paths = shard_paths[: int(shard_limit)]
    for shard_path in shard_paths:
        for record_index, payload in enumerate(iter_serialized_records(shard_path)):
            yield shard_path, record_index, payload


def main() -> None:
    args = parse_args()
    splits = parse_split_list(args.splits)
    ensure_dir(args.out_root)

    global_manifest = {
        "dataset_root": str(args.dataset_root),
        "out_root": str(args.out_root),
        "splits": splits,
        "processed": {},
    }

    for split in splits:
        split_dir = args.dataset_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        processed: List[dict] = []
        converted = 0
        for shard_path, record_index, payload in iter_split_records(split_dir, int(args.shard_limit)):
            features = parse_example(payload)
            video_name = decode_text_feature(features["metadata/video_name"])[0]
            sample_id = f"movi_d_{split}__video_{video_name}"
            if args.sample_filter and args.sample_filter not in sample_id:
                continue
            result = convert_record(
                features=features,
                split=split,
                shard_path=shard_path,
                record_index=record_index,
                out_root=args.out_root,
                skip_existing=bool(args.skip_existing),
                save_dense_modalities=bool(args.save_dense_modalities),
            )
            processed.append(result)
            if not result.get("skipped_existing"):
                converted += 1
            if int(args.max_samples_per_split) > 0 and len(processed) >= int(args.max_samples_per_split):
                break

        summary = build_summary(processed, split)
        global_manifest["processed"][split] = summary
        split_manifest_path = args.out_root / f"{split}_manifest.json"
        split_manifest_path.write_text(
            json.dumps({"summary": summary, "samples": processed}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[{split}] processed={summary['processed_samples']} skipped_existing={summary['skipped_existing']} "
            f"total_listed={len(processed)}"
        )

    (args.out_root / "summary.json").write_text(
        json.dumps(global_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"DONE out_root={args.out_root}")


if __name__ == "__main__":
    main()
