#!/usr/bin/env python3
"""Prepare MOVI-D test and Genesis held-out sets in a unified mytest/meta.json format."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import struct
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

try:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
except ImportError as exc:  # pragma: no cover
    raise SystemExit("protobuf is required to prepare MOVI-D benchmark samples.") from exc


TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
DEFAULT_FULL_META_LIST_PATH = TRAIN0419_ROOT / "benchmark_meta_json_paths_full.txt"
MOVI_D_ROOT = Path("/data/gaoya/dataset/kubric_tfds_movi-d")
MOVI_D_MYTEST_ROOT = MOVI_D_ROOT / "mytest"
GENESIS_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"
)
GENESIS_RIGID_ROOT = GENESIS_ROOT / "train" / "rigid"
GENESIS_MYTEST_ROOT = GENESIS_ROOT / "mytest"

GENESIS_HELDOUT_POOL_NAME = "benchmark_v1_reserved_seedspace_try1"
GENESIS_HELDOUT_SEED = 20260421
GENESIS_HELDOUT_COUNT = 8
DEFAULT_CONTEXT_FRAMES = 8
DEFAULT_MOVID_FPS = 12
MOVI_D_TFRECORD_MIN_BYTES = 1_000_000
MOVI_D_TFRECORD_MAX_BYTES = 100_000_000


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def clean_object_name(name: str) -> str:
    return clean_text(str(name).replace("_", " ").replace("-", " "))


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


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_frame_png(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def read_video_frames(video_path: Path) -> tuple[list[np.ndarray], int]:
    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data()
        fps = int(round(float(meta.get("fps", 12)))) if meta.get("fps") else 12
        frames = [np.asarray(frame, dtype=np.uint8) for frame in reader]
        return frames, max(1, fps)
    finally:
        reader.close()


def decode_utf8_list(values):
    return [value.decode("utf-8", errors="replace") for value in values]


def iter_tfrecord_records(tfrecord_path: Path):
    file_size = tfrecord_path.stat().st_size
    with tfrecord_path.open("rb") as handle:
        while True:
            header = handle.read(12)
            if not header or len(header) != 12:
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


def parse_tf_example(payload: bytes):
    example = TF_EXAMPLE()
    example.ParseFromString(payload)
    return example


def build_movid_prompt(features) -> str:
    background = clean_object_name(decode_utf8_list(features["background"].bytes_list.value)[0])
    asset_ids = [
        clean_object_name(value)
        for value in decode_utf8_list(features["instances/asset_id"].bytes_list.value)
    ]
    unique_assets = []
    for asset_name in asset_ids:
        if asset_name and asset_name not in unique_assets:
            unique_assets.append(asset_name)
    object_phrase = ", ".join(unique_assets[:4]) if unique_assets else "assorted objects"
    num_instances = int(features["metadata/num_instances"].int64_list.value[0])
    dynamic = sum(int(value) for value in features["instances/is_dynamic"].int64_list.value)
    return clean_text(
        f"A synthetic Kubric scene on a {background} background with "
        f"{num_instances} object(s), including {object_phrase}. "
        f"{dynamic} object(s) are dynamic."
    )


def prepare_movi_d_mytest(context_frames: int, overwrite: bool) -> list[Path]:
    output_root = MOVI_D_MYTEST_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    meta_paths: list[Path] = []
    created = 0
    for shard_path in sorted((MOVI_D_ROOT / "test").glob("*.tfrecord-*")):
        for record_index, payload in enumerate(iter_tfrecord_records(shard_path)):
            example = parse_tf_example(payload)
            features = example.features.feature
            video_name = decode_utf8_list(features["metadata/video_name"].bytes_list.value)[0]
            sample_id = f"movi_d_test_{created:04d}__video_{video_name}"
            sample_dir = output_root / sample_id
            meta_path = sample_dir / "meta.json"
            if meta_path.exists() and not overwrite:
                meta_paths.append(meta_path)
                created += 1
                continue

            frames = []
            for encoded in features["video"].bytes_list.value:
                frame = Image.open(io.BytesIO(encoded)).convert("RGB")
                frames.append(np.asarray(frame, dtype=np.uint8))

            keep_context = min(context_frames, max(1, len(frames) - 1))
            context = frames[:keep_context]
            future = frames[keep_context:]
            if not future:
                future = frames[-1:]

            sample_dir.mkdir(parents=True, exist_ok=True)
            context_video_path = sample_dir / "context_video.mp4"
            future_gt_video_path = sample_dir / "future_gt_video.mp4"
            full_video_path = sample_dir / "full_video.mp4"
            first_frame_path = sample_dir / "first_frame.png"

            write_video(context_video_path, context, fps=DEFAULT_MOVID_FPS)
            write_video(future_gt_video_path, future, fps=DEFAULT_MOVID_FPS)
            write_video(full_video_path, frames, fps=DEFAULT_MOVID_FPS)
            save_frame_png(context[0], first_frame_path)

            caption = build_movid_prompt(features)
            metadata = {
                "sample_id": sample_id,
                "caption": caption,
                "description": caption,
                "dataset": "MOVI-D",
                "split": "test",
                "fps": DEFAULT_MOVID_FPS,
                "context_frames": len(context),
                "future_frames": len(future),
                "raw_frames": len(frames),
                "video_name": video_name,
                "paths": {
                    "sample_dir": str(sample_dir),
                    "future_gt_video_path": str(future_gt_video_path),
                    "full_video_path": str(full_video_path),
                    "context_video_path": str(context_video_path),
                    "first_frame_path": str(first_frame_path),
                },
                "source_paths": {
                    "meta_json_path": str(meta_path),
                    "tfrecord_path": str(shard_path),
                    "tfrecord_record_index": record_index,
                },
            }
            write_json(meta_path, metadata)
            append_jsonl(manifest_path, metadata)
            meta_paths.append(meta_path)
            created += 1
            if created % 50 == 0:
                print(f"[MOVI-D] prepared {created} samples", flush=True)

    write_json(
        output_root / "summary.json",
        {
            "dataset": "MOVI-D",
            "split": "test",
            "num_samples": len(meta_paths),
            "output_root": str(output_root),
            "manifest_path": str(manifest_path),
            "context_frames": context_frames,
            "fps": DEFAULT_MOVID_FPS,
        },
    )
    return meta_paths


def infer_genesis_prompt(sample_dir: Path, metadata: dict) -> str:
    for name in ("caption.txt", "caption_simple.txt"):
        path = sample_dir / name
        if path.exists():
            text = clean_text(path.read_text(encoding="utf-8"))
            if text:
                return text
    caption_json = sample_dir / "caption.json"
    if caption_json.exists():
        payload = json.loads(caption_json.read_text(encoding="utf-8"))
        text = clean_text(payload.get("caption") or payload.get("simple_caption") or "")
        if text:
            return text

    names = []
    for item in metadata.get("objects", []):
        name = item.get("name") or item.get("category") or item.get("source_object_id")
        if name:
            names.append(clean_object_name(name))
    object_phrase = ", ".join(names[:4]) if names else "rigid objects"
    return clean_text(
        f"A Genesis rigid scene showing {object_phrase}. "
        f"Motion category: {metadata.get('motion_category', 'unknown')}."
    )


def select_genesis_heldout_ids(entries: list[dict]) -> list[str]:
    object_ids = sorted({entry["object_id"] for entry in entries if entry["object_id"]})
    eligible = [
        object_id
        for object_id in object_ids
        if stable_hash(object_id + GENESIS_HELDOUT_POOL_NAME) % 5 == 0
    ]
    if not eligible:
        eligible = object_ids
    import random

    random.Random(GENESIS_HELDOUT_SEED).shuffle(eligible)
    return eligible[: max(0, min(GENESIS_HELDOUT_COUNT, len(eligible)))]


def gather_genesis_entries() -> list[dict]:
    entries = []
    for metadata_path in GENESIS_RIGID_ROOT.rglob("metadata.json"):
        sample_dir = metadata_path.parent
        rgb_video_path = sample_dir / "videos" / "rgb.mp4"
        if not rgb_video_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "sample_dir": sample_dir,
                "metadata": metadata,
                "object_id": str(metadata.get("object_id", "")),
                "video_path": rgb_video_path,
            }
        )
    entries.sort(key=lambda item: str(item["sample_dir"]))
    return entries


def prepare_genesis_mytest(context_frames: int, overwrite: bool) -> list[Path]:
    output_root = GENESIS_MYTEST_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    all_entries = gather_genesis_entries()
    heldout_ids = set(select_genesis_heldout_ids(all_entries))
    test_entries = [entry for entry in all_entries if entry["object_id"] in heldout_ids]

    meta_paths: list[Path] = []
    for index, entry in enumerate(test_entries):
        metadata = entry["metadata"]
        source_dir = entry["sample_dir"]
        sample_id = f"genesis_heldout_{index:04d}__{source_dir.name}"
        sample_dir = output_root / sample_id
        meta_path = sample_dir / "meta.json"
        if meta_path.exists() and not overwrite:
            meta_paths.append(meta_path)
            continue

        frames, fps = read_video_frames(entry["video_path"])
        keep_context = min(context_frames, max(1, len(frames) - 1))
        context = frames[:keep_context]
        future = frames[keep_context:]
        if not future:
            future = frames[-1:]

        sample_dir.mkdir(parents=True, exist_ok=True)
        context_video_path = sample_dir / "context_video.mp4"
        future_gt_video_path = sample_dir / "future_gt_video.mp4"
        full_video_path = sample_dir / "full_video.mp4"
        first_frame_path = sample_dir / "first_frame.png"

        write_video(context_video_path, context, fps=fps)
        write_video(future_gt_video_path, future, fps=fps)
        shutil.copy2(entry["video_path"], full_video_path)
        save_frame_png(context[0], first_frame_path)

        caption = infer_genesis_prompt(source_dir, metadata)
        payload = {
            "sample_id": sample_id,
            "caption": caption,
            "description": caption,
            "dataset": "GenesisRigid",
            "split": "heldout",
            "fps": fps,
            "context_frames": len(context),
            "future_frames": len(future),
            "raw_frames": len(frames),
            "object_id": entry["object_id"],
            "scene_id": metadata.get("scene_id"),
            "scene_composition": metadata.get("scene_composition"),
            "interaction_pattern": metadata.get("interaction_pattern"),
            "paths": {
                "sample_dir": str(sample_dir),
                "future_gt_video_path": str(future_gt_video_path),
                "full_video_path": str(full_video_path),
                "context_video_path": str(context_video_path),
                "first_frame_path": str(first_frame_path),
            },
            "source_paths": {
                "meta_json_path": str(meta_path),
                "source_sample_dir": str(source_dir),
                "source_rgb_video_path": str(entry["video_path"]),
                "source_metadata_json_path": str(source_dir / "metadata.json"),
            },
        }
        write_json(meta_path, payload)
        append_jsonl(manifest_path, payload)
        meta_paths.append(meta_path)
        if len(meta_paths) % 50 == 0:
            print(f"[Genesis held-out] prepared {len(meta_paths)} samples", flush=True)

    write_json(
        output_root / "summary.json",
        {
            "dataset": "GenesisRigid",
            "split": "heldout",
            "num_samples": len(meta_paths),
            "output_root": str(output_root),
            "manifest_path": str(manifest_path),
            "context_frames": context_frames,
            "heldout_ids": sorted(heldout_ids),
        },
    )
    return meta_paths


def update_full_meta_list(meta_list_path: Path, new_meta_paths: list[Path]) -> None:
    existing = []
    if meta_list_path.exists():
        existing = [
            line.strip()
            for line in meta_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    merged = sorted({*existing, *(str(path) for path in new_meta_paths)})
    meta_list_path.write_text("\n".join(merged) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MOVI-D test and Genesis held-out mytest folders plus benchmark_meta_json_paths_full.txt."
    )
    parser.add_argument(
        "--meta-list-path",
        type=Path,
        default=DEFAULT_FULL_META_LIST_PATH,
        help=f"Path to benchmark meta list. Default: {DEFAULT_FULL_META_LIST_PATH}",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=DEFAULT_CONTEXT_FRAMES,
        help="Number of context frames to materialize into context_video.mp4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing meta/video files under the generated mytest roots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    movid_meta_paths = prepare_movi_d_mytest(
        context_frames=args.context_frames,
        overwrite=args.overwrite,
    )
    genesis_meta_paths = prepare_genesis_mytest(
        context_frames=args.context_frames,
        overwrite=args.overwrite,
    )
    all_new = movid_meta_paths + genesis_meta_paths
    update_full_meta_list(args.meta_list_path, all_new)
    print(
        json.dumps(
            {
                "movid_test_samples": len(movid_meta_paths),
                "genesis_heldout_samples": len(genesis_meta_paths),
                "meta_list_path": str(args.meta_list_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
