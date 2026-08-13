#!/usr/bin/env python3
"""Extract decoder-free, repeated-prefix causal xSSC trajectories."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional


STAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_ROOT.parent
UPSTREAM_ROOT = PROJECT_ROOT / "upstream"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UPSTREAM_ROOT))
sys.path.insert(0, "/home/gaoya/Code_Video/vjepa2-main")

from stage1_causal_state_probe import (  # noqa: E402
    CALIBRATION_STATES,
    LABEL_FRAME_INDICES,
    NUM_OBJECTS,
    NUM_RAW_FRAMES,
    NUM_SLOTS,
    NUM_STATES,
    SLOT_DIM,
)
from stage1_causal_state_probe.alignment import calibrate_identity  # noqa: E402
from stage1_causal_state_probe.io_utils import (  # noqa: E402
    atomic_torch_save,
    atomic_write_json,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=PROJECT_ROOT
        / "upstream/config-randsfq/"
        "rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-"
        "prefix-causal-stage1.py",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/gaoya/agent-data/cache/xssc_stage1_causal_state"),
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pad_objects(value: torch.Tensor, target: int = NUM_OBJECTS) -> torch.Tensor:
    if value.shape[1] > target:
        raise ValueError(f"MOVi-C object count {value.shape[1]} exceeds {target}")
    shape = list(value.shape)
    shape[1] = target - value.shape[1]
    if shape[1] == 0:
        return value
    return torch.cat([value, value.new_zeros(shape)], dim=1)


def downsample_masks(masks: torch.Tensor, size=(16, 16)) -> torch.Tensor:
    time_steps, objects, height, width = masks.shape
    resized = functional.interpolate(
        masks.reshape(time_steps * objects, 1, height, width).float(),
        size=size,
        mode="nearest",
    )
    return resized.reshape(time_steps, objects, *size).bool()


def load_checkpoint(model, checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("m.encode_backbone.")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state
    gc.collect()


def build_dataset(cfg, split: str, data_root: Path):
    from object_centric_bench.datum import MOViTFRecord
    from object_centric_bench.util import Config, build_from_config

    dataset_cfg = Config(dict(cfg.dataset_v))
    dataset_cfg.type = MOViTFRecord
    dataset_cfg.split = split
    dataset_cfg.base_dir = data_root
    dataset_cfg.extra_keys = [
        "segment",
        "bbox",
        "position",
        "velocity",
        "image_position",
        "visibility",
        "video_name",
    ]
    dataset_cfg.preserve_instance_ids = True
    dataset_cfg.require_complete = True
    return build_from_config(dataset_cfg)


def write_manifest(entries: list[dict], path: Path):
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
    os.replace(temporary, path)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.device == "cuda:4":
        raise ValueError("GPU 4 is prohibited by workspace policy")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(args.config_file.resolve())
    if cfg.model.encode_backbone.temporal_mode != "prefix_causal":
        raise RuntimeError("Stage-1 cache requires temporal_mode='prefix_causal'")
    if int(cfg.raw_clip_frames) != NUM_RAW_FRAMES:
        raise RuntimeError(f"Stage-1 cache requires {NUM_RAW_FRAMES} raw frames")

    dataset = build_dataset(cfg, args.split, args.data_root.resolve())
    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    load_checkpoint(model, args.checkpoint.resolve())
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = model.to(device).eval()
    dtype = getattr(torch, args.amp_dtype)

    split_root = args.cache_root.resolve() / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    stop = len(dataset)
    if args.max_cases is not None:
        stop = min(stop, args.start_index + args.max_cases)
    if not 0 <= args.start_index < stop:
        raise ValueError(f"Invalid extraction range [{args.start_index}, {stop})")

    for index in range(args.start_index, stop):
        record_name = f"case_{index:06d}.pt"
        output_file = split_root / record_name
        if output_file.is_file() and not args.overwrite:
            print(f"[cache] skip existing {output_file}", flush=True)
            continue
        sample = dataset[index]
        video = sample["video"]
        if tuple(video.shape[:2]) != (NUM_RAW_FRAMES, 3):
            raise RuntimeError(f"Unexpected video shape at {index}: {video.shape}")
        num_objects = int(sample["position"].shape[1])
        if not 1 <= num_objects <= NUM_OBJECTS:
            raise RuntimeError(f"Unexpected object count: {num_objects}")

        initial_bbox = torch.zeros(NUM_SLOTS, 4, dtype=torch.float32)
        initial_bbox[:num_objects] = sample["bbox"][0, :num_objects]
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            feature, slots, attention = model.m.extract_slot_trajectory(
                video[None].to(device, non_blocking=True),
                initial_condit=initial_bbox[None].to(device, non_blocking=True),
            )
        feature = feature[0].float().cpu()
        slots = slots[0].float().cpu()
        attention = attention[0].float().cpu()
        if tuple(slots.shape) != (NUM_STATES, NUM_SLOTS, SLOT_DIM):
            raise RuntimeError(f"Unexpected causal slot shape: {slots.shape}")

        # The configured V-JEPA transform selects the second frame only for
        # segmentation/bbox. Official float states retain all 24 raw frames.
        gt_mask = sample["segment"][..., 1 : num_objects + 1].permute(0, 3, 1, 2)
        if gt_mask.shape[0] != NUM_STATES:
            raise RuntimeError(f"Expected {NUM_STATES} tubelet GT masks")
        gt_mask = downsample_masks(pad_objects(gt_mask))
        label_indices = torch.tensor(LABEL_FRAME_INDICES, dtype=torch.long)
        gt_position = pad_objects(sample["position"].index_select(0, label_indices))
        gt_velocity = pad_objects(sample["velocity"].index_select(0, label_indices))
        gt_image_position = pad_objects(
            sample["image_position"].index_select(0, label_indices)
        )
        gt_visibility = pad_objects(
            sample["visibility"].index_select(0, label_indices)[..., None]
        )[..., 0]
        gt_bbox = pad_objects(sample["bbox"][:, :num_objects])
        object_valid = torch.zeros(NUM_OBJECTS, dtype=torch.bool)
        object_valid[:num_objects] = True

        prefix = calibrate_identity(
            attention,
            gt_mask,
            object_valid,
            calibration_states=CALIBRATION_STATES,
            mode="prefix_oracle",
        )
        boundary = calibrate_identity(
            attention,
            gt_mask,
            object_valid,
            calibration_states=CALIBRATION_STATES,
            mode="boundary_frozen",
        )
        record = {
            "slots": slots.half(),
            "slot_attention": attention.half(),
            "gt_mask": gt_mask,
            "gt_position": gt_position.float(),
            "gt_velocity": gt_velocity.float(),
            "gt_image_position": gt_image_position.float(),
            "gt_bbox": gt_bbox.float(),
            "gt_visibility": gt_visibility.long(),
            "object_valid": object_valid,
            "prefix_slot_to_object": prefix.slot_to_object,
            "boundary_slot_to_object": boundary.slot_to_object,
            "source": {
                "split": args.split,
                "index": index,
                "video_name": sample["video_name"],
                "label_frame_indices": list(LABEL_FRAME_INDICES),
                "num_objects": num_objects,
                "feature_shape": list(feature.shape),
                "prefix_coverage": prefix.coverage,
                "prefix_mean_iou": prefix.mean_matched_iou,
                "boundary_coverage": boundary.coverage,
                "boundary_mean_iou": boundary.mean_matched_iou,
            },
        }
        atomic_torch_save(record, output_file)
        print(
            f"[cache] split={args.split} index={index} "
            f"prefix_coverage={prefix.coverage:.3f} file={output_file}",
            flush=True,
        )

    entries = []
    for record_file in sorted(split_root.glob("case_*.pt")):
        record = torch.load(record_file, map_location="cpu", weights_only=True)
        entries.append(
            {
                "index": int(record["source"]["index"]),
                "record": record_file.name,
                "video_name": record["source"]["video_name"],
            }
        )
    entries.sort(key=lambda item: item["index"])
    write_manifest(entries, split_root / "records.jsonl")
    provenance = {
        "format": "xssc_stage1_causal_slots_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "config": str(args.config_file.resolve()),
        "split": args.split,
        "dataset_size": len(dataset),
        "cached_records": len(entries),
        "temporal_mode": cfg.model.encode_backbone.temporal_mode,
        "causal_method": "repeated_prefix_last_tubelet",
        "decoder_called": False,
        "raw_frames": NUM_RAW_FRAMES,
        "states": NUM_STATES,
        "label_frame_indices": list(LABEL_FRAME_INDICES),
        "slot_shape": [NUM_STATES, NUM_SLOTS, SLOT_DIM],
        "amp_dtype": args.amp_dtype,
    }
    atomic_write_json(provenance, split_root / "provenance.json")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
