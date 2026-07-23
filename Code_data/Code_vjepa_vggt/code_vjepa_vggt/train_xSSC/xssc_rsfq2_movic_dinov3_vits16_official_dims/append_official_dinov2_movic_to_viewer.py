#!/usr/bin/env python3
"""Append official DINOv2 MOVi-C xSSC overlays to an existing viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))

IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)
PALETTE = np.asarray(
    [
        [239, 68, 68],
        [59, 130, 246],
        [34, 197, 94],
        [250, 204, 21],
        [168, 85, 247],
        [6, 182, 212],
        [249, 115, 22],
        [236, 72, 153],
        [132, 204, 22],
        [20, 184, 166],
        [251, 146, 60],
    ],
    dtype=np.uint8,
)

DEFAULT_VIEWER_DIR = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_slot_overlay_test5_crop_padding_compare_plus24000"
)
DEFAULT_OUTPUTS_ROOT = Path("/data/gaoya/agent-data/outputs")
DEFAULT_CONFIG = ROOT / "upstream/config-randsfq/rsfq2_c-movi_c.py"
DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/agent-data/weights/xssc_official_archive_rsfq2/"
    "rsfq2_c-movi_c/42-0035.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--label", default="official_dinov2_movic_42_0035")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--quality", type=int, default=88)
    return parser.parse_args()


def normalize_rgb_frames(frames: np.ndarray) -> torch.Tensor:
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()[None]
    return (video - IMAGENET_MEAN) / IMAGENET_STD


def read_frame_sequence(frame_root: Path, frame_count: int) -> np.ndarray:
    frames = []
    for frame_id in range(frame_count):
        path = frame_root / f"{frame_id:04d}.webp"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(np.asarray(Image.open(path).convert("RGB")))
    return np.stack(frames, axis=0).astype(np.uint8)


def boxes_from_metadata(amg: dict, num_slots: int, frame_count: int, height: int, width: int) -> torch.Tensor:
    boxes = np.zeros((1, frame_count, num_slots, 4), dtype=np.float32)
    for slot_id, xywh in enumerate(amg.get("selected_boxes_xywh", [])[:num_slots]):
        x, y, box_w, box_h = [float(value) for value in xywh]
        boxes[0, :, slot_id] = np.asarray(
            [x / width, y / height, (x + box_w) / width, (y + box_h) / height],
            dtype=np.float32,
        )
    return torch.from_numpy(boxes)


def overlay(frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels_full = labels.repeat(16, axis=0).repeat(16, axis=1)
    colors = PALETTE[labels_full % len(PALETTE)]
    result = (
        frame.astype(np.float32) * 0.43 + colors.astype(np.float32) * 0.57
    ).round().clip(0, 255).astype(np.uint8)
    for pos in range(16, result.shape[0], 16):
        result[pos, :, :] = (result[pos, :, :].astype(np.float32) * 0.72).astype(np.uint8)
        result[:, pos, :] = (result[:, pos, :].astype(np.float32) * 0.72).astype(np.uint8)
    return result


def save_webp(path: Path, array: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="WEBP", quality=quality, method=4)


def load_checkpoint(model, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch {checkpoint}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def build_model(config_file: Path, checkpoint: Path, device: torch.device):
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(config_file)
    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    model.freez(cfg.freez, verbose=False)
    load_checkpoint(model, checkpoint)
    return cfg, model.to(device).eval()


def infer_labels(model, video: torch.Tensor, boxes: torch.Tensor, device: torch.device, amp_dtype: torch.dtype) -> np.ndarray:
    batch = {
        "video": video.to(device, non_blocking=True),
        "bbox": boxes.to(device, non_blocking=True),
    }
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
        out = model(batch=batch)
    attentd = out["attentd"][0].detach().float().cpu()
    return attentd.argmax(dim=1).to(torch.uint8).numpy()


def replace_model_row(models: list[dict], row: dict) -> None:
    models[:] = [model for model in models if model.get("label") != row["label"]]
    models.append(row)


def update_source_metadata(source_root: Path, label: str, config: Path, checkpoint: Path) -> None:
    metadata_path = source_root / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoints = [item for item in metadata.get("checkpoints", []) if item.get("label") != label]
    checkpoints.append({"label": label, "config": str(config), "checkpoint": str(checkpoint)})
    metadata["checkpoints"] = checkpoints
    for case in metadata.get("cases", []):
        row = {
            "label": label,
            "config": config.name,
            "slots": 11,
            "condition": "official DINOv2 MOVi-C AMG pseudo boxes",
            "frame_pattern": f"cases/{case['case_id']}/{label}/{{frame}}.webp",
        }
        replace_model_row(case["models"], row)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is not available")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = getattr(torch, args.amp_dtype)

    combined_path = viewer_dir / "combined_metadata.json"
    metadata = json.loads(combined_path.read_text(encoding="utf-8"))
    cfg, model = build_model(config, checkpoint, device)
    num_slots = int(cfg.max_num)
    label = args.label

    for case_index, case in enumerate(metadata["cases"], start=1):
        case_id = case["case_id"]
        frame_count = int(case["frames"])
        for mode, source_key in (("crop", "crop_dir"), ("padding", "padding_dir")):
            source_root = outputs_root / metadata[source_key]
            rgb = read_frame_sequence(source_root / "cases" / case_id / "original", frame_count)
            video = normalize_rgb_frames(rgb)
            height, width = rgb.shape[1:3]
            boxes = boxes_from_metadata(case[mode]["amg"], num_slots, frame_count, height, width)
            labels = infer_labels(model, video, boxes, device, amp_dtype)
            out_dir = source_root / "cases" / case_id / label
            for frame_id, frame in enumerate(rgb):
                save_webp(out_dir / f"{frame_id:04d}.webp", overlay(frame, labels[frame_id]), args.quality)
            row = {
                "label": label,
                "config": config.name,
                "slots": int(labels.shape[1]) if labels.ndim == 4 else num_slots,
                "condition": "official DINOv2 MOVi-C AMG pseudo boxes",
                "frame_pattern": f"cases/{case_id}/{label}/{{frame}}.webp",
            }
            replace_model_row(case[mode]["models"], row)
            print(f"[infer] {case_index}/{len(metadata['cases'])} {case_id} {mode} {label}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata["official_dinov2_movic"] = {
        "label": label,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "note": "Official xSSC DINOv2 MOVi-C checkpoint appended using the same frame-0 AMG pseudo-box condition stored in this viewer metadata.",
    }
    combined_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for source_key in ("crop_dir", "padding_dir"):
        update_source_metadata(outputs_root / metadata[source_key], label, config, checkpoint)
    print(
        json.dumps(
            {
                "combined_metadata": str(combined_path),
                "label": label,
                "cases": len(metadata["cases"]),
                "frames": sum(int(case["frames"]) for case in metadata["cases"]),
                "checkpoint": str(checkpoint),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
