#!/usr/bin/env python3
"""Compare official DINOv2 and trained DINOv3 xSSC slots on source videos."""

from __future__ import annotations

import argparse
import av
import gc
import hashlib
import html
import imageio_ffmpeg
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DINOV3_EXPERIMENT = ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
sys.path.insert(0, str(DINOV3_EXPERIMENT / "third_party/dinov3"))
sys.path.insert(0, str(DINOV3_EXPERIMENT / "upstream"))

MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
PALETTE = np.asarray(
    [
        [239, 68, 68],
        [59, 130, 246],
        [34, 197, 94],
        [250, 204, 21],
        [168, 85, 247],
        [6, 182, 212],
        [249, 115, 22],
    ],
    dtype=np.uint8,
)

DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_test5_source_slot_compare_5weights_ctx8_full"
)
DEFAULT_CACHE = Path(
    "/data/gaoya/agent-data/cache/"
    "xssc_test5_source_slot_compare_5weights_ctx8_full"
)
OFFICIAL_CONFIG = Path("/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/rsfq2_r-ytvis.py")
DINOV3_CONFIG = (
    DINOV3_EXPERIMENT
    / "upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py"
)
DINOV3_CHECKPOINT_DIR = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/"
    "dinov3_xSSC/restart_save1000_20260720T140029Z/"
    "rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512/42"
)

MODEL_SPECS = [
    {
        "id": "official-42",
        "label": "Official 42-0130",
        "architecture": "dinov2",
        "config": OFFICIAL_CONFIG,
        "checkpoint": Path("/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth"),
    },
    {
        "id": "official-43",
        "label": "Official 43-0091",
        "architecture": "dinov2",
        "config": OFFICIAL_CONFIG,
        "checkpoint": Path("/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/43-0091.pth"),
    },
    {
        "id": "official-44",
        "label": "Official 44-0101",
        "architecture": "dinov2",
        "config": OFFICIAL_CONFIG,
        "checkpoint": Path("/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/44-0101.pth"),
    },
    {
        "id": "step-4000",
        "label": "DINOv3 step-4000",
        "architecture": "dinov3",
        "config": DINOV3_CONFIG,
        "checkpoint": DINOV3_CHECKPOINT_DIR / "step-004000.pth",
    },
    {
        "id": "step-15000",
        "label": "DINOv3 step-15000",
        "architecture": "dinov3",
        "config": DINOV3_CONFIG,
        "checkpoint": DINOV3_CHECKPOINT_DIR / "step-015000.pth",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-list",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctx-frames", type=int, default=8)
    parser.add_argument("--feature-batch", type=int, default=24)
    parser.add_argument("--decoder-batch", type=int, default=16)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def safe_stem(path: Path):
    text = path.stem
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in text)


def resolve_source_video(payload, json_path):
    for key in ("source_video", "input_video"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (json_path.parent / path).resolve()
            return path.resolve()
    raise ValueError(f"{json_path} has no source_video or input_video")


def read_cases(json_list, max_cases=0):
    raw_lines = [line.strip() for line in json_list.read_text().splitlines() if line.strip()]
    unique_lines = list(dict.fromkeys(raw_lines))
    if max_cases > 0:
        unique_lines = unique_lines[:max_cases]
    cases = []
    used_ids = set()
    for position, line in enumerate(unique_lines):
        json_path = Path(line).expanduser().resolve()
        payload = json.loads(json_path.read_text())
        source_video = resolve_source_video(payload, json_path)
        case_id = safe_stem(json_path)
        if case_id in used_ids:
            case_id = f"{case_id}_{position:02d}"
        used_ids.add(case_id)
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        cases.append(
            {
                "position": position,
                "id": case_id,
                "json": str(json_path),
                "source_video": str(source_video),
            }
        )
    duplicate_lines = [
        {"line": index + 1, "json": line, "first_line": raw_lines.index(line) + 1}
        for index, line in enumerate(raw_lines)
        if line in raw_lines[:index]
    ]
    return raw_lines, duplicate_lines, cases


def geometry_for_source(source_h, source_w, target_h=512, target_w=896):
    scale = max(target_h / float(source_h), target_w / float(source_w))
    resized_h = max(target_h, int(round(source_h * scale)))
    resized_w = max(target_w, int(round(source_w * scale)))
    cover_top = max(0, (resized_h - target_h) // 2)
    cover_left = max(0, (resized_w - target_w) // 2)
    square = min(target_h, target_w)
    square_top = (target_h - square) // 2
    square_left = (target_w - square) // 2
    return {
        "source_hw": [source_h, source_w],
        "cover_target_hw": [target_h, target_w],
        "cover_scale": scale,
        "resized_hw": [resized_h, resized_w],
        "cover_crop_yxhw": [cover_top, cover_left, target_h, target_w],
        "square_crop_yxhw_in_cover": [square_top, square_left, square, square],
        "model_input_hw": [256, 256],
    }


def preprocess_frame_chunk(frames, geometry):
    tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    resized_h, resized_w = geometry["resized_hw"]
    tensor = F.interpolate(
        tensor,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    top, left, height, width = geometry["cover_crop_yxhw"]
    tensor = tensor[:, :, top : top + height, left : left + width]
    top, left, height, width = geometry["square_crop_yxhw_in_cover"]
    tensor = tensor[:, :, top : top + height, left : left + width]
    pixels = F.interpolate(
        tensor,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    normalized = (pixels - MEAN) / STD
    rgb = pixels.round().clamp(0, 255).to(torch.uint8)
    rgb = rgb.permute(0, 2, 3, 1).contiguous().numpy()
    return normalized.to(torch.bfloat16), rgb


def prepare_case_input(case, cache_dir, decode_batch=32):
    case_dir = cache_dir / "inputs" / case["id"]
    tensor_path = case_dir / "normalized.pt"
    rgb_path = case_dir / "rgb.npy"
    metadata_path = case_dir / "metadata.json"
    if tensor_path.is_file() and rgb_path.is_file() and metadata_path.is_file():
        return json.loads(metadata_path.read_text())

    case_dir.mkdir(parents=True, exist_ok=True)
    normalized_parts = []
    rgb_parts = []
    with av.open(case["source_video"]) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 8.0
        frame_buffer = []
        geometry = None
        for frame in container.decode(stream):
            array = frame.to_ndarray(format="rgb24")
            if geometry is None:
                geometry = geometry_for_source(array.shape[0], array.shape[1])
            frame_buffer.append(array)
            if len(frame_buffer) == decode_batch:
                normalized, rgb = preprocess_frame_chunk(frame_buffer, geometry)
                normalized_parts.append(normalized)
                rgb_parts.append(rgb)
                frame_buffer = []
        if frame_buffer:
            normalized, rgb = preprocess_frame_chunk(frame_buffer, geometry)
            normalized_parts.append(normalized)
            rgb_parts.append(rgb)
    if not normalized_parts or geometry is None:
        raise RuntimeError(f"no video frames decoded: {case['source_video']}")
    normalized = torch.cat(normalized_parts)
    rgb = np.concatenate(rgb_parts)
    torch.save(normalized, tensor_path)
    np.save(rgb_path, rgb)
    metadata = {
        "frames": int(rgb.shape[0]),
        "fps": fps,
        "geometry": geometry,
        "normalized_shape": list(normalized.shape),
        "rgb_shape": list(rgb.shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def normalize_state_dict(state):
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint object: {type(state)!r}")
    if state and all(str(key).startswith("m.") for key in state):
        state = {str(key)[2:]: value for key, value in state.items()}
    return state


def backbone_signature(state):
    digest = hashlib.sha256()
    count = 0
    for key in sorted(state):
        if not key.startswith("encode_backbone."):
            continue
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(memoryview(tensor.view(torch.uint8).numpy()))
        count += 1
    if count == 0:
        raise RuntimeError("checkpoint contains no encode_backbone parameters")
    return {"sha256": digest.hexdigest(), "tensor_count": count}


def load_state(model, checkpoint):
    state = normalize_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    signature = backbone_signature(state)
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    return signature


def build_model(spec, device):
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(spec["config"])
    if spec["architecture"] == "dinov2":
        import timm

        original_create_model = timm.create_model

        def create_model_offline(*args, **kwargs):
            kwargs["pretrained"] = False
            return original_create_model(*args, **kwargs)

        timm.create_model = create_model_offline
        try:
            model = build_from_config(cfg.model)
        finally:
            timm.create_model = original_create_model
    else:
        model = build_from_config(cfg.model)
    signature = load_state(model, spec["checkpoint"])
    model.requires_grad_(False)
    model.to(device).eval()
    return model, cfg, signature


def load_case_normalized(cache_dir, case_id):
    return torch.load(
        cache_dir / "inputs" / case_id / "normalized.pt",
        map_location="cpu",
        weights_only=True,
    )


@torch.inference_mode()
def extract_backbone_feature(model, normalized, device, batch_size):
    parts = []
    for start in range(0, len(normalized), batch_size):
        video = normalized[start : start + batch_size].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feature = model.encode_backbone(video).detach()
        parts.append(feature.to(device="cpu", dtype=torch.bfloat16))
        del video, feature
    return torch.cat(parts)


def ensure_feature_cache(model, architecture, signature, cases, cache_dir, device, batch_size):
    architecture_dir = cache_dir / "features" / architecture
    architecture_dir.mkdir(parents=True, exist_ok=True)
    for position, case in enumerate(cases, start=1):
        path = architecture_dir / f"{case['id']}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("backbone_sha256") != signature["sha256"]:
                raise RuntimeError(f"stale feature cache with different backbone: {path}")
            print(f"[feature:{architecture}] {position}/{len(cases)} cached {case['id']}", flush=True)
            continue
        normalized = load_case_normalized(cache_dir, case["id"])
        feature = extract_backbone_feature(model, normalized, device, batch_size)
        torch.save(
            {
                "feature": feature,
                "backbone_sha256": signature["sha256"],
                "shape": list(feature.shape),
            },
            path,
        )
        print(
            f"[feature:{architecture}] {position}/{len(cases)} {case['id']} {tuple(feature.shape)}",
            flush=True,
        )
        del normalized, feature
        torch.cuda.empty_cache()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def attention_from_feature(model, feature, device, seed, decoder_batch):
    encoded_parts = []
    for start in range(0, len(feature), decoder_batch):
        current = feature[start : start + decoder_batch].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = current.permute(0, 2, 3, 1)
            encoded = model.encode_posit_embed(encoded).flatten(1, 2)
            encoded = model.encode_project(encoded)
        encoded_parts.append(encoded.to(device="cpu", dtype=torch.bfloat16))
    encoded = torch.cat(encoded_parts)

    set_seed(seed)
    slot_parts = []
    slot_window = []
    encode_window = []
    transition_dt = int(model.transit.dt)
    for frame_id in range(len(encoded)):
        encoded_i = encoded[frame_id : frame_id + 1].to(device, non_blocking=True)
        encode_window.append(encoded_i)
        encode_window = encode_window[-transition_dt:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if frame_id == 0:
                query = model.initializ(1)
            else:
                query = model.transit(
                    torch.stack(slot_window, dim=1),
                    torch.stack(encode_window, dim=1),
                )
            slots_i, _ = model.aggregat(
                encoded_i,
                query,
                num_iter=None if frame_id == 0 else 1,
            )
        slot_parts.append(slots_i.to(device="cpu", dtype=torch.bfloat16))
        slot_window.append(slots_i)
        slot_window = slot_window[-(transition_dt - 1) :]
    slots = torch.stack(slot_parts, dim=1)

    attention_parts = []
    feature_h, feature_w = feature.shape[-2:]
    for start in range(0, len(feature), decoder_batch):
        feature_i = feature[start : start + decoder_batch].to(device, non_blocking=True)
        clue = feature_i.permute(0, 2, 3, 1).flatten(1, 2)[None]
        slots_i = slots[:, start : start + decoder_batch].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, attention_i, _ = model.decode(clue, slots_i)
        attention_parts.append(attention_i[0].to(device="cpu", dtype=torch.float32))
    attention = torch.cat(attention_parts)
    attention = attention.reshape(len(feature), attention.shape[1], feature_h, feature_w)
    return attention, slots


@torch.inference_mode()
def direct_attention(model, normalized, device, seed):
    set_seed(seed)
    video = normalized[None].to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(video)
    return output[-1][0].detach().float().cpu()


def smoke_validate(model, feature, normalized, device, seed, ctx_frames, decoder_batch):
    count = min(ctx_frames, len(feature))
    custom_attention, _ = attention_from_feature(
        model, feature[:count], device, seed, decoder_batch
    )
    direct = direct_attention(model, normalized[:count], device, seed)
    custom_labels = custom_attention.argmax(1)
    direct_labels = direct.argmax(1)
    label_agreement = float((custom_labels == direct_labels).float().mean())
    max_abs = float((custom_attention - direct).abs().max())

    full_attention, _ = attention_from_feature(
        model, feature, device, seed, decoder_batch
    )
    prefix_labels = full_attention[:count].argmax(1)
    prefix_agreement = float((prefix_labels == custom_labels).float().mean())
    if label_agreement != 1.0 or prefix_agreement != 1.0:
        raise RuntimeError(
            "smoke mismatch: "
            f"direct={label_agreement}, full_prefix={prefix_agreement}, max_abs={max_abs}"
        )
    return {
        "frames": count,
        "custom_vs_direct_label_agreement": label_agreement,
        "custom_vs_direct_attention_max_abs": max_abs,
        "full_prefix_vs_ctx_label_agreement": prefix_agreement,
    }


def infer_weight(model, spec, cases, cache_dir, output_dir, device, seed, decoder_batch):
    label_dir = output_dir / "labels" / spec["id"]
    label_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for position, case in enumerate(cases, start=1):
        output_path = label_dir / f"{case['id']}.npz"
        if output_path.is_file():
            with np.load(output_path) as payload:
                labels_shape = list(payload["labels"].shape)
            records.append({"case_id": case["id"], "labels_shape": labels_shape, "cached": True})
            print(f"[infer:{spec['id']}] {position}/{len(cases)} cached {case['id']}", flush=True)
            continue
        feature_path = cache_dir / "features" / spec["architecture"] / f"{case['id']}.pt"
        feature = torch.load(feature_path, map_location="cpu", weights_only=True)["feature"]
        attention, slots = attention_from_feature(
            model,
            feature,
            device,
            seed + case["position"],
            decoder_batch,
        )
        labels = attention.argmax(1).to(torch.uint8).numpy()
        occupancy = np.bincount(labels.reshape(-1), minlength=7) / labels.size
        np.savez_compressed(
            output_path,
            labels=labels,
            occupancy=occupancy,
            attention_shape=np.asarray([1, *attention.shape], dtype=np.int64),
            slots_shape=np.asarray(slots.shape, dtype=np.int64),
        )
        records.append({"case_id": case["id"], "labels_shape": list(labels.shape), "cached": False})
        print(
            f"[infer:{spec['id']}] {position}/{len(cases)} {case['id']} {tuple(labels.shape)}",
            flush=True,
        )
        del feature, attention, slots
        torch.cuda.empty_cache()
    return records


def pairwise_iou(anchor, target, num_slots=7):
    matrix = np.zeros((num_slots, num_slots), dtype=np.float64)
    for anchor_id in range(num_slots):
        anchor_mask = anchor == anchor_id
        for target_id in range(num_slots):
            target_mask = target == target_id
            union = np.logical_or(anchor_mask, target_mask).sum()
            if union:
                matrix[anchor_id, target_id] = (
                    np.logical_and(anchor_mask, target_mask).sum() / union
                )
    return matrix


def align_labels(anchor, target):
    matrix = pairwise_iou(anchor, target)
    anchor_ids, target_ids = linear_sum_assignment(-matrix)
    raw_to_aligned = {
        int(target_id): int(anchor_id)
        for anchor_id, target_id in zip(anchor_ids, target_ids)
    }
    aligned_to_raw = {aligned_id: raw_id for raw_id, aligned_id in raw_to_aligned.items()}
    aligned = np.empty_like(target)
    for raw_id, aligned_id in raw_to_aligned.items():
        aligned[target == raw_id] = aligned_id
    matched = {
        int(anchor_id): float(matrix[anchor_id, target_id])
        for anchor_id, target_id in zip(anchor_ids, target_ids)
    }
    return aligned, matrix, raw_to_aligned, aligned_to_raw, matched


def upscale(labels):
    return labels.repeat(16, axis=1).repeat(16, axis=2)


def add_grid(frames, strength=0.28):
    frames = frames.copy()
    for position in range(16, 256, 16):
        frames[:, position, :, :] = (
            frames[:, position, :, :].astype(np.float32) * (1 - strength)
        ).astype(np.uint8)
        frames[:, :, position, :] = (
            frames[:, :, position, :].astype(np.float32) * (1 - strength)
        ).astype(np.uint8)
    return frames


def combined_overlay(rgb, labels):
    colors = PALETTE[upscale(labels)]
    output = rgb.astype(np.float32) * 0.43 + colors.astype(np.float32) * 0.57
    return add_grid(output.round().clip(0, 255).astype(np.uint8))


def slot_overlay(rgb, labels, slot_id):
    full_labels = upscale(labels)
    selected = full_labels == slot_id
    dimmed = (rgb.astype(np.float32) * 0.24).round().astype(np.uint8)
    colored = (
        rgb.astype(np.float32) * 0.36 + PALETTE[slot_id].astype(np.float32) * 0.64
    ).round().clip(0, 255).astype(np.uint8)
    return add_grid(np.where(selected[..., None], colored, dimmed))


def write_h264(path, frames, fps, ffmpeg):
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "256x256",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.run(
        command,
        input=np.ascontiguousarray(frames).tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"ffmpeg failed for {path}: {process.stderr.decode(errors='replace')}")


def rel(path, root):
    return path.relative_to(root).as_posix()


def render_cases(cases, specs, cache_dir, output_dir, ctx_frames):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    rendered_cases = []
    for position, case in enumerate(cases, start=1):
        input_meta = json.loads(
            (cache_dir / "inputs" / case["id"] / "metadata.json").read_text()
        )
        rgb = np.load(cache_dir / "inputs" / case["id"] / "rgb.npy")
        raw_labels = {}
        occupancies = {}
        for spec in specs:
            with np.load(output_dir / "labels" / spec["id"] / f"{case['id']}.npz") as payload:
                raw_labels[spec["id"]] = payload["labels"]
                occupancies[spec["id"]] = payload["occupancy"].tolist()
        anchor = raw_labels["step-4000"]
        aligned = {}
        alignments = {}
        for spec in specs:
            model_id = spec["id"]
            if model_id == "step-4000":
                aligned[model_id] = anchor
                alignments[model_id] = {
                    "raw_to_aligned": {str(index): index for index in range(7)},
                    "aligned_to_raw": {str(index): index for index in range(7)},
                    "matched_iou": {str(index): 1.0 for index in range(7)},
                    "mean_matched_iou": 1.0,
                    "pairwise_iou": np.eye(7).tolist(),
                }
            else:
                labels, matrix, raw_to_aligned, aligned_to_raw, matched = align_labels(
                    anchor, raw_labels[model_id]
                )
                aligned[model_id] = labels
                alignments[model_id] = {
                    "raw_to_aligned": {str(key): value for key, value in raw_to_aligned.items()},
                    "aligned_to_raw": {str(key): value for key, value in aligned_to_raw.items()},
                    "matched_iou": {str(key): value for key, value in matched.items()},
                    "mean_matched_iou": float(np.mean(list(matched.values()))),
                    "pairwise_iou": matrix.tolist(),
                }

        case_dir = output_dir / "cases" / case["id"]
        original_path = case_dir / "model_input.mp4"
        write_h264(original_path, rgb, input_meta["fps"], ffmpeg)
        model_rows = []
        for spec in specs:
            model_id = spec["id"]
            model_dir = case_dir / model_id
            combined_path = model_dir / "all_slots.mp4"
            write_h264(combined_path, combined_overlay(rgb, aligned[model_id]), input_meta["fps"], ffmpeg)
            aligned_occupancy = np.bincount(aligned[model_id].reshape(-1), minlength=7) / aligned[model_id].size
            slots = []
            for slot_id in range(7):
                slot_path = model_dir / f"slot_{slot_id}.mp4"
                write_h264(
                    slot_path,
                    slot_overlay(rgb, aligned[model_id], slot_id),
                    input_meta["fps"],
                    ffmpeg,
                )
                alignment = alignments[model_id]
                slots.append(
                    {
                        "aligned_slot": slot_id,
                        "raw_slot": int(alignment["aligned_to_raw"][str(slot_id)]),
                        "occupancy": float(aligned_occupancy[slot_id]),
                        "match_iou": float(alignment["matched_iou"][str(slot_id)]),
                        "video": rel(slot_path, output_dir),
                    }
                )
            model_rows.append(
                {
                    "id": model_id,
                    "label": spec["label"],
                    "architecture": spec["architecture"],
                    "combined_video": rel(combined_path, output_dir),
                    "slots": slots,
                    "alignment": alignments[model_id],
                    "raw_occupancy": occupancies[model_id],
                }
            )
        rendered_cases.append(
            {
                **case,
                "frames": input_meta["frames"],
                "fps": input_meta["fps"],
                "ctx_frames": min(ctx_frames, input_meta["frames"]),
                "geometry": input_meta["geometry"],
                "original_video": rel(original_path, output_dir),
                "models": model_rows,
            }
        )
        print(f"[render] {position}/{len(cases)} {case['id']}", flush=True)
    return rendered_cases


def build_html(metadata):
    data = json.dumps(metadata, separators=(",", ":"))
    palette = json.dumps(PALETTE.tolist())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>xSSC source slot comparison</title>
<style>
*{{box-sizing:border-box}}:root{{color-scheme:dark}}body{{margin:0;background:#101214;color:#f3f4f6;font:14px system-ui,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:4;background:rgba(16,18,20,.98);border-bottom:1px solid #363b41}}.bar{{max-width:1900px;margin:auto;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}h1{{font-size:18px;margin:0 auto 0 0}}button,select,input{{font:inherit}}select,.icon{{height:34px;border:1px solid #4b525a;border-radius:5px;background:#202429;color:#f5f6f7}}select{{padding:0 28px 0 9px}}.icon{{width:34px;cursor:pointer}}.icon:hover{{background:#2b3036}}#seek{{min-width:170px;flex:0 1 260px;accent-color:#38bdf8}}.mode{{display:flex;border:1px solid #4b525a;border-radius:5px;overflow:hidden}}.mode button{{height:32px;border:0;border-right:1px solid #4b525a;background:#202429;color:#c4c9cf;padding:0 11px;cursor:pointer}}.mode button:last-child{{border-right:0}}.mode button.active{{background:#0369a1;color:#fff}}main{{max-width:1900px;margin:auto;padding:16px}}.meta{{display:flex;gap:16px;color:#aeb5bd;white-space:nowrap;overflow:auto;padding-bottom:12px}}.original{{width:min(360px,100%);margin-bottom:18px}}h2{{font-size:14px;margin:0 0 7px}}video{{display:block;width:100%;aspect-ratio:1;background:#050607;border:1px solid #34383d;border-radius:4px}}.table-wrap{{overflow:auto}}.head,.row{{display:grid;grid-template-columns:105px repeat(5,minmax(230px,1fr));gap:12px;min-width:1320px}}.head{{position:sticky;top:55px;z-index:3;background:rgba(16,18,20,.97);padding:9px 0;border-top:1px solid #34383d;border-bottom:1px solid #34383d;font-weight:680}}.row{{padding:13px 0;border-bottom:1px solid #2b2f34}}.label{{padding-top:2px}}.swatch{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}}figure{{margin:0;min-width:0}}figcaption{{padding-top:5px;color:#aeb5bd;font-size:12px;min-height:22px}}.metric{{color:#7dd3fc}}@media(max-width:760px){{h1{{width:100%}}main{{padding:11px}}.head{{top:101px}}}}
</style>
</head>
<body>
<header><div class="bar"><h1>xSSC source slot comparison</h1><select id="caseSelect" aria-label="Case"></select><div class="mode"><button id="ctxMode" class="active">ctx8</button><button id="fullMode">full</button></div><button id="restart" class="icon" title="Restart" aria-label="Restart">&#8634;</button><button id="play" class="icon" title="Play all" aria-label="Play all">&#9654;</button><input id="seek" type="range" min="0" max="1" step="0.001" value="0" aria-label="Timeline"><select id="speed" aria-label="Playback speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select><label><input id="loop" type="checkbox" checked> Loop</label></div></header>
<main id="app"></main>
<script>
const DATA={data};const PALETTE={palette};const app=document.getElementById('app');const caseSelect=document.getElementById('caseSelect');const playButton=document.getElementById('play');const restartButton=document.getElementById('restart');const seek=document.getElementById('seek');const speed=document.getElementById('speed');const loop=document.getElementById('loop');const ctxMode=document.getElementById('ctxMode');const fullMode=document.getElementById('fullMode');let videos=[],master=null,mode='ctx8',raf=null;
function video(src){{return `<video muted playsinline preload="metadata" src="${{src}}"></video>`}}
function limit(item){{return (mode==='ctx8'?item.ctx_frames:item.frames)/item.fps}}
function render(index){{if(raf)cancelAnimationFrame(raf);const item=DATA.cases[index];const rows=[];rows.push(`<section class="row"><div class="label"><h2>All slots</h2><span>16 x 16</span></div>${{item.models.map(m=>`<figure>${{video(m.combined_video)}}<figcaption>${{m.architecture}} | mean IoU <span class="metric">${{m.alignment.mean_matched_iou.toFixed(3)}}</span></figcaption></figure>`).join('')}}</section>`);for(let sid=0;sid<7;sid++){{const color=`rgb(${{PALETTE[sid].join(',')}})`;rows.push(`<section class="row"><div class="label"><h2><span class="swatch" style="background:${{color}}"></span>Slot ${{sid}}</h2></div>${{item.models.map(m=>{{const s=m.slots[sid];return `<figure>${{video(s.video)}}<figcaption>raw ${{s.raw_slot}} | occ <span class="metric">${{(s.occupancy*100).toFixed(1)}}%</span> | IoU <span class="metric">${{s.match_iou.toFixed(3)}}</span></figcaption></figure>`}}).join('')}}</section>`);}}app.innerHTML=`<div class="meta"><span>${{item.id}}</span><span>${{item.frames}} frames</span><span>${{item.fps.toFixed(2)}} fps</span><span>${{item.source_video}}</span></div><section class="original"><h2>Effective xSSC input</h2>${{video(item.original_video)}}</section><div class="table-wrap"><div class="head"><span>View</span>${{item.models.map(m=>`<span>${{m.label}}</span>`).join('')}}</div>${{rows.join('')}}</div>`;videos=Array.from(app.querySelectorAll('video'));master=videos[0];videos.forEach(v=>{{v.playbackRate=Number(speed.value);v.loop=false}});seek.max=String(limit(item));seek.value='0';setIcon(false);tick();}}
function item(){{return DATA.cases[Number(caseSelect.value)]}}function setIcon(on){{playButton.innerHTML=on?'&#10074;&#10074;':'&#9654;';playButton.title=on?'Pause all':'Play all'}}async function playAll(){{const end=limit(item());if(master.currentTime>=end-.02)videos.forEach(v=>v.currentTime=0);videos.forEach(v=>{{if(Math.abs(v.currentTime-master.currentTime)>.04)v.currentTime=master.currentTime}});await Promise.all(videos.map(v=>v.play().catch(()=>null)));setIcon(true)}}function pauseAll(){{videos.forEach(v=>v.pause());setIcon(false)}}function tick(){{if(master){{const end=limit(item());if(!master.paused&&master.currentTime>=end-.015){{if(loop.checked){{videos.forEach(v=>v.currentTime=0)}}else pauseAll()}}seek.value=String(Math.min(master.currentTime,end));if(!master.paused)videos.slice(1).forEach(v=>{{if(Math.abs(v.currentTime-master.currentTime)>.08)v.currentTime=master.currentTime}})}}raf=requestAnimationFrame(tick)}}function setMode(next){{mode=next;ctxMode.classList.toggle('active',mode==='ctx8');fullMode.classList.toggle('active',mode==='full');const end=limit(item());seek.max=String(end);if(master.currentTime>end)videos.forEach(v=>v.currentTime=0)}}
DATA.cases.forEach((c,i)=>{{const o=document.createElement('option');o.value=String(i);o.textContent=`${{String(i+1).padStart(2,'0')}} | ${{c.id}}`;caseSelect.appendChild(o)}});caseSelect.addEventListener('change',()=>render(Number(caseSelect.value)));ctxMode.addEventListener('click',()=>setMode('ctx8'));fullMode.addEventListener('click',()=>setMode('full'));playButton.addEventListener('click',()=>master.paused?playAll():pauseAll());restartButton.addEventListener('click',()=>videos.forEach(v=>v.currentTime=0));seek.addEventListener('input',()=>videos.forEach(v=>v.currentTime=Number(seek.value)));speed.addEventListener('change',()=>videos.forEach(v=>v.playbackRate=Number(speed.value)));render(0);
</script>
</body>
</html>"""


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for spec in MODEL_SPECS:
        if not spec["config"].is_file() or not spec["checkpoint"].is_file():
            raise FileNotFoundError(f"missing model input: {spec}")

    raw_lines, duplicates, cases = read_cases(args.json_list.resolve(), args.max_cases)
    print(
        f"[data] list_lines={len(raw_lines)} unique_cases={len(cases)} duplicates={len(duplicates)}",
        flush=True,
    )
    for position, case in enumerate(cases, start=1):
        metadata = prepare_case_input(case, cache_dir)
        case.update({"frames": metadata["frames"], "fps": metadata["fps"]})
        print(f"[input] {position}/{len(cases)} {case['id']} frames={metadata['frames']}", flush=True)

    smoke = {}
    signatures = {}
    inference_records = {}
    for architecture in ("dinov2", "dinov3"):
        architecture_specs = [spec for spec in MODEL_SPECS if spec["architecture"] == architecture]
        model, _, signature = build_model(architecture_specs[0], device)
        signatures[architecture] = signature
        ensure_feature_cache(
            model,
            architecture,
            signature,
            cases,
            cache_dir,
            device,
            args.feature_batch,
        )
        if not args.skip_smoke:
            first_case = cases[0]
            normalized = load_case_normalized(cache_dir, first_case["id"])
            feature = torch.load(
                cache_dir / "features" / architecture / f"{first_case['id']}.pt",
                map_location="cpu",
                weights_only=True,
            )["feature"]
            smoke[architecture] = smoke_validate(
                model,
                feature,
                normalized,
                device,
                args.seed + first_case["position"],
                args.ctx_frames,
                args.decoder_batch,
            )
            print(f"[smoke:{architecture}] {smoke[architecture]}", flush=True)
            del normalized, feature
        for spec_index, spec in enumerate(architecture_specs):
            if spec_index > 0:
                current_signature = load_state(model, spec["checkpoint"])
                if current_signature != signature:
                    raise RuntimeError(
                        f"backbone differs within {architecture}: {signature} != {current_signature}"
                    )
            inference_records[spec["id"]] = infer_weight(
                model,
                spec,
                cases,
                cache_dir,
                output_dir,
                device,
                args.seed,
                args.decoder_batch,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    rendered_cases = [] if args.skip_render else render_cases(
        cases, MODEL_SPECS, cache_dir, output_dir, args.ctx_frames
    )
    metadata = {
        "title": "xSSC source slot comparison",
        "json_list": str(args.json_list.resolve()),
        "json_list_lines": len(raw_lines),
        "unique_cases": len(cases),
        "duplicate_entries": duplicates,
        "seed": args.seed,
        "ctx_frames": args.ctx_frames,
        "anchor_model": "step-4000",
        "attention": "decoder final cross-attention (attentd)",
        "assignment": "argmax over 7 slots on native 16x16 patch grid",
        "alignment": "full-sequence Hungarian IoU to step-4000; reused by ctx8 view",
        "ctx8_derivation": "causal prefix of full inference, verified by smoke",
        "preprocessing": "512x896 cover resize/crop -> center 512x512 -> 256x256 -> ImageNet normalize",
        "backbone_signatures": signatures,
        "smoke": smoke,
        "models": [
            {
                "id": spec["id"],
                "label": spec["label"],
                "architecture": spec["architecture"],
                "config": str(spec["config"]),
                "checkpoint": str(spec["checkpoint"]),
            }
            for spec in MODEL_SPECS
        ],
        "cases": rendered_cases,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.skip_render:
        (output_dir / "index.html").write_text(build_html(metadata))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "cases": len(cases),
                "models": len(MODEL_SPECS),
                "videos": 0 if args.skip_render else len(cases) * (1 + len(MODEL_SPECS) * 8),
                "smoke": smoke,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
