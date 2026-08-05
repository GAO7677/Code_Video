#!/usr/bin/env python3
"""Compute V-JEPA2 feature MSE for tiny-VAE-decoded visualization videos.

Default comparison:
  step_*/gt_x0.mp4 vs step_*/pred_x0.mp4

The script discovers the latest xSSC visualization run unless --viz-dir is set.
Large outputs are written under /data/gaoya/agent-data/outputs by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parent
VJEPA2_DIR = PROJECT_DIR / "vjepa2"
DEFAULT_VIZ_BASE = Path(
    "/data/gaoya/agent-data/checkpoints/xssc_viz/full_sa_no_object_gpu27_formal"
)
DEFAULT_OUTPUT_BASE = Path("/data/gaoya/agent-data/outputs/vjepa2_tinyvae_mse")
DEFAULT_CHECKPOINT = Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


MODEL_FACTORIES = {
    "vjepa2-vitl-256": ("vjepa2_vit_large", 256),
    "vjepa2-vith-256": ("vjepa2_vit_huge", 256),
    "vjepa2-vitg-256": ("vjepa2_vit_giant", 256),
    "vjepa2-vitg-384": ("vjepa2_vit_giant_384", 384),
    "vjepa2.1-vitb-384": ("vjepa2_1_vit_base_384", 384),
    "vjepa2.1-vitl-384": ("vjepa2_1_vit_large_384", 384),
    "vjepa2.1-vitg-384": ("vjepa2_1_vit_giant_384", 384),
    "vjepa2.1-vitG-384": ("vjepa2_1_vit_gigantic_384", 384),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA2 encoder features from tiny-VAE decoded mp4s and compute MSE."
    )
    parser.add_argument("--vjepa2-dir", type=Path, default=VJEPA2_DIR)
    parser.add_argument("--viz-base", type=Path, default=DEFAULT_VIZ_BASE)
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="Specific visualization run directory. Defaults to newest records.json under --viz-base.",
    )
    parser.add_argument(
        "--reference-name",
        default="gt_x0.mp4",
        help="Reference video name inside each step directory.",
    )
    parser.add_argument(
        "--candidate-name",
        default="pred_x0.mp4",
        help="Candidate video name inside each step directory.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FACTORIES),
        default="vjepa2.1-vitl-384",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Limit number of pairs; 0 means all discovered pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to timestamped directory under /data/gaoya/agent-data/outputs.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="Model/input dtype on CUDA. CPU always uses fp32. V-JEPA2.1 attention is safest in fp32.",
    )
    return parser.parse_args()


def resolve_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type == "cuda" and device.index == 4:
        raise ValueError("GPU 4 is prohibited by workspace rules; choose another device.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return device


def dtype_for(device: torch.device, dtype_text: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_text]


def find_latest_viz_dir(viz_base: Path) -> Path:
    candidates = sorted(
        viz_base.glob("*/records.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No records.json found under {viz_base}")
    return candidates[0].parent


def discover_pairs(
    viz_dir: Path,
    reference_name: str,
    candidate_name: str,
    max_pairs: int,
) -> list[dict[str, Any]]:
    records_path = viz_dir / "records.json"
    records: list[dict[str, Any]] = []
    if records_path.is_file():
        payload = json.loads(records_path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            step_dir = viz_dir / str(record.get("step_dir", ""))
            records.append(
                {
                    "step": record.get("step"),
                    "timestep": record.get("timestep"),
                    "sigma": record.get("sigma"),
                    "fps": record.get("fps"),
                    "case_id": record.get("sample", {}).get("case_id"),
                    "step_dir": step_dir,
                }
            )
    else:
        for step_dir in sorted(viz_dir.glob("step_*")):
            if step_dir.is_dir():
                records.append({"step": step_dir.name, "step_dir": step_dir})

    pairs: list[dict[str, Any]] = []
    for record in records:
        ref = record["step_dir"] / reference_name
        cand = record["step_dir"] / candidate_name
        if not ref.is_file() or not cand.is_file():
            continue
        item = dict(record)
        item["reference"] = ref
        item["candidate"] = cand
        pairs.append(item)
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
    if not pairs:
        raise FileNotFoundError(
            f"No complete pairs found in {viz_dir}: {reference_name} vs {candidate_name}"
        )
    return pairs


def load_video_uint8(path: Path, num_frames: int) -> torch.Tensor:
    from decord import VideoReader

    reader = VideoReader(str(path))
    total = len(reader)
    if total <= 0:
        raise ValueError(f"Video has no frames: {path}")
    indices = np.linspace(0, total - 1, num=max(1, int(num_frames)))
    indices = np.rint(indices).astype(np.int64).clip(0, total - 1)
    frames = reader.get_batch(indices).asnumpy()
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()


def preprocess_video(frames: torch.Tensor, img_size: int) -> torch.Tensor:
    if frames.ndim != 4 or int(frames.shape[1]) != 3:
        raise ValueError(f"Expected [T,3,H,W] video tensor, got {tuple(frames.shape)}")
    frames = frames.float().div_(255.0)
    _, _, height, width = frames.shape
    short_side = int(round(256.0 / 224.0 * img_size))
    if height <= width:
        new_height = short_side
        new_width = int(round(width * short_side / height))
    else:
        new_width = short_side
        new_height = int(round(height * short_side / width))
    frames = F.interpolate(
        frames,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )
    top = max(0, (new_height - img_size) // 2)
    left = max(0, (new_width - img_size) // 2)
    frames = frames[:, :, top : top + img_size, left : left + img_size]
    if frames.shape[-2:] != (img_size, img_size):
        frames = F.interpolate(
            frames,
            size=(img_size, img_size),
            mode="bicubic",
            align_corners=False,
        )
    mean = torch.tensor(IMAGENET_MEAN, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=frames.dtype).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def clean_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def select_encoder_state(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("ema_encoder", "target_encoder", "encoder", "backbone", "model"):
            value = raw.get(key)
            if isinstance(value, dict):
                return clean_state_dict(value)
        if raw and all(torch.is_tensor(value) for value in raw.values()):
            return clean_state_dict(raw)
    raise ValueError(
        "Could not find encoder state dict. Expected one of: "
        "ema_encoder, target_encoder, encoder, backbone, model, or a raw tensor state_dict."
    )


def load_encoder(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    if not args.vjepa2_dir.is_dir():
        raise FileNotFoundError(f"V-JEPA2 repo not found: {args.vjepa2_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    sys.path.insert(0, str(args.vjepa2_dir.resolve()))

    from src.hub import backbones

    factory_name, img_size = MODEL_FACTORIES[args.model]
    factory = getattr(backbones, factory_name)
    built = factory(pretrained=False)
    encoder = built[0] if isinstance(built, tuple) else built
    if isinstance(built, tuple) and len(built) > 1:
        del built

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = select_encoder_state(raw)
    msg = encoder.load_state_dict(state, strict=False)
    print(f"Loaded encoder checkpoint: {args.checkpoint}")
    print(f"load_state_dict message: {msg}")

    encoder.eval().to(device=device, dtype=dtype)
    return encoder, img_size


@torch.inference_mode()
def extract_features(
    encoder,
    path: Path,
    *,
    num_frames: int,
    img_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    frames = load_video_uint8(path, num_frames=num_frames)
    video = preprocess_video(frames, img_size=img_size).to(device=device, dtype=dtype)
    features = encoder(video)
    if isinstance(features, (list, tuple)):
        features = features[-1]
    return features.detach().float().cpu()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = dtype_for(device, args.dtype)
    viz_dir = args.viz_dir.expanduser().resolve() if args.viz_dir else find_latest_viz_dir(args.viz_base)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DEFAULT_OUTPUT_BASE / stamp
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(
        viz_dir=viz_dir,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
        max_pairs=args.max_pairs,
    )
    print(f"Visualization dir: {viz_dir}")
    print(f"Pairs: {len(pairs)}")
    print(f"Device/dtype: {device}/{dtype}")

    encoder, img_size = load_encoder(args, device=device, dtype=dtype)
    results: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        ref_feat = extract_features(
            encoder,
            pair["reference"],
            num_frames=args.num_frames,
            img_size=img_size,
            device=device,
            dtype=dtype,
        )
        cand_feat = extract_features(
            encoder,
            pair["candidate"],
            num_frames=args.num_frames,
            img_size=img_size,
            device=device,
            dtype=dtype,
        )
        if ref_feat.shape != cand_feat.shape:
            raise ValueError(
                f"Feature shape mismatch for {pair['step_dir']}: "
                f"{tuple(ref_feat.shape)} vs {tuple(cand_feat.shape)}"
            )
        mse = torch.mean((ref_feat - cand_feat) ** 2).item()
        result = {
            "index": index,
            "step": pair.get("step"),
            "case_id": pair.get("case_id"),
            "timestep": pair.get("timestep"),
            "sigma": pair.get("sigma"),
            "fps": pair.get("fps"),
            "feature_shape": list(ref_feat.shape),
            "mse": float(mse),
            "reference": str(pair["reference"]),
            "candidate": str(pair["candidate"]),
        }
        results.append(result)
        print(
            f"[{index}/{len(pairs)}] step={result['step']} "
            f"sigma={result['sigma']} mse={mse:.8f}"
        )

    mse_values = [item["mse"] for item in results]
    summary = {
        "count": len(results),
        "mean_mse": float(np.mean(mse_values)),
        "std_mse": float(np.std(mse_values)),
        "min_mse": float(np.min(mse_values)),
        "max_mse": float(np.max(mse_values)),
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "vjepa2_dir": str(args.vjepa2_dir.resolve()),
        "viz_dir": str(viz_dir),
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "reference_name": args.reference_name,
        "candidate_name": args.candidate_name,
        "num_frames": args.num_frames,
        "img_size": img_size,
        "device": str(device),
        "dtype": str(dtype),
        "summary": summary,
        "results": results,
    }
    result_path = output_dir / "mse_results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
