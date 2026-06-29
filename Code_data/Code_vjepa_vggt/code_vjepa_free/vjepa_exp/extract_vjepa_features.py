from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_ANALYSIS_ROOT = Path("/data/gaoya/agent-data/outputs/vjepa_wan_precheck")
VJEPA2_REPO = Path("/home/gaoya/Code_Video/vjepa2-main")
VJEPA2_1_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")


def add_repo_to_path() -> None:
    repo = str(VJEPA2_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def load_video_frames(video_path: Path, target_frames: int) -> np.ndarray:
    from decord import VideoReader

    vr = VideoReader(str(video_path))
    num_frames = len(vr)
    if num_frames == 0:
        raise ValueError(f"Empty video: {video_path}")
    if num_frames >= target_frames:
        frame_idx = np.linspace(0, num_frames - 1, target_frames, dtype=int)
    else:
        frame_idx = np.arange(num_frames, dtype=int)
    return vr.get_batch(frame_idx).asnumpy()


def load_vjepa21_encoder(device: torch.device, out_layers: list[int]):
    add_repo_to_path()
    from local_vjepa21_backbone import build_vjepa2_1_vit_large_384_encoder

    encoder = build_vjepa2_1_vit_large_384_encoder(
        checkpoint_path=str(VJEPA2_1_CKPT),
        map_location="cpu",
        out_layers=out_layers,
    )
    encoder = encoder.to(device).eval()
    return encoder


def validate_out_layers(encoder: torch.nn.Module, out_layers: list[int]) -> None:
    num_layers = int(encoder.get_num_layers())
    invalid = [layer for layer in out_layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            "Unsupported out_layers for the local V-JEPA 2.1 Large backbone copy. "
            f"requested={out_layers}, invalid={invalid}, valid_range=[0, {num_layers - 1}]."
        )


def load_vjepa21_preprocessor():
    add_repo_to_path()
    from evals.hub.preprocessor import vjepa2_preprocessor

    return vjepa2_preprocessor(crop_size=384)


def to_model_input(frames_thwc: np.ndarray, processor, device: torch.device) -> torch.Tensor:
    video = torch.from_numpy(frames_thwc).permute(0, 3, 1, 2)
    x = processor(video)[0].to(device).unsqueeze(0)
    return x


def summarize_layer_tokens(layer_tokens: torch.Tensor) -> dict[str, Any]:
    x = layer_tokens.detach().float().cpu()
    temporal_diff = None
    if x.shape[1] > 1:
        temporal_diff = torch.norm(x[:, 1:] - x[:, :-1], dim=-1).mean().item()
    else:
        temporal_diff = 0.0

    affinity_probe = None
    if x.shape[1] >= 2:
        a = torch.nn.functional.normalize(x[:, :-1], dim=-1)
        b = torch.nn.functional.normalize(x[:, 1:], dim=-1)
        affinity_probe = (a * b).sum(dim=-1).mean().item()
    else:
        affinity_probe = 0.0

    return {
        "shape": list(x.shape),
        "token_mean": float(x.mean().item()),
        "token_std": float(x.std().item()),
        "motion_saliency_mean": float(temporal_diff),
        "adjacent_affinity_mean": float(affinity_probe),
    }


def extract_case(
    video_path: Path,
    output_dir: Path,
    device: torch.device,
    target_frames: int,
    out_layers: list[int],
) -> Path:
    processor = load_vjepa21_preprocessor()
    encoder = load_vjepa21_encoder(device=device, out_layers=out_layers)
    validate_out_layers(encoder, out_layers)

    frames = load_video_frames(video_path, target_frames=target_frames)
    x = to_model_input(frames, processor=processor, device=device)

    with torch.inference_mode():
        layer_outputs = encoder(x)

    if len(layer_outputs) != len(out_layers):
        raise RuntimeError(f"Unexpected number of layer outputs: {len(layer_outputs)} vs {len(out_layers)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "video_path": str(video_path),
        "target_frames": target_frames,
        "out_layers": out_layers,
        "num_sampled_frames": int(frames.shape[0]),
        "layers": {},
    }

    for layer_idx, tokens in zip(out_layers, layer_outputs):
        layer_key = f"layer_{layer_idx}"
        summary["layers"][layer_key] = summarize_layer_tokens(tokens)
        torch.save(tokens.detach().cpu(), output_dir / f"{layer_key}.pt")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract multi-layer V-JEPA features from a Wan baseline video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--out-layers",
        nargs="+",
        type=int,
        default=[5, 7, 8, 9, 11, 17, 23],
        help="V-JEPA block indices to export from the local V-JEPA 2.1 Large backbone copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    video_path = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_path = extract_case(
        video_path=video_path,
        output_dir=output_dir,
        device=device,
        target_frames=args.target_frames,
        out_layers=args.out_layers,
    )
    print(str(summary_path))


if __name__ == "__main__":
    main()
