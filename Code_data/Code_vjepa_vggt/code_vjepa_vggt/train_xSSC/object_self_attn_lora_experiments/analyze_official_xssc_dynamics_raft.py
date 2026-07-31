#!/usr/bin/env python3
"""Analyze whether official xSSC slots encode video dynamics measured by RAFT.

This is a pure xSSC/RAFT analysis script: no Wan model is loaded.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
TRAIN_XSSC_ROOT = ROOT.parent
PROJECT_ROOT = TRAIN_XSSC_ROOT.parent
PACKAGE_PARENT = PROJECT_ROOT.parent
DEFAULT_TRAIN_CONFIG = ROOT / "configs/formal_full_sa_slot_dedup_merge_gpu67.json"
DEFAULT_OUTPUT_DIR = Path("/data/gaoya/agent-data/outputs/official_xssc_dynamics_raft")
OFFICIAL_XSSC_ROOT = Path("/home/gaoya/Code_Video/xSSC-main")
RAFT_ROOT = Path("/home/gaoya/Code_Video/DreamWorld-main/extract/RAFT")
RAFT_CHECKPOINT = Path("/data/gaoya/ckpt/RAFT-Things/models/raft-things.pth")

for item in (PACKAGE_PARENT, PROJECT_ROOT, TRAIN_XSSC_ROOT, ROOT):
    text = str(item)
    if text not in sys.path:
        sys.path.insert(0, text)


class ContainsNamespace(SimpleNamespace):
    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--official-root", type=Path, default=Path("/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--indices", default="808,58755,142643")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--xssc-input-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--xssc-batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def load_dataset_args(config_path: Path):
    import launch_slot_dedup_from_config as config_launcher
    import train_xssc_object_self_attn_lora as object_train
    import train_xssc_object_self_attn_lora_slot_dedup as dedup_train

    raw, _ = config_launcher.base.load_config(config_path)
    config = config_launcher.validate_config(raw, config_path.expanduser().resolve().parent)
    command = config_launcher.build_command(config, Path("/tmp/unused_xssc_dynamics"))
    script_index = next(index for index, token in enumerate(command) if str(token).endswith(".py"))
    argv = [str(item) for item in command[script_index + 1 :]]
    args = object_train.tvn.prepare_args(dedup_train.build_parser().parse_args(argv))
    args.no_context_ratio = 0.0
    return args, object_train


def build_official_model(checkpoint: Path, device: torch.device):
    if str(OFFICIAL_XSSC_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_XSSC_ROOT))
    from object_centric_bench.util import Config, build_from_config
    import timm

    config_path = OFFICIAL_XSSC_ROOT / "config-randsfq/rsfq2_r-ytvis.py"
    cfg = Config.fromfile(config_path)
    original_create_model = timm.create_model

    def create_model_offline(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = create_model_offline
    try:
        model = build_from_config(cfg.model)
    finally:
        timm.create_model = original_create_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if state and all(str(key).startswith("m.") for key in state):
        state = {str(key)[2:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval().to(device=device)
    return model, cfg


def build_raft(device: torch.device, iters: int):
    core_root = RAFT_ROOT / "core"
    for item in (RAFT_ROOT, core_root):
        text = str(item)
        if text not in sys.path:
            sys.path.insert(0, text)
    from raft import RAFT

    args = ContainsNamespace(
        small=False,
        mixed_precision=True,
        alternate_corr=False,
        dropout=0.0,
    )
    wrapper = torch.nn.DataParallel(RAFT(args))
    state = torch.load(RAFT_CHECKPOINT, map_location="cpu")
    wrapper.load_state_dict(state)
    model = wrapper.module
    model.to(device=device).eval()
    model.freeze_bn()
    return model


def preprocess_video_for_xssc(
    video: torch.Tensor,
    input_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return normalized [T,3,256,256] and RGB uint8 [T,256,256,3]."""
    if video.ndim != 4:
        raise ValueError(f"video must be [C,T,H,W], got {tuple(video.shape)}")
    frames = video.permute(1, 0, 2, 3).float()
    time_steps, channels, height, width = frames.shape
    crop_size = min(int(height), int(width))
    top = (int(height) - crop_size) // 2
    left = (int(width) - crop_size) // 2
    frames = frames[:, :, top : top + crop_size, left : left + crop_size]
    frames = F.interpolate(
        frames,
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    rgb = (frames + 1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    mean = frames.new_tensor((123.675, 116.28, 103.53)).view(1, 3, 1, 1)
    std = frames.new_tensor((58.395, 57.12, 57.375)).view(1, 3, 1, 1)
    normalized = (rgb.float() - mean) / std
    rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy()
    return normalized, rgb_np


@torch.inference_mode()
def extract_official_slots(
    model: torch.nn.Module,
    normalized: torch.Tensor,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract full-video official xSSC slots with the corrected sliding transition."""
    features = []
    for start in range(0, len(normalized), batch_size):
        batch = normalized[start : start + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            feature = model.encode_backbone(batch).detach()
        features.append(feature.to(device=device, dtype=torch.bfloat16))
    feature = torch.cat(features, dim=0)

    encoded_parts = []
    for start in range(0, len(feature), batch_size):
        current = feature[start : start + batch_size]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoded = current.permute(0, 2, 3, 1)
            encoded = model.encode_posit_embed(encoded).flatten(1, 2)
            encoded = model.encode_project(encoded)
        encoded_parts.append(encoded)
    encoded_all = torch.cat(encoded_parts, dim=0)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    slot_parts = []
    attn_parts = []
    slot_window = []
    encode_window = []
    transition_dt = int(model.transit.dt)
    for frame_id in range(len(encoded_all)):
        encoded_i = encoded_all[frame_id : frame_id + 1]
        encode_window.append(encoded_i)
        encode_window = encode_window[-transition_dt:]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if frame_id == 0:
                query = model.initializ(1)
            else:
                query = model.transit(
                    torch.stack(slot_window, dim=1),
                    torch.stack(encode_window, dim=1),
                )
            slots_i, attn_i = model.aggregat(
                encoded_i,
                query,
                num_iter=None if frame_id == 0 else 1,
            )
        slot_parts.append(slots_i[0].detach().float().cpu())
        attn_parts.append(attn_i[0].detach().float().cpu())
        slot_window.append(slots_i)
        slot_window = slot_window[-(transition_dt - 1) :]

    slots = torch.stack(slot_parts, dim=0)
    attention = torch.stack(attn_parts, dim=0)
    side = int(round(attention.shape[-1] ** 0.5))
    attention = attention.view(attention.shape[0], attention.shape[1], side, side)
    return slots, attention


@torch.inference_mode()
def compute_raft_flow(
    raft: torch.nn.Module,
    rgb: np.ndarray,
    device: torch.device,
    iters: int,
) -> np.ndarray:
    from utils.utils import InputPadder

    flows = []
    for frame_id in range(len(rgb) - 1):
        image1 = torch.from_numpy(rgb[frame_id]).permute(2, 0, 1).float()[None].to(device)
        image2 = torch.from_numpy(rgb[frame_id + 1]).permute(2, 0, 1).float()[None].to(device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, flow_up = raft(image1, image2, iters=iters, test_mode=True)
        flow_up = padder.unpad(flow_up)
        flows.append(flow_up[0].permute(1, 2, 0).detach().float().cpu().numpy())
    return np.stack(flows, axis=0)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def rankdata_simple(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return pearson(rankdata_simple(x[mask]), rankdata_simple(y[mask]))


def cosine_pairwise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    flat = flat / np.maximum(norm, 1e-12)
    return flat @ flat.T


def slot_cosine_pairwise(slots: np.ndarray) -> np.ndarray:
    # slots: [T,S,D], compare frames by averaging same-slot cosine.
    normalized = slots / np.maximum(np.linalg.norm(slots, axis=-1, keepdims=True), 1e-12)
    return np.einsum("tsd,usd->tu", normalized, normalized) / slots.shape[1]


def attention_centroids(attention: np.ndarray) -> np.ndarray:
    time_steps, num_slots, height, width = attention.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    mass = attention.sum(axis=(2, 3), keepdims=True)
    norm = attention / np.maximum(mass, 1e-12)
    cx = (norm * xx[None, None]).sum(axis=(2, 3))
    cy = (norm * yy[None, None]).sum(axis=(2, 3))
    return np.stack([cx, cy], axis=-1)


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    flow_root = RAFT_ROOT / "core"
    if str(flow_root) not in sys.path:
        sys.path.insert(0, str(flow_root))
    from utils import flow_viz

    return flow_viz.flow_to_image(flow)


def downsample_flow_mag(flow_mag: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.stack(
        [
            cv2.resize(item, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            for item in flow_mag
        ],
        axis=0,
    )


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = float(np.nanstd(x))
    if std <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - float(np.nanmean(x))) / std


def analyze_arrays(slots_tsd: np.ndarray, attention_tshw: np.ndarray, flow: np.ndarray) -> dict[str, Any]:
    time_steps, num_slots, dim = slots_tsd.shape
    static = slots_tsd.mean(axis=0, keepdims=True)
    dynamic = slots_tsd - static
    static_repeated = np.repeat(static, time_steps, axis=0)
    flow_mag = np.linalg.norm(flow, axis=-1)
    global_flow = flow_mag.mean(axis=(1, 2))
    flow_low = downsample_flow_mag(flow_mag, tuple(attention_tshw.shape[-2:]))
    attn_for_flow = attention_tshw[:-1].copy()
    attn_for_flow = attn_for_flow / np.maximum(attn_for_flow.sum(axis=(2, 3), keepdims=True), 1e-12)
    slot_flow = (attn_for_flow * flow_low[:, None]).sum(axis=(2, 3))

    raw_delta_l2 = np.linalg.norm(slots_tsd[1:] - slots_tsd[:-1], axis=-1) / math.sqrt(dim)
    raw_cos = np.sum(slots_tsd[1:] * slots_tsd[:-1], axis=-1) / np.maximum(
        np.linalg.norm(slots_tsd[1:], axis=-1) * np.linalg.norm(slots_tsd[:-1], axis=-1),
        1e-12,
    )
    raw_one_minus_cos = 1.0 - raw_cos
    dyn_cos = np.sum(dynamic[1:] * dynamic[:-1], axis=-1) / np.maximum(
        np.linalg.norm(dynamic[1:], axis=-1) * np.linalg.norm(dynamic[:-1], axis=-1),
        1e-12,
    )
    dyn_one_minus_cos = 1.0 - dyn_cos
    dyn_energy = np.linalg.norm(dynamic, axis=-1) / math.sqrt(dim)
    dyn_energy_mid = 0.5 * (dyn_energy[1:] + dyn_energy[:-1])
    static_alignment = np.sum(slots_tsd * static[0][None], axis=-1) / np.maximum(
        np.linalg.norm(slots_tsd, axis=-1) * np.linalg.norm(static[0][None], axis=-1),
        1e-12,
    )

    attn_delta_l1 = np.abs(attention_tshw[1:] - attention_tshw[:-1]).mean(axis=(2, 3))
    centroids = attention_centroids(attention_tshw)
    centroid_shift = np.linalg.norm(centroids[1:] - centroids[:-1], axis=-1)
    combined = zscore(dyn_one_minus_cos) + zscore(centroid_shift)

    adjacent_candidates = {
        "feature_delta_l2": raw_delta_l2.mean(axis=1),
        "raw_one_minus_cos": raw_one_minus_cos.mean(axis=1),
        "dynamic_one_minus_cos": dyn_one_minus_cos.mean(axis=1),
        "dynamic_energy_mid": dyn_energy_mid.mean(axis=1),
        "attention_delta_l1": attn_delta_l1.mean(axis=1),
        "attention_centroid_shift": centroid_shift.mean(axis=1),
        "combined_dynamic_cos_centroid": combined.mean(axis=1),
    }
    slot_candidates = {
        "feature_delta_l2": raw_delta_l2,
        "raw_one_minus_cos": raw_one_minus_cos,
        "dynamic_one_minus_cos": dyn_one_minus_cos,
        "dynamic_energy_mid": dyn_energy_mid,
        "attention_delta_l1": attn_delta_l1,
        "attention_centroid_shift": centroid_shift,
        "combined_dynamic_cos_centroid": combined,
    }
    adjacent_correlations = {
        name: {
            "pearson": pearson(values, global_flow),
            "spearman": spearman(values, global_flow),
        }
        for name, values in adjacent_candidates.items()
    }
    slot_correlations = {
        name: {
            "pearson": pearson(values, slot_flow),
            "spearman": spearman(values, slot_flow),
        }
        for name, values in slot_candidates.items()
    }

    raw_frame_similarity = slot_cosine_pairwise(slots_tsd)
    static_frame_similarity = slot_cosine_pairwise(static_repeated)
    dynamic_frame_similarity = slot_cosine_pairwise(dynamic)
    static_alignment_by_frame = static_alignment.mean(axis=1)
    pair_xssc_raw_dist = 1.0 - raw_frame_similarity
    pair_xssc_dyn_dist = 1.0 - dynamic_frame_similarity
    pair_flow = np.zeros((time_steps, time_steps), dtype=np.float64)
    cumsum = np.concatenate([[0.0], np.cumsum(global_flow)])
    for start in range(time_steps):
        for end in range(start + 1, time_steps):
            pair_flow[start, end] = cumsum[end] - cumsum[start]
            pair_flow[end, start] = pair_flow[start, end]
    triu = np.triu_indices(time_steps, k=1)
    pair_correlations = {
        "raw_pair_distance_vs_cumulative_flow": {
            "pearson": pearson(pair_xssc_raw_dist[triu], pair_flow[triu]),
            "spearman": spearman(pair_xssc_raw_dist[triu], pair_flow[triu]),
        },
        "dynamic_pair_distance_vs_cumulative_flow": {
            "pearson": pearson(pair_xssc_dyn_dist[triu], pair_flow[triu]),
            "spearman": spearman(pair_xssc_dyn_dist[triu], pair_flow[triu]),
        },
    }

    dyn_fft = np.fft.rfft(dynamic, axis=0)
    raw_fft = np.fft.rfft(slots_tsd, axis=0)
    flow_fft = np.fft.rfft(global_flow)
    def band_energy(arr: np.ndarray) -> dict[str, float]:
        energy = np.abs(arr) ** 2
        by_freq = energy.reshape(energy.shape[0], -1).mean(axis=1)
        total = float(by_freq.sum()) + 1e-12
        n = len(by_freq)
        low_end = max(1, math.ceil(n * 0.25))
        mid_end = max(low_end + 1, math.ceil(n * 0.60))
        return {
            "dc": float(by_freq[0] / total),
            "low": float(by_freq[1:low_end].sum() / total) if low_end > 1 else 0.0,
            "mid": float(by_freq[low_end:mid_end].sum() / total),
            "high": float(by_freq[mid_end:].sum() / total),
        }

    return {
        "time_steps": time_steps,
        "num_slots": num_slots,
        "slot_dim": dim,
        "global_flow": global_flow.tolist(),
        "slot_flow": slot_flow.tolist(),
        "raw_delta_l2": raw_delta_l2.tolist(),
        "raw_one_minus_cos": raw_one_minus_cos.tolist(),
        "dynamic_one_minus_cos": dyn_one_minus_cos.tolist(),
        "dynamic_energy": dyn_energy.tolist(),
        "static_alignment_by_frame": static_alignment_by_frame.tolist(),
        "attention_delta_l1": attn_delta_l1.tolist(),
        "attention_centroid_shift": centroid_shift.tolist(),
        "combined_dynamic_cos_centroid": combined.tolist(),
        "raw_frame_similarity": raw_frame_similarity.tolist(),
        "static_frame_similarity": static_frame_similarity.tolist(),
        "dynamic_frame_similarity": dynamic_frame_similarity.tolist(),
        "pair_flow": pair_flow.tolist(),
        "adjacent_correlations": adjacent_correlations,
        "slot_correlations": slot_correlations,
        "pair_correlations": pair_correlations,
        "frequency": {
            "raw_slot": band_energy(raw_fft),
            "dynamic_slot": band_energy(dyn_fft),
            "global_flow": band_energy(flow_fft),
        },
    }


def best_metric(correlations: dict[str, dict[str, float]], key: str = "spearman") -> tuple[str, float]:
    valid = [
        (name, values.get(key, float("nan")))
        for name, values in correlations.items()
        if np.isfinite(values.get(key, float("nan")))
    ]
    if not valid:
        return "", float("nan")
    return max(valid, key=lambda item: abs(item[1]))


def write_video(path: Path, frames_rgb: np.ndarray, fps: float = 8.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb = np.ascontiguousarray(frames_rgb.astype(np.uint8))
    height, width = frames_rgb.shape[1:3]
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(path),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(frames_rgb.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def plot_heatmap(path: Path, matrix: np.ndarray, title: str, cmap: str = "viridis", vmin=None, vmax=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=150)
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("frame")
    ax.set_ylabel("frame")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_adjacent(path: Path, analysis: dict[str, Any], title: str) -> None:
    flow = np.asarray(analysis["global_flow"], dtype=np.float64)
    candidates = {
        "dyn 1-cos": np.asarray(analysis["dynamic_one_minus_cos"]).mean(axis=1),
        "feat delta": np.asarray(analysis["raw_delta_l2"]).mean(axis=1),
        "centroid shift": np.asarray(analysis["attention_centroid_shift"]).mean(axis=1),
        "combo": np.asarray(analysis["combined_dynamic_cos_centroid"]).mean(axis=1),
    }
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=150)
    ax.plot(zscore(flow), label="RAFT global flow", linewidth=2.0, color="#111827")
    for name, values in candidates.items():
        ax.plot(zscore(values), label=name, linewidth=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("adjacent transition t -> t+1")
    ax.set_ylabel("z-score")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_slot_time(path: Path, analysis: dict[str, Any], title: str) -> None:
    dyn = np.asarray(analysis["dynamic_one_minus_cos"], dtype=np.float64).T
    slot_flow = np.asarray(analysis["slot_flow"], dtype=np.float64).T
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.8), dpi=150, sharex=True)
    images = [
        axes[0].imshow(dyn, aspect="auto", cmap="magma"),
        axes[1].imshow(slot_flow, aspect="auto", cmap="magma"),
    ]
    axes[0].set_title(title + " | xSSC dyn 1-cos", fontsize=9)
    axes[1].set_title("RAFT slot-weighted flow", fontsize=9)
    axes[1].set_xlabel("adjacent transition t -> t+1")
    for ax in axes:
        ax.set_ylabel("slot")
    for ax, image in zip(axes, images):
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_frequency(path: Path, analysis: dict[str, Any], title: str) -> None:
    labels = ["dc", "low", "mid", "high"]
    groups = ["raw_slot", "dynamic_slot", "global_flow"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=150)
    for offset, group in enumerate(groups):
        values = [analysis["frequency"][group][label] for label in labels]
        ax.bar(x + (offset - 1) * width, values, width=width, label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("energy ratio")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_case_model_assets(case_dir: Path, model_name: str, analysis: dict[str, Any]) -> dict[str, str]:
    model_dir = case_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "adjacent_plot": model_dir / "adjacent_dynamics_vs_flow.png",
        "raw_pair": model_dir / "raw_pair_similarity.png",
        "static_pair": model_dir / "static_pair_similarity.png",
        "dynamic_pair": model_dir / "dynamic_pair_similarity.png",
        "pair_flow": model_dir / "pair_cumulative_flow.png",
        "slot_time": model_dir / "slot_time_xssc_vs_raft.png",
        "frequency": model_dir / "frequency_energy.png",
    }
    plot_adjacent(paths["adjacent_plot"], analysis, model_name)
    plot_heatmap(paths["raw_pair"], np.asarray(analysis["raw_frame_similarity"]), "raw slot frame similarity", "coolwarm", -1, 1)
    plot_heatmap(paths["static_pair"], np.asarray(analysis["static_frame_similarity"]), "static component frame similarity", "coolwarm", -1, 1)
    plot_heatmap(paths["dynamic_pair"], np.asarray(analysis["dynamic_frame_similarity"]), "dynamic residual frame similarity", "coolwarm", -1, 1)
    plot_heatmap(paths["pair_flow"], np.asarray(analysis["pair_flow"]), "RAFT cumulative adjacent-flow between frame pairs", "viridis")
    plot_slot_time(paths["slot_time"], analysis, model_name)
    plot_frequency(paths["frequency"], analysis, model_name)
    return {key: str(path.relative_to(case_dir.parents[1])) for key, path in paths.items()}


def build_html(output_dir: Path, report: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    sections = []
    for case in cases:
        model_cards = []
        for model in case["models"]:
            adj_best = model["best_adjacent"]
            slot_best = model["best_slot"]
            pair = model["pair_correlations"]
            assets = model["assets"]
            corr_rows = []
            for name, values in model["adjacent_correlations"].items():
                corr_rows.append(
                    f"<tr><td>{html.escape(name)}</td><td>{values['pearson']:.3f}</td><td>{values['spearman']:.3f}</td></tr>"
                )
            model_cards.append(
                f"""
                <article class="model">
                  <h3>{html.escape(model['name'])}</h3>
                  <p class="small">best adjacent: <b>{html.escape(adj_best[0])}</b> Spearman={adj_best[1]:.3f};
                  best slot-local: <b>{html.escape(slot_best[0])}</b> Spearman={slot_best[1]:.3f}</p>
                  <p class="small">pair raw-vs-flow Spearman={pair['raw_pair_distance_vs_cumulative_flow']['spearman']:.3f};
                  pair dynamic-vs-flow Spearman={pair['dynamic_pair_distance_vs_cumulative_flow']['spearman']:.3f}</p>
                  <div class="plots">
                    <figure><img src="{assets['adjacent_plot']}" loading="lazy"><figcaption>time-domain adjacent dynamics</figcaption></figure>
                    <figure><img src="{assets['slot_time']}" loading="lazy"><figcaption>slot-time xSSC vs RAFT</figcaption></figure>
                    <figure><img src="{assets['raw_pair']}" loading="lazy"><figcaption>all-pair raw similarity</figcaption></figure>
                    <figure><img src="{assets['static_pair']}" loading="lazy"><figcaption>all-pair static similarity</figcaption></figure>
                    <figure><img src="{assets['dynamic_pair']}" loading="lazy"><figcaption>all-pair dynamic similarity</figcaption></figure>
                    <figure><img src="{assets['pair_flow']}" loading="lazy"><figcaption>all-pair RAFT cumulative flow</figcaption></figure>
                    <figure><img src="{assets['frequency']}" loading="lazy"><figcaption>frequency-domain energy</figcaption></figure>
                  </div>
                  <details><summary>Adjacent correlations vs RAFT global flow</summary>
                    <table><thead><tr><th>xSSC function</th><th>Pearson</th><th>Spearman</th></tr></thead><tbody>{''.join(corr_rows)}</tbody></table>
                  </details>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="case">
              <h2>case {case['index']} | {html.escape(case['source'])}</h2>
              <div class="casegrid">
                <figure><video src="{case['input_video']}" controls muted preload="metadata"></video><figcaption>49-frame xSSC input</figcaption></figure>
                <figure><video src="{case['flow_video']}" controls muted preload="metadata"></video><figcaption>RAFT adjacent flow</figcaption></figure>
              </div>
              <div class="models">{''.join(model_cards)}</div>
            </section>
            """
        )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Official xSSC Dynamics vs RAFT</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:#101214; color:#eef2f7; font:13px system-ui,sans-serif; letter-spacing:0; }}
    header {{ position:sticky; top:0; z-index:5; padding:12px 16px; background:#16191c; border-bottom:1px solid #333b44; }}
    h1 {{ margin:0 0 6px; font-size:20px; }}
    h2 {{ margin:0 0 12px; font-size:17px; }}
    h3 {{ margin:0 0 6px; font-size:14px; }}
    main {{ max-width:2100px; margin:0 auto; padding:16px; }}
    code {{ color:#d5f5ff; }}
    .summary,.small {{ color:#bdc7d1; }}
    .case {{ padding:18px 0 28px; border-top:1px solid #30363d; }}
    .casegrid {{ display:grid; grid-template-columns:repeat(2,minmax(0,420px)); gap:12px; margin-bottom:12px; }}
    .models {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:12px; }}
    .model {{ border:1px solid #333b44; background:#14191e; padding:10px; border-radius:8px; }}
    .plots {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    figure {{ margin:0; min-width:0; }}
    img,video {{ display:block; width:100%; background:#000; border:1px solid #303942; }}
    figcaption {{ padding:4px 1px; color:#aeb8c2; font-size:11px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
    th,td {{ border:1px solid #303942; padding:5px 7px; text-align:left; }}
    th {{ background:#192027; }}
    td {{ background:#12171c; color:#cbd5df; }}
    details {{ margin-top:8px; }}
    @media(max-width:900px) {{ .models,.casegrid,.plots {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Official xSSC dynamics vs RAFT</h1>
    <div class="summary">{html.escape(json.dumps(report, ensure_ascii=False))}</div>
  </header>
  <main>
    <p class="small"><b>Candidate xSSC-only dynamics function:</b>
    D(t)=mean_s zscore(1-cos(z_dyn[t,s], z_dyn[t+1,s])) + zscore(||centroid_attn[t+1,s]-centroid_attn[t,s]||).
    It is evaluated against RAFT only for analysis; it does not use RAFT as input.</p>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_args, object_train = load_dataset_args(args.train_config)
    dataset = object_train.base.build_dataset(dataset_args)
    indices = [int(item) for item in str(args.indices).replace(",", " ").split()]
    checkpoints = sorted(args.official_root.expanduser().resolve().glob("*.pth"))
    if len(checkpoints) != 3:
        raise RuntimeError(f"Expected 3 official xSSC weights, found {len(checkpoints)} under {args.official_root}")

    raft = build_raft(device, args.raft_iters)
    case_cache: dict[int, dict[str, Any]] = {}
    cases_out = []
    for case_position, index in enumerate(indices, start=1):
        sample = dataset[index]
        video = sample.get("video", sample["context_video"])[:, : args.num_frames]
        normalized, rgb = preprocess_video_for_xssc(video, args.xssc_input_size)
        flow = compute_raft_flow(raft, rgb, device, args.raft_iters)
        case_id = f"case_{case_position:02d}_index_{index:06d}"
        case_dir = output_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        input_video = case_dir / "xssc_input_49f.mp4"
        flow_video = case_dir / "raft_flow.mp4"
        write_video(input_video, rgb, fps=8.0)
        flow_rgb = np.stack([flow_to_rgb(item) for item in flow], axis=0)
        write_video(flow_video, flow_rgb, fps=8.0)
        np.savez_compressed(
            case_dir / "raft_flow.npz",
            flow=flow.astype(np.float16),
            global_flow=np.linalg.norm(flow, axis=-1).mean(axis=(1, 2)),
        )
        metadata = dict(sample.get("metadata", {}))
        case_cache[index] = {
            "normalized": normalized,
            "rgb": rgb,
            "flow": flow,
            "case_dir": case_dir,
            "case_id": case_id,
            "source": str(metadata.get("dataset_source", "unknown")),
            "input_video": input_video.relative_to(output_dir).as_posix(),
            "flow_video": flow_video.relative_to(output_dir).as_posix(),
        }
        print(f"[raft] {case_id} flow={flow.shape}", flush=True)

    del raft
    torch.cuda.empty_cache()

    model_summaries = []
    for checkpoint in checkpoints:
        model_name = f"official_{checkpoint.stem}"
        model, cfg = build_official_model(checkpoint, device)
        model_summary = {
            "name": model_name,
            "checkpoint": str(checkpoint),
            "slot_shape": [args.num_frames, int(cfg.max_num), int(cfg.emb_dim)],
            "cases": [],
        }
        for case_position, index in enumerate(indices, start=1):
            cache = case_cache[index]
            seed = int(args.seed) + case_position * 1000 + int(re.search(r"^(\d+)", checkpoint.stem).group(1))
            slots, attention = extract_official_slots(
                model,
                cache["normalized"],
                device,
                seed,
                args.xssc_batch_size,
            )
            slots_np = slots.numpy()
            attention_np = attention.numpy()
            analysis = analyze_arrays(slots_np, attention_np, cache["flow"])
            case_dir = cache["case_dir"]
            asset_paths = render_case_model_assets(case_dir, model_name, analysis)
            np.savez_compressed(
                case_dir / model_name / "xssc_slots_attention_analysis.npz",
                slots=slots_np.astype(np.float16),
                attention=attention_np.astype(np.float16),
                global_flow=np.asarray(analysis["global_flow"], dtype=np.float32),
            )
            best_adj = best_metric(analysis["adjacent_correlations"], "spearman")
            best_slot = best_metric(analysis["slot_correlations"], "spearman")
            record = {
                "case_index": index,
                "case_id": cache["case_id"],
                "source": cache["source"],
                "best_adjacent": [best_adj[0], best_adj[1]],
                "best_slot": [best_slot[0], best_slot[1]],
                "adjacent_correlations": analysis["adjacent_correlations"],
                "slot_correlations": analysis["slot_correlations"],
                "pair_correlations": analysis["pair_correlations"],
                "frequency": analysis["frequency"],
                "assets": asset_paths,
            }
            model_summary["cases"].append(record)
            print(
                f"[xssc] {model_name} case={index} "
                f"best_adj={best_adj[0]}:{best_adj[1]:.3f} "
                f"best_slot={best_slot[0]}:{best_slot[1]:.3f}",
                flush=True,
            )
        model_summaries.append(model_summary)
        del model
        torch.cuda.empty_cache()

    cases_for_html = []
    for index in indices:
        cache = case_cache[index]
        models = []
        for model_summary in model_summaries:
            record = next(item for item in model_summary["cases"] if item["case_index"] == index)
            models.append({"name": model_summary["name"], **record})
        cases_for_html.append(
            {
                "index": index,
                "source": cache["source"],
                "input_video": cache["input_video"],
                "flow_video": cache["flow_video"],
                "models": models,
            }
        )

    report = {
        "official_weights": [str(path) for path in checkpoints],
        "indices": indices,
        "num_frames": int(args.num_frames),
        "xssc_shape": "[T,7,256]",
        "raft": str(RAFT_CHECKPOINT),
        "function_note": "RAFT is used only as an external validation target; xSSC dynamic candidates are computed from slots/slot attention.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "report": report,
                "models": model_summaries,
                "cases": cases_for_html,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    build_html(output_dir, report, cases_for_html)
    print(f"viewer={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
