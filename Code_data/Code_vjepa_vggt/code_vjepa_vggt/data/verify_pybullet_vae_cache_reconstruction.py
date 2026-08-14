#!/usr/bin/env python3
"""Decode cached PyBullet Wan latents and compare them with source video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from code_vjepa_vggt.data.prepare_pybullet_vae_cache import (
    _encode_sample,
    _latent_comparison_metrics,
    _load_vae,
)
from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import (
    PyBullet0713NoGTBoxDataset,
)
from code_vjepa_vggt.data.pybullet_vae_cache import sha256_file


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"
)
DEFAULT_CACHE_NAME = "vae_latents_wan22_512x896_49f_prefix_bf16"
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/physrvg-openvid-rl/vae-cache-reconstruction-check"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument(
        "--logical-key",
        action="append",
        default=[],
        help="Explicit family/case key. Repeat to check multiple samples.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _select_keys(index_rows: list[dict[str, Any]], explicit: list[str], count: int) -> list[str]:
    available = {str(row["logical_key"]) for row in index_rows}
    if explicit:
        missing = [key for key in explicit if key not in available]
        if missing:
            raise KeyError(f"Logical keys are absent from the cache index: {missing}")
        return list(dict.fromkeys(explicit))
    if count <= 0:
        raise ValueError(f"num_samples must be positive, got {count}")
    positions = np.linspace(0, len(index_rows) - 1, min(count, len(index_rows)))
    indices = list(dict.fromkeys(int(round(value)) for value in positions))
    return [str(index_rows[index]["logical_key"]) for index in indices]


def _to_unit_video(video_cthw: torch.Tensor) -> torch.Tensor:
    return ((video_cthw.detach().float().cpu().permute(1, 0, 2, 3) + 1.0) / 2.0).clamp(0, 1)


def _ssim_per_frame(reference: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    kernel_size = 11
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(reference, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(reconstruction, kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(reference.square(), kernel_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(reconstruction.square(), kernel_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(reference * reconstruction, kernel_size, 1, padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(torch.finfo(torch.float32).eps)).mean(
        dim=(1, 2, 3)
    )


def _comparison_metrics(reference: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, Any]:
    if reference.shape != reconstruction.shape:
        raise ValueError(
            f"RGB reconstruction shape mismatch: source={tuple(reference.shape)}, "
            f"decoded={tuple(reconstruction.shape)}"
        )
    delta = reconstruction - reference
    abs_delta = delta.abs()
    mse_per_frame = delta.square().mean(dim=(1, 2, 3))
    mae_per_frame = abs_delta.mean(dim=(1, 2, 3))
    psnr_per_frame = -10.0 * torch.log10(mse_per_frame.clamp_min(1e-12))
    ssim_per_frame = _ssim_per_frame(reference, reconstruction)

    lag_mae: dict[str, float] = {}
    for lag in range(-4, 5):
        if lag < 0:
            lhs, rhs = reconstruction[-lag:], reference[:lag]
        elif lag > 0:
            lhs, rhs = reconstruction[:-lag], reference[lag:]
        else:
            lhs, rhs = reconstruction, reference
        lag_mae[str(lag)] = float((lhs - rhs).abs().mean().item())

    temporal_reference = reference[1:] - reference[:-1]
    temporal_reconstruction = reconstruction[1:] - reconstruction[:-1]
    return {
        "mae": float(abs_delta.mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "max_abs_error": float(abs_delta.max().item()),
        "psnr_db": float(psnr_per_frame.mean().item()),
        "ssim": float(ssim_per_frame.mean().item()),
        "per_frame_mae": [float(value) for value in mae_per_frame.tolist()],
        "per_frame_psnr_db": [float(value) for value in psnr_per_frame.tolist()],
        "per_frame_ssim": [float(value) for value in ssim_per_frame.tolist()],
        "channel_mean_error_rgb": [
            float(value) for value in delta.mean(dim=(0, 2, 3)).tolist()
        ],
        "temporal_delta_mae": float(
            (temporal_reconstruction - temporal_reference).abs().mean().item()
        ),
        "lag_mae_minus4_to_plus4": lag_mae,
        "aligned_mae_is_best_lag": min(lag_mae, key=lag_mae.get) == "0",
        "reverse_order_mae": float((reconstruction - reference.flip(0)).abs().mean().item()),
    }


def _add_header(frame: np.ndarray, labels: list[str]) -> np.ndarray:
    height, width, _ = frame.shape
    panel_width = width // len(labels)
    canvas = Image.new("RGB", (width, height + 28), color="black")
    canvas.paste(Image.fromarray(frame), (0, 28))
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(labels):
        draw.text((index * panel_width + 8, 7), label, fill="white")
    return np.asarray(canvas)


def _write_comparison_video(
    output_path: Path,
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    fps: float,
) -> None:
    reference_np = (reference.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    reconstruction_np = (
        reconstruction.permute(0, 2, 3, 1).numpy() * 255.0
    ).round().astype(np.uint8)
    error_np = (
        (reconstruction - reference).abs().permute(0, 2, 3, 1).numpy() * 4.0 * 255.0
    ).clip(0, 255).round().astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, quality=8, macro_block_size=1) as writer:
        for source_frame, decoded_frame, error_frame in zip(
            reference_np, reconstruction_np, error_np, strict=True
        ):
            combined = np.concatenate([source_frame, decoded_frame, error_frame], axis=1)
            writer.append_data(
                _add_header(combined, ["preprocessed source", "cached latent decoded", "abs error x4"])
            )


def _read_cache_index(cache_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((cache_dir / "cache_config.json").read_text(encoding="utf-8"))
    index_path = cache_dir / str(config["index_file"])
    index_rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(index_rows) != int(config["num_samples"]):
        raise RuntimeError(
            f"Cache index count mismatch: config={config['num_samples']}, rows={len(index_rows)}"
        )
    if sha256_file(index_path) != str(config["index_sha256"]):
        raise RuntimeError("Cache index SHA-256 does not match cache_config.json")
    return config, index_rows


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    cache_dir = (args.cache_dir or dataset_root / DEFAULT_CACHE_NAME).expanduser().resolve()
    wan_root = args.wan_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    vae_path = wan_root / "Wan2.2_VAE.pth"
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Wan VAE reconstruction verification requires a CUDA device")

    config, index_rows = _read_cache_index(cache_dir)
    selected_keys = _select_keys(index_rows, args.logical_key, args.num_samples)
    rows_by_key = {str(row["logical_key"]): row for row in index_rows}

    dataset = PyBullet0713NoGTBoxDataset(
        root=dataset_root,
        split="all",
        resolution=(args.height, args.width),
        num_frames=args.num_frames,
        num_context_frames=min(8, args.num_frames),
        sampling_strategy="prefix",
        vae_cache_dir=cache_dir,
        vae_checkpoint_path=vae_path,
    )
    records_by_key = {record.key: record for record in dataset.samples}
    missing_records = [key for key in selected_keys if key not in records_by_key]
    if missing_records:
        raise RuntimeError(f"Selected cache keys are absent from the dataset: {missing_records}")

    device = torch.device(args.device)
    pipe = _load_vae(vae_path, device=device, dtype=torch.bfloat16)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_reports: list[dict[str, Any]] = []

    for sample_number, logical_key in enumerate(selected_keys, start=1):
        row = rows_by_key[logical_key]
        record = records_by_key[logical_key]
        source_path = Path(record.video_path).resolve()
        source_stat = source_path.stat()
        if source_path.relative_to(dataset_root).as_posix() != str(row["source_relpath"]):
            raise RuntimeError(f"Source path mismatch for {logical_key}")
        if source_stat.st_size != int(row["source_size"]):
            raise RuntimeError(f"Source size mismatch for {logical_key}")
        if sha256_file(source_path) != str(row["source_sha256"]):
            raise RuntimeError(f"Source SHA-256 mismatch for {logical_key}")
        if list(row["sampled_frame_indices"]) != list(range(args.num_frames)):
            raise RuntimeError(f"Unexpected cached frame indices for {logical_key}")

        sample = dataset._load_sample(record)
        latent = sample["precomputed_input_latents"]
        online_context_latent = _encode_sample(
            pipe,
            sample["context_video"],
            torch.bfloat16,
        )
        cached_context_latent = latent[:, : online_context_latent.shape[1]]
        context_prefix_metrics = _latent_comparison_metrics(
            online_context_latent,
            cached_context_latent,
        )
        if not context_prefix_metrics["within_tolerance"]:
            raise RuntimeError(
                f"Cached prefix is inconsistent with online context encoding for {logical_key}: "
                f"{context_prefix_metrics}"
            )
        decoded = pipe.vae.decode(latent.unsqueeze(0), device=device, tiled=False)[0]
        reference_video = _to_unit_video(sample["video"])
        decoded_video = _to_unit_video(decoded)
        metrics = _comparison_metrics(reference_video, decoded_video)

        safe_name = logical_key.replace("/", "__")
        comparison_path = output_dir / f"{sample_number:02d}_{safe_name}.mp4"
        _write_comparison_video(comparison_path, reference_video, decoded_video, args.fps)
        sample_report = {
            "logical_key": logical_key,
            "source_video": str(source_path),
            "source_sha256": str(row["source_sha256"]),
            "sampled_frame_indices": list(row["sampled_frame_indices"]),
            "latent_file": str(cache_dir / str(row["latent_file"])),
            "latent_shape": list(latent.shape),
            "latent_dtype": str(latent.dtype).removeprefix("torch."),
            "context_rgb_frames": int(sample["context_video"].shape[1]),
            "context_latent_frames": int(online_context_latent.shape[1]),
            "cached_prefix_vs_online_context_encode": context_prefix_metrics,
            "decoded_shape_tchw": list(decoded_video.shape),
            "comparison_video": str(comparison_path),
            "metrics": metrics,
        }
        sample_reports.append(sample_report)
        print(
            f"[vae-cache-reconstruction] {sample_number}/{len(selected_keys)} "
            f"key={logical_key} mae={metrics['mae']:.6f} "
            f"psnr={metrics['psnr_db']:.3f} ssim={metrics['ssim']:.6f} "
            f"aligned_best_lag={metrics['aligned_mae_is_best_lag']} "
            f"prefix_rel_l2={context_prefix_metrics['relative_l2_error']:.6f}",
            flush=True,
        )

    aggregate = {
        "mean_mae": float(np.mean([row["metrics"]["mae"] for row in sample_reports])),
        "mean_psnr_db": float(
            np.mean([row["metrics"]["psnr_db"] for row in sample_reports])
        ),
        "mean_ssim": float(np.mean([row["metrics"]["ssim"] for row in sample_reports])),
        "all_aligned_mae_best_lag": all(
            bool(row["metrics"]["aligned_mae_is_best_lag"]) for row in sample_reports
        ),
        "all_cached_prefixes_match_online_context_encode": all(
            bool(row["cached_prefix_vs_online_context_encode"]["within_tolerance"])
            for row in sample_reports
        ),
    }
    report = {
        "status": "completed",
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "cache_encoding_id": str(config["encoding_id"]),
        "vae_checkpoint": str(vae_path),
        "vae_checkpoint_sha256": sha256_file(vae_path),
        "selection": "explicit" if args.logical_key else "evenly_spaced_index_rows",
        "num_samples": len(sample_reports),
        "aggregate": aggregate,
        "samples": sample_reports,
    }
    report_path = output_dir / "reconstruction_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[vae-cache-reconstruction] report={report_path}", flush=True)
    print(f"[vae-cache-reconstruction] aggregate={json.dumps(aggregate, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
