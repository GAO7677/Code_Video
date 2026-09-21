"""Estimate RGB7 state and static point maps from RGB0--RGB7 only.

This entrypoint intentionally has no data-root, blueprint, or GT-state
argument.  It runs official VGGT directly (no Utonia) and the existing SAM2
video tracker seeded by an RGB temporal-motion prompt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from context_rgb_pybullet_common import (
    OBSERVED_FRAMES,
    crop_transform,
    dump_json,
    estimate_metric_centers,
    load_json,
    pixel_sha256,
    rgb_motion_circle_prompt,
    resize_crop_mask,
    robust_terminal_velocity,
    sha256_file,
    static_point_cloud,
)


VGGT_ROOT = Path("/home/gaoya/vggt_official")
VGGT_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-VGGT-1B")
SAM2_SOURCE = Path("/home/gaoya/Code_Video/phys_state_video/src")
SAM2_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt")
SAM2_CONFIG = Path("/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml")
FORBIDDEN_GPU_UUID = "GPU-4a8abb69-6a43-4b79-5713-31979b8d6d75"
INPUT_KEYS = {
    "schema", "case_id", "family", "frames", "observed_indices", "time_s",
    "shape", "pixels_sha256", "calibration", "allowed_estimator_inputs",
}
FORBIDDEN_NAMES = ("future", "state", "trajectory", "blueprint", "contact", "interaction")


def validate_gpu(gpu_uuid: str) -> tuple[int, str]:
    pattern = r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    if re.fullmatch(pattern, gpu_uuid) is None:
        raise ValueError("--gpu-uuid must be a canonical full UUID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal --gpu-uuid exactly")
    if gpu_uuid == FORBIDDEN_GPU_UUID:
        raise ValueError("physical GPU4 is forbidden")
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    rows = list(csv.reader(io.StringIO(inventory)))
    matches = [(int(row[0].strip()), row[1].strip()) for row in rows if len(row) == 2 and row[1].strip() == gpu_uuid]
    if len(matches) != 1 or matches[0][0] == 4:
        raise ValueError("GPU UUID does not resolve uniquely to an allowed physical GPU")
    return matches[0][0], inventory


def validate_input_tree(input_root: Path, manifest: dict) -> None:
    if manifest.get("schema") != "context_rgb8_pybullet_vision_manifest_v1":
        raise ValueError("unexpected vision manifest schema")
    if int(manifest.get("case_count", -1)) != 36 or len(manifest.get("records", [])) != 36:
        raise ValueError("inference is fixed to the admitted 36-case pilot")
    for path in input_root.rglob("*"):
        if path.is_file() and any(token in path.name.lower() for token in FORBIDDEN_NAMES):
            raise ValueError(f"forbidden estimator input filename: {path}")
    for record in manifest["records"]:
        case_dir = input_root / record["input_dir"]
        allowed = {"input.json", "calibration.json", *(f"rgb_{index:02d}.png" for index in range(OBSERVED_FRAMES))}
        actual = {path.name for path in case_dir.iterdir() if path.is_file()}
        if actual != allowed:
            raise ValueError(f"unexpected files for {record['case_id']}: {sorted(actual ^ allowed)}")


def load_case(input_root: Path, record: dict) -> tuple[np.ndarray, np.ndarray, dict, list[Path]]:
    input_path = input_root / record["input_json"]
    payload = load_json(input_path)
    if set(payload) != INPUT_KEYS or payload["schema"] != "context_rgb8_pybullet_input_v1":
        raise ValueError(f"unexpected input schema for {record['case_id']}")
    if payload["case_id"] != record["case_id"] or payload["family"] != record["family"]:
        raise ValueError("manifest/input case identity mismatch")
    expected = [f"rgb_{index:02d}.png" for index in range(OBSERVED_FRAMES)]
    if payload["frames"] != expected or payload["observed_indices"] != list(range(OBSERVED_FRAMES)):
        raise ValueError("only RGB0-RGB7 are accepted")
    paths = [input_path.parent / name for name in expected]
    rgb = np.stack([np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths])
    if list(rgb.shape) != payload["shape"] or pixel_sha256(rgb) != payload["pixels_sha256"]:
        raise ValueError("RGB shape/hash mismatch")
    times = np.asarray(payload["time_s"], dtype=np.float64)
    if times.shape != (OBSERVED_FRAMES,) or not np.all(np.diff(times) > 0):
        raise ValueError("invalid observed timestamps")
    calibration = load_json(input_path.parent / payload["calibration"])
    if calibration.get("schema") != "calibrated_fixed_context_camera_v1":
        raise ValueError("unexpected camera calibration schema")
    return rgb, times, calibration, paths


def run_vggt_phase(input_root: Path, records: list[dict], output: Path, torch) -> dict:
    sys.path.insert(0, str(VGGT_ROOT))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = VGGT.from_pretrained(str(VGGT_CHECKPOINT)).to("cuda:0").eval().requires_grad_(False)
    load_seconds = time.perf_counter() - load_started
    rows = []
    raw_root = output / "raw_vggt"
    raw_root.mkdir(parents=True)
    for index, record in enumerate(records, start=1):
        rgb, _, _, paths = load_case(input_root, record)
        started = time.perf_counter()
        images = load_and_preprocess_images([str(path) for path in paths]).to("cuda:0")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction["pose_enc"], images.shape[-2:])
        arrays = {
            "depth": prediction["depth"].detach().float().cpu().numpy().squeeze(0),
            "depth_conf": prediction["depth_conf"].detach().float().cpu().numpy().squeeze(0),
            "world_points": prediction["world_points"].detach().float().cpu().numpy().squeeze(0),
            "world_points_conf": prediction["world_points_conf"].detach().float().cpu().numpy().squeeze(0),
            "extrinsic": extrinsic.detach().float().cpu().numpy().squeeze(0),
            "intrinsic": intrinsic.detach().float().cpu().numpy().squeeze(0),
            "processed_rgb": np.clip(
                prediction["images"].detach().float().cpu().numpy().squeeze(0).transpose(0, 2, 3, 1) * 255.0,
                0,
                255,
            ).astype(np.uint8),
        }
        if arrays["depth"].shape[0] != OBSERVED_FRAMES or arrays["world_points"].shape[-1] != 3:
            raise ValueError(f"invalid VGGT output shapes for {record['case_id']}")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"nonfinite VGGT output for {record['case_id']}")
        path = raw_root / f"{record['case_id']}.npz"
        np.savez_compressed(
            path,
            **{
                key: value.astype(np.float16) if key in {
                    "depth", "depth_conf", "world_points", "world_points_conf"
                } else value
                for key, value in arrays.items()
            },
        )
        elapsed = time.perf_counter() - started
        rows.append({
            "case_id": record["case_id"],
            "seconds": elapsed,
            "shape": {key: list(value.shape) for key, value in arrays.items()},
            "npz": str(path.relative_to(output)),
            "sha256": sha256_file(path),
            "source_rgb_sha256": pixel_sha256(rgb),
        })
        del images, prediction, extrinsic, intrinsic, arrays
        torch.cuda.empty_cache()
        print(f"VGGT_OK {index:02d}/36 {record['case_id']} {elapsed:.3f}s", flush=True)
    phase = {
        "status": "EXECUTED",
        "model_load_seconds": load_seconds,
        "case_count": len(rows),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "records": rows,
    }
    del model
    torch.cuda.empty_cache()
    return phase


def run_sam2_phase(input_root: Path, records: list[dict], output: Path, torch) -> dict:
    sys.path.insert(0, str(SAM2_SOURCE))
    from phys_state_video.mask_tracking import SAM2VideoMaskTracker

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tracker = SAM2VideoMaskTracker(
        device="cuda:0",
        model_id=None,
        model_cfg=str(SAM2_CONFIG),
        checkpoint_path=SAM2_CHECKPOINT,
    )
    rows = []
    mask_root = output / "sam2_masks"
    mask_root.mkdir(parents=True)
    for index, record in enumerate(records, start=1):
        rgb, _, _, _ = load_case(input_root, record)
        frames = rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        height, width = rgb.shape[1:3]
        started = time.perf_counter()
        prompt, prompt_report = rgb_motion_circle_prompt(rgb)
        masks = tracker.track_from_boxes(
            frames,
            prompt_frame_idx=7,
            boxes_xyxy=prompt[None],
        )[:, 0].astype(bool)
        areas = masks.sum(axis=(1, 2))
        if np.any(areas < 20) or np.any(areas > 0.12 * height * width):
            raise ValueError(f"invalid SAM2 mask areas for {record['case_id']}: {areas.tolist()}")
        path = mask_root / f"{record['case_id']}.npz"
        np.savez_compressed(path, masks=masks, prompt_box_xyxy=prompt)
        elapsed = time.perf_counter() - started
        rows.append({
            "case_id": record["case_id"],
            "seconds": elapsed,
            "npz": str(path.relative_to(output)),
            "sha256": sha256_file(path),
            "prompt_frame": 7,
            "prompt_mode": "RGB_motion_circle_then_SAM2_bidirectional",
            "prompt_box_xyxy": prompt.tolist(),
            "prompt_detector": prompt_report,
            "mask_area_pixels": areas.astype(int).tolist(),
        })
        print(f"SAM2_OK {index:02d}/36 {record['case_id']} {elapsed:.3f}s", flush=True)
    phase = {
        "status": "EXECUTED",
        "case_count": len(rows),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "records": rows,
    }
    del tracker
    torch.cuda.empty_cache()
    return phase


def make_preview(rgb: np.ndarray, masks: np.ndarray, centers: np.ndarray, calibration: dict, path: Path) -> None:
    intrinsic = np.asarray(calibration["intrinsic_K"], dtype=np.float64)
    world_to_camera = np.asarray(calibration["world_to_camera_3x4"], dtype=np.float64)
    camera_centers = centers @ world_to_camera[:, :3].T + world_to_camera[:, 3]
    pixels = camera_centers @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    height, width = rgb.shape[1:3]
    canvas = Image.new("RGB", (4 * width, 2 * (height + 22)), "white")
    draw = ImageDraw.Draw(canvas)
    for frame_index in range(OBSERVED_FRAMES):
        image = rgb[frame_index].copy()
        image[masks[frame_index]] = (
            0.55 * image[masks[frame_index]] + 0.45 * np.asarray([20, 240, 130])
        ).astype(np.uint8)
        panel = Image.fromarray(image)
        overlay = ImageDraw.Draw(panel)
        x, y = pixels[frame_index]
        overlay.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 210, 40))
        col, row = frame_index % 4, frame_index // 4
        canvas.paste(panel, (col * width, row * (height + 22) + 22))
        draw.text((col * width + 5, row * (height + 22) + 4), f"RGB{frame_index}", fill="black")
    canvas.save(path)


def combine_estimates(input_root: Path, records: list[dict], output: Path) -> dict:
    rows = []
    estimate_root = output / "estimates"
    estimate_root.mkdir(parents=True)
    for index, record in enumerate(records, start=1):
        rgb, times, calibration, _ = load_case(input_root, record)
        key = record["case_id"]
        with np.load(output / "raw_vggt" / f"{key}.npz", allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        with np.load(output / "sam2_masks" / f"{key}.npz", allow_pickle=False) as archive:
            masks_source = archive["masks"].astype(bool)
            prompt = archive["prompt_box_xyxy"].astype(np.float32)
        transform = crop_transform(tuple(rgb.shape[1:3]))
        masks_processed = np.stack([resize_crop_mask(mask, transform) for mask in masks_source])
        intrinsic_source = np.asarray(calibration["intrinsic_K"], dtype=np.float64)
        intrinsic_processed = transform["affine"] @ intrinsic_source
        world_to_camera = np.asarray(calibration["world_to_camera_3x4"], dtype=np.float64)
        depth = raw["depth"].astype(np.float32)
        confidence = raw["depth_conf"].astype(np.float32)
        centers, depth_report = estimate_metric_centers(
            depth,
            masks_processed,
            intrinsic_processed,
            world_to_camera,
            radius_m=float(calibration["known_object_prior"]["radius_m"]),
        )
        velocity, fit_report = robust_terminal_velocity(centers, times, degree=2)
        points, colors, point_confidence, cloud_report = static_point_cloud(
            depth,
            confidence,
            raw["processed_rgb"],
            masks_processed,
            intrinsic_processed,
            world_to_camera,
            float(depth_report["metric_scale"]),
        )
        case_root = estimate_root / key
        case_root.mkdir()
        np.savez_compressed(
            case_root / "estimate.npz",
            estimated_centers=centers.astype(np.float32),
            estimated_p7=centers[-1].astype(np.float32),
            estimated_v7=velocity.astype(np.float32),
            masks_source=masks_source,
            masks_processed=masks_processed,
            static_points=points,
            static_colors=colors,
            static_confidence=point_confidence,
            frame_times=times.astype(np.float32),
            intrinsic_processed=intrinsic_processed.astype(np.float32),
            world_to_camera=world_to_camera.astype(np.float32),
        )
        report = {
            "schema": "context_rgb8_metric_state_pointmap_v1",
            "status": "EXECUTED",
            "case_id": key,
            "family": record["family"],
            "estimated_p7_m": centers[-1].tolist(),
            "estimated_v7_mps": velocity.tolist(),
            "mask_prompt_xyxy": prompt.tolist(),
            "depth_scale": depth_report,
            "velocity_fit": fit_report,
            "static_cloud": cloud_report,
            "future_rgb_used": False,
            "gt_state_used": False,
            "blueprint_used": False,
            "utonia_used": False,
            "predictor_used": False,
        }
        dump_json(case_root / "report.json", report)
        make_preview(rgb, masks_source, centers, calibration, case_root / "preview.png")
        rows.append({
            "case_id": key,
            "family": record["family"],
            "estimate": str((case_root / "estimate.npz").relative_to(output)),
            "report": str((case_root / "report.json").relative_to(output)),
            "preview": str((case_root / "preview.png").relative_to(output)),
            "p7": centers[-1].tolist(),
            "v7": velocity.tolist(),
        })
        print(f"ESTIMATE_OK {index:02d}/36 {key}", flush=True)
    return {"status": "EXECUTED", "case_count": len(rows), "records": rows}


def run(args: argparse.Namespace) -> dict:
    input_root = args.input_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    physical_index, inventory = validate_gpu(args.gpu_uuid)
    manifest_path = input_root / "manifest.json"
    manifest = load_json(manifest_path)
    validate_input_tree(input_root, manifest)
    if not VGGT_CHECKPOINT.joinpath("model.safetensors").is_file():
        raise FileNotFoundError(VGGT_CHECKPOINT / "model.safetensors")
    if not SAM2_CHECKPOINT.is_file() or not SAM2_CONFIG.is_file():
        raise FileNotFoundError("SAM2 checkpoint/config missing")
    output.mkdir(parents=True)
    dump_json(
        output / "admission.json",
        {
            "status": "PASS",
            "input_root": str(input_root),
            "input_manifest_sha256": sha256_file(manifest_path),
            "case_count": 36,
            "physical_gpu_index": physical_index,
            "gpu_uuid": args.gpu_uuid,
            "nvidia_inventory": inventory.splitlines(),
            "future_inputs_present": False,
            "gt_state_inputs_present": False,
            "blueprint_inputs_present": False,
        },
    )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    started = time.perf_counter()
    records = manifest["records"]
    vggt_phase = run_vggt_phase(input_root, records, output, torch)
    sam2_phase = run_sam2_phase(input_root, records, output, torch)
    estimates = combine_estimates(input_root, records, output)
    report = {
        "schema": "context_rgb8_vision_inference_report_v1",
        "status": "EXECUTED",
        "case_count": len(records),
        "gpu_uuid": args.gpu_uuid,
        "physical_gpu_index": physical_index,
        "elapsed_seconds": time.perf_counter() - started,
        "vggt": vggt_phase,
        "sam2": sam2_phase,
        "estimates": estimates,
        "models": {
            "vggt": {
                "source": str(VGGT_ROOT),
                "source_git_commit": subprocess.run(
                    ["git", "-C", str(VGGT_ROOT), "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=True,
                ).stdout.strip(),
                "checkpoint": str(VGGT_CHECKPOINT / "model.safetensors"),
                "checkpoint_bytes": (VGGT_CHECKPOINT / "model.safetensors").stat().st_size,
                "direct_official_import": True,
                "utonia_used": False,
            },
            "sam2": {
                "source": str(SAM2_SOURCE),
                "checkpoint": str(SAM2_CHECKPOINT),
                "checkpoint_bytes": SAM2_CHECKPOINT.stat().st_size,
                "prompt": "RGB-only temporal motion proxy at RGB7",
                "tracking": "bidirectional RGB7->RGB0 SAM2 video propagation",
            },
        },
        "future_rgb_used": False,
        "gt_state_used": False,
        "blueprint_used": False,
        "dit_loaded": False,
        "utonia_loaded": False,
        "dynamics_predictor_loaded": False,
    }
    dump_json(output / "inference_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "case_count": result["case_count"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))
