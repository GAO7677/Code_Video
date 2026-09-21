"""CPU refinement of frozen VGGT/SAM2 outputs for a known spherical object."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from context_rgb_pybullet_common import (
    crop_transform,
    dump_json,
    estimate_metric_centers,
    metric_centers_from_sphere_circles,
    regularize_sphere_masks,
    resize_crop_mask,
    robust_terminal_velocity,
    sha256_file,
    static_point_cloud,
)
from infer_context_rgb_pybullet import load_case, make_preview, validate_input_tree


def refine_case(input_root: Path, source: Path, output: Path, record: dict) -> dict:
    key = str(record["case_id"])
    rgb, times, calibration, _ = load_case(input_root, record)
    with np.load(source / "raw_vggt" / f"{key}.npz", allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    with np.load(source / "sam2_masks" / f"{key}.npz", allow_pickle=False) as archive:
        sam2_masks = archive["masks"].astype(bool)
        prompt = archive["prompt_box_xyxy"].astype(np.float32)

    circles, sphere_masks, circle_report = regularize_sphere_masks(rgb, sam2_masks)
    intrinsic_source = np.asarray(calibration["intrinsic_K"], dtype=np.float64)
    world_to_camera = np.asarray(calibration["world_to_camera_3x4"], dtype=np.float64)
    radius_m = float(calibration["known_object_prior"]["radius_m"])
    centers, center_report = metric_centers_from_sphere_circles(
        circles,
        intrinsic_source,
        world_to_camera,
        radius_m=radius_m,
        horizontal_support_regularization=True,
    )
    velocity, fit_report = robust_terminal_velocity(centers, times, degree=2)

    transform = crop_transform(tuple(rgb.shape[1:3]))
    masks_processed = np.stack([resize_crop_mask(mask, transform) for mask in sphere_masks])
    intrinsic_processed = transform["affine"] @ intrinsic_source
    depth = raw["depth"].astype(np.float32)
    confidence = raw["depth_conf"].astype(np.float32)
    _, depth_report = estimate_metric_centers(
        depth,
        masks_processed,
        intrinsic_processed,
        world_to_camera,
        radius_m=radius_m,
    )
    points, colors, point_confidence, cloud_report = static_point_cloud(
        depth,
        confidence,
        raw["processed_rgb"],
        masks_processed,
        intrinsic_processed,
        world_to_camera,
        float(depth_report["metric_scale"]),
    )

    case_root = output / "estimates" / key
    case_root.mkdir(parents=True)
    estimate_path = case_root / "estimate.npz"
    np.savez_compressed(
        estimate_path,
        estimated_centers=centers.astype(np.float32),
        estimated_p7=centers[-1].astype(np.float32),
        estimated_v7=velocity.astype(np.float32),
        sphere_circles_source=circles.astype(np.float32),
        sam2_masks_source=sam2_masks,
        masks_source=sphere_masks,
        masks_processed=masks_processed,
        static_points=points,
        static_colors=colors,
        static_confidence=point_confidence,
        frame_times=times.astype(np.float32),
        intrinsic_processed=intrinsic_processed.astype(np.float32),
        world_to_camera=world_to_camera.astype(np.float32),
    )
    report = {
        "schema": "context_rgb8_metric_state_pointmap_v2",
        "status": "EXECUTED",
        "case_id": key,
        "family": record["family"],
        "estimated_p7_m": centers[-1].tolist(),
        "estimated_v7_mps": velocity.tolist(),
        "mask_prompt_xyxy": prompt.tolist(),
        "sphere_mask_regularization": circle_report,
        "metric_center": center_report,
        "vggt_depth_scale": depth_report,
        "velocity_fit": fit_report,
        "static_cloud": cloud_report,
        "source_vision_output": str(source),
        "source_raw_vggt_sha256": sha256_file(source / "raw_vggt" / f"{key}.npz"),
        "source_sam2_sha256": sha256_file(source / "sam2_masks" / f"{key}.npz"),
        "future_rgb_used": False,
        "gt_state_used": False,
        "blueprint_used": False,
        "utonia_used": False,
        "predictor_used": False,
    }
    dump_json(case_root / "report.json", report)
    make_preview(rgb, sphere_masks, centers, calibration, case_root / "preview.png")
    return {
        "case_id": key,
        "family": record["family"],
        "estimate": str(estimate_path.relative_to(output)),
        "report": str((case_root / "report.json").relative_to(output)),
        "preview": str((case_root / "preview.png").relative_to(output)),
        "p7": centers[-1].tolist(),
        "v7": velocity.tolist(),
    }


def run(args: argparse.Namespace) -> dict:
    input_root = args.input_root.resolve()
    source = args.source_vision.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    validate_input_tree(input_root, manifest)
    source_report = json.loads((source / "inference_report.json").read_text(encoding="utf-8"))
    if source_report.get("status") != "EXECUTED" or source_report.get("case_count") != 36:
        raise ValueError("source GPU inference is incomplete")
    started = time.perf_counter()
    rows = []
    for index, record in enumerate(manifest["records"], start=1):
        row = refine_case(input_root, source, output, record)
        rows.append(row)
        print(f"REFINE_OK {index:02d}/36 {row['case_id']}", flush=True)
    report = {
        "schema": "context_rgb8_vision_refinement_report_v2",
        "status": "EXECUTED",
        "case_count": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "source_gpu_inference": str(source),
        "gpu_used": False,
        "cpu_threads": 2,
        "estimates": {"status": "EXECUTED", "case_count": len(rows), "records": rows},
        "sphere_prior": "known fixed radius 0.11 m and spherical silhouette",
        "support_prior": "one horizontal support height during RGB0-RGB7, estimated from silhouette centers",
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
    parser.add_argument("--source-vision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "case_count": result["case_count"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))
