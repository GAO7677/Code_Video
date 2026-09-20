"""Observed-only SAM2 masks prompted by permitted context geometry, not GT masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from prepare_context import load_model_input, pixel_hash
from probe_vggt import validate_gpu

CHECKPOINT = Path("/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt")
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GEOMETRY_KEYS = {
    "positions_world", "size_m", "camera_K", "camera_world_to_view",
    "mask_boxes_xyxy", "center_pixels", "frame_times",
}
MIN_BOX_CONTAINMENT = 0.95


def validate_geometry(geometry, frame_times):
    """Reject undeclared inputs, future frames, and incompatible camera conventions."""
    if set(geometry) != GEOMETRY_KEYS:
        raise ValueError("Geometry must contain exactly the seven permitted context fields")
    positions = geometry["positions_world"]
    if positions.ndim != 3 or positions.shape[0] != 8 or positions.shape[2] != 3 or positions.shape[1] < 1:
        raise ValueError("positions_world must have shape [8,K,3], K >= 1")
    k = positions.shape[1]
    shapes = dict(size_m=(k, 3), camera_K=(3, 3), camera_world_to_view=(3, 4),
                  mask_boxes_xyxy=(8, k, 4), center_pixels=(8, k, 2), frame_times=(8,))
    for key in GEOMETRY_KEYS:
        value = geometry[key]
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"Nonfinite or nonnumeric geometry: {key}")
        if key in shapes and value.shape != shapes[key]:
            raise ValueError(f"Wrong {key} shape: {value.shape}, expected {shapes[key]}")
    if not (geometry["size_m"] > 0).all():
        raise ValueError("Object sizes must be positive")
    if not np.allclose(geometry["frame_times"], frame_times, rtol=0, atol=1e-7):
        raise ValueError("Geometry times do not match the eight RGB observation times")
    intrinsic = geometry["camera_K"]
    rotation = geometry["camera_world_to_view"][:, :3]
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], [0, 0, 1]):
        raise ValueError("Expected positive-focal OpenCV camera intrinsics")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("Expected a proper world-to-view rotation")
    boxes = geometry["mask_boxes_xyxy"]
    if not ((boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])).all():
        raise ValueError("Prompt boxes must have positive width and height")
    centers_camera = positions @ rotation.T + geometry["camera_world_to_view"][:, 3]
    if not (centers_camera[..., 2] > 0).all():
        raise ValueError("Object centers must be in front of the OpenCV camera")
    projected = centers_camera @ intrinsic.T
    projected = projected[..., :2] / projected[..., 2:]
    if not np.allclose(projected, geometry["center_pixels"], rtol=0, atol=1e-3):
        raise ValueError("Center pixels disagree with context positions and camera")


def load_geometry(path, frame_times):
    with np.load(path, allow_pickle=False) as archive:
        geometry = {key: archive[key] for key in archive.files}
    validate_geometry(geometry, frame_times)
    return geometry


def make_prompt(box, center, image_hw):
    """Clip only prompts, never a predicted mask; skip offscreen positive points."""
    h, w = image_hw
    box = np.asarray(box, dtype=np.float32).copy()
    center = np.asarray(center, dtype=np.float32).copy()
    if box.shape != (4,) or center.shape != (2,) or not np.isfinite(box).all() or not np.isfinite(center).all():
        raise ValueError("Invalid prompt arrays")
    if h < 2 or w < 2 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Invalid image extent or prompt box")
    if not (0 <= center[0] < w and 0 <= center[1] < h):
        return None
    box[[0, 2]] = np.clip(box[[0, 2]], 0, w - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, h - 1)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    if not (box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]):
        return None
    return box, center[None], np.ones(1, dtype=np.int32)


def select_mask(candidates, scores, box, center):
    """Choose a SAM mask using prompt agreement, not silhouette GT or clipping."""
    candidates = np.asarray(candidates)
    scores = np.asarray(scores)
    if candidates.ndim != 3 or candidates.shape[0] == 0 or scores.shape != (candidates.shape[0],):
        raise ValueError("Expected candidate masks [N,H,W] and scores [N]")
    if not np.isfinite(candidates).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite SAM2 output")
    if candidates.dtype != np.bool_ and not np.isin(candidates, [0, 1]).all():
        raise ValueError("Expected binary SAM2 masks, not unthresholded logits")
    masks = candidates.astype(bool)
    h, w = masks.shape[1:]
    prompt = make_prompt(box, center, (h, w))
    if prompt is None:
        raise ValueError("Selection requires an on-image prompt")
    box = prompt[0]
    x0, y0 = np.floor(box[:2]).astype(int)
    x1, y1 = np.ceil(box[2:]).astype(int)
    cx, cy = np.clip(np.rint(center).astype(int), [0, 0], [w - 1, h - 1])
    area = masks.sum(axis=(1, 2))
    inside = masks[:, y0:y1 + 1, x0:x1 + 1].sum(axis=(1, 2))
    containment = inside / np.maximum(area, 1)
    center_inside = masks[:, cy, cx]
    accepted = (area > 0) & center_inside & (containment >= MIN_BOX_CONTAINMENT)
    # Agreement gates precede model score; scores are not calibrated probabilities.
    ranking = [(True, float(scores[i]), float(containment[i]), -i) if accepted[i] else
               (False, bool(area[i]), bool(center_inside[i]), float(containment[i]), float(scores[i]), -i)
               for i in range(len(masks))]
    chosen = max(range(len(masks)), key=ranking.__getitem__)
    rows = [dict(candidate=i, model_score=float(scores[i]), pixels=int(area[i]),
                 box_containment=float(containment[i]), center_inside=bool(center_inside[i]),
                 accepted=bool(accepted[i])) for i in range(len(masks))]
    quality = dict(rows[chosen], candidate_count=len(masks), candidates=rows,
                   prompt_box_area_pixels=int((x1 - x0 + 1) * (y1 - y0 + 1)),
                   box_fill_fraction=float(inside[chosen] / max((x1 - x0 + 1) * (y1 - y0 + 1), 1)))
    return masks[chosen].copy(), quality


def validate_masks(per_object, union_dynamic, accepted):
    if per_object.ndim != 4 or per_object.shape[0] != 8 or per_object.dtype != np.bool_:
        raise ValueError("Expected bool per_object [8,K,H,W]")
    if per_object.shape[1] < 1 or min(per_object.shape[2:]) < 2:
        raise ValueError("Empty object or image dimensions")
    if union_dynamic.shape != (8, *per_object.shape[2:]) or union_dynamic.dtype != np.bool_:
        raise ValueError("Expected bool union_dynamic [8,H,W]")
    if not np.array_equal(union_dynamic, per_object.any(axis=1)):
        raise ValueError("Union mask does not match per-object masks")
    if accepted.shape != per_object.shape[:2] or accepted.dtype != np.bool_:
        raise ValueError("Expected bool accepted [8,K]")
    if np.any(accepted & ~per_object.any(axis=(2, 3))):
        raise ValueError("Empty masks cannot be accepted")


def preview(rgb, masks, geometry, quality, path):
    h, w = rgb.shape[1:3]
    canvas = Image.new("RGB", (2 * w, 4 * (h + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    colors = np.asarray([[55, 220, 120], [235, 90, 90], [100, 150, 255]], dtype=np.float32)
    k = masks.shape[1]
    for t in range(8):
        pixels = rgb[t].copy()
        for j, mask in enumerate(masks[t]):
            pixels[mask] = (pixels[mask] * 0.55 + colors[j % len(colors)] * 0.45).astype(np.uint8)
        im = Image.fromarray(pixels)
        overlay = ImageDraw.Draw(im)
        for j in range(k):
            prompt = make_prompt(geometry["mask_boxes_xyxy"][t, j], geometry["center_pixels"][t, j], (h, w))
            if prompt is None:
                continue
            box, point, _ = prompt
            overlay.rectangle(box.tolist(), outline=(255, 215, 50), width=1)
            x, y = point[0]
            overlay.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 215, 50))
        x, y = (t % 2) * w, (t // 2) * (h + 24)
        accepted_count = sum(row["accepted"] for row in quality if row["frame"] == t)
        draw.text((x + 6, y + 5), f"RGB{t}: SAM2 candidates / prompt gates {accepted_count}/{k}", fill="black")
        canvas.paste(im, (x, y + 24))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=30).stdout
    physical_index = validate_gpu(args.gpu_uuid, os.environ.get("CUDA_VISIBLE_DEVICES"), inventory)
    args.input, args.geometry, args.output = args.input.resolve(), args.geometry.resolve(), args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    rgb, frame_times = load_model_input(args.input)
    geometry = load_geometry(args.geometry, frame_times)
    geometry_hash = hashlib.sha256(args.geometry.read_bytes()).hexdigest()
    h, w = rgb.shape[1:3]
    k = geometry["positions_world"].shape[1]
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one allowed visible CUDA device")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model = build_sam2(CONFIG, str(CHECKPOINT), device="cuda:0").eval().requires_grad_(False)
    predictor = SAM2ImagePredictor(model)
    loaded = time.monotonic()
    per_object = np.zeros((8, k, h, w), dtype=bool)
    accepted = np.zeros((8, k), dtype=bool)
    selected_scores = np.zeros((8, k), dtype=np.float32)
    containment = np.zeros((8, k), dtype=np.float32)
    quality = []
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for t in range(8):
            predictor.set_image(rgb[t])
            for j in range(k):
                prompt = make_prompt(geometry["mask_boxes_xyxy"][t, j], geometry["center_pixels"][t, j], (h, w))
                if prompt is None:
                    quality.append(dict(frame=t, object_index=j, accepted=False, pixels=0,
                                        model_score=0., box_containment=0., center_inside=False,
                                        reason="no_visible_center_prompt; mask is unknown, not verified empty"))
                    continue
                box, points, labels = prompt
                masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels,
                                                     box=box, multimask_output=True)
                mask, record = select_mask(masks, scores, box, points[0])
                if mask.shape != (h, w):
                    raise ValueError("SAM2 changed the source image resolution")
                per_object[t, j] = mask
                accepted[t, j] = record["accepted"]
                selected_scores[t, j] = record["model_score"]
                containment[t, j] = record["box_containment"]
                quality.append(dict(frame=t, object_index=j, **record))
            print(f"Observed RGB{t}: prompt gates {int(accepted[t].sum())}/{k}", flush=True)
    torch.cuda.synchronize()
    inferred = time.monotonic()
    union_dynamic = per_object.any(axis=1)
    validate_masks(per_object, union_dynamic, accepted)
    rgb_verified, _ = load_model_input(args.input)
    if pixel_hash(rgb_verified) != pixel_hash(rgb) or hashlib.sha256(args.geometry.read_bytes()).hexdigest() != geometry_hash:
        raise RuntimeError("Inputs changed during inference")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "observed_masks.npz", per_object=per_object,
                        union_dynamic=union_dynamic, accepted=accepted,
                        model_scores=selected_scores, box_containment=containment,
                        area_pixels=per_object.sum(axis=(2, 3)), frame_times=frame_times)
    preview(rgb, per_object, geometry, quality, args.output / "preview.png")
    stat = CHECKPOINT.stat()
    report = dict(schema="observed_sam2_rgb8_v1", status="candidate_masks_complete",
                  input_json=str(args.input), geometry_npz=str(args.geometry),
                  input_pixels_sha256=pixel_hash(rgb), geometry_sha256=geometry_hash,
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checkpoint=str(CHECKPOINT), checkpoint_bytes=stat.st_size,
                  checkpoint_mtime_ns=stat.st_mtime_ns, config=CONFIG,
                  gpu_uuid=args.gpu_uuid, physical_gpu_index=physical_index,
                  load_seconds=loaded - start, inference_seconds=inferred - loaded,
                  peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  observed_indices=list(range(8)), per_object_shape=list(per_object.shape),
                  union_dynamic_shape=list(union_dynamic.shape),
                  all_prompt_gates_passed=bool(accepted.all()), quality=quality,
                  mask_source="SAM2 per-frame RGB with context geometry box and positive center prompt",
                  minimum_box_containment=MIN_BOX_CONTAINMENT,
                  temporal_propagation=False, predicted_mask_clipping=False,
                  static_scene_gt_used=False, gt_mask_used=False, future_frames_used=False,
                  permitted_context_geometry_used=True, needs_review=True, training_ready=False,
                  limitations=["Prompt agreement is not segmentation accuracy or calibrated confidence.",
                               "A projected center may be occluded; accepted masks still need visual review.",
                               "Rejected/empty masks mean unknown, not a verified static region.",
                               "No segmentation-GT IoU was computed; downstream must check accepted and review status."])
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
