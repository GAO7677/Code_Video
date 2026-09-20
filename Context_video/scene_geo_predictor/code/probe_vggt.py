"""One frozen VGGT RGB8 probe; raw geometry is neither static nor metric."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import csv
import io
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from prepare_context import load_model_input

SOURCE = Path("/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/data")
CHECKPOINT = Path("/data/gaoya/ckpt/facebook-VGGT-1B")
FORBIDDEN_UUID = "GPU-4a8abb69-6a43-4b79-5713-31979b8d6d75"
FIELDS = ("world_points", "world_points_conf", "depth", "depth_conf", "extrinsic", "intrinsic", "images")


def validate_gpu(gpu_uuid, visible, inventory):
    pattern = r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    if re.fullmatch(pattern, gpu_uuid) is None:
        raise ValueError("A canonical full GPU UUID is required, not an index or prefix")
    if gpu_uuid == FORBIDDEN_UUID or visible != gpu_uuid:
        raise ValueError("Forbidden GPU or mismatched CUDA_VISIBLE_DEVICES")
    rows = list(csv.reader(io.StringIO(inventory)))
    matches = [int(row[0].strip()) for row in rows if len(row) == 2 and row[1].strip() == gpu_uuid]
    if len(matches) != 1 or matches[0] == 4:
        raise ValueError("GPU must resolve uniquely to a physical index other than 4")
    return matches[0]


def validate_predictions(pred):
    missing = set(FIELDS) - set(pred)
    if missing:
        raise ValueError(f"Missing VGGT outputs: {sorted(missing)}")
    points = pred["world_points"]
    if points.ndim != 4 or points.shape[0] != 8 or points.shape[-1] != 3:
        raise ValueError("Expected exactly eight point maps")
    h, w = points.shape[1:3]
    shapes = dict(world_points_conf=(8,h,w), depth=(8,h,w,1), depth_conf=(8,h,w),
                  extrinsic=(8,3,4), intrinsic=(8,3,3), images=(8,3,h,w))
    for name, shape in shapes.items():
        if pred[name].shape != shape:
            raise ValueError(f"Bad {name} shape: {pred[name].shape}, expected {shape}")
    for name in FIELDS:
        if not np.isfinite(pred[name]).all():
            raise ValueError(f"Nonfinite {name}")
    if not (pred["depth"] > 0).all():
        raise ValueError("Nonpositive predicted depth")


def preview(pred, path):
    h, w = pred["world_points"].shape[1:3]
    canvas = Image.new("RGB", (3*w, 2*(h+28)), "white")
    draw = ImageDraw.Draw(canvas)
    depth = pred["depth"][...,0]
    confidence = pred["world_points_conf"]
    dmin, dmax = np.quantile(depth, [.02,.98])
    cmin, cmax = np.quantile(confidence, [.02,.98])
    for row, frame in enumerate((0,7)):
        rgb = np.clip(pred["images"][frame].transpose(1,2,0)*255,0,255).astype(np.uint8)
        d = np.clip((depth[frame]-dmin)/max(dmax-dmin,1e-8),0,1)
        c = np.clip((confidence[frame]-cmin)/max(cmax-cmin,1e-8),0,1)
        panels = [Image.fromarray(rgb), Image.fromarray((d*255).astype(np.uint8)).convert("RGB"),
                  Image.fromarray((c*255).astype(np.uint8)).convert("RGB")]
        titles = [f"RGB{frame} / model input", "Depth / unscaled", "Point confidence / not probability"]
        for col, (panel, title) in enumerate(zip(panels,titles)):
            xy = (col*w, row*(h+28))
            draw.text((xy[0]+6,xy[1]+6), title, fill="black")
            canvas.paste(panel,(xy[0],xy[1]+28))
    canvas.save(path)
    return dict(depth_display_quantiles=[float(dmin),float(dmax)],
                confidence_display_quantiles=[float(cmin),float(cmax)],
                shared_display_scale_across_all_eight_frames=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=10).stdout
    physical_gpu_index = validate_gpu(args.gpu_uuid, os.environ.get("CUDA_VISIBLE_DEVICES"), inventory)
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    load_model_input(args.input)
    payload = json.loads(args.input.read_text())
    frames = [args.input.parent / name for name in payload["frames"]]
    import torch
    sys.path.insert(0, str(SOURCE))
    from utonia_dense_pointmap import VGGT, run_vggt
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible CUDA device")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    print("Loading local frozen VGGT; RGB0-7 only", flush=True)
    start = time.monotonic()
    model = VGGT.from_pretrained(str(CHECKPOINT)).to("cuda:0").eval().requires_grad_(False)
    loaded = time.monotonic()
    pred = run_vggt(model, frames, torch.device("cuda:0"))
    inferred = time.monotonic()
    validate_predictions(pred)
    load_model_input(args.input)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "raw_vggt.npz", **{k:pred[k] for k in FIELDS})
    display = preview(pred, args.output / "preview.png")
    weight = CHECKPOINT / "model.safetensors"
    stat = weight.stat()
    report = dict(status="raw_rgb8_probe_complete", input_json=str(args.input),
                  input_pixels_sha256=payload["pixels_sha256"], observed_indices=list(range(8)),
                  model_source=str(SOURCE/"utonia_dense_pointmap.py"),
                  source_sha256=hashlib.sha256((SOURCE/"utonia_dense_pointmap.py").read_bytes()).hexdigest(),
                  checkpoint=str(weight), checkpoint_bytes=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns,
                  gpu_uuid=args.gpu_uuid, physical_gpu_index=physical_gpu_index, visible_device_count=1, load_seconds=loaded-start,
                  inference_seconds=inferred-loaded, elapsed_seconds=time.monotonic()-start,
                  peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  shapes={k:list(pred[k].shape) for k in FIELDS},
                  depth_quantiles=np.quantile(pred["depth"],[0,.01,.5,.99,1]).tolist(),
                  coordinate_frame="VGGT predicted coordinates; not simulator world",
                  depth_units="unscaled model depth; not validated meters",
                  static_dynamic_separation=False, metric_alignment=False,
                  utonia_loaded=False, dit_loaded=False, predictor_loaded=False,
                  static_scene_gt_used=False, gt_mask_used=False, training_ready=False,
                  display=display)
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()

