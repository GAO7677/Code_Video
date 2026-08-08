#!/usr/bin/env python3
"""Rerun selected PhysicIQ67 samples and capture provisional S039 Top10 heatmaps."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for path in (ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (  # noqa: E402
    _run_pipe_once,
    build_wan_ti2v_pipeline,
    ensure_firstframe_image,
)
from AAA_my_test.build_legacy_ti2v_firstlatent_physiciq67_visual_samples import (  # noqa: E402
    MANIFEST_PATH,
    VISUAL_ROOT,
)
from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import (  # noqa: E402
    CASES,
    REGION_CACHE_ROOT,
    read_payload,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_pck_worker import (  # noqa: E402
    build_args,
    object_queries,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_top10_heatmaps_worker import (  # noqa: E402
    SelectedObjectHeatmapCapture,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def process(pipe, entries: list[dict], case, seed: int, overwrite: bool) -> None:
    output = VISUAL_ROOT / "heatmaps" / case.key / f"seed_{seed:05d}"
    if (output / "complete.json").is_file() and not overwrite:
        print(f"skip visual heatmap {case.key} seed={seed}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    cache = load_region_cache(REGION_CACHE_ROOT, case.key)
    points, query_regions = object_queries(cache)
    region_slices = [point_slice for _, point_slice in query_regions]
    payload = read_payload(case)
    payload, firstframe = ensure_firstframe_image(case.json_path, payload)
    args = build_args(seed)
    image = Image.open(firstframe).convert("RGB").resize((1280, 704), Image.Resampling.LANCZOS)
    capture = SelectedObjectHeatmapCapture(
        pipe.pipe, entries, points, region_slices, (704, 1280)
    )
    capture.install()
    try:
        _run_pipe_once(
            pipe=pipe,
            prompt=str(payload["input_caption"]),
            negative_prompt=args.negative_prompt,
            seed=seed,
            input_image=image,
            height=704,
            width=1280,
            num_frames=49,
            cfg_scale=5.0,
            num_inference_steps=40,
            sample_shift=5.0,
            sample_solver="unipc",
            offload_model=False,
        )
    finally:
        capture.remove()
    if capture.maps is None or len(capture.filled) != len(entries):
        raise RuntimeError(f"captured {len(capture.filled)}/{len(entries)} selected heads")
    temporary = output / "attention_maps.npy.tmp"
    with temporary.open("wb") as handle:
        import numpy as np

        np.save(handle, capture.maps)
    temporary.replace(output / "attention_maps.npy")
    latent_time, height, width = capture.grid
    metadata = {
        "case": case.key,
        "seed": int(seed),
        "selection": "provisional PhysicIQ67 aggregate S039 Top10",
        "query_latent_index": 0,
        "query_pixel_frame": 0,
        "normalization": "per target-frame spatial softmax, then mean over object query points",
        "entries": entries,
        "regions": [region.region_name for region, _ in query_regions],
        "grid": [latent_time, height, width],
        "latent_anchor_pixel_frames": list(range(0, 49, 4)),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "complete.json").write_text(
        json.dumps(
            {"case": case.key, "seed": seed, "captured_heads": len(entries)}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    case_lookup = {case.key: case for case in CASES}
    tasks = [
        (case_lookup[str(row["case"])], int(row["seed"]))
        for row in manifest["samples"]
    ][args.worker_id :: args.num_workers]
    if not tasks:
        return
    pipe = build_wan_ti2v_pipeline(build_args(tasks[0][1]))
    for index, (case, seed) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] visual heatmap {case.key} seed={seed}", flush=True)
        try:
            process(pipe, entries, case, seed, bool(args.overwrite))
        except Exception:
            output = VISUAL_ROOT / "heatmaps" / case.key / f"seed_{seed:05d}"
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
