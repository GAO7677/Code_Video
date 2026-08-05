#!/usr/bin/env python3
"""Rerun deterministic legacy TI2V jobs and capture compact final-Top10 object heatmaps."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for path in (ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (
    _run_pipe_once,
    build_wan_ti2v_pipeline,
    ensure_firstframe_image,
)
from AAA_my_test.legacy_ti2v_firstlatent_common import (
    OUTPUT_ROOT,
    REGION_CACHE_ROOT,
    all_tasks,
    heatmap_dir,
    read_payload,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import build_args, object_queries
from AAA_my_test.sam2_region_query_utils import load_region_cache


class SelectedObjectHeatmapCapture:
    def __init__(self, pipe, entries: list[dict], query_points, region_slices, pixel_hw):
        self.pipe = pipe
        self.entries = entries
        self.query_points = torch.from_numpy(query_points).float()
        self.region_slices = region_slices
        self.pixel_height, self.pixel_width = pixel_hw
        self.by_step_layer: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for rank, entry in enumerate(entries):
            self.by_step_layer.setdefault((int(entry["step"]), int(entry["block"])), []).append(
                (rank, int(entry["head"]))
            )
        self.maps = None
        self.grid = None
        self.call_counts: dict[int, int] = {}
        self.current_step = -1
        self.current_grid = None
        self.active = False
        self.filled = set()
        self._handles: list[Any] = []
        self._original_model_fn = None

    def _step(self, timestep):
        value = float(timestep.detach().flatten()[0].float().cpu())
        schedule = self.pipe.scheduler.timesteps.detach().float().cpu()
        return int(torch.argmin((schedule - value).abs()).item())

    def _wrap(self, original):
        selected_steps = {step for step, _ in self.by_step_layer}

        def wrapped(*args, **kwargs):
            timestep, latents = kwargs.get("timestep"), kwargs.get("latents")
            if timestep is None or latents is None:
                return original(*args, **kwargs)
            step = self._step(timestep)
            call = self.call_counts.get(step, 0)
            self.call_counts[step] = call + 1
            patch = tuple(int(value) for value in kwargs["dit"].patch_size)
            self.current_grid = (
                int(latents.shape[2] // patch[0]),
                int(latents.shape[3] // patch[1]),
                int(latents.shape[4] // patch[2]),
            )
            self.current_step = step
            self.active = call == 0 and step in selected_steps
            try:
                return original(*args, **kwargs)
            finally:
                self.active = False

        return wrapped

    def install(self):
        self._original_model_fn = self.pipe.model_fn
        self.pipe.model_fn = self._wrap(self.pipe.model_fn)
        layers = {layer for _, layer in self.by_step_layer}
        models = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            models.append(self.pipe.dit2)
        for model in models:
            for layer in sorted(layers):
                self._handles.append(
                    model.blocks[layer].self_attn.attn.register_forward_pre_hook(self._hook(layer))
                )

    def remove(self):
        for handle in self._handles:
            handle.remove()
        if self._original_model_fn is not None:
            self.pipe.model_fn = self._original_model_fn

    def _hook(self, layer):
        def hook(module, inputs):
            selected = self.by_step_layer.get((self.current_step, layer), [])
            selected = [(rank, head) for rank, head in selected if rank not in self.filled]
            if not self.active or not selected or self.current_grid is None:
                return
            q, k = inputs[:2]
            time, height, width = self.current_grid
            heads = int(module.num_heads)
            dim = q.shape[-1] // heads
            q_frames = q[0].view(time, height * width, heads, dim)
            k_frames = k[0].view(time, height * width, heads, dim)
            if self.maps is None:
                self.maps = np.full(
                    (len(self.entries), len(self.region_slices), time, height, width),
                    np.nan,
                    dtype=np.float16,
                )
                self.grid = (time, height, width)
            points = self.query_points.to(q.device)
            x = torch.floor(points[:, 0] * width / self.pixel_width).long().clamp(0, width - 1)
            y = torch.floor(points[:, 1] * height / self.pixel_height).long().clamp(0, height - 1)
            source_indices = y * width + x
            scale = math.sqrt(dim)
            for rank, head in selected:
                for region_index, point_slice in enumerate(self.region_slices):
                    source = q_frames[0, source_indices[point_slice], head].float()
                    for target_time in range(time):
                        scores = torch.einsum(
                            "pd,sd->ps", source, k_frames[target_time, :, head].float()
                        ) / scale
                        probability = scores.softmax(dim=-1).mean(dim=0)
                        self.maps[rank, region_index, target_time] = (
                            probability.view(height, width).cpu().numpy().astype(np.float16)
                        )
                self.filled.add(rank)

        return hook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def process(pipe, entries, case, seed, overwrite):
    output = heatmap_dir(case.key, seed)
    if (output / "complete.json").is_file() and not overwrite:
        print(f"skip heatmap {case.key} seed={seed}", flush=True)
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
        raise RuntimeError(f"captured {len(capture.filled)}/{len(entries)} selected combinations")
    tmp = output / "attention_maps.npy.tmp"
    with tmp.open("wb") as handle:
        np.save(handle, capture.maps)
    tmp.replace(output / "attention_maps.npy")
    latent_time, height, width = capture.grid
    metadata = {
        "case": case.key,
        "seed": int(seed),
        "query_latent_index": 0,
        "query_pixel_frame": 0,
        "normalization": "per target-frame spatial softmax, then mean over object query points",
        "entries": entries,
        "regions": [region.region_name for region, _ in query_regions],
        "grid": [latent_time, height, width],
        "latent_anchor_pixel_frames": (np.arange(latent_time) * 4).tolist(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "complete.json").write_text(
        json.dumps({"case": case.key, "seed": seed, "top_n": len(entries)}, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    top_path = OUTPUT_ROOT / "aggregate" / "final_top10.json"
    entries = json.loads(top_path.read_text(encoding="utf-8"))["entries"]
    pipe = build_wan_ti2v_pipeline(build_args(all_tasks()[0][1]))
    tasks = all_tasks()[args.worker_id :: args.num_workers]
    for index, (case, seed) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] heatmap start {case.key} seed={seed}", flush=True)
        try:
            process(pipe, entries, case, seed, bool(args.overwrite))
        except Exception:
            output = heatmap_dir(case.key, seed)
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
            raise
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
