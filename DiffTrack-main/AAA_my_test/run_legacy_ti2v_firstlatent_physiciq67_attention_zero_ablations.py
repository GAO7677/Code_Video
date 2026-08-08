#!/usr/bin/env python3
"""Generate post-softmax attention-zero ablations for fixed PhysicIQ67 samples."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

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
    save_video_np,
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
from AAA_my_test.sam2_region_query_utils import load_region_cache  # noqa: E402


OUTPUT_ROOT = VISUAL_ROOT / "attention_zero_ablations"
TOP_COUNTS = (30, 50, 100)
MODES = ("single_object", "all_objects", "full_qk")


def variant_id(mode: str, top_n: int, region: str | None = None) -> str:
    if mode == "single_object":
        if not region:
            raise ValueError("single_object requires a region")
        target = region
    elif mode == "all_objects":
        target = "all_objects"
    elif mode == "full_qk":
        target = "all_queries"
    else:
        raise ValueError(f"unknown mode: {mode}")
    return f"{mode}__{target}__top{int(top_n):03d}"


def build_tasks(manifest: dict) -> list[dict]:
    tasks = []
    for top_n in TOP_COUNTS:
        for sample in manifest["samples"]:
            regions = [
                str(row["region_name"])
                for row in sample["regions"]
                if row.get("region_type") == "object"
            ]
            for region in regions:
                tasks.append(
                    {
                        "case": str(sample["case"]),
                        "seed": int(sample["seed"]),
                        "mode": "single_object",
                        "region": region,
                        "top_n": top_n,
                    }
                )
            for mode in ("all_objects", "full_qk"):
                tasks.append(
                    {
                        "case": str(sample["case"]),
                        "seed": int(sample["seed"]),
                        "mode": mode,
                        "region": None,
                        "top_n": top_n,
                    }
                )
    return tasks


def task_root(task: dict) -> Path:
    return (
        OUTPUT_ROOT
        / str(task["case"])
        / f"seed_{int(task['seed']):05d}"
        / variant_id(str(task["mode"]), int(task["top_n"]), task.get("region"))
    )


class PostSoftmaxAttentionZeroer:
    """Zero selected A@V rows, exactly equivalent to setting those A rows to zero."""

    def __init__(
        self,
        pipe,
        entries: list[dict],
        query_points,
        region_slices: dict[str, slice],
        pixel_hw: tuple[int, int],
        mode: str,
        region: str | None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unsupported mode: {mode}")
        if mode == "single_object" and region not in region_slices:
            raise ValueError(f"unknown object region: {region}")
        self.pipe = pipe
        self.entries = entries
        self.query_points = torch.as_tensor(query_points, dtype=torch.float32)
        self.region_slices = region_slices
        self.pixel_height, self.pixel_width = pixel_hw
        self.mode = mode
        self.region = region
        self.by_block: dict[int, list[int]] = {}
        for entry in entries:
            self.by_block.setdefault(int(entry["block"]), []).append(int(entry["head"]))
        for block in self.by_block:
            self.by_block[block] = sorted(set(self.by_block[block]))
        if sum(len(heads) for heads in self.by_block.values()) != len(entries):
            raise RuntimeError("selected ranking contains duplicate physical heads")
        self.current_step = -1
        self.current_grid: tuple[int, int, int] | None = None
        self.active = False
        self.model_call_counts: dict[int, int] = {}
        self.modified_head_events = 0
        self.modified_forward_calls = 0
        self.zeroed_output_vectors = 0
        self.query_token_indices: list[int] | None = None
        self._original_model_fn = None
        self._original_forwards: list[tuple[Any, Any]] = []

    def _step(self, timestep: torch.Tensor) -> int:
        value = float(timestep.detach().flatten()[0].float().cpu())
        schedule = self.pipe.scheduler.timesteps.detach().float().cpu()
        return int(torch.argmin((schedule - value).abs()).item())

    def _wrapped_model_fn(self, *args, **kwargs):
        timestep, latents = kwargs.get("timestep"), kwargs.get("latents")
        if timestep is None or latents is None:
            return self._original_model_fn(*args, **kwargs)
        patch = tuple(int(value) for value in kwargs["dit"].patch_size)
        self.current_grid = (
            int(latents.shape[2] // patch[0]),
            int(latents.shape[3] // patch[1]),
            int(latents.shape[4] // patch[2]),
        )
        self.current_step = self._step(timestep)
        self.model_call_counts[self.current_step] = self.model_call_counts.get(self.current_step, 0) + 1
        self.active = True
        try:
            return self._original_model_fn(*args, **kwargs)
        finally:
            self.active = False
            self.current_step = -1

    def _rows(self, device: torch.device) -> torch.Tensor | None:
        if self.mode == "full_qk":
            return None
        if self.current_grid is None:
            raise RuntimeError("attention grid is unavailable")
        _, height, width = self.current_grid
        points = self.query_points.to(device)
        x = torch.floor(points[:, 0] * width / self.pixel_width).long().clamp(0, width - 1)
        y = torch.floor(points[:, 1] * height / self.pixel_height).long().clamp(0, height - 1)
        source_indices = y * width + x
        if self.mode == "single_object":
            source_indices = source_indices[self.region_slices[str(self.region)]]
        rows = torch.unique(source_indices, sorted=True)
        values = [int(value) for value in rows.detach().cpu().tolist()]
        if self.query_token_indices is None:
            self.query_token_indices = values
        elif self.query_token_indices != values:
            raise RuntimeError("object query token mapping changed during generation")
        return rows

    def _attention(self, q, k, v, original, block: int):
        output = original(q, k, v)
        heads = self.by_block.get(block, ())
        if not self.active or not heads:
            return output
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError(f"unexpected attention output: {type(output)}")
        num_heads = int(q.shape[-1] // 128)
        if output.shape[-1] % num_heads:
            raise RuntimeError(f"attention width {output.shape[-1]} is not head-aligned")
        head_dim = output.shape[-1] // num_heads
        output_heads = output.reshape(output.shape[0], output.shape[1], num_heads, head_dim)
        if self.mode == "full_qk":
            row_count = int(output.shape[1])
            output_heads[:, :, heads, :] = 0
        else:
            rows = self._rows(output.device)
            if rows is None or not rows.numel():
                raise RuntimeError("no object query tokens selected")
            row_count = int(rows.numel())
            for head in heads:
                output_heads[:, rows, head, :] = 0
        self.modified_forward_calls += 1
        self.modified_head_events += len(heads)
        self.zeroed_output_vectors += output.shape[0] * row_count * len(heads)
        return output

    def install(self) -> None:
        self._original_model_fn = self.pipe.model_fn
        self.pipe.model_fn = self._wrapped_model_fn
        models = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            models.append(self.pipe.dit2)
        for model in models:
            for block, heads in self.by_block.items():
                module = model.blocks[block].self_attn.attn
                for head in heads:
                    if not 0 <= head < int(module.num_heads):
                        raise ValueError(f"head {head} outside L{block} head range")
                original = module.forward
                self._original_forwards.append((module, original))

                def wrapped(q, k, v, *, _original=original, _block=block):
                    return self._attention(q, k, v, _original, _block)

                module.forward = wrapped

    def remove(self) -> None:
        if self._original_model_fn is not None:
            self.pipe.model_fn = self._original_model_fn
        for module, original in self._original_forwards:
            module.forward = original
        self._original_forwards.clear()

    def audit(self) -> dict:
        expected_model_calls = 40 * 2
        expected_head_events = len(self.entries) * expected_model_calls
        if sorted(self.model_call_counts) != list(range(40)):
            raise RuntimeError(f"expected steps 0..39, got {sorted(self.model_call_counts)}")
        if any(count != 2 for count in self.model_call_counts.values()):
            raise RuntimeError(f"expected two CFG calls per step, got {self.model_call_counts}")
        if self.modified_head_events != expected_head_events:
            raise RuntimeError(
                f"modified {self.modified_head_events} head events, expected {expected_head_events}"
            )
        if self.mode != "full_qk" and not self.query_token_indices:
            raise RuntimeError("object query token indices were not resolved")
        return {
            "model_call_counts": self.model_call_counts,
            "modified_forward_calls": self.modified_forward_calls,
            "modified_head_events": self.modified_head_events,
            "expected_head_events": expected_head_events,
            "zeroed_output_vectors": self.zeroed_output_vectors,
            "query_token_indices": self.query_token_indices,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def process(pipe, manifest: dict, task: dict, case, overwrite: bool) -> None:
    output = task_root(task)
    complete_path = output / "complete.json"
    ready = all(
        (output / name).is_file()
        for name in ("complete.json", "manifest.json", "generated.mp4")
    )
    if ready and not overwrite:
        print(f"skip {output.relative_to(OUTPUT_ROOT)}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    (output / "error.txt").unlink(missing_ok=True)
    cache = load_region_cache(REGION_CACHE_ROOT, case.key)
    if int(cache.metadata.get("query_context_frame", -1)) != 0:
        raise RuntimeError(f"{case.key}: expected query frame 0 cache")
    points, query_regions = object_queries(cache)
    region_slices = {region.region_name: point_slice for region, point_slice in query_regions}
    payload = read_payload(case)
    payload, firstframe = ensure_firstframe_image(case.json_path, payload)
    args = build_args(int(task["seed"]))
    image = Image.open(firstframe).convert("RGB").resize((1280, 704), Image.Resampling.LANCZOS)
    entries = manifest["entries"][: int(task["top_n"])]
    zeroer = PostSoftmaxAttentionZeroer(
        pipe.pipe,
        entries,
        points,
        region_slices,
        (704, 1280),
        str(task["mode"]),
        task.get("region"),
    )
    zeroer.install()
    try:
        video = _run_pipe_once(
            pipe=pipe,
            prompt=str(payload["input_caption"]),
            negative_prompt=args.negative_prompt,
            seed=int(task["seed"]),
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
        zeroer.remove()
    audit = zeroer.audit()
    temporary_video = output / "generated.tmp.mp4"
    save_video_np(video, temporary_video, fps=30)
    temporary_video.replace(output / "generated.mp4")
    metadata = {
        **task,
        "variant_id": variant_id(str(task["mode"]), int(task["top_n"]), task.get("region")),
        "protocol": "post_softmax_attention_rows_zero_without_renormalization",
        "mathematical_equivalence": "A[selected_rows,:]=0 => (A@V)[selected_rows,:]=0",
        "qkv_modified": False,
        "output_projection_location": "selected self-attention head outputs before o projection",
        "denoising_steps": list(range(40)),
        "cfg_branches": ["conditional", "unconditional"],
        "ranking_snapshot_completed_runs": int(manifest["completed_runs_at_selection"]),
        "selected_entries": entries,
        "regions": list(region_slices),
        "audit": audit,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete_path.write_text(
        json.dumps(
            {
                "case": task["case"],
                "seed": task["seed"],
                "variant_id": metadata["variant_id"],
                "modified_head_events": audit["modified_head_events"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del video


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if len(manifest.get("entries", [])) < max(TOP_COUNTS):
        raise RuntimeError("visual sample manifest does not contain Top100 entries")
    tasks = build_tasks(manifest)
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            raise ValueError(f"task-index must be in [0, {len(tasks)})")
        tasks = [tasks[args.task_index]]
    else:
        tasks = tasks[args.worker_id :: args.num_workers]
    if not tasks:
        return
    case_lookup = {case.key: case for case in CASES}
    pipe = build_wan_ti2v_pipeline(build_args(int(tasks[0]["seed"])))
    for index, task in enumerate(tasks, start=1):
        output = task_root(task)
        print(f"[{index}/{len(tasks)}] start {output.relative_to(OUTPUT_ROOT)}", flush=True)
        try:
            process(pipe, manifest, task, case_lookup[str(task["case"])], bool(args.overwrite))
        except Exception:
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(tasks)}] complete {output.relative_to(OUTPUT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
