#!/usr/bin/env python3
"""Generate audited attention-matrix ablations for fixed PhysicIQ67 samples."""

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
)
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_pck_worker import (  # noqa: E402
    build_args,
    object_queries,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache  # noqa: E402


OUTPUT_ROOT = VISUAL_ROOT / "attention_matrix_ablations_v2"
TOP_COUNTS = (30, 50, 100)
TARGET_SCOPES = ("single_object", "all_objects")
MATRIX_MASKS = (
    "self_only",
    "incoming_only",
    "outgoing_only",
    "query_row",
    "key_value_column",
    "cross_boundary",
    "row_and_column",
)
ALL_TOKEN_CONTROLS = ("qk_logits_zero", "full_head_output")
CONTROL_MASKS = ("literal_kv_zero",) + ALL_TOKEN_CONTROLS
MASK_BLOCKS = {
    "self_only": ("S",),
    "incoming_only": ("I",),
    "outgoing_only": ("O",),
    "query_row": ("S", "I"),
    "key_value_column": ("S", "O"),
    "cross_boundary": ("I", "O"),
    "row_and_column": ("S", "I", "O"),
    "literal_kv_zero": (),
    "qk_logits_zero": (),
    "full_head_output": ("S", "I", "O", "C,C"),
}
PROTOCOL = "attention_matrix_ablation_v2"


def variant_id(
    target_scope: str, mask_mode: str, top_n: int, region: str | None = None
) -> str:
    if mask_mode in ALL_TOKEN_CONTROLS:
        if target_scope != "all_tokens":
            raise ValueError(f"{mask_mode} requires all_tokens target scope")
        return f"{mask_mode}__all_tokens__top{int(top_n):03d}"
    if target_scope == "single_object":
        if not region:
            raise ValueError("single_object requires a region")
        target = region
    elif target_scope == "all_objects":
        target = "all_objects"
    else:
        raise ValueError(f"unknown target scope: {target_scope}")
    if mask_mode not in MATRIX_MASKS + ("literal_kv_zero",):
        raise ValueError(f"unknown mask mode: {mask_mode}")
    return f"{target_scope}__{target}__{mask_mode}__top{int(top_n):03d}"


def build_tasks(manifest: dict) -> list[dict]:
    tasks = []
    for top_n in TOP_COUNTS:
        for sample in manifest["samples"]:
            regions = [
                str(row["region_name"])
                for row in sample["regions"]
                if row.get("region_type") == "object"
            ]
            targets = [("single_object", region) for region in regions]
            targets.append(("all_objects", None))
            for target_scope, region in targets:
                for mask_mode in MATRIX_MASKS + ("literal_kv_zero",):
                    tasks.append(
                        {
                            "case": str(sample["case"]),
                            "seed": int(sample["seed"]),
                            "target_scope": target_scope,
                            "mask_mode": mask_mode,
                            "region": region,
                            "top_n": top_n,
                        }
                    )
            for mask_mode in ALL_TOKEN_CONTROLS:
                tasks.append(
                    {
                        "case": str(sample["case"]),
                        "seed": int(sample["seed"]),
                        "target_scope": "all_tokens",
                        "mask_mode": mask_mode,
                        "region": None,
                        "top_n": top_n,
                    }
                )
    return tasks


def task_root(task: dict, output_root: Path = OUTPUT_ROOT) -> Path:
    return (
        output_root
        / str(task["case"])
        / f"seed_{int(task['seed']):05d}"
        / variant_id(
            str(task["target_scope"]),
            str(task["mask_mode"]),
            int(task["top_n"]),
            task.get("region"),
        )
    )


def baseline_tasks(manifest: dict) -> list[dict]:
    tasks = []
    for sample in manifest["samples"]:
        baseline_video = sample.get("baseline_video")
        if baseline_video and not Path(str(baseline_video)).is_file():
            tasks.append(
                {
                    "case": str(sample["case"]),
                    "seed": int(sample["seed"]),
                    "mode": "baseline",
                }
            )
    return tasks


class AttentionMatrixAblator:
    """Apply exact block ablations to A@V for a fixed sparse F00 token set."""

    def __init__(
        self,
        pipe,
        entries: list[dict],
        query_points,
        region_slices: dict[str, slice],
        pixel_hw: tuple[int, int],
        target_scope: str,
        mask_mode: str,
        region: str | None,
        extra_mask_modes: tuple[str, ...] = (),
    ) -> None:
        if target_scope not in TARGET_SCOPES + ("all_tokens",):
            raise ValueError(f"unsupported target scope: {target_scope}")
        if mask_mode not in MATRIX_MASKS + CONTROL_MASKS + extra_mask_modes:
            raise ValueError(f"unsupported mask mode: {mask_mode}")
        if mask_mode in ALL_TOKEN_CONTROLS and target_scope != "all_tokens":
            raise ValueError(f"{mask_mode} requires target_scope=all_tokens")
        if target_scope == "all_tokens" and mask_mode not in ALL_TOKEN_CONTROLS:
            raise ValueError("all_tokens is reserved for all-token controls")
        if target_scope == "single_object" and region not in region_slices:
            raise ValueError(f"unknown object region: {region}")
        self.pipe = pipe
        self.entries = entries
        self.query_points = torch.as_tensor(query_points, dtype=torch.float32)
        self.region_slices = region_slices
        self.pixel_height, self.pixel_width = pixel_hw
        self.target_scope = target_scope
        self.mask_mode = mask_mode
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
        self.auxiliary_attention_calls = 0
        self.affected_query_vectors = 0
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
        if self.target_scope == "all_tokens":
            return None
        if self.current_grid is None:
            raise RuntimeError("attention grid is unavailable")
        _, height, width = self.current_grid
        points = self.query_points.to(device)
        x = torch.floor(points[:, 0] * width / self.pixel_width).long().clamp(0, width - 1)
        y = torch.floor(points[:, 1] * height / self.pixel_height).long().clamp(0, height - 1)
        source_indices = y * width + x
        if self.target_scope == "single_object":
            source_indices = source_indices[self.region_slices[str(self.region)]]
        rows = torch.unique(source_indices, sorted=True)
        values = [int(value) for value in rows.detach().cpu().tolist()]
        if self.query_token_indices is None:
            self.query_token_indices = values
        elif self.query_token_indices != values:
            raise RuntimeError("object query token mapping changed during generation")
        return rows

    @staticmethod
    def _head_view(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[-1] % num_heads:
            raise RuntimeError(f"tensor shape {tuple(tensor.shape)} is not head-aligned")
        return tensor.reshape(
            tensor.shape[0], tensor.shape[1], num_heads, tensor.shape[-1] // num_heads
        )

    @staticmethod
    def _selected_values(
        v: torch.Tensor, rows: torch.Tensor, heads: tuple[int, ...] | list[int], num_heads: int
    ) -> torch.Tensor:
        selected = torch.zeros_like(v)
        source_heads = AttentionMatrixAblator._head_view(v, num_heads)
        selected_heads = AttentionMatrixAblator._head_view(selected, num_heads)
        for head in heads:
            selected_heads[:, rows, head, :] = source_heads[:, rows, head, :]
        return selected

    @staticmethod
    def _zero_selected_kv(
        tensor: torch.Tensor,
        rows: torch.Tensor,
        heads: tuple[int, ...] | list[int],
        num_heads: int,
    ) -> torch.Tensor:
        modified = tensor.clone()
        modified_heads = AttentionMatrixAblator._head_view(modified, num_heads)
        for head in heads:
            modified_heads[:, rows, head, :] = 0
        return modified

    @staticmethod
    def _zero_selected_heads(
        tensor: torch.Tensor,
        heads: tuple[int, ...] | list[int],
        num_heads: int,
    ) -> torch.Tensor:
        """Zero complete physical heads at every token position."""
        modified = tensor.clone()
        modified_heads = AttentionMatrixAblator._head_view(modified, num_heads)
        for head in heads:
            modified_heads[:, :, head, :] = 0
        return modified

    def _attention(self, q, k, v, original, block: int):
        heads = self.by_block.get(block, ())
        if not self.active or not heads:
            return original(q, k, v)
        num_heads = int(q.shape[-1] // 128)
        if num_heads <= 0 or q.shape[-1] % num_heads:
            raise RuntimeError(f"query width {q.shape[-1]} is not head-aligned")

        rows = self._rows(q.device)
        if rows is None:
            row_count = int(q.shape[1])
        elif not rows.numel():
            raise RuntimeError("no object query tokens selected")
        else:
            row_count = int(rows.numel())

        if self.mask_mode == "qk_logits_zero":
            if rows is not None:
                raise RuntimeError("qk_logits_zero requires all-token scope")
            # q_h=0 for every token makes every selected-head QK^T logit
            # exactly zero. softmax(0) is uniform, so the head returns mean(V),
            # rather than a zero output.
            modified_q = self._zero_selected_heads(q, heads, num_heads)
            output = original(modified_q, k, v)
            affected_rows = int(q.shape[1])
        elif self.mask_mode == "literal_kv_zero":
            if rows is None:
                raise RuntimeError("literal_kv_zero requires selected object tokens")
            modified_k = self._zero_selected_kv(k, rows, heads, num_heads)
            modified_v = self._zero_selected_kv(v, rows, heads, num_heads)
            output = original(q, modified_k, modified_v)
            affected_rows = int(q.shape[1])
        elif self.mask_mode in {"key_value_column", "row_and_column"}:
            if rows is None:
                raise RuntimeError(f"{self.mask_mode} requires selected object tokens")
            modified_v = self._zero_selected_kv(v, rows, heads, num_heads)
            output = original(q, k, modified_v)
            output_heads = self._head_view(output, num_heads)
            if self.mask_mode == "row_and_column":
                for head in heads:
                    output_heads[:, rows, head, :] = 0
            affected_rows = int(q.shape[1])
        else:
            output = original(q, k, v)
            output_heads = self._head_view(output, num_heads)
            if self.mask_mode == "full_head_output":
                for head in heads:
                    output_heads[:, :, head, :] = 0
                affected_rows = int(q.shape[1])
            elif self.mask_mode == "query_row":
                if rows is None:
                    raise RuntimeError("query_row requires selected object tokens")
                for head in heads:
                    output_heads[:, rows, head, :] = 0
                affected_rows = row_count
            else:
                if rows is None:
                    raise RuntimeError(f"{self.mask_mode} requires selected object tokens")
                selected_v = self._selected_values(v, rows, heads, num_heads)
                selected_contribution = original(q, k, selected_v)
                self.auxiliary_attention_calls += 1
                contribution_heads = self._head_view(selected_contribution, num_heads)
                for head in heads:
                    if self.mask_mode == "self_only":
                        output_heads[:, rows, head, :] -= contribution_heads[:, rows, head, :]
                    elif self.mask_mode == "incoming_only":
                        output_heads[:, rows, head, :] = contribution_heads[:, rows, head, :]
                    elif self.mask_mode == "outgoing_only":
                        original_rows = output_heads[:, rows, head, :].clone()
                        output_heads[:, :, head, :] -= contribution_heads[:, :, head, :]
                        output_heads[:, rows, head, :] = original_rows
                    elif self.mask_mode == "cross_boundary":
                        output_heads[:, :, head, :] -= contribution_heads[:, :, head, :]
                        output_heads[:, rows, head, :] = contribution_heads[:, rows, head, :]
                    else:
                        raise RuntimeError(f"unhandled mask mode: {self.mask_mode}")
                affected_rows = row_count if self.mask_mode in {"self_only", "incoming_only"} else int(q.shape[1])

        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError(f"unexpected attention output: {type(output)}")
        self.modified_forward_calls += 1
        self.modified_head_events += len(heads)
        self.affected_query_vectors += output.shape[0] * affected_rows * len(heads)
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
        if self.target_scope != "all_tokens" and not self.query_token_indices:
            raise RuntimeError("object query token indices were not resolved")
        return {
            "model_call_counts": self.model_call_counts,
            "modified_forward_calls": self.modified_forward_calls,
            "modified_head_events": self.modified_head_events,
            "expected_head_events": expected_head_events,
            "auxiliary_attention_calls": self.auxiliary_attention_calls,
            "affected_query_vectors": self.affected_query_vectors,
            "query_token_indices": self.query_token_indices,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--task-indices", type=int, nargs="+", default=None)
    parser.add_argument("--top-counts", type=int, nargs="+", choices=TOP_COUNTS, default=None)
    parser.add_argument(
        "--object-dependent-only",
        action="store_true",
        help="run M1-M7/C1 targets only; omit the R-independent C2/C3 controls",
    )
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--generate-missing-baselines", action="store_true")
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="generate only missing seed-matched baselines from the manifest",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_inputs(sample: dict, case_lookup: dict) -> tuple[Path, Path]:
    case_key = str(sample["case"])
    case = case_lookup.get(case_key)
    input_json = sample.get("input_json")
    query_cache_dir = sample.get("query_cache_dir")
    if input_json is None:
        if case is None:
            raise KeyError(f"{case_key}: manifest has no input_json and case is unknown")
        input_json = case.json_path
    if query_cache_dir is None:
        query_cache_dir = REGION_CACHE_ROOT / case_key
    return Path(str(input_json)), Path(str(query_cache_dir))


def generation_inputs(sample: dict, case_lookup: dict, seed: int):
    json_path, cache_dir = sample_inputs(sample, case_lookup)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload, firstframe = ensure_firstframe_image(json_path, payload)
    args = build_args(seed)
    image = (
        Image.open(firstframe)
        .convert("RGB")
        .resize((1280, 704), Image.Resampling.LANCZOS)
    )
    return json_path, cache_dir, payload, args, image


def generate_video(pipe, payload: dict, args, image: Image.Image, seed: int):
    return _run_pipe_once(
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


def process_baseline(
    pipe, task: dict, sample: dict, case_lookup: dict, overwrite: bool
) -> None:
    output_video = Path(str(sample["baseline_video"]))
    output = output_video.parent
    ready = all(
        (output / name).is_file() for name in ("complete.json", "manifest.json")
    )
    if output_video.is_file() and ready and not overwrite:
        print(f"skip baseline {output_video}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "complete.json").unlink(missing_ok=True)
    (output / "error.txt").unlink(missing_ok=True)
    json_path, _, payload, args, image = generation_inputs(
        sample, case_lookup, int(task["seed"])
    )
    video = generate_video(pipe, payload, args, image, int(task["seed"]))
    temporary_video = output_video.with_name(
        f"{output_video.stem}.tmp{output_video.suffix}"
    )
    save_video_np(video, temporary_video, fps=30)
    temporary_video.replace(output_video)
    metadata = {
        **task,
        "input_json": str(json_path),
        "output_video": str(output_video),
        "height": 704,
        "width": 1280,
        "num_frames": 49,
        "fps": 30,
        "sampling_steps": 40,
        "cfg_scale": 5.0,
        "sample_shift": 5.0,
        "sample_solver": "unipc",
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "complete.json").write_text(
        json.dumps(
            {"case": task["case"], "seed": task["seed"], "mode": "baseline"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del video


def process(
    pipe,
    manifest: dict,
    task: dict,
    sample: dict,
    case_lookup: dict,
    output_root: Path,
    overwrite: bool,
) -> None:
    output = task_root(task, output_root)
    complete_path = output / "complete.json"
    ready = all(
        (output / name).is_file()
        for name in ("complete.json", "manifest.json", "generated.mp4")
    )
    if ready and not overwrite:
        print(f"skip {output.relative_to(output_root)}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    (output / "error.txt").unlink(missing_ok=True)
    json_path, cache_dir, payload, args, image = generation_inputs(
        sample, case_lookup, int(task["seed"])
    )
    cache = load_region_cache(cache_dir.parent, cache_dir.name)
    if int(cache.metadata.get("query_context_frame", -1)) != 0:
        raise RuntimeError(f"{task['case']}: expected query frame 0 cache")
    points, query_regions = object_queries(cache)
    region_slices = {region.region_name: point_slice for region, point_slice in query_regions}
    entries = manifest["entries"][: int(task["top_n"])]
    ablator = AttentionMatrixAblator(
        pipe.pipe,
        entries,
        points,
        region_slices,
        (704, 1280),
        str(task["target_scope"]),
        str(task["mask_mode"]),
        task.get("region"),
    )
    ablator.install()
    try:
        video = generate_video(pipe, payload, args, image, int(task["seed"]))
    finally:
        ablator.remove()
    audit = ablator.audit()
    temporary_video = output / "generated.tmp.mp4"
    save_video_np(video, temporary_video, fps=30)
    temporary_video.replace(output / "generated.mp4")
    metadata = {
        **task,
        "variant_id": variant_id(
            str(task["target_scope"]),
            str(task["mask_mode"]),
            int(task["top_n"]),
            task.get("region"),
        ),
        "protocol": PROTOCOL,
        "attention_definition": "A=softmax(QK^T/sqrt(d)); Y=A@V",
        "selected_token_definition": (
            "unique latent spatial tokens mapped from sparse object points at "
            "query pixel frame F00; not a full object mask or temporal tube"
        ),
        "matrix_partition": {
            "R": "selected sparse F00 object tokens",
            "C": "all unselected tokens",
            "S": "A[R,R]",
            "I": "A[R,C] (unselected K/V -> selected queries)",
            "O": "A[C,R] (selected K/V -> unselected queries)",
        },
        "zeroed_matrix_blocks": list(MASK_BLOCKS[str(task["mask_mode"])]),
        "semantic_qkv_projection_intervention": (
            str(task["mask_mode"]) in {"literal_kv_zero", "qk_logits_zero"}
        ),
        "post_mask_renormalization": False,
        "softmax_recomputed_after_k_intervention": (
            str(task["mask_mode"]) == "literal_kv_zero"
        ),
        "softmax_recomputed_after_q_intervention": (
            str(task["mask_mode"]) == "qk_logits_zero"
        ),
        "implementation": (
            "literal selected K/V vectors set to zero before attention"
            if str(task["mask_mode"]) == "literal_kv_zero"
            else (
                "selected physical heads have q_h=0 at every token, so QK^T=0, "
                "softmax is uniform, and Y_h=mean(V_h)"
                if str(task["mask_mode"]) == "qk_logits_zero"
                else (
                    "selected physical head A@V output is zero at every token"
                    if str(task["mask_mode"]) == "full_head_output"
                    else "post-softmax A@V block decomposition; column masks use the exact V_R=0 equivalence"
                )
            )
        ),
        "output_projection_location": "selected self-attention head outputs before o projection",
        "denoising_steps": list(range(40)),
        "cfg_branches": ["conditional", "unconditional"],
        "input_json": str(json_path),
        "query_cache_dir": str(cache_dir),
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
                "protocol": PROTOCOL,
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
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("entries", [])) < max(TOP_COUNTS):
        raise RuntimeError("visual sample manifest does not contain Top100 entries")
    tasks = build_tasks(manifest)
    if args.top_counts is not None:
        selected_top_counts = set(args.top_counts)
        tasks = [task for task in tasks if int(task["top_n"]) in selected_top_counts]
    if args.object_dependent_only:
        tasks = [task for task in tasks if task["target_scope"] != "all_tokens"]
    if args.baselines_only:
        tasks = baseline_tasks(manifest)
    elif args.generate_missing_baselines:
        tasks = baseline_tasks(manifest) + tasks
    if args.task_index is not None and args.task_indices is not None:
        raise ValueError("--task-index and --task-indices are mutually exclusive")
    if args.task_indices is not None:
        if len(set(args.task_indices)) != len(args.task_indices):
            raise ValueError("--task-indices contains duplicates")
        invalid = [index for index in args.task_indices if not 0 <= index < len(tasks)]
        if invalid:
            raise ValueError(f"task-indices outside [0, {len(tasks)}): {invalid}")
        tasks = [tasks[index] for index in args.task_indices]
    elif args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            raise ValueError(f"task-index must be in [0, {len(tasks)})")
        tasks = [tasks[args.task_index]]
    else:
        tasks = tasks[args.worker_id :: args.num_workers]
    if not tasks:
        return
    case_lookup = {case.key: case for case in CASES}
    sample_lookup = {
        (str(sample["case"]), int(sample["seed"])): sample
        for sample in manifest["samples"]
    }
    pipe = build_wan_ti2v_pipeline(build_args(int(tasks[0]["seed"])))
    for index, task in enumerate(tasks, start=1):
        sample = sample_lookup[(str(task["case"]), int(task["seed"]))]
        if task.get("mode") == "baseline":
            output = Path(str(sample["baseline_video"])).parent
            label = f"baseline/{task['case']}/seed_{int(task['seed']):05d}"
        else:
            output = task_root(task, args.output_root)
            label = str(output.relative_to(args.output_root))
        print(f"[{index}/{len(tasks)}] start {label}", flush=True)
        try:
            if task.get("mode") == "baseline":
                process_baseline(pipe, task, sample, case_lookup, bool(args.overwrite))
            else:
                process(
                    pipe,
                    manifest,
                    task,
                    sample,
                    case_lookup,
                    args.output_root,
                    bool(args.overwrite),
                )
        except Exception:
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(tasks)}] complete {label}", flush=True)


if __name__ == "__main__":
    main()
