#!/usr/bin/env python3
"""Replay the 108 M1/M2/M3 runs and capture fixed-F04 S039 Top100 means.

The generated videos already exist.  This worker replays the deterministic
inference only to recover attention probabilities and never overwrites those
videos.  `before` is live softmax attention in the replay.  `effective_after`
sets exactly the coefficients whose A@V contributions the active M1/M2/M3
operator removes to zero, without softmax renormalization.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from AAA_my_test.render_m123_s039_top100_mean_overlay import render_variant
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (
    AttentionMatrixAblator,
    build_args,
    generate_video,
    generation_inputs,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import object_queries
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (
    DEFAULT_HEAD_RANKING,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT as DEFAULT_ABLATION_ROOT,
    HEAD_SCOPE_COUNTS,
    HEAD_SCOPES,
    TEMPORAL_DIRECTIONAL_MODES,
    TEMPORAL_DIRECTIONAL_SPECS,
    TemporalObjectTubeAblator,
    prepare_tracks,
    selected_head_entries,
    task_root as ablation_task_root,
    temporal_variant_id,
    validate_head_ranking,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache
from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import build_wan_ti2v_pipeline


CASE = "0613pybullet_sample_001460_w002"
SEED = 47326
CAPTURE_STEP = 39
LATENT_FRAMES = 13
LATENT_HEIGHT = 16
LATENT_WIDTH = 28
FRAME_TOKEN_COUNT = LATENT_HEIGHT * LATENT_WIDTH
SEQUENCE = LATENT_FRAMES * FRAME_TOKEN_COUNT
F04_QUERY_CACHE = Path(
    "/data/gaoya/agent-data/cache/test100_51_grounded_sam2_regions/"
    "case_test100_51_048_0613pybullet_sample_001460_w002"
)
DEFAULT_CAPTURE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_attention_overlays/"
    "m123_head_scope_s039_top100_mean_v1"
)
M123_MODES = (
    "self_only",
    "self_same",
    "self_future",
    "self_past",
    "incoming_only",
    "incoming_same",
    "incoming_future",
    "incoming_past",
    "outgoing_only",
    "outgoing_same",
    "outgoing_future",
    "outgoing_past",
)
OPERATOR_IDS = {"self": "M1", "incoming": "M2", "outgoing": "M3"}
TEMPORAL_LABELS = {
    "only": "All-time",
    "same": "Same",
    "future": "Future",
    "past": "Past",
}


def mode_parts(mask_mode: str) -> tuple[str, str, str, str]:
    base = next((name for name in OPERATOR_IDS if mask_mode.startswith(name)), None)
    if base is None:
        raise ValueError(f"not an M1/M2/M3 mode: {mask_mode}")
    temporal = mask_mode.removeprefix(f"{base}_")
    if temporal not in TEMPORAL_LABELS:
        raise ValueError(f"unknown temporal scope: {mask_mode}")
    target_partition = "C" if base == "outgoing" else "R"
    source_partition = "C" if base == "incoming" else "R"
    return base, temporal, target_partition, source_partition


def exact_tasks(sample: dict) -> list[dict]:
    regions = [
        str(row["region_name"])
        for row in sample.get("regions", [])
        if row.get("region_type") == "object"
    ]
    if regions != ["object_A", "object_B"]:
        raise RuntimeError(f"expected Object A/B for {CASE}, got {regions}")
    targets = [("single_object", region) for region in regions] + [("all_objects", None)]
    tasks = [
        {
            "case": str(sample["case"]),
            "seed": int(sample["seed"]),
            "target_scope": target_scope,
            "mask_mode": mask_mode,
            "region": region,
            "top_n": HEAD_SCOPE_COUNTS[head_scope],
            "head_scope": head_scope,
            "ranking_tag": "",
        }
        for head_scope in HEAD_SCOPES
        for target_scope, region in targets
        for mask_mode in M123_MODES
    ]
    if len(tasks) != 108 or len({temporal_variant_id(task) for task in tasks}) != 108:
        raise RuntimeError("M1/M2/M3 Head-Scope matrix is not exactly 108 unique tasks")
    return tasks


def task_formula(task: dict) -> dict:
    base, temporal, target, source = mode_parts(str(task["mask_mode"]))
    predicate = {
        "only": "all tk",
        "same": "tk=tq",
        "future": "tk<tq",
        "past": "tk>tq",
    }[temporal]
    operator = OPERATOR_IDS[base]
    return {
        "operator_id": operator,
        "temporal_scope": TEMPORAL_LABELS[temporal],
        "target_partition": target,
        "source_partition": source,
        "time_predicate": predicate,
        "formula": (
            f"Y'_{target}(tq)=Y_{target}(tq)-"
            f"sum_{{{predicate}}} A[{target}_tq,{source}_tk]V_{source}(tk)"
        ),
    }


def capture_task_root(capture_root: Path, task: dict) -> Path:
    return (
        Path(capture_root)
        / str(task["case"])
        / f"seed_{int(task['seed']):05d}"
        / temporal_variant_id(task)
    )


def build_delete_mask(
    tokens_by_time: list[list[int]],
    frame_token_count: int,
    query_indices: np.ndarray | list[int],
    mask_mode: str,
) -> np.ndarray:
    """Return exact post-softmax coefficient entries deleted for fixed queries."""
    _, temporal, target_partition, source_partition = mode_parts(mask_mode)
    time_count = len(tokens_by_time)
    sequence = time_count * frame_token_count
    r_sets = [set(int(value) for value in values) for values in tokens_by_time]
    for time_index, rows in enumerate(r_sets):
        start, stop = time_index * frame_token_count, (time_index + 1) * frame_token_count
        if any(row < start or row >= stop for row in rows):
            raise ValueError(f"R rows for latent time {time_index} escape [{start}, {stop})")
    result = np.zeros((len(query_indices), sequence), dtype=np.bool_)
    for query_offset, query_index_value in enumerate(query_indices):
        query_index = int(query_index_value)
        if not 0 <= query_index < sequence:
            raise ValueError(f"fixed query index outside sequence: {query_index}")
        query_time = query_index // frame_token_count
        query_is_r = query_index in r_sets[query_time]
        if (target_partition == "R") != query_is_r:
            continue
        if temporal == "only":
            source_times = range(time_count)
        elif temporal == "same":
            source_times = (query_time,)
        elif temporal == "future":
            source_times = range(query_time)
        elif temporal == "past":
            source_times = range(query_time + 1, time_count)
        else:  # guarded by mode_parts
            raise AssertionError(temporal)
        for source_time in source_times:
            start = source_time * frame_token_count
            stop = start + frame_token_count
            if source_partition == "C":
                result[query_offset, start:stop] = True
                if r_sets[source_time]:
                    result[query_offset, list(r_sets[source_time])] = False
            elif r_sets[source_time]:
                result[query_offset, list(r_sets[source_time])] = True
    return result


def load_fixed_f04_regions(cache_dir: Path) -> tuple[list[dict], np.ndarray]:
    cache = load_region_cache(cache_dir.parent, cache_dir.name)
    if int(cache.metadata.get("query_context_frame", -1)) != 4:
        raise RuntimeError("fixed Object Query cache must use pixel frame F04")
    height, width = cache.context_frame_rgb.shape[:2]
    if (height, width) != (512, 896):
        raise RuntimeError(f"expected F04 query cache at 512x896, got {height}x{width}")
    regions = []
    for region_index, region in enumerate(cache.regions):
        if region.region_type != "object":
            continue
        points = cache.query_points[region.point_start : region.point_end].copy()
        token_x = np.floor(points[:, 0] * LATENT_WIDTH / width).astype(np.int64)
        token_y = np.floor(points[:, 1] * LATENT_HEIGHT / height).astype(np.int64)
        local = np.clip(token_y, 0, LATENT_HEIGHT - 1) * LATENT_WIDTH + np.clip(
            token_x, 0, LATENT_WIDTH - 1
        )
        indices = FRAME_TOKEN_COUNT + local
        regions.append(
            {
                "name": region.region_name,
                "phrase": region.region_phrase or region.region_name,
                "points": points,
                "mask": cache.masks_rhw[region_index].copy(),
                "token_indices": indices.astype(np.int64),
            }
        )
    if [row["name"] for row in regions] != ["object_A", "object_B"]:
        raise RuntimeError("F04 cache must contain Object A/B")
    return regions, cache.context_frame_rgb.copy()


class S039Top100MeanAblator(TemporalObjectTubeAblator):
    """Run the original ablation and audit its effective coefficients at S039."""

    def __init__(
        self,
        *args,
        capture_entries: list[dict],
        capture_regions: list[dict],
        capture_context_frame: np.ndarray,
        capture_step: int = CAPTURE_STEP,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.capture_step = int(capture_step)
        self.capture_regions = capture_regions
        self.capture_context_frame = capture_context_frame
        self.capture_by_block: dict[int, list[int]] = {}
        for entry in capture_entries:
            self.capture_by_block.setdefault(int(entry["block"]), []).append(int(entry["head"]))
        for block in self.capture_by_block:
            self.capture_by_block[block] = sorted(set(self.capture_by_block[block]))
        if sum(map(len, self.capture_by_block.values())) != 100:
            raise RuntimeError("capture ranking must contain exactly 100 unique Top100 heads")
        self.capture_sums: dict[str, dict[str, Any]] = {}
        self.capture_head_calls: dict[tuple[int, int], int] = {}
        self._delete_masks: dict[str, torch.Tensor] = {}

    def install(self) -> None:
        self._original_model_fn = self.pipe.model_fn
        self.pipe.model_fn = self._wrapped_model_fn
        models = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            models.append(self.pipe.dit2)
        blocks = sorted(set(self.by_block) | set(self.capture_by_block))
        for model in models:
            for block in blocks:
                module = model.blocks[block].self_attn.attn
                for head in set(self.by_block.get(block, ())) | set(
                    self.capture_by_block.get(block, ())
                ):
                    if not 0 <= head < int(module.num_heads):
                        raise ValueError(f"head {head} outside L{block} head range")
                original = module.forward
                self._original_forwards.append((module, original))

                def wrapped(q, k, v, *, _original=original, _block=block):
                    return self._attention(q, k, v, _original, _block)

                module.forward = wrapped

    def _capture_attention(self, q: torch.Tensor, k: torch.Tensor, block: int) -> None:
        if not self.active or self.current_step != self.capture_step:
            return
        capture_heads = self.capture_by_block.get(block, ())
        if not capture_heads:
            return
        if self.current_grid != (LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH):
            raise RuntimeError(f"unexpected capture grid: {self.current_grid}")
        self._rows(q.device)
        if not self.query_token_indices_by_latent_frame:
            raise RuntimeError("temporal R tube was not resolved before capture")
        num_heads = int(q.shape[-1] // 128)
        head_dim = int(q.shape[-1] // num_heads)
        if q.shape[1] != SEQUENCE:
            raise RuntimeError(f"expected {SEQUENCE} attention tokens, got {q.shape[1]}")
        qh = self._head_view(q, num_heads).permute(0, 2, 1, 3)
        kh = self._head_view(k, num_heads).permute(0, 2, 1, 3)
        selected_q = qh[:, capture_heads]
        selected_k_t = kh[:, capture_heads].transpose(-1, -2)
        selected_for_ablation = set(self.by_block.get(block, ()))
        local_offsets = [
            offset for offset, head in enumerate(capture_heads) if head in selected_for_ablation
        ]
        scale = 1.0 / math.sqrt(head_dim)
        for region in self.capture_regions:
            indices = torch.as_tensor(
                region["token_indices"], device=q.device, dtype=torch.long
            )
            probabilities = torch.softmax(
                torch.matmul(selected_q[:, :, indices], selected_k_t).float() * scale,
                dim=-1,
            )
            effective = probabilities.clone()
            if local_offsets:
                mask = self._delete_masks.get(region["name"])
                if mask is None or mask.device != q.device:
                    mask = torch.from_numpy(
                        build_delete_mask(
                            self.query_token_indices_by_latent_frame,
                            FRAME_TOKEN_COUNT,
                            region["token_indices"],
                            self.mask_mode,
                        )
                    ).to(q.device)
                    self._delete_masks[region["name"]] = mask
                head_offsets = torch.as_tensor(local_offsets, device=q.device, dtype=torch.long)
                effective[:, head_offsets] = effective[:, head_offsets].masked_fill(
                    mask[None, None], 0.0
                )
            else:
                mask = torch.zeros(
                    (len(region["token_indices"]), SEQUENCE),
                    device=q.device,
                    dtype=torch.bool,
                )
            before_map = probabilities.sum(dim=2).reshape(
                probabilities.shape[0], len(capture_heads), LATENT_FRAMES,
                LATENT_HEIGHT, LATENT_WIDTH
            ).mean(dim=0)
            after_map = effective.sum(dim=2).reshape(
                effective.shape[0], len(capture_heads), LATENT_FRAMES,
                LATENT_HEIGHT, LATENT_WIDTH
            ).mean(dim=0)
            entry = self.capture_sums.setdefault(
                region["name"],
                {
                    "before": torch.zeros((LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)),
                    "effective_after": torch.zeros(
                        (LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
                    ),
                    "head_instances": 0,
                    "locally_ablated_head_instances": 0,
                    "locally_ablated_query_rows": int(mask.any(dim=1).sum().item()),
                    "region": region,
                },
            )
            entry["before"] += before_map.sum(dim=0).detach().cpu()
            entry["effective_after"] += after_map.sum(dim=0).detach().cpu()
            entry["head_instances"] += len(capture_heads)
            entry["locally_ablated_head_instances"] += len(local_offsets)
        for head in capture_heads:
            key = (block, int(head))
            self.capture_head_calls[key] = self.capture_head_calls.get(key, 0) + 1

    def _attention(self, q, k, v, original, block: int):
        self._capture_attention(q, k, block)
        return super()._attention(q, k, v, original, block)

    def capture_audit(self) -> dict:
        if len(self.capture_head_calls) != 100:
            raise RuntimeError(f"captured {len(self.capture_head_calls)}/100 physical heads")
        bad = {key: count for key, count in self.capture_head_calls.items() if count != 2}
        if bad:
            raise RuntimeError(f"S039 expected two CFG calls per Top100 head, got {bad}")
        for name, entry in self.capture_sums.items():
            if int(entry["head_instances"]) != 200:
                raise RuntimeError(f"{name}: expected 200 head instances")
        return {
            "capture_step": self.capture_step,
            "physical_top100_heads": 100,
            "cfg_calls_per_head": 2,
            "head_instances_per_object": 200,
            "captured_regions": sorted(self.capture_sums),
        }

    def flush(self, output: Path, task: dict, source_video: Path) -> dict:
        output.mkdir(parents=True, exist_ok=True)
        audit = self.capture_audit()
        formula = task_formula(task)
        for region_name, entry in self.capture_sums.items():
            count = float(entry["head_instances"])
            before = (entry["before"] / count).numpy()
            after = (entry["effective_after"] / count).numpy()
            region = entry["region"]
            path = output / f"{region_name}.npz"
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    before=before,
                    effective_after=after,
                    removed=np.maximum(before - after, 0.0),
                    query_context_frame=self.capture_context_frame,
                    query_mask=region["mask"],
                    query_points=region["points"],
                    query_token_indices=region["token_indices"],
                    query_pixel_frame=np.int32(4),
                    query_latent_frame=np.int32(1),
                    region_name=np.asarray(region_name),
                    region_phrase=np.asarray(region["phrase"]),
                    experiment_id=np.asarray(temporal_variant_id(task)),
                    operator_id=np.asarray(formula["operator_id"]),
                    temporal_scope=np.asarray(formula["temporal_scope"]),
                    head_scope=np.asarray(task["head_scope"]),
                    intervention_head_count=np.int32(task["top_n"]),
                    locally_ablated_top100_heads=np.int32(
                        entry["locally_ablated_head_instances"] // 2
                    ),
                    locally_ablated_query_rows=np.int32(
                        entry["locally_ablated_query_rows"]
                    ),
                    step=np.int32(self.capture_step),
                    seed=np.int32(task["seed"]),
                    protocol=np.asarray("m123_fixed_f04_s039_top100_effective_coefficients_v1"),
                    query_aggregation=np.asarray(
                        "sum8_queries_then_mean100_heads_and_2_cfg_calls"
                    ),
                )
            temporary.replace(path)
        overlay = render_variant(output, source_video)
        return {"capture_audit": audit, "formula": formula, "overlay": overlay}


def validate_source_task(task: dict, root: Path, ranking_entries: list[dict]) -> tuple[Path, dict]:
    source = ablation_task_root(task, root)
    video = source / "generated.mp4"
    manifest_path = source / "manifest.json"
    complete_path = source / "complete.json"
    if not all(path.is_file() for path in (video, manifest_path, complete_path)):
        raise FileNotFoundError(f"source ablation is incomplete: {source}")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_entries = selected_head_entries(ranking_entries, str(task["head_scope"]))
    checks = {
        "case": task["case"],
        "seed": task["seed"],
        "target_scope": task["target_scope"],
        "mask_mode": task["mask_mode"],
    }
    if any(metadata.get(key) != value for key, value in checks.items()):
        raise RuntimeError(f"source manifest identity mismatch: {source}")
    if str(metadata.get("region") or "") != str(task.get("region") or ""):
        raise RuntimeError(f"source manifest region mismatch: {source}")
    if str(metadata.get("head_scope", "top100")) != str(task["head_scope"]):
        raise RuntimeError(f"source manifest head-scope mismatch: {source}")
    if metadata.get("selected_entries") != expected_entries:
        raise RuntimeError(f"source manifest ranking mismatch: {source}")
    return video, metadata


def write_experiment_list(
    capture_root: Path,
    tasks: list[dict],
    ablation_root: Path,
    ranking_entries: list[dict],
) -> dict:
    capture_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, task in enumerate(tasks, start=1):
        source = ablation_task_root(task, ablation_root)
        formula = task_formula(task)
        rows.append(
            {
                "id": index,
                "variant_id": temporal_variant_id(task),
                **task,
                **formula,
                "source_dir": str(source),
                "source_video": str(source / "generated.mp4"),
                "source_ready": all(
                    (source / name).is_file()
                    for name in ("generated.mp4", "manifest.json", "complete.json")
                ),
                "capture_dir": str(capture_task_root(capture_root, task)),
                "top100_observation_heads": [
                    {"block": int(row["block"]), "head": int(row["head"])}
                    for row in ranking_entries[:100]
                ],
            }
        )
    payload = {
        "case": CASE,
        "seed": SEED,
        "capture_step": CAPTURE_STEP,
        "query_definition": "fixed Object A/B F04 SAM2 query points; 8 points per object",
        "observation_head_definition": "frozen S039 PCK Top100, independent of intervention head scope",
        "matrix_dimensions": {
            "targets": 3,
            "head_scopes": 3,
            "operators": 3,
            "temporal_scopes": 4,
            "total": 108,
        },
        "source_ready": sum(int(row["source_ready"]) for row in rows),
        "experiments": rows,
    }
    (capture_root / "experiment_list.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# S039 M1/M2/M3 Head-Scope Top100 Mean Overlay — 108 Experiments",
        "",
        f"- Case: `{CASE}`",
        f"- Seed: `{SEED}`",
        "- Matrix: `3 targets × 3 head scopes × 3 operators × 4 temporal scopes = 108`",
        "- Observation: fixed F04 Object A/B queries, S039, frozen PCK Top100 heads",
        "- Effective after: exact deleted attention coefficients are set to zero; no renormalization",
        "",
        "| # | Target | Intervention heads | ID | Time | Implementation | Variant | Ready |",
        "|---:|---|---|---|---|---|---|:---:|",
    ]
    for row in rows:
        target = row["region"] or "all_objects"
        md.append(
            f"| {row['id']} | `{target}` | `{row['head_scope']}` | {row['operator_id']} | "
            f"{row['temporal_scope']} | `{row['mask_mode']}` | `{row['variant_id']}` | "
            f"{'✓' if row['source_ready'] else '✗'} |"
        )
    (capture_root / "EXPERIMENT_LIST.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=CASE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--head-ranking-path", type=Path, default=DEFAULT_HEAD_RANKING)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--query-cache", type=Path, default=F04_QUERY_CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    if (cfg.case, cfg.seed) != (CASE, SEED):
        raise ValueError(f"this 108-capture worker is fixed to {CASE}/seed {SEED}")
    if not 0 <= cfg.worker_id < cfg.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    sample = next(
        (
            row for row in manifest["samples"]
            if str(row["case"]) == cfg.case and int(row["seed"]) == cfg.seed
        ),
        None,
    )
    if sample is None:
        raise KeyError(f"manifest has no {cfg.case}/seed {cfg.seed}")
    ranking = json.loads(cfg.head_ranking_path.read_text(encoding="utf-8"))
    ranking_entries = validate_head_ranking(manifest, ranking)
    tasks = exact_tasks(sample)
    listing = write_experiment_list(cfg.capture_root, tasks, cfg.ablation_root, ranking_entries)
    if listing["source_ready"] != 108:
        raise RuntimeError(f"only {listing['source_ready']}/108 source ablations are complete")
    if cfg.list_only:
        print(cfg.capture_root / "EXPERIMENT_LIST.md")
        return
    if cfg.task_index is not None:
        if not 0 <= cfg.task_index < len(tasks):
            raise ValueError("task-index must be in [0,108)")
        tasks = [tasks[cfg.task_index]]
    else:
        tasks = tasks[cfg.worker_id :: cfg.num_workers]
    if not tasks:
        return

    from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import CASES

    case_lookup = {case.key: case for case in CASES}
    track_path = prepare_tracks(
        sample, case_lookup, cfg.ablation_root, str(cfg.device), overwrite=False
    )
    fixed_regions, fixed_context = load_fixed_f04_regions(cfg.query_cache)
    pipe = build_wan_ti2v_pipeline(build_args(cfg.seed))
    for progress, task in enumerate(tasks, start=1):
        variant = temporal_variant_id(task)
        output = capture_task_root(cfg.capture_root, task)
        ready = all(
            (output / name).is_file()
            for name in (
                "complete.json",
                "manifest.json",
                "overlay_manifest.json",
                "object_A__s039_top100_mean_comparison.jpg",
                "object_B__s039_top100_mean_comparison.jpg",
            )
        )
        if ready and not cfg.overwrite:
            print(f"[{progress}/{len(tasks)}] skip {variant}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        (output / "complete.json").unlink(missing_ok=True)
        (output / "error.txt").unlink(missing_ok=True)
        print(f"[{progress}/{len(tasks)}] start {variant}", flush=True)
        try:
            source_video, source_manifest = validate_source_task(
                task, cfg.ablation_root, ranking_entries
            )
            json_path, base_cache_dir, payload, run_args, image = generation_inputs(
                sample, case_lookup, cfg.seed
            )
            base_cache = load_region_cache(base_cache_dir.parent, base_cache_dir.name)
            points, query_regions = object_queries(base_cache)
            region_slices = {
                region.region_name: point_slice
                for region, point_slice in query_regions
            }
            with np.load(track_path, allow_pickle=False) as arrays:
                tracks = arrays["tracks"].astype(np.float32)
                anchors = arrays["anchor_pixel_frames"].astype(np.int64)
            entries = selected_head_entries(ranking_entries, str(task["head_scope"]))
            ablator = S039Top100MeanAblator(
                pipe.pipe,
                entries,
                points,
                region_slices,
                (704, 1280),
                str(task["target_scope"]),
                str(task["mask_mode"]),
                task.get("region"),
                tracks=tracks,
                anchor_frames=anchors,
                capture_entries=ranking_entries[:100],
                capture_regions=fixed_regions,
                capture_context_frame=fixed_context,
            )
            ablator.install()
            try:
                replay_video = generate_video(pipe, payload, run_args, image, cfg.seed)
            finally:
                ablator.remove()
            ablation_audit = ablator.audit()
            capture_payload = ablator.flush(output, task, source_video)
            metadata = {
                **task,
                "variant_id": variant,
                "capture_protocol": "m123_fixed_f04_s039_top100_effective_coefficients_v1",
                "capture_step": CAPTURE_STEP,
                "source_ablation_dir": str(ablation_task_root(task, cfg.ablation_root)),
                "source_video": str(source_video),
                "source_protocol": source_manifest.get("protocol"),
                "source_input_json": str(json_path),
                "f04_query_cache": str(cfg.query_cache),
                "observation_heads": "frozen S039 PCK Top100 ranks 1-100",
                "intervention_heads": str(task["head_scope"]),
                "before_definition": (
                    "live A=softmax(QK^T/sqrt(d)) at S039 in the replayed ablation run"
                ),
                "effective_after_definition": (
                    "before coefficients with exact M1/M2/M3 entries set to zero only "
                    "for heads selected by the intervention scope; no renormalization"
                ),
                "fixed_query_definition": "8 SAM2 Object Query points at F04 (latent tq=1)",
                "key_frames": list(range(0, 49, 4)),
                "cfg_branches_averaged": ["conditional", "unconditional"],
                "replay_video_shape": list(np.asarray(replay_video).shape),
                "ablation_audit": ablation_audit,
                **capture_payload,
            }
            (output / "manifest.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (output / "complete.json").write_text(
                json.dumps(
                    {
                        "variant_id": variant,
                        "case": cfg.case,
                        "seed": cfg.seed,
                        "step": CAPTURE_STEP,
                        "regions": ["object_A", "object_B"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            del replay_video
        except Exception:
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{progress}/{len(tasks)}] complete {variant}", flush=True)


if __name__ == "__main__":
    main()
