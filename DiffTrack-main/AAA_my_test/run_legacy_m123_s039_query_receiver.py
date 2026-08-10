#!/usr/bin/env python3
"""Replay 108 M1/M2/M3 runs and capture S039 query-side receiver maps."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from AAA_my_test.render_m123_s039_query_receiver_overlay import render_receiver
from AAA_my_test.run_legacy_m123_head_scope_s039_top100_mean import (
    CAPTURE_STEP,
    CASE,
    FRAME_TOKEN_COUNT,
    LATENT_FRAMES,
    LATENT_HEIGHT,
    LATENT_WIDTH,
    SEED,
    capture_task_root,
    exact_tasks,
    mode_parts,
    task_formula,
    validate_source_task,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import object_queries
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (
    AttentionMatrixAblator,
    build_args,
    generate_video,
    generation_inputs,
)
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (
    DEFAULT_HEAD_RANKING,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT as DEFAULT_ABLATION_ROOT,
    TemporalObjectTubeAblator,
    prepare_tracks,
    selected_head_entries,
    temporal_variant_id,
    validate_head_ranking,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache
from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import build_wan_ti2v_pipeline


DEFAULT_CAPTURE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_attention_overlays/"
    "m123_head_scope_s039_query_receiver_v1"
)


def receiver_groups(
    tokens_by_time: list[list[int]],
    frame_token_count: int,
    mask_mode: str,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return exact (receiver Q rows, selected source K/V rows) groups."""
    _, temporal, target_partition, source_partition = mode_parts(mask_mode)
    partitions: dict[str, list[torch.Tensor]] = {"R": [], "C": []}
    for time_index, values in enumerate(tokens_by_time):
        start = time_index * frame_token_count
        frame_rows = torch.arange(
            start, start + frame_token_count, device=device, dtype=torch.long
        )
        r_rows = torch.as_tensor(values, device=device, dtype=torch.long)
        c_rows = frame_rows[~torch.isin(frame_rows, r_rows)] if r_rows.numel() else frame_rows
        partitions["R"].append(r_rows)
        partitions["C"].append(c_rows)

    groups = []
    time_count = len(tokens_by_time)
    for target_time in range(time_count):
        if temporal == "only":
            source_times = range(time_count)
        elif temporal == "same":
            source_times = (target_time,)
        elif temporal == "future":
            source_times = range(target_time)
        elif temporal == "past":
            source_times = range(target_time + 1, time_count)
        else:
            raise ValueError(f"unsupported temporal scope: {temporal}")
        source_parts = [partitions[source_partition][index] for index in source_times]
        target_rows = partitions[target_partition][target_time]
        if not source_parts or not target_rows.numel():
            continue
        source_rows = torch.cat(source_parts)
        if source_rows.numel():
            groups.append((target_rows, source_rows))
    return groups


def attention_receiver_values(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    original,
    heads: tuple[int, ...] | list[int],
    num_heads: int,
    target_rows: torch.Tensor,
    source_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return per-query sums of S(q), E(q), and their head/CFG instance count."""
    indicator_v = torch.zeros_like(v)
    indicator_heads = AttentionMatrixAblator._head_view(indicator_v, num_heads)
    for head in heads:
        indicator_heads[:, source_rows, head, :] = 1
    coefficient_output = original(q[:, target_rows, :], k, indicator_v)

    selected_v = AttentionMatrixAblator._selected_values(v, source_rows, heads, num_heads)
    value_output = original(q[:, target_rows, :], k, selected_v)

    head_indices = torch.as_tensor(heads, device=q.device, dtype=torch.long)
    coefficient_heads = AttentionMatrixAblator._head_view(
        coefficient_output, num_heads
    )[:, :, head_indices]
    value_heads = AttentionMatrixAblator._head_view(value_output, num_heads)[
        :, :, head_indices
    ]
    coefficient_sum = coefficient_heads.mean(dim=-1).sum(dim=(0, 2))
    value_norm_sum = torch.linalg.vector_norm(value_heads.float(), dim=-1).sum(dim=(0, 2))
    instances = int(q.shape[0] * len(heads))
    return coefficient_sum, value_norm_sum, instances


class S039QueryReceiverAblator(TemporalObjectTubeAblator):
    """Capture query-side S/E on the same heads used by each intervention."""

    def __init__(self, *args, capture_step: int = CAPTURE_STEP, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.capture_step = int(capture_step)
        self.receiver_coefficient_sum = torch.zeros(
            LATENT_FRAMES * FRAME_TOKEN_COUNT, dtype=torch.float64
        )
        self.receiver_value_norm_sum = torch.zeros_like(self.receiver_coefficient_sum)
        self.receiver_head_instances = 0
        self.receiver_auxiliary_calls = 0

    def _capture_receiver(self, q, k, v, original, block: int) -> None:
        if not self.active or self.current_step != self.capture_step:
            return
        heads = self.by_block.get(block, ())
        if not heads:
            return
        if self.current_grid != (LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH):
            raise RuntimeError(f"unexpected capture grid: {self.current_grid}")
        self._rows(q.device)
        if not self.query_token_indices_by_latent_frame:
            raise RuntimeError("temporal R tube was not resolved before receiver capture")
        num_heads = int(q.shape[-1] // 128)
        groups = receiver_groups(
            self.query_token_indices_by_latent_frame,
            FRAME_TOKEN_COUNT,
            self.mask_mode,
            q.device,
        )
        for target_rows, source_rows in groups:
            coefficient, value_norm, _ = attention_receiver_values(
                q,
                k,
                v,
                original,
                heads,
                num_heads,
                target_rows,
                source_rows,
            )
            cpu_rows = target_rows.detach().cpu()
            self.receiver_coefficient_sum[cpu_rows] += coefficient.detach().double().cpu()
            self.receiver_value_norm_sum[cpu_rows] += value_norm.detach().double().cpu()
            self.receiver_auxiliary_calls += 2
        self.receiver_head_instances += int(q.shape[0] * len(heads))

    def _attention(self, q, k, v, original, block: int):
        self._capture_receiver(q, k, v, original, block)
        return super()._attention(q, k, v, original, block)

    def flush(self, output: Path, task: dict[str, Any], source_video: Path) -> dict:
        expected_instances = 2 * len(self.entries)
        if self.receiver_head_instances != expected_instances:
            raise RuntimeError(
                f"captured {self.receiver_head_instances} receiver head instances, "
                f"expected {expected_instances}"
            )
        if not self.query_token_indices_by_latent_frame:
            raise RuntimeError("missing temporal R tube rows")
        formula = task_formula(task)
        coefficient = (
            self.receiver_coefficient_sum / self.receiver_head_instances
        ).reshape(LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
        value_norm = (
            self.receiver_value_norm_sum / self.receiver_head_instances
        ).reshape(LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
        r_tube_mask = np.zeros(
            (LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH), dtype=np.uint8
        )
        for time_index, rows in enumerate(self.query_token_indices_by_latent_frame):
            local = np.asarray(rows, dtype=np.int64) - time_index * FRAME_TOKEN_COUNT
            r_tube_mask[time_index].flat[local] = 1

        output.mkdir(parents=True, exist_ok=True)
        receiver_path = output / "receiver.npz"
        temporary = receiver_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                coefficient_mass=coefficient.float().numpy(),
                value_contribution_norm=value_norm.float().numpy(),
                r_tube_mask=r_tube_mask,
                experiment_id=np.asarray(temporal_variant_id(task)),
                operator_id=np.asarray(formula["operator_id"]),
                temporal_scope=np.asarray(formula["temporal_scope"]),
                head_scope=np.asarray(task["head_scope"]),
                intervention_head_count=np.int32(task["top_n"]),
                target_partition=np.asarray(formula["target_partition"]),
                source_partition=np.asarray(formula["source_partition"]),
                time_predicate=np.asarray(formula["time_predicate"]),
                step=np.int32(self.capture_step),
                seed=np.int32(task["seed"]),
                protocol=np.asarray("m123_s039_query_receiver_coefficient_and_value_v1"),
            )
        temporary.replace(receiver_path)
        overlay = render_receiver(output, source_video)
        return {
            "capture_step": self.capture_step,
            "receiver_head_instances": self.receiver_head_instances,
            "receiver_auxiliary_attention_calls": self.receiver_auxiliary_calls,
            "coefficient_definition": "mean_heads_cfg sum_selected_source_keys A[q,k]",
            "value_definition": "mean_heads_cfg norm2(sum_selected_source_keys A[q,k]V[k])",
            "overlay": overlay,
        }


def write_experiment_list(capture_root: Path, tasks: list[dict]) -> None:
    capture_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": index,
            "variant_id": temporal_variant_id(task),
            **task,
            **task_formula(task),
        }
        for index, task in enumerate(tasks, start=1)
    ]
    payload = {
        "case": CASE,
        "seed": SEED,
        "capture_step": CAPTURE_STEP,
        "total": len(rows),
        "protocol": "m123_s039_query_receiver_coefficient_and_value_v1",
        "experiments": rows,
    }
    (capture_root / "experiment_list.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=CASE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--head-ranking-path", type=Path, default=DEFAULT_HEAD_RANKING)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
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
        raise ValueError(f"this worker is fixed to {CASE}/seed {SEED}")
    if not 0 <= cfg.worker_id < cfg.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    sample = next(
        (
            row
            for row in manifest["samples"]
            if str(row["case"]) == cfg.case and int(row["seed"]) == cfg.seed
        ),
        None,
    )
    if sample is None:
        raise KeyError(f"manifest has no {cfg.case}/seed {cfg.seed}")
    ranking = json.loads(cfg.head_ranking_path.read_text(encoding="utf-8"))
    ranking_entries = validate_head_ranking(manifest, ranking)
    tasks = exact_tasks(sample)
    write_experiment_list(cfg.capture_root, tasks)
    if cfg.list_only:
        print(cfg.capture_root / "experiment_list.json")
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
                "receiver.npz",
                "receiver__s039_query_side_comparison.jpg",
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
            json_path, cache_dir, payload, run_args, image = generation_inputs(
                sample, case_lookup, cfg.seed
            )
            cache = load_region_cache(cache_dir.parent, cache_dir.name)
            points, query_regions = object_queries(cache)
            region_slices = {
                region.region_name: point_slice for region, point_slice in query_regions
            }
            with np.load(track_path, allow_pickle=False) as arrays:
                tracks = arrays["tracks"].astype(np.float32)
                anchors = arrays["anchor_pixel_frames"].astype(np.int64)
            entries = selected_head_entries(ranking_entries, str(task["head_scope"]))
            ablator = S039QueryReceiverAblator(
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
            )
            ablator.install()
            try:
                replay_video = generate_video(pipe, payload, run_args, image, cfg.seed)
            finally:
                ablator.remove()
            ablation_audit = ablator.audit()
            receiver_payload = ablator.flush(output, task, source_video)
            metadata = {
                **task,
                "variant_id": variant,
                "capture_protocol": "m123_s039_query_receiver_coefficient_and_value_v1",
                "capture_step": CAPTURE_STEP,
                "source_video": str(source_video),
                "source_protocol": source_manifest.get("protocol"),
                "source_input_json": str(json_path),
                "receiver_query_frames": list(range(0, 49, 4)),
                "intervention_heads": str(task["head_scope"]),
                "cfg_branches_averaged": ["conditional", "unconditional"],
                "replay_video_shape": list(np.asarray(replay_video).shape),
                "ablation_audit": ablation_audit,
                **receiver_payload,
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
