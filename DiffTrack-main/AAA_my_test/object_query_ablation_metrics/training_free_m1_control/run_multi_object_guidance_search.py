#!/usr/bin/env python3
"""Run Top100 guidance or direct generation with an audited attention perturbation.

The default ``m1_multi_object_blockdiag`` mode preserves the original experiment.
For object-token tubes R_i, its perturbed conditional forward subtracts

    sum_i A[R_i, R_i] V[R_i]

from the corresponding query rows.  Cross-object pairs R_i <- R_j (i != j)
remain untouched.  If two objects map to the same latent token, deleted token
pairs are set-unioned so no A_qk V_k term is subtracted twice.

The ``full_head_output_zero`` control instead sets the complete post-attention
output O_h=A_hV_h of every selected physical head to zero at every query token.
It therefore removes R->R, C->R, R->C, and C->C together before the heads are
concatenated and passed through the attention output projection.

The M2/M3 multi-object controls use the same object-wise union semantics as M1:

    M2: union_i R_i x C_i, where C_i = Omega \\ R_i
    M3: union_i C_i x R_i

All object-specific blocks are removed in one perturbed forward.  Duplicate
token pairs caused by overlapping object tubes are removed exactly once.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import imageio.v3 as iio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
for import_root in (REPO_ROOT, CODE_ROOT, COTRACKER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (  # noqa: E402
    _run_pipe_once,
    build_wan_ti2v_pipeline,
    save_video_np,
)
from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import CASES  # noqa: E402
from AAA_my_test.object_query_ablation_metrics.run_top100_m1_perturbed_attention_guidance import (  # noqa: E402
    adjusted_conditional_prediction,
    scale_tag,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (  # noqa: E402
    load_cotracker,
    run_cotracker,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (  # noqa: E402
    AttentionMatrixAblator,
    generation_inputs,
)
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (  # noqa: E402
    TemporalObjectTubeAblator,
    atomic_npz,
    selected_head_entries,
    sha256_file,
    validate_head_ranking,
)


EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1"
)
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "training_free_m1_multi_object_search_v1"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "search_manifest.json"
DEFAULT_HEAD_RANKING = EXPERIMENT_ROOT / "head_scopes_latest3350_with_random100.json"
PROTOCOL = "wan_top100_m1_multi_object_blockdiag_contrast_guidance_v1"
M2_PROTOCOL = "wan_top100_m2_multi_object_independent_contrast_guidance_v1"
M3_PROTOCOL = "wan_top100_m3_multi_object_independent_contrast_guidance_v1"
FULL_HEAD_PROTOCOL = "wan_top100_full_head_output_zero_contrast_guidance_v1"
DIRECT_PROTOCOL = "wan_top100_m1_multi_object_blockdiag_direct_ablation_v1"
M2_DIRECT_PROTOCOL = "wan_top100_m2_multi_object_independent_direct_ablation_v1"
M3_DIRECT_PROTOCOL = "wan_top100_m3_multi_object_independent_direct_ablation_v1"
FULL_HEAD_DIRECT_PROTOCOL = "wan_top100_full_head_output_zero_direct_ablation_v1"
EXECUTION_MODES = ("guidance", "direct")
PERTURBATION_MODES = (
    "m1_multi_object_blockdiag",
    "m2_multi_object_independent",
    "m3_multi_object_independent",
    "full_head_output_zero",
)
MULTI_OBJECT_FLOW_CONFIG = {
    "m1_multi_object_blockdiag": {
        "id": "M1",
        "mask_mode": "self_only",
        "formula": "union_i A[R_i,R_i]V[R_i]",
        "protocol": PROTOCOL,
        "direct_protocol": DIRECT_PROTOCOL,
        "symbol": "eps_M1_multi",
    },
    "m2_multi_object_independent": {
        "id": "M2",
        "mask_mode": "incoming_only",
        "formula": "union_i A[R_i,C_i]V[C_i]",
        "protocol": M2_PROTOCOL,
        "direct_protocol": M2_DIRECT_PROTOCOL,
        "symbol": "eps_M2_multi",
    },
    "m3_multi_object_independent": {
        "id": "M3",
        "mask_mode": "outgoing_only",
        "formula": "union_i A[C_i,R_i]V[R_i]",
        "protocol": M3_PROTOCOL,
        "direct_protocol": M3_DIRECT_PROTOCOL,
        "symbol": "eps_M3_multi",
    },
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_window(start: int, end: int, steps: int = 40) -> None:
    if not 0 <= start <= end < steps:
        raise ValueError(f"invalid denoising window {start}..{end} for {steps} steps")


def block_diagonal_groups(
    object_rows: dict[str, torch.Tensor], device: torch.device
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Return disjoint query groups representing union_i R_i x R_i."""
    normalized = {
        name: tuple(sorted({int(value) for value in rows.detach().cpu().tolist()}))
        for name, rows in object_rows.items()
    }
    if not normalized or any(not rows for rows in normalized.values()):
        raise RuntimeError("every multi-object M1 region must contain at least one token")
    memberships: dict[int, list[str]] = defaultdict(list)
    for name, rows in normalized.items():
        for row in rows:
            memberships[row].append(name)

    targets_by_source: dict[tuple[int, ...], list[int]] = defaultdict(list)
    overlap_rows: dict[int, list[str]] = {}
    for target, names in memberships.items():
        if len(names) > 1:
            overlap_rows[target] = sorted(names)
        sources = tuple(sorted({row for name in names for row in normalized[name]}))
        targets_by_source[sources].append(target)

    groups = [
        (
            torch.as_tensor(sorted(targets), device=device, dtype=torch.long),
            torch.as_tensor(sources, device=device, dtype=torch.long),
        )
        for sources, targets in sorted(
            targets_by_source.items(), key=lambda item: min(item[1])
        )
    ]
    deleted_pairs = sum(int(target.numel() * source.numel()) for target, source in groups)
    naive_pairs = sum(len(rows) ** 2 for rows in normalized.values())
    audit = {
        "object_token_counts": {name: len(rows) for name, rows in normalized.items()},
        "object_token_indices": {name: list(rows) for name, rows in normalized.items()},
        "union_token_count": len(memberships),
        "overlap_token_count": len(overlap_rows),
        "overlap_tokens": {str(row): names for row, names in sorted(overlap_rows.items())},
        "deleted_pair_count_per_head": deleted_pairs,
        "naive_unchecked_pair_count_per_head": naive_pairs,
        "duplicate_pair_subtractions_prevented": naive_pairs - deleted_pairs,
        "group_count": len(groups),
    }
    return groups, audit


def multi_object_flow_groups(
    object_rows: dict[str, torch.Tensor],
    token_count: int,
    flow_id: str,
    device: torch.device,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Return the set-union of per-object M1/M2/M3 query-key blocks."""
    normalized = {
        name: frozenset(int(value) for value in rows.detach().cpu().tolist())
        for name, rows in object_rows.items()
    }
    if not normalized or any(not rows for rows in normalized.values()):
        raise RuntimeError("every multi-object flow region must contain at least one token")
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    universe = frozenset(range(token_count))
    if any(value not in universe for rows in normalized.values() for value in rows):
        raise ValueError("object token index is outside the attention token grid")
    if flow_id not in {"M1", "M2", "M3"}:
        raise ValueError(f"unsupported multi-object flow: {flow_id}")

    memberships: dict[frozenset[str], list[int]] = defaultdict(list)
    overlap_rows: dict[int, list[str]] = {}
    for target in range(token_count):
        names = frozenset(name for name, rows in normalized.items() if target in rows)
        memberships[names].append(target)
        if len(names) > 1:
            overlap_rows[target] = sorted(names)

    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    for names, targets in sorted(memberships.items(), key=lambda item: min(item[1])):
        if flow_id == "M1":
            sources = set().union(*(normalized[name] for name in names)) if names else set()
        elif flow_id == "M2":
            sources = (
                set().union(*(universe - normalized[name] for name in names))
                if names
                else set()
            )
        else:
            sources = set().union(
                *(rows for name, rows in normalized.items() if name not in names)
            )
        if not sources:
            continue
        groups.append(
            (
                torch.as_tensor(targets, device=device, dtype=torch.long),
                torch.as_tensor(sorted(sources), device=device, dtype=torch.long),
            )
        )

    deleted_pairs = sum(int(target.numel() * source.numel()) for target, source in groups)
    if flow_id == "M1":
        naive_pairs = sum(len(rows) ** 2 for rows in normalized.values())
    else:
        naive_pairs = sum(len(rows) * (token_count - len(rows)) for rows in normalized.values())
    audit = {
        "flow_id": flow_id,
        "object_token_counts": {name: len(rows) for name, rows in normalized.items()},
        "object_token_indices": {
            name: sorted(rows) for name, rows in normalized.items()
        },
        "token_count": token_count,
        "union_token_count": len(set().union(*normalized.values())),
        "overlap_token_count": len(overlap_rows),
        "overlap_tokens": {str(row): names for row, names in sorted(overlap_rows.items())},
        "deleted_pair_count_per_head": deleted_pairs,
        "naive_unchecked_pair_count_per_head": naive_pairs,
        "duplicate_pair_subtractions_prevented": naive_pairs - deleted_pairs,
        "group_count": len(groups),
        "object_specific_complements": flow_id in {"M2", "M3"},
    }
    return groups, audit


def apply_grouped_m1_ablation(
    *,
    output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    original: Callable[..., torch.Tensor],
    heads: list[int] | tuple[int, ...],
    num_heads: int,
    groups: list[tuple[torch.Tensor, torch.Tensor]],
    group_batch_size: int,
    dose_recorder: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]
    | None = None,
) -> tuple[int, int, int]:
    """Subtract exact grouped A@V contributions using batched fused-attention calls."""
    if int(q.shape[0]) != 1 or int(k.shape[0]) != 1 or int(v.shape[0]) != 1:
        raise RuntimeError("multi-object grouped M1 currently requires Wan batch size 1")
    if group_batch_size <= 0:
        raise ValueError("group_batch_size must be positive")
    output_heads = AttentionMatrixAblator._head_view(output, num_heads)
    original_heads = output_heads.clone() if dose_recorder is not None else output_heads
    value_heads = AttentionMatrixAblator._head_view(v, num_heads)
    removed_full = torch.zeros_like(output_heads) if dose_recorder is not None else None
    mass_full = torch.zeros_like(output_heads) if dose_recorder is not None else None
    target_parts: list[torch.Tensor] = []
    auxiliary_calls = 0

    for offset in range(0, len(groups), group_batch_size):
        chunk = groups[offset : offset + group_batch_size]
        max_targets = max(int(target.numel()) for target, _ in chunk)
        padded_targets = []
        for target_rows, _ in chunk:
            padding = max_targets - int(target_rows.numel())
            if padding:
                target_rows = torch.cat((target_rows, target_rows[:1].repeat(padding)))
            padded_targets.append(target_rows)
        query_batch = torch.stack([q[0, rows] for rows in padded_targets], dim=0)
        key_batch = k.expand(len(chunk), -1, -1)
        selected_values = torch.zeros(
            (len(chunk), int(v.shape[1]), int(v.shape[2])),
            device=v.device,
            dtype=v.dtype,
        )
        selected_heads = AttentionMatrixAblator._head_view(selected_values, num_heads)
        for group_index, (_, source_rows) in enumerate(chunk):
            for head in heads:
                selected_heads[group_index, source_rows, head, :] = value_heads[
                    0, source_rows, head, :
                ]
        contribution = original(query_batch, key_batch, selected_values)
        auxiliary_calls += 1
        contribution_heads = AttentionMatrixAblator._head_view(contribution, num_heads)

        mass_heads = None
        if dose_recorder is not None:
            selected_values.zero_()
            for group_index, (_, source_rows) in enumerate(chunk):
                for head in heads:
                    selected_heads[group_index, source_rows, head, :] = 1
            mass = original(query_batch, key_batch, selected_values)
            auxiliary_calls += 1
            mass_heads = AttentionMatrixAblator._head_view(mass, num_heads)

        for group_index, (target_rows, _) in enumerate(chunk):
            target_count = int(target_rows.numel())
            for head in heads:
                delta = contribution_heads[group_index, :target_count, head, :]
                output_heads[0, target_rows, head, :] -= delta
                if removed_full is not None:
                    removed_full[0, target_rows, head, :] = delta
                    assert mass_full is not None and mass_heads is not None
                    mass_full[0, target_rows, head, :] = mass_heads[
                        group_index, :target_count, head, :
                    ]
            target_parts.append(target_rows)

    if dose_recorder is not None:
        assert removed_full is not None and mass_full is not None
        dose_recorder(
            torch.cat(target_parts), removed_full, original_heads, mass_full
        )
    affected_rows = sum(int(target.numel()) for target, _ in groups)
    deleted_pairs = sum(int(target.numel() * source.numel()) for target, source in groups)
    return auxiliary_calls, affected_rows, deleted_pairs


class MultiObjectIndependentFlowAblator(TemporalObjectTubeAblator):
    """Delete the set-union of per-object M1, M2, or M3 attention blocks."""

    def __init__(
        self,
        *args,
        object_regions: Iterable[str],
        flow_id: str = "M1",
        group_batch_size: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        expected_mode = {"M1": "self_only", "M2": "incoming_only", "M3": "outgoing_only"}.get(
            flow_id
        )
        if expected_mode is None:
            raise ValueError(f"unsupported multi-object flow: {flow_id}")
        if self.mask_mode != expected_mode or self.target_scope != "all_objects":
            raise ValueError(
                f"multi-object {flow_id} requires all_objects/{expected_mode}"
            )
        self.flow_id = flow_id
        self.object_regions = tuple(str(name) for name in object_regions)
        if not self.object_regions:
            raise ValueError("multi-object flow needs at least one object region")
        if any(name not in self.region_slices for name in self.object_regions):
            raise ValueError("object region missing from region_slices")
        self.group_batch_size = int(group_batch_size)
        self.object_token_indices: dict[str, list[int]] | None = None
        self.object_token_indices_by_latent_frame: dict[str, list[list[int]]] | None = None
        self.block_diagonal_group_audit: dict[str, Any] | None = None
        self.temporal_zeroed_entries_per_head = None

    def _object_rows(self, device: torch.device) -> dict[str, torch.Tensor]:
        if self.current_grid is None:
            raise RuntimeError("attention grid is unavailable")
        time, height, width = self.current_grid
        if time != len(self.anchor_frames):
            raise RuntimeError("latent-time and frozen-track anchors disagree")
        tracks = self.tracks[self.anchor_frames].to(device)
        offsets = torch.arange(time, device=device, dtype=torch.long)[:, None] * height * width
        result: dict[str, torch.Tensor] = {}
        by_time: dict[str, list[list[int]]] = {}
        for name in self.object_regions:
            region_tracks = tracks[:, self.region_slices[name]]
            x = torch.floor(region_tracks[..., 0] * width / self.pixel_width).long().clamp(0, width - 1)
            y = torch.floor(region_tracks[..., 1] * height / self.pixel_height).long().clamp(0, height - 1)
            tokens = y * width + x + offsets
            rows = torch.unique(tokens.flatten(), sorted=True)
            result[name] = rows
            by_time[name] = [
                [int(value) for value in torch.unique(frame, sorted=True).detach().cpu().tolist()]
                for frame in tokens
            ]
        groups, audit = multi_object_flow_groups(
            result,
            time * height * width,
            self.flow_id,
            device,
        )
        del groups
        values = {
            name: [int(value) for value in rows.detach().cpu().tolist()]
            for name, rows in result.items()
        }
        union = sorted({value for rows in values.values() for value in rows})
        if self.object_token_indices is None:
            self.object_token_indices = values
            self.object_token_indices_by_latent_frame = by_time
            self.block_diagonal_group_audit = audit
            self.query_token_indices = union
            union_by_time = []
            for time_index in range(time):
                union_by_time.append(
                    sorted(
                        {
                            value
                            for name in self.object_regions
                            for value in by_time[name][time_index]
                        }
                    )
                )
            self.query_token_indices_by_latent_frame = union_by_time
        elif self.object_token_indices != values:
            raise RuntimeError("multi-object token mapping changed during generation")
        return result

    def _rows(self, device: torch.device) -> torch.Tensor:
        object_rows = self._object_rows(device)
        return torch.unique(torch.cat(list(object_rows.values())), sorted=True)

    def _attention(self, q, k, v, original, block: int):
        heads = self.by_block.get(block, ())
        if not self.active or not heads:
            return original(q, k, v)
        num_heads = int(q.shape[-1] // 128)
        if num_heads <= 0 or q.shape[-1] % num_heads:
            raise RuntimeError(f"query width {q.shape[-1]} is not head-aligned")
        object_rows = self._object_rows(q.device)
        groups, audit = multi_object_flow_groups(
            object_rows,
            int(q.shape[1]),
            self.flow_id,
            q.device,
        )
        if self.block_diagonal_group_audit != audit:
            raise RuntimeError("multi-object flow group definition changed")
        output = original(q, k, v)
        dose_recorder = None
        if self.record_dose:
            dose_recorder = lambda target, removed, original_output, mass: (
                self._record_removed_dose(
                    block, heads, target, removed, original_output, mass
                )
            )
        auxiliary_calls, affected_rows, deleted_pairs = apply_grouped_m1_ablation(
            output=output,
            q=q,
            k=k,
            v=v,
            original=original,
            heads=heads,
            num_heads=num_heads,
            groups=groups,
            group_batch_size=self.group_batch_size,
            dose_recorder=dose_recorder,
        )
        if self.temporal_zeroed_entries_per_head is None:
            self.temporal_zeroed_entries_per_head = deleted_pairs
        elif self.temporal_zeroed_entries_per_head != deleted_pairs:
            raise RuntimeError("multi-object deleted pair count changed")
        self.auxiliary_attention_calls += auxiliary_calls
        self.temporal_auxiliary_attention_calls += auxiliary_calls
        self.modified_forward_calls += 1
        self.modified_head_events += len(heads)
        self.affected_query_vectors += output.shape[0] * affected_rows * len(heads)
        return output

    def block_audit(self) -> dict[str, Any]:
        if self.object_token_indices is None or self.block_diagonal_group_audit is None:
            raise RuntimeError("multi-object rows were never resolved")
        return {
            **self.block_diagonal_group_audit,
            "object_token_indices_by_latent_frame": self.object_token_indices_by_latent_frame,
            "union_token_indices_by_latent_frame": self.query_token_indices_by_latent_frame,
            "temporal_zeroed_entries_per_head": self.temporal_zeroed_entries_per_head,
            "group_batch_size": self.group_batch_size,
        }


MultiObjectBlockDiagonalM1Ablator = MultiObjectIndependentFlowAblator


class WindowedMultiObjectM1Guidance:
    """Apply the clean-minus-perturbed conditional contrast in one step window."""

    def __init__(
        self,
        pipe: Any,
        ablator: MultiObjectBlockDiagonalM1Ablator,
        *,
        cfg_scale: float,
        pag_scale: float,
        denoise_start: int,
        denoise_end: int,
        expected_steps: int = 40,
    ) -> None:
        validate_window(denoise_start, denoise_end, expected_steps)
        if not math.isfinite(pag_scale) or pag_scale == 0:
            raise ValueError("pag_scale must be finite and non-zero")
        self.pipe = pipe
        self.ablator = ablator
        self.cfg_scale = float(cfg_scale)
        self.pag_scale = float(pag_scale)
        self.denoise_start = int(denoise_start)
        self.denoise_end = int(denoise_end)
        self.expected_steps = int(expected_steps)
        self._clean_model_fn = None
        self._perturbed_model_fn = None
        self.pipeline_calls_by_step: dict[int, int] = {}
        self.guided_calls_by_step: dict[int, int] = {}
        self.delta_l2_by_step: dict[int, float] = {}

    @property
    def active_steps(self) -> list[int]:
        return list(range(self.denoise_start, self.denoise_end + 1))

    def install(self) -> None:
        self.ablator.install()
        self._clean_model_fn = self.ablator._original_model_fn
        self._perturbed_model_fn = self.pipe.model_fn
        if self._clean_model_fn is None:
            raise RuntimeError("ablator did not capture clean model_fn")
        self.pipe.model_fn = self

    def remove(self) -> None:
        self.ablator.remove()

    def __call__(self, *args, **kwargs):
        if self._clean_model_fn is None or self._perturbed_model_fn is None:
            raise RuntimeError("guidance is not installed")
        timestep = kwargs.get("timestep")
        latents = kwargs.get("latents")
        if timestep is None or latents is None:
            return self._clean_model_fn(*args, **kwargs)
        step = self.ablator._step(timestep)
        branch = self.pipeline_calls_by_step.get(step, 0)
        if branch not in (0, 1):
            raise RuntimeError(f"unexpected CFG call {branch} at step {step}")
        clean = self._clean_model_fn(*args, **kwargs)
        if branch == 0 and self.denoise_start <= step <= self.denoise_end:
            perturbed = self._perturbed_model_fn(*args, **kwargs)
            result = adjusted_conditional_prediction(
                clean,
                perturbed,
                cfg_scale=self.cfg_scale,
                pag_scale=self.pag_scale,
            )
            self.delta_l2_by_step[step] = float(
                torch.linalg.vector_norm((clean - perturbed).detach().float()).cpu()
            )
            self.guided_calls_by_step[step] = self.guided_calls_by_step.get(step, 0) + 1
        else:
            result = clean
        self.pipeline_calls_by_step[step] = branch + 1
        return result

    def audit(self) -> dict[str, Any]:
        all_steps = list(range(self.expected_steps))
        active = self.active_steps
        if sorted(self.pipeline_calls_by_step) != all_steps or any(
            count != 2 for count in self.pipeline_calls_by_step.values()
        ):
            raise RuntimeError(f"invalid pipeline call coverage: {self.pipeline_calls_by_step}")
        if sorted(self.guided_calls_by_step) != active or any(
            count != 1 for count in self.guided_calls_by_step.values()
        ):
            raise RuntimeError(f"invalid guided call coverage: {self.guided_calls_by_step}")
        if sorted(self.ablator.model_call_counts) != active or any(
            count != 1 for count in self.ablator.model_call_counts.values()
        ):
            raise RuntimeError(
                f"invalid perturbed call coverage: {self.ablator.model_call_counts}"
            )
        expected_events = len(self.ablator.entries) * len(active)
        if self.ablator.modified_head_events != expected_events:
            raise RuntimeError(
                f"modified {self.ablator.modified_head_events} head events, "
                f"expected {expected_events}"
            )
        dose_finite = int(np.isfinite(self.ablator.dose_attention_mass).sum())
        if self.ablator.record_dose and dose_finite != expected_events:
            raise RuntimeError(f"dose coverage {dose_finite} != {expected_events}")
        result = {
            "pipeline_calls_by_step": self.pipeline_calls_by_step,
            "guided_calls_by_step": self.guided_calls_by_step,
            "perturbed_calls_by_step": self.ablator.model_call_counts,
            "active_denoising_steps": active,
            "inactive_denoising_steps": [step for step in all_steps if step not in active],
            "modified_head_events": self.ablator.modified_head_events,
            "expected_modified_head_events": expected_events,
            "dose_finite_events": dose_finite,
            "perturbation_delta_l2_by_step": self.delta_l2_by_step,
        }
        if hasattr(self.ablator, "block_audit"):
            flow_audit = self.ablator.block_audit()
            result["multi_object_flow"] = flow_audit
            if getattr(self.ablator, "flow_id", None) == "M1":
                result["block_diagonal"] = flow_audit
        elif (
            self.ablator.target_scope == "all_tokens"
            and self.ablator.mask_mode == "full_head_output"
        ):
            result["all_token_head_output_zero"] = {
                "all_query_tokens": True,
                "selected_head_count": len(self.ablator.entries),
                "removed_flows": ["R->R", "C->R", "R->C", "C->C"],
            }
        else:
            raise RuntimeError("ablator does not provide an audit for this perturbation")
        return result


class WindowedDirectAttentionAblation:
    """Run the perturbed model directly for both CFG branches in one step window."""

    def __init__(
        self,
        pipe: Any,
        ablator: MultiObjectBlockDiagonalM1Ablator,
        *,
        denoise_start: int,
        denoise_end: int,
        expected_steps: int = 40,
    ) -> None:
        validate_window(denoise_start, denoise_end, expected_steps)
        self.pipe = pipe
        self.ablator = ablator
        self.denoise_start = int(denoise_start)
        self.denoise_end = int(denoise_end)
        self.expected_steps = int(expected_steps)
        self._clean_model_fn = None
        self._perturbed_model_fn = None
        self.pipeline_calls_by_step: dict[int, int] = {}
        self.direct_calls_by_step: dict[int, int] = {}

    @property
    def active_steps(self) -> list[int]:
        return list(range(self.denoise_start, self.denoise_end + 1))

    def install(self) -> None:
        self._clean_model_fn = self.pipe.model_fn
        self.ablator.install()
        self._perturbed_model_fn = self.pipe.model_fn
        self.pipe.model_fn = self

    def remove(self) -> None:
        self.ablator.remove()

    def __call__(self, *args, **kwargs):
        if self._clean_model_fn is None or self._perturbed_model_fn is None:
            raise RuntimeError("direct ablation is not installed")
        timestep = kwargs.get("timestep")
        latents = kwargs.get("latents")
        if timestep is None or latents is None:
            return self._clean_model_fn(*args, **kwargs)
        step = self.ablator._step(timestep)
        branch = self.pipeline_calls_by_step.get(step, 0)
        if branch not in (0, 1):
            raise RuntimeError(f"unexpected CFG call {branch} at step {step}")
        if self.denoise_start <= step <= self.denoise_end:
            result = self._perturbed_model_fn(*args, **kwargs)
            self.direct_calls_by_step[step] = self.direct_calls_by_step.get(step, 0) + 1
        else:
            result = self._clean_model_fn(*args, **kwargs)
        self.pipeline_calls_by_step[step] = branch + 1
        return result

    def audit(self) -> dict[str, Any]:
        all_steps = list(range(self.expected_steps))
        active = self.active_steps
        if sorted(self.pipeline_calls_by_step) != all_steps or any(
            count != 2 for count in self.pipeline_calls_by_step.values()
        ):
            raise RuntimeError(f"invalid pipeline call coverage: {self.pipeline_calls_by_step}")
        if sorted(self.direct_calls_by_step) != active or any(
            count != 2 for count in self.direct_calls_by_step.values()
        ):
            raise RuntimeError(f"invalid direct call coverage: {self.direct_calls_by_step}")
        if sorted(self.ablator.model_call_counts) != active or any(
            count != 2 for count in self.ablator.model_call_counts.values()
        ):
            raise RuntimeError(
                f"invalid perturbed call coverage: {self.ablator.model_call_counts}"
            )
        expected_events = len(self.ablator.entries) * len(active) * 2
        if self.ablator.modified_head_events != expected_events:
            raise RuntimeError(
                f"modified {self.ablator.modified_head_events} head events, "
                f"expected {expected_events}"
            )
        dose_finite = int(np.isfinite(self.ablator.dose_attention_mass).sum())
        if self.ablator.record_dose and dose_finite != expected_events:
            raise RuntimeError(f"dose coverage {dose_finite} != {expected_events}")
        result = {
            "pipeline_calls_by_step": self.pipeline_calls_by_step,
            "direct_calls_by_step": self.direct_calls_by_step,
            "perturbed_calls_by_step": self.ablator.model_call_counts,
            "active_denoising_steps": active,
            "inactive_denoising_steps": [step for step in all_steps if step not in active],
            "modified_cfg_branches": ["conditional", "unconditional"],
            "modified_head_events": self.ablator.modified_head_events,
            "expected_modified_head_events": expected_events,
            "dose_finite_events": dose_finite,
        }
        if hasattr(self.ablator, "block_audit"):
            flow_audit = self.ablator.block_audit()
            result["multi_object_flow"] = flow_audit
            if getattr(self.ablator, "flow_id", None) == "M1":
                result["block_diagonal"] = flow_audit
        elif (
            self.ablator.target_scope == "all_tokens"
            and self.ablator.mask_mode == "full_head_output"
        ):
            result["all_token_head_output_zero"] = {
                "all_query_tokens": True,
                "selected_head_count": len(self.ablator.entries),
                "removed_flows": ["R->R", "C->R", "R->C", "C->C"],
            }
        else:
            raise RuntimeError("ablator does not provide an audit for this perturbation")
        return result


def tracks_directory(root: Path, case: str, seed: int) -> Path:
    return root / case / f"seed_{seed:05d}" / "frozen_baseline_tracks"


def load_source_queries(sample: dict[str, Any]) -> tuple[np.ndarray, dict[str, slice], tuple[str, ...]]:
    tube_path = Path(str(sample["source_query_tube"]))
    with np.load(tube_path) as arrays:
        points = arrays["query_points_n2"].astype(np.float32)
        names = tuple(str(value) for value in arrays["region_names"].tolist())
        starts = arrays["point_starts"].astype(np.int64)
        ends = arrays["point_ends"].astype(np.int64)
        height = int(arrays["pixel_height"])
        width = int(arrays["pixel_width"])
    if (height, width) != (704, 1280):
        raise RuntimeError(f"source queries are {height}x{width}, expected 704x1280")
    slices = {
        name: slice(int(start), int(end))
        for name, start, end in zip(names, starts, ends)
    }
    if set(slices) != {
        str(row["region_name"])
        for row in sample.get("regions", [])
        if row.get("region_type") == "object"
    }:
        raise RuntimeError("source tube object slices disagree with search manifest")
    return points, slices, names


def prepare_tracks(
    model: Any,
    sample: dict[str, Any],
    tracks_root: Path,
    device: str,
) -> Path:
    case, seed = str(sample["case"]), int(sample["seed"])
    output = tracks_directory(tracks_root, case, seed)
    path = output / "tracks.npz"
    complete = output / "complete.json"
    if path.is_file() and complete.is_file():
        return path
    baseline = Path(str(sample["baseline_video"]))
    if not baseline.is_file():
        raise FileNotFoundError(f"baseline required before tracking: {baseline}")
    points, slices, names = load_source_queries(sample)
    frames = iio.imread(baseline)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"unexpected baseline shape {frames.shape}: {baseline}")
    tracks, visibility = run_cotracker(model, frames, points, device)
    anchors = np.arange(13, dtype=np.int64) * 4
    if int(anchors[-1]) >= len(frames):
        anchors = np.rint(np.linspace(0, len(frames) - 1, 13)).astype(np.int64)
    if not np.isfinite(tracks[anchors]).all():
        raise RuntimeError(f"{case}/seed_{seed:05d}: non-finite baseline tracks")
    output.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        path,
        tracks=np.asarray(tracks, dtype=np.float32),
        visibility=np.asarray(visibility, dtype=np.bool_),
        anchor_pixel_frames=anchors,
        query_points=points,
        region_names=np.asarray(names),
        point_starts=np.asarray([slices[name].start for name in names], dtype=np.int32),
        point_ends=np.asarray([slices[name].stop for name in names], dtype=np.int32),
        source_video=np.asarray(str(baseline)),
        source_query_tube=np.asarray(str(sample["source_query_tube"])),
        pixel_height=np.int32(frames.shape[1]),
        pixel_width=np.int32(frames.shape[2]),
        seed=np.int32(seed),
    )
    atomic_json(
        output / "manifest.json",
        {
            "case": case,
            "seed": seed,
            "source_video": str(baseline),
            "source_query_tube": str(sample["source_query_tube"]),
            "trajectory_source": "CoTracker on same-seed no-intervention Baseline",
            "future_source_gt_used_by_guidance": False,
            "objects": [
                {
                    "name": name,
                    "point_start": int(slices[name].start),
                    "point_end": int(slices[name].stop),
                }
                for name in names
            ],
            "anchor_pixel_frames": anchors.tolist(),
        },
    )
    atomic_json(complete, {"case": case, "seed": seed, "point_count": len(points)})
    return path


def baseline_ready(sample: dict[str, Any]) -> bool:
    return Path(str(sample["baseline_video"])).is_file()


def generate_baseline(pipe_wrapper: Any, sample: dict[str, Any], case_lookup: dict[str, Any]) -> None:
    output_video = Path(str(sample["baseline_video"]))
    if output_video.is_file():
        return
    case, seed = str(sample["case"]), int(sample["seed"])
    json_path, _, payload, wan_args, image = generation_inputs(sample, case_lookup, seed)
    video = _run_pipe_once(
        pipe=pipe_wrapper,
        prompt=str(payload["input_caption"]),
        negative_prompt=str(wan_args.negative_prompt),
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
    output_video.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_video.with_name("generated.tmp.mp4")
    save_video_np(video, temporary, fps=30)
    temporary.replace(output_video)
    atomic_json(
        output_video.parent / "manifest.json",
        {
            "protocol": PROTOCOL,
            "mode": "baseline",
            "case": case,
            "seed": seed,
            "input_json": str(json_path),
            "output_video": str(output_video),
            "sampling_steps": 40,
            "cfg_scale": 5.0,
            "sample_shift": 5.0,
            "solver": "unipc",
        },
    )
    atomic_json(output_video.parent / "complete.json", {"case": case, "seed": seed})
    del video


def variant_directory(
    output_root: Path,
    case: str,
    seed: int,
    pag_scale: float,
    denoise_start: int,
    denoise_end: int,
    perturbation_mode: str = "m1_multi_object_blockdiag",
) -> Path:
    if perturbation_mode == "m1_multi_object_blockdiag":
        prefix = "multi_object_blockdiag__m1_all_time__top100"
    elif perturbation_mode == "m2_multi_object_independent":
        prefix = "multi_object_independent__m2_all_time__top100"
    elif perturbation_mode == "m3_multi_object_independent":
        prefix = "multi_object_independent__m3_all_time__top100"
    elif perturbation_mode == "full_head_output_zero":
        prefix = "all_token__full_head_output_zero__top100"
    else:
        raise ValueError(f"unsupported perturbation mode: {perturbation_mode}")
    variant = (
        f"{prefix}__pag{scale_tag(pag_scale)}"
        f"__denoise_{denoise_start:02d}_{denoise_end:02d}"
    )
    return output_root / "guided" / case / f"seed_{seed:05d}" / variant


def direct_variant_directory(
    output_root: Path,
    case: str,
    seed: int,
    denoise_start: int,
    denoise_end: int,
    perturbation_mode: str,
) -> Path:
    prefixes = {
        "m1_multi_object_blockdiag": "direct_multi_object_blockdiag__m1_all_time__top100",
        "m2_multi_object_independent": "direct_multi_object_independent__m2_all_time__top100",
        "m3_multi_object_independent": "direct_multi_object_independent__m3_all_time__top100",
        "full_head_output_zero": "direct_all_token__full_head_output_zero__top100",
    }
    try:
        prefix = prefixes[perturbation_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported perturbation mode: {perturbation_mode}") from exc
    variant = f"{prefix}__denoise_{denoise_start:02d}_{denoise_end:02d}"
    return output_root / "direct" / case / f"seed_{seed:05d}" / variant


def process_variant(
    pipe_wrapper: Any,
    sample: dict[str, Any],
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    pag_scale: float | None,
    denoise_start: int,
    denoise_end: int,
) -> None:
    case, seed = str(sample["case"]), int(sample["seed"])
    direct = args.execution_mode == "direct"
    if direct:
        output = direct_variant_directory(
            args.output_root,
            case,
            seed,
            denoise_start,
            denoise_end,
            args.perturbation_mode,
        )
    else:
        assert pag_scale is not None
        output = variant_directory(
            args.output_root,
            case,
            seed,
            pag_scale,
            denoise_start,
            denoise_end,
            args.perturbation_mode,
        )
    required = (output / "generated.mp4", output / "manifest.json", output / "complete.json")
    if all(path.is_file() for path in required) and not args.overwrite:
        print(f"[{args.execution_mode}] skip {output.relative_to(args.output_root)}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "complete.json").unlink(missing_ok=True)
    (output / "error.txt").unlink(missing_ok=True)
    points, region_slices, names = load_source_queries(sample)
    track_path = tracks_directory(args.tracks_root, case, seed) / "tracks.npz"
    tracks = None
    anchors = None
    if args.perturbation_mode in MULTI_OBJECT_FLOW_CONFIG:
        if not track_path.is_file():
            raise FileNotFoundError(track_path)
        with np.load(track_path) as arrays:
            tracks = arrays["tracks"].astype(np.float32)
            anchors = arrays["anchor_pixel_frames"].astype(np.int64)
    json_path, _, payload, wan_args, image = generation_inputs(sample, {}, seed)
    wan_args.cfg_scale = 5.0
    wan_args.sampling_steps = 40
    if args.perturbation_mode in MULTI_OBJECT_FLOW_CONFIG:
        assert tracks is not None and anchors is not None
        flow = MULTI_OBJECT_FLOW_CONFIG[args.perturbation_mode]
        ablator = MultiObjectIndependentFlowAblator(
            pipe_wrapper.pipe,
            entries,
            points,
            region_slices,
            (704, 1280),
            "all_objects",
            str(flow["mask_mode"]),
            None,
            tracks=tracks,
            anchor_frames=anchors,
            object_regions=names,
            flow_id=str(flow["id"]),
            group_batch_size=args.group_batch_size,
            record_dose=bool(args.record_dose),
        )
        protocol = str(flow["direct_protocol"] if direct else flow["protocol"])
        target_scope = f"all objects, simultaneous object-wise {flow['id']} union"
        perturbation = f"subtract {flow['formula']} without renormalization"
        if flow["id"] == "M1":
            preserved = (
                "A[R_i,R_j]V[R_j] for i != j, except inherently shared token cells"
            )
        elif flow["id"] == "M2":
            preserved = "each object's R_i->R_i; queries outside every object are unchanged"
        else:
            preserved = "each object's R_i->R_i; values outside every object are unchanged"
        perturbed_symbol = str(flow["symbol"])
    else:
        ablator = AttentionMatrixAblator(
            pipe_wrapper.pipe,
            entries,
            points,
            region_slices,
            (704, 1280),
            "all_tokens",
            "full_head_output",
            None,
            record_dose=False,
        )
        protocol = FULL_HEAD_DIRECT_PROTOCOL if direct else FULL_HEAD_PROTOCOL
        target_scope = "all query tokens and all latent-frame token positions"
        perturbation = (
            "set O_h=A_hV_h to zero for every selected head and every query token "
            "before head concatenation/output projection"
        )
        preserved = "unselected 620 physical heads only"
        perturbed_symbol = "eps_head_zero"
    if direct:
        controller = WindowedDirectAttentionAblation(
            pipe_wrapper.pipe,
            ablator,
            denoise_start=denoise_start,
            denoise_end=denoise_end,
            expected_steps=40,
        )
    else:
        assert pag_scale is not None
        controller = WindowedMultiObjectM1Guidance(
            pipe_wrapper.pipe,
            ablator,
            cfg_scale=5.0,
            pag_scale=pag_scale,
            denoise_start=denoise_start,
            denoise_end=denoise_end,
            expected_steps=40,
        )
    controller.install()
    try:
        video = _run_pipe_once(
            pipe=pipe_wrapper,
            prompt=str(payload["input_caption"]),
            negative_prompt=str(wan_args.negative_prompt),
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
        controller.remove()
    audit = controller.audit()
    temporary = output / "generated.tmp.mp4"
    save_video_np(video, temporary, fps=30)
    temporary.replace(output / "generated.mp4")
    if args.record_dose:
        atomic_npz(output / "dose_metrics.npz", **ablator.dose_arrays())
    metadata = {
        "protocol": protocol,
        "case": case,
        "seed": seed,
        "input_json": str(json_path),
        "baseline_video": str(sample["baseline_video"]),
        "tracks_npz": str(track_path) if tracks is not None else None,
        "source_query_tube": str(sample["source_query_tube"]),
        "head_scope": "latest3350 Top100",
        "selected_head_count": len(entries),
        "head_ranking_path": str(args.head_ranking_path),
        "head_ranking_sha256": sha256_file(args.head_ranking_path),
        "target_scope": target_scope,
        "object_regions": list(names),
        "object_count": len(names),
        "execution_mode": args.execution_mode,
        "perturbation_mode": args.perturbation_mode,
        "perturbation": perturbation,
        "preserved": preserved,
        "inactive_steps_are_clean_cfg": True,
        "future_source_gt_used_by_generation": False,
        "future_source_gt_used_by_guidance": False,
        "trajectory_source": (
            "same-seed Baseline CoTracker tube"
            if tracks is not None
            else "not used by all-token full-head perturbation"
        ),
        "selected_entries": entries,
        "audit": audit,
        "output_video": str(output / "generated.mp4"),
    }
    if direct:
        metadata.update(
            {
                "sampling_equation": "eps_u_deleted + 5*(eps_c_deleted-eps_u_deleted)",
                "ablation_step_range_inclusive": [denoise_start, denoise_end],
                "cfg_branches_modified": ["conditional", "unconditional"],
                "uses_prediction_contrast_guidance": False,
                "pag_scale": None,
            }
        )
    else:
        metadata.update(
            {
                "guidance_equation": (
                    f"eps_u + 5*(eps_c-eps_u) + lambda*(eps_c-{perturbed_symbol})"
                ),
                "pag_scale": pag_scale,
                "guidance_step_range_inclusive": [denoise_start, denoise_end],
            }
        )
    atomic_json(output / "manifest.json", metadata)
    complete_payload = {
        "protocol": protocol,
        "case": case,
        "seed": seed,
        "execution_mode": args.execution_mode,
        "object_count": len(names),
        "perturbation_mode": args.perturbation_mode,
        "modified_head_events": audit["modified_head_events"],
    }
    if direct:
        complete_payload["ablation_step_range_inclusive"] = [denoise_start, denoise_end]
    else:
        complete_payload["pag_scale"] = pag_scale
        complete_payload["guidance_step_range_inclusive"] = [denoise_start, denoise_end]
    atomic_json(output / "complete.json", complete_payload)
    del video
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{args.execution_mode}] complete {output.relative_to(args.output_root)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--stage", choices=("baselines", "tracks", "guidance", "all"), default="all"
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--head-ranking-path", type=Path, default=DEFAULT_HEAD_RANKING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tracks-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "tracks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-batch-size", type=int, default=4)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="guidance")
    parser.add_argument(
        "--perturbation-mode",
        choices=PERTURBATION_MODES,
        default="m1_multi_object_blockdiag",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--pag-scale", type=float, action="append", default=[])
    parser.add_argument(
        "--guidance-window",
        nargs=2,
        type=int,
        action="append",
        metavar=("START", "END"),
        default=[],
    )
    parser.add_argument("--max-guided-tasks", type=int, default=None)
    parser.add_argument("--record-dose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execution_mode == "direct" and args.pag_scale:
        raise ValueError("direct execution has no PAG scale")
    if args.perturbation_mode == "full_head_output_zero" and args.record_dose:
        raise ValueError(
            "full_head_output_zero does not expose A-block mass; run without --record-dose"
        )
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    ranking = json.loads(args.head_ranking_path.read_text(encoding="utf-8"))
    ranking_entries = validate_head_ranking(
        manifest, ranking, allow_tagged_snapshot_change=True
    )
    entries = selected_head_entries(
        ranking_entries, "top100", dict(ranking.get("head_scopes") or {})
    )
    all_samples = list(manifest.get("samples") or [])
    if args.case:
        selected_cases = set(args.case)
        all_samples = [row for row in all_samples if str(row["case"]) in selected_cases]
    if args.seed:
        selected_seeds = set(args.seed)
        all_samples = [row for row in all_samples if int(row["seed"]) in selected_seeds]
    samples = all_samples[args.worker_id :: args.num_workers]
    if not samples:
        raise RuntimeError("worker filters selected no samples")
    scales: list[float | None] = (
        [None]
        if args.execution_mode == "direct"
        else (
            list(args.pag_scale)
            if args.pag_scale
            else [float(value) for value in manifest["search_grid"]["pag_scales"]]
        )
    )
    windows = (
        [(int(value[0]), int(value[1])) for value in args.guidance_window]
        if args.guidance_window
        else [
        (int(value[0]), int(value[1]))
        for value in manifest["search_grid"]["guidance_windows_inclusive"]
        ]
    )
    for denoise_start, denoise_end in windows:
        validate_window(denoise_start, denoise_end)
    summary = {
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "sample_count": len(samples),
        "task_count": len(samples) * len(scales) * len(windows),
        "guided_task_count": (
            len(samples) * len(scales) * len(windows)
            if args.execution_mode == "guidance"
            else 0
        ),
        "direct_task_count": (
            len(samples) * len(windows) if args.execution_mode == "direct" else 0
        ),
        "cases": sorted({str(row["case"]) for row in samples}),
        "seeds": sorted({int(row["seed"]) for row in samples}),
        "scales": scales,
        "windows": windows,
        "perturbation_mode": args.perturbation_mode,
        "execution_mode": args.execution_mode,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    case_lookup = {case.key: case for case in CASES}
    if args.stage in {"baselines", "all"}:
        missing = [sample for sample in samples if not baseline_ready(sample)]
        if missing:
            pipe = build_wan_ti2v_pipeline(
                generation_inputs(missing[0], case_lookup, int(missing[0]["seed"]))[3]
            )
            try:
                for index, sample in enumerate(missing, start=1):
                    print(
                        f"[baseline {index}/{len(missing)}] {sample['case']} "
                        f"seed={sample['seed']}",
                        flush=True,
                    )
                    generate_baseline(pipe, sample, case_lookup)
            finally:
                del pipe
                gc.collect()
                torch.cuda.empty_cache()

    if (
        args.perturbation_mode in MULTI_OBJECT_FLOW_CONFIG
        and args.stage in {"tracks", "all"}
    ):
        missing_tracks = [
            sample
            for sample in samples
            if not (
                tracks_directory(
                    args.tracks_root, str(sample["case"]), int(sample["seed"])
                )
                / "complete.json"
            ).is_file()
        ]
        if missing_tracks:
            tracker = load_cotracker(args.device)
            try:
                for index, sample in enumerate(missing_tracks, start=1):
                    print(
                        f"[tracks {index}/{len(missing_tracks)}] {sample['case']} "
                        f"seed={sample['seed']}",
                        flush=True,
                    )
                    prepare_tracks(tracker, sample, args.tracks_root, args.device)
            finally:
                del tracker
                gc.collect()
                torch.cuda.empty_cache()

    if args.stage in {"guidance", "all"}:
        for sample in samples:
            if not baseline_ready(sample):
                raise FileNotFoundError(sample["baseline_video"])
            if args.perturbation_mode in MULTI_OBJECT_FLOW_CONFIG:
                track_path = (
                    tracks_directory(
                        args.tracks_root, str(sample["case"]), int(sample["seed"])
                    )
                    / "tracks.npz"
                )
                if not track_path.is_file():
                    raise FileNotFoundError(track_path)
        pipe = build_wan_ti2v_pipeline(
            generation_inputs(samples[0], case_lookup, int(samples[0]["seed"]))[3]
        )
        try:
            total = len(samples) * len(scales) * len(windows)
            task_index = 0
            stop = False
            for sample in samples:
                for denoise_start, denoise_end in windows:
                    for pag_scale in scales:
                        if (
                            args.max_guided_tasks is not None
                            and task_index >= args.max_guided_tasks
                        ):
                            stop = True
                            break
                        task_index += 1
                        scale_text = (
                            ""
                            if pag_scale is None
                            else f" lambda={pag_scale:g}"
                        )
                        print(
                            f"[{args.execution_mode} {task_index}/{total}] "
                            f"case={sample['case']} seed={sample['seed']}"
                            f"{scale_text} window={denoise_start}..{denoise_end}",
                            flush=True,
                        )
                        try:
                            process_variant(
                                pipe,
                                sample,
                                entries,
                                args,
                                pag_scale,
                                denoise_start,
                                denoise_end,
                            )
                        except Exception:
                            if args.execution_mode == "direct":
                                output = direct_variant_directory(
                                    args.output_root,
                                    str(sample["case"]),
                                    int(sample["seed"]),
                                    denoise_start,
                                    denoise_end,
                                    args.perturbation_mode,
                                )
                            else:
                                assert pag_scale is not None
                                output = variant_directory(
                                    args.output_root,
                                    str(sample["case"]),
                                    int(sample["seed"]),
                                    pag_scale,
                                    denoise_start,
                                    denoise_end,
                                    args.perturbation_mode,
                                )
                            output.mkdir(parents=True, exist_ok=True)
                            (output / "error.txt").write_text(
                                traceback.format_exc(), encoding="utf-8"
                            )
                            raise
                    if stop:
                        break
                if stop:
                    break
        finally:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

    completion_name = (
        f"worker_{args.worker_id}_complete.json"
        if args.execution_mode == "guidance"
        else f"worker_{args.worker_id}_direct_{args.perturbation_mode}_complete.json"
    )
    atomic_json(
        args.output_root / "logs" / completion_name,
        {**summary, "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )


if __name__ == "__main__":
    main()
