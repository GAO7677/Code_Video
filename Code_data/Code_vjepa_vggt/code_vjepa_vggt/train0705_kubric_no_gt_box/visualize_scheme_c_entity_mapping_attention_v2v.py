#!/usr/bin/env python3
"""Validate Scheme-C text/object entity mapping inside Wan DiT blocks.

The script reuses the existing predicted-x0 text-attention visualizer, adds an
equivalent recorder for every object_cross_attn branch, and compares both maps
on the shared Wan video-query grid. It loads the real Scheme-C entity wrapper.
"""
from __future__ import annotations

import csv
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)


def _pop_option(argv: list[str], name: str, default: str) -> str:
    values: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            values.append(argv[index + 1])
            del argv[index : index + 2]
            continue
        if token.startswith(f"{name}="):
            values.append(token.split("=", 1)[1])
            del argv[index]
            continue
        index += 1
    if not values:
        return default
    if len(set(values)) != 1:
        raise SystemExit(f"conflicting {name} values: {values}")
    return values[-1]


OBJECT_QUERY_CHUNK = int(
    _pop_option(sys.argv, "--entity-mapping-object-query-chunk", "128")
)
ENTITY_TOP_PAIRS = int(_pop_option(sys.argv, "--entity-mapping-top-pairs", "3"))
SAVE_ALL_OBJECT_MAPS = bool(
    int(_pop_option(sys.argv, "--entity-mapping-save-all-object-maps", "0"))
)
ZERO_COMPACT_SLOT_RANKS = tuple(
    sorted(
        {
            int(value.strip())
            for value in _pop_option(
                sys.argv, "--entity-mapping-zero-compact-slot-ranks", ""
            ).split(",")
            if value.strip()
        }
    )
)
SWAP_SLOT_ENTITY_IDS = tuple(
    int(value.strip())
    for value in _pop_option(
        sys.argv, "--entity-mapping-swap-slot-entity-ids", ""
    ).split(",")
    if value.strip()
)
if SWAP_SLOT_ENTITY_IDS and len(SWAP_SLOT_ENTITY_IDS) != 2:
    raise SystemExit("--entity-mapping-swap-slot-entity-ids requires exactly two slot IDs")

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    visualize_text_noun_attention_x0_v2v as textvis,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_entity_id_binding_v2v as entity_v2v,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_scheme_c_entity_caption_physical_v2v as scheme_c,
)


base = textvis.base

# Keep this first diagnostic deliberately small and unambiguous.
textvis.NOUN_SPECS.update(
    {
        "physicIQ_025_Solid_Mechanics_0002_perspective-center-trimmed-ball-and-block-fall_motion_to_end": {
            "tennis_ball": ("brown tennis ball", "tennis ball", "ball"),
            "block": ("orange block", "block"),
        },
        "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end": {
            "tennis_ball": ("brown tennis ball", "tennis ball", "ball"),
            "block": ("orange block", "block"),
        },
        "phyco_kubric_pool_table_force_2025-09-27_fef01f": {
            "balls": ("balls", "ball"),
        },
        "phyco_kubric_ball_drop_soft_v4_2025-09-05_0144a4": {
            "ball": ("deformable elastic ball", "elastic ball", "ball"),
        },
    }
)


class EntityMappingAttentionRecorder(textvis.AttentionRecorder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.object_key_maps: dict[tuple[int, int], np.ndarray] = {}
        self.object_key_counts: dict[tuple[int, int], int] = {}
        self.text_contribution_maps: dict[tuple[int, int, str], np.ndarray] = {}
        self.object_contribution_maps: dict[tuple[int, int], np.ndarray] = {}
        self.object_gate_abs: dict[int, float] = {}
        self.valid_slot_count: int | None = None
        self._text_contribution_forwards: list[tuple[Any, Any]] = []

    def install(self, dit) -> None:
        super().install(dit)
        for layer_id, block in enumerate(dit.blocks):
            text_cross_attn = getattr(block, "cross_attn", None)
            text_attn = getattr(text_cross_attn, "attn", None)
            if text_attn is None:
                raise RuntimeError(f"Wan block {layer_id} has no text attention module")
            parent_text_forward = text_attn.forward

            def wrapped_text(
                q,
                k,
                v,
                *,
                _parent=parent_text_forward,
                _cross=text_cross_attn,
                _layer_id=layer_id,
            ):
                output = _parent(q, k, v)
                if self.active:
                    self.capture_text_contribution(_layer_id, q, k, v, _cross)
                return output

            text_attn.forward = wrapped_text
            self._text_contribution_forwards.append(
                (text_attn, parent_text_forward)
            )

            object_cross_attn = getattr(block, "object_cross_attn", None)
            attn = getattr(object_cross_attn, "attn", None)
            if attn is None:
                raise RuntimeError(
                    f"Wan block {layer_id} has no object cross-attention module"
                )
            original = attn.forward
            gate = getattr(block, "object_gate", None)
            if gate is not None:
                self.object_gate_abs[int(layer_id)] = float(
                    torch.tanh(gate.detach().float()).abs().mean().item()
                )

            def wrapped(
                q,
                k,
                v,
                *,
                _original=original,
                _cross=object_cross_attn,
                _gate=gate,
                _layer_id=layer_id,
            ):
                output = _original(q, k, v)
                if self.active:
                    self.capture_object(_layer_id, q, k, v, _cross, _gate)
                return output

            attn.forward = wrapped
            self._original_forwards.append((attn, original))

    def restore(self) -> None:
        # Restore the contribution wrappers to the parent text recorder first;
        # the parent restore then returns every module to its original forward.
        for module, parent_forward in self._text_contribution_forwards:
            module.forward = parent_forward
        self._text_contribution_forwards.clear()
        super().restore()

    @staticmethod
    def _project_contribution(
        weighted: torch.Tensor,
        cross_attn: Any,
        *,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        merged = weighted.permute(0, 2, 1, 3).reshape(
            int(weighted.shape[0]), int(weighted.shape[2]), -1
        )
        output_weight = cross_attn.o.weight
        projected = cross_attn.o(
            merged.to(device=output_weight.device, dtype=output_weight.dtype)
        ).float()
        if gate is not None:
            projected = projected * torch.tanh(gate.detach().float()).to(
                device=projected.device
            )
        return projected.norm(dim=-1)

    @torch.no_grad()
    def capture_text_contribution(
        self,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cross_attn: Any,
    ) -> None:
        if self.grid is None:
            raise RuntimeError("attention grid missing for text contribution")
        frames, grid_h, grid_w = self.grid
        num_heads = 24
        head_dim = int(q.shape[-1]) // num_heads
        keys = k.view(1, int(k.shape[1]), num_heads, head_dim).permute(0, 2, 1, 3).float()
        values = v.view(1, int(v.shape[1]), num_heads, head_dim).permute(0, 2, 1, 3).float()
        key_t = keys.transpose(-1, -2)
        chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
        for start in range(0, int(q.shape[1]), OBJECT_QUERY_CHUNK):
            stop = min(start + OBJECT_QUERY_CHUNK, int(q.shape[1]))
            queries = q[:, start:stop].view(
                1, stop - start, num_heads, head_dim
            ).permute(0, 2, 1, 3).float()
            probs = torch.softmax(
                torch.matmul(queries, key_t) / math.sqrt(float(head_dim)),
                dim=-1,
            )
            for noun, details in self.noun_details.items():
                token_ids = torch.as_tensor(
                    details["token_positions"], device=probs.device, dtype=torch.long
                )
                weighted = torch.matmul(
                    probs.index_select(-1, token_ids),
                    values.index_select(-2, token_ids),
                )
                chunks[noun].append(
                    self._project_contribution(weighted, cross_attn).cpu()
                )
            del probs, queries
        for noun, noun_chunks in chunks.items():
            flat = torch.cat(noun_chunks, dim=1)[0]
            self.text_contribution_maps[
                (int(self.step_index), int(layer_id), noun)
            ] = flat.reshape(frames, grid_h, grid_w).numpy().astype(np.float16)

    @torch.no_grad()
    def capture_object(
        self,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cross_attn: Any,
        gate: torch.Tensor | None,
    ) -> None:
        if self.grid is None:
            raise RuntimeError("attention grid was not configured before object capture")
        frames, grid_h, grid_w = self.grid
        expected_queries = int(frames * grid_h * grid_w)
        if int(q.shape[0]) != 1 or int(q.shape[1]) != expected_queries:
            raise RuntimeError(
                f"object attention q={list(q.shape)} does not match grid={self.grid}"
            )
        num_heads = int(
            getattr(getattr(q, "_attention_module", None), "num_heads", 0)
        )
        if num_heads <= 0:
            num_heads = 24
        if int(q.shape[-1]) % num_heads != 0:
            raise RuntimeError(
                f"object attention dim={q.shape[-1]} not divisible by heads={num_heads}"
            )
        head_dim = int(q.shape[-1]) // num_heads
        keys = (
            k.view(1, int(k.shape[1]), num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .float()
        )
        values = (
            v.view(1, int(v.shape[1]), num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .float()
        )
        key_t = keys.transpose(-1, -2)
        chunks: list[torch.Tensor] = []
        contribution_chunks: list[list[torch.Tensor]] | None = None
        if self.valid_slot_count is not None and self.valid_slot_count > 0:
            if int(k.shape[1]) % int(self.valid_slot_count) != 0:
                raise RuntimeError(
                    f"object keys={k.shape[1]} not divisible by valid slots={self.valid_slot_count}"
                )
            contribution_chunks = [list() for _ in range(int(self.valid_slot_count))]
        for start in range(0, int(q.shape[1]), OBJECT_QUERY_CHUNK):
            stop = min(start + OBJECT_QUERY_CHUNK, int(q.shape[1]))
            queries = (
                q[:, start:stop]
                .view(1, stop - start, num_heads, head_dim)
                .permute(0, 2, 1, 3)
                .float()
            )
            scores = torch.matmul(queries, key_t) / math.sqrt(float(head_dim))
            probs_by_head = torch.softmax(scores, dim=-1)
            chunks.append(probs_by_head.mean(dim=1)[0].cpu())
            if contribution_chunks is not None:
                for compact_rank in range(int(self.valid_slot_count)):
                    key_ids = torch.arange(
                        compact_rank,
                        int(k.shape[1]),
                        int(self.valid_slot_count),
                        device=probs_by_head.device,
                    )
                    weighted = torch.matmul(
                        probs_by_head.index_select(-1, key_ids),
                        values.index_select(-2, key_ids),
                    )
                    contribution_chunks[compact_rank].append(
                        self._project_contribution(
                            weighted, cross_attn, gate=gate
                        ).cpu()
                    )
            del scores, probs_by_head, queries
        flat = torch.cat(chunks, dim=0)
        key_count = int(flat.shape[-1])
        self.object_key_maps[(int(self.step_index), int(layer_id))] = (
            flat.reshape(frames, grid_h, grid_w, key_count).numpy().astype(np.float16)
        )
        self.object_key_counts[(int(self.step_index), int(layer_id))] = key_count
        if contribution_chunks is not None:
            slot_arrays = []
            for slot_chunks in contribution_chunks:
                slot_flat = torch.cat(slot_chunks, dim=1)[0]
                slot_arrays.append(
                    slot_flat.reshape(frames, grid_h, grid_w).numpy()
                )
            self.object_contribution_maps[(int(self.step_index), int(layer_id))] = (
                np.stack(slot_arrays, axis=-1).astype(np.float16)
            )


def _normalize_phrase(value: str) -> str:
    value = re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())
    return " ".join(value.split())


def _binding_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    binding = result.get("object_debug", {}).get("entity_id_binding", {})
    records: list[dict[str, Any]] = []
    for matched in binding.get("matched", []):
        record = dict(matched)
        record["prompt_matched"] = True
        records.append(record)
    for unmatched in binding.get("unmatched", []):
        record = dict(unmatched)
        record["prompt_matched"] = False
        record.setdefault("matched_candidate", None)
        record.setdefault("prompt_span_count", 0)
        records.append(record)
    records.sort(key=lambda item: int(item["slot_id"]))
    return records


def _expected_nouns_for_record(
    record: dict[str, Any],
    noun_spec: dict[str, tuple[str, ...]],
) -> list[str]:
    phrase = _normalize_phrase(
        str(record.get("matched_candidate") or record.get("grounding_phrase") or "")
    )
    matches: list[tuple[int, str]] = []
    for noun, aliases in noun_spec.items():
        for alias in aliases:
            normalized = _normalize_phrase(alias)
            if normalized and (
                normalized == phrase
                or normalized in phrase
                or phrase in normalized
            ):
                matches.append((len(normalized), noun))
    if not matches:
        return []
    longest = max(length for length, _ in matches)
    return sorted({noun for length, noun in matches if length == longest})


def _distribution(array: np.ndarray) -> np.ndarray:
    flat = np.maximum(array.astype(np.float64).reshape(-1), 0.0)
    total = float(flat.sum())
    if total <= 0.0:
        return np.full(flat.shape, 1.0 / max(flat.size, 1), dtype=np.float64)
    return flat / total


def _map_metrics(text_map: np.ndarray, object_map: np.ndarray) -> dict[str, float]:
    text_flat = text_map.astype(np.float64).reshape(-1)
    object_flat = object_map.astype(np.float64).reshape(-1)
    cosine = float(
        np.dot(text_flat, object_flat)
        / max(np.linalg.norm(text_flat) * np.linalg.norm(object_flat), 1.0e-12)
    )
    p = _distribution(text_map)
    q = _distribution(object_map)
    m = 0.5 * (p + q)
    js = 0.5 * float(np.sum(p * np.log(np.maximum(p, 1.0e-12) / m)))
    js += 0.5 * float(np.sum(q * np.log(np.maximum(q, 1.0e-12) / m)))
    text_centered = text_flat - text_flat.mean()
    object_centered = object_flat - object_flat.mean()
    centered_cosine = float(
        np.dot(text_centered, object_centered)
        / max(
            np.linalg.norm(text_centered) * np.linalg.norm(object_centered),
            1.0e-12,
        )
    )
    return {
        "cosine_similarity": cosine,
        "centered_cosine_similarity": centered_cosine,
        "js_divergence": js,
        "text_peak_to_mean": float(
            text_flat.max() / max(text_flat.mean(), 1.0e-12)
        ),
        "object_peak_to_mean": float(
            object_flat.max() / max(object_flat.mean(), 1.0e-12)
        ),
        "object_coefficient_of_variation": float(
            object_flat.std() / max(object_flat.mean(), 1.0e-12)
        ),
    }


def _pool_slot_maps(
    raw: np.ndarray,
    *,
    valid_slot_count: int,
) -> tuple[list[np.ndarray], int]:
    key_count = int(raw.shape[-1])
    if valid_slot_count <= 0 or key_count % valid_slot_count != 0:
        raise ValueError(
            f"object key count={key_count} cannot be grouped into slots={valid_slot_count}"
        )
    time_steps = key_count // valid_slot_count
    grouped = raw.reshape(*raw.shape[:-1], time_steps, valid_slot_count)
    return [grouped[..., slot_id].sum(axis=-1) for slot_id in range(valid_slot_count)], time_steps


def _write_pair_video(
    *,
    path: Path,
    frames: list[np.ndarray],
    text_map: np.ndarray,
    object_map: np.ndarray,
    noun: str,
    slot_id: int,
    entity_id: int,
    layer_id: int,
    remaining_steps: int,
    fps: int,
) -> None:
    text_aligned = textvis._temporal_resize_lowres(text_map, len(frames))
    object_aligned = textvis._temporal_resize_lowres(object_map, len(frames))
    text_low, text_high = np.percentile(text_aligned, [2.0, 99.0]).tolist()
    object_low, object_high = np.percentile(object_aligned, [2.0, 99.0]).tolist()
    output: list[np.ndarray] = []
    for frame_id, frame in enumerate(frames):
        text_overlay = textvis._heat_overlay(
            frame, text_aligned[frame_id], float(text_low), float(text_high)
        )
        object_overlay = textvis._heat_overlay(
            frame, object_aligned[frame_id], float(object_low), float(object_high)
        )
        text_overlay = cv2.resize(text_overlay, (448, 256), interpolation=cv2.INTER_AREA)
        object_overlay = cv2.resize(object_overlay, (448, 256), interpolation=cv2.INTER_AREA)
        text_overlay = textvis._label(
            text_overlay,
            [
                f"TEXT noun={noun} | layer={layer_id} | remaining={remaining_steps}",
                f"x0 frame={frame_id:02d}",
            ],
        )
        object_overlay = textvis._label(
            object_overlay,
            [
                f"OBJECT S{slot_id}/E{entity_id} | layer={layer_id} | remaining={remaining_steps}",
                f"x0 frame={frame_id:02d}",
            ],
        )
        output.append(np.concatenate([text_overlay, object_overlay], axis=1))
    textvis._write_h264(path, output, fps=int(fps))


def _render_entity_mapping(
    *,
    output_dir: Path,
    case_stem: str,
    result: dict[str, Any],
    recorder: EntityMappingAttentionRecorder,
    noun_details: dict[str, dict[str, Any]],
    noun_spec: dict[str, tuple[str, ...]],
    capture_indices: list[int],
    total_steps: int,
    x0_videos: dict[int, list[np.ndarray]],
    fps: int,
) -> dict[str, Any]:
    records = _binding_records(result)
    if not records:
        raise RuntimeError("entity mapping validation requires at least one bound slot")
    valid_slot_count = len(records)
    rows: list[dict[str, Any]] = []
    raw_selected: dict[str, np.ndarray] = {}
    expected_pairs: list[tuple[str, int, int]] = []
    for compact_rank, record in enumerate(records):
        for noun in _expected_nouns_for_record(record, noun_spec):
            expected_pairs.append(
                (noun, compact_rank, int(record.get("entity_id", -1)))
            )

    for step_index in capture_indices:
        for layer_id in range(len(recorder.object_gate_abs)):
            raw = recorder.object_key_maps[(step_index, layer_id)].astype(np.float32)
            slot_maps, object_time_steps = _pool_slot_maps(
                raw, valid_slot_count=valid_slot_count
            )
            object_contributions = recorder.object_contribution_maps[
                (step_index, layer_id)
            ].astype(np.float32)
            for noun in noun_details:
                text_map = recorder.maps[(step_index, layer_id, noun)].astype(np.float32)
                text_contribution = recorder.text_contribution_maps[
                    (step_index, layer_id, noun)
                ].astype(np.float32)
                pair_rows: list[dict[str, Any]] = []
                for compact_rank, (record, object_map) in enumerate(zip(records, slot_maps)):
                    metrics = _map_metrics(text_map, object_map)
                    contribution_metrics = _map_metrics(
                        text_contribution,
                        object_contributions[..., compact_rank],
                    )
                    expected_nouns = _expected_nouns_for_record(record, noun_spec)
                    row = {
                        "case": case_stem,
                        "step_index": int(step_index),
                        "remaining_steps": int(total_steps - step_index),
                        "layer_id": int(layer_id),
                        "noun": noun,
                        "compact_slot_rank": int(compact_rank),
                        "slot_id": int(record["slot_id"]),
                        "entity_id": int(record.get("entity_id", -1)),
                        "grounding_phrase": str(record.get("grounding_phrase", "")),
                        "matched_candidate": record.get("matched_candidate"),
                        "expected_nouns": ",".join(expected_nouns),
                        "is_expected_pair": noun in expected_nouns,
                        "object_key_count": int(raw.shape[-1]),
                        "object_time_steps": int(object_time_steps),
                        "object_gate_tanh_abs": float(
                            recorder.object_gate_abs.get(layer_id, 0.0)
                        ),
                        **metrics,
                        **{
                            f"contribution_{key}": value
                            for key, value in contribution_metrics.items()
                        },
                    }
                    pair_rows.append(row)
                    rows.append(row)
                ranked_attention = sorted(
                    pair_rows,
                    key=lambda item: float(item["centered_cosine_similarity"]),
                    reverse=True,
                )
                for rank, item in enumerate(ranked_attention, 1):
                    item["slot_rank_by_centered_cosine"] = int(rank)
                    item["noun_top1_centered_correct"] = bool(
                        rank == 1 and item["is_expected_pair"]
                    )
                ranked_contribution = sorted(
                    pair_rows,
                    key=lambda item: float(
                        item["contribution_centered_cosine_similarity"]
                    ),
                    reverse=True,
                )
                for rank, item in enumerate(ranked_contribution, 1):
                    item["slot_rank_by_contribution_centered_cosine"] = int(rank)
                    item["noun_top1_contribution_centered_correct"] = bool(
                        rank == 1 and item["is_expected_pair"]
                    )
                unexpected_attention = [
                    float(item["centered_cosine_similarity"])
                    for item in pair_rows
                    if not item["is_expected_pair"]
                ]
                best_unexpected_attention = (
                    max(unexpected_attention) if unexpected_attention else None
                )
                unexpected_contribution = [
                    float(item["contribution_centered_cosine_similarity"])
                    for item in pair_rows
                    if not item["is_expected_pair"]
                ]
                best_unexpected_contribution = (
                    max(unexpected_contribution)
                    if unexpected_contribution
                    else None
                )
                for item in pair_rows:
                    item["centered_margin_vs_best_unexpected"] = (
                        None
                        if not item["is_expected_pair"]
                        or best_unexpected_attention is None
                        else float(item["centered_cosine_similarity"])
                        - best_unexpected_attention
                    )
                    item["contribution_centered_margin_vs_best_unexpected"] = (
                        None
                        if not item["is_expected_pair"]
                        or best_unexpected_contribution is None
                        else float(
                            item["contribution_centered_cosine_similarity"]
                        ) - best_unexpected_contribution
                    )

    if not rows:
        raise RuntimeError("no entity mapping metrics were captured")
    metrics_path = output_dir / "entity_mapping_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pair_outputs: list[dict[str, Any]] = []
    for noun, compact_rank, entity_id in expected_pairs:
        record = records[compact_rank]
        pair_rows = [
            row
            for row in rows
            if row["noun"] == noun
            and int(row["compact_slot_rank"]) == compact_rank
        ]
        top = sorted(
            pair_rows,
            key=lambda item: (
                float(item["contribution_centered_cosine_similarity"]),
                -float(item["js_divergence"]),
            ),
            reverse=True,
        )[:ENTITY_TOP_PAIRS]
        visualizations: list[dict[str, Any]] = []
        for rank, row in enumerate(top, 1):
            step_index = int(row["step_index"])
            layer_id = int(row["layer_id"])
            raw = recorder.object_key_maps[(step_index, layer_id)].astype(np.float32)
            _pool_slot_maps(raw, valid_slot_count=valid_slot_count)
            text_map = recorder.text_contribution_maps[
                (step_index, layer_id, noun)
            ].astype(np.float32)
            object_map = recorder.object_contribution_maps[
                (step_index, layer_id)
            ][..., compact_rank].astype(np.float32)
            raw_selected[
                f"text__{noun}__layer_{layer_id:02d}__progress_{step_index:02d}"
            ] = text_map.astype(np.float16)
            raw_selected[
                f"object__slot_{record['slot_id']}__layer_{layer_id:02d}__progress_{step_index:02d}"
            ] = object_map.astype(np.float16)
            video_path = output_dir / (
                f"pair_{noun}_S{record['slot_id']}_E{entity_id}_rank{rank}_"
                f"layer{layer_id:02d}_remaining{total_steps-step_index:02d}.mp4"
            )
            _write_pair_video(
                path=video_path,
                frames=x0_videos[step_index],
                text_map=text_map,
                object_map=object_map,
                noun=noun,
                slot_id=int(record["slot_id"]),
                entity_id=int(entity_id),
                layer_id=layer_id,
                remaining_steps=int(total_steps - step_index),
                fps=int(fps),
            )
            visualizations.append({"rank": rank, "video": video_path.name, **row})
        pair_outputs.append(
            {
                "noun": noun,
                "compact_slot_rank": int(compact_rank),
                "slot_id": int(record["slot_id"]),
                "entity_id": int(entity_id),
                "grounding_phrase": str(record.get("grounding_phrase", "")),
                "matched_candidate": record.get("matched_candidate"),
                "top_block_steps": visualizations,
            }
        )

    npz_path = output_dir / "selected_entity_mapping_maps_fp16.npz"
    np.savez_compressed(npz_path, **raw_selected)
    all_object_path = None
    if SAVE_ALL_OBJECT_MAPS:
        all_object_path = output_dir / "all_object_key_attention_maps_fp16.npz"
        np.savez_compressed(
            all_object_path,
            **{
                f"layer_{layer:02d}__progress_{step:02d}": array
                for (step, layer), array in recorder.object_key_maps.items()
            },
        )

    expected_rows = [row for row in rows if row["is_expected_pair"]]
    unique_semantic_nouns = {
        noun
        for noun in noun_details
        if sum(
            int(noun in _expected_nouns_for_record(record, noun_spec))
            for record in records
        ) == 1
    }
    unique_top1_rows = [
        row
        for row in rows
        if row["noun"] in unique_semantic_nouns
        and int(row.get("slot_rank_by_contribution_centered_cosine", 0)) == 1
    ]
    summary = {
        "case": case_stem,
        "binding_records": records,
        "valid_slot_count": valid_slot_count,
        "expected_pairs": [
            {"noun": noun, "compact_slot_rank": rank, "entity_id": entity_id}
            for noun, rank, entity_id in expected_pairs
        ],
        "capture_indices": capture_indices,
        "capture_remaining_steps": [total_steps - step for step in capture_indices],
        "captured_layer_count": len(recorder.object_gate_abs),
        "captured_object_map_count": len(recorder.object_key_maps),
        "object_gate_tanh_abs_by_layer": recorder.object_gate_abs,
        "expected_pair_cosine_mean": float(
            np.mean([row["cosine_similarity"] for row in expected_rows])
        ) if expected_rows else None,
        "expected_pair_centered_cosine_mean": float(
            np.mean([row["centered_cosine_similarity"] for row in expected_rows])
        ) if expected_rows else None,
        "expected_pair_contribution_centered_cosine_mean": float(
            np.mean([
                row["contribution_centered_cosine_similarity"]
                for row in expected_rows
            ])
        ) if expected_rows else None,
        "expected_pair_centered_margin_mean": float(
            np.mean([
                row["centered_margin_vs_best_unexpected"]
                for row in expected_rows
                if row["centered_margin_vs_best_unexpected"] is not None
            ])
        ) if any(
            row["centered_margin_vs_best_unexpected"] is not None
            for row in expected_rows
        ) else None,
        "expected_pair_contribution_centered_margin_mean": float(
            np.mean([
                row["contribution_centered_margin_vs_best_unexpected"]
                for row in expected_rows
                if row["contribution_centered_margin_vs_best_unexpected"] is not None
            ])
        ) if any(
            row["contribution_centered_margin_vs_best_unexpected"] is not None
            for row in expected_rows
        ) else None,
        "expected_pair_js_mean": float(
            np.mean([row["js_divergence"] for row in expected_rows])
        ) if expected_rows else None,
        "unique_noun_contribution_centered_top1_accuracy": float(
            np.mean([row.get("is_expected_pair", False) for row in unique_top1_rows])
        ) if unique_top1_rows else None,
        "identical_noun_policy": (
            "set_level_mapping; multiple slots may share one caption span"
        ),
        "metrics_csv": metrics_path.name,
        "selected_maps": npz_path.name,
        "all_object_maps": None if all_object_path is None else all_object_path.name,
        "pairs": pair_outputs,
    }
    summary_path = output_dir / "entity_mapping_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cards: list[str] = []
    for pair in pair_outputs:
        videos = "".join(
            f"<figure><video controls src='{html.escape(item['video'])}'></video>"
            f"<figcaption>rank {item['rank']} | layer {item['layer_id']} | "
            f"remaining {item['remaining_steps']} | contribution centered cosine {item['contribution_centered_cosine_similarity']:.4f} | "
            f"JS {item['js_divergence']:.4f}</figcaption></figure>"
            for item in pair["top_block_steps"]
        )
        cards.append(
            f"<section><h2>{html.escape(pair['noun'])} -> "
            f"S{pair['slot_id']}/E{pair['entity_id']}</h2>"
            f"<p>grounding={html.escape(pair['grounding_phrase'])}; "
            f"caption={html.escape(str(pair['matched_candidate']))}</p>"
            f"<div class='media'>{videos}</div></section>"
        )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Entity mapping {html.escape(case_stem)}</title>
<style>body{{margin:0;background:#edf0ec;color:#20231f;font:15px Georgia,serif}}main{{max-width:1600px;margin:auto;padding:24px}}h1,h2{{font-family:Verdana,sans-serif;letter-spacing:0}}section{{background:#fff;border:1px solid #c9cec6;padding:16px;margin:16px 0;border-radius:4px}}.media{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}figure{{margin:0}}video{{width:100%;background:#000}}figcaption{{padding:7px 0}}@media(max-width:900px){{.media{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Scheme-C entity mapping: {html.escape(case_stem)}</h1><p>Each video compares positive text cross-attention (left) with the mapped object-slot cross-attention (right) on the same predicted-x0 frames. Attention agreement is evidence of shared spatial routing, not sufficient causal proof.</p>{''.join(cards)}<p><a href='{summary_path.name}'>summary JSON</a> | <a href='{metrics_path.name}'>all metrics CSV</a></p></main></body></html>"""
    index_path = output_dir / "entity_mapping_index.html"
    index_path.write_text(page, encoding="utf-8")
    return {"summary": str(summary_path), "index": str(index_path)}


_original_run_single_case = textvis._original_run_single_case
_original_has_complete_output = textvis._original_has_complete_output


def _mapping_dir(output_video: Path) -> Path:
    return output_video.with_name(f"{output_video.stem}_entity_mapping_attention")


def _has_complete_mapping_output(output_video: Path, output_json: Path) -> bool:
    if not _original_has_complete_output(output_video, output_json):
        return False
    return (_mapping_dir(output_video) / "entity_mapping_summary.json").is_file()


def _run_single_case_with_entity_mapping(*args, **kwargs):
    model = kwargs["model"]
    pipe = model.pipe
    input_json_path = Path(kwargs["input_json_path"])
    case_stem = input_json_path.stem
    noun_spec = textvis.NOUN_SPECS.get(case_stem)
    if noun_spec is None:
        raise RuntimeError(f"no entity noun specification configured for {case_stem}")
    prompt = str(kwargs.get("input_caption", ""))
    sampling_steps = int(kwargs.get("sampling_steps", 40))
    cfg_scale = float(kwargs.get("cfg_scale", 5.0))
    capture_indices = textvis._capture_indices(textvis.CAPTURE_SPEC, sampling_steps)
    noun_details = textvis._resolve_noun_tokens(pipe, prompt, noun_spec)
    attention = EntityMappingAttentionRecorder(
        noun_details=noun_details,
        capture_indices=capture_indices,
        query_chunk=textvis.QUERY_CHUNK,
    )
    model_fn = textvis.ModelFnRecorder(
        pipe=pipe,
        attention=attention,
        capture_indices=capture_indices,
        cfg_scale=cfg_scale,
        total_steps=sampling_steps,
    )
    original_object_builder = entity_v2v.infer0705._build_object_context
    entity_adapter = model.object_adapter
    if not hasattr(entity_adapter, "set_entity_binding_context"):
        raise TypeError(
            f"object adapter {type(entity_adapter).__name__} is not entity-bound"
        )
    original_set_binding_context = entity_adapter.set_entity_binding_context

    def intercepted_set_binding_context(
        *,
        entity_text_by_id,
        entity_text_match_mask,
        slot_entity_ids,
        text_token_entity_ids=None,
    ):
        routed_ids = slot_entity_ids
        if SWAP_SLOT_ENTITY_IDS:
            first, second = SWAP_SLOT_ENTITY_IDS
            routed_ids = slot_entity_ids.clone()
            if not (
                0 <= first < int(routed_ids.shape[1])
                and 0 <= second < int(routed_ids.shape[1])
            ):
                raise ValueError(
                    f"swap slots={SWAP_SLOT_ENTITY_IDS} exceed shape={list(routed_ids.shape)}"
                )
            first_value = routed_ids[:, first].clone()
            routed_ids[:, first] = routed_ids[:, second]
            routed_ids[:, second] = first_value
        return original_set_binding_context(
            entity_text_by_id=entity_text_by_id,
            entity_text_match_mask=entity_text_match_mask,
            slot_entity_ids=routed_ids,
            text_token_entity_ids=text_token_entity_ids,
        )

    entity_adapter.set_entity_binding_context = intercepted_set_binding_context

    def captured_object_builder(*builder_args, **builder_kwargs):
        object_context, debug = original_object_builder(
            *builder_args, **builder_kwargs
        )
        binding = debug.get("entity_id_binding", {})
        recorder_count = len(binding.get("matched", [])) + len(
            binding.get("unmatched", [])
        )
        if recorder_count <= 0:
            raise RuntimeError("object builder produced no entity-bound slots")
        attention.valid_slot_count = int(recorder_count)
        if object_context is not None and ZERO_COMPACT_SLOT_RANKS:
            sequence_length = int(object_context.shape[1])
            if sequence_length % recorder_count != 0:
                raise RuntimeError(
                    f"object context length={sequence_length} not divisible by slots={recorder_count}"
                )
            invalid = [
                rank
                for rank in ZERO_COMPACT_SLOT_RANKS
                if rank < 0 or rank >= recorder_count
            ]
            if invalid:
                raise ValueError(
                    f"zero compact ranks={invalid} exceed valid slots={recorder_count}"
                )
            time_steps = sequence_length // recorder_count
            context_bto = object_context.view(
                int(object_context.shape[0]),
                time_steps,
                recorder_count,
                int(object_context.shape[-1]),
            ).clone()
            context_bto[:, :, list(ZERO_COMPACT_SLOT_RANKS), :] = 0
            object_context = context_bto.reshape_as(object_context)
        debug["entity_mapping_causal_intervention"] = {
            "zero_compact_slot_ranks": list(ZERO_COMPACT_SLOT_RANKS),
            "swap_slot_entity_ids": list(SWAP_SLOT_ENTITY_IDS),
        }
        return object_context, debug

    entity_v2v.infer0705._build_object_context = captured_object_builder
    attention.install(pipe.dit)
    model_fn.install()
    try:
        result, logs = _original_run_single_case(*args, **kwargs)
    finally:
        model_fn.restore()
        attention.restore()
        entity_v2v.infer0705._build_object_context = original_object_builder
        entity_adapter.set_entity_binding_context = original_set_binding_context
    if set(model_fn.x0_latents) != set(capture_indices):
        raise RuntimeError(
            f"predicted-x0 snapshots mismatch: got={sorted(model_fn.x0_latents)} "
            f"expected={capture_indices}"
        )
    expected_map_keys = {
        (step, layer)
        for step in capture_indices
        for layer in range(len(pipe.dit.blocks))
    }
    if set(attention.object_key_maps) != expected_map_keys:
        missing = sorted(expected_map_keys - set(attention.object_key_maps))[:10]
        raise RuntimeError(f"object attention capture incomplete; first missing={missing}")
    if set(attention.object_contribution_maps) != expected_map_keys:
        missing = sorted(
            expected_map_keys - set(attention.object_contribution_maps)
        )[:10]
        raise RuntimeError(
            f"object contribution capture incomplete; first missing={missing}"
        )
    x0_videos = textvis._decode_x0(pipe, model_fn.x0_latents)
    output_dir = _mapping_dir(Path(kwargs["output_video"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = _render_entity_mapping(
        output_dir=output_dir,
        case_stem=case_stem,
        result=result,
        recorder=attention,
        noun_details=noun_details,
        noun_spec=noun_spec,
        capture_indices=capture_indices,
        total_steps=sampling_steps,
        x0_videos=x0_videos,
        fps=int(kwargs["fps"]),
    )
    result["entity_mapping_attention"] = mapping
    logs.append(
        "[entity-mapping-attention] "
        f"capture_indices={capture_indices} nouns={list(noun_details)} "
        f"zero_compact_slot_ranks={list(ZERO_COMPACT_SLOT_RANKS)} "
        f"swap_slot_entity_ids={list(SWAP_SLOT_ENTITY_IDS)} "
        f"summary={mapping['summary']}"
    )
    return result, logs


textvis.AttentionRecorder = EntityMappingAttentionRecorder
base._run_single_case_in_process = _run_single_case_with_entity_mapping
base._has_complete_existing_output = _has_complete_mapping_output


def main() -> None:
    weights_root = scheme_c._option_value(sys.argv, "--weights-root")
    if weights_root is None:
        raise ValueError("--weights-root is required")
    audit = scheme_c.audit_entity_checkpoint(weights_root)
    print(f"[entity-checkpoint-audit] {audit}", flush=True)
    scheme_c._install_training_matched_defaults(sys.argv)
    base._install_kubric_runtime_hooks = entity_v2v._install_entity_runtime_hooks
    base.main()


if __name__ == "__main__":
    main()
