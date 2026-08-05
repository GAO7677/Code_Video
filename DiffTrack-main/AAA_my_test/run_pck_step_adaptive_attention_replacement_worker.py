#!/usr/bin/env python3
#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np


BASE_WORKER = Path(__file__).with_name("run_pck_extreme_head_zero_ablation_worker.py")
RANKING_CSV = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "three_model_combined_summary.csv"
)
NUM_STEPS = 40
SIGMA = 0.30
QUERY_CHUNK = 128
NOISE_MODE = os.environ.get("ATTENTION_NOISE_MODE", "logit").strip().lower()
ATTENTION_ALPHA = float(os.environ.get("ATTENTION_NOISE_ALPHA", "0.30"))
CFG_BRANCH_MODE = os.environ.get("ATTENTION_CFG_BRANCH_MODE", "both").strip().lower()
if CFG_BRANCH_MODE not in {"both", "conditional", "unconditional"}:
    raise ValueError(f"Unsupported ATTENTION_CFG_BRANCH_MODE: {CFG_BRANCH_MODE}")
if NOISE_MODE not in {
    "logit",
    "probability_additive",
    "probability_zero",
    "probability_uniform",
    "probability_temporal_causal",
    "probability_strict_past",
    "probability_strict_future",
    "probability_exclude_current",
    "probability_context_only",
    "probability_object_query_continuity",
    "probability_object_query_main_component_continuity",
    "probability_object_query_identity",
    "probability_identity",
}:
    raise ValueError(f"Unsupported ATTENTION_NOISE_MODE: {NOISE_MODE}")

_CAPTURE_PROMPT_CASES: dict[str, list[str]] = {}
_CAPTURE_PROMPT_POSITIONS: dict[str, int] = {}


def load_capture_prompt_cases() -> None:
    if "--input-json-list" not in sys.argv:
        return
    input_list = Path(sys.argv[sys.argv.index("--input-json-list") + 1])
    seen_cases = set()
    for line in input_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        case_path = Path(line)
        case_key = case_path.stem
        if case_key in seen_cases:
            continue
        seen_cases.add(case_key)
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        prompt = payload.get("caption", payload.get("input_caption"))
        if not prompt:
            raise KeyError(
                f"Neither caption nor input_caption exists in test case: {case_path}"
            )
        prompt = str(prompt)
        _CAPTURE_PROMPT_CASES.setdefault(prompt, []).append(case_key)

spec = importlib.util.spec_from_file_location("pck_qk_base_worker", BASE_WORKER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base worker: {BASE_WORKER}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def select_step_heads(ranking_pool: str, extreme_count: int) -> dict[str, list[dict]]:
    if ranking_pool != "all720" or extreme_count not in {30, 100}:
        raise ValueError(
            "Attention noise experiment requires all720 and extreme-count in {30, 100}"
        )
    by_step: dict[int, list[dict[str, Any]]] = {step: [] for step in range(NUM_STEPS)}
    with RANKING_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scope") != "objects":
                continue
            step = int(row["step"])
            if step in by_step:
                by_step[step].append(
                    {
                        "step": step,
                        "block": int(row["block"]),
                        "head": int(row["head"]),
                        "macro_pck32": float(row["macro_pck32"]),
                        "timestep": float(row["timestep"]),
                        "sigma": float(row["sigma"]),
                    }
                )
    top_group = f"top{extreme_count}"
    bottom_group = f"bottom{extreme_count}"
    groups: dict[str, list[dict]] = {top_group: [], bottom_group: []}
    for step, rows in by_step.items():
        if len(rows) != 720 or len({(r["block"], r["head"]) for r in rows}) != 720:
            raise RuntimeError(f"step {step} does not contain 720 unique object heads")
        ranked = sorted(rows, key=lambda r: (-r["macro_pck32"], r["block"], r["head"]))
        groups[f"{top_group}_step_{step:02d}"] = [
            dict(row, rank_within_step=index + 1)
            for index, row in enumerate(ranked[:extreme_count])
        ]
        groups[f"{bottom_group}_step_{step:02d}"] = [
            dict(row, rank_within_step=720 - index)
            for index, row in enumerate(
                sorted(
                    ranked[-extreme_count:],
                    key=lambda r: (r["macro_pck32"], r["block"], r["head"]),
                )
            )
        ]
    return groups


class AdaptiveQKLogitNoise:
    def __init__(self, pipe, groups: dict[str, list[dict]]) -> None:
        self.pipe = pipe
        self.group: str | None = None
        self.adaptive_prefix: str | None = None
        self.active_steps: set[int] = set()
        self.current_step = -1
        self.current_cfg_branch = "conditional"
        self._cfg_step = -1
        self._cfg_call_index = 0
        self.call_count = 0
        self.noise_context = "unset"
        capture_root = os.environ.get("QK_ATTENTION_CAPTURE_ROOT", "").strip()
        self.capture_root = Path(capture_root) if capture_root else None
        self.capture_step = int(os.environ.get("QK_ATTENTION_CAPTURE_STEP", "39"))
        self.capture_model = os.environ.get("QK_ATTENTION_CAPTURE_MODEL", "baseline")
        self.capture_case = os.environ.get("QK_ATTENTION_CAPTURE_CASE", "case")
        self.noise_mode = NOISE_MODE
        self.attention_alpha = ATTENTION_ALPHA
        self.capture_groups: dict[str, dict[str, Any]] = {}
        self.object_continuity_regions = []
        self.object_continuity_context_frame = None
        self.object_continuity_entries: dict[tuple, dict[str, Any]] = {}
        self.object_continuity_capture_root = None
        self.object_continuity_capture_heads: dict[tuple[int, int], float] = {}
        if self.noise_mode in {
            "probability_object_query_continuity",
            "probability_object_query_main_component_continuity",
            "probability_object_query_identity",
        }:
            from AAA_my_test.object_query_attention_capture_headwise_pck import pck_query_regions

            self.object_continuity_regions, self.object_continuity_context_frame = pck_query_regions()
            capture_root = os.environ.get("OBJECT_CONTINUITY_CAPTURE_ROOT", "").strip()
            self.object_continuity_capture_root = Path(capture_root) if capture_root else None
            capture_head_limit = int(
                os.environ.get("OBJECT_CONTINUITY_CAPTURE_HEAD_LIMIT", "10")
            )
            for name, entries in groups.items():
                if name.startswith("top100_step_"):
                    for entry in entries[:capture_head_limit]:
                        key = (int(entry["block"]), int(entry["head"]))
                        self.object_continuity_capture_heads[key] = float(
                            entry.get("ranking_score", entry.get("macro_pck32", 0.0))
                        )
                    break
        self.original_model_fn = pipe.model_fn
        self.original_forwards: list[tuple[Any, Any]] = []
        by_block: dict[int, dict[str, list[int]]] = {}
        for group, entries in groups.items():
            for entry in entries:
                by_block.setdefault(int(entry["block"]), {}).setdefault(group, []).append(
                    int(entry["head"])
                )
        for dit in (pipe.dit, getattr(pipe, "dit2", None)):
            if dit is None:
                continue
            for block_index, block in enumerate(dit.blocks):
                attn = block.self_attn.attn
                original = attn.forward
                self.original_forwards.append((attn, original))
                block_groups = by_block.get(block_index, {})

                def wrapped(q, k, v, *, _original=original, _groups=block_groups, _block=block_index):
                    return self._attention(q, k, v, _original, _groups, _block)

                attn.forward = wrapped
        pipe.model_fn = self._wrapped_model_fn

    def set_noise_context(self, prompt: str, group: str | None) -> None:
        self.noise_context = f"{prompt}|{group or 'original'}"

    def set_variant(self, group: str | None, steps: tuple[int, ...]) -> None:
        self.adaptive_prefix = group
        self.group = group
        self.active_steps = set(range(NUM_STEPS))

    def _scheduler_step(self, timestep: torch.Tensor) -> int:
        value = float(timestep.detach().flatten()[0].cpu().item())
        timesteps = self.pipe.scheduler.timesteps
        return int(torch.argmin(torch.abs(timesteps.float().cpu() - value)).item())

    def _wrapped_model_fn(self, *args, **kwargs):
        timestep = kwargs.get("timestep")
        self.current_step = self._scheduler_step(timestep) if timestep is not None else -1
        if self.current_step != self._cfg_step:
            self._cfg_step = self.current_step
            self._cfg_call_index = 0
        else:
            self._cfg_call_index += 1
        self.current_cfg_branch = (
            "conditional" if self._cfg_call_index % 2 == 0 else "unconditional"
        )
        branch_active = (
            CFG_BRANCH_MODE == "both" or CFG_BRANCH_MODE == self.current_cfg_branch
        )
        if self.adaptive_prefix is None or self.current_step < 0 or not branch_active:
            self.group = None
        else:
            self.group = f"{self.adaptive_prefix}_step_{self.current_step:02d}"
        return self.original_model_fn(*args, **kwargs)

    def _seed(self, block: int, heads: list[int], chunk_start: int) -> int:
        key = (
            f"{self.noise_context}|step={self.current_step}|block={block}|"
            f"heads={','.join(map(str, heads))}|chunk={chunk_start}|"
            f"mode={self.noise_mode}|sigma={SIGMA}|alpha={self.attention_alpha}"
        )
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")

    def _object_continuity_probabilities(
        self, selected_q, key_t, scale: float, heads: list[int], block: int
    ):
        latent_frames = int(os.environ.get("ATTENTION_MASK_LATENT_FRAMES", "13"))
        context_frames = int(os.environ.get("ATTENTION_MASK_CONTEXT_LATENT_FRAMES", "2"))
        sequence = int(selected_q.shape[2])
        if sequence % latent_frames != 0:
            raise RuntimeError(
                f"Sequence {sequence} is not divisible by {latent_frames} latent frames"
            )
        spatial_tokens = sequence // latent_frames
        if spatial_tokens != 16 * 28:
            raise RuntimeError(f"Expected 16x28 spatial tokens, got {spatial_tokens}")
        query_rows = sorted(
            {
                int(index)
                for region in self.object_continuity_regions
                for index in region["token_indices"]
            }
        )
        row_tensor = torch.as_tensor(query_rows, device=selected_q.device, dtype=torch.long)
        logits = torch.matmul(selected_q[:, :, row_tensor], key_t).float() * scale
        before = torch.softmax(logits, dim=-1)
        after = before.clone()
        row_positions = {row: index for index, row in enumerate(query_rows)}
        quantile = float(os.environ.get("OBJECT_CONTINUITY_HIGH_QUANTILE", "0.90"))
        radius = int(os.environ.get("OBJECT_CONTINUITY_NEIGHBOR_RADIUS", "1"))
        capture_step = int(os.environ.get("OBJECT_CONTINUITY_CAPTURE_STEP", "39"))

        for region in self.object_continuity_regions:
            region_rows = [int(index) for index in region["token_indices"]]
            positions = torch.as_tensor(
                [row_positions[index] for index in region_rows],
                device=selected_q.device,
                dtype=torch.long,
            )
            region_before = before[:, :, positions]
            response = region_before.mean(dim=2).reshape(
                region_before.shape[0], region_before.shape[1], latent_frames, 16, 28
            )
            thresholds = torch.quantile(
                response.flatten(-2), quantile, dim=-1, keepdim=True
            ).unsqueeze(-1)
            high = response >= thresholds
            top_k = min(
                int(os.environ.get("OBJECT_CONTINUITY_MAIN_COMPONENT_TOPK", "5")),
                16 * 28,
            )
            flat_response = response.flatten(-2)
            top_indices = torch.topk(flat_response, k=top_k, dim=-1).indices
            topk_candidate = torch.zeros_like(flat_response, dtype=torch.bool)
            topk_candidate.scatter_(-1, top_indices, True)
            topk_candidate = topk_candidate.reshape_as(response)
            component_shape = topk_candidate.shape
            candidate_2d = topk_candidate.reshape(-1, 16, 28)
            peak_indices = flat_response.argmax(dim=-1).reshape(-1, 1)
            main_component = torch.zeros_like(
                candidate_2d.reshape(-1, 16 * 28), dtype=torch.bool
            )
            main_component.scatter_(-1, peak_indices, True)
            main_component = main_component.reshape(-1, 16, 28)
            for _ in range(top_k):
                main_component = candidate_2d & (
                    torch.nn.functional.max_pool2d(
                        main_component.float().unsqueeze(1),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ).squeeze(1) > 0
                )
            main_component = main_component.reshape(component_shape)
            if self.noise_mode == "probability_object_query_continuity":
                rejected = torch.zeros_like(high)
                previous = high[:, :, context_frames - 1]
                kernel = 2 * radius + 1
                for frame in range(context_frames, latent_frames):
                    neighboring = torch.nn.functional.max_pool2d(
                        previous.float(), kernel_size=kernel, stride=1, padding=radius
                    ) > 0
                    kept = high[:, :, frame] & neighboring
                    rejected[:, :, frame] = high[:, :, frame] & ~neighboring
                    previous = kept
                rejected_flat = rejected.flatten(-2).reshape(
                    rejected.shape[0], rejected.shape[1], latent_frames * spatial_tokens
                )
                region_after = region_before.masked_fill(rejected_flat.unsqueeze(2), 0.0)
                row_sum = region_after.sum(dim=-1, keepdim=True)
                region_after = torch.where(
                    row_sum > 0,
                    region_after / row_sum.clamp_min(1e-12),
                    region_before,
                )
            elif self.noise_mode == "probability_object_query_main_component_continuity":
                rejected = torch.zeros_like(high)
                kernel = 2 * radius + 1
                for frame in range(context_frames, latent_frames):
                    previous_component = main_component[:, :, frame - 1]
                    neighboring = torch.nn.functional.max_pool2d(
                        previous_component.float(),
                        kernel_size=kernel,
                        stride=1,
                        padding=radius,
                    ) > 0
                    rejected[:, :, frame] = high[:, :, frame] & ~neighboring
                rejected_flat = rejected.flatten(-2).reshape(
                    rejected.shape[0], rejected.shape[1],
                    latent_frames * spatial_tokens,
                )
                region_after = region_before.masked_fill(
                    rejected_flat.unsqueeze(2), 0.0
                )
                row_sum = region_after.sum(dim=-1, keepdim=True)
                region_after = torch.where(
                    row_sum > 0,
                    region_after / row_sum.clamp_min(1e-12),
                    region_before,
                )
            else:
                rejected = torch.zeros_like(high)
                region_after = region_before
            after[:, :, positions] = region_after

            if (
                self.object_continuity_capture_root is not None
                and self.current_step == capture_step
            ):
                prefix = (self.group or "").split("_step_", 1)[0]
                for head_offset, head in enumerate(heads):
                    score = self.object_continuity_capture_heads.get((block, head))
                    if score is None:
                        continue
                    key = (prefix, block, head, region["name"])
                    entry = self.object_continuity_entries.setdefault(
                        key,
                        {
                            "before": torch.zeros(
                                (len(region_rows), latent_frames, 16, 28)
                            ),
                            "after": torch.zeros(
                                (len(region_rows), latent_frames, 16, 28)
                            ),
                            "removed": torch.zeros(
                                (len(region_rows), latent_frames, 16, 28)
                            ),
                            "p90_frequency": torch.zeros((latent_frames, 16, 28)),
                            "main_component_frequency": torch.zeros(
                                (latent_frames, 16, 28)
                            ),
                            "count": 0,
                            "region": region,
                            "pck32": score,
                        },
                    )
                    before_map = region_before[:, head_offset].mean(dim=0).reshape(
                        len(region_rows), latent_frames, 16, 28
                    ).detach().cpu()
                    after_map = region_after[:, head_offset].mean(dim=0).reshape(
                        len(region_rows), latent_frames, 16, 28
                    ).detach().cpu()
                    entry["before"] += before_map
                    entry["after"] += after_map
                    entry["removed"] += (before_map - after_map).clamp_min(0)
                    entry["p90_frequency"] += high[:, head_offset].float().mean(dim=0).detach().cpu()
                    entry["main_component_frequency"] += (
                        main_component[:, head_offset].float().mean(dim=0).detach().cpu()
                    )
                    entry["count"] += 1
        return query_rows, after

    def flush_object_continuity_capture(self, group: str | None) -> None:
        if not group or self.object_continuity_capture_root is None:
            return
        prefix = group.split("_step_", 1)[0]
        keys = [key for key in self.object_continuity_entries if key[0] == prefix]
        if not keys:
            return
        self.object_continuity_capture_root.mkdir(parents=True, exist_ok=True)
        capture_step = int(os.environ.get("OBJECT_CONTINUITY_CAPTURE_STEP", "39"))
        case = os.environ.get("QK_ATTENTION_CAPTURE_CASE", "case")
        seed = int(os.environ.get("ATTENTION_NOISE_SEED", "0"))
        for key in keys:
            _prefix, block, head, region_name = key
            entry = self.object_continuity_entries.pop(key)
            count = max(1, int(entry["count"]))
            region = entry["region"]
            filename = (
                f"{case}__seed{seed:06d}__{prefix}__{region_name}"
                f"__step{capture_step:02d}__b{block:02d}_h{head:02d}.npz"
            )
            np.savez_compressed(
                self.object_continuity_capture_root / filename,
                before=(entry["before"] / count).numpy(),
                after=(entry["after"] / count).numpy(),
                removed=(entry["removed"] / count).numpy(),
                p90_frequency=(entry["p90_frequency"] / count).numpy(),
                main_component_frequency=(
                    entry["main_component_frequency"] / count
                ).numpy(),
                query_points=region["points"],
                query_mask=region["mask"],
                query_token_indices=region["token_indices"],
                query_context_frame=self.object_continuity_context_frame,
                query_latent_frame=np.int32(1),
                query_pixel_frame=np.int32(4),
                region_name=np.asarray(region_name),
                region_phrase=np.asarray(region["phrase"]),
                pck32=np.float32(entry["pck32"]),
                block=np.int32(block),
                head=np.int32(head),
                step=np.int32(capture_step),
                seed=np.int32(seed),
                high_quantile=np.float32(
                    float(os.environ.get("OBJECT_CONTINUITY_HIGH_QUANTILE", "0.90"))
                ),
                neighbor_radius=np.int32(
                    int(os.environ.get("OBJECT_CONTINUITY_NEIGHBOR_RADIUS", "1"))
                ),
                main_component_topk=np.int32(
                    int(os.environ.get("OBJECT_CONTINUITY_MAIN_COMPONENT_TOPK", "5"))
                ),
                mode=np.asarray(self.noise_mode),
                protocol=np.asarray("headwise_pck_sam2_context_f04"),
                query_aggregation=np.asarray("preserve_then_sum"),
            )

    def _attention(self, q, k, v, original, groups: dict[str, list[int]], block: int):
        heads = sorted(set(groups.get(self.group or "", ())))
        if not heads or self.current_step not in self.active_steps:
            return original(q, k, v)
        num_heads = int(q.shape[-1] // 128)
        head_dim = int(q.shape[-1] // num_heads)
        batch, sequence, _ = q.shape
        qh = q.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3)
        kh = k.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3)
        vh = v.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3)
        selected_q = qh[:, heads]
        selected_k = kh[:, heads]
        selected_v = vh[:, heads]
        key_t = selected_k.transpose(-1, -2)
        selected_output = torch.empty_like(selected_q)
        scale = 1.0 / math.sqrt(head_dim)
        continuity_rows = []
        continuity_after = None
        if self.noise_mode in {
            "probability_object_query_continuity",
            "probability_object_query_main_component_continuity",
            "probability_object_query_identity",
        }:
            continuity_rows, continuity_after = self._object_continuity_probabilities(
                selected_q, key_t, scale, heads, block
            )
        capture_prefix = (self.group or "").split("_step_", 1)[0]
        capture_enabled = (
            self.capture_root is not None
            and self.current_step == self.capture_step
            and capture_prefix in {"top30", "bottom30", "top100", "bottom100"}
        )
        capture_entry = None
        if capture_enabled:
            capture_entry = self.capture_groups.setdefault(
                capture_prefix,
                {
                    "before": torch.zeros((sequence, sequence), dtype=torch.float32),
                    "after": torch.zeros((sequence, sequence), dtype=torch.float32),
                    "abs_delta": torch.zeros((sequence, sequence), dtype=torch.float32),
                    "head_instances": 0,
                    "selected": set(),
                    "forward_calls": 0,
                    "clipped_elements": 0,
                    "total_elements": 0,
                    "max_row_sum_error": 0.0,
                },
            )
        for start in range(0, sequence, QUERY_CHUNK):
            end = min(start + QUERY_CHUNK, sequence)
            logits = torch.matmul(selected_q[:, :, start:end], key_t).float() * scale
            before_probabilities = torch.softmax(logits, dim=-1)
            row_std = logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(self._seed(block, heads, start))
            noise = torch.randn(
                logits.shape,
                generator=generator,
                device=logits.device,
                dtype=torch.float32,
            )
            if self.noise_mode == "probability_identity":
                probabilities = before_probabilities
            elif self.noise_mode in {
                "probability_object_query_continuity",
                "probability_object_query_identity",
            }:
                probabilities = before_probabilities.clone()
                pairs = [
                    (row - start, index)
                    for index, row in enumerate(continuity_rows)
                    if start <= row < end
                ]
                if pairs:
                    local_rows = torch.as_tensor(
                        [pair[0] for pair in pairs], device=logits.device, dtype=torch.long
                    )
                    source_rows = torch.as_tensor(
                        [pair[1] for pair in pairs], device=logits.device, dtype=torch.long
                    )
                    probabilities[:, :, local_rows] = continuity_after[:, :, source_rows]
            elif self.noise_mode in {
                "probability_temporal_causal",
                "probability_strict_past",
                "probability_strict_future",
                "probability_exclude_current",
                "probability_context_only",
            }:
                latent_frames = int(os.environ.get("ATTENTION_MASK_LATENT_FRAMES", "7"))
                if sequence % latent_frames != 0:
                    raise RuntimeError(
                        f"Sequence {sequence} is not divisible by {latent_frames} latent frames"
                    )
                spatial_tokens = sequence // latent_frames
                query_frames = (
                    torch.arange(start, end, device=logits.device) // spatial_tokens
                )
                key_frames = (
                    torch.arange(sequence, device=logits.device) // spatial_tokens
                )
                if self.noise_mode == "probability_temporal_causal":
                    allowed = key_frames.unsqueeze(0) <= query_frames.unsqueeze(1)
                elif self.noise_mode == "probability_strict_past":
                    allowed = key_frames.unsqueeze(0) < query_frames.unsqueeze(1)
                elif self.noise_mode == "probability_strict_future":
                    allowed = key_frames.unsqueeze(0) > query_frames.unsqueeze(1)
                elif self.noise_mode == "probability_exclude_current":
                    allowed = key_frames.unsqueeze(0) != query_frames.unsqueeze(1)
                else:
                    context_frames = int(
                        os.environ.get("ATTENTION_MASK_CONTEXT_LATENT_FRAMES", "2")
                    )
                    if not 0 < context_frames <= latent_frames:
                        raise RuntimeError(
                            f"Invalid context latent frame count {context_frames}; "
                            f"expected 1..{latent_frames}"
                        )
                    allowed = (key_frames < context_frames).unsqueeze(0).expand(
                        end - start, -1
                    )
                expanded = allowed.unsqueeze(0).unsqueeze(0)
                valid_rows = allowed.any(dim=-1)
                probabilities = torch.zeros_like(before_probabilities)
                if valid_rows.any():
                    probabilities[:, :, valid_rows] = torch.softmax(
                        logits[:, :, valid_rows].masked_fill(
                            ~expanded[:, :, valid_rows], -torch.inf
                        ),
                        dim=-1,
                    )
                if capture_entry is not None:
                    capture_entry["max_row_sum_error"] = max(
                        capture_entry["max_row_sum_error"],
                        float((probabilities.sum(dim=-1) - 1.0).abs().max().item()),
                    )
            elif self.noise_mode == "probability_zero":
                probabilities = torch.zeros_like(before_probabilities)
                if capture_entry is not None:
                    capture_entry["max_row_sum_error"] = 1.0
            elif self.noise_mode == "probability_uniform":
                probabilities = torch.full_like(before_probabilities, 1.0 / sequence)
                if capture_entry is not None:
                    capture_entry["max_row_sum_error"] = max(
                        capture_entry["max_row_sum_error"],
                        float((probabilities.sum(dim=-1) - 1.0).abs().max().item()),
                    )
            elif self.noise_mode == "probability_additive":
                perturbed = before_probabilities + (self.attention_alpha / sequence) * noise
                clipped = perturbed.clamp_min(0.0)
                row_sum = clipped.sum(dim=-1, keepdim=True)
                probabilities = torch.where(
                    row_sum > 0,
                    clipped / row_sum.clamp_min(1e-12),
                    before_probabilities,
                )
                if capture_entry is not None:
                    capture_entry["clipped_elements"] += int((perturbed < 0).sum().item())
                    capture_entry["total_elements"] += perturbed.numel()
                    capture_entry["max_row_sum_error"] = max(
                        capture_entry["max_row_sum_error"],
                        float((probabilities.sum(dim=-1) - 1.0).abs().max().item()),
                    )
            else:
                noise = noise - noise.mean(dim=-1, keepdim=True)
                noise = noise / noise.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                probabilities = torch.softmax(logits + SIGMA * row_std * noise, dim=-1)
            if capture_entry is not None:
                capture_entry["before"][start:end] += (
                    before_probabilities.sum(dim=(0, 1)).detach().cpu()
                )
                capture_entry["after"][start:end] += (
                    probabilities.sum(dim=(0, 1)).detach().cpu()
                )
                capture_entry["abs_delta"][start:end] += (
                    (probabilities - before_probabilities).abs().sum(dim=(0, 1)).detach().cpu()
                )
            selected_output[:, :, start:end] = torch.matmul(
                probabilities.to(dtype=selected_v.dtype), selected_v
            )
        if capture_entry is not None:
            capture_entry["head_instances"] += batch * len(heads)
            capture_entry["selected"].update((block, head) for head in heads)
            capture_entry["forward_calls"] += 1
        fused_output = original(q, k, v)
        if self.noise_mode == "probability_identity":
            return fused_output
        if self.noise_mode == "probability_object_query_identity":
            self.call_count += 1
            return fused_output
        output_heads = fused_output.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3).clone()
        if self.noise_mode == "probability_object_query_continuity":
            row_tensor = torch.as_tensor(
                continuity_rows, device=output_heads.device, dtype=torch.long
            )
            selected_heads_output = output_heads[:, heads].clone()
            selected_heads_output[:, :, row_tensor] = selected_output[:, :, row_tensor]
            output_heads[:, heads] = selected_heads_output
        else:
            output_heads[:, heads] = selected_output
        self.call_count += 1
        return output_heads.permute(0, 2, 1, 3).reshape(batch, sequence, -1)

    @staticmethod
    def _downsample(matrix: torch.Tensor, size: int = 512):
        return torch.nn.functional.adaptive_avg_pool2d(
            matrix[None, None], (size, size)
        )[0, 0].numpy()

    @staticmethod
    def _frame_matrix(matrix: torch.Tensor, frame_count: int = 7):
        sequence = matrix.shape[0]
        if sequence % frame_count:
            return None
        spatial_tokens = sequence // frame_count
        return (
            matrix.reshape(frame_count, spatial_tokens, frame_count, spatial_tokens)
            .sum(dim=-1)
            .mean(dim=1)
            .numpy()
        )

    def flush_capture(self, group: str | None) -> None:
        if self.capture_root is None or group not in self.capture_groups:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        entry = self.capture_groups.pop(group)
        count = max(1, int(entry["head_instances"]))
        before = entry["before"] / count
        after = entry["after"] / count
        delta = after - before
        mean_abs_delta = entry["abs_delta"] / count
        prefix = (
            f"{self.capture_model}__{self.capture_case}__{group}"
            f"__step{self.capture_step:02d}"
        )
        self.capture_root.mkdir(parents=True, exist_ok=True)

        before_small = self._downsample(before)
        after_small = self._downsample(after)
        log_before = np.log10(before_small + 1e-9)
        log_after = np.log10(after_small + 1e-9)
        color_values = np.concatenate((log_before.ravel(), log_after.ravel()))
        vmin, vmax = np.percentile(color_values, [1.0, 99.5])
        delta_small = self._downsample(mean_abs_delta).astype(np.float32)
        delta_limit = float(np.percentile(delta_small, 99.5)) or 1e-9
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
        panels = (
            (log_before, "Before: log10 attention", "magma", vmin, vmax),
            (log_after, "After: log10 attention", "magma", vmin, vmax),
            (delta_small, "Mean per-head |After - Before|", "viridis", 0.0, delta_limit),
        )
        for axis, (values, title, cmap, low, high) in zip(axes, panels):
            image = axis.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect="auto")
            axis.set_title(title)
            axis.set_xlabel("Key token index (downsampled)")
            axis.set_ylabel("Query token index (downsampled)")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        if self.noise_mode == "probability_temporal_causal":
            strength_label = "temporal causal mask"
        elif self.noise_mode == "probability_zero":
            strength_label = "A'=0"
        elif self.noise_mode == "probability_uniform":
            strength_label = "A'=1/N_K"
        elif self.noise_mode == "probability_additive":
            strength_label = f"alpha={self.attention_alpha:.2f}"
        else:
            strength_label = f"sigma={SIGMA:.2f}"
        fig.suptitle(
            f"{group.upper()} | S{self.capture_step:03d} | "
            f"{self.noise_mode} | {strength_label}"
        )
        fig.savefig(self.capture_root / f"{prefix}__all_token.png", dpi=160)
        plt.close(fig)

        frame_before = self._frame_matrix(before)
        frame_after = self._frame_matrix(after)
        if frame_before is not None and frame_after is not None:
            frame_delta = frame_after - frame_before
            frame_limit = float(np.max(np.abs(frame_delta))) or 1e-9
            fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
            frame_panels = (
                (frame_before, "Before", "magma", 0.0, 1.0),
                (frame_after, "After", "magma", 0.0, 1.0),
                (frame_delta, "After - Before", "coolwarm", -frame_limit, frame_limit),
            )
            for axis, (values, title, cmap, low, high) in zip(axes, frame_panels):
                image = axis.imshow(values, cmap=cmap, vmin=low, vmax=high)
                axis.set_title(title)
                axis.set_xlabel("Key latent frame")
                axis.set_ylabel("Query latent frame")
                axis.set_xticks(range(7))
                axis.set_yticks(range(7))
                fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            fig.suptitle(f"Frame-level attention mass | {group.upper()} | S{self.capture_step:03d}")
            fig.savefig(self.capture_root / f"{prefix}__frame.png", dpi=180)
            plt.close(fig)

        before_entropy = float((-(before * (before + 1e-12).log()).sum(dim=-1)).mean())
        after_entropy = float((-(after * (after + 1e-12).log()).sum(dim=-1)).mean())
        metadata = {
            "model": self.capture_model,
            "case": self.capture_case,
            "group": group,
            "step": self.capture_step,
            "intervention": self.noise_mode,
            "cfg_branch_mode": CFG_BRANCH_MODE,
            "sigma": SIGMA if self.noise_mode == "logit" else None,
            "alpha": self.attention_alpha if self.noise_mode == "probability_additive" else None,
            "noise_scale_per_attention_element": (
                self.attention_alpha / int(before.shape[0])
                if self.noise_mode == "probability_additive"
                else None
            ),
            "qkv_modified": False,
            "sequence_tokens": int(before.shape[0]),
            "latent_frames": 7 if before.shape[0] % 7 == 0 else None,
            "unique_block_heads": len(entry["selected"]),
            "head_instances": entry["head_instances"],
            "forward_calls": entry["forward_calls"],
            "selected_block_heads": [
                {"block": block, "head": head}
                for block, head in sorted(entry["selected"])
            ],
            "mean_abs_attention_delta": float(mean_abs_delta.mean()),
            "max_mean_abs_attention_delta": float(mean_abs_delta.max()),
            "signed_aggregate_mean_abs_delta": float(delta.abs().mean()),
            "clipped_fraction": (
                entry["clipped_elements"] / entry["total_elements"]
                if entry["total_elements"]
                else None
            ),
            "max_row_sum_error": entry["max_row_sum_error"],
            "before_mean_row_entropy": before_entropy,
            "after_mean_row_entropy": after_entropy,
            "entropy_change": after_entropy - before_entropy,
            "all_token_image": f"{prefix}__all_token.png",
            "frame_image": f"{prefix}__frame.png",
        }
        (self.capture_root / f"{prefix}.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        print(f"[qk-capture] wrote {group} attention comparison to {self.capture_root}", flush=True)

    def remove(self) -> None:
        self.pipe.model_fn = self.original_model_fn
        for attn, original in self.original_forwards:
            attn.forward = original


original_generate = base.generate


def generate_with_context(pipe, zeroer, context, prompt, group, steps):
    if group is None or group == "original":
        cases = _CAPTURE_PROMPT_CASES.get(prompt, ())
        position = _CAPTURE_PROMPT_POSITIONS.get(prompt, 0)
        if cases:
            zeroer.capture_case = cases[min(position, len(cases) - 1)]
            _CAPTURE_PROMPT_POSITIONS[prompt] = position + 1
    zeroer.set_noise_context(prompt, group)
    result = original_generate(pipe, zeroer, context, prompt, group, steps)
    zeroer.flush_object_continuity_capture(group)
    zeroer.flush_capture(group)
    if NOISE_MODE == "probability_identity" and group is not None:
        zeroer.call_count += 1
    return result


base.select_heads = select_step_heads
base.ExtremeHeadZeroer = AdaptiveQKLogitNoise
base.generate = generate_with_context
base.STAGE_RANGES = (("steps_00_40", tuple(range(NUM_STEPS))),)


def write_experiment_metadata() -> None:
    if "--output-root" not in sys.argv:
        return
    root = Path(sys.argv[sys.argv.index("--output-root") + 1]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "ATTENTION_PROBABILITY_REPLACEMENT_EXPERIMENT.json").write_text(
        json.dumps(
            {
                "intervention": {
                    "probability_identity": "no_attention_intervention",
                    "probability_zero": "direct_attention_probability_zero",
                    "probability_uniform": "direct_attention_probability_uniform",
                    "probability_temporal_causal": "temporal_causal_attention_mask",
                    "probability_strict_past": "strict_past_attention_mask",
                    "probability_strict_future": "strict_future_attention_mask",
                    "probability_exclude_current": "exclude_current_frame_attention_mask",
                    "probability_context_only": "context_frames_only_attention_mask",
                    "probability_object_query_continuity": "object_query_temporal_spatial_continuity",
                    "probability_object_query_main_component_continuity": "object_query_main_component_temporal_spatial_continuity",
                    "probability_object_query_identity": "object_query_identity_no_intervention",
                    "probability_additive": "direct_attention_probability_additive_noise",
                    "logit": "direct_qk_logit_noise",
                }[NOISE_MODE],
                "qkv_modified": False,
                "sigma_relative_to_per_query_logit_std": (
                    SIGMA if NOISE_MODE == "logit" else None
                ),
                "attention_alpha": (
                    ATTENTION_ALPHA if NOISE_MODE == "probability_additive" else None
                ),
                "attention_noise_scale": (
                    "alpha / num_key_tokens"
                    if NOISE_MODE == "probability_additive"
                    else None
                ),
                "probability_postprocess": (
                    "clamp_min_0_then_row_normalize"
                    if NOISE_MODE == "probability_additive"
                    else None
                ),
                "query_chunk": QUERY_CHUNK,
                "selection": "per-step three-model combined object PCK@32 adaptive extremes",
                "denoising_steps": list(range(NUM_STEPS)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    load_capture_prompt_cases()
    write_experiment_metadata()
    base.main()
