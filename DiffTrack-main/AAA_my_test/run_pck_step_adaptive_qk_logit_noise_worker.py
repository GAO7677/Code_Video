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
if NOISE_MODE not in {"logit", "probability_additive", "probability_mono_scale"}:
    raise ValueError(f"Unsupported ATTENTION_NOISE_MODE: {NOISE_MODE}")
if NOISE_MODE == "probability_mono_scale" and ATTENTION_ALPHA < 0:
    raise ValueError("probability_mono_scale requires ATTENTION_NOISE_ALPHA >= 0")

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
        self.capture_per_head = os.environ.get(
            "QK_ATTENTION_CAPTURE_PER_HEAD", "0"
        ).strip().lower() in {"1", "true", "yes", "y"}
        self.capture_small_size = int(
            os.environ.get("QK_ATTENTION_CAPTURE_SMALL_SIZE", "512")
        )
        self.capture_latent_frames = int(
            os.environ.get("QK_ATTENTION_CAPTURE_LATENT_FRAMES", "7")
        )
        self.capture_groups: dict[str, dict[str, Any]] = {}
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
                    "before": None
                    if self.capture_per_head
                    else torch.zeros((sequence, sequence), dtype=torch.float32),
                    "after": None
                    if self.capture_per_head
                    else torch.zeros((sequence, sequence), dtype=torch.float32),
                    "head_instances": 0,
                    "selected": set(),
                    "forward_calls": 0,
                    "clipped_elements": 0,
                    "total_elements": 0,
                    "max_row_sum_error": 0.0,
                    "per_head": self.capture_per_head,
                    "sequence_tokens": sequence,
                    "heads": {},
                },
            )
        for start in range(0, sequence, QUERY_CHUNK):
            end = min(start + QUERY_CHUNK, sequence)
            logits = torch.matmul(selected_q[:, :, start:end], key_t).float() * scale
            before_probabilities = torch.softmax(logits, dim=-1)
            row_std = None
            noise = None
            if self.noise_mode != "probability_mono_scale":
                row_std = logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                generator = torch.Generator(device=logits.device)
                generator.manual_seed(self._seed(block, heads, start))
                noise = torch.randn(
                    logits.shape,
                    generator=generator,
                    device=logits.device,
                    dtype=torch.float32,
                )
            if self.noise_mode == "probability_additive":
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
            elif self.noise_mode == "probability_mono_scale":
                exponent = 1.0 + float(self.attention_alpha)
                probs = before_probabilities.clamp_min(1e-12)
                if exponent != 1.0:
                    probs = torch.pow(probs, exponent)
                    probabilities = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                else:
                    probabilities = before_probabilities
            else:
                noise = noise - noise.mean(dim=-1, keepdim=True)
                noise = noise / noise.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                probabilities = torch.softmax(logits + SIGMA * row_std * noise, dim=-1)
            if capture_entry is not None:
                capture_entry["max_row_sum_error"] = max(
                    capture_entry["max_row_sum_error"],
                    float((probabilities.sum(dim=-1) - 1.0).abs().max().item()),
                )
                if self.capture_per_head:
                    for local_head_idx, head in enumerate(heads):
                        key = f"b{block:02d}_h{head:02d}"
                        head_entry = capture_entry["heads"].setdefault(
                            key,
                            {
                                "before_small": torch.zeros(
                                    (self.capture_small_size, self.capture_small_size),
                                    dtype=torch.float32,
                                ),
                                "after_small": torch.zeros(
                                    (self.capture_small_size, self.capture_small_size),
                                    dtype=torch.float32,
                                ),
                                "small_query_counts": torch.zeros(
                                    self.capture_small_size, dtype=torch.float32
                                ),
                                "before_frame": torch.zeros(
                                    (self.capture_latent_frames, self.capture_latent_frames),
                                    dtype=torch.float32,
                                ),
                                "after_frame": torch.zeros(
                                    (self.capture_latent_frames, self.capture_latent_frames),
                                    dtype=torch.float32,
                                ),
                                "frame_query_counts": torch.zeros(
                                    self.capture_latent_frames, dtype=torch.float32
                                ),
                                "before_entropy_sum": 0.0,
                                "after_entropy_sum": 0.0,
                                "entropy_count": 0,
                                "abs_delta_sum": 0.0,
                                "signed_delta_sum": 0.0,
                                "delta_elements": 0,
                                "max_abs_delta": 0.0,
                                "head_instances": 0,
                                "clipped_elements": 0,
                                "total_elements": 0,
                                "max_row_sum_error": 0.0,
                            },
                        )
                        if sequence % self.capture_small_size:
                            raise RuntimeError(
                                f"sequence {sequence} is not divisible by capture size "
                                f"{self.capture_small_size}"
                            )
                        if sequence % self.capture_latent_frames:
                            raise RuntimeError(
                                f"sequence {sequence} is not divisible by latent frames "
                                f"{self.capture_latent_frames}"
                            )
                        before_head = before_probabilities[:, local_head_idx]
                        after_head = probabilities[:, local_head_idx]
                        delta_head = after_head - before_head
                        query_count = end - start
                        token_bin = sequence // self.capture_small_size
                        query_bins = torch.arange(start, end, dtype=torch.long) // token_bin
                        before_small_rows = (
                            before_head.reshape(
                                batch,
                                query_count,
                                self.capture_small_size,
                                token_bin,
                            )
                            .mean(dim=-1)
                            .sum(dim=0)
                            .detach()
                            .cpu()
                        )
                        after_small_rows = (
                            after_head.reshape(
                                batch,
                                query_count,
                                self.capture_small_size,
                                token_bin,
                            )
                            .mean(dim=-1)
                            .sum(dim=0)
                            .detach()
                            .cpu()
                        )
                        head_entry["before_small"].index_add_(
                            0, query_bins, before_small_rows
                        )
                        head_entry["after_small"].index_add_(
                            0, query_bins, after_small_rows
                        )
                        head_entry["small_query_counts"].index_add_(
                            0,
                            query_bins,
                            torch.full((query_count,), float(batch)),
                        )
                        spatial_tokens = sequence // self.capture_latent_frames
                        query_frames = (
                            torch.arange(start, end, dtype=torch.long) // spatial_tokens
                        )
                        before_frame_rows = (
                            before_head.reshape(
                                batch,
                                query_count,
                                self.capture_latent_frames,
                                spatial_tokens,
                            )
                            .sum(dim=-1)
                            .sum(dim=0)
                            .detach()
                            .cpu()
                        )
                        after_frame_rows = (
                            after_head.reshape(
                                batch,
                                query_count,
                                self.capture_latent_frames,
                                spatial_tokens,
                            )
                            .sum(dim=-1)
                            .sum(dim=0)
                            .detach()
                            .cpu()
                        )
                        head_entry["before_frame"].index_add_(
                            0, query_frames, before_frame_rows
                        )
                        head_entry["after_frame"].index_add_(
                            0, query_frames, after_frame_rows
                        )
                        head_entry["frame_query_counts"].index_add_(
                            0,
                            query_frames,
                            torch.full((query_count,), float(batch)),
                        )
                        head_entry["before_entropy_sum"] += float(
                            (-(before_head * before_head.clamp_min(1e-12).log()).sum(dim=-1))
                            .sum()
                            .item()
                        )
                        head_entry["after_entropy_sum"] += float(
                            (-(after_head * after_head.clamp_min(1e-12).log()).sum(dim=-1))
                            .sum()
                            .item()
                        )
                        head_entry["entropy_count"] += batch * query_count
                        head_entry["abs_delta_sum"] += float(delta_head.abs().sum().item())
                        head_entry["signed_delta_sum"] += float(delta_head.sum().item())
                        head_entry["delta_elements"] += delta_head.numel()
                        head_entry["max_abs_delta"] = max(
                            head_entry["max_abs_delta"], float(delta_head.abs().max().item())
                        )
                        head_entry["max_row_sum_error"] = max(
                            head_entry["max_row_sum_error"],
                            float(
                                (probabilities[:, local_head_idx].sum(dim=-1) - 1.0)
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                        if self.noise_mode == "probability_additive":
                            perturbed_head = (
                                before_probabilities[:, local_head_idx]
                                + (self.attention_alpha / sequence) * noise[:, local_head_idx]
                            )
                            clipped_head = perturbed_head.clamp_min(0.0)
                            row_sum_head = clipped_head.sum(dim=-1, keepdim=True)
                            probabilities_head = torch.where(
                                row_sum_head > 0,
                                clipped_head / row_sum_head.clamp_min(1e-12),
                                before_probabilities[:, local_head_idx],
                            )
                            head_entry["clipped_elements"] += int((perturbed_head < 0).sum().item())
                            head_entry["total_elements"] += perturbed_head.numel()
                            head_entry["max_row_sum_error"] = max(
                                head_entry["max_row_sum_error"],
                                float((probabilities_head.sum(dim=-1) - 1.0).abs().max().item()),
                            )
                else:
                    capture_entry["before"][start:end] += (
                        before_probabilities.sum(dim=(0, 1)).detach().cpu()
                    )
                    capture_entry["after"][start:end] += (
                        probabilities.sum(dim=(0, 1)).detach().cpu()
                    )
            selected_output[:, :, start:end] = torch.matmul(
                probabilities.to(dtype=selected_v.dtype), selected_v
            )
        if capture_entry is not None:
            if self.capture_per_head:
                for head in heads:
                    capture_entry["heads"][f"b{block:02d}_h{head:02d}"][
                        "head_instances"
                    ] += batch
            capture_entry["head_instances"] += batch * len(heads)
            capture_entry["selected"].update((block, head) for head in heads)
            capture_entry["forward_calls"] += 1
        fused_output = original(q, k, v)
        output_heads = fused_output.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3).clone()
        output_heads[:, heads] = selected_output
        self.call_count += 1
        return output_heads.permute(0, 2, 1, 3).reshape(batch, sequence, -1)

    @staticmethod
    def _downsample(matrix: torch.Tensor, size: int = 512):
        return torch.nn.functional.adaptive_avg_pool2d(
            matrix[None, None], (size, size)
        )[0, 0].numpy()

    def _frame_matrix(self, matrix: torch.Tensor):
        frame_count = self.capture_latent_frames
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
        self.capture_root.mkdir(parents=True, exist_ok=True)

        def _write_one(
            prefix: str,
            before_tensor: torch.Tensor,
            after_tensor: torch.Tensor,
            head_label: str | None = None,
            frame_before_override=None,
            frame_after_override=None,
            already_downsampled: bool = False,
        ):
            before_mat = before_tensor
            after_mat = after_tensor
            delta = after_mat - before_mat
            before_small = (
                before_mat.numpy() if already_downsampled else self._downsample(before_mat)
            )
            after_small = (
                after_mat.numpy() if already_downsampled else self._downsample(after_mat)
            )
            log_before = np.log10(before_small + 1e-9)
            log_after = np.log10(after_small + 1e-9)
            color_values = np.concatenate((log_before.ravel(), log_after.ravel()))
            vmin, vmax = np.percentile(color_values, [1.0, 99.5])
            delta_small = (
                delta.abs().numpy()
                if already_downsampled
                else self._downsample(delta.abs())
            ).astype(np.float32)
            delta_limit = float(np.percentile(delta_small, 99.5)) or 1e-9
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
            panels = (
                (log_before, "Before: log10 attention", "magma", vmin, vmax),
                (log_after, "After: log10 attention", "magma", vmin, vmax),
                (delta_small, "Mean |After - Before|", "viridis", 0.0, delta_limit),
            )
            for axis, (values, title, cmap, low, high) in zip(axes, panels):
                image = axis.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect="auto")
                axis.set_title(title)
                axis.set_xlabel("Key token index (downsampled)")
                axis.set_ylabel("Query token index (downsampled)")
                fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            strength_label = (
                f"alpha={self.attention_alpha:.2f}"
                if self.noise_mode == "probability_additive"
                else f"mono_scale_alpha={self.attention_alpha:.2f}"
                if self.noise_mode == "probability_mono_scale"
                else f"sigma={SIGMA:.2f}"
            )
            tag = f" {head_label}" if head_label else ""
            fig.suptitle(
                f"{group.upper()}{tag} | S{self.capture_step:03d} | "
                f"{self.noise_mode} | {strength_label}"
            )
            fig.savefig(self.capture_root / f"{prefix}__all_token.png", dpi=160)
            plt.close(fig)

            frame_before = (
                frame_before_override
                if frame_before_override is not None
                else self._frame_matrix(before_mat)
            )
            frame_after = (
                frame_after_override
                if frame_after_override is not None
                else self._frame_matrix(after_mat)
            )
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
                    axis.set_xticks(range(self.capture_latent_frames))
                    axis.set_yticks(range(self.capture_latent_frames))
                    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
                fig.suptitle(
                    f"Frame-level attention mass | {group.upper()}{tag} | S{self.capture_step:03d}"
                )
                fig.savefig(self.capture_root / f"{prefix}__frame.png", dpi=180)
                plt.close(fig)
        if not entry.get("per_head", False):
            before = entry["before"] / max(1, int(entry["head_instances"]))
            after = entry["after"] / max(1, int(entry["head_instances"]))
            prefix = (
                f"{self.capture_model}__{self.capture_case}__{group}"
                f"__{self.noise_mode}"
                f"__step{self.capture_step:02d}"
            )
            _write_one(prefix, before, after)
            delta = after - before
            before_entropy = float((-(before * (before + 1e-12).log()).sum(dim=-1)).mean())
            after_entropy = float((-(after * (after + 1e-12).log()).sum(dim=-1)).mean())
            mean_abs = float(delta.abs().mean())
            mean_delta = float(delta.mean())
            max_delta = float(delta.abs().max())
            metadata = {
                "model": self.capture_model,
                "case": self.capture_case,
                "group": group,
                "step": self.capture_step,
                "intervention": self.noise_mode,
                "sigma": SIGMA if self.noise_mode == "logit" else None,
                "alpha": (
                    self.attention_alpha
                    if self.noise_mode in {"probability_additive", "probability_mono_scale"}
                    else None
                ),
                "noise_scale_per_attention_element": (
                    self.attention_alpha / int(before.shape[0])
                    if self.noise_mode == "probability_additive"
                    else None
                ),
                "qkv_modified": False,
                "sequence_tokens": int(before.shape[0]),
                "latent_frames": self.capture_latent_frames,
                "unique_block_heads": len(entry["selected"]),
                "head_instances": entry["head_instances"],
                "forward_calls": entry["forward_calls"],
                "selected_block_heads": [
                    {"block": block, "head": head}
                    for block, head in sorted(entry["selected"])
                ],
                "mean_abs_attention_delta": float(mean_abs),
                "max_mean_abs_attention_delta": float(max_delta),
                "signed_aggregate_mean_abs_delta": float(mean_delta),
                "clipped_fraction": (
                    entry["clipped_elements"] / entry["total_elements"]
                    if entry["total_elements"]
                    else 0.0
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
            print(
                f"[qk-capture] wrote {group} attention comparison to {self.capture_root}",
                flush=True,
            )
            return

        for key, head_entry in sorted(entry["heads"].items()):
            parts = key.split("_")
            block = int(parts[0][1:])
            head = int(parts[1][1:])
            label = f"{key.replace('_', ' ').upper()}"
            small_counts = head_entry["small_query_counts"].clamp_min(1.0)[:, None]
            frame_counts = head_entry["frame_query_counts"].clamp_min(1.0)[:, None]
            before = head_entry["before_small"] / small_counts
            after = head_entry["after_small"] / small_counts
            frame_before = (head_entry["before_frame"] / frame_counts).numpy()
            frame_after = (head_entry["after_frame"] / frame_counts).numpy()
            prefix = (
                f"{self.capture_model}__{self.capture_case}__{group}_{key}"
                f"__{self.noise_mode}"
                f"__step{self.capture_step:02d}"
            )
            _write_one(
                prefix,
                before,
                after,
                label,
                frame_before_override=frame_before,
                frame_after_override=frame_after,
                already_downsampled=True,
            )
            entropy_count = max(1, int(head_entry["entropy_count"]))
            delta_elements = max(1, int(head_entry["delta_elements"]))
            before_entropy = head_entry["before_entropy_sum"] / entropy_count
            after_entropy = head_entry["after_entropy_sum"] / entropy_count
            mean_abs = head_entry["abs_delta_sum"] / delta_elements
            mean_delta = head_entry["signed_delta_sum"] / delta_elements
            max_delta = head_entry["max_abs_delta"]
            metadata = {
                "model": self.capture_model,
                "case": self.capture_case,
                "group": f"{group}_{key}",
                "step": self.capture_step,
                "intervention": self.noise_mode,
                "sigma": SIGMA if self.noise_mode == "logit" else None,
                "alpha": (
                    self.attention_alpha
                    if self.noise_mode in {"probability_additive", "probability_mono_scale"}
                    else None
                ),
                "noise_scale_per_attention_element": (
                    self.attention_alpha / int(before.shape[0])
                    if self.noise_mode == "probability_additive"
                    else None
                ),
                "qkv_modified": False,
                "sequence_tokens": int(entry["sequence_tokens"]),
                "heatmap_tokens": self.capture_small_size,
                "latent_frames": self.capture_latent_frames,
                "unique_block_heads": 1,
                "head_instances": head_entry["head_instances"],
                "forward_calls": entry["forward_calls"],
                "selected_block_heads": [{"block": block, "head": head}],
                "mean_abs_attention_delta": float(mean_abs),
                "max_mean_abs_attention_delta": float(max_delta),
                "signed_aggregate_mean_abs_delta": float(mean_delta),
                "clipped_fraction": (
                    head_entry["clipped_elements"] / head_entry["total_elements"]
                    if head_entry["total_elements"]
                    else 0.0
                ),
                "max_row_sum_error": head_entry["max_row_sum_error"],
                "before_mean_row_entropy": before_entropy,
                "after_mean_row_entropy": after_entropy,
                "entropy_change": after_entropy - before_entropy,
                "all_token_image": f"{prefix}__all_token.png",
                "frame_image": f"{prefix}__frame.png",
            }
            (self.capture_root / f"{prefix}.json").write_text(
                json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
            )
            print(
                f"[qk-capture] wrote {group}/{key} attention comparison to {self.capture_root}",
                flush=True,
            )

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
    zeroer.flush_capture(group)
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
    (root / "QK_LOGIT_NOISE_EXPERIMENT.json").write_text(
        json.dumps(
            {
                "intervention": (
                    "direct_attention_probability_additive_noise"
                    if NOISE_MODE == "probability_additive"
                    else (
                        "direct_attention_probability_mono_scale"
                        if NOISE_MODE == "probability_mono_scale"
                        else "direct_qk_logit_noise"
                    )
                ),
                "qkv_modified": False,
                "sigma_relative_to_per_query_logit_std": (
                    SIGMA if NOISE_MODE == "logit" else None
                ),
                "attention_alpha": (
                    ATTENTION_ALPHA
                    if NOISE_MODE in {"probability_additive", "probability_mono_scale"}
                    else None
                ),
                "attention_noise_scale": (
                    "alpha / num_key_tokens"
                    if NOISE_MODE == "probability_additive"
                    else ("probability power transformation" if NOISE_MODE == "probability_mono_scale" else None)
                ),
                "probability_postprocess": (
                    "clamp_min_0_then_row_normalize"
                    if NOISE_MODE == "probability_additive"
                    else (
                        "power_transform_then_renormalize"
                        if NOISE_MODE == "probability_mono_scale"
                        else None
                    )
                ),
                "query_chunk": QUERY_CHUNK,
                "selection": "per-step three-model combined object PCK@32 adaptive top/bottom extremes",
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
