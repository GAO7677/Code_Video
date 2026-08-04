#!/usr/bin/env python3
"""Inference entry point restricted to Full-SA checkpoints without object modules."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from pathlib import Path

import torch

import infer_xssc_object_self_attn_lora as common

_COMMON_LOAD_RESOLVED_CONFIG = common._load_resolved_config
_COMMON_BUILD_RUNTIME_MODEL = common._build_runtime_model
RANKING_CSV = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "three_model_combined_summary.csv"
)
QUERY_CHUNK = 128
RANKING_STEPS = 40


def _load_resolved_config(checkpoint):
    config, manifest_path = _COMMON_LOAD_RESOLVED_CONFIG(checkpoint)
    if config["adaptation"]["mode"] != "full_sa":
        raise ValueError("Full-SA-only inference requires adaptation.mode='full_sa'")
    if bool(config["adaptation"].get("enable_object_branch", True)):
        raise ValueError(
            "Full-SA-only inference refuses a checkpoint with object branch enabled"
        )
    return config, manifest_path


class AdaptiveAttentionProbabilityNoise:
    def __init__(self, pipe, group: str, alpha: float, seed: int) -> None:
        if not (group.startswith("top") or group.startswith("bottom")):
            raise ValueError(f"Invalid ATTENTION_NOISE_GROUP: {group}")
        self.pipe = pipe
        self.group = group
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.extreme_count = int(group.removeprefix("top").removeprefix("bottom"))
        if self.extreme_count not in {30, 100}:
            raise ValueError("ATTENTION_NOISE_GROUP must be top30/bottom30/top100/bottom100")
        self.heads_by_step_block = self._load_heads()
        self.current_ranking_step = -1
        self.original_model_fn = pipe.model_fn
        self.original_forwards = []
        for dit in (pipe.dit, getattr(pipe, "dit2", None)):
            if dit is None:
                continue
            for block_index, block in enumerate(dit.blocks):
                attention = block.self_attn.attn
                original = attention.forward
                self.original_forwards.append((attention, original))

                def wrapped(q, k, v, *, _original=original, _block=block_index):
                    return self._attention(q, k, v, _original, _block)

                attention.forward = wrapped
        pipe.model_fn = self._wrapped_model_fn

    def _load_heads(self):
        rows_by_step = {step: [] for step in range(RANKING_STEPS)}
        with RANKING_CSV.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("scope") != "objects":
                    continue
                step = int(row["step"])
                if step in rows_by_step:
                    rows_by_step[step].append(
                        (float(row["macro_pck32"]), int(row["block"]), int(row["head"]))
                    )
        selected = {}
        descending = self.group.startswith("top")
        for step, rows in rows_by_step.items():
            if len(rows) != 720:
                raise RuntimeError(f"Ranking step {step} has {len(rows)} rows, expected 720")
            rows.sort(key=lambda item: ((-item[0]) if descending else item[0], item[1], item[2]))
            for _score, block, head in rows[: self.extreme_count]:
                selected.setdefault((step, block), []).append(head)
        return selected

    def _wrapped_model_fn(self, *args, **kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None:
            self.current_ranking_step = -1
        else:
            value = float(timestep.detach().flatten()[0].cpu().item())
            scheduler_steps = self.pipe.scheduler.timesteps.float().cpu()
            runtime_step = int(torch.argmin(torch.abs(scheduler_steps - value)).item())
            denominator = max(1, len(scheduler_steps) - 1)
            self.current_ranking_step = int(
                round(runtime_step * (RANKING_STEPS - 1) / denominator)
            )
        return self.original_model_fn(*args, **kwargs)

    def _noise_seed(self, block: int, heads: list[int], start: int) -> int:
        key = (
            f"seed={self.seed}|group={self.group}|alpha={self.alpha}|"
            f"step={self.current_ranking_step}|block={block}|"
            f"heads={','.join(map(str, heads))}|chunk={start}"
        )
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")

    def _attention(self, q, k, v, original, block: int):
        heads = self.heads_by_step_block.get((self.current_ranking_step, block), ())
        if not heads:
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
        for start in range(0, sequence, QUERY_CHUNK):
            end = min(start + QUERY_CHUNK, sequence)
            logits = torch.matmul(selected_q[:, :, start:end], key_t).float() * scale
            attention = torch.softmax(logits, dim=-1)
            generator = torch.Generator(device=attention.device)
            generator.manual_seed(self._noise_seed(block, list(heads), start))
            noise = torch.randn(
                attention.shape,
                generator=generator,
                device=attention.device,
                dtype=torch.float32,
            )
            perturbed = (attention + (self.alpha / sequence) * noise).clamp_min(0.0)
            row_sum = perturbed.sum(dim=-1, keepdim=True)
            perturbed = torch.where(
                row_sum > 0,
                perturbed / row_sum.clamp_min(1e-12),
                attention,
            )
            selected_output[:, :, start:end] = torch.matmul(
                perturbed.to(selected_v.dtype), selected_v
            )
        fused = original(q, k, v)
        fused_heads = fused.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3).clone()
        fused_heads[:, heads] = selected_output
        return fused_heads.permute(0, 2, 1, 3).reshape(batch, sequence, -1)


def _build_runtime_model(args):
    model, model_args, runtime_info = _COMMON_BUILD_RUNTIME_MODEL(args)
    group = os.environ.get("ATTENTION_NOISE_GROUP", "").strip().lower()
    if group:
        alpha = float(os.environ["ATTENTION_NOISE_ALPHA"])
        seed = int(os.environ.get("ATTENTION_NOISE_SEED", "851"))
        model._attention_probability_noise = AdaptiveAttentionProbabilityNoise(
            model.pipe, group, alpha, seed
        )
        runtime_info["experiment_info"]["attention_probability_noise"] = {
            "enabled": True,
            "group": group,
            "alpha": alpha,
            "seed": seed,
            "formula": "normalize(clamp(A + alpha / K * epsilon, min=0))",
            "ranking": "per-step three-model combined object PCK@32",
        }
    else:
        runtime_info["experiment_info"]["attention_probability_noise"] = {
            "enabled": False
        }
    return model, model_args, runtime_info


def _install_runtime_hooks() -> None:
    common._load_resolved_config = _load_resolved_config
    common._build_runtime_model = _build_runtime_model
    common._install_runtime_hooks()


def main() -> None:
    common.batch_base._install_kubric_runtime_hooks = _install_runtime_hooks
    common.batch_base.main()


if __name__ == "__main__":
    main()
