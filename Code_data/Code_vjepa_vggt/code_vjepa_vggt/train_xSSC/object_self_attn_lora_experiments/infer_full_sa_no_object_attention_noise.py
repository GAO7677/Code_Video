#!/usr/bin/env python3
"""Full-SA/no-object inference with direct attention-probability perturbation."""

from __future__ import annotations

import csv
import atexit
import hashlib
import json
import math
import os
from pathlib import Path

import torch

import infer_full_sa_no_object_lora as full_sa
import infer_xssc_object_self_attn_lora as common

_ORIGINAL_BUILD_RUNTIME_MODEL = common._build_runtime_model
RANKING_CSV = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "three_model_combined_summary.csv"
)
QUERY_CHUNK = 128
RANKING_STEPS = 40


class AdaptiveAttentionProbabilityNoise:
    def __init__(self, pipe, group: str, alpha: float, seed: int) -> None:
        self.pipe = pipe
        self.group = group
        self.alpha = float(alpha)
        self.seed = int(seed)
        if group.startswith("top"):
            self.descending = True
            count_text = group.removeprefix("top")
        elif group.startswith("bottom"):
            self.descending = False
            count_text = group.removeprefix("bottom")
        else:
            raise ValueError(f"Invalid ATTENTION_NOISE_GROUP: {group}")
        self.extreme_count = int(count_text)
        if self.extreme_count not in {30, 100}:
            raise ValueError("Group must be top30, bottom30, top100, or bottom100")
        self.heads_by_step_block = self._load_heads()
        self.current_ranking_step = -1
        self.last_runtime_step = -1
        self.capture_root = None
        capture_root = os.environ.get("ATTENTION_CAPTURE_ROOT", "").strip()
        if capture_root:
            self.capture_root = Path(capture_root)
            self.capture_root.mkdir(parents=True, exist_ok=True)
        self.capture_cases = self._load_capture_cases()
        self.capture_case_index = -1
        self.current_case = "case"
        self.capture_entry = None
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
        atexit.register(self.flush_capture)

    @staticmethod
    def _load_capture_cases():
        input_list = os.environ.get("ATTENTION_CAPTURE_TEST_LIST", "").strip()
        if not input_list:
            return []
        cases = []
        for line in Path(input_list).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                case = Path(line).stem
                if case not in cases:
                    cases.append(case)
        return cases

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
        for step, rows in rows_by_step.items():
            if len(rows) != 720:
                raise RuntimeError(f"Ranking step {step} has {len(rows)} rows, expected 720")
            rows.sort(
                key=lambda item: (
                    -item[0] if self.descending else item[0],
                    item[1],
                    item[2],
                )
            )
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
            if runtime_step == 0 and self.last_runtime_step not in {-1, 0}:
                self.flush_capture()
                self.capture_case_index += 1
            elif runtime_step == 0 and self.capture_case_index < 0:
                self.capture_case_index = 0
            if self.capture_cases and self.capture_case_index < len(self.capture_cases):
                self.current_case = self.capture_cases[self.capture_case_index]
            denominator = max(1, len(scheduler_steps) - 1)
            self.current_ranking_step = int(
                round(runtime_step * (RANKING_STEPS - 1) / denominator)
            )
            self.last_runtime_step = runtime_step
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
        capture_enabled = self.capture_root is not None and self.current_ranking_step == 39
        if capture_enabled and self.capture_entry is None:
            self.capture_entry = {
                "before": torch.zeros((sequence, sequence), dtype=torch.float32),
                "after": torch.zeros((sequence, sequence), dtype=torch.float32),
                "abs_delta": torch.zeros((sequence, sequence), dtype=torch.float32),
                "head_instances": 0,
                "selected": set(),
                "clipped_elements": 0,
                "total_elements": 0,
                "max_row_sum_error": 0.0,
            }
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
            if capture_enabled:
                entry = self.capture_entry
                entry["before"][start:end] += attention.sum(dim=(0, 1)).detach().cpu()
                entry["after"][start:end] += perturbed.sum(dim=(0, 1)).detach().cpu()
                entry["abs_delta"][start:end] += (
                    (perturbed - attention).abs().sum(dim=(0, 1)).detach().cpu()
                )
                entry["clipped_elements"] += int(
                    ((attention + (self.alpha / sequence) * noise) < 0).sum().item()
                )
                entry["total_elements"] += noise.numel()
                entry["max_row_sum_error"] = max(
                    entry["max_row_sum_error"],
                    float((perturbed.sum(dim=-1) - 1.0).abs().max().item()),
                )
            selected_output[:, :, start:end] = torch.matmul(
                perturbed.to(selected_v.dtype), selected_v
            )
        if capture_enabled:
            self.capture_entry["head_instances"] += batch * len(heads)
            self.capture_entry["selected"].update((block, head) for head in heads)
        fused = original(q, k, v)
        fused_heads = fused.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3).clone()
        fused_heads[:, heads] = selected_output
        return fused_heads.permute(0, 2, 1, 3).reshape(batch, sequence, -1)

    def flush_capture(self) -> None:
        if self.capture_root is None or self.capture_entry is None:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        entry = self.capture_entry
        self.capture_entry = None
        count = max(1, int(entry["head_instances"]))
        before = entry["before"] / count
        after = entry["after"] / count
        mean_abs_delta = entry["abs_delta"] / count
        size = min(512, before.shape[0])
        pool = torch.nn.functional.adaptive_avg_pool2d
        before_small = pool(before[None, None], (size, size))[0, 0].numpy()
        after_small = pool(after[None, None], (size, size))[0, 0].numpy()
        delta_small = pool(mean_abs_delta[None, None], (size, size))[0, 0].numpy()
        log_before = np.log10(before_small + 1e-9)
        log_after = np.log10(after_small + 1e-9)
        values = np.concatenate((log_before.ravel(), log_after.ravel()))
        vmin, vmax = np.percentile(values, [1.0, 99.5])
        delta_max = float(np.percentile(delta_small, 99.5)) or 1e-9
        prefix = f"full_sa__{self.current_case}__{self.group}__step39"
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
        panels = (
            (log_before, "Before: log10 attention", "magma", vmin, vmax),
            (log_after, "After: log10 attention", "magma", vmin, vmax),
            (delta_small, "Mean per-head |After - Before|", "viridis", 0.0, delta_max),
        )
        for axis, (matrix, title, cmap, low, high) in zip(axes, panels):
            image = axis.imshow(matrix, cmap=cmap, vmin=low, vmax=high, aspect="auto")
            axis.set_title(title)
            axis.set_xlabel("Key token index")
            axis.set_ylabel("Query token index")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        fig.suptitle(f"Full-SA | {self.group.upper()} | alpha={self.alpha:.1f} | S039")
        all_token_image = f"{prefix}__all_token.png"
        fig.savefig(self.capture_root / all_token_image, dpi=160)
        plt.close(fig)

        frame_image = ""
        if before.shape[0] % 7 == 0:
            spatial = before.shape[0] // 7
            frame_before = before.reshape(7, spatial, 7, spatial).sum(-1).mean(1).numpy()
            frame_after = after.reshape(7, spatial, 7, spatial).sum(-1).mean(1).numpy()
            frame_delta = frame_after - frame_before
            limit = float(np.max(np.abs(frame_delta))) or 1e-9
            fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
            for axis, matrix, title, cmap, low, high in (
                (axes[0], frame_before, "Before", "magma", 0.0, 1.0),
                (axes[1], frame_after, "After", "magma", 0.0, 1.0),
                (axes[2], frame_delta, "After - Before", "coolwarm", -limit, limit),
            ):
                image = axis.imshow(matrix, cmap=cmap, vmin=low, vmax=high)
                axis.set_title(title)
                axis.set_xlabel("Key latent frame")
                axis.set_ylabel("Query latent frame")
                fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            frame_image = f"{prefix}__frame.png"
            fig.savefig(self.capture_root / frame_image, dpi=180)
            plt.close(fig)
        metadata = {
            "model": "Full-SA no-object step-002500",
            "case": self.current_case,
            "group": self.group,
            "step": 39,
            "alpha": self.alpha,
            "unique_block_heads": len(entry["selected"]),
            "selected_block_heads": [
                {"block": block, "head": head}
                for block, head in sorted(entry["selected"])
            ],
            "mean_abs_attention_delta": float(mean_abs_delta.mean()),
            "clipped_fraction": (
                entry["clipped_elements"] / entry["total_elements"]
                if entry["total_elements"]
                else 0.0
            ),
            "max_row_sum_error": entry["max_row_sum_error"],
            "all_token_image": all_token_image,
            "frame_image": frame_image,
        }
        (self.capture_root / f"{prefix}.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        print(f"[attention-capture] wrote {self.current_case} {self.group}", flush=True)


def _build_runtime_model(args):
    model, model_args, runtime_info = _ORIGINAL_BUILD_RUNTIME_MODEL(args)
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
    return model, model_args, runtime_info


def _install_runtime_hooks() -> None:
    common._load_resolved_config = full_sa._load_resolved_config
    common._build_runtime_model = _build_runtime_model
    common._install_runtime_hooks()


def main() -> None:
    common.batch_base._install_kubric_runtime_hooks = _install_runtime_hooks
    common.batch_base.main()


if __name__ == "__main__":
    main()
