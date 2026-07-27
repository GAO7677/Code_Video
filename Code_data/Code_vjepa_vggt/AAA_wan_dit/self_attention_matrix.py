#!/usr/bin/env python3
"""Exact all-token self-attention pooling and heatmap rendering for Wan DiT."""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass(frozen=True)
class MatrixCaptureConfig:
    block_id: int = 17
    step_numbers: tuple[int, ...] = (5, 15, 25, 35)
    output_bins: int = 512
    query_chunk: int = 128

    def validate(self) -> None:
        if self.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if not self.step_numbers or any(step <= 0 for step in self.step_numbers):
            raise ValueError("step_numbers must contain positive, one-based steps")
        if len(set(self.step_numbers)) != len(self.step_numbers):
            raise ValueError("step_numbers must be unique")
        if self.output_bins <= 0 or self.query_chunk <= 0:
            raise ValueError("output_bins and query_chunk must be positive")


def parse_step_numbers(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("attention step list is empty")
    return values


def _as_heads(
    tensor: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    if tensor.ndim == 4:
        if tensor.shape[0] != 1 or tensor.shape[2] != num_heads:
            raise ValueError(
                f"expected [1, tokens, {num_heads}, head_dim], got {list(tensor.shape)}"
            )
        return tensor[0].permute(1, 0, 2).contiguous()
    if tensor.ndim == 3:
        if tensor.shape[0] != 1 or tensor.shape[-1] % num_heads != 0:
            raise ValueError(
                f"expected [1, tokens, heads*head_dim], got {list(tensor.shape)}"
            )
        tokens = int(tensor.shape[1])
        head_dim = int(tensor.shape[-1]) // int(num_heads)
        return (
            tensor[0]
            .view(tokens, int(num_heads), head_dim)
            .permute(1, 0, 2)
            .contiguous()
        )
    raise ValueError(f"unsupported Q/K rank: {tensor.ndim}")


@torch.no_grad()
def _pool_full_attention_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    output_bins: int,
    query_chunk: int,
    temporal_grid: tuple[int, int, int] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, np.ndarray] | None,
]:
    """Compute exact softmax over all keys, then pool every query/key token.

    The returned block_mean is the mean attention value for every query-bin and
    key-bin pair. key_mass multiplies block_mean by the number of keys in each
    key bin, so each row approximately sums to one and is easier to visualize.
    """

    q_heads = _as_heads(q.detach(), num_heads=num_heads)
    k_heads = _as_heads(k.detach(), num_heads=num_heads)
    if q_heads.shape != k_heads.shape:
        raise ValueError(
            f"self-attention Q/K shapes differ: {list(q_heads.shape)} vs "
            f"{list(k_heads.shape)}"
        )

    heads, token_count, head_dim = (int(value) for value in q_heads.shape)
    bins = min(int(output_bins), token_count)
    device = q_heads.device
    token_ids = torch.arange(token_count, device=device, dtype=torch.long)
    token_bins = torch.div(token_ids * bins, token_count, rounding_mode="floor")
    token_bin_counts = torch.bincount(token_bins, minlength=bins).float()
    pooled_sum = torch.zeros(
        (heads, bins, bins), device=device, dtype=torch.float32
    )
    scale = 1.0 / math.sqrt(float(head_dim))

    temporal_sum = None
    exact_self_sum = None
    same_frame_wins = None
    key_time_ids = None
    tokens_per_frame = None
    temporal_frames = None
    if temporal_grid is not None:
        temporal_frames, grid_h, grid_w = (
            int(value) for value in temporal_grid
        )
        tokens_per_frame = grid_h * grid_w
        if temporal_frames * tokens_per_frame != token_count:
            raise ValueError(
                f"temporal grid {temporal_grid} has "
                f"{temporal_frames * tokens_per_frame} tokens, "
                f"expected {token_count}"
            )
        key_time_ids = torch.div(
            token_ids, tokens_per_frame, rounding_mode="floor"
        )
        temporal_sum = torch.zeros(
            (heads, temporal_frames, temporal_frames),
            device=device,
            dtype=torch.float32,
        )
        exact_self_sum = torch.zeros(
            (heads, temporal_frames), device=device, dtype=torch.float32
        )
        same_frame_wins = torch.zeros(
            (heads, temporal_frames), device=device, dtype=torch.float32
        )

    k_t = k_heads.transpose(-1, -2)
    for start in range(0, token_count, int(query_chunk)):
        stop = min(start + int(query_chunk), token_count)
        query = q_heads[:, start:stop]
        scores = torch.matmul(query, k_t) * scale
        probabilities = torch.softmax(scores.float(), dim=-1)

        chunk = stop - start
        key_pooled = torch.zeros(
            (heads, chunk, bins), device=device, dtype=torch.float32
        )
        key_index = token_bins.view(1, 1, token_count).expand(
            heads, chunk, token_count
        )
        key_pooled.scatter_add_(2, key_index, probabilities)
        pooled_sum.index_add_(1, token_bins[start:stop], key_pooled)

        if temporal_sum is not None:
            assert key_time_ids is not None
            assert tokens_per_frame is not None
            assert temporal_frames is not None
            assert exact_self_sum is not None
            assert same_frame_wins is not None
            query_ids = token_ids[start:stop]
            query_times = torch.div(
                query_ids, tokens_per_frame, rounding_mode="floor"
            )
            key_temporal = torch.zeros(
                (heads, chunk, temporal_frames),
                device=device,
                dtype=torch.float32,
            )
            temporal_index = key_time_ids.view(
                1, 1, token_count
            ).expand(heads, chunk, token_count)
            key_temporal.scatter_add_(2, temporal_index, probabilities)
            exact_self = probabilities.gather(
                2,
                query_ids.view(1, chunk, 1).expand(heads, chunk, 1),
            ).squeeze(2)
            chunk_ids = torch.arange(chunk, device=device)
            key_temporal[
                :, chunk_ids, query_times
            ] -= exact_self
            key_temporal /= (1.0 - exact_self).clamp_min(1.0e-12).unsqueeze(2)
            same_mass = key_temporal[
                :, chunk_ids, query_times
            ]
            other_temporal = key_temporal.clone()
            other_temporal[
                :, chunk_ids, query_times
            ] = -1.0
            wins = same_mass > other_temporal.max(dim=2).values
            temporal_sum.index_add_(1, query_times, key_temporal)
            exact_self_sum.index_add_(1, query_times, exact_self)
            same_frame_wins.index_add_(
                1, query_times, wins.to(torch.float32)
            )
            del (
                query_ids,
                query_times,
                key_temporal,
                temporal_index,
                exact_self,
                same_mass,
                other_temporal,
                wins,
                chunk_ids,
            )

        del scores, probabilities, key_pooled, key_index, query

    denominator = (
        token_bin_counts.view(1, bins, 1)
        * token_bin_counts.view(1, 1, bins)
    ).clamp_min_(1.0)
    block_mean = pooled_sum / denominator
    key_mass = block_mean * token_bin_counts.view(1, 1, bins)

    metadata = {
        "num_heads": heads,
        "token_count": token_count,
        "head_dim": head_dim,
        "output_bins": bins,
        "query_chunk": int(query_chunk),
        "query_bin_counts": token_bin_counts.cpu().to(torch.int64).tolist(),
        "key_bin_counts": token_bin_counts.cpu().to(torch.int64).tolist(),
        "softmax_axis": "all_key_tokens",
        "query_sampling": "none",
        "pooling": "contiguous_token_block_mean",
    }
    temporal_statistics = None
    if temporal_sum is not None:
        assert exact_self_sum is not None
        assert same_frame_wins is not None
        assert tokens_per_frame is not None
        denominator = float(tokens_per_frame)
        temporal_statistics = {
            "time_matrix_no_exact_self": (
                temporal_sum / denominator
            ).cpu().numpy().astype(np.float32),
            "exact_self_mass": (
                exact_self_sum / denominator
            ).cpu().numpy().astype(np.float32),
            "same_frame_win_rate": (
                same_frame_wins / denominator
            ).cpu().numpy().astype(np.float32),
        }

    return (
        block_mean.cpu().numpy().astype(np.float32),
        key_mass.cpu().numpy().astype(np.float32),
        metadata,
        temporal_statistics,
    )


def pool_full_attention_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    output_bins: int,
    query_chunk: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    block_mean, key_mass, metadata, _ = _pool_full_attention_matrix(
        q,
        k,
        num_heads=num_heads,
        output_bins=output_bins,
        query_chunk=query_chunk,
        temporal_grid=None,
    )
    return block_mean, key_mass, metadata


def pool_full_attention_matrix_with_temporal(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    output_bins: int,
    query_chunk: int,
    temporal_grid: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    block_mean, key_mass, metadata, temporal = _pool_full_attention_matrix(
        q,
        k,
        num_heads=num_heads,
        output_bins=output_bins,
        query_chunk=query_chunk,
        temporal_grid=temporal_grid,
    )
    if temporal is None:
        raise RuntimeError("temporal attention statistics were not produced")
    return block_mean, key_mass, metadata, temporal


def _frame_boundaries(
    *,
    grid: tuple[int, int, int],
    token_count: int,
    bins: int,
) -> list[float]:
    frames, grid_h, grid_w = grid
    if frames * grid_h * grid_w != token_count:
        raise ValueError(
            f"grid {grid} has {frames * grid_h * grid_w} tokens, "
            f"expected {token_count}"
        )
    tokens_per_frame = grid_h * grid_w
    return [
        (frame * tokens_per_frame * bins / token_count) - 0.5
        for frame in range(1, frames)
    ]


def _render_head(
    *,
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    boundaries: Iterable[float],
) -> None:
    positive = matrix[matrix > 0]
    epsilon = float(positive.min()) * 0.5 if positive.size else 1.0e-12
    display = np.log10(np.maximum(matrix, epsilon))
    finite = display[np.isfinite(display)]
    low, high = np.percentile(finite, [1.0, 99.8]).tolist()
    if high <= low:
        high = low + 1.0

    figure, axis = plt.subplots(figsize=(7.2, 6.5), dpi=150)
    image = axis.imshow(
        display,
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
        vmin=float(low),
        vmax=float(high),
        origin="upper",
    )
    for boundary in boundaries:
        axis.axhline(boundary, color="white", linewidth=0.35, alpha=0.55)
        axis.axvline(boundary, color="white", linewidth=0.35, alpha=0.55)
    axis.set_title(title)
    axis.set_xlabel("Key-token bin")
    axis.set_ylabel("Query-token bin")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("log10(mean attention mass)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def _render_contact_sheet(
    *,
    matrices: np.ndarray,
    output_path: Path,
    title: str,
    boundaries: Iterable[float],
) -> None:
    heads = int(matrices.shape[0])
    columns = min(6, heads)
    rows = int(math.ceil(heads / columns))
    positive = matrices[matrices > 0]
    epsilon = float(positive.min()) * 0.5 if positive.size else 1.0e-12
    display = np.log10(np.maximum(matrices, epsilon))
    finite = display[np.isfinite(display)]
    low, high = np.percentile(finite, [1.0, 99.8]).tolist()
    if high <= low:
        high = low + 1.0

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.25 * rows),
        dpi=120,
        squeeze=False,
    )
    last_image = None
    for head in range(rows * columns):
        axis = axes.flat[head]
        if head >= heads:
            axis.axis("off")
            continue
        last_image = axis.imshow(
            display[head],
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
            vmin=float(low),
            vmax=float(high),
            origin="upper",
        )
        for boundary in boundaries:
            axis.axhline(boundary, color="white", linewidth=0.25, alpha=0.45)
            axis.axvline(boundary, color="white", linewidth=0.25, alpha=0.45)
        axis.set_title(f"head {head:02d}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    if last_image is not None:
        colorbar_axis = figure.add_axes((0.965, 0.12, 0.012, 0.76))
        figure.colorbar(
            last_image,
            cax=colorbar_axis,
            label="log10(mean attention mass)",
        )
    figure.subplots_adjust(
        left=0.02, right=0.95, bottom=0.02, top=0.94, wspace=0.08, hspace=0.14
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


class FullTokenSelfAttentionRecorder:
    def __init__(
        self,
        *,
        config: MatrixCaptureConfig,
        model_label: str,
        output_root: Path,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.model_label = str(model_label)
        self.output_root = output_root.expanduser().resolve()
        self.grid = grid
        self.active = False
        self.current_step: int | None = None
        self.case_key: str | None = None
        self.case_metadata: dict[str, Any] = {}
        self.captures: dict[int, dict[str, Any]] = {}

    def begin_case(
        self,
        case_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.active = False
        self.current_step = None
        self.case_key = str(case_key)
        self.case_metadata = dict(metadata or {})
        self.captures = {}

    def set_grid(self, grid: tuple[int, int, int]) -> None:
        candidate = tuple(int(value) for value in grid)
        if self.grid is not None and self.grid != candidate:
            raise RuntimeError(f"attention grid changed from {self.grid} to {candidate}")
        self.grid = candidate

    @torch.no_grad()
    def capture(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        num_heads: int,
    ) -> None:
        step = self.current_step
        if not self.active or step is None or step not in self.config.step_numbers:
            return
        if step in self.captures:
            raise RuntimeError(
                f"captured Block {self.config.block_id} step {step} more than once"
            )
        if self.grid is None:
            raise RuntimeError("latent token grid is not configured")

        print(
            f"[self-attn-matrix] model={self.model_label} case={self.case_key} "
            f"block={self.config.block_id} step={step} q={list(q.shape)} "
            f"k={list(k.shape)} heads={num_heads}",
            flush=True,
        )
        block_mean, key_mass, metadata = pool_full_attention_matrix(
            q,
            k,
            num_heads=int(num_heads),
            output_bins=int(self.config.output_bins),
            query_chunk=int(self.config.query_chunk),
        )
        expected_tokens = math.prod(self.grid)
        if int(metadata["token_count"]) != expected_tokens:
            raise RuntimeError(
                f"captured {metadata['token_count']} tokens but grid {self.grid} "
                f"contains {expected_tokens}"
            )
        self.captures[int(step)] = {
            "block_mean": block_mean,
            "key_mass": key_mass,
            "metadata": metadata,
        }

    def finalize_case(self) -> Path:
        if self.case_key is None:
            raise RuntimeError("begin_case must be called before finalize_case")
        missing = sorted(set(self.config.step_numbers) - set(self.captures))
        if missing:
            raise RuntimeError(
                f"missing Block {self.config.block_id} captures for steps {missing}"
            )
        if self.grid is None:
            raise RuntimeError("latent grid is missing")

        case_dir = self.output_root / self.model_label / self.case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        step_entries: list[dict[str, Any]] = []
        for step in self.config.step_numbers:
            capture = self.captures[int(step)]
            block_mean = capture["block_mean"]
            key_mass = capture["key_mass"]
            metadata = capture["metadata"]
            step_dir = case_dir / f"step_{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            npz_path = step_dir / "block17_all_heads_token_matrix.npz"
            np.savez_compressed(
                npz_path,
                block_mean=block_mean.astype(np.float32),
                key_mass=key_mass.astype(np.float32),
            )
            boundaries = _frame_boundaries(
                grid=self.grid,
                token_count=int(metadata["token_count"]),
                bins=int(metadata["output_bins"]),
            )
            head_images: list[str] = []
            for head in range(int(metadata["num_heads"])):
                image_name = f"head_{head:02d}_token_attention_matrix.png"
                _render_head(
                    matrix=key_mass[head],
                    output_path=step_dir / image_name,
                    title=(
                        f"{self.model_label} | Block {self.config.block_id} | "
                        f"denoise step {step} | head {head:02d}"
                    ),
                    boundaries=boundaries,
                )
                head_images.append(image_name)
            contact_name = "all_heads_token_attention_matrix.png"
            _render_contact_sheet(
                matrices=key_mass,
                output_path=step_dir / contact_name,
                title=(
                    f"{self.model_label} | Block {self.config.block_id} | "
                    f"denoise step {step}"
                ),
                boundaries=boundaries,
            )
            step_entries.append(
                {
                    "step_number_one_based": int(step),
                    "step_index_zero_based": int(step - 1),
                    "directory": step_dir.name,
                    "matrix_npz": npz_path.name,
                    "contact_sheet": contact_name,
                    "head_images": head_images,
                    "matrix_metadata": metadata,
                }
            )

        summary = {
            "model": self.model_label,
            "case": self.case_key,
            "block_id": int(self.config.block_id),
            "step_numbers_one_based": list(self.config.step_numbers),
            "latent_grid": list(self.grid),
            "token_order": "time-major, then row-major height and width",
            "query_sampling": "none; all query and key tokens contribute",
            "softmax": "exact over all key tokens for each query and head",
            "display_matrix": (
                "512x512 contiguous-token pooling by default; PNG uses log10 "
                "of mean attention mass, while NPZ stores linear values"
            ),
            "case_metadata": self.case_metadata,
            "steps": step_entries,
        }
        summary_path = case_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_case_html(case_dir, summary)
        print(f"[self-attn-matrix] wrote {summary_path}", flush=True)
        return summary_path


def _write_case_html(case_dir: Path, summary: dict[str, Any]) -> None:
    sections: list[str] = []
    for step in summary["steps"]:
        figures = "".join(
            "<figure>"
            f"<a href='{html.escape(step['directory'] + '/' + image)}'>"
            f"<img loading='lazy' src='{html.escape(step['directory'] + '/' + image)}'></a>"
            f"<figcaption>{html.escape(Path(image).stem)}</figcaption>"
            "</figure>"
            for image in step["head_images"]
        )
        contact = html.escape(step["directory"] + "/" + step["contact_sheet"])
        sections.append(
            "<section>"
            f"<h2>Denoise step {int(step['step_number_one_based'])}</h2>"
            f"<a href='{contact}'><img class='contact' loading='lazy' src='{contact}'></a>"
            f"<div class='grid'>{figures}</div>"
            "</section>"
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(summary['model'])} attention</title>
<style>
body{{margin:0;background:#f5f5f3;color:#20231f;font:14px Arial,sans-serif}}
main{{max-width:1900px;margin:auto;padding:20px}}
h1,h2{{letter-spacing:0}} section{{border-top:1px solid #c9ccc7;margin-top:22px;padding-top:14px}}
.contact{{display:block;max-width:100%;background:#fff;border:1px solid #d5d8d2}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:12px;margin-top:14px}}
figure{{margin:0;padding:7px;background:#fff;border:1px solid #d5d8d2;border-radius:4px;min-width:0}}
figure img{{width:100%;height:auto;display:block}} figcaption{{padding-top:5px}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,minmax(220px,1fr))}}}}
@media(max-width:620px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>{html.escape(summary['model'])} · {html.escape(summary['case'])}</h1>
<p>Block {summary['block_id']} full-token self-attention. Grid {html.escape(str(summary['latent_grid']))};
all query/key tokens included; heatmaps show log-scaled 512×512 pooled matrices.</p>
{''.join(sections)}
</main></body></html>"""
    (case_dir / "index.html").write_text(page, encoding="utf-8")


class DiffSynthAttentionScope:
    """Activate a recorder only on the positive CFG branch of selected steps."""

    def __init__(
        self,
        *,
        pipe: Any,
        recorder: FullTokenSelfAttentionRecorder,
        cfg_scale: float,
    ) -> None:
        self.pipe = pipe
        self.recorder = recorder
        self.calls_per_step = 1 if abs(float(cfg_scale) - 1.0) < 1.0e-8 else 2
        self.call_index = 0
        self.original = pipe.model_fn

    def install(self) -> None:
        self.pipe.model_fn = self

    def restore(self) -> None:
        self.pipe.model_fn = self.original

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        step_index = self.call_index // self.calls_per_step
        branch_index = self.call_index % self.calls_per_step
        step_number = int(step_index + 1)
        selected = (
            branch_index == 0 and step_number in self.recorder.config.step_numbers
        )
        self.recorder.active = bool(selected)
        self.recorder.current_step = step_number if selected else None
        if selected:
            latents = kwargs.get("latents")
            dit = kwargs.get("dit")
            if latents is None or dit is None:
                raise RuntimeError("model_fn did not expose latents and dit")
            patch = tuple(int(value) for value in getattr(dit, "patch_size", (1, 2, 2)))
            self.recorder.set_grid(
                (
                    int(latents.shape[2]) // patch[0],
                    int(latents.shape[3]) // patch[1],
                    int(latents.shape[4]) // patch[2],
                )
            )
        try:
            return self.original(*args, **kwargs)
        finally:
            self.recorder.active = False
            self.recorder.current_step = None
            self.call_index += 1


def install_diffsynth_block_recorder(
    dit: Any,
    recorder: FullTokenSelfAttentionRecorder,
) -> Callable[[], None]:
    blocks = getattr(dit, "blocks", None)
    if blocks is None or recorder.config.block_id >= len(blocks):
        raise AttributeError("Wan DiT blocks are unavailable")
    self_attn = getattr(blocks[recorder.config.block_id], "self_attn", None)
    inner = getattr(self_attn, "attn", None)
    if self_attn is None or inner is None:
        raise AttributeError(
            f"Block {recorder.config.block_id} self-attention inner module is unavailable"
        )
    num_heads = int(getattr(self_attn, "num_heads"))
    original = inner.forward

    def wrapped(q, k, v, *args, **kwargs):
        output = original(q, k, v, *args, **kwargs)
        recorder.capture(q=q, k=k, num_heads=num_heads)
        return output

    inner.forward = wrapped

    def restore() -> None:
        inner.forward = original

    return restore


class PhysRVGAttentionProcessorRecorder:
    """Wrap PhysRVG's official Wan attention processor without changing output."""

    def __init__(
        self,
        *,
        recorder: FullTokenSelfAttentionRecorder,
        model_module: Any,
    ) -> None:
        self.recorder = recorder
        self.model_module = model_module
        self.original: Any | None = None
        self.attention: Any | None = None
        self.call_index = 0

    def install(self, transformer: Any) -> None:
        candidates = (
            transformer,
            getattr(getattr(transformer, "base_model", None), "model", None),
            getattr(transformer, "model", None),
            getattr(transformer, "module", None),
        )
        blocks = next(
            (
                candidate.blocks
                for candidate in candidates
                if candidate is not None and getattr(candidate, "blocks", None) is not None
            ),
            None,
        )
        if blocks is None or self.recorder.config.block_id >= len(blocks):
            raise AttributeError("PhysRVG transformer blocks are unavailable")
        self.attention = blocks[self.recorder.config.block_id].attn1
        self.original = self.attention.processor
        self.attention.processor = self

    def restore(self) -> None:
        if self.attention is not None and self.original is not None:
            self.attention.processor = self.original

    def begin_case(self) -> None:
        self.call_index = 0

    @staticmethod
    def _apply_rotary(
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        output = torch.empty_like(hidden_states)
        output[..., 0::2] = x1 * cos - x2 * sin
        output[..., 1::2] = x1 * sin + x2 * cos
        return output.type_as(hidden_states)

    @torch.no_grad()
    def _capture_qk(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        query, key, _ = self.model_module._get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = self._apply_rotary(query, *rotary_emb)
            key = self._apply_rotary(key, *rotary_emb)
        self.recorder.capture(q=query, k=key, num_heads=int(attn.heads))

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.original is None:
            raise RuntimeError("PhysRVG processor recorder is not installed")
        output = self.original(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            **kwargs,
        )
        self.call_index += 1
        step_number = int(self.call_index)
        selected = step_number in self.recorder.config.step_numbers
        self.recorder.active = selected
        self.recorder.current_step = step_number if selected else None
        try:
            if selected:
                self._capture_qk(
                    attn,
                    hidden_states,
                    encoder_hidden_states,
                    rotary_emb,
                )
        finally:
            self.recorder.active = False
            self.recorder.current_step = None
        return output
