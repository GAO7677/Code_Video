#!/usr/bin/env python3
"""Exact selected-query self-attention capture and visualization."""

from __future__ import annotations

import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from self_attention_matrix import MatrixCaptureConfig, _as_heads


def parse_query_coords(text: str) -> tuple[tuple[int, int, int], ...]:
    coords: list[tuple[int, int, int]] = []
    for item in text.split(","):
        parts = tuple(int(value) for value in item.strip().split(":"))
        if len(parts) != 3:
            raise ValueError(
                "query coordinates must use comma-separated time:row:column"
            )
        coords.append(parts)
    if not coords or len(set(coords)) != len(coords):
        raise ValueError("query coordinates must be non-empty and unique")
    return tuple(coords)


def _query_indices(
    coords: tuple[tuple[int, int, int], ...],
    grid: tuple[int, int, int],
) -> tuple[int, ...]:
    frames, height, width = grid
    indices: list[int] = []
    for time, row, column in coords:
        if not (0 <= time < frames and 0 <= row < height and 0 <= column < width):
            raise ValueError(f"query coordinate {(time, row, column)} outside {grid}")
        indices.append(time * height * width + row * width + column)
    return tuple(indices)


@torch.no_grad()
def selected_query_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    query_indices: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    q_heads = _as_heads(q.detach(), num_heads=num_heads)
    k_heads = _as_heads(k.detach(), num_heads=num_heads)
    if q_heads.shape != k_heads.shape:
        raise ValueError(
            f"self-attention Q/K shapes differ: {list(q_heads.shape)} vs "
            f"{list(k_heads.shape)}"
        )
    heads, token_count, head_dim = (int(value) for value in q_heads.shape)
    if not query_indices or min(query_indices) < 0 or max(query_indices) >= token_count:
        raise ValueError("query token index is outside captured Q/K")

    indices = torch.tensor(query_indices, device=q_heads.device, dtype=torch.long)
    query = q_heads.index_select(1, indices)
    scores = torch.matmul(query, k_heads.transpose(-1, -2))
    scores = scores * (1.0 / math.sqrt(float(head_dim)))
    probabilities = torch.softmax(scores.float(), dim=-1).mean(dim=1)
    metadata = {
        "num_heads": heads,
        "token_count": token_count,
        "head_dim": head_dim,
        "query_token_indices": list(query_indices),
        "query_token_count": len(query_indices),
        "softmax_axis": "all_key_tokens",
        "query_reduction": "mean probability over selected query tokens",
    }
    return probabilities.cpu().numpy().astype(np.float32), metadata


def _display_limits(attention: np.ndarray) -> tuple[np.ndarray, float, float]:
    positive = attention[attention > 0]
    epsilon = float(positive.min()) * 0.5 if positive.size else 1.0e-12
    display = np.log10(np.maximum(attention, epsilon))
    finite = display[np.isfinite(display)]
    low, high = np.percentile(finite, [1.0, 99.7]).tolist()
    if high <= low:
        high = low + 1.0
    return display, float(low), float(high)


def _render_head(
    *,
    attention: np.ndarray,
    output_path: Path,
    title: str,
    query_coords: tuple[tuple[int, int, int], ...],
    low: float,
    high: float,
) -> None:
    frames, height, width = attention.shape
    columns = 5
    rows = int(math.ceil(frames / columns))
    display, _, _ = _display_limits(attention)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.05 * columns, 2.35 * rows),
        dpi=135,
        squeeze=False,
    )
    image = None
    for frame in range(rows * columns):
        axis = axes.flat[frame]
        if frame >= frames:
            axis.axis("off")
            continue
        image = axis.imshow(
            display[frame],
            cmap="magma",
            interpolation="nearest",
            vmin=low,
            vmax=high,
            origin="upper",
            aspect="equal",
        )
        for query_time, row, column in query_coords:
            if query_time == frame:
                axis.add_patch(
                    plt.Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#35e68a",
                        linewidth=1.2,
                    )
                )
        axis.set_title(f"key latent t={frame} (video f~{frame * 4})", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title, fontsize=12)
    if image is not None:
        colorbar_axis = figure.add_axes((0.94, 0.12, 0.012, 0.76))
        figure.colorbar(image, cax=colorbar_axis, label="log10 attention probability")
    figure.subplots_adjust(
        left=0.025,
        right=0.92,
        bottom=0.035,
        top=0.91,
        wspace=0.09,
        hspace=0.25,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def _render_contact(
    *,
    attention: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    temporal_sum = attention.sum(axis=1)
    display, low, high = _display_limits(temporal_sum)
    heads = int(display.shape[0])
    columns = 6
    rows = int(math.ceil(heads / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.25 * columns, 2.45 * rows),
        dpi=125,
        squeeze=False,
    )
    image = None
    for head in range(rows * columns):
        axis = axes.flat[head]
        if head >= heads:
            axis.axis("off")
            continue
        image = axis.imshow(
            display[head],
            cmap="magma",
            interpolation="nearest",
            vmin=low,
            vmax=high,
            origin="upper",
            aspect="equal",
        )
        axis.set_title(f"head {head:02d}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    if image is not None:
        colorbar_axis = figure.add_axes((0.965, 0.12, 0.012, 0.76))
        figure.colorbar(
            image,
            cax=colorbar_axis,
            label="log10 attention summed over key time",
        )
    figure.subplots_adjust(
        left=0.02, right=0.95, bottom=0.025, top=0.92, wspace=0.1, hspace=0.2
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


class BallQuerySelfAttentionRecorder:
    def __init__(
        self,
        *,
        config: MatrixCaptureConfig,
        model_label: str,
        output_root: Path,
        query_coords: tuple[tuple[int, int, int], ...],
        query_video_frame: int,
        query_preview: Path | None = None,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.model_label = str(model_label)
        self.output_root = output_root.expanduser().resolve()
        self.query_coords = query_coords
        self.query_video_frame = int(query_video_frame)
        self.query_preview = (
            query_preview.expanduser().resolve() if query_preview is not None else None
        )
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
        _query_indices(self.query_coords, candidate)
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
        indices = _query_indices(self.query_coords, self.grid)
        attention, metadata = selected_query_attention(
            q,
            k,
            num_heads=int(num_heads),
            query_indices=indices,
        )
        expected_tokens = math.prod(self.grid)
        if int(metadata["token_count"]) != expected_tokens:
            raise RuntimeError(
                f"captured {metadata['token_count']} tokens but grid {self.grid} "
                f"contains {expected_tokens}"
            )
        print(
            f"[ball-query-attn] model={self.model_label} case={self.case_key} "
            f"block={self.config.block_id} step={step} queries={list(indices)} "
            f"attention={list(attention.shape)}",
            flush=True,
        )
        self.captures[int(step)] = {
            "attention": attention.reshape(
                int(metadata["num_heads"]), *self.grid
            ),
            "metadata": metadata,
        }

    def finalize_case(self) -> Path:
        if self.case_key is None or self.grid is None:
            raise RuntimeError("begin_case and set_grid are required")
        missing = sorted(set(self.config.step_numbers) - set(self.captures))
        if missing:
            raise RuntimeError(f"missing attention captures for steps {missing}")

        case_dir = self.output_root / self.model_label / self.case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        preview_name = None
        if self.query_preview is not None:
            if not self.query_preview.is_file():
                raise FileNotFoundError(self.query_preview)
            preview_name = "ball_query_patch_preview.png"
            shutil.copy2(self.query_preview, case_dir / preview_name)

        step_entries: list[dict[str, Any]] = []
        for step in self.config.step_numbers:
            capture = self.captures[int(step)]
            attention = capture["attention"]
            metadata = capture["metadata"]
            step_dir = case_dir / f"step_{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            npz_name = (
                f"block{int(self.config.block_id):02d}_ball_query_attention.npz"
            )
            np.savez_compressed(
                step_dir / npz_name,
                attention=attention.astype(np.float32),
                query_token_indices=np.asarray(
                    metadata["query_token_indices"], dtype=np.int64
                ),
                query_coords=np.asarray(self.query_coords, dtype=np.int64),
            )
            _, low, high = _display_limits(attention)
            head_images: list[str] = []
            for head in range(int(metadata["num_heads"])):
                image_name = f"head_{head:02d}_ball_query_attention.png"
                _render_head(
                    attention=attention[head],
                    output_path=step_dir / image_name,
                    title=(
                        f"{self.model_label} | Block {self.config.block_id} | "
                        f"ball query | denoise step {step} | head {head:02d}"
                    ),
                    query_coords=self.query_coords,
                    low=low,
                    high=high,
                )
                head_images.append(image_name)
            contact_name = "all_heads_ball_query_attention.png"
            _render_contact(
                attention=attention,
                output_path=step_dir / contact_name,
                title=(
                    f"{self.model_label} | Block {self.config.block_id} | "
                    f"ball query | denoise step {step} | temporal sum"
                ),
            )
            step_entries.append(
                {
                    "step_number_one_based": int(step),
                    "directory": step_dir.name,
                    "matrix_npz": npz_name,
                    "contact_sheet": contact_name,
                    "head_images": head_images,
                    "attention_shape": list(attention.shape),
                    "metadata": metadata,
                }
            )

        summary = {
            "model": self.model_label,
            "case": self.case_key,
            "block_id": int(self.config.block_id),
            "step_numbers_one_based": list(self.config.step_numbers),
            "latent_grid": list(self.grid),
            "query_video_frame": self.query_video_frame,
            "query_latent_coords": [list(coord) for coord in self.query_coords],
            "query_token_indices": list(_query_indices(self.query_coords, self.grid)),
            "query_reduction": "mean attention probability over four ball patches",
            "key_layout": "13 latent frames, each with a 16x28 spatial grid",
            "query_preview": preview_name,
            "case_metadata": self.case_metadata,
            "steps": step_entries,
        }
        summary_path = case_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_case_html(case_dir, summary)
        print(f"[ball-query-attn] wrote {summary_path}", flush=True)
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
    preview = ""
    if summary.get("query_preview"):
        preview = (
            "<figure class='query'><img src='"
            f"{html.escape(str(summary['query_preview']))}'>"
            "<figcaption>Green cells are the four ball query patches.</figcaption>"
            "</figure>"
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(summary['model'])} ball query</title>
<style>
body{{margin:0;background:#f1f3f2;color:#20231f;font:14px Arial,sans-serif;letter-spacing:0}}
main{{max-width:2100px;margin:auto;padding:20px}} h1,h2{{letter-spacing:0}}
section{{border-top:2px solid #356b50;margin-top:24px;padding-top:14px}}
.query{{max-width:900px;margin:14px 0;padding:8px;background:#fff;border:1px solid #d1d7d3;border-radius:6px}}
.query img,.contact{{display:block;max-width:100%;background:#fff}}
.contact{{border:1px solid #d1d7d3}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(300px,1fr));gap:10px;margin-top:14px}}
figure{{margin:0;padding:7px;background:#fff;border:1px solid #d1d7d3;border-radius:6px;min-width:0}}
figure img{{width:100%;height:auto;display:block}} figcaption{{padding-top:5px;color:#5e6963}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,minmax(240px,1fr))}}}}
@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>{html.escape(summary['model'])} | {html.escape(summary['case'])}</h1>
<p>Block {summary['block_id']}; video frame {summary['query_video_frame']} maps to
latent t={summary['query_latent_coords'][0][0]}. Attention is exact over all
5824 key tokens. Four ball patches are averaged as the query.</p>
{preview}{''.join(sections)}
</main></body></html>"""
    (case_dir / "index.html").write_text(page, encoding="utf-8")
