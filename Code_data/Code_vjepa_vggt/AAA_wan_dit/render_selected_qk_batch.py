#!/usr/bin/env python3
"""Render saved all-token raw-QK and softmax matrices for selected batch heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _render(
    *,
    model: str,
    case: str,
    role: str,
    block: int,
    head: int,
    path: Path,
    output: Path,
) -> None:
    with np.load(path, allow_pickle=False) as data:
        selected = data["selected_heads"].astype(int).tolist()
        if head not in selected:
            raise RuntimeError(f"{path} does not contain selected head {head}")
        head_index = selected.index(head)
        steps = data["steps_one_based"].astype(int)
        raw = data["raw_qk_mean"][:, head_index].astype(np.float32)
        attention = data["softmax_attention_mass"][:, head_index].astype(np.float32)
    log_attention = np.log10(np.maximum(attention, 1.0e-8))
    raw_limit = max(float(np.percentile(np.abs(raw), 99.5)), 1.0e-6)
    attention_min = float(np.percentile(log_attention, 2.0))
    attention_max = max(
        float(np.percentile(log_attention, 99.5)), attention_min + 1.0e-6
    )
    fig, axes = plt.subplots(
        len(steps), 2, figsize=(11.0, 2.9 * len(steps)), constrained_layout=True
    )
    bins = int(raw.shape[-1])
    boundaries = [time * bins / 13.0 - 0.5 for time in range(1, 13)]
    for row, step in enumerate(steps):
        raw_image = axes[row, 0].imshow(
            raw[row],
            cmap="coolwarm",
            vmin=-raw_limit,
            vmax=raw_limit,
            interpolation="nearest",
            aspect="equal",
        )
        attention_image = axes[row, 1].imshow(
            log_attention[row],
            cmap="magma",
            vmin=attention_min,
            vmax=attention_max,
            interpolation="nearest",
            aspect="equal",
        )
        axes[row, 0].set_title(f"step {step}: raw QK / sqrt(d)")
        axes[row, 1].set_title(f"step {step}: log10 softmax attention mass")
        for axis in axes[row]:
            for boundary in boundaries:
                axis.axhline(boundary, color="white", linewidth=0.18, alpha=0.45)
                axis.axvline(boundary, color="white", linewidth=0.18, alpha=0.45)
            axis.set_xlabel("pooled key-token bin")
            axis.set_ylabel("pooled query-token bin")
        fig.colorbar(raw_image, ax=axes[row, 0], fraction=0.046)
        fig.colorbar(attention_image, ax=axes[row, 1], fraction=0.046)
    fig.suptitle(
        f"{model} | {role} | block {block:02d}, head {head:02d}\n"
        f"{case}\n"
        "all 5824 query/key tokens pooled to 512x512",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selection = json.loads(
        args.selection.expanduser().resolve().read_text(encoding="utf-8")
    )["samples"]
    capture_root = args.capture_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rendered = []
    for model, cases in selection.items():
        for case, item in cases.items():
            by_pair: dict[tuple[int, int], list[str]] = {}
            for role, pair in item["roles"].items():
                key = (int(pair["block"]), int(pair["head"]))
                by_pair.setdefault(key, []).append(role)
            for (block, head), roles in by_pair.items():
                path = (
                    capture_root
                    / model
                    / f"block{block:02d}"
                    / "matrices"
                    / model
                    / case
                    / f"block{block:02d}_selected_qk.npz"
                )
                role_text = "_".join(sorted(roles))
                output = (
                    output_dir
                    / model
                    / case
                    / f"{role_text}_block{block:02d}_head{head:02d}.png"
                )
                _render(
                    model=model,
                    case=case,
                    role=role_text,
                    block=block,
                    head=head,
                    path=path,
                    output=output,
                )
                rendered.append(str(output))
    (output_dir / "manifest.json").write_text(
        json.dumps({"count": len(rendered), "figures": rendered}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"rendered {len(rendered)} selected all-token QK figures")


if __name__ == "__main__":
    main()
