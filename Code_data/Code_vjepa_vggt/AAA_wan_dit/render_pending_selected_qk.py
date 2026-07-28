#!/usr/bin/env python3
"""Render softmax-only Q@K figures for one pending model-seed replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROLES = ("S", "T", "P", "C", "G")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def _load_head(
    capture_root: Path,
    *,
    model: str,
    case: str,
    block: int,
    head: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = (
        capture_root
        / model
        / f"block{block:02d}"
        / "matrices"
        / model
        / case
        / f"block{block:02d}_selected_qk.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        selected = data["selected_heads"].astype(int).tolist()
        if head not in selected:
            raise RuntimeError(f"{path}: selected heads are {selected}, missing {head}")
        index = selected.index(head)
        steps = data["steps_one_based"].astype(int)
        attention = data["softmax_attention_mass"][:, index].astype(np.float32)
        if "temporal_matrix" not in data:
            raise RuntimeError(f"{path}: missing exact 13x13 temporal_matrix")
        temporal = data["temporal_matrix"][:, index].astype(np.float32)
    return steps, attention, temporal


def _limits(attention: np.ndarray) -> tuple[float, float]:
    values = np.log10(np.maximum(attention, 1.0e-8))
    lower = float(np.percentile(values, 2.0))
    upper = max(float(np.percentile(values, 99.5)), lower + 1.0e-6)
    return lower, upper


def _decorate(axis: plt.Axes, bins: int) -> None:
    for time in range(1, 13):
        boundary = time * bins / 13.0 - 0.5
        axis.axhline(boundary, color="white", linewidth=0.18, alpha=0.45)
        axis.axvline(boundary, color="white", linewidth=0.18, alpha=0.45)
    axis.set_xticks([])
    axis.set_yticks([])


def _render_head(
    *,
    model: str,
    case: str,
    roles: list[str],
    block: int,
    head: int,
    steps: np.ndarray,
    attention: np.ndarray,
    output: Path,
) -> None:
    values = np.log10(np.maximum(attention, 1.0e-8))
    lower, upper = _limits(attention)
    fig, axes = plt.subplots(
        len(steps), 1, figsize=(6.0, 5.4 * len(steps)), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    image = None
    for row, step in enumerate(steps):
        image = axes[row].imshow(
            values[row],
            cmap="magma",
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
            aspect="equal",
        )
        axes[row].set_title(f"step {step}: softmax attention mass (log10 color)")
        axes[row].set_xlabel("pooled key-token bin")
        axes[row].set_ylabel("pooled query-token bin")
        _decorate(axes[row], int(values.shape[-1]))
    assert image is not None
    fig.colorbar(image, ax=list(axes), fraction=0.025)
    fig.suptitle(
        f"{model} | {case} | {'/'.join(roles)} | "
        f"block {block:02d}, head {head:02d} | 512x512 softmax Q@K",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=145)
    plt.close(fig)


def _render_contact(
    *,
    model: str,
    case: str,
    role_data: dict[str, dict],
    output: Path,
) -> None:
    steps = next(iter(role_data.values()))["steps"]
    fig, axes = plt.subplots(
        len(steps),
        len(ROLES),
        figsize=(17.5, 13.5),
        constrained_layout=True,
    )
    for column, role in enumerate(ROLES):
        item = role_data[role]
        attention = item["attention"]
        values = np.log10(np.maximum(attention, 1.0e-8))
        lower, upper = _limits(attention)
        image = None
        for row, step in enumerate(steps):
            axis = axes[row, column]
            image = axis.imshow(
                values[row],
                cmap="magma",
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
                aspect="equal",
            )
            _decorate(axis, int(values.shape[-1]))
            if row == 0:
                axis.set_title(
                    f"{role}: B{item['block']:02d} H{item['head']:02d}",
                    fontsize=10,
                )
            if column == 0:
                axis.set_ylabel(f"step {int(step)}", fontsize=9)
        assert image is not None
        fig.colorbar(image, ax=list(axes[:, column]), fraction=0.020)
    fig.suptitle(
        f"{model} | {case} | S/T/P/C/G representative heads | "
        "512x512 softmax Q@K (log10 color)",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def _render_temporal_contact(
    *,
    model: str,
    case: str,
    role_data: dict[str, dict],
    output: Path,
) -> None:
    steps = next(iter(role_data.values()))["steps"]
    fig, axes = plt.subplots(
        len(steps),
        len(ROLES),
        figsize=(15.5, 11.5),
        constrained_layout=True,
    )
    for column, role in enumerate(ROLES):
        item = role_data[role]
        temporal = item["temporal"]
        upper = max(float(np.percentile(temporal, 99.5)), 1.0e-6)
        image = None
        for row, step in enumerate(steps):
            axis = axes[row, column]
            image = axis.imshow(
                temporal[row],
                cmap="viridis",
                vmin=0.0,
                vmax=upper,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_xticks(range(13))
            axis.set_yticks(range(13))
            axis.tick_params(labelsize=6)
            if row == 0:
                axis.set_title(
                    f"{role}: B{item['block']:02d} H{item['head']:02d}",
                    fontsize=10,
                )
            if column == 0:
                axis.set_ylabel(
                    f"step {int(step)}\nquery latent time",
                    fontsize=8,
                )
            if row == len(steps) - 1:
                axis.set_xlabel("key latent time", fontsize=8)
        assert image is not None
        fig.colorbar(image, ax=list(axes[:, column]), fraction=0.025)
    fig.suptitle(
        f"{model} | {case} | S/T/P/C/G representative heads | "
        "exact 13x13 softmax temporal attention",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selection = json.loads(
        args.selection.expanduser().resolve().read_text(encoding="utf-8")
    )["samples"]
    model = str(args.model)
    if model not in selection:
        raise KeyError(f"selection has no model {model}")
    capture_root = args.capture_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    figures = []
    contacts = []
    temporal_contacts = []
    for case, case_item in selection[model].items():
        role_data = {}
        by_pair: dict[tuple[int, int], list[str]] = {}
        for role in ROLES:
            pair = case_item["roles"][role]
            block, head = int(pair["block"]), int(pair["head"])
            by_pair.setdefault((block, head), []).append(role)
            steps, attention, temporal = _load_head(
                capture_root,
                model=model,
                case=case,
                block=block,
                head=head,
            )
            role_data[role] = {
                "block": block,
                "head": head,
                "steps": steps,
                "attention": attention,
                "temporal": temporal,
            }
        for (block, head), roles in by_pair.items():
            item = role_data[roles[0]]
            output = (
                output_dir
                / model
                / case
                / f"{'_'.join(roles)}_block{block:02d}_head{head:02d}.png"
            )
            _render_head(
                model=model,
                case=case,
                roles=roles,
                block=block,
                head=head,
                steps=item["steps"],
                attention=item["attention"],
                output=output,
            )
            figures.append(str(output))
        contact = output_dir / model / case / "all_roles_softmax_qk.png"
        _render_contact(
            model=model,
            case=case,
            role_data=role_data,
            output=contact,
        )
        contacts.append(str(contact))
        temporal_contact = (
            output_dir / model / case / "all_roles_temporal_13x13.png"
        )
        _render_temporal_contact(
            model=model,
            case=case,
            role_data=role_data,
            output=temporal_contact,
        )
        temporal_contacts.append(str(temporal_contact))
    manifest = output_dir / model / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "model": model,
                "softmax_only": True,
                "figures": figures,
                "contact_sheets": contacts,
                "temporal_contact_sheets": temporal_contacts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[pending-qk-render] model={model} "
        f"heads={len(figures)} qk_contacts={len(contacts)} "
        f"temporal_contacts={len(temporal_contacts)}"
    )


if __name__ == "__main__":
    main()
