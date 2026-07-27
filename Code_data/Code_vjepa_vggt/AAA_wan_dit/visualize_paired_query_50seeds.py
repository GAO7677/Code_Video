#!/usr/bin/env python3
"""Visualize partial or complete paired-query head-role statistics."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from analyze_multiblock_ball_query_heads import ROLE_LABELS, _role_scores
from moving_query_attention import FEATURE_NAMES


ROLES = tuple(ROLE_LABELS)
PROTOCOLS = ("moving", "anchor_t2")
PROTOCOL_LABELS = {
    "moving": "Moving query",
    "anchor_t2": "Fixed query at latent t=2",
}
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
ROLE_COLORS = ("#2672B8", "#D1495B", "#2A9D6F", "#E69F3A", "#737B86")
ROLE_DESCRIPTIONS = {
    "S": "within-frame spatial",
    "T": "moving-object trajectory",
    "P": "fixed-position temporal",
    "C": "history/context",
    "G": "global aggregation",
}
BLOCK_PATTERN = re.compile(r"block(\d{2})_paired_query_features\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def classify(arrays: np.lib.npyio.NpzFile, protocol: str) -> np.ndarray:
    features = {
        name: arrays[f"{protocol}__{name}"].astype(np.float64).mean(axis=0)
        for name in FEATURE_NAMES
    }
    scores = _role_scores(features)
    matrix = np.stack([scores[role] for role in ROLES], axis=1)
    return matrix.argmax(axis=1)


def complete_seeds(root: Path, model: str) -> list[int]:
    seeds = []
    for path in sorted((root / "state" / model).glob("seed-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            seeds.append(int(payload["seed"]))
    return seeds


def package_paths(root: Path, model: str, seed: int) -> list[Path]:
    return sorted(
        (
            root
            / "capture"
            / model
            / f"seed-{seed:06d}"
        ).glob(
            f"block*/matrices/{model}/*/block*_paired_query_features.npz"
        )
    )


def configure_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=10, pad=7)
    axis.set_xlabel("Head")
    axis.set_ylabel("Block")
    axis.set_xticks(np.arange(0, 24, 2))
    axis.set_yticks(np.arange(0, 30, 2))
    axis.tick_params(labelsize=7)


def plot_model(
    output: Path,
    model: str,
    counts: dict[tuple[str, str], np.ndarray],
    agreements: dict[str, tuple[np.ndarray, np.ndarray]],
    seed_count: int,
) -> str | None:
    if seed_count == 0:
        return None
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(15, 14),
        dpi=160,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.0, 0.92)},
    )
    role_cmap = ListedColormap(ROLE_COLORS)
    role_norm = BoundaryNorm(np.arange(-0.5, len(ROLES) + 0.5), len(ROLES))

    for column, protocol in enumerate(PROTOCOLS):
        values = counts[(model, protocol)]
        samples = values.sum(axis=2)
        dominant = values.argmax(axis=2).astype(np.float64)
        dominant[samples == 0] = np.nan
        consistency = values.max(axis=2) / np.maximum(samples, 1)
        consistency[samples == 0] = np.nan

        axes[0, column].imshow(
            dominant,
            cmap=role_cmap,
            norm=role_norm,
            interpolation="nearest",
            aspect="auto",
        )
        configure_axis(
            axes[0, column],
            f"{PROTOCOL_LABELS[protocol]}: dominant role",
        )
        image = axes[1, column].imshow(
            consistency,
            cmap="viridis",
            vmin=0.2,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )
        configure_axis(
            axes[1, column],
            f"{PROTOCOL_LABELS[protocol]}: role consistency",
        )
        figure.colorbar(image, ax=axes[1, column], fraction=0.025, pad=0.02)

    numerator, denominator = agreements[model]
    agreement = numerator / np.maximum(denominator, 1)
    agreement[denominator == 0] = np.nan
    image = axes[2, 0].imshow(
        agreement,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    configure_axis(axes[2, 0], "Fixed vs moving role agreement")
    figure.colorbar(image, ax=axes[2, 0], fraction=0.025, pad=0.02)

    role_totals = np.stack(
        [counts[(model, protocol)].sum(axis=(0, 1)) for protocol in PROTOCOLS]
    ).astype(np.float64)
    role_fractions = role_totals / np.maximum(role_totals.sum(axis=1)[:, None], 1)
    left = np.zeros(len(PROTOCOLS))
    for index, role in enumerate(ROLES):
        axes[2, 1].barh(
            np.arange(len(PROTOCOLS)),
            role_fractions[:, index],
            left=left,
            color=ROLE_COLORS[index],
            height=0.55,
            label=role,
        )
        left += role_fractions[:, index]
    axes[2, 1].set_yticks(
        np.arange(len(PROTOCOLS)),
        [PROTOCOL_LABELS[protocol] for protocol in PROTOCOLS],
    )
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].set_xlabel("Fraction of block/head/sample assignments")
    axes[2, 1].set_title("Role distribution", fontsize=10, pad=7)
    axes[2, 1].grid(axis="x", alpha=0.25)
    axes[2, 1].legend(ncol=5, loc="lower center", fontsize=8)

    figure.suptitle(
        f"{MODEL_LABELS[model]} paired-query head roles | "
        f"{seed_count}/50 complete seeds",
        fontsize=14,
        y=1.035,
    )
    legend = [
        Patch(facecolor=color, label=f"{role}: {ROLE_DESCRIPTIONS[role]}")
        for role, color in zip(ROLES, ROLE_COLORS)
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=5,
        fontsize=8,
    )
    filename = f"{model}_head_role_statistics.png"
    figure.savefig(output / filename, bbox_inches="tight")
    plt.close(figure)
    return filename


def write_csv(
    path: Path,
    models: tuple[str, ...],
    counts: dict[tuple[str, str], np.ndarray],
) -> None:
    fields = [
        "model",
        "protocol",
        "block",
        "head",
        "valid_samples",
        "dominant_role",
        "role_consistency",
        *[f"{role}_fraction" for role in ROLES],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in models:
            for protocol in PROTOCOLS:
                values = counts[(model, protocol)]
                for block in range(30):
                    for head in range(24):
                        role_counts = values[block, head]
                        total = int(role_counts.sum())
                        dominant = (
                            ROLES[int(role_counts.argmax())] if total else ""
                        )
                        row = {
                            "model": model,
                            "protocol": protocol,
                            "block": block,
                            "head": head,
                            "valid_samples": total,
                            "dominant_role": dominant,
                            "role_consistency": (
                                float(role_counts.max() / total) if total else ""
                            ),
                        }
                        row.update(
                            {
                                f"{role}_fraction": (
                                    float(role_counts[index] / total)
                                    if total
                                    else ""
                                )
                                for index, role in enumerate(ROLES)
                            }
                        )
                        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = json.loads(
        args.config.expanduser().resolve().read_text(encoding="utf-8")
    )
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = tuple(config["models"])
    expected_seeds = len(config["seed_sampling"]["seeds"])

    counts = {
        (model, protocol): np.zeros((30, 24, len(ROLES)), dtype=np.int64)
        for model in models
        for protocol in PROTOCOLS
    }
    agreements = {
        model: (
            np.zeros((30, 24), dtype=np.float64),
            np.zeros((30, 24), dtype=np.float64),
        )
        for model in models
    }
    seeds_by_model = {model: complete_seeds(root, model) for model in models}
    invalid_anchors: dict[str, set[tuple[int, str]]] = {
        model: set() for model in models
    }

    for model in models:
        for seed in seeds_by_model[model]:
            packages = package_paths(root, model, seed)
            if len(packages) != 600:
                raise RuntimeError(
                    f"{model}/seed-{seed:06d}: expected 600 packages, "
                    f"found {len(packages)}"
                )
            for path in packages:
                match = BLOCK_PATTERN.fullmatch(path.name)
                if match is None:
                    raise ValueError(path)
                block = int(match.group(1))
                case = path.parent.name
                with np.load(path) as arrays:
                    moving = classify(arrays, "moving")
                    for head, role_index in enumerate(moving):
                        counts[(model, "moving")][block, head, role_index] += 1
                    anchor_valid = bool(arrays["anchor_t2_valid"])
                    if not anchor_valid:
                        invalid_anchors[model].add((seed, case))
                        continue
                    anchor = classify(arrays, "anchor_t2")
                    for head, role_index in enumerate(anchor):
                        counts[(model, "anchor_t2")][
                            block, head, role_index
                        ] += 1
                    numerator, denominator = agreements[model]
                    numerator[block] += moving == anchor
                    denominator[block] += 1

    write_csv(output / "head_role_statistics.csv", models, counts)
    images = {
        model: plot_model(
            output,
            model,
            counts,
            agreements,
            len(seeds_by_model[model]),
        )
        for model in models
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "generated_at": generated_at,
        "expected_seeds": expected_seeds,
        "complete_seeds": {
            model: seeds_by_model[model] for model in models
        },
        "invalid_anchor_model_case_seed_count": {
            model: len(invalid_anchors[model]) for model in models
        },
        "images": images,
        "classification": {
            "roles": ROLE_DESCRIPTIONS,
            "scope": (
                "relative rank among 24 heads for each "
                "model/seed/case/block"
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    progress_rows = "".join(
        "<tr>"
        f"<td>{html.escape(MODEL_LABELS[model])}</td>"
        f"<td>{len(seeds_by_model[model])}/{expected_seeds}</td>"
        f"<td>{len(seeds_by_model[model]) * 20}</td>"
        f"<td>{len(invalid_anchors[model])}</td>"
        "</tr>"
        for model in models
    )
    sections = "".join(
        (
            f"<section><h2>{html.escape(MODEL_LABELS[model])}</h2>"
            f"<img src='{html.escape(images[model])}' "
            f"alt='{html.escape(MODEL_LABELS[model])} head role statistics'>"
            "</section>"
            if images[model]
            else (
                f"<section><h2>{html.escape(MODEL_LABELS[model])}</h2>"
                "<p class='pending'>No complete seed is available yet.</p>"
                "</section>"
            )
        )
        for model in models
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="600">
<title>Paired-query head stability</title>
<style>
:root{{--ink:#17212b;--muted:#5c6773;--line:#d8dde3;--paper:#f6f7f8;--panel:#fff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:14px/1.45 Arial,sans-serif;letter-spacing:0}} header{{background:#1f3430;color:#fff;
padding:18px 28px}} h1{{font-size:22px;margin:0 0 5px}} header p{{margin:0;color:#dbe6e2}}
main{{max-width:1500px;margin:0 auto;padding:20px 24px 40px}} table{{border-collapse:collapse;
width:100%;background:var(--panel);margin-bottom:20px}} th,td{{padding:9px 12px;
border-bottom:1px solid var(--line);text-align:left}} th{{background:#edf0f2;font-size:12px}}
section{{border-top:1px solid var(--line);padding:18px 0 24px}} h2{{font-size:17px;margin:0 0 12px}}
img{{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--line)}}
.pending{{padding:28px;background:#fff;border:1px dashed #9da7b1;color:var(--muted)}}
.note{{color:var(--muted);font-size:12px;margin:8px 0 18px}} a{{color:#126a58}}
</style>
</head>
<body>
<header><h1>50-seed paired-query head-role statistics</h1>
<p>Fixed latent-t2 query compared with per-frame moving query</p></header>
<main>
<table><thead><tr><th>Model</th><th>Complete seeds</th><th>Complete generated samples</th>
<th>Invalid fixed anchors</th></tr></thead><tbody>{progress_rows}</tbody></table>
<p class="note">Generated {html.escape(generated_at)}. Only complete seeds are included.
Fixed-query statistics exclude samples without a valid object anchor at latent t=2.
S/T/P/C/G are relative role assignments among the 24 heads, not causal labels.
The page refreshes every 10 minutes; regenerate the artifacts to update the data.</p>
{sections}
</main></body></html>
"""
    (output / "index.html").write_text(document, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
