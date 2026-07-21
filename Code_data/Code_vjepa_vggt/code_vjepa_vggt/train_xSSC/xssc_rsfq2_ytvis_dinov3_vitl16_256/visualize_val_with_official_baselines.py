#!/usr/bin/env python3
"""Plot DINOv3 xSSC validation curves against official xSSC baselines."""

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VAL_PATTERN = re.compile(r"^(?P<step>\d+)-val (?P<metrics>\{.*\})$")
DISCOVERY_METRICS = {
    "ari": ("ARI", "higher is better"),
    "ari_fg": ("ARI-FG", "higher is better"),
    "mbo": ("mBO", "higher is better"),
    "miou": ("mIoU", "higher is better"),
}
COLORS = {
    "DINOv3 slot-512": "#2563EB",
    "rsfq2_r-ytvis": "#D97706",
    "rsfq2_r-ytvis_hq": "#059669",
    "rsfq2_c-movi_c": "#7C3AED",
    "rsfq2_c-movi_e": "#DC2626",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--official-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def load_dinov3_history(path):
    rows = []
    with path.open(errors="replace") as handle:
        for raw_line in handle:
            match = VAL_PATTERN.match(raw_line.strip())
            if match:
                metrics = ast.literal_eval(match.group("metrics"))
                rows.append({"step": int(match.group("step")), **metrics})
    if not rows:
        raise RuntimeError(f"No validation rows found in {path}")
    return rows


def load_official_results(path):
    with path.open() as handle:
        results = json.load(handle)["results"]
    discovery = defaultdict(list)
    recognition = []
    for result in results:
        if result["task"] == "discovery":
            discovery[result["model"]].append(result)
        elif result["task"] == "recognition":
            recognition.append(result)
    return dict(discovery), recognition


def official_stats(discovery, metric):
    stats = {}
    for model, results in discovery.items():
        values = np.asarray([item["metrics"][metric] for item in results])
        stats[model] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return stats


def plot_discovery(history, discovery, output_path):
    steps = np.asarray([row["step"] for row in history])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(
        steps,
        [row["recon"] for row in history],
        color=COLORS["DINOv3 slot-512"],
        linewidth=2.4,
        marker="o",
        label="DINOv3 slot-512",
    )
    ax.set_title("Reconstruction loss (lower is better)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("recon")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    metric_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    for ax, (metric, (label, direction)) in zip(metric_axes, DISCOVERY_METRICS.items()):
        ax.plot(
            steps,
            [row[metric] for row in history],
            color=COLORS["DINOv3 slot-512"],
            linewidth=2.4,
            marker="o",
            label="DINOv3 slot-512",
            zorder=5,
        )
        for model in COLORS:
            if model == "DINOv3 slot-512" or model not in discovery:
                continue
            stats = official_stats(discovery, metric)[model]
            color = COLORS[model]
            ax.axhspan(stats["min"], stats["max"], color=color, alpha=0.08)
            ax.axhline(
                stats["mean"],
                color=color,
                linestyle="--",
                linewidth=1.7,
                label=f"{model} mean",
            )
        ax.set_title(f"{label} ({direction})")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    note_ax = axes[1, 2]
    note_ax.axis("off")
    note_ax.text(
        0.0,
        0.95,
        "Comparison notes",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    note_ax.text(
        0.0,
        0.82,
        "Solid blue: current DINOv3 validation curve\n"
        "Dashed line: official family mean over 3 seeds\n"
        "Shaded band: official seed min-max\n\n"
        "MOVi-C/E use official bbox-conditioned initialization.\n"
        "Their segmentation scores are not a like-for-like\n"
        "comparison with unconditional YTVIS models.\n\n"
        "DINOv2 and DINOv3 reconstruction losses have different\n"
        "feature scales, so official recon baselines are omitted.",
        fontsize=11,
        linespacing=1.4,
        va="top",
    )

    fig.suptitle("xSSC DINOv3 Validation vs Official xSSC Weights", fontsize=17)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_recognition(recognition, output_path):
    metrics = {
        "ce": ("Cross entropy", "lower is better"),
        "l1": ("Box L1", "lower is better"),
        "top1": ("Top-1", "higher is better"),
        "top3": ("Top-3", "higher is better"),
        "iou": ("Box IoU", "higher is better"),
    }
    seeds = [str(item["seed"]) for item in recognition]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (metric, (label, direction)), color in zip(
        axes.flat,
        metrics.items(),
        ["#D97706", "#DC2626", "#2563EB", "#7C3AED", "#059669"],
    ):
        values = [item["metrics"][metric] for item in recognition]
        ax.bar(seeds, values, color=color, width=0.62)
        ax.set_title(f"{label} ({direction})")
        ax.set_xlabel("official seed")
        ax.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    axes[1, 2].axis("off")
    fig.suptitle("Official xSSC Recognition Weights on Current YTVIS-HQ Val", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def log_to_wandb(args, history, discovery, discovery_plot, recognition_plot):
    import wandb

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name="val-comparison-official-baselines",
        job_type="evaluation",
        dir=str(args.output_dir),
        config={
            "train_log": str(args.train_log),
            "official_results": str(args.official_results),
            "note": "MOVi baselines are bbox-conditioned; DINOv2/DINOv3 recon is not comparable.",
        },
    )
    steps = [row["step"] for row in history]
    payload = {
        "comparison/discovery_plot": wandb.Image(str(discovery_plot)),
        "comparison/recognition_plot": wandb.Image(str(recognition_plot)),
    }
    for metric, (label, direction) in DISCOVERY_METRICS.items():
        series = [[row[metric] for row in history]]
        keys = ["DINOv3 slot-512"]
        for model in COLORS:
            if model == "DINOv3 slot-512" or model not in discovery:
                continue
            mean = official_stats(discovery, metric)[model]["mean"]
            series.append([mean] * len(steps))
            keys.append(f"{model} mean")
        payload[f"comparison/{metric}"] = wandb.plot.line_series(
            xs=steps,
            ys=series,
            keys=keys,
            title=f"{label}: DINOv3 vs official baselines ({direction})",
            xname="optimizer_step",
        )
    run.log(payload)
    run.finish()
    return run.url


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = load_dinov3_history(args.train_log)
    discovery, recognition = load_official_results(args.official_results)
    discovery_plot = args.output_dir / "dinov3_val_vs_official_discovery.png"
    recognition_plot = args.output_dir / "official_recognition_val.png"
    plot_discovery(history, discovery, discovery_plot)
    plot_recognition(recognition, recognition_plot)

    summary = {
        "val_points": len(history),
        "latest_step": history[-1]["step"],
        "discovery_plot": str(discovery_plot),
        "recognition_plot": str(recognition_plot),
    }
    if args.wandb_project:
        summary["wandb_url"] = log_to_wandb(
            args, history, discovery, discovery_plot, recognition_plot
        )
    with (args.output_dir / "visualization_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
