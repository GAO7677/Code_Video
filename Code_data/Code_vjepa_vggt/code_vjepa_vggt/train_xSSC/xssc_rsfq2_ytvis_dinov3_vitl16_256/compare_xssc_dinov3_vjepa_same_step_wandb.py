#!/usr/bin/env python3
"""Compare exact-step YTVIS segmentation logs from DINOv3 and V-JEPA xSSC."""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path

import wandb


METRICS = (
    "train/mbo_epoch",
    "train/recon_epoch",
    "val/ari",
    "val/ari_fg",
    "val/mbo",
    "val/miou",
    "val/recon",
)
SEGMENTATION_METRICS = (
    "train/mbo_epoch",
    "val/ari",
    "val/ari_fg",
    "val/mbo",
    "val/miou",
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dinov3-run", default="875222004-gy/xssc_dinov3/jyweols1"
    )
    parser.add_argument(
        "--vjepa-run", default="875222004-gy/xssc_vjepa2_1_video/sfrma3mp"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/xssc_dinov3_vjepa_same_step_compare"
        ),
    )
    return parser.parse_args()


def metric_history(run, metric):
    result = {}
    for row in run.scan_history(keys=["optimizer_step", metric], page_size=1000):
        step, value = row.get("optimizer_step"), row.get(metric)
        if step is not None and value is not None:
            result[int(step)] = float(value)
    return result


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=90)
    runs = {
        "dinov3": api.run(args.dinov3_run),
        "vjepa": api.run(args.vjepa_run),
    }
    histories = {
        label: {metric: metric_history(run, metric) for metric in METRICS}
        for label, run in runs.items()
    }
    common_steps = sorted(
        set(histories["dinov3"]["val/mbo"])
        & set(histories["vjepa"]["val/mbo"])
    )
    rows = []
    for step in common_steps:
        row = {"optimizer_step": step}
        for metric in METRICS:
            dinov3 = histories["dinov3"][metric].get(step)
            vjepa = histories["vjepa"][metric].get(step)
            row[f"dinov3:{metric}"] = dinov3
            row[f"vjepa:{metric}"] = vjepa
            row[f"delta_vjepa_minus_dinov3:{metric}"] = (
                None if dinov3 is None or vjepa is None else vjepa - dinov3
            )
        rows.append(row)

    config_keys = (
        "variant_name",
        "train_clip_frames",
        "raw_clip_frames",
        "label_frame_indices",
        "batch_size_t",
        "expected_world_size",
        "gradient_accumulation_steps",
        "effective_global_batch_size",
        "lr",
        "val_interval",
    )
    report = {
        "runs": {
            label: {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "url": run.url,
                "config": {key: run.config.get(key) for key in config_keys},
            }
            for label, run in runs.items()
        },
        "common_exact_validation_steps": common_steps,
        "rows": rows,
        "interpretation": {
            "higher_is_better": list(SEGMENTATION_METRICS),
            "recon_warning": (
                "Absolute recon losses are not comparable because each frozen "
                "backbone defines a different feature target and scale."
            ),
            "temporal_warning": (
                "DINOv3 evaluates all video frames; V-JEPA uses second-frame "
                "tubelet targets, so this is not a pure backbone-only ablation."
            ),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        with (output_dir / "same_step_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# xSSC DINOv3 vs V-JEPA: exact-step YTVIS comparison",
        "",
        f"- DINOv3: [{runs['dinov3'].id}]({runs['dinov3'].url})",
        f"- V-JEPA: [{runs['vjepa'].id}]({runs['vjepa'].url})",
        "- Positive delta means V-JEPA is higher.",
        "- Do not compare absolute reconstruction losses across backbones.",
        "",
        "| step | DINO mBO | V-JEPA mBO | delta | DINO ARI-FG | V-JEPA ARI-FG | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {optimizer_step} | {d_mbo:.6f} | {v_mbo:.6f} | {x_mbo:+.6f} "
            "| {d_fg:.6f} | {v_fg:.6f} | {x_fg:+.6f} |".format(
                optimizer_step=row["optimizer_step"],
                d_mbo=row["dinov3:val/mbo"],
                v_mbo=row["vjepa:val/mbo"],
                x_mbo=row["delta_vjepa_minus_dinov3:val/mbo"],
                d_fg=row["dinov3:val/ari_fg"],
                v_fg=row["vjepa:val/ari_fg"],
                x_fg=row["delta_vjepa_minus_dinov3:val/ari_fg"],
            )
        )
    lines.extend(
        [
            "",
            "| step | DINO train mBO | V-JEPA train mBO | delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {optimizer_step} | {d_mbo:.6f} | {v_mbo:.6f} | {x_mbo:+.6f} |".format(
                optimizer_step=row["optimizer_step"],
                d_mbo=row["dinov3:train/mbo_epoch"],
                v_mbo=row["vjepa:train/mbo_epoch"],
                x_mbo=row["delta_vjepa_minus_dinov3:train/mbo_epoch"],
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "DINOv3 uses five frame-level xSSC steps per training clip. V-JEPA uses six raw frames but three native tubelet steps and second-frame labels. Segmentation metrics share the same YTVIS-HQ validation source, but the temporal sampling is different.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))
    print(json.dumps({"output_dir": str(output_dir), "steps": common_steps}, indent=2))


if __name__ == "__main__":
    main()
