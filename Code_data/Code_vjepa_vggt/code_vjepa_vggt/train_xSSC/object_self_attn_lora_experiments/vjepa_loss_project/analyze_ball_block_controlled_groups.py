#!/usr/bin/env python3
"""Analyze controlled ball-block videos with cached V-JEPA2 patch tokens."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import compute_vjepa2_feature_mse as common
import visualize_ball_block_pairwise as pairwise
from visualize_ball_block_temporal_similarity import expand_tubelets, token_cosine_curve


DATASET_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block")
RESULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/vjepa_ball_block_pairwise/"
    "ball_block49_native_rect_vitl_with_raw_20260808"
)
VJEPA_REPO = Path("/home/gaoya/Code_Video/vjepa2_tinyvae_mse/vjepa2")
CHECKPOINT = Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")
BASELINE = "e07_mu05_m1"


GROUPS = [
    {
        "id": "restitution",
        "title": "Restitution",
        "symbol": "e",
        "unit": "",
        "settings": [
            (0.0, "extreme_restitution_000"),
            (0.3, "e03_mu05_m1"),
            (0.5, "e05_mu05_m1"),
            (0.7, BASELINE),
            (0.9, "e09_mu05_m1"),
            (1.0, "extreme_restitution_100"),
        ],
    },
    {
        "id": "friction",
        "title": "Lateral friction",
        "symbol": "mu",
        "unit": "",
        "settings": [
            (0.0, "extreme_friction_000"),
            (0.1, "e07_mu01_m1"),
            (0.5, BASELINE),
            (1.0, "e07_mu10_m1"),
            (2.0, "extreme_friction_200"),
        ],
    },
    {
        "id": "mass",
        "title": "Ball mass",
        "symbol": "m",
        "unit": "x",
        "settings": [
            (0.05, "extreme_mass_005x"),
            (0.1, "e07_mu05_m01"),
            (1.0, BASELINE),
            (5.0, "e07_mu05_m5"),
            (10.0, "extreme_mass_10x"),
        ],
    },
    {
        "id": "speed",
        "title": "Initial speed scale",
        "symbol": "speed",
        "unit": "x",
        "settings": [
            (0.25, "extreme_speed_025x"),
            (0.5, "motion_speed_050x"),
            (0.75, "motion_speed_075x"),
            (1.0, BASELINE),
            (1.25, "motion_speed_125x"),
            (1.5, "motion_speed_150x"),
            (2.0, "extreme_speed_200x"),
        ],
    },
    {
        "id": "direction",
        "title": "Direction yaw",
        "symbol": "yaw",
        "unit": "deg",
        "settings": [
            (-25.0, "extreme_direction_yaw_m25"),
            (-10.0, "motion_direction_yaw_m10"),
            (0.0, BASELINE),
            (10.0, "motion_direction_yaw_p10"),
            (25.0, "extreme_direction_yaw_p25"),
        ],
    },
    {
        "id": "distance",
        "title": "Initial distance scale",
        "symbol": "distance",
        "unit": "x",
        "settings": [
            (0.35, "extreme_distance_035x"),
            (0.5, "motion_distance_050x"),
            (0.75, "motion_distance_075x"),
            (1.0, BASELINE),
            (1.25, "motion_distance_125x"),
            (1.5, "motion_distance_150x"),
            (2.0, "extreme_distance_200x"),
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vjepa-repo", type=Path, default=VJEPA_REPO)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--model", default="vjepa2.1-vitl-384")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--input-height", type=int, default=384)
    parser.add_argument("--input-width", type=int, default=672)
    return parser.parse_args()


def setting_label(group: dict, value: float) -> str:
    if group["unit"] == "deg":
        return f"{value:+g} deg"
    if group["unit"] == "x":
        return f"{value:g}x"
    return f"{group['symbol']}={value:g}"


def feature_path(result_root: Path, stem: str) -> Path:
    return result_root / "features" / f"{stem}.npy"


def ensure_features(args: argparse.Namespace, stems: list[str]) -> None:
    missing = [stem for stem in stems if not feature_path(args.result_root, stem).exists()]
    if not missing:
        print("All feature caches already exist; encoder loading is skipped.")
        return

    for stem in missing:
        video_path = args.dataset_dir / f"{stem}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing controlled video: {video_path}")

    device = common.resolve_device(args.device)
    dtype = common.dtype_for(device, args.dtype)
    args.vjepa2_dir = args.vjepa_repo
    encoder, _ = common.load_encoder(args, device, dtype)
    encoder.eval()
    args.result_root.joinpath("features").mkdir(parents=True, exist_ok=True)

    print(f"Extracting {len(missing)} missing feature caches: {', '.join(missing)}")
    for index, stem in enumerate(missing, start=1):
        video_path = args.dataset_dir / f"{stem}.mp4"
        print(f"[{index}/{len(missing)}] {stem}")
        pairwise.extract_video_features(
            encoder,
            video_path,
            feature_path(args.result_root, stem),
            num_frames=args.num_frames,
            input_height=args.input_height,
            input_width=args.input_width,
            device=device,
            dtype=dtype,
        )

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_feature(result_root: Path, stem: str) -> Path:
    path = feature_path(result_root, stem)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def frame_curve(feature_a: Path, feature_b: Path, num_frames: int) -> np.ndarray:
    tubelet_mean, _ = token_cosine_curve(feature_a, feature_b)
    return np.asarray(expand_tubelets(tubelet_mean, num_frames), dtype=np.float32)


def analyze_groups(args: argparse.Namespace) -> list[dict]:
    analyzed = []
    cache: dict[str, Path] = {}

    def get(stem: str) -> Path:
        if stem not in cache:
            cache[stem] = load_feature(args.result_root, stem)
        return cache[stem]

    baseline_feature = get(BASELINE)
    for group in GROUPS:
        settings = []
        curves: dict[str, np.ndarray] = {}
        stems = [stem for _, stem in group["settings"]]

        for value, stem in group["settings"]:
            if stem == BASELINE:
                curve = np.ones(args.num_frames, dtype=np.float32)
            else:
                curve = frame_curve(baseline_feature, get(stem), args.num_frames)
            curves[stem] = curve
            min_frame = int(np.argmin(curve))
            settings.append(
                {
                    "value": value,
                    "stem": stem,
                    "label": setting_label(group, value),
                    "is_baseline": stem == BASELINE,
                    "mean_similarity": float(np.mean(curve)),
                    "min_similarity": float(curve[min_frame]),
                    "min_frame": min_frame,
                    "min_time_seconds": min_frame / 60.0,
                    "frame_similarity": curve.tolist(),
                }
            )

        matrix = np.eye(len(stems), dtype=np.float32)
        for row in range(len(stems)):
            for column in range(row + 1, len(stems)):
                curve = frame_curve(get(stems[row]), get(stems[column]), args.num_frames)
                matrix[row, column] = matrix[column, row] = float(np.mean(curve))

        analyzed.append(
            {
                "id": group["id"],
                "title": group["title"],
                "symbol": group["symbol"],
                "unit": group["unit"],
                "settings": settings,
                "similarity_matrix": matrix.tolist(),
            }
        )
    return analyzed


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#f5f1e8")
    axis.grid(True, color="#d8d0bf", linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#7b7468")


def plot_temporal(groups: list[dict], output_path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    frames = np.arange(49)
    for axis, group in zip(axes.flat, groups):
        style_axis(axis)
        plotted = 0
        for setting in group["settings"]:
            if setting["is_baseline"]:
                continue
            axis.plot(
                frames,
                setting["frame_similarity"],
                linewidth=2.0,
                color=colors[plotted],
                label=setting["label"],
            )
            plotted += 1
        axis.set_title(group["title"], loc="left", fontsize=14, fontweight="bold")
        axis.set_xlabel("Frame (60 FPS)")
        axis.set_ylabel("Patch cosine vs baseline")
        axis.legend(frameon=False, ncols=2, fontsize=9)
    figure.suptitle(
        "V-JEPA2 controlled-variable temporal similarity",
        fontsize=20,
        fontweight="bold",
        color="#19352f",
    )
    figure.savefig(output_path, dpi=180, facecolor="#eee8db")
    plt.close(figure)


def plot_summary(groups: list[dict], output_path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    for axis, group in zip(axes.flat, groups):
        style_axis(axis)
        values = np.array([item["value"] for item in group["settings"]], dtype=np.float32)
        means = np.array([item["mean_similarity"] for item in group["settings"]], dtype=np.float32)
        axis.plot(values, means, color="#d45d3f", linewidth=2.4, marker="o", markersize=7)
        baseline_index = next(i for i, item in enumerate(group["settings"]) if item["is_baseline"])
        axis.scatter(
            values[baseline_index],
            means[baseline_index],
            s=115,
            color="#1f6f62",
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
            label="baseline",
        )
        for value, mean in zip(values, means):
            axis.annotate(f"{mean:.4f}", (value, mean), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        axis.set_xticks(values)
        axis.set_title(group["title"], loc="left", fontsize=14, fontweight="bold")
        axis.set_xlabel(f"{group['symbol']} ({group['unit'] or 'value'})")
        axis.set_ylabel("Mean cosine vs baseline")
        axis.legend(frameon=False)
    figure.suptitle(
        "Setting effect relative to e07_mu05_m1",
        fontsize=20,
        fontweight="bold",
        color="#19352f",
    )
    figure.savefig(output_path, dpi=180, facecolor="#eee8db")
    plt.close(figure)


def plot_matrices(groups: list[dict], output_path: Path) -> None:
    off_diagonal = []
    for group in groups:
        matrix = np.asarray(group["similarity_matrix"])
        off_diagonal.extend(matrix[~np.eye(matrix.shape[0], dtype=bool)].tolist())
    vmin = min(off_diagonal)

    figure, axes = plt.subplots(3, 2, figsize=(16, 15), constrained_layout=True)
    image = None
    for axis, group in zip(axes.flat, groups):
        matrix = np.asarray(group["similarity_matrix"])
        labels = [setting["label"] for setting in group["settings"]]
        image = axis.imshow(matrix, vmin=vmin, vmax=1.0, cmap="YlGnBu")
        axis.set_title(group["title"], loc="left", fontsize=14, fontweight="bold")
        axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels=labels)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = "white" if matrix[row, column] < (vmin + 1.0) / 2 else "#19352f"
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color=color, fontsize=8)
    figure.colorbar(image, ax=axes, shrink=0.72, label="Mean corresponding-patch cosine")
    figure.suptitle(
        "Within-group pairwise V-JEPA2 similarity",
        fontsize=20,
        fontweight="bold",
        color="#19352f",
    )
    figure.savefig(output_path, dpi=180, facecolor="#eee8db")
    plt.close(figure)


def write_csv(groups: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "group",
                "setting",
                "value",
                "video",
                "is_baseline",
                "mean_similarity",
                "min_similarity",
                "min_frame",
                "min_time_seconds",
            ),
        )
        writer.writeheader()
        for group in groups:
            for setting in group["settings"]:
                writer.writerow(
                    {
                        "group": group["id"],
                        "setting": setting["label"],
                        "value": setting["value"],
                        "video": setting["stem"],
                        "is_baseline": setting["is_baseline"],
                        "mean_similarity": f"{setting['mean_similarity']:.8f}",
                        "min_similarity": f"{setting['min_similarity']:.8f}",
                        "min_frame": setting["min_frame"],
                        "min_time_seconds": f"{setting['min_time_seconds']:.6f}",
                    }
                )


def write_html(groups: list[dict], output_path: Path) -> None:
    sections = []
    for group in groups:
        rows = []
        for setting in group["settings"]:
            badge = '<span class="baseline">baseline</span>' if setting["is_baseline"] else ""
            rows.append(
                "<tr>"
                f"<td>{html.escape(setting['label'])} {badge}</td>"
                f"<td><code>{html.escape(setting['stem'])}</code></td>"
                f"<td>{setting['mean_similarity']:.5f}</td>"
                f"<td>{setting['min_similarity']:.5f}</td>"
                f"<td>{setting['min_frame']} ({setting['min_time_seconds']:.3f}s)</td>"
                "</tr>"
            )
        sections.append(
            f"<section><h2>{html.escape(group['title'])}</h2>"
            "<div class=table-wrap><table><thead><tr>"
            "<th>Setting</th><th>Video</th><th>Mean vs baseline</th><th>Minimum</th><th>Minimum time</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
        )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA controlled groups</title>
<style>
:root{{--ink:#19352f;--paper:#eee8db;--card:#fffdf7;--accent:#d45d3f;--line:#d8d0bf}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 85% 5%,#d8ddc8 0,transparent 28%),var(--paper);color:var(--ink);font-family:Georgia,'Times New Roman',serif}}
main{{width:min(1440px,94vw);margin:auto;padding:52px 0 80px}} .eyebrow{{font:700 12px/1.2 sans-serif;letter-spacing:.18em;text-transform:uppercase;color:var(--accent)}}
h1{{font-size:clamp(36px,6vw,76px);line-height:.95;margin:12px 0 20px;max-width:1050px}} .intro{{font-size:18px;line-height:1.6;max-width:920px}}
.notice{{margin:28px 0;padding:18px 22px;border-left:5px solid var(--accent);background:#fff7e8;font:15px/1.55 sans-serif}}
.plot{{display:block;width:100%;margin:28px 0;background:var(--card);border:1px solid var(--line);box-shadow:0 14px 40px #19352f18}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;margin-top:34px}} section{{background:var(--card);padding:24px;border:1px solid var(--line)}}
h2{{margin:0 0 15px;font-size:25px}} table{{width:100%;border-collapse:collapse;font:13px/1.4 sans-serif}} th,td{{padding:9px 8px;text-align:left;border-bottom:1px solid #e8e1d4}} th{{color:#61706a;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
code{{font-size:11px}} .baseline{{display:inline-block;background:#1f6f62;color:white;border-radius:999px;padding:2px 7px;font:9px sans-serif;text-transform:uppercase}} nav{{display:flex;gap:12px;margin-top:26px}} nav a{{color:var(--ink);font:700 13px sans-serif;text-decoration:none;border-bottom:2px solid var(--accent);padding:5px 0}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}} .table-wrap{{overflow-x:auto}} main{{padding-top:32px}}}}
</style></head><body><main>
<div class="eyebrow">V-JEPA2 / controlled-variable audit</div><h1>What changes when one physical control changes?</h1>
<p class="intro">All comparisons use the first 49 frames at native rectangular resize, normalized V-JEPA2 ViT-L patch tokens, and corresponding-patch cosine similarity. The common baseline is <code>{BASELINE}</code>.</p>
<div class="notice"><strong>Interpretation boundary:</strong> each setting contains one deterministic video and no repeated random seed. These are descriptive representation differences, not uncertainty-calibrated causal estimates. The encoder sees the full clip, so an early-frame token can also contain information about later events.</div>
<nav><a href="index.html">Pairwise gallery</a><a href="temporal_similarity.html">All pair curves</a><a href="controlled_groups/controlled_group_similarity.csv">Download CSV</a><a href="controlled_groups/controlled_group_similarity.json">Download JSON</a></nav>
<img class="plot" src="controlled_groups/controlled_group_temporal_curves.png" alt="Temporal similarity curves">
<img class="plot" src="controlled_groups/controlled_group_setting_summary.png" alt="Setting summaries">
<img class="plot" src="controlled_groups/controlled_group_similarity_matrices.png" alt="Similarity matrices">
<div class="grid">{''.join(sections)}</div>
</main></body></html>"""
    output_path.write_text(page, encoding="utf-8")


def print_summary(groups: list[dict]) -> None:
    print("\nControlled-group summary (corresponding-patch cosine vs baseline)")
    for group in groups:
        print(f"\n[{group['title']}]")
        for setting in group["settings"]:
            if setting["is_baseline"]:
                continue
            print(
                f"  {setting['label']:>10}  mean={setting['mean_similarity']:.6f}  "
                f"min={setting['min_similarity']:.6f} at frame {setting['min_frame']}"
            )


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or args.result_root / "controlled_groups"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted({stem for group in GROUPS for _, stem in group["settings"]})
    ensure_features(args, stems)
    groups = analyze_groups(args)

    payload = {
        "baseline": BASELINE,
        "num_frames": args.num_frames,
        "fps": 60.0,
        "feature_metric": "mean corresponding-patch cosine on normalized V-JEPA2 tokens",
        "groups": groups,
    }
    (args.output_dir / "controlled_group_similarity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(groups, args.output_dir / "controlled_group_similarity.csv")
    plot_temporal(groups, args.output_dir / "controlled_group_temporal_curves.png")
    plot_summary(groups, args.output_dir / "controlled_group_setting_summary.png")
    plot_matrices(groups, args.output_dir / "controlled_group_similarity_matrices.png")
    write_html(groups, args.result_root / "controlled_groups.html")
    print_summary(groups)
    print(f"\nDashboard: {args.result_root / 'controlled_groups.html'}")


if __name__ == "__main__":
    main()
