#!/usr/bin/env python3
"""Run the existing decoder static/dynamic analysis over all MOVi-C checkpoints."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analyze_decoder_static_dynamic_viewer import analyze_case  # noqa: E402
from analyze_slot_temporal_similarity_viewer import (  # noqa: E402
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_VIEWER_DIR,
    boxes_from_metadata,
    build_model,
    load_checkpoint,
    normalize_rgb_frames,
    read_frame_sequence,
)


DEFAULT_CHECKPOINT_DIR = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/"
    "dinov3_xSSC/restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42"
)
DEFAULT_CONFIG = (
    ROOT
    / "upstream/config-randsfq/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
)
OUTPUT_NAME = "decoder_static_dynamic_all_movic_checkpoints"
OFFICIAL_LABEL = "official_dinov2_movic_42_0035"
OFFICIAL_DISPLAY = "Official DINOv2 MOVi-C 42-0035"
OFFICIAL_CHECKPOINT = Path(
    "/data/gaoya/agent-data/weights/xssc_official_archive_rsfq2/"
    "rsfq2_c-movi_c/42-0035.pth"
)
METRIC_FIELDS = (
    "static_adjacent",
    "dynamic_adjacent",
    "static_frame0_final",
    "dynamic_frame0_final",
    "static_dc_ratio",
    "dynamic_dc_ratio",
    "full_mse",
    "dynamic_frozen_mse",
    "static_frozen_mse",
    "dynamic_freeze_delta",
    "static_freeze_delta",
    "full_temporal_delta_rms",
    "dynamic_frozen_temporal_delta_rms",
    "static_frozen_temporal_delta_rms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--stage", choices=("analyze", "report", "all"), default="all")
    parser.add_argument("--min-step", type=int, default=16000)
    parser.add_argument("--max-step", type=int, default=38000)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"step-(\d+)", path.stem)
    if not match:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints = sorted(
        (
            path.resolve()
            for path in args.checkpoint_dir.glob("step-*.pth")
            if args.min_step <= checkpoint_step(path) <= args.max_step
        ),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No step checkpoints in {args.checkpoint_dir} within "
            f"[{args.min_step}, {args.max_step}]"
        )
    expected = list(
        range(checkpoint_step(checkpoints[0]), checkpoint_step(checkpoints[-1]) + 1, 1000)
    )
    actual = [checkpoint_step(path) for path in checkpoints]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise RuntimeError(f"Checkpoint sequence has gaps: {missing}")
    return checkpoints


def validate(args: argparse.Namespace) -> None:
    required = [
        args.viewer_dir / "combined_metadata.json",
        args.outputs_root,
        args.checkpoint_dir,
        args.config,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))


def summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    full_mse = float(summary["decoder"]["full"]["mse_to_target"])
    dynamic_frozen_mse = float(
        summary["decoder"]["dynamic_frozen"]["mse_to_target"]
    )
    static_frozen_mse = float(
        summary["decoder"]["static_frozen"]["mse_to_target"]
    )
    return {
        "static_adjacent": float(
            summary["static"]["temporal"]["adjacent_fixed_mean"]
        ),
        "dynamic_adjacent": float(
            summary["dynamic"]["temporal"]["adjacent_fixed_mean"]
        ),
        "static_frame0_final": float(
            summary["static"]["temporal"]["frame0_final_fixed"]
        ),
        "dynamic_frame0_final": float(
            summary["dynamic"]["temporal"]["frame0_final_fixed"]
        ),
        "static_dc_ratio": float(
            summary["static"]["frequency"]["dc_energy_ratio"]
        ),
        "dynamic_dc_ratio": float(
            summary["dynamic"]["frequency"]["dc_energy_ratio"]
        ),
        "full_mse": full_mse,
        "dynamic_frozen_mse": dynamic_frozen_mse,
        "static_frozen_mse": static_frozen_mse,
        "dynamic_freeze_delta": dynamic_frozen_mse - full_mse,
        "static_freeze_delta": static_frozen_mse - full_mse,
        "full_temporal_delta_rms": float(
            summary["decoder"]["full"]["temporal_delta_rms"]
        ),
        "dynamic_frozen_temporal_delta_rms": float(
            summary["decoder"]["dynamic_frozen"]["temporal_delta_rms"]
        ),
        "static_frozen_temporal_delta_rms": float(
            summary["decoder"]["static_frozen"]["temporal_delta_rms"]
        ),
    }


def read_case_inputs(
    case: dict[str, Any],
    mode: str,
    combined: dict[str, Any],
    outputs_root: Path,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_key = "crop_dir" if mode == "crop" else "padding_dir"
    frame_root = (
        outputs_root
        / combined[source_key]
        / "cases"
        / case["case_id"]
        / "original"
    )
    rgb = read_frame_sequence(frame_root, int(case["frames"]))
    video = normalize_rgb_frames(rgb)
    boxes = boxes_from_metadata(
        case[mode]["amg"],
        num_slots,
        len(rgb),
        rgb.shape[1],
        rgb.shape[2],
    )
    return video, boxes


def analyze_all(
    args: argparse.Namespace,
    combined: dict[str, Any],
    cases: list[dict[str, Any]],
    checkpoints: list[Path],
    output_root: Path,
) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = getattr(torch, args.amp_dtype)

    cfg, model = build_model(args.config.resolve(), checkpoints[0], device)
    num_slots = int(cfg.max_num)
    input_cache = {}
    for case_index, case in enumerate(cases, start=1):
        for mode in ("crop",):
            input_cache[(case["case_id"], mode)] = read_case_inputs(
                case, mode, combined, args.outputs_root, num_slots
            )
        print(
            f"[preload] case={case_index}/{len(cases)} {case['case_id']}",
            flush=True,
        )
    for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
        if checkpoint_index > 1:
            load_checkpoint(model, checkpoint)
            model.eval()
        step = checkpoint_step(checkpoint)
        label = f"movi_step_{step:06d}"
        print(
            f"[checkpoint] {checkpoint_index}/{len(checkpoints)} "
            f"step={step} path={checkpoint}",
            flush=True,
        )
        for case_index, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            for mode in ("crop",):
                output_path = (
                    output_root / "cases" / case_id / mode / f"{label}.png"
                )
                metrics_path = output_path.with_suffix(".json")
                if (
                    output_path.is_file()
                    and metrics_path.is_file()
                    and not args.force
                ):
                    print(
                        f"[cached] checkpoint={checkpoint_index}/{len(checkpoints)} "
                        f"case={case_index}/{len(cases)} {case_id} {mode}",
                        flush=True,
                    )
                    continue
                video, boxes = input_cache[(case_id, mode)]
                analyze_case(
                    model,
                    video,
                    boxes,
                    device,
                    amp_dtype,
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "label": label,
                        "checkpoint": str(checkpoint),
                        "step": step,
                    },
                    output_path,
                )
                print(
                    f"[analyze] checkpoint={checkpoint_index}/{len(checkpoints)} "
                    f"case={case_index}/{len(cases)} {case_id} {mode}",
                    flush=True,
                )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def collect_records(
    cases: list[dict[str, Any]],
    checkpoints: list[Path],
    output_root: Path,
) -> list[dict[str, Any]]:
    records = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        label = f"movi_step_{step:06d}"
        for case in cases:
            case_id = case["case_id"]
            for mode in ("crop",):
                chart = output_root / "cases" / case_id / mode / f"{label}.png"
                metrics_path = chart.with_suffix(".json")
                if not chart.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(
                        f"Missing analysis artifact: {metrics_path}"
                    )
                summary = json.loads(metrics_path.read_text())
                records.append(
                    {
                        "series_key": label,
                        "display_label": f"DINOv3 step-{step:06d}",
                        "step": step,
                        "label": label,
                        "checkpoint": str(checkpoint),
                        "case_id": case_id,
                        "mode": mode,
                        "chart": chart.relative_to(output_root).as_posix(),
                        "metrics": summary_metrics(summary),
                        "partition": summary["partition"],
                    }
                )
    official_root = output_root.parent / "decoder_static_dynamic"
    for case in cases:
        case_id = case["case_id"]
        chart = (
            official_root
            / "cases"
            / case_id
            / "crop"
            / f"{OFFICIAL_LABEL}.png"
        )
        metrics_path = chart.with_suffix(".json")
        if not chart.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Missing official artifact: {metrics_path}")
        summary = json.loads(metrics_path.read_text())
        records.append(
            {
                "series_key": OFFICIAL_LABEL,
                "display_label": OFFICIAL_DISPLAY,
                "step": None,
                "label": OFFICIAL_LABEL,
                "checkpoint": str(OFFICIAL_CHECKPOINT),
                "case_id": case_id,
                "mode": "crop",
                "chart": (
                    Path("..")
                    / "decoder_static_dynamic"
                    / "cases"
                    / case_id
                    / "crop"
                    / f"{OFFICIAL_LABEL}.png"
                ).as_posix(),
                "metrics": summary_metrics(summary),
                "partition": summary["partition"],
            }
        )
    return records


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = [OFFICIAL_LABEL] + [
        f"movi_step_{step:06d}"
        for step in sorted(
            {
                int(record["step"])
                for record in records
                if record["step"] is not None
            }
        )
    ]
    for series_key in keys:
        selected = [
            record
            for record in records
            if record["series_key"] == series_key
        ]
        row: dict[str, Any] = {
            "series_key": series_key,
            "label": selected[0]["display_label"],
            "step": selected[0]["step"],
            "cases": len(selected),
            "static_dim": int(selected[0]["partition"]["static_dim"]),
            "dynamic_dim": int(selected[0]["partition"]["dynamic_dim"]),
        }
        for field in METRIC_FIELDS:
            values = np.asarray(
                [record["metrics"][field] for record in selected],
                dtype=np.float64,
            )
            row[field] = float(values.mean())
            row[f"{field}_std"] = float(values.std())
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_trends(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=150)
    colors = {"static": "#15803d", "dynamic": "#dc2626"}
    official = next(row for row in rows if row["series_key"] == OFFICIAL_LABEL)
    selected = sorted(
        (row for row in rows if row["step"] is not None),
        key=lambda row: row["step"],
    )
    steps = [row["step"] for row in selected]

    def reference(axis: Any, field: str, color: str, label: str) -> None:
        axis.axhline(
            official[field],
            linestyle=":",
            linewidth=1.7,
            color=color,
            alpha=0.9,
            label=f"official {label}",
        )

    for partition in ("static", "dynamic"):
        axes[0, 0].plot(
            steps,
            [row[f"{partition}_adjacent"] for row in selected],
            color=colors[partition],
            label=f"DINOv3 {partition}",
        )
        reference(
            axes[0, 0],
            f"{partition}_adjacent",
            colors[partition],
            partition,
        )
        axes[0, 1].plot(
            steps,
            [row[f"{partition}_frame0_final"] for row in selected],
            color=colors[partition],
            label=f"DINOv3 {partition}",
        )
        reference(
            axes[0, 1],
            f"{partition}_frame0_final",
            colors[partition],
            partition,
        )
        axes[1, 0].plot(
            steps,
            [row[f"{partition}_dc_ratio"] for row in selected],
            color=colors[partition],
            label=f"DINOv3 {partition}",
        )
        reference(
            axes[1, 0],
            f"{partition}_dc_ratio",
            colors[partition],
            partition,
        )

    axes[1, 1].plot(
        steps,
        [row["full_mse"] for row in selected],
        color="#2563eb",
        label="DINOv3 full",
    )
    reference(axes[1, 1], "full_mse", "#2563eb", "full")

    for field, color, label in (
        ("dynamic_freeze_delta", "#dc2626", "freeze dynamic"),
        ("static_freeze_delta", "#15803d", "freeze static"),
    ):
        axes[2, 0].plot(
            steps,
            [row[field] for row in selected],
            color=color,
            label=f"DINOv3 {label}",
        )
        reference(axes[2, 0], field, color, label)

    for field, color, label in (
        ("full_temporal_delta_rms", "#2563eb", "full"),
        (
            "dynamic_frozen_temporal_delta_rms",
            "#dc2626",
            "dynamic frozen",
        ),
        (
            "static_frozen_temporal_delta_rms",
            "#15803d",
            "static frozen",
        ),
    ):
        axes[2, 1].plot(
            steps,
            [row[field] for row in selected],
            color=color,
            label=f"DINOv3 {label}",
        )
        reference(axes[2, 1], field, color, label)

    titles = (
        "Adjacent projected-memory cosine",
        "Projected-memory similarity: frame 0 to final",
        "Projected-memory DC energy ratio",
        "Full all-mask decoder MSE",
        "MSE increase when a partition is frozen",
        "Decoded temporal delta RMS",
    )
    ylabels = (
        "cosine similarity",
        "cosine similarity",
        "energy ratio",
        "MSE",
        "delta MSE",
        "RMS",
    )
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("optimizer step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle(
        "MOVi-C decoder static/dynamic evolution (center crop only)\n"
        "solid=DINOv3 checkpoints, dotted=official DINOv2 MOVi-C 42-0035"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_html(output_root: Path, metadata: dict[str, Any]) -> None:
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOVi-C decoder checkpoint evolution</title>
<style>
:root{--bg:#f4f6f8;--ink:#17202a;--muted:#65717e;--line:#cbd2da}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{background:#17202a;color:#fff;padding:18px 24px}header h1{font-size:22px;margin:0 0 5px}header p{margin:0;color:#cbd5e1}
main{max-width:1500px;margin:auto;padding:20px 24px 40px}h2{font-size:17px;margin:25px 0 10px}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));border:1px solid var(--line);background:#fff}
.metric{padding:14px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric b{display:block;font-size:20px}.metric span,.note{color:var(--muted)}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0}select{padding:8px;border:1px solid #98a3af;background:#fff}
img{display:block;width:100%;border:1px solid var(--line);background:#fff}
table{width:100%;border-collapse:collapse;background:#fff;font-variant-numeric:tabular-nums}th,td{padding:7px 9px;border:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
a{color:#93c5fd}@media(max-width:800px){.metrics{grid-template-columns:1fr 1fr}.controls{grid-template-columns:1fr}}
</style></head><body>
<header><h1>MOVi-C Decoder Static / Dynamic Evolution</h1><p>Center crop only: official DINOv2 MOVi-C 42-0035 and DINOv3 step-016000 through step-038000. <a href="../">Back to comparison viewer</a></p></header>
<main><div class="metrics">
 <div class="metric"><span>Model states</span><b id="series-count">-</b></div>
 <div class="metric"><span>Cases</span><b id="case-count">-</b></div>
 <div class="metric"><span>Analyses</span><b id="analysis-count">-</b></div>
 <div class="metric"><span>Decoder splits</span><b>288+96 / 768+256</b></div>
</div>
<h2>Checkpoint trends</h2><img src="assets/checkpoint_trends.png" alt="checkpoint trends">
<p class="note">Solid lines are DINOv3 checkpoints; dotted lines are the official DINOv2 MOVi-C reference. Freeze ablations run the full four-layer decoder with one partition replaced by its per-slot temporal mean.</p>
<h2>Case analysis</h2><div class="controls">
 <select id="case-select"></select><select id="series-select"></select>
</div><p id="detail" class="note"></p><img id="chart" alt="decoder static dynamic analysis">
<h2>Model-state means</h2><table><thead><tr><th>Model state</th><th>Split</th><th>Static adjacent</th><th>Dynamic adjacent</th><th>Static frame0-final</th><th>Dynamic frame0-final</th><th>Full MSE</th><th>Freeze dynamic ΔMSE</th><th>Freeze static ΔMSE</th></tr></thead><tbody id="summary-body"></tbody></table>
</main><script>
let DATA=null;const caseSelect=document.getElementById("case-select");const seriesSelect=document.getElementById("series-select");
const chart=document.getElementById("chart");const detail=document.getElementById("detail");
function option(value,label){const node=document.createElement("option");node.value=value;node.textContent=label;return node}
function key(seriesKey,caseId){return `${seriesKey}|${caseId}`}
function update(){const item=DATA.record_index[key(seriesSelect.value,caseSelect.value)];chart.src=item.chart;const m=item.metrics;detail.textContent=`${item.display_label} | ${item.case_id} | split ${item.partition.static_dim}+${item.partition.dynamic_dim} | static adj ${m.static_adjacent.toFixed(5)} | dynamic adj ${m.dynamic_adjacent.toFixed(5)} | full MSE ${m.full_mse.toFixed(2)}`}
fetch("metadata.json").then(r=>r.json()).then(data=>{DATA=data;document.getElementById("series-count").textContent=data.series.length;document.getElementById("case-count").textContent=data.cases.length;document.getElementById("analysis-count").textContent=data.records.length;
data.cases.forEach(c=>caseSelect.appendChild(option(c.case_id,c.case_id)));data.series.forEach(s=>seriesSelect.appendChild(option(s.series_key,s.label)));
data.summary.forEach(r=>{const tr=document.createElement("tr");tr.innerHTML=`<td>${r.label}</td><td>${r.static_dim}+${r.dynamic_dim}</td><td>${r.static_adjacent.toFixed(5)}</td><td>${r.dynamic_adjacent.toFixed(5)}</td><td>${r.static_frame0_final.toFixed(5)}</td><td>${r.dynamic_frame0_final.toFixed(5)}</td><td>${r.full_mse.toFixed(2)}</td><td>${r.dynamic_freeze_delta.toFixed(2)}</td><td>${r.static_freeze_delta.toFixed(2)}</td>`;document.getElementById("summary-body").appendChild(tr)});update()});
caseSelect.addEventListener("change",update);seriesSelect.addEventListener("change",update);
</script></body></html>"""
    (output_root / "index.html").write_text(page)


def write_readme(output_root: Path, metadata: dict[str, Any]) -> None:
    text = f"""# MOVi-C decoder static/dynamic checkpoint evolution

- Model states: {len(metadata['series'])} (official DINOv2 plus
  {len(metadata['steps'])} DINOv3 checkpoints, `step-{metadata['steps'][0]:06d}` to
  `step-{metadata['steps'][-1]:06d}`)
- Cases: {len(metadata['cases'])}
- Preprocessing: center crop only
- Total analyses: {len(metadata['records'])}

The implementation imports `analyze_case` from
`analyze_decoder_static_dynamic_viewer.py`, so the projected-memory metrics,
frequency analysis, all-mask decoding and static/dynamic temporal-mean freeze
ablations are identical to the existing comparison viewer. The official
DINOv2 MOVi-C results reuse the exact existing 22 center-crop artifacts from
the parent comparison; DINOv3 uses the 23 MOVi-C transfer checkpoints.

Artifacts:
- `index.html`: interactive model-state/case viewer
- `metadata.json`: complete records and model-state summaries
- `summary.csv`: model-state mean metrics
- `assets/checkpoint_trends.png`: aggregate evolution curves
- `cases/`: one six-panel PNG and JSON per DINOv3 checkpoint/case
"""
    (output_root / "README.md").write_text(text)


def add_parent_link(viewer_dir: Path) -> None:
    index = viewer_dir / "index.html"
    if not index.is_file():
        return
    marker = "<!-- ALL_MOVIC_DECODER_CHECKPOINTS_LINK -->"
    text = index.read_text()
    if marker in text:
        return
    link = (
        marker
        + "<p style='margin:8px 0'><a href='"
        + OUTPUT_NAME
        + "/'>All MOVi-C decoder checkpoint evolution</a></p>"
    )
    text = text.replace("<main>", "<main>" + link, 1)
    index.write_text(text)


def build_report(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    checkpoints: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    records = collect_records(cases, checkpoints, output_root)
    summary = aggregate_records(records)
    record_index = {
        f"{record['series_key']}|{record['case_id']}": record
        for record in records
    }
    series = [
        {
            "series_key": row["series_key"],
            "label": row["label"],
            "step": row["step"],
            "static_dim": row["static_dim"],
            "dynamic_dim": row["dynamic_dim"],
        }
        for row in summary
    ]
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Exact analyze_case implementation from "
            "analyze_decoder_static_dynamic_viewer.py, using center-crop "
            "inputs only. Official DINOv2 project2 is split 288 static + 96 "
            "dynamic; DINOv3 project2 is split 768 static + 256 dynamic. "
            "All-mask ablations freeze one partition to its per-slot temporal "
            "mean before the full decoder."
        ),
        "official_checkpoint": str(OFFICIAL_CHECKPOINT),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "config": str(args.config.resolve()),
        "steps": [checkpoint_step(path) for path in checkpoints],
        "series": series,
        "cases": [
            {"case_id": case["case_id"], "frames": int(case["frames"])}
            for case in cases
        ],
        "records": records,
        "record_index": record_index,
        "summary": summary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, separators=(",", ":")) + "\n"
    )
    write_csv(output_root / "summary.csv", summary)
    plot_trends(summary, output_root / "assets/checkpoint_trends.png")
    write_html(output_root, metadata)
    write_readme(output_root, metadata)
    add_parent_link(args.viewer_dir)
    return metadata


def main() -> None:
    args = parse_args()
    validate(args)
    args.viewer_dir = args.viewer_dir.resolve()
    args.outputs_root = args.outputs_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.config = args.config.resolve()
    combined = json.loads(
        (args.viewer_dir / "combined_metadata.json").read_text()
    )
    cases = combined["cases"]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    checkpoints = discover_checkpoints(args)
    output_root = args.viewer_dir / OUTPUT_NAME
    print(
        f"[setup] checkpoints={len(checkpoints)} cases={len(cases)} "
        f"dinov3_analyses={len(checkpoints) * len(cases)} "
        f"visible_analyses={(len(checkpoints) + 1) * len(cases)}",
        flush=True,
    )
    if args.stage in ("analyze", "all"):
        analyze_all(args, combined, cases, checkpoints, output_root)
    if args.stage in ("report", "all"):
        metadata = build_report(args, cases, checkpoints, output_root)
        print(
            f"[report] records={len(metadata['records'])} "
            f"path={output_root / 'index.html'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
