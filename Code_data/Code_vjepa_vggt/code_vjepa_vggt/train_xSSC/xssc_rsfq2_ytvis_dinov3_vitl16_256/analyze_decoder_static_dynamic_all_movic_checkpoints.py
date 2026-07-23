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
        for mode in ("crop", "padding"):
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
            for mode in ("crop", "padding"):
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
            for mode in ("crop", "padding"):
                chart = output_root / "cases" / case_id / mode / f"{label}.png"
                metrics_path = chart.with_suffix(".json")
                if not chart.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(
                        f"Missing analysis artifact: {metrics_path}"
                    )
                summary = json.loads(metrics_path.read_text())
                records.append(
                    {
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
    return records


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted({(record["step"], record["mode"]) for record in records})
    for step, mode in keys:
        selected = [
            record
            for record in records
            if record["step"] == step and record["mode"] == mode
        ]
        row: dict[str, Any] = {
            "step": step,
            "mode": mode,
            "cases": len(selected),
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
    styles = {"crop": "-", "padding": "--"}
    for mode in ("crop", "padding"):
        selected = sorted(
            (row for row in rows if row["mode"] == mode),
            key=lambda row: row["step"],
        )
        steps = [row["step"] for row in selected]
        for partition in ("static", "dynamic"):
            axes[0, 0].plot(
                steps,
                [row[f"{partition}_adjacent"] for row in selected],
                styles[mode],
                color=colors[partition],
                label=f"{partition} {mode}",
            )
            axes[0, 1].plot(
                steps,
                [row[f"{partition}_frame0_final"] for row in selected],
                styles[mode],
                color=colors[partition],
                label=f"{partition} {mode}",
            )
            axes[1, 0].plot(
                steps,
                [row[f"{partition}_dc_ratio"] for row in selected],
                styles[mode],
                color=colors[partition],
                label=f"{partition} {mode}",
            )
        axes[1, 1].plot(
            steps,
            [row["full_mse"] for row in selected],
            styles[mode],
            label=f"full {mode}",
        )
        axes[2, 0].plot(
            steps,
            [row["dynamic_freeze_delta"] for row in selected],
            styles[mode],
            color="#dc2626",
            label=f"freeze dynamic {mode}",
        )
        axes[2, 0].plot(
            steps,
            [row["static_freeze_delta"] for row in selected],
            styles[mode],
            color="#15803d",
            label=f"freeze static {mode}",
        )
        axes[2, 1].plot(
            steps,
            [row["full_temporal_delta_rms"] for row in selected],
            styles[mode],
            color="#2563eb",
            label=f"full {mode}",
        )
        axes[2, 1].plot(
            steps,
            [row["dynamic_frozen_temporal_delta_rms"] for row in selected],
            styles[mode],
            color="#dc2626",
            label=f"dynamic frozen {mode}",
        )
        axes[2, 1].plot(
            steps,
            [row["static_frozen_temporal_delta_rms"] for row in selected],
            styles[mode],
            color="#15803d",
            label=f"static frozen {mode}",
        )

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
        "MOVi-C DINOv3 decoder static/dynamic evolution\n"
        "solid=center crop, dashed=padding"
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
.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:10px;margin:12px 0}select,button{padding:8px;border:1px solid #98a3af;background:#fff}
.modes{display:flex}.modes button{min-width:85px}.modes button.active{background:#17202a;color:#fff}
img{display:block;width:100%;border:1px solid var(--line);background:#fff}
table{width:100%;border-collapse:collapse;background:#fff;font-variant-numeric:tabular-nums}th,td{padding:7px 9px;border:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
a{color:#93c5fd}@media(max-width:800px){.metrics{grid-template-columns:1fr 1fr}.controls{grid-template-columns:1fr}.modes{width:100%}.modes button{flex:1}}
</style></head><body>
<header><h1>MOVi-C DINOv3 Decoder Static / Dynamic Evolution</h1><p>All step-016000 through step-038000 checkpoints, using the exact existing six-panel analysis. <a href="../">Back to comparison viewer</a></p></header>
<main><div class="metrics">
 <div class="metric"><span>Checkpoints</span><b id="checkpoint-count">-</b></div>
 <div class="metric"><span>Cases</span><b id="case-count">-</b></div>
 <div class="metric"><span>Analyses</span><b id="analysis-count">-</b></div>
 <div class="metric"><span>Decoder split</span><b>768 + 256</b></div>
</div>
<h2>Checkpoint trends</h2><img src="assets/checkpoint_trends.png" alt="checkpoint trends">
<p class="note">Adjacent/frame-0 similarities and spectra describe the projected decoder memory. Freeze ablations run the full four-layer decoder with one partition replaced by its per-slot temporal mean.</p>
<h2>Case analysis</h2><div class="controls">
 <select id="case-select"></select><select id="step-select"></select>
 <div class="modes"><button id="crop-button" class="active">Crop</button><button id="padding-button">Padding</button></div>
</div><p id="detail" class="note"></p><img id="chart" alt="decoder static dynamic analysis">
<h2>Checkpoint means</h2><table><thead><tr><th>Step</th><th>Mode</th><th>Static adjacent</th><th>Dynamic adjacent</th><th>Static frame0-final</th><th>Dynamic frame0-final</th><th>Full MSE</th><th>Freeze dynamic ΔMSE</th><th>Freeze static ΔMSE</th></tr></thead><tbody id="summary-body"></tbody></table>
</main><script>
let DATA=null;let mode="crop";const caseSelect=document.getElementById("case-select");const stepSelect=document.getElementById("step-select");
const chart=document.getElementById("chart");const detail=document.getElementById("detail");
function option(value,label){const node=document.createElement("option");node.value=value;node.textContent=label;return node}
function key(step,caseId,mode){return `${step}|${caseId}|${mode}`}
function update(){const item=DATA.record_index[key(Number(stepSelect.value),caseSelect.value,mode)];chart.src=item.chart;const m=item.metrics;detail.textContent=`${item.label} | ${item.case_id} | ${mode} | static adj ${m.static_adjacent.toFixed(5)} | dynamic adj ${m.dynamic_adjacent.toFixed(5)} | full MSE ${m.full_mse.toFixed(2)}`}
function setMode(value){mode=value;document.getElementById("crop-button").classList.toggle("active",mode==="crop");document.getElementById("padding-button").classList.toggle("active",mode==="padding");update()}
fetch("metadata.json").then(r=>r.json()).then(data=>{DATA=data;document.getElementById("checkpoint-count").textContent=data.steps.length;document.getElementById("case-count").textContent=data.cases.length;document.getElementById("analysis-count").textContent=data.records.length;
data.cases.forEach(c=>caseSelect.appendChild(option(c.case_id,c.case_id)));data.steps.forEach(s=>stepSelect.appendChild(option(s,`step-${String(s).padStart(6,"0")}`)));
data.summary.forEach(r=>{const tr=document.createElement("tr");tr.innerHTML=`<td>${r.step}</td><td>${r.mode}</td><td>${r.static_adjacent.toFixed(5)}</td><td>${r.dynamic_adjacent.toFixed(5)}</td><td>${r.static_frame0_final.toFixed(5)}</td><td>${r.dynamic_frame0_final.toFixed(5)}</td><td>${r.full_mse.toFixed(2)}</td><td>${r.dynamic_freeze_delta.toFixed(2)}</td><td>${r.static_freeze_delta.toFixed(2)}</td>`;document.getElementById("summary-body").appendChild(tr)});update()});
caseSelect.addEventListener("change",update);stepSelect.addEventListener("change",update);document.getElementById("crop-button").addEventListener("click",()=>setMode("crop"));document.getElementById("padding-button").addEventListener("click",()=>setMode("padding"));
</script></body></html>"""
    (output_root / "index.html").write_text(page)


def write_readme(output_root: Path, metadata: dict[str, Any]) -> None:
    text = f"""# MOVi-C DINOv3 decoder static/dynamic checkpoint evolution

- Checkpoints: {len(metadata['steps'])} (`step-{metadata['steps'][0]:06d}` to
  `step-{metadata['steps'][-1]:06d}`)
- Cases: {len(metadata['cases'])}
- Modes: center crop and padding
- Total analyses: {len(metadata['records'])}

The implementation imports `analyze_case` from
`analyze_decoder_static_dynamic_viewer.py`, so the projected-memory metrics,
frequency analysis, all-mask decoding and static/dynamic temporal-mean freeze
ablations are identical to the existing comparison viewer.

Artifacts:
- `index.html`: interactive checkpoint/case/mode viewer
- `metadata.json`: complete records and checkpoint summaries
- `summary.csv`: checkpoint mean metrics
- `assets/checkpoint_trends.png`: aggregate evolution curves
- `cases/`: one six-panel PNG and JSON per checkpoint/case/mode
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
        f"{record['step']}|{record['case_id']}|{record['mode']}": record
        for record in records
    }
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Exact analyze_case implementation from "
            "analyze_decoder_static_dynamic_viewer.py. Decoder project2 is "
            "split 768 static + 256 dynamic; all-mask ablations freeze one "
            "partition to its per-slot temporal mean before the full decoder."
        ),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "config": str(args.config.resolve()),
        "steps": [checkpoint_step(path) for path in checkpoints],
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
        f"analyses={len(checkpoints) * len(cases) * 2}",
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
