#!/usr/bin/env python3
"""Build a focused viewer for fixed-slot temporal change."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

from analyze_decoder_static_dynamic_viewer import (  # noqa: E402
    project_slots,
    run_xssc,
)
from analyze_slot_temporal_similarity_viewer import (  # noqa: E402
    DEFAULT_MOVIC_CKPT_DIR,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_VIEWER_DIR,
    boxes_from_metadata,
    build_model,
    latest_checkpoint,
    load_specs,
    normalize_rgb_frames,
    read_frame_sequence,
)


REPRESENTATIONS = ("xssc", "static", "dynamic")
REPRESENTATION_LABELS = {
    "xssc": "xSSC slotz",
    "static": "Decoder static",
    "dynamic": "Decoder dynamic",
}
REPRESENTATION_COLORS = {
    "xssc": "#2563eb",
    "static": "#15803d",
    "dynamic": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--latest-movic-checkpoint", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def fixed_slot_metrics(features: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    values = torch.from_numpy(features).float()
    normalized = F.normalize(values, dim=-1)
    time_matrix = torch.einsum(
        "tsc,usc->tu", normalized, normalized
    ) / normalized.shape[1]
    time_matrix = time_matrix.numpy()
    frame0_curve = time_matrix[0]
    adjacent = np.diag(time_matrix, k=1)
    off_diagonal = time_matrix[~np.eye(time_matrix.shape[0], dtype=bool)]
    summary = {
        "frames": int(features.shape[0]),
        "slots": int(features.shape[1]),
        "dim": int(features.shape[2]),
        "adjacent_cosine": float(adjacent.mean()),
        "frame0_final_cosine": float(frame0_curve[-1]),
        "all_pairs_cosine": float(off_diagonal.mean()),
        "adjacent_change": float(1.0 - adjacent.mean()),
        "frame0_final_change": float(1.0 - frame0_curve[-1]),
        "all_pairs_change": float(1.0 - off_diagonal.mean()),
    }
    arrays = {
        "time_matrix": time_matrix.astype(np.float32),
        "frame0_curve": frame0_curve.astype(np.float32),
    }
    return summary, arrays


def resample_curve(values: np.ndarray, size: int = 100) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, size)
    return np.interp(target, source, values).astype(np.float32)


def resample_matrix(values: np.ndarray, size: int = 64) -> np.ndarray:
    tensor = torch.from_numpy(values)[None, None].float()
    resized = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=True,
    )
    return resized[0, 0].numpy()


def plot_focused(
    arrays: dict[str, dict[str, np.ndarray]],
    summary: dict[str, dict],
    output_path: Path,
    title: str,
    aggregate: bool,
) -> None:
    fig = plt.figure(figsize=(13.2, 7.3), dpi=145)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 0.72),
        hspace=0.34,
        wspace=0.20,
    )
    heat_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    curve_axis = fig.add_subplot(grid[1, :])
    matrix_min = min(
        float(arrays[key]["time_matrix"].min()) for key in REPRESENTATIONS
    )
    vmin = min(0.9, np.floor(matrix_min * 20.0) / 20.0)
    vmin = max(-1.0, vmin)
    heat = None
    for axis, key in zip(heat_axes, REPRESENTATIONS):
        matrix = arrays[key]["time_matrix"]
        heat = axis.imshow(
            matrix,
            cmap="viridis",
            vmin=vmin,
            vmax=1.0,
            interpolation="nearest",
            origin="lower",
            aspect="equal",
            extent=(0.0, 1.0, 0.0, 1.0) if aggregate else None,
        )
        axis.set_title(
            f"{REPRESENTATION_LABELS[key]}\n"
            f"all-pairs change {summary[key]['all_pairs_change']:.4f}",
            color=REPRESENTATION_COLORS[key],
        )
        axis.set_xlabel("normalized time" if aggregate else "frame")
        axis.set_ylabel("normalized time" if aggregate else "frame")
    if heat is not None:
        fig.colorbar(
            heat,
            ax=heat_axes,
            label="same-slot cosine",
            fraction=0.024,
            pad=0.02,
        )

    for key in REPRESENTATIONS:
        curve = arrays[key]["frame0_curve"]
        x_axis = (
            np.linspace(0.0, 1.0, len(curve))
            if aggregate
            else np.arange(len(curve))
        )
        curve_axis.plot(
            x_axis,
            curve,
            color=REPRESENTATION_COLORS[key],
            label=REPRESENTATION_LABELS[key],
            linewidth=2.0,
        )
    curve_min = min(
        float(arrays[key]["frame0_curve"].min()) for key in REPRESENTATIONS
    )
    curve_axis.set_ylim(max(-1.0, curve_min - 0.05), 1.015)
    curve_axis.set_title("Same fixed slot: cosine similarity to frame 0")
    curve_axis.set_xlabel("normalized time" if aggregate else "frame")
    curve_axis.set_ylabel("cosine similarity")
    curve_axis.grid(alpha=0.20)
    curve_axis.legend(loc="lower left", ncol=3)
    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_case_result(
    output_path: Path,
    arrays: dict[str, dict[str, np.ndarray]],
    summary: dict,
) -> None:
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        **{
            f"{key}_{array_name}": value
            for key, representation in arrays.items()
            for array_name, value in representation.items()
        },
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def load_case_result(
    output_path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    archive = np.load(output_path.with_suffix(".npz"))
    arrays = {
        key: {
            "time_matrix": archive[f"{key}_time_matrix"],
            "frame0_curve": archive[f"{key}_frame0_curve"],
        }
        for key in REPRESENTATIONS
    }
    summary = json.loads(
        output_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    return arrays, summary


def analyze_video(
    model,
    video: torch.Tensor,
    boxes: torch.Tensor | None,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    feature, slots = run_xssc(model, video, boxes, device, amp_dtype)
    projected = project_slots(model.m.decode, slots, amp_dtype)
    decoder_dim = int(projected.shape[-1])
    dynamic_dim = int(decoder_dim * float(model.m.decode.rd))
    static_dim = decoder_dim - dynamic_dim
    representations = {
        "xssc": slots[0].detach().float().cpu().numpy(),
        "static": projected[0, :, :, :static_dim].detach().float().cpu().numpy(),
        "dynamic": projected[0, :, :, static_dim:].detach().float().cpu().numpy(),
    }
    arrays = {}
    metrics = {}
    for key, values in representations.items():
        metrics[key], arrays[key] = fixed_slot_metrics(values)
    del feature, slots, projected
    return arrays, {
        "partition": {
            "decoder_dim": decoder_dim,
            "static_dim": static_dim,
            "dynamic_dim": dynamic_dim,
        },
        "representations": metrics,
    }


def average_results(
    results: list[tuple[dict[str, dict[str, np.ndarray]], dict]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    arrays = {}
    metrics = {}
    for key in REPRESENTATIONS:
        matrices = np.stack(
            [
                resample_matrix(case_arrays[key]["time_matrix"])
                for case_arrays, _ in results
            ]
        )
        curves = np.stack(
            [
                resample_curve(case_arrays[key]["frame0_curve"])
                for case_arrays, _ in results
            ]
        )
        arrays[key] = {
            "time_matrix": matrices.mean(axis=0),
            "frame0_curve": curves.mean(axis=0),
        }
        source_metrics = [
            case_summary["representations"][key] for _, case_summary in results
        ]
        metrics[key] = {
            field: float(np.mean([item[field] for item in source_metrics]))
            for field in (
                "adjacent_cosine",
                "frame0_final_cosine",
                "all_pairs_cosine",
                "adjacent_change",
                "frame0_final_change",
                "all_pairs_change",
            )
        }
        metrics[key]["dim"] = int(source_metrics[0]["dim"])
    partition = results[0][1]["partition"]
    return arrays, {
        "case_count": len(results),
        "partition": partition,
        "representations": metrics,
    }


def relative_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def build_html(metadata: dict) -> str:
    data = json.dumps(metadata, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC Same-Slot Temporal Change</title>
<style>
:root{{--ink:#171717;--muted:#666;--line:#d4d4d4;--paper:#fff;--soft:#f5f5f4}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:3;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}}
.bar{{max-width:1500px;margin:auto;padding:14px 20px;display:flex;align-items:end;gap:18px;flex-wrap:wrap}}
h1{{font-size:20px;margin:0 auto 1px 0;letter-spacing:0}}
label{{display:grid;gap:5px;color:var(--muted);font-size:12px;font-weight:650}}
select{{min-width:190px;height:36px;border:1px solid #aaa;border-radius:5px;background:#fff;padding:0 30px 0 10px;color:var(--ink)}}
main{{max-width:1500px;margin:auto;padding:20px}}
.status{{display:flex;align-items:center;gap:10px;margin-bottom:12px;color:var(--muted)}}
.dot{{width:10px;height:10px;border-radius:50%}}
.figure{{margin:0;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#fff}}
.figure img{{display:block;width:100%;height:auto}}
table{{width:100%;margin-top:18px;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left}}
thead th{{background:var(--soft);color:#444;font-size:12px}}
.rep{{display:inline-flex;align-items:center;gap:8px;font-weight:700}}
.swatch{{width:12px;height:12px;border-radius:2px}}
.method{{margin-top:12px;color:var(--muted);font-size:12px}}
@media(max-width:760px){{.bar{{align-items:stretch}}h1{{width:100%}}label,select{{width:100%}}main{{padding:12px}}th,td{{padding:8px 6px;font-size:12px}}}}
</style>
</head>
<body>
<header><div class="bar">
  <h1>xSSC Same-Slot Temporal Change</h1>
  <label>Checkpoint<select id="model"></select></label>
  <label>Preprocessing<select id="mode"><option value="crop">Center crop</option><option value="padding">Resize + padding</option></select></label>
  <label>Cases<select id="case"></select></label>
</div></header>
<main>
  <div class="status"><span id="dot" class="dot"></span><span id="status"></span></div>
  <figure class="figure"><img id="chart" alt="same-slot temporal similarity"></figure>
  <table>
    <thead><tr><th>Representation</th><th>Dimension</th><th>Adjacent cosine</th><th>Frame 0 to final cosine</th><th>All-pairs change (1-cos)</th></tr></thead>
    <tbody id="metrics"></tbody>
  </table>
  <div class="method">Fixed slot IDs, without Hungarian rematching. The average view is the arithmetic mean over 22 cases after normalizing each video timeline.</div>
</main>
<script>
const DATA={data};
const COLORS={{xssc:'#2563eb',static:'#15803d',dynamic:'#dc2626'}};
const LABELS={{xssc:'xSSC slotz',static:'Decoder static',dynamic:'Decoder dynamic'}};
const model=document.getElementById('model');
const mode=document.getElementById('mode');
const caseSelect=document.getElementById('case');
const chart=document.getElementById('chart');
const metrics=document.getElementById('metrics');
const status=document.getElementById('status');
const dot=document.getElementById('dot');
DATA.models.forEach(item=>{{const option=document.createElement('option');option.value=item.label;option.textContent=item.label;model.appendChild(option);}});
DATA.cases.forEach(item=>{{const option=document.createElement('option');option.value=item.id;option.textContent=item.label;caseSelect.appendChild(option);}});
model.value=DATA.default_model;
function render(){{
  const entry=DATA.entries[caseSelect.value][mode.value][model.value];
  chart.src=entry.chart;
  chart.alt=`${{model.value}} ${{mode.value}} ${{caseSelect.value}}`;
  const reps=entry.metrics.representations;
  metrics.replaceChildren(...['xssc','static','dynamic'].map(key=>{{
    const row=document.createElement('tr');
    row.innerHTML=`<td><span class="rep"><span class="swatch" style="background:${{COLORS[key]}}"></span>${{LABELS[key]}}</span></td><td>${{reps[key].dim}}</td><td>${{reps[key].adjacent_cosine.toFixed(4)}}</td><td>${{reps[key].frame0_final_cosine.toFixed(4)}}</td><td>${{reps[key].all_pairs_change.toFixed(4)}}</td>`;
    return row;
  }}));
  const count=entry.metrics.case_count || 1;
  status.textContent=count>1?`${{count}}-case normalized-time average`:`${{entry.metrics.frames}} frames`;
  dot.style.background=COLORS.dynamic;
}}
[model,mode,caseSelect].forEach(element=>element.addEventListener('change',render));
render();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else viewer_dir / "focused_slot_temporal"
    )
    combined = json.loads(
        (viewer_dir / "combined_metadata.json").read_text(encoding="utf-8")
    )
    cases = combined["cases"]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = getattr(torch, args.amp_dtype)
    latest = (
        args.latest_movic_checkpoint.resolve()
        if args.latest_movic_checkpoint is not None
        else latest_checkpoint(DEFAULT_MOVIC_CKPT_DIR)
    )
    specs = load_specs(viewer_dir, outputs_root, latest)
    entries: dict[str, dict] = {
        "average": {"crop": {}, "padding": {}},
        **{
            case["case_id"]: {"crop": {}, "padding": {}}
            for case in cases
        },
    }
    grouped_results: dict[
        tuple[str, str],
        list[tuple[dict[str, dict[str, np.ndarray]], dict]],
    ] = {}

    for model_index, spec in enumerate(specs, start=1):
        cfg, model_instance = build_model(
            spec["config"], spec["checkpoint"], device
        )
        num_slots = int(cfg.max_num)
        print(
            f"[model] {model_index}/{len(specs)} {spec['label']}",
            flush=True,
        )
        for case_index, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            for mode, source_key in (
                ("crop", "crop_dir"),
                ("padding", "padding_dir"),
            ):
                output_path = (
                    output_dir
                    / "cases"
                    / case_id
                    / mode
                    / f"{spec['label']}.png"
                )
                cache_exists = all(
                    output_path.with_suffix(suffix).is_file()
                    for suffix in (".png", ".json", ".npz")
                )
                if cache_exists and not args.force:
                    arrays, summary = load_case_result(output_path)
                else:
                    frame_root = (
                        outputs_root
                        / combined[source_key]
                        / "cases"
                        / case_id
                        / "original"
                    )
                    rgb = read_frame_sequence(
                        frame_root, int(case["frames"])
                    )
                    video = normalize_rgb_frames(rgb)
                    boxes = None
                    if spec["conditioned"]:
                        boxes = boxes_from_metadata(
                            case[mode]["amg"],
                            num_slots,
                            len(rgb),
                            rgb.shape[1],
                            rgb.shape[2],
                        )
                    arrays, analysis = analyze_video(
                        model_instance,
                        video,
                        boxes,
                        device,
                        amp_dtype,
                    )
                    summary = {
                        "case_id": case_id,
                        "mode": mode,
                        "label": spec["label"],
                        "checkpoint": str(spec["checkpoint"]),
                        "frames": int(case["frames"]),
                        **analysis,
                    }
                    plot_focused(
                        arrays,
                        summary["representations"],
                        output_path,
                        f"{case_id} | {mode} | {spec['label']}",
                        aggregate=False,
                    )
                    save_case_result(output_path, arrays, summary)
                grouped_results.setdefault((spec["label"], mode), []).append(
                    (arrays, summary)
                )
                entries[case_id][mode][spec["label"]] = {
                    "chart": relative_path(output_path, output_dir),
                    "metrics": summary,
                }
                print(
                    f"[case] model={model_index}/{len(specs)} "
                    f"case={case_index}/{len(cases)} {mode}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del model_instance
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for spec in specs:
        for mode in ("crop", "padding"):
            results = grouped_results[(spec["label"], mode)]
            arrays, summary = average_results(results)
            output_path = (
                output_dir / "averages" / mode / f"{spec['label']}.png"
            )
            plot_focused(
                arrays,
                summary["representations"],
                output_path,
                f"{len(cases)}-case average | {mode} | {spec['label']}",
                aggregate=True,
            )
            summary_path = output_path.with_suffix(".json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            entries["average"][mode][spec["label"]] = {
                "chart": relative_path(output_path, output_dir),
                "metrics": summary,
            }

    default_model = next(
        (
            spec["label"]
            for spec in reversed(specs)
            if "movi_current" in spec["label"]
        ),
        specs[-1]["label"],
    )
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Fixed-ID same-slot cosine over time for raw slotz and the trained "
            "decoder project2 static/dynamic channel partitions."
        ),
        "default_model": default_model,
        "models": [
            {
                "label": spec["label"],
                "checkpoint": str(spec["checkpoint"]),
                "config": str(spec["config"]),
            }
            for spec in specs
        ],
        "cases": [
            {"id": "average", "label": f"{len(cases)}-case average"},
            *[
                {
                    "id": case["case_id"],
                    "label": f"{index:02d} | {case['case_id']}",
                }
                for index, case in enumerate(cases, start=1)
            ],
        ],
        "entries": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        build_html(metadata),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# xSSC same-slot temporal change viewer\n\n"
        "The viewer compares fixed slot IDs across frames without Hungarian "
        "rematching. It displays raw xSSC `slotz`, the leading static channels "
        "of `decoder.project2(slotz)`, and the trailing dynamic channels. The "
        "22-case average resamples every timeline to normalized time before "
        "averaging matrices and curves.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index": str(output_dir / "index.html"),
                "models": len(specs),
                "cases": len(cases),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
