#!/usr/bin/env python3
"""Analyze whether temporal variation is rotated across decoder channels."""

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


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

from analyze_decoder_static_dynamic_viewer import project_slots, run_xssc  # noqa: E402
from analyze_slot_temporal_similarity_viewer import (  # noqa: E402
    DEFAULT_MOVIC_CKPT_DIR,
    DEFAULT_MOVIC_CONFIG,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_VIEWER_DIR,
    boxes_from_metadata,
    build_model,
    checkpoint_step,
    latest_checkpoint,
    normalize_rgb_frames,
    read_frame_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_MOVIC_CONFIG)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def svd_temporal_analysis(values: np.ndarray, dynamic_ratio: float, top_k: int) -> dict:
    """Run SVD on per-slot time-centered decoder vectors.

    values is [T, S, D]. Centering is done per slot, so the SVD focuses on
    within-slot temporal changes rather than persistent slot identity.
    """
    time_steps, num_slots, decoder_dim = values.shape
    dynamic_dim = int(decoder_dim * dynamic_ratio)
    static_dim = decoder_dim - dynamic_dim
    centered = values - values.mean(axis=0, keepdims=True)
    matrix = centered.reshape(time_steps * num_slots, decoder_dim)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    variance = singular_values**2
    variance_ratio = variance / max(float(variance.sum()), np.finfo(np.float64).eps)
    components = vh[:top_k]
    loading_power = components**2
    static_loading = loading_power[:, :static_dim].sum(axis=1)
    dynamic_loading = loading_power[:, static_dim:].sum(axis=1)
    total_temporal_power = (centered**2).sum(axis=(0, 1))
    static_temporal_power = float(total_temporal_power[:static_dim].sum())
    dynamic_temporal_power = float(total_temporal_power[static_dim:].sum())
    total_power = static_temporal_power + dynamic_temporal_power
    scores = matrix @ components[:2].T

    return {
        "summary": {
            "frames": int(time_steps),
            "slots": int(num_slots),
            "decoder_dim": int(decoder_dim),
            "static_dim": int(static_dim),
            "dynamic_dim": int(dynamic_dim),
            "dynamic_ratio": float(dynamic_ratio),
            "top1_variance_ratio": float(variance_ratio[0]),
            "top2_cumulative_variance_ratio": float(variance_ratio[:2].sum()),
            "top4_cumulative_variance_ratio": float(variance_ratio[:4].sum()),
            "top8_cumulative_variance_ratio": float(variance_ratio[:8].sum()),
            "top16_cumulative_variance_ratio": float(
                variance_ratio[: min(16, len(variance_ratio))].sum()
            ),
            "static_temporal_power_ratio": float(static_temporal_power / total_power),
            "dynamic_temporal_power_ratio": float(dynamic_temporal_power / total_power),
            "top1_dynamic_loading_ratio": float(dynamic_loading[0]),
            "top4_dynamic_loading_ratio_mean": float(dynamic_loading[:4].mean()),
            "top8_dynamic_loading_ratio_mean": float(dynamic_loading[:8].mean()),
            "expected_dynamic_ratio_if_unstructured": float(dynamic_dim / decoder_dim),
        },
        "arrays": {
            "variance_ratio": variance_ratio.astype(np.float32),
            "cumulative_variance": np.cumsum(variance_ratio).astype(np.float32),
            "static_loading": static_loading.astype(np.float32),
            "dynamic_loading": dynamic_loading.astype(np.float32),
            "scores_2d": scores.astype(np.float32),
        },
    }


def plot_case(analysis: dict, output_path: Path, title: str) -> None:
    arrays = analysis["arrays"]
    summary = analysis["summary"]
    top_k = len(arrays["dynamic_loading"])
    x = np.arange(1, top_k + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), dpi=145)

    axes[0, 0].bar(x, arrays["variance_ratio"][:top_k], color="#2563eb")
    axes[0, 0].plot(x, arrays["cumulative_variance"][:top_k], color="#dc2626")
    axes[0, 0].set_title("Temporal SVD variance")
    axes[0, 0].set_xlabel("component")
    axes[0, 0].set_ylabel("explained variance")
    axes[0, 0].grid(alpha=0.22)

    axes[0, 1].bar(x, arrays["static_loading"], label="static channels", color="#15803d")
    axes[0, 1].bar(
        x,
        arrays["dynamic_loading"],
        bottom=arrays["static_loading"],
        label="dynamic channels",
        color="#dc2626",
    )
    axes[0, 1].axhline(
        summary["expected_dynamic_ratio_if_unstructured"],
        color="#525252",
        linestyle="--",
        linewidth=1.3,
        label="dynamic dim ratio",
    )
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Component loading split")
    axes[0, 1].set_xlabel("component")
    axes[0, 1].set_ylabel("squared loading ratio")
    axes[0, 1].legend(loc="lower right", fontsize=8)

    scores = arrays["scores_2d"]
    axes[1, 0].scatter(scores[:, 0], scores[:, 1], s=5, alpha=0.45, color="#2563eb")
    axes[1, 0].set_title("Temporal-change samples in PC1/PC2")
    axes[1, 0].set_xlabel("PC1 score")
    axes[1, 0].set_ylabel("PC2 score")
    axes[1, 0].grid(alpha=0.22)

    labels = ["static channels", "dynamic channels"]
    values = [
        summary["static_temporal_power_ratio"],
        summary["dynamic_temporal_power_ratio"],
    ]
    axes[1, 1].bar(labels, values, color=["#15803d", "#dc2626"])
    axes[1, 1].axhline(
        summary["expected_dynamic_ratio_if_unstructured"],
        color="#525252",
        linestyle="--",
        linewidth=1.3,
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Total temporal power by hard split")
    axes[1, 1].tick_params(axis="x", rotation=8)
    axes[1, 1].set_ylabel("ratio")

    fig.suptitle(
        f"{title}\n"
        f"top4 cum {summary['top4_cumulative_variance_ratio']:.3f}, "
        f"dynamic power {summary['dynamic_temporal_power_ratio']:.3f}, "
        f"top4 dynamic loading {summary['top4_dynamic_loading_ratio_mean']:.3f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def average_case_summaries(items: list[dict]) -> dict:
    numeric_keys = [
        key for key, value in items[0].items() if isinstance(value, (int, float))
    ]
    return {
        key: float(np.mean([float(item[key]) for item in items]))
        for key in numeric_keys
    }


def build_html(metadata: dict) -> str:
    data = json.dumps(metadata, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC Decoder PCA/SVD Temporal Analysis</title>
<style>
:root{{--ink:#171717;--muted:#666;--line:#d4d4d4;--paper:#fff;--soft:#f5f5f4}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:3;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}}
.bar{{max-width:1500px;margin:auto;padding:14px 20px;display:flex;align-items:end;gap:18px;flex-wrap:wrap}}
h1{{font-size:20px;margin:0 auto 1px 0;letter-spacing:0}}
label{{display:grid;gap:5px;color:var(--muted);font-size:12px;font-weight:650}}
select{{min-width:220px;height:36px;border:1px solid #aaa;border-radius:5px;background:#fff;padding:0 30px 0 10px;color:var(--ink)}}
main{{max-width:1500px;margin:auto;padding:20px}}
.figure{{margin:0;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#fff}}
.figure img{{display:block;width:100%;height:auto}}
table{{width:100%;margin-top:18px;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left}}
thead th{{background:var(--soft);color:#444;font-size:12px}}
.method{{margin-top:12px;color:var(--muted);font-size:12px}}
@media(max-width:760px){{.bar{{align-items:stretch}}h1{{width:100%}}label,select{{width:100%}}main{{padding:12px}}th,td{{padding:8px 6px;font-size:12px}}}}
</style>
</head>
<body>
<header><div class="bar">
  <h1>xSSC Decoder PCA/SVD Temporal Analysis</h1>
  <label>Preprocessing<select id="mode"><option value="crop">Center crop</option><option value="padding">Resize + padding</option></select></label>
  <label>Cases<select id="case"></select></label>
</div></header>
<main>
  <figure class="figure"><img id="chart" alt="PCA SVD temporal analysis"></figure>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody id="metrics"></tbody>
  </table>
  <div class="method">SVD is applied to per-slot time-centered `decoder.project2(slotz)` vectors. If dynamic information is merely rotated, the leading temporal components should carry meaningful variance while their loadings are spread across the hard static/dynamic channel split.</div>
</main>
<script>
const DATA={data};
const mode=document.getElementById('mode');
const caseSelect=document.getElementById('case');
const chart=document.getElementById('chart');
const metrics=document.getElementById('metrics');
DATA.cases.forEach(item=>{{const option=document.createElement('option');option.value=item.id;option.textContent=item.label;caseSelect.appendChild(option);}});
function row(name,value){{const tr=document.createElement('tr');tr.innerHTML=`<td>${{name}}</td><td>${{Number(value).toFixed(4)}}</td>`;return tr;}}
function render(){{
  const entry=DATA.entries[caseSelect.value][mode.value];
  chart.src=entry.chart;
  const m=entry.metrics;
  metrics.replaceChildren(
    row('top1 variance ratio',m.top1_variance_ratio),
    row('top4 cumulative variance',m.top4_cumulative_variance_ratio),
    row('top16 cumulative variance',m.top16_cumulative_variance_ratio),
    row('dynamic temporal power ratio',m.dynamic_temporal_power_ratio),
    row('top1 dynamic loading ratio',m.top1_dynamic_loading_ratio),
    row('top4 dynamic loading ratio mean',m.top4_dynamic_loading_ratio_mean),
    row('expected dynamic dim ratio',m.expected_dynamic_ratio_if_unstructured)
  );
}}
[mode,caseSelect].forEach(element=>element.addEventListener('change',render));
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
        else viewer_dir / "pca_svd_temporal"
    )
    combined = json.loads(
        (viewer_dir / "combined_metadata.json").read_text(encoding="utf-8")
    )
    cases = combined["cases"]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else latest_checkpoint(DEFAULT_MOVIC_CKPT_DIR)
    )
    config = args.config.resolve()
    label = f"movi_current_{checkpoint_step(checkpoint):06d}"

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = getattr(torch, args.amp_dtype)

    cfg, model = build_model(config, checkpoint, device)
    num_slots = int(cfg.max_num)
    dynamic_ratio = float(model.m.decode.rd)
    entries: dict[str, dict] = {
        "average": {"crop": {}, "padding": {}},
        **{case["case_id"]: {"crop": {}, "padding": {}} for case in cases},
    }
    grouped: dict[str, list[dict]] = {"crop": [], "padding": []}

    for case_index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        for mode, source_key in (("crop", "crop_dir"), ("padding", "padding_dir")):
            output_path = output_dir / "cases" / case_id / mode / f"{label}.png"
            json_path = output_path.with_suffix(".json")
            if output_path.is_file() and json_path.is_file() and not args.force:
                summary = json.loads(json_path.read_text(encoding="utf-8"))
            else:
                frame_root = outputs_root / combined[source_key] / "cases" / case_id / "original"
                rgb = read_frame_sequence(frame_root, int(case["frames"]))
                video = normalize_rgb_frames(rgb)
                boxes = boxes_from_metadata(
                    case[mode]["amg"], num_slots, len(rgb), rgb.shape[1], rgb.shape[2]
                )
                feature, slots = run_xssc(model, video, boxes, device, amp_dtype)
                projected = project_slots(model.m.decode, slots, amp_dtype)
                values = projected[0].detach().float().cpu().numpy().astype(np.float64)
                del feature, slots, projected
                analysis = svd_temporal_analysis(values, dynamic_ratio, args.top_k)
                summary = {
                    "case_id": case_id,
                    "mode": mode,
                    "label": label,
                    "checkpoint": str(checkpoint),
                    **analysis["summary"],
                }
                plot_case(analysis, output_path, f"{case_id} | {mode} | {label}")
                json_path.write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )
            grouped[mode].append(summary)
            entries[case_id][mode] = {
                "chart": str(output_path.relative_to(output_dir)),
                "metrics": summary,
            }
            print(f"[case] {case_index}/{len(cases)} {mode} {case_id}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for mode in ("crop", "padding"):
        summary = average_case_summaries(grouped[mode])
        output_path = output_dir / "averages" / mode / f"{label}.png"
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=145)
        names = ["top1", "top2", "top4", "top8", "top16"]
        values = [
            summary["top1_variance_ratio"],
            summary["top2_cumulative_variance_ratio"],
            summary["top4_cumulative_variance_ratio"],
            summary["top8_cumulative_variance_ratio"],
            summary["top16_cumulative_variance_ratio"],
        ]
        axes[0].bar(names, values, color="#2563eb")
        axes[0].set_ylim(0.0, max(0.05, max(values) * 1.15))
        axes[0].set_title("Average temporal variance concentration")
        axes[0].set_ylabel("explained variance")
        axes[0].grid(alpha=0.22)
        axes[1].bar(
            ["dynamic power", "top1 dyn load", "top4 dyn load"],
            [
                summary["dynamic_temporal_power_ratio"],
                summary["top1_dynamic_loading_ratio"],
                summary["top4_dynamic_loading_ratio_mean"],
            ],
            color=["#dc2626", "#d97706", "#7c3aed"],
        )
        axes[1].axhline(
            summary["expected_dynamic_ratio_if_unstructured"],
            color="#525252",
            linestyle="--",
            linewidth=1.3,
            label="dynamic dim ratio",
        )
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Average loading against hard split")
        axes[1].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"{len(cases)}-case average | {mode} | {label}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        avg_summary = {"case_count": len(cases), **summary}
        (output_path.with_suffix(".json")).write_text(
            json.dumps(avg_summary, indent=2) + "\n", encoding="utf-8"
        )
        entries["average"][mode] = {
            "chart": str(output_path.relative_to(output_dir)),
            "metrics": avg_summary,
        }

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Per-slot time-centered SVD/PCA on decoder.project2(slotz). The "
            "hard split is only used afterward to measure where each principal "
            "direction places squared loading mass."
        ),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "label": label,
        "cases": [
            {"id": "average", "label": f"{len(cases)}-case average"},
            *[
                {"id": case["case_id"], "label": f"{idx:02d} | {case['case_id']}"}
                for idx, case in enumerate(cases, start=1)
            ],
        ],
        "entries": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(build_html(metadata), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# xSSC decoder PCA/SVD temporal analysis\n\n"
        "This analysis runs SVD on per-slot time-centered "
        "`decoder.project2(slotz)` vectors. It checks whether temporal change "
        "is concentrated in a few rotated directions and whether those "
        "directions align with or ignore the hard static/dynamic channel split.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index": str(output_dir / "index.html"),
                "checkpoint": str(checkpoint),
                "cases": len(cases),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
