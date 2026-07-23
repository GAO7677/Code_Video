#!/usr/bin/env python3
"""Analyze frozen DINOv3 features with 3D FFT over time and patch grid."""

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

from analyze_dinov3_feature_frequency_temporal import (  # noqa: E402
    extract_dinov3_features,
    resample,
)
from analyze_slot_temporal_similarity_viewer import (  # noqa: E402
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_VIEWER_DIR,
    normalize_rgb_frames,
    read_frame_sequence,
)
from object_centric_bench.model.dinov3_backbone import DINO3ViT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch-frames", type=int, default=16)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def radial_spatial_bins(height: int, width: int, rfft_width: bool = False) -> tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(height)
    fx = np.fft.rfftfreq(width) if rfft_width else np.fft.fftfreq(width)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(yy**2 + xx**2)
    bins = np.linspace(0.0, float(radius.max()) + 1e-8, 13)
    ids = np.digitize(radius.reshape(-1), bins) - 1
    ids = np.clip(ids, 0, len(bins) - 2)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return ids, centers


def analyze_3dfft(features: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    values = features.astype(np.float64)
    dynamic = values - values.mean(axis=0, keepdims=True)
    time_steps, channels, height, width = dynamic.shape

    window_t = np.hanning(time_steps)[:, None, None, None]
    window_y = np.hanning(height)[None, None, :, None]
    window_x = np.hanning(width)[None, None, None, :]
    windowed = dynamic * window_t * window_y * window_x
    spectrum = np.fft.fftn(windowed, axes=(0, 2, 3))
    power = np.abs(spectrum) ** 2

    temporal_abs_freq = np.abs(np.fft.fftfreq(time_steps, d=1.0))
    temporal_freq = np.unique(temporal_abs_freq)
    temporal_power = np.asarray(
        [
            power[np.isclose(temporal_abs_freq, freq)].sum()
            for freq in temporal_freq
        ],
        dtype=np.float64,
    )
    temporal_power = temporal_power / max(temporal_power.sum(), np.finfo(np.float64).eps)

    spatial_ids, spatial_centers = radial_spatial_bins(height, width, rfft_width=False)
    flat_power = power.reshape(power.shape[0], channels, -1)
    spatial_power = np.zeros(len(spatial_centers), dtype=np.float64)
    joint_power = np.zeros((len(temporal_freq), len(spatial_centers)), dtype=np.float64)
    for bin_id in range(len(spatial_centers)):
        mask = spatial_ids == bin_id
        by_signed_time = flat_power[:, :, mask].sum(axis=(1, 2))
        joint_power[:, bin_id] = np.asarray(
            [
                by_signed_time[np.isclose(temporal_abs_freq, freq)].sum()
                for freq in temporal_freq
            ],
            dtype=np.float64,
        )
        spatial_power[bin_id] = joint_power[:, bin_id].sum()
    spatial_power = spatial_power / max(spatial_power.sum(), np.finfo(np.float64).eps)
    joint_power = joint_power / max(joint_power.sum(), np.finfo(np.float64).eps)

    nonzero_temporal = temporal_freq > 0
    nonzero_spatial = spatial_centers > 0
    dynamic_temporal_power = temporal_power[nonzero_temporal]
    dynamic_temporal_freq = temporal_freq[nonzero_temporal]
    temporal_centroid = float((temporal_freq * temporal_power).sum())
    spatial_centroid = float((spatial_centers * spatial_power).sum())

    static_energy = float(temporal_power[0])
    low_temporal = float(temporal_power[(temporal_freq > 0) & (temporal_freq <= 0.10)].sum())
    high_temporal = float(temporal_power[temporal_freq >= 0.25].sum())
    low_spatial = float(spatial_power[nonzero_spatial & (spatial_centers <= 0.12)].sum())
    high_spatial = float(spatial_power[spatial_centers >= 0.30].sum())

    summary = {
        "frames": int(time_steps),
        "channels": int(channels),
        "grid_h": int(height),
        "grid_w": int(width),
        "static_ft0_energy_ratio": static_energy,
        "dynamic_ft_nonzero_energy_ratio": float(1.0 - static_energy),
        "temporal_centroid_cycles_per_frame": temporal_centroid,
        "temporal_dominant_nonzero_frequency_cycles_per_frame": float(
            dynamic_temporal_freq[int(dynamic_temporal_power.argmax())]
        )
        if len(dynamic_temporal_freq)
        else 0.0,
        "low_temporal_energy_0_0_to_0_10": low_temporal,
        "high_temporal_energy_ge_0_25": high_temporal,
        "spatial_centroid_cycles_per_patch": spatial_centroid,
        "low_spatial_energy_le_0_12": low_spatial,
        "high_spatial_energy_ge_0_30": high_spatial,
    }
    arrays = {
        "temporal_freq": temporal_freq.astype(np.float32),
        "temporal_power": temporal_power.astype(np.float32),
        "spatial_freq": spatial_centers.astype(np.float32),
        "spatial_power": spatial_power.astype(np.float32),
        "joint_power": joint_power.astype(np.float32),
    }
    return summary, arrays


def average_results(results: list[tuple[dict, dict[str, np.ndarray]]]) -> tuple[dict, dict[str, np.ndarray]]:
    freq_grid = np.linspace(0.0, 0.5, 100, dtype=np.float32)
    spatial_grid = np.linspace(0.0, 0.72, 64, dtype=np.float32)
    temporal = []
    spatial = []
    joint = []
    for _, arrays in results:
        temporal.append(np.interp(freq_grid, arrays["temporal_freq"], arrays["temporal_power"], left=0.0, right=0.0))
        spatial.append(np.interp(spatial_grid, arrays["spatial_freq"], arrays["spatial_power"], left=0.0, right=0.0))
        resized = torch.from_numpy(arrays["joint_power"])[None, None].float()
        joint.append(
            torch.nn.functional.interpolate(
                resized, size=(100, 64), mode="bilinear", align_corners=True
            )[0, 0].numpy()
        )
    arrays = {
        "temporal_freq": freq_grid,
        "temporal_power": np.stack(temporal).mean(axis=0).astype(np.float32),
        "spatial_freq": spatial_grid,
        "spatial_power": np.stack(spatial).mean(axis=0).astype(np.float32),
        "joint_power": np.stack(joint).mean(axis=0).astype(np.float32),
    }
    fields = (
        "static_ft0_energy_ratio",
        "dynamic_ft_nonzero_energy_ratio",
        "temporal_centroid_cycles_per_frame",
        "temporal_dominant_nonzero_frequency_cycles_per_frame",
        "low_temporal_energy_0_0_to_0_10",
        "high_temporal_energy_ge_0_25",
        "spatial_centroid_cycles_per_patch",
        "low_spatial_energy_le_0_12",
        "high_spatial_energy_ge_0_30",
    )
    summary = {field: float(np.mean([item[0][field] for item in results])) for field in fields}
    first = results[0][0]
    summary.update({"case_count": len(results), "channels": first["channels"], "grid_h": first["grid_h"], "grid_w": first["grid_w"]})
    return summary, arrays


def plot_3dfft(arrays: dict[str, np.ndarray], summary: dict, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), dpi=145)
    axes[0].plot(arrays["temporal_freq"], arrays["temporal_power"], color="#2563eb", linewidth=1.9)
    axes[0].axvspan(0.0, 0.10, color="#16a34a", alpha=0.12)
    axes[0].axvspan(0.25, 0.5, color="#dc2626", alpha=0.10)
    axes[0].set_title("3D FFT temporal power")
    axes[0].set_xlabel("temporal frequency (cycles/frame)")
    axes[0].set_ylabel("relative energy")
    axes[0].grid(alpha=0.22)

    axes[1].plot(arrays["spatial_freq"], arrays["spatial_power"], color="#7c3aed", linewidth=1.9)
    axes[1].axvspan(0.0, 0.12, color="#16a34a", alpha=0.12)
    axes[1].axvspan(0.30, float(arrays["spatial_freq"][-1]), color="#dc2626", alpha=0.10)
    axes[1].set_title("3D FFT radial spatial power")
    axes[1].set_xlabel("spatial frequency (cycles/patch)")
    axes[1].set_ylabel("relative energy")
    axes[1].grid(alpha=0.22)

    joint = axes[2].imshow(
        np.log10(np.maximum(arrays["joint_power"], 1e-10)),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="magma",
        extent=[
            float(arrays["spatial_freq"][0]),
            float(arrays["spatial_freq"][-1]),
            float(arrays["temporal_freq"][0]),
            float(arrays["temporal_freq"][-1]),
        ],
    )
    axes[2].set_title("Joint time-space energy, log10")
    axes[2].set_xlabel("spatial frequency")
    axes[2].set_ylabel("temporal frequency")
    fig.colorbar(joint, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_result(output_path: Path, summary: dict, arrays: dict[str, np.ndarray]) -> None:
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output_path.with_suffix(".npz"), **arrays)


def load_result(output_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    archive = np.load(output_path.with_suffix(".npz"))
    arrays = {key: archive[key] for key in archive.files}
    return summary, arrays


def relative_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def build_html(metadata: dict) -> str:
    data = json.dumps(metadata, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DINOv3 Feature 3D FFT</title>
<style>
body{{margin:0;color:#171717;background:#fff;font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;background:rgba(255,255,255,.96);border-bottom:1px solid #d4d4d4;z-index:2}}
.bar{{max-width:1440px;margin:auto;padding:14px 20px;display:flex;gap:18px;align-items:end;flex-wrap:wrap}}
h1{{font-size:20px;margin:0 auto 1px 0;letter-spacing:0}}
label{{display:grid;gap:5px;color:#666;font-size:12px;font-weight:650}}
select{{min-width:210px;height:36px;border:1px solid #aaa;border-radius:5px;background:#fff;padding:0 10px}}
main{{max-width:1440px;margin:auto;padding:20px}}
.figure{{border:1px solid #d4d4d4;border-radius:6px;overflow:hidden}}
img{{display:block;width:100%;height:auto}}
table{{width:100%;margin-top:18px;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:10px 12px;border-bottom:1px solid #d4d4d4;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
thead th{{background:#f5f5f4;color:#444;font-size:12px}}
.method{{margin-top:12px;color:#666;font-size:12px}}
</style>
</head>
<body>
<header><div class="bar">
  <h1>DINOv3 Feature 3D FFT</h1>
  <label>Preprocessing<select id="mode"><option value="crop">Center crop</option><option value="padding">Resize + padding</option></select></label>
  <label>Cases<select id="case"></select></label>
</div></header>
<main>
  <figure class="figure"><img id="chart" alt="DINOv3 3D FFT chart"></figure>
  <table><thead><tr><th>Scope</th><th>Channels</th><th>Grid</th><th>ft=0 energy</th><th>dynamic ft energy</th><th>temporal centroid</th><th>low temporal</th><th>high temporal</th><th>spatial centroid</th><th>high spatial</th></tr></thead><tbody id="metrics"></tbody></table>
  <div class="method">DINOv3 features `[T,1024,16,16]` are mean-centered and Hann-windowed, then 3D FFT is applied over `[T,H,W]` per channel. Energy is summed over channels; the channel axis itself is not Fourier transformed.</div>
</main>
<script>
const DATA={data};
const mode=document.getElementById('mode');
const caseSelect=document.getElementById('case');
const chart=document.getElementById('chart');
const metrics=document.getElementById('metrics');
DATA.cases.forEach(item=>{{const option=document.createElement('option');option.value=item.id;option.textContent=item.label;caseSelect.appendChild(option);}});
function render(){{
  const entry=DATA.entries[caseSelect.value][mode.value];
  chart.src=entry.chart;
  const m=entry.metrics;
  metrics.innerHTML=`<tr><td>${{caseSelect.value==='average' ? m.case_count + '-case average' : m.frames + ' frames'}}</td><td>${{m.channels}}</td><td>${{m.grid_h}}x${{m.grid_w}}</td><td>${{m.static_ft0_energy_ratio.toFixed(4)}}</td><td>${{m.dynamic_ft_nonzero_energy_ratio.toFixed(4)}}</td><td>${{m.temporal_centroid_cycles_per_frame.toFixed(4)}}</td><td>${{m.low_temporal_energy_0_0_to_0_10.toFixed(4)}}</td><td>${{m.high_temporal_energy_ge_0_25.toFixed(4)}}</td><td>${{m.spatial_centroid_cycles_per_patch.toFixed(4)}}</td><td>${{m.high_spatial_energy_ge_0_30.toFixed(4)}}</td></tr>`;
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
    output_dir = args.output_dir.resolve() if args.output_dir else viewer_dir / "dinov3_feature_3dfft"
    combined = json.loads((viewer_dir / "combined_metadata.json").read_text(encoding="utf-8"))
    cases = combined["cases"][: args.max_cases] if args.max_cases > 0 else combined["cases"]

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = getattr(torch, args.amp_dtype)
    backbone = DINO3ViT(rearrange=True, norm_out=False).to(device).eval()

    entries = {"average": {"crop": {}, "padding": {}}}
    grouped: dict[str, list[tuple[dict, dict[str, np.ndarray]]]] = {"crop": [], "padding": []}
    for case in cases:
        entries[case["case_id"]] = {"crop": {}, "padding": {}}

    for case_index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        for mode, source_key in (("crop", "crop_dir"), ("padding", "padding_dir")):
            output_path = output_dir / "cases" / case_id / mode / "dinov3_feature_3dfft.png"
            complete = output_path.is_file() and output_path.with_suffix(".json").is_file() and output_path.with_suffix(".npz").is_file()
            if complete and not args.force:
                summary, arrays = load_result(output_path)
            else:
                frame_root = outputs_root / combined[source_key] / "cases" / case_id / "original"
                rgb = read_frame_sequence(frame_root, int(case["frames"]))
                video = normalize_rgb_frames(rgb)
                features = extract_dinov3_features(backbone, video, device, amp_dtype, args.batch_frames)
                summary, arrays = analyze_3dfft(features)
                summary.update({"case_id": case_id, "mode": mode, "backbone": "dinov3_vitl16_lvd1689m", "feature_shape": list(features.shape)})
                plot_3dfft(arrays, summary, output_path, f"{case_id} | {mode} | DINOv3 feature 3D FFT")
                save_result(output_path, summary, arrays)
            grouped[mode].append((summary, arrays))
            entries[case_id][mode] = {"chart": relative_path(output_path, output_dir), "metrics": summary}
            print(f"[case] {case_index}/{len(cases)} {case_id} {mode}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for mode in ("crop", "padding"):
        summary, arrays = average_results(grouped[mode])
        output_path = output_dir / "averages" / mode / "dinov3_feature_3dfft.png"
        plot_3dfft(arrays, summary, output_path, f"{len(cases)}-case average | {mode} | DINOv3 feature 3D FFT")
        save_result(output_path, summary, arrays)
        entries["average"][mode] = {"chart": relative_path(output_path, output_dir), "metrics": summary}

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Frozen DINOv3 ViT-L/16 patch features are extracted for every frame. "
            "Each channel's [T,16,16] feature volume is transformed with 3D FFT over time and patch grid; energy is summed over channels."
        ),
        "cases": [{"id": "average", "label": f"{len(cases)}-case average"}]
        + [{"id": case["case_id"], "label": f"{index:02d} | {case['case_id']}"} for index, case in enumerate(cases, 1)],
        "entries": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(build_html(metadata), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# DINOv3 feature 3D FFT\n\n"
        "This analysis uses the same 22 cases as the existing xSSC visualization. "
        "For every video, frozen DINOv3 features `[T,1024,16,16]` are mean-centered, Hann-windowed, and transformed by 3D FFT over `[T,H,W]` for each channel. "
        "Energy is summed over channels. The channel axis is not Fourier transformed because it has no geometric ordering.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "index": str(output_dir / "index.html"), "cases": len(cases)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
