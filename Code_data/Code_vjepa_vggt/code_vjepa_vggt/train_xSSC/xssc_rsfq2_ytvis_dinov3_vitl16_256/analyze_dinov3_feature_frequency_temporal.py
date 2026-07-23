#!/usr/bin/env python3
"""Analyze temporal changes of DINOv3 patch features in the spatial frequency domain."""

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
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

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


def extract_dinov3_features(
    backbone: DINO3ViT,
    video: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    batch_frames: int,
) -> np.ndarray:
    frames = video[0]
    chunks = []
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        for start in range(0, frames.shape[0], batch_frames):
            chunk = frames[start : start + batch_frames].to(device, non_blocking=True)
            feat = backbone(chunk)
            chunks.append(feat.detach().float().cpu())
    return torch.cat(chunks, dim=0).numpy()


def phase_coherence(a: np.ndarray, b: np.ndarray) -> float:
    amplitude_a = np.abs(a)
    amplitude_b = np.abs(b)
    weights = amplitude_a * amplitude_b
    unit_delta = a * np.conjugate(b) / np.maximum(weights, np.finfo(np.float64).eps)
    return float(np.abs((weights * unit_delta).sum()) / max(weights.sum(), np.finfo(np.float64).eps))


def frequency_curves(features: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    values = features.astype(np.float64)
    spectrum = np.fft.fft2(values, axes=(-2, -1))
    amplitude = np.abs(spectrum)
    flat_amplitude = amplitude.reshape(amplitude.shape[0], -1)
    flat_amplitude = flat_amplitude / np.maximum(
        np.linalg.norm(flat_amplitude, axis=1, keepdims=True),
        np.finfo(np.float64).eps,
    )

    frame0_amplitude = flat_amplitude @ flat_amplitude[0]
    adjacent_amplitude = np.sum(flat_amplitude[:-1] * flat_amplitude[1:], axis=1)
    frame0_phase = np.asarray(
        [phase_coherence(spectrum[index], spectrum[0]) for index in range(spectrum.shape[0])],
        dtype=np.float64,
    )
    adjacent_phase = np.asarray(
        [
            phase_coherence(spectrum[index + 1], spectrum[index])
            for index in range(spectrum.shape[0] - 1)
        ],
        dtype=np.float64,
    )

    dynamic = values - values.mean(axis=0, keepdims=True)
    temporal_spectrum = np.fft.rfft(dynamic, axis=0)[1:]
    temporal_freq = np.fft.rfftfreq(values.shape[0], d=1.0)[1:]
    temporal_power = np.abs(temporal_spectrum) ** 2
    temporal_power_curve = temporal_power.sum(axis=(1, 2, 3))
    temporal_power_curve = temporal_power_curve / max(
        temporal_power_curve.sum(), np.finfo(np.float64).eps
    )

    arrays = {
        "frame0_amplitude": frame0_amplitude.astype(np.float32),
        "frame0_phase": frame0_phase.astype(np.float32),
        "adjacent_amplitude": adjacent_amplitude.astype(np.float32),
        "adjacent_phase": adjacent_phase.astype(np.float32),
        "temporal_freq": temporal_freq.astype(np.float32),
        "temporal_power": temporal_power_curve.astype(np.float32),
    }
    summary = {
        "frames": int(features.shape[0]),
        "channels": int(features.shape[1]),
        "grid_h": int(features.shape[2]),
        "grid_w": int(features.shape[3]),
        "frame0_final_amplitude_similarity": float(frame0_amplitude[-1]),
        "frame0_final_phase_coherence": float(frame0_phase[-1]),
        "adjacent_amplitude_similarity_mean": float(adjacent_amplitude.mean()),
        "adjacent_phase_coherence_mean": float(adjacent_phase.mean()),
        "temporal_dominant_frequency_cycles_per_frame": float(
            temporal_freq[int(temporal_power_curve.argmax())]
        )
        if len(temporal_freq)
        else 0.0,
        "temporal_low_frequency_energy_le_0_10": float(
            temporal_power_curve[temporal_freq <= 0.10].sum()
        )
        if len(temporal_freq)
        else 0.0,
        "temporal_high_frequency_energy_ge_0_25": float(
            temporal_power_curve[temporal_freq >= 0.25].sum()
        )
        if len(temporal_freq)
        else 0.0,
    }
    return summary, arrays


def resample(values: np.ndarray, size: int = 100) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, size)
    return np.interp(target, source, values).astype(np.float32)


def average_results(results: list[tuple[dict, dict[str, np.ndarray]]]) -> tuple[dict, dict[str, np.ndarray]]:
    arrays = {
        key: np.stack([resample(item[1][key]) for item in results]).mean(axis=0)
        for key in ("frame0_amplitude", "frame0_phase", "adjacent_amplitude", "adjacent_phase")
    }
    max_freq_len = max(len(item[1]["temporal_freq"]) for item in results)
    freq_grid = np.linspace(0.0, 0.5, max_freq_len, dtype=np.float32)
    powers = []
    for _, case_arrays in results:
        powers.append(
            np.interp(
                freq_grid,
                case_arrays["temporal_freq"],
                case_arrays["temporal_power"],
                left=0.0,
                right=0.0,
            )
        )
    arrays["temporal_freq"] = freq_grid
    arrays["temporal_power"] = np.stack(powers).mean(axis=0).astype(np.float32)
    metrics = {
        field: float(np.mean([item[0][field] for item in results]))
        for field in (
            "frame0_final_amplitude_similarity",
            "frame0_final_phase_coherence",
            "adjacent_amplitude_similarity_mean",
            "adjacent_phase_coherence_mean",
            "temporal_dominant_frequency_cycles_per_frame",
            "temporal_low_frequency_energy_le_0_10",
            "temporal_high_frequency_energy_ge_0_25",
        )
    }
    first = results[0][0]
    metrics.update(
        {
            "case_count": len(results),
            "channels": first["channels"],
            "grid_h": first["grid_h"],
            "grid_w": first["grid_w"],
        }
    )
    return metrics, arrays


def plot_curves(arrays: dict[str, np.ndarray], summary: dict, output_path: Path, title: str, aggregate: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=145)
    x0 = np.linspace(0.0, 1.0, len(arrays["frame0_amplitude"])) if aggregate else np.arange(len(arrays["frame0_amplitude"]))
    xa = np.linspace(0.0, 1.0, len(arrays["adjacent_amplitude"])) if aggregate else np.arange(1, len(arrays["adjacent_amplitude"]) + 1)
    axes[0].plot(x0, arrays["frame0_amplitude"], color="#2563eb", label="amplitude")
    axes[0].plot(x0, arrays["frame0_phase"], color="#dc2626", label="phase")
    axes[0].set_title("Spatial FFT similarity to frame 0")
    axes[0].set_xlabel("normalized time" if aggregate else "frame")
    axes[0].set_ylabel("similarity / coherence")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.22)
    axes[0].legend(loc="lower left")

    axes[1].plot(xa, arrays["adjacent_amplitude"], color="#2563eb", label="amplitude")
    axes[1].plot(xa, arrays["adjacent_phase"], color="#dc2626", label="phase")
    axes[1].set_title("Spatial FFT adjacent-frame similarity")
    axes[1].set_xlabel("normalized time" if aggregate else "destination frame")
    axes[1].set_ylabel("similarity / coherence")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.22)
    axes[1].legend(loc="lower left")

    axes[2].plot(arrays["temporal_freq"], arrays["temporal_power"], color="#111827")
    axes[2].axvspan(0.0, min(0.10, float(arrays["temporal_freq"][-1])), color="#16a34a", alpha=0.12)
    axes[2].axvspan(0.25, 0.5, color="#dc2626", alpha=0.10)
    axes[2].set_title("Temporal FFT power of DINOv3 features")
    axes[2].set_xlabel("frequency (cycles/frame)")
    axes[2].set_ylabel("relative dynamic power")
    axes[2].grid(alpha=0.22)

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


def build_html(metadata: dict) -> str:
    data = json.dumps(metadata, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DINOv3 Feature Frequency Temporal Curves</title>
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
  <h1>DINOv3 Feature Frequency Temporal Curves</h1>
  <label>Preprocessing<select id="mode"><option value="crop">Center crop</option><option value="padding">Resize + padding</option></select></label>
  <label>Cases<select id="case"></select></label>
</div></header>
<main>
  <figure class="figure"><img id="chart" alt="DINOv3 frequency temporal chart"></figure>
  <table><thead><tr><th>Scope</th><th>Channels</th><th>Grid</th><th>amp final vs f0</th><th>phase final vs f0</th><th>adj amp mean</th><th>adj phase mean</th><th>low freq <=0.10</th><th>high freq >=0.25</th></tr></thead><tbody id="metrics"></tbody></table>
  <div class="method">Per frame, the frozen DINOv3 `[1024,16,16]` patch feature map is transformed by spatial 2D FFT. Amplitude uses cosine similarity of magnitude spectra; phase uses amplitude-weighted phase coherence. The average view resamples timelines to normalized time before averaging 22 cases.</div>
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
  metrics.innerHTML=`<tr><td>${{caseSelect.value==='average' ? m.case_count + '-case average' : m.frames + ' frames'}}</td><td>${{m.channels}}</td><td>${{m.grid_h}}x${{m.grid_w}}</td><td>${{m.frame0_final_amplitude_similarity.toFixed(4)}}</td><td>${{m.frame0_final_phase_coherence.toFixed(4)}}</td><td>${{m.adjacent_amplitude_similarity_mean.toFixed(4)}}</td><td>${{m.adjacent_phase_coherence_mean.toFixed(4)}}</td><td>${{m.temporal_low_frequency_energy_le_0_10.toFixed(4)}}</td><td>${{m.temporal_high_frequency_energy_ge_0_25.toFixed(4)}}</td></tr>`;
}}
[mode,caseSelect].forEach(element=>element.addEventListener('change',render));
render();
</script>
</body>
</html>
"""


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else viewer_dir / "dinov3_feature_frequency_temporal")
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
            output_path = output_dir / "cases" / case_id / mode / "dinov3_feature_frequency.png"
            complete = output_path.is_file() and output_path.with_suffix(".json").is_file() and output_path.with_suffix(".npz").is_file()
            if complete and not args.force:
                summary, arrays = load_result(output_path)
            else:
                frame_root = outputs_root / combined[source_key] / "cases" / case_id / "original"
                rgb = read_frame_sequence(frame_root, int(case["frames"]))
                video = normalize_rgb_frames(rgb)
                features = extract_dinov3_features(backbone, video, device, amp_dtype, args.batch_frames)
                summary, arrays = frequency_curves(features)
                summary.update(
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "backbone": "dinov3_vitl16_lvd1689m",
                        "feature_shape": list(features.shape),
                    }
                )
                plot_curves(arrays, summary, output_path, f"{case_id} | {mode} | DINOv3 feature FFT", aggregate=False)
                save_result(output_path, summary, arrays)
            grouped[mode].append((summary, arrays))
            entries[case_id][mode] = {"chart": rel(output_path, output_dir), "metrics": summary}
            print(f"[case] {case_index}/{len(cases)} {case_id} {mode}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for mode in ("crop", "padding"):
        summary, arrays = average_results(grouped[mode])
        output_path = output_dir / "averages" / mode / "dinov3_feature_frequency.png"
        plot_curves(arrays, summary, output_path, f"{len(cases)}-case average | {mode} | DINOv3 feature FFT", aggregate=True)
        save_result(output_path, summary, arrays)
        entries["average"][mode] = {"chart": rel(output_path, output_dir), "metrics": summary}

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Frozen DINOv3 ViT-L/16 patch features are extracted for every frame. "
            "Each [1024,16,16] feature map is transformed with spatial 2D FFT. "
            "Curves report amplitude-spectrum cosine and amplitude-weighted phase coherence over time."
        ),
        "cases": [{"id": "average", "label": f"{len(cases)}-case average"}]
        + [{"id": case["case_id"], "label": f"{idx:02d} | {case['case_id']}"} for idx, case in enumerate(cases, 1)],
        "entries": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(build_html(metadata), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# DINOv3 feature frequency temporal curves\n\n"
        "This viewer uses the same 22 cases as the slot overlay comparison. "
        "For every frame, frozen DINOv3 ViT-L/16 patch features `[1024,16,16]` are transformed by spatial 2D FFT. "
        "Amplitude curves are cosine similarities between magnitude spectra; phase curves are amplitude-weighted phase coherence. "
        "The average view resamples each case to normalized time before averaging.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "index": str(output_dir / "index.html"), "cases": len(cases)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
