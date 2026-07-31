#!/usr/bin/env python3
"""Analyze official xSSC decoder static/dynamic channel split.

This script reuses the slot/attention/RAFT artifacts produced by
analyze_official_xssc_dynamics_raft.py, then applies the official decoder
project2(slotz) and splits the resulting decoder memory by MarkovRarDecoder.rd.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_official_xssc_dynamics_raft as base  # noqa: E402


DEFAULT_INPUT_DIR = Path("/data/gaoya/agent-data/outputs/official_xssc_dynamics_raft")
DEFAULT_OUTPUT_DIR = Path("/data/gaoya/agent-data/outputs/official_xssc_decoder_static_dynamic_raft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def band_energy(arr: np.ndarray) -> dict[str, float]:
    fft = np.fft.rfft(arr, axis=0)
    energy = np.abs(fft) ** 2
    by_freq = energy.reshape(energy.shape[0], -1).mean(axis=1)
    total = float(by_freq.sum()) + 1e-12
    n = len(by_freq)
    low_end = max(1, math.ceil(n * 0.25))
    mid_end = max(low_end + 1, math.ceil(n * 0.60))
    return {
        "dc": float(by_freq[0] / total),
        "low": float(by_freq[1:low_end].sum() / total) if low_end > 1 else 0.0,
        "mid": float(by_freq[low_end:mid_end].sum() / total),
        "high": float(by_freq[mid_end:].sum() / total),
    }


def dc_ratio_by_slot(features: np.ndarray) -> np.ndarray:
    fft = np.fft.rfft(features, axis=0)
    energy = np.abs(fft) ** 2
    total = energy.sum(axis=0)
    ratio = energy[0] / np.maximum(total, 1e-12)
    return ratio.mean(axis=-1)


def adjacent_metrics(features: np.ndarray) -> dict[str, np.ndarray]:
    dim = features.shape[-1]
    delta = np.linalg.norm(features[1:] - features[:-1], axis=-1) / math.sqrt(dim)
    cos = np.sum(features[1:] * features[:-1], axis=-1) / np.maximum(
        np.linalg.norm(features[1:], axis=-1) * np.linalg.norm(features[:-1], axis=-1),
        1e-12,
    )
    mean = features.mean(axis=0, keepdims=True)
    residual = features - mean
    residual_cos = np.sum(residual[1:] * residual[:-1], axis=-1) / np.maximum(
        np.linalg.norm(residual[1:], axis=-1) * np.linalg.norm(residual[:-1], axis=-1),
        1e-12,
    )
    energy = np.linalg.norm(residual, axis=-1) / math.sqrt(dim)
    return {
        "delta_l2": delta,
        "one_minus_cos": 1.0 - cos,
        "residual_one_minus_cos": 1.0 - residual_cos,
        "residual_energy_mid": 0.5 * (energy[1:] + energy[:-1]),
    }


def cumulative_flow(global_flow: np.ndarray, time_steps: int) -> np.ndarray:
    pair_flow = np.zeros((time_steps, time_steps), dtype=np.float64)
    cumsum = np.concatenate([[0.0], np.cumsum(global_flow)])
    for start in range(time_steps):
        for end in range(start + 1, time_steps):
            pair_flow[start, end] = cumsum[end] - cumsum[start]
            pair_flow[end, start] = pair_flow[start, end]
    return pair_flow


def all_pair_distance(features: np.ndarray) -> np.ndarray:
    return 1.0 - base.slot_cosine_pairwise(features)


def xssc_residual_motion_from_slots(slots: np.ndarray, attention: np.ndarray) -> np.ndarray:
    slot_dyn = slots - slots.mean(axis=0, keepdims=True)
    dyn_cos = np.sum(slot_dyn[1:] * slot_dyn[:-1], axis=-1) / np.maximum(
        np.linalg.norm(slot_dyn[1:], axis=-1) * np.linalg.norm(slot_dyn[:-1], axis=-1),
        1e-12,
    )
    dyn_one_minus_cos = 1.0 - dyn_cos
    centroids = base.attention_centroids(attention)
    centroid_shift = np.linalg.norm(centroids[1:] - centroids[:-1], axis=-1)
    return (base.zscore(dyn_one_minus_cos) + base.zscore(centroid_shift)).mean(axis=1)


def xssc_feature_motion_from_slots(slots: np.ndarray) -> np.ndarray:
    dim = slots.shape[-1]
    delta = np.linalg.norm(slots[1:] - slots[:-1], axis=-1) / math.sqrt(dim)
    slot_dyn = slots - slots.mean(axis=0, keepdims=True)
    dyn_cos = np.sum(slot_dyn[1:] * slot_dyn[:-1], axis=-1) / np.maximum(
        np.linalg.norm(slot_dyn[1:], axis=-1) * np.linalg.norm(slot_dyn[:-1], axis=-1),
        1e-12,
    )
    return (base.zscore(delta) + base.zscore(1.0 - dyn_cos)).mean(axis=1)


@torch.inference_mode()
def decoder_memory_split(
    model: torch.nn.Module,
    slots: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    slot_tensor = torch.from_numpy(slots).to(device=device, dtype=torch.float32)
    memory = model.decode.project2(slot_tensor)
    split_static = int(memory.size(-1) * (1.0 - float(model.decode.rd)))
    static = memory[..., :split_static].detach().float().cpu().numpy()
    dynamic = memory[..., split_static:].detach().float().cpu().numpy()
    info = {
        "slot_dim": int(slot_tensor.shape[-1]),
        "decoder_memory_dim": int(memory.size(-1)),
        "rd": float(model.decode.rd),
        "static_dim": int(static.shape[-1]),
        "dynamic_dim": int(dynamic.shape[-1]),
    }
    return static, dynamic, info


def plot_frequency(path: Path, stats: dict[str, dict[str, float]], title: str) -> None:
    labels = ["dc", "low", "mid", "high"]
    groups = ["raw_slot", "decoder_static", "decoder_dynamic", "global_flow"]
    x = np.arange(len(labels))
    width = 0.20
    fig, ax = plt.subplots(figsize=(7.8, 3.4), dpi=150)
    for offset, group in enumerate(groups):
        values = [stats[group][label] for label in labels]
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("energy ratio")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_curves(path: Path, curves: dict[str, np.ndarray], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.6), dpi=150)
    for name, values in curves.items():
        width = 2.1 if name == "RAFT global flow" else 1.2
        color = "#111827" if name == "RAFT global flow" else None
        ax.plot(base.zscore(values), label=name, linewidth=width, color=color)
    ax.set_xlabel("adjacent transition t -> t+1")
    ax.set_ylabel("z-score")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(path: Path, matrix: np.ndarray, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=150)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("frame")
    ax.set_ylabel("frame")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analyze_one(
    slots: np.ndarray,
    attention: np.ndarray,
    flow: np.ndarray,
    static: np.ndarray,
    dynamic: np.ndarray,
) -> dict[str, Any]:
    time_steps = slots.shape[0]
    global_flow = np.linalg.norm(flow, axis=-1).mean(axis=(1, 2))
    pair_flow = cumulative_flow(global_flow, time_steps)
    triu = np.triu_indices(time_steps, k=1)

    raw_metrics = adjacent_metrics(slots)
    static_metrics = adjacent_metrics(static)
    dynamic_metrics = adjacent_metrics(dynamic)
    d_xssc_residual = xssc_residual_motion_from_slots(slots, attention)
    d_xssc_feature = xssc_feature_motion_from_slots(slots)
    d_decoder_dynamic = (
        base.zscore(dynamic_metrics["delta_l2"]) + base.zscore(dynamic_metrics["one_minus_cos"])
    ).mean(axis=1)
    d_decoder_static = (
        base.zscore(static_metrics["delta_l2"]) + base.zscore(static_metrics["one_minus_cos"])
    ).mean(axis=1)

    adjacent = {
        "raw_slot_delta_l2": raw_metrics["delta_l2"].mean(axis=1),
        "decoder_static_delta_l2": static_metrics["delta_l2"].mean(axis=1),
        "decoder_static_one_minus_cos": static_metrics["one_minus_cos"].mean(axis=1),
        "decoder_dynamic_delta_l2": dynamic_metrics["delta_l2"].mean(axis=1),
        "decoder_dynamic_one_minus_cos": dynamic_metrics["one_minus_cos"].mean(axis=1),
        "decoder_dynamic_residual_one_minus_cos": dynamic_metrics["residual_one_minus_cos"].mean(axis=1),
        "decoder_dynamic_residual_energy_mid": dynamic_metrics["residual_energy_mid"].mean(axis=1),
        "D_xSSC_residual_centroid": d_xssc_residual,
        "D_xSSC_feature_residual": d_xssc_feature,
        "D_decoder_dynamic": d_decoder_dynamic,
        "D_decoder_static": d_decoder_static,
    }
    adjacent_vs_flow = {
        name: {
            "pearson": base.pearson(values, global_flow),
            "spearman": base.spearman(values, global_flow),
        }
        for name, values in adjacent.items()
    }
    adjacent_vs_d_xssc = {
        name: {
            "pearson": base.pearson(values, d_xssc_feature),
            "spearman": base.spearman(values, d_xssc_feature),
        }
        for name, values in adjacent.items()
        if name != "D_xSSC_feature_residual"
    }

    raw_pair_dist = all_pair_distance(slots)
    static_pair_dist = all_pair_distance(static)
    dynamic_pair_dist = all_pair_distance(dynamic)
    pair_correlations = {
        "raw_pair_distance_vs_cumulative_flow": {
            "pearson": base.pearson(raw_pair_dist[triu], pair_flow[triu]),
            "spearman": base.spearman(raw_pair_dist[triu], pair_flow[triu]),
        },
        "decoder_static_pair_distance_vs_cumulative_flow": {
            "pearson": base.pearson(static_pair_dist[triu], pair_flow[triu]),
            "spearman": base.spearman(static_pair_dist[triu], pair_flow[triu]),
        },
        "decoder_dynamic_pair_distance_vs_cumulative_flow": {
            "pearson": base.pearson(dynamic_pair_dist[triu], pair_flow[triu]),
            "spearman": base.spearman(dynamic_pair_dist[triu], pair_flow[triu]),
        },
        "raw_pair_distance_vs_decoder_static_pair_distance": {
            "pearson": base.pearson(raw_pair_dist[triu], static_pair_dist[triu]),
            "spearman": base.spearman(raw_pair_dist[triu], static_pair_dist[triu]),
        },
        "raw_pair_distance_vs_decoder_dynamic_pair_distance": {
            "pearson": base.pearson(raw_pair_dist[triu], dynamic_pair_dist[triu]),
            "spearman": base.spearman(raw_pair_dist[triu], dynamic_pair_dist[triu]),
        },
    }

    raw_dc_slot = dc_ratio_by_slot(slots)
    static_dc_slot = dc_ratio_by_slot(static)
    dynamic_dc_slot = dc_ratio_by_slot(dynamic)

    return {
        "global_flow": global_flow,
        "pair_flow": pair_flow,
        "raw_pair_dist": raw_pair_dist,
        "static_pair_dist": static_pair_dist,
        "dynamic_pair_dist": dynamic_pair_dist,
        "curves": {
            "RAFT global flow": global_flow,
            "D_xSSC feature+residual": d_xssc_feature,
            "D_xSSC residual+centroid": d_xssc_residual,
            "decoder dynamic": d_decoder_dynamic,
            "decoder static": d_decoder_static,
            "decoder dyn delta": dynamic_metrics["delta_l2"].mean(axis=1),
        },
        "frequency": {
            "raw_slot": band_energy(slots),
            "decoder_static": band_energy(static),
            "decoder_dynamic": band_energy(dynamic),
            "global_flow": band_energy(global_flow),
        },
        "dc_ratio_by_slot": {
            "raw_slot": raw_dc_slot.tolist(),
            "decoder_static": static_dc_slot.tolist(),
            "decoder_dynamic": dynamic_dc_slot.tolist(),
        },
        "dc_slot_correlations": {
            "raw_vs_decoder_static": {
                "pearson": base.pearson(raw_dc_slot, static_dc_slot),
                "spearman": base.spearman(raw_dc_slot, static_dc_slot),
            },
            "raw_vs_decoder_dynamic": {
                "pearson": base.pearson(raw_dc_slot, dynamic_dc_slot),
                "spearman": base.spearman(raw_dc_slot, dynamic_dc_slot),
            },
        },
        "adjacent_vs_flow": adjacent_vs_flow,
        "adjacent_vs_d_xssc": adjacent_vs_d_xssc,
        "pair_correlations": pair_correlations,
    }


def best_by_abs_spearman(rows: dict[str, dict[str, float]]) -> tuple[str, float]:
    valid = [(k, v["spearman"]) for k, v in rows.items() if np.isfinite(v["spearman"])]
    if not valid:
        return "", float("nan")
    return max(valid, key=lambda item: abs(item[1]))


def render_assets(case_model_dir: Path, analysis: dict[str, Any]) -> dict[str, str]:
    case_model_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "curves": case_model_dir / "decoder_split_motion_curves.png",
        "frequency": case_model_dir / "decoder_split_frequency.png",
        "raw_pair": case_model_dir / "raw_slot_pair_distance.png",
        "static_pair": case_model_dir / "decoder_static_pair_distance.png",
        "dynamic_pair": case_model_dir / "decoder_dynamic_pair_distance.png",
        "pair_flow": case_model_dir / "pair_cumulative_flow.png",
    }
    plot_curves(paths["curves"], analysis["curves"], "xSSC decoder split dynamics vs RAFT")
    plot_frequency(paths["frequency"], analysis["frequency"], "frequency energy split")
    plot_heatmap(paths["raw_pair"], analysis["raw_pair_dist"], "raw slot pair distance")
    plot_heatmap(paths["static_pair"], analysis["static_pair_dist"], "decoder static pair distance")
    plot_heatmap(paths["dynamic_pair"], analysis["dynamic_pair_dist"], "decoder dynamic pair distance")
    plot_heatmap(paths["pair_flow"], analysis["pair_flow"], "RAFT cumulative flow")
    return {key: path.name for key, path in paths.items()}


def copy_or_link_static_inputs(input_dir: Path, output_dir: Path, case_id: str) -> dict[str, str]:
    src_case = input_dir / "cases" / case_id
    dst_case = output_dir / "cases" / case_id
    dst_case.mkdir(parents=True, exist_ok=True)
    links = {}
    for name in ["xssc_input_49f.mp4", "raft_flow.mp4"]:
        dst = dst_case / name
        src = src_case / name
        if not dst.exists():
            try:
                dst.symlink_to(src)
            except FileExistsError:
                pass
        links[name] = f"cases/{case_id}/{name}"
    return links


def build_html(output_dir: Path, report: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    sections = []
    for case in cases:
        cards = []
        for model in case["models"]:
            rows = []
            for name, values in model["adjacent_vs_flow"].items():
                rows.append(
                    f"<tr><td>{html.escape(name)}</td><td>{values['pearson']:.3f}</td><td>{values['spearman']:.3f}</td></tr>"
                )
            pair_rows = []
            for name, values in model["pair_correlations"].items():
                pair_rows.append(
                    f"<tr><td>{html.escape(name)}</td><td>{values['pearson']:.3f}</td><td>{values['spearman']:.3f}</td></tr>"
                )
            asset_base = f"cases/{case['case_id']}/{model['name']}"
            assets = model["assets"]
            cards.append(
                f"""
                <article class="model">
                  <h3>{html.escape(model['name'])}</h3>
                  <p class="small">decoder split: slot {model['split_info']['slot_dim']} -> memory {model['split_info']['decoder_memory_dim']}
                  = static {model['split_info']['static_dim']} + dynamic {model['split_info']['dynamic_dim']} (rd={model['split_info']['rd']:.2f})</p>
                  <p class="small">best vs RAFT: <b>{html.escape(model['best_vs_flow'][0])}</b> Spearman={model['best_vs_flow'][1]:.3f};
                  decoder dynamic vs D_xSSC feature Spearman={model['dynamic_vs_d_xssc_spearman']:.3f}</p>
                  <p class="small">DC ratio: raw={model['dc_mean']['raw_slot']:.3f},
                  decoder-static={model['dc_mean']['decoder_static']:.3f},
                  decoder-dynamic={model['dc_mean']['decoder_dynamic']:.3f}</p>
                  <div class="plots">
                    <figure><img src="{asset_base}/{assets['curves']}" loading="lazy"><figcaption>adjacent curves</figcaption></figure>
                    <figure><img src="{asset_base}/{assets['frequency']}" loading="lazy"><figcaption>frequency split</figcaption></figure>
                    <figure><img src="{asset_base}/{assets['raw_pair']}" loading="lazy"><figcaption>raw slot pair distance</figcaption></figure>
                    <figure><img src="{asset_base}/{assets['static_pair']}" loading="lazy"><figcaption>decoder static pair distance</figcaption></figure>
                    <figure><img src="{asset_base}/{assets['dynamic_pair']}" loading="lazy"><figcaption>decoder dynamic pair distance</figcaption></figure>
                    <figure><img src="{asset_base}/{assets['pair_flow']}" loading="lazy"><figcaption>RAFT cumulative flow</figcaption></figure>
                  </div>
                  <details><summary>Adjacent correlations vs RAFT</summary>
                    <table><thead><tr><th>function</th><th>Pearson</th><th>Spearman</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
                  </details>
                  <details><summary>Pairwise correlations</summary>
                    <table><thead><tr><th>relation</th><th>Pearson</th><th>Spearman</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table>
                  </details>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="case">
              <h2>{html.escape(case['case_id'])} | {html.escape(case['source'])}</h2>
              <div class="casegrid">
                <figure><video src="{case['input_video']}" controls muted preload="metadata"></video><figcaption>xSSC input 49f</figcaption></figure>
                <figure><video src="{case['flow_video']}" controls muted preload="metadata"></video><figcaption>RAFT flow</figcaption></figure>
              </div>
              <div class="models">{''.join(cards)}</div>
            </section>
            """
        )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC Decoder Static/Dynamic Split</title>
  <style>
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#101214; color:#eef2f7; font:13px system-ui,sans-serif; letter-spacing:0; }}
    header {{ position:sticky; top:0; z-index:10; padding:12px 16px; background:#15191d; border-bottom:1px solid #303942; }}
    h1 {{ margin:0 0 6px; font-size:20px; }}
    h2 {{ margin:0 0 12px; font-size:17px; }}
    h3 {{ margin:0 0 6px; font-size:14px; }}
    main {{ max-width:2100px; margin:0 auto; padding:16px; }}
    .small,.summary {{ color:#bdc7d1; }}
    .case {{ padding:18px 0 28px; border-top:1px solid #30363d; }}
    .casegrid {{ display:grid; grid-template-columns:repeat(2,minmax(0,420px)); gap:12px; margin-bottom:12px; }}
    .models {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(560px,1fr)); gap:12px; }}
    .model {{ border:1px solid #333b44; background:#14191e; padding:10px; border-radius:8px; }}
    .plots {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    figure {{ margin:0; min-width:0; }}
    img,video {{ display:block; width:100%; background:#000; border:1px solid #303942; }}
    figcaption {{ padding:4px 1px; color:#aeb8c2; font-size:11px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
    th,td {{ border:1px solid #303942; padding:5px 7px; text-align:left; }}
    th {{ background:#192027; }}
    td {{ background:#12171c; color:#cbd5df; }}
    details {{ margin-top:8px; }}
    @media(max-width:900px) {{ .models,.casegrid,.plots {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>xSSC decoder static/dynamic split vs slot DC and RAFT</h1>
    <div class="summary">{html.escape(json.dumps(report, ensure_ascii=False))}</div>
  </header>
  <main>
    <p class="small">Official split is applied after <code>memory = decoder.project2(slotz)</code>.
    With <code>rd=0.25</code>, decoder memory <code>[T,S,384]</code> is split into static
    <code>[T,S,288]</code> and dynamic <code>[T,S,96]</code>. RAFT is only an external validation signal.</p>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    metadata = json.loads((input_dir / "metadata.json").read_text())
    checkpoint_by_model = {
        model["name"]: Path(model["checkpoint"])
        for model in metadata["models"]
    }
    cases_out: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for model_name, checkpoint in checkpoint_by_model.items():
        model, _ = base.build_official_model(checkpoint, device)
        for case in metadata["cases"]:
            case_id = case["models"][0]["case_id"]
            case_dir = input_dir / "cases" / case_id
            model_npz = case_dir / model_name / "xssc_slots_attention_analysis.npz"
            arrays = np.load(model_npz)
            slots = arrays["slots"].astype(np.float32)
            attention = arrays["attention"].astype(np.float32)
            flow = np.load(case_dir / "raft_flow.npz")["flow"].astype(np.float32)
            static, dynamic, split_info = decoder_memory_split(model, slots, device)
            analysis = analyze_one(slots, attention, flow, static, dynamic)

            dst_model_dir = output_dir / "cases" / case_id / model_name
            assets = render_assets(dst_model_dir, analysis)
            np.savez_compressed(
                dst_model_dir / "decoder_split_features_analysis.npz",
                decoder_static=static.astype(np.float16),
                decoder_dynamic=dynamic.astype(np.float16),
                global_flow=analysis["global_flow"].astype(np.float32),
                raw_dc_by_slot=np.asarray(analysis["dc_ratio_by_slot"]["raw_slot"], dtype=np.float32),
                static_dc_by_slot=np.asarray(analysis["dc_ratio_by_slot"]["decoder_static"], dtype=np.float32),
                dynamic_dc_by_slot=np.asarray(analysis["dc_ratio_by_slot"]["decoder_dynamic"], dtype=np.float32),
            )
            best_flow = best_by_abs_spearman(analysis["adjacent_vs_flow"])
            dynamic_vs_d = analysis["adjacent_vs_d_xssc"]["D_decoder_dynamic"]["spearman"]
            record = {
                "case_id": case_id,
                "case_index": int(case["index"]),
                "source": str(case["source"]),
                "name": model_name,
                "checkpoint": str(checkpoint),
                "split_info": split_info,
                "best_vs_flow": [best_flow[0], best_flow[1]],
                "dynamic_vs_d_xssc_spearman": dynamic_vs_d,
                "dc_mean": {
                    key: float(np.mean(value))
                    for key, value in analysis["dc_ratio_by_slot"].items()
                },
                "frequency": analysis["frequency"],
                "dc_ratio_by_slot": analysis["dc_ratio_by_slot"],
                "dc_slot_correlations": analysis["dc_slot_correlations"],
                "adjacent_vs_flow": analysis["adjacent_vs_flow"],
                "adjacent_vs_d_xssc": analysis["adjacent_vs_d_xssc"],
                "pair_correlations": analysis["pair_correlations"],
                "assets": assets,
            }
            records.append(record)
            print(
                f"[split] {model_name} {case_id} "
                f"dc raw/static/dyn={record['dc_mean']['raw_slot']:.3f}/"
                f"{record['dc_mean']['decoder_static']:.3f}/"
                f"{record['dc_mean']['decoder_dynamic']:.3f} "
                f"best={best_flow[0]}:{best_flow[1]:.3f} "
                f"dyn_vs_D={dynamic_vs_d:.3f}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    by_case: dict[str, dict[str, Any]] = {}
    for case in metadata["cases"]:
        case_id = case["models"][0]["case_id"]
        links = copy_or_link_static_inputs(input_dir, output_dir, case_id)
        by_case[case_id] = {
            "case_id": case_id,
            "index": int(case["index"]),
            "source": str(case["source"]),
            "input_video": links["xssc_input_49f.mp4"],
            "flow_video": links["raft_flow.mp4"],
            "models": [],
        }
    for record in records:
        by_case[record["case_id"]]["models"].append(record)
    cases_out = list(by_case.values())

    report = {
        "input": str(input_dir),
        "models": list(checkpoint_by_model.keys()),
        "decoder_split": "project2(slotz) [T,S,384] -> static [T,S,288] + dynamic [T,S,96]",
        "note": "This is the official MarkovRarDecoder channel split, not the post-hoc mean-subtracted slot residual.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps({"report": report, "records": records, "cases": cases_out}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_html(output_dir, report, cases_out)

    # Compact text summary for terminal use.
    print("\nSummary", flush=True)
    for key in ["raw_slot", "decoder_static", "decoder_dynamic"]:
        vals = [record["dc_mean"][key] for record in records]
        print(f"dc_mean/{key}={np.mean(vals):.3f}", flush=True)
    for name in ["D_decoder_dynamic", "decoder_dynamic_delta_l2", "decoder_dynamic_one_minus_cos"]:
        vals = [record["adjacent_vs_flow"][name]["spearman"] for record in records]
        print(f"spearman_vs_raft/{name}={np.nanmean(vals):.3f}", flush=True)
    dyn_vs_d = [record["dynamic_vs_d_xssc_spearman"] for record in records]
    print(f"spearman_decoder_dynamic_vs_D_xSSC_feature={np.nanmean(dyn_vs_d):.3f}", flush=True)
    print(f"viewer={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

