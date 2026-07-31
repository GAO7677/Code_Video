#!/usr/bin/env python3
"""Rank xSSC-only dynamics proxies on full training videos.

RAFT is used only as an external validation signal. The candidate proxy
functions are computed from official xSSC slots, slot attention, and the
official MarkovRarDecoder static/dynamic channel split.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import random
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
import analyze_official_xssc_decoder_split_dynamics as split_base  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("/data/gaoya/agent-data/outputs/xssc_dynamics_proxy_train_subset_n90")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=Path, default=base.DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--official-root", type=Path, default=Path("/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--xssc-input-size", type=int, default=256)
    parser.add_argument("--xssc-batch-size", type=int, default=16)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--samples-per-source", type=int, default=30)
    parser.add_argument("--indices", default="")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--visualize-samples", type=int, default=9)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def sample_indices_by_source(dataset: Any, samples_per_source: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if not all(hasattr(dataset, name) for name in ["source_names", "source_lengths", "cumulative_lengths"]):
        total = min(int(samples_per_source), len(dataset))
        indices = rng.choice(len(dataset), size=total, replace=False)
        return [
            {"index": int(index), "source": "unknown", "source_local_index": int(index)}
            for index in sorted(indices.tolist())
        ]

    selected: list[dict[str, Any]] = []
    previous = 0
    for source_id, (source_name, source_length) in enumerate(zip(dataset.source_names, dataset.source_lengths)):
        count = min(int(samples_per_source), int(source_length))
        local = rng.choice(int(source_length), size=count, replace=False)
        for value in sorted(local.tolist()):
            selected.append(
                {
                    "index": int(previous + value),
                    "source": str(source_name),
                    "source_id": int(source_id),
                    "source_local_index": int(value),
                }
            )
        previous += int(source_length)
    return sorted(selected, key=lambda item: (item["source"], item["source_local_index"]))


def parse_indices(indices: str) -> list[dict[str, Any]]:
    values = [int(item) for item in str(indices).replace(",", " ").split() if item.strip()]
    return [{"index": int(value), "source": "manual", "source_local_index": int(value)} for value in values]


def safe_case_id(position: int, index: int, source: str) -> str:
    return f"case_{position:04d}_{base.safe_id(source)}_idx_{int(index):06d}"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_attention(attention: np.ndarray) -> np.ndarray:
    return attention / np.maximum(attention.sum(axis=(2, 3), keepdims=True), 1e-12)


def slot_flow_from_attention(flow: np.ndarray, attention: np.ndarray) -> np.ndarray:
    flow_mag = np.linalg.norm(flow, axis=-1)
    low = base.downsample_flow_mag(flow_mag, tuple(attention.shape[-2:]))
    attn = normalize_attention(attention[:-1])
    return (attn * low[:, None]).sum(axis=(2, 3))


def attention_metrics(attention: np.ndarray) -> dict[str, np.ndarray]:
    delta_l1 = np.abs(attention[1:] - attention[:-1]).mean(axis=(2, 3))
    centroids = base.attention_centroids(attention)
    centroid_shift = np.linalg.norm(centroids[1:] - centroids[:-1], axis=-1)
    return {
        "attention_delta_l1": delta_l1,
        "attention_centroid_shift": centroid_shift,
    }


def combined_mean_slot(*items: np.ndarray) -> np.ndarray:
    return combined_slot(*items).mean(axis=1)


def combined_slot(*items: np.ndarray) -> np.ndarray:
    return np.stack([base.zscore(item) for item in items], axis=0).sum(axis=0)


def pair_distance(features: np.ndarray) -> np.ndarray:
    return 1.0 - base.slot_cosine_pairwise(features)


@torch.inference_mode()
def decoder_split(model: torch.nn.Module, slots: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(slots).to(device=device, dtype=torch.float32)
    memory = model.decode.project2(tensor)
    split_static = int(memory.size(-1) * (1.0 - float(model.decode.rd)))
    static = memory[..., :split_static].detach().float().cpu().numpy()
    dynamic = memory[..., split_static:].detach().float().cpu().numpy()
    return static, dynamic


def compute_candidate_record(
    model: torch.nn.Module,
    slots: np.ndarray,
    attention: np.ndarray,
    flow: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    time_steps = slots.shape[0]
    global_flow = np.linalg.norm(flow, axis=-1).mean(axis=(1, 2))
    slot_flow = slot_flow_from_attention(flow, attention)
    pair_flow = split_base.cumulative_flow(global_flow, time_steps)
    triu = np.triu_indices(time_steps, k=1)

    raw = split_base.adjacent_metrics(slots)
    slot_residual = slots - slots.mean(axis=0, keepdims=True)
    decoder_static, decoder_dynamic = decoder_split(model, slots, device)
    stat = split_base.adjacent_metrics(decoder_static)
    dyn = split_base.adjacent_metrics(decoder_dynamic)
    attn = attention_metrics(attention)
    combo_raw_decoder_delta = combined_slot(raw["delta_l2"], dyn["delta_l2"])
    combo_raw_decoder_cos = combined_slot(raw["delta_l2"], dyn["one_minus_cos"])
    combo_raw_residual_decoder = combined_slot(
        raw["delta_l2"],
        raw["residual_one_minus_cos"],
        dyn["delta_l2"],
    )

    global_candidates: dict[str, np.ndarray] = {
        "raw_feature_delta_l2": raw["delta_l2"].mean(axis=1),
        "raw_one_minus_cos": raw["one_minus_cos"].mean(axis=1),
        "posthoc_residual_one_minus_cos": raw["residual_one_minus_cos"].mean(axis=1),
        "posthoc_residual_energy_mid": raw["residual_energy_mid"].mean(axis=1),
        "attention_delta_l1": attn["attention_delta_l1"].mean(axis=1),
        "attention_centroid_shift": attn["attention_centroid_shift"].mean(axis=1),
        "D_residual_centroid": split_base.xssc_residual_motion_from_slots(slots, attention),
        "D_feature_residual": split_base.xssc_feature_motion_from_slots(slots),
        "decoder_static_delta_l2": stat["delta_l2"].mean(axis=1),
        "decoder_static_one_minus_cos": stat["one_minus_cos"].mean(axis=1),
        "decoder_dynamic_delta_l2": dyn["delta_l2"].mean(axis=1),
        "decoder_dynamic_one_minus_cos": dyn["one_minus_cos"].mean(axis=1),
        "decoder_dynamic_residual_one_minus_cos": dyn["residual_one_minus_cos"].mean(axis=1),
        "decoder_dynamic_residual_energy_mid": dyn["residual_energy_mid"].mean(axis=1),
        "D_decoder_dynamic": combined_mean_slot(dyn["delta_l2"], dyn["one_minus_cos"]),
        "D_decoder_dynamic_residual": combined_mean_slot(dyn["residual_one_minus_cos"], dyn["residual_energy_mid"]),
        "D_raw_decoder_delta": combo_raw_decoder_delta.mean(axis=1),
        "D_raw_decoder_cos": combo_raw_decoder_cos.mean(axis=1),
        "D_raw_residual_decoder_delta": combo_raw_residual_decoder.mean(axis=1),
    }
    slot_candidates: dict[str, np.ndarray] = {
        "raw_feature_delta_l2": raw["delta_l2"],
        "raw_one_minus_cos": raw["one_minus_cos"],
        "posthoc_residual_one_minus_cos": raw["residual_one_minus_cos"],
        "posthoc_residual_energy_mid": raw["residual_energy_mid"],
        "attention_delta_l1": attn["attention_delta_l1"],
        "attention_centroid_shift": attn["attention_centroid_shift"],
        "decoder_static_delta_l2": stat["delta_l2"],
        "decoder_static_one_minus_cos": stat["one_minus_cos"],
        "decoder_dynamic_delta_l2": dyn["delta_l2"],
        "decoder_dynamic_one_minus_cos": dyn["one_minus_cos"],
        "decoder_dynamic_residual_one_minus_cos": dyn["residual_one_minus_cos"],
        "decoder_dynamic_residual_energy_mid": dyn["residual_energy_mid"],
        "D_raw_decoder_delta": combo_raw_decoder_delta,
        "D_raw_decoder_cos": combo_raw_decoder_cos,
        "D_raw_residual_decoder_delta": combo_raw_residual_decoder,
    }
    raw_pair = pair_distance(slots)
    residual_pair = pair_distance(slot_residual)
    decoder_static_pair = pair_distance(decoder_static)
    decoder_dynamic_pair = pair_distance(decoder_dynamic)
    pair_candidates: dict[str, np.ndarray] = {
        "raw_pair_distance": raw_pair,
        "posthoc_residual_pair_distance": residual_pair,
        "decoder_static_pair_distance": decoder_static_pair,
        "decoder_dynamic_pair_distance": decoder_dynamic_pair,
        "combo_raw_decoder_dynamic_pair_distance": 0.5 * (raw_pair + decoder_dynamic_pair),
        "combo_residual_decoder_dynamic_pair_distance": 0.5 * (residual_pair + decoder_dynamic_pair),
    }

    adjacent_vs_flow = {
        name: {
            "pearson": base.pearson(values, global_flow),
            "spearman": base.spearman(values, global_flow),
        }
        for name, values in global_candidates.items()
    }
    slot_vs_flow = {
        name: {
            "pearson": base.pearson(values, slot_flow),
            "spearman": base.spearman(values, slot_flow),
        }
        for name, values in slot_candidates.items()
    }
    pair_vs_flow = {
        name: {
            "pearson": base.pearson(values[triu], pair_flow[triu]),
            "spearman": base.spearman(values[triu], pair_flow[triu]),
        }
        for name, values in pair_candidates.items()
    }
    xssc_internal = {
        name: {
            "pearson": base.pearson(values, global_candidates["D_feature_residual"]),
            "spearman": base.spearman(values, global_candidates["D_feature_residual"]),
        }
        for name, values in global_candidates.items()
        if name != "D_feature_residual"
    }
    return {
        "global_flow_mean": float(global_flow.mean()),
        "global_flow_std": float(global_flow.std()),
        "adjacent_vs_flow": adjacent_vs_flow,
        "slot_vs_flow": slot_vs_flow,
        "pair_vs_flow": pair_vs_flow,
        "xssc_internal": xssc_internal,
        "curves": {name: values.astype(np.float32) for name, values in global_candidates.items()},
        "global_flow": global_flow.astype(np.float32),
    }


def load_existing_records(records_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not records_path.is_file():
        return {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            records[(str(item["case_id"]), str(item["model_name"]))] = item
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_names = sorted({name for record in records for name in record["adjacent_vs_flow"]})
    pair_names = sorted({name for record in records for name in record["pair_vs_flow"]})
    slot_names = sorted({name for record in records for name in record["slot_vs_flow"]})
    rows = []
    for name in candidate_names:
        adjacent = np.asarray([record["adjacent_vs_flow"][name]["spearman"] for record in records], dtype=np.float64)
        adjacent = adjacent[np.isfinite(adjacent)]
        slot_match = name if name in slot_names else None
        slot = (
            np.asarray([record["slot_vs_flow"][slot_match]["spearman"] for record in records], dtype=np.float64)
            if slot_match is not None
            else np.asarray([], dtype=np.float64)
        )
        slot = slot[np.isfinite(slot)]
        pair_match = {
            "raw_feature_delta_l2": "raw_pair_distance",
            "raw_one_minus_cos": "raw_pair_distance",
            "posthoc_residual_one_minus_cos": "posthoc_residual_pair_distance",
            "posthoc_residual_energy_mid": "posthoc_residual_pair_distance",
            "decoder_static_delta_l2": "decoder_static_pair_distance",
            "decoder_static_one_minus_cos": "decoder_static_pair_distance",
            "decoder_dynamic_delta_l2": "decoder_dynamic_pair_distance",
            "decoder_dynamic_one_minus_cos": "decoder_dynamic_pair_distance",
            "decoder_dynamic_residual_one_minus_cos": "decoder_dynamic_pair_distance",
            "decoder_dynamic_residual_energy_mid": "decoder_dynamic_pair_distance",
            "D_decoder_dynamic": "decoder_dynamic_pair_distance",
            "D_decoder_dynamic_residual": "decoder_dynamic_pair_distance",
            "D_feature_residual": "posthoc_residual_pair_distance",
            "D_residual_centroid": "posthoc_residual_pair_distance",
            "D_raw_decoder_delta": "combo_raw_decoder_dynamic_pair_distance",
            "D_raw_decoder_cos": "combo_raw_decoder_dynamic_pair_distance",
            "D_raw_residual_decoder_delta": "combo_residual_decoder_dynamic_pair_distance",
        }.get(name)
        pair = (
            np.asarray([record["pair_vs_flow"][pair_match]["spearman"] for record in records], dtype=np.float64)
            if pair_match in pair_names
            else np.asarray([], dtype=np.float64)
        )
        pair = pair[np.isfinite(pair)]

        mean_adj = float(np.mean(adjacent)) if len(adjacent) else float("nan")
        median_adj = float(np.median(adjacent)) if len(adjacent) else float("nan")
        std_adj = float(np.std(adjacent)) if len(adjacent) else float("nan")
        positive_rate = float(np.mean(adjacent > 0.0)) if len(adjacent) else float("nan")
        mean_slot = float(np.mean(slot)) if len(slot) else float("nan")
        mean_pair = float(np.mean(pair)) if len(pair) else float("nan")
        score = (
            0.50 * mean_adj
            + 0.20 * (0.0 if not np.isfinite(mean_slot) else mean_slot)
            + 0.20 * (0.0 if not np.isfinite(mean_pair) else mean_pair)
            + 0.10 * positive_rate
            - 0.10 * std_adj
        )
        source_means: dict[str, float] = {}
        for source in sorted({str(record["source"]) for record in records}):
            vals = np.asarray(
                [
                    record["adjacent_vs_flow"][name]["spearman"]
                    for record in records
                    if str(record["source"]) == source
                ],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            source_means[source] = float(np.mean(vals)) if len(vals) else float("nan")
        model_means: dict[str, float] = {}
        for model_name in sorted({str(record["model_name"]) for record in records}):
            vals = np.asarray(
                [
                    record["adjacent_vs_flow"][name]["spearman"]
                    for record in records
                    if str(record["model_name"]) == model_name
                ],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            model_means[model_name] = float(np.mean(vals)) if len(vals) else float("nan")
        rows.append(
            {
                "candidate": name,
                "proxy_score": float(score),
                "adjacent_spearman_mean": mean_adj,
                "adjacent_spearman_median": median_adj,
                "adjacent_spearman_std": std_adj,
                "adjacent_positive_rate": positive_rate,
                "slot_spearman_mean": mean_slot,
                "pair_spearman_mean": mean_pair,
                "source_adjacent_means": source_means,
                "model_adjacent_means": model_means,
            }
        )
    rows = sorted(rows, key=lambda item: item["proxy_score"], reverse=True)
    pair_rows = []
    for name in pair_names:
        vals = np.asarray([record["pair_vs_flow"][name]["spearman"] for record in records], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        pair_rows.append(
            {
                "candidate": name,
                "spearman_mean": float(np.mean(vals)) if len(vals) else float("nan"),
                "spearman_median": float(np.median(vals)) if len(vals) else float("nan"),
                "spearman_std": float(np.std(vals)) if len(vals) else float("nan"),
            }
        )
    pair_rows = sorted(pair_rows, key=lambda item: item["spearman_mean"], reverse=True)
    return {"ranking": rows, "pair_ranking": pair_rows}


def write_ranking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate",
        "proxy_score",
        "adjacent_spearman_mean",
        "adjacent_spearman_median",
        "adjacent_spearman_std",
        "adjacent_positive_rate",
        "slot_spearman_mean",
        "pair_spearman_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_top_candidates(path: Path, summary: dict[str, Any], top_k: int = 12) -> None:
    rows = summary["ranking"][:top_k]
    labels = [row["candidate"] for row in rows][::-1]
    scores = [row["proxy_score"] for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(8.6, max(4.2, 0.34 * len(rows))), dpi=150)
    ax.barh(np.arange(len(labels)), scores, color="#4f8cff")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("proxy score")
    ax.set_title("Top xSSC-only dynamics proxies", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_source_heatmap(path: Path, summary: dict[str, Any], top_k: int = 12) -> None:
    rows = summary["ranking"][:top_k]
    sources = sorted({source for row in rows for source in row["source_adjacent_means"]})
    matrix = np.asarray([[row["source_adjacent_means"].get(source, np.nan) for source in sources] for row in rows])
    fig, ax = plt.subplots(figsize=(1.7 * len(sources) + 4.0, max(4.0, 0.34 * len(rows))), dpi=150)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(sources)))
    ax.set_xticklabels(sources, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["candidate"] for row in rows], fontsize=8)
    ax.set_title("Adjacent Spearman vs RAFT by source", fontsize=11)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            ax.text(x, y, "nan" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_case_curves(path: Path, global_flow: np.ndarray, curves: dict[str, np.ndarray], title: str, top_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.6), dpi=150)
    ax.plot(base.zscore(global_flow), label="RAFT global flow", color="#111827", linewidth=2.2)
    for name in top_names:
        if name in curves:
            ax.plot(base.zscore(curves[name]), label=name, linewidth=1.1)
    ax.set_xlabel("adjacent transition t -> t+1")
    ax.set_ylabel("z-score")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_html(output_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]], visual_items: list[dict[str, Any]]) -> None:
    ranking_rows = []
    for row in summary["ranking"]:
        sources = " / ".join(f"{k}:{v:.3f}" for k, v in row["source_adjacent_means"].items())
        ranking_rows.append(
            f"<tr><td>{html.escape(row['candidate'])}</td><td>{row['proxy_score']:.3f}</td>"
            f"<td>{row['adjacent_spearman_mean']:.3f}</td><td>{row['adjacent_spearman_std']:.3f}</td>"
            f"<td>{row['slot_spearman_mean']:.3f}</td><td>{row['pair_spearman_mean']:.3f}</td>"
            f"<td>{row['adjacent_positive_rate']:.2f}</td><td>{html.escape(sources)}</td></tr>"
        )
    visuals = []
    for item in visual_items:
        visuals.append(
            f"""
            <article class="visual">
              <h3>{html.escape(item['title'])}</h3>
              <div class="visualgrid">
                <figure><video src="{item['input_video']}" controls muted preload="metadata"></video><figcaption>49-frame xSSC input</figcaption></figure>
                <figure><video src="{item['flow_video']}" controls muted preload="metadata"></video><figcaption>RAFT adjacent flow</figcaption></figure>
                <figure><img src="{item['curve_plot']}" loading="lazy"><figcaption>top proxy curves vs RAFT</figcaption></figure>
              </div>
            </article>
            """
        )
    top = summary["ranking"][0] if summary["ranking"] else {}
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC Dynamics Proxy Ranking</title>
  <style>
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#101214; color:#eef2f7; font:13px system-ui,sans-serif; letter-spacing:0; }}
    header {{ position:sticky; top:0; z-index:10; background:#15191d; border-bottom:1px solid #303942; padding:12px 16px; }}
    main {{ max-width:1800px; margin:0 auto; padding:16px; }}
    h1 {{ margin:0 0 6px; font-size:20px; }}
    h2 {{ margin:18px 0 10px; font-size:17px; }}
    h3 {{ margin:0 0 8px; font-size:14px; }}
    .small {{ color:#bdc7d1; }}
    .plots {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin:12px 0; }}
    .visual {{ border-top:1px solid #303942; padding:14px 0; }}
    .visualgrid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    img,video {{ display:block; width:100%; background:#000; border:1px solid #303942; }}
    figure {{ margin:0; }}
    figcaption {{ padding:4px 1px; color:#aeb8c2; font-size:11px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
    th,td {{ border:1px solid #303942; padding:5px 7px; text-align:left; vertical-align:top; }}
    th {{ background:#192027; }}
    td {{ background:#12171c; color:#cbd5df; }}
    code {{ color:#d5f5ff; }}
    @media(max-width:950px) {{ .plots,.visualgrid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>xSSC-only dynamics proxy ranking on training full videos</h1>
    <div class="small">records={len(records)}; best candidate={html.escape(str(top.get('candidate', 'n/a')))}; score={top.get('proxy_score', float('nan')):.3f}</div>
  </header>
  <main>
    <p class="small">Proxy score = 0.50 adjacent Spearman vs RAFT + 0.20 slot-local Spearman + 0.20 pairwise cumulative-flow Spearman + 0.10 positive-rate - 0.10 adjacent std. RAFT is not used by the proxy itself; it is only the external validation signal.</p>
    <div class="plots">
      <figure><img src="top_proxy_scores.png" loading="lazy"><figcaption>ranking score</figcaption></figure>
      <figure><img src="source_heatmap.png" loading="lazy"><figcaption>source-wise adjacent correlation</figcaption></figure>
    </div>
    <h2>Ranking</h2>
    <table>
      <thead><tr><th>candidate</th><th>score</th><th>adj mean</th><th>adj std</th><th>slot mean</th><th>pair mean</th><th>positive</th><th>source adj means</th></tr></thead>
      <tbody>{''.join(ranking_rows)}</tbody>
    </table>
    <h2>Representative Curves</h2>
    {''.join(visuals)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset_args, object_train = base.load_dataset_args(args.train_config)
    dataset = object_train.base.build_dataset(dataset_args)
    selected = parse_indices(args.indices) if args.indices.strip() else sample_indices_by_source(dataset, args.samples_per_source, args.seed)
    for position, item in enumerate(selected, start=1):
        item["case_id"] = safe_case_id(position, item["index"], item["source"])

    checkpoints = sorted(args.official_root.expanduser().resolve().glob("*.pth"))
    if len(checkpoints) != 3:
        raise RuntimeError(f"Expected 3 official xSSC weights, found {len(checkpoints)} under {args.official_root}")

    records_path = output_dir / "records.jsonl"
    if records_path.exists() and not args.resume:
        records_path.unlink()
    existing = load_existing_records(records_path) if args.resume else {}
    visual_case_ids = {item["case_id"] for item in selected[: max(0, int(args.visualize_samples))]}

    model_entries = []
    for checkpoint in checkpoints:
        model_name = f"official_{checkpoint.stem}"
        model, _ = base.build_official_model(checkpoint, device)
        model_entries.append((model_name, checkpoint, model))

    raft = base.build_raft(device, args.raft_iters)
    visual_items = []
    with records_path.open("a", encoding="utf-8"):
        pass

    for position, item in enumerate(selected, start=1):
        index = int(item["index"])
        case_id = str(item["case_id"])
        source = str(item["source"])
        sample_seed = int(args.seed) + index * 17 + position * 9973
        set_all_seeds(sample_seed)
        sample = dataset[index]
        metadata = dict(sample.get("metadata", {}))
        source = str(metadata.get("dataset_source", source))
        item["source"] = source
        video = sample.get("video", sample["context_video"])[:, : args.num_frames]
        normalized, rgb = base.preprocess_video_for_xssc(video, args.xssc_input_size)
        flow = base.compute_raft_flow(raft, rgb, device, args.raft_iters)
        is_visual = case_id in visual_case_ids
        visual_dir = output_dir / "visual_cases" / case_id
        if is_visual:
            visual_dir.mkdir(parents=True, exist_ok=True)
            input_video = visual_dir / "xssc_input_49f.mp4"
            flow_video = visual_dir / "raft_flow.mp4"
            if not input_video.is_file():
                base.write_video(input_video, rgb, fps=8.0)
            if not flow_video.is_file():
                flow_rgb = np.stack([base.flow_to_rgb(frame_flow) for frame_flow in flow], axis=0)
                base.write_video(flow_video, flow_rgb, fps=8.0)

        print(f"[case] {position}/{len(selected)} {case_id} source={source} flow={flow.shape}", flush=True)
        for model_position, (model_name, checkpoint, model) in enumerate(model_entries, start=1):
            key = (case_id, model_name)
            if key in existing:
                continue
            seed = int(args.seed) + position * 1000 + model_position * 100 + int(checkpoint.stem.split("-")[0])
            slots, attention = base.extract_official_slots(
                model,
                normalized,
                device,
                seed=seed,
                batch_size=args.xssc_batch_size,
            )
            slots_np = slots.numpy().astype(np.float32)
            attention_np = attention.numpy().astype(np.float32)
            result = compute_candidate_record(model, slots_np, attention_np, flow, device)
            record = {
                "case_position": int(position),
                "case_id": case_id,
                "case_index": index,
                "source": source,
                "model_name": model_name,
                "checkpoint": str(checkpoint),
                "seed": int(seed),
                "sample_seed": int(sample_seed),
                "global_flow_mean": result["global_flow_mean"],
                "global_flow_std": result["global_flow_std"],
                "adjacent_vs_flow": result["adjacent_vs_flow"],
                "slot_vs_flow": result["slot_vs_flow"],
                "pair_vs_flow": result["pair_vs_flow"],
                "xssc_internal": result["xssc_internal"],
                "metadata": {
                    "sample_key": metadata.get("sample_key", metadata.get("case_id", "")),
                    "dataset_source": source,
                },
            }
            append_record(records_path, record)
            existing[key] = record
            print(
                f"[model] {model_name} {case_id} "
                f"D_feat={record['adjacent_vs_flow']['D_feature_residual']['spearman']:.3f} "
                f"D_dec={record['adjacent_vs_flow']['D_decoder_dynamic']['spearman']:.3f}",
                flush=True,
            )

            if is_visual and model_position == 1:
                curves_path = visual_dir / "curves_model_official_42-0130.png"
                plot_case_curves(
                    curves_path,
                    result["global_flow"],
                    result["curves"],
                    f"{case_id} | {source}",
                    [
                        "D_raw_decoder_delta",
                        "D_raw_residual_decoder_delta",
                        "D_feature_residual",
                        "D_decoder_dynamic",
                        "decoder_dynamic_delta_l2",
                        "raw_feature_delta_l2",
                        "attention_centroid_shift",
                    ],
                )
                visual_items.append(
                    {
                        "title": f"{case_id} | {source} | official_42-0130",
                        "input_video": str(input_video.relative_to(output_dir)),
                        "flow_video": str(flow_video.relative_to(output_dir)),
                        "curve_plot": str(curves_path.relative_to(output_dir)),
                    }
                )

        del normalized, rgb, flow
        if device.type == "cuda":
            torch.cuda.empty_cache()

    records = list(existing.values())
    summary = summarize_records(records)
    write_ranking_csv(output_dir / "proxy_ranking.csv", summary["ranking"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    (output_dir / "selected_indices.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    plot_top_candidates(output_dir / "top_proxy_scores.png", summary)
    plot_source_heatmap(output_dir / "source_heatmap.png", summary)
    write_html(output_dir, summary, records, visual_items)

    top = summary["ranking"][0]
    print(
        f"[done] records={len(records)} best={top['candidate']} score={top['proxy_score']:.3f} "
        f"adj={top['adjacent_spearman_mean']:.3f} slot={top['slot_spearman_mean']:.3f} pair={top['pair_spearman_mean']:.3f}",
        flush=True,
    )
    print(f"viewer={output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
