#!/usr/bin/env python3
"""Extract full-video xSSC slot embeddings and render temporal cosine curves."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from infer_vjepa_xssc_video_slot_overlay import (
    SLOT_COLORS,
    checkpoint_load_summary,
    load_model,
    set_seed,
)
from visualize_vjepa_xssc_latest_val5_test5 import (
    decode_external_video,
    infer_window,
    load_test5_amg_conditions,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--plots-only", action="store_true")
    return parser.parse_args()


def normalize_embeddings(slotz):
    norms = np.linalg.norm(slotz, axis=-1, keepdims=True)
    return slotz / np.clip(norms, 1e-12, None)


def temporal_similarities(slotz):
    normalized = normalize_embeddings(slotz.astype(np.float64))
    to_t0 = np.einsum("tsd,sd->ts", normalized, normalized[0])
    adjacent = np.full(to_t0.shape, np.nan, dtype=np.float64)
    adjacent[1:] = np.einsum(
        "tsd,tsd->ts", normalized[1:], normalized[:-1]
    )
    return to_t0.astype(np.float32), adjacent.astype(np.float32)


def plot_similarity(values, case_id, step, kind, output_file, dpi):
    steps, slots = values.shape
    x = np.arange(steps)
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.22)
    for slot in range(slots):
        color = SLOT_COLORS[slot % len(SLOT_COLORS)] / 255.0
        linestyle = "-" if slot < len(SLOT_COLORS) else "--"
        ax.plot(
            x,
            values[:, slot],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            alpha=0.92,
            label=f"S{slot}",
        )
    ax.set_xlim(0, max(1, steps - 1))
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel("xSSC tubelet step t  (2 consecutive raw frames per step)")
    ax.set_ylabel("cosine similarity")
    ax.grid(True, color="#CBD5E1", linewidth=0.65, alpha=0.48)
    ax.axhline(0.0, color="#64748B", linewidth=0.9, alpha=0.7)
    ax.axhline(1.0, color="#334155", linewidth=0.9, alpha=0.55)
    reference = (
        "cos(slot[t,s], slot[0,s]); t0 represents raw frames f0–f1"
        if kind == "to_t0"
        else "cos(slot[t,s], slot[t−1,s]); t0 has no previous step"
    )
    title = f"step-{step} · {reference}\n{textwrap.shorten(case_id, width=150)}"
    ax.set_title(title, fontsize=13, pad=12)
    secondary = ax.secondary_xaxis(
        "top", functions=(lambda value: value * 2, lambda value: value / 2)
    )
    secondary.set_xlabel("raw-frame start index of each tubelet")
    fig.legend(
        ncol=min(11, slots),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        frameon=False,
        fontsize=9,
        handlelength=2.5,
    )
    fig.text(
        0.995,
        0.025,
        "Colors exactly match slot overlay; S7–S10 reuse the modulo-7 overlay "
        "palette and are dashed.",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#475569",
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    report_file = args.report.resolve()
    report = json.loads(report_file.read_text())
    output_dir = report_file.parent
    if args.plots_only:
        cases = [case for case in report["cases"] if case["source"] == "test5"]
        for ordinal, case in enumerate(cases, start=1):
            case_dir = output_dir / "cases" / case["case_id"]
            data = np.load(case_dir / "slot_embedding_similarity.npz")
            plot_similarity(
                data["cosine_to_t0"],
                case["case_id"],
                report["latest_complete_step"],
                "to_t0",
                case_dir / "slot_similarity_to_t0.png",
                args.dpi,
            )
            plot_similarity(
                data["cosine_adjacent"],
                case["case_id"],
                report["latest_complete_step"],
                "adjacent",
                case_dir / "slot_similarity_adjacent.png",
                args.dpi,
            )
            print(f"[plot-only] {ordinal}/{len(cases)} {case['case_id']}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    cfg, model, load_report = load_model(
        Path(report["config"]), Path(report["checkpoint"]), device
    )
    summary = checkpoint_load_summary(load_report)
    step = summary["source_optimizer_step"]
    if step != report["latest_complete_step"]:
        raise RuntimeError(
            f"checkpoint/report step mismatch: {step} != {report['latest_complete_step']}"
        )
    uses_bbox_condition = "condit" in cfg.model_imap
    conditions = {}
    if uses_bbox_condition:
        conditions, _ = load_test5_amg_conditions(
            Path(report["test5_amg_metadata"]), int(cfg.max_num)
        )

    cases = [case for case in report["cases"] if case["source"] == "test5"]
    for ordinal, case in enumerate(cases, start=1):
        source = str(Path(case["source_key"]).resolve())
        frames, _, _, _ = decode_external_video(Path(source), cfg)
        condition = conditions.get(source)
        if uses_bbox_condition and condition is None:
            raise KeyError(f"missing cached first-frame condition: {source}")
        _, slotz, attention_shape, original_count = infer_window(
            model, frames, condition, device, getattr(torch, cfg.amp_dtype)
        )
        if original_count != case["frames"]:
            raise RuntimeError(
                f"frame count changed for {case['case_id']}: "
                f"{original_count} != {case['frames']}"
            )
        if list(attention_shape) != case["attention_shapes"][0]:
            raise RuntimeError(
                f"attention shape changed for {case['case_id']}: "
                f"{attention_shape} != {case['attention_shapes'][0]}"
            )
        to_t0, adjacent = temporal_similarities(slotz)
        tubelet_count = len(slotz)
        raw_frame_pairs = np.stack(
            [
                np.arange(tubelet_count, dtype=np.int32) * 2,
                np.minimum(
                    np.arange(tubelet_count, dtype=np.int32) * 2 + 1,
                    original_count - 1,
                ),
            ],
            axis=1,
        )
        case_dir = output_dir / "cases" / case["case_id"]
        data_file = case_dir / "slot_embedding_similarity.npz"
        np.savez_compressed(
            data_file,
            slot_embeddings=slotz.astype(np.float32),
            cosine_to_t0=to_t0,
            cosine_adjacent=adjacent,
            tubelet_indices=np.arange(tubelet_count, dtype=np.int32),
            raw_frame_pairs=raw_frame_pairs,
            slot_color_rgb=SLOT_COLORS[
                np.arange(slotz.shape[1]) % len(SLOT_COLORS)
            ],
        )
        to_t0_file = case_dir / "slot_similarity_to_t0.png"
        adjacent_file = case_dir / "slot_similarity_adjacent.png"
        plot_similarity(
            to_t0, case["case_id"], step, "to_t0", to_t0_file, args.dpi
        )
        plot_similarity(
            adjacent, case["case_id"], step, "adjacent", adjacent_file, args.dpi
        )
        case["assets"]["slot_similarity_data"] = (
            f"cases/{case['case_id']}/{data_file.name}"
        )
        case["assets"]["slot_similarity_to_t0"] = (
            f"cases/{case['case_id']}/{to_t0_file.name}"
        )
        case["assets"]["slot_similarity_adjacent"] = (
            f"cases/{case['case_id']}/{adjacent_file.name}"
        )
        case["slot_embedding_shape"] = list(slotz.shape)
        case["slot_similarity_time_axis"] = {
            "unit": "V-JEPA/xSSC tubelet",
            "raw_frames_per_step": 2,
            "t0_raw_frames": [0, min(1, original_count - 1)],
        }
        print(
            f"[slot-sim] {ordinal}/{len(cases)} {case['case_id']} "
            f"shape={tuple(slotz.shape)}",
            flush=True,
        )

    report["slot_embedding_similarity"] = {
        "scope": "complete-video single-forward test_5 cases",
        "embedding": "output.slotz before visualization",
        "to_t0": "cos(slot[t,s], slot[0,s])",
        "adjacent": "cos(slot[t,s], slot[t-1,s]); t=0 undefined",
        "time_axis": "one V-JEPA tubelet step = two consecutive raw frames",
        "color_policy": (
            "exact overlay palette SLOT_COLORS[slot_id % 7]; "
            "S7-S10 use dashed lines"
        ),
    }
    temporary = report_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(report_file)
    print(
        json.dumps(
            {"report": str(report_file), "step": step, "cases": len(cases)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
