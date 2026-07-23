#!/usr/bin/env python3
"""Measure xSSC slot redundancy before and after per-frame mean removal."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from analyze_slot_temporal_similarity_viewer import (
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_VIEWER_DIR,
    boxes_from_metadata,
    build_model,
    infer_slots,
    load_specs,
    normalize_rgb_frames,
    read_frame_sequence,
)


DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
    "restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/"
    "step-034000.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--latest-movic-checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def pairwise_cosine(values: np.ndarray) -> dict:
    norms = np.linalg.norm(values, axis=-1)
    scale = np.maximum(norms.max(axis=1, keepdims=True), 1e-12)
    valid = norms > scale * 1e-6
    normalized = values / np.maximum(norms[..., None], 1e-12)
    matrices = np.einsum(
        "tsc,tuc->tsu", normalized, normalized, optimize=True
    )
    num_slots = values.shape[1]
    offdiag = ~np.eye(num_slots, dtype=bool)
    signed: list[float] = []
    absolute: list[float] = []
    valid_pair_count = 0
    for frame_id in range(values.shape[0]):
        pair_mask = valid[frame_id, :, None] & valid[frame_id, None, :] & offdiag
        frame_values = matrices[frame_id][pair_mask]
        if frame_values.size:
            signed.append(float(frame_values.mean()))
            absolute.append(float(np.abs(frame_values).mean()))
            valid_pair_count += int(frame_values.size)
    return {
        "signed_offdiag_mean": finite_mean(signed),
        "absolute_offdiag_mean": finite_mean(absolute),
        "valid_pair_count": valid_pair_count,
        "valid_slot_fraction": float(valid.mean()),
    }


def effective_rank(values: np.ndarray) -> dict:
    entropy_ranks: list[float] = []
    participation_ranks: list[float] = []
    for frame in values:
        singular_values = np.linalg.svd(frame, compute_uv=False)
        energy = singular_values**2
        total = energy.sum()
        if total <= 1e-12:
            continue
        probability = energy / total
        nonzero = probability > 1e-12
        entropy_ranks.append(
            float(np.exp(-(probability[nonzero] * np.log(probability[nonzero])).sum()))
        )
        participation_ranks.append(float(1.0 / np.square(probability).sum()))
    return {
        "entropy_rank_mean": finite_mean(entropy_ranks),
        "participation_rank_mean": finite_mean(participation_ranks),
        "valid_frames": len(entropy_ranks),
    }


def phase_coherence(values: np.ndarray) -> dict:
    time_steps, num_slots, _ = values.shape
    centered = values - values.mean(axis=0, keepdims=True)
    window = np.hanning(time_steps)[:, None, None]
    spectrum = np.fft.rfft(centered * window, axis=0)[1:]
    amplitude = np.abs(spectrum)
    energy = np.square(amplitude).sum(axis=(0, 2))
    active = energy > max(float(energy.max()) * 1e-10, 1e-12)
    unit_phase = spectrum / np.maximum(amplitude, np.finfo(np.float64).eps)
    coherences: list[float] = []
    for slot_a in range(num_slots):
        if not active[slot_a]:
            continue
        for slot_b in range(slot_a + 1, num_slots):
            if not active[slot_b]:
                continue
            weights = amplitude[:, slot_a] * amplitude[:, slot_b]
            denominator = float(weights.sum())
            if denominator <= 1e-12:
                continue
            phase_delta = (
                unit_phase[:, slot_a] * unit_phase[:, slot_b].conjugate()
            )
            coherences.append(float(np.abs((weights * phase_delta).sum()) / denominator))
    return {
        "mean": finite_mean(coherences),
        "valid_unordered_pairs": len(coherences),
        "active_slots": int(active.sum()),
    }


def analyze_slots(slots: np.ndarray) -> dict:
    values = slots.astype(np.float64)
    common = values.mean(axis=1, keepdims=True)
    residual = values - common
    common_energy = float(values.shape[1] * np.square(common).sum())
    residual_energy = float(np.square(residual).sum())
    total_energy = common_energy + residual_energy
    raw_cosine = pairwise_cosine(values)
    residual_cosine = pairwise_cosine(residual)
    return {
        "frames": int(values.shape[0]),
        "slots": int(values.shape[1]),
        "slot_dim": int(values.shape[2]),
        "common_energy_ratio": common_energy / max(total_energy, 1e-12),
        "residual_energy_ratio": residual_energy / max(total_energy, 1e-12),
        "raw_cosine": raw_cosine,
        "residual_cosine": residual_cosine,
        "residual_rank": effective_rank(residual),
        "raw_phase_coherence": phase_coherence(values),
        "residual_phase_coherence": phase_coherence(residual),
    }


def aggregate(records: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["label"], record["mode"]), []).append(record)
    rows = []
    for (label, mode), items in sorted(groups.items()):
        rows.append(
            {
                "label": label,
                "mode": mode,
                "cases": len(items),
                "common_energy_ratio": finite_mean(
                    [item["common_energy_ratio"] for item in items]
                ),
                "raw_cosine": finite_mean(
                    [
                        item["raw_cosine"]["signed_offdiag_mean"]
                        for item in items
                    ]
                ),
                "residual_abs_cosine": finite_mean(
                    [
                        item["residual_cosine"]["absolute_offdiag_mean"]
                        for item in items
                    ]
                ),
                "residual_effective_rank": finite_mean(
                    [
                        item["residual_rank"]["entropy_rank_mean"]
                        for item in items
                    ]
                ),
                "raw_phase_coherence": finite_mean(
                    [item["raw_phase_coherence"]["mean"] for item in items]
                ),
                "residual_phase_coherence": finite_mean(
                    [
                        item["residual_phase_coherence"]["mean"]
                        for item in items
                    ]
                ),
                "residual_phase_valid_cases": sum(
                    math.isfinite(item["residual_phase_coherence"]["mean"])
                    for item in items
                ),
            }
        )
    return {"groups": rows}


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    metadata = json.loads(
        (viewer_dir / "combined_metadata.json").read_text(encoding="utf-8")
    )
    cases = metadata["cases"]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = getattr(torch, args.amp_dtype)
    specs = load_specs(
        viewer_dir,
        outputs_root,
        args.latest_movic_checkpoint,
    )
    records: list[dict] = []
    for model_index, spec in enumerate(specs, start=1):
        cfg, model = build_model(spec["config"], spec["checkpoint"], device)
        num_slots = int(cfg.max_num)
        print(f"[model] {model_index}/{len(specs)} {spec['label']}", flush=True)
        for case_index, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            for mode, source_key in (
                ("crop", "crop_dir"),
                ("padding", "padding_dir"),
            ):
                frame_root = (
                    outputs_root
                    / metadata[source_key]
                    / "cases"
                    / case_id
                    / "original"
                )
                rgb = read_frame_sequence(frame_root, int(case["frames"]))
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
                slots = infer_slots(model, video, boxes, device, amp_dtype)
                record = {
                    "case_id": case_id,
                    "mode": mode,
                    "label": spec["label"],
                    "checkpoint": str(spec["checkpoint"]),
                    "conditioned": bool(spec["conditioned"]),
                    "selected_boxes": len(
                        case[mode].get("amg", {}).get("selected_boxes_xywh", [])
                    ),
                    **analyze_slots(slots),
                }
                records.append(record)
                print(
                    f"[infer] model={model_index}/{len(specs)} "
                    f"case={case_index}/{len(cases)} {mode}",
                    flush=True,
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir = viewer_dir / "common_mode_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "For each frame t, residual[t,s] = slot[t,s] - mean_s(slot[t,s]). "
            "Common energy uses the exact ANOVA decomposition "
            "sum||slot||^2 = S*sum||mean||^2 + sum||residual||^2."
        ),
        "records": len(records),
        **aggregate(records),
    }
    (output_dir / "records.json").write_text(
        json.dumps(records, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    csv_fields = list(summary["groups"][0])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summary["groups"])
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
