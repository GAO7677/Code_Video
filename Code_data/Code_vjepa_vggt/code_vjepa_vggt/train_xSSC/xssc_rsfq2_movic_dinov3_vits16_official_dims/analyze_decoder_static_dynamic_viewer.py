#!/usr/bin/env python3
"""Analyze xSSC decoder static/dynamic partitions and controlled ablations."""

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

from analyze_slot_temporal_similarity_viewer import (  # noqa: E402
    DEFAULT_MOVIC_CKPT_DIR,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_VIEWER_DIR,
    boxes_from_metadata,
    build_model,
    checkpoint_step,
    cosine_metrics,
    frequency_metrics,
    latest_checkpoint,
    load_specs,
    normalize_rgb_frames,
    read_frame_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--latest-movic-checkpoint", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def run_xssc(
    model,
    video: torch.Tensor,
    boxes: torch.Tensor | None,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = {"video": video.to(device, non_blocking=True)}
    if boxes is not None:
        batch["bbox"] = boxes.to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        output = model(batch=batch)
    return output["feature"], output["slotz"]


def project_slots(
    decoder,
    slots: torch.Tensor,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    batch, time_steps, num_slots, slot_dim = slots.shape
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=amp_dtype, enabled=slots.device.type == "cuda"
    ):
        projected = decoder.project2(
            slots.reshape(batch * time_steps, num_slots, slot_dim)
        )
    return projected.reshape(batch, time_steps, num_slots, -1)


def freeze_partition(
    memory: torch.Tensor,
    static_dim: int,
    mode: str,
) -> torch.Tensor:
    if mode == "full":
        return memory
    frozen = memory.clone()
    temporal_mean = memory.mean(dim=1, keepdim=True)
    if mode == "dynamic_frozen":
        frozen[..., static_dim:] = temporal_mean[..., static_dim:]
    elif mode == "static_frozen":
        frozen[..., :static_dim] = temporal_mean[..., :static_dim]
    else:
        raise ValueError(mode)
    return frozen


def decode_all_masked(
    decoder,
    feature: torch.Tensor,
    projected_slots: torch.Tensor,
    static_dim: int,
    mode: str,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    batch, time_steps, channels, height, width = feature.shape
    num_patches = height * width
    decoder_dim = projected_slots.shape[-1]
    memory = freeze_partition(projected_slots, static_dim, mode)
    memory = memory.reshape(
        batch * time_steps, projected_slots.shape[2], decoder_dim
    )

    query = decoder.mask_token.expand(
        batch * time_steps, num_patches, decoder_dim
    )
    query = query + decoder.posit_embed.pe[:, :num_patches]
    time_token = decoder.te.weight[0][None, None].expand(
        batch * time_steps, -1, -1
    )
    query = torch.cat([time_token, query], dim=1)
    memory = torch.cat([time_token, memory], dim=1)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=amp_dtype, enabled=feature.device.type == "cuda"
    ):
        decoded = decoder.backbone(
            decoder.norm0(query),
            memory=memory,
            memory_key_padding_mask=None,
        )
        decoded = decoder.readout(decoded)[:, 1:]
    return decoded.reshape(
        batch, time_steps, num_patches, channels
    )


def decoder_metrics(
    target: torch.Tensor,
    outputs: dict[str, torch.Tensor],
) -> dict:
    target = target.detach().float().cpu()
    outputs_cpu = {
        key: value.detach().float().cpu() for key, value in outputs.items()
    }
    full = outputs_cpu["full"]
    result = {}
    for key, output in outputs_cpu.items():
        result[key] = {
            "mse_to_target": float(F.mse_loss(output, target).item()),
            "cosine_to_target": float(
                F.cosine_similarity(output, target, dim=-1).mean().item()
            ),
            "adjacent_cosine": float(
                F.cosine_similarity(
                    output[:, :-1], output[:, 1:], dim=-1
                ).mean().item()
            ),
            "temporal_delta_rms": float(
                (output[:, 1:] - output[:, :-1])
                .square()
                .mean()
                .sqrt()
                .item()
            ),
            "cosine_to_full": float(
                F.cosine_similarity(output, full, dim=-1).mean().item()
            ),
        }
    return result


def plot_analysis(
    static_arrays: dict[str, np.ndarray],
    dynamic_arrays: dict[str, np.ndarray],
    static_frequency_arrays: dict[str, np.ndarray],
    dynamic_frequency_arrays: dict[str, np.ndarray],
    summary: dict,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 11.2), dpi=145)
    for axis, arrays, name in (
        (axes[0, 0], static_arrays, "Static partition"),
        (axes[0, 1], dynamic_arrays, "Dynamic partition"),
    ):
        image = axis.imshow(
            arrays["adjacent_same"],
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(f"{name}: adjacent cosine by slot")
        axis.set_xlabel("transition t -> t+1")
        axis.set_ylabel("slot ID")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axes[1, 0].plot(
        static_arrays["frame0_fixed_mean"],
        label="static partition",
        linewidth=1.8,
    )
    axes[1, 0].plot(
        dynamic_arrays["frame0_fixed_mean"],
        label="dynamic partition",
        linewidth=1.8,
    )
    axes[1, 0].set_title("Projected-slot similarity to frame 0")
    axes[1, 0].set_xlabel("frame")
    axes[1, 0].set_ylabel("cosine similarity")
    axes[1, 0].set_ylim(-0.05, 1.02)
    axes[1, 0].grid(alpha=0.22)
    axes[1, 0].legend(loc="lower left")

    axes[1, 1].plot(
        static_frequency_arrays["frequencies"],
        static_frequency_arrays["global_relative_power"],
        label="static partition",
        linewidth=1.8,
    )
    axes[1, 1].plot(
        dynamic_frequency_arrays["frequencies"],
        dynamic_frequency_arrays["global_relative_power"],
        label="dynamic partition",
        linewidth=1.8,
    )
    axes[1, 1].set_title("Projected-slot dynamic power spectrum")
    axes[1, 1].set_xlabel("frequency (cycles/frame)")
    axes[1, 1].set_ylabel("relative dynamic power")
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].legend(loc="upper right")

    variants = ["full", "dynamic_frozen", "static_frozen"]
    labels = ["full", "dynamic frozen", "static frozen"]
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    decoder = summary["decoder"]
    axes[2, 0].bar(
        labels,
        [decoder[key]["mse_to_target"] for key in variants],
        color=colors,
    )
    axes[2, 0].set_title("All-mask decoder reconstruction MSE")
    axes[2, 0].set_ylabel("MSE to DINO target (lower is better)")
    axes[2, 0].tick_params(axis="x", rotation=12)

    x = np.arange(len(variants))
    width = 0.25
    axes[2, 1].bar(
        x - width,
        [decoder[key]["cosine_to_target"] for key in variants],
        width,
        label="target cosine",
    )
    axes[2, 1].bar(
        x,
        [decoder[key]["adjacent_cosine"] for key in variants],
        width,
        label="temporal cosine",
    )
    axes[2, 1].bar(
        x + width,
        [decoder[key]["cosine_to_full"] for key in variants],
        width,
        label="full-output cosine",
    )
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(labels, rotation=12)
    axes[2, 1].set_ylim(-0.05, 1.02)
    axes[2, 1].set_title("Decoder output similarity")
    axes[2, 1].legend(loc="lower left", fontsize=8)

    partition = summary["partition"]
    fig.suptitle(
        f"{title}\n"
        f"decoder split {partition['static_dim']} static + "
        f"{partition['dynamic_dim']} dynamic",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def analyze_case(
    model,
    video: torch.Tensor,
    boxes: torch.Tensor | None,
    device: torch.device,
    amp_dtype: torch.dtype,
    metadata: dict,
    output_path: Path,
) -> dict:
    feature, slots = run_xssc(model, video, boxes, device, amp_dtype)
    decoder = model.m.decode
    projected = project_slots(decoder, slots, amp_dtype)
    decoder_dim = int(projected.shape[-1])
    dynamic_dim = int(decoder_dim * float(decoder.rd))
    static_dim = decoder_dim - dynamic_dim
    static_slots = projected[0, :, :, :static_dim].detach().float().cpu().numpy()
    dynamic_slots = (
        projected[0, :, :, static_dim:].detach().float().cpu().numpy()
    )

    static_summary, static_arrays = cosine_metrics(static_slots)
    dynamic_summary, dynamic_arrays = cosine_metrics(dynamic_slots)
    static_frequency_summary, static_frequency_arrays = frequency_metrics(
        static_slots
    )
    dynamic_frequency_summary, dynamic_frequency_arrays = frequency_metrics(
        dynamic_slots
    )

    outputs = {
        mode: decode_all_masked(
            decoder,
            feature,
            projected,
            static_dim,
            mode,
            amp_dtype,
        )
        for mode in ("full", "dynamic_frozen", "static_frozen")
    }
    target = feature.permute(0, 1, 3, 4, 2).flatten(2, 3)
    summary = {
        **metadata,
        "partition": {
            "decoder_dim": decoder_dim,
            "static_dim": static_dim,
            "dynamic_dim": dynamic_dim,
            "dynamic_ratio": float(decoder.rd),
        },
        "static": {
            "temporal": static_summary,
            "frequency": static_frequency_summary,
        },
        "dynamic": {
            "temporal": dynamic_summary,
            "frequency": dynamic_frequency_summary,
        },
        "decoder": decoder_metrics(target, outputs),
        "ablation": {
            "query": (
                "All patch queries use the learned mask token plus positional "
                "embedding for deterministic slot-conditioned decoding."
            ),
            "dynamic_frozen": (
                "Dynamic projected-memory channels are replaced by their "
                "per-slot temporal mean."
            ),
            "static_frozen": (
                "Static projected-memory channels are replaced by their "
                "per-slot temporal mean."
            ),
        },
    }
    plot_analysis(
        static_arrays,
        dynamic_arrays,
        static_frequency_arrays,
        dynamic_frequency_arrays,
        summary,
        output_path,
        f"{metadata['case_id']} | {metadata['mode']} | {metadata['label']}",
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    combined_path = viewer_dir / "combined_metadata.json"
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
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
    output_root = viewer_dir / "decoder_static_dynamic"
    case_results: dict[str, dict] = {}

    for model_index, spec in enumerate(specs, start=1):
        cfg, model = build_model(spec["config"], spec["checkpoint"], device)
        num_slots = int(cfg.max_num)
        print(
            f"[model] {model_index}/{len(specs)} {spec['label']} "
            f"checkpoint={spec['checkpoint']}",
            flush=True,
        )
        for case_index, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            case_results.setdefault(case_id, {"crop": [], "padding": []})
            for mode, source_key in (
                ("crop", "crop_dir"),
                ("padding", "padding_dir"),
            ):
                output_path = (
                    output_root / "cases" / case_id / mode
                    / f"{spec['label']}.png"
                )
                metrics_path = output_path.with_suffix(".json")
                if (
                    output_path.is_file()
                    and metrics_path.is_file()
                    and not args.force
                ):
                    summary = json.loads(
                        metrics_path.read_text(encoding="utf-8")
                    )
                else:
                    frame_root = (
                        outputs_root
                        / combined[source_key]
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
                    summary = analyze_case(
                        model,
                        video,
                        boxes,
                        device,
                        amp_dtype,
                        {
                            "case_id": case_id,
                            "mode": mode,
                            "label": spec["label"],
                            "checkpoint": str(spec["checkpoint"]),
                        },
                        output_path,
                    )
                case_results[case_id][mode].append(
                    {
                        "label": spec["label"],
                        "aliases": spec["aliases"],
                        "chart": str(output_path.relative_to(viewer_dir)),
                        "metrics": summary,
                    }
                )
                print(
                    f"[analyze] model={model_index}/{len(specs)} "
                    f"case={case_index}/{len(cases)} {case_id} {mode}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined["decoder_static_dynamic"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "The trained MarkovRarDecoder project2 output is split according "
            "to rd into static and dynamic channel partitions. Controlled "
            "all-mask decoding freezes one partition to its per-slot temporal "
            "mean; it is an ablation, not an additive feature decomposition."
        ),
        "models": [
            {
                "label": spec["label"],
                "checkpoint": str(spec["checkpoint"]),
                "config": str(spec["config"]),
            }
            for spec in specs
        ],
        "cases": case_results,
    }
    combined_path.write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "viewer_dir": str(viewer_dir),
                "models": len(specs),
                "cases": len(cases),
                "latest_movic_checkpoint": str(
                    max(
                        (
                            spec["checkpoint"]
                            for spec in specs
                            if "transfer15000" in str(spec["config"])
                        ),
                        key=checkpoint_step,
                    )
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
