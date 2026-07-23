#!/usr/bin/env python3
"""Add temporal slot-embedding similarity plots to the crop/padding viewer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

DEFAULT_VIEWER_DIR = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_slot_overlay_test5_crop_padding_compare_plus24000"
)
DEFAULT_OUTPUTS_ROOT = Path("/data/gaoya/agent-data/outputs")
DEFAULT_MOVIC_CKPT_DIR = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
    "restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42"
)
DEFAULT_MOVIC_CONFIG = (
    ROOT
    / "upstream/config-randsfq/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
)
IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--latest-movic-checkpoint",
        type=Path,
        default=None,
        help="Defaults to the highest numbered checkpoint present at startup.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def normalize_rgb_frames(frames: np.ndarray) -> torch.Tensor:
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()[None]
    return (video - IMAGENET_MEAN) / IMAGENET_STD


def read_frame_sequence(frame_root: Path, frame_count: int) -> np.ndarray:
    frames = []
    for frame_id in range(frame_count):
        path = frame_root / f"{frame_id:04d}.webp"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(np.asarray(Image.open(path).convert("RGB")))
    return np.stack(frames, axis=0).astype(np.uint8)


def boxes_from_metadata(
    amg: dict,
    num_slots: int,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    boxes = np.zeros((1, frame_count, num_slots, 4), dtype=np.float32)
    for slot_id, xywh in enumerate(amg.get("selected_boxes_xywh", [])[:num_slots]):
        x, y, box_w, box_h = [float(value) for value in xywh]
        boxes[0, :, slot_id] = np.asarray(
            [x / width, y / height, (x + box_w) / width, (y + box_h) / height],
            dtype=np.float32,
        )
    return torch.from_numpy(boxes)


def load_checkpoint(model, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("m.encode_backbone.")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch {checkpoint}: "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )


def build_model(config_file: Path, checkpoint: Path, device: torch.device):
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(config_file)
    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    model.freez(cfg.freez, verbose=False)
    load_checkpoint(model, checkpoint)
    return cfg, model.to(device).eval()


def infer_slots(
    model,
    video: torch.Tensor,
    boxes: torch.Tensor | None,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    batch = {"video": video.to(device, non_blocking=True)}
    if boxes is not None:
        batch["bbox"] = boxes.to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        output = model(batch=batch)
    slots = output["slotz"][0].detach().float().cpu()
    if slots.ndim != 3:
        raise RuntimeError(f"expected slotz [T,S,C], got {tuple(slots.shape)}")
    return slots.numpy()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step-(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def latest_checkpoint(directory: Path) -> Path:
    checkpoints = list(directory.glob("step-*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"no step checkpoints under {directory}")
    return max(checkpoints, key=checkpoint_step)


def load_specs(
    viewer_dir: Path,
    outputs_root: Path,
    latest_movic_checkpoint: Path | None,
) -> list[dict]:
    combined = json.loads(
        (viewer_dir / "combined_metadata.json").read_text(encoding="utf-8")
    )
    crop_metadata = json.loads(
        (outputs_root / combined["crop_dir"] / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    unique: dict[str, dict] = {}
    for item in crop_metadata["checkpoints"]:
        checkpoint = Path(item["checkpoint"]).resolve()
        key = str(checkpoint)
        if key in unique:
            unique[key]["aliases"].append(item["label"])
            continue
        config = Path(item["config"])
        if not config.is_absolute():
            config = ROOT / "upstream/config-randsfq" / config
        unique[key] = {
            "label": item["label"],
            "aliases": [item["label"]],
            "config": config.resolve(),
            "checkpoint": checkpoint,
        }

    latest = (
        latest_movic_checkpoint.resolve()
        if latest_movic_checkpoint is not None
        else latest_checkpoint(DEFAULT_MOVIC_CKPT_DIR)
    )
    if str(latest) not in unique:
        step = checkpoint_step(latest)
        unique[str(latest)] = {
            "label": f"movi_current_{step:06d}",
            "aliases": [f"movi_current_{step:06d}"],
            "config": DEFAULT_MOVIC_CONFIG.resolve(),
            "checkpoint": latest,
        }
    specs = list(unique.values())
    for spec in specs:
        if not spec["config"].is_file():
            raise FileNotFoundError(spec["config"])
        if not spec["checkpoint"].is_file():
            raise FileNotFoundError(spec["checkpoint"])
        spec["conditioned"] = "c-movi" in spec["config"].stem
    return specs


def cosine_metrics(slots: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    normalized = F.normalize(torch.from_numpy(slots).float(), dim=-1).numpy()
    time_steps, num_slots, _ = normalized.shape
    adjacent_same = np.sum(normalized[:-1] * normalized[1:], axis=-1)
    adjacent_fixed_mean = adjacent_same.mean(axis=1)
    adjacent_matched_mean = []
    adjacent_identity_rate = []
    for frame_id in range(time_steps - 1):
        cross = normalized[frame_id] @ normalized[frame_id + 1].T
        rows, cols = linear_sum_assignment(-cross)
        adjacent_matched_mean.append(float(cross[rows, cols].mean()))
        adjacent_identity_rate.append(float(np.mean(cols == rows)))

    time_matrix = np.einsum(
        "tsc,usc->tus", normalized, normalized, optimize=True
    ).mean(axis=-1)
    frame0_fixed = time_matrix[0]
    frame0_matched = []
    for frame_id in range(time_steps):
        cross = normalized[0] @ normalized[frame_id].T
        rows, cols = linear_sum_assignment(-cross)
        frame0_matched.append(float(cross[rows, cols].mean()))

    arrays = {
        "adjacent_same": adjacent_same.T,
        "adjacent_fixed_mean": np.asarray(adjacent_fixed_mean),
        "adjacent_matched_mean": np.asarray(adjacent_matched_mean),
        "adjacent_identity_rate": np.asarray(adjacent_identity_rate),
        "frame0_fixed_mean": np.asarray(frame0_fixed),
        "frame0_matched_mean": np.asarray(frame0_matched),
        "time_matrix": np.asarray(time_matrix),
    }
    summary = {
        "frames": int(time_steps),
        "slots": int(num_slots),
        "slot_dim": int(slots.shape[-1]),
        "adjacent_fixed_mean": float(arrays["adjacent_fixed_mean"].mean()),
        "adjacent_matched_mean": float(arrays["adjacent_matched_mean"].mean()),
        "adjacent_identity_rate": float(arrays["adjacent_identity_rate"].mean()),
        "frame0_final_fixed": float(arrays["frame0_fixed_mean"][-1]),
        "frame0_final_matched": float(arrays["frame0_matched_mean"][-1]),
        "minimum_fixed_adjacent": float(arrays["adjacent_same"].min()),
    }
    return summary, arrays


def plot_similarity(
    arrays: dict[str, np.ndarray],
    summary: dict,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), dpi=145)
    heat = axes[0, 0].imshow(
        arrays["adjacent_same"],
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 0].set_title("Adjacent-frame cosine by fixed slot ID")
    axes[0, 0].set_xlabel("transition t -> t+1")
    axes[0, 0].set_ylabel("slot ID")
    fig.colorbar(heat, ax=axes[0, 0], fraction=0.046, pad=0.04)

    x_adjacent = np.arange(1, len(arrays["adjacent_fixed_mean"]) + 1)
    axes[0, 1].plot(
        x_adjacent,
        arrays["adjacent_fixed_mean"],
        label="fixed slot ID",
        linewidth=1.8,
    )
    axes[0, 1].plot(
        x_adjacent,
        arrays["adjacent_matched_mean"],
        label="Hungarian matched",
        linewidth=1.8,
    )
    axes[0, 1].set_title(
        "Adjacent-frame mean "
        f"(ID retention {summary['adjacent_identity_rate']:.1%})"
    )
    axes[0, 1].set_xlabel("destination frame")
    axes[0, 1].set_ylabel("cosine similarity")
    axes[0, 1].set_ylim(-0.05, 1.02)
    axes[0, 1].grid(alpha=0.22)
    axes[0, 1].legend(loc="lower left")

    x_time = np.arange(len(arrays["frame0_fixed_mean"]))
    axes[1, 0].plot(
        x_time,
        arrays["frame0_fixed_mean"],
        label="fixed slot ID",
        linewidth=1.8,
    )
    axes[1, 0].plot(
        x_time,
        arrays["frame0_matched_mean"],
        label="Hungarian matched",
        linewidth=1.8,
    )
    axes[1, 0].set_title("Similarity to frame 0")
    axes[1, 0].set_xlabel("frame")
    axes[1, 0].set_ylabel("cosine similarity")
    axes[1, 0].set_ylim(-0.05, 1.02)
    axes[1, 0].grid(alpha=0.22)
    axes[1, 0].legend(loc="lower left")

    matrix = axes[1, 1].imshow(
        arrays["time_matrix"],
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 1].set_title("All-frame fixed-ID mean cosine")
    axes[1, 1].set_xlabel("frame")
    axes[1, 1].set_ylabel("frame")
    fig.colorbar(matrix, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    combined_path = viewer_dir / "combined_metadata.json"
    metadata = json.loads(combined_path.read_text(encoding="utf-8"))
    cases = metadata["cases"]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = getattr(torch, args.amp_dtype)
    specs = load_specs(
        viewer_dir, outputs_root, args.latest_movic_checkpoint
    )

    temporal_root = viewer_dir / "temporal_similarity"
    temporal_cases: dict[str, dict] = {}
    model_rows = [
        {
            "label": spec["label"],
            "aliases": spec["aliases"],
            "config": str(spec["config"]),
            "checkpoint": str(spec["checkpoint"]),
            "conditioned": spec["conditioned"],
        }
        for spec in specs
    ]

    for model_index, spec in enumerate(specs, start=1):
        cfg, model = build_model(
            spec["config"], spec["checkpoint"], device
        )
        num_slots = int(cfg.max_num)
        print(
            f"[model] {model_index}/{len(specs)} {spec['label']} "
            f"slots={num_slots} checkpoint={spec['checkpoint']}",
            flush=True,
        )
        for case_index, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            temporal_cases.setdefault(case_id, {"crop": [], "padding": []})
            for mode, source_key in (
                ("crop", "crop_dir"),
                ("padding", "padding_dir"),
            ):
                output_path = (
                    temporal_root / "cases" / case_id / mode
                    / f"{spec['label']}.png"
                )
                metrics_path = output_path.with_suffix(".json")
                if output_path.is_file() and metrics_path.is_file() and not args.force:
                    summary = json.loads(
                        metrics_path.read_text(encoding="utf-8")
                    )
                else:
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
                    slots = infer_slots(
                        model, video, boxes, device, amp_dtype
                    )
                    summary, arrays = cosine_metrics(slots)
                    summary.update(
                        {
                            "case_id": case_id,
                            "mode": mode,
                            "label": spec["label"],
                            "checkpoint": str(spec["checkpoint"]),
                        }
                    )
                    plot_similarity(
                        arrays,
                        summary,
                        output_path,
                        f"{case_id} | {mode} | {spec['label']}",
                    )
                    metrics_path.write_text(
                        json.dumps(summary, indent=2) + "\n",
                        encoding="utf-8",
                    )
                temporal_cases[case_id][mode].append(
                    {
                        "label": spec["label"],
                        "aliases": spec["aliases"],
                        "chart": str(output_path.relative_to(viewer_dir)),
                        "metrics": summary,
                    }
                )
                print(
                    f"[infer] model={model_index}/{len(specs)} "
                    f"case={case_index}/{len(cases)} {case_id} {mode}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata["temporal_similarity"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Cosine similarity on final xSSC slotz embeddings. Fixed-ID curves "
            "retain the original slot index; matched curves use independent "
            "Hungarian maximum-cosine matching for each frame pair."
        ),
        "models": model_rows,
        "cases": temporal_cases,
    }
    combined_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
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
