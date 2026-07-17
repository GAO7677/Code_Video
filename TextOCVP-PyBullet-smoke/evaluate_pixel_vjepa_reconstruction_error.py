#!/usr/bin/env python3
"""Compare Pixel-SAVi RGB error and V-JEPA-SAVi feature error spatially."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parent
TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
os.chdir(TEXTOCVP_ROOT)

from data.Stage1Indexed import Stage1Indexed  # noqa: E402
from feature_space_stage1.backbones import FrozenVJEPA2Extractor  # noqa: E402
from feature_space_stage1.model import FeatureSlotDecomposer  # noqa: E402
from lib.setup_model import load_checkpoint, setup_model  # noqa: E402


PALETTE = np.asarray(
    [
        [230, 57, 70], [29, 154, 108], [43, 116, 189], [244, 162, 54],
        [138, 79, 191], [0, 168, 181], [241, 91, 181], [126, 130, 122],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pixel", "vjepa"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--external-json", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--num-validation", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


class ExternalDataset(Stage1Indexed):
    def __init__(self, paths: list[Path], *, preprocess_mode: str, img_size: tuple[int, int]):
        self.dataset_mode = "external"
        self.split = "valid"
        self.num_frames = 10
        self.img_size = img_size
        self.frame_stride = 1
        self.random_start = False
        self.preprocess_mode = preprocess_mode
        self.vjepa_short_side = 438
        self.vjepa_crop_size = 384
        self.load_masks = False
        self.records = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            video_path = Path(payload["source_video"])
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            self.records.append(
                {
                    "source": "physiq_external",
                    "video_path": str(video_path),
                    "metadata_path": str(path),
                    "group": "physiq_external",
                    "sample_id": path.stem,
                    "sampling_frame_range": [0, 49],
                }
            )


def write_h264(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def add_header(panels: list[np.ndarray], names: list[str], detail: str) -> np.ndarray:
    body = np.concatenate(panels, axis=1)
    header = np.full((66, body.shape[1], 3), 250, dtype=np.uint8)
    panel_width = panels[0].shape[1]
    for index, name in enumerate(names):
        cv2.putText(
            header, name, (index * panel_width + 8, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA,
        )
    cv2.putText(
        header, detail, (8, 53), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (70, 70, 70), 1, cv2.LINE_AA,
    )
    return np.concatenate((header, body), axis=0)


def heat_rgb(values: np.ndarray, scale: float, output_hw: tuple[int, int]) -> np.ndarray:
    encoded = np.clip(values / max(scale, 1e-8), 0.0, 1.0)
    heat = cv2.applyColorMap((encoded * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.resize(heat, (output_hw[1], output_hw[0]), interpolation=cv2.INTER_NEAREST)


def heat_overlay(frame: np.ndarray, heat: np.ndarray) -> np.ndarray:
    return np.clip(
        frame.astype(np.float32) * 0.45 + heat.astype(np.float32) * 0.55,
        0,
        255,
    ).astype(np.uint8)


def slot_overlay(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    labels = masks.argmax(axis=0).astype(np.int64)
    colors = PALETTE[labels % len(PALETTE)]
    colors = cv2.resize(colors, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.clip(frame.astype(np.float32) * 0.55 + colors * 0.45, 0, 255).astype(np.uint8)


def motion_mask(video: torch.Tensor, quantile: float = 0.80) -> torch.Tensor:
    gray = video.float().mean(dim=1)
    difference = torch.zeros_like(gray)
    difference[1:] = (gray[1:] - gray[:-1]).abs()
    positive = difference[difference > 0]
    threshold = torch.quantile(positive, quantile) if positive.numel() else torch.tensor(0.0)
    return difference >= threshold


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else float("nan")


def to_uint8(video: torch.Tensor) -> np.ndarray:
    return video.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).mul(255).round().byte().numpy()


def load_model(args: argparse.Namespace, config: dict, device: torch.device):
    if args.mode == "pixel":
        model = setup_model(config["model"])
        model = load_checkpoint(
            checkpoint_path=str(args.checkpoint), model=model, only_model=True, map_cpu=True
        )
        return model.eval().to(device), None
    train_config = config.get("config", config)
    extractor = FrozenVJEPA2Extractor(
        Path(train_config["checkpoint"]), device, num_frames=int(train_config["num_frames"])
    )
    model = FeatureSlotDecomposer(
        feature_dim=extractor.feature_dim,
        num_slots=int(train_config["num_slots"]),
        slot_dim=int(train_config["slot_dim"]),
    ).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(payload["decomposer_state_dict"], strict=True)
    return model.eval(), extractor


def kubric_regions(
    dataset: Stage1Indexed,
    index: int,
    frame_ids: list[int],
) -> dict[str, torch.Tensor] | None:
    record = dataset.records[index]
    targets, _ = dataset._load_kubric_mask_targets(
        record, np.asarray(frame_ids, dtype=np.int64)
    )
    if not bool(targets["mask_supervision_valid"]):
        return None
    dynamic_occupancy = targets["dynamic_union_mask"][:, 0]
    static_occupancy = targets["static_geometry_mask"][:, 0]
    dynamic = (dynamic_occupancy > static_occupancy) & (dynamic_occupancy > 0.10)
    static_geometry = (static_occupancy >= dynamic_occupancy) & (static_occupancy > 0.10)
    return {"dynamic": dynamic, "static_geometry": static_geometry}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    preprocess_mode = "resize" if args.mode == "pixel" else "vjepa"
    img_size = (216, 384) if args.mode == "pixel" else (384, 384)
    validation = Stage1Indexed(
        index_root=args.index_root,
        dataset_mode="kubric",
        split="valid",
        num_frames=10,
        img_size=img_size,
        preprocess_mode=preprocess_mode,
        random_start=False,
        load_masks=True,
        max_mask_instances=6,
        mask_temporal_stride=1 if args.mode == "pixel" else 2,
        mask_spatial_stride=1 if args.mode == "pixel" else 16,
        max_samples=None,
    )
    if args.num_validation > len(validation):
        raise ValueError(
            f"Requested {args.num_validation} validation samples, only {len(validation)} exist"
        )
    validation_indices = np.random.default_rng(args.seed).choice(
        len(validation), size=args.num_validation, replace=False
    ).tolist()
    external_paths = [path.resolve() for path in args.external_json]
    external = ExternalDataset(external_paths, preprocess_mode=preprocess_mode, img_size=img_size)
    items = [
        ("kubric_val", validation, index) for index in validation_indices
    ] + [("physiq", external, index) for index in range(len(external))]
    model, extractor = load_model(args, config, device)

    raw_records = []
    for kind, dataset, index in items:
        video, metadata = dataset[index]
        batch = video.unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            if args.mode == "pixel":
                output = model(x=batch, num_imgs=10, decode=True)
                reconstruction = output["recons_imgs"][0].float().cpu().clamp(0, 1)
                loss_map = (reconstruction - video).square().mean(dim=1)
                secondary_map = (reconstruction - video).abs().mean(dim=1)
            else:
                features = extractor(batch)
                output = model(features)
                reconstruction_features = output["reconstructed_features"].float()
                target_features = features.float()
                loss_map = (reconstruction_features - target_features).square().mean(dim=-1)[0].cpu()
                secondary_map = (
                    1.0 - F.cosine_similarity(reconstruction_features, target_features, dim=-1)
                )[0].cpu()
                reconstruction = None
        masks = output["masks"][0].detach().float().cpu().numpy()
        regions = (
            kubric_regions(validation, index, metadata["frame_ids"])
            if kind == "kubric_val"
            else None
        )
        if regions is not None and regions["dynamic"].shape != loss_map.shape:
            raise RuntimeError(
                f"Region/loss shape mismatch: {regions['dynamic'].shape} vs {loss_map.shape}"
            )
        motion = motion_mask(video)
        if loss_map.shape[0] != motion.shape[0]:
            motion = F.avg_pool3d(
                motion[None, None].float(), kernel_size=(2, 16, 16), stride=(2, 16, 16)
            )[0, 0] >= 0.20
        raw_records.append(
            {
                "kind": kind,
                "metadata": metadata,
                "video": video,
                "reconstruction": reconstruction,
                "loss_map": loss_map,
                "secondary_map": secondary_map,
                "masks": masks,
                "regions": regions,
                "motion": motion,
            }
        )

    loss_scale = float(torch.quantile(torch.cat([r["loss_map"].reshape(-1) for r in raw_records]), 0.99))
    secondary_scale = float(
        torch.quantile(torch.cat([r["secondary_map"].reshape(-1) for r in raw_records]), 0.99)
    )
    reports = []
    for record in raw_records:
        metadata = record["metadata"]
        sample_id = safe_name(metadata["sample_id"])
        relative_dir = Path(record["kind"]) / sample_id
        output_case = args.output_dir / relative_dir
        output_case.mkdir(parents=True, exist_ok=True)
        input_rgb = to_uint8(record["video"])
        reconstruction_rgb = (
            to_uint8(record["reconstruction"])
            if record["reconstruction"] is not None
            else None
        )
        rendered = []
        for frame_index, frame in enumerate(input_rgb):
            latent_index = (
                frame_index if args.mode == "pixel" else min(frame_index // 2, record["loss_map"].shape[0] - 1)
            )
            loss_heat = heat_overlay(
                frame,
                heat_rgb(
                    record["loss_map"][latent_index].numpy(), loss_scale, frame.shape[:2]
                ),
            )
            second_heat = heat_overlay(
                frame,
                heat_rgb(
                    record["secondary_map"][latent_index].numpy(),
                    secondary_scale,
                    frame.shape[:2],
                ),
            )
            slot = slot_overlay(frame, record["masks"][latent_index])
            if args.mode == "pixel":
                panels = [frame, reconstruction_rgb[frame_index], loss_heat, slot]
                names = ["GT input", "Pixel reconstruction", "RGB MSE overlay", "SAVi slot masks"]
            else:
                panels = [frame, loss_heat, second_heat, slot]
                names = ["GT input", "Feature MSE overlay", "Cosine-loss overlay", "SAVi slot masks"]
            rendered.append(
                add_header(
                    panels, names,
                    f"source frame={metadata['frame_ids'][frame_index]} | shared p99 scale | {args.mode}",
                )
            )
        video_rel = relative_dir / f"{args.mode}_reconstruction_error_overlay_h264.mp4"
        write_h264(args.output_dir / video_rel, rendered, args.fps)
        np.savez_compressed(
            output_case / f"{args.mode}_loss_maps.npz",
            loss_map=record["loss_map"].numpy().astype(np.float16),
            secondary_map=record["secondary_map"].numpy().astype(np.float16),
            motion_mask=record["motion"].numpy().astype(np.uint8),
        )
        loss_map = record["loss_map"]
        metrics = {"global_loss": float(loss_map.mean().item())}
        if record["regions"] is not None:
            dynamic = record["regions"]["dynamic"]
            static_geometry = record["regions"]["static_geometry"]
            background = ~(dynamic | static_geometry)
            metrics.update(
                {
                    "dynamic_loss": masked_mean(loss_map, dynamic),
                    "static_geometry_loss": masked_mean(loss_map, static_geometry),
                    "background_loss": masked_mean(loss_map, background),
                    "dynamic_area": float(dynamic.float().mean().item()),
                    "static_geometry_area": float(static_geometry.float().mean().item()),
                }
            )
            metrics["dynamic_to_background_ratio"] = (
                metrics["dynamic_loss"] / max(metrics["background_loss"], 1e-12)
            )
            metrics["dynamic_to_static_ratio"] = (
                metrics["dynamic_loss"] / max(metrics["static_geometry_loss"], 1e-12)
            )
        else:
            metrics.update(
                {
                    "motion_proxy_loss": masked_mean(loss_map, record["motion"]),
                    "non_motion_proxy_loss": masked_mean(loss_map, ~record["motion"]),
                }
            )
            metrics["motion_to_non_motion_ratio"] = (
                metrics["motion_proxy_loss"]
                / max(metrics["non_motion_proxy_loss"], 1e-12)
            )
        reports.append(
            {
                "kind": record["kind"],
                "sample_id": metadata["sample_id"],
                "source_video": metadata["video_path"],
                "frame_ids": metadata["frame_ids"],
                "source_resolution_hw": metadata["source_resolution_hw"],
                "model_input_shape": metadata["training_shape"],
                "validation_index": index if record["kind"] == "kubric_val" else None,
                "video": video_rel.as_posix(),
                "metrics": metrics,
            }
        )

    summary = {
        "mode": args.mode,
        "evaluation_gpu": args.gpu,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": (
            int(match.group(1))
            if (match := re.search(r"checkpoint_step_(\d+)", args.checkpoint.name))
            else None
        ),
        "config": str(args.config.resolve()),
        "loss_definition": (
            "mean RGB squared error over channels" if args.mode == "pixel"
            else "mean squared error over 1408-D frozen V-JEPA target features"
        ),
        "secondary_definition": (
            "mean RGB absolute error over channels" if args.mode == "pixel"
            else "1-cosine(reconstructed V-JEPA feature, target V-JEPA feature)"
        ),
        "shared_loss_p99_scale": loss_scale,
        "shared_secondary_p99_scale": secondary_scale,
        "validation_seed": args.seed,
        "validation_indices": validation_indices,
        "model_input_policy": (
            "bilinear resize to 216x384"
            if args.mode == "pixel"
            else "resize short side to 438 then center crop to 384x384"
        ),
        "region_policy": {
            "kubric_val": "GT segmentation: dynamic/static_geometry/background",
            "physiq": "top-20% temporal RGB difference as motion; no segmentation GT",
        },
        "samples": reports,
    }
    (args.output_dir / f"{args.mode}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    cards = []
    for report in reports:
        cards.append(
            f"<article><h2>{html.escape(report['kind'])} / {html.escape(report['sample_id'])}</h2>"
            f"<p><code>{html.escape(report['source_video'])}</code></p>"
            f"<p><code>{html.escape(json.dumps(report['metrics']))}</code></p>"
            f"<video controls loop muted preload='metadata' src='{html.escape(report['video'])}'></video></article>"
        )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>{args.mode} reconstruction error</title>
<style>body{{margin:0;background:#eef0ed;color:#17201b;font-family:Verdana,sans-serif}}header,article{{padding:18px 24px;border-bottom:1px solid #bbc1bc}}article{{background:white;margin:18px}}video{{display:block;width:100%;background:#111}}code{{overflow-wrap:anywhere;font-size:12px}}</style></head><body>
<header><h1>{args.mode} reconstruction error analysis</h1><p>Checkpoint: <code>{html.escape(str(args.checkpoint))}</code></p><p>Heatmaps use one shared p99 scale across all 14 cases. Titles are outside image pixels.</p></header>{''.join(cards)}</body></html>"""
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps({"mode": args.mode, "output_dir": str(args.output_dir), "samples": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
