#!/usr/bin/env python3
"""Audit and visualize GT latent-mask Q/K correspondence on PyBullet cases."""

from __future__ import annotations

import argparse
import gc
import html
import json
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

import run_pybullet_correspondence_diagnostics as single
import run_pybullet_multiobject_correspondence_diagnostics as multi
import train_xssc_object_self_attn_lora as train_core
from AAA_my_test import analyze_wan_gt_toy_worker as wan_tools
from code_vjepa_vggt.train0419_reference.batch_eval_lora import build_pipeline
from frozen_motion_probe import load_pck_head_weights, ordered_head_pairs
from noise_gated_correspondence import (
    cross_frame_mask_terms,
    masks_to_token_occupancy,
    token_occupancy_to_pixel,
    uniform_object_region_correspondence_objective,
)
from run_training_case_diagnostics import add_label, atomic_json, write_video


DEFAULT_SOURCE_CACHE = Path(
    "/data/gaoya/agent-data/cache/frozen_motion_probe_training_diagnostics"
)
DEFAULT_SOURCE_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/frozen_motion_probe_training_diagnostics"
)
DEFAULT_MASK_CACHE = Path(
    "/data/gaoya/agent-data/cache/uniform_multiobject_correspondence_diagnostics"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/latent_mask_correspondence_diagnostics"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_OPENVID_LORA = Path(
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/"
    "openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors"
)
DEFAULT_HEAD_CONFIG = (
    single.EXPERIMENT_ROOT
    / "configs/physiciq67_pck32_s039_latest3350_top100_heads.json"
)
DEFAULT_TIMESTEPS = (100.0, 300.0, 500.0, 700.0, 900.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GT latent-mask correspondence geometry and forward diagnostic."
    )
    parser.add_argument("mode", choices=("mapping", "forward", "render", "all"))
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument("--mask-cache", type=Path, default=DEFAULT_MASK_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--openvid-lora", type=Path, default=DEFAULT_OPENVID_LORA)
    parser.add_argument("--head-config", type=Path, default=DEFAULT_HEAD_CONFIG)
    parser.add_argument(
        "--head-subset", default="T_physiciq67_pck32_s039_latest3350_top100"
    )
    parser.add_argument(
        "--head-subtype", default="physiciq67_pck32_s039_latest3350"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--training-timesteps", type=float, nargs="+", default=DEFAULT_TIMESTEPS
    )
    parser.add_argument("--source-pixel-frame", type=int, default=4)
    parser.add_argument("--source-latent-frame", type=int, default=1)
    parser.add_argument("--lambda-corr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if str(args.device).startswith("cuda:4"):
        raise ValueError("GPU 4 is prohibited by workspace rules")
    if int(args.source_pixel_frame) != 4 or int(args.source_latent_frame) != 1:
        raise ValueError("this diagnostic is fixed to F04 / latent-1")
    if float(args.lambda_corr) <= 0.0:
        raise ValueError("lambda-corr must be positive")
    for timestep in args.training_timesteps:
        if float(timestep) not in DEFAULT_TIMESTEPS:
            raise ValueError(
                f"timestep={timestep} has no controlled cached x_t; expected {DEFAULT_TIMESTEPS}"
            )
    required = [args.source_cache / "manifest.json"]
    if args.mode in {"forward", "all"}:
        required.extend((args.openvid_lora, args.head_config))
    for path in required:
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)


def load_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(
        (args.source_cache.resolve() / "manifest.json").read_text(encoding="utf-8")
    )


def load_case_masks(
    args: argparse.Namespace,
    case_key: str,
) -> tuple[np.ndarray, list[str]]:
    case_cache = args.mask_cache.resolve() / "cases" / case_key
    meta_path = case_cache / "objects_complete.json"
    mask_path = case_cache / "object_masks.npz"
    if not meta_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(
            f"missing object masks for {case_key}; run the multi-object prepare-objects stage"
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(mask_path) as arrays:
        masks = arrays["masks_othw"].astype(np.uint8)
    phrases = [str(value) for value in metadata["object_phrases"]]
    if masks.shape[0] != len(phrases) or masks.shape[1:] != (49, 512, 896):
        raise RuntimeError(f"{case_key}: unexpected mask geometry {masks.shape}")
    return masks, phrases


def mapping_frame_metrics(
    gt_mask: np.ndarray,
    reverse_soft: np.ndarray,
    reverse_support: np.ndarray,
    occupancy: np.ndarray,
) -> dict[str, float | int]:
    gt = np.asarray(gt_mask, dtype=bool)
    support = np.asarray(reverse_support, dtype=bool)
    intersection = int(np.logical_and(gt, support).sum())
    union = int(np.logical_or(gt, support).sum())
    predicted = int(support.sum())
    gt_area = int(gt.sum())
    false_negative = int(np.logical_and(gt, ~support).sum())
    return {
        "gt_pixels": gt_area,
        "support_pixels": predicted,
        "support_tokens": int((occupancy > 0).sum()),
        "occupancy_sum_tokens": float(occupancy.sum()),
        "iou": float(intersection / max(union, 1)),
        "precision": float(intersection / max(predicted, 1)),
        "recall": float(intersection / max(gt_area, 1)),
        "false_negative_pixels": false_negative,
        "soft_roundtrip_mae": float(
            np.abs(np.asarray(reverse_soft, dtype=np.float32) - gt.astype(np.float32)).mean()
        ),
        "area_ratio": float(occupancy.sum() * 32.0 * 32.0 / max(gt_area, 1)),
    }


def render_mapping_error(
    frame: np.ndarray,
    gt_mask: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    output = (np.asarray(frame, dtype=np.float32) * 0.38).astype(np.uint8)
    gt = np.asarray(gt_mask, dtype=bool)
    support = np.asarray(support_mask, dtype=bool)
    true_positive = gt & support
    false_positive = ~gt & support
    false_negative = gt & ~support
    output[true_positive] = np.asarray([35, 196, 115], dtype=np.uint8)
    output[false_positive] = np.asarray([238, 75, 71], dtype=np.uint8)
    output[false_negative] = np.asarray([38, 220, 255], dtype=np.uint8)
    return output


def run_mapping(args: argparse.Namespace) -> None:
    manifest = load_manifest(args)
    output_root = args.output_root.resolve()
    cases = []
    for case_index, case in enumerate(manifest["cases"], start=1):
        case_key = str(case["case_key"])
        masks, phrases = load_case_masks(args, case_key)
        frames = multi.load_source_frames(case)
        aligned_masks = masks[:, single.ANCHOR_PIXEL_FRAMES]
        occupancy = masks_to_token_occupancy(
            torch.from_numpy(aligned_masks), token_hw=single.TOKEN_HW
        )
        reverse_soft = token_occupancy_to_pixel(
            occupancy, pixel_hw=frames.shape[1:3]
        ).numpy()
        reverse_support = token_occupancy_to_pixel(
            occupancy > 0, pixel_hw=frames.shape[1:3]
        ).numpy().astype(bool)
        case_output = output_root / "cases" / case_key
        mapping_output = case_output / "mapping"
        mapping_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mapping_output / "latent_masks.npz",
            anchor_pixel_frames=single.ANCHOR_PIXEL_FRAMES,
            gt_masks_othw=aligned_masks,
            occupancy_othw=occupancy.numpy(),
            reverse_soft_othw=reverse_soft,
            reverse_support_othw=reverse_support.astype(np.uint8),
        )

        object_rows = []
        for object_index, phrase in enumerate(phrases):
            frame_rows = []
            rendered = []
            color = multi.OBJECT_COLORS[object_index % len(multi.OBJECT_COLORS)]
            for latent_index, pixel_index in enumerate(single.ANCHOR_PIXEL_FRAMES):
                frame = frames[int(pixel_index)]
                gt = aligned_masks[object_index, latent_index].astype(bool)
                support = reverse_support[object_index, latent_index]
                row = mapping_frame_metrics(
                    gt,
                    reverse_soft[object_index, latent_index],
                    support,
                    occupancy[object_index, latent_index].numpy(),
                )
                row.update(
                    {
                        "latent_frame": latent_index,
                        "pixel_frame": int(pixel_index),
                    }
                )
                if int(row["false_negative_pixels"]) != 0:
                    raise RuntimeError(
                        f"{case_key}/O{object_index + 1}/L{latent_index}: "
                        "reverse token support lost GT pixels"
                    )
                frame_rows.append(row)

                gt_panel = add_label(
                    multi.overlay_mask(frame, gt, color),
                    f"SAM2 supervision mask | L{latent_index:02d}/F{pixel_index:02d}",
                )
                occupancy_panel = single.overlay_heatmap(
                    frame, occupancy[object_index, latent_index].numpy(), vmax=1.0
                )
                occupancy_panel = add_label(
                    occupancy_panel,
                    f"latent occupancy | {int(row['support_tokens'])} cells",
                )
                reverse_panel = add_label(
                    multi.overlay_mask(frame, support, color),
                    f"reverse support | IoU {float(row['iou']):.3f}",
                )
                error_panel = add_label(
                    render_mapping_error(frame, gt, support),
                    "roundtrip: green TP | red extra | cyan missed",
                )
                rendered.append(
                    np.concatenate(
                        [
                            single.resize_panel(gt_panel),
                            single.resize_panel(occupancy_panel),
                            single.resize_panel(reverse_panel),
                            single.resize_panel(error_panel),
                        ],
                        axis=1,
                    )
                )
            write_video(
                mapping_output / f"object_{object_index:02d}_mapping_audit.mp4",
                rendered,
                fps=float(args.fps),
            )
            object_rows.append(
                {
                    "object_index": object_index,
                    "phrase": phrase,
                    "mean_iou": float(np.mean([row["iou"] for row in frame_rows])),
                    "min_iou": float(np.min([row["iou"] for row in frame_rows])),
                    "mean_precision": float(
                        np.mean([row["precision"] for row in frame_rows])
                    ),
                    "min_recall": float(np.min([row["recall"] for row in frame_rows])),
                    "max_false_negative_pixels": int(
                        np.max([row["false_negative_pixels"] for row in frame_rows])
                    ),
                    "mean_soft_roundtrip_mae": float(
                        np.mean([row["soft_roundtrip_mae"] for row in frame_rows])
                    ),
                    "max_area_ratio_error": float(
                        np.max([abs(float(row["area_ratio"]) - 1.0) for row in frame_rows])
                    ),
                    "frames": frame_rows,
                }
            )
        summary = {
            "schema_version": 1,
            "case_key": case_key,
            "family": case["family"],
            "caption": case["caption"],
            "object_count": len(phrases),
            "object_phrases": phrases,
            "pixel_geometry": [512, 896],
            "latent_geometry": [13, 16, 28],
            "temporal_mapping": "latent L00..L12 maps to pixel F00,F04,...,F48",
            "spatial_mapping": "32x32 pixel area average per latent token",
            "mask_source": (
                "GroundingDINO identity detection plus SAM2 per-frame tracking; "
                "pseudo-mask treated as GT-role supervision"
            ),
            "reverse_mapping": "nearest expansion of each token cell to its 32x32 pixel footprint",
            "support_rule": "occupancy > 0; exact GT-pixel recall is required",
            "objects": object_rows,
        }
        atomic_json(mapping_output / "metrics.json", summary)
        atomic_json(mapping_output / "mapping_complete.json", {"state": "complete"})
        cases.append(summary)
        print(
            f"[mapping {case_index}/3] {case_key}: {len(phrases)} objects, "
            "reverse recall verified",
            flush=True,
        )
    atomic_json(
        output_root / "mapping_status.json",
        {"state": "complete", "case_count": len(cases), "cases": cases},
    )


class LatentMaskPCKCollector:
    def __init__(
        self,
        dit: torch.nn.Module,
        selected_heads_by_block: Mapping[int, Sequence[int]],
        pck_weights: torch.Tensor,
        *,
        object_token_occupancy_othw: torch.Tensor,
        source_frame: int,
    ) -> None:
        self.dit = dit
        self.selected_heads_by_block = {
            int(block): tuple(sorted(map(int, heads)))
            for block, heads in selected_heads_by_block.items()
        }
        pairs = ordered_head_pairs(self.selected_heads_by_block)
        if len(pairs) != int(pck_weights.numel()):
            raise ValueError(f"head/weight mismatch: {len(pairs)}/{pck_weights.numel()}")
        weights_by_pair = {
            pair: float(weight)
            for pair, weight in zip(pairs, pck_weights.detach().cpu().tolist())
        }
        self.weights_by_block = {
            block: torch.tensor(
                [weights_by_pair[(block, head)] for head in heads], dtype=torch.float32
            )
            for block, heads in self.selected_heads_by_block.items()
        }
        self.object_token_occupancy_othw = object_token_occupancy_othw
        self.source_frame = int(source_frame)
        self.attention: torch.Tensor | None = None
        self.target: torch.Tensor | None = None
        self.valid: torch.Tensor | None = None
        self.source_token_count: torch.Tensor | None = None
        self.head_count = 0
        self.originals: list[tuple[torch.nn.Module, Any]] = []

    def _capture(self, q: torch.Tensor, k: torch.Tensor, block_id: int) -> None:
        heads = self.selected_heads_by_block[block_id]
        num_heads = int(q.shape[-1] // 128)
        expected_tokens = 13 * single.TOKEN_HW[0] * single.TOKEN_HW[1]
        if num_heads != 24 or q.shape[1] != expected_tokens:
            raise RuntimeError(
                f"unexpected Wan Q/K geometry: q={tuple(q.shape)}, heads={num_heads}"
            )
        selected = torch.as_tensor(heads, device=q.device, dtype=torch.long)
        q_heads = q.reshape(q.shape[0], q.shape[1], num_heads, 128).index_select(
            2, selected
        )
        k_heads = k.reshape(k.shape[0], k.shape[1], num_heads, 128).index_select(
            2, selected
        )
        terms = cross_frame_mask_terms(
            q_heads,
            k_heads,
            object_token_occupancy_othw=self.object_token_occupancy_othw,
            source_frame=self.source_frame,
            future_only=True,
        )
        weights = self.weights_by_block[block_id].to(
            device=q.device, dtype=terms["attention"].dtype
        )
        weighted_attention = (
            terms["attention"] * weights.reshape(1, 1, 1, -1, 1)
        ).sum(dim=3)
        self.attention = (
            weighted_attention
            if self.attention is None
            else self.attention + weighted_attention
        )
        self.target = terms["target"]
        self.valid = terms["valid"]
        self.source_token_count = terms["source_token_count"]
        self.head_count += len(heads)

    def install(self) -> None:
        for block_id in self.selected_heads_by_block:
            attention = self.dit.blocks[block_id].self_attn.attn
            original = attention.forward

            def wrapped(
                module,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                *,
                _block_id: int = block_id,
                _original=original,
            ):
                self._capture(q, k, _block_id)
                return _original(q, k, v)

            self.originals.append((attention, original))
            attention.forward = MethodType(wrapped, attention)

    def remove(self) -> None:
        for attention, original in self.originals:
            attention.forward = original
        self.originals.clear()

    def finalize(
        self,
        *,
        scheduler_sigma: float,
        lambda_corr: float,
        object_phrases: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.head_count != 100:
            raise RuntimeError(f"captured {self.head_count} heads, expected 100")
        if (
            self.attention is None
            or self.target is None
            or self.valid is None
            or self.source_token_count is None
        ):
            raise RuntimeError("collector captured no latent-mask correspondence maps")
        attention = self.attention[0].float()
        target = self.target.float()
        valid = self.valid.bool()
        probability_error = float((attention.sum(dim=-1) - 1.0).abs().max().item())
        if probability_error > 2.0e-5:
            raise RuntimeError(f"PCK aggregate probability error: {probability_error}")
        objective = uniform_object_region_correspondence_objective(
            attention,
            target,
            valid,
            lambda_corr=float(lambda_corr),
        )
        support = target > 0
        attention_mass = (attention * support.to(attention.dtype)).sum(dim=-1)
        distribution_l1 = (attention - target).abs().sum(dim=-1)
        top1 = attention.argmax(dim=-1)
        top1_inside = support.gather(dim=-1, index=top1[..., None]).squeeze(-1)

        object_metrics = []
        for object_index, phrase in enumerate(object_phrases):
            selected = valid[object_index]
            object_metrics.append(
                {
                    "object_index": object_index,
                    "phrase": phrase,
                    "source_support_tokens": int(
                        self.source_token_count[object_index].item()
                    ),
                    "valid_target_frames": int(selected.sum().item()),
                    "raw_soft_ce": float(
                        objective["raw_soft_ce_per_object"][object_index].item()
                    ),
                    "correspondence_loss": float(
                        lambda_corr
                        * objective["raw_soft_ce_per_object"][object_index].item()
                    ),
                    "mean_attention_mass_in_gt_support": float(
                        attention_mass[object_index][selected].mean().item()
                    ),
                    "top1_in_gt_support_rate": float(
                        top1_inside[object_index][selected].float().mean().item()
                    ),
                    "mean_distribution_l1": float(
                        distribution_l1[object_index][selected].mean().item()
                    ),
                }
            )
        maps = {
            "attention_ots": attention.cpu().numpy(),
            "target_ots": target.cpu().numpy(),
            "ce_contribution_ots": objective["ce_contribution"].cpu().numpy(),
            "ce_ot": objective["ce"].cpu().numpy(),
            "valid_ot": valid.cpu().numpy().astype(np.uint8),
            "attention_mass_in_gt_support_ot": attention_mass.cpu().numpy(),
            "top1_in_gt_support_ot": top1_inside.cpu().numpy().astype(np.uint8),
            "object_token_occupancy_othw": self.object_token_occupancy_othw
            .float()
            .cpu()
            .numpy(),
        }
        metrics = {
            "scheduler_sigma": float(scheduler_sigma),
            "raw_soft_ce": float(objective["raw_soft_ce"].item()),
            "lambda_corr": float(lambda_corr),
            "correspondence_loss": float(objective["loss"].item()),
            "valid_object_frames": int(valid.sum().item()),
            "mean_attention_mass_in_gt_support": float(
                np.mean(
                    [row["mean_attention_mass_in_gt_support"] for row in object_metrics]
                )
            ),
            "top1_in_gt_support_rate": float(
                np.mean([row["top1_in_gt_support_rate"] for row in object_metrics])
            ),
            "mean_distribution_l1": float(
                np.mean([row["mean_distribution_l1"] for row in object_metrics])
            ),
            "aggregate_probability_max_error": probability_error,
            "object_reduction": "mean valid frame CE per object, then equal mean over objects",
            "objects": object_metrics,
        }
        return maps, metrics


def run_forward(args: argparse.Namespace) -> None:
    manifest = load_manifest(args)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    print("[forward] loading Wan2.2 + OpenVid LoRA step-10000", flush=True)
    pipe = build_pipeline(
        args.wan_root.resolve(), str(device), args.openvid_lora.resolve()
    )
    pipe.dit.requires_grad_(False).eval()
    selected_heads, head_metadata = train_core.load_head_selection_config(
        args.head_config,
        expected_subset_id=args.head_subset,
        expected_role="T",
        expected_feature_subtype=args.head_subtype,
        expected_num_heads=100,
        num_blocks=30,
        num_heads=24,
    )
    pck_weights, pck_audit = load_pck_head_weights(head_metadata, selected_heads)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    try:
        for case_index, case in enumerate(manifest["cases"], start=1):
            case_key = str(case["case_key"])
            masks, phrases = load_case_masks(args, case_key)
            aligned_masks = masks[:, single.ANCHOR_PIXEL_FRAMES]
            occupancy = masks_to_token_occupancy(
                torch.from_numpy(aligned_masks), token_hw=single.TOKEN_HW
            ).to(device)
            source_frames = wan_tools.load_video_prefix(
                Path(case["source_video"]), 49, 512, 896, "cache"
            )
            inputs_shared, inputs_positive = wan_tools.prepare_conditioning(
                pipe,
                prompt=case["caption"],
                context_video=source_frames[:8],
                height=512,
                width=896,
                num_frames=49,
                sampling_steps=40,
                sigma_shift=5.0,
                cfg_scale=5.0,
                seed=int(args.seed) + case_index,
            )
            captured_inputs = dict(inputs_shared)
            captured_inputs.update(inputs_positive)
            pipe.scheduler.set_timesteps(1000, training=True)
            pipe.load_models_to_device(pipe.in_iteration_models)
            models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
            case_output = output_root / "cases" / case_key
            case_output.mkdir(parents=True, exist_ok=True)
            stage_metrics = []
            for timestep_value in args.training_timesteps:
                sid = single.stage_id(timestep_value)
                stage_output = case_output / sid
                complete = stage_output / "forward_complete.json"
                if complete.is_file() and not args.overwrite:
                    stage_metrics.append(
                        json.loads(
                            (stage_output / "metrics.json").read_text(encoding="utf-8")
                        )
                    )
                    print(f"[forward {case_index}/3] skip {case_key} {sid}", flush=True)
                    continue
                stage_output.mkdir(parents=True, exist_ok=True)
                source_stage = (
                    args.source_output.resolve()
                    / "cases"
                    / case_key
                    / "noise_sweep"
                    / "stages"
                    / sid
                )
                latent_state = torch.load(
                    source_stage / "latents.pt", map_location="cpu", weights_only=True
                )
                source_metrics = json.loads(
                    (source_stage / "metrics.json").read_text(encoding="utf-8")
                )
                latent_xt = latent_state["training_xt"].to(
                    device=device, dtype=pipe.torch_dtype
                )
                timestep = torch.full(
                    (1,), float(timestep_value), device=device, dtype=pipe.torch_dtype
                )
                sigma = float(source_metrics["scheduler_sigma"])
                collector = LatentMaskPCKCollector(
                    pipe.dit,
                    selected_heads,
                    pck_weights,
                    object_token_occupancy_othw=occupancy,
                    source_frame=int(args.source_latent_frame),
                )
                print(
                    f"[forward {case_index}/3] {case_key} {sid} "
                    f"objects={len(phrases)} sigma={sigma:.4f}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                main_inputs = dict(captured_inputs)
                main_inputs["latents"] = latent_xt
                collector.install()
                try:
                    with torch.no_grad():
                        velocity = pipe.model_fn(
                            **models, **main_inputs, timestep=timestep
                        )
                finally:
                    collector.remove()
                del velocity, latent_xt, latent_state
                maps, metrics = collector.finalize(
                    scheduler_sigma=sigma,
                    lambda_corr=float(args.lambda_corr),
                    object_phrases=phrases,
                )
                np.savez_compressed(stage_output / "loss_maps.npz", **maps)
                metrics.update(
                    {
                        "schema_version": 1,
                        "case_key": case_key,
                        "family": case["family"],
                        "caption": case["caption"],
                        "dataset_split": "train",
                        "dataset_index": int(case["dataset_index"]),
                        "training_timestep": float(timestep_value),
                        "stage_id": sid,
                        "model": (
                            "Wan2.2-TI2V-5B + OpenVid LoRA step-10000; "
                            "equivalent to fresh zero-delta Full-SA initialization"
                        ),
                        "execution_scope": "one forward per case/stage; no optimizer step",
                        "supervision": (
                            "cached GroundingDINO + SAM2 tracked per-frame pseudo-mask, "
                            "treated as GT-role supervision and area-pooled to latent tokens"
                        ),
                        "source_frame": "L01/F04",
                        "target_frames": "future L02/F08 through L12/F48",
                        "loss_formula": "0.01 * object_equal_mean(region_soft_label_CE)",
                        "noise_weighting": "uniform; no SNR gate and no hard cutoff",
                        "head_weighting": "Top100 normalized linear PCK32 mixture",
                        "head_reduction": "CE(target, sum_h weight_h * region_attention_h)",
                        "source_query_aggregation": (
                            "per-token post-RoPE Q attention, weighted by source-mask occupancy"
                        ),
                        "target_distribution": (
                            "target-frame token occupancy normalized over 16x28 cells"
                        ),
                        "pixel_token_mapping": "32x32 area occupancy; no point sampling",
                        "pck_score_min": float(pck_audit["score_min"]),
                        "pck_score_max": float(pck_audit["score_max"]),
                        "peak_gpu_memory_mib": float(
                            torch.cuda.max_memory_allocated(device) / (1024**2)
                            if device.type == "cuda"
                            else 0.0
                        ),
                    }
                )
                atomic_json(stage_output / "metrics.json", metrics)
                atomic_json(complete, {"state": "complete", "stage_id": sid})
                stage_metrics.append(metrics)
                del collector, maps
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            case_summary = {
                "case_key": case_key,
                "family": case["family"],
                "caption": case["caption"],
                "object_count": len(phrases),
                "object_phrases": phrases,
                "stages": stage_metrics,
            }
            atomic_json(case_output / "metrics.json", case_summary)
            summaries.append(case_summary)
    finally:
        del pipe
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_json(
        output_root / "forward_status.json",
        {"state": "complete", "case_count": len(summaries), "cases": summaries},
    )


def render_stage(
    stage_output: Path,
    frames: np.ndarray,
    masks_othw: np.ndarray,
    metrics: dict[str, Any],
    phrases: list[str],
    *,
    fps: float,
) -> None:
    with np.load(stage_output / "loss_maps.npz") as arrays:
        attention = arrays["attention_ots"].astype(np.float32)
        target = arrays["target_ots"].astype(np.float32)
        ce_contribution = arrays["ce_contribution_ots"].astype(np.float32)
        valid = arrays["valid_ot"].astype(bool)
        attention_mass = arrays["attention_mass_in_gt_support_ot"].astype(np.float32)
        occupancy = arrays["object_token_occupancy_othw"].astype(np.float32)
    object_count, time_count, spatial_count = attention.shape
    if (time_count, spatial_count) != (13, 16 * 28):
        raise RuntimeError(f"unexpected attention map shape {attention.shape}")
    attention_maps = attention.reshape(object_count, time_count, 16, 28)
    target_maps = target.reshape(object_count, time_count, 16, 28)
    ce_maps = (
        ce_contribution * float(metrics["lambda_corr"])
    ).reshape(object_count, time_count, 16, 28)
    object_dir = stage_output / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)

    for object_index, phrase in enumerate(phrases):
        attention_scale = multi.positive_scale(
            attention_maps[object_index][valid[object_index]]
        )
        ce_scale = multi.positive_scale(ce_maps[object_index][valid[object_index]])
        rendered = []
        color = multi.OBJECT_COLORS[object_index % len(multi.OBJECT_COLORS)]
        for latent_index, pixel_index in enumerate(single.ANCHOR_PIXEL_FRAMES):
            frame = frames[int(pixel_index)]
            gt_panel = multi.overlay_mask(
                frame, masks_othw[object_index, int(pixel_index)], color
            )
            gt_panel = add_label(
                gt_panel,
                f"SAM2 supervision mask | L{latent_index:02d}/F{pixel_index:02d}",
            )
            target_panel = single.overlay_heatmap(
                frame, occupancy[object_index, latent_index], vmax=1.0
            )
            target_panel = add_label(
                target_panel,
                f"latent occupancy target | sum {occupancy[object_index, latent_index].sum():.2f}",
            )
            attention_panel = single.overlay_heatmap(
                frame,
                attention_maps[object_index, latent_index],
                vmax=attention_scale,
            )
            role = "source (not in loss)" if not valid[object_index, latent_index] else "target"
            attention_panel = add_label(
                attention_panel,
                f"post-RoPE QK region attention | {role} | GT mass {attention_mass[object_index, latent_index]:.3f}",
            )
            ce_panel = single.overlay_heatmap(
                frame, ce_maps[object_index, latent_index], vmax=ce_scale
            )
            ce_panel = add_label(
                ce_panel,
                f"lambda-weighted region CE | lambda {float(metrics['lambda_corr']):.3f}",
            )
            rendered.append(
                np.concatenate(
                    [
                        single.resize_panel(gt_panel),
                        single.resize_panel(target_panel),
                        single.resize_panel(attention_panel),
                        single.resize_panel(ce_panel),
                    ],
                    axis=1,
                )
            )
        write_video(
            object_dir / f"object_{object_index:02d}_latent_mask_loss.mp4",
            rendered,
            fps=float(fps),
        )

    combined_attention = attention_maps.mean(axis=0)
    combined_occupancy = occupancy.mean(axis=0)
    combined_ce = ce_maps.mean(axis=0)
    attention_scale = multi.positive_scale(combined_attention[:, :, :])
    ce_scale = multi.positive_scale(combined_ce[:, :, :])
    combined_frames = []
    for latent_index, pixel_index in enumerate(single.ANCHOR_PIXEL_FRAMES):
        frame = frames[int(pixel_index)]
        gt_panel = frame.copy()
        for object_index in range(object_count):
            gt_panel = multi.overlay_mask(
                gt_panel,
                masks_othw[object_index, int(pixel_index)],
                multi.OBJECT_COLORS[object_index % len(multi.OBJECT_COLORS)],
            )
        gt_panel = add_label(
            gt_panel,
            f"all {object_count} GT masks | L{latent_index:02d}/F{pixel_index:02d}",
        )
        target_panel = add_label(
            single.overlay_heatmap(
                frame, combined_occupancy[latent_index], vmax=1.0
            ),
            "equal-object latent occupancy",
        )
        attention_panel = add_label(
            single.overlay_heatmap(
                frame, combined_attention[latent_index], vmax=attention_scale
            ),
            "equal-object post-RoPE QK region attention",
        )
        ce_panel = add_label(
            single.overlay_heatmap(frame, combined_ce[latent_index], vmax=ce_scale),
            "equal-object lambda-weighted region CE",
        )
        combined_frames.append(
            np.concatenate(
                [
                    single.resize_panel(gt_panel),
                    single.resize_panel(target_panel),
                    single.resize_panel(attention_panel),
                    single.resize_panel(ce_panel),
                ],
                axis=1,
            )
        )
    write_video(
        stage_output / "object_equal_latent_mask_loss.mp4",
        combined_frames,
        fps=float(fps),
    )
    atomic_json(
        stage_output / "render_complete.json",
        {"state": "complete", "training_timestep": metrics["training_timestep"]},
    )


def build_report(output_root: Path, cases: list[dict[str, Any]]) -> None:
    sections = []
    for case in cases:
        case_key = str(case["case_key"])
        mapping = case["mapping"]
        mapping_figures = []
        for row in mapping["objects"]:
            object_index = int(row["object_index"])
            source = (
                f"cases/{case_key}/mapping/"
                f"object_{object_index:02d}_mapping_audit.mp4"
            )
            mapping_figures.append(
                "<figure>"
                f"<figcaption><strong>O{object_index + 1}: {html.escape(row['phrase'])}</strong>"
                f"<span>mean IoU {row['mean_iou']:.3f} | min recall {row['min_recall']:.3f} | "
                f"max missed pixels {row['max_false_negative_pixels']}</span></figcaption>"
                f"<video controls muted loop playsinline preload='metadata' src='{html.escape(source)}'></video>"
                "</figure>"
            )

        stage_rows = []
        stage_blocks = []
        for stage_index, stage in enumerate(case["stages"]):
            object_losses = " / ".join(
                f"O{row['object_index'] + 1} {row['correspondence_loss']:.5f}"
                for row in stage["objects"]
            )
            stage_rows.append(
                "<tr>"
                f"<td>{stage['training_timestep']:.0f}</td>"
                f"<td>{stage['scheduler_sigma']:.4f}</td>"
                f"<td>{stage['raw_soft_ce']:.4f}</td>"
                f"<td>{stage['correspondence_loss']:.6f}</td>"
                f"<td>{stage['mean_attention_mass_in_gt_support']:.3f}</td>"
                f"<td>{stage['top1_in_gt_support_rate']:.3f}</td>"
                f"<td>{html.escape(object_losses)}</td>"
                "</tr>"
            )
            sid = str(stage["stage_id"])
            object_figures = []
            for row in stage["objects"]:
                object_index = int(row["object_index"])
                source = (
                    f"cases/{case_key}/{sid}/objects/"
                    f"object_{object_index:02d}_latent_mask_loss.mp4"
                )
                object_figures.append(
                    "<figure>"
                    f"<figcaption><strong>O{object_index + 1}: {html.escape(row['phrase'])}</strong>"
                    f"<span>loss {row['correspondence_loss']:.6f} | "
                    f"GT mass {row['mean_attention_mass_in_gt_support']:.3f}</span></figcaption>"
                    f"<video controls muted loop playsinline preload='metadata' src='{html.escape(source)}'></video>"
                    "</figure>"
                )
            combined_source = f"cases/{case_key}/{sid}/object_equal_latent_mask_loss.mp4"
            stage_blocks.append(
                f"<details {'open' if stage_index == 0 else ''}>"
                f"<summary><strong>t={stage['training_timestep']:.0f}</strong>"
                f"<span>sigma {stage['scheduler_sigma']:.4f} | object-equal loss "
                f"{stage['correspondence_loss']:.6f}</span></summary>"
                "<div class='stage-body'>"
                "<figure class='combined'><figcaption><strong>All objects, equal reduction</strong>"
                "<span>one forward; no optimizer step</span></figcaption>"
                f"<video controls muted loop playsinline preload='metadata' src='{html.escape(combined_source)}'></video>"
                "</figure>"
                f"<div class='figures'>{''.join(object_figures)}</div>"
                f"<div class='links'><a href='cases/{case_key}/{sid}/metrics.json'>metrics.json</a>"
                f"<a href='cases/{case_key}/{sid}/loss_maps.npz'>loss_maps.npz</a></div>"
                "</div></details>"
            )
        phrases = " | ".join(
            f"O{index + 1}: {phrase}"
            for index, phrase in enumerate(case["object_phrases"])
        )
        sections.append(
            "<section>"
            f"<div class='case-title'><span>{case['object_count']} objects</span><div>"
            f"<h2>{html.escape(case_key)}</h2><p>{html.escape(case['caption'])}</p>"
            f"<small>{html.escape(phrases)}</small></div></div>"
            "<div class='mapping-head'><h3>1. Pixel-to-latent roundtrip audit</h3>"
            "<p>The available object masks are GroundingDINO + SAM2 tracked pseudo-labels "
            "treated as GT-role supervision; these cases do not export native simulator instance "
            "masks. Area pooling maps each 32x32 pixel cell to one occupancy value. The reverse "
            "support expands every occupied token cell; it must contain every GT pixel, while "
            "red pixels expose the expected boundary over-coverage.</p></div>"
            f"<div class='figures'>{''.join(mapping_figures)}</div>"
            f"<div class='links'><a href='cases/{case_key}/mapping/metrics.json'>mapping metrics</a>"
            f"<a href='cases/{case_key}/mapping/latent_masks.npz'>latent masks</a></div>"
            "<div class='attention-head'><h3>2. GT latent-mask Q/K correspondence</h3>"
            "<p>For each object, every occupied F04 source token supplies post-RoPE Q. Its "
            "framewise attention distributions are occupancy-weighted, then compared with the "
            "normalized target-frame GT occupancy. CE is averaged over future frames per object "
            "and then equally across objects.</p></div>"
            "<div class='table-wrap'><table><thead><tr><th>t</th><th>sigma</th>"
            "<th>object-equal CE</th><th>loss</th><th>GT support mass</th>"
            f"<th>top1 in GT</th><th>per-object loss</th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table></div>"
            f"{''.join(stage_blocks)}</section>"
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GT latent-mask correspondence</title><style>
:root{{--bg:#edf0ee;--paper:#fff;--ink:#17201d;--muted:#64706b;--line:#c6ceca;--accent:#08756a;--red:#b0473c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans SC","Source Han Sans SC",sans-serif;letter-spacing:0}}header{{padding:27px max(20px,calc((100vw - 1580px)/2));background:#17201d;color:#fff}}h1{{font-size:28px;margin:0 0 8px}}header p{{max-width:1180px;margin:0;color:#c8d1cd;line-height:1.6}}main{{max-width:1580px;margin:auto;padding:20px}}section{{margin-bottom:24px;padding:18px;background:var(--paper);border:1px solid var(--line)}}.case-title{{display:grid;grid-template-columns:auto minmax(0,1fr);gap:13px;align-items:start;padding-bottom:13px;border-bottom:1px solid var(--line)}}.case-title>span{{padding:5px 9px;background:var(--accent);color:#fff;font-weight:800}}h2{{font-size:21px;margin:0 0 4px}}h3{{font-size:16px;margin:0 0 5px}}p{{line-height:1.55}}.case-title p{{margin:0 0 5px;color:var(--muted)}}small{{line-height:1.5}}.mapping-head,.attention-head{{margin-top:18px;border-left:4px solid var(--accent);padding-left:10px}}.mapping-head p,.attention-head p{{max-width:1120px;margin:0;color:var(--muted);font-size:13px}}.figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:10px;margin-top:11px}}figure{{margin:0;border:1px solid var(--line);background:#fff;padding:7px;min-width:0}}figcaption{{display:flex;justify-content:space-between;gap:12px;min-height:24px;margin-bottom:5px}}figcaption strong{{font-size:12px}}figcaption span{{font-size:11px;color:var(--muted);text-align:right}}video{{display:block;width:100%;height:auto;background:#111}}.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;margin:14px 0;font-variant-numeric:tabular-nums}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right;font-size:12px;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}details{{border-top:3px solid var(--accent);background:#f7f9f8;margin-top:12px}}summary{{display:flex;justify-content:space-between;gap:15px;padding:11px 13px;cursor:pointer}}summary span{{font-size:12px;color:var(--muted)}}.stage-body{{padding:0 12px 13px}}.combined{{margin-bottom:10px}}.links{{display:flex;gap:14px;margin-top:9px}}a{{color:var(--accent);font-size:12px}}@media(max-width:800px){{main{{padding:8px}}section{{padding:10px}}.figures{{grid-template-columns:1fr}}summary,figcaption{{display:block}}summary span,figcaption span{{display:block;text-align:left;margin-top:3px}}}}
</style></head><body><header><h1>GT-role latent-mask correspondence audit</h1><p>PyBullet train cases using cached GroundingDINO + SAM2 tracked pseudo-masks as supervision GT. Wan2.2-TI2V-5B with OpenVid LoRA step-10000; exact F00..F48 to L00..L12 mapping; 512x896 pixels to a 16x28 token grid; F04/L01 source queries; Top100 PCK32 attention mixture; object-equal region CE. This page is a forward-only preflight diagnostic, not a training run.</p></header><main>{''.join(sections)}</main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def run_render(args: argparse.Namespace) -> None:
    manifest = load_manifest(args)
    output_root = args.output_root.resolve()
    cases = []
    for case_index, case in enumerate(manifest["cases"], start=1):
        case_key = str(case["case_key"])
        case_output = output_root / "cases" / case_key
        case_metrics = json.loads(
            (case_output / "metrics.json").read_text(encoding="utf-8")
        )
        mapping_metrics = json.loads(
            (case_output / "mapping" / "metrics.json").read_text(encoding="utf-8")
        )
        frames = multi.load_source_frames(case)
        masks, phrases = load_case_masks(args, case_key)
        print(
            f"[render {case_index}/3] {case_key}: {len(phrases)} objects",
            flush=True,
        )
        for stage in case_metrics["stages"]:
            stage_output = case_output / stage["stage_id"]
            complete = stage_output / "render_complete.json"
            if complete.is_file() and not args.overwrite:
                continue
            render_stage(
                stage_output,
                frames,
                masks,
                stage,
                phrases,
                fps=float(args.fps),
            )
        case_metrics["mapping"] = mapping_metrics
        cases.append(case_metrics)
    build_report(output_root, cases)
    atomic_json(
        output_root / "render_status.json",
        {"state": "complete", "case_count": len(cases)},
    )


def main() -> None:
    args = parse_args()
    check_args(args)
    if args.mode in {"mapping", "all"}:
        run_mapping(args)
    if args.mode in {"forward", "all"}:
        run_forward(args)
    if args.mode in {"render", "all"}:
        run_render(args)


if __name__ == "__main__":
    main()
