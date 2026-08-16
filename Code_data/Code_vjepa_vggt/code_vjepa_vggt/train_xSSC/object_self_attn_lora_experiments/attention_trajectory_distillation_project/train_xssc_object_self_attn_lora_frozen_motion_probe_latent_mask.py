#!/usr/bin/env python3
"""Train Wan self-attention LoRA with GT-role latent-mask correspondence CE.

The main Student keeps the original flow-matching objective. A separate,
LoRA-free Wan2.2 Frozen Motion Probe reuses detached Teacher post-RoPE Q and
computes frame-conditional attention against Student K. GroundingDINO + SAM2
tracked masks are area-pooled to the Wan latent grid and used as soft spatial
targets. The probe is frozen; only the main Student adapter is optimized.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
for _path in (EXPERIMENT_ROOT, TRAIN_XSSC_ROOT, REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import code_vjepa_vggt.context_wan_v_newtrain as context_wan
import train_xssc_object_self_attn_lora as core
import train_xssc_object_self_attn_lora_frozen_motion_probe as legacy
from attention_trajectory_distillation_project.frozen_motion_probe import (
    blend_with_fixed_probe_noise,
)
from attention_trajectory_distillation_project.noise_gated_correspondence import (
    masks_to_token_occupancy,
    token_occupancy_to_pixel,
    uniform_object_region_correspondence_objective,
)


DEFAULT_MASK_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/uniform_multiobject_correspondence_diagnostics"
)


def _as_numpy_masks(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    masks = np.asarray(value)
    if masks.ndim == 3:
        masks = masks[None]
    if masks.ndim != 4:
        raise ValueError(f"expected [O,T,H,W] or [T,H,W] masks, got {masks.shape}")
    masks = masks.astype(np.float32, copy=False)
    if not np.isfinite(masks).all() or np.any(masks < 0.0) or np.any(masks > 1.0):
        raise ValueError("tracked masks must be finite and lie in [0,1]")
    return masks


def _sample_metadata(raw_sample: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = raw_sample.get("metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _cache_case_root(cache_root: Path, sample_key: str) -> Path:
    cases_root = (cache_root.expanduser().resolve() / "cases").resolve()
    case_root = (cases_root / sample_key).resolve()
    try:
        case_root.relative_to(cases_root)
    except ValueError as exc:
        raise ValueError(f"unsafe sample_key for mask cache: {sample_key!r}") from exc
    return case_root


def _align_masks_to_sampled_frames(
    masks_othw: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    expected_frames: int,
    mask_frame_indices: Any = None,
) -> tuple[np.ndarray, str]:
    raw_indices = metadata.get("sampled_frame_indices")
    if raw_indices is None:
        if masks_othw.shape[1] != int(expected_frames):
            raise ValueError(
                "tracked mask frame count cannot be aligned without "
                "metadata.sampled_frame_indices"
            )
        return masks_othw, "identity_without_sample_indices"
    indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
    if indices.size != int(expected_frames):
        raise ValueError(
            f"expected {expected_frames} sampled frame indices, got {indices.size}"
        )
    if np.any(indices < 0):
        raise ValueError("sampled frame indices must be non-negative")
    if mask_frame_indices is not None:
        mask_indices = np.asarray(mask_frame_indices, dtype=np.int64).reshape(-1)
        if mask_indices.size != masks_othw.shape[1]:
            raise ValueError(
                "tracked mask frame_indices length does not match mask frames: "
                f"{mask_indices.size}/{masks_othw.shape[1]}"
            )
        if np.any(mask_indices < 0) or np.unique(mask_indices).size != mask_indices.size:
            raise ValueError("tracked mask frame_indices must be unique and non-negative")
        positions = {int(frame_id): position for position, frame_id in enumerate(mask_indices)}
        missing = [int(frame_id) for frame_id in indices if int(frame_id) not in positions]
        if missing:
            raise ValueError(
                "tracked masks do not cover sampled frame indices: "
                f"missing={missing[:8]}"
            )
        selected = np.asarray([positions[int(frame_id)] for frame_id in indices])
        return masks_othw[:, selected], "explicit_frame_indices"
    if np.array_equal(indices, np.arange(int(expected_frames), dtype=np.int64)):
        if masks_othw.shape[1] != int(expected_frames):
            raise ValueError(
                f"identity sampled frames require {expected_frames} masks, "
                f"got {masks_othw.shape[1]}"
            )
        return masks_othw, "identity_sampled_frames"
    source_frame_count = metadata.get("source_frame_count")
    if source_frame_count is None or int(source_frame_count) != masks_othw.shape[1]:
        raise ValueError(
            "non-identity sampled frame indices require either tracked mask "
            "frame_indices or a full source-timeline mask with matching "
            "metadata.source_frame_count"
        )
    if int(indices.max()) >= masks_othw.shape[1]:
        raise ValueError(
            "sampled frame indices are incompatible with tracked mask frames: "
            f"max_index={int(indices.max())}, mask_frames={masks_othw.shape[1]}"
        )
    return masks_othw[:, indices], "source_timeline_indices"


def load_sample_gt_role_masks(
    raw_sample: Mapping[str, Any],
    *,
    cache_root: Path,
    mask_key: str,
    object_index: int,
    expected_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one object's sampled-frame masks from the sample or offline cache."""
    if not isinstance(raw_sample, Mapping):
        raise TypeError("latent-mask training requires a raw_sample mapping")
    metadata = _sample_metadata(raw_sample)
    direct = legacy._unwrap_query_payload(dict(raw_sample), str(mask_key))
    sample_key = str(metadata.get("sample_key", "")).strip()
    if direct is not None:
        masks_othw = _as_numpy_masks(direct)
        mask_frame_indices = legacy._unwrap_query_payload(
            dict(raw_sample),
            f"{mask_key}_frame_indices",
        )
        source = "raw_sample"
        source_path = None
    else:
        if not sample_key:
            raise KeyError(
                "raw_sample.metadata.sample_key is required to locate tracked masks"
            )
        source_path = _cache_case_root(Path(cache_root), sample_key) / "object_masks.npz"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"missing GroundingDINO + SAM2 mask cache for {sample_key}: {source_path}"
            )
        with np.load(source_path) as arrays:
            if "masks_othw" not in arrays:
                raise KeyError(f"{source_path} does not contain masks_othw")
            masks_othw = _as_numpy_masks(arrays["masks_othw"])
            mask_frame_indices = (
                arrays["frame_indices"] if "frame_indices" in arrays else None
            )
        source = "cache"
    masks_othw, frame_alignment = _align_masks_to_sampled_frames(
        masks_othw,
        metadata,
        expected_frames=int(expected_frames),
        mask_frame_indices=mask_frame_indices,
    )
    object_index = int(object_index)
    if not 0 <= object_index < masks_othw.shape[0]:
        raise IndexError(
            f"object_index={object_index} outside cached objects={masks_othw.shape[0]}"
        )
    selected = masks_othw[object_index]
    video = raw_sample.get("video")
    if isinstance(video, torch.Tensor) and tuple(video.shape[-2:]) != tuple(
        selected.shape[-2:]
    ):
        raise ValueError(
            "tracked mask/video geometry mismatch: "
            f"mask={selected.shape[-2:]}, video={tuple(video.shape[-2:])}"
        )
    frame_area = selected.reshape(selected.shape[0], -1).sum(axis=1)
    if np.any(frame_area <= 0.0):
        invalid = np.flatnonzero(frame_area <= 0.0).tolist()
        raise RuntimeError(
            f"tracked object {object_index} has empty sampled masks at frames {invalid}"
        )
    return selected, {
        "sample_key": sample_key,
        "source": source,
        "source_path": str(source_path) if source_path is not None else None,
        "frame_alignment": frame_alignment,
        "object_index": object_index,
        "pixel_frames": int(selected.shape[0]),
        "pixel_height": int(selected.shape[1]),
        "pixel_width": int(selected.shape[2]),
        "min_frame_area": float(frame_area.min()),
        "max_frame_area": float(frame_area.max()),
    }


def build_latent_mask_supervision(
    masks_thw: np.ndarray,
    *,
    grid: tuple[int, int, int],
    source_frame: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    """Area-pool sampled pixel masks and build fixed source query weights."""
    masks = np.asarray(masks_thw, dtype=np.float32)
    if masks.ndim != 3:
        raise ValueError(f"expected [T,H,W] selected-object masks, got {masks.shape}")
    latent_frames, token_height, token_width = map(int, grid)
    required_pixel_frames = 4 * (latent_frames - 1) + 1
    if masks.shape[0] != required_pixel_frames:
        raise ValueError(
            f"grid={grid} requires {required_pixel_frames} sampled masks, got {masks.shape[0]}"
        )
    source_frame = int(source_frame)
    if not 0 <= source_frame < latent_frames:
        raise ValueError(f"source_frame={source_frame} outside T={latent_frames}")
    anchor_indices = np.arange(latent_frames, dtype=np.int64) * 4
    aligned = torch.from_numpy(masks[anchor_indices]).unsqueeze(0)
    occupancy_cpu = masks_to_token_occupancy(
        aligned,
        token_hw=(token_height, token_width),
    )
    reverse_support = token_occupancy_to_pixel(
        (occupancy_cpu > 0).to(torch.float32),
        pixel_hw=tuple(map(int, masks.shape[-2:])),
    ).to(torch.bool)
    gt_support = aligned.to(torch.bool)
    missed_pixels = gt_support & ~reverse_support
    if bool(missed_pixels.any()):
        raise RuntimeError(
            "latent-mask reverse mapping missed GT foreground pixels: "
            f"count={int(missed_pixels.sum().item())}"
        )
    intersection = (gt_support & reverse_support).sum().to(torch.float64)
    reverse_area = reverse_support.sum().to(torch.float64)
    union = (gt_support | reverse_support).sum().to(torch.float64)
    mapping_audit = {
        "reverse_recall": float(intersection / gt_support.sum().clamp_min(1)),
        "reverse_precision": float(intersection / reverse_area.clamp_min(1)),
        "reverse_iou": float(intersection / union.clamp_min(1)),
        "missed_gt_pixels": float(missed_pixels.sum().item()),
    }
    occupancy = occupancy_cpu.to(device=device, dtype=torch.float32)
    flat = occupancy.flatten(2)
    occupancy_sum = flat.sum(dim=-1)
    if float(occupancy_sum[0, source_frame].item()) <= 0.0:
        raise RuntimeError("GT-role mask has no occupied source latent tokens")
    spatial = token_height * token_width
    source_occupancy = flat[0, source_frame]
    source_rows = (source_occupancy > 0).nonzero(as_tuple=False).flatten()
    query_rows = source_frame * spatial + source_rows.cpu()
    query_weights = source_occupancy[source_rows]
    query_weights = query_weights / query_weights.sum().clamp_min(1.0e-12)
    frame_ids = torch.arange(latent_frames, device=device)[None]
    valid = (occupancy_sum > 0) & (frame_ids > source_frame)
    if int(valid.sum().item()) == 0:
        raise RuntimeError("GT-role mask has no valid future latent-frame targets")
    return occupancy, query_rows, query_weights, valid, mapping_audit


def compute_latent_mask_training_objective(
    teacher_frame_head_probabilities: torch.Tensor,
    student_frame_head_probabilities: torch.Tensor,
    *,
    object_token_occupancy_bthw: torch.Tensor,
    valid_frames: torch.Tensor,
    head_weights: torch.Tensor,
    lambda_mask: float,
) -> dict[str, torch.Tensor]:
    """Compute PCK-head-mixture region CE for one object and one video."""
    teacher = teacher_frame_head_probabilities
    student = student_frame_head_probabilities
    if teacher.shape != student.shape or teacher.ndim != 4:
        raise ValueError(
            "expected matching [B,H,T,S] Teacher/Student maps, got "
            f"{teacher.shape}/{student.shape}"
        )
    if object_token_occupancy_bthw.ndim != 4:
        raise ValueError(
            "expected [B,T,H,W] object occupancy, got "
            f"{object_token_occupancy_bthw.shape}"
        )
    batch, heads, frames, spatial = student.shape
    if batch != 1:
        raise ValueError("latent-mask training currently requires batch size 1")
    if object_token_occupancy_bthw.shape[:2] != (batch, frames):
        raise ValueError("occupancy and frame-head maps use different batch/time geometry")
    if math.prod(object_token_occupancy_bthw.shape[-2:]) != spatial:
        raise ValueError("occupancy and frame-head maps use different spatial geometry")
    weights = torch.as_tensor(
        head_weights,
        device=student.device,
        dtype=student.dtype,
    ).flatten()
    if weights.numel() != heads:
        raise ValueError(f"head weight/count mismatch: {weights.numel()}/{heads}")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("head weights must be finite and non-negative")
    weights = weights / weights.sum().clamp_min(1.0e-12)
    teacher_attention = (
        teacher * weights.reshape(1, heads, 1, 1)
    ).sum(dim=1)
    student_attention = (
        student * weights.reshape(1, heads, 1, 1)
    ).sum(dim=1)
    teacher_attention = teacher_attention / teacher_attention.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    student_attention = student_attention / student_attention.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    occupancy = object_token_occupancy_bthw.to(
        device=student.device,
        dtype=student.dtype,
    ).flatten(2)
    target = occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    valid = valid_frames.to(device=student.device, dtype=torch.bool)
    student_objective = uniform_object_region_correspondence_objective(
        student_attention,
        target,
        valid,
        lambda_corr=float(lambda_mask),
    )
    teacher_objective = uniform_object_region_correspondence_objective(
        teacher_attention,
        target,
        valid,
        lambda_corr=float(lambda_mask),
    )
    support = occupancy > 0
    student_mass = (student_attention * support.to(student_attention.dtype)).sum(-1)
    teacher_mass = (teacher_attention * support.to(teacher_attention.dtype)).sum(-1)
    student_top1 = support.gather(
        -1, student_attention.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    teacher_top1 = support.gather(
        -1, teacher_attention.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    return {
        "loss": student_objective["loss"],
        "raw_soft_ce": student_objective["raw_soft_ce"],
        "teacher_raw_soft_ce": teacher_objective["raw_soft_ce"],
        "student_attention": student_attention,
        "teacher_attention": teacher_attention,
        "target": target,
        "valid": valid,
        "student_attention_mass_in_support": student_mass,
        "teacher_attention_mass_in_support": teacher_mass,
        "student_top1_in_support": student_top1,
        "teacher_top1_in_support": teacher_top1,
    }


class FrozenMotionProbeLatentMaskWanModule(legacy.FrozenMotionProbeWanModule):
    """Formal flow + GT-role latent-mask CE training module."""

    def __init__(
        self,
        *args,
        motion_probe_latent_mask_weight: float,
        motion_probe_mask_cache_root: str,
        motion_probe_tracking_mask_key: str,
        motion_probe_expected_pixel_frames: int,
        **kwargs,
    ) -> None:
        self.motion_probe_latent_mask_weight = float(motion_probe_latent_mask_weight)
        self.motion_probe_mask_cache_root = Path(
            motion_probe_mask_cache_root
        ).expanduser().resolve()
        self.motion_probe_tracking_mask_key = str(motion_probe_tracking_mask_key)
        self.motion_probe_expected_pixel_frames = int(
            motion_probe_expected_pixel_frames
        )
        if (
            not math.isfinite(self.motion_probe_latent_mask_weight)
            or self.motion_probe_latent_mask_weight <= 0.0
        ):
            raise ValueError("motion_probe_latent_mask_weight must be positive and finite")
        if self.motion_probe_expected_pixel_frames <= 0:
            raise ValueError("motion_probe_expected_pixel_frames must be positive")
        if not self.motion_probe_mask_cache_root.is_dir():
            raise FileNotFoundError(
                f"tracked mask cache does not exist: {self.motion_probe_mask_cache_root}"
            )
        super().__init__(*args, **kwargs)

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        captured: list[dict[str, Any]] = []
        original_model_fn = pipe.model_fn

        def capture_main_model_fn(*args, **kwargs):
            output = original_model_fn(*args, **kwargs)
            captured.append(
                {
                    "model_output": output,
                    "latents": kwargs.get("latents"),
                    "timestep": kwargs.get("timestep"),
                    "inputs": kwargs,
                }
            )
            return output

        pipe.model_fn = capture_main_model_fn
        try:
            flow_loss, metrics = core.DINOv3XSSCContextSlotsWanModule._compute_object_losses(
                self,
                pipe,
                inputs_shared,
                inputs_posi,
            )
        finally:
            pipe.model_fn = original_model_fn
        if len(captured) != 1:
            raise RuntimeError(
                f"expected exactly one main Student DiT forward, captured {len(captured)}"
            )
        record = captured[0]
        latent_xt = record["latents"]
        model_output = record["model_output"]
        if not isinstance(latent_xt, torch.Tensor) or not isinstance(
            model_output, torch.Tensor
        ):
            raise RuntimeError("main Student capture did not contain latent x_t and v_pred")
        if latent_xt.shape[0] != 1:
            raise RuntimeError("latent-mask training currently requires batch size 1")

        sigma = context_wan._diffsynth_sigma_for_timestep(
            pipe.scheduler,
            record["timestep"],
        ).to(device=latent_xt.device, dtype=latent_xt.dtype)
        while sigma.ndim < latent_xt.ndim:
            sigma = sigma.unsqueeze(-1)
        pred_x0 = latent_xt - sigma * model_output
        target_x0 = inputs_shared["input_latents"].detach()
        pred_x0 = self._restore_condition_latents(
            pred_x0,
            target_x0,
            record["inputs"],
        )
        grid = legacy._probe_grid(self._motion_probe_dit, target_x0)
        if grid[0] != self.motion_probe_expected_latent_frames:
            raise RuntimeError(
                f"Frozen Motion Probe expected {self.motion_probe_expected_latent_frames} "
                f"latent frames, got grid={grid}"
            )
        raw_sample = inputs_shared.get("raw_sample")
        masks_thw, mask_audit = load_sample_gt_role_masks(
            raw_sample,
            cache_root=self.motion_probe_mask_cache_root,
            mask_key=self.motion_probe_tracking_mask_key,
            object_index=self.motion_probe_query_object_index,
            expected_frames=self.motion_probe_expected_pixel_frames,
        )
        (
            occupancy,
            query_rows,
            query_weights,
            valid,
            mapping_audit,
        ) = build_latent_mask_supervision(
            masks_thw,
            grid=grid,
            source_frame=self.motion_probe_query_latent_frame,
            device=target_x0.device,
        )

        epsilon_p = torch.randn_like(target_x0)
        teacher_probe_input = blend_with_fixed_probe_noise(
            target_x0,
            epsilon_p,
            noise_level=self.motion_probe_noise_level,
        )
        student_probe_input = blend_with_fixed_probe_noise(
            pred_x0,
            epsilon_p,
            noise_level=self.motion_probe_noise_level,
        )
        teacher_probe_input = self._restore_condition_latents(
            teacher_probe_input,
            target_x0,
            record["inputs"],
        )
        student_probe_input = self._restore_condition_latents(
            student_probe_input,
            target_x0,
            record["inputs"],
        )
        probe_timestep = torch.full(
            (target_x0.shape[0],),
            self.motion_probe_timestep,
            device=target_x0.device,
            dtype=pipe.torch_dtype,
        )

        with torch.no_grad():
            (
                teacher_heatmap,
                _,
                gt_query_by_block,
                teacher_frame_heads,
            ) = self._run_frozen_probe(
                latents=teacher_probe_input,
                timestep=probe_timestep,
                captured_inputs=record["inputs"],
                query_rows=query_rows,
                frame_query_weights=query_weights,
                return_frame_head_probabilities=True,
                grid=grid,
                retain_input_gradient=False,
                fixed_query_by_block=None,
            )
            teacher_frame_heads = teacher_frame_heads.detach()
            gt_query_by_block = {
                block_id: query.detach()
                for block_id, query in gt_query_by_block.items()
            }
        (
            student_heatmap,
            _,
            _,
            student_frame_heads,
        ) = self._run_frozen_probe(
            latents=student_probe_input,
            timestep=probe_timestep,
            captured_inputs=record["inputs"],
            query_rows=query_rows,
            frame_query_weights=query_weights,
            return_frame_head_probabilities=True,
            grid=grid,
            retain_input_gradient=True,
            fixed_query_by_block=gt_query_by_block,
        )
        if teacher_heatmap.requires_grad:
            raise RuntimeError("teacher Frozen Motion Probe branch must be stop-gradient")
        if not student_heatmap.requires_grad or not student_frame_heads.requires_grad:
            raise RuntimeError("latent-mask Student attention lost its gradient to x0_pred")

        objective = compute_latent_mask_training_objective(
            teacher_frame_heads,
            student_frame_heads,
            object_token_occupancy_bthw=occupancy,
            valid_frames=valid,
            head_weights=self.motion_probe_pck_weights,
            lambda_mask=self.motion_probe_latent_mask_weight,
        )
        mask_loss = objective["loss"]
        self._motion_probe_forward_count += 1
        metrics["train/motion_probe_grad_diag_applied"] = 0.0
        if (
            self._motion_probe_forward_count
            % self.motion_probe_gradient_diagnostics_every_n_forwards
            == 0
        ):
            mask_gradient = torch.autograd.grad(
                mask_loss,
                model_output,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if mask_gradient is None:
                raise RuntimeError(
                    "GT latent-mask loss has no gradient to first-pass v_pred"
                )
            metrics["train/motion_probe_grad_diag_applied"] = 1.0
            metrics["train/motion_probe_grad_v_norm"] = float(
                mask_gradient.detach().float().norm().item()
            )

        total = flow_loss + mask_loss
        selected = objective["valid"]
        student_mass = objective["student_attention_mass_in_support"][selected]
        teacher_mass = objective["teacher_attention_mass_in_support"][selected]
        student_top1 = objective["student_top1_in_support"][selected].float()
        teacher_top1 = objective["teacher_top1_in_support"][selected].float()
        scheduler_sigma = context_wan._diffsynth_sigma_for_timestep(
            pipe.scheduler,
            probe_timestep,
        )
        metrics.update(
            {
                "train/loss_flow": float(flow_loss.detach().item()),
                "train/loss_motion_probe_latent_mask_ce_raw": float(
                    objective["raw_soft_ce"].detach().item()
                ),
                "train/loss_motion_probe_latent_mask_ce_weighted": float(
                    mask_loss.detach().item()
                ),
                "train/loss_motion_probe_teacher_mask_ce_raw": float(
                    objective["teacher_raw_soft_ce"].detach().item()
                ),
                "train/motion_probe_latent_mask_weight": (
                    self.motion_probe_latent_mask_weight
                ),
                "train/motion_probe_valid_target_frames": float(selected.sum().item()),
                "train/motion_probe_query_token_count": float(query_rows.numel()),
                "train/motion_probe_student_attention_mass_in_mask": float(
                    student_mass.mean().detach().item()
                ),
                "train/motion_probe_teacher_attention_mass_in_mask": float(
                    teacher_mass.mean().detach().item()
                ),
                "train/motion_probe_student_top1_in_mask_rate": float(
                    student_top1.mean().detach().item()
                ),
                "train/motion_probe_teacher_top1_in_mask_rate": float(
                    teacher_top1.mean().detach().item()
                ),
                "train/motion_probe_uses_teacher_q_for_student_map": 1.0,
                "train/motion_probe_mask_source_cache": float(
                    mask_audit["source"] == "cache"
                ),
                "train/motion_probe_mask_min_frame_area": float(
                    mask_audit["min_frame_area"]
                ),
                "train/motion_probe_mask_max_frame_area": float(
                    mask_audit["max_frame_area"]
                ),
                "train/motion_probe_mask_reverse_recall": mapping_audit[
                    "reverse_recall"
                ],
                "train/motion_probe_mask_reverse_precision": mapping_audit[
                    "reverse_precision"
                ],
                "train/motion_probe_mask_reverse_iou": mapping_audit["reverse_iou"],
                "train/motion_probe_mask_reverse_missed_gt_pixels": mapping_audit[
                    "missed_gt_pixels"
                ],
                "train/motion_probe_timestep": self.motion_probe_timestep,
                "train/motion_probe_noise_level": self.motion_probe_noise_level,
                "train/motion_probe_scheduler_sigma": float(
                    scheduler_sigma.detach().float().mean().item()
                ),
                "train/motion_probe_pck_weight_power": float(
                    self.motion_probe_pck_weight_power
                ),
                "train/motion_probe_pck_weight_min": float(
                    self.motion_probe_pck_audit["weight_min"]
                ),
                "train/motion_probe_pck_weight_max": float(
                    self.motion_probe_pck_audit["weight_max"]
                ),
                "train/motion_probe_trainable_params": 0.0,
                "train/loss_total": float(total.detach().item()),
            }
        )
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description += (
        " Adds a LoRA-free Frozen Motion Probe with GT-role latent-mask CE."
    )
    group = parser.add_argument_group("frozen_motion_probe_latent_mask")
    group.add_argument(
        "--motion_probe_wan_root",
        default=legacy.DEFAULT_WAN22_BASELINE_ROOT,
    )
    group.add_argument(
        "--motion_probe_head_config",
        default=legacy.DEFAULT_TOP100_CONFIG,
    )
    group.add_argument(
        "--motion_probe_head_subset_id",
        default=legacy.DEFAULT_TOP100_SUBSET,
    )
    group.add_argument(
        "--motion_probe_head_feature_subtype",
        default=legacy.DEFAULT_TOP100_SUBTYPE,
    )
    group.add_argument("--probe_timestep", type=float, required=True)
    group.add_argument("--probe_noise_level", type=float, required=True)
    group.add_argument("--motion_probe_pck_weight_power", type=float, default=30.0)
    group.add_argument("--motion_probe_latent_mask_weight", type=float, default=0.01)
    group.add_argument(
        "--motion_probe_mask_cache_root",
        default=str(DEFAULT_MASK_CACHE_ROOT),
    )
    group.add_argument(
        "--motion_probe_tracking_mask_key",
        default="object_tracking_masks",
        help="Optional raw_sample field; offline cache is used when this field is absent.",
    )
    group.add_argument("--motion_probe_query_latent_frame", type=int, default=1)
    group.add_argument("--motion_probe_query_object_index", type=int, default=0)
    group.add_argument("--motion_probe_expected_latent_frames", type=int, default=13)
    group.add_argument("--motion_probe_expected_pixel_frames", type=int, default=49)
    group.add_argument(
        "--motion_probe_gradient_diagnostics_every_n_forwards",
        type=int,
        default=400,
    )
    group.add_argument(
        "--disable_motion_probe_gradient_checkpointing_offload",
        action="store_true",
    )
    return parser


def build_model(args: argparse.Namespace, accelerator):
    legacy._reject_loaded_lora(args)
    legacy._assert_main_student_uses_same_baseline(args)
    if not args.disable_object_branch:
        raise ValueError("GT latent-mask entry requires --disable_object_branch")
    if int(args.train_batch_size) != 1:
        raise ValueError("GT latent-mask entry currently requires --train_batch_size 1")
    return core.build_model(
        args,
        accelerator,
        model_class=FrozenMotionProbeLatentMaskWanModule,
        extra_model_kwargs={
            "motion_probe_wan_root": args.motion_probe_wan_root,
            "motion_probe_head_config": args.motion_probe_head_config,
            "motion_probe_head_subset_id": args.motion_probe_head_subset_id,
            "motion_probe_head_feature_subtype": (
                args.motion_probe_head_feature_subtype
            ),
            "motion_probe_timestep": args.probe_timestep,
            "motion_probe_noise_level": args.probe_noise_level,
            "motion_probe_pck_weight_power": args.motion_probe_pck_weight_power,
            # Parent-only compatibility values; this subclass never evaluates
            # the legacy KL or trajectory objective.
            "motion_probe_heatmap_weight": 1.0,
            "motion_probe_trajectory_weight": 0.0,
            "motion_probe_trajectory_huber_delta": 0.05,
            "motion_probe_query_latent_frame": args.motion_probe_query_latent_frame,
            "motion_probe_query_object_index": args.motion_probe_query_object_index,
            "motion_probe_query_token_key": "unused_token_indices",
            "motion_probe_query_mask_key": "unused_query_mask",
            "motion_probe_query_points_key": "unused_query_points",
            "motion_probe_expected_latent_frames": (
                args.motion_probe_expected_latent_frames
            ),
            "motion_probe_gradient_checkpointing_offload": (
                not args.disable_motion_probe_gradient_checkpointing_offload
            ),
            "motion_probe_gradient_diagnostics_every_n_forwards": (
                args.motion_probe_gradient_diagnostics_every_n_forwards
            ),
            "motion_probe_device": accelerator.device,
            "motion_probe_latent_mask_weight": args.motion_probe_latent_mask_weight,
            "motion_probe_mask_cache_root": args.motion_probe_mask_cache_root,
            "motion_probe_tracking_mask_key": args.motion_probe_tracking_mask_key,
            "motion_probe_expected_pixel_frames": (
                args.motion_probe_expected_pixel_frames
            ),
        },
    )


def log_stage_summary(accelerator, model, args: argparse.Namespace) -> None:
    core._log_stage_summary(accelerator, model, args)
    if accelerator.is_main_process:
        accelerator.print(
            "Frozen Motion Probe GT latent-mask CE: official Wan2.2 baseline; "
            "loaded LoRA=none; trainable probe params=0; "
            f"probe_timestep={args.probe_timestep:g}; "
            f"probe_noise_level={args.probe_noise_level:g}; "
            f"PCK power={args.motion_probe_pck_weight_power:g}; "
            f"lambda_mask={args.motion_probe_latent_mask_weight:g}; "
            f"mask_cache={Path(args.motion_probe_mask_cache_root).resolve()}; "
            "loss=flow + lambda_mask*future_frame_region_CE"
        )


def main() -> None:
    core.main(
        build_parser_fn=build_parser,
        build_model_fn=build_model,
        log_stage_summary_fn=log_stage_summary,
        require_pretrained_lora=False,
    )


if __name__ == "__main__":
    main()
