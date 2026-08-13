#!/usr/bin/env python3
"""Equal-budget point-correspondence guidance for two Wan context protocols.

Protocol A (``firstframe_ti2v``) uses the legacy first-frame TI2V model and
guides the observed point Query at R0 toward the same tracked point at future
Keys R1..R12.

Protocol B (``context8_v2v``) loads the requested Full-SA xSSC checkpoint,
actually encodes eight RGB context frames into clean prefix latents R0,R1, and
guides observed point Queries R0,R1 toward an equal mixture of the same point
Keys at R2..R12.  The latest3350 ranking is intentionally transferred unchanged into
this checkpoint and is therefore a ranking-transfer diagnostic, not a claim
that these are the checkpoint's native Top100 heads.

At every guided denoising step, the current noisy latent is updated directly:

    x'_s = x_s - eta * grad(L) / RMS_mutable(grad(L))

The clean context prefix is immutable.  The positive and negative CFG branches
are then recomputed at x'_s before the ordinary FlowMatch scheduler step.  All
head groups receive exactly the same mutable-latent RMS update budget.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint


DIFFTRACK_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
for import_root in (DIFFTRACK_ROOT, CODE_ROOT, DIFFSYNTH_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from AAA_my_test import run_wan_gt_spatiotemporal_correspondence_guidance as legacy  # noqa: E402
from AAA_my_test.wan_context_point_guidance.attention_audit_visualization import (  # noqa: E402
    write_step_attention_audit,
)
from AAA_my_test.wan_context_point_guidance.protocol_core import (  # noqa: E402
    fixed_mutable_rms_delta,
    forward_attention_audit_sums,
    global_forward_point_loss,
    load_head_groups,
    points_to_token_rows,
    transform_points_stretch_to_cover_crop,
    valid_correspondence_count,
)
from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import save_video_np  # noqa: E402
from code_vjepa_vggt.utils.video_io import (  # noqa: E402
    preprocess_video_rgb_uint8,
    read_video_prefix,
)


PROTOCOL = "wan_equal_budget_forward_point_guidance_attention_audit_v3"
DEFAULT_INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
DEFAULT_RANKING = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/"
    "visual_samples/attention_zero_seed47326/pck_head_scopes_s039_latest3350.json"
)
DEFAULT_SCOPES = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/head_scopes_latest3350_with_random100.json"
)
DEFAULT_TUBE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2"
)
DEFAULT_TARGET_MAP = DEFAULT_TUBE_ROOT / "screening/seed_47326/baseline_eligibility.json"
DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/agent-data/checkpoints/xssc_feature_loss/"
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000/formal_gpu01/"
    "checkpoints/step-000500"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/attention_audit_v3"
)
NEGATIVE_PROMPT = "模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"
HEAD_DIM = 128
LATENT_FRAMES = 13
PIXEL_FRAMES = 49
LATENT_ANCHORS = np.arange(LATENT_FRAMES, dtype=np.int64) * 4
DEFAULT_ATTENTION_CAPTURE_STEPS = tuple(range(5, 41, 5))


@dataclass(frozen=True)
class BackendSpec:
    name: str
    height: int
    width: int
    context_rgb_frames: int
    context_latent_frames: int
    query_times: tuple[int, ...]
    key_times: tuple[int, ...]
    ranking_semantics: str


BACKENDS = {
    "firstframe_ti2v": BackendSpec(
        name="firstframe_ti2v",
        height=704,
        width=1280,
        context_rgb_frames=1,
        context_latent_frames=1,
        query_times=tuple(range(1, 13)),
        key_times=(0,),
        ranking_semantics="native latest3350 ranking on the legacy TI2V model",
    ),
    "context8_v2v": BackendSpec(
        name="context8_v2v",
        height=512,
        width=896,
        context_rgb_frames=8,
        context_latent_frames=2,
        query_times=tuple(range(2, 13)),
        key_times=(0, 1),
        ranking_semantics=(
            "latest3350 legacy ranking transferred unchanged into the Full-SA "
            "xSSC checkpoint; not checkpoint-native PCK ranking"
        ),
    ),
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serializable(item) for item in value]
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    return value


def source_geometry(source_video: Path) -> tuple[int, int]:
    frame = np.asarray(iio.imread(source_video, index=0))
    if frame.ndim != 3:
        raise RuntimeError(f"invalid source frame shape: {frame.shape}")
    return int(frame.shape[0]), int(frame.shape[1])


def source_anchor_frames(
    tube: legacy.FrozenTube,
    spec: BackendSpec,
) -> np.ndarray:
    """Decode the same 13 source frames in the exact backend pixel geometry."""
    frames = np.asarray(iio.imread(tube.source_video))[:PIXEL_FRAMES, ..., :3]
    anchors = np.asarray(tube.anchor_source_frames, dtype=np.int64)
    if anchors.shape != (LATENT_FRAMES,):
        raise RuntimeError(f"expected 13 source anchors, got {anchors.shape}")
    if anchors.min() < 0 or anchors.max() >= len(frames):
        raise RuntimeError(
            f"source anchors {anchors.tolist()} exceed {len(frames)} decoded frames"
        )
    if spec.name == "firstframe_ti2v":
        resized = legacy.resize_frames(frames, spec.height, spec.width)
    else:
        tensor = preprocess_video_rgb_uint8(
            frames,
            (spec.height, spec.width),
            resize_mode="cover_crop",
            cover_crop_hw=(spec.height, spec.width),
        )
        resized = (
            ((tensor.permute(1, 2, 3, 0).float() + 1.0) * 127.5)
            .round()
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
        )
    return np.asarray(resized, dtype=np.uint8)[anchors]


def backend_tracks(
    tube: legacy.FrozenTube,
    spec: BackendSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if spec.name == "firstframe_ti2v":
        return (
            tube.tracks_tn2.copy(),
            tube.visibility_tn.copy(),
            {"mapping": "identity legacy 704x1280 stretch geometry"},
        )
    transformed, in_frame, geometry = transform_points_stretch_to_cover_crop(
        tube.tracks_tn2,
        source_hw=source_geometry(tube.source_video),
        stretched_hw=(tube.pixel_height, tube.pixel_width),
        crop_hw=(spec.height, spec.width),
        output_hw=(spec.height, spec.width),
    )
    visibility = tube.visibility_tn & in_frame
    return transformed, visibility, {
        "mapping": "legacy stretch coordinates -> original source -> exact Wan cover crop",
        **geometry,
    }


def target_point_arrays(
    tube: legacy.FrozenTube,
    target: legacy.GuidanceTarget,
    spec: BackendSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tracks, visibility, geometry = backend_tracks(tube, spec)
    point_indices = np.concatenate(
        [
            np.arange(tube.point_starts[index], tube.point_ends[index], dtype=np.int64)
            for index in target.object_indices
        ]
    )
    selected_tracks = tracks[:, point_indices]
    selected_visibility = visibility[:, point_indices]
    context_visible = np.logical_or.reduce(
        selected_visibility[list(spec.key_times)], axis=0
    )
    count = sum(
        int((selected_visibility[query_time] & context_visible).sum())
        for query_time in spec.query_times
    )
    if count <= 0:
        raise RuntimeError(
            f"no visible correspondences for {tube.case}/{target.name}/{spec.name}"
        )
    return selected_tracks, selected_visibility, {
        **geometry,
        "selected_point_count": int(len(point_indices)),
        "visible_future_context_correspondences": int(count),
    }


class GlobalPointCollector:
    """Collect context-Query to future-Key CE from selected attention heads."""

    def __init__(
        self,
        pipe: Any,
        entries: list[dict[str, Any]],
        tracks_tn2: np.ndarray,
        visibility_tn: np.ndarray,
        pixel_hw: tuple[int, int],
        query_times: tuple[int, ...],
        key_times: tuple[int, ...],
        sigma_tokens: float,
    ) -> None:
        self.pipe = pipe
        self.entries = entries
        self.tracks_tn2 = np.asarray(tracks_tn2, dtype=np.float32)
        self.visibility_tn = torch.from_numpy(
            np.asarray(visibility_tn, dtype=bool)
        )
        self.pixel_hw = tuple(int(value) for value in pixel_hw)
        self.query_times = tuple(int(value) for value in query_times)
        self.key_times = tuple(int(value) for value in key_times)
        self.sigma_tokens = float(sigma_tokens)
        self.by_block: dict[int, list[int]] = {}
        for row in entries:
            self.by_block.setdefault(int(row["block"]), []).append(int(row["head"]))
        self.active = False
        self.current_grid: tuple[int, int, int] | None = None
        self.losses: list[tuple[torch.Tensor, int]] = []
        self.head_events = 0
        self.term_count = 0
        self._geometry_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._originals: list[tuple[Any, Any]] = []
        self._capture_phase: str | None = None
        self._capture_step: int | None = None
        self._capture_seen_blocks: set[int] = set()
        self._capture_sums: dict[str, np.ndarray] = {}

    def point_rows(self, token_hw: tuple[int, int]) -> torch.Tensor:
        cached = self._geometry_cache.get(token_hw)
        if cached is None:
            cached = points_to_token_rows(self.tracks_tn2, self.pixel_hw, token_hw)
            self._geometry_cache[token_hw] = cached
        return cached

    def reset(self, grid: tuple[int, int, int]) -> None:
        if int(grid[0]) != LATENT_FRAMES:
            raise RuntimeError(f"expected 13 latent frames, got {grid}")
        self.current_grid = tuple(int(value) for value in grid)
        self.losses.clear()
        self.head_events = 0
        self.term_count = 0

    def begin_capture(self, step: int, phase: str) -> None:
        if phase not in {"pre", "post"}:
            raise ValueError(f"unknown attention capture phase: {phase}")
        if self.current_grid is None:
            raise RuntimeError("collector grid is unset")
        self._capture_phase = phase
        self._capture_step = int(step)
        self._capture_seen_blocks.clear()
        self._capture_sums.clear()

    def _capture_attention(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        rows: torch.Tensor,
        token_hw: tuple[int, int],
        block: int,
    ) -> None:
        if self._capture_phase is None or block in self._capture_seen_blocks:
            return
        self._capture_seen_blocks.add(block)
        with torch.no_grad():
            values = forward_attention_audit_sums(
                q_heads,
                k_heads,
                rows,
                self.visibility_tn,
                token_hw,
                self.key_times,
                self.query_times,
                self.sigma_tokens,
            )
        for name, value in values.items():
            array = value.detach().float().cpu().numpy()
            if name in self._capture_sums:
                self._capture_sums[name] += array
            else:
                self._capture_sums[name] = array.copy()

    def end_capture(self) -> dict[str, Any]:
        if self._capture_phase is None or self._capture_step is None:
            raise RuntimeError("attention capture was not started")
        if not self._capture_sums:
            raise RuntimeError("attention capture produced no selected-head values")
        heat_count = self._capture_sums["heatmap_pair_count"]
        metric_count = self._capture_sums["metric_pair_count"]
        heat_denominator = np.maximum(heat_count, 1.0)
        metric_denominator = np.maximum(metric_count, 1.0)
        result: dict[str, Any] = {
            "step_index": int(self._capture_step),
            "step_1based": int(self._capture_step + 1),
            "phase": self._capture_phase,
            "heatmap": (
                self._capture_sums["heatmap_sum"]
                / heat_denominator[:, None, None]
            ).astype(np.float32),
            "heatmap_pair_count": heat_count.astype(np.int64),
            "frame_mass": (
                self._capture_sums["frame_mass_sum"] / heat_denominator
            ).astype(np.float32),
            "localized_mass": np.where(
                metric_count > 0,
                self._capture_sums["localized_mass_sum"] / metric_denominator,
                np.nan,
            ).astype(np.float32),
            "peak_distance_tokens": np.where(
                metric_count > 0,
                self._capture_sums["peak_distance_sum"] / metric_denominator,
                np.nan,
            ).astype(np.float32),
            "peak_hit_rate_2sigma": np.where(
                metric_count > 0,
                self._capture_sums["peak_hit_sum"] / metric_denominator,
                np.nan,
            ).astype(np.float32),
            "metric_pair_count": metric_count.astype(np.int64),
            "captured_blocks": int(len(self._capture_seen_blocks)),
        }
        self._capture_phase = None
        self._capture_step = None
        self._capture_seen_blocks.clear()
        self._capture_sums.clear()
        return result

    def cancel_capture(self) -> None:
        self._capture_phase = None
        self._capture_step = None
        self._capture_seen_blocks.clear()
        self._capture_sums.clear()

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original: Any,
        block: int,
    ) -> torch.Tensor:
        heads = self.by_block.get(block, ())
        if not self.active or not heads:
            return original(q, k, v)
        if self.current_grid is None:
            raise RuntimeError("collector grid is unset")
        time_count, token_height, token_width = self.current_grid
        if q.shape[1] != time_count * token_height * token_width:
            raise RuntimeError(
                f"Q geometry mismatch: {q.shape} vs {self.current_grid}"
            )
        num_heads = int(q.shape[-1] // HEAD_DIM)
        if num_heads != 24:
            raise RuntimeError(f"expected 24 Wan heads, got {num_heads}")
        selected = torch.as_tensor(heads, device=q.device, dtype=torch.long)
        # Advanced indexing creates compact copies so the side loss does not
        # retain every unselected Q/K activation until backward.
        q_heads = q.view(q.shape[0], q.shape[1], num_heads, HEAD_DIM)[
            :, :, selected
        ].contiguous()
        k_heads = k.view(k.shape[0], k.shape[1], num_heads, HEAD_DIM)[
            :, :, selected
        ].contiguous()
        rows = self.point_rows((token_height, token_width))
        self._capture_attention(
            q_heads,
            k_heads,
            rows,
            (token_height, token_width),
            block,
        )

        def compute_loss(q_selected: torch.Tensor, k_selected: torch.Tensor) -> torch.Tensor:
            return global_forward_point_loss(
                q_selected,
                k_selected,
                rows,
                self.visibility_tn,
                (token_height, token_width),
                self.key_times,
                self.query_times,
                self.sigma_tokens,
            )

        if torch.is_grad_enabled() and (q_heads.requires_grad or k_heads.requires_grad):
            loss = checkpoint(compute_loss, q_heads, k_heads, use_reentrant=False)
        else:
            loss = compute_loss(q_heads, k_heads)
        self.losses.append((loss, len(heads)))
        self.head_events += len(heads)
        self.term_count += valid_correspondence_count(
            self.visibility_tn, self.key_times, self.query_times
        ) * len(heads)
        return original(q, k, v)

    def install(self) -> None:
        models = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            models.append(self.pipe.dit2)
        for model in models:
            for block in self.by_block:
                module = model.blocks[block].self_attn.attn
                original = module.forward
                self._originals.append((module, original))

                def wrapped(q, k, v, *, _original=original, _block=block):
                    return self._attention(q, k, v, _original, _block)

                module.forward = wrapped

    def remove(self) -> None:
        for module, original in self._originals:
            module.forward = original
        self._originals.clear()

    def total_loss(self) -> torch.Tensor:
        if self.head_events != len(self.entries):
            raise RuntimeError(
                f"expected {len(self.entries)} selected-head events, got {self.head_events}"
            )
        if not self.losses:
            raise RuntimeError("no global point loss was collected")
        numerator = torch.stack(
            [loss * head_count for loss, head_count in self.losses]
        ).sum()
        return numerator / float(sum(count for _, count in self.losses))


def freeze_pipe(pipe: Any) -> None:
    pipe.eval()
    pipe.requires_grad_(False)
    if any(parameter.requires_grad for parameter in pipe.parameters()):
        raise RuntimeError("failed to freeze Wan parameters")


def prepare_context8_inputs(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    context_pil: list[Image.Image],
    seed: int,
    cfg_scale: float,
    height: int,
    width: int,
    steps: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pipe.scheduler.set_timesteps(steps, denoising_strength=1.0, shift=5.0)
    inputs_posi = {
        "prompt": prompt,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": steps,
    }
    inputs_nega = {
        "negative_prompt": negative_prompt,
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": steps,
    }
    inputs_shared = {
        "input_image": context_pil[0],
        "end_image": None,
        "input_video": None,
        "context_video": context_pil,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (
            0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 1, 0,
        ),
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1.0,
        "seed": int(seed),
        "rand_device": "cpu",
        "height": int(height),
        "width": int(width),
        "num_frames": PIXEL_FRAMES,
        "cfg_scale": float(cfg_scale),
        "cfg_merge": False,
        "sigma_shift": 5.0,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
        "wantodance_music_path": None,
        "wantodance_reference_image": None,
        "wantodance_fps": 30,
        "wantodance_keyframes": None,
        "wantodance_keyframes_mask": None,
        "object_context": None,
        "framewise_decoding": False,
    }
    with torch.no_grad():
        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )
    clean = inputs_shared.get("clean_prefix_latents")
    if clean is None or int(clean.shape[2]) != 2:
        raise RuntimeError(
            f"8-frame V2V must produce two clean prefix latents, got "
            f"{None if clean is None else tuple(clean.shape)}"
        )
    return inputs_shared, inputs_posi, inputs_nega


def load_context_pil(tube: legacy.FrozenTube, spec: BackendSpec) -> list[Image.Image]:
    frames, indices = read_video_prefix(tube.source_video, spec.context_rgb_frames)
    if len(frames) != spec.context_rgb_frames:
        raise RuntimeError(
            f"{tube.case}: requested {spec.context_rgb_frames} context frames, got {len(frames)}"
        )
    context_tensor = preprocess_video_rgb_uint8(
        frames,
        (spec.height, spec.width),
        resize_mode="cover_crop",
        cover_crop_hw=(spec.height, spec.width),
    )
    pixels = (
        ((context_tensor.permute(1, 2, 3, 0).float() + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )
    return [Image.fromarray(frame, mode="RGB") for frame in pixels]


def prepare_backend_inputs(
    pipe: Any,
    spec: BackendSpec,
    payload: dict[str, Any],
    tube: legacy.FrozenTube,
    seed: int,
    cfg_scale: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if spec.name == "firstframe_ti2v":
        source_frames = legacy.read_source_prefix(tube.source_video)
        input_image = legacy.resolve_condition_image(payload, source_frames)
        return legacy.prepare_wan_inputs(
            pipe,
            str(payload["input_caption"]),
            NEGATIVE_PROMPT,
            input_image,
            seed,
            cfg_scale,
            40,
            5.0,
        )
    return prepare_context8_inputs(
        pipe,
        str(payload["input_caption"]),
        NEGATIVE_PROMPT,
        load_context_pil(tube, spec),
        seed,
        cfg_scale,
        spec.height,
        spec.width,
        40,
    )


def active_grid(latent: torch.Tensor, dit: Any) -> tuple[int, int, int]:
    patch = tuple(int(value) for value in dit.patch_size)
    return (
        int(latent.shape[2] // patch[0]),
        int(latent.shape[3] // patch[1]),
        int(latent.shape[4] // patch[2]),
    )


def restore_context(inputs_shared: dict[str, Any], context_latent_frames: int) -> None:
    if inputs_shared.get("clean_prefix_latents") is not None:
        clean = inputs_shared["clean_prefix_latents"]
        if int(clean.shape[2]) != context_latent_frames:
            raise RuntimeError("clean prefix length differs from frozen protocol")
        inputs_shared["latents"][:, :, :context_latent_frames] = clean
    elif "first_frame_latents" in inputs_shared:
        if context_latent_frames != 1:
            raise RuntimeError("first-frame condition is only valid for one latent")
        inputs_shared["latents"][:, :, :1] = inputs_shared["first_frame_latents"]
    else:
        raise RuntimeError("backend did not expose an immutable context latent")


def model_forward(
    pipe: Any,
    models: dict[str, Any],
    shared: dict[str, Any],
    branch: dict[str, Any],
    timestep: torch.Tensor,
    gradient_checkpointing: bool = False,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if gradient_checkpointing:
        kwargs["use_gradient_checkpointing"] = True
    return pipe.model_fn(**models, **shared, **branch, timestep=timestep, **kwargs)


def direct_update_at_step(
    pipe: Any,
    models: dict[str, Any],
    inputs_shared: dict[str, Any],
    inputs_posi: dict[str, Any],
    timestep: torch.Tensor,
    collector: GlobalPointCollector,
    spec: BackendSpec,
    update_rms: float,
    use_gradient_checkpointing: bool,
    step: int,
    capture_attention: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any] | None]:
    latents = inputs_shared["latents"].detach()
    latent_leaf = latents.requires_grad_(True)
    inputs_shared["latents"] = latent_leaf
    grid = active_grid(latent_leaf, models["dit"])
    collector.reset(grid)
    if capture_attention:
        collector.begin_capture(step, "pre")
    collector.active = True
    try:
        with torch.enable_grad():
            pre_noise = model_forward(
                pipe,
                models,
                inputs_shared,
                inputs_posi,
                timestep,
                gradient_checkpointing=use_gradient_checkpointing,
            )
            pre_loss = collector.total_loss()
            head_events = collector.head_events
            term_count = collector.term_count
            gradient = torch.autograd.grad(pre_loss, latent_leaf, only_inputs=True)[0]
    except BaseException:
        collector.cancel_capture()
        raise
    finally:
        collector.active = False
    pre_capture = collector.end_capture() if capture_attention else None
    delta, update_audit = fixed_mutable_rms_delta(
        gradient, spec.context_latent_frames, update_rms
    )
    updated = (latents + delta).detach()
    inputs_shared["latents"] = updated
    restore_context(inputs_shared, spec.context_latent_frames)
    updated = inputs_shared["latents"].detach()
    # Reuse the required post-update positive CFG forward to verify that the
    # direct latent step actually decreases the registered correspondence loss.
    collector.reset(grid)
    if capture_attention:
        collector.begin_capture(step, "post")
    collector.active = True
    try:
        with torch.no_grad():
            post_noise = model_forward(
                pipe, models, inputs_shared, inputs_posi, timestep, False
            )
            post_loss = collector.total_loss()
    except BaseException:
        collector.cancel_capture()
        raise
    finally:
        collector.active = False
        collector.losses.clear()
    post_capture = collector.end_capture() if capture_attention else None
    pre_value = float(pre_loss.detach().cpu())
    post_value = float(post_loss.detach().cpu())
    audit = {
        "pre_update_loss": pre_value,
        "post_update_loss": post_value,
        "loss_change": post_value - pre_value,
        "loss_decreased": bool(post_value < pre_value),
        "selected_head_events": int(head_events),
        "correspondence_terms": int(term_count),
        **update_audit,
    }
    del pre_noise, pre_loss, post_loss, gradient, delta, latent_leaf
    captures = (
        {"pre": pre_capture, "post": post_capture}
        if pre_capture is not None and post_capture is not None
        else None
    )
    return updated, post_noise, audit, captures


def run_denoising(
    pipe: Any,
    spec: BackendSpec,
    inputs_shared: dict[str, Any],
    inputs_posi: dict[str, Any],
    inputs_nega: dict[str, Any],
    collector: GlobalPointCollector | None,
    cfg_scale: float,
    update_rms: float,
    guidance_start: int,
    guidance_end: int,
    use_gradient_checkpointing: bool,
    attention_capture_steps_1based: tuple[int, ...] = (),
    stop_after_step: int | None = None,
) -> tuple[
    np.ndarray | None,
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, np.ndarray],
]:
    freeze_pipe(pipe)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    audit: list[dict[str, Any]] = []
    attention_captures: dict[int, dict[str, Any]] = {}
    predicted_x0_latents: dict[int, torch.Tensor] = {}
    capture_steps = {int(value) for value in attention_capture_steps_1based}
    for step, scheduler_timestep in enumerate(pipe.scheduler.timesteps):
        if (
            scheduler_timestep.item() < 0.875 * 1000
            and getattr(pipe, "dit2", None) is not None
            and models["dit"] is not pipe.dit2
        ):
            pipe.load_models_to_device(pipe.in_iteration_models_2)
            models["dit"] = pipe.dit2
            models["vace"] = pipe.vace2
        timestep = scheduler_timestep.unsqueeze(0).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        latents = inputs_shared["latents"].detach()
        guided = collector is not None and guidance_start <= step <= guidance_end
        capture_attention = guided and (step + 1) in capture_steps
        if guided:
            latents_for_step, noise_pos, row, capture_row = direct_update_at_step(
                pipe,
                models,
                inputs_shared,
                inputs_posi,
                timestep,
                collector,
                spec,
                update_rms,
                use_gradient_checkpointing,
                step,
                capture_attention,
            )
            if capture_row is not None:
                attention_captures[step + 1] = capture_row
        else:
            inputs_shared["latents"] = latents
            with torch.no_grad():
                noise_pos = model_forward(
                    pipe, models, inputs_shared, inputs_posi, timestep, False
                )
            latents_for_step = latents
            row = {
                "pre_update_loss": None,
                "post_update_loss": None,
                "loss_change": None,
                "loss_decreased": None,
                "selected_head_events": 0,
                "correspondence_terms": 0,
                "raw_mutable_gradient_rms": 0.0,
                "requested_mutable_update_rms": 0.0,
                "actual_mutable_update_rms": 0.0,
                "context_update_abs_max": 0.0,
            }
        inputs_shared["latents"] = latents_for_step
        with torch.no_grad():
            noise_neg = model_forward(
                pipe, models, inputs_shared, inputs_nega, timestep, False
            )
            noise_cfg = noise_neg + float(cfg_scale) * (noise_pos - noise_neg)
            if capture_attention:
                sigma = float(pipe.scheduler.sigmas[step])
                predicted_x0 = (latents_for_step - sigma * noise_cfg).detach().clone()
                predicted_x0[:, :, : spec.context_latent_frames] = latents_for_step[
                    :, :, : spec.context_latent_frames
                ]
                predicted_x0_latents[step + 1] = predicted_x0.to(
                    device="cpu", dtype=torch.float16
                )
                del predicted_x0
            inputs_shared["latents"] = pipe.scheduler.step(
                noise_cfg, scheduler_timestep, latents_for_step
            )
            restore_context(inputs_shared, spec.context_latent_frames)
        audit.append(
            {
                "step": int(step),
                "timestep": float(scheduler_timestep.detach().cpu()),
                "sigma": float(pipe.scheduler.sigmas[step]),
                "guided": bool(guided),
                **row,
            }
        )
        print(
            f"[denoise] backend={spec.name} step={step:02d} guided={guided} "
            f"pre={row['pre_update_loss']} post={row['post_update_loss']}",
            flush=True,
        )
        del noise_pos, noise_neg, noise_cfg
        if stop_after_step is not None and step >= stop_after_step:
            return None, audit, attention_captures, {}
    with torch.no_grad():
        for unit in pipe.post_units:
            inputs_shared, _, _ = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )
        pipe.load_models_to_device(["vae"])
        decoded = pipe.vae.decode(
            inputs_shared["latents"],
            device=pipe.device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        video = pipe.vae_output_to_video(decoded)
        predicted_x0_frames: dict[int, np.ndarray] = {}
        for capture_step, latent_cpu in sorted(predicted_x0_latents.items()):
            latent = latent_cpu.to(device=pipe.device, dtype=pipe.torch_dtype)
            predicted_decoded = pipe.vae.decode(
                latent,
                device=pipe.device,
                tiled=True,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )
            predicted_video = pipe.vae_output_to_video(predicted_decoded)
            if len(predicted_video) < PIXEL_FRAMES:
                raise RuntimeError(
                    f"predicted-x0 decode returned {len(predicted_video)} frames"
                )
            predicted_x0_frames[capture_step] = np.stack(
                [
                    np.asarray(predicted_video[int(index)].convert("RGB"), dtype=np.uint8)
                    for index in LATENT_ANCHORS
                ]
            )
            del latent, predicted_decoded, predicted_video
        pipe.load_models_to_device([])
    frames = np.stack(
        [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in video]
    )
    return frames, audit, attention_captures, predicted_x0_frames


def build_context8_model(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from code_vjepa_vggt.train0705_kubric_no_gt_box import (
        wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch,
    )
    from code_vjepa_vggt.train_xSSC.object_self_attn_lora_experiments import (
        infer_xssc_object_self_attn_lora as experiment,
    )

    argv = [
        "run_dual_protocol.py",
        "--weights-root", str(args.checkpoint),
        "--input-json-list-path", str(args.input_list),
        "--model-name", "equal_budget_context8_v2v",
        "--output-root", str(args.output_root / "_runtime"),
        "--device", str(args.device),
        "--aux-device", str(args.device),
        "--height", "512",
        "--width", "896",
        "--num-frames", "49",
        "--context-frames", "8",
        "--sampling-mode", "prefix",
        "--num-inference-steps", "40",
        "--negative-prompt", NEGATIVE_PROMPT,
    ]
    original_argv = sys.argv
    try:
        sys.argv = argv
        cli_args = batch.parse_args()
    finally:
        sys.argv = original_argv
    experiment._install_runtime_hooks()
    runtime_args = batch._build_runtime_args(
        cli_args, args.checkpoint.resolve(), args.output_root / "_runtime"
    )
    model, _, info = experiment._build_runtime_model(runtime_args)
    if bool(getattr(model, "enable_object_branch", True)):
        raise RuntimeError("requested context8 checkpoint must have object branch disabled")
    return model, info


def build_backend(args: argparse.Namespace, spec: BackendSpec) -> tuple[Any, Any, dict[str, Any]]:
    if spec.name == "firstframe_ti2v":
        wrapper = legacy.build_pipeline(args.seed)
        return wrapper, wrapper.pipe, {"model": "legacy Wan2.2 TI2V-5B"}
    model, info = build_context8_model(args)
    return model, model.pipe, info


def release_backend(owner: Any) -> None:
    if owner is None:
        return
    if hasattr(owner, "pipe"):
        try:
            owner.pipe.load_models_to_device([])
        except Exception:
            pass
    del owner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generation_dir(
    root: Path,
    spec: BackendSpec,
    case: str,
    seed: int,
    variant: str,
) -> Path:
    return root / spec.name / "generations" / case / f"seed_{seed:05d}" / variant


def load_cases(args: argparse.Namespace) -> tuple[list[Path], dict[str, tuple[str, ...]]]:
    paths = legacy.deduplicated_json_paths(args.input_list.resolve())
    target_map = legacy.load_target_map(args.target_map.resolve())
    selected = set(args.case_keys or target_map)
    unknown = selected - {path.stem for path in paths}
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    paths = [path for path in paths if path.stem in selected]
    return paths[args.worker_id :: args.num_workers], target_map


def task_manifest(
    args: argparse.Namespace,
    spec: BackendSpec,
    case_paths: list[Path],
    target_map: dict[str, tuple[str, ...]],
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    target_count = sum(len(target_map[path.stem]) for path in case_paths)
    return {
        "protocol": PROTOCOL,
        "backend": serializable(spec.__dict__),
        "case_count": len(case_paths),
        "target_count": int(target_count),
        "seed_count_per_case": 1,
        "seed": int(args.seed),
        "baseline_video_count": len(case_paths),
        "guided_video_count": int(target_count * len(groups)),
        "total_video_count": int(len(case_paths) + target_count * len(groups)),
        "head_groups": {name: rows for name, rows in groups.items()},
        "equal_budget": {
            "mutable_latent_update_rms_per_guided_step": float(args.latent_update_rms),
            "context_latent_update": 0.0,
            "guided_step_range_inclusive": [args.guidance_start, args.guidance_end],
        },
        "loss": {
            "normalization": "one softmax over all T*H*W Wan keys",
            "direction": "observed-context Query -> future Key",
            "target": "equal mixture of same-ID CoTracker point Gaussians at visible future latents",
            "context_query_times": list(spec.key_times),
            "future_key_times": list(spec.query_times),
            "sigma_tokens": float(args.gaussian_sigma_tokens),
        },
        "attention_audit": {
            "denoising_step_numbering": "1-based",
            "capture_steps": list(args.attention_capture_steps),
            "pre": "same guided run at x_s before the normalized latent update",
            "post": "same guided step after x_s' = x_s - eta*normalized_gradient",
            "normalization": "Wan global softmax over all 13*H*W Keys",
            "overlay_background": "the same source/GT frame for Pre and Post",
            "panels": [
                "source GT/pseudo-GT",
                "Pre original attention",
                "Post constrained attention",
                "Post-Pre attention",
                "post-guidance predicted x0",
            ],
        },
        "cases": [
            {"case": path.stem, "targets": list(target_map[path.stem])}
            for path in case_paths
        ],
    }


def build_collector(
    pipe: Any,
    spec: BackendSpec,
    tube: legacy.FrozenTube,
    target: legacy.GuidanceTarget,
    entries: list[dict[str, Any]],
    sigma_tokens: float,
) -> tuple[GlobalPointCollector, dict[str, Any]]:
    tracks, visibility, geometry = target_point_arrays(tube, target, spec)
    collector = GlobalPointCollector(
        pipe,
        entries,
        tracks,
        visibility,
        (spec.height, spec.width),
        spec.query_times,
        spec.key_times,
        sigma_tokens,
    )
    return collector, geometry


def run_sanity(
    args: argparse.Namespace,
    spec: BackendSpec,
    pipe: Any,
    case_path: Path,
    target_map: dict[str, tuple[str, ...]],
    groups: dict[str, list[dict[str, Any]]],
) -> None:
    tube = legacy.load_frozen_tube(args.tube_root, case_path.stem)
    payload = legacy.load_payload(case_path)
    target = legacy.selected_target_specs(tube, target_map[tube.case])[0]
    group_name = next(iter(groups))
    collector, geometry = build_collector(
        pipe,
        spec,
        tube,
        target,
        groups[group_name],
        args.gaussian_sigma_tokens,
    )
    collector.install()
    try:
        inputs = prepare_backend_inputs(pipe, spec, payload, tube, args.seed, args.cfg_scale)
        _, audit, captures, _ = run_denoising(
            pipe,
            spec,
            *inputs,
            collector,
            args.cfg_scale,
            args.latent_update_rms,
            0,
            0,
            not args.no_gradient_checkpointing,
            attention_capture_steps_1based=(1,),
            stop_after_step=0,
        )
    finally:
        collector.remove()
    report = {
        "protocol": PROTOCOL,
        "backend": spec.name,
        "case": tube.case,
        "target": target.name,
        "head_group": group_name,
        "geometry": geometry,
        "step": audit[0],
        "attention_capture": {
            "captured": 1 in captures,
            "pre_blocks": int(captures.get(1, {}).get("pre", {}).get("captured_blocks", 0)),
            "post_blocks": int(captures.get(1, {}).get("post", {}).get("captured_blocks", 0)),
        },
        "passed": bool(
            audit[0]["loss_decreased"]
            and audit[0]["context_update_abs_max"] == 0.0
            and abs(
                audit[0]["actual_mutable_update_rms"] - args.latent_update_rms
            ) < 1.0e-5
            and 1 in captures
            and captures[1]["pre"]["captured_blocks"] > 0
            and captures[1]["post"]["captured_blocks"] > 0
        ),
    }
    atomic_json(args.output_root / spec.name / "sanity.json", report)
    if not report["passed"]:
        raise RuntimeError(f"sanity check failed: {report}")


def run_generate(
    args: argparse.Namespace,
    spec: BackendSpec,
    pipe: Any,
    case_paths: list[Path],
    target_map: dict[str, tuple[str, ...]],
    groups: dict[str, list[dict[str, Any]]],
    runtime_info: dict[str, Any],
) -> None:
    for case_path in case_paths:
        tube = legacy.load_frozen_tube(args.tube_root, case_path.stem)
        payload = legacy.load_payload(case_path)
        targets = legacy.selected_target_specs(tube, target_map[tube.case])
        source_anchors: np.ndarray | None = None
        tasks: list[tuple[str, legacy.GuidanceTarget | None, list[dict[str, Any]] | None]] = []
        if not args.no_baseline:
            tasks.append(("baseline", None, None))
        for target in targets:
            tasks.extend((group_name, target, entries) for group_name, entries in groups.items())
        for group_name, target, entries in tasks:
            variant = "baseline" if target is None else f"{group_name}__{target.name}"
            output = generation_dir(
                args.output_root, spec, tube.case, args.seed, variant
            )
            required = [output / "generated.mp4", output / "manifest.json", output / "complete.json"]
            if target is not None:
                required.extend(
                    output / "attention_audit" / f"step_{step:02d}" / "complete.json"
                    for step in args.attention_capture_steps
                )
            if all(path.is_file() for path in required) and not args.overwrite:
                print(f"[generate] skip {spec.name}/{tube.case}/{variant}", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            (output / "complete.json").unlink(missing_ok=True)
            collector = None
            geometry: dict[str, Any] | None = None
            if target is not None and entries is not None:
                collector, geometry = build_collector(
                    pipe,
                    spec,
                    tube,
                    target,
                    entries,
                    args.gaussian_sigma_tokens,
                )
                collector.install()
            try:
                inputs = prepare_backend_inputs(
                    pipe, spec, payload, tube, args.seed, args.cfg_scale
                )
                frames, audit, captures, predicted_x0 = run_denoising(
                    pipe,
                    spec,
                    *inputs,
                    collector,
                    args.cfg_scale,
                    args.latent_update_rms,
                    args.guidance_start,
                    args.guidance_end,
                    not args.no_gradient_checkpointing,
                    attention_capture_steps_1based=(
                        () if target is None else args.attention_capture_steps
                    ),
                )
            finally:
                if collector is not None:
                    collector.remove()
            if frames is None:
                raise RuntimeError("full generation unexpectedly returned no frames")
            temporary = output / "generated.tmp.mp4"
            save_video_np(frames, temporary, fps=30)
            temporary.replace(output / "generated.mp4")
            attention_reports: dict[str, Any] = {}
            if target is not None:
                tracks, visibility, _ = target_point_arrays(tube, target, spec)
                if source_anchors is None:
                    source_anchors = source_anchor_frames(tube, spec)
                missing_capture = sorted(
                    set(args.attention_capture_steps) - set(captures)
                )
                missing_prediction = sorted(
                    set(args.attention_capture_steps) - set(predicted_x0)
                )
                if missing_capture or missing_prediction:
                    raise RuntimeError(
                        "incomplete attention audit: "
                        f"capture={missing_capture}, predicted_x0={missing_prediction}"
                    )
                for capture_step in args.attention_capture_steps:
                    step_audit = audit[capture_step - 1]
                    report = write_step_attention_audit(
                        output / "attention_audit",
                        capture_step,
                        source_anchors,
                        np.asarray(tube.anchor_source_frames, dtype=np.int64),
                        tracks,
                        visibility,
                        captures[capture_step],
                        predicted_x0[capture_step],
                        spec.context_latent_frames,
                        float(step_audit["pre_update_loss"]),
                        float(step_audit["post_update_loss"]),
                    )
                    attention_reports[str(capture_step)] = report["summary"]
            atomic_json(
                output / "manifest.json",
                {
                    "protocol": PROTOCOL,
                    "backend": serializable(spec.__dict__),
                    "case": tube.case,
                    "seed": int(args.seed),
                    "variant": variant,
                    "target": None if target is None else target.name,
                    "target_object_indices": [] if target is None else list(target.object_indices),
                    "head_group": None if target is None else group_name,
                    "selected_heads": [] if entries is None else entries,
                    "source_json": str(case_path),
                    "source_video": str(tube.source_video),
                    "checkpoint": (
                        None if spec.name == "firstframe_ti2v" else str(args.checkpoint)
                    ),
                    "ranking": str(args.head_ranking),
                    "ranking_semantics": spec.ranking_semantics,
                    "geometry": geometry,
                    "runtime_info": runtime_info,
                    "model_parameters_updated": False,
                    "latent_update": (
                        "x_s' = x_s - eta * grad(L)/RMS_mutable(grad(L)); "
                        "re-forward positive and negative CFG at x_s'; ordinary FlowMatch step"
                    ),
                    "mutable_latent_update_rms": (
                        0.0 if target is None else float(args.latent_update_rms)
                    ),
                    "audit": audit,
                    "attention_audit": {
                        "capture_steps": (
                            [] if target is None else list(args.attention_capture_steps)
                        ),
                        "step_numbering": "1-based",
                        "normalization": "global softmax over all 13*H*W Keys",
                        "summaries": attention_reports,
                    },
                },
            )
            atomic_json(output / "complete.json", {"variant": variant})
            print(f"[generate] complete {spec.name}/{tube.case}/{variant}", flush=True)
            del frames, captures, predicted_x0
            gc.collect()
            torch.cuda.empty_cache()


def evaluation_d0(tube: legacy.FrozenTube, target: legacy.GuidanceTarget, spec: BackendSpec) -> float:
    mask = np.logical_or.reduce(tube.masks_othw[list(target.object_indices), 0], axis=0)
    yx = np.argwhere(mask)
    if not len(yx):
        return 1.0
    corners = np.asarray(
        [[[yx[:, 1].min(), yx[:, 0].min()], [yx[:, 1].max(), yx[:, 0].max()]]],
        dtype=np.float32,
    )
    if spec.name == "context8_v2v":
        corners, _, _ = transform_points_stretch_to_cover_crop(
            corners,
            source_hw=source_geometry(tube.source_video),
            stretched_hw=(tube.pixel_height, tube.pixel_width),
            crop_hw=(spec.height, spec.width),
            output_hw=(spec.height, spec.width),
        )
    return max(float(np.linalg.norm(corners[0, 1] - corners[0, 0])), 1.0)


def trajectory_row(
    candidate_tracks: np.ndarray,
    candidate_visibility: np.ndarray,
    reference_tracks: np.ndarray,
    reference_visibility: np.ndarray,
    tube: legacy.FrozenTube,
    target: legacy.GuidanceTarget,
    spec: BackendSpec,
) -> dict[str, Any]:
    point_indices = np.concatenate(
        [
            np.arange(tube.point_starts[index], tube.point_ends[index], dtype=np.int64)
            for index in target.object_indices
        ]
    )
    candidate = candidate_tracks[LATENT_ANCHORS][:, point_indices]
    candidate_visible = candidate_visibility[LATENT_ANCHORS][:, point_indices]
    reference = reference_tracks[:, point_indices]
    reference_visible = reference_visibility[:, point_indices]
    start = spec.context_latent_frames
    error = np.linalg.norm(candidate - reference, axis=-1)[start:]
    finite = (
        candidate_visible[start:]
        & reference_visible[start:]
        & np.isfinite(error)
    )
    min_points = min(4, len(point_indices))
    reference_anchor_valid = reference_visible[start:].sum(axis=1) >= min_points
    common_anchor_valid = finite.sum(axis=1) >= min_points
    reference_anchor_count = int(reference_anchor_valid.sum())
    common_anchor_count = int((reference_anchor_valid & common_anchor_valid).sum())
    coverage = common_anchor_count / max(reference_anchor_count, 1)
    quality_pass = common_anchor_count >= 4 and coverage >= 0.8
    visible_error = error[finite]
    d0 = evaluation_d0(tube, target, spec)
    final_valid = finite[-1]
    fde = float(error[-1, final_valid].mean()) if int(final_valid.sum()) >= min_points else None
    ade = float(visible_error.mean()) if len(visible_error) else None
    return {
        "target": target.name,
        "point_count": int(len(point_indices)),
        "future_start_latent": int(start),
        "valid_comparisons": int(finite.sum()),
        "future_reference_anchor_count": reference_anchor_count,
        "future_common_anchor_count": common_anchor_count,
        "future_common_anchor_coverage": float(coverage),
        "future_track_loss_score_0_100": float(100.0 * (1.0 - coverage)),
        "quality_pass": bool(quality_pass),
        "ade_px": ade if quality_pass else None,
        "ade_d0": ade / d0 if quality_pass and ade is not None else None,
        "fde_px": fde if quality_pass else None,
        "fde_d0": fde / d0 if quality_pass and fde is not None else None,
        "pck_10pct_d0": (
            float((visible_error <= 0.10 * d0).mean()) if quality_pass and len(visible_error) else None
        ),
        "pck_20pct_d0": (
            float((visible_error <= 0.20 * d0).mean()) if quality_pass and len(visible_error) else None
        ),
        "d0_px": float(d0),
    }


def run_evaluate(
    args: argparse.Namespace,
    spec: BackendSpec,
    case_paths: list[Path],
    target_map: dict[str, tuple[str, ...]],
) -> None:
    model = legacy.load_cotracker(args.device)
    try:
        for case_path in case_paths:
            tube = legacy.load_frozen_tube(args.tube_root, case_path.stem)
            reference_tracks, reference_visibility, geometry = backend_tracks(tube, spec)
            query_points = reference_tracks[0]
            root = args.output_root / spec.name / "generations" / tube.case / f"seed_{args.seed:05d}"
            reports: dict[str, dict[str, Any]] = {}
            for video_path in sorted(root.glob("*/generated.mp4")):
                metrics_path = video_path.parent / "trajectory_metrics.json"
                if metrics_path.is_file() and not args.overwrite:
                    reports[video_path.parent.name] = json.loads(metrics_path.read_text(encoding="utf-8"))
                    continue
                frames = np.asarray(iio.imread(video_path))[:PIXEL_FRAMES, ..., :3]
                if len(frames) != PIXEL_FRAMES:
                    raise RuntimeError(f"expected 49 frames: {video_path}")
                tracks, visibility = legacy.run_cotracker(
                    model, frames, query_points, args.device
                )
                rows = [
                    trajectory_row(
                        tracks,
                        visibility,
                        reference_tracks,
                        reference_visibility,
                        tube,
                        target,
                        spec,
                    )
                    for target in legacy.selected_target_specs(tube, target_map[tube.case])
                ]
                report = {
                    "protocol": PROTOCOL,
                    "backend": spec.name,
                    "case": tube.case,
                    "variant": video_path.parent.name,
                    "reference": "source-video CoTracker point trajectories in backend coordinates",
                    "geometry": geometry,
                    "metrics": rows,
                }
                atomic_json(metrics_path, report)
                reports[video_path.parent.name] = report
            baseline = reports.get("baseline")
            if baseline is None:
                continue
            baseline_rows = {row["target"]: row for row in baseline["metrics"]}
            comparisons = []
            for variant, report in reports.items():
                if variant == "baseline":
                    continue
                for row in report["metrics"]:
                    base = baseline_rows[row["target"]]
                    deltas = {}
                    for metric in (
                        "ade_d0", "fde_d0", "pck_10pct_d0", "pck_20pct_d0",
                        "future_track_loss_score_0_100",
                    ):
                        left, right = row.get(metric), base.get(metric)
                        deltas[f"delta_{metric}"] = (
                            float(left) - float(right)
                            if left is not None and right is not None
                            else None
                        )
                    comparisons.append(
                        {
                            "variant": variant,
                            "target": row["target"],
                            "baseline_quality_pass": base["quality_pass"],
                            "guided_quality_pass": row["quality_pass"],
                            **deltas,
                        }
                    )
            atomic_json(root / "comparison_to_baseline.json", {
                "protocol": PROTOCOL,
                "backend": spec.name,
                "case": tube.case,
                "comparisons": comparisons,
            })
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--stage", choices=("dry-run", "sanity", "generate", "evaluate", "all"), default="all")
    parser.add_argument("--input-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--head-ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--head-scopes", type=Path, default=DEFAULT_SCOPES)
    parser.add_argument("--tube-root", type=Path, default=DEFAULT_TUBE_ROOT)
    parser.add_argument("--target-map", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--head-groups", nargs="+", choices=("top100", "bottom100", "random100"), default=("top100", "bottom100", "random100"))
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--latent-update-rms", type=float, default=0.01)
    parser.add_argument("--guidance-start", type=int, default=0)
    parser.add_argument("--guidance-end", type=int, default=39)
    parser.add_argument("--gaussian-sigma-tokens", type=float, default=1.5)
    parser.add_argument(
        "--attention-capture-steps",
        nargs="*",
        type=int,
        default=DEFAULT_ATTENTION_CAPTURE_STEPS,
        help="1-based denoising steps for Pre/Post attention and predicted-x0 audit",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.device in {"cuda:4", "4"}:
        raise ValueError("GPU 4 is prohibited")
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0,num-workers)")
    if not 0 <= args.guidance_start <= args.guidance_end < 40:
        raise ValueError("guidance range must lie in [0,39]")
    if args.latent_update_rms <= 0 or args.gaussian_sigma_tokens <= 0:
        raise ValueError("update RMS and Gaussian sigma must be positive")
    capture_steps = tuple(sorted(set(args.attention_capture_steps)))
    if not capture_steps:
        raise ValueError("at least one attention capture step is required")
    if capture_steps[0] < 1 or capture_steps[-1] > 40:
        raise ValueError("attention capture steps must lie in 1..40")
    outside_guidance = [
        step
        for step in capture_steps
        if not args.guidance_start <= step - 1 <= args.guidance_end
    ]
    if outside_guidance:
        raise ValueError(
            f"attention capture steps outside the guidance range: {outside_guidance}"
        )
    for path in (args.input_list, args.head_ranking, args.head_scopes, args.target_map):
        if not path.expanduser().is_file():
            raise FileNotFoundError(path)
    if args.backend == "context8_v2v" and not (args.checkpoint / "checkpoint.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint / "checkpoint.safetensors")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args = argparse.Namespace(
        **{
            **vars(args),
            "input_list": args.input_list.expanduser().resolve(),
            "head_ranking": args.head_ranking.expanduser().resolve(),
            "head_scopes": args.head_scopes.expanduser().resolve(),
            "tube_root": args.tube_root.expanduser().resolve(),
            "target_map": args.target_map.expanduser().resolve(),
            "checkpoint": args.checkpoint.expanduser().resolve(),
            "output_root": args.output_root.expanduser().resolve(),
            "attention_capture_steps": tuple(
                sorted(set(args.attention_capture_steps))
            ),
        }
    )
    spec = BACKENDS[args.backend]
    case_paths, target_map = load_cases(args)
    groups = load_head_groups(args.head_ranking, args.head_scopes, args.head_groups)
    manifest = task_manifest(args, spec, case_paths, target_map, groups)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / spec.name / "task_manifest.json", manifest)
    atomic_json(args.output_root / spec.name / "run_config.json", {
        "protocol": PROTOCOL,
        "arguments": serializable(vars(args)),
        **manifest,
    })
    if args.stage == "dry-run":
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.stage == "evaluate":
        run_evaluate(args, spec, case_paths, target_map)
        return
    error_path = args.output_root / spec.name / "run_error.txt"
    error_path.unlink(missing_ok=True)
    owner = None
    try:
        owner, pipe, runtime_info = build_backend(args, spec)
        if args.stage in {"sanity", "all"}:
            run_sanity(args, spec, pipe, case_paths[0], target_map, groups)
        if args.stage in {"generate", "all"}:
            run_generate(
                args, spec, pipe, case_paths, target_map, groups, runtime_info
            )
    except Exception:
        error = traceback.format_exc()
        error_path.write_text(error, encoding="utf-8")
        print(error, flush=True)
        raise
    finally:
        release_backend(owner)
    if args.stage == "all":
        run_evaluate(args, spec, case_paths, target_map)


if __name__ == "__main__":
    main()
