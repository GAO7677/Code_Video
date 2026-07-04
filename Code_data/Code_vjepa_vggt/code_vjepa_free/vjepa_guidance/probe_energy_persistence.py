#!/usr/bin/env python3
"""
probe_energy_persistence.py

Multi-phase sweep to find which V-JEPA guidance configuration produces a
durable, measurable energy signal.

Phase 1: timing sweep  -- single step at 5 positions (p20/35/50/65/80)
Phase 2: step-size sweep -- at best timing from Phase 1
Phase 3: step-count + inner-K sweep -- at best timing+size from Phase 2
Phase 4: context-anchored persistence sweep
Phase 5: context-anchored strength ladder
Phase 6: context-anchored knee refinement around the first wmreward-positive regime
Phase 7: context-anchored target-shape sweep (wider future windows)
Phase 8: context-anchored anti-artifact guard sweep

Each phase runs baseline + N guided conditions, computes persistence_score,
saves per-condition JSON and delta-curve plots.

Example (Phase 1):
  PYTHONPATH=.../Code_vjepa_vggt:.../DiffSynth-Studio-main:.../train_0419 \
  CUDA_VISIBLE_DEVICES=2,3 \
  python probe_energy_persistence.py \
    --weights-root .../step-000500 \
    --input-json .../physicIQ_025_...trimmed.json \
    --context-path .../context_video_8f.mp4 \
    --output-dir /data/gaoya/agent-data/outputs/probe_sweep \
    --phase 1 \
    --device cuda:0 --vjepa-device cuda:1 \
    --probe-every-n 2 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

_VJEPA_GUIDANCE_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _VJEPA_GUIDANCE_DIR.parents[1]
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "DiffSynth-Studio-main")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .vjepa_surprise import VJEPASurpriseEnergy, build_context_future_clip
    from .wan_latent_guidance import WanVJEPAConfig, pick_guidance_step_indices
    from .wan_openvid_0613pybullet_lorav2v_vjepa import (
        ContextAwareWanVideoPipelineVJEPA,
        _diffsynth_sigma_for_timestep,
        _predict_x0_from_diffsynth_flow,
        _apply_diffsynth_vjepa_guidance,
    )
except ImportError:
    from vjepa_surprise import VJEPASurpriseEnergy, build_context_future_clip
    from wan_latent_guidance import WanVJEPAConfig, pick_guidance_step_indices
    from wan_openvid_0613pybullet_lorav2v_vjepa import (
        ContextAwareWanVideoPipelineVJEPA,
        _diffsynth_sigma_for_timestep,
        _predict_x0_from_diffsynth_flow,
        _apply_diffsynth_vjepa_guidance,
    )

from code_vjepa_vggt.train0419_reference import batch_eval_lora as core

log = logging.getLogger(__name__)


def _pil_frames_to_tensor(frames: list) -> torch.Tensor:
    """Convert a list of PIL RGB frames to a [1,3,T,H,W] tensor in [-1,1].

    Matches DiffSynth's preprocess_video normalization so the anchor context
    lives in the same value range as the VAE-decoded generated frames.
    """
    import numpy as np

    arrs = []
    for frame in frames:
        arr = torch.from_numpy(np.asarray(frame.convert("RGB"))).float()  # [H,W,3] in 0..255
        arr = arr.permute(2, 0, 1) / 127.5 - 1.0  # [3,H,W] in [-1,1]
        arrs.append(arr)
    stacked = torch.stack(arrs, dim=1)  # [3,T,H,W]
    return stacked.unsqueeze(0)  # [1,3,T,H,W]


# ---------------------------------------------------------------------------
# Scheduler proxy
# ---------------------------------------------------------------------------

class _SchedulerEnergyProbe:
    """
    Transparent proxy around a DiffSynth scheduler.

    Intercepts every .step() call to passively measure V-JEPA energy on the
    current x0 prediction. Does NOT modify the latent in any way.

    Usage:
        probe = _SchedulerEnergyProbe(pipe.scheduler, energy_fn, pipe, config)
        pipe.scheduler = probe
        # run generation ...
        pipe.scheduler = probe.inner   # restore
        records = probe.records
    """

    def __init__(
        self,
        inner_scheduler,
        energy_fn: VJEPASurpriseEnergy,
        pipe: ContextAwareWanVideoPipelineVJEPA,
        config: WanVJEPAConfig,
        *,
        probe_every_n: int = 1,
        restore_model_names: tuple[str, ...] = ("dit",),
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        framewise_decoding: bool = False,
    ) -> None:
        self.inner = inner_scheduler
        self._energy_fn = energy_fn
        self._pipe = pipe
        self._config = config
        self._probe_every_n = max(1, probe_every_n)
        self._restore_model_names = restore_model_names
        self._tiled = tiled
        self._tile_size = tile_size
        self._tile_stride = tile_stride
        self._framewise_decoding = framewise_decoding
        self._step_counter = 0
        self.records: list[dict] = []
        self.guidance_step_indices: set[int] = set()
        # Anchored-probe state: when enabled, the probe measures the same
        # context-anchored energy that context_anchored guidance optimizes,
        # against a fixed precomputed reference (so baseline and guided runs
        # are directly comparable).
        self.anchored_probe: bool = False
        self._anchor_context_pixel: Optional[torch.Tensor] = None
        self._anchor_future_ref: Optional[torch.Tensor] = None

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor, **kwargs):
        idx = self._step_counter
        self._step_counter += 1

        if idx % self._probe_every_n == 0:
            record = self._probe(idx, timestep, model_output, sample)
            self.records.append(record)

        return self.inner.step(model_output, timestep, sample, **kwargs)

    def _probe(self, step_idx: int, timestep, model_output: torch.Tensor, latent_xt: torch.Tensor) -> dict:
        t0 = time.time()
        try:
            with torch.no_grad():
                # x0_pred via flow-matching formula
                sigma_t = _diffsynth_sigma_for_timestep(self.inner, timestep)
                sigma_t = sigma_t.to(device=latent_xt.device, dtype=torch.float32)
                while sigma_t.ndim < latent_xt.ndim:
                    sigma_t = sigma_t.unsqueeze(-1)
                x0_pred = latent_xt.detach().float() - sigma_t * model_output.detach().float()

                if self.anchored_probe and self._anchor_context_pixel is not None:
                    # Anchored energy: decode full video, take future frames, prepend
                    # the fixed real context, measure mismatch vs the fixed prediction.
                    full_video = self._pipe._decode_preview_video(
                        x0_pred,
                        preview_downsample_factor=self._config.preview_downsample_factor,
                        preview_frame_stride=1,
                        tiled=self._tiled,
                        tile_size=self._tile_size,
                        tile_stride=self._tile_stride,
                        framewise_decoding=self._framewise_decoding,
                        restore_model_names=self._restore_model_names,
                    )
                    n_ctx = int(self._config.context_frames)
                    generated_future = full_video[:, :, n_ctx:]
                    clip = build_context_future_clip(
                        context_btchw=self._anchor_context_pixel.to(
                            device=generated_future.device, dtype=generated_future.dtype
                        ),
                        future_btchw=generated_future,
                        window_size=self._config.window_size,
                        context_frames=n_ctx,
                    )
                    energy = float(
                        self._energy_fn.context_anchored(
                            clip,
                            window_size=self._config.window_size,
                            context_frames=n_ctx,
                            predicted_future_ref=self._anchor_future_ref,
                        ).item()
                    )
                else:
                    # decode low-res preview via VAE
                    preview = self._pipe._decode_preview_video(
                        x0_pred,
                        preview_downsample_factor=self._config.preview_downsample_factor,
                        preview_frame_stride=self._config.preview_frame_stride,
                        tiled=self._tiled,
                        tile_size=self._tile_size,
                        tile_stride=self._tile_stride,
                        framewise_decoding=self._framewise_decoding,
                        restore_model_names=self._restore_model_names,
                    )

                    energy = float(
                        self._energy_fn(
                            preview,
                            window_size=self._config.window_size,
                            context_frames=self._config.context_frames,
                            stride=self._config.stride,
                            reduction=self._config.reduction,
                        ).item()
                    )

            timestep_val = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
            return {
                "step": step_idx,
                "timestep": timestep_val,
                "energy": energy,
                "was_guidance_step": step_idx in self.guidance_step_indices,
                "probe_elapsed_s": round(time.time() - t0, 3),
            }
        except Exception as exc:
            log.warning("Probe failed at step %d: %s", step_idx, exc)
            return {
                "step": step_idx,
                "timestep": int(timestep.item()) if hasattr(timestep, "item") else -1,
                "energy": None,
                "was_guidance_step": step_idx in self.guidance_step_indices,
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------

def _resolve_lora_path(weights_root: Path) -> Path:
    p = weights_root.expanduser().resolve() / "checkpoint.safetensors"
    if not p.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {p}")
    return p


def _build_pipeline(
    wan_root: Path,
    device: str,
    lora_path: Path,
    vjepa_model: str,
    vjepa_ckpt: Path,
    vjepa_device: str,
    vjepa_config: WanVJEPAConfig,
) -> ContextAwareWanVideoPipelineVJEPA:
    pipe = ContextAwareWanVideoPipelineVJEPA.from_pretrained_vjepa(
        wan_root=wan_root,
        device=device,
        lora_path=lora_path,
        vjepa_model_name=vjepa_model,
        vjepa_checkpoint_path=vjepa_ckpt,
        vjepa_device=vjepa_device,
        vjepa_config=vjepa_config,
        enable_vjepa_guidance=True,
    )
    return pipe


def _load_case(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def _persistence_score(baseline_records: list[dict], guided_records: list[dict]) -> tuple[float, float]:
    """Fraction of post-guidance probes where delta < 0, and mean delta."""
    baseline_map = {r["step"]: r["energy"] for r in baseline_records if r.get("energy") is not None}
    guided_steps = [r["step"] for r in guided_records if r.get("was_guidance_step")]
    first_g = min(guided_steps) if guided_steps else 0
    post = [(r["step"], r["energy"]) for r in guided_records
            if r.get("energy") is not None and r["step"] >= first_g]
    deltas = [e - baseline_map[s] for s, e in post if s in baseline_map]
    if not deltas:
        return 0.0, 0.0
    score = sum(1 for d in deltas if d < 0) / len(deltas)
    mean_delta = sum(deltas) / len(deltas)
    return score, mean_delta


def _run_condition(
    pipe: ContextAwareWanVideoPipelineVJEPA,
    case: dict,
    *,
    seed: int,
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int,
    cfg_scale: float,
    negative_prompt: str,
    context_path: Path,
    guidance_step_percents: list[float],  # empty = baseline
    vjepa_config: WanVJEPAConfig,
    probe_every_n: int,
    condition_label: str,
    latent_step_size: Optional[float] = None,   # overrides vjepa_config value
    inner_k: int = 1,                            # guidance repetitions per step
    save_video_path: Optional[Path] = None,     # if provided, save generated video
    guidance_mode: str = "surprise",            # "surprise" | "context_anchored"
    backtracking: bool = False,                 # auto-pick step size via line search
    backtracking_taps: Optional[list[float]] = None,
    grad_norm: Optional[str] = None,            # overrides vjepa_config.gradient_normalization
    window_size: Optional[int] = None,          # overrides vjepa_config.window_size
    context_frames_vjepa: Optional[int] = None, # overrides vjepa_config.context_frames
    stride: Optional[int] = None,               # overrides vjepa_config.stride
    reduction: Optional[str] = None,            # overrides vjepa_config.reduction
    max_correction_ratio: Optional[float] = None,
    stay_close_max_video_l1: Optional[float] = None,
    artifact_guard_mode: Optional[str] = None,
) -> tuple[list, list[dict]]:
    """Run one generation condition with per-step energy probing. Returns (video, records).

    When ``guidance_mode == "context_anchored"``, both the guidance and the probe
    measure the context-anchored energy (generated future vs V-JEPA's prediction
    from the real conditioning frames), so baseline and guided runs stay comparable.
    """

    effective_step_size = latent_step_size if latent_step_size is not None else vjepa_config.latent_step_size
    effective_grad_norm = grad_norm if grad_norm is not None else vjepa_config.gradient_normalization
    effective_window_size = int(window_size) if window_size is not None else int(vjepa_config.window_size)
    effective_context_frames = (
        int(context_frames_vjepa) if context_frames_vjepa is not None else int(vjepa_config.context_frames)
    )
    effective_stride = int(stride) if stride is not None else int(vjepa_config.stride)
    effective_reduction = str(reduction) if reduction is not None else str(vjepa_config.reduction)
    effective_max_correction_ratio = (
        float(max_correction_ratio)
        if max_correction_ratio is not None
        else vjepa_config.max_correction_ratio
    )
    effective_stay_close_max_video_l1 = (
        float(stay_close_max_video_l1)
        if stay_close_max_video_l1 is not None
        else vjepa_config.stay_close_max_video_l1
    )
    effective_artifact_guard_mode = (
        str(artifact_guard_mode)
        if artifact_guard_mode is not None
        else str(vjepa_config.artifact_guard_mode)
    )

    def _make_config(*, steps: int, min_p: float, max_p: float, step_size: float) -> WanVJEPAConfig:
        return WanVJEPAConfig(
            guidance_steps=steps,
            min_step_percent=min_p,
            max_step_percent=max_p,
            latent_step_size=step_size,
            preview_downsample_factor=vjepa_config.preview_downsample_factor,
            preview_frame_stride=vjepa_config.preview_frame_stride,
            window_size=effective_window_size,
            context_frames=effective_context_frames,
            stride=effective_stride,
            reduction=effective_reduction,
            gradient_normalization=effective_grad_norm,
            max_grad_norm=vjepa_config.max_grad_norm,
            max_correction_ratio=effective_max_correction_ratio,
            stay_close_max_video_l1=effective_stay_close_max_video_l1,
            artifact_guard_mode=effective_artifact_guard_mode,
            guidance_mode=guidance_mode,
        )

    # Determine guidance steps
    if guidance_step_percents:
        guidance_step_indices = set(
            pick_guidance_step_indices(
                total_steps=num_inference_steps,
                count=len(guidance_step_percents),
                min_step_percent=min(guidance_step_percents),
                max_step_percent=max(guidance_step_percents),
            )
        )
        pipe.vjepa_config = _make_config(
            steps=len(guidance_step_percents),
            min_p=min(guidance_step_percents),
            max_p=max(guidance_step_percents),
            step_size=effective_step_size,
        )
        pipe.enable_vjepa_guidance = True
        pipe.vjepa_inner_k = max(1, int(inner_k))
    else:
        guidance_step_indices = set()
        # Baseline still needs guidance_mode on the config so the probe knows which
        # energy to measure; enable_vjepa_guidance stays False so no correction runs.
        pipe.vjepa_config = _make_config(
            steps=1,
            min_p=0.5,
            max_p=0.5,
            step_size=effective_step_size,
        )
        pipe.enable_vjepa_guidance = False
        pipe.vjepa_inner_k = 1

    pipe.configure_target_step_indices([])
    pipe.configure_target_timesteps([])

    # Ensure V-JEPA energy model is loaded (needed for probe in both conditions)
    if pipe._vjepa_energy is None:
        pipe._vjepa_energy = VJEPASurpriseEnergy(
            model_name=pipe.vjepa_model_name,
            device=pipe.vjepa_device,
            local_torchhub=True,
            checkpoint_path=pipe.vjepa_checkpoint_path,
        )
    energy_fn = pipe._vjepa_energy

    # For context-anchored mode, precompute the fixed anchor (real context frames
    # in pixel space + V-JEPA's future prediction from them). The SAME anchor is
    # used to probe baseline and every guided condition, so the delta curve is
    # measured against one consistent reference.
    anchor_context_pixel = None
    anchor_future_ref = None
    if guidance_mode == "context_anchored":
        ctx_frames = core.load_context_frames(
            context_path,
            context_frames=effective_context_frames,
            height=height,
            width=width,
        )
        anchor_context_pixel = _pil_frames_to_tensor(ctx_frames).to(pipe.vjepa_device)
        n_ctx = effective_context_frames
        future_frames = effective_window_size - n_ctx
        placeholder = torch.zeros(
            1, 3, future_frames,
            anchor_context_pixel.shape[3],
            anchor_context_pixel.shape[4],
            device=anchor_context_pixel.device,
            dtype=anchor_context_pixel.dtype,
        )
        precompute_clip = build_context_future_clip(
            context_btchw=anchor_context_pixel,
            future_btchw=placeholder,
            window_size=effective_window_size,
            context_frames=n_ctx,
        )
        anchor_future_ref = energy_fn.precompute_future_prediction(
            precompute_clip,
            window_size=effective_window_size,
            context_frames=n_ctx,
        )
        log.info("[%s] anchored probe ready: ctx=%d future=%d ref=%s",
                 condition_label, n_ctx, future_frames, tuple(anchor_future_ref.shape))
        # Share the SAME anchor with the pipeline guidance so guidance descends
        # exactly the energy the probe measures (Bug 1 fix: previously guidance
        # re-derived a shorter context by decoding the compressed latent prefix).
        pipe.set_external_anchor(
            context_frames_pixel=anchor_context_pixel,
            predicted_future_ref=anchor_future_ref,
        )
    else:
        pipe.set_external_anchor(context_frames_pixel=None, predicted_future_ref=None)

    # Per-step backtracking: pick the tap that most lowers anchored energy instead
    # of a fixed step (the fixed 0.02 overshot the shallow descent basin). Only
    # meaningful in context_anchored mode.
    pipe.set_backtracking(backtracking and guidance_mode == "context_anchored")

    probe = _SchedulerEnergyProbe(
        inner_scheduler=pipe.scheduler,
        energy_fn=energy_fn,
        pipe=pipe,
        config=pipe.vjepa_config,
        probe_every_n=probe_every_n,
        restore_model_names=tuple(pipe.in_iteration_models),
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        framewise_decoding=False,
    )
    probe.guidance_step_indices = guidance_step_indices
    probe.anchored_probe = guidance_mode == "context_anchored"
    probe._anchor_context_pixel = anchor_context_pixel
    probe._anchor_future_ref = anchor_future_ref
    pipe.scheduler = probe

    log.info("[%s] guidance_steps=%s  probe_every_n=%d", condition_label, sorted(guidance_step_indices), probe_every_n)

    try:
        prompt = case.get("input_caption", case.get("caption", case.get("prompt", "")))
        context_frames_list = core.load_context_frames(
            context_path,
            context_frames=8,
            height=height,
            width=width,
        )

        video, _ = core.generate_one_video(
            pipe=pipe,
            context_path=context_path,
            first_frame_path=None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=16,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            context_frames=8,
            output_num_frames=num_frames,
        )
    finally:
        pipe.scheduler = probe.inner
        pipe.enable_vjepa_guidance = bool(guidance_step_percents)

    records = probe.records
    log.info("[%s] collected %d probe records", condition_label, len(records))

    # Save video if requested
    if save_video_path is not None:
        save_video_path.parent.mkdir(parents=True, exist_ok=True)
        core.save_video(video, save_video_path, fps=16)
        log.info("[%s] saved video to %s", condition_label, save_video_path)

    return video, records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_phase(
    records_per_condition: dict[str, list[dict]],
    output_dir: Path,
    phase_name: str,
    prompt_short: str = "",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping plot")
        return

    baseline_records = records_per_condition.get("baseline", [])
    baseline_map = {r["step"]: r["energy"] for r in baseline_records if r.get("energy") is not None}
    colors = plt.cm.tab10.colors
    guided_labels = [lb for lb in records_per_condition if lb != "baseline"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{phase_name} -- V-JEPA Energy Persistence\n{prompt_short}", fontsize=10)

    # Top: raw energy curves
    if baseline_records:
        steps_b = [r["step"] for r in baseline_records if r.get("energy") is not None]
        energies_b = [r["energy"] for r in baseline_records if r.get("energy") is not None]
        ax1.plot(steps_b, energies_b, color="black", linewidth=2.0, label="baseline", zorder=10)

    for i, label in enumerate(guided_labels):
        records = records_per_condition[label]
        steps = [r["step"] for r in records if r.get("energy") is not None]
        energies = [r["energy"] for r in records if r.get("energy") is not None]
        color = colors[i % len(colors)]
        ax1.plot(steps, energies, color=color, linewidth=1.2, label=label, alpha=0.8)
        gs = [r["step"] for r in records if r.get("was_guidance_step") and r.get("energy") is not None]
        ge = [r["energy"] for r in records if r.get("was_guidance_step") and r.get("energy") is not None]
        if gs:
            ax1.scatter(gs, ge, marker="*", s=120, color=color, zorder=5)

    ax1.set_ylabel("V-JEPA surprise energy")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Bottom: delta curves (guided - baseline)
    ax2.axhline(0, color="black", linewidth=1.0, linestyle="--")
    for i, label in enumerate(guided_labels):
        records = records_per_condition[label]
        guided_steps_set = {r["step"] for r in records if r.get("was_guidance_step")}
        first_g = min(guided_steps_set) if guided_steps_set else 0
        steps_d, deltas = [], []
        for r in records:
            if r.get("energy") is not None and r["step"] in baseline_map:
                steps_d.append(r["step"])
                deltas.append(r["energy"] - baseline_map[r["step"]])
        if not steps_d:
            continue
        color = colors[i % len(colors)]
        ax2.plot(steps_d, deltas, color=color, linewidth=1.2, label=label, alpha=0.8)
        # shade post-guidance region
        post_x = [s for s in steps_d if s >= first_g]
        if post_x:
            ax2.axvspan(first_g, max(post_x), alpha=0.05, color=color)

    ax2.set_xlabel("Denoising step index")
    ax2.set_ylabel("delta = guided - baseline")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / f"{phase_name}_delta_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Plot saved: %s", out)


# ---------------------------------------------------------------------------
# Phase condition definitions
# ---------------------------------------------------------------------------

def _phase1_conditions() -> list[dict]:
    return [
        {"label": "timing_p20", "guidance_step_percents": [0.20], "latent_step_size": 0.02, "inner_k": 1},
        {"label": "timing_p35", "guidance_step_percents": [0.35], "latent_step_size": 0.02, "inner_k": 1},
        {"label": "timing_p50", "guidance_step_percents": [0.50], "latent_step_size": 0.02, "inner_k": 1},
        {"label": "timing_p65", "guidance_step_percents": [0.65], "latent_step_size": 0.02, "inner_k": 1},
        {"label": "timing_p80", "guidance_step_percents": [0.80], "latent_step_size": 0.02, "inner_k": 1},
    ]


def _phase2_conditions(p_best: float) -> list[dict]:
    return [
        {"label": "stepsize_001", "guidance_step_percents": [p_best], "latent_step_size": 0.01, "inner_k": 1},
        {"label": "stepsize_005", "guidance_step_percents": [p_best], "latent_step_size": 0.05, "inner_k": 1},
        {"label": "stepsize_010", "guidance_step_percents": [p_best], "latent_step_size": 0.10, "inner_k": 1},
        {"label": "stepsize_020", "guidance_step_percents": [p_best], "latent_step_size": 0.20, "inner_k": 1},
    ]


def _phase3_conditions(p_best: float, ss_best: float) -> list[dict]:
    lo = max(0.05, p_best - 0.15)
    hi = min(0.95, p_best + 0.15)
    lo2 = max(0.05, p_best - 0.10)
    hi2 = min(0.95, p_best + 0.10)
    return [
        # step count
        {"label": "count2", "guidance_step_percents": [lo2, hi2], "latent_step_size": ss_best, "inner_k": 1},
        {"label": "count4", "guidance_step_percents": [lo, lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3, hi],
         "latent_step_size": ss_best, "inner_k": 1},
        {"label": "count6", "guidance_step_percents": [lo + i * (hi - lo) / 5 for i in range(6)],
         "latent_step_size": ss_best, "inner_k": 1},
        # inner k
        {"label": "inner_k2", "guidance_step_percents": [p_best], "latent_step_size": ss_best, "inner_k": 2},
        {"label": "inner_k4", "guidance_step_percents": [p_best], "latent_step_size": ss_best, "inner_k": 4},
        {"label": "inner_k2_half", "guidance_step_percents": [p_best], "latent_step_size": ss_best / 2, "inner_k": 2},
    ]


def _dense_percents(lo: float, hi: float, n: int) -> list[float]:
    """n evenly-spaced guidance-step percents in [lo, hi] (inclusive)."""
    if n <= 1:
        return [round((lo + hi) / 2, 3)]
    return [round(lo + i * (hi - lo) / (n - 1), 3) for i in range(n)]


def _phase4_conditions(p_center: float = 0.50, ss_base: float = 0.005) -> list[dict]:
    """Context-anchored sweep built on the line-search finding (2026-07-03):

    The gradient DIRECTION is correct, but a single fixed step of 0.02 overshoots the
    sharp/shallow energy basin (only tap<=0.005 lowers energy). A single small step
    also gets washed out by the next diffusion step. So this sweep tests the two ways
    to make the correction PERSIST:

      1. CONTINUOUS small-step guidance -- apply a small (~0.005) step at many
         consecutive denoising steps so the pull is re-applied faster than the DiT
         prior can erase it.
      2. BACKTRACKING -- at each guidance step, auto-pick the tap that most lowers the
         anchored energy (never climb). Robust to the basin being step-dependent.

    The whole phase runs in guidance_mode='context_anchored' so baseline + all
    conditions are probed with the same anchored energy.

    ``p_center`` centers the timing band; ``ss_base`` is the corrected small step.
    """
    lo = max(0.05, p_center - 0.15)   # ~p35
    hi = min(0.95, p_center + 0.30)   # ~p80 -- extend into low-noise region
    return [
        # A single small-step reference (the verified fix, one step).
        {"label": "anch_single_s005", "guidance_step_percents": [p_center],
         "latent_step_size": ss_base, "inner_k": 1},
        # Continuous small-step guidance at increasing density across p35..p80.
        {"label": "anch_dense6_s005", "guidance_step_percents": _dense_percents(lo, hi, 6),
         "latent_step_size": ss_base, "inner_k": 1},
        {"label": "anch_dense12_s005", "guidance_step_percents": _dense_percents(lo, hi, 12),
         "latent_step_size": ss_base, "inner_k": 1},
        {"label": "anch_dense20_s003", "guidance_step_percents": _dense_percents(lo, hi, 20),
         "latent_step_size": 0.003, "inner_k": 1},
        # Backtracking (auto step size) at medium and high density.
        {"label": "anch_dense12_bt", "guidance_step_percents": _dense_percents(lo, hi, 12),
         "latent_step_size": ss_base, "inner_k": 1, "backtracking": True},
        {"label": "anch_dense20_bt", "guidance_step_percents": _dense_percents(lo, hi, 20),
         "latent_step_size": ss_base, "inner_k": 1, "backtracking": True},
    ]


def _phase5_conditions(p_center: float = 0.50, ss_base: float = 0.005) -> list[dict]:
    """Strength ladder (Phase 0a finding, 2026-07-03): reducing the anchored energy
    by ~0.005 does NOT move wmreward, and every phase-4 video is visually identical to
    baseline. The guidance is simply too weak to change the output. Before revisiting
    the energy target, we must first find the step size at which the decoded video
    ACTUALLY diverges from baseline.

    Timing is held at the dense band (p35..p80, 12 steps) that gave the most
    persistent energy drop in phase 4. We sweep only the strength axis:

      latent_step_size in {0.01, 0.02, 0.05, 0.1, 0.2}  (rms normalization)

    plus two normalization variants at a promising step size to test whether rms
    rescaling is what caps the effect. Every condition records energy trajectory,
    applied-correction L2 (Phase 0b stats), and produces a video for wmreward +
    pixel-delta scoring downstream.

    Expected three regimes: too-weak (no change) -> useful (physics moves) ->
    too-strong (artifacts, wmreward worsens). We want the knee.
    """
    lo = max(0.05, p_center - 0.15)   # ~p35
    hi = min(0.95, p_center + 0.30)   # ~p80
    band12 = _dense_percents(lo, hi, 12)
    return [
        {"label": "ladder_s01", "guidance_step_percents": band12,
         "latent_step_size": 0.01, "inner_k": 1},
        {"label": "ladder_s02", "guidance_step_percents": band12,
         "latent_step_size": 0.02, "inner_k": 1},
        {"label": "ladder_s05", "guidance_step_percents": band12,
         "latent_step_size": 0.05, "inner_k": 1},
        {"label": "ladder_s10", "guidance_step_percents": band12,
         "latent_step_size": 0.10, "inner_k": 1},
        {"label": "ladder_s20", "guidance_step_percents": band12,
         "latent_step_size": 0.20, "inner_k": 1},
        # Normalization variants at 0.05 to test whether rms rescaling caps the effect.
        {"label": "ladder_s05_none", "guidance_step_percents": band12,
         "latent_step_size": 0.05, "inner_k": 1, "grad_norm": "none"},
        {"label": "ladder_s05_l2", "guidance_step_percents": band12,
         "latent_step_size": 0.05, "inner_k": 1, "grad_norm": "l2"},
    ]


def _phase6_conditions() -> list[dict]:
    """Refine the first wmreward-positive region found in phase 5.

    Phase 5 showed that weak steps barely write into the latent, while stronger
    rms-normalized steps (0.10 / 0.20) finally move wmreward. This phase
    narrows in on the knee by varying:
      - strength between 0.10 and 0.20,
      - timing band shifts (early / mid / late),
      - inner-K repeats vs one larger update,
      - backtracking at a knee-sized step.
    """
    early12 = _dense_percents(0.10, 0.55, 12)
    mid12 = _dense_percents(0.20, 0.65, 12)
    late12 = _dense_percents(0.35, 0.80, 12)
    return [
        {"label": "knee_mid_s12", "guidance_step_percents": mid12, "latent_step_size": 0.12, "inner_k": 1},
        {"label": "knee_mid_s15", "guidance_step_percents": mid12, "latent_step_size": 0.15, "inner_k": 1},
        {"label": "knee_mid_s18", "guidance_step_percents": mid12, "latent_step_size": 0.18, "inner_k": 1},
        {"label": "knee_mid_s15_bt", "guidance_step_percents": mid12, "latent_step_size": 0.15, "inner_k": 1, "backtracking": True},
        {"label": "knee_early_s15", "guidance_step_percents": early12, "latent_step_size": 0.15, "inner_k": 1},
        {"label": "knee_late_s15", "guidance_step_percents": late12, "latent_step_size": 0.15, "inner_k": 1},
        {"label": "knee_mid_s10_k2", "guidance_step_percents": mid12, "latent_step_size": 0.10, "inner_k": 2},
        {"label": "knee_mid_s075_k2", "guidance_step_percents": mid12, "latent_step_size": 0.075, "inner_k": 2},
    ]


def _phase7_conditions(p_center: float = 0.35) -> list[dict]:
    """Phase 3 entry point: keep the strongest fixed-step guidance schedule and
    vary the *anchored target shape* by widening the future horizon that V-JEPA
    must explain from the real context.

    Important: `reduction=max` is only meaningful for the legacy sliding-window
    surprise energy today. The current context-anchored loss is a single anchored
    clip, so the first low-cost Phase 3 probe is to widen `window_size`, not to
    toggle `reduction`.

    All conditions reuse the phase-5 winner schedule (`ladder_s20` / dense mid
    band, step size 0.20) so any wmreward change can be attributed to the target
    horizon, not to a weaker/stronger latent correction.
    """
    lo = max(0.05, p_center - 0.15)   # default p_center=0.35 -> lo=0.20
    hi = min(0.95, p_center + 0.30)   # default -> hi=0.65
    band12 = _dense_percents(lo, hi, 12)
    return [
        {"label": "target_w16", "guidance_step_percents": band12,
         "latent_step_size": 0.20, "inner_k": 1, "window_size": 16},
        {"label": "target_w24", "guidance_step_percents": band12,
         "latent_step_size": 0.20, "inner_k": 1, "window_size": 24},
        {"label": "target_w32", "guidance_step_percents": band12,
         "latent_step_size": 0.20, "inner_k": 1, "window_size": 32},
    ]


def _phase8_conditions(p_center: float = 0.35) -> list[dict]:
    """Stage 1 anti-artifact sweep.

    Hold the current single-case metric winner skeleton (target_w24) fixed and
    add trust-region style guards that try to reduce the visible color-block
    failure mode:
      - correction-ratio cap in latent space
      - decoded-video L1 backoff against the unguided x0 preview
      - both combined
    """
    lo = max(0.05, p_center - 0.15)
    hi = min(0.95, p_center + 0.30)
    band12 = _dense_percents(lo, hi, 12)
    return [
        {"label": "target_w24_old", "guidance_step_percents": band12,
         "latent_step_size": 0.20, "inner_k": 1, "window_size": 24},
        {"label": "target_w24_ratio_005", "guidance_step_percents": band12,
         "latent_step_size": 0.20, "inner_k": 1, "window_size": 24,
         "max_correction_ratio": 0.05},
        {"label": "target_w24_guard_l1_003", "guidance_step_percents": band12,
         "latent_step_size": 0.20, "inner_k": 1, "window_size": 24,
         "artifact_guard_mode": "video_l1_backoff", "stay_close_max_video_l1": 0.03},
        {"label": "target_w24_guard_combo", "guidance_step_percents": band12,
         "latent_step_size": 0.20, "inner_k": 1, "window_size": 24,
         "max_correction_ratio": 0.05,
         "artifact_guard_mode": "video_l1_backoff", "stay_close_max_video_l1": 0.03},
    ]


def _rank_phase_scores(scores: list[dict], ranking_mode: str) -> list[dict]:
    if ranking_mode == "abs_mean_delta_desc":
        return sorted(
            scores,
            key=lambda x: (
                abs(float(x.get("mean_delta_post", 0.0))),
                float(x.get("persistence_score", 0.0)),
            ),
            reverse=True,
        )
    return sorted(scores, key=lambda x: x["persistence_score"], reverse=True)


def _energy_signature(
    *,
    guidance_mode: str,
    window_size: int,
    context_frames: int,
    stride: int,
    reduction: str,
) -> dict[str, str | int]:
    return {
        "guidance_mode": str(guidance_mode),
        "window_size": int(window_size),
        "context_frames": int(context_frames),
        "stride": int(stride),
        "reduction": str(reduction),
    }


def _signature_key(signature: dict[str, str | int]) -> str:
    return (
        f"{signature['guidance_mode']}"
        f"_w{signature['window_size']}"
        f"_c{signature['context_frames']}"
        f"_s{signature['stride']}"
        f"_{signature['reduction']}"
    )


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def _run_phase(
    phase_num: int,
    conditions: list[dict],
    *,
    pipe,
    case: dict,
    base_run_kwargs: dict,
    vjepa_config: WanVJEPAConfig,
    output_dir: Path,
    probe_every_n: int,
    baseline_records_path: Optional[Path] = None,
    guidance_mode: str = "surprise",
    ranking_mode: str = "persistence_desc",
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Run baseline + conditions, return (scores_list, all_records_dict).

    ``guidance_mode`` selects the energy/guidance mechanism for the whole phase
    ("surprise" = legacy self-consistency, "context_anchored" = anchored to the
    real conditioning frames). Individual conditions may override via a
    "guidance_mode" key.
    """

    phase_dir = output_dir / f"phase{phase_num}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = phase_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    all_records: dict[str, list[dict]] = {}
    baseline_video_path = videos_dir / "baseline.mp4"
    default_signature = _energy_signature(
        guidance_mode=guidance_mode,
        window_size=int(vjepa_config.window_size),
        context_frames=int(vjepa_config.context_frames),
        stride=int(vjepa_config.stride),
        reduction=str(vjepa_config.reduction),
    )
    default_sig_key = _signature_key(default_signature)
    baseline_cache: dict[str, list[dict]] = {}
    baseline_sig_map: dict[str, dict[str, str | int]] = {}

    def _baseline_json_path(sig_key: str) -> Path:
        if sig_key == default_sig_key:
            return phase_dir / "baseline_records.json"
        return phase_dir / f"baseline_{sig_key}_records.json"

    def _load_or_run_baseline(sig: dict[str, str | int]) -> tuple[str, list[dict]]:
        sig_key = _signature_key(sig)
        if sig_key in baseline_cache:
            return sig_key, baseline_cache[sig_key]

        baseline_json = _baseline_json_path(sig_key)
        if sig_key == default_sig_key and baseline_records_path is not None and baseline_records_path.exists():
            log.info("Reusing baseline from: %s", baseline_records_path)
            baseline_records = json.loads(baseline_records_path.read_text())
        elif baseline_json.exists():
            log.info("Reusing existing baseline: %s", baseline_json)
            baseline_records = json.loads(baseline_json.read_text())
        else:
            log.info("=== Running BASELINE [%s] ===", sig_key)
            _, baseline_records = _run_condition(
                **base_run_kwargs,
                guidance_step_percents=[],
                vjepa_config=vjepa_config,
                probe_every_n=probe_every_n,
                condition_label=f"baseline__{sig_key}",
                save_video_path=baseline_video_path if not baseline_video_path.exists() else None,
                guidance_mode=str(sig["guidance_mode"]),
                window_size=int(sig["window_size"]),
                context_frames_vjepa=int(sig["context_frames"]),
                stride=int(sig["stride"]),
                reduction=str(sig["reduction"]),
            )
            baseline_json.write_text(json.dumps(baseline_records, indent=2), encoding="utf-8")

        baseline_cache[sig_key] = baseline_records
        baseline_sig_map[sig_key] = sig
        return sig_key, baseline_records

    default_baseline_key, default_baseline_records = _load_or_run_baseline(default_signature)
    all_records["baseline"] = default_baseline_records

    scores: list[dict] = []
    for cond in conditions:
        label = cond["label"]
        rec_path = phase_dir / f"{label}_records.json"
        video_path = videos_dir / f"{label}.mp4"
        sig = _energy_signature(
            guidance_mode=cond.get("guidance_mode", guidance_mode),
            window_size=int(cond.get("window_size", vjepa_config.window_size)),
            context_frames=int(cond.get("context_frames_vjepa", vjepa_config.context_frames)),
            stride=int(cond.get("stride", vjepa_config.stride)),
            reduction=str(cond.get("reduction", vjepa_config.reduction)),
        )
        sig_key, baseline_records = _load_or_run_baseline(sig)

        if rec_path.exists():
            log.info("Reusing existing condition: %s", label)
            records = json.loads(rec_path.read_text())
        else:
            log.info("=== Running condition: %s ===", label)
            _, records = _run_condition(
                **base_run_kwargs,
                guidance_step_percents=cond["guidance_step_percents"],
                vjepa_config=vjepa_config,
                probe_every_n=probe_every_n,
                condition_label=label,
                latent_step_size=cond.get("latent_step_size"),
                inner_k=cond.get("inner_k", 1),
                save_video_path=video_path,
                guidance_mode=cond.get("guidance_mode", guidance_mode),
                backtracking=cond.get("backtracking", False),
                grad_norm=cond.get("grad_norm"),
                window_size=cond.get("window_size"),
                context_frames_vjepa=cond.get("context_frames_vjepa"),
                stride=cond.get("stride"),
                reduction=cond.get("reduction"),
                max_correction_ratio=cond.get("max_correction_ratio"),
                stay_close_max_video_l1=cond.get("stay_close_max_video_l1"),
                artifact_guard_mode=cond.get("artifact_guard_mode"),
            )
            rec_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

        all_records[label] = records
        score, mean_delta = _persistence_score(baseline_records, records)
        scores.append({
            "label": label,
            "persistence_score": round(score, 4),
            "mean_delta_post": round(mean_delta, 6),
            "baseline_signature_key": sig_key,
            "energy_signature": sig,
            **{k: v for k, v in cond.items() if k != "label"},
        })
        log.info("[%s] persistence=%.3f  mean_delta=%.6f", label, score, mean_delta)

    # Print ranking
    ranked = _rank_phase_scores(scores, ranking_mode)
    print(f"\n=== Phase {phase_num} results ===")
    print(f"{'Label':<20}  {'persist':>7}  {'mean_delta':>12}  {'step_percents'}")
    for s in ranked:
        print(f"{s['label']:<20}  {s['persistence_score']:>7.3f}  {s['mean_delta_post']:>12.6f}  "
              f"{s.get('guidance_step_percents', [])}")

    # Save summary
    summary = {"ranking_mode": ranking_mode, "ranked": ranked, "best": ranked[0] if ranked else {}}
    (phase_dir / f"phase{phase_num}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Phase %d summary saved to %s", phase_num, phase_dir)

    # Plot
    prompt_short = case.get("input_caption", case.get("caption", ""))[:60]
    if len(baseline_cache) == 1:
        _plot_phase(all_records, phase_dir, f"Phase{phase_num}", prompt_short)
    else:
        note = phase_dir / f"phase{phase_num}_plot_skipped.txt"
        note.write_text(
            "Skipped combined delta plot because this phase used multiple energy signatures "
            "with different baselines; see per-condition baseline_signature_key in the summary JSON.\n",
            encoding="utf-8",
        )
        log.info("Skipped combined plot for phase %d because %d baseline signatures were used",
                 phase_num, len(baseline_cache))

    return scores, all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe V-JEPA energy persistence -- multi-phase sweep.")
    p.add_argument("--weights-root", type=Path, required=True)
    p.add_argument("--input-json", type=Path, required=True, help="Single case JSON file")
    p.add_argument("--context-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("/data/gaoya/agent-data/outputs/probe_sweep"))
    p.add_argument("--wan-root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--vjepa-device", type=str, default=None)
    p.add_argument("--vjepa-model", type=str, default="vith")
    p.add_argument("--vjepa-ckpt", type=Path, default=Path("/data/gaoya/ckpt/VJEPA2/vith.pt"))
    p.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], default=1,
                   help="Which experiment phase to run (1=timing, 2=step-size, "
                        "3=count+inner_k, 4=mechanism compare: surprise vs context_anchored, "
                        "5=strength ladder: latent_step_size sweep in context_anchored, "
                        "6=knee refinement: timing/inner-k/backtracking near the first wmreward-positive regime, "
                        "7=target-shape sweep: wider context-anchored future windows, "
                        "8=anti-artifact guard sweep on target_w24)")
    p.add_argument("--probe-every-n", type=int, default=2)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-inference-steps", type=int, default=40)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--preview-downsample-factor", type=int, default=4)
    p.add_argument("--preview-frame-stride", type=int, default=2)
    p.add_argument("--window-size", type=int, default=16)
    p.add_argument("--context-frames-vjepa", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--anchor-timing", type=float, default=0.35,
                   help="Phase 4: guidance timing percent for context-anchored conditions")
    p.add_argument("--anchor-step-size", type=float, default=0.005,
                   help="Phase 4: latent step size for context-anchored conditions "
                        "(0.005 = verified descent-basin step; 0.02 overshoots)")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lora_path = _resolve_lora_path(args.weights_root)

    # Base vjepa_config (probe settings, not guidance settings -- those come from conditions)
    vjepa_config = WanVJEPAConfig(
        guidance_steps=1,
        min_step_percent=0.50,
        max_step_percent=0.50,
        latent_step_size=0.02,
        preview_downsample_factor=args.preview_downsample_factor,
        preview_frame_stride=args.preview_frame_stride,
        window_size=args.window_size,
        context_frames=args.context_frames_vjepa,
        stride=args.stride,
    )

    vjepa_device = args.vjepa_device or args.device
    log.info("Building pipeline (device=%s vjepa_device=%s)...", args.device, vjepa_device)
    pipe = _build_pipeline(
        wan_root=args.wan_root.expanduser().resolve(),
        device=args.device,
        lora_path=lora_path,
        vjepa_model=args.vjepa_model,
        vjepa_ckpt=args.vjepa_ckpt.expanduser().resolve() if args.vjepa_ckpt else None,
        vjepa_device=vjepa_device,
        vjepa_config=vjepa_config,
    )

    case = _load_case(args.input_json)
    context_path = args.context_path or Path(case.get("input_video", case.get("context_path", "")))
    if not context_path.is_file():
        raise FileNotFoundError(f"Context path not found: {context_path}")

    base_run_kwargs = dict(
        pipe=pipe,
        case=case,
        seed=args.seed,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        negative_prompt=args.negative_prompt,
        context_path=context_path,
    )

    # Reuse baseline from phase1 if available
    p1_baseline = args.output_dir / "phase1" / "baseline_records.json"

    if args.phase == 1:
        conditions = _phase1_conditions()
        _run_phase(1, conditions, pipe=pipe, case=case,
                   base_run_kwargs=base_run_kwargs, vjepa_config=vjepa_config,
                   output_dir=args.output_dir, probe_every_n=args.probe_every_n)

    elif args.phase == 2:
        p1_summary = args.output_dir / "phase1" / "phase1_summary.json"
        if not p1_summary.exists():
            raise FileNotFoundError(f"Phase 1 summary not found: {p1_summary} -- run --phase 1 first")
        p1 = json.loads(p1_summary.read_text())
        p_best = float(p1["best"]["guidance_step_percents"][0])
        log.info("Phase 1 best timing: p_best=%.2f (label=%s, score=%.3f)",
                 p_best, p1["best"]["label"], p1["best"]["persistence_score"])
        conditions = _phase2_conditions(p_best)
        _run_phase(2, conditions, pipe=pipe, case=case,
                   base_run_kwargs=base_run_kwargs, vjepa_config=vjepa_config,
                   output_dir=args.output_dir, probe_every_n=args.probe_every_n,
                   baseline_records_path=p1_baseline)

    elif args.phase == 3:
        p1_summary = args.output_dir / "phase1" / "phase1_summary.json"
        p2_summary = args.output_dir / "phase2" / "phase2_summary.json"
        if not p1_summary.exists():
            raise FileNotFoundError(f"Phase 1 summary not found -- run --phase 1 first")
        if not p2_summary.exists():
            raise FileNotFoundError(f"Phase 2 summary not found -- run --phase 2 first")
        p1 = json.loads(p1_summary.read_text())
        p2 = json.loads(p2_summary.read_text())
        p_best = float(p1["best"]["guidance_step_percents"][0])
        ss_best = float(p2["best"]["latent_step_size"])
        log.info("Phase 3 using p_best=%.2f ss_best=%.3f", p_best, ss_best)
        conditions = _phase3_conditions(p_best, ss_best)
        _run_phase(3, conditions, pipe=pipe, case=case,
                   base_run_kwargs=base_run_kwargs, vjepa_config=vjepa_config,
                   output_dir=args.output_dir, probe_every_n=args.probe_every_n,
                   baseline_records_path=p1_baseline)

    elif args.phase == 4:
        # Mechanism comparison: context-anchored guidance vs the legacy
        # self-consistency surprise, at a shared timing/step-size. Runs entirely
        # in context_anchored mode so baseline + all conditions are probed with
        # the anchored energy (the quantity the new mechanism optimizes). The
        # anchored baseline is NOT reused from phase1 (different energy).
        p_best = args.anchor_timing
        ss_best = args.anchor_step_size
        log.info("Phase 4 (context-anchored) using p_best=%.2f ss_best=%.3f", p_best, ss_best)
        conditions = _phase4_conditions(p_best, ss_best)
        _run_phase(4, conditions, pipe=pipe, case=case,
                   base_run_kwargs=base_run_kwargs, vjepa_config=vjepa_config,
                   output_dir=args.output_dir, probe_every_n=args.probe_every_n,
                   guidance_mode="context_anchored")

    elif args.phase == 5:
        # Strength ladder (Phase 0a finding): energy reduction of ~0.005 did NOT move
        # wmreward and every phase-4 video is visually identical to baseline. Sweep the
        # step-size axis in context_anchored mode to find where the decoded video
        # actually diverges from baseline. Timing held at the p35..p80 dense band.
        p_best = args.anchor_timing
        ss_best = args.anchor_step_size
        log.info("Phase 5 (strength ladder, context-anchored) p_center=%.2f", p_best)
        conditions = _phase5_conditions(p_best, ss_best)
        _run_phase(5, conditions, pipe=pipe, case=case,
                   base_run_kwargs=base_run_kwargs, vjepa_config=vjepa_config,
                   output_dir=args.output_dir, probe_every_n=args.probe_every_n,
                   guidance_mode="context_anchored",
                   ranking_mode="abs_mean_delta_desc")

    elif args.phase == 6:
        log.info("Phase 6 (knee refinement, context-anchored)")
        conditions = _phase6_conditions()
        _run_phase(6, conditions, pipe=pipe, case=case,
                   base_run_kwargs=base_run_kwargs, vjepa_config=vjepa_config,
                   output_dir=args.output_dir, probe_every_n=args.probe_every_n,
                   guidance_mode="context_anchored",
                   ranking_mode="abs_mean_delta_desc")

    elif args.phase == 7:
        # Phase 3, low-cost path: before reworking the anchored objective itself,
        # hold the strongest fixed-step schedule constant and widen the future
        # horizon V-JEPA must explain from the real context.
        p_best = args.anchor_timing
        log.info("Phase 7 (target-shape sweep, context-anchored) p_center=%.2f", p_best)
        conditions = _phase7_conditions(p_best)
        _run_phase(7, conditions, pipe=pipe, case=case,
                   base_run_kwargs=base_run_kwargs, vjepa_config=vjepa_config,
                   output_dir=args.output_dir, probe_every_n=args.probe_every_n,
                   guidance_mode="context_anchored",
                   ranking_mode="abs_mean_delta_desc")

    elif args.phase == 8:
        p_best = args.anchor_timing
        log.info("Phase 8 (anti-artifact guard sweep, context-anchored) p_center=%.2f", p_best)
        conditions = _phase8_conditions(p_best)
        _run_phase(8, conditions, pipe=pipe, case=case,
                   base_run_kwargs=base_run_kwargs, vjepa_config=vjepa_config,
                   output_dir=args.output_dir, probe_every_n=args.probe_every_n,
                   guidance_mode="context_anchored",
                   ranking_mode="abs_mean_delta_desc")

    log.info("Done. Output dir: %s", args.output_dir)


if __name__ == "__main__":
    main()
