from __future__ import annotations

from dataclasses import asdict, dataclass


PILOT_PROMPT_IDS = [
    "001",  # microgravity liquid
    "006",  # buoyancy / sinking
    "011",  # squeeze / stress
    "021",  # puncture / pressure
    "026",  # bounce / elasticity
    "036",  # surface tension
    "071",  # refraction
    "091",  # freezing
    "096",  # melting
    "101",  # condensation
    "106",  # boiling
    "116",  # sublimation / fog
    "126",  # flame reaction
    "131",  # brittle collision
    "136",  # immiscible liquids
    "151",  # sulfuric acid reaction
]


@dataclass(frozen=True)
class ExperimentMode:
    mode_id: str
    description: str
    disable_vjepa_guidance: bool
    vjepa_model: str = "vith"
    vjepa_guidance_steps: int = 0
    vjepa_min_step_percent: float = 0.0
    vjepa_max_step_percent: float = 0.0
    vjepa_latent_step_size: float = 0.0
    vjepa_preview_downsample_factor: int = 4
    vjepa_preview_frame_stride: int = 2
    vjepa_window_size: int = 16
    vjepa_context_frames: int = 8
    vjepa_stride: int = 4
    vjepa_reduction: str = "mean"
    vjepa_grad_norm_mode: str = "rms"
    vjepa_max_grad_norm: float = 10.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ROUND1_MODES = [
    ExperimentMode(
        mode_id="baseline",
        description="Wan2.2 TI2V baseline without V-JEPA guidance.",
        disable_vjepa_guidance=True,
    ),
    ExperimentMode(
        mode_id="g1_mid1_s001",
        description="Single mid-step guidance with light latent correction.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=1,
        vjepa_min_step_percent=0.50,
        vjepa_max_step_percent=0.50,
        vjepa_latent_step_size=0.01,
    ),
    ExperimentMode(
        mode_id="g2_mid2_s001",
        description="Two mid-range guidance steps with light latent correction.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=2,
        vjepa_min_step_percent=0.35,
        vjepa_max_step_percent=0.65,
        vjepa_latent_step_size=0.01,
    ),
    ExperimentMode(
        mode_id="g3_mid2_s002",
        description="Two mid-range guidance steps with moderate latent correction.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=2,
        vjepa_min_step_percent=0.35,
        vjepa_max_step_percent=0.65,
        vjepa_latent_step_size=0.02,
    ),
    ExperimentMode(
        mode_id="g4_wide4_s001",
        description="Four wide-range guidance steps with light latent correction.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=4,
        vjepa_min_step_percent=0.20,
        vjepa_max_step_percent=0.80,
        vjepa_latent_step_size=0.01,
    ),
    ExperimentMode(
        mode_id="g5_wide4_s002",
        description="Four wide-range guidance steps with moderate latent correction.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=4,
        vjepa_min_step_percent=0.20,
        vjepa_max_step_percent=0.80,
        vjepa_latent_step_size=0.02,
    ),
    ExperimentMode(
        mode_id="g6_wide6_s002",
        description="Six wide-range guidance steps with moderate latent correction.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=6,
        vjepa_min_step_percent=0.20,
        vjepa_max_step_percent=0.80,
        vjepa_latent_step_size=0.02,
    ),
]


MODE_MAP = {mode.mode_id: mode for mode in ROUND1_MODES}


MID12_INDICES = (8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25)
EARLY12_INDICES = (4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21)


@dataclass(frozen=True)
class Train0705GuidancePreset:
    mode_id: str
    description: str
    disable_vjepa_guidance: bool
    vjepa_model: str = "vith"
    vjepa_guidance_mode: str = "context_anchored"
    vjepa_guidance_steps: int = 12
    vjepa_min_step_percent: float = 0.35
    vjepa_max_step_percent: float = 0.80
    vjepa_target_step_indices: tuple[int, ...] = ()
    vjepa_target_timesteps: tuple[int, ...] = ()
    vjepa_latent_step_size: float = 0.20
    vjepa_inner_k: int = 1
    vjepa_backtracking: bool = False
    vjepa_preview_downsample_factor: int = 4
    vjepa_preview_frame_stride: int = 1
    vjepa_window_size: int = 24
    vjepa_context_frames: int = 8
    vjepa_stride: int = 4
    vjepa_reduction: str = "mean"
    vjepa_grad_norm_mode: str = "rms"
    vjepa_max_grad_norm: float = 10.0
    vjepa_max_correction_ratio: float = 0.05
    vjepa_stay_close_max_video_l1: float = 0.03
    vjepa_artifact_guard_mode: str = "video_l1_backoff"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TRAIN0705_CURRENT_MODES = [
    Train0705GuidancePreset(
        mode_id="baseline",
        description="train0705 Wan2.2 baseline without V-JEPA guidance.",
        disable_vjepa_guidance=True,
    ),
    Train0705GuidancePreset(
        mode_id="ladder_s20",
        description="Current wmreward winner: dense mid-band guidance, step size 0.20.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=len(MID12_INDICES),
        vjepa_target_step_indices=MID12_INDICES,
        vjepa_latent_step_size=0.20,
    ),
    Train0705GuidancePreset(
        mode_id="knee_mid_s18",
        description="Phase-6 near-winner: dense mid-band guidance, step size 0.18.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=len(MID12_INDICES),
        vjepa_target_step_indices=MID12_INDICES,
        vjepa_latent_step_size=0.18,
    ),
    Train0705GuidancePreset(
        mode_id="knee_early_s15",
        description="Timing variant: dense early-band guidance, step size 0.15.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=len(EARLY12_INDICES),
        vjepa_target_step_indices=EARLY12_INDICES,
        vjepa_latent_step_size=0.15,
    ),
    Train0705GuidancePreset(
        mode_id="knee_mid_s10_k2",
        description="Inner-k variant: dense mid-band guidance, step size 0.10, inner_k=2.",
        disable_vjepa_guidance=False,
        vjepa_guidance_steps=len(MID12_INDICES),
        vjepa_target_step_indices=MID12_INDICES,
        vjepa_latent_step_size=0.10,
        vjepa_inner_k=2,
    ),
]


TRAIN0705_MODE_MAP = {mode.mode_id: mode for mode in TRAIN0705_CURRENT_MODES}


def apply_train0705_preset(args, preset_name: str):
    if preset_name not in TRAIN0705_MODE_MAP:
        raise ValueError(
            f"Unknown train0705 V-JEPA preset: {preset_name}. "
            f"Available: {sorted(TRAIN0705_MODE_MAP)}"
        )
    preset = TRAIN0705_MODE_MAP[preset_name]
    args.vjepa_preset = preset.mode_id
    args.enable_vjepa_guidance = not bool(preset.disable_vjepa_guidance)
    args.vjepa_model = str(preset.vjepa_model)
    args.vjepa_guidance_mode = str(preset.vjepa_guidance_mode)
    args.vjepa_guidance_steps = int(preset.vjepa_guidance_steps)
    args.vjepa_min_step_percent = float(preset.vjepa_min_step_percent)
    args.vjepa_max_step_percent = float(preset.vjepa_max_step_percent)
    args.vjepa_target_step_indices = [int(value) for value in preset.vjepa_target_step_indices]
    args.vjepa_target_timesteps = [int(value) for value in preset.vjepa_target_timesteps]
    args.vjepa_latent_step_size = float(preset.vjepa_latent_step_size)
    args.vjepa_inner_k = int(preset.vjepa_inner_k)
    args.vjepa_backtracking = bool(preset.vjepa_backtracking)
    args.vjepa_preview_downsample_factor = int(preset.vjepa_preview_downsample_factor)
    args.vjepa_preview_frame_stride = int(preset.vjepa_preview_frame_stride)
    args.vjepa_window_size = int(preset.vjepa_window_size)
    args.vjepa_context_frames = int(preset.vjepa_context_frames)
    args.vjepa_stride = int(preset.vjepa_stride)
    args.vjepa_reduction = str(preset.vjepa_reduction)
    args.vjepa_grad_norm_mode = str(preset.vjepa_grad_norm_mode)
    args.vjepa_max_grad_norm = float(preset.vjepa_max_grad_norm)
    args.vjepa_max_correction_ratio = float(preset.vjepa_max_correction_ratio)
    args.vjepa_stay_close_max_video_l1 = float(preset.vjepa_stay_close_max_video_l1)
    args.vjepa_artifact_guard_mode = str(preset.vjepa_artifact_guard_mode)
    return preset
