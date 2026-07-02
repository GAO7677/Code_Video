from __future__ import annotations

from dataclasses import dataclass, asdict


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
