from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PredictorConfig:
    state_dim: int = 10
    appearance_dim: int = 64
    camera_dim: int = 8
    prompt_vocab_size: int = 4096
    prompt_embed_dim: int = 64
    hidden_dim: int = 192
    num_layers: int = 2
    future_steps: int = 12
    dropout: float = 0.1
    interaction_radius: float = 0.2
    uncertainty_bias: float = 0.05


@dataclass(slots=True)
class ProjectionConfig:
    velocity_smooth_weight: float = 0.5
    depth_smooth_weight: float = 0.25
    visibility_on_threshold: float = 0.55
    visibility_off_threshold: float = 0.45
    scale_depth_weight: float = 0.35
    max_scale_delta: float = 0.6
    max_position_delta: float = 0.3
    low_confidence_threshold: float = 0.45


@dataclass(slots=True)
class ConditioningConfig:
    frame_height: int = 128
    frame_width: int = 128
    heatmap_sigma: float = 4.0
    include_velocity_maps: bool = True
    include_existence_map: bool = True


@dataclass(slots=True)
class AdapterConfig:
    context_channels: int = 3
    cond_channels: int = 7
    memory_dim: int = 96
    latent_dim: int = 128
    num_heads: int = 4
    future_steps: int = 12
    freeze_backbone: bool = False
    dropout: float = 0.1
    temporal_kernel_size: int = 3
