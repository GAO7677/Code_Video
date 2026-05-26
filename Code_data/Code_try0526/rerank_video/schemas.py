from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InputSpec:
    prompt: str
    context_video_path: Path


@dataclass
class GeneratorConfig:
    key: str
    type: str
    enabled: bool = True
    device: str = "cuda"
    model_root: Path | None = None
    lora_path: Path | None = None
    num_candidates: int = 4
    base_seed: int = 42
    height: int = 384
    width: int = 672
    fps: int = 16
    num_frames: int = 49
    context_frames: int = 8
    num_inference_steps: int = 50
    cfg_scale: float = 5.0
    quality: int = 5
    negative_prompt: str = ""
    conditioning_mode: str = "context_aware"


@dataclass
class LatentMotionConfig:
    device: str = "cuda"
    vae_root: Path | None = None
    max_frames: int = 49
    tile_size: tuple[int, int] = (34, 34)
    tile_stride: tuple[int, int] = (18, 16)


@dataclass
class GeometryConfig:
    mode: str = "proxy"
    backend: str = "legacy_motion"
    device: str = "cuda"
    max_frames: int = 49
    diff_threshold: float = 18.0
    min_component_area: int = 192
    min_mask_pixels: int = 64
    pdi_repo_root: Path | None = None
    sam_ckpt: Path | None = None
    sam_cfg: Path | None = None
    tracker_ckpt: Path | None = None
    depth_anything_repo_root: Path | None = None
    depth_anything_ckpt: Path | None = None


@dataclass
class JEPAScoreConfig:
    backend: str = "vjepa2"
    device: str = "cuda"
    max_frames: int = 48
    context_frames: int = 16
    future_frames: int = 16
    context_repeat_frames: int = 8
    crop_size: int = 384
    vjepa_checkpoint: Path | None = None
    vjepa_repo_root: Path | None = None
    vjepa_model_name: str = "vjepa2_1_vit_large_384"
    videomae_model_id: str = ""


@dataclass
class ScoringConfig:
    weights: dict[str, float]
    latent_motion: LatentMotionConfig
    geometry: GeometryConfig
    jepa: JEPAScoreConfig


@dataclass
class CandidateRecord:
    candidate_id: str
    generator_key: str
    generator_type: str
    seed: int
    video_path: Path
    used_context_frames: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateScore:
    candidate_id: str
    raw_scores: dict[str, float]
    normalized_scores: dict[str, float]
    weighted_total: float
    details: dict[str, Any]


@dataclass
class RunConfig:
    run_name: str
    output_root: Path
    tmp_root: Path
    input: InputSpec
    generators: list[GeneratorConfig]
    scoring: ScoringConfig


def _path_or_none(value: Any) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(value).expanduser().resolve()


def _as_tuple2(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return default


def load_run_config(payload: dict[str, Any]) -> RunConfig:
    input_cfg = payload["input"]
    generators: list[GeneratorConfig] = []
    for item in payload.get("generators", []):
        generators.append(
            GeneratorConfig(
                key=str(item["key"]),
                type=str(item["type"]),
                enabled=bool(item.get("enabled", True)),
                device=str(item.get("device", "cuda")),
                model_root=_path_or_none(item.get("model_root")),
                lora_path=_path_or_none(item.get("lora_path")),
                num_candidates=int(item.get("num_candidates", 4)),
                base_seed=int(item.get("base_seed", 42)),
                height=int(item.get("height", 384)),
                width=int(item.get("width", 672)),
                fps=int(item.get("fps", 16)),
                num_frames=int(item.get("num_frames", 49)),
                context_frames=int(item.get("context_frames", 8)),
                num_inference_steps=int(item.get("num_inference_steps", 50)),
                cfg_scale=float(item.get("cfg_scale", 5.0)),
                quality=int(item.get("quality", 5)),
                negative_prompt=str(item.get("negative_prompt", "")),
                conditioning_mode=str(item.get("conditioning_mode", "context_aware")),
            )
        )

    scoring_cfg = payload["scoring"]
    latent_motion_cfg = scoring_cfg.get("latent_motion", {})
    geometry_cfg = scoring_cfg.get("geometry", {})
    jepa_cfg = scoring_cfg.get("jepa", {})
    return RunConfig(
        run_name=str(payload["run_name"]),
        output_root=Path(payload["output_root"]).expanduser().resolve(),
        tmp_root=Path(payload["tmp_root"]).expanduser().resolve(),
        input=InputSpec(
            prompt=str(input_cfg["prompt"]),
            context_video_path=Path(input_cfg["context_video_path"]).expanduser().resolve(),
        ),
        generators=generators,
        scoring=ScoringConfig(
            weights={str(k): float(v) for k, v in scoring_cfg.get("weights", {}).items()},
            latent_motion=LatentMotionConfig(
                device=str(latent_motion_cfg.get("device", "cuda")),
                vae_root=_path_or_none(latent_motion_cfg.get("vae_root")),
                max_frames=int(latent_motion_cfg.get("max_frames", 49)),
                tile_size=_as_tuple2(latent_motion_cfg.get("tile_size"), (34, 34)),
                tile_stride=_as_tuple2(latent_motion_cfg.get("tile_stride"), (18, 16)),
            ),
            geometry=GeometryConfig(
                mode=str(geometry_cfg.get("mode", "proxy")),
                backend=str(geometry_cfg.get("backend", "legacy_motion")),
                device=str(geometry_cfg.get("device", "cuda")),
                max_frames=int(geometry_cfg.get("max_frames", 49)),
                diff_threshold=float(geometry_cfg.get("diff_threshold", 18.0)),
                min_component_area=int(geometry_cfg.get("min_component_area", 192)),
                min_mask_pixels=int(geometry_cfg.get("min_mask_pixels", 64)),
                pdi_repo_root=_path_or_none(geometry_cfg.get("pdi_repo_root")),
                sam_ckpt=_path_or_none(geometry_cfg.get("sam_ckpt")),
                sam_cfg=_path_or_none(geometry_cfg.get("sam_cfg")),
                tracker_ckpt=_path_or_none(geometry_cfg.get("tracker_ckpt")),
                depth_anything_repo_root=_path_or_none(geometry_cfg.get("depth_anything_repo_root")),
                depth_anything_ckpt=_path_or_none(geometry_cfg.get("depth_anything_ckpt")),
            ),
            jepa=JEPAScoreConfig(
                backend=str(jepa_cfg.get("backend", "vjepa2")),
                device=str(jepa_cfg.get("device", "cuda")),
                max_frames=int(jepa_cfg.get("max_frames", 48)),
                context_frames=int(jepa_cfg.get("context_frames", 16)),
                future_frames=int(jepa_cfg.get("future_frames", 16)),
                context_repeat_frames=int(jepa_cfg.get("context_repeat_frames", 8)),
                crop_size=int(jepa_cfg.get("crop_size", 384)),
                vjepa_checkpoint=_path_or_none(jepa_cfg.get("vjepa_checkpoint")),
                vjepa_repo_root=_path_or_none(jepa_cfg.get("vjepa_repo_root")),
                vjepa_model_name=str(jepa_cfg.get("vjepa_model_name", "vjepa2_1_vit_large_384")),
                videomae_model_id=str(jepa_cfg.get("videomae_model_id", "")),
            ),
        ),
    )
