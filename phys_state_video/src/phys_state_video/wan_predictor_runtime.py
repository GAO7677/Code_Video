from __future__ import annotations

from typing import Any

from .checkpoint_io import load_torch_checkpoint
from .predictor_wan_state import WanStateLatentPredictor, WanStateLatentPredictorConfig
from .predictor_wan_state_v2 import WanStateLatentPredictorV2, WanStateLatentPredictorV2Config
from .wan_bridge import WanLatentExtractor
from .wan_state_v2_helpers import WanPromptContextEncoder


def load_wan_state_predictor(checkpoint_path: str, device: str) -> tuple[Any, dict[str, Any]]:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    predictor_version = checkpoint.get("predictor_version", "wan_state_v1")
    if predictor_version == "wan_state_v2_latent_time":
        predictor = WanStateLatentPredictorV2(WanStateLatentPredictorV2Config(**checkpoint["config"])).to(device)
        predictor.load_state_dict(checkpoint["model"])
        predictor.eval()
        return predictor, checkpoint
    predictor = WanStateLatentPredictor(WanStateLatentPredictorConfig(**checkpoint["config"])).to(device)
    predictor.load_state_dict(checkpoint["model"])
    predictor.eval()
    return predictor, checkpoint


def resolve_predictor_wan_task(
    predictor_ckpt: dict[str, Any],
    *,
    default_wan_task: str,
    predictor_wan_task: str | None = None,
) -> str:
    if predictor_wan_task:
        return str(predictor_wan_task)
    return str(predictor_ckpt.get("wan_task") or default_wan_task)


def validate_v2_predictor_checkpoint(predictor_ckpt: dict[str, Any], *, context: str) -> None:
    if predictor_ckpt.get("predictor_version", "wan_state_v1") != "wan_state_v2_latent_time":
        return
    latent_source = predictor_ckpt.get("latent_source")
    if latent_source != "wan":
        raise ValueError(
            f"{context} requires wan_state_v2_latent_time checkpoints to declare latent_source='wan', "
            f"got latent_source={latent_source!r}"
        )


def build_predictor_latent_extractor(
    *,
    wan_ckpt_dir: str | None,
    wan_repo_root: str,
    device: str,
    predictor_ckpt: dict[str, Any],
    default_wan_task: str,
    predictor_wan_task: str | None = None,
    context: str,
) -> WanLatentExtractor:
    validate_v2_predictor_checkpoint(predictor_ckpt, context=context)
    if wan_ckpt_dir is None:
        raise ValueError(f"--wan-ckpt-dir is required because {context} uses Wan VAE latents")
    predictor_task = resolve_predictor_wan_task(
        predictor_ckpt,
        default_wan_task=default_wan_task,
        predictor_wan_task=predictor_wan_task,
    )
    return WanLatentExtractor(
        ckpt_dir=wan_ckpt_dir,
        wan_repo_root=wan_repo_root,
        task=predictor_task,
        device=device,
    )


def build_predictor_prompt_context_encoder(
    *,
    wan_ckpt_dir: str | None,
    wan_repo_root: str,
    device: str,
    predictor_ckpt: dict[str, Any],
    default_wan_task: str,
    predictor_wan_task: str | None = None,
    context: str,
) -> WanPromptContextEncoder:
    if wan_ckpt_dir is None:
        raise ValueError(f"--wan-ckpt-dir is required because {context} uses frozen Wan T5 prompt context")
    predictor_task = resolve_predictor_wan_task(
        predictor_ckpt,
        default_wan_task=default_wan_task,
        predictor_wan_task=predictor_wan_task,
    )
    return WanPromptContextEncoder(
        ckpt_dir=wan_ckpt_dir,
        wan_repo_root=wan_repo_root,
        task=predictor_task,
        device=device,
    )
