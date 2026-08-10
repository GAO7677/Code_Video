#!/usr/bin/env python3
"""Validate an experiment JSON and launch the shared training entry point."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_xssc_object_self_attn_lora.py"
PHYSRVG_DIT_TRAIN_SCRIPT = ROOT / "train_xssc_object_self_attn_lora_physrvg_dit.py"
VJEPA_LOSS_TRAIN_SCRIPT = (
    ROOT / "vjepa_loss_project/train_xssc_object_self_attn_lora_vjepa_loss.py"
)
XSSC_LOSS_TRAIN_SCRIPT = (
    ROOT / "xssc_loss_project/train_xssc_object_self_attn_lora_xssc_loss.py"
)
OFFICIAL_XSSC_OBJECT_ONLY_TRAIN_SCRIPT = ROOT / "train_official_xssc_object_only.py"
VALID_MODES = {"object_only", "full_sa", "s_head", "t_head"}
HEAD_SELECTIVE_MODES = {"s_head", "t_head"}
VALID_XSSC_BACKENDS = {"dinov3_movic", "official_dinov2"}
VALID_INITIALIZATIONS = {"openvid_lora", "physrvg_dit"}


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path, stack: tuple[Path, ...] = ()) -> tuple[dict, list[str]]:
    resolved = path.expanduser().resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Config inheritance cycle: {chain}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Config does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    parent = payload.pop("extends", None)
    if parent is None:
        config = payload
        sources = [str(resolved)]
    else:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        parent_config, sources = load_config(parent_path, (*stack, resolved))
        config = deep_merge(parent_config, payload)
        sources.append(str(resolved))
    return config, sources


def require(mapping: dict, dotted_key: str):
    value = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing required config key: {dotted_key}")
        value = value[key]
    return value


def resolve_config_path(value: str, config_dir: Path) -> str:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def validate_config(config: dict, config_dir: Path) -> dict:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Experiment config schema_version must be 1")
    mode = str(require(config, "adaptation.mode"))
    if mode not in VALID_MODES:
        raise ValueError(f"adaptation.mode must be one of {sorted(VALID_MODES)}")
    enable_object_branch = require(config, "adaptation.enable_object_branch")
    if not isinstance(enable_object_branch, bool):
        raise TypeError("adaptation.enable_object_branch must be a boolean")
    if not enable_object_branch and mode not in {"full_sa", *HEAD_SELECTIVE_MODES}:
        raise ValueError(
            "Disabling the object branch requires a self-attention adaptation mode"
        )
    xssc_backend = str(require(config, "model.xssc_backend"))
    if xssc_backend not in VALID_XSSC_BACKENDS:
        raise ValueError(
            f"model.xssc_backend must be one of {sorted(VALID_XSSC_BACKENDS)}"
        )
    xssc_loss = config.get("xssc_loss", {})
    xssc_loss_enabled = bool(xssc_loss.get("enabled", False))
    if xssc_backend == "official_dinov2" and not (
        (mode == "object_only" and enable_object_branch)
        or (xssc_loss_enabled and mode == "full_sa" and not enable_object_branch)
    ):
        raise ValueError(
            "official_dinov2 supports either object_only with the object branch "
            "enabled, or Full-SA no-object with xssc_loss enabled"
        )
    expected_role_by_mode = {"s_head": "S", "t_head": "T"}
    if mode in HEAD_SELECTIVE_MODES:
        configured_role = str(require(config, "adaptation.head_selection_expected_role"))
        if configured_role != expected_role_by_mode[mode]:
            raise ValueError(
                f"{mode} requires role={expected_role_by_mode[mode]}, "
                f"got {configured_role!r}"
            )
    name = str(require(config, "experiment.name")).strip()
    if not name or name == "override_in_child_config":
        raise ValueError("experiment.name must be set by the child config")

    gpu_ids = [
        item.strip() for item in str(require(config, "launch.gpu_set")).split(",")
        if item.strip()
    ]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Invalid launch.gpu_set: {gpu_ids}")
    if "4" in gpu_ids:
        raise ValueError("GPU 4 is prohibited by workspace rules")
    if int(require(config, "launch.num_processes")) != len(gpu_ids):
        raise ValueError(
            "launch.num_processes must match the number of visible GPUs: "
            f"{require(config, 'launch.num_processes')} vs {gpu_ids}"
        )
    main_process_port = config["launch"].get("main_process_port")
    if main_process_port is not None and not 1 <= int(main_process_port) <= 65535:
        raise ValueError(
            f"launch.main_process_port must be in [1, 65535], got {main_process_port}"
        )

    ratios = [
        float(require(config, "data.mixture_pybullet_ratio")),
        float(require(config, "data.mixture_kubric_ratio")),
        float(require(config, "data.mixture_openvid_ratio")),
    ]
    if any(value < 0.0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"Dataset mixture ratios must be nonnegative and sum to 1: {ratios}")
    if int(require(config, "model.fixed_num_context_frames")) != 8:
        raise ValueError("xSSC experiments require exactly 8 context frames")
    if int(require(config, "model.num_frames")) <= 8:
        raise ValueError("model.num_frames must be greater than the context length")
    if int(require(config, "optimization.train_batch_size_per_gpu")) <= 0:
        raise ValueError("optimization.train_batch_size_per_gpu must be positive")
    if int(require(config, "optimization.gradient_accumulation_steps")) <= 0:
        raise ValueError("optimization.gradient_accumulation_steps must be positive")
    if int(require(config, "adaptation.self_attn_lora_rank")) <= 0:
        raise ValueError("adaptation.self_attn_lora_rank must be positive")

    initialization_type = str(
        config.get("initialization", {}).get("type", "openvid_lora")
    )
    if initialization_type not in VALID_INITIALIZATIONS:
        raise ValueError(
            "initialization.type must be one of "
            f"{sorted(VALID_INITIALIZATIONS)}, got {initialization_type!r}"
        )
    if initialization_type == "physrvg_dit" and (
        xssc_backend != "dinov3_movic" or mode != "full_sa"
    ):
        raise ValueError(
            "PhysRVG DiT initialization currently supports only "
            "dinov3_movic Full-SA experiments"
        )

    vjepa_loss = config.get("vjepa_loss", {})
    vjepa_loss_enabled = bool(vjepa_loss.get("enabled", False))
    if vjepa_loss_enabled and xssc_loss_enabled:
        raise ValueError("vjepa_loss and xssc_loss cannot be enabled together")
    if vjepa_loss_enabled:
        if xssc_backend != "dinov3_movic":
            raise ValueError(
                "vjepa_loss currently requires model.xssc_backend=dinov3_movic"
            )
        if float(require(config, "vjepa_loss.weight")) <= 0.0:
            raise ValueError("vjepa_loss.weight must be positive")
        sigma_min = float(require(config, "vjepa_loss.sigma_min"))
        sigma_max = float(require(config, "vjepa_loss.sigma_max"))
        if not 0.0 <= sigma_min < sigma_max <= 1.0:
            raise ValueError(
                "vjepa_loss sigma range must satisfy "
                f"0 <= sigma_min < sigma_max <= 1, got [{sigma_min}, {sigma_max}]"
            )
        if int(require(config, "vjepa_loss.every_n_forwards")) <= 0:
            raise ValueError("vjepa_loss.every_n_forwards must be positive")
        num_frames = int(require(config, "vjepa_loss.num_frames"))
        frame_sampling = str(require(config, "vjepa_loss.frame_sampling"))
        if num_frames <= 0:
            raise ValueError("vjepa_loss.num_frames must be positive")
        if frame_sampling != "full" and num_frames % 2:
            raise ValueError(
                "vjepa_loss.num_frames must be even unless frame_sampling=full"
            )
        if int(require(config, "vjepa_loss.input_size")) != 384:
            raise ValueError(
                "The configured V-JEPA2.1 ViT-L checkpoint requires input_size=384"
            )
        if not isinstance(require(config, "vjepa_loss.tiny_vae_parallel"), bool):
            raise TypeError("vjepa_loss.tiny_vae_parallel must be a boolean")
        if float(require(config, "vjepa_loss.range_penalty_weight")) < 0.0:
            raise ValueError("vjepa_loss.range_penalty_weight must be non-negative")
        if frame_sampling not in {"global", "local", "mixed", "full"}:
            raise ValueError(
                "vjepa_loss.frame_sampling must be global/local/mixed/full"
            )
        local_probability = float(
            require(config, "vjepa_loss.local_sampling_probability")
        )
        if not 0.0 <= local_probability <= 1.0:
            raise ValueError(
                "vjepa_loss.local_sampling_probability must be in [0, 1]"
            )
        local_context_frames = int(
            require(config, "vjepa_loss.local_context_frames")
        )
        if not 0 < local_context_frames < num_frames:
            raise ValueError(
                "vjepa_loss.local_context_frames must be in [1, num_frames)"
            )
        if int(
            require(config, "vjepa_loss.gradient_diagnostics_every_n_forwards")
        ) <= 0:
            raise ValueError(
                "vjepa_loss.gradient_diagnostics_every_n_forwards must be positive"
            )

    if xssc_loss_enabled:
        if mode != "full_sa" or enable_object_branch:
            raise ValueError(
                "xssc_loss requires adaptation.mode=full_sa and the object branch disabled"
            )
        backend = str(require(config, "xssc_loss.backend"))
        if backend != xssc_backend:
            raise ValueError(
                "xssc_loss.backend must equal model.xssc_backend: "
                f"{backend!r} vs {xssc_backend!r}"
            )
        if float(require(config, "xssc_loss.weight")) <= 0.0:
            raise ValueError("xssc_loss.weight must be positive")
        future_start = int(require(config, "xssc_loss.future_start_frame"))
        if future_start != int(require(config, "model.fixed_num_context_frames")):
            raise ValueError(
                "xssc_loss.future_start_frame must equal the 8-frame context length"
            )
        if int(require(config, "xssc_loss.backbone_chunk_size")) <= 0:
            raise ValueError("xssc_loss.backbone_chunk_size must be positive")
        if int(
            require(config, "xssc_loss.gradient_diagnostics_every_n_forwards")
        ) <= 0:
            raise ValueError(
                "xssc_loss.gradient_diagnostics_every_n_forwards must be positive"
            )

    path_keys = [
        "paths.project_root",
        "paths.diffsynth_root",
        "paths.wan_root",
        "paths.pybullet_root",
        "paths.kubric_root",
        "paths.openvid_root",
    ]
    if initialization_type == "openvid_lora":
        path_keys.append("paths.pretrained_lora_checkpoint")
    else:
        path_keys.append("paths.physrvg_dit_checkpoint")
    if enable_object_branch or xssc_loss_enabled:
        path_keys.extend(
            [
                "paths.xssc_root",
                "paths.xssc_config",
                "paths.xssc_checkpoint",
            ]
        )
        if xssc_backend == "dinov3_movic":
            path_keys.extend(
                [
                    "paths.dinov3_root",
                    "paths.dinov3_checkpoint",
                    "paths.sam2_config",
                    "paths.sam2_checkpoint",
                ]
            )
    normalized = copy.deepcopy(config)
    normalized.setdefault("initialization", {})["type"] = initialization_type
    for dotted_key in path_keys:
        keys = dotted_key.split(".")
        normalized[keys[0]][keys[1]] = resolve_config_path(
            str(require(config, dotted_key)), config_dir
        )
        if not Path(normalized[keys[0]][keys[1]]).exists():
            raise FileNotFoundError(
                f"Configured path does not exist ({dotted_key}): "
                f"{normalized[keys[0]][keys[1]]}"
            )

    if vjepa_loss_enabled:
        for key in (
            "vjepa_repo",
            "vjepa_checkpoint",
            "tiny_vae_root",
            "tiny_vae_checkpoint",
        ):
            normalized["vjepa_loss"][key] = resolve_config_path(
                str(require(config, f"vjepa_loss.{key}")), config_dir
            )
            if not Path(normalized["vjepa_loss"][key]).exists():
                raise FileNotFoundError(
                    f"Configured V-JEPA loss path does not exist ({key}): "
                    f"{normalized['vjepa_loss'][key]}"
                )

    normalized["paths"]["output_root"] = resolve_config_path(
        str(require(config, "paths.output_root")), config_dir
    )
    normalized["paths"]["cache_root"] = resolve_config_path(
        str(require(config, "paths.cache_root")), config_dir
    )
    normalized["paths"]["xssc_box_cache_dir"] = resolve_config_path(
        str(require(config, "paths.xssc_box_cache_dir")), config_dir
    )
    normalized["paths"]["head_selection_config"] = resolve_config_path(
        str(require(config, "paths.head_selection_config")), config_dir
    )
    resume_from = normalized["checkpointing"].get("resume_from")
    if resume_from:
        normalized["checkpointing"]["resume_from"] = resolve_config_path(
            str(resume_from), config_dir
        )
        if not Path(normalized["checkpointing"]["resume_from"]).is_file():
            raise FileNotFoundError(
                "Resume training state does not exist: "
                f"{normalized['checkpointing']['resume_from']}"
            )
    wandb_run_id = str(normalized["logging"].get("wandb_run_id", "")).strip()
    wandb_resume = str(normalized["logging"].get("wandb_resume", "")).strip()
    if bool(wandb_run_id) != bool(wandb_resume):
        raise ValueError(
            "logging.wandb_run_id and logging.wandb_resume must be set together"
        )
    if wandb_resume and wandb_resume not in {"allow", "must", "never", "auto"}:
        raise ValueError(
            "logging.wandb_resume must be one of allow/must/never/auto"
        )
    if mode in HEAD_SELECTIVE_MODES and not Path(
        normalized["paths"]["head_selection_config"]
    ).is_file():
        raise FileNotFoundError(
            "Head-selection config does not exist: "
            f"{normalized['paths']['head_selection_config']}"
        )
    return normalized


def add_option(command: list[str], name: str, value) -> None:
    command.extend((name, str(value)))


def snapshot_head_selection_config(
    config: dict,
    output_dir: Path,
) -> dict[str, object] | None:
    if config["adaptation"]["mode"] not in HEAD_SELECTIVE_MODES:
        return None
    source_path = Path(config["paths"]["head_selection_config"]).resolve()
    content = source_path.read_bytes()
    payload = json.loads(content.decode("utf-8"))
    snapshot_path = output_dir / "head_selection_config.json"
    snapshot_path.write_bytes(content)
    config["paths"]["head_selection_config"] = str(snapshot_path)
    return {
        "source_path": str(source_path),
        "snapshot_path": str(snapshot_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "subset_id": payload.get("subset_id"),
        "role": payload.get("role"),
        "feature_subtype": payload.get("feature_subtype"),
        "num_heads": payload.get("num_heads"),
    }


def build_command(config: dict, output_dir: Path) -> list[str]:
    launch = config["launch"]
    paths = config["paths"]
    model = config["model"]
    adaptation = config["adaptation"]
    conditioning = config["conditioning"]
    data = config["data"]
    optim = config["optimization"]
    checkpointing = config["checkpointing"]
    logging = config["logging"]
    vjepa_loss = config.get("vjepa_loss", {})
    vjepa_loss_enabled = bool(vjepa_loss.get("enabled", False))
    xssc_loss = config.get("xssc_loss", {})
    xssc_loss_enabled = bool(xssc_loss.get("enabled", False))

    xssc_backend = str(model["xssc_backend"])
    initialization_type = str(
        config.get("initialization", {}).get("type", "openvid_lora")
    )
    if xssc_loss_enabled:
        train_script = XSSC_LOSS_TRAIN_SCRIPT
    elif vjepa_loss_enabled:
        train_script = VJEPA_LOSS_TRAIN_SCRIPT
    elif initialization_type == "physrvg_dit":
        train_script = PHYSRVG_DIT_TRAIN_SCRIPT
    else:
        train_script = (
            OFFICIAL_XSSC_OBJECT_ONLY_TRAIN_SCRIPT
            if xssc_backend == "official_dinov2"
            else TRAIN_SCRIPT
        )
    command = [
        str(launch["accelerate_bin"]),
        "launch",
    ]
    if int(launch["num_processes"]) > 1:
        command.append("--multi_gpu")
    if launch.get("main_process_port") is not None:
        command.extend([
            "--main_process_port",
            str(launch["main_process_port"]),
        ])
    command.extend([
        "--num_processes",
        str(launch["num_processes"]),
        "--num_machines",
        "1",
        "--mixed_precision",
        str(launch["mixed_precision"]),
        str(train_script),
    ])
    options = {
        "--diffsynth_root": paths["diffsynth_root"],
        "--wan_root": paths["wan_root"],
        "--expected_trainable_params": config["experiment"][
            "expected_trainable_params"
        ],
        "--dataset_type": data["dataset_type"],
        "--pybullet0713_root": paths["pybullet_root"],
        "--pybullet0713_split": data["pybullet_split"],
        "--pybullet0713_sampling_strategy": data["pybullet_sampling_strategy"],
        "--kubric_root": paths["kubric_root"],
        "--kubric_split": data["kubric_split"],
        "--kubric_sampling_strategy": data["kubric_sampling_strategy"],
        "--kubric_cache_root": data["kubric_cache_root"],
        "--kubric_replay_index_num_frames": data["kubric_replay_index_num_frames"],
        "--kubric_replay_index_num_context_frames": data[
            "kubric_replay_index_num_context_frames"
        ],
        "--openvid_root": paths["openvid_root"],
        "--mixture_pybullet_ratio": data["mixture_pybullet_ratio"],
        "--mixture_kubric_ratio": data["mixture_kubric_ratio"],
        "--mixture_openvid_ratio": data["mixture_openvid_ratio"],
        "--height": model["height"],
        "--width": model["width"],
        "--num_frames": model["num_frames"],
        "--fixed_num_context_frames": model["fixed_num_context_frames"],
        "--train_batch_size": optim["train_batch_size_per_gpu"],
        "--no_context_ratio": conditioning["no_context_ratio"],
        "--max_train_steps": optim["max_train_steps"],
        "--num_epochs": optim["num_epochs"],
        "--dataset_num_workers": data["dataset_num_workers"],
        "--learning_rate": optim["learning_rate"],
        "--weight_decay": optim["weight_decay"],
        "--gradient_accumulation_steps": optim["gradient_accumulation_steps"],
        "--optimizer_type": optim["optimizer_type"],
        "--max_grad_norm": optim["max_grad_norm"],
        "--save_steps": checkpointing["save_steps"],
        "--max_checkpoints_keep": checkpointing["max_checkpoints_keep"],
        "--remove_prefix_in_ckpt": checkpointing["remove_prefix_in_ckpt"],
        "--output_path": output_dir,
        "--lora_base_model": "dit",
        "--lora_target_modules": model["pretrained_lora_target_modules"],
        "--lora_rank": model["pretrained_lora_rank"],
        "--lora_alpha": model["pretrained_lora_alpha"],
        "--extra_inputs": "input_image",
        "--lambda_main": conditioning["lambda_main"],
        "--report_to": logging["report_to"],
        "--wandb_project": logging["wandb_project"],
        "--wandb_name": logging.get(
            "wandb_name",
            f"{config['experiment']['name']}_{output_dir.name}",
        ),
        "--wandb_mode": logging["wandb_mode"],
    }
    if initialization_type == "openvid_lora":
        options["--lora_checkpoint"] = paths["pretrained_lora_checkpoint"]
    else:
        options["--physrvg_dit_checkpoint"] = paths["physrvg_dit_checkpoint"]
    if xssc_backend == "dinov3_movic" or xssc_loss_enabled:
        options.update(
            {
                "--self_attn_adaptation_mode": adaptation["mode"],
                "--pretrained_lora_expected_modules": model[
                    "pretrained_lora_expected_modules"
                ],
                "--self_attn_expected_num_blocks": model[
                    "self_attn_expected_num_blocks"
                ],
                "--self_attn_expected_num_heads": model[
                    "self_attn_expected_num_heads"
                ],
                "--self_attn_lora_rank": adaptation["self_attn_lora_rank"],
                "--self_attn_lora_alpha": adaptation["self_attn_lora_alpha"],
                "--self_attn_lora_dropout": adaptation[
                    "self_attn_lora_dropout"
                ],
                "--experiment_seed": optim["seed"],
            }
        )
    if vjepa_loss_enabled:
        options.update(
            {
                "--vjepa_loss_weight": vjepa_loss["weight"],
                "--vjepa_sigma_min": vjepa_loss["sigma_min"],
                "--vjepa_sigma_max": vjepa_loss["sigma_max"],
                "--vjepa_every_n_forwards": vjepa_loss["every_n_forwards"],
                "--vjepa_num_frames": vjepa_loss["num_frames"],
                "--vjepa_input_size": vjepa_loss["input_size"],
                "--vjepa_repo": vjepa_loss["vjepa_repo"],
                "--vjepa_checkpoint": vjepa_loss["vjepa_checkpoint"],
                "--tiny_vae_root": vjepa_loss["tiny_vae_root"],
                "--tiny_vae_checkpoint": vjepa_loss["tiny_vae_checkpoint"],
                "--vjepa_range_penalty_weight": vjepa_loss[
                    "range_penalty_weight"
                ],
                "--vjepa_frame_sampling": vjepa_loss["frame_sampling"],
                "--vjepa_local_sampling_probability": vjepa_loss[
                    "local_sampling_probability"
                ],
                "--vjepa_local_context_frames": vjepa_loss[
                    "local_context_frames"
                ],
                "--vjepa_gradient_diagnostics_every_n_forwards": vjepa_loss[
                    "gradient_diagnostics_every_n_forwards"
                ],
            }
        )
    if xssc_loss_enabled:
        options.update(
            {
                "--xssc_loss_backend": xssc_loss["backend"],
                "--xssc_loss_weight": xssc_loss["weight"],
                "--xssc_loss_future_start_frame": xssc_loss[
                    "future_start_frame"
                ],
                "--xssc_loss_backbone_chunk_size": xssc_loss[
                    "backbone_chunk_size"
                ],
                "--xssc_loss_gradient_diagnostics_every_n_forwards": xssc_loss[
                    "gradient_diagnostics_every_n_forwards"
                ],
                "--xssc_root": paths["xssc_root"],
                "--xssc_config": paths["xssc_config"],
                "--xssc_checkpoint": paths["xssc_checkpoint"],
                "--xssc_input_size": model["xssc_input_size"],
                "--xssc_max_time_steps": model["xssc_max_time_steps"],
            }
        )
        if xssc_backend == "dinov3_movic":
            options.update(
                {
                    "--dinov3_root": paths["dinov3_root"],
                    "--dinov3_checkpoint": paths["dinov3_checkpoint"],
                    "--xssc_sam2_config": paths["sam2_config"],
                    "--xssc_sam2_checkpoint": paths["sam2_checkpoint"],
                    "--xssc_box_cache_dir": paths["xssc_box_cache_dir"],
                }
            )
    if not adaptation["enable_object_branch"]:
        options["--disable_object_branch"] = None
    else:
        options.update(
            {
                "--xssc_root": paths["xssc_root"],
                "--xssc_config": paths["xssc_config"],
                "--xssc_checkpoint": paths["xssc_checkpoint"],
                "--xssc_input_size": model["xssc_input_size"],
                "--xssc_max_time_steps": model["xssc_max_time_steps"],
                "--object_lora_rank": adaptation["object_lora_rank"],
                "--object_lora_alpha": adaptation["object_lora_alpha"],
                "--object_lora_dropout": adaptation["object_lora_dropout"],
                "--xssc_slot_track_dropout": conditioning["slot_track_dropout"],
                "--object_gate_init": adaptation["object_gate_init"],
                "--lambda_object_context_reg": conditioning[
                    "lambda_object_context_reg"
                ],
            }
        )
        if xssc_backend == "dinov3_movic":
            options.update(
                {
                "--dinov3_root": paths["dinov3_root"],
                "--dinov3_checkpoint": paths["dinov3_checkpoint"],
                "--xssc_sam2_config": paths["sam2_config"],
                "--xssc_sam2_checkpoint": paths["sam2_checkpoint"],
                "--xssc_box_source": conditioning["xssc_box_source"],
                "--xssc_box_cache_dir": paths["xssc_box_cache_dir"],
                "--xssc_empty_amg_max_resample_attempts": conditioning[
                    "empty_amg_max_resample_attempts"
                ],
                }
            )
    if adaptation["mode"] in HEAD_SELECTIVE_MODES:
        options.update(
            {
                "--head_selection_config": paths["head_selection_config"],
                "--head_selection_subset_id": adaptation[
                    "head_selection_subset_id"
                ],
                "--head_selection_expected_role": adaptation[
                    "head_selection_expected_role"
                ],
                "--head_selection_feature_subtype": adaptation[
                    "head_selection_feature_subtype"
                ],
                "--head_selection_expected_num_heads": adaptation[
                    "head_selection_expected_num_heads"
                ],
            }
        )
    amg_option_names = {
        "max_selected": "--xssc_amg_max_selected",
        "min_area_ratio": "--xssc_amg_min_area_ratio",
        "max_area_ratio": "--xssc_amg_max_area_ratio",
        "min_bbox_side": "--xssc_amg_min_bbox_side",
        "background_area_ratio": "--xssc_amg_background_area_ratio",
        "background_span_ratio": "--xssc_amg_background_span_ratio",
        "border_area_ratio": "--xssc_amg_border_area_ratio",
        "border_occupancy_ratio": "--xssc_amg_border_occupancy_ratio",
        "opposite_edge_area_ratio": "--xssc_amg_opposite_edge_area_ratio",
        "shadow_min_area_ratio": "--xssc_amg_shadow_min_area_ratio",
        "shadow_max_luminance_ratio": "--xssc_amg_shadow_max_luminance_ratio",
        "shadow_max_chromaticity_distance": "--xssc_amg_shadow_max_chromaticity_distance",
        "shadow_max_gradient_mean": "--xssc_amg_shadow_max_gradient_mean",
        "duplicate_iou": "--xssc_amg_duplicate_iou",
        "duplicate_containment": "--xssc_amg_duplicate_containment",
    }
    if (
        adaptation["enable_object_branch"] or xssc_loss_enabled
    ) and xssc_backend == "dinov3_movic":
        amg_filters = conditioning["amg_filters"]
        missing_amg = sorted(set(amg_option_names) - set(amg_filters))
        if missing_amg:
            raise KeyError(f"Missing conditioning.amg_filters values: {missing_amg}")
        for config_name, option_name in amg_option_names.items():
            options[option_name] = amg_filters[config_name]
    for name, value in options.items():
        if value is None:
            command.append(name)
        else:
            add_option(command, name, value)

    if (
        adaptation["enable_object_branch"]
        and xssc_backend == "dinov3_movic"
        and conditioning["filter_empty_amg"]
    ):
        command.append("--xssc_filter_empty_amg")
    if optim["fail_on_nonfinite_train_values"]:
        command.append("--fail_on_nonfinite_train_values")
    if vjepa_loss_enabled and vjepa_loss["tiny_vae_parallel"]:
        command.append("--tiny_vae_parallel")
    if adaptation["enable_object_branch"] and logging["debug_print_object_regularization"]:
        command.append("--debug_print_object_regularization")
    if checkpointing.get("resume_from"):
        add_option(command, "--stage2_resume_from", checkpointing["resume_from"])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Output run tag; defaults to the current UTC timestamp.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, sources = load_config(args.config)
    config = validate_config(raw, args.config.expanduser().resolve().parent)
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(config["paths"]["output_root"])
        / config["experiment"]["name"]
        / run_tag
    )
    head_selection_snapshot = None
    if not (args.validate_only or args.dry_run):
        output_dir.mkdir(parents=True, exist_ok=False)
        head_selection_snapshot = snapshot_head_selection_config(
            config,
            output_dir,
        )
    command = build_command(config, output_dir)
    effective_batch = (
        int(config["launch"]["num_processes"])
        * int(config["optimization"]["train_batch_size_per_gpu"])
        * int(config["optimization"]["gradient_accumulation_steps"])
    )
    summary = {
        "experiment": config["experiment"]["name"],
        "mode": config["adaptation"]["mode"],
        "xssc_backend": config["model"]["xssc_backend"],
        "enable_object_branch": config["adaptation"]["enable_object_branch"],
        "initialization": config["initialization"]["type"],
        "vjepa_loss_enabled": bool(
            config.get("vjepa_loss", {}).get("enabled", False)
        ),
        "xssc_loss_enabled": bool(
            config.get("xssc_loss", {}).get("enabled", False)
        ),
        "config_sources": sources,
        "gpu_set": config["launch"]["gpu_set"],
        "effective_batch": effective_batch,
        "output_dir": str(output_dir),
        "command": shlex.join(command),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    if args.validate_only or args.dry_run:
        return

    cache_root = Path(config["paths"]["cache_root"])
    cache_dirs = {
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    required_cache_dirs = list(cache_dirs.values())
    if (
        config["adaptation"]["enable_object_branch"]
        or bool(config.get("xssc_loss", {}).get("enabled", False))
    ):
        required_cache_dirs.append(Path(config["paths"]["xssc_box_cache_dir"]))
    for path in required_cache_dirs:
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sources": sources,
        "resolved_config": config,
        "head_selection_snapshot": head_selection_snapshot,
        "launch_summary": summary,
    }
    (output_dir / "resolved_experiment_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "launch_command.txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update({key: str(value) for key, value in cache_dirs.items()})
    # Keep the training environment isolated from incompatible packages in ~/.local.
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(config["launch"]["gpu_set"])
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if config["logging"].get("wandb_run_id"):
        env["WANDB_RUN_ID"] = str(config["logging"]["wandb_run_id"])
        env["WANDB_RESUME"] = str(config["logging"]["wandb_resume"])
    env["PYTHONPATH"] = os.pathsep.join(
        [
            config["paths"]["project_root"],
            config["paths"]["diffsynth_root"],
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
