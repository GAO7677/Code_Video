from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import accelerate
import torch
from tqdm import tqdm

from diffsynth.diffusion import ModelLogger

from code_vjepa_vggt.object_token_teacher_student.runtime_stage1b_context_only import (
    ContextOnlyInjectionTrainer,
)
from code_vjepa_vggt.train_v_newtrain import (
    _collect_trainable_grad_stats,
    format_step_tag,
    get_checkpoint_dir,
    initialize_deepspeed_gradient_checkpointing,
    load_training_state,
    resolve_lora_checkpoint_for_resume,
    resolve_resume_state_path,
    restore_rng_state,
    save_training_checkpoint_bundle,
)
from code_vjepa_vggt.utils.config import load_yaml_config


DEFAULT_CONFIG = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "object_token_teacher_student/config_stage1b_context_only_template.yaml"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_JEPA_CKPT = Path("/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth")
DEFAULT_COTRACKER_CKPT = Path("/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
DEFAULT_VGGT_ROOT = Path("/data/gaoya/ckpt/facebook-VGGT-1B")


def _normalize_checkpoint_key(key: str) -> str:
    normalized = str(key)
    prefixes = (
        "module.",
        "base_model.model.",
        "dit.base_model.model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def _resolve_checkpoint_file(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path.is_dir():
        candidate = checkpoint_path / "checkpoint.safetensors"
        if candidate.is_file():
            return candidate
        candidates = sorted(checkpoint_path.rglob("checkpoint.safetensors"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")


def _load_trainable_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    resolved = _resolve_checkpoint_file(checkpoint_path)
    if resolved.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file

        return load_safetensors_file(str(resolved), device="cpu")
    state = torch.load(resolved, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
        return state
    raise RuntimeError(f"unsupported checkpoint format: {resolved}")


def _maybe_normalize_jepa_root(raw_path: str | Path | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser().resolve()
    if path.name == "model.pth" and path.parent.name == "original":
        return str(path.parent.parent)
    return str(path)


def _load_matching_state_into_model(
    model: ContextOnlyInjectionTrainer,
    checkpoint_path: str | Path,
    *,
    include_prefixes: tuple[str, ...] | None = None,
    include_substrings: tuple[str, ...] | None = None,
) -> dict[str, object]:
    state_dict = _load_trainable_state(checkpoint_path)

    latent_key = None
    for candidate in ("object_pooler.latent_proj.weight", "bundle.object_pooler.latent_proj.weight"):
        if candidate in state_dict:
            latent_key = candidate
            break
    if latent_key is not None and hasattr(model.object_pooler, "_ensure_latent_proj"):
        latent_dim = int(state_dict[latent_key].shape[1])
        model.object_pooler._ensure_latent_proj(latent_dim, model.device_obj)

    include_prefixes = tuple(include_prefixes or ())
    include_substrings = tuple(include_substrings or ())
    if include_prefixes or include_substrings:
        filtered_source_state = {}
        for key, value in state_dict.items():
            key_str = str(key)
            if include_prefixes and any(key_str.startswith(prefix) for prefix in include_prefixes):
                filtered_source_state[key_str] = value
                continue
            if include_substrings and any(token in key_str for token in include_substrings):
                filtered_source_state[key_str] = value
    else:
        filtered_source_state = {str(key): value for key, value in state_dict.items()}

    if not filtered_source_state:
        return {
            "loaded_count": 0,
            "missing_keys": [],
            "unexpected_keys": [],
            "skipped_shape_mismatch": [],
            "selected_source_keys": 0,
        }

    model_state = model.state_dict()
    normalized_model_keys = {_normalize_checkpoint_key(key): key for key in model_state.keys()}
    normalized_checkpoint_keys = {_normalize_checkpoint_key(key): key for key in filtered_source_state.keys()}
    overlapping = sorted(set(normalized_model_keys.keys()) & set(normalized_checkpoint_keys.keys()))
    if not overlapping:
        return {
            "loaded_count": 0,
            "missing_keys": [],
            "unexpected_keys": [],
            "skipped_shape_mismatch": [],
            "selected_source_keys": len(filtered_source_state),
        }

    filtered_state = {}
    skipped_shape_mismatch = []
    for norm_key in overlapping:
        model_key = normalized_model_keys[norm_key]
        ckpt_key = normalized_checkpoint_keys[norm_key]
        model_value = model_state[model_key]
        ckpt_value = filtered_source_state[ckpt_key]
        if tuple(model_value.shape) != tuple(ckpt_value.shape):
            skipped_shape_mismatch.append(
                {
                    "model_key": model_key,
                    "checkpoint_key": ckpt_key,
                    "model_shape": list(model_value.shape),
                    "checkpoint_shape": list(ckpt_value.shape),
                }
            )
            continue
        filtered_state[model_key] = ckpt_value
    missing = model.load_state_dict(filtered_state, strict=False)
    return {
        "loaded_count": len(filtered_state),
        "missing_keys": list(missing.missing_keys),
        "unexpected_keys": list(missing.unexpected_keys),
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "selected_source_keys": len(filtered_source_state),
    }


def _build_optimizer(
    optimizer_name: str,
    parameters,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
):
    name = optimizer_name.strip().lower()
    if name in {"adamw", "torch_adamw", "torch"}:
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    if name in {"adamw8bit", "8bit_adamw"}:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    if name in {"paged_adamw8bit", "pagedadamw8bit"}:
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    raise ValueError(f"unsupported optimizer_type: {optimizer_name}")


def _build_accelerator(args: argparse.Namespace) -> accelerate.Accelerator:
    log_with = args.report_to if args.report_to != "none" else None
    return accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
        log_with=log_with,
    )


def _init_trackers(accelerator: accelerate.Accelerator, args: argparse.Namespace) -> None:
    if args.report_to == "none":
        return
    if args.wandb_mode is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={
            "wandb": {
                "entity": args.wandb_entity,
                "name": args.wandb_name or os.path.basename(args.output_path.rstrip("/")),
            }
        },
    )


class ContextOnlyDiffSynthModule(ContextOnlyInjectionTrainer):
    def __init__(self, cfg: dict[str, object], device: str | torch.device) -> None:
        super().__init__(cfg, build_optimizer=True, device=device)
        self.enable_object_branch = True

    def trainable_modules(self):
        return self.trainable_parameters()

    def export_trainable_state_dict(self, state_dict=None, remove_prefix=None):
        if state_dict is None:
            state_dict = self.state_dict()
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        filtered = {
            name: tensor
            for name, tensor in state_dict.items()
            if name in trainable_names
        }
        if remove_prefix is not None:
            filtered = {
                (name[len(remove_prefix):] if name.startswith(remove_prefix) else name): tensor
                for name, tensor in filtered.items()
            }
        return filtered


def _base_config(args: argparse.Namespace) -> dict[str, object]:
    cfg = load_yaml_config(Path(args.config).expanduser().resolve())
    if args.output_path is None:
        args.output_path = str(Path(cfg["experiment"]["output_dir"]).expanduser().resolve())
    cfg["experiment"]["name"] = str(args.experiment_name or Path(args.output_path).name)
    cfg["experiment"]["output_dir"] = str(Path(get_checkpoint_dir(args)).expanduser().resolve())
    cfg["experiment"]["seed"] = int(args.seed)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    opt_cfg = cfg["optimization"]
    loss_cfg = cfg["loss"]
    log_cfg = cfg["logging"]

    model_cfg["wan_ckpt_dir"] = str(Path(args.wan_root).expanduser().resolve())
    model_cfg["init_wan_lora_from_checkpoint"] = str(Path(args.lora_checkpoint).expanduser().resolve())
    model_cfg["wan_lora_rank"] = int(args.lora_rank)
    model_cfg["wan_lora_alpha"] = int(args.lora_alpha)
    model_cfg["wan_lora_dropout"] = float(args.lora_dropout)
    model_cfg["wan_lora_init"] = str(args.lora_init)
    model_cfg["sam2_max_objects"] = int(args.aux_max_objects)
    model_cfg["object_num_queries"] = int(args.object_num_queries)
    model_cfg["je_pa_ckpt_dir"] = str(_maybe_normalize_jepa_root(args.jepa_ckpt_path))
    model_cfg["vggt_model_path"] = str(Path(args.vggt_model_path).expanduser().resolve())
    model_cfg["vggt_input_hw"] = [int(args.vggt_input_h), int(args.vggt_input_w)]
    model_cfg["cotracker_checkpoint"] = str(Path(args.cotracker_checkpoint).expanduser().resolve())
    model_cfg["cotracker_input_hw"] = [int(args.cotracker_input_h), int(args.cotracker_input_w)]
    model_cfg["cotracker_window_len"] = int(args.cotracker_window_len)
    model_cfg["jepa_input_size"] = int(args.jepa_input_size)
    model_cfg["jepa_patch_size"] = int(args.jepa_patch_size)
    model_cfg["jepa_tubelet_size"] = int(args.jepa_tubelet_size)
    model_cfg["jepa_window_radius"] = int(args.jepa_window_radius)
    model_cfg["latent_window_radius"] = int(args.latent_window_radius)
    model_cfg["cond_proj_dim"] = int(args.cond_proj_dim)

    data_cfg["root"] = str(Path(args.phys_state_root).expanduser().resolve())
    data_cfg["split"] = str(args.phys_state_split)
    data_cfg["resolution"] = [int(args.height), int(args.width)]
    data_cfg["num_context_frames"] = int(args.fixed_num_context_frames)
    data_cfg["random_context_frames"] = bool(args.random_context_frames)
    data_cfg["batch_size"] = int(args.batch_size)
    data_cfg["num_workers"] = int(args.dataset_num_workers)
    data_cfg["fps"] = int(args.fps)

    opt_cfg["lr"] = float(args.learning_rate)
    opt_cfg["weight_decay"] = float(args.weight_decay)
    opt_cfg["betas"] = [float(args.adam_beta1), float(args.adam_beta2)]
    opt_cfg["eps"] = float(args.adam_epsilon)
    opt_cfg["optimizer_type"] = str(args.optimizer_type)
    opt_cfg["max_steps"] = int(args.max_train_steps)
    opt_cfg["grad_accum_steps"] = int(args.gradient_accumulation_steps)
    opt_cfg["max_grad_norm"] = float(args.max_grad_norm)
    opt_cfg["mixed_precision"] = str(args.mixed_precision)

    loss_cfg["lambda_main"] = float(args.lambda_main)
    loss_cfg["lambda_track_aux"] = float(args.lambda_track_aux)
    loss_cfg["lambda_box_aux"] = float(args.lambda_box_aux)
    loss_cfg["lambda_depth_aux"] = float(args.lambda_depth_aux)

    log_cfg["log_every"] = int(args.log_every_steps)
    log_cfg["save_every"] = int(args.save_steps)
    log_cfg["max_checkpoints"] = int(args.max_checkpoints_keep)
    log_cfg["use_wandb"] = bool(args.report_to == "wandb")
    log_cfg["wandb_project"] = str(args.wandb_project)
    log_cfg["wandb_run_name"] = str(args.wandb_name or cfg["experiment"]["name"])
    if args.wandb_dir is not None:
        log_cfg["wandb_dir"] = str(Path(args.wandb_dir).expanduser().resolve())
    else:
        log_cfg["wandb_dir"] = str((Path(args.output_path).expanduser().resolve() / "wandb"))

    return cfg


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.stage2_resume_from = resolve_resume_state_path(args.stage2_resume_from)
    return args


def _build_model(args: argparse.Namespace, accelerator: accelerate.Accelerator) -> ContextOnlyDiffSynthModule:
    cfg = _base_config(args)
    return ContextOnlyDiffSynthModule(cfg, device=accelerator.device)


def train_loop(
    accelerator: accelerate.Accelerator,
    model: ContextOnlyDiffSynthModule,
    model_logger: ModelLogger,
    args: argparse.Namespace,
) -> dict[str, int]:
    optimizer = _build_optimizer(
        args.optimizer_type,
        model.trainable_modules(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = model.build_dataloader(num_workers=args.dataset_num_workers)

    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)
    optimizer.zero_grad(set_to_none=True)

    start_epoch = 0
    resume_batch_in_epoch = 0
    global_step = 0
    if args.stage2_resume_from is not None:
        resume_payload = load_training_state(args.stage2_resume_from)
        optimizer_state_restored = True
        scheduler_state_restored = True
        try:
            optimizer.load_state_dict(resume_payload["optimizer"])
        except ValueError as exc:
            optimizer_state_restored = False
            accelerator.print(
                "Skipping optimizer state restore because the current trainable parameter groups "
                f"do not match the saved state. Optimizer error: {exc}"
            )
        if optimizer_state_restored:
            try:
                scheduler.load_state_dict(resume_payload["scheduler"])
            except Exception as exc:
                scheduler_state_restored = False
                accelerator.print(
                    "Failed to restore scheduler state; keeping a fresh scheduler instead. "
                    f"Scheduler error: {exc}"
                )
        else:
            scheduler_state_restored = False
        global_step = int(resume_payload.get("global_step", 0))
        start_epoch = int(resume_payload.get("epoch_id", 0))
        resume_batch_in_epoch = int(resume_payload.get("batch_in_epoch", 0))
        model_logger.num_steps = int(resume_payload.get("model_logger_num_steps", global_step))
        restore_rng_state(resume_payload)
        accelerator.wait_for_everyone()
        accelerator.print(
            "Restored training state: "
            f"global_step={global_step}, epoch_id={start_epoch}, batch_in_epoch={resume_batch_in_epoch}, "
            f"model_logger_num_steps={model_logger.num_steps}, "
            f"optimizer_state_restored={optimizer_state_restored}, "
            f"scheduler_state_restored={scheduler_state_restored}"
        )

    progress = {
        "global_step": global_step,
        "epoch_id": start_epoch,
        "batch_in_epoch": resume_batch_in_epoch,
        "model_logger_num_steps": model_logger.num_steps,
    }
    for epoch_id in range(start_epoch, args.num_epochs):
        skip_batches = resume_batch_in_epoch if epoch_id == start_epoch else 0
        progress_bar = tqdm(
            total=len(dataloader),
            initial=skip_batches,
            disable=not accelerator.is_local_main_process,
            desc=f"epoch {epoch_id} | global_step {global_step}",
        )
        if skip_batches > 0:
            accelerator.print(
                f"Resuming epoch {epoch_id}: skipping the first {skip_batches} batches before continuing training."
            )
        for batch_index, batch in enumerate(dataloader):
            if batch_index < skip_batches:
                if accelerator.is_local_main_process:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model):
                loss = model(batch)
                accelerator.backward(loss)
                grad_stats = {}
                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(
                            accelerator.unwrap_model(model).trainable_parameters(),
                            args.max_grad_norm,
                        )
                    grad_stats = _collect_trainable_grad_stats(accelerator.unwrap_model(model))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    model_logger.num_steps = global_step
                    metrics = {
                        "train/loss": loss.detach().float().item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch_id,
                    }
                    extra_metrics = getattr(accelerator.unwrap_model(model), "last_train_metrics", {})
                    metrics.update(extra_metrics)
                    metrics.update(grad_stats)
                    accelerator.log(metrics, step=global_step)

                progress["global_step"] = global_step
                progress["epoch_id"] = epoch_id
                progress["batch_in_epoch"] = batch_index + 1
                progress["model_logger_num_steps"] = model_logger.num_steps

                if accelerator.is_local_main_process:
                    progress_bar.set_description(f"epoch {epoch_id} | global_step {global_step}")
                    progress_bar.set_postfix(model_step=model_logger.num_steps, refresh=False)

                if (
                    accelerator.sync_gradients
                    and args.save_steps is not None
                    and model_logger.num_steps > 0
                    and model_logger.num_steps % args.save_steps == 0
                ):
                    checkpoint_tag = format_step_tag(model_logger.num_steps)
                    save_training_checkpoint_bundle(
                        accelerator=accelerator,
                        model=model,
                        model_logger=model_logger,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch_id=epoch_id,
                        batch_in_epoch=batch_index + 1,
                        checkpoint_root=get_checkpoint_dir(args),
                        checkpoint_tag=checkpoint_tag,
                        max_checkpoints_keep=args.max_checkpoints_keep,
                    )
            progress_bar.update(1)
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break
        progress_bar.close()

        accelerator.log({"train/epoch_end": epoch_id}, step=global_step)
        progress["global_step"] = global_step
        progress["epoch_id"] = epoch_id + 1
        progress["batch_in_epoch"] = 0
        progress["model_logger_num_steps"] = model_logger.num_steps
        resume_batch_in_epoch = 0
        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    if (
        args.save_steps is not None
        and model_logger.num_steps > 0
        and model_logger.num_steps % args.save_steps != 0
    ):
        checkpoint_tag = format_step_tag(model_logger.num_steps)
        save_training_checkpoint_bundle(
            accelerator=accelerator,
            model=model,
            model_logger=model_logger,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch_id=progress["epoch_id"],
            batch_in_epoch=progress["batch_in_epoch"],
            checkpoint_root=get_checkpoint_dir(args),
            checkpoint_tag=checkpoint_tag,
            max_checkpoints_keep=args.max_checkpoints_keep,
        )
    return progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Stage1B context-only branch with DiffSynth/v_newtrain-style argparse/checkpoint framework."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--checkpoint_output_subdir", default="checkpoints")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wan_root", default=str(DEFAULT_WAN_ROOT))
    parser.add_argument("--lora_checkpoint", default=None)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_init", default="gaussian")

    parser.add_argument("--phys_state_root", required=True)
    parser.add_argument("--phys_state_split", default="train")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fixed_num_context_frames", type=int, default=8)
    parser.add_argument("--random_context_frames", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_num_workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--object_num_queries", type=int, default=8)
    parser.add_argument("--aux_max_objects", type=int, default=4)
    parser.add_argument("--jepa_ckpt_path", default=str(DEFAULT_JEPA_CKPT))
    parser.add_argument("--jepa_input_size", type=int, default=384)
    parser.add_argument("--jepa_patch_size", type=int, default=16)
    parser.add_argument("--jepa_tubelet_size", type=int, default=2)
    parser.add_argument("--jepa_window_radius", type=int, default=1)
    parser.add_argument("--latent_window_radius", type=int, default=1)
    parser.add_argument("--vggt_model_path", default=str(DEFAULT_VGGT_ROOT))
    parser.add_argument("--vggt_input_h", type=int, default=420)
    parser.add_argument("--vggt_input_w", type=int, default=728)
    parser.add_argument("--cotracker_checkpoint", default=str(DEFAULT_COTRACKER_CKPT))
    parser.add_argument("--cotracker_input_h", type=int, default=384)
    parser.add_argument("--cotracker_input_w", type=int, default=512)
    parser.add_argument("--cotracker_window_len", type=int, default=60)
    parser.add_argument("--cond_proj_dim", type=int, default=4096)

    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1.0e-8)
    parser.add_argument("--optimizer_type", default="paged_adamw8bit")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_checkpoints_keep", type=int, default=10)
    parser.add_argument("--log_every_steps", type=int, default=10)

    parser.add_argument("--lambda_main", type=float, default=1.0)
    parser.add_argument("--lambda_track_aux", type=float, default=0.0)
    parser.add_argument("--lambda_box_aux", type=float, default=0.0)
    parser.add_argument("--lambda_depth_aux", type=float, default=0.0)

    parser.add_argument("--report_to", choices=["none", "wandb"], default="wandb")
    parser.add_argument("--wandb_project", default="vjepa_vggt_wan")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", default="online")
    parser.add_argument("--wandb_dir", default=None)

    parser.add_argument(
        "--head_resume_from",
        "--init_from",
        dest="head_resume_from",
        default=None,
        help="Load an init checkpoint before training starts. Supports old .pt or new checkpoint.safetensors.",
    )
    parser.add_argument(
        "--stage2_resume_from",
        "--resume_from",
        dest="stage2_resume_from",
        default=None,
        help="Resume from a DiffSynth training_state.pt / checkpoint dir / checkpoint.safetensors.",
    )
    parser.add_argument("--remove_prefix_in_ckpt", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = _prepare_args(parser.parse_args())

    template_cfg = load_yaml_config(Path(args.config).expanduser().resolve())
    if args.lora_checkpoint is None:
        args.lora_checkpoint = str(template_cfg["model"]["init_wan_lora_from_checkpoint"])
    if args.output_path is None:
        args.output_path = str(Path(template_cfg["experiment"]["output_dir"]).expanduser().resolve())

    accelerator = _build_accelerator(args)
    _init_trackers(accelerator, args)

    model = _build_model(args, accelerator)
    resolved_cfg = _base_config(args)
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "resolved_stage1b_context_only_config.json").write_text(
        json.dumps(resolved_cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.head_resume_from is not None:
        init_info = _load_matching_state_into_model(model, args.head_resume_from)
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded init checkpoint: "
                f"loaded_count={init_info['loaded_count']}, "
                f"selected_source_keys={init_info['selected_source_keys']}, "
                f"shape_mismatch={len(init_info['skipped_shape_mismatch'])}"
            )

    if args.stage2_resume_from is not None:
        resume_ckpt = resolve_lora_checkpoint_for_resume(args.stage2_resume_from)
        resume_info = _load_matching_state_into_model(model, resume_ckpt)
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded resume checkpoint weights: "
                f"loaded_count={resume_info['loaded_count']}, "
                f"selected_source_keys={resume_info['selected_source_keys']}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )

    model_logger = ModelLogger(
        output_path=get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    progress = train_loop(
        accelerator=accelerator,
        model=model,
        model_logger=model_logger,
        args=args,
    )
    if accelerator.is_main_process:
        (output_path / "train_summary.json").write_text(
            json.dumps({"progress": progress}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
