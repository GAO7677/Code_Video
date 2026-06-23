from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from tqdm import tqdm


def _first_nonfinite_named_tensor(named_tensors: list[tuple[str, torch.Tensor]]) -> dict[str, object] | None:
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        tensor_f = tensor.detach().float()
        if tensor_f.numel() == 0:
            continue
        finite_mask = torch.isfinite(tensor_f)
        if bool(finite_mask.all()):
            continue
        return {
            "name": name,
            "shape": list(tensor.shape),
            "bad_count": int((~finite_mask).sum().item()),
            "has_nan": bool(torch.isnan(tensor_f).any().item()),
            "has_posinf": bool(torch.isposinf(tensor_f).any().item()),
            "has_neginf": bool(torch.isneginf(tensor_f).any().item()),
        }
    return None


def _max_abs_named_tensor(named_tensors: list[tuple[str, torch.Tensor]]) -> float:
    max_abs = 0.0
    for _, tensor in named_tensors:
        if tensor is None:
            continue
        tensor_f = tensor.detach().float()
        if tensor_f.numel() == 0:
            continue
        finite_mask = torch.isfinite(tensor_f)
        if not bool(finite_mask.any()):
            continue
        value = float(tensor_f[finite_mask].abs().max().item())
        if value > max_abs:
            max_abs = value
    return max_abs


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


def launch_training_task(
    accelerator: Accelerator,
    model: torch.nn.Module,
    *,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    num_workers: int,
    save_every: int,
    max_steps: int,
    grad_accum_steps: int,
    max_grad_norm: float | None,
    resume_checkpoint: str | Path | None = None,
) -> None:
    debug_runner = os.environ.get("CODEX_DEBUG_RUNNER_INIT", "").strip() not in {"", "0", "false", "False"}
    runner_t0 = time.perf_counter()

    def _debug_log(message: str) -> None:
        if debug_runner:
            elapsed = time.perf_counter() - runner_t0
            rank = int(os.environ.get("RANK", "0"))
            print(f"[runner +{elapsed:.2f}s rank={rank}] {message}", flush=True)

    base_model = accelerator.unwrap_model(model)
    cfg = base_model.cfg
    _debug_log("build optimizer start")
    optimizer = _build_optimizer(
        optimizer_type,
        model.trainable_parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    _debug_log("build optimizer done")
    _debug_log("build dataloader start")
    dataloader = model.build_dataloader(num_workers=num_workers)
    _debug_log("build dataloader done")
    _debug_log(f"model.to start device={accelerator.device}")
    model.to(device=accelerator.device)
    _debug_log("model.to done")
    _debug_log("accelerator.prepare start")
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    _debug_log("accelerator.prepare done")

    step = 0
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"unsupported resume checkpoint format in {resume_path}")
        if "model" not in state:
            raise RuntimeError(f"resume checkpoint missing 'model' key: {resume_path}")
        model_state = state["model"]
        if not isinstance(model_state, dict):
            raise RuntimeError(f"resume checkpoint 'model' entry must be a dict: {resume_path}")
        unwrapped = accelerator.unwrap_model(model)
        missing = unwrapped.load_state_dict(model_state, strict=False)
        if accelerator.is_main_process:
            print(
                f"resumed model weights from {resume_path}; "
                f"missing_keys={len(missing.missing_keys)} unexpected_keys={len(missing.unexpected_keys)}",
                flush=True,
            )
        step = int(state.get("step", 0))
    ckpt_dir = Path(cfg["experiment"]["output_dir"])
    _debug_log(f"mkdir checkpoint dir start path={ckpt_dir}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _debug_log("mkdir checkpoint dir done")
    log_cfg = cfg.get("logging", {})
    log_every = int(log_cfg.get("log_every", 10))
    save_every = int(log_cfg.get("save_every", save_every))
    use_wandb = bool(log_cfg.get("use_wandb", False))
    nonfinite_probe = bool(log_cfg.get("nonfinite_probe", False))
    nonfinite_probe_every = int(log_cfg.get("nonfinite_probe_every", 1))
    wandb_run = None
    if use_wandb and accelerator.is_main_process:
        import wandb

        wandb_dir = Path(log_cfg.get("wandb_dir", ckpt_dir.parent / "wandb"))
        _debug_log(f"wandb init start dir={wandb_dir}")
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=str(log_cfg.get("wandb_project", "vjepa-vggt-wan")),
            entity=log_cfg.get("wandb_entity"),
            name=str(log_cfg.get("wandb_run_name", cfg["experiment"]["name"])),
            dir=str(wandb_dir),
            config=json.loads(json.dumps(cfg)),
            resume="allow",
        )
        _debug_log("wandb init done")

    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    optimizer.zero_grad(set_to_none=True)
    saw_first_batch = False
    while step < max_steps:
        for batch in dataloader:
            if not saw_first_batch:
                _debug_log("first batch fetched")
                saw_first_batch = True
            with accelerator.accumulate(model):
                if step == 0:
                    _debug_log("first forward start")
                loss = model(batch)
                if step == 0:
                    _debug_log("first forward done")
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss detected at step={step + 1}: "
                        f"{float(loss.detach().cpu().item())}"
                    )
                if step == 0:
                    _debug_log("first backward start")
                accelerator.backward(loss)
                if step == 0:
                    _debug_log("first backward done")
                if accelerator.sync_gradients:
                    unwrapped = accelerator.unwrap_model(model)
                    trainable_named_params = [
                        (name, param)
                        for name, param in unwrapped.named_parameters()
                        if param.requires_grad
                    ]
                    if nonfinite_probe and step % max(1, nonfinite_probe_every) == 0:
                        grad_info = _first_nonfinite_named_tensor(
                            [(name, param.grad) for name, param in trainable_named_params]
                        )
                        if grad_info is not None:
                            raise RuntimeError(
                                "non-finite gradient detected before optimizer.step; "
                                f"step={step + 1}, param={grad_info['name']}, "
                                f"shape={grad_info['shape']}, bad_count={grad_info['bad_count']}, "
                                f"has_nan={grad_info['has_nan']}, has_posinf={grad_info['has_posinf']}, "
                                f"has_neginf={grad_info['has_neginf']}"
                            )
                    if max_grad_norm is not None and max_grad_norm > 0:
                        accelerator.clip_grad_norm_(unwrapped.trainable_parameters(), max_grad_norm)
                    optimizer.step()
                    if step == 0:
                        _debug_log("first optimizer.step done")
                    if nonfinite_probe and step % max(1, nonfinite_probe_every) == 0:
                        param_info = _first_nonfinite_named_tensor(trainable_named_params)
                        if param_info is not None:
                            if accelerator.is_local_main_process:
                                state = {
                                    "step": step + 1,
                                    "model": unwrapped.export_trainable_state_dict(),
                                }
                                torch.save(state, ckpt_dir / f"step_{step + 1:07d}_nonfinite.pt")
                            raise RuntimeError(
                                "non-finite trainable parameter detected after optimizer.step; "
                                f"step={step + 1}, param={param_info['name']}, "
                                f"shape={param_info['shape']}, bad_count={param_info['bad_count']}, "
                                f"has_nan={param_info['has_nan']}, has_posinf={param_info['has_posinf']}, "
                                f"has_neginf={param_info['has_neginf']}"
                            )
                    optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                progress.update(1)
                loss_value = float(loss.detach().item())
                unwrapped = accelerator.unwrap_model(model)
                trainable_named_params = [
                    (name, param)
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                ]
                trainable_param_abs_max = _max_abs_named_tensor(trainable_named_params)
                progress.set_postfix(loss=f"{loss_value:.4f}", pmax=f"{trainable_param_abs_max:.4f}")
                if accelerator.is_main_process and wandb_run is not None:
                    extra_metrics = {}
                    if hasattr(unwrapped, "last_train_metrics") and isinstance(unwrapped.last_train_metrics, dict):
                        extra_metrics = {
                            key: float(value)
                            for key, value in unwrapped.last_train_metrics.items()
                        }
                    wandb_run.log(
                        {
                            "train/loss": loss_value,
                            "train/step": step,
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/loss_is_finite": float(torch.isfinite(loss.detach()).item()),
                            "train/trainable_param_abs_max": float(trainable_param_abs_max),
                            **extra_metrics,
                        },
                        step=step,
                    )
                if accelerator.is_local_main_process and step % max(1, save_every) == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    state = {
                        "step": step,
                        "model": unwrapped.export_trainable_state_dict(),
                    }
                    torch.save(state, ckpt_dir / f"step_{step:07d}.pt")
                if step >= max_steps:
                    break
    progress.close()
    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()
