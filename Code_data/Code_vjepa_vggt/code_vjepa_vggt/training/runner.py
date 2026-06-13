from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm


def launch_training_task(
    accelerator: Accelerator,
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    save_every: int,
    max_steps: int,
    grad_accum_steps: int,
    max_grad_norm: float | None,
) -> None:
    base_model = accelerator.unwrap_model(model)
    cfg = base_model.cfg
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=learning_rate, weight_decay=weight_decay)
    dataloader = model.build_dataloader(num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    step = 0
    ckpt_dir = Path(cfg["experiment"]["output_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_cfg = cfg.get("logging", {})
    log_every = int(log_cfg.get("log_every", 10))
    use_wandb = bool(log_cfg.get("use_wandb", False))
    wandb_run = None
    if use_wandb and accelerator.is_main_process:
        import wandb

        wandb_dir = Path(log_cfg.get("wandb_dir", ckpt_dir.parent / "wandb"))
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=str(log_cfg.get("wandb_project", "vjepa-vggt-wan")),
            entity=log_cfg.get("wandb_entity"),
            name=str(log_cfg.get("wandb_run_name", cfg["experiment"]["name"])),
            dir=str(wandb_dir),
            config=json.loads(json.dumps(cfg)),
            resume="allow",
        )

    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    while step < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                loss = model(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss detected at step={step + 1}: "
                        f"{float(loss.detach().cpu().item())}"
                    )
                accelerator.backward(loss)
                if max_grad_norm is not None and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(accelerator.unwrap_model(model).trainable_parameters(), max_grad_norm)
                optimizer.step()
            step += 1
            progress.update(1)
            if step % max(1, log_every) == 0:
                loss_value = float(loss.detach().item())
                progress.set_postfix(loss=f"{loss_value:.4f}")
                if accelerator.is_main_process and wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss_value,
                            "train/step": step,
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/loss_is_finite": float(torch.isfinite(loss.detach()).item()),
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
