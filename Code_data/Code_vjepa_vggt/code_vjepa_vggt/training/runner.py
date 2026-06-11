from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator
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
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=learning_rate, weight_decay=weight_decay)
    dataloader = model.build_dataloader(num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    step = 0
    ckpt_dir = Path(model.cfg["experiment"]["output_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    while step < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                loss = model(batch)
                accelerator.backward(loss)
                if max_grad_norm is not None and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.trainable_parameters(), max_grad_norm)
                optimizer.step()
            step += 1
            progress.update(1)
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
