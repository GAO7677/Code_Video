#!/usr/bin/env python3
"""
Stage1b Context-Only Training Script (DiffSynth-Native Architecture)

Key differences from original training script:
1. Uses WanVideoPipeline from DiffSynth-Studio-main directly
2. Monkey-patches DiTBlock to add object_cross_attn support
3. No bootstrap.py WanContextVideoModel wrapper
4. Ensures trained weights are compatible with DiffSynth inference pipeline
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import yaml
from tqdm import tqdm
from safetensors.torch import save_file, load_file

# Add DiffSynth-Studio-main to path
DIFFSYNTH_PATH = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
if str(DIFFSYNTH_PATH) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_PATH))

# Add code_vjepa_vggt to path
VJEPA_PATH = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
if str(VJEPA_PATH) not in sys.path:
    sys.path.insert(0, str(VJEPA_PATH))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--gpu", type=int, default=7, help="GPU device ID")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_pipeline(config: dict, device: torch.device):
    """Initialize WanVideoPipeline (without object branch - injected after LoRA)."""
    from diffsynth.pipelines.wan_video import WanVideoPipeline
    from diffsynth.core import ModelConfig
    from code_vjepa_vggt.training.flow_match import WanFlowMatchScheduler

    print(f"Loading WanVideoPipeline from {config['model']['wan_ckpt_dir']}...")

    # Load pretrained Wan2.2 pipeline
    wan_ckpt = config['model']['wan_ckpt_dir']
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=[
                wan_ckpt + "/diffusion_pytorch_model-00001-of-00003.safetensors",
                wan_ckpt + "/diffusion_pytorch_model-00002-of-00003.safetensors",
                wan_ckpt + "/diffusion_pytorch_model-00003-of-00003.safetensors",
            ]),
            ModelConfig(path=wan_ckpt + "/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=wan_ckpt + "/Wan2.2_VAE.pth"),
        ],
    )

    # Replace scheduler with WanFlowMatchScheduler
    pipe.scheduler = WanFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    # NOTE: object branch injection happens AFTER LoRA injection in train()
    # to avoid LoRA wrapping object_cross_attn.q/k/v/o

    # Freeze VAE and text encoder
    if config['model']['freeze_vae']:
        for param in pipe.vae.parameters():
            param.requires_grad = False
        print("VAE frozen")

    if config['model']['freeze_text_encoder']:
        for param in pipe.text_encoder.parameters():
            param.requires_grad = False
        print("Text encoder frozen")

    return pipe


def setup_lora(pipe: WanVideoPipeline, config: dict, device: torch.device):
    """Setup LoRA for DiT and load pretrained weights."""
    from peft import LoraConfig, inject_adapter_in_model

    lora_rank = config['model']['wan_lora_rank']
    lora_alpha = config['model']['wan_lora_alpha']

    print(f"Injecting LoRA (rank={lora_rank}, alpha={lora_alpha}) to DiT...")

    # Inject LoRA to DiT (standard injection, object branch will be frozen)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q", "k", "v", "o"],
        lora_dropout=config['model']['wan_lora_dropout'],
        bias="none",
    )
    pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)

    # Load pretrained LoRA weights from stage1a
    if config['model'].get('init_wan_lora_from_checkpoint'):
        ckpt_path = config['model']['init_wan_lora_from_checkpoint']
        print(f"Loading LoRA weights from {ckpt_path}...")

        state_dict = load_file(ckpt_path)

        # Filter LoRA keys
        lora_keys = {k: v for k, v in state_dict.items() if "lora_" in k}
        print(f"Found {len(lora_keys)} LoRA keys in checkpoint")

        # Load with strict=False
        missing, unexpected = pipe.dit.load_state_dict(lora_keys, strict=False)
        print(f"LoRA loaded: {len(lora_keys) - len(missing)} keys matched")

        if config['model']['init_wan_lora_zero_missing']:
            print("Zero-initializing missing LoRA keys")
            # Already handled by LoRA initialization

    # Freeze LoRA if specified
    if config['model']['freeze_wan_lora']:
        for name, param in pipe.dit.named_parameters():
            if "lora_" in name:
                param.requires_grad = False
        print("LoRA frozen")

    # Freeze DiT base parameters
    if config['model']['freeze_wan_dit']:
        for name, param in pipe.dit.named_parameters():
            # Keep trainable: lora_*
            if "lora_" not in name:
                param.requires_grad = False
        print("DiT base parameters frozen (except LoRA)")

    # Note: Object branch is frozen (requires_grad=False) but will be manually saved
    object_params = sum(1 for n, p in pipe.dit.named_parameters() if "object_" in n or "norm4" in n)
    total_trainable = sum(p.numel() for p in pipe.dit.parameters() if p.requires_grad)
    print(f"Object branch parameters (frozen, will save manually): {object_params}")
    print(f"Total trainable DiT parameters: {total_trainable:,}")


def setup_object_modules(config: dict, device: torch.device):
    """Initialize SimpleBoxEncoder for testing."""
    from code_vjepa_vggt.models.simple_box_encoder import SimpleBoxEncoder

    print("Initializing object modules (simplified box encoder for testing)...")

    # SimpleBoxEncoder for architecture testing
    box_encoder = SimpleBoxEncoder(
        box_dim=4,
        hidden_dim=256,
        out_dim=config['model']['cond_proj_dim'],  # 4096
    ).to(device)

    print("Using SimpleBoxEncoder (bypasses ObjectTubeProjector for testing)")

    return box_encoder


def setup_dataloader(config: dict) -> DataLoader:
    """Setup training dataloader."""
    from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset

    print(f"Loading dataset from {config['data']['root']}...")

    dataset = PhysStateEpisodeDataset(
        root=config['data']['root'],
        split=config['data']['split'],
        resolution=tuple(config['data']['resolution']),
        num_context_frames=config['data']['num_context_frames'],
        context_fraction=config['data']['context_fraction'],
        random_context_frames=config['data']['random_context_frames'],
        init_scan_limit=config['data']['init_scan_limit'],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    return dataloader


def setup_optimizer(trainer, config: dict):
    """Setup optimizer for trainable parameters."""
    trainable_params = [p for p in trainer.parameters() if p.requires_grad]

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer_type = config['optimization']['optimizer_type']

    if optimizer_type == "paged_adamw8bit":
        from bitsandbytes.optim import PagedAdamW8bit
        optimizer = PagedAdamW8bit(
            trainable_params,
            lr=config['optimization']['lr'],
            weight_decay=config['optimization']['weight_decay'],
            betas=config['optimization']['betas'],
            eps=config['optimization']['eps'],
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config['optimization']['lr'],
            weight_decay=config['optimization']['weight_decay'],
            betas=config['optimization']['betas'],
            eps=config['optimization']['eps'],
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def save_checkpoint(
    trainer,
    optimizer,
    step: int,
    output_dir: str,
):
    """Save checkpoint to disk (manually extract object branch weights)."""
    ckpt_dir = Path(output_dir) / "checkpoints" / f"step-{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Collect state dict
    state_dict = {}

    # 1. Box encoder (standard requires_grad check)
    for name, param in trainer.box_encoder.named_parameters():
        if param.requires_grad:
            state_dict[f"box_encoder.{name}"] = param.cpu()

    # 2. Object branch weights (manually extract, bypass requires_grad check)
    object_count = 0
    for name, param in trainer.pipe.dit.named_parameters():
        if "object_" in name or "norm4" in name:
            state_dict[f"dit.{name}"] = param.cpu()
            object_count += 1

    print(f"DEBUG: Found {object_count} object branch parameters in dit.named_parameters()")
    if object_count == 0:
        print("DEBUG: First 10 dit parameter names:")
        for i, (n, _) in enumerate(trainer.pipe.dit.named_parameters()):
            if i >= 10:
                break
            print(f"  {n}")

    # Save with safetensors
    save_path = ckpt_dir / "checkpoint.safetensors"
    save_file(state_dict, str(save_path))

    print(f"Checkpoint saved: {save_path} ({len(state_dict)} keys)")


def train(args, config: dict):
    """Main training loop."""
    from code_vjepa_vggt.trainers.diffsynth_context_trainer import DiffSynthContextTrainer

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # Setup output directory
    output_dir = config['experiment']['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup pipeline
    pipe = setup_pipeline(config, device)
    setup_lora(pipe, config, device)

    # Inject object branch AFTER LoRA (so LoRA doesn't wrap object_cross_attn)
    from code_vjepa_vggt.models.diffsynth_object_injection import inject_object_branch_to_dit
    print("Injecting object branch to DiT blocks (after LoRA)...")
    inject_object_branch_to_dit(
        pipe.dit,
        object_cross_attn_dim=config['model']['cond_proj_dim'],
        object_gate_init=0.1,
    )
    # Freeze object branch (saves initialized weights without gradient computation)
    for name, param in pipe.dit.named_parameters():
        if "object_" in name or "norm4" in name:
            param.requires_grad = False
    obj_count = sum(1 for n, p in pipe.dit.named_parameters() if "object_" in n or "norm4" in n)
    print(f"Object branch frozen (will save {obj_count} params manually): {obj_count} params")

    # Setup object modules
    box_encoder = setup_object_modules(config, device)

    # Setup trainer
    trainer = DiffSynthContextTrainer(
        pipe=pipe,
        box_encoder=box_encoder,
        vae_stride_t=4,
        num_context_frames=config['data']['num_context_frames'],
    )

    # Setup optimizer
    optimizer = setup_optimizer(trainer, config)

    # Setup dataloader
    dataloader = setup_dataloader(config)

    # Setup mixed precision
    scaler = GradScaler(enabled=(config['optimization']['mixed_precision'] == 'fp16'))

    # Training loop
    max_steps = config['optimization']['max_steps']
    grad_accum_steps = config['optimization']['grad_accum_steps']
    log_every = config['logging']['log_every']
    save_every = config['logging']['save_every']

    global_step = 0
    trainer.train()

    pbar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        for batch in dataloader:
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = trainer.forward(batch, cfg_dropout_prob=0.0)

            # Backward pass
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            # Optimizer step (every grad_accum_steps)
            if (global_step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainer.parameters(),
                    config['optimization']['max_grad_norm']
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Logging
            if global_step % log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

            # Checkpointing
            if global_step % save_every == 0 and global_step > 0:
                save_checkpoint(trainer, optimizer, global_step, output_dir)

            global_step += 1
            pbar.update(1)

            if global_step >= max_steps:
                break

    # Final checkpoint
    save_checkpoint(trainer, optimizer, global_step, output_dir)
    pbar.close()

    print(f"Training completed: {global_step} steps")


def main():
    args = parse_args()
    config = load_config(args.config)

    print("=" * 80)
    print("Stage1b Context-Only Training (DiffSynth-Native)")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"GPU: {args.gpu}")
    print(f"Output: {config['experiment']['output_dir']}")
    print("=" * 80)

    train(args, config)


if __name__ == "__main__":
    main()
