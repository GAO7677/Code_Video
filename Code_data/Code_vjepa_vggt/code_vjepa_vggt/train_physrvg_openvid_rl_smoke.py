from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from safetensors.torch import save_file


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_OPENVID_LORA = Path(
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/"
    "openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors"
)
DEFAULT_DIFFSYNTH_ROOT = Path(
    "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main"
)
DEFAULT_OUTPUT_BASE = Path("/data/gaoya/agent-data/outputs/physrvg-openvid-rl")
LORA_TARGET_MODULES = "q,k,v,o,ffn.0,ffn.2"
NUM_CONTEXT_FRAMES = 8
NUM_CLEAN_PREFIX_LATENTS = 2


@dataclass
class RolloutTransition:
    generation_id: int
    timestep_index: int
    timestep: float
    sigma: float
    sigma_next: float
    latents: torch.Tensor
    next_latents: torch.Tensor
    old_log_prob: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-update PhysRVG RL smoke adaptation for DiffSynth Wan2.2 TI2V-5B "
            "with the OpenVid LoRA and cached PyBullet VAE latents."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--vae-cache-dir", type=Path, default=None)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--openvid-lora", type=Path, default=DEFAULT_OPENVID_LORA)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--sde-start-index", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--bestofn", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--timestep-fraction", type=float, default=1.0)
    parser.add_argument("--hybrid-train-threshold", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--clip-range", type=float, default=1.0e-4)
    parser.add_argument("--adv-clip-max", type=float, default=5.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--no-save-lora", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    required_dirs = {
        "dataset root": args.dataset_root,
        "Wan root": args.wan_root,
        "DiffSynth root": args.diffsynth_root,
    }
    if args.vae_cache_dir is not None:
        required_dirs["VAE cache"] = args.vae_cache_dir
    for label, path in required_dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not args.openvid_lora.is_file():
        raise FileNotFoundError(f"OpenVid LoRA not found: {args.openvid_lora}")
    if args.num_generations < 2:
        raise ValueError("num_generations must be at least two")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("height and width must be divisible by the Wan VAE factor 16")
    if args.vae_cache_dir is not None and (args.height, args.width) != (512, 896):
        raise ValueError(
            "the supplied VAE cache is encoded only for 512x896; omit it for official 480x832"
        )
    if args.bestofn != args.num_generations:
        raise ValueError("this one-prompt smoke requires bestofn == num_generations")
    if args.sampling_steps < 2:
        raise ValueError("sampling_steps must be at least 2")
    max_sde_start = args.sampling_steps // 4
    if args.sde_start_index is not None and not 0 <= args.sde_start_index <= max_sde_start:
        raise ValueError(
            f"sde_start_index must be in [0, {max_sde_start}] for the official selection rule"
        )
    if args.eta <= 0:
        raise ValueError("eta must be positive so rollout log-probabilities are defined")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.adv_clip_max <= 0:
        raise ValueError("adv_clip_max must be positive")
    if args.timestep_fraction != 1.0:
        raise ValueError("the official-aligned smoke requires timestep_fraction=1.0")
    if args.hybrid_train_threshold <= 0:
        raise ValueError("hybrid_train_threshold must be positive")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must select one allowed physical GPU explicitly; GPU 4 is forbidden"
        )
    selected_devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if len(selected_devices) != 1:
        raise RuntimeError(
            f"expected exactly one CUDA_VISIBLE_DEVICES entry, got {visible_devices!r}"
        )
    if selected_devices[0] == "4":
        raise RuntimeError("GPU 4 is forbidden for this workspace")


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    stamp = time.strftime("smoke-%Y%m%d-%H%M%S", time.gmtime())
    return (DEFAULT_OUTPUT_BASE / stamp).resolve()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cuda_memory() -> dict[str, float]:
    gib = 1024.0**3
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "free_gib": free / gib,
        "total_gib": total / gib,
    }


def _transition_mean(
    latents: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float,
    sigma_next: float,
    eta: float,
) -> tuple[torch.Tensor, float]:
    """PhysRVG's released SDE transition mean, evaluated in float32."""
    sigma_value = float(sigma)
    sigma_next_value = float(sigma_next)
    delta_t = sigma_value - sigma_next_value
    if sigma_value <= 0 or delta_t <= 0:
        raise ValueError(
            f"expected sigma > sigma_next >= 0, got {sigma_value}, {sigma_next_value}"
        )
    x_t = latents.float()
    velocity = model_output.float()
    dsigma = sigma_next_value - sigma_value
    prev_mean = x_t + dsigma * velocity
    pred_x0 = x_t - sigma_value * velocity
    score = -(x_t - pred_x0 * (1.0 - sigma_value)) / (sigma_value**2)
    prev_mean = prev_mean + (-0.5 * eta**2 * score) * dsigma
    return prev_mean, eta * math.sqrt(delta_t)


def _euler_transition_mean(
    latents: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    sigma_value = float(sigma)
    sigma_next_value = float(sigma_next)
    if sigma_value <= 0 or sigma_value <= sigma_next_value:
        raise ValueError(
            f"expected sigma > sigma_next >= 0, got {sigma_value}, {sigma_next_value}"
        )
    return latents.float() + (sigma_next_value - sigma_value) * model_output.float()


def _future_log_prob(
    next_latents: torch.Tensor,
    transition_mean: torch.Tensor,
    std: float,
    prefix_latents: int = NUM_CLEAN_PREFIX_LATENTS,
) -> torch.Tensor:
    if std <= 0:
        raise ValueError(f"transition standard deviation must be positive, got {std}")
    residual = (
        next_latents[:, :, prefix_latents:].float()
        - transition_mean[:, :, prefix_latents:].float()
    )
    elementwise = (
        -(residual.square()) / (2.0 * std**2)
        - math.log(std)
        - 0.5 * math.log(2.0 * math.pi)
    )
    return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))


def _write_video(path: Path, video_cthw: torch.Tensor, fps: float) -> None:
    video = video_cthw.detach().float().cpu().clamp(-1.0, 1.0)
    frames = (
        ((video.permute(1, 2, 3, 0) + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _standardized_advantages(losses: list[float]) -> tuple[list[float], list[float]]:
    if len(losses) < 2:
        raise ValueError("group-relative advantages require at least two losses")
    rewards = -torch.tensor(losses, dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1.0e-8)
    if not bool(torch.isfinite(advantages).all().item()):
        raise RuntimeError("non-finite group-relative advantages")
    return rewards.tolist(), advantages.tolist()


def _merge_injected_lora_into_base(module: torch.nn.Module) -> int:
    from peft.tuners.lora.layer import LoraLayer

    lora_layers = [
        (name, child)
        for name, child in module.named_modules()
        if name and isinstance(child, LoraLayer)
    ]
    for name, layer in lora_layers:
        layer.merge(safe_merge=True, adapter_names=["default"])
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        base_layer = layer.get_base_layer()
        for parameter in base_layer.parameters():
            parameter.requires_grad = False
        parent._modules[child_name] = base_layer
    return len(lora_layers)


def _trainable_report(module: torch.nn.Module) -> dict[str, Any]:
    names: list[str] = []
    numel = 0
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            names.append(name)
            numel += parameter.numel()
    suspicious = [
        name
        for name in names
        if any(token in name.lower() for token in ("xssc", "dinov3", "object_pooler", "object_adapter"))
    ]
    return {
        "tensor_count": len(names),
        "numel": numel,
        "all_lora": bool(names) and all("lora_" in name for name in names),
        "forbidden_branch_trainables": suspicious,
        "first_names": names[:8],
    }


def _gradient_report(parameters: list[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
    with_grad = 0
    finite = True
    nonzero = 0
    max_abs = 0.0
    first_nonzero_name = None
    for name, parameter in parameters:
        if parameter.grad is None:
            continue
        with_grad += 1
        grad = parameter.grad.detach().float()
        finite = finite and bool(torch.isfinite(grad).all().item())
        grad_max = float(grad.abs().max().item())
        max_abs = max(max_abs, grad_max)
        if grad_max > 0:
            nonzero += 1
            if first_nonzero_name is None:
                first_nonzero_name = name
    return {
        "with_grad": with_grad,
        "nonzero_grad_tensors": nonzero,
        "all_finite": finite,
        "max_abs": max_abs,
        "first_nonzero_name": first_nonzero_name,
    }


def _model_output(
    pipe: Any,
    latents: torch.Tensor,
    timestep: float,
    prompt_embedding: torch.Tensor,
    clean_prefix_latents: torch.Tensor,
    *,
    gradient_checkpointing: bool,
) -> torch.Tensor:
    timestep_tensor = torch.tensor(
        [timestep], device=latents.device, dtype=pipe.torch_dtype
    )
    return pipe.model_fn(
        dit=pipe.dit,
        latents=latents,
        timestep=timestep_tensor,
        context=prompt_embedding,
        clean_prefix_latents=clean_prefix_latents,
        num_clean_prefix_latents=NUM_CLEAN_PREFIX_LATENTS,
        use_gradient_checkpointing=gradient_checkpointing,
        use_gradient_checkpointing_offload=False,
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)

    diffsynth_root = str(args.diffsynth_root.resolve())
    if diffsynth_root not in sys.path:
        sys.path.insert(0, diffsynth_root)
    os.environ["DIFFSYNTH_ROOT"] = diffsynth_root

    from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import (
        PyBullet0713NoGTBoxDataset,
    )
    from code_vjepa_vggt.train_v_newtrain import (
        WanTrainingModule,
        build_wan_model_paths,
        find_tokenizer_path,
    )
    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
    from peft import LoraConfig, inject_adapter_in_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke train")
    device = torch.device("cuda:0")
    _set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"[smoke] output_dir={output_dir}", flush=True)
    print("[smoke] loading one real PyBullet sample", flush=True)
    dataset = PyBullet0713NoGTBoxDataset(
        root=args.dataset_root,
        split="train",
        resolution=(args.height, args.width),
        num_frames=49,
        num_context_frames=NUM_CONTEXT_FRAMES,
        sampling_strategy="prefix",
        init_scan_limit=max(1, args.sample_index + 1),
        vae_cache_dir=args.vae_cache_dir,
        vae_checkpoint_path=args.wan_root / "Wan2.2_VAE.pth",
    )
    sample = dataset[args.sample_index]
    expected_shape = (48, 13, args.height // 16, args.width // 16)
    cached_latents = sample.get("precomputed_input_latents")
    cache_metadata = sample["metadata"].get("vae_cache", {})
    if args.vae_cache_dir is not None:
        if cached_latents is None or not cache_metadata.get("hit", False):
            raise RuntimeError("dataset sample did not report the requested VAE cache hit")
        if tuple(cached_latents.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected cache latent shape: {tuple(cached_latents.shape)} != {expected_shape}"
            )
        if cached_latents.dtype != torch.bfloat16:
            raise RuntimeError(f"unexpected cache dtype: {cached_latents.dtype}")

    print(
        "[smoke] loading Wan2.2, merging OpenVid 32/32, then injecting fresh RL LoRA 32/64",
        flush=True,
    )
    model = WanTrainingModule(
        model_paths=build_wan_model_paths(str(args.wan_root)),
        tokenizer_path=find_tokenizer_path(str(args.wan_root)),
        trainable_models=None,
        lora_base_model="dit",
        lora_target_modules=LORA_TARGET_MODULES,
        lora_rank=32,
        lora_alpha=32,
        lora_checkpoint=str(args.openvid_lora),
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        task="sft",
        device=device,
        fixed_num_context_frames=NUM_CONTEXT_FRAMES,
        enable_object_branch=False,
    )
    pipe = model.pipe
    openvid_trainable_report = _trainable_report(model)
    if not openvid_trainable_report["all_lora"]:
        raise RuntimeError(
            f"unexpected trainables before OpenVid merge: {openvid_trainable_report}"
        )
    merged_openvid_modules = _merge_injected_lora_into_base(pipe.dit)
    if merged_openvid_modules * 2 != openvid_trainable_report["tensor_count"]:
        raise RuntimeError(
            "OpenVid merge did not consume exactly one A/B pair per LoRA module: "
            f"modules={merged_openvid_modules}, tensors={openvid_trainable_report['tensor_count']}"
        )
    if any(parameter.requires_grad for parameter in pipe.dit.parameters()):
        raise RuntimeError("trainable parameters remain after merging the OpenVid adapter")

    resolved_targets = model.parse_lora_target_modules(pipe.dit, LORA_TARGET_MODULES)
    pipe.dit = inject_adapter_in_model(
        LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=resolved_targets,
        ),
        pipe.dit,
    )
    for parameter in pipe.dit.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.to(pipe.torch_dtype)
    pipe.dit.train()
    trainable_report = _trainable_report(model)
    if not trainable_report["all_lora"]:
        raise RuntimeError(f"non-LoRA trainables detected: {trainable_report}")
    if trainable_report["forbidden_branch_trainables"]:
        raise RuntimeError(f"forbidden xSSC/object trainables detected: {trainable_report}")
    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("no trainable LoRA parameters were found")

    prompt_unit = next(
        (unit for unit in pipe.units if isinstance(unit, WanVideoUnit_PromptEmbedder)),
        None,
    )
    if prompt_unit is None:
        raise RuntimeError("Wan prompt embedder unit was not found")
    with torch.no_grad():
        prompt_embedding = prompt_unit.encode_prompt(pipe, sample["caption"])
    prompt_embedding = prompt_embedding.to(device=device, dtype=pipe.torch_dtype)

    if cached_latents is not None:
        clean_prefix_latents = (
            cached_latents[:, :NUM_CLEAN_PREFIX_LATENTS]
            .unsqueeze(0)
            .to(device=device, dtype=pipe.torch_dtype)
        )
        if not torch.equal(
            clean_prefix_latents.detach().cpu(),
            cached_latents[:, :NUM_CLEAN_PREFIX_LATENTS].unsqueeze(0),
        ):
            raise RuntimeError("cached clean-prefix latent changed during device transfer")
        context_latent_source = "validated 512x896 VAE cache"
    else:
        with torch.no_grad():
            clean_prefix_latents = pipe.vae.encode(
                sample["context_video"].unsqueeze(0).to(dtype=pipe.torch_dtype),
                device=device,
                tiled=False,
            ).to(device=device, dtype=pipe.torch_dtype)
        context_latent_source = "online Wan VAE encoding of the first 8 frames"
    expected_prefix_shape = (1, 48, NUM_CLEAN_PREFIX_LATENTS, expected_shape[2], expected_shape[3])
    if tuple(clean_prefix_latents.shape) != expected_prefix_shape:
        raise RuntimeError(
            f"unexpected context latent shape: {tuple(clean_prefix_latents.shape)} "
            f"!= {expected_prefix_shape}"
        )

    pipe.scheduler.set_timesteps(
        args.sampling_steps,
        denoising_strength=1.0,
        shift=args.sigma_shift,
    )
    timesteps = [float(value.item()) for value in pipe.scheduler.timesteps]
    sigmas = [float(value.item()) for value in pipe.scheduler.sigmas]
    sigma_next_values = sigmas[1:] + [0.0]
    if args.sde_start_index is None:
        sde_index_generator = torch.Generator(device="cpu").manual_seed(args.seed)
        sde_start_index = int(
            torch.randint(
                0,
                args.sampling_steps // 4 + 1,
                (1,),
                generator=sde_index_generator,
            ).item()
        )
    else:
        sde_start_index = int(args.sde_start_index)
    sde_timestep_indices = [sde_start_index, sde_start_index + 1]

    initial_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial_latents = torch.randn(
        (1, *expected_shape),
        generator=initial_generator,
        device=device,
        dtype=pipe.torch_dtype,
    )
    initial_latents[:, :, :NUM_CLEAN_PREFIX_LATENTS] = clean_prefix_latents

    print(
        f"[smoke] running {args.num_generations} x {args.sampling_steps}-step rollouts; "
        f"SDE transitions={sde_timestep_indices}",
        flush=True,
    )
    transitions: list[RolloutTransition] = []
    final_latents: list[torch.Tensor] = []
    pipe.dit.eval()
    with torch.no_grad():
        for generation_id in range(args.num_generations):
            latents = initial_latents.clone()
            rollout_generator = torch.Generator(device=device).manual_seed(
                args.seed + 1000 + generation_id
            )
            for timestep_index, (timestep, sigma, sigma_next) in enumerate(
                zip(timesteps, sigmas, sigma_next_values, strict=True)
            ):
                x_t = latents
                model_output = _model_output(
                    pipe,
                    x_t,
                    timestep,
                    prompt_embedding,
                    clean_prefix_latents,
                    gradient_checkpointing=False,
                )
                is_sde_transition = timestep_index in sde_timestep_indices
                if is_sde_transition:
                    prev_mean, std = _transition_mean(
                        x_t, model_output, sigma, sigma_next, args.eta
                    )
                    transition_noise = torch.randn(
                        prev_mean.shape,
                        generator=rollout_generator,
                        device=device,
                        dtype=torch.float32,
                    )
                    next_latents = (prev_mean + transition_noise * std).to(pipe.torch_dtype)
                    old_log_prob = _future_log_prob(next_latents, prev_mean, std)
                    if not bool(torch.isfinite(old_log_prob).all().item()):
                        raise RuntimeError("non-finite rollout log-probability")
                else:
                    prev_mean = _euler_transition_mean(
                        x_t, model_output, sigma, sigma_next
                    )
                    next_latents = prev_mean.to(pipe.torch_dtype)
                    old_log_prob = None
                next_latents[:, :, :NUM_CLEAN_PREFIX_LATENTS] = clean_prefix_latents
                if not torch.equal(
                    next_latents[:, :, :NUM_CLEAN_PREFIX_LATENTS],
                    clean_prefix_latents,
                ):
                    raise RuntimeError("clean context prefix changed during rollout")
                if is_sde_transition:
                    transitions.append(
                        RolloutTransition(
                            generation_id=generation_id,
                            timestep_index=timestep_index,
                            timestep=timestep,
                            sigma=sigma,
                            sigma_next=sigma_next,
                            latents=x_t.detach().cpu(),
                            next_latents=next_latents.detach().cpu(),
                            old_log_prob=float(old_log_prob.item()),
                        )
                    )
                latents = next_latents
            final_latents.append(latents.detach().cpu())

    expected_transitions = args.num_generations * len(sde_timestep_indices)
    if len(transitions) != expected_transitions:
        raise RuntimeError(f"expected {expected_transitions} stored transitions")

    print(
        "[smoke] decoding rollouts and computing future-only pixel rewards",
        flush=True,
    )
    gt_video = sample["video"].float().cpu()
    future_pixel_losses: list[float] = []
    pipe.dit.eval()
    with torch.no_grad():
        for generation_id, latent in enumerate(final_latents):
            decoded = pipe.vae.decode(
                latent.to(device=device, dtype=pipe.torch_dtype),
                device=device,
                tiled=False,
            )[0].float().cpu()
            if tuple(decoded.shape) != tuple(gt_video.shape):
                raise RuntimeError(
                    f"decoded shape {tuple(decoded.shape)} != GT shape {tuple(gt_video.shape)}"
                )
            pixel_loss = (
                decoded[:, NUM_CONTEXT_FRAMES:] - gt_video[:, NUM_CONTEXT_FRAMES:]
            ).abs().mean()
            if not bool(torch.isfinite(pixel_loss).item()):
                raise RuntimeError("non-finite pixel reward loss")
            future_pixel_losses.append(float(pixel_loss.item()))
            _write_video(
                output_dir / f"rollout_{generation_id}.mp4",
                decoded,
                args.fps,
            )
            del decoded
            torch.cuda.empty_cache()

    rewards, advantages = _standardized_advantages(future_pixel_losses)
    if max(advantages) <= 0 or min(advantages) >= 0:
        raise RuntimeError(f"advantages do not contain both signs: {advantages}")
    hybrid_sft_triggered = (
        sum(future_pixel_losses) / len(future_pixel_losses)
        > args.hybrid_train_threshold
    )
    if hybrid_sft_triggered:
        raise RuntimeError(
            "official hybrid SFT branch was triggered, but this RL smoke only validates the RL update"
        )

    print(
        f"[smoke] losses={future_pixel_losses} advantages={advantages}; running one optimizer update",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_parameters],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    optimizer.zero_grad(set_to_none=True)
    pipe.dit.train()
    per_transition_losses: list[float] = []
    ratios: list[float] = []
    new_log_probs: list[float] = []
    clipped_advantages: list[float] = []
    loss_divisor = args.gradient_accumulation_steps * len(sde_timestep_indices)
    for transition in transitions:
        advantage = advantages[transition.generation_id]
        clipped_advantage = max(
            -args.adv_clip_max,
            min(args.adv_clip_max, advantage),
        )
        x_t = transition.latents.to(device=device, dtype=pipe.torch_dtype)
        x_next = transition.next_latents.to(device=device, dtype=pipe.torch_dtype)
        model_output = _model_output(
            pipe,
            x_t,
            transition.timestep,
            prompt_embedding,
            clean_prefix_latents,
            gradient_checkpointing=True,
        )
        prev_mean, std = _transition_mean(
            x_t,
            model_output,
            transition.sigma,
            transition.sigma_next,
            args.eta,
        )
        new_log_prob = _future_log_prob(x_next, prev_mean, std)
        old_log_prob = torch.tensor(
            [transition.old_log_prob], device=device, dtype=torch.float32
        )
        ratio = torch.exp(new_log_prob - old_log_prob)
        advantage_tensor = torch.tensor(
            [clipped_advantage], device=device, dtype=torch.float32
        )
        unclipped_loss = -advantage_tensor * ratio
        clipped_loss = -advantage_tensor * torch.clamp(
            ratio, 1.0 - args.clip_range, 1.0 + args.clip_range
        )
        rl_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
        if not bool(torch.isfinite(rl_loss).item()):
            raise RuntimeError("non-finite clipped-ratio RL loss")
        (rl_loss / loss_divisor).backward()
        per_transition_losses.append(float(rl_loss.detach().item()))
        ratios.append(float(ratio.detach().item()))
        new_log_probs.append(float(new_log_prob.detach().item()))
        clipped_advantages.append(float(clipped_advantage))
        del x_t, x_next, model_output, prev_mean, new_log_prob, ratio, rl_loss

    gradient_report = _gradient_report(trainable_parameters)
    if not gradient_report["all_finite"]:
        raise RuntimeError(f"non-finite LoRA gradients: {gradient_report}")
    if gradient_report["nonzero_grad_tensors"] == 0:
        raise RuntimeError(f"all LoRA gradients are zero: {gradient_report}")

    changed_name = gradient_report["first_nonzero_name"]
    changed_parameter = dict(trainable_parameters)[changed_name]
    parameter_before = changed_parameter.detach().float().cpu().clone()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in trainable_parameters],
        max_norm=args.max_grad_norm,
    )
    if not bool(torch.isfinite(grad_norm).item()) or float(grad_norm.item()) <= 0:
        raise RuntimeError(f"invalid LoRA gradient norm: {float(grad_norm.item())}")
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    parameter_delta = float(
        (changed_parameter.detach().float().cpu() - parameter_before).abs().max().item()
    )
    if parameter_delta <= 0 or not math.isfinite(parameter_delta):
        raise RuntimeError(
            f"optimizer step did not change sampled LoRA parameter {changed_name}: {parameter_delta}"
        )

    lora_output = None
    if not args.no_save_lora:
        lora_output = output_dir / "lora_after_smoke.safetensors"
        trainable_state = {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in trainable_parameters
        }
        save_file(
            trainable_state,
            str(lora_output),
            metadata={
                "format": "physrvg-openvid-rl-smoke-trainables",
                "base": str(args.wan_root),
                "merged_source_lora": str(args.openvid_lora),
                "rl_lora_rank": "32",
                "rl_lora_alpha": "64",
            },
        )

    summary = {
        "status": "passed",
        "scope": {
            "algorithm": "PhysRVG SDE log-prob + clipped-ratio RL update",
            "model_backend": "DiffSynth Wan2.2 TI2V-5B",
            "initial_adapter": "OpenVid 32/32 LoRA merged into Wan base",
            "trainable_adapter": "fresh PhysRVG RL LoRA rank/alpha 32/64",
            "xssc_loaded": False,
            "object_branch_loaded": False,
            "reward": "negative future RGB pixel MAE, then group-standardized (smoke substitute)",
            "formal_physrvg_position_reward": False,
            "official_hyperparameters_aligned": True,
            "documented_exceptions": [
                "PyBullet dataset without two object-mask videos",
                "future pixel reward instead of SAM2 position reward",
                "DiffSynth backend and scheduler instead of Diffusers/FSDP",
                "8-frame clean-prefix V2V input instead of the official Diffusers condition path",
            ],
        },
        "paths": {
            "dataset_root": str(args.dataset_root.resolve()),
            "vae_cache_dir": (
                str(args.vae_cache_dir.resolve()) if args.vae_cache_dir is not None else None
            ),
            "wan_root": str(args.wan_root.resolve()),
            "openvid_lora": str(args.openvid_lora.resolve()),
            "output_dir": str(output_dir),
            "lora_after_smoke": str(lora_output) if lora_output is not None else None,
        },
        "sample": {
            "sample_index": args.sample_index,
            "sample_key": sample["metadata"]["sample_key"],
            "caption": sample["caption"],
            "video_path": sample["video_path"],
            "height": args.height,
            "width": args.width,
            "cache_hit": bool(cache_metadata.get("hit", False)),
            "cache_encoding_id": cache_metadata.get("encoding_id"),
            "context_latent_source": context_latent_source,
            "cached_latent_shape": (
                list(cached_latents.shape) if cached_latents is not None else None
            ),
            "cached_latent_dtype": (
                str(cached_latents.dtype) if cached_latents is not None else None
            ),
            "context_latent_shape": list(clean_prefix_latents.shape),
            "clean_context_frames": NUM_CONTEXT_FRAMES,
            "clean_prefix_latents": NUM_CLEAN_PREFIX_LATENTS,
            "future_latents_initialized_from_noise": expected_shape[1]
            - NUM_CLEAN_PREFIX_LATENTS,
        },
        "rollout": {
            "num_generations": args.num_generations,
            "bestofn": args.bestofn,
            "sampling_steps": args.sampling_steps,
            "sigma_shift": args.sigma_shift,
            "eta": args.eta,
            "guidance_scale_config": args.guidance_scale,
            "do_cfg": False,
            "full_trajectory_sde": False,
            "sde_start_index": sde_start_index,
            "sde_timestep_indices": sde_timestep_indices,
            "deterministic_euler_steps": args.sampling_steps - len(sde_timestep_indices),
            "timesteps": timesteps,
            "sigmas": sigmas,
            "old_log_probs": [item.old_log_prob for item in transitions],
            "new_log_probs": new_log_probs,
            "ratios": ratios,
            "future_pixel_mae": future_pixel_losses,
            "rewards": rewards,
            "advantages": advantages,
            "clipped_advantages_per_transition": clipped_advantages,
            "prefix_preserved": True,
            "log_prob_excludes_prefix": True,
        },
        "update": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "lr_scheduler": "constant_with_zero_warmup",
            "post_step_learning_rate": optimizer.param_groups[0]["lr"],
            "weight_decay": args.weight_decay,
            "clip_range": args.clip_range,
            "adv_clip_max": args.adv_clip_max,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "train_timesteps": len(sde_timestep_indices),
            "timestep_fraction": args.timestep_fraction,
            "hybrid_train_enabled": True,
            "hybrid_train_threshold": args.hybrid_train_threshold,
            "hybrid_sft_triggered": hybrid_sft_triggered,
            "loss_divisor": loss_divisor,
            "per_transition_rl_losses": per_transition_losses,
            "mean_rl_loss": float(sum(per_transition_losses) / len(per_transition_losses)),
            "preclip_grad_norm": float(grad_norm.item()),
            "gradient_report": gradient_report,
            "sampled_changed_parameter": changed_name,
            "sampled_parameter_max_abs_delta": parameter_delta,
            "optimizer_steps": 1,
        },
        "adapter_initialization": {
            "openvid_rank": 32,
            "openvid_alpha": 32,
            "merged_openvid_modules": merged_openvid_modules,
            "rl_lora_rank": 32,
            "rl_lora_alpha": 64,
            "rl_lora_init": "gaussian",
            "openvid_trainables_before_merge": openvid_trainable_report,
        },
        "trainables": trainable_report,
        "cuda": {
            "visible_physical_device": os.environ["CUDA_VISIBLE_DEVICES"],
            "logical_device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "memory": _cuda_memory(),
        },
        "seed": args.seed,
    }
    summary_path = output_dir / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
