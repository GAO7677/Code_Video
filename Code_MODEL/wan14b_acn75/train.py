"""CUDA-only, two-expert FSDP2 LoRA training on verified Wan14B caches.

All ranks execute the same expert on each microbatch.  Gradients are reduced
on every microbatch: disabling synchronization until the last microbatch is
incorrect when that last microbatch does not use both experts.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import random
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from noise import flow_batch, future_loss, training_scheduler


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(value):
    objects = [value]
    if dist.is_initialized():
        dist.broadcast_object_list(objects, src=0)
    return objects[0]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def expert_name(name):
    parts = name.split(".")
    for expert in ("high", "low"):
        if expert in parts or any(part.startswith(expert + "_") for part in parts):
            return expert
    raise RuntimeError(f"cannot identify expert for trainable parameter {name}")


def adapter_hashes(model):
    """Exact local-shard hashes, used only for the smoke round-trip check."""
    digests = {expert: hashlib.sha256() for expert in ("high", "low")}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest = digests[expert_name(name)]
            digest.update(name.encode())
            tensor = local_tensor(parameter.detach()).contiguous().cpu()
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {key: value.hexdigest() for key, value in digests.items()}


class LocalShardOptimizer:
    """Let bitsandbytes update ordinary CUDA tensors sharing DTensor storage.

    Each FSDP2 parameter remains owned by the model.  The optimizer owns a
    parameter view of its local shard, rebound and checked before every step.
    This avoids sending DTensor through bitsandbytes CUDA kernels.
    """
    def __init__(self, model, training):
        import bitsandbytes as bnb

        if training.get("optimizer", "paged_adamw8bit").lower() != "paged_adamw8bit":
            raise ValueError("this runner requires paged_adamw8bit")
        self.bindings = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                shard = local_tensor(parameter.detach())
                if shard.device.type != "cuda" or shard.dtype != torch.float32:
                    raise RuntimeError(f"LoRA shard must be CUDA float32: {name}, {shard.device}, {shard.dtype}")
                view = torch.nn.Parameter(shard, requires_grad=True)
                self.bindings.append((name, parameter, view))
        if not self.bindings:
            raise RuntimeError("no trainable LoRA parameters")
        self.optimizer = bnb.optim.PagedAdamW8bit(
            [view for _, _, view in self.bindings],
            lr=float(training["learning_rate"]),
            betas=tuple(training.get("betas", (0.9, 0.999))),
            eps=float(training.get("adam_epsilon", 1e-8)),
            weight_decay=float(training.get("weight_decay", 0.01)),
        )
        self.rebindings = 0

    def step(self):
        active = set()
        for name, parameter, view in self.bindings:
            current = local_tensor(parameter.detach())
            if current.shape != view.shape or current.dtype != view.dtype:
                raise RuntimeError(f"FSDP local shard contract changed: {name}")
            if current.data_ptr() != view.data_ptr():
                view.data = current
                self.rebindings += 1
            gradient = parameter.grad
            view.grad = None if gradient is None else local_tensor(gradient).detach()
            if view.grad is not None:
                if view.grad.shape != view.shape:
                    raise RuntimeError(f"local gradient shape mismatch: {name}")
                active.add(expert_name(name))
        self.optimizer.step()
        return active

    def zero_grad(self, model):
        self.optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)


class OffsetSampler(DistributedSampler):
    """Resume a deterministic epoch without loading already-consumed caches."""
    offset = 0

    def __iter__(self):
        return iter(list(super().__iter__())[self.offset:])

    def __len__(self):
        return max(0, super().__len__() - self.offset)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())


def run_signature(config):
    relevant = {key: config[key] for key in ("model", "data", "cache", "training")}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def save_checkpoint(model, optimizer, output, state, config, wandb_id):
    from model import save_adapter

    checkpoint = output / "checkpoints" / f"step_{state['step']:06d}"
    if rank() == 0:
        checkpoint.mkdir(parents=True, exist_ok=True)
    barrier()
    adapter_path = checkpoint / "adapters.safetensors"
    metadata = {
        "step": str(state["step"]), "training_mode": config["training"]["mode"],
        "run_signature": run_signature(config),
        "expert_optimizer_steps": json.dumps(state["expert_optimizer_steps"], sort_keys=True),
        "expert_microbatches": json.dumps(state["expert_microbatches"], sort_keys=True),
    }
    save_adapter(model, adapter_path, metadata=metadata)
    name = broadcast_object(f"latest.tmp.{uuid.uuid4().hex}" if rank() == 0 else None)
    temporary = output / name
    if rank() == 0:
        temporary.mkdir()
    barrier()
    # Only latest contains optimizer state.  Historical 500-step checkpoints
    # contain the paired adapters and a small manifest, avoiding huge repeats.
    payload = {
        "optimizer": cpu_tree(optimizer.optimizer.state_dict()),
        "rng": rng_state(), "rank": rank(), "world_size": world_size(),
    }
    torch.save(payload, temporary / f"rank_{rank():05d}.pt")
    del payload
    barrier()
    complete = {
        **state, "world_size": world_size(), "run_signature": run_signature(config),
        "adapter_path": str(adapter_path.resolve()), "wandb_run_id": wandb_id,
        "scheduler": {"type": "constant", "warmup_steps": 0},
    }
    if rank() == 0:
        atomic_json(checkpoint / "manifest.json", complete)
        atomic_json(temporary / "state.json", complete)
        previous = output / "latest.previous"
        if previous.exists():
            shutil.rmtree(previous)
        latest = output / "latest"
        if latest.exists():
            latest.rename(previous)
        temporary.rename(latest)
        if previous.exists():
            shutil.rmtree(previous)
    barrier()
    return output / "latest"


def load_checkpoint(model, optimizer, checkpoint, config, device):
    from model import load_adapter

    state = json.loads((checkpoint / "state.json").read_text())
    if state["world_size"] != world_size():
        raise RuntimeError("optimizer resume requires the original world size")
    if state["run_signature"] != run_signature(config):
        raise RuntimeError("resume model/data/cache/training configuration differs")
    load_adapter(model, Path(state["adapter_path"]), strict=True)
    optimizer.optimizer.state.clear()
    gc.collect()
    torch.cuda.empty_cache()
    payload = torch.load(checkpoint / f"rank_{rank():05d}.pt", map_location=device, weights_only=False)
    if payload["rank"] != rank() or payload["world_size"] != world_size():
        raise RuntimeError("optimizer shard rank mismatch")
    optimizer.optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    return state


def setup_wandb(config, output, resume_state, smoke):
    run, error, run_id = None, None, None
    if rank() == 0:
        try:
            import wandb

            settings = config.get("wandb", {})
            if not settings.get("enabled", True) or settings.get("mode", "online") != "online":
                raise RuntimeError("online W&B is required; offline/disabled mode is not accepted")
            if os.environ.get("WANDB_MODE", "online") != "online" or os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1"}:
                raise RuntimeError("environment disables online W&B")
            if not wandb.login(relogin=False, force=True, timeout=60):
                raise RuntimeError("W&B credentials unavailable")
            run_id = resume_state.get("wandb_run_id") if resume_state else None
            run = wandb.init(
                project=settings.get("project", "wan14b-acn75"),
                entity=settings.get("entity"), name=settings.get("name", config["training"]["mode"]),
                id=run_id, resume="must" if run_id else None, mode="online",
                dir=str(output), config=config,
                tags=["smoke" if smoke else "train", "pusa-wan2.2-14b", config["training"]["mode"]],
                settings=wandb.Settings(init_timeout=120),
            )
            if run is None or run.offline:
                raise RuntimeError("W&B did not create an online run")
            run_id = run.id
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error, run_id = broadcast_object((error, run_id))
    if error:
        raise RuntimeError(f"W&B online initialization failed: {error}")
    return run, run_id


def assert_all_finite(value, description, device):
    finite = torch.isfinite(value.detach()).all().to(device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError(f"nonfinite {description}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", default="auto", help="auto, none, or a latest directory")
    parser.add_argument("--smoke", action="store_true", help="three updates, all modes and both experts, checkpoint round-trip")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    if config_path.suffix in {".yaml", ".yml"}:
        import yaml
        config = yaml.safe_load(config_path.read_text())
    else:
        config = json.loads(config_path.read_text())
    config = copy.deepcopy(config)
    if args.output_dir:
        config["paths"]["output_dir"] = str(Path(args.output_dir).resolve())
    output = Path(config["paths"]["output_dir"]).resolve()
    if str(output).startswith("/home/"):
        raise ValueError("training artifacts must be stored under /data, not /home")
    training = config["training"]
    if training["mode"] not in {"A", "C", "N75"}:
        raise ValueError("training.mode must be A, C, or N75")
    if training.get("scheduler", "constant") != "constant" or int(training.get("warmup_steps", 0)) != 0:
        raise ValueError("the aligned recipe requires a constant LR with zero warmup")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU model loading or fallback is implemented")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl", device_id=device)
    random.seed(int(training.get("seed", 42)) + rank())
    torch.manual_seed(int(training.get("seed", 42)) + rank())
    torch.cuda.manual_seed(int(training.get("seed", 42)) + rank())
    micro = int(training.get("micro_batch_size", 1))
    effective = int(training.get("effective_batch_size", 8))
    if effective % (micro * world_size()):
        raise ValueError("effective batch must be divisible by micro batch times world size")
    accumulation = effective // (micro * world_size())
    if args.smoke and accumulation < 2:
        raise ValueError("smoke requires at least two accumulation microbatches to cover both experts")
    config["execution"] = {
        "world_size": world_size(), "gradient_accumulation_steps": accumulation,
        "timestep_sampling": "uniform paired training index, one shared index per microbatch across ranks",
        "gradient_sync": "every microbatch, including both expert routes",
        "effective_training_shift": float(training.get("flow_shift", 1.0)),
        "smoke": args.smoke,
    }
    if rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
    barrier()
    if args.resume == "auto":
        resume = output / "latest" if (output / "latest" / "state.json").exists() else None
    else:
        resume = None if args.resume == "none" else Path(args.resume).resolve()
    resume_header = json.loads((resume / "state.json").read_text()) if resume else None
    from data import Wan14BCachedDataset
    from model import build_model

    # Fail on a missing/invalid online-equivalence cache audit before allocating
    # the large model or creating an experiment run.
    dataset = Wan14BCachedDataset(config, require_verified=True)
    run, wandb_id = setup_wandb(config, output, resume_header, args.smoke)
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (world_size(),)) if dist.is_initialized() else None
    torch.cuda.reset_peak_memory_stats(device)
    model, audit = build_model(config, device, mesh=mesh)
    model.train()
    optimizer = LocalShardOptimizer(model, training)
    state = {
        "step": 0, "data_epoch": 0, "data_batch": 0,
        "expert_optimizer_steps": {"high": 0, "low": 0},
        "expert_microbatches": {"high": 0, "low": 0},
    }
    if resume:
        state.update(load_checkpoint(model, optimizer, resume, config, device))
    if rank() == 0:
        atomic_json(output / "resolved_config.json", config)
        atomic_json(output / "model_audit.json", audit)
        print(json.dumps({"event": "ready", "step": state["step"], "samples": len(dataset), **config["execution"]}), flush=True)
    sampler = OffsetSampler(dataset, num_replicas=world_size(), rank=rank(), shuffle=True,
                            seed=int(training.get("seed", 42)), drop_last=True)
    sampler.set_epoch(state["data_epoch"])
    sampler.offset = state["data_batch"] * micro
    # A separate generator prevents iterator creation on resume from changing
    # the model/noise RNG.  Cached items themselves are deterministic.
    loader_generator = torch.Generator().manual_seed(int(training.get("seed", 42)))
    loader = DataLoader(dataset, batch_size=micro, sampler=sampler, num_workers=0,
                        pin_memory=True, drop_last=True, generator=loader_generator)
    iterator = iter(loader)
    scheduler = training_scheduler(device, int(training.get("num_train_timesteps", 1000)),
                                   float(training.get("flow_shift", 1.0)))
    context_frames = int(training.get("context_latent_frames", 2))
    max_steps = 3 if args.smoke else int(training.get("max_steps", 3000))
    save_every = int(training.get("save_every", 500))
    if max_steps <= 0 or save_every <= 0:
        raise ValueError("max_steps and save_every must be positive")
    smoke_before = adapter_hashes(model) if args.smoke else None
    start = time.monotonic()
    last_checkpoint = None
    while state["step"] < max_steps:
        optimizer.zero_grad(model)
        loss_total = torch.zeros((), device=device)
        diagnostic_sums = Counter()
        micro_counts = Counter()
        mode = ("A", "C", "N75")[state["step"]] if args.smoke else training["mode"]
        step_started = time.monotonic()
        for micro_index in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                state["data_epoch"] += 1
                state["data_batch"] = 0
                sampler.set_epoch(state["data_epoch"])
                sampler.offset = 0
                iterator = iter(loader)
                batch = next(iterator)
            state["data_batch"] += 1
            clean = batch["latents"].to(device, non_blocking=True)
            text = batch["text"].to(device, non_blocking=True)
            if args.smoke:
                sampled_index = 56 if micro_index % 2 == 0 else 500
                index = torch.tensor([sampled_index], device=device, dtype=torch.long)
            else:
                index = torch.randint(len(scheduler.timesteps), (1,), device=device) if rank() == 0 else torch.zeros(1, device=device, dtype=torch.long)
            if dist.is_initialized():
                dist.broadcast(index, src=0)
            flow = flow_batch(clean, scheduler, mode, int(index.item()), context_frames=context_frames,
                              boundary=float(config["model"].get("boundary", 0.875)))
            # Always synchronize: a later microbatch may select the other expert.
            model.set_requires_gradient_sync(True)
            prediction = model(flow.latents, flow.timesteps, text, expert=flow.expert)
            loss = future_loss(prediction, flow.target, context_frames)
            assert_all_finite(loss, "loss", device)
            (loss / accumulation).backward()
            loss_total += loss.detach() / accumulation
            micro_counts[flow.expert] += 1
            diagnostic_sums["future_sigma"] += float(flow.future_sigma)
            diagnostic_sums["context_sigma"] += float(flow.context_sigma)
            diagnostic_sums["context_sigma_target"] += float(flow.context_sigma_target)
            del clean, text, flow, prediction, loss, batch
        grad_norm = model.clip_grad_norm_(float(training.get("max_grad_norm", 1.0)))
        assert_all_finite(grad_norm, "gradient norm", device)
        if args.smoke and grad_norm.item() <= 0:
            raise RuntimeError("smoke produced zero gradient norm")
        active = optimizer.step()
        if active != set(micro_counts):
            raise RuntimeError(f"expert gradient routing mismatch: expected {set(micro_counts)}, got {active}")
        state["step"] += 1
        for expert in ("high", "low"):
            state["expert_microbatches"][expert] += micro_counts[expert]
            state["expert_optimizer_steps"][expert] += int(expert in active)
        if dist.is_initialized():
            dist.all_reduce(loss_total)
            loss_total /= world_size()
        memory = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                              device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        metrics = {
            "train/loss": float(loss_total), "train/grad_norm": float(grad_norm),
            "train/lr": float(training["learning_rate"]), "train/step_seconds": time.monotonic() - step_started,
            "train/effective_batch_size": effective, "train/epoch": state["data_epoch"],
            "memory/max_allocated_gib": float(memory[0]) / 2**30,
            "memory/max_reserved_gib": float(memory[1]) / 2**30,
            "optimizer/storage_rebindings": optimizer.rebindings,
            **{f"noise/{key}": value / accumulation for key, value in diagnostic_sums.items()},
            **{f"expert/{key}_updates": value for key, value in state["expert_optimizer_steps"].items()},
            **{f"expert/{key}_microbatches": value for key, value in state["expert_microbatches"].items()},
        }
        if rank() == 0:
            run.log(metrics, step=state["step"])
            print(json.dumps({"event": "step", "step": state["step"], "mode": mode, **metrics}), flush=True)
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": state["step"], "mode": mode, **metrics}) + "\n")
        if state["step"] % save_every == 0 or state["step"] == max_steps:
            last_checkpoint = save_checkpoint(model, optimizer, output, state, config, wandb_id)
    if args.smoke:
        after = adapter_hashes(model)
        if any(after[expert] == smoke_before[expert] for expert in ("high", "low")):
            raise RuntimeError("smoke did not update both experts")
        expected_rng = rng_state()
        with torch.no_grad():
            for _, parameter, _ in optimizer.bindings:
                shard = local_tensor(parameter)
                if shard.numel():
                    shard.reshape(-1)[0].add_(1.0)
        load_checkpoint(model, optimizer, last_checkpoint, config, device)
        if adapter_hashes(model) != after:
            raise RuntimeError("adapter checkpoint reload changed local shard values")
        actual_rng = rng_state()
        if (actual_rng["python"] != expected_rng["python"] or
                not torch.equal(actual_rng["torch"], expected_rng["torch"]) or
                not torch.equal(actual_rng["cuda"], expected_rng["cuda"])):
            raise RuntimeError("checkpoint RNG round-trip failed")
        report = {
            "status": "passed", "modes": ["A", "C", "N75"], "experts": ["high", "low"],
            "adapter_reload_exact": True, "rng_reload_exact": True,
            "world_size": world_size(), "effective_batch_size": effective,
            "optimizer_updates": state["expert_optimizer_steps"],
            "config_signature": run_signature(config),
            "elapsed_seconds": time.monotonic() - start,
            "max_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "max_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        reports = [None] * world_size() if rank() == 0 else None
        if dist.is_initialized():
            dist.gather_object(report, reports, dst=0)
        else:
            reports = [report]
        if rank() == 0:
            atomic_json(output / "smoke_passed.json", {"status": "passed", "ranks": reports})
    if rank() == 0:
        atomic_json(output / "completed.json", {"step": state["step"], "expert_updates": state["expert_optimizer_steps"], "smoke": args.smoke})
        run.finish()
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
