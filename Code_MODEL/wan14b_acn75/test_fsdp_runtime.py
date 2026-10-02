"""Two-GPU runtime probe using tiny random Pusa experts, never real weights.

Run explicitly through torchrun with CUDA_VISIBLE_DEVICES=0,1 or 2,3.
Default hidden size 256 makes rank-32 local adapter shards reach the 4096
parameter threshold, exercising actual bitsandbytes 8-bit optimizer states.
This validates distributed mechanics, not full A14B memory feasibility.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from model import (ARCHITECTURE, DEFAULT_TARGETS, PusaDualExpert,
                   _inject_block, shard_block, load_upstream, load_adapter, save_adapter,
                   trainable_parameter_manifest)
from train import LocalShardOptimizer, adapter_hashes, cpu_tree, local_tensor, optimizer_hash


def emit(stage, **values):
    if dist.get_rank() == 0:
        print(json.dumps(dict(stage=stage, **values), sort_keys=True), flush=True)


def initialize(module, device, generator):
    module.to_empty(device=device)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("bias"):
                param.zero_()
            elif param.ndim == 1:
                param.fill_(1)
            else:
                nn.init.normal_(param, mean=0, std=0.02, generator=generator)
    module.requires_grad_(False)


def build_tiny(device, mesh):
    upstream, provenance = load_upstream()
    architecture = dict(ARCHITECTURE, dim=256, ffn_dim=512, num_heads=2, num_layers=2)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                  cast_forward_inputs=False)
    experts = {}
    for index, name in enumerate(("high", "low")):
        generator = torch.Generator(device=device).manual_seed(1234 + index)
        with torch.device("meta"):
            expert = upstream.WanModelPusa(**architecture).to(dtype=torch.bfloat16)
        with torch.device(device):
            expert.freqs = upstream.precompute_freqs_cis_3d(architecture["dim"] // architecture["num_heads"])
        for block in expert.blocks:
            initialize(block, device, generator)
            _inject_block(block, DEFAULT_TARGETS, 32, 64, 0, generator)
            shard_block(block, mesh, policy)
        for name_, child in expert.named_children():
            if name_ != "blocks":
                initialize(child, device, generator)
        fully_shard(expert, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        experts[name] = expert
    model = PusaDualExpert(experts, gradient_checkpointing=True)
    states_seen = set()
    for module in model.modules():
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if id(state) not in states_seen and state._fsdp_param_group is not None:
                states_seen.add(id(state))
                dtypes = {p.sharded_param.dtype for p in state._fsdp_param_group.fsdp_params}
                if len(dtypes) != 1:
                    raise AssertionError(f"Mixed original dtypes in one FSDP group: {dtypes}")
    model.initialization_identity = json.loads(json.dumps(dict(
        probe="random_tiny_two_expert_v1", architecture=architecture,
        source_commit=provenance["commit"], lora_rank=32, lora_alpha=64,
        lora_targets=DEFAULT_TARGETS,
    )))
    return model


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            digest.update(name.encode())
            tensor = local_tensor(param.detach()).contiguous().cpu()
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def full_adapter_hash(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not isinstance(param, DTensor):
                raise AssertionError(f"Expected sharded trainable parameter: {name}")
            digest.update(name.encode())
            tensor = param.full_tensor().detach().contiguous().cpu()
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, digest.hexdigest())
    if len(set(hashes)) != 1:
        raise AssertionError(f"full_tensor differs across ranks: {hashes}")
    return hashes[0]


def reshard(model):
    # FSDP roots may retain frozen root parameters after eval forward.
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def evaluate(model, device, seed=111):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((1, 16, 3, 4, 4), device=device, generator=generator)
    text = torch.randn((1, 8, 4096), device=device, generator=generator)
    model.eval()
    output = {}
    with torch.no_grad():
        for name in ("high", "low"):
            t = torch.tensor([[0, 500, 500]], device=device, dtype=torch.float32)
            output[name] = model(x, t, text, expert=name).detach().clone()
            same_t = torch.full_like(t, 500)
            changed = model(x, same_t, text, expert=name)
            if torch.equal(output[name], changed):
                raise AssertionError("Per-frame timestep changes did not affect output")
    reshard(model)
    model.train()
    return output


def train_step(model, optimizer, device, route, seed):
    optimizer.zero_grad(model)
    before = adapter_hashes(model)
    generator = torch.Generator(device=device).manual_seed(seed + dist.get_rank())
    losses = []
    for expert in route:
        x = torch.randn((1, 16, 3, 4, 4), device=device, generator=generator)
        text = torch.randn((1, 8, 4096), device=device, generator=generator)
        target = torch.randn(x.shape, device=device, generator=generator)
        t = torch.tensor([[0, 950, 950] if expert == "high" else [0, 500, 500]],
                         device=device, dtype=torch.float32)
        model.set_requires_gradient_sync(True)
        pred = model(x, t, text, expert=expert)
        loss = (pred[:, :, 1:].float() - target[:, :, 1:]).square().mean()
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite tiny-model loss")
        (loss / len(route)).backward()
        losses.append(float(loss.detach()))
    norm = model.clip_grad_norm_(1.0)
    if not torch.isfinite(norm) or norm <= 0:
        raise AssertionError(f"Invalid global gradient norm: {norm}")
    active = optimizer.step()
    if active != set(route):
        raise AssertionError(f"Wrong expert gradients: {active} != {set(route)}")
    after = adapter_hashes(model)
    for expert in ("high", "low"):
        if (before[expert] != after[expert]) != (expert in active):
            raise AssertionError(f"Incorrect active/inactive parameter update: {expert}")
    return dict(route=list(route), loss=losses, grad_norm=float(norm), active=sorted(active))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in ("0,1", "2,3"):
        raise ValueError("Explicit CUDA_VISIBLE_DEVICES=0,1 or 2,3 is required; GPU4 is forbidden")
    if not str(args.output_dir.resolve()).startswith("/data/gaoya/agent-data/"):
        raise ValueError("Probe artifacts must be placed under /data/gaoya/agent-data")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This probe requires exactly two torchrun ranks")
        if dist.get_rank() == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        mesh = init_device_mesh("cuda", (2,))
        emit("build_tiny_model")
        model = build_tiny(device, mesh)
        model.train()
        optimizer = LocalShardOptimizer(model, dict(
            optimizer="paged_adamw8bit", learning_rate=1e-3,
            betas=[0.9, 0.999], adam_epsilon=1e-8, weight_decay=0.01,
        ))
        frozen_before = frozen_hash(model)
        emit("per_frame_timestep_forward")
        evaluate(model, device)
        if frozen_hash(model) != frozen_before:
            raise AssertionError("Frozen model changed during eval")
        reports = []
        for index, route in enumerate((("high",), ("low",), ("high", "low", "high", "low"))):
            emit("train", route=route)
            reports.append(train_step(model, optimizer, device, route, seed=200 + index))
        if frozen_hash(model) != frozen_before:
            raise AssertionError("Frozen model received optimizer updates")
        eight_bit_states = sum(
            isinstance(state.get("state1"), torch.Tensor) and state["state1"].dtype == torch.uint8
            for state in optimizer.optimizer.state.values()
        )
        if eight_bit_states == 0:
            raise AssertionError("Probe did not exercise any bitsandbytes 8-bit optimizer states")
        emit("adapter_full_tensor_and_save", eight_bit_states=eight_bit_states)
        before_local = adapter_hashes(model)
        before_full = full_adapter_hash(model)
        before_outputs = evaluate(model, device)
        adapter_path = args.output_dir / "tiny-adapters.safetensors"
        save_adapter(model, adapter_path, metadata={"probe": True})
        optimizer_path = args.output_dir / f"tiny-optimizer-rank-{dist.get_rank()}.pt"
        optimizer_before = optimizer_hash(optimizer)
        torch.save(cpu_tree(optimizer.optimizer.state_dict()), optimizer_path)
        # Make a detectable mutation then restore through the production loader.
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.to_local().add_(0.125)
        if adapter_hashes(model) == before_local:
            raise AssertionError("Deliberate adapter mutation was ineffective")
        identity = model.initialization_identity
        model.initialization_identity = dict(identity, deliberately_wrong=True)
        rejected = False
        try:
            load_adapter(model, adapter_path, strict=True)
        except ValueError as error:
            if "initialization identity" not in str(error):
                raise
            rejected = True
        finally:
            model.initialization_identity = identity
        if not rejected:
            raise AssertionError("Strict loader accepted the wrong initialization identity")
        load_adapter(model, adapter_path, strict=True)
        if adapter_hashes(model) != before_local or full_adapter_hash(model) != before_full:
            raise AssertionError("Adapter full/local tensor round-trip mismatch")
        after_outputs = evaluate(model, device)
        for name in before_outputs:
            torch.testing.assert_close(before_outputs[name], after_outputs[name], atol=0, rtol=0)
        optimizer.optimizer.state.clear()
        payload = torch.load(optimizer_path, map_location=device, weights_only=True)
        optimizer.optimizer.load_state_dict(payload)
        if optimizer_hash(optimizer) != optimizer_before:
            raise AssertionError("Local optimizer state exact round-trip mismatch")
        emit("after_resume_cross_expert_step")
        reports.append(train_step(model, optimizer, device, ("low", "high"), seed=300))
        if frozen_hash(model) != frozen_before:
            raise AssertionError("Frozen model changed after resume")
        per_rank = dict(
            rank=dist.get_rank(), max_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            max_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
            eight_bit_optimizer_states=eight_bit_states,
            storage_rebindings=optimizer.rebindings, steps=reports,
        )
        results = [None, None]
        dist.all_gather_object(results, per_rank)
        if dist.get_rank() == 0:
            result = dict(
                status="PASS", tested_at=datetime.now(timezone.utc).isoformat(),
                physical_gpus=visible, torch_version=torch.__version__,
                trainable_parameters=sum(p["numel"] for p in trainable_parameter_manifest(model)),
                checks=["meta_to_cuda_initialization", "uniform_original_dtype_per_fsdp_group",
                        "per_frame_timesteps", "checkpoint_backward",
                        "high_only_update", "low_only_update", "cross_micro_expert_routing",
                        "fsdp2_global_gradient_clip", "bnb_uint8_optimizer_states",
                        "frozen_parameters_unchanged", "full_tensor_rank_equality",
                        "adapter_exact_roundtrip", "strict_identity_rejection",
                        "optimizer_exact_roundtrip", "optimizer_local_shard_reload_and_update"],
                full_adapter_sha256_before_resume=before_full, ranks=results,
                limitation="Tiny random models only; not A14B weight loading, cache equivalence or full-model VRAM proof",
            )
            (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
