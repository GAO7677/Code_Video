#!/usr/bin/env python3
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import timedelta
from itertools import islice
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, Subset
import tqdm


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))


class DistributedModelProxy:
    def __init__(self, ddp_model, module):
        self.ddp_model = ddp_model
        self.module = module

    def __call__(self, **pack):
        return self.ddp_model(batch=dict(pack["batch"]))

    def train(self):
        self.ddp_model.train()

    def eval(self):
        self.ddp_model.eval()

    def parameters(self):
        return self.module.parameters()

    def save(self, save_file, weights_only=True, key=r".*"):
        self.module.save(save_file, weights_only, key)


class WandbCallbackProxy:
    """Translate official callback logs onto an optimizer-step W&B axis."""

    def __init__(self, run):
        self.run = run

    def log(self, data, step=None):
        payload = {}
        if step is not None:
            payload["optimizer_step"] = int(step)
        for key, value in data.items():
            if key.endswith("-val"):
                payload[f"val/{key[:-4]}"] = value
            else:
                payload[f"train/{key}_epoch"] = value
        self.run.log(payload)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cfg-file",
        type=Path,
        default=Path(
            "upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--ckpt-file", type=Path, default=None)
    parser.add_argument("--resume-file", type=Path, default=None)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    return parser.parse_args()


def setup_distributed(cfg):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=cfg.distributed_backend,
        timeout=timedelta(minutes=cfg.distributed_timeout_minutes),
        device_id=device,
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != cfg.expected_world_size:
        raise RuntimeError(
            f"expected {cfg.expected_world_size} processes for GPUs {cfg.gpu_ids}, "
            f"got {world_size}"
        )
    return rank, local_rank, world_size, device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gather_metrics(metrics):
    gathered = {}
    for key, (value, valid) in metrics.items():
        value = value.detach()
        valid = valid.detach()
        values = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        valids = [torch.empty_like(valid) for _ in range(dist.get_world_size())]
        dist.all_gather(values, value)
        dist.all_gather(valids, valid)
        gathered[key] = (torch.cat(values, dim=0), torch.cat(valids, dim=0))
    return gathered


def metric_means(metrics):
    means = {}
    for key, (value, valid) in metrics.items():
        selected = value[valid]
        if selected.numel() == 0:
            raise RuntimeError(f"metric {key} has no valid values")
        means[key] = float(selected.float().mean().item())
    return means


def checkpoint_path(save_path, step):
    return save_path / f"step-{step:06d}.pth"


def checkpoint_metadata_path(checkpoint_file):
    return checkpoint_file.with_suffix(".metadata.json")


def save_checkpoint(model, save_file, key=r".*", metadata=None):
    temporary_file = save_file.with_suffix(f"{save_file.suffix}.tmp")
    model.save(temporary_file, key=key)
    os.replace(temporary_file, save_file)
    if metadata is not None:
        metadata_file = checkpoint_metadata_path(save_file)
        temporary_metadata_file = metadata_file.with_suffix(
            f"{metadata_file.suffix}.tmp"
        )
        temporary_metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary_metadata_file, metadata_file)
    size_gib = save_file.stat().st_size / 1024**3
    print(
        f"[checkpoint] step={metadata['optimizer_step']} file={save_file} "
        f"size_gib={size_gib:.3f}",
        flush=True,
    )
    return save_file


def read_checkpoint_metadata(checkpoint_file):
    metadata_file = checkpoint_metadata_path(checkpoint_file)
    if not metadata_file.is_file():
        return {}
    return json.loads(metadata_file.read_text())


def load_model_state(checkpoint_file):
    checkpoint_file = checkpoint_file.resolve()
    source = torch.load(
        checkpoint_file, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(source, dict) or not all(
        isinstance(key, str) for key in source
    ):
        raise TypeError(f"Invalid model checkpoint: {checkpoint_file}")
    return source


def load_matching_checkpoint(
    model,
    checkpoint_file,
    exclude_patterns,
    allowed_missing_patterns=(),
    expected_source_variant=None,
):
    checkpoint_file = checkpoint_file.resolve()
    metadata = read_checkpoint_metadata(checkpoint_file)
    source_variant = metadata.get("variant_name")
    if expected_source_variant is not None and source_variant != expected_source_variant:
        raise RuntimeError(
            "Transfer checkpoint variant mismatch: "
            f"expected {expected_source_variant!r}, got {source_variant!r} from "
            f"{checkpoint_file}"
        )
    source = load_model_state(checkpoint_file)
    target = model.state_dict()
    excludes = [re.compile(pattern) for pattern in exclude_patterns]
    allowed_missing = [re.compile(pattern) for pattern in allowed_missing_patterns]
    matched = {}
    excluded = []
    unexpected = []
    shape_mismatch = []
    for key, value in source.items():
        if any(pattern.match(key) for pattern in excludes):
            excluded.append(key)
        elif key not in target:
            unexpected.append(key)
        elif target[key].shape != value.shape:
            shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(target[key].shape),
                }
            )
        else:
            matched[key] = value

    load_result = model.load_state_dict(matched, strict=False)
    missing_keys = sorted(load_result.missing_keys)
    disallowed_missing = [
        key
        for key in missing_keys
        if not any(pattern.match(key) for pattern in allowed_missing)
    ]
    report = {
        "checkpoint": str(checkpoint_file),
        "source_variant": source_variant,
        "matched_key_count": len(matched),
        "matched_parameter_count": sum(value.numel() for value in matched.values()),
        "excluded_keys": sorted(excluded),
        "unexpected_keys": sorted(unexpected),
        "shape_mismatches": shape_mismatch,
        "missing_keys": missing_keys,
        "disallowed_missing_keys": disallowed_missing,
    }
    if not matched or unexpected or shape_mismatch or disallowed_missing:
        raise RuntimeError(
            "Checkpoint did not pass strict compatibility checks:\n"
            + json.dumps(report, indent=2)
        )
    return report


def capture_runtime_state(pack):
    numpy_state = np.random.get_state()
    return {
        "python_random": random.getstate(),
        "numpy_random": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(pack.device),
        "data_generators": {
            key: generator.get_state()
            for key, generator in pack.data_generators.items()
        },
    }


def restore_runtime_state(runtime_state, generators, device):
    random.setstate(runtime_state["python_random"])
    numpy_state = runtime_state["numpy_random"]
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_state["keys"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(runtime_state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(runtime_state["torch_cuda"].cpu(), device)
    for key, state in runtime_state["data_generators"].items():
        generators[key].set_state(state.cpu())


def save_checkpoint_bundle(pack, save_file=None):
    runtime_states = [None] * pack.world_size if pack.rank == 0 else None
    dist.gather_object(
        capture_runtime_state(pack), runtime_states, dst=0
    )
    final_checkpoint = None
    if pack.rank == 0:
        if save_file is None:
            save_file = checkpoint_path(pack.save_path, pack.step_count)
        metadata = {
            "format": "xssc_model_checkpoint_v1",
            "variant_name": pack.variant_name,
            "config_file": str(pack.config_file),
            "optimizer_step": pack.step_count,
            "epoch": pack.epoch,
            "seed": pack.seed,
            "world_size": pack.world_size,
            "effective_global_batch_size": (
                pack.batch_size_per_gpu
                * pack.world_size
                * pack.gradient_accumulation_steps
            ),
        }
        final_checkpoint = save_checkpoint(
            pack.model,
            save_file,
            key=pack.checkpoint_key,
            metadata=metadata,
        )
        resume_state = {
            "format": "xssc_training_state_v1",
            **metadata,
            "model_checkpoint": final_checkpoint.name,
            "optimizer": pack.optimiz.state_dict(),
            "scaler": pack.scaler.state_dict(),
            "runtime_states": runtime_states,
            # Dataloader worker prefetch state is not serializable. Resume at
            # the next sampler epoch to avoid replaying a partial epoch.
            "resume_epoch": pack.epoch + 1,
            "data_resume_policy": "next_sampler_epoch",
        }
        resume_file = pack.save_path / "resume-latest.pth"
        temporary_resume_file = resume_file.with_suffix(".pth.tmp")
        torch.save(resume_state, temporary_resume_file)
        os.replace(temporary_resume_file, resume_file)
        current_step = pack.step_count
        for old_checkpoint in pack.save_path.glob("step-*.pth"):
            match = re.fullmatch(r"step-(\d+)\.pth", old_checkpoint.name)
            if match is None:
                continue
            old_step = int(match.group(1))
            if old_step == current_step or old_step in pack.checkpoint_keep_steps:
                continue
            old_metadata = checkpoint_metadata_path(old_checkpoint)
            old_checkpoint.unlink()
            if old_metadata.is_file():
                old_metadata.unlink()
            print(
                f"[checkpoint-retention] removed superseded step {old_step}: "
                f"{old_checkpoint}",
                flush=True,
            )
    dist.barrier()
    return final_checkpoint


def load_training_state(resume_file):
    resume_file = resume_file.resolve()
    state = torch.load(
        resume_file, map_location="cpu", weights_only=True, mmap=True
    )
    if state.get("format") != "xssc_training_state_v1":
        raise ValueError(f"Not an xSSC training-state checkpoint: {resume_file}")
    model_checkpoint = resume_file.parent / state["model_checkpoint"]
    if not model_checkpoint.is_file():
        raise FileNotFoundError(
            f"Resume model checkpoint is missing: {model_checkpoint}"
        )
    return state, model_checkpoint


def concatenate_metrics(metric_batches):
    return {
        key: (
            torch.cat([batch[key][0] for batch in metric_batches], dim=0),
            torch.cat([batch[key][1] for batch in metric_batches], dim=0),
        )
        for key in metric_batches[0]
    }


def run_after_epoch_callbacks(callbacks, pack):
    for callback in callbacks:
        if callback.__class__.__name__ in ("AverageLog", "HandleLog"):
            callback.idx = pack.step_count
        callback.after_epoch(**pack)


def build_callbacks(cfg, save_path, project, rank):
    import wandb

    from object_centric_bench.util import build_from_config

    wabrun = None
    if rank == 0 and project:
        wabrun = wandb.init(
            project=project,
            group=f"{Path('').cwd().name}/{save_path.parent.name}",
            name=f"{save_path.parent.name}-seed{save_path.name}",
            config=json.loads(json.dumps(cfg.__dict__, default=str)),
            dir=str(save_path),
            reinit="finish_previous",
        )
        wabrun.define_metric("optimizer_step")
        for metric_glob in ("train/*", "val/*", "system/*"):
            wabrun.define_metric(metric_glob, step_metric="optimizer_step")
        callback_wabrun = WandbCallbackProxy(wabrun)
        wandb_info = {
            "project": wabrun.project,
            "entity": wabrun.entity,
            "run_id": wabrun.id,
            "run_name": wabrun.name,
            "url": wabrun.url,
        }
        (save_path / "wandb_run.json").write_text(
            json.dumps(wandb_info, indent=2) + "\n"
        )
    else:
        callback_wabrun = None
    for callback_cfg in cfg.callback_t + cfg.callback_v:
        if callback_cfg.type.__name__ in ["AverageLog", "HandleLog"]:
            callback_cfg.log_file = f"{save_path}.txt"
            callback_cfg.epoch_key = "step_count"
            if callback_wabrun is not None:
                callback_cfg.wabrun = callback_wabrun
        elif callback_cfg.type.__name__ == "SaveModel":
            callback_cfg.save_dir = save_path
    callback_t = build_from_config(cfg.callback_t)
    callback_v = build_from_config(cfg.callback_v)
    if rank != 0:
        callback_t = callback_t[:1]
        callback_v = callback_v[:1]
    return callback_t, callback_v, wabrun


def train_epoch(pack, sampler, epoch):
    sampler.set_epoch(epoch)
    pack.model.train()
    pack.isval = False
    [callback.before_epoch(**pack) for callback in pack.callback_t]
    start_time = time.time()

    data_iterator = iter(pack.dataset_t)
    iterator = tqdm.tqdm(
        total=len(pack.dataset_t), disable=pack.rank != 0
    )
    while True:
        if pack.step_count >= pack.max_step:
            break
        batches = list(islice(data_iterator, pack.gradient_accumulation_steps))
        if not batches:
            break
        if (
            len(batches) < pack.gradient_accumulation_steps
            and pack.drop_incomplete_accumulation
        ):
            iterator.update(len(batches))
            break
        accumulation_count = len(batches)
        pack.optimiz.zero_grad(set_to_none=True)
        loss_batches = []
        acc_batches = []
        for accumulation_index, batch in enumerate(batches):
            pack.batch = batch
            [callback.before_step(**pack) for callback in pack.callback_t]
            synchronize = accumulation_index + 1 == accumulation_count
            sync_context = nullcontext() if synchronize else pack.model.ddp_model.no_sync()
            with sync_context:
                with torch.autocast("cuda", dtype=pack.amp_dtype):
                    pack.output = pack.model(**pack)
                    local_loss = pack.loss_fn_t(**pack)
                    valid_loss = all(
                        valid.sum() > 0 for _, valid in local_loss.values()
                    )
                    if not valid_loss:
                        raise RuntimeError(
                            f"rank {pack.rank}: batch has no valid loss"
                        )
                    loss = sum(
                        value[valid].mean()
                        for value, valid in local_loss.values()
                    ) / accumulation_count
                if pack.use_scaler:
                    pack.scaler.scale(loss).backward()
                else:
                    loss.backward()
            loss_batches.append(
                {
                    key: (value.detach(), valid.detach())
                    for key, (value, valid) in local_loss.items()
                }
            )
            # Segmentation metrics do not contribute gradients.
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=pack.amp_dtype
            ):
                [callback.after_forward(**pack) for callback in pack.callback_t]
                acc_batches.append(pack.acc_fn_t(**pack))
            iterator.update(1)

        if pack.use_scaler:
            pack.scaler.unscale_(pack.optimiz)
        gradient_norm = pack.gclip(pack.model.parameters())
        gradients_finite = bool(torch.isfinite(gradient_norm).item())
        if pack.use_scaler:
            scale_before = pack.scaler.get_scale()
            pack.scaler.step(pack.optimiz)
            pack.scaler.update()
            optimizer_step = pack.scaler.get_scale() >= scale_before
        else:
            pack.optimiz.step()
            optimizer_step = True
        if not gradients_finite or not optimizer_step:
            raise RuntimeError(
                f"rank {pack.rank}: non-finite gradient or skipped optimizer step"
            )

        pack.loss = gather_metrics(concatenate_metrics(loss_batches))
        pack.acc = gather_metrics(concatenate_metrics(acc_batches))
        [callback.after_step(**pack) for callback in pack.callback_t]
        pack.step_count += 1
        pack.last_gradient_norm = float(gradient_norm.detach().float().item())
        pack.last_lr = float(pack.optimiz.param_groups[0]["lr"])
        if pack.rank == 0:
            loss_means = metric_means(pack.loss)
            loss_total = sum(loss_means.values())
            clip_coefficient = min(
                1.0,
                pack.gclip.max_norm / (pack.last_gradient_norm + 1e-6),
            )
            step_metrics = {
                "optimizer_step": pack.step_count,
                "epoch": epoch,
                "system/gradient_accumulation_steps": (
                    pack.gradient_accumulation_steps
                ),
                "system/effective_global_batch_size": (
                    pack.batch_size_per_gpu
                    * pack.world_size
                    * pack.gradient_accumulation_steps
                ),
                "train/loss_total_step": loss_total,
                **{
                    f"train/{key}_step": value
                    for key, value in loss_means.items()
                },
                "train/gradient_l2_norm_before_clip": pack.last_gradient_norm,
                "train/gradient_clip_max_norm": float(pack.gclip.max_norm),
                "train/gradient_clip_coefficient": clip_coefficient,
                "train/lr": pack.last_lr,
                "system/peak_memory_reserved_gib_rank0": (
                    torch.cuda.max_memory_reserved() / 1024**3
                ),
            }
            with pack.step_metrics_file.open("a") as file:
                file.write(json.dumps(step_metrics) + "\n")
            if pack.wabrun is not None:
                pack.wabrun.log(step_metrics)
            iterator.set_postfix(
                step=pack.step_count,
                loss=loss_total,
                grad=pack.last_gradient_norm,
                lr=pack.last_lr,
            )

        if pack.step_count % pack.checkpoint_interval == 0:
            save_checkpoint_bundle(pack)

    run_after_epoch_callbacks(pack.callback_t, pack)
    if pack.rank == 0:
        elapsed = time.time() - start_time
        print(f"[train] epoch={epoch} step={pack.step_count} seconds={elapsed:.3f}")


@torch.inference_mode()
def val_epoch(pack):
    pack.model.eval()
    pack.isval = True
    [callback.before_epoch(**pack) for callback in pack.callback_v]

    iterator = tqdm.tqdm(pack.dataset_v, disable=pack.rank != 0)
    for batch in iterator:
        pack.batch = batch
        [callback.before_step(**pack) for callback in pack.callback_v]
        with torch.autocast("cuda", dtype=pack.amp_dtype):
            pack.output = pack.model(**pack)
            [callback.after_forward(**pack) for callback in pack.callback_v]
            local_loss = pack.loss_fn_v(**pack)
        local_acc = pack.acc_fn_v(**pack)
        pack.loss = gather_metrics(local_loss)
        pack.acc = gather_metrics(local_acc)
        [callback.after_step(**pack) for callback in pack.callback_v]

    run_after_epoch_callbacks(pack.callback_v, pack)


def main():
    args = parse_args()
    from object_centric_bench.datum import DataLoader
    from object_centric_bench.learn import MetricWrap
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    if args.ckpt_file is not None and args.resume_file is not None:
        raise ValueError("--ckpt-file and --resume-file are mutually exclusive")

    config_file = args.cfg_file
    if not config_file.is_absolute():
        config_file = (ROOT / config_file).resolve()
    cfg = Config.fromfile(config_file)
    rank, local_rank, world_size, device = setup_distributed(cfg)
    max_step = cfg.max_step if args.max_step is None else args.max_step
    resume_state = None
    resume_model_checkpoint = None
    if args.resume_file is not None:
        resume_state, resume_model_checkpoint = load_training_state(args.resume_file)
        if resume_state.get("variant_name") != cfg.variant_name:
            raise RuntimeError(
                "Resume variant mismatch: "
                f"checkpoint={resume_state.get('variant_name')!r}, "
                f"config={cfg.variant_name!r}"
            )
        start_step = int(resume_state["optimizer_step"])
        if args.start_step is not None and args.start_step != start_step:
            raise ValueError(
                f"--start-step={args.start_step} conflicts with resume step {start_step}"
            )
    else:
        start_step = cfg.start_step if args.start_step is None else args.start_step
    if not 1 <= max_step <= cfg.total_step:
        raise ValueError(
            f"max_step must be in [1, {cfg.total_step}], got {max_step}"
        )
    if not 0 <= start_step < max_step:
        raise ValueError(
            f"start_step must be in [0, {max_step}), got {start_step}"
        )
    checkpoint_interval = int(cfg.checkpoint_interval)
    if checkpoint_interval <= 0:
        raise ValueError(
            f"checkpoint_interval must be positive, got {checkpoint_interval}"
        )

    save_path = (args.save_dir / config_file.stem / str(args.seed)).resolve()
    if rank == 0:
        save_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_file, save_path.parent)
    dist.barrier()

    set_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.use_deterministic_algorithms(
        cfg.use_deterministic_algorithms, warn_only=True
    )
    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = args.data_dir

    dataset_t = build_from_config(cfg.dataset_t)
    sampler_t = DistributedSampler(
        dataset_t,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=cfg.train_sampler_drop_last,
    )
    worker_seed = args.seed + rank

    def worker_init_fn(_):
        seed = torch.initial_seed() % 2**32
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    generator_t = torch.Generator().manual_seed(worker_seed)
    generator_v = torch.Generator().manual_seed(worker_seed + 10_000)
    dataload_t = DataLoader(
        dataset_t,
        batch_size=cfg.batch_size_t,
        sampler=sampler_t,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_t),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator_t,
        drop_last=cfg.train_loader_drop_last,
    )

    dataset_v = build_from_config(cfg.dataset_v)
    all_val_indices = list(range(len(dataset_v)))
    val_subset_size = int(getattr(cfg, "val_subset_size", 0))
    val_subset_seed = int(getattr(cfg, "val_subset_seed", args.seed))
    if val_subset_size:
        if val_subset_size > len(all_val_indices):
            raise ValueError(
                f"val_subset_size={val_subset_size} exceeds dataset size "
                f"{len(all_val_indices)}"
            )
        subset_rng = random.Random(val_subset_seed)
        selected_val_indices = sorted(
            subset_rng.sample(all_val_indices, val_subset_size)
        )
    else:
        selected_val_indices = all_val_indices
    if rank == 0:
        (save_path / "val_subset.json").write_text(
            json.dumps(
                {
                    "dataset_size": len(dataset_v),
                    "subset_size": len(selected_val_indices),
                    "seed": val_subset_seed,
                    "indices": selected_val_indices,
                },
                indent=2,
            )
            + "\n"
        )
    val_indices = selected_val_indices[rank::world_size]
    dataload_v = DataLoader(
        Subset(dataset_v, val_indices),
        batch_size=cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator_v,
    )

    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    if resume_model_checkpoint is not None:
        load_report = load_matching_checkpoint(
            model,
            resume_model_checkpoint,
            exclude_patterns=(),
            allowed_missing_patterns=cfg.checkpoint_allowed_missing,
            expected_source_variant=cfg.variant_name,
        )
    elif args.ckpt_file is not None:
        transfer_load_exclude = getattr(cfg, "transfer_load_exclude", None)
        if transfer_load_exclude is None:
            ckpt_map = getattr(cfg, "ckpt_map", None)
            if ckpt_map:
                model.load(args.ckpt_file, ckpt_map)
                load_report = None
            else:
                load_report = load_matching_checkpoint(
                    model,
                    args.ckpt_file,
                    exclude_patterns=(),
                    allowed_missing_patterns=getattr(
                        cfg, "checkpoint_allowed_missing", ()
                    ),
                )
        else:
            load_report = load_matching_checkpoint(
                model,
                args.ckpt_file,
                transfer_load_exclude,
                allowed_missing_patterns=cfg.transfer_allowed_missing,
                expected_source_variant=cfg.transfer_expected_source_variant,
            )
    else:
        load_report = None
    if load_report is not None and rank == 0:
        (save_path / "checkpoint_load_report.json").write_text(
            json.dumps(load_report, indent=2) + "\n"
        )
        print(json.dumps(load_report, indent=2), flush=True)
    model.freez(cfg.freez, verbose=False)
    model = model.to(device)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    model_proxy = DistributedModelProxy(ddp_model, model)

    cfg.optimiz.params = model.parameters()
    optimiz = build_from_config(cfg.optimiz)
    amp_dtype = getattr(torch, cfg.amp_dtype)
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if resume_state is not None:
        optimiz.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
    gclip = build_from_config(cfg.gclip)
    loss_fn_t = MetricWrap(**build_from_config(cfg.loss_fn_t))
    loss_fn_v = MetricWrap(**build_from_config(cfg.loss_fn_v))
    acc_fn_t = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_t))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))
    callback_t, callback_v, wabrun = build_callbacks(
        cfg, save_path, args.project, rank
    )

    pack = Config({})
    pack.rank = rank
    pack.device = device
    pack.world_size = world_size
    pack.seed = args.seed
    pack.variant_name = cfg.variant_name
    pack.config_file = config_file
    pack.batch_size_per_gpu = cfg.batch_size_t
    pack.dataset_t = dataload_t
    pack.dataset_v = dataload_v
    pack.model = model_proxy
    pack.optimiz = optimiz
    pack.scaler = scaler
    pack.data_generators = {"train": generator_t, "val": generator_v}
    pack.use_scaler = use_scaler
    pack.gclip = gclip
    pack.amp_dtype = amp_dtype
    pack.loss_fn_t = loss_fn_t
    pack.loss_fn_v = loss_fn_v
    pack.acc_fn_t = acc_fn_t
    pack.acc_fn_v = acc_fn_v
    pack.callback_t = callback_t
    pack.callback_v = callback_v
    pack.total_step = cfg.total_step
    pack.max_step = max_step
    pack.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
    if pack.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    pack.drop_incomplete_accumulation = bool(
        getattr(cfg, "drop_incomplete_accumulation", False)
    )
    pack.checkpoint_interval = checkpoint_interval
    pack.checkpoint_keep_steps = set(
        int(step) for step in getattr(cfg, "checkpoint_keep_steps", ())
    )
    pack.checkpoint_key = cfg.checkpoint_key
    pack.save_path = save_path
    pack.val_interval = cfg.val_interval
    pack.step_count = start_step
    pack.last_gradient_norm = None
    pack.last_lr = None
    pack.step_metrics_file = save_path / "step_metrics.jsonl"
    pack.wabrun = wabrun

    if resume_state is not None:
        runtime_states = resume_state["runtime_states"]
        if len(runtime_states) != world_size:
            raise RuntimeError(
                f"Resume world size changed: {len(runtime_states)} -> {world_size}"
            )
        restore_runtime_state(
            runtime_states[rank], pack.data_generators, device
        )

    [callback.before_train(**pack) for callback in pack.callback_t]
    if rank == 0 and wabrun is not None:
        wabrun.log(
            {
                "optimizer_step": start_step,
                "system/transfer_start": 1,
                "system/effective_global_batch_size": (
                    cfg.batch_size_t
                    * world_size
                    * pack.gradient_accumulation_steps
                ),
            }
        )
    epoch = 0 if resume_state is None else int(resume_state["resume_epoch"])
    validation_count = start_step // pack.val_interval
    torch.cuda.reset_peak_memory_stats(device)
    while pack.step_count < pack.max_step:
        pack.epoch = epoch
        train_epoch(pack, sampler_t, epoch)
        should_validate = (
            pack.step_count >= (validation_count + 1) * pack.val_interval
            or pack.step_count >= pack.max_step
        )
        if should_validate:
            val_epoch(pack)
            validation_count += 1
        epoch += 1

    [callback.after_train(**pack) for callback in pack.callback_t]
    torch.cuda.synchronize(device)
    peak_reserved = torch.tensor(
        torch.cuda.max_memory_reserved(device) / 1024**3, device=device
    )
    dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)
    if pack.step_count % checkpoint_interval == 0:
        final_checkpoint = checkpoint_path(save_path, pack.step_count)
    else:
        final_checkpoint = save_path / "last.pth"
        save_checkpoint_bundle(pack, save_file=final_checkpoint)
    if rank == 0:
        summary = {
            "config": str(config_file),
            "gpu_ids": cfg.gpu_ids,
            "world_size": world_size,
            "batch_size_per_gpu": cfg.batch_size_t,
            "gradient_accumulation_steps": pack.gradient_accumulation_steps,
            "drop_incomplete_accumulation": pack.drop_incomplete_accumulation,
            "effective_global_batch_size": (
                cfg.batch_size_t
                * world_size
                * pack.gradient_accumulation_steps
            ),
            "total_step": cfg.total_step,
            "start_step": start_step,
            "max_step": max_step,
            "completed_step": pack.step_count,
            "checkpoint_interval": checkpoint_interval,
            "amp_dtype": cfg.amp_dtype,
            "last_lr": pack.last_lr,
            "last_gradient_l2_norm_before_clip": pack.last_gradient_norm,
            "peak_reserved_gib_max_rank": float(peak_reserved.item()),
            "checkpoint": str(final_checkpoint),
            "wandb_url": None if wabrun is None else wabrun.url,
        }
        (save_path / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
        if wabrun is not None:
            wabrun.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
