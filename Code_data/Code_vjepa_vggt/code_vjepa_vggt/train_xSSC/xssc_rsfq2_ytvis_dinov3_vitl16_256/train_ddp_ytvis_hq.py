#!/usr/bin/env python3
from argparse import ArgumentParser
from datetime import timedelta
import json
import os
from pathlib import Path
import random
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

    iterator = tqdm.tqdm(pack.dataset_t, disable=pack.rank != 0)
    for batch in iterator:
        if pack.step_count >= pack.max_step:
            break
        pack.batch = batch
        [callback.before_step(**pack) for callback in pack.callback_t]

        with torch.autocast("cuda", dtype=pack.amp_dtype):
            pack.output = pack.model(**pack)
            local_loss = pack.loss_fn_t(**pack)
        valid_loss = all(valid.sum() > 0 for _, valid in local_loss.values())
        if not valid_loss:
            raise RuntimeError(f"rank {pack.rank}: batch has no valid loss")
        with torch.autocast("cuda", dtype=pack.amp_dtype):
            loss = sum(value[valid].mean() for value, valid in local_loss.values())

        pack.optimiz.zero_grad(set_to_none=True)
        if pack.use_scaler:
            pack.scaler.scale(loss).backward()
            pack.scaler.unscale_(pack.optimiz)
        else:
            loss.backward()
        logged_loss = {
            key: (value.detach(), valid.detach())
            for key, (value, valid) in local_loss.items()
        }
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

        # Training segmentation metrics are detached from the loss. Compute them
        # after backward so the full-resolution mask does not overlap the graph.
        with torch.inference_mode(), torch.autocast("cuda", dtype=pack.amp_dtype):
            [callback.after_forward(**pack) for callback in pack.callback_t]
            local_acc = pack.acc_fn_t(**pack)
        pack.loss = gather_metrics(logged_loss)
        pack.acc = gather_metrics(local_acc)
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

    config_file = args.cfg_file
    if not config_file.is_absolute():
        config_file = (ROOT / config_file).resolve()
    cfg = Config.fromfile(config_file)
    rank, local_rank, world_size, device = setup_distributed(cfg)
    max_step = cfg.max_step if args.max_step is None else args.max_step
    if not 1 <= max_step <= cfg.total_step:
        raise ValueError(
            f"max_step must be in [1, {cfg.total_step}], got {max_step}"
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
        set_seed(worker_seed)

    generator = torch.Generator().manual_seed(worker_seed)
    dataload_t = DataLoader(
        dataset_t,
        batch_size=cfg.batch_size_t,
        sampler=sampler_t,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_t),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        drop_last=cfg.train_loader_drop_last,
    )

    dataset_v = build_from_config(cfg.dataset_v)
    val_indices = list(range(len(dataset_v)))[rank::world_size]
    dataload_v = DataLoader(
        Subset(dataset_v, val_indices),
        batch_size=cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    if args.ckpt_file is not None:
        model.load(args.ckpt_file, cfg.ckpt_map)
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
    pack.world_size = world_size
    pack.dataset_t = dataload_t
    pack.dataset_v = dataload_v
    pack.model = model_proxy
    pack.optimiz = optimiz
    pack.scaler = scaler
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
    pack.val_interval = cfg.val_interval
    pack.step_count = 0
    pack.last_gradient_norm = None
    pack.last_lr = None
    pack.step_metrics_file = save_path / "step_metrics.jsonl"
    pack.wabrun = wabrun

    [callback.before_train(**pack) for callback in pack.callback_t]
    epoch = 0
    validation_count = 0
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
    if rank == 0:
        final_checkpoint = save_path / "last.pth"
        model.save(final_checkpoint)
        summary = {
            "config": str(config_file),
            "gpu_ids": cfg.gpu_ids,
            "world_size": world_size,
            "batch_size_per_gpu": cfg.batch_size_t,
            "global_batch_size": cfg.batch_size_t * world_size,
            "total_step": cfg.total_step,
            "max_step": max_step,
            "completed_step": pack.step_count,
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
