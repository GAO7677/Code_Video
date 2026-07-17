#!/usr/bin/env python3
"""Official SAVi Stage 1 training with step validation and gradient accumulation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
os.chdir(TEXTOCVP_ROOT)

from base.baseTrainer import BaseTrainer  # noqa: E402
from CONFIG import CONFIG  # noqa: E402
import data as datalib  # noqa: E402
from data.load_data import unwrap_batch_data  # noqa: E402
from lib.loss import LossTracker  # noqa: E402
from lib.logger import Logger, print_  # noqa: E402
from lib.schedulers import WarmupVSScehdule  # noqa: E402
import lib.setup_model as setup_model_lib  # noqa: E402
from lib.setup_model import save_checkpoint  # noqa: E402
import lib.utils as utils  # noqa: E402
from feature_space_stage1.mask_loss import MaskLossWeights, compute_mask_loss  # noqa: E402


MASK_TARGET_KEYS = (
    "dynamic_instance_masks",
    "dynamic_instance_valid",
    "dynamic_union_mask",
    "static_geometry_mask",
    "mask_supervision_valid",
    "instance_supervision_valid",
)


class StepValidationTrainer(BaseTrainer):
    def __init__(self, exp_path, max_optimizer_steps=None):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl", device_id=torch.device("cuda", self.local_rank)
            )
        self.is_main = self.rank == 0
        super().__init__(exp_path=exp_path, checkpoint=None, resume_training=False)
        training = self.exp_params["training"]
        self.effective_batch_size = int(training["effective_batch_size"])
        if self.effective_batch_size % self.world_size != 0:
            raise ValueError("effective batch size must be divisible by DDP world size")
        self.effective_local_batch_size = self.effective_batch_size // self.world_size
        self.use_bf16 = str(training.get("mixed_precision", "none")).lower() == "bf16"
        self.validation_frequency = int(training["validation_frequency_steps"])
        self.mask_loss_weight = float(training.get("mask_loss_weight", 0.0))
        self.mask_loss_warmup_steps = int(training.get("mask_loss_warmup_steps", 500))
        self.mask_loss_weights = MaskLossWeights(
            union=float(training.get("mask_union_weight", 0.20)),
            instance=float(training.get("mask_instance_weight", 0.10)),
            static=float(training.get("mask_static_weight", 0.02)),
            background=float(training.get("mask_background_weight", 0.01)),
            unused=float(training.get("mask_unused_weight", 0.01)),
            focal_bce=float(training.get("mask_focal_bce_weight", 0.25)),
        )
        if self.mask_loss_weight < 0 or self.mask_loss_warmup_steps < 0:
            raise ValueError("mask loss weight and warmup steps must be non-negative")
        if self.mask_loss_weight > 0 and not self.exp_params["dataset"].get("load_masks", False):
            raise ValueError("mask loss requires dataset.load_masks=true")
        self.max_optimizer_steps = max_optimizer_steps
        self.global_step = 0
        self.best_val_loss = math.inf
        self.no_improvement_count = 0
        self.overfit_patience = int(training.get("overfit_patience_validations", 3))
        self.overfit_relative_degradation = float(
            training.get("overfit_relative_degradation", 0.02)
        )
        self.recent_step_losses = deque(maxlen=self.validation_frequency)
        self.previous_train_loss = None
        self.metrics_dir = Path(exp_path) / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl = self.metrics_dir / "step_metrics.jsonl"
        self.metrics_csv = self.metrics_dir / "step_metrics.csv"
        tracking = self.exp_params.get("wandb", {})
        self.wandb_run = None
        if self.is_main and tracking.get("enabled", False):
            wandb_dir = Path(exp_path) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            self.wandb_run = wandb.init(
                project=tracking["project"],
                entity=tracking.get("entity"),
                name=tracking["run_name"],
                group=tracking.get("group"),
                job_type="savi_stage1",
                dir=str(wandb_dir),
                config=self.exp_params,
                resume="allow",
            )
            self.wandb_run.define_metric("global_step")
            self.wandb_run.define_metric("train/*", step_metric="global_step")
            self.wandb_run.define_metric("val/*", step_metric="global_step")
            self.wandb_run.define_metric("monitor/*", step_metric="global_step")

    def setup_model(self):
        self.device = torch.device(
            "cuda", self.local_rank if self.distributed else 0
        )
        if self.is_main:
            print_(
                f"Using DDP world_size={self.world_size}, per-GPU device={self.device}, "
                f"bf16={self.use_bf16}"
            )
        model = setup_model_lib.setup_model(model_params=self.exp_params["model"])
        if self.is_main:
            utils.log_architecture(model, exp_path=self.exp_path)
        model = model.eval().to(self.device)
        optimizer, scheduler, lr_warmup = setup_model_lib.setup_optimizer(
            exp_params=self.exp_params, model=model
        )
        self.loss_tracker = LossTracker(loss_params=self.exp_params["loss"])
        self.epoch = 0
        if self.distributed:
            self.model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
        else:
            self.model = torch.nn.DataParallel(
                model, device_ids=list(range(torch.cuda.device_count()))
            ).to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = WarmupVSScehdule(
            optimizer=optimizer,
            lr_warmup=lr_warmup,
            scheduler=scheduler,
        )

    def load_data(self):
        if not self.distributed:
            return super().load_data()
        batch_size = int(self.exp_params["training"]["batch_size"])
        train_set = datalib.load_data(exp_params=self.exp_params, split="train")
        valid_set = datalib.load_data(exp_params=self.exp_params, split="valid")
        self.train_sampler = DistributedSampler(
            train_set,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=False,
        )
        self.valid_sampler = DistributedSampler(
            valid_set,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        )
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": CONFIG["num_workers"],
            "collate_fn": train_set.collate_fn,
            "pin_memory": True,
        }
        self.train_loader = DataLoader(
            train_set, sampler=self.train_sampler, **loader_kwargs
        )
        self.valid_loader = DataLoader(
            valid_set,
            sampler=self.valid_sampler,
            **{**loader_kwargs, "collate_fn": valid_set.collate_fn},
        )
        if self.is_main:
            print_(f"Examples in training set: {len(train_set)}")
            print_(f"Examples in validation set: {len(valid_set)}")

    def _mask_loss_ramp(self):
        if self.mask_loss_weight <= 0:
            return 0.0
        if self.mask_loss_warmup_steps == 0:
            return 1.0
        return min(1.0, self.global_step / self.mask_loss_warmup_steps)

    def _forward_per_sample_loss(self, batch_data, mask_ramp=None):
        videos, others = unwrap_batch_data(self.exp_params, batch_data)
        metadata = others.pop("metadata", None)
        sources = others.pop("sources", None)
        mask_targets = {
            key: others.pop(key)
            for key in MASK_TARGET_KEYS
            if key in others
        }
        videos = videos.to(self.device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16
        ):
            output = self.model(x=videos, num_imgs=videos.shape[1], **others)
        reconstruction = output["recons_imgs"].clamp(0, 1)
        target = videos.clamp(0, 1)
        reconstruction_mse = F.mse_loss(
            reconstruction.float(), target.float(), reduction="none"
        ).mean((1, 2, 3, 4))
        components = {"reconstruction_mse": reconstruction_mse}
        total = reconstruction_mse
        if self.mask_loss_weight > 0:
            missing = [key for key in MASK_TARGET_KEYS if key not in mask_targets]
            if missing:
                raise KeyError(f"Mask loss is enabled but batch is missing targets: {missing}")
            mask_targets = {
                key: value.to(self.device, non_blocking=True)
                for key, value in mask_targets.items()
            }
            mask_losses = compute_mask_loss(
                predicted_masks=output["masks"],
                weights=self.mask_loss_weights,
                **mask_targets,
            )
            ramp = self._mask_loss_ramp() if mask_ramp is None else float(mask_ramp)
            mask_scaled = self.mask_loss_weight * ramp * mask_losses["mask_total"]
            total = total + mask_scaled
            components.update(
                {
                    "mask_loss": mask_losses["mask_total"],
                    "mask_loss_scaled": mask_scaled,
                    "mask_union": mask_losses["mask_union"],
                    "mask_instance": mask_losses["mask_instance"],
                    "mask_static": mask_losses["mask_static"],
                    "mask_background": mask_losses["mask_background"],
                    "mask_unused": mask_losses["mask_unused"],
                    "mask_supervision_rate": mask_losses["mask_supervision_rate"],
                    "mask_instance_supervision_rate": mask_losses[
                        "mask_instance_supervision_rate"
                    ],
                    "mask_loss_ramp": torch.full_like(total, ramp),
                }
            )
        components["loss_total"] = total
        return total, sources, metadata, components

    @torch.no_grad()
    def validate_monitor(self):
        self.model.eval()
        sums_by_metric = defaultdict(float)
        counts_by_metric = defaultdict(int)
        sums_by_source = defaultdict(lambda: defaultdict(float))
        counts_by_source = defaultdict(lambda: defaultdict(int))
        for batch_data in tqdm(
            self.valid_loader,
            desc=f"validate step {self.global_step}",
            disable=not self.is_main,
        ):
            _, sources, _, components = self._forward_per_sample_loss(
                batch_data, mask_ramp=1.0
            )
            for name, tensor in components.items():
                values = tensor.detach().float().cpu().tolist()
                sums_by_metric[name] += sum(values)
                counts_by_metric[name] += len(values)
                if sources is not None:
                    for source, value in zip(sources, values):
                        sums_by_source[source][name] += value
                        counts_by_source[source][name] += 1
        self.model.train()
        metric_names = sorted(sums_by_metric)
        source_names = ("pybullet", "kubric")
        flat = []
        for name in metric_names:
            flat.extend((sums_by_metric[name], counts_by_metric[name]))
        for source in source_names:
            for name in metric_names:
                flat.extend(
                    (
                        sums_by_source[source][name],
                        counts_by_source[source][name],
                    )
                )
        reduced = torch.tensor(flat, device=self.device, dtype=torch.float64)
        if self.distributed:
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        values = reduced.cpu().tolist()
        offset = 0

        def consume():
            nonlocal offset
            result = {}
            for name in metric_names:
                value_sum, count = values[offset : offset + 2]
                offset += 2
                result[name] = value_sum / count if count else float("nan")
            return result

        aggregate = consume()
        source_metrics = {source: consume() for source in source_names}
        return aggregate, source_metrics

    def _save_step_checkpoint(self, epoch, name):
        if not self.is_main:
            return
        save_checkpoint(
            model=self.model.module,
            optimizer=self.optimizer,
            scheduler=self.warmup_scheduler.scheduler,
            lr_warmup=self.warmup_scheduler.lr_warmup,
            epoch=epoch,
            exp_path=self.exp_path,
            savedir="models",
            savename=name,
        )
        sidecar = {
            "global_step": self.global_step,
            "epoch": epoch,
            "effective_batch_size": self.effective_batch_size,
            "per_gpu_batch_size": self.exp_params["training"]["batch_size"],
            "world_size": self.world_size,
            "mixed_precision": "bf16" if self.use_bf16 else "none",
        }
        (Path(self.models_path) / f"{name}.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )

    def _append_metrics(self, record):
        if not self.is_main:
            return
        with self.metrics_jsonl.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
        fieldnames = list(record.keys())
        write_header = not self.metrics_csv.exists()
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    def validate_save_and_monitor(self, epoch, save_regular=True):
        train_loss = (
            float(sum(self.recent_step_losses) / len(self.recent_step_losses))
            if self.recent_step_losses
            else float("nan")
        )
        val_metrics, source_metrics = self.validate_monitor()
        val_loss = val_metrics["loss_total"]
        val_reconstruction = val_metrics["reconstruction_mse"]
        improved = val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
            self.no_improvement_count = 0
            self._save_step_checkpoint(epoch, "checkpoint_best_val.pth")
        else:
            self.no_improvement_count += 1
        train_still_improving = (
            self.previous_train_loss is not None
            and math.isfinite(train_loss)
            and train_loss < self.previous_train_loss
        )
        degraded = val_loss > self.best_val_loss * (1 + self.overfit_relative_degradation)
        overfit_warning = bool(
            self.no_improvement_count >= self.overfit_patience
            and degraded
            and train_still_improving
        )
        record = {
            "global_step": self.global_step,
            "epoch": epoch,
            "train_loss_recent": train_loss,
            "val_loss": val_loss,
            "val_reconstruction_mse": val_reconstruction,
            "val_psnr": -10.0 * math.log10(max(val_reconstruction, 1e-12)),
            "generalization_gap": val_loss - train_loss,
            "best_val_loss": self.best_val_loss,
            "no_improvement_count": self.no_improvement_count,
            "overfit_warning": overfit_warning,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "val_loss_pybullet": source_metrics.get("pybullet", {}).get(
                "loss_total", float("nan")
            ),
            "val_loss_kubric": source_metrics.get("kubric", {}).get(
                "loss_total", float("nan")
            ),
        }
        for name, value in val_metrics.items():
            record[f"val_{name}"] = value
        self._append_metrics(record)
        if self.is_main:
            self.writer.log_full_dictionary(
                record, step=self.global_step, plot_name="Step Monitor"
            )
        if self.is_main and self.wandb_run is not None:
            wandb_metrics = {
                "global_step": self.global_step,
                "val/loss_total": val_loss,
                "val/reconstruction_mse": val_reconstruction,
                "val/psnr": record["val_psnr"],
                "val/generalization_gap": record["generalization_gap"],
                "monitor/best_val_mse": self.best_val_loss,
                "monitor/no_improvement_count": self.no_improvement_count,
                "monitor/overfit_warning": int(overfit_warning),
                "train/reconstruction_mse_recent": train_loss,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            }
            for source, metrics in source_metrics.items():
                value = metrics["reconstruction_mse"]
                wandb_metrics[f"val/{source}_reconstruction_mse"] = value
                wandb_metrics[f"val/{source}_loss_total"] = metrics["loss_total"]
                wandb_metrics[f"val/{source}_psnr"] = -10.0 * math.log10(
                    max(value, 1e-12)
                )
            for name, value in val_metrics.items():
                wandb_metrics[f"val/{name}"] = value
            self.wandb_run.log(wandb_metrics, step=self.global_step)
            self.wandb_run.summary["best_val_mse"] = self.best_val_loss
        if self.is_main:
            print_(json.dumps(record, ensure_ascii=True))
        if save_regular and self.global_step > 0:
            self._save_step_checkpoint(
                epoch, f"checkpoint_step_{self.global_step:06d}.pth"
            )
        if math.isfinite(train_loss):
            self.previous_train_loss = train_loss
        self.recent_step_losses.clear()

    @staticmethod
    def _rescale_gradients(model, scale):
        if scale == 1.0:
            return
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    def training_loop(self):
        num_epochs = int(self.exp_params["training"]["num_epochs"])
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.validate_save_and_monitor(epoch=0, save_regular=False)
        stop = False
        for epoch in range(num_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            accumulation_samples = 0
            accumulation_component_sums = defaultdict(float)
            progress = tqdm(
                self.train_loader,
                desc=f"epoch {epoch + 1}/{num_epochs}",
                disable=not self.is_main,
            )
            for batch_index, batch_data in enumerate(progress):
                local_batch_size = int(batch_data[0].shape[0])
                last_batch = batch_index + 1 == len(self.train_loader)
                update_now = (
                    accumulation_samples + local_batch_size
                    >= self.effective_local_batch_size
                    or last_batch
                )
                sync_context = (
                    nullcontext()
                    if update_now or not self.distributed
                    else self.model.no_sync()
                )
                with sync_context:
                    per_sample, _, _, components = self._forward_per_sample_loss(batch_data)
                    micro_samples = int(per_sample.shape[0])
                    loss = per_sample.mean()
                    scaled_loss = loss * (
                        micro_samples / self.effective_local_batch_size
                    )
                    scaled_loss.backward()
                accumulation_samples += micro_samples
                for name, tensor in components.items():
                    accumulation_component_sums[name] += float(tensor.detach().sum())
                if not update_now:
                    continue

                if accumulation_samples != self.effective_local_batch_size:
                    self._rescale_gradients(
                        self.model,
                        self.effective_local_batch_size / accumulation_samples,
                    )
                self.warmup_scheduler(
                    iter=self.global_step,
                    epoch=epoch,
                    exp_params=self.exp_params,
                    end_epoch=False,
                )
                if self.exp_params["training"]["gradient_clipping"]:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.exp_params["training"]["clipping_max_value"],
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                step_metrics = {
                    name: value / accumulation_samples
                    for name, value in accumulation_component_sums.items()
                }
                if self.distributed:
                    names = sorted(step_metrics)
                    tensor = torch.tensor(
                        [step_metrics[name] for name in names],
                        device=self.device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    tensor.div_(self.world_size)
                    step_metrics = dict(zip(names, tensor.cpu().tolist()))
                step_loss = step_metrics["loss_total"]
                self.recent_step_losses.append(step_loss)
                if self.is_main:
                    self.writer.add_scalar(
                        name="Train Loss/optimizer_step",
                        val=step_loss,
                        step=self.global_step,
                    )
                    for name, value in step_metrics.items():
                        self.writer.add_scalar(
                            name=f"Train Components/{name}",
                            val=value,
                            step=self.global_step,
                        )
                if self.is_main and self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "global_step": self.global_step,
                            **{f"train/{name}": value for name, value in step_metrics.items()},
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch + 1,
                            "train/accumulated_samples": accumulation_samples,
                        },
                        step=self.global_step,
                    )
                if self.is_main:
                    progress.set_postfix(step=self.global_step, loss=f"{step_loss:.5f}")
                accumulation_samples = 0
                accumulation_component_sums.clear()

                if self.global_step % self.validation_frequency == 0:
                    self.validate_save_and_monitor(epoch=epoch, save_regular=True)
                if (
                    self.max_optimizer_steps is not None
                    and self.global_step >= self.max_optimizer_steps
                ):
                    stop = True
                    break
            if stop:
                break

        if self.global_step % self.validation_frequency != 0:
            self.validate_save_and_monitor(epoch=epoch, save_regular=False)
        self._save_step_checkpoint(epoch, "checkpoint_final.pth")
        if self.is_main and self.wandb_run is not None:
            self.wandb_run.summary["final_global_step"] = self.global_step
            self.wandb_run.finish()
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-directory", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    utils.set_random_seed(14 + rank)
    Logger(
        exp_path=str(args.exp_directory),
        file_name="logs.txt" if rank == 0 else f"logs_rank{rank}.txt",
    )
    trainer = StepValidationTrainer(
        exp_path=str(args.exp_directory),
        max_optimizer_steps=args.max_optimizer_steps,
    )
    trainer.setup_model()
    trainer.load_data()
    trainer.training_loop()


if __name__ == "__main__":
    main()
