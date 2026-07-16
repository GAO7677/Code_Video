from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .backbones import build_frozen_extractor
from .model import FeatureSlotDecomposer, feature_space_losses


TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
from data.Stage1Indexed import Stage1Indexed  # noqa: E402


@dataclass
class TrainConfig:
    space: str
    checkpoint: str
    output_dir: str
    index_root: str
    dataset_mode: str
    num_frames: int
    image_height: int
    image_width: int
    num_slots: int
    slot_dim: int
    per_gpu_batch_size: int
    effective_batch_size: int
    epochs: int
    validation_frequency_steps: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    warmup_ratio: float
    num_workers: int
    max_train_samples: int | None
    max_valid_samples: int | None
    max_optimizer_steps: int | None
    seed: int
    wandb_project: str
    wandb_group: str | None
    wandb_name: str | None
    disable_wandb: bool


class StridedDistributedSampler(Sampler[int]):
    """Shard validation without padding or duplicated examples."""

    def __init__(self, dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def parse_args(space: str) -> TrainConfig:
    default_frames = 10 if space == "vjepa" else 9
    default_checkpoint = (
        "/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth"
        if space == "vjepa"
        else "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    )
    parser = argparse.ArgumentParser(
        description=f"Train TextOCVP Stage 1 in frozen {space.upper()} feature space"
    )
    parser.add_argument("--checkpoint", default=default_checkpoint)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--index-root", default="/data/gaoya/AAA_test_video/0623_savi/indices"
    )
    parser.add_argument("--dataset-mode", choices=("pybullet", "kubric", "mixed"), default="mixed")
    parser.add_argument("--num-frames", type=int, default=default_frames)
    parser.add_argument("--image-height", type=int, default=216)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=256)
    parser.add_argument("--per-gpu-batch-size", type=int, default=1)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--validation-frequency-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--wandb-project", default="textocvp_feature_space_stage1")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()
    return TrainConfig(space=space, **vars(args))


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_sums(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def unwrap_model(model: torch.nn.Module) -> FeatureSlotDecomposer:
    return model.module if isinstance(model, DistributedDataParallel) else model


class FeatureSpaceTrainer:
    metric_names = (
        "loss",
        "feature_mse",
        "feature_cosine",
        "mask_entropy",
        "slot_usage_min",
        "slot_usage_max",
    )

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.rank, self.local_rank, self.world_size, self.device = setup_distributed()
        seed_everything(config.seed, self.rank)
        self.is_main = self.rank == 0
        self.output_dir = Path(config.output_dir).resolve()
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "models").mkdir(exist_ok=True)
            (self.output_dir / "metrics").mkdir(exist_ok=True)
            (self.output_dir / "config.json").write_text(
                json.dumps(asdict(config), indent=2), encoding="utf-8"
            )
        if dist.is_initialized():
            dist.barrier()

        global_micro_batch = config.per_gpu_batch_size * self.world_size
        if config.effective_batch_size % global_micro_batch != 0:
            raise ValueError(
                f"effective batch {config.effective_batch_size} must be divisible by "
                f"per_gpu_batch {config.per_gpu_batch_size} * world_size {self.world_size}"
            )
        self.accumulation_steps = config.effective_batch_size // global_micro_batch
        self._build_data()
        if len(self.train_loader) % self.accumulation_steps != 0:
            raise ValueError(
                f"train loader length {len(self.train_loader)} must divide accumulation "
                f"steps {self.accumulation_steps}; use a compatible batch size"
            )

        self.extractor = build_frozen_extractor(
            config.space,
            Path(config.checkpoint),
            self.device,
            config.num_frames,
        )
        core = FeatureSlotDecomposer(
            feature_dim=self.extractor.feature_dim,
            num_slots=config.num_slots,
            slot_dim=config.slot_dim,
        ).to(self.device)
        trainable_parameters = [parameter for parameter in core.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.steps_per_epoch = len(self.train_loader) // self.accumulation_steps
        planned_steps = self.steps_per_epoch * config.epochs
        self.total_steps = min(planned_steps, config.max_optimizer_steps or planned_steps)
        self.warmup_steps = max(1, round(self.total_steps * config.warmup_ratio))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._learning_rate_factor
        )
        if self.world_size > 1:
            self.decomposer = DistributedDataParallel(
                core,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.decomposer = core

        self.global_step = 0
        self.best_val_loss = math.inf
        self.no_improvement_count = 0
        self.previous_train_loss = None
        self.recent_train_losses = deque(maxlen=config.validation_frequency_steps)
        self.wandb_run = None
        if self.is_main and not config.disable_wandb:
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                group=config.wandb_group,
                name=config.wandb_name or f"{self.output_dir.name}-{config.dataset_mode}",
                job_type=f"{config.space}_space_stage1",
                dir=str(self.output_dir),
                config=asdict(config),
            )
            self.wandb_run.define_metric("global_step")
            self.wandb_run.define_metric("train/*", step_metric="global_step")
            self.wandb_run.define_metric("val/*", step_metric="global_step")
            self.wandb_run.define_metric("monitor/*", step_metric="global_step")

        if self.is_main:
            trainable = sum(parameter.numel() for parameter in trainable_parameters)
            runtime = {
                "world_size": self.world_size,
                "global_micro_batch": global_micro_batch,
                "accumulation_steps": self.accumulation_steps,
                "effective_batch_size": config.effective_batch_size,
                "steps_per_epoch": self.steps_per_epoch,
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps,
                "trainable_parameters": trainable,
                "frozen_checkpoint": config.checkpoint,
            }
            (self.output_dir / "runtime.json").write_text(
                json.dumps(runtime, indent=2), encoding="utf-8"
            )
            print(json.dumps(runtime, indent=2), flush=True)

    def _learning_rate_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step + 1) / self.warmup_steps
        denominator = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / denominator)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _build_data(self) -> None:
        dataset_kwargs = {
            "index_root": self.config.index_root,
            "dataset_mode": self.config.dataset_mode,
            "num_frames": self.config.num_frames,
            "img_size": (self.config.image_height, self.config.image_width),
            "frame_stride": 1,
            "random_start": True,
        }
        train_set = Stage1Indexed(
            split="train", max_samples=self.config.max_train_samples, **dataset_kwargs
        )
        valid_set = Stage1Indexed(
            split="valid", max_samples=self.config.max_valid_samples, **dataset_kwargs
        )
        self.train_sampler = DistributedSampler(
            train_set,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.config.seed,
            drop_last=False,
        )
        valid_sampler = StridedDistributedSampler(valid_set, self.rank, self.world_size)
        loader_kwargs = {
            "batch_size": self.config.per_gpu_batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": True,
            "persistent_workers": self.config.num_workers > 0,
            "collate_fn": Stage1Indexed.collate_fn,
        }
        self.train_loader = DataLoader(
            train_set, sampler=self.train_sampler, drop_last=False, **loader_kwargs
        )
        self.valid_loader = DataLoader(
            valid_set, sampler=valid_sampler, drop_last=False, **loader_kwargs
        )

    def _forward(self, videos: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        features = self.extractor(videos)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.decomposer(features)
        losses = feature_space_losses(output, features, self.config.space)
        if self.is_main and not (self.output_dir / "shape_trace.json").exists():
            trace = {
                "video": list(videos.shape),
                "frozen_features": list(features.shape),
                "slots": list(output["slots"].shape),
                "masks": list(output["masks"].shape),
                "reconstructed_features": list(output["reconstructed_features"].shape),
            }
            (self.output_dir / "shape_trace.json").write_text(
                json.dumps(trace, indent=2), encoding="utf-8"
            )
        return losses, output

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        core = unwrap_model(self.decomposer)
        core.eval()
        aggregate = defaultdict(float)
        by_source = {
            "pybullet": defaultdict(float),
            "kubric": defaultdict(float),
        }
        progress = tqdm(
            self.valid_loader,
            desc=f"validate step {self.global_step}",
            disable=not self.is_main,
        )
        for videos, info in progress:
            features = self.extractor(videos)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = core(features)
            losses = feature_space_losses(output, features, self.config.space)
            batch_size = videos.shape[0]
            for name, key in zip(self.metric_names, ("total", *self.metric_names[1:])):
                values = losses[key].detach().float().cpu()
                aggregate[name] += values.sum().item()
                for sample_index, source in enumerate(info["sources"]):
                    by_source[source][name] += values[sample_index].item()
            aggregate["count"] += batch_size
            for source in info["sources"]:
                by_source[source]["count"] += 1

        flat = [aggregate[name] for name in self.metric_names] + [aggregate["count"]]
        for source in ("pybullet", "kubric"):
            flat.extend(by_source[source][name] for name in self.metric_names)
            flat.append(by_source[source]["count"])
        reduced = reduce_sums(flat, self.device)
        offset = 0

        def consume() -> dict[str, float]:
            nonlocal offset
            sums = reduced[offset:offset + len(self.metric_names)]
            count = reduced[offset + len(self.metric_names)]
            offset += len(self.metric_names) + 1
            if count == 0:
                return {name: float("nan") for name in self.metric_names}
            return {name: value / count for name, value in zip(self.metric_names, sums)}

        metrics = {f"val/{key}": value for key, value in consume().items()}
        for source in ("pybullet", "kubric"):
            metrics.update(
                {f"val/{source}_{key}": value for key, value in consume().items()}
            )
        core.train()
        return metrics

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        if not self.is_main:
            return
        checkpoint = {
            "space": self.config.space,
            "global_step": self.global_step,
            "epoch": epoch,
            "decomposer_state_dict": unwrap_model(self.decomposer).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config),
        }
        path = self.output_dir / "models" / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def _record_validation(self, metrics: dict[str, float], epoch: int) -> None:
        val_loss = metrics["val/loss"]
        recent_train = (
            sum(self.recent_train_losses) / len(self.recent_train_losses)
            if self.recent_train_losses
            else float("nan")
        )
        improved = val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
            self.no_improvement_count = 0
            self._save_checkpoint("checkpoint_best_val.pth", epoch)
        else:
            self.no_improvement_count += 1
        train_improving = (
            self.previous_train_loss is not None
            and math.isfinite(recent_train)
            and recent_train < self.previous_train_loss
        )
        overfit_warning = (
            self.no_improvement_count >= 3
            and val_loss > self.best_val_loss * 1.02
            and train_improving
        )
        record = {
            "global_step": self.global_step,
            "epoch": epoch,
            "train/loss_recent": recent_train,
            **metrics,
            "monitor/generalization_gap": val_loss - recent_train,
            "monitor/best_val_loss": self.best_val_loss,
            "monitor/no_improvement_count": self.no_improvement_count,
            "monitor/overfit_warning": int(overfit_warning),
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        if self.is_main:
            jsonl_path = self.output_dir / "metrics" / "validation_metrics.jsonl"
            with jsonl_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, allow_nan=True) + "\n")
            csv_path = self.output_dir / "metrics" / "validation_metrics.csv"
            with csv_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(record))
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(record)
            if self.wandb_run is not None:
                self.wandb_run.log(record, step=self.global_step)
                self.wandb_run.summary["best_val_loss"] = self.best_val_loss
            print(json.dumps(record, allow_nan=True), flush=True)
        if math.isfinite(recent_train):
            self.previous_train_loss = recent_train
        self.recent_train_losses.clear()

    def run(self) -> None:
        initial_metrics = self.validate()
        self._record_validation(initial_metrics, epoch=0)
        self.optimizer.zero_grad(set_to_none=True)
        stop = False
        for epoch in range(self.config.epochs):
            self.train_sampler.set_epoch(epoch)
            self.decomposer.train()
            accumulation = defaultdict(list)
            progress = tqdm(
                self.train_loader,
                desc=f"epoch {epoch + 1}/{self.config.epochs}",
                disable=not self.is_main,
            )
            for batch_index, (videos, _) in enumerate(progress):
                update_now = (batch_index + 1) % self.accumulation_steps == 0
                sync_context = (
                    nullcontext()
                    if update_now or not isinstance(self.decomposer, DistributedDataParallel)
                    else self.decomposer.no_sync()
                )
                with sync_context:
                    losses, _ = self._forward(videos)
                    (losses["total"].mean() / self.accumulation_steps).backward()
                for key, value in losses.items():
                    accumulation[key].append(value.detach())
                if not update_now:
                    continue

                torch.nn.utils.clip_grad_norm_(
                    unwrap_model(self.decomposer).parameters(), self.config.gradient_clip
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

                local_values = []
                for key in ("total", *self.metric_names[1:]):
                    values = torch.cat(accumulation[key])
                    local_values.extend((values.sum().item(), values.numel()))
                reduced = reduce_sums(local_values, self.device)
                train_metrics = {}
                for metric_index, name in enumerate(self.metric_names):
                    value_sum, count = reduced[2 * metric_index:2 * metric_index + 2]
                    train_metrics[f"train/{name}"] = value_sum / count
                accumulation.clear()
                step_loss = train_metrics["train/loss"]
                self.recent_train_losses.append(step_loss)
                train_metrics.update(
                    {
                        "global_step": self.global_step,
                        "train/epoch": epoch + 1,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )
                if self.is_main:
                    progress.set_postfix(step=self.global_step, loss=f"{step_loss:.5f}")
                    if self.wandb_run is not None:
                        self.wandb_run.log(train_metrics, step=self.global_step)

                if self.global_step % self.config.validation_frequency_steps == 0:
                    metrics = self.validate()
                    self._record_validation(metrics, epoch=epoch + 1)
                    self._save_checkpoint(
                        f"checkpoint_step_{self.global_step:06d}.pth", epoch + 1
                    )
                if self.global_step >= self.total_steps:
                    stop = True
                    break
            if stop:
                break

        if self.global_step % self.config.validation_frequency_steps != 0:
            metrics = self.validate()
            self._record_validation(metrics, epoch=epoch + 1)
        self._save_checkpoint("checkpoint_final.pth", epoch + 1)
        if self.wandb_run is not None:
            self.wandb_run.summary["final_global_step"] = self.global_step
            self.wandb_run.finish()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def main(space: str) -> None:
    config = parse_args(space)
    if config.space == "vjepa" and config.num_frames % 2 != 0:
        raise ValueError("V-JEPA tubelet_size=2 requires an even number of frames")
    if config.space == "vae" and (config.num_frames - 1) % 4 != 0:
        raise ValueError("Wan VAE requires num_frames=4n+1, for example 9 or 13")
    trainer = FeatureSpaceTrainer(config)
    trainer.run()
