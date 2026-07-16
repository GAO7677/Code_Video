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
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm


TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
os.chdir(TEXTOCVP_ROOT)

from base.baseTrainer import BaseTrainer  # noqa: E402
from data.load_data import unwrap_batch_data  # noqa: E402
from lib.logger import Logger, print_  # noqa: E402
from lib.setup_model import save_checkpoint  # noqa: E402
import lib.utils as utils  # noqa: E402


class StepValidationTrainer(BaseTrainer):
    def __init__(self, exp_path, max_optimizer_steps=None):
        super().__init__(exp_path=exp_path, checkpoint=None, resume_training=False)
        training = self.exp_params["training"]
        self.effective_batch_size = int(training["effective_batch_size"])
        self.validation_frequency = int(training["validation_frequency_steps"])
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
        if tracking.get("enabled", False):
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

    def _forward_per_sample_loss(self, batch_data):
        videos, others = unwrap_batch_data(self.exp_params, batch_data)
        metadata = others.pop("metadata", None)
        sources = others.pop("sources", None)
        videos = videos.to(self.device, non_blocking=True)
        output = self.model(x=videos, num_imgs=videos.shape[1], **others)
        reconstruction = output["recons_imgs"].clamp(0, 1)
        target = videos.clamp(0, 1)
        per_sample = F.mse_loss(reconstruction, target, reduction="none").mean((1, 2, 3, 4))
        return per_sample, sources, metadata

    @torch.no_grad()
    def validate_monitor(self):
        self.model.eval()
        losses_by_source = defaultdict(list)
        all_losses = []
        for batch_data in tqdm(self.valid_loader, desc=f"validate step {self.global_step}"):
            per_sample, sources, _ = self._forward_per_sample_loss(batch_data)
            values = per_sample.detach().cpu().tolist()
            all_losses.extend(values)
            if sources is not None:
                for source, value in zip(sources, values):
                    losses_by_source[source].append(value)
        self.model.train()
        aggregate = float(sum(all_losses) / len(all_losses))
        source_metrics = {
            source: float(sum(values) / len(values))
            for source, values in sorted(losses_by_source.items())
        }
        return aggregate, source_metrics

    def _save_step_checkpoint(self, epoch, name):
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
            "micro_global_batch_size": self.exp_params["training"]["batch_size"],
        }
        (Path(self.models_path) / f"{name}.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )

    def _append_metrics(self, record):
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
        val_loss, source_metrics = self.validate_monitor()
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
            "val_psnr": -10.0 * math.log10(max(val_loss, 1e-12)),
            "generalization_gap": val_loss - train_loss,
            "best_val_loss": self.best_val_loss,
            "no_improvement_count": self.no_improvement_count,
            "overfit_warning": overfit_warning,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "val_loss_pybullet": source_metrics.get("pybullet", float("nan")),
            "val_loss_kubric": source_metrics.get("kubric", float("nan")),
        }
        self._append_metrics(record)
        self.writer.log_full_dictionary(record, step=self.global_step, plot_name="Step Monitor")
        if self.wandb_run is not None:
            wandb_metrics = {
                "global_step": self.global_step,
                "val/reconstruction_mse": val_loss,
                "val/psnr": record["val_psnr"],
                "val/generalization_gap": record["generalization_gap"],
                "monitor/best_val_mse": self.best_val_loss,
                "monitor/no_improvement_count": self.no_improvement_count,
                "monitor/overfit_warning": int(overfit_warning),
                "train/reconstruction_mse_recent": train_loss,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            }
            for source, value in source_metrics.items():
                wandb_metrics[f"val/{source}_reconstruction_mse"] = value
                wandb_metrics[f"val/{source}_psnr"] = -10.0 * math.log10(
                    max(value, 1e-12)
                )
            self.wandb_run.log(wandb_metrics, step=self.global_step)
            self.wandb_run.summary["best_val_mse"] = self.best_val_loss
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
            accumulation_samples = 0
            accumulation_loss_sum = 0.0
            progress = tqdm(self.train_loader, desc=f"epoch {epoch + 1}/{num_epochs}")
            for batch_index, batch_data in enumerate(progress):
                per_sample, _, _ = self._forward_per_sample_loss(batch_data)
                micro_samples = int(per_sample.shape[0])
                loss = per_sample.mean()
                scaled_loss = loss * (micro_samples / self.effective_batch_size)
                scaled_loss.backward()
                accumulation_samples += micro_samples
                accumulation_loss_sum += float(loss.detach()) * micro_samples
                last_batch = batch_index + 1 == len(self.train_loader)
                if accumulation_samples < self.effective_batch_size and not last_batch:
                    continue

                if accumulation_samples != self.effective_batch_size:
                    self._rescale_gradients(
                        self.model, self.effective_batch_size / accumulation_samples
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
                step_loss = accumulation_loss_sum / accumulation_samples
                self.recent_step_losses.append(step_loss)
                self.writer.add_scalar(
                    name="Train Loss/optimizer_step",
                    val=step_loss,
                    step=self.global_step,
                )
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "global_step": self.global_step,
                            "train/reconstruction_mse": step_loss,
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch + 1,
                            "train/accumulated_samples": accumulation_samples,
                        },
                        step=self.global_step,
                    )
                progress.set_postfix(step=self.global_step, loss=f"{step_loss:.5f}")
                accumulation_samples = 0
                accumulation_loss_sum = 0.0

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
        if self.wandb_run is not None:
            self.wandb_run.summary["final_global_step"] = self.global_step
            self.wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-directory", type=Path, required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    utils.set_random_seed()
    Logger(exp_path=str(args.exp_directory))
    trainer = StepValidationTrainer(
        exp_path=str(args.exp_directory),
        max_optimizer_steps=args.max_optimizer_steps,
    )
    trainer.setup_model()
    trainer.load_data()
    trainer.training_loop()


if __name__ == "__main__":
    main()
