from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from code_vjepa_vggt.utils.paths import ensure_upstream_paths
from code_vjepa_vggt.wan_like.bootstrap import load_wan_config, load_wan_model, load_wan_t5_encoder, load_wan_vae

ensure_upstream_paths()


@dataclass
class WanBackboneBundle:
    config: Any
    vae: Any
    text_encoder: Any
    dit: Any


class WanContextVideoModel(nn.Module):
    def __init__(
        self,
        ckpt_dir: str,
        task: str = "ti2v-5B",
        device: str = "cuda",
        load_dit: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        lora_init: str = "gaussian",
    ) -> None:
        super().__init__()
        self.config = load_wan_config(task)
        self.device_obj = torch.device(device)
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha if lora_alpha > 0 else lora_rank)
        self.lora_dropout = float(lora_dropout)
        self.lora_init = str(lora_init)
        T5EncoderModel = load_wan_t5_encoder()
        Wan2_2_VAE = load_wan_vae()

        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=f"{ckpt_dir}/{self.config.t5_checkpoint}",
            tokenizer_path=f"{ckpt_dir}/{self.config.t5_tokenizer}",
            shard_fn=None,
        )
        self.vae = Wan2_2_VAE(
            vae_pth=f"{ckpt_dir}/{self.config.vae_checkpoint}",
            device=self.device_obj,
        )
        self.dit = None
        if load_dit:
            self.ensure_dit_loaded()

    @staticmethod
    def _wan_lora_target_modules(model: nn.Module) -> list[str]:
        targets: list[str] = []
        for name, module in model.named_modules():
            if "blocks" not in name:
                continue
            if "modulation" in name:
                continue
            if isinstance(module, nn.Linear):
                targets.append(name)
        if not targets:
            raise RuntimeError("no Wan transformer linear modules found for LoRA injection")
        return targets

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        if self.lora_rank <= 0:
            return model
        target_modules = self._wan_lora_target_modules(model)
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            init_lora_weights=self.lora_init,
            target_modules=target_modules,
        )
        model.requires_grad_(False)
        model = get_peft_model(model, lora_config)
        return model

    def ensure_dit_loaded(self) -> None:
        if self.dit is not None:
            return
        WanModel = load_wan_model()
        target_dtype = getattr(self.config, "param_dtype", None)
        pretrained_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if target_dtype is not None:
            pretrained_kwargs["torch_dtype"] = target_dtype
        dit = WanModel.from_pretrained(self.ckpt_dir, **pretrained_kwargs)
        dit = self._apply_lora(dit)
        self.dit = dit
        if target_dtype is not None:
            self.dit.to(device=self.device_obj, dtype=target_dtype)
        else:
            self.dit.to(self.device_obj)

    def load_lora_checkpoint(self, checkpoint_path: str | Path | None) -> dict[str, int] | None:
        if checkpoint_path is None:
            return None
        if self.dit is None:
            self.ensure_dit_loaded()
        if self.dit is None:
            raise RuntimeError("WAN DIT must be loaded before applying a LoRA checkpoint")
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {path}")

        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path), device="cpu")
        else:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise RuntimeError(f"unsupported LoRA checkpoint format: {path}")

        def _normalize_key(key: str) -> str:
            prefixes = ("module.", "base_model.", "model.", "bundle.", "bundle.dit.")
            normalized = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if normalized.startswith(prefix):
                        normalized = normalized[len(prefix) :]
                        changed = True
            return normalized

        checkpoint_by_normalized: dict[str, str] = {}
        for key in state.keys():
            if "lora_" not in key:
                continue
            checkpoint_by_normalized[_normalize_key(str(key))] = str(key)

        loaded = 0
        missing: list[str] = []
        trainable_named_params = [
            (name, param)
            for name, param in self.dit.named_parameters()
            if "lora_" in name
        ]
        for name, param in trainable_named_params:
            norm_name = _normalize_key(name)
            state_key = checkpoint_by_normalized.get(norm_name)
            if state_key is None:
                missing.append(name)
                continue
            tensor = state[state_key]
            if tuple(tensor.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"shape mismatch when loading LoRA checkpoint for {name}: "
                    f"checkpoint_shape={list(tensor.shape)} model_shape={list(param.shape)}"
                )
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            loaded += 1

        if missing:
            raise RuntimeError(
                f"LoRA checkpoint {path} is missing {len(missing)} trainable tensors; "
                f"first_missing={missing[0]}"
            )
        return {
            "loaded_lora_tensors": loaded,
            "checkpoint_lora_tensors": len(checkpoint_by_normalized),
        }

    def freeze_parts(
        self,
        freeze_vae: bool,
        freeze_text_encoder: bool,
        freeze_dit: bool,
        freeze_lora: bool = False,
    ) -> None:
        if freeze_vae:
            self.vae.model.eval().requires_grad_(False)
        if freeze_text_encoder:
            self.text_encoder.model.eval().requires_grad_(False)
        if freeze_dit and self.dit is not None:
            # Keep activation checkpointing enabled for frozen DIT training so
            # eval-mode forwards do not explode memory inside Wan attention.
            self.dit._codex_force_checkpointing = True
            base_dit = self.dit.get_base_model() if hasattr(self.dit, "get_base_model") else self.dit
            setattr(base_dit, "_codex_force_checkpointing", True)
            if self.lora_rank > 0:
                self.dit.eval()
                for name, param in self.dit.named_parameters():
                    is_lora_param = "lora_" in name
                    param.requires_grad = is_lora_param and not freeze_lora
                    if is_lora_param and not freeze_lora:
                        # Trainable LoRA weights must stay in fp32, otherwise AMP/GradScaler
                        # can hit bfloat16 unscale paths and abort during backward.
                        param.data = param.data.float()
            else:
                self.dit.eval().requires_grad_(False)
