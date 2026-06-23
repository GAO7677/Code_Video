from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import time
import math

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
        object_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        debug_init = os.environ.get("CODEX_DEBUG_TRAINER_INIT", "").strip() not in {"", "0", "false", "False"}
        t0 = time.perf_counter()
        def _debug_log(message: str) -> None:
            if debug_init:
                elapsed = time.perf_counter() - t0
                print(f"[wan_bundle_init +{elapsed:.2f}s] {message}", flush=True)
        self.config = load_wan_config(task)
        self.device_obj = torch.device(device)
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha if lora_alpha > 0 else lora_rank)
        self.lora_dropout = float(lora_dropout)
        self.lora_init = str(lora_init)
        self.object_gate_init = float(object_gate_init)
        _debug_log("load wan helper classes")
        T5EncoderModel = load_wan_t5_encoder()
        Wan2_2_VAE = load_wan_vae()

        _debug_log("build text_encoder start")
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=f"{ckpt_dir}/{self.config.t5_checkpoint}",
            tokenizer_path=f"{ckpt_dir}/{self.config.t5_tokenizer}",
            shard_fn=None,
        )
        _debug_log("build text_encoder done")
        _debug_log("build vae start")
        self.vae = Wan2_2_VAE(
            vae_pth=f"{ckpt_dir}/{self.config.vae_checkpoint}",
            device=self.device_obj,
        )
        _debug_log("build vae done")
        self.dit = None
        if load_dit:
            _debug_log("ensure_dit_loaded start")
            self.ensure_dit_loaded()
            _debug_log("ensure_dit_loaded done")

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

    @staticmethod
    def _reinit_linear(module: nn.Module, *, std: float | None = None) -> None:
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if weight is None:
            return
        if std is None:
            nn.init.xavier_uniform_(weight)
        else:
            nn.init.normal_(weight, std=std)
        if bias is not None:
            nn.init.zeros_(bias)

    def _reinitialize_missing_object_branch(self, model: nn.Module) -> None:
        # The upstream Wan checkpoint has no object branch. Some missing tensors
        # end up effectively zeroed after loading, which collapses object
        # conditioning. Reinitialize those add-on layers explicitly.
        object_embedding = getattr(model, "object_embedding", None)
        if isinstance(object_embedding, nn.Sequential):
            for module in object_embedding.modules():
                if isinstance(module, nn.Linear):
                    self._reinit_linear(module, std=0.02)

        for block in getattr(model, "blocks", []):
            norm4 = getattr(block, "norm4", None)
            if isinstance(norm4, nn.LayerNorm):
                if norm4.weight is not None:
                    nn.init.ones_(norm4.weight)
                if norm4.bias is not None:
                    nn.init.zeros_(norm4.bias)

            object_gate = getattr(block, "object_gate", None)
            if isinstance(object_gate, torch.nn.Parameter):
                object_gate.data.fill_(self.object_gate_init)

            object_cross_attn = getattr(block, "object_cross_attn", None)
            if object_cross_attn is None:
                continue
            for attr in ("q", "k", "v", "o"):
                module = getattr(object_cross_attn, attr, None)
                if module is None:
                    continue
                if hasattr(module, "base_layer"):
                    self._reinit_linear(module.base_layer)
                elif isinstance(module, nn.Linear):
                    self._reinit_linear(module)
            for attr in ("norm_q", "norm_k"):
                module = getattr(object_cross_attn, attr, None)
                weight = getattr(module, "weight", None)
                if weight is not None:
                    nn.init.ones_(weight)

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
        debug_init = os.environ.get("CODEX_DEBUG_TRAINER_INIT", "").strip() not in {"", "0", "false", "False"}
        t0 = time.perf_counter()
        def _debug_log(message: str) -> None:
            if debug_init:
                elapsed = time.perf_counter() - t0
                print(f"[wan_dit_load +{elapsed:.2f}s] {message}", flush=True)
        WanModel = load_wan_model()
        target_dtype = getattr(self.config, "param_dtype", None)
        # The WAN backbone now includes extra object-conditioning layers that do
        # not exist in the upstream checkpoint. Loading with low_cpu_mem_usage
        # can leave those newly added parameters on the meta device, which then
        # breaks the subsequent .to(...) move during training/inference startup.
        pretrained_kwargs: dict[str, Any] = {"low_cpu_mem_usage": False}
        if target_dtype is not None:
            pretrained_kwargs["torch_dtype"] = target_dtype
        _debug_log(f"from_pretrained start ckpt_dir={self.ckpt_dir}")
        dit = WanModel.from_pretrained(self.ckpt_dir, **pretrained_kwargs)
        _debug_log("from_pretrained done")
        dit = self._apply_lora(dit)
        base_dit = dit.get_base_model() if hasattr(dit, "get_base_model") else dit
        self._reinitialize_missing_object_branch(base_dit)
        _debug_log(f"lora_applied rank={self.lora_rank}")
        self.dit = dit
        if target_dtype is not None:
            _debug_log(f"dit.to start device={self.device_obj} dtype={target_dtype}")
            self.dit.to(device=self.device_obj, dtype=target_dtype)
        else:
            _debug_log(f"dit.to start device={self.device_obj}")
            self.dit.to(self.device_obj)
        _debug_log("dit.to done")

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
                    is_object_param = (
                        ".object_cross_attn." in name
                        or ".object_gate" in name
                        or ".norm4." in name
                        or "object_embedding." in name
                    )
                    should_train = (is_lora_param and not freeze_lora) or is_object_param
                    param.requires_grad = should_train
                    if should_train:
                        # Trainable add-on weights must stay in fp32, otherwise AMP/GradScaler
                        # can hit bfloat16 unscale paths and abort during backward.
                        param.data = param.data.float()
            else:
                self.dit.eval().requires_grad_(False)
