from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

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
    def __init__(self, ckpt_dir: str, task: str = "ti2v-5B", device: str = "cuda", load_dit: bool = True) -> None:
        super().__init__()
        self.config = load_wan_config(task)
        self.device_obj = torch.device(device)
        self.ckpt_dir = ckpt_dir
        self.task = task
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

    def ensure_dit_loaded(self) -> None:
        if self.dit is not None:
            return
        WanModel = load_wan_model()
        self.dit = WanModel.from_pretrained(self.ckpt_dir)
        self.dit.to(self.device_obj)

    def freeze_parts(self, freeze_vae: bool, freeze_text_encoder: bool, freeze_dit: bool) -> None:
        if freeze_vae:
            self.vae.model.eval().requires_grad_(False)
        if freeze_text_encoder:
            self.text_encoder.model.eval().requires_grad_(False)
        if freeze_dit and self.dit is not None:
            self.dit.eval().requires_grad_(False)
