from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from code_vjepa_vggt.utils.paths import ensure_upstream_paths
from code_vjepa_vggt.wan_like.bootstrap import load_wan_config

ensure_upstream_paths()

from wan.modules.model import WanModel  # type: ignore
from wan.modules.t5 import T5EncoderModel  # type: ignore
from wan.modules.vae2_2 import Wan2_2_VAE  # type: ignore


@dataclass
class WanBackboneBundle:
    config: Any
    vae: Wan2_2_VAE
    text_encoder: T5EncoderModel
    dit: WanModel


class WanContextVideoModel(nn.Module):
    def __init__(self, ckpt_dir: str, task: str = "ti2v-5B", device: str = "cuda") -> None:
        super().__init__()
        self.config = load_wan_config(task)
        self.device_obj = torch.device(device)

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
        self.dit = WanModel.from_pretrained(ckpt_dir)
        self.dit.to(self.device_obj)

    def freeze_parts(self, freeze_vae: bool, freeze_text_encoder: bool, freeze_dit: bool) -> None:
        if freeze_vae:
            self.vae.model.eval().requires_grad_(False)
        if freeze_text_encoder:
            self.text_encoder.model.eval().requires_grad_(False)
        if freeze_dit:
            self.dit.eval().requires_grad_(False)

