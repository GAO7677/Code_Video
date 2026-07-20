"""DINOv3 ViT-L/16 adapter for the official xSSC model registry."""

import os
from pathlib import Path
import sys

from einops import rearrange
from safetensors import safe_open
import torch as pt
import torch.nn as nn


_EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DINOV3_ROOT = _EXPERIMENT_ROOT / "third_party" / "dinov3"
DEFAULT_DINOV3_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-sat493m/model.safetensors"
)


class DINO3ViT(nn.Module):
    """Frozen-feature wrapper backed by Meta's official DINOv3 implementation."""

    def __init__(
        self,
        model_name="dinov3_vitl16",
        checkpoint=DEFAULT_DINOV3_CHECKPOINT,
        source_root=DEFAULT_DINOV3_ROOT,
        in_size=256,
        rearrange=True,
        norm_out=False,
    ):
        super().__init__()
        if model_name != "dinov3_vitl16":
            raise ValueError(f"Unsupported DINOv3 model: {model_name}")
        if int(in_size) != 256:
            raise ValueError(f"This controlled experiment requires in_size=256, got {in_size}")

        source_root = Path(os.environ.get("DINOV3_ROOT", source_root)).expanduser().resolve()
        checkpoint = Path(os.environ.get("DINOV3_CHECKPOINT", checkpoint)).expanduser().resolve()
        if not (source_root / "dinov3").is_dir():
            raise FileNotFoundError(f"Official DINOv3 package not found: {source_root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

        from dinov3.hub.backbones import Weights, dinov3_vitl16

        model = dinov3_vitl16(pretrained=False, weights=Weights.SAT493M)
        load_report = self._load_huggingface_sat_weights(model, checkpoint)

        self.model = model
        self.in_size = int(in_size)
        self.patch_size = int(model.patch_size)
        self.out_size = self.in_size // self.patch_size
        self.embed_dim = int(model.embed_dim)
        self.rearrange = bool(rearrange)
        self.norm_out = bool(norm_out)
        self.checkpoint = str(checkpoint)
        self.source_root = str(source_root)
        self.load_report = load_report

        if self.patch_size != 16 or self.out_size != 16 or self.embed_dim != 1024:
            raise RuntimeError(
                "Unexpected DINOv3-L/16 geometry: "
                f"patch={self.patch_size}, grid={self.out_size}, dim={self.embed_dim}"
            )

    @staticmethod
    def _load_huggingface_sat_weights(model: nn.Module, checkpoint: Path) -> dict:
        target = model.state_dict()
        consumed = set()

        with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
            source_keys = set(source.keys())

            def copy(target_key, source_key, transform=None):
                value = source.get_tensor(source_key)
                consumed.add(source_key)
                if transform is not None:
                    value = transform(value)
                expected = target[target_key]
                if tuple(value.shape) != tuple(expected.shape):
                    raise RuntimeError(
                        f"DINOv3 shape mismatch for {source_key} -> {target_key}: "
                        f"{tuple(value.shape)} != {tuple(expected.shape)}"
                    )
                expected.copy_(value.to(dtype=expected.dtype))

            with pt.no_grad():
                copy("cls_token", "embeddings.cls_token")
                copy("storage_tokens", "embeddings.register_tokens")
                copy("mask_token", "embeddings.mask_token", lambda value: value.squeeze(1))
                copy("patch_embed.proj.weight", "embeddings.patch_embeddings.weight")
                copy("patch_embed.proj.bias", "embeddings.patch_embeddings.bias")
                copy("norm.weight", "norm.weight")
                copy("norm.bias", "norm.bias")

                for block_id in range(len(model.blocks)):
                    src = f"layer.{block_id}"
                    dst = f"blocks.{block_id}"
                    copy(f"{dst}.norm1.weight", f"{src}.norm1.weight")
                    copy(f"{dst}.norm1.bias", f"{src}.norm1.bias")
                    copy(f"{dst}.norm2.weight", f"{src}.norm2.weight")
                    copy(f"{dst}.norm2.bias", f"{src}.norm2.bias")
                    copy(f"{dst}.attn.proj.weight", f"{src}.attention.o_proj.weight")
                    copy(f"{dst}.attn.proj.bias", f"{src}.attention.o_proj.bias")
                    copy(f"{dst}.ls1.gamma", f"{src}.layer_scale1.lambda1")
                    copy(f"{dst}.ls2.gamma", f"{src}.layer_scale2.lambda1")
                    copy(f"{dst}.mlp.fc1.weight", f"{src}.mlp.up_proj.weight")
                    copy(f"{dst}.mlp.fc1.bias", f"{src}.mlp.up_proj.bias")
                    copy(f"{dst}.mlp.fc2.weight", f"{src}.mlp.down_proj.weight")
                    copy(f"{dst}.mlp.fc2.bias", f"{src}.mlp.down_proj.bias")

                    qkv_weight = target[f"{dst}.attn.qkv.weight"]
                    qkv_bias = target[f"{dst}.attn.qkv.bias"]
                    hidden_dim = qkv_weight.shape[1]
                    for index, projection in enumerate(("q", "k", "v")):
                        source_key = f"{src}.attention.{projection}_proj.weight"
                        value = source.get_tensor(source_key)
                        consumed.add(source_key)
                        expected = qkv_weight[index * hidden_dim : (index + 1) * hidden_dim]
                        if tuple(value.shape) != tuple(expected.shape):
                            raise RuntimeError(
                                f"DINOv3 QKV shape mismatch for {source_key}: "
                                f"{tuple(value.shape)} != {tuple(expected.shape)}"
                            )
                        expected.copy_(value.to(dtype=expected.dtype))

                    q_bias_key = f"{src}.attention.q_proj.bias"
                    v_bias_key = f"{src}.attention.v_proj.bias"
                    q_bias = source.get_tensor(q_bias_key)
                    v_bias = source.get_tensor(v_bias_key)
                    consumed.update((q_bias_key, v_bias_key))
                    qkv_bias[:hidden_dim].copy_(q_bias.to(dtype=qkv_bias.dtype))
                    qkv_bias[hidden_dim : 2 * hidden_dim].zero_()
                    qkv_bias[2 * hidden_dim :].copy_(v_bias.to(dtype=qkv_bias.dtype))

                if model.local_cls_norm is not None:
                    model.local_cls_norm.load_state_dict(model.norm.state_dict())

            unused = sorted(source_keys - consumed)
            missing = sorted(consumed - source_keys)
            if unused or missing:
                raise RuntimeError(
                    f"Incomplete DINOv3 conversion: unused={unused}, missing={missing}"
                )

        return {
            "format": "huggingface_safetensors",
            "source_tensor_count": len(consumed),
            "source_tensor_count_expected": 415,
            "all_source_tensors_consumed": len(consumed) == 415,
            "qkv_order": "q,k,v",
            "key_bias": False,
        }

    def forward(self, input):
        if input.ndim != 4 or tuple(input.shape[-2:]) != (self.in_size, self.in_size):
            raise ValueError(
                f"DINOv3 expects [B,3,{self.in_size},{self.in_size}], got {tuple(input.shape)}"
            )
        output = self.model.forward_features(input)
        if self.norm_out:
            feature = output["x_norm_patchtokens"]
        else:
            feature = output["x_prenorm"][:, self.model.n_storage_tokens + 1 :]
        if self.rearrange:
            feature = rearrange(feature, "b (h w) c -> b c h w", h=self.out_size, w=self.out_size)
        return feature
