from __future__ import annotations

import importlib.util
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from code_vjepa_vggt.utils.paths import ensure_upstream_paths

ensure_upstream_paths()


WAN_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main/wan")
WAN_MODULES_ROOT = WAN_ROOT / "modules"
WAN_CONFIGS_ROOT = WAN_ROOT / "configs"


def _ensure_fake_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _patch_wan_attention_fallback() -> None:
    attention_module = _load_module("wan.modules.attention", WAN_MODULES_ROOT / "attention.py")
    model_module = _load_module("wan.modules.model", WAN_MODULES_ROOT / "model.py")

    flash_attention_fn = attention_module.flash_attention
    attention_fn = attention_module.attention

    def safe_flash_attention(*args: Any, **kwargs: Any):
        try:
            return flash_attention_fn(*args, **kwargs)
        except AssertionError:
            return attention_fn(*args, **kwargs)

    attention_module.flash_attention = safe_flash_attention
    model_module.flash_attention = safe_flash_attention

    ulysses_name = "wan.distributed.ulysses"
    ulysses_path = WAN_ROOT / "distributed" / "ulysses.py"
    if ulysses_path.exists():
        ulysses_module = _load_module(ulysses_name, ulysses_path)
        ulysses_module.flash_attention = safe_flash_attention

    if not getattr(model_module, "_codex_bf16_activation_patch", False):
        import torch.nn.functional as F

        rope_apply = model_module.rope_apply
        WanLayerNorm = model_module.WanLayerNorm
        WanSelfAttention = model_module.WanSelfAttention
        WanCrossAttention = model_module.WanCrossAttention
        WanAttentionBlock = model_module.WanAttentionBlock
        Head = model_module.Head

        def safe_layer_norm_forward(self, x):
            weight = self.weight.float() if self.weight is not None else None
            bias = self.bias.float() if self.bias is not None else None
            out = F.layer_norm(
                x.float(),
                self.normalized_shape,
                weight,
                bias,
                self.eps,
            )
            return out.type_as(x)

        def safe_self_attn_forward(self, x, seq_lens, grid_sizes, freqs):
            b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
            q = self.norm_q(self.q(x.to(self.q.weight.dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(self.k.weight.dtype))).view(b, s, n, d)
            v = self.v(x.to(self.v.weight.dtype)).view(b, s, n, d)
            x_out = model_module.flash_attention(
                q=rope_apply(q, grid_sizes, freqs),
                k=rope_apply(k, grid_sizes, freqs),
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
            )
            x_out = x_out.flatten(2)
            x_out = self.o(x_out.to(self.o.weight.dtype))
            return x_out

        def safe_cross_attn_forward(self, x, context, context_lens):
            b, n, d = x.size(0), self.num_heads, self.head_dim
            q = self.norm_q(self.q(x.to(self.q.weight.dtype))).view(b, -1, n, d)
            k = self.norm_k(self.k(context.to(self.k.weight.dtype))).view(b, -1, n, d)
            v = self.v(context.to(self.v.weight.dtype)).view(b, -1, n, d)
            x_out = model_module.flash_attention(q, k, v, k_lens=context_lens)
            x_out = x_out.flatten(2)
            x_out = self.o(x_out.to(self.o.weight.dtype))
            return x_out

        def safe_block_forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
            assert e.dtype == torch.float32
            with torch.amp.autocast("cuda", dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            y = self.self_attn(
                self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
                seq_lens,
                grid_sizes,
                freqs,
            )
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[2].squeeze(2)

            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            ffn_in = self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
            ffn_in = ffn_in.to(self.ffn[0].weight.dtype)
            y = self.ffn(ffn_in)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        def safe_head_forward(self, x, e):
            assert e.dtype == torch.float32
            with torch.amp.autocast("cuda", dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            head_in = self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
            head_in = head_in.to(self.head.weight.dtype)
            return self.head(head_in)

        WanLayerNorm.forward = safe_layer_norm_forward
        WanSelfAttention.forward = safe_self_attn_forward
        WanCrossAttention.forward = safe_cross_attn_forward
        WanAttentionBlock.forward = safe_block_forward
        Head.forward = safe_head_forward
        model_module._codex_bf16_activation_patch = True


def ensure_wan_module_packages() -> None:
    _ensure_fake_package("wan", WAN_ROOT)
    _ensure_fake_package("wan.modules", WAN_MODULES_ROOT)
    _ensure_fake_package("wan.configs", WAN_CONFIGS_ROOT)


def load_wan_config(task: str):
    ensure_wan_module_packages()
    _load_module("wan.configs.shared_config", WAN_CONFIGS_ROOT / "shared_config.py")
    config_module_map = {
        "ti2v-5B": ("wan.configs.wan_ti2v_5B", WAN_CONFIGS_ROOT / "wan_ti2v_5B.py", "ti2v_5B"),
        "t2v-14B": ("wan.configs.wan_t2v_14B", WAN_CONFIGS_ROOT / "wan_t2v_14B.py", "t2v_14B"),
        "i2v-14B": ("wan.configs.wan_i2v_14B", WAN_CONFIGS_ROOT / "wan_i2v_14B.py", "i2v_14B"),
        "ti2v-14B": ("wan.configs.wan_ti2v_14B", WAN_CONFIGS_ROOT / "wan_ti2v_14B.py", "ti2v_14B"),
    }
    if task not in config_module_map:
        raise KeyError(f"unsupported Wan task config: {task}")
    module_name, module_path, attr_name = config_module_map[task]
    module = _load_module(module_name, module_path)
    return deepcopy(getattr(module, attr_name))


def load_wan_t5_encoder():
    ensure_wan_module_packages()
    _load_module("wan.modules.tokenizers", WAN_MODULES_ROOT / "tokenizers.py")
    module = _load_module("wan.modules.t5", WAN_MODULES_ROOT / "t5.py")
    return module.T5EncoderModel


def load_wan_vae():
    ensure_wan_module_packages()
    module = _load_module("wan.modules.vae2_2", WAN_MODULES_ROOT / "vae2_2.py")
    return module.Wan2_2_VAE


def load_wan_model():
    ensure_wan_module_packages()
    _patch_wan_attention_fallback()
    module = _load_module("wan.modules.model", WAN_MODULES_ROOT / "model.py")
    return module.WanModel
