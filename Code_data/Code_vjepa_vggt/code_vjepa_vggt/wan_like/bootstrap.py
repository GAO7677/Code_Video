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


WAN22_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main/wan")
WAN21_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.1-main/wan")
WAN_ROOT = WAN22_ROOT
WAN_MODULES_ROOT = WAN_ROOT / "modules"
WAN_CONFIGS_ROOT = WAN_ROOT / "configs"


def _ensure_fake_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _reset_fake_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _drop_wan_module_cache(prefix: str) -> None:
    stale_keys = [name for name in sys.modules if name == prefix or name.startswith(prefix + ".")]
    for key in stale_keys:
        sys.modules.pop(key, None)


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


def _patch_wan_attention_fallback(modules_root: Path, wan_root: Path) -> None:
    attention_module = _load_module("wan.modules.attention", modules_root / "attention.py")
    model_module = _load_module("wan.modules.model", modules_root / "model.py")

    flash_attention_fn = attention_module.flash_attention
    rope_apply_fn = model_module.rope_apply

    def safe_flash_attention(*args: Any, **kwargs: Any):
        return flash_attention_fn(*args, **kwargs)

    def safe_rope_apply(x, grid_sizes, freqs):
        out = rope_apply_fn(x, grid_sizes, freqs)
        return out.to(dtype=x.dtype)

    attention_module.flash_attention = safe_flash_attention
    model_module.flash_attention = safe_flash_attention
    model_module.rope_apply = safe_rope_apply

    ulysses_name = "wan.distributed.ulysses"
    ulysses_path = wan_root / "distributed" / "ulysses.py"
    if ulysses_path.exists():
        ulysses_module = _load_module(ulysses_name, ulysses_path)
        ulysses_module.flash_attention = safe_flash_attention

    if not getattr(model_module, "_codex_bf16_activation_patch", False):
        import torch.nn.functional as F
        import torch.utils.checkpoint as checkpoint

        rope_apply = safe_rope_apply
        WanLayerNorm = model_module.WanLayerNorm
        WanSelfAttention = model_module.WanSelfAttention
        WanCrossAttention = model_module.WanCrossAttention
        WanAttentionBlock = model_module.WanAttentionBlock
        WanModel = model_module.WanModel
        Head = model_module.Head
        sinusoidal_embedding_1d = model_module.sinusoidal_embedding_1d

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
            x_bf16 = x.to(torch.bfloat16)
            q = self.norm_q(self.q(x_bf16)).view(b, s, n, d).to(torch.bfloat16)
            k = self.norm_k(self.k(x_bf16)).view(b, s, n, d).to(torch.bfloat16)
            v = self.v(x_bf16).view(b, s, n, d).to(torch.bfloat16)
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
            x_bf16 = x.to(torch.bfloat16)
            context_bf16 = context.to(torch.bfloat16)
            q = self.norm_q(self.q(x_bf16)).view(b, -1, n, d).to(torch.bfloat16)
            k = self.norm_k(self.k(context_bf16)).view(b, -1, n, d).to(torch.bfloat16)
            v = self.v(context_bf16).view(b, -1, n, d).to(torch.bfloat16)
            x_out = model_module.flash_attention(q, k, v, k_lens=context_lens)
            x_out = x_out.flatten(2)
            x_out = self.o(x_out.to(self.o.weight.dtype))
            return x_out

        def safe_block_forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            object_context=None,
            object_context_lens=None,
        ):
            assert e.dtype == torch.float32
            with torch.amp.autocast("cuda", dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                y = self.self_attn(
                    self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
                    seq_lens,
                    grid_sizes,
                    freqs,
                )
                x = x + y * e[2].squeeze(2)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x = x + self.cross_attn(self.norm3(x), context, context_lens)
                if object_context is not None:
                    object_delta = self.object_cross_attn(
                        self.norm4(x),
                        object_context,
                        object_context_lens,
                    )
                    with torch.amp.autocast("cuda", dtype=torch.float32):
                        x = x + object_delta * torch.tanh(self.object_gate)
                ffn_in = self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
                ffn_in = ffn_in.to(self.ffn[0].weight.dtype)
                y = self.ffn(ffn_in)
                x = x + y * e[5].squeeze(2)
            return x

        def safe_head_forward(self, x, e):
            assert e.dtype == torch.float32
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            head_in = self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
            head_in = head_in.to(self.head.weight.dtype)
            return self.head(head_in)

        def safe_model_forward(
            self,
            x,
            t,
            context=None,
            seq_len=None,
            y=None,
            text_context=None,
            object_context=None,
        ):
            if self.model_type == "i2v":
                assert y is not None
            if seq_len is None:
                raise ValueError("WanModel.forward requires seq_len")
            device = self.patch_embedding.weight.device
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)

            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            x = [self.patch_embedding(u.unsqueeze(0).to(self.patch_embedding.weight.dtype)) for u in x]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat(
                [
                    torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                    for u in x
                ]
            )

            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))

            if text_context is None:
                text_context = context
            if text_context is None:
                raise ValueError("WanModel.forward requires either context or text_context")

            def _embed_context(context_list, embedding, fixed_len=None):
                if context_list is None:
                    return None, None
                if len(context_list) == 0:
                    raise ValueError("context list must not be empty")
                lengths = torch.tensor(
                    [int(u.size(0)) for u in context_list],
                    dtype=torch.long,
                    device=device,
                )
                max_len = int(lengths.max().item()) if fixed_len is None else int(fixed_len)
                padded = []
                for u in context_list:
                    if int(u.size(0)) > max_len:
                        padded.append(u[:max_len])
                    else:
                        padded.append(
                            torch.cat([u, u.new_zeros(max_len - u.size(0), u.size(1))], dim=0)
                        )
                stacked = torch.stack(padded)
                embedded = embedding(stacked.to(embedding[0].weight.dtype))
                lens = lengths.clamp_max(max_len)
                if fixed_len is not None and bool(torch.all(lens == max_len)):
                    lens = None
                return embedded, lens

            context, context_lens = _embed_context(
                text_context,
                self.text_embedding,
                fixed_len=self.text_len,
            )
            object_context, object_context_lens = _embed_context(
                object_context,
                self.object_embedding,
                fixed_len=None,
            )

            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                object_context=object_context,
                object_context_lens=object_context_lens,
            )

            force_checkpoint = bool(getattr(self, "_codex_force_checkpointing", False))
            for block in self.blocks:
                if self.training or force_checkpoint:
                    x = checkpoint.checkpoint(
                        lambda x_in: block(x_in, **kwargs),
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, **kwargs)

            x = self.head(x, e)
            x = self.unpatchify(x, grid_sizes)
            return [u.float() for u in x]

        WanLayerNorm.forward = safe_layer_norm_forward
        WanSelfAttention.forward = safe_self_attn_forward
        WanCrossAttention.forward = safe_cross_attn_forward
        WanAttentionBlock.forward = safe_block_forward
        Head.forward = safe_head_forward
        WanModel.forward = safe_model_forward
        model_module._codex_bf16_activation_patch = True


def _select_wan_roots(task: str) -> tuple[Path, Path, Path]:
    if task == "t2v-1.3B":
        root = WAN21_ROOT
    else:
        root = WAN22_ROOT
    return root, root / "modules", root / "configs"


def ensure_wan_module_packages(task: str = "ti2v-5B") -> tuple[Path, Path, Path]:
    wan_root, modules_root, configs_root = _select_wan_roots(task)
    _drop_wan_module_cache("wan")
    _reset_fake_package("wan", wan_root)
    _reset_fake_package("wan.modules", modules_root)
    _reset_fake_package("wan.configs", configs_root)
    return wan_root, modules_root, configs_root


def load_wan_config(task: str):
    _, _, configs_root = ensure_wan_module_packages(task)
    _load_module("wan.configs.shared_config", configs_root / "shared_config.py")
    config_module_map = {
        "t2v-1.3B": ("wan.configs.wan_t2v_1_3B", configs_root / "wan_t2v_1_3B.py", "t2v_1_3B"),
        "ti2v-5B": ("wan.configs.wan_ti2v_5B", configs_root / "wan_ti2v_5B.py", "ti2v_5B"),
        "t2v-14B": ("wan.configs.wan_t2v_14B", configs_root / "wan_t2v_14B.py", "t2v_14B"),
        "i2v-14B": ("wan.configs.wan_i2v_14B", configs_root / "wan_i2v_14B.py", "i2v_14B"),
        "ti2v-14B": ("wan.configs.wan_ti2v_14B", configs_root / "wan_ti2v_14B.py", "ti2v_14B"),
    }
    if task not in config_module_map:
        raise KeyError(f"unsupported Wan task config: {task}")
    module_name, module_path, attr_name = config_module_map[task]
    module = _load_module(module_name, module_path)
    return deepcopy(getattr(module, attr_name))


def load_wan_t5_encoder():
    _, modules_root, _ = ensure_wan_module_packages("ti2v-5B")
    _load_module("wan.modules.tokenizers", modules_root / "tokenizers.py")
    module = _load_module("wan.modules.t5", modules_root / "t5.py")
    return module.T5EncoderModel


def load_wan_vae(task: str = "ti2v-5B"):
    _, modules_root, _ = ensure_wan_module_packages(task)
    if task == "t2v-1.3B":
        module = _load_module("wan.modules.vae2_1", modules_root / "vae.py")
        return module.WanVAE
    module = _load_module("wan.modules.vae2_2", modules_root / "vae2_2.py")
    return module.Wan2_2_VAE


def load_wan_model(task: str = "ti2v-5B"):
    # Always use the 2.2 WanModel implementation because this repo injects
    # object-conditioning layers into that backbone. For 1.3B we only swap the
    # config and VAE while keeping the object-branch-capable transformer class.
    wan_root, modules_root, _ = ensure_wan_module_packages("ti2v-5B")
    _patch_wan_attention_fallback(modules_root, wan_root)
    module = _load_module("wan.modules.model", modules_root / "model.py")
    return module.WanModel
