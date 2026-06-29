from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.utils.tensors import trunc_normal_

from app.vjepa_2_1.models.utils.modules import Block
from app.vjepa_2_1.models.utils.patch_embed import PatchEmbed, PatchEmbed3D


class VisionTransformerArbitraryOut(nn.Module):
    """
    Local copy of the V-JEPA 2.1 video backbone with one targeted change:
    `out_layers` can be any valid block indices instead of only the official
    hierarchical export layers.

    For official hierarchical layers we preserve the learned output norms from
    the checkpoint. For arbitrary intermediate blocks we apply a parameter-free
    layer norm at readout time so checkpoint loading stays exact.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        is_causal=False,
        use_rope=False,
        init_type: str = "default",
        handle_nonsquare_inputs=True,
        img_temporal_dim_size=None,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        n_output_distillation=4,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.init_type = init_type
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (
                (num_frames // tubelet_size)
                * (img_size[0] // patch_size)
                * (img_size[1] // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        if self.img_temporal_dim_size is not None:
            if not isinstance(self.img_temporal_dim_size, int):
                raise ValueError(
                    f"img_temporal_dim_size must be an int, got {self.img_temporal_dim_size}"
                )
            self.patch_embed_img = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=1,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed_img = None

        self.uniform_power = uniform_power
        self.use_rope = use_rope
        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    n_registers=n_registers,
                    has_cls_first=has_cls_first,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

        self.attn_out = False
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        if depth == 12:
            self.hierarchical_layers = [2, 5, 8, 11]
            self.out_layers_distillation = [2, 5, 8, 11] if n_output_distillation == 4 else [11]
        elif depth == 24:
            self.hierarchical_layers = [5, 11, 17, 23]
            self.out_layers_distillation = [5, 11, 17, 23] if n_output_distillation == 4 else [23]
        elif depth == 40:
            self.hierarchical_layers = [9, 19, 29, 39]
            self.out_layers_distillation = [9, 19, 29, 39] if n_output_distillation == 4 else [39]
        elif depth == 48:
            self.hierarchical_layers = [11, 23, 37, 47]
            self.out_layers_distillation = [11, 23, 37, 47] if n_output_distillation == 4 else [47]
        else:
            raise ValueError(f"Unsupported depth for local V-JEPA backbone copy: {depth}")

        self.norms_block = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))])

        self.cls_token = None
        self.return_hierarchical = False

        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return
        if self.init_type == "default":
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif self.init_type == "xavier_uniform":
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif self.init_type == "xavier_normal":
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            raise ValueError(f"Unknown init type {self.init_type}")

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def check_temporal_dim(self, shape) -> bool:
        if self.img_temporal_dim_size is not None and shape[2] == self.img_temporal_dim_size:
            return True
        return False

    def _normalize_out_tokens(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx in self.hierarchical_layers:
            out_idx = self.hierarchical_layers.index(layer_idx)
            return self.norms_block[out_idx](x)
        return torch.nn.functional.layer_norm(x, (x.shape[-1],))

    def forward(self, x, masks=None, training=False):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            T = T // 1 if self.check_temporal_dim(x.shape) else T // self.tubelet_size
        else:
            raise ValueError(f"Unsupported input ndim: {x.ndim}")

        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if self.check_temporal_dim(x.shape):
            assert self.patch_embed_img is not None
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x += self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x += self.video_mod_embed.repeat(x.shape[0], 1, 1)

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        outs = []
        hier = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x, attn = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    use_reentrant=False,
                    return_attn=self.attn_out,
                    mode=mode,
                )
            else:
                x, attn = blk(
                    x,
                    mask=masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    return_attn=self.attn_out,
                    mode=mode,
                )

            if self.out_layers is not None and i in self.out_layers:
                outs.append(self._normalize_out_tokens(x, i))

            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if self.out_layers is not None:
            return outs

        if training or self.return_hierarchical:
            return torch.cat(hier, dim=2)
        return self.norms_block[-1](x)


def vit_large(patch_size=16, **kwargs):
    return VisionTransformerArbitraryOut(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def _clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = val
    return cleaned


def build_vjepa2_1_vit_large_384_encoder(
    *,
    checkpoint_path: str | Path,
    map_location: str = "cpu",
    out_layers: list[int] | None = None,
    img_size: int = 384,
    patch_size: int = 16,
    tubelet_size: int = 2,
    num_frames: int = 64,
) -> VisionTransformerArbitraryOut:
    encoder = vit_large(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        out_layers=out_layers,
        n_output_distillation=1,
    )

    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location=map_location)
    encoder_state_dict = _clean_backbone_key(checkpoint["ema_encoder"])
    msg = encoder.load_state_dict(encoder_state_dict, strict=True)
    print("Loaded local V-JEPA 2.1 Large encoder from ema_encoder")
    print(msg)
    return encoder
