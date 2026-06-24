"""该模块用于给 Wan 视频管线补充多帧上下文条件训练与推理逻辑；输入为 Wan 管线、上下文/噪声 latent 与条件张量，输出为上下文感知的损失计算和生成所需的中间结果。"""
import types
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from typing_extensions import Literal

from diffsynth.core import ModelConfig, gradient_checkpoint_forward
from diffsynth.core.device.npu_compatible_device import get_device_type
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.models.longcat_video_dit import LongCatVideoTransformer3DModel
from diffsynth.models.wan_video_dit import CrossAttention, WanModel, modulate, sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video import (
    TeaCache,
    TemporalTiler_BCTHW,
    WanVideoPipeline,
    model_fn_longcat_video,
    model_fn_wans2v,
    wantodance_get_single_freqs,
)


def _reinit_linear(module: nn.Module, std: float = 0.02) -> None:
    if not isinstance(module, nn.Linear):
        return
    nn.init.normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def enable_object_condition_branch(
    dit: WanModel,
    *,
    object_gate_init: float = 0.1,
    reinitialize_object_branch: bool = True,
) -> WanModel:
    if getattr(dit, "_codex_object_branch_enabled", False):
        return dit

    if not hasattr(dit, "text_embedding"):
        raise RuntimeError("Expected Wan DIT with text_embedding before enabling object branch.")

    text_embedding = getattr(dit, "text_embedding")
    if not isinstance(text_embedding, nn.Sequential):
        raise RuntimeError("Expected dit.text_embedding to be nn.Sequential.")
    first_linear = None
    for module in text_embedding.modules():
        if isinstance(module, nn.Linear):
            first_linear = module
            break
    if first_linear is None:
        raise RuntimeError("Failed to infer object embedding input dim from dit.text_embedding.")
    text_dim = int(first_linear.in_features)
    dim = int(dit.dim)
    num_heads = int(dit.blocks[0].num_heads)

    dit.object_embedding = nn.Sequential(
        nn.Linear(text_dim, dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(dim, dim),
    ).to(device=dit.patch_embedding.weight.device, dtype=dit.patch_embedding.weight.dtype)
    if reinitialize_object_branch:
        for module in dit.object_embedding.modules():
            _reinit_linear(module)

    for block in dit.blocks:
        block.norm4 = nn.LayerNorm(dim, eps=1e-6).to(
            device=dit.patch_embedding.weight.device,
            dtype=dit.patch_embedding.weight.dtype,
        )
        block.object_cross_attn = CrossAttention(dim, num_heads, eps=1e-6).to(
            device=dit.patch_embedding.weight.device,
            dtype=dit.patch_embedding.weight.dtype,
        )
        block.object_gate = nn.Parameter(
            torch.full(
                (1, 1, dim),
                float(object_gate_init),
                device=dit.patch_embedding.weight.device,
                dtype=dit.patch_embedding.weight.dtype,
            )
        )
        if reinitialize_object_branch:
            for attr in ("q", "k", "v", "o"):
                _reinit_linear(getattr(block.object_cross_attn, attr, None))
            if block.norm4.weight is not None:
                nn.init.ones_(block.norm4.weight)
            if block.norm4.bias is not None:
                nn.init.zeros_(block.norm4.bias)

    def block_forward(self, x, context, t_mod, freqs, object_context=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        if object_context is not None and getattr(self, "object_cross_attn", None) is not None:
            object_delta = self.object_cross_attn(self.norm4(x), object_context)
            x = x + torch.tanh(self.object_gate) * object_delta
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x

    for block in dit.blocks:
        block.forward = types.MethodType(block_forward, block)

    dit._codex_object_branch_enabled = True
    return dit


def resolve_num_clean_prefix_latents(
    clean_prefix_latents: Optional[torch.Tensor],
    num_clean_prefix_latents: Optional[int],
) -> int:
    if num_clean_prefix_latents is not None:
        return int(num_clean_prefix_latents)
    if clean_prefix_latents is not None:
        return int(clean_prefix_latents.shape[2])
    return 0


def apply_clean_prefix_to_latents(
    latents: torch.Tensor,
    clean_prefix_latents: Optional[torch.Tensor],
) -> torch.Tensor:
    if clean_prefix_latents is None:
        return latents
    prefix_len = clean_prefix_latents.shape[2]
    latents = latents.clone()
    latents[:, :, :prefix_len] = clean_prefix_latents
    return latents


def apply_clean_latents_at_indices(
    latents: torch.Tensor,
    clean_latents: torch.Tensor,
    latent_indices: list[int],
) -> torch.Tensor:
    if not latent_indices:
        return latents
    latents = latents.clone()
    index_tensor = torch.tensor(latent_indices, device=latents.device, dtype=torch.long)
    latents[:, :, index_tensor] = clean_latents[:, :, index_tensor]
    return latents


def resolve_context_latent_indices_from_frames(
    raw_frame_indices: Optional[list[int]],
    raw_num_frames: Optional[int],
    latent_length: int,
) -> list[int]:
    if not raw_frame_indices:
        return []
    if raw_num_frames is None or raw_num_frames < 1:
        raise ValueError(
            "num_frames must be provided to map context_frame_indices onto latent indices."
        )
    if latent_length < 1:
        raise ValueError(f"latent_length must be positive, got {latent_length}.")

    max_raw_index = max(int(index) for index in raw_frame_indices)
    if max_raw_index >= int(raw_num_frames):
        raise ValueError(
            "context_frame_indices contains an out-of-range frame index: "
            f"max={max_raw_index}, num_frames={raw_num_frames}."
        )

    if raw_num_frames == 1 or latent_length == 1:
        return [0]

    latent_indices = []
    scale = (latent_length - 1) / (int(raw_num_frames) - 1)
    for frame_index in raw_frame_indices:
        latent_index = int(round(int(frame_index) * scale))
        latent_index = max(0, min(latent_index, latent_length - 1))
        if latent_index not in latent_indices:
            latent_indices.append(latent_index)
    latent_indices = sorted(latent_indices)
    # 24 raw frames are rounded to 25 for Wan, so aggressive raw-frame context selections
    # can collapse onto every latent timestep after frame->latent mapping. Keep the latest
    # latent step available for supervision instead of failing the training step.
    if len(latent_indices) >= latent_length:
        latent_indices = latent_indices[:-1]
    return latent_indices


def slice_non_context_latents(
    tensor: torch.Tensor,
    latent_length: int,
    context_latent_indices: list[int],
) -> torch.Tensor:
    if not context_latent_indices:
        return tensor
    keep_indices = [
        index for index in range(latent_length) if index not in set(context_latent_indices)
    ]
    if not keep_indices:
        raise ValueError("Context covers all latent steps, leaving no future steps to predict.")
    index_tensor = torch.tensor(keep_indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(2, index_tensor)


def flow_match_context_sft_loss(pipe: WanVideoPipeline, **inputs):
    max_timestep_boundary = int(
        inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)
    )
    min_timestep_boundary = int(
        inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps)
    )

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype,
        device=pipe.device,
    )

    input_latents = inputs["input_latents"]
    noise = torch.randn_like(input_latents)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

    clean_prefix_latents = inputs.get("clean_prefix_latents")
    num_clean_prefix_latents = resolve_num_clean_prefix_latents(
        clean_prefix_latents=clean_prefix_latents,
        num_clean_prefix_latents=inputs.get("num_clean_prefix_latents"),
    )
    context_latent_indices = resolve_context_latent_indices_from_frames(
        raw_frame_indices=inputs.get("context_frame_indices"),
        raw_num_frames=inputs.get("num_frames"),
        latent_length=input_latents.shape[2],
    )

    if num_clean_prefix_latents < 0 or num_clean_prefix_latents >= input_latents.shape[2]:
        raise ValueError(
            "num_clean_prefix_latents must be in [0, latent_length). "
            f"Got {num_clean_prefix_latents} for latent length {input_latents.shape[2]}."
        )
    if context_latent_indices and len(context_latent_indices) >= input_latents.shape[2]:
        raise ValueError(
            "context_latent_indices must leave at least one latent step for supervision. "
            f"Got {context_latent_indices} for latent length {input_latents.shape[2]}."
        )

    if context_latent_indices:
        inputs["latents"] = pipe.scheduler.add_noise(input_latents, noise, timestep)
        inputs["latents"] = apply_clean_latents_at_indices(
            inputs["latents"],
            input_latents,
            context_latent_indices,
        )
    elif num_clean_prefix_latents > 0:
        latents = input_latents.clone()
        latents[:, :, num_clean_prefix_latents:] = pipe.scheduler.add_noise(
            input_latents[:, :, num_clean_prefix_latents:],
            noise[:, :, num_clean_prefix_latents:],
            timestep,
        )
        latents = apply_clean_prefix_to_latents(latents, clean_prefix_latents)
        inputs["latents"] = latents
    else:
        inputs["latents"] = pipe.scheduler.add_noise(input_latents, noise, timestep)
        if "first_frame_latents" in inputs:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    if context_latent_indices:
        noise_pred = slice_non_context_latents(
            noise_pred,
            latent_length=input_latents.shape[2],
            context_latent_indices=context_latent_indices,
        )
        training_target = slice_non_context_latents(
            training_target,
            latent_length=input_latents.shape[2],
            context_latent_indices=context_latent_indices,
        )
    elif num_clean_prefix_latents > 0:
        noise_pred = noise_pred[:, :, num_clean_prefix_latents:]
        training_target = training_target[:, :, num_clean_prefix_latents:]
    elif "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    loss = torch.nn.functional.mse_loss(
        noise_pred.float(),
        training_target.float(),
    )
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


class WanVideoUnit_ContextVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "context_video",
                "latents",
                "tiled",
                "tile_size",
                "tile_stride",
                "framewise_decoding",
            ),
            output_params=("latents", "clean_prefix_latents", "num_clean_prefix_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        context_video,
        latents,
        tiled,
        tile_size,
        tile_stride,
        framewise_decoding,
    ):
        if context_video is None:
            return {}
        if len(context_video) == 0:
            raise ValueError("context_video cannot be empty.")

        pipe.load_models_to_device(self.onload_model_names)
        context_video = pipe.preprocess_video(context_video)
        if framewise_decoding:
            clean_prefix_latents = pipe.vae.encode_framewise(context_video, device=pipe.device)
        else:
            clean_prefix_latents = pipe.vae.encode(
                context_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

        prefix_len = clean_prefix_latents.shape[2]
        if prefix_len >= latents.shape[2]:
            raise ValueError(
                "Context covers all latent steps, leaving no future steps to predict. "
                f"context_latents={prefix_len}, total_latents={latents.shape[2]}"
            )

        latents = apply_clean_prefix_to_latents(latents, clean_prefix_latents)
        return {
            "latents": latents,
            "clean_prefix_latents": clean_prefix_latents,
            "num_clean_prefix_latents": prefix_len,
        }


def model_fn_wan_video_with_context(
    dit: WanModel,
    motion_controller=None,
    vace=None,
    vap=None,
    animate_adapter=None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state=None,
    vap_clip_feature=None,
    context_vap=None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    wantodance_refimage_feature=None,
    wantodance_fps: float = 30.0,
    music_feature=None,
    skip_9th_layer: bool = False,
    clean_prefix_latents: Optional[torch.Tensor] = None,
    num_clean_prefix_latents: Optional[int] = None,
    object_context: Optional[torch.Tensor] = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            vap=vap,
            animate_adapter=animate_adapter,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
            control_camera_latents_input=control_camera_latents_input,
            clean_prefix_latents=clean_prefix_latents,
            num_clean_prefix_latents=num_clean_prefix_latents,
            object_context=object_context,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video_with_context,
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )

    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )

    clean_prefix_len = resolve_num_clean_prefix_latents(
        clean_prefix_latents=clean_prefix_latents,
        num_clean_prefix_latents=num_clean_prefix_latents,
    )
    if clean_prefix_len > 0:
        latents = apply_clean_prefix_to_latents(latents, clean_prefix_latents)

    if dit.seperated_timestep and (
        fuse_vae_embedding_in_latents or clean_prefix_len > 0
    ):
        clean_steps = clean_prefix_len if clean_prefix_len > 0 else 1
        token_count_per_latent = latents.shape[3] * latents.shape[4] // 4
        timestep = torch.concat(
            [
                torch.zeros(
                    (clean_steps * token_count_per_latent,),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    ((latents.shape[2] - clean_steps) * token_count_per_latent,),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
        )
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [
                torch.nn.functional.pad(
                    chunk,
                    (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]),
                    value=0,
                )
                for chunk in t_chunks
            ]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    if hasattr(dit, "wantodance_enable_global") and dit.wantodance_enable_global and int(
        wantodance_fps + 0.5
    ) != 30:
        x = dit.patchify(
            x,
            control_camera_latents_input,
            enable_wantodance_global=True,
        )
    else:
        x = dit.patchify(x, control_camera_latents_input)

    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(
            x,
            pose_latents,
            face_pixel_values,
        )

    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    if vap is not None:
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, "b c f h w -> b (f h w) c").contiguous()
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(
            timestep.dtype
        )
        t = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t).unflatten(1, (6, vap.dim))
        freqs_vap = vap.compute_freqs_mot(f, h, w).to(x.device)
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)

    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    if vace_context is not None:
        vace_hints = vace(
            x,
            vace_context,
            context,
            t_mod,
            freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    if hasattr(dit, "wantodance_enable_global") and dit.wantodance_enable_global:
        if wantodance_refimage_feature is not None:
            refimage_feature_embedding = dit.img_emb_refimage(wantodance_refimage_feature)
            context = torch.cat([refimage_feature_embedding, context], dim=1)
        if (
            dit.wantodance_enable_dynamicfps or dit.wantodance_enable_unimodel
        ) and int(wantodance_fps + 0.5) != 30:
            freqs_0 = wantodance_get_single_freqs(dit.freqs[0], f, wantodance_fps)
            freqs = torch.cat(
                [
                    freqs_0.view(f, 1, 1, -1).expand(f, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(f * h * w, 1, -1).to(x.device)
        if (
            dit.wantodance_enable_global
            or dit.wantodance_enable_dynamicfps
            or dit.wantodance_enable_unimodel
        ):
            if use_unified_sequence_parallel:
                length = (
                    int(float(music_feature.shape[0]) / get_sequence_parallel_world_size())
                    * get_sequence_parallel_world_size()
                )
                music_feature = music_feature[:length]
                music_feature = torch.chunk(
                    music_feature,
                    get_sequence_parallel_world_size(),
                    dim=0,
                )[get_sequence_parallel_rank()]
            if not dit.training:
                dit.music_encoder.to(x.device, dtype=x.dtype)
            music_feature = music_feature.to(x.device, dtype=x.dtype)
            music_feature = dit.music_projection(music_feature)
            music_feature = dit.music_encoder(music_feature)
            if music_feature.dim() == 2:
                music_feature = music_feature.unsqueeze(0)
            if use_unified_sequence_parallel:
                if dist.is_initialized() and dist.get_world_size() > 1:
                    music_feature = get_sp_group().all_gather(music_feature, dim=1)
            music_feature = music_feature.unsqueeze(1)
            music_feature = torch.nn.functional.interpolate(
                music_feature,
                size=(149, 4800),
                mode="bilinear",
            )
            music_feature = music_feature.squeeze(1)
        if music_feature is not None:
            if music_feature.dim() == 2:
                music_feature = music_feature.unsqueeze(0)
            music_feature = music_feature.to(x.device, dtype=x.dtype)
            frame_num = latents.shape[2] if len(latents.shape) == 5 else latents.shape[1]
            context_shape_end = context.shape[2]
            music_feature = music_feature.unsqueeze(1)
            if use_unified_sequence_parallel:
                frame_interp = (
                    int(float(frame_num * 8) / get_sequence_parallel_world_size())
                    * get_sequence_parallel_world_size()
                )
            else:
                frame_interp = frame_num * 8
            music_feature = torch.nn.functional.interpolate(
                music_feature,
                size=(frame_interp, context_shape_end),
                mode="bilinear",
            )
            music_feature = music_feature.squeeze(1)
            if use_unified_sequence_parallel:
                dit.merged_audio_emb = torch.chunk(
                    music_feature,
                    get_sequence_parallel_world_size(),
                    dim=1,
                )[get_sequence_parallel_rank()]
            else:
                dit.merged_audio_emb = music_feature

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [
                torch.nn.functional.pad(
                    chunk,
                    (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]),
                    value=0,
                )
                for chunk in chunks
            ]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        if object_context is not None:
            object_context = object_context.to(device=x.device, dtype=context.dtype)
            if getattr(dit, "object_embedding", None) is not None:
                object_context = dit.object_embedding(object_context.to(dit.object_embedding[0].weight.dtype))
                object_context = object_context.to(dtype=context.dtype, device=x.device)

        def create_custom_forward_vap(block, vap_module):
            def custom_forward(*inputs):
                return vap_module(block, *inputs)

            return custom_forward

        for block_id, block in enumerate(dit.blocks):
            if skip_9th_layer and block_id == 9:
                continue
            if vap is not None and block_id in vap.mot_layers_mapping:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x,
                            context,
                            t_mod,
                            freqs,
                            x_vap,
                            context_vap,
                            t_mod_vap,
                            freqs_vap,
                            block_id,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x,
                        context,
                        t_mod,
                        freqs,
                        x_vap,
                        context_vap,
                        t_mod_vap,
                        freqs_vap,
                        block_id,
                        use_reentrant=False,
                    )
                else:
                    x, x_vap = vap(
                        block,
                        x,
                        context,
                        t_mod,
                        freqs,
                        x_vap,
                        context_vap,
                        t_mod_vap,
                        freqs_vap,
                        block_id,
                    )
            else:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x,
                    context,
                    t_mod,
                    freqs,
                    object_context=object_context,
                )

            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(
                        current_vace_hint,
                        get_sequence_parallel_world_size(),
                        dim=1,
                    )[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(
                        current_vace_hint,
                        (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]),
                        value=0,
                    )
                x = x + current_vace_hint * vace_scale

            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)

            if hasattr(dit, "wantodance_enable_music_inject") and dit.wantodance_enable_music_inject:
                x = dit.wantodance_after_transformer_block(block_id, x)
        if tea_cache is not None:
            tea_cache.store(x)

    if hasattr(dit, "wantodance_enable_unimodel") and dit.wantodance_enable_unimodel and int(
        wantodance_fps + 0.5
    ) != 30:
        x = dit.head_global(x, t)
    else:
        x = dit.head(x, t)

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1] :]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


class ContextAwareWanVideoPipeline(WanVideoPipeline):
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            redirect_common_files=redirect_common_files,
            use_usp=use_usp,
            vram_limit=vram_limit,
        )
        pipe.__class__ = ContextAwareWanVideoPipeline
        pipe.enable_context_prefix_conditioning()
        return pipe

    def enable_context_prefix_conditioning(self):
        if any(isinstance(unit, WanVideoUnit_ContextVideoEmbedder) for unit in self.units):
            self.model_fn = model_fn_wan_video_with_context
            return

        insert_at = None
        for index, unit in enumerate(self.units):
            if unit.__class__.__name__ == "WanVideoUnit_InputVideoEmbedder":
                insert_at = index + 1
                break
        if insert_at is None:
            raise RuntimeError("Failed to locate WanVideoUnit_InputVideoEmbedder in pipeline units.")

        self.units.insert(insert_at, WanVideoUnit_ContextVideoEmbedder())
        self.model_fn = model_fn_wan_video_with_context

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_image: Optional[Image.Image] = None,
        end_image: Optional[Image.Image] = None,
        input_video: Optional[list[Image.Image]] = None,
        context_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        input_audio=None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video=None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        camera_control_direction: Optional[
            Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]
        ] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[tuple] = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ),
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        vap_video: Optional[list[Image.Image]] = None,
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        switch_DiT_boundary: Optional[float] = 0.875,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        motion_bucket_id: Optional[int] = None,
        longcat_video: Optional[list[Image.Image]] = None,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        wantodance_music_path: Optional[str] = None,
        wantodance_reference_image: Optional[Image.Image] = None,
        wantodance_fps: Optional[float] = 30,
        wantodance_keyframes: Optional[list[Image.Image]] = None,
        wantodance_keyframes_mask: Optional[list[int]] = None,
        object_context: Optional[torch.Tensor] = None,
        framewise_decoding: bool = False,
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        if context_video is not None and input_image is None and len(context_video) > 0:
            input_image = context_video[0]

        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "context_video": context_video,
            "denoising_strength": denoising_strength,
            "control_video": control_video,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio,
            "audio_sample_rate": audio_sample_rate,
            "s2v_pose_video": s2v_pose_video,
            "audio_embeds": audio_embeds,
            "s2v_pose_latents": s2v_pose_latents,
            "motion_video": motion_video,
            "animate_pose_video": animate_pose_video,
            "animate_face_video": animate_face_video,
            "animate_inpaint_video": animate_inpaint_video,
            "animate_mask_video": animate_mask_video,
            "vap_video": vap_video,
            "wantodance_music_path": wantodance_music_path,
            "wantodance_reference_image": wantodance_reference_image,
            "wantodance_fps": wantodance_fps,
            "wantodance_keyframes": wantodance_keyframes,
            "wantodance_keyframes_mask": wantodance_keyframes_mask,
            "object_context": object_context,
            "framewise_decoding": framewise_decoding,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit,
                self,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            if (
                timestep.item() < switch_DiT_boundary * 1000
                and self.dit2 is not None
                and models["dit"] is not self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2

            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        timestep=timestep,
                    )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
            if inputs_shared.get("clean_prefix_latents") is not None:
                prefix_len = inputs_shared["clean_prefix_latents"].shape[2]
                inputs_shared["latents"][:, :, :prefix_len] = inputs_shared["clean_prefix_latents"]
            elif "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        if vace_reference_image is not None or (
            animate_pose_video is not None and animate_face_video is not None
        ):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]

        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(["vae"])
        if framewise_decoding:
            video = self.vae.decode_framewise(inputs_shared["latents"], device=self.device)
        else:
            video = self.vae.decode(
                inputs_shared["latents"],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video
