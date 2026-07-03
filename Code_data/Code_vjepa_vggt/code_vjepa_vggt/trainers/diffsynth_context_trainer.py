"""
DiffSynth-native Trainer with Simplified Object Conditioning and Context Mask Injection.

This trainer integrates:
1. SimpleBoxEncoder (boxes → object tokens, bypasses JEPA/CoTracker for testing)
2. Context mask injection (replace context frames with clean latents)
3. Flow matching loss (only on future frames)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..utils.masks import latent_frame_mask, broadcast_latent_mask


class DiffSynthContextTrainer(nn.Module):
    """
    Trainer for stage1b context-only object conditioning.

    Wraps WanVideoPipeline and adds:
    - Simplified object conditioning (box encoder)
    - Context mask injection during training
    - Future-only loss computation
    """

    def __init__(
        self,
        pipe,  # WanVideoPipeline instance
        box_encoder,  # SimpleBoxEncoder instance
        vae_stride_t: int = 4,
        num_context_frames: int = 16,
    ):
        super().__init__()
        self.pipe = pipe
        self.box_encoder = box_encoder
        self.vae_stride_t = vae_stride_t
        self.num_context_frames = num_context_frames

        # Move box encoder to same device as pipe
        device = next(pipe.dit.parameters()).device
        self.box_encoder = self.box_encoder.to(device)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        cfg_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            batch: Dictionary containing:
                - video: [B, C, T, H, W] full video
                - context_video: [B, C, Tc, H, W] context frames
                - context_boxes: [B, Tc, O, 4] normalized xyxy boxes
                - prompt_emb: [B, L, D] text embeddings (optional)
            cfg_dropout_prob: Probability to drop object_context for CFG training

        Returns:
            loss: Scalar tensor
        """
        device = next(self.pipe.dit.parameters()).device

        # 1. Encode video to latents
        with torch.no_grad():
            # Context frames: [B, C, Tc, H, W] → [B, C_lat, Tc_lat, H_lat, W_lat]
            context_video = batch["context_video"].to(device)
            context_latents = self._encode_video(context_video)

            # Full video: [B, C, T, H, W] → [B, C_lat, T_lat, H_lat, W_lat]
            full_video = batch["video"].to(device)
            full_latents = self._encode_video(full_video)

        # 2. Extract object_context
        object_context = None
        if cfg_dropout_prob == 0.0 or torch.rand(1).item() > cfg_dropout_prob:
            # Simple box encoding: [B, Tc, O, 4] -> [B, Tc*O, D]
            object_context = self.box_encoder(batch["context_boxes"].to(device))
            # Project to DiT hidden dim via global object_embedding
            object_context = self.pipe.dit.object_embedding(object_context)

        # 3. Sample timestep and add noise
        batch_size = full_latents.shape[0]
        timestep_id = torch.randint(
            0,
            len(self.pipe.scheduler.timesteps),
            (1,),
            device="cpu"  # Keep on CPU for indexing
        )
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype,
            device=device
        )

        noise = torch.randn_like(full_latents)
        noisy_latents = self.pipe.scheduler.add_noise(full_latents, noise, timestep)

        # 4. Context mask injection
        num_video_frames = full_video.shape[2]
        context_mask, future_mask = latent_frame_mask(
            num_video_frames,
            self.num_context_frames,
            self.vae_stride_t,
            device=device,
        )

        # Expand context_latents to full time length
        clean_latents = self._expand_context_latents_to_full(
            context_latents,
            full_latents
        )

        # Mix: context frames = clean, future frames = noisy
        context_mask_bc = broadcast_latent_mask(context_mask, full_latents)
        future_mask_bc = broadcast_latent_mask(future_mask, full_latents)
        mixed_latents = clean_latents * context_mask_bc + noisy_latents * future_mask_bc

        # 5. DiT forward with object_context
        # Get prompt embedding (use empty prompt if not provided)
        if "prompt_emb" in batch:
            prompt_emb = batch["prompt_emb"].to(device)
        else:
            # Use text encoder to get empty prompt embedding
            # Tokenize empty prompt
            ids, mask = self.pipe.tokenizer(
                [""] * batch_size,
                return_mask=True,
                add_special_tokens=True
            )
            ids = ids.to(device)
            mask = mask.to(device)
            prompt_emb = self.pipe.text_encoder(ids, mask)

        # Forward through DiT (monkey-patched forward accepts object_context)
        pred = self._model_forward(
            latents=mixed_latents,
            timestep=timestep,
            prompt_emb=prompt_emb,
            object_context=object_context,
        )

        # 6. Compute loss (only on future frames)
        training_target = self.pipe.scheduler.training_target(
            full_latents,
            noise,
            timestep
        )

        # Apply future mask to loss
        loss = F.mse_loss(
            pred * future_mask_bc,
            training_target * future_mask_bc,
            reduction="none"
        )

        # Weight by timestep
        timestep_weight = self.pipe.scheduler.training_weight(
            timestep,
            device=device,
            dtype=self.pipe.torch_dtype
        )
        loss = (loss * timestep_weight).mean()

        return loss

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video [B,C,T,H,W] to latents [B,C_lat,T_lat,H_lat,W_lat]."""
        device = next(self.pipe.dit.parameters()).device
        # DiffSynth VAE.encode expects list of [C,T,H,W] tensors
        video_list = [video[i] for i in range(video.shape[0])]
        latents = self.pipe.vae.encode(video_list, device)
        return latents

    def _expand_context_latents_to_full(
        self,
        context_latents: torch.Tensor,
        full_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Expand context_latents to full time length by zero-padding."""
        # context_latents: [B, C, Tc, H, W]
        # full_latents: [B, C, T, H, W]
        out = torch.zeros_like(full_latents)
        copy_t = min(context_latents.shape[2], full_latents.shape[2])
        out[:, :, :copy_t] = context_latents[:, :, :copy_t]
        return out

    def _model_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        object_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward through DiT model.

        Manual implementation to handle patchify/unpatchify and object_context injection.
        """
        device = latents.device
        B, C, T, H, W = latents.shape

        # 1. Timestep embedding
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
        t = self.pipe.dit.time_embedding(
            sinusoidal_embedding_1d(self.pipe.dit.freq_dim, timestep).to(latents.dtype)
        )
        t_mod = self.pipe.dit.time_projection(t).unflatten(1, (6, self.pipe.dit.dim))

        # 2. Text embedding
        context = self.pipe.dit.text_embedding(prompt_emb)

        # 3. Patchify
        x = self.pipe.dit.patch_embedding(latents)  # [B, D, T', H', W']
        _, D, T_p, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, T'*H'*W', D]

        # 4. Prepare RoPE frequencies
        from einops import rearrange
        freqs = torch.cat([
            self.pipe.dit.freqs[0][:T_p].view(T_p, 1, 1, -1).expand(T_p, H_p, W_p, -1),
            self.pipe.dit.freqs[1][:H_p].view(1, H_p, 1, -1).expand(T_p, H_p, W_p, -1),
            self.pipe.dit.freqs[2][:W_p].view(1, 1, W_p, -1).expand(T_p, H_p, W_p, -1)
        ], dim=-1).reshape(T_p * H_p * W_p, 1, -1).to(device)

        # 5. Set object_context in holder
        self.pipe.dit._object_context_holder["context"] = object_context

        # 6. Forward through blocks
        try:
            for block in self.pipe.dit.blocks:
                x = block(x, context, t_mod, freqs)
        finally:
            # Clear holder
            self.pipe.dit._object_context_holder["context"] = None

        # 7. Head
        x = self.pipe.dit.head(x, t)

        # 8. Unpatchify
        x = rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=T_p, h=H_p, w=W_p,
            x=self.pipe.dit.patch_size[0],
            y=self.pipe.dit.patch_size[1],
            z=self.pipe.dit.patch_size[2]
        )

        return x
