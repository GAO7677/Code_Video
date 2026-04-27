"""Small oracle future-state adapter for Wan temporal modulation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class FrameObjectAttentionPool(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, num_objects, hidden_dim = tokens.shape
        flat_tokens = tokens.reshape(batch * frames, num_objects, hidden_dim)
        flat_valid = valid_mask.bool().reshape(batch * frames, num_objects)
        # MultiheadAttention expects at least one unmasked token. When a whole frame
        # is invisible, keep the first object slot alive and let the state MLP learn
        # a neutral representation from the zeroed state.
        empty_rows = ~flat_valid.any(dim=1)
        if torch.any(empty_rows):
            flat_valid[empty_rows, 0] = True
        flat_mask = ~flat_valid
        query = self.query.expand(batch * frames, 1, hidden_dim)
        pooled, _ = self.attn(
            query=query,
            key=flat_tokens,
            value=flat_tokens,
            key_padding_mask=flat_mask,
            need_weights=False,
        )
        return pooled[:, 0].reshape(batch, frames, hidden_dim)


class OracleStateAdapter(torch.nn.Module):
    def __init__(
        self,
        dit_dim: int,
        num_layers: int,
        state_dim: int = 9,
        hidden_dim: int = 1024,
        mlp_hidden_dim: int = 512,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        pool_heads: int = 4,
        condition_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dit_dim = int(dit_dim)
        self.condition_dropout = float(condition_dropout)
        self.state_mlp = torch.nn.Sequential(
            torch.nn.Linear(state_dim, mlp_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self.frame_pool = FrameObjectAttentionPool(hidden_dim=hidden_dim, num_heads=pool_heads)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=temporal_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
        )
        self.modulation_heads = torch.nn.ModuleList()
        for _ in range(num_layers):
            head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, dit_dim * 2),
            )
            # Start from an exact no-op so Wan behavior is unchanged at step 0.
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)
            self.modulation_heads.append(head)

    def _maybe_dropout(self, token_seq: torch.Tensor) -> torch.Tensor:
        if self.training and self.condition_dropout > 0.0:
            if torch.rand((), device=token_seq.device) < self.condition_dropout:
                return torch.zeros_like(token_seq)
        return token_seq

    def encode_future_plan(
        self,
        oracle_state: torch.Tensor,
        target_frames: int,
        oracle_visibility: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if oracle_state.dim() == 3:
            oracle_state = oracle_state.unsqueeze(0)
        if oracle_visibility is not None and oracle_visibility.dim() == 2:
            oracle_visibility = oracle_visibility.unsqueeze(0)

        batch, raw_frames, num_objects, _ = oracle_state.shape
        if target_frames <= 0:
            return oracle_state.new_zeros((batch, 0, self.hidden_dim))

        dynamic_tokens = self.state_mlp(oracle_state)
        if oracle_visibility is None:
            valid_mask = oracle_state[..., -1] > 0.5
        else:
            valid_mask = oracle_visibility > 0.5
        frame_tokens = self.frame_pool(dynamic_tokens, valid_mask=valid_mask)
        frame_tokens = self.temporal_encoder(frame_tokens)
        frame_tokens = self._maybe_dropout(frame_tokens)

        if frame_tokens.shape[1] != target_frames:
            pooled = F.adaptive_avg_pool1d(
                frame_tokens.transpose(1, 2),
                output_size=target_frames,
            )
            frame_tokens = pooled.transpose(1, 2)
        return frame_tokens

    def apply_block_modulation(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        future_plan_tokens: torch.Tensor,
        total_frames: int,
        clean_prefix_len: int,
        spatial_tokens_per_frame: int,
    ) -> torch.Tensor:
        if future_plan_tokens is None or future_plan_tokens.numel() == 0:
            return hidden_states

        batch, _, channels = hidden_states.shape
        modulation = self.modulation_heads[block_idx](future_plan_tokens)
        gamma, beta = modulation.chunk(2, dim=-1)
        if clean_prefix_len > 0:
            zeros = hidden_states.new_zeros((batch, clean_prefix_len, channels))
            gamma = torch.cat([zeros, gamma], dim=1)
            beta = torch.cat([zeros, beta], dim=1)
        gamma = gamma[:, :total_frames]
        beta = beta[:, :total_frames]

        x = hidden_states.reshape(batch, total_frames, spatial_tokens_per_frame, channels)
        x_norm = F.layer_norm(x, (channels,))
        x = x + gamma.unsqueeze(2) * x_norm + beta.unsqueeze(2)
        return x.reshape(batch, total_frames * spatial_tokens_per_frame, channels)
