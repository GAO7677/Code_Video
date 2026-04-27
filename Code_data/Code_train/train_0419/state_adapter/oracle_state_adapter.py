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
        object_vocab_size: int = 65536,
        text_vocab_size: int = 4096,
        object_embed_dim: int = 64,
        role_embed_dim: int = 32,
        source_embed_dim: int = 32,
        category_embed_dim: int = 64,
        condition_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dit_dim = int(dit_dim)
        self.condition_dropout = float(condition_dropout)

        self.object_embed = torch.nn.Embedding(object_vocab_size, object_embed_dim)
        self.role_embed = torch.nn.Embedding(text_vocab_size, role_embed_dim)
        self.source_embed = torch.nn.Embedding(text_vocab_size, source_embed_dim)
        self.category_embed = torch.nn.Embedding(text_vocab_size, category_embed_dim)

        static_in_dim = object_embed_dim + role_embed_dim + source_embed_dim + category_embed_dim
        self.static_proj = torch.nn.Sequential(
            torch.nn.Linear(static_in_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
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
        self.modulation_heads = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, dit_dim * 2),
                )
                for _ in range(num_layers)
            ]
        )

    def _maybe_dropout(self, token_seq: torch.Tensor) -> torch.Tensor:
        if self.training and self.condition_dropout > 0.0:
            if torch.rand((), device=token_seq.device) < self.condition_dropout:
                return torch.zeros_like(token_seq)
        return token_seq

    def encode_future_plan(
        self,
        oracle_state: torch.Tensor,
        oracle_visibility: torch.Tensor,
        object_id_tokens: torch.Tensor,
        role_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        category_tokens: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        if oracle_state.dim() == 3:
            oracle_state = oracle_state.unsqueeze(0)
        if oracle_visibility.dim() == 2:
            oracle_visibility = oracle_visibility.unsqueeze(0)
        if object_id_tokens.dim() == 1:
            object_id_tokens = object_id_tokens.unsqueeze(0)
            role_tokens = role_tokens.unsqueeze(0)
            source_tokens = source_tokens.unsqueeze(0)
            category_tokens = category_tokens.unsqueeze(0)

        batch, raw_frames, num_objects, _ = oracle_state.shape
        if target_frames <= 0:
            return oracle_state.new_zeros((batch, 0, self.hidden_dim))

        static_tokens = torch.cat(
            [
                self.object_embed(object_id_tokens),
                self.role_embed(role_tokens),
                self.source_embed(source_tokens),
                self.category_embed(category_tokens),
            ],
            dim=-1,
        )
        static_tokens = self.static_proj(static_tokens).unsqueeze(1).expand(batch, raw_frames, num_objects, self.hidden_dim)
        dynamic_tokens = self.state_mlp(oracle_state)
        object_tokens = dynamic_tokens + static_tokens
        valid_mask = oracle_visibility > 0.5
        frame_tokens = self.frame_pool(object_tokens, valid_mask=valid_mask)
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
