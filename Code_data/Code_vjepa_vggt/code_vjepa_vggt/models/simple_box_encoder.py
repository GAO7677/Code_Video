"""
Simple box encoder for testing DiffSynth-native architecture.

This is a minimal object conditioning module that encodes boxes directly,
bypassing the full ObjectTubeProjector pipeline (JEPA + CoTracker + VAE pooling).
Used for architecture validation only.
"""
import torch
import torch.nn as nn


class SimpleBoxEncoder(nn.Module):
    """
    Encode normalized xyxy boxes to object tokens.

    Args:
        box_dim: 4 (xyxy format)
        hidden_dim: intermediate dimension
        out_dim: output dimension matching object_cross_attn
    """

    def __init__(self, box_dim: int = 4, hidden_dim: int = 256, out_dim: int = 4096):
        super().__init__()
        self.out_dim = out_dim

        # Encode box coordinates
        self.box_encoder = nn.Sequential(
            nn.Linear(box_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Learnable time embedding
        self.time_embed = nn.Embedding(64, out_dim)  # max 64 frames

        # Learnable slot embedding
        self.slot_embed = nn.Embedding(16, out_dim)  # max 16 objects

    def forward(
        self,
        boxes: torch.Tensor,  # [B, T, O, 4] normalized xyxy
    ) -> torch.Tensor:
        """
        Args:
            boxes: [B, T, O, 4] normalized xyxy boxes

        Returns:
            object_context: [B, T*O, D]
        """
        B, T, O, _ = boxes.shape
        device = boxes.device

        # Encode boxes: [B, T, O, 4] -> [B, T, O, D]
        box_features = self.box_encoder(boxes)

        # Add time embedding
        time_ids = torch.arange(T, device=device).unsqueeze(0).unsqueeze(2)  # [1, T, 1]
        time_ids = time_ids.expand(B, T, O)  # [B, T, O]
        time_features = self.time_embed(time_ids)  # [B, T, O, D]

        # Add slot embedding
        slot_ids = torch.arange(O, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, O]
        slot_ids = slot_ids.expand(B, T, O)  # [B, T, O]
        slot_features = self.slot_embed(slot_ids)  # [B, T, O, D]

        # Combine
        object_tokens = box_features + time_features + slot_features  # [B, T, O, D]

        # Flatten time and object dimensions
        object_context = object_tokens.reshape(B, T * O, self.out_dim)  # [B, T*O, D]

        return object_context
