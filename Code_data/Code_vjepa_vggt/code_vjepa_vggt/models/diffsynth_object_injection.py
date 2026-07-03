"""
Monkey-patch DiffSynth DiTBlock to add object conditioning support.

This module does NOT modify DiffSynth-Studio-main source code.
Instead, it dynamically replaces DiTBlock.forward at runtime.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

try:
    from diffsynth.models.wan_video_dit import CrossAttention
except ImportError:
    raise ImportError(
        "Cannot import CrossAttention from DiffSynth. "
        "Ensure /home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main is in PYTHONPATH."
    )


def inject_object_branch_to_dit(
    model: nn.Module,
    object_cross_attn_dim: int = 1536,
    object_gate_init: float = 0.1,
) -> nn.Module:
    """
    Inject object conditioning branch to all DiTBlocks in a WanModel.

    Args:
        model: WanModel instance (pipe.dit)
        object_cross_attn_dim: Hidden dim for object_context (default 1536 for Wan2.2)
        object_gate_init: Initial value for object_gate parameter

    Returns:
        Modified model (in-place operation)
    """
    # Create global object_embedding layer (shared by all blocks)
    # This projects object_context [B, T*O, object_cross_attn_dim] to hidden_dim
    hidden_dim = model.dim  # Use model.dim instead of in_dim
    object_embedding = nn.Linear(object_cross_attn_dim, hidden_dim, bias=True)
    nn.init.xavier_uniform_(object_embedding.weight)
    nn.init.zeros_(object_embedding.bias)

    # Register as model attribute so it appears in state_dict
    model.object_embedding = object_embedding.to(device=next(model.parameters()).device)

    # Create shared context holder (thread-safe for training)
    model._object_context_holder = {"context": None}

    # Inject object branch to each DiTBlock
    for block in model.blocks:
        _inject_object_branch_to_block(block, hidden_dim, object_gate_init, model._object_context_holder)

    return model


def _inject_object_branch_to_block(
    block: nn.Module,
    hidden_dim: int,
    object_gate_init: float,
    context_holder: dict,
) -> None:
    """Add object conditioning modules to a single DiTBlock."""
    device = next(block.parameters()).device

    # Add norm4 (LayerNorm for object_cross_attn query)
    block.norm4 = nn.LayerNorm(hidden_dim, eps=1e-6).to(device)

    # Add object_cross_attn (CrossAttention with kv from object_context)
    block.object_cross_attn = CrossAttention(
        hidden_dim,
        num_heads=block.num_heads,
        eps=1e-6,
        has_image_input=False
    ).to(device)

    # Add object_gate (learnable scalar, initialized to object_gate_init)
    block.object_gate = nn.Parameter(
        torch.full((1,), object_gate_init, dtype=torch.float32, device=device)
    )

    # Replace forward method
    original_forward = block.forward
    block._original_forward = original_forward  # Keep reference for debugging

    def forward_with_object(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extended DiTBlock forward with object conditioning.

        Args:
            x: [B, N, D] patch tokens
            context: [B, L, D] text context
            t_mod: [B, 6, D] timestep modulation
            freqs: Positional encoding frequencies

        Returns:
            x: [B, N, D]
        """
        # Call original forward (self_attn + cross_attn + ffn)
        x = self._original_forward(x, context, t_mod, freqs)

        # Add object conditioning branch if provided via context_holder
        object_context = context_holder.get("context", None)
        if object_context is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Query: norm4(x), Key/Value: object_context
                object_delta = self.object_cross_attn(
                    self.norm4(x),
                    object_context
                )
                # Gate with tanh activation
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    x = x + object_delta * torch.tanh(self.object_gate)

        return x

    # Bind the new forward method
    block.forward = forward_with_object.__get__(block, type(block))


def remove_object_branch_from_dit(model: nn.Module) -> nn.Module:
    """
    Remove object conditioning branch (restore original DiTBlock.forward).

    Useful for debugging or switching back to baseline model.
    """
    # Remove global object_embedding
    if hasattr(model, 'object_embedding'):
        delattr(model, 'object_embedding')

    # Restore original forward for each block
    for block in model.blocks:
        if hasattr(block, '_original_forward'):
            block.forward = block._original_forward
            delattr(block, '_original_forward')

        # Remove added modules
        for attr in ['norm4', 'object_cross_attn', 'object_gate']:
            if hasattr(block, attr):
                delattr(block, attr)

    return model
