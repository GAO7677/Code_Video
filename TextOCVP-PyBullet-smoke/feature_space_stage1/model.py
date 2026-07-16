from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))

from models.Blocks.attention import SlotAttention  # noqa: E402
from models.Blocks.initializers import get_initializer  # noqa: E402
from models.Blocks.transition_models import get_transition_module  # noqa: E402


class SpatialBroadcastFeatureDecoder(nn.Module):
    """Decode each slot at every latent-grid location and alpha-compose slots."""

    def __init__(self, slot_dim: int, feature_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.position_mlp = nn.Sequential(
            nn.Linear(2, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim + 1),
        )
        self.feature_dim = feature_dim

    def forward(
        self, slots: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_slots, slot_dim = slots.shape
        ys = torch.linspace(-1.0, 1.0, height, device=slots.device, dtype=slots.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=slots.device, dtype=slots.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((grid_x, grid_y), dim=-1).reshape(height * width, 2)
        positions = self.position_mlp(coords).view(1, 1, height * width, slot_dim)
        broadcast = slots[:, :, None, :] + positions
        decoded = self.decoder(broadcast)
        object_features, mask_logits = decoded.split((self.feature_dim, 1), dim=-1)
        masks = F.softmax(mask_logits, dim=1)
        reconstruction = (object_features * masks).sum(dim=1)
        reconstruction = reconstruction.view(batch, height, width, self.feature_dim)
        object_features = object_features.view(
            batch, num_slots, height, width, self.feature_dim
        )
        masks = masks.view(batch, num_slots, height, width, 1)
        return reconstruction, object_features, masks


class FeatureSlotDecomposer(nn.Module):
    """SAVi-style recurrent slots operating on a frozen latent feature grid."""

    def __init__(
        self,
        feature_dim: int,
        num_slots: int = 8,
        slot_dim: int = 256,
        slot_iterations_first: int = 3,
        slot_iterations: int = 1,
        slot_mlp_hidden: int = 512,
        decoder_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.initializer = get_initializer("LearnedRandom", slot_dim, num_slots)
        self.slot_attention = SlotAttention(
            dim_feats=feature_dim,
            dim_slots=slot_dim,
            num_slots=num_slots,
            num_iters_first=slot_iterations_first,
            num_iters=slot_iterations,
            mlp_hidden=slot_mlp_hidden,
        )
        self.transition = get_transition_module(
            model_name="TransformerBlock",
            slot_dim=slot_dim,
            num_heads=4,
            mlp_size=512,
        )
        self.decoder = SpatialBroadcastFeatureDecoder(
            slot_dim=slot_dim,
            feature_dim=feature_dim,
            hidden_dim=decoder_hidden,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 5:
            raise ValueError(f"Expected [B,T,H,W,D] features, got {tuple(features.shape)}")
        batch, latent_time, height, width, feature_dim = features.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {feature_dim}")

        predicted_slots = self.initializer(batch_size=batch)
        slots_history = []
        reconstruction_history = []
        objects_history = []
        masks_history = []
        for time_index in range(latent_time):
            tokens = features[:, time_index].reshape(batch, height * width, feature_dim)
            slots = self.slot_attention(
                inputs=tokens,
                slots=predicted_slots,
                step=time_index,
            )
            predicted_slots = self.transition(slots)
            reconstruction, objects, masks = self.decoder(slots, height, width)
            slots_history.append(slots)
            reconstruction_history.append(reconstruction)
            objects_history.append(objects)
            masks_history.append(masks)

        return {
            "reconstructed_features": torch.stack(reconstruction_history, dim=1),
            "object_features": torch.stack(objects_history, dim=1),
            "masks": torch.stack(masks_history, dim=1),
            "slots": torch.stack(slots_history, dim=1),
        }


def feature_space_losses(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    space: str,
) -> dict[str, torch.Tensor]:
    prediction = output["reconstructed_features"].float()
    target = target.float()
    reduce_dims = (1, 2, 3, 4)
    mse = F.mse_loss(prediction, target, reduction="none").mean(dim=reduce_dims)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    cosine = cosine.mean(dim=(1, 2, 3))
    if space == "vjepa":
        total = mse + 0.1 * cosine
    elif space == "vae":
        total = mse
    else:
        raise ValueError(f"Unsupported feature space: {space}")

    masks = output["masks"].float().squeeze(-1)
    entropy = -(masks.clamp_min(1e-8) * masks.clamp_min(1e-8).log()).sum(dim=2)
    normalized_entropy = entropy.mean(dim=(1, 2, 3)) / torch.log(
        torch.tensor(masks.shape[2], device=masks.device, dtype=masks.dtype)
    )
    usage = masks.mean(dim=(1, 3, 4))
    return {
        "total": total,
        "feature_mse": mse,
        "feature_cosine": cosine,
        "mask_entropy": normalized_entropy,
        "slot_usage_min": usage.min(dim=1).values,
        "slot_usage_max": usage.max(dim=1).values,
    }

