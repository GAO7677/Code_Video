from __future__ import annotations

from .utils import require_torch


def compute_interaction_features(centers, existence):
    torch = require_torch()
    batch, num_objects, _ = centers.shape
    rel = centers[:, :, None, :] - centers[:, None, :, :]
    dist = torch.linalg.norm(rel, dim=-1)
    eye = torch.eye(num_objects, device=centers.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))

    existence_mask = existence[:, None, :].expand(batch, num_objects, num_objects) > 0.5
    dist = dist.masked_fill(~existence_mask, float("inf"))
    nearest_dist, nearest_idx = dist.min(dim=-1)
    nearest_idx_safe = nearest_idx.clamp(min=0)
    gather_idx = nearest_idx_safe.unsqueeze(-1).expand(-1, -1, 2)
    nearest_rel = torch.gather(rel, dim=2, index=gather_idx.unsqueeze(2)).squeeze(2)
    nearest_dist = torch.where(torch.isfinite(nearest_dist), nearest_dist, torch.zeros_like(nearest_dist))
    nearest_rel = torch.where(
        torch.isfinite(nearest_dist).unsqueeze(-1),
        nearest_rel,
        torch.zeros_like(nearest_rel),
    )
    return torch.cat([nearest_rel, nearest_dist.unsqueeze(-1)], dim=-1)
