from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from wan_phyco_train0716.property_maps import BRANCH_SLICES


@dataclass(frozen=True)
class ControlResidualStats:
    block_id: int
    branch_name: str
    residual_to_hidden_rms: float
    active_fraction: float


class ZeroLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class WanPhyCoControlBranch(nn.Module):
    """ControlNet-XS branch with a property-map encoder and zero outputs."""

    def __init__(
        self,
        *,
        name: str,
        map_channels: int,
        wan_dim: int,
        hidden_dim: int,
        block_ids: Iterable[int],
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.wan_dim = int(wan_dim)
        self.hidden_dim = int(hidden_dim)
        self.block_ids = tuple(int(value) for value in block_ids)
        self.condition_encoder = nn.Sequential(
            nn.Conv3d(map_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.hidden_norms = nn.ModuleDict(
            {str(block_id): nn.LayerNorm(wan_dim) for block_id in self.block_ids}
        )
        self.hidden_down = nn.ModuleDict(
            {str(block_id): nn.Linear(wan_dim, hidden_dim) for block_id in self.block_ids}
        )
        self.control_blocks = nn.ModuleDict(
            {
                str(block_id): nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                )
                for block_id in self.block_ids
            }
        )
        self.zero_outputs = nn.ModuleDict(
            {str(block_id): ZeroLinear(hidden_dim, wan_dim) for block_id in self.block_ids}
        )
        self.register_buffer("conditioning_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    def _condition_tokens(
        self,
        maps: torch.Tensor,
        *,
        token_count: int,
        patch_size: tuple[int, int, int],
    ) -> torch.Tensor:
        encoded = self.condition_encoder(maps.float())
        target_h = max(1, int(maps.shape[-2]) // int(patch_size[1]))
        target_w = max(1, int(maps.shape[-1]) // int(patch_size[2]))
        spatial = target_h * target_w
        if token_count % spatial != 0:
            raise ValueError(
                f"Wan token count {token_count} is not divisible by property grid {target_h}x{target_w}"
            )
        target_f = token_count // spatial
        encoded = F.interpolate(
            encoded,
            size=(target_f, target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
        return encoded.flatten(2).transpose(1, 2).contiguous()

    def forward(
        self,
        block_id: int,
        hidden: torch.Tensor,
        maps: torch.Tensor,
        active: torch.Tensor,
        *,
        patch_size: tuple[int, int, int],
    ) -> torch.Tensor:
        key = str(int(block_id))
        condition = self._condition_tokens(
            maps,
            token_count=int(hidden.shape[1]),
            patch_size=patch_size,
        ).to(device=hidden.device, dtype=hidden.dtype)
        state = self.hidden_down[key](self.hidden_norms[key](hidden)) + condition
        state = state + self.control_blocks[key](state)
        residual = self.zero_outputs[key](state)
        residual = residual * active[:, None, None].to(device=residual.device, dtype=residual.dtype)
        return residual * self.conditioning_scale.to(device=residual.device, dtype=residual.dtype)


class WanPhyCoMultiControlNet(nn.Module):
    """Three independent PhyCo branches injected into frozen Wan blocks."""

    BRANCH_NAMES = ("rigid", "deformation", "force_motion")

    def __init__(
        self,
        *,
        wan_dim: int,
        hidden_dim: int = 128,
        block_ids: Iterable[int] = (3, 8, 13, 18, 23, 28),
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        self.wan_dim = int(wan_dim)
        self.hidden_dim = int(hidden_dim)
        self.block_ids = tuple(int(value) for value in block_ids)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.branches = nn.ModuleDict(
            {
                name: WanPhyCoControlBranch(
                    name=name,
                    map_channels=end - start,
                    wan_dim=self.wan_dim,
                    hidden_dim=self.hidden_dim,
                    block_ids=self.block_ids,
                )
                for name, (start, end) in zip(self.BRANCH_NAMES, BRANCH_SLICES)
            }
        )
        self._last_stats: list[ControlResidualStats] = []

    def set_branch_scales(self, scales: Iterable[float]) -> None:
        values = tuple(float(value) for value in scales)
        if len(values) != len(self.BRANCH_NAMES):
            raise ValueError("branch scales must contain rigid,deformation,force_motion")
        for name, scale in zip(self.BRANCH_NAMES, values):
            self.branches[name].conditioning_scale.fill_(scale)

    def residual(
        self,
        block_id: int,
        hidden: torch.Tensor,
        maps: torch.Tensor | None,
        branch_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        if maps is None or int(block_id) not in self.block_ids:
            return torch.zeros_like(hidden)
        if maps.ndim == 4:
            maps = maps.unsqueeze(0)
        if branch_valid is None:
            branch_valid = torch.ones(maps.shape[0], len(self.BRANCH_NAMES), device=maps.device, dtype=torch.bool)
        elif branch_valid.ndim == 1:
            branch_valid = branch_valid.unsqueeze(0)
        maps = maps.to(device=hidden.device)
        branch_valid = branch_valid.to(device=hidden.device)
        total = torch.zeros_like(hidden)
        for branch_index, (name, (start, end)) in enumerate(zip(self.BRANCH_NAMES, BRANCH_SLICES)):
            residual = self.branches[name](
                int(block_id),
                hidden,
                maps[:, start:end],
                branch_valid[:, branch_index],
                patch_size=self.patch_size,
            )
            total = total + residual
            with torch.no_grad():
                hidden_rms = hidden.float().square().mean().sqrt().clamp_min(1.0e-12)
                residual_rms = residual.float().square().mean().sqrt()
                self._last_stats.append(
                    ControlResidualStats(
                        block_id=int(block_id),
                        branch_name=name,
                        residual_to_hidden_rms=float((residual_rms / hidden_rms).item()),
                        active_fraction=float(branch_valid[:, branch_index].float().mean().item()),
                    )
                )
        return total

    def pop_stats(self) -> list[ControlResidualStats]:
        stats, self._last_stats = self._last_stats, []
        return stats


def inject_wan_phyco_controlnet(
    dit: nn.Module,
    *,
    hidden_dim: int = 128,
    block_ids: Iterable[int] = (3, 8, 13, 18, 23, 28),
) -> WanPhyCoMultiControlNet:
    if hasattr(dit, "phyco_controlnet"):
        return dit.phyco_controlnet
    block_ids = tuple(int(value) for value in block_ids)
    if not block_ids or min(block_ids) < 0 or max(block_ids) >= len(dit.blocks):
        raise ValueError(f"invalid PhyCo block ids {block_ids} for {len(dit.blocks)} Wan blocks")
    controller = WanPhyCoMultiControlNet(
        wan_dim=int(dit.dim),
        hidden_dim=int(hidden_dim),
        block_ids=block_ids,
        patch_size=tuple(int(value) for value in dit.patch_size),
    ).to(device=dit.patch_embedding.weight.device, dtype=dit.patch_embedding.weight.dtype)
    dit.phyco_controlnet = controller
    dit._phyco_control_maps = None
    dit._phyco_branch_valid = None
    dit._phyco_original_block_forwards = {}
    for block_id in block_ids:
        block = dit.blocks[block_id]
        original_forward = block.forward
        dit._phyco_original_block_forwards[block_id] = original_forward

        def controlled_forward(self, x, context, t_mod, freqs, _block_id=block_id, _original=original_forward):
            residual = dit.phyco_controlnet.residual(
                _block_id,
                x,
                getattr(dit, "_phyco_control_maps", None),
                getattr(dit, "_phyco_branch_valid", None),
            )
            return _original(x + residual, context, t_mod, freqs)

        block.forward = types.MethodType(controlled_forward, block)

    for parameter in dit.parameters():
        parameter.requires_grad = False
    controller.requires_grad_(True)
    return controller


def controller_parameter_report(controller: WanPhyCoMultiControlNet) -> dict[str, object]:
    branches = {}
    for name, branch in controller.branches.items():
        branches[name] = {
            "trainable_tensors": sum(1 for p in branch.parameters() if p.requires_grad),
            "trainable_parameters": sum(p.numel() for p in branch.parameters() if p.requires_grad),
        }
    return {
        "architecture": "Wan2.2 PhyCo multi-branch ControlNet-XS",
        "block_ids": list(controller.block_ids),
        "hidden_dim": controller.hidden_dim,
        "branches": branches,
        "total_trainable_parameters": sum(p.numel() for p in controller.parameters() if p.requires_grad),
    }

