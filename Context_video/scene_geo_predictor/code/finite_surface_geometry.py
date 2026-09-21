"""Versioned finite-surface geometry adapter and predictor.

The point-geometry arm in this project consumes padded ``scene_xyz`` tokens.
This module adds a separate, explicit rectangular-surface interface for the
phase-one structured-geometry pilot.  It intentionally contains no future
state, event, collider-id, or family-id input.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import torch
from torch import nn

from future_query_predictor import common, interval_velocity, motion_features


SURFACE_SCHEMA = "finite_surface_geometry_v1"
SURFACE_FEATURE_DIM = 11
SURFACE_RELATION_DIM = 9
SURFACE_DIM = 128
SURFACE_HEADS = 4
SURFACE_HEAD_DIM = SURFACE_DIM // SURFACE_HEADS
SURFACE_TOKEN_COUNT = 48
POINT_TOKEN_COUNT = 1792
POINT_VALID_COUNT = 512
FPS = 30
STEPS = 41
BASE_LENGTH_M = 0.25


def _unit(value: np.ndarray, name: str, *, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(value, axis=axis, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 1e-8):
        raise ValueError(f"{name} contains a zero or non-finite vector")
    return value / norm


def canonicalize_surfaces(
    center: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    half_extents: np.ndarray,
    surface_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Validate and orthogonalize an explicit finite-rectangle batch.

    Arrays are ``[S, ...]`` and padding is represented by ``surface_mask``.
    Valid surface centers and directions remain in world coordinates; the
    predictor forms the object-relative relation to ``last_position``.
    """
    center = np.asarray(center, dtype=np.float32)
    normal = np.asarray(normal, dtype=np.float32)
    tangent_u = np.asarray(tangent_u, dtype=np.float32)
    half_extents = np.asarray(half_extents, dtype=np.float32)
    surface_mask = np.asarray(surface_mask, dtype=bool)
    if center.ndim != 2 or center.shape[-1] != 3:
        raise ValueError("center must have shape [S,3]")
    s = center.shape[0]
    if normal.shape != (s, 3) or tangent_u.shape != (s, 3):
        raise ValueError("normal and tangent_u must have shape [S,3]")
    if half_extents.shape != (s, 2) or surface_mask.shape != (s,):
        raise ValueError("half_extents/mask shape mismatch")
    if not all(np.isfinite(x).all() for x in (center, normal, tangent_u, half_extents)):
        raise ValueError("surface arrays must be finite")
    if np.any(half_extents[surface_mask] <= 0):
        raise ValueError("valid surface half extents must be positive")
    # Directions are normalized after removing any accidental normal component.
    n = np.zeros_like(normal)
    u = np.zeros_like(tangent_u)
    v = np.zeros_like(tangent_u)
    if np.any(surface_mask):
        n_valid = _unit(normal[surface_mask], "normal")
        u_raw = tangent_u[surface_mask]
        u_proj = u_raw - (u_raw * n_valid).sum(-1, keepdims=True) * n_valid
        u_valid = _unit(u_proj, "tangent_u")
        v_valid = _unit(np.cross(n_valid, u_valid), "tangent_v")
        n[surface_mask], u[surface_mask], v[surface_mask] = n_valid, u_valid, v_valid
    return {
        "surface_center": center,
        "surface_normal": n,
        "surface_tangent_u": u,
        "surface_tangent_v": v,
        "surface_half_extents": half_extents,
        "surface_mask": surface_mask,
    }


def rectangular_surface_points(
    center: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    half_extents: np.ndarray,
    *,
    resolution: int = 8,
) -> np.ndarray:
    """Return a finite, edge-inclusive grid for one rectangle."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    clean = canonicalize_surfaces(
        np.asarray(center)[None], np.asarray(normal)[None],
        np.asarray(tangent_u)[None], np.asarray(half_extents)[None],
        np.ones(1, dtype=bool),
    )
    c = clean["surface_center"][0]
    n = clean["surface_normal"][0]
    u = clean["surface_tangent_u"][0]
    v = clean["surface_tangent_v"][0]
    uv = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    uu, vv = np.meshgrid(uv, uv, indexing="xy")
    return (c[None, None] + uu[..., None] * clean["surface_half_extents"][0, 0] * u
            + vv[..., None] * clean["surface_half_extents"][0, 1] * v).reshape(-1, 3)


def surface_point_pool(surfaces: Mapping[str, np.ndarray], resolution: int = 8):
    """Build a deterministic candidate point pool from the same finite faces."""
    clean = canonicalize_surfaces(
        surfaces["surface_center"], surfaces["surface_normal"],
        surfaces["surface_tangent_u"], surfaces["surface_half_extents"],
        surfaces["surface_mask"],
    )
    points, source = [], []
    for index in np.flatnonzero(clean["surface_mask"]):
        points.append(rectangular_surface_points(
            clean["surface_center"][index], clean["surface_normal"][index],
            clean["surface_tangent_u"][index], clean["surface_half_extents"][index],
            resolution=resolution,
        ))
        source.extend([int(index)] * (resolution * resolution))
    if not points:
        raise ValueError("at least one valid surface is required")
    points = np.concatenate(points, axis=0).astype(np.float32)
    source = np.asarray(source, dtype=np.int16)
    return points, source, hashlib.sha256(points.tobytes() + source.tobytes()).hexdigest()


def sample_surface_points(
    surfaces: Mapping[str, np.ndarray], seed: int, count: int = POINT_VALID_COUNT,
    resolution: int = 8,
) -> dict[str, np.ndarray]:
    """Sample exactly ``count`` points while touching every valid surface first."""
    points, source, pool_hash = surface_point_pool(surfaces, resolution=resolution)
    valid_surface = np.flatnonzero(np.asarray(surfaces["surface_mask"], dtype=bool))
    if count < len(valid_surface) or count > len(points):
        raise ValueError(f"point count {count} incompatible with {len(valid_surface)} surfaces/{len(points)} candidates")
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    # One candidate per surface is a predeclared coverage rule, not outcome
    # selection.  The remaining points use one global deterministic shuffle.
    for surface_id in valid_surface:
        ids = np.flatnonzero(source == surface_id)
        chosen.append(int(rng.permutation(ids)[0]))
    remaining = np.setdiff1d(np.arange(len(points), dtype=np.int64), np.asarray(chosen), assume_unique=False)
    chosen.extend(rng.permutation(remaining)[: count - len(chosen)].tolist())
    chosen = np.asarray(chosen, dtype=np.int64)
    chosen = chosen[rng.permutation(len(chosen))]
    if len(np.unique(chosen)) != count:
        raise AssertionError("point sampler returned duplicate candidate indices")
    return {
        "point_xyz": points[chosen].astype(np.float32),
        "point_source_surface": source[chosen].astype(np.int16),
        "candidate_pool_hash": pool_hash,
        "candidate_pool_count": np.asarray([len(points)], dtype=np.int64),
    }


def _surface_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return common._mlp(input_dim, SURFACE_DIM, output_dim)


class _SurfaceAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(SURFACE_DIM)
        self.query = nn.Linear(SURFACE_DIM, SURFACE_DIM)
        self.geometry_key = _surface_mlp(SURFACE_RELATION_DIM, SURFACE_DIM)
        self.geometry_value = _surface_mlp(SURFACE_RELATION_DIM, SURFACE_DIM)
        self.geometry_bias = _surface_mlp(SURFACE_RELATION_DIM, SURFACE_HEADS)
        # The shared surface encoder produces the 128-D token consumed by
        # every reader layer; keeping this projection 128-D avoids bypassing
        # the explicit finite-surface encoder.
        self.surface_key = nn.Linear(SURFACE_DIM, SURFACE_DIM, bias=False)
        self.surface_value = nn.Linear(SURFACE_DIM, SURFACE_DIM, bias=False)
        self.output = nn.Linear(SURFACE_DIM, SURFACE_DIM)
        self.ff_norm = nn.LayerNorm(SURFACE_DIM)
        self.ff = common._mlp(SURFACE_DIM, 2 * SURFACE_DIM, SURFACE_DIM)

    def forward(self, q, relation, valid, features):
        b, k, t, _ = q.shape
        s = relation.shape[2]
        encoded_key = self.surface_key(features)
        encoded_value = self.surface_value(features)
        # Surface features may be shared as [B,S,D] or object-relative as
        # [B,K,S,D].  The latter is used here because the relative delta is
        # formed against each object's last position.
        if encoded_key.ndim == 3:
            encoded_key = encoded_key[:, None]
            encoded_value = encoded_value[:, None]
        key = self.geometry_key(relation) + encoded_key
        value = self.geometry_value(relation) + encoded_value
        query = self.query(self.query_norm(q)).reshape(b, k, t, SURFACE_HEADS, SURFACE_HEAD_DIM)
        key = key.reshape(b, k, s, SURFACE_HEADS, SURFACE_HEAD_DIM)
        value = value.reshape(b, k, s, SURFACE_HEADS, SURFACE_HEAD_DIM)
        logits = torch.einsum("bkthd,bkshd->bkhts", query, key) / (SURFACE_HEAD_DIM ** 0.5)
        logits = logits + self.geometry_bias(relation).permute(0, 1, 3, 2)[:, :, :, None]
        logits = logits.masked_fill(~valid[:, :, None, None], -torch.inf)
        null = torch.where(valid.any(-1), -torch.inf, 0.0)[:, :, None, None, None]
        logits = torch.cat((logits, null.expand(b, k, SURFACE_HEADS, t, 1)), -1)
        weights = logits.softmax(-1)[..., :s]
        pooled = torch.einsum("bkhts,bkshd->bkthd", weights, value).flatten(-2)
        q = q + self.output(pooled)
        return q + self.ff(self.ff_norm(q))


class FiniteSurfacePredictor(nn.Module):
    """Same motion/time/query trunk with explicit finite surface tokens."""

    INPUTS = (
        "motion", "object_mask", "last_position", "last_velocity", "future_dt",
        "surface_center", "surface_normal", "surface_tangent_u",
        "surface_half_extents", "surface_mask",
    )

    def __init__(self, motion_mean: torch.Tensor, motion_std: torch.Tensor,
                 seed: int = 42, strict_future_grid: bool = True):
        super().__init__()
        if motion_mean.shape != (65,) or motion_std.shape != (65,):
            raise ValueError("expected 65-D motion statistics")
        if not torch.isfinite(motion_mean).all() or not torch.isfinite(motion_std).all() or not (motion_std > 0).all():
            raise ValueError("invalid motion statistics")
        self.strict_future_grid = bool(strict_future_grid)
        self.register_buffer("motion_mean", motion_mean.detach().clone())
        self.register_buffer("motion_std", motion_std.detach().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.motion_encoder = common._mlp(65, SURFACE_DIM, SURFACE_DIM)
            self.time_encoder = common._mlp(3, SURFACE_DIM, SURFACE_DIM)
            self.surface_encoder = nn.Sequential(
                nn.LayerNorm(SURFACE_FEATURE_DIM), nn.Linear(SURFACE_FEATURE_DIM, SURFACE_DIM)
            )
            self.reader = nn.ModuleList([_SurfaceAttention(), _SurfaceAttention()])
            self.position = common._mlp(SURFACE_DIM, SURFACE_DIM, 3, zero_output=True)

    def active_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _validate(batch):
        if set(batch) != set(FiniteSurfacePredictor.INPUTS):
            raise ValueError("forward accepts only declared observation and surface inputs")
        mask = batch["object_mask"].bool()
        if mask.ndim != 2:
            raise ValueError("object_mask must have shape [B,K]")
        b, k = mask.shape
        if batch["motion"].shape != (b, k, 65):
            raise ValueError("motion must have shape [B,K,65]")
        for name in ("last_position", "last_velocity"):
            if batch[name].shape != (b, k, 3):
                raise ValueError(f"{name} shape mismatch")
        time = batch["future_dt"].float()
        if time.shape != (b, STEPS) or not torch.isfinite(time).all() or not (time > 0).all() or not (time.diff(dim=-1) > 0).all():
            raise ValueError("future_dt must be 41 increasing positive times")
        if not torch.isfinite(batch["motion"]).all() or not torch.isfinite(batch["last_position"]).all() or not torch.isfinite(batch["last_velocity"]).all():
            raise ValueError("non-finite motion input")
        center = batch["surface_center"]
        smask = batch["surface_mask"].bool()
        if center.ndim != 3 or center.shape[0] != b or center.shape[-1] != 3:
            raise ValueError("surface_center must have shape [B,S,3]")
        s = center.shape[1]
        for name in ("surface_normal", "surface_tangent_u"):
            if batch[name].shape != (b, s, 3):
                raise ValueError(f"{name} shape mismatch")
        if batch["surface_half_extents"].shape != (b, s, 2) or smask.shape != (b, s):
            raise ValueError("surface extent/mask shape mismatch")
        for name in ("surface_center", "surface_normal", "surface_tangent_u", "surface_half_extents"):
            if not torch.isfinite(batch[name]).all():
                raise ValueError(f"non-finite {name}")
        if (batch["surface_half_extents"][smask] <= 0).any():
            raise ValueError("valid surface extents must be positive")
        if not torch.allclose(time, torch.arange(1, STEPS + 1, device=time.device, dtype=time.dtype)[None] / FPS, atol=1e-6, rtol=0):
            raise ValueError("future_dt must map RGB8-RGB48 at 30 FPS")
        return mask, time, smask

    def forward(self, batch):
        mask, time, smask = self._validate(batch)
        b, k = mask.shape
        h = self.motion_encoder((batch["motion"].float() - self.motion_mean) / self.motion_std)
        e = self.time_encoder(torch.stack((time, time.square(), torch.sin(torch.pi * time)), -1))
        q = h[:, :, None] + e[:, None]
        center = batch["surface_center"].float()
        normal = batch["surface_normal"].float()
        tangent_u = batch["surface_tangent_u"].float()
        tangent_v = torch.linalg.cross(normal, tangent_u, dim=-1)
        tangent_v = tangent_v / tangent_v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        half = batch["surface_half_extents"].float()
        delta = center[:, None] - batch["last_position"].float()[:, :, None]
        distance = delta.norm(dim=-1, keepdim=True)
        scale = BASE_LENGTH_M + distance
        relation = torch.cat((delta / scale, distance / scale,
                              (delta * normal[:, None]).sum(-1, keepdim=True) / scale,
                              (delta * tangent_u[:, None]).sum(-1, keepdim=True) / scale,
                              (delta * tangent_v[:, None]).sum(-1, keepdim=True) / scale,
                              half[:, None] / scale), -1)
        raw = torch.cat((delta / scale, normal[:, None].expand(-1, k, -1, -1),
                         tangent_u[:, None].expand(-1, k, -1, -1),
                         half[:, None] / scale), -1)
        valid = smask[:, None] & mask[:, :, None]
        relation = torch.where(valid[..., None], relation, 0.0)
        raw = torch.where(valid[..., None], raw, 0.0)
        features = self.surface_encoder(raw)
        for layer in self.reader:
            q = layer(q, relation, valid, features)
        residual = self.position(q) * time[:, None, :, None]
        cv = batch["last_velocity"][:, :, None] * time[:, None, :, None]
        result = batch["last_position"][:, :, None] + cv + residual
        return result * mask[:, :, None, None]


__all__ = [
    "FiniteSurfacePredictor", "SURFACE_SCHEMA", "SURFACE_TOKEN_COUNT",
    "POINT_TOKEN_COUNT", "POINT_VALID_COUNT", "canonicalize_surfaces",
    "rectangular_surface_points", "surface_point_pool", "sample_surface_points",
    "motion_features", "interval_velocity",
]
