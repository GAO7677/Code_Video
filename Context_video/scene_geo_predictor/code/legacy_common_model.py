"""Predictor-only SG-O models; no pretrained models or video modules are loaded."""

from contextlib import nullcontext
import math

import torch
from torch import nn


VARIANTS = ("surface", "geometry_raw", "geometry_bounded", "geometry_capacity", "utonia_bounded")
DIM = 128
FUTURE = 41
LENGTH_SCALE_M = 0.25


def _mlp(input_dim, hidden_dim, output_dim, *, zero_output=False):
    module = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                           nn.Linear(hidden_dim, output_dim))
    if zero_output:
        nn.init.zeros_(module[-1].weight)
        nn.init.zeros_(module[-1].bias)
    return module


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.node = nn.Sequential(_mlp(90, DIM, DIM), nn.SiLU())
        self.edge = nn.Sequential(_mlp(54, DIM, DIM), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(2 * DIM, DIM), nn.SiLU())
        self.position = nn.Linear(DIM, FUTURE * 3)
        self.contact = nn.Linear(DIM, FUTURE)

    def encode(self, node, edge, mask):
        keep = mask[:, :, None] & mask[:, None, :]
        keep = keep & ~torch.eye(2, dtype=torch.bool, device=mask.device)[None]
        messages = self.edge(edge) * keep[..., None]
        messages = messages.sum(2) / keep.sum(2).clamp_min(1)[..., None]
        h = self.fuse(torch.cat((self.node(node), messages), dim=-1))
        return h * mask[..., None]


class GeometryAttention(nn.Module):
    """One motion query per object, with query-dependent geometric keys."""

    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(DIM)
        self.query = nn.Linear(DIM, DIM)
        self.geometry_key = _mlp(4, DIM, DIM)
        self.geometry_value = _mlp(4, DIM, DIM)
        self.geometry_bias = _mlp(4, 32, 4)
        self.feature_key = nn.Linear(DIM, DIM, bias=False)
        self.feature_value = nn.Linear(DIM, DIM, bias=False)
        self.output = nn.Linear(DIM, DIM)
        self.ff_norm = nn.LayerNorm(DIM)
        self.ff = _mlp(DIM, 2 * DIM, DIM)

    def forward(self, h, relation, valid, features=None):
        batch, objects, points, _ = relation.shape
        key, value = self.geometry_key(relation), self.geometry_value(relation)
        if features is not None:
            key = key + self.feature_key(features)
            value = value + self.feature_value(features)
        query = self.query(self.query_norm(h)).reshape(batch, objects, 4, 32)
        key = key.reshape(batch, objects, points, 4, 32)
        value = value.reshape(batch, objects, points, 4, 32)
        logits = torch.einsum("bkhd,bkmhd->bkhm", query, key) / (32 ** 0.5)
        logits = logits + self.geometry_bias(relation).permute(0, 1, 3, 2)
        logits = logits.masked_fill(~valid[:, :, None], -torch.inf)
        # Enable a null value only for empty rows, avoiding an all-infinite softmax.
        null_logit = torch.where(valid.any(-1), -torch.inf, 0.0)
        logits = torch.cat((logits, null_logit[:, :, None, None].expand(-1, -1, 4, 1)), -1)
        weights = logits.softmax(-1)[..., :points]
        pooled = torch.einsum("bkhm,bkmhd->bkhd", weights, value).flatten(2)
        h = h + self.output(pooled)
        return h + self.ff(self.ff_norm(h))


def canonical_to_query(residual, logits, future_dt):
    """Interpolate a 24 Hz residual grid in meters, preserving native sample times."""
    if future_dt.shape != (residual.shape[0], FUTURE):
        raise ValueError("future_dt must have shape [B,41]")
    if not bool(torch.isfinite(future_dt).all() and (future_dt > 0).all()
                and (future_dt.diff(dim=-1) > 0).all()
                and (future_dt <= FUTURE / 24 + 1e-6).all()):
        raise ValueError("Future times must increase within the 24 Hz head's time range")
    grid = torch.arange(FUTURE + 1, device=residual.device, dtype=residual.dtype) / 24
    query = future_dt.clamp(0, grid[-1]).contiguous()
    hi = torch.searchsorted(grid, query).clamp(1, FUTURE)
    lo = hi - 1
    query = torch.where((query - grid[lo]).abs() <= 1e-6, grid[lo], query)
    query = torch.where((query - grid[hi]).abs() <= 1e-6, grid[hi], query)
    weight = (query - grid[lo]) / (grid[hi] - grid[lo])
    residual = torch.cat((torch.zeros_like(residual[:, :, :1]), residual), dim=2)
    contact = torch.cat((logits[:, :, :1], logits), dim=2)[..., None]

    def interpolate(values):
        shape = (values.shape[0], values.shape[1], FUTURE, values.shape[-1])
        a = values.gather(2, lo[:, None, :, None].expand(shape))
        b = values.gather(2, hi[:, None, :, None].expand(shape))
        return a * (1 - weight[:, None, :, None]) + b * weight[:, None, :, None]

    return interpolate(residual), interpolate(contact).squeeze(-1)


class Predictor(nn.Module):
    """Five matched variants with identical initial common parameters.

    Coordinates and times retain meters and seconds. All geometry variants use
    the same object-centered radius mask without changing coordinates or rows.
    """

    def __init__(self, variant, stats, seed=42, radius_m=15.0, length_scale_m=LENGTH_SCALE_M):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown predictor variant: {variant}")
        self.variant = variant
        for name, value in (("radius_m", radius_m), ("length_scale_m", length_scale_m)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        for name, shape in (("node_mean", (90,)), ("node_std", (90,)),
                            ("edge_mean", (54,)), ("edge_std", (54,)),
                            ("target_mean", (FUTURE, 3)), ("target_std", (FUTURE, 3))):
            tensor = torch.as_tensor(stats[name], dtype=torch.float32).detach().clone()
            if tensor.shape != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid normalization array {name}: expected {shape}")
            if name.endswith("std") and not bool((tensor > 0).all()):
                raise ValueError(f"Normalization standard deviation must be positive: {name}")
            self.register_buffer(name, tensor)
        # Identical registration and isolated CPU RNG preserve all shared initial
        # weights regardless of which paths a variant activates.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.core = Dynamics()
            self.history_time = nn.Linear(8, DIM, bias=False)
            nn.init.zeros_(self.history_time.weight)
            self.history_residual = _mlp(DIM, DIM, DIM, zero_output=True)
            self.surface_encoder = _mlp(3, DIM, DIM)
            self.surface_residual = _mlp(2 * DIM, DIM, DIM, zero_output=True)
            self.scene_projection = nn.Sequential(nn.LayerNorm(1386), nn.Linear(1386, DIM))
            self.geometry_attention = nn.ModuleList([GeometryAttention(), GeometryAttention()])
            self.geometry_residual = _mlp(2 * DIM, DIM, DIM, zero_output=True)
            self.query_correction = _mlp(DIM + 3, DIM, 4, zero_output=True)
        # Append capacity under a separate RNG so every original draw is retained.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) + 104729)
            self.capacity_adapter = _mlp(DIM, 956, DIM)
        self.use_surface = True
        self.use_geometry = variant != "surface"
        self.use_features = variant == "utonia_bounded"
        self.use_capacity = variant == "geometry_capacity"
        if not self.use_capacity:
            self.capacity_adapter.requires_grad_(False)
        if not self.use_geometry:
            self.geometry_attention.requires_grad_(False)
            self.geometry_residual.requires_grad_(False)
        if not self.use_features:
            self.scene_projection.requires_grad_(False)
            for layer in self.geometry_attention:
                layer.feature_key.requires_grad_(False)
                layer.feature_value.requires_grad_(False)

    def active_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def history_features(self, batch):
        mask = batch["mask"].bool()
        node = torch.where(mask[..., None], batch["node"].float(), 0.0)
        edge_mask = mask[:, :, None] & mask[:, None, :]
        edge = torch.where(edge_mask[..., None], batch["edge"].float(), 0.0)
        node = (node - self.node_mean) / self.node_std
        edge = (edge - self.edge_mean) / self.edge_std
        h = self.core.encode(node, edge, mask)
        clock = torch.arange(-7, 1, device=h.device, dtype=h.dtype) / 24
        h = h + self.history_time(batch["history_dt"].float() - clock)[:, None]
        return (h + self.history_residual(h)) * mask[..., None]

    def surface_features(self, batch):
        mask = batch["surface_mask"].bool() & batch["mask"].bool()[..., None]
        relative = (batch["surface"].float() - batch["last_position"].float()[:, :, None]) / LENGTH_SCALE_M
        relative = torch.where(mask[..., None], relative, 0.0)
        encoded = self.surface_encoder(relative)
        pooled = (encoded * mask[..., None]).sum(2) / mask.sum(2).clamp_min(1)[..., None]
        return pooled, mask.any(-1)

    def scene_inputs(self, batch):
        object_mask = batch["mask"].bool()
        surface_mask = batch["surface_mask"].bool() & object_mask[..., None]
        scene_mask = batch["scene_mask"].bool()
        xyz = torch.where(scene_mask[..., None], batch["scene_xyz"].float(), 0.0)
        last = batch["last_position"].float()[:, :, None]
        surface = torch.where(surface_mask[..., None], batch["surface"].float() - last, 0.0)
        batch_size, points, _ = xyz.shape
        if points == 0:
            return (xyz.new_zeros((batch_size, 2, 0, 4)),
                    scene_mask.new_zeros((batch_size, 2, 0)),
                    xyz.new_zeros((batch_size, 2, 0, DIM)) if self.use_features else None)
        delta = xyz[:, None] - last
        center_distance = torch.linalg.vector_norm(delta, dim=-1)
        valid = scene_mask[:, None] & surface_mask.any(-1)[..., None] & object_mask[..., None]
        valid = valid & (center_distance <= self.radius_m)
        # Clear rejected rows before cdist or an MLP, including extreme padded data.
        delta = torch.where(valid[..., None], delta, 0.0)
        distance = torch.cdist(delta, surface)
        distance = distance.masked_fill(~surface_mask[:, :, None], torch.inf).amin(-1)
        distance = torch.where(valid, distance, 0.0)
        if self.variant == "geometry_raw":
            offset = delta / LENGTH_SCALE_M
            proximity = torch.log1p(distance / LENGTH_SCALE_M)
        else:
            offset = delta / (self.length_scale_m + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))
            proximity = distance / (self.length_scale_m + distance)
        relation = torch.cat((offset, proximity[..., None]), dim=-1)
        relation = torch.where(valid[..., None], relation, 0.0)
        features = None
        if self.use_features:
            raw = torch.where(valid.any(1)[..., None], batch["scene_features"].float(), 0.0)
            features = self.scene_projection(raw)[:, None].expand(-1, 2, -1, -1)
            features = torch.where(valid[..., None], features, 0.0)
        return relation, valid, features

    def geometry_features(self, h, batch):
        relation, valid, features = self.scene_inputs(batch)
        result = h
        for layer in self.geometry_attention:
            result = layer(result, relation, valid, features)
        if self.use_capacity:
            result = result + self.capacity_adapter(result)
        return result, valid.any(-1)

    def forward(self, batch):
        device_type = batch["node"].device.type
        precision = torch.autocast(device_type, enabled=False) if device_type in ("cpu", "cuda") else nullcontext()
        with precision:
            h = self.history_features(batch)
            if self.use_surface:
                surface, valid = self.surface_features(batch)
                h = h + self.surface_residual(torch.cat((h, surface), -1)) * valid[..., None]
            if self.use_geometry:
                geometry, valid = self.geometry_features(h, batch)
                h = h + self.geometry_residual(torch.cat((h, geometry), -1)) * valid[..., None]
            prior = self.core.position(h).reshape(-1, 2, FUTURE, 3)
            prior = prior * self.target_std + self.target_mean
            future_dt = batch["future_dt"].float()
            residual, contact = canonical_to_query(prior, self.core.contact(h), future_dt)
            tau = future_dt / (FUTURE / 24)
            time = torch.stack((tau, tau.square(), torch.sin(torch.pi * tau)), -1)
            query = torch.cat((h[:, :, None].expand(-1, -1, FUTURE, -1),
                               time[:, None].expand(-1, 2, -1, -1)), -1)
            correction = self.query_correction(query)
            residual = residual + correction[..., :3] * tau[:, None, :, None]
            contact = contact + correction[..., 3]
            relative = residual + batch["cv"].float()
            mask = batch["mask"].bool()
            return {
                "relative": relative * mask[:, :, None, None],
                "position": (relative + batch["last_position"].float()[:, :, None]) * mask[:, :, None, None],
                "contact_logits": contact * mask[:, :, None],
                "normalized_residual": ((residual - self.target_mean) / self.target_std) * mask[:, :, None, None],
            }
