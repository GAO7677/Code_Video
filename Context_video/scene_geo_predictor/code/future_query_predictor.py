"""Small observation-motion + visual-scene predictor, with 41 parallel queries."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]/'p4_v2_sg_o_revised_20260916/round1/scripts/model.py'
spec = importlib.util.spec_from_file_location('_visual_scene_common', BASE)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
DIM, STEPS = 128, 41
FPS = 30
SCENE_MODES = ('visual', 'constant', 'motion_only', 'geometry_only')
INPUTS = ('motion', 'object_mask', 'last_position', 'last_velocity', 'future_dt',
          'scene_xyz', 'scene_features', 'scene_mask')


def motion_features(positions, size, times):
    if positions.ndim != 4 or positions.shape[1] != 8 or positions.shape[-1] != 3:
        raise ValueError('Expected [B,8,K,3] observed positions')
    b, _, k, _ = positions.shape
    if size.shape != (b, k, 3) or times.shape != (b, 8):
        raise ValueError('Size/time shape mismatch')
    if not all(torch.isfinite(x).all() for x in (positions, size, times)):
        raise ValueError('Nonfinite motion context')
    dt = times.diff(dim=1)
    if not (dt > 0).all() or not (size > 0).all():
        raise ValueError('Increasing observation times and positive sizes required')
    velocity = positions.diff(dim=1)/dt[:, :, None, None]
    velocity8 = torch.cat((velocity[:, :1], velocity), 1)
    last = positions[:, -1]
    relative = positions-last[:, None]
    clock = (times-times[:, -1:])[:, None].expand(-1, k, -1)
    gravity = positions.new_tensor([0., 0., -9.81]).expand(b, k, -1)
    feature = torch.cat((relative.permute(0, 2, 1, 3).flatten(2),
        velocity8.permute(0, 2, 1, 3).flatten(2), size, last, clock, gravity), -1)
    return feature, last, velocity[:, -1]


def interval_velocity(position, last_position, future_dt):
    before = torch.cat((last_position[:, :, None], position[:, :, :-1]), 2)
    dt = torch.cat((future_dt[:, :1], future_dt.diff(dim=1)), 1)
    if not (dt > 0).all():
        raise ValueError('Future times must be positive and increasing')
    return (position-before)/dt[:, None, :, None]


class FutureAttention(common.GeometryAttention):
    def forward(self, q, relation, valid, features=None):
        b, k, t, _ = q.shape
        n = relation.shape[2]
        key = self.geometry_key(relation)
        value = self.geometry_value(relation)
        if features is not None:
            key = key+self.feature_key(features)[:, None]
            value = value+self.feature_value(features)[:, None]
        query = self.query(self.query_norm(q)).reshape(b, k, t, 4, 32)
        key, value = key.reshape(b, k, n, 4, 32), value.reshape(b, k, n, 4, 32)
        logits = torch.einsum('bkthd,bknhd->bkhtn', query, key)/(32**.5)
        logits = logits+self.geometry_bias(relation).permute(0, 1, 3, 2)[:, :, :, None]
        logits = logits.masked_fill(~valid[:, :, None, None], -torch.inf)
        null = torch.where(valid.any(-1), -torch.inf, 0.)[:, :, None, None, None]
        logits = torch.cat((logits, null.expand(b, k, 4, t, 1)), -1)
        weights = logits.softmax(-1)[..., :n]
        pooled = torch.einsum('bkhtn,bknhd->bkthd', weights, value).flatten(-2)
        q = q+self.output(pooled)
        return q+self.ff(self.ff_norm(q))


class Predictor(nn.Module):
    def __init__(self, motion_mean, motion_std, scene_mode='visual', seed=42,
                 strict_future_grid=True):
        super().__init__()
        if scene_mode not in SCENE_MODES:
            raise ValueError('Unknown scene mode')
        self.scene_mode = scene_mode
        self.strict_future_grid = bool(strict_future_grid)
        if motion_mean.shape != (65,) or motion_std.shape != (65,):
            raise ValueError('Expected training-only 65-D motion statistics')
        if not torch.isfinite(motion_mean).all() or not torch.isfinite(motion_std).all() or not (motion_std > 0).all():
            raise ValueError('Invalid training motion statistics')
        self.register_buffer('motion_mean', motion_mean.detach().clone())
        self.register_buffer('motion_std', motion_std.detach().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.motion_encoder = common._mlp(65, DIM, DIM)
            self.time_encoder = common._mlp(3, DIM, DIM)
            self.scene_projection = nn.Sequential(nn.LayerNorm(1386), nn.Linear(1386, DIM))
            self.reader = nn.ModuleList([FutureAttention(), FutureAttention()])
            self.position = common._mlp(DIM, DIM, 3, zero_output=True)

        # Experimental controls use the same registered modules and initialization
        # as the legacy arms, but remove unused parameters from the optimizer.
        if scene_mode == 'geometry_only':
            self.scene_projection.requires_grad_(False)
            for layer in self.reader:
                layer.feature_key.requires_grad_(False)
                layer.feature_value.requires_grad_(False)
        elif scene_mode == 'motion_only':
            self.scene_projection.requires_grad_(False)
            self.reader.requires_grad_(False)

    def active_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, batch):
        if set(batch) != set(INPUTS):
            raise ValueError('Forward accepts only declared observation inputs, never targets or oracle geometry')
        mask = batch['object_mask'].bool()
        if mask.ndim != 2:
            raise ValueError('Expected object_mask [B,K]')
        b, k = mask.shape
        if batch['motion'].shape != (b, k, 65):
            raise ValueError('Expected motion [B,K,65]')
        for name in ('last_position', 'last_velocity'):
            if batch[name].shape != (b, k, 3):
                raise ValueError(f'Expected {name} [B,K,3]')
        for name in ('motion', 'last_position', 'last_velocity'):
            if not torch.isfinite(batch[name]).all():
                raise ValueError(f'Nonfinite {name}')
        time = batch['future_dt'].float()
        if time.shape != (mask.shape[0], STEPS) or not torch.isfinite(time).all() or not (time > 0).all() or not (time.diff(dim=-1) > 0).all():
            raise ValueError('Expected 41 positive increasing future times')
        if self.strict_future_grid:
            expected = torch.arange(1, STEPS + 1, device=time.device,
                                    dtype=time.dtype)[None] / FPS
            if not torch.allclose(time, expected.expand_as(time), atol=1e-6, rtol=0):
                raise ValueError('Expected future_dt to map RGB8-RGB48 at 30 FPS')
        h = self.motion_encoder((batch['motion'].float()-self.motion_mean)/self.motion_std)
        e = self.time_encoder(torch.stack((time, time.square(), torch.sin(torch.pi*time)), -1))
        q = h[:, :, None]+e[:, None]
        if self.scene_mode == 'motion_only':
            residual = self.position(q)*time[:, None, :, None]
            cv = batch['last_velocity'][:, :, None]*time[:, None, :, None]
            result = batch['last_position'][:, :, None]+cv+residual
            return result*mask[:, :, None, None]
        scene_mask = batch['scene_mask'].bool()
        if scene_mask.ndim != 2 or scene_mask.shape[0] != b:
            raise ValueError('Expected scene_mask [B,N]')
        n = scene_mask.shape[1]
        if batch['scene_xyz'].shape != (b, n, 3):
            raise ValueError('Expected scene_xyz [B,N,3]')
        if batch['scene_features'].shape != (b, n, 1386):
            raise ValueError('Expected scene_features [B,N,1386]')
        if not torch.isfinite(batch['scene_xyz']).all() or not torch.isfinite(batch['scene_features']).all():
            raise ValueError('Nonfinite scene input')
        raw = torch.where(scene_mask[..., None], batch['scene_features'].float(), 0.)
        xyz = torch.where(scene_mask[..., None], batch['scene_xyz'].float(), 0.)
        delta = xyz[:, None]-batch['last_position'][:, :, None]
        distance = delta.norm(dim=-1, keepdim=True)
        relation = torch.cat((delta/(.25+distance), distance/(.25+distance)), -1)
        valid = scene_mask[:, None]&mask[:, :, None]
        relation = torch.where(valid[..., None], relation, 0.)
        if self.scene_mode == 'constant':
            raw = torch.zeros_like(raw)
            relation = torch.zeros_like(relation)
            valid = mask[:, :, None].expand(-1, -1, raw.shape[1])
        if self.scene_mode == 'geometry_only':
            features = None
        else:
            features = self.scene_projection(raw)
        for layer in self.reader:
            q = layer(q, relation, valid, features)
        # One query corresponds directly to one real time; no legacy 24-Hz resampling.
        residual = self.position(q)*time[:, None, :, None]
        cv = batch['last_velocity'][:, :, None]*time[:, None, :, None]
        result = batch['last_position'][:, :, None]+cv+residual
        return result*mask[:, :, None, None]


def objective(prediction, target, batch):
    mask = batch['object_mask'].bool()
    if not mask.any():
        return prediction.sum()*0
    position = nn.functional.smooth_l1_loss(prediction[mask], target[mask])
    pred_v = interval_velocity(prediction, batch['last_position'], batch['future_dt'])
    true_v = interval_velocity(target, batch['last_position'], batch['future_dt'])
    return position+.2*nn.functional.smooth_l1_loss(pred_v[mask], true_v[mask])
