"""Bounded CPU toy: identical observed motion, two fixed-wall futures.

This is a computation-graph smoke/fit diagnostic, not evidence of simulator
generalization. The wall is represented by observed scene points only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from future_query_predictor import Predictor, motion_features, objective


def make_fixture():
    torch.set_num_threads(2)
    times = torch.arange(8, dtype=torch.float32) / 30
    observed = torch.zeros(2, 8, 1, 3)
    observed[:, :, 0, 0] = -0.60 + 1.50 * times
    observed[:, :, 0, 2] = 0.10
    size = torch.full((2, 1, 3), .10)
    motion, last, velocity = motion_features(observed, size, times[None].expand(2, -1))
    future_dt = torch.arange(1, 42, dtype=torch.float32)[None].expand(2, -1) / 30
    # Case 0 hits a wall at x=0 and reverses; case 1 has a far wall and keeps
    # moving. Both cases have exactly the same observation motion.
    free = last[:, :, None] + velocity[:, :, None] * future_dt[:, None, :, None]
    target = free.clone()
    hit = target[0, 0, :, 0] > -.05
    target[0, 0, hit, 0] = -.05 - (target[0, 0, hit, 0] + .05)
    target[0, 0, :, 2] = .10
    scene_xyz = torch.zeros(2, 4, 3)
    scene_xyz[0, :, 0] = 0.0
    scene_xyz[1, :, 0] = 0.8
    scene_xyz[:, :, 1] = torch.tensor([-.3, -.1, .1, .3])
    scene_xyz[:, :, 2] = .10
    batch = dict(motion=motion, object_mask=torch.ones(2, 1, dtype=torch.bool),
                 last_position=last, last_velocity=velocity, future_dt=future_dt,
                 scene_xyz=scene_xyz, scene_features=torch.zeros(2, 4, 1386),
                 scene_mask=torch.ones(2, 4, dtype=torch.bool))
    return batch, target


def run(steps: int):
    batch, target = make_fixture()
    model = Predictor(torch.zeros(65), torch.ones(65), scene_mode='geometry_only')
    optimizer = torch.optim.AdamW(model.active_parameters(), lr=2e-3, weight_decay=0.)
    with torch.no_grad():
        initial_prediction = model(batch)
        initial_loss = float(objective(initial_prediction, target, batch))
    losses = [initial_loss]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = objective(prediction, target, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite toy loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.active_parameters()), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        final_prediction = model(batch)
        final_loss = float(objective(final_prediction, target, batch))
        scene_difference = float((final_prediction[0] - final_prediction[1]).norm(dim=-1).mean())
    return dict(schema='future_query_toy_smoke_v1', steps=steps, batch=2,
                initial_loss=initial_loss, final_loss=final_loss,
                loss_reduction_fraction=1. - final_loss / max(initial_loss, 1e-12),
                same_observation_motion=True, different_scene_points=True,
                final_scene_conditioned_prediction_difference_m=scene_difference,
                finite=True, no_gpu=True,
                interpretation='bounded two-case CPU fit only; not simulator evidence',
                loss_curve=losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 128:
        raise ValueError('steps must be in [1,128]')
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
