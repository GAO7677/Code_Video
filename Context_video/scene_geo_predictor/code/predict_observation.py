"""CPU inference from observed geometry and cached visual features, without labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from future_query_predictor import Predictor, motion_features
from prepare_small_trial import sha


def observation_batch(geometry_path, scene_path):
    with np.load(geometry_path, allow_pickle=False) as data:
        position = np.array(data['positions_world'], dtype=np.float32)
        size = np.array(data['size_m'], dtype=np.float32)
        times = np.array(data['frame_times'], dtype=np.float32)
    if position.shape != (8, 1, 3) or size.shape != (1, 3) or times.shape != (8,):
        raise ValueError('Supported context: one object, eight positions, one size, eight times')
    if not np.allclose(times-times[0], np.arange(8)/30, atol=1e-6, rtol=0):
        raise ValueError('This checkpoint requires a 30 FPS context')
    with np.load(scene_path, allow_pickle=False) as data:
        scene = {k: np.array(data[k]) for k in ('scene_xyz', 'scene_features', 'scene_mask')}
    n = len(scene['scene_xyz'])
    if not 0 < n <= 1792 or scene['scene_xyz'].shape != (n, 3) or scene['scene_features'].shape != (n, 1386) or scene['scene_mask'].shape != (n,):
        raise ValueError('Expected up to 1792 scene tokens with XYZ and 1386-D features')
    if not np.isin(scene['scene_mask'], [0, 1]).all() or not scene['scene_mask'].any():
        raise ValueError('Scene mask must contain valid binary tokens')
    if not all(np.isfinite(v).all() for v in scene.values()):
        raise ValueError('Nonfinite scene input')
    motion, last, velocity = motion_features(torch.from_numpy(position)[None],
        torch.from_numpy(size)[None], torch.from_numpy(times)[None])
    batch = dict(motion=motion, last_position=last, last_velocity=velocity,
        object_mask=torch.ones(1, 1, dtype=torch.bool), future_dt=torch.arange(1, 42)[None].float()/30,
        scene_xyz=torch.from_numpy(scene['scene_xyz']).float()[None],
        scene_features=torch.from_numpy(scene['scene_features']).half()[None],
        scene_mask=torch.from_numpy(scene['scene_mask']).bool()[None])
    return batch, times[-1] + np.arange(1, 42, dtype=np.float32)/30


@torch.no_grad()
def predict(geometry_path, scene_path, checkpoint, mode='visual'):
    batch, times = observation_batch(geometry_path, scene_path)
    state = load_file(str(checkpoint), device='cpu')
    model = Predictor(state['motion_mean'], state['motion_std'], mode).eval()
    model.load_state_dict(state, strict=True)
    prediction = model(batch)[0].numpy()
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite prediction')
    return prediction, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['visual', 'constant', 'motion_only', 'geometry_only'], default='visual')
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        raise ValueError('Output must end in .npz')
    receipt = args.output.with_suffix('.json')
    if args.output.exists() or receipt.exists():
        raise FileExistsError('Refusing to overwrite prediction or receipt')
    torch.set_num_threads(2)
    prediction, times = predict(args.geometry, args.scene, args.checkpoint, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, positions_world=prediction, frame_times=times)
    receipt.write_text(json.dumps(dict(shape=list(prediction.shape), mode=args.mode,
        device='cpu', future_labels_loaded=False, dit_loaded=False,
        inputs={name: dict(path=str(path.resolve()), sha256=sha(path)) for name, path in
                [('geometry', args.geometry), ('scene', args.scene), ('checkpoint', args.checkpoint)]},
        limitations='Single sphere/upright puck scope; calibrated metric context; no rotation prediction',
        output_sha256=sha(args.output)), indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
