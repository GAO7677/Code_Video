"""Replay frozen grounded masks through the unchanged generic state/physics path.

Estimation consumes anonymous observation caches only. Evaluation runs after all
estimates and rollout decisions are frozen, including failed attempts.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from context_rgb_pybullet_common import dump_json, sha256_file, crop_transform, resize_crop_mask
from generic_context_geometry import sphere_fit, finite_mesh
from lower_plane_gravity import estimate as estimate_gravity
from finish_generic_pilot36 import rollout, evaluate


def verify(root, filename):
    hashes = json.loads((root / filename).read_text())
    for name, expected in hashes.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return hashes


def run(source, masks_root, output):
    cv2.setNumThreads(2)
    source_hashes = verify(source, 'input_freeze.json')
    verify(masks_root, 'prediction_freeze.json')
    rgb_hashes = json.loads((masks_root / 'input_rgb_hashes.json').read_text())
    clips = sorted((source / 'inputs').iterdir())
    if len(clips) != 36:
        raise ValueError('Expected the complete frozen 36-case cohort')
    for clip in clips:
        for frame in range(8):
            path = clip / f'rgb_{frame:02d}.png'
            if rgb_hashes.get(str(path)) != sha256_file(path):
                raise ValueError(f'Mask/RGB identity mismatch: {path}')
        raw_key = f'vggt_cache/{clip.name}.npz'
        if raw_key not in source_hashes:
            raise ValueError(f'Unfrozen reconstruction: {raw_key}')
    output.mkdir(exist_ok=False)
    (output / 'inputs').symlink_to(source / 'inputs', target_is_directory=True)
    protocol = json.loads((source / 'protocol.json').read_text())
    protocol.update(target='Grounding DINO unique RGB7 box + SAM2 bidirectional tracking',
                    mask_source=str(masks_root), reconstruction_source=str(source),
                    model_execution='Reuse verified frozen observation outputs; no model inference in this run',
                    target_prior='Previously context-tuned ball/brown ball phrases; not blind target discovery',
                    gt_boundary='No evaluation mapping/state/blueprint/future reads until rollout_freeze',
                    gravity='Explicit lower-image horizontal-plane prior, observation admission; magnitude 9.81')
    dump_json(output / 'protocol.json', protocol)
    dump_json(output / 'source_audit.json', {
        'input_freeze_sha256': sha256_file(source / 'input_freeze.json'),
        'mask_freeze_sha256': sha256_file(masks_root / 'prediction_freeze.json'),
        'matched_rgb_frames': 288, 'cases': 36})
    scale = protocol['scale_config']['meters_per_reconstruction_unit']
    started = time.perf_counter()
    for clip in clips:
        tick = time.perf_counter()
        dst = output / 'estimates' / clip.name
        dst.mkdir(parents=True)
        (dst / 'vggt_raw.npz').symlink_to(source / 'vggt_cache' / f'{clip.name}.npz')
        shutil.copyfile(masks_root / 'predictions' / clip.name / 'tracking.npz', dst / 'tracking.npz')
        detection = json.loads((masks_root / 'predictions' / clip.name / 'result.json').read_text())
        if detection['status'] != 'EXECUTED' or detection['fallback_used']:
            raise ValueError(f'Invalid grounded mask cache: {clip.name}')
        with np.load(dst / 'tracking.npz') as a:
            masks = a['masks']
        transform = crop_transform(masks.shape[1:])
        processed = np.stack([resize_crop_mask(m, transform) for m in masks])
        with np.load(dst / 'vggt_raw.npz') as a:
            depth, k, e = a['depth'][..., 0], a['intrinsic'], a['extrinsic']
        times = json.loads((clip / 'timestamps.json').read_text())['time_s']
        row = {'scale': scale, 'target_detection': detection['detection'],
               'sources': {'target': 'cached Grounding DINO + SAM2', 'camera_depth': 'cached VGGT',
                           'radius': 'unknown-radius joint surface fit', 'velocity': 'robust linear fit of 8 centers',
                           'geometry': 'observed finite mesh', 'omega': 'zero prior'},
               'fallback': {'used': False, 'gt': False, 'blueprint': False, 'family': False,
                            'default_radius': False, 'default_support': False, 'position_alignment': False}}
        try:
            row['state'] = sphere_fit(depth, k, e, processed, scale, times)
        except ValueError as exc:
            row['state'] = {'status': 'FAIL', 'reason': str(exc)}
        try:
            mesh, _ = finite_mesh(depth, k, e, processed, scale)
            dump_json(dst / 'collision_primitive.json', mesh)
            row['geometry'] = {'status': 'ESTIMATED', 'triangles': len(mesh['faces']), 'confidence': 'UNVALIDATED'}
        except ValueError as exc:
            row['geometry'] = {'status': 'FAIL', 'reason': str(exc)}
        row['status'] = 'ESTIMATED' if all(row[x]['status'] == 'ESTIMATED' for x in ['state', 'geometry']) else 'FAIL'
        row['estimation_seconds'] = time.perf_counter() - tick
        dump_json(dst / 'result.json', row)
        print(clip.name, row['state']['status'], row['state'].get('reasons', row['state'].get('reason')), flush=True)
    frozen = {str(p.relative_to(output)): sha256_file(p) for p in sorted((output / 'estimates').glob('*/*'))}
    frozen['protocol.json'] = sha256_file(output / 'protocol.json')
    dump_json(output / 'estimate_freeze.json', frozen)
    estimate_gravity(output, output / 'lower_plane_gravity_v1')
    rollout(output)
    # First access to named evaluation metadata occurs only after rollout freeze.
    shutil.copyfile(source / 'evaluation_mapping.json', output / 'evaluation_mapping.json')
    evaluate(output)
    dump_json(output / 'runtime.json', {'cpu_threads': 2, 'gpu_used': False,
              'seconds_estimate_gravity_rollout_evaluate': time.perf_counter() - started,
              'model_inference': 'CACHED_NOT_TIMED'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.masks, a.output)
