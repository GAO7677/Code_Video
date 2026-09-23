"""Frozen grounded test70 observations -> generic geometry/state -> zero-omega Bullet.

No named evaluation mapping is read until all predictions have been frozen.
Unsupported ordinals follow the previously declared context-only visual screening.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from context_rgb_pybullet_common import dump_json, sha256_file, crop_transform, resize_crop_mask
from generic_context_geometry import sphere_fit, finite_mesh_observed
from run_grounded_generic_pilot36 import verify


def support_reason(i):
    if 5 <= i < 15 or 20 <= i < 30:
        return 'UNSUPPORTED_non_sphere_target'
    if 35 <= i < 40:
        return 'UNSUPPORTED_multiple_dynamic_objects'
    if 55 <= i < 70:
        return 'UNSUPPORTED_joint_or_moving_support'
    return None


def prepare(source, root):
    verify(source, 'prediction_freeze.json')
    hashes = json.loads((source / 'input_rgb_hashes.json').read_text())
    clips = sorted((source / 'inputs').iterdir())
    assert len(clips) == 70
    for clip in clips:
        for t in range(8):
            p = clip / f'rgb_{t:02d}.png'
            # Source hashes use the original resolved input tree.
            candidates = [v for k, v in hashes.items() if Path(k).parent.name == clip.name and Path(k).name == p.name]
            if candidates != [sha256_file(p)]:
                raise ValueError(f'RGB identity mismatch: {p}')
    root.mkdir(exist_ok=False)
    (root / 'inputs').symlink_to((source / 'inputs').resolve(), target_is_directory=True)
    protocol = json.loads(Path('/data/gaoya/agent-data/outputs/context_generic_six_20260922_v1/protocol.json').read_text())
    for key in ['selection', 'selected_ordinals', 'screened_only', 'reuse_frozen_anonymous_vggt']:
        protocol.pop(key, None)
    protocol.update(cases=70, source=str(source), date='2026-09-23',
        support_protocol='Prior context-screening ordinals expanded to all five variants; all attempts retained; not automatic scene understanding',
        target='Frozen group-conditioned Grounding DINO RGB7 unique box + SAM2 bidirectional; context-tuned appearance prior',
        geometry_mode='local_plane_completion',
        gravity='Authorized lower-image horizontal-plane prior, admitted only with observation consensus',
        unsupported_geometry='Diagnostic only: target-only mask does not remove all other moving objects',
        gt_boundary='GT, named mapping and future only read by post-freeze evaluation',
        support_by_id={c.name: support_reason(i) for i, c in enumerate(clips)},
        model_weights='/data/gaoya/ckpt/facebook-VGGT-1B', gpu_physical_index=6)
    dump_json(root / 'protocol.json', protocol)
    for clip in clips:
        dst = root / 'estimates' / clip.name
        dst.mkdir(parents=True)
        for name in ['tracking.npz', 'result.json']:
            shutil.copyfile(source / 'predictions' / clip.name / name, dst / ('mask_result.json' if name == 'result.json' else name))
        detection = json.loads((dst / 'mask_result.json').read_text())
        if detection['status'] != 'EXECUTED' or detection['fallback_used']:
            raise ValueError(f'Invalid frozen mask: {clip.name}')
    freeze = {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / 'inputs').glob('*/*'))}
    freeze.update({str(p.relative_to(root)): sha256_file(p) for p in sorted((root / 'estimates').glob('*/*'))})
    freeze['protocol.json'] = sha256_file(root / 'protocol.json')
    dump_json(root / 'input_freeze.json', freeze)


def depth(root):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-7f6fbc40-3594-2c34-8557-422621355ff9':
        raise ValueError('Expected authorized physical GPU6 UUID')
    torch.set_num_threads(2)
    verify(root, 'input_freeze.json')
    sys.path.insert(0, '/home/gaoya/vggt_official')
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    model = VGGT.from_pretrained('/data/gaoya/ckpt/facebook-VGGT-1B').to('cuda:0').eval().requires_grad_(False)
    times = {}
    for clip in sorted((root / 'inputs').iterdir()):
        dst = root / 'estimates' / clip.name / 'vggt_raw.npz'
        if dst.exists():
            raise FileExistsError(f'Refusing to overwrite reconstruction: {dst}')
        start = time.perf_counter()
        imgs = load_and_preprocess_images([str(clip / f'rgb_{t:02d}.png') for t in range(8)]).to('cuda:0')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(imgs)
        e, k = pose_encoding_to_extri_intri(pred['pose_enc'], imgs.shape[-2:])
        arrays = {n: pred[n].detach().float().cpu().numpy().squeeze(0) for n in ['depth', 'depth_conf', 'world_points', 'world_points_conf']}
        arrays.update(extrinsic=e.detach().float().cpu().numpy().squeeze(0), intrinsic=k.detach().float().cpu().numpy().squeeze(0))
        np.savez_compressed(dst, **arrays)
        times[clip.name] = time.perf_counter() - start
        print('VGGT', clip.name, round(times[clip.name], 2), flush=True)
        del imgs, pred, e, k, arrays
        torch.cuda.empty_cache()
    dump_json(root / 'depth_runtime.json', times)
    dump_json(root / 'depth_freeze.json', {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / 'estimates').glob('*/vggt_raw.npz'))})


def estimate(root):
    from lower_plane_gravity import estimate as gravity
    from generic_bullet_input import run
    cv2.setNumThreads(2)
    verify(root, 'input_freeze.json')
    verify(root, 'depth_freeze.json')
    protocol = json.loads((root / 'protocol.json').read_text())
    scale = protocol['scale_config']['meters_per_reconstruction_unit']
    for folder in sorted((root / 'estimates').iterdir()):
        tick = time.perf_counter()
        with np.load(folder / 'tracking.npz') as a:
            masks = a['masks']
        with np.load(folder / 'vggt_raw.npz') as a:
            d, k, e = a['depth'][..., 0], a['intrinsic'], a['extrinsic']
        tr = crop_transform(masks.shape[1:])
        masks = np.stack([resize_crop_mask(m, tr) for m in masks])
        times = json.loads((root / 'inputs' / folder.name / 'timestamps.json').read_text())['time_s']
        reason = protocol['support_by_id'][folder.name]
        row = {'scale': scale, 'support_reason': reason,
               'target_detection': json.loads((folder / 'mask_result.json').read_text())['detection'],
               'fallback': {'used': False, 'gt': False, 'family_geometry': False, 'default_radius': False, 'alignment': False},
               'sources': {'state': 'mask + VGGT visible sphere surface fit', 'geometry': 'VGGT static fusion + labelled local plane prior',
                           'omega': 'zero prior', 'scale': 'fixed pilot median baseline', 'support_scope': 'predeclared context-screening ordinal manifest'}}
        if reason:
            row['state'] = {'status': 'UNSUPPORTED', 'reason': reason}
        else:
            try:
                row['state'] = sphere_fit(d, k, e, masks, scale, times)
            except ValueError as exc:
                row['state'] = {'status': 'FAIL', 'reason': str(exc)}
        try:
            mesh, _ = finite_mesh_observed(d, k, e, masks, scale, complete_local_planes=True)
            dump_json(folder / 'collision_primitive.json', mesh)
            row['geometry'] = {'status': 'ESTIMATED', 'triangles': len(mesh['faces']), 'confidence': 'UNVALIDATED',
                               'diagnostic_only': bool(reason)}
        except ValueError as exc:
            row['geometry'] = {'status': 'FAIL', 'reason': str(exc)}
        row['status'] = row['state']['status']
        row['seconds'] = time.perf_counter() - tick
        dump_json(folder / 'result.json', row)
        print('ESTIMATE', folder.name, row['status'], row['state'].get('reasons', row['state'].get('reason')), flush=True)
    dump_json(root / 'estimate_freeze.json', {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / 'estimates').glob('*/*'))})
    gravity(root, root / 'lower_plane_gravity_v1')
    for entry in json.loads((root / 'lower_plane_gravity_v1/candidates.json').read_text()):
        folder = root / 'estimates' / entry['id']
        row = json.loads((folder / 'result.json').read_text())
        if row['state']['status'] != 'ESTIMATED':
            out = {'status': row['state']['status'], 'reason': row['state'].get('reason', row['state'].get('reasons')), 'step_calls': 0}
        elif row['geometry']['status'] != 'ESTIMATED':
            out = {'status': 'FAIL', 'reason': 'geometry_failed', 'step_calls': 0}
        else:
            out = run(row['state'], [json.loads((folder / 'collision_primitive.json').read_text())], protocol['physics'], entry['gravity'], log_api_contacts=True)
        dump_json(root / 'rollouts' / (entry['id'] + '.json'), out)
        print('ROLLOUT', entry['id'], out['status'], out.get('reason'), flush=True)
    dump_json(root / 'rollout_freeze.json', {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / 'rollouts').glob('*.json'))})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['prepare', 'depth', 'estimate'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('/data/gaoya/agent-data/outputs/test70_grounding_sam2_20260923_v2'))
    a = p.parse_args()
    prepare(a.source, a.root) if a.phase == 'prepare' else globals()[a.phase](a.root)
