"""Geometry-only ablation against frozen recovery v2; no GT before freezes."""
import argparse
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from context_rgb_pybullet_common import dump_json, sha256_file, crop_transform, resize_crop_mask
from run_grounded_generic_pilot36 import verify
from generic_context_geometry import finite_mesh_observed
from generic_bullet_input import run


def main(source, out, surface_first=False):
    cv2.setNumThreads(2)
    for name in ['input_freeze.json', 'depth_freeze.json', 'estimate_freeze.json', 'rollout_freeze.json']:
        verify(source, name)
    out.mkdir(exist_ok=False)
    (out/'inputs').symlink_to(source/'inputs', target_is_directory=True)
    protocol = json.loads((source/'protocol.json').read_text())
    protocol.update(parent_baseline=str(source), recovery_method='geometry_only_segmented_plane_v1',
        frozen_changes={'state': 'exact frozen recovery v2, no refitting',
                        'camera_scale_gravity_physics': 'unchanged',
                        'geometry': 'per-frame visible fusion + segmented surrounding-plane completion',
                        'selection': 'all 30 predeclared sphere candidates, including state failures; other 40 retained unsupported',
                        'evaluation': 'freeze first; paired common rollout means plus all failures; no GT tuning'})
    protocol['code_hashes'] = {n: sha256_file(Path(__file__).parent/n) for n in
        ['run_segmented_mesh_ablation.py', 'segmented_plane_completion.py', 'generic_context_geometry.py','finite_surface_completion.py']}
    if surface_first:
        protocol['recovery_method']='geometry_only_finite_surface_first_v1'
        protocol['frozen_changes']['geometry']='connected normal-consistent visible surfaces; bounded per-surface occlusion continuation; ambiguous ownership UNKNOWN'
    dump_json(out/'protocol.json', protocol)
    for folder in sorted((source/'estimates').iterdir()):
        dst = out/'estimates'/folder.name
        dst.mkdir(parents=True)
        for name in ['tracking.npz', 'mask_result.json', 'vggt_raw.npz']:
            (dst/name).symlink_to(folder/name)
        row = json.loads((folder/'result.json').read_text())
        if not row['support_reason']:
            with np.load(folder/'vggt_raw.npz') as a:
                depth, k, e = a['depth'][...,0], a['intrinsic'], a['extrinsic']
            with np.load(folder/'tracking.npz') as a:
                masks = a['masks']
            transform = crop_transform(masks.shape[1:])
            masks = np.stack([resize_crop_mask(m, transform) for m in masks])
            mesh, _ = finite_mesh_observed(depth, k, e, masks, row['scale'],
                complete_local_planes=True, segmented_completion=not surface_first,surface_completion=surface_first)
            dump_json(dst/'collision_primitive.json', mesh)
            row['geometry'].update(triangles=len(mesh['faces']), confidence='PARTLY_INFERRED_UNVALIDATED')
            print(folder.name, 'inferred_pixels', mesh['completion_audit']['inferred_pixels'], flush=True)
        else:
            (dst/'collision_primitive.json').symlink_to(folder/'collision_primitive.json')
        dump_json(dst/'result.json', row)
    hashes = {str(p.relative_to(out)): sha256_file(p) for p in sorted((out/'inputs').glob('*/*'))}
    hashes['protocol.json'] = sha256_file(out/'protocol.json')
    dump_json(out/'input_freeze.json', hashes)
    dump_json(out/'depth_freeze.json', {str(p.relative_to(out)): sha256_file(p) for p in sorted((out/'estimates').glob('*/vggt_raw.npz'))})
    dump_json(out/'estimate_freeze.json', {str(p.relative_to(out)): sha256_file(p) for p in sorted((out/'estimates').glob('*/*'))})
    shutil.copytree(source/'lower_plane_gravity_v1', out/'lower_plane_gravity_v1')
    verify(out/'lower_plane_gravity_v1', 'freeze.json')
    for gravity in json.loads((out/'lower_plane_gravity_v1/candidates.json').read_text()):
        folder = out/'estimates'/gravity['id']
        row = json.loads((folder/'result.json').read_text())
        if row['state']['status'] != 'ESTIMATED':
            result = {'status': row['state']['status'], 'step_calls': 0,
                      'reason': row['state'].get('reason', row['state'].get('reasons'))}
        else:
            mesh = json.loads((folder/'collision_primitive.json').read_text())
            result = run(row['state'], [mesh], protocol['physics'], gravity['gravity'], log_api_contacts=True)
        dump_json(out/'rollouts'/(gravity['id']+'.json'), result)
        print(gravity['id'], result['status'], flush=True)
    dump_json(out/'rollout_freeze.json', {str(p.relative_to(out)): sha256_file(p) for p in sorted((out/'rollouts').glob('*.json'))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--surface-first',action='store_true')
    args = parser.parse_args()
    main(args.source, args.output,args.surface_first)
