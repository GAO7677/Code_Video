"""CPU audit of Utonia feature/coordinate row correspondence.

The check is deliberately read-only. It validates the stored representative
pixel mapping and reports legacy caches without silently upgrading them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_case(root: Path, key: str) -> dict:
    aligned_dir = root / 'aligned_scene' / key
    scene_dir = root / 'scene_features' / key
    scene_path = scene_dir / 'scene_tokens.npz'
    scene_report_path = scene_dir / 'report.json'
    aligned_path = aligned_dir / 'coarse_static_points.npz'
    aligned_report_path = aligned_dir / 'report.json'
    scene_report = json.loads(scene_report_path.read_text())
    aligned_report = json.loads(aligned_report_path.read_text())
    with np.load(scene_path, allow_pickle=False) as data:
        required = {'scene_features', 'scene_xyz', 'scene_mask', 'source_flat_indices',
                    'source_frame_y_x'}
        if not required.issubset(data.files):
            raise ValueError(f'{key}: cache is missing correspondence fields')
        features = np.asarray(data['scene_features'])
        xyz = np.asarray(data['scene_xyz'])
        mask = np.asarray(data['scene_mask'])
        source_flat = np.asarray(data['source_flat_indices'])
        source_tyx = np.asarray(data['source_frame_y_x'])
    with np.load(aligned_path, allow_pickle=False) as data:
        points = np.asarray(data['world_midpoint'])
        valid = np.asarray(data['static_valid'])
        times = np.asarray(data['frame_times'])
    if features.ndim != 2 or features.shape[1] != 1386:
        raise ValueError(f'{key}: expected [N,1386] features, got {features.shape}')
    n = features.shape[0]
    if xyz.shape != (n, 3) or mask.shape != (n,) or mask.dtype != np.bool_:
        raise ValueError(f'{key}: feature/coordinate/mask shapes disagree')
    if source_flat.shape != (n,) or source_tyx.shape != (n, 3):
        raise ValueError(f'{key}: missing one-to-one source mapping')
    if not np.isfinite(features).all() or not np.isfinite(xyz).all():
        raise ValueError(f'{key}: nonfinite feature or coordinate')
    if points.ndim != 4 or points.shape[0] != 8 or points.shape[-1] != 3:
        raise ValueError(f'{key}: aligned point map is not [8,H,W,3]')
    flat = points.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    active = np.flatnonzero(mask)
    if len(active) and (source_flat[active].min() < 0 or source_flat[active].max() >= len(flat)):
        raise ValueError(f'{key}: source index out of range')
    if len(active):
        if not flat_valid[source_flat[active]].all():
            raise ValueError(f'{key}: token representative is not a valid static point')
        np.testing.assert_array_equal(xyz[active], flat[source_flat[active]])
        np.testing.assert_array_equal(source_tyx[active], np.stack(np.unravel_index(
            source_flat[active], valid.shape), axis=-1))
    if aligned_report.get('observed_indices') != list(range(8)):
        raise ValueError(f'{key}: aligned cache does not declare RGB0-7')
    for name in ('future_frames_used', 'gt_masks_used', 'static_scene_gt_used'):
        if aligned_report.get(name) is not False:
            raise ValueError(f'{key}: unsafe aligned provenance flag {name}')
    for name in ('future_rgb_used', 'gt_mask_used', 'static_scene_gt_used'):
        if scene_report.get(name) is not False:
            raise ValueError(f'{key}: unsafe scene provenance flag {name}')
    if scene_report.get('feature_shape') != [n, 1386]:
        raise ValueError(f'{key}: report feature shape mismatch')
    if scene_report.get('output_sha256') != sha256(scene_path):
        raise ValueError(f'{key}: scene cache hash mismatch')
    if scene_report.get('aligned_points_sha256') != sha256(aligned_path):
        raise ValueError(f'{key}: aligned cache hash mismatch')
    status = 'PASS_VERSIONED' if scene_report.get('cache_schema') == 'observed_utonia_tokens_v2' else 'LEGACY_UNVERIFIED'
    return dict(key=key, status=status, feature_shape=list(features.shape),
                valid_tokens=int(mask.sum()), source_index_check=True,
                feature_coordinate_rows_same_source=True,
                observed_times=times.tolist(),
                scene_report_schema=scene_report.get('cache_schema'),
                world_coordinates=scene_report.get('world_coordinates'),
                encoder_coordinates=scene_report.get('encoder_coordinates'),
                static_scene_gt_used=False, future_rgb_used=False, gt_mask_used=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [audit_case(args.root, key) for key in args.cases]
    result = dict(schema='predictor_scene_cache_audit_v1', root=str(args.root.resolve()),
                  records=rows, legacy_present=any(r['status'] == 'LEGACY_UNVERIFIED' for r in rows),
                  scope='CPU read-only row correspondence/provenance audit; no cache rewrite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
