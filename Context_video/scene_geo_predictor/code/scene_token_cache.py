"""Frozen Utonia on observed, coarse metric points; no simulator scene input."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from probe_vggt import SOURCE, validate_gpu

FEATURE_DIM = 1386
CHECKPOINT = Path('/data/gaoya/ckpt/Pointcept-Utonia/utonia.pth')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def observed_normals(points, valid, camera_origin, max_relative_step=.1):
    """Central differences, oriented toward the camera in the same world frame."""
    if points.ndim != 4 or points.shape[-1] != 3 or valid.shape != points.shape[:-1]:
        raise ValueError('Expected [T,H,W,3] points and aligned masks')
    if valid.dtype != np.bool_ or not np.isfinite(points).all():
        raise ValueError('Nonfinite points or invalid mask dtype')
    normal = np.zeros_like(points, dtype=np.float32)
    keep = np.zeros_like(valid)
    center = points[:, 1:-1, 1:-1]
    neighbors = (points[:, 1:-1, 2:], points[:, 1:-1, :-2],
                 points[:, 2:, 1:-1], points[:, :-2, 1:-1])
    good = (valid[:, 1:-1, 1:-1] & valid[:, 1:-1, 2:] & valid[:, 1:-1, :-2]
            & valid[:, 2:, 1:-1] & valid[:, :-2, 1:-1])
    distance = np.linalg.norm(center - camera_origin, axis=-1)
    for neighbor in neighbors:
        good &= np.linalg.norm(neighbor - center, axis=-1) <= max_relative_step * distance
    cross = np.cross(neighbors[0] - neighbors[1], neighbors[2] - neighbors[3])
    length = np.linalg.norm(cross, axis=-1)
    good &= length > 1e-10
    cross /= np.maximum(length[..., None], 1e-10)
    flip = np.sum(cross * (camera_origin - center), axis=-1) < 0
    cross[flip] *= -1
    normal[:, 1:-1, 1:-1] = np.where(good[..., None], cross, 0)
    keep[:, 1:-1, 1:-1] = good
    return normal, keep


def point_inputs(aligned, raw, max_points=65536):
    points = aligned['world_midpoint']
    valid = aligned['static_valid']
    if points.shape[0] != 8 or valid.shape != points.shape[:-1]:
        raise ValueError('Exactly eight aligned observation point maps are required')
    if not np.allclose(aligned['frame_times'], np.arange(8)/30, atol=1e-6, rtol=0):
        raise ValueError('Only observation RGB0-7 at 30 FPS is supported by this pilot')
    images = raw['images']
    confidence = raw['depth_conf']
    if images.shape != (8, 3, *points.shape[1:3]) or confidence.shape != valid.shape:
        raise ValueError('RGB/depth confidence must match the exact point grid')
    if not np.isfinite(images).all() or not np.isfinite(confidence).all():
        raise ValueError('Nonfinite RGB/confidence')
    if images.min() < 0 or images.max() > 1:
        raise ValueError('Expected RGB in [0,1]')
    rt = aligned['RT']
    origin = -rt[:, :3].T @ rt[:, 3]
    normals, keep = observed_normals(points, valid, origin)
    available = np.flatnonzero(keep.reshape(-1))
    if len(available) < 64 or max_points < 64:
        raise ValueError('Too few observed static points with valid normals')
    count = min(len(available), int(max_points))
    ids = available[np.linspace(0, len(available)-1, count, dtype=np.int64)]
    world = points.reshape(-1, 3)[ids].astype(np.float32)
    rgb = images.transpose(0, 2, 3, 1).reshape(-1, 3)[ids] * 255
    point = dict(coord=world.copy(), color=rgb.astype(np.float32),
                 normal=normals.reshape(-1, 3)[ids].astype(np.float32))
    return point, ids, confidence, dict(
        source_shape=list(valid.shape), observed_static_pixels=int(valid.sum()),
        normal_valid_pixels=int(keep.sum()), encoder_input_points=count,
        input_subsample='uniform valid observed pixels in frame/y/x order',
        max_relative_normal_step=.1, camera_origin_world=origin.tolist())


def representative_indices(inverse, source_ids, confidence, token_count):
    inverse = np.asarray(inverse, dtype=np.int64)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    if inverse.shape != source_ids.shape or inverse.ndim != 1 or inverse.size == 0:
        raise ValueError('Invalid raw-point/root-token correspondence')
    if inverse.min() < 0 or inverse.max() >= token_count:
        raise ValueError('Root inverse is out of range')
    score = confidence.reshape(-1)[source_ids]
    order = np.lexsort((np.arange(inverse.size), -score, inverse))
    first = np.r_[True, np.diff(inverse[order]) != 0]
    chosen = np.full(token_count, -1, dtype=np.int64)
    chosen[inverse[order[first]]] = source_ids[order[first]]
    if np.any(chosen < 0):
        raise ValueError('Missing representative for an encoder root token')
    return chosen


def encode_case(model, aligned_dir, raw_path, output, device, max_points, max_tokens, selection_geometry=None):
    import torch
    import utonia
    from utonia_dense_pointmap import full_upcast

    if output.exists():
        raise FileExistsError(output)
    start = time.monotonic()
    report_path = aligned_dir / 'report.json'
    geometry_path = aligned_dir / 'coarse_static_points.npz'
    report = json.loads(report_path.read_text())
    if report.get('status') != 'coarse_interval_alignment_complete':
        raise ValueError('An existing successful coarse-alignment report is required')
    if sha256(geometry_path) != report['output_npz_sha256']:
        raise ValueError('Observed geometry hash changed')
    raw_hash = sha256(raw_path)
    if raw_hash != report['provenance']['raw']['sha256']:
        raise ValueError('Raw RGB/depth cache does not match this alignment')
    with np.load(geometry_path, allow_pickle=False) as data:
        aligned = {k: data[k] for k in data.files}
    with np.load(raw_path, allow_pickle=False) as data:
        raw = {k: data[k] for k in ('images', 'depth_conf')}
    point, source_ids, confidence, audit = point_inputs(aligned, raw, max_points)
    utonia.utils.set_seed(53124)
    transformed = utonia.transform.default(scale=1., normalize_coord=False)(point)
    transformed = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in transformed.items()}
    with torch.inference_mode():
        root, stages = full_upcast(model(transformed))
        features = root.feat.detach()
        inverse = root.inverse.detach().long().cpu().numpy()
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError('Unexpected Utonia feature dimension')
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite Utonia features')
        native = int(features.shape[0])
        representative = representative_indices(inverse, source_ids, confidence, native)
        selection = dict(method='uniform_root_index')
        keep = np.linspace(0, native-1, min(native, max_tokens), dtype=np.int64)
        if selection_geometry is not None:
            from motion_token_selection import select_tokens
            if sha256(selection_geometry) != report['provenance']['geometry']['sha256']:
                raise ValueError('Motion selection and alignment context disagree')
            with np.load(selection_geometry, allow_pickle=False) as context:
                keep, selection = select_tokens(
                    aligned['world_midpoint'].reshape(-1, 3)[representative],
                    context['positions_world'], context['size_m'], context['frame_times'], max_tokens)
            selection['geometry_sha256'] = sha256(selection_geometry)
            selection['source_sha256'] = sha256(Path(__file__).with_name('motion_token_selection.py'))
        feat = features[torch.as_tensor(keep, device=device)].half().cpu().numpy()
        encoding_xyz = root.coord[torch.as_tensor(keep, device=device)].cpu().numpy()
    representative = representative[keep]
    xyz = aligned['world_midpoint'].reshape(-1, 3)[representative]
    if not aligned['static_valid'].reshape(-1)[representative].all():
        raise ValueError('Dynamic/invalid representative unexpectedly retained')
    indices = np.stack(np.unravel_index(representative, aligned['static_valid'].shape), -1)
    output.mkdir(parents=True)
    np.savez_compressed(output / 'scene_tokens.npz', scene_features=feat,
        scene_xyz=xyz, scene_mask=np.ones(len(keep), dtype=bool),
        scene_confidence=confidence.reshape(-1)[representative],
        encoder_coords=encoding_xyz, source_flat_indices=representative,
        source_frame_y_x=indices, scale_interval=aligned['scale_interval'])
    result = dict(status='observed_utonia_tokens_complete',
        cache_schema='observed_utonia_tokens_v2', cache_version=2,
        case=aligned_dir.name, observed_indices=list(range(8)),
        observed_frame_times_s=(np.arange(8, dtype=np.float64) / 30).tolist(),
        aligned_points=str(geometry_path), aligned_points_sha256=sha256(geometry_path),
        alignment_report_sha256=sha256(report_path), input_pixels_sha256=report['input_pixels_sha256'],
        raw_cache=str(raw_path), raw_sha256=raw_hash, feature_shape=list(feat.shape),
        native_root_tokens=native, stored_tokens=len(keep), encoder_stage_shapes=stages,
        token_selection=selection,
        checkpoint=str(CHECKPOINT), frozen=True, vggt_reloaded=False, dit_loaded=False,
        feature_dim=FEATURE_DIM, feature_coordinate_row_mapping='scene_features[j] and scene_xyz[j] share the same selected Utonia root representative',
        scene_xyz_units='meters', scene_xyz_coordinate_frame='observed simulator world frame via actual camera RT',
        scene_xyz_source='observed RGB8-7 depth midpoint after dynamic-mask exclusion and observed-object AABB scale interval midpoint',
        scene_xyz_is_collision_sdf=False, sampling_version='uniform_root_index_v1',
        static_scene_gt_used=False, future_rgb_used=False, gt_mask_used=False,
        observed_position_size_camera_assistance=True, world_coordinates='coarse observed depth midpoint in meters',
        encoder_coordinates='official translation/grid transform, scale=1; NOT used as world coordinates',
        normals='world-frame finite differences, facing actual camera; discontinuities excluded',
        token_geometry='same-root highest-depth-confidence observed pixel, not GT collider intersection',
        predictive_quality_validated=False, full_training_ready=False,
        intended_use='small feature/predictor pilot; not a precise collision SDF',
        elapsed_seconds=time.monotonic()-start, point_audit=audit,
        output_sha256=sha256(output/'scene_tokens.npz'))
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--max-points', type=int, default=65536)
    p.add_argument('--max-tokens', type=int, default=1792)
    p.add_argument('--token-selection', choices=('uniform', 'motion-corridor'), default='uniform')
    args = p.parse_args()
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, check=True, timeout=30).stdout
    index = validate_gpu(args.gpu_uuid, os.environ.get('CUDA_VISIBLE_DEVICES'), inventory)
    if args.max_points < 64 or args.max_tokens < 1 or not CHECKPOINT.is_file():
        raise ValueError('Invalid point budget or missing local checkpoint')
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    sys.path.insert(0, str(SOURCE))
    import utonia_dense_pointmap  # Registers the existing vendored dependencies.
    import utonia
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one permitted visible GPU')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model = utonia.load(str(CHECKPOINT),
        custom_config={'enc_patch_size': [1024]*5, 'enable_flash': False}).to('cuda:0').eval().requires_grad_(False)
    loaded = time.monotonic()
    args.output.mkdir(parents=True)
    rows = [encode_case(model, args.root/'aligned_scene'/case,
            args.root/'raw_probe'/case/'raw_vggt.npz', args.output/case,
            'cuda:0', args.max_points, args.max_tokens,
            args.root/'observed_context'/case/'context_geometry.npz'
                if args.token_selection == 'motion-corridor' else None) for case in args.cases]
    summary = dict(status='feature_pilot_complete', cases=len(rows), physical_gpu_index=index,
        gpu_uuid=args.gpu_uuid, checkpoint_sha256=sha256(CHECKPOINT),
        source_sha256=sha256(Path(__file__)), full_upcast_source_sha256=sha256(SOURCE/'utonia_dense_pointmap.py'),
        load_seconds=loaded-start, elapsed_seconds=time.monotonic()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, rows=rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
