"""Reuse frozen models across cases; stop before any unreviewed alignment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from prepare_small_trial import ROOT, dump, sha
from prepare_context import load_model_input, pixel_hash
import probe_vggt as depth_api
import observed_masks as mask_api


def wait_for_context(root, key, seconds):
    wait_for_file(root/'observed_context'/key/'report.json',seconds)


def wait_for_file(path, seconds):
    start = time.monotonic()
    while not path.exists():
        if time.monotonic()-start >= seconds:
            raise TimeoutError(f'Preprocessing dependency not ready: {path}')
        time.sleep(2)


def depth_cases(root, records, uuid, wait_seconds=0):
    import torch
    sys.path.insert(0, str(depth_api.SOURCE))
    from utonia_dense_pointmap import VGGT, run_vggt
    torch.manual_seed(42)
    model = VGGT.from_pretrained(str(depth_api.CHECKPOINT)).to('cuda:0').eval().requires_grad_(False)
    for row in records:
        key = row['key']
        wait_for_context(root, key, wait_seconds)
        output = root/'raw_probe'/key
        if (output/'report.json').exists():
            proof = json.loads((output/'report.json').read_text())
            if sha(output/'raw_vggt.npz') != proof['output_sha256']:
                raise ValueError('Depth cache changed')
            continue
        if output.exists():
            raise FileExistsError(output)
        path = root/'context_inputs'/key/'input.json'
        rgb, _ = load_model_input(path)
        payload = json.loads(path.read_text())
        start = time.monotonic()
        pred = run_vggt(model, [path.parent/name for name in payload['frames']], torch.device('cuda:0'))
        depth_api.validate_predictions(pred)
        if pixel_hash(load_model_input(path)[0]) != pixel_hash(rgb):
            raise ValueError('RGB changed during extraction')
        output.mkdir(parents=True)
        np.savez_compressed(output/'raw_vggt.npz', **{k:pred[k] for k in depth_api.FIELDS})
        depth_api.preview(pred, output/'preview.png')
        dump(output/'report.json', dict(status='raw_rgb8_probe_complete', observed_indices=list(range(8)),
            input_pixels_sha256=pixel_hash(rgb), input_json=str(path), checkpoint=str(depth_api.CHECKPOINT),
            source_sha256=sha(depth_api.SOURCE/'utonia_dense_pointmap.py'), frozen=True,
            elapsed_seconds=time.monotonic()-start, gpu_uuid=uuid, static_scene_gt_used=False,
            gt_mask_used=False, future_rgb_used=False, output_sha256=sha(output/'raw_vggt.npz')))
        print('DEPTH_READY', key, flush=True)


def mask_cases(root, records, uuid, wait_seconds=0):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.manual_seed(42)
    model = build_sam2(mask_api.CONFIG, str(mask_api.CHECKPOINT), device='cuda:0').eval().requires_grad_(False)
    predictor = SAM2ImagePredictor(model)
    for row in records:
        key = row['key']
        wait_for_context(root, key, wait_seconds)
        output = root/'observed_masks'/key
        if (output/'report.json').exists():
            proof = json.loads((output/'report.json').read_text())
            if sha(output/'observed_masks.npz') != proof['output_sha256']:
                raise ValueError('Mask cache changed')
            continue
        if output.exists():
            raise FileExistsError(output)
        path = root/'context_inputs'/key/'input.json'
        geometry_path = root/'observed_context'/key/'context_geometry.npz'
        rgb, times = load_model_input(path)
        geometry = mask_api.load_geometry(geometry_path, times)
        h, w = rgb.shape[1:3]
        k = geometry['positions_world'].shape[1]
        per_object = np.zeros((8,k,h,w), dtype=bool)
        accepted = np.zeros((8,k), dtype=bool)
        scores_out = np.zeros((8,k), dtype=np.float32)
        containment = np.zeros((8,k), dtype=np.float32)
        quality = []
        start = time.monotonic()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
            for t in range(8):
                predictor.set_image(rgb[t])
                for j in range(k):
                    prompt = mask_api.make_prompt(geometry['mask_boxes_xyxy'][t,j], geometry['center_pixels'][t,j], (h,w))
                    if prompt is None:
                        quality.append(dict(frame=t, object_index=j, accepted=False, reason='no_visible_center_prompt'))
                        continue
                    box, points, labels = prompt
                    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, box=box, multimask_output=True)
                    chosen, record = mask_api.select_mask(masks, scores, box, points[0])
                    if chosen.shape != (h,w):
                        raise ValueError('SAM resolution changed')
                    per_object[t,j], accepted[t,j] = chosen, record['accepted']
                    scores_out[t,j], containment[t,j] = record['model_score'], record['box_containment']
                    quality.append(dict(frame=t, object_index=j, **record))
        union = per_object.any(1)
        mask_api.validate_masks(per_object, union, accepted)
        output.mkdir(parents=True)
        np.savez_compressed(output/'observed_masks.npz', per_object=per_object, union_dynamic=union,
            accepted=accepted, model_scores=scores_out, box_containment=containment,
            area_pixels=per_object.sum((2,3)), frame_times=times)
        mask_api.preview(rgb, per_object, geometry, quality, output/'preview.png')
        dump(output/'report.json', dict(schema='observed_sam2_rgb8_v1', status='candidate_masks_complete',
            input_pixels_sha256=pixel_hash(rgb), geometry_sha256=sha(geometry_path), checkpoint=str(mask_api.CHECKPOINT),
            config=mask_api.CONFIG, adapter_sha256=sha(Path(mask_api.__file__)), gpu_uuid=uuid,
            observed_indices=list(range(8)), all_prompt_gates_passed=bool(accepted.all()), quality=quality,
            static_scene_gt_used=False, gt_mask_used=False, future_frames_used=False,
            permitted_context_geometry_used=True, needs_review=True, training_ready=False,
            elapsed_seconds=time.monotonic()-start, output_sha256=sha(output/'observed_masks.npz')))
        print('MASK_CANDIDATE', key, int(accepted.sum()), '/', accepted.size, flush=True)


def feature_cases(root, records, uuid, wait_seconds=0):
    import scene_token_cache as feature_api
    sys.path.insert(0, str(feature_api.SOURCE))
    import utonia_dense_pointmap
    import utonia
    selection = json.loads((root/'manifest.json').read_text()).get('token_selection', 'uniform')
    if selection not in ('uniform', 'motion_corridor'):
        raise ValueError('Unknown token-selection protocol')
    model = utonia.load(str(feature_api.CHECKPOINT),
        custom_config={'enc_patch_size':[1024]*5, 'enable_flash':False}).to('cuda:0').eval().requires_grad_(False)
    for row in records:
        key = row['key']
        output = root/'scene_features'/key
        wait_for_file(root/'aligned_scene'/key/'report.json',wait_seconds)
        if (output/'report.json').exists():
            proof = json.loads((output/'report.json').read_text())
            if sha(output/'scene_tokens.npz') != proof['output_sha256']:
                raise ValueError('Utonia cache changed')
            expected = 'uniform_root_index' if selection == 'uniform' else 'half_global_half_observed_cv_corridor_fps_v1'
            if proof.get('token_selection', {}).get('method', 'uniform_root_index') != expected:
                raise ValueError('Cached token selection does not match manifest')
            if (sha(root/'aligned_scene'/key/'report.json') != proof['alignment_report_sha256'] or
                sha(root/'aligned_scene'/key/'coarse_static_points.npz') != proof['aligned_points_sha256'] or
                sha(root/'raw_probe'/key/'raw_vggt.npz') != proof['raw_sha256']):
                raise ValueError('Utonia input provenance changed')
            continue
        feature_api.encode_case(model, root/'aligned_scene'/key,
            root/'raw_probe'/key/'raw_vggt.npz', output, 'cuda:0', 65536, 1792,
            root/'observed_context'/key/'context_geometry.npz' if selection == 'motion_corridor' else None)
        print('SCENE_READY', key, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['depth', 'masks', 'features'])
    parser.add_argument('--root', type=Path, default=ROOT/'small_trial_120')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--wait-for-context', type=int, default=0)
    parser.add_argument('--wait-for-alignment', type=int, default=0)
    args = parser.parse_args()
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, check=True, timeout=30).stdout
    depth_api.validate_gpu(args.gpu_uuid, os.environ.get('CUDA_VISIBLE_DEVICES'), inventory)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one allowed GPU')
    torch.set_num_threads(2)
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard')
    rows = json.loads((args.root/'manifest.json').read_text())['records']
    if args.preflight:
        rows = [next(r for r in rows if r['family'] == f) for f in ('barrier','door','gap')]
    {'depth':depth_cases, 'masks':mask_cases, 'features':feature_cases}[args.stage](
        args.root, rows[args.shard::args.shards], args.gpu_uuid,
        args.wait_for_alignment if args.stage == 'features' else args.wait_for_context)
