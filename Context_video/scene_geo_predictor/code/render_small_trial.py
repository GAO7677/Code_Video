"""Foreground, resumable observation-only Cycles render worker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from context_geometry import extract_observed
from prepare_context import load_model_input, prepare_case
from prepare_small_trial import ROOT, dump, sha
from probe_vggt import validate_gpu

BLENDER = '/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender'
RENDERER = '/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/render_physv_cycles.py'


def render(root, record, uuid):
    key = record['key']
    sample = root/'samples'/key
    proof = json.loads((sample/'replay.json').read_text())
    for name, expected in proof['output_sha256'].items():
        if sha(sample/name) != expected:
            raise ValueError(f'Replay artifact changed: {key}/{name}')
    output = root/'render_context8'/key
    receipt = output/'execution.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['returncode'] != 0 or saved['renderer_sha256'] != sha(RENDERER):
            raise ValueError(f'Failed or changed render: {key}')
        for name, expected in saved['output_sha256'].items():
            if sha(output/name) != expected:
                raise ValueError(f'Render artifact changed: {key}/{name}')
    else:
        if output.exists():
            raise FileExistsError(f'Inspect incomplete render before retry: {output}')
        if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ['CUDA_VISIBLE_DEVICES'] != uuid:
            raise ValueError('Render GPU binding changed')
        output.mkdir(parents=True)
        argv = [BLENDER, '-b', '-t', '4', '--python', RENDERER, '--',
                '--sample-dir', str(sample), '--trajectory-json', str(sample/'raw/observed_trajectories.json'),
                '--output-dir', str(output/'cycles_frames'), '--width', '1280', '--height', '720',
                '--samples', '64', '--frame-limit', '8', '--engine', 'CYCLES', '--device', 'CUDA']
        dump(output/'command.json', dict(argv=argv, gpu_uuid=uuid, timeout_seconds=300,
                                       trajectory_sha256=sha(sample/'raw/observed_trajectories.json')))
        started = time.monotonic()
        with (output/'render.log').open('w') as stream:
            proc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=300)
        files = [f'cycles_frames/frame_{i:04d}.png' for i in range(1, 9)]+['cycles_frames/render_metadata.json']
        dump(receipt, dict(returncode=proc.returncode, elapsed_seconds=time.monotonic()-started,
                          renderer_sha256=sha(RENDERER), gpu_uuid=uuid,
                          output_sha256={name:sha(output/name) for name in files if (output/name).exists()}))
        if proc.returncode != 0 or not all((output/name).is_file() for name in files):
            raise RuntimeError(f'Render failed: {key}, see {output}/render.log')
    inputs = root/'context_inputs'/key/'input.json'
    if not inputs.exists():
        prepare_case(dict(case_id=key, family=record['family'], frames_dir=str(output/'cycles_frames')), root/'context_inputs')
    rgb, times = load_model_input(inputs)
    geometry_dir = root/'observed_context'/key
    if not (geometry_dir/'report.json').exists():
        metadata_path = output/'cycles_frames/render_metadata.json'
        geometry, audit = extract_observed(sample, json.loads(metadata_path.read_text()), times)
        if list(rgb.shape[1:3]) != audit['source_hw']:
            raise ValueError('RGB/camera resolution mismatch')
        geometry_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(geometry_dir/'context_geometry.npz', **geometry)
        dump(geometry_dir/'report.json', dict(audit, sample=str(sample), input_json=str(inputs),
            render_metadata=str(metadata_path), split=record['split'], input_pixels_sha256=json.loads(inputs.read_text())['pixels_sha256']))
    print('RGB8_READY', key, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT/'small_trial_120')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        check=True, text=True, capture_output=True, timeout=30).stdout
    validate_gpu(args.gpu_uuid, os.environ.get('CUDA_VISIBLE_DEVICES'), inventory)
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard')
    records = json.loads((args.root/'manifest.json').read_text())['records']
    if args.preflight:
        records = [next(r for r in records if r['family'] == f) for f in ('barrier', 'door', 'gap')]
    for record in records[args.shard::args.shards]:
        render(args.root, record, args.gpu_uuid)
