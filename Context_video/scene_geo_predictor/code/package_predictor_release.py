"""Package the validation-selected predictor and an observation-only example."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from future_query_predictor import BASE
from predict_observation import predict
from prepare_small_trial import ROOT, dump, sha
from summarize_two_rounds import ARMS, OUT


def main():
    comparison = json.loads((OUT/'reports/comparison.json').read_text())
    selected = comparison['selected']
    train, control = ARMS[selected]
    target = OUT/'release'
    if target.exists():
        raise FileExistsError(target)
    target.mkdir()
    weights = target/'predictor.safetensors'
    shutil.copy2(train/'runs/visual/best.safetensors', weights)
    runtime = target/'runtime'/ROOT.name
    runtime.mkdir(parents=True)
    for name in ('future_query_predictor.py', 'predict_observation.py', 'prepare_small_trial.py'):
        shutil.copy2(ROOT/name, runtime/name)
    common = runtime.parent/'p4_v2_sg_o_revised_20260916/round1/scripts/model.py'
    common.parent.mkdir(parents=True)
    shutil.copy2(BASE, common)
    torch.set_num_threads(2)
    maximum = 0.
    with np.load(train/'runs/visual/validation_predictions.npz') as data:
        keys, expected = data['keys'].tolist(), data['prediction']
    for key, saved in zip(keys, expected):
        actual, _ = predict(train/'observed_context'/key/'context_geometry.npz',
            train/'scene_features'/key/'scene_tokens.npz', weights)
        maximum = max(maximum, float(np.max(np.abs(actual-saved))))
        np.testing.assert_allclose(actual, saved, atol=2e-6, rtol=1e-5)
    key = keys[0]
    example = target/'example'
    example.mkdir()
    shutil.copy2(train/'observed_context'/key/'context_geometry.npz', example/'context_geometry.npz')
    shutil.copy2(train/'scene_features'/key/'scene_tokens.npz', example/'scene_tokens.npz')
    command = [sys.executable, '-B', str(runtime/'predict_observation.py'),
        '--geometry', str(example/'context_geometry.npz'), '--scene', str(example/'scene_tokens.npz'),
        '--checkpoint', str(weights), '--output', str(example/'prediction.npz')]
    subprocess.run(command, check=True, timeout=90, cwd=target)
    with np.load(example/'prediction.npz') as data:
        np.testing.assert_allclose(data['positions_world'], expected[0], atol=2e-6, rtol=1e-5)
    feature_report = json.loads((train/'scene_features'/key/'report.json').read_text())
    alignment_report = json.loads((train/'aligned_scene'/key/'report.json').read_text())
    manifest = dict(selected=selected, best_step=comparison['results'][selected]['best_step'],
        checkpoint_sha256=sha(weights), source_training_root=str(train), source_control_root=str(control),
        scene_token_selection=feature_report.get('token_selection', dict(method='uniform_root_index'))['method'],
        alignment_schema=alignment_report.get('schema'), validation_cases_verified=len(keys),
        inference_max_abs_difference_m=maximum, no_label_example_verified=True,
        runtime_source_sha256={str(p.relative_to(target)):sha(p) for p in [*runtime.glob('*.py'), common]},
        required_runtime_packages={name:__import__(name).__version__ for name in ('torch', 'numpy', 'safetensors')},
        independent_test=False, generalization_solved=False, dit_loaded=False,
        supported_scope='Single sphere/upright puck, 8 observed frames at 30 FPS, 41 future positions; calibrated world meters',
        deployment_type='Runnable research predictor, not an end-to-end RGB-only production system')
    candidate_train = ARMS['Round 2 local'][0]
    candidate = target/'round2_candidate'
    candidate.mkdir()
    candidate_weights = candidate/'predictor.safetensors'
    shutil.copy2(candidate_train/'runs/visual/best.safetensors', candidate_weights)
    with np.load(candidate_train/'runs/visual/validation_predictions.npz') as data:
        candidate_keys, candidate_expected = data['keys'].tolist(), data['prediction']
    candidate_maximum = 0.
    for candidate_key, saved in zip(candidate_keys, candidate_expected):
        actual, _ = predict(candidate_train/'observed_context'/candidate_key/'context_geometry.npz',
            candidate_train/'scene_features'/candidate_key/'scene_tokens.npz', candidate_weights)
        candidate_maximum = max(candidate_maximum, float(np.max(np.abs(actual-saved))))
        np.testing.assert_allclose(actual, saved, atol=2e-6, rtol=1e-5)
    candidate_key = candidate_keys[0]
    shutil.copy2(candidate_train/'observed_context'/candidate_key/'context_geometry.npz', candidate/'context_geometry.npz')
    shutil.copy2(candidate_train/'scene_features'/candidate_key/'scene_tokens.npz', candidate/'scene_tokens.npz')
    subprocess.run([sys.executable, '-B', str(runtime/'predict_observation.py'),
        '--geometry', str(candidate/'context_geometry.npz'), '--scene', str(candidate/'scene_tokens.npz'),
        '--checkpoint', str(candidate_weights), '--output', str(candidate/'prediction.npz')],
        check=True, timeout=90, cwd=target)
    manifest['round2_candidate'] = dict(checkpoint_sha256=sha(candidate_weights),
        source_training_root=str(candidate_train), best_step=comparison['results']['Round 2 local']['best_step'],
        validation_cases_verified=len(candidate_keys), inference_max_abs_difference_m=candidate_maximum,
        alignment_schema='observed_surface_scale_v1',
        scene_token_selection='half_global_half_observed_cv_corridor_fps_v1',
        note='Separate matched preprocessing and weights; not selected as an across-metric winner')
    dump(target/'release.json', manifest)
    dump(OUT/'reports/runtime_verification.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
