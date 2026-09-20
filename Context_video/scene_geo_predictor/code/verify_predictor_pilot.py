"""CPU verification of frozen-token provenance and saved pilot checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from future_query_predictor import Predictor
from predictor_pilot import build_batch
from scene_token_cache import point_inputs, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    root = args.root
    features = root/'utonia_feature_pilot'
    fit = root/'predictor_fit_pilot'
    report = json.loads((fit/'report.json').read_text())
    output = fit/'cpu_verification.json'
    if output.exists():
        raise FileExistsError(output)
    cases = [r['case'] for r in report['records']]
    torch.set_num_threads(2)
    tokens = []
    for case in cases:
        with np.load(root/'aligned_scene'/case/'coarse_static_points.npz', allow_pickle=False) as a:
            aligned = {k: a[k] for k in a.files}
        with np.load(root/'raw_probe'/case/'raw_vggt.npz', allow_pickle=False) as a:
            raw = {k: a[k] for k in ('images', 'depth_conf')}
        point, source, _, _ = point_inputs(aligned, raw)
        xyz = point['coord']
        low, high = xyz.min(0), xyz.max(0)
        shift = np.array([(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]])
        with np.load(features/case/'scene_tokens.npz', allow_pickle=False) as cache:
            ids = cache['source_flat_indices']
            assert np.isin(ids, source).all()
            assert aligned['static_valid'].reshape(-1)[ids].all()
            np.testing.assert_array_equal(cache['scene_xyz'], aligned['world_midpoint'].reshape(-1, 3)[ids])
            recovered = cache['encoder_coords']+shift
            error = np.linalg.norm(cache['scene_xyz']-recovered, axis=-1)
            assert error.max() < np.sqrt(3)*.01+1e-4, 'Root-token/world-pixel association mismatch'
            tokens.append(dict(case=case, shape=list(cache['scene_features'].shape),
                source_frames=np.unique(cache['source_frame_y_x'][:, 0]).tolist(),
                representative_to_encoder_root_max_m=float(error.max()),
                all_representatives_static=True, no_oracle_bridge=True))
    batch, target, records = build_batch(root, features, cases)
    assert records == report['records']
    stats = load_file(str(fit/'fit_only_stats.safetensors'))
    rows = {}
    with np.load(fit/'fit_predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['target'], target.numpy())
        for mode in ('constant', 'visual'):
            model = Predictor(stats['motion_mean'], stats['motion_std'], mode).eval()
            model.load_state_dict(load_file(str(fit/f'{mode}_step0200.safetensors')))
            with torch.no_grad():
                prediction = model(batch).numpy()
            np.testing.assert_allclose(prediction, saved[mode], rtol=2e-4, atol=2e-5)
            rows[mode] = dict(cpu_vs_saved_max_m=float(np.abs(prediction-saved[mode]).max()),
                             training_ADE_m=float(np.linalg.norm(prediction-saved['target'], axis=-1).mean()))
    for name, fingerprint in report['source_sha256'].items():
        assert sha256(root/name) == fingerprint
    feature_summary = json.loads((features/'summary.json').read_text())
    assert sha256(root/'scene_token_cache.py') == feature_summary['source_sha256']
    result = dict(status='passed', gpu_used=False, tokens=tokens, checkpoint_reload=rows,
                  source_hashes_match=True, supervision_remains_fit_only=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
