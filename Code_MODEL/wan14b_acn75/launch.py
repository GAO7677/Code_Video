"""Start only validated Wan14B jobs, in tmux, after the requested GPUs are idle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent



def verified_assets(config):
    root = Path(config['model']['base_dir']).parent
    proof = json.loads((root / 'assets_verified.json').read_text())
    manifest = ROOT / 'configs/assets.json'
    if not proof.get('complete') or proof.get('manifest_sha256') != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise RuntimeError('Missing or stale pinned model asset audit')
    expected_revisions = {config['model']['base_revision'], config['model']['pusa_revision']}
    if {item['revision'] for item in proof['files']} != expected_revisions:
        raise RuntimeError('Asset audit revisions differ from training initialization')
    verified_paths = {Path(item['path']).resolve(): item for item in proof['files']}
    for expert in ('high', 'low'):
        files = list((Path(config['model']['base_dir']) / f'{expert}_noise_model').glob('*.safetensors'))
        if not files:
            raise RuntimeError('Missing expert weights: ' + expert)
        files.append(Path(config['model']['pusa_dir']) / f'{expert}_noise_pusa.safetensors')
        for path in files:
            if path.resolve() not in verified_paths:
                raise RuntimeError('Model will read an unaudited weight file: ' + str(path))
    for item in proof['files']:
        stat = Path(item['path']).stat()
        if stat.st_size != item['size'] or stat.st_mtime_ns != item.get('mtime_ns'):
            raise RuntimeError('Model asset changed after SHA256 verification: ' + item['path'])


def validate_report(config, report_path):
    from train import validation_fingerprint
    report = json.loads(Path(report_path).read_text())
    verified_assets(config)
    required = {
        'passed': True, 'adapter_reload_pass': True, 'optimizer_reload_pass': True,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise RuntimeError(f'Smoke gate failed: {key}')
    if set(report.get('covered_modes', [])) != {'A', 'C', 'N75'}:
        raise RuntimeError('Smoke did not cover all three modes')
    if set(report.get('covered_experts', [])) != {'high', 'low'}:
        raise RuntimeError('Smoke did not cover both experts')
    if report.get('latents_shape') != [1, 16, 13, 64, 112]:
        raise RuntimeError('Smoke did not use the full training resolution')
    for key, value in validation_fingerprint(config).items():
        if report.get(key) != value:
            raise RuntimeError(f'Smoke is stale or has a different configuration: {key}')
    return report


def complete_cache(config, report=None):
    from data import Wan14BCachedDataset
    dataset = Wan14BCachedDataset(config, require_verified=True)
    if len(dataset) != config['data']['expected_samples']:
        raise RuntimeError('Formal training requires the complete verified dataset')
    # Dataset construction checks every entry/provenance. Tensor integrity is
    # checked when each entry is read, rather than loading 12GiB into RAM here.
    if report is not None:
        ranks = report.get('ranks', [])
        if len(ranks) != config['runtime']['world_size'] or any(
                item.get('cache_contract') != dataset.contracts for item in ranks):
            raise RuntimeError('Smoke and formal cache contracts differ')
    return len(dataset)


def idle(gpus):
    command = ['nvidia-smi', '-i', ','.join(map(str, gpus)),
               '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits']
    result = subprocess.run(command, text=True, capture_output=True, timeout=25, check=True)
    rows = list(csv.reader(result.stdout.strip().splitlines()))
    return (len(rows) == len(gpus) and {int(row[0]) for row in rows} == set(gpus)
            and all(int(row[1]) < 512 and int(row[2]) == 0 for row in rows))


def worker(args):
    from train import run_signature, validation_fingerprint
    configs = [json.loads(path.read_text()) for path in args.config]
    for config in configs:
        report = validate_report(config, args.smoke_report)
        complete_cache(config, report)
    for config_path, config in zip(args.config, configs):
        report = validate_report(config, args.smoke_report)
        output = Path(config['paths']['output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        complete = output / 'completed.json'
        def is_complete():
            if not complete.is_file():
                return False
            state = json.loads(complete.read_text())
            return (state.get('smoke') is False
                    and state.get('step') == config['training']['max_steps']
                    and state.get('training_mode') == config['training']['mode']
                    and state.get('run_signature') == run_signature(config)
                    and state.get('validation_fingerprint') == validation_fingerprint(config))
        if is_complete():
            print(f'Already complete: {config["training"]["mode"]}', flush=True)
            continue
        consecutive = 0
        while consecutive < 3:
            try:
                ready = idle(args.gpus)
            except (subprocess.SubprocessError, ValueError):
                ready = False
            consecutive = consecutive + 1 if ready else 0
            print(f'Waiting for GPUs {args.gpus}: idle checks {consecutive}/3', flush=True)
            if consecutive < 3:
                time.sleep(20)
        # Recheck the actual file after the potentially long resource wait.
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Queued configuration changed; revalidate before restarting the queue')
        validate_report(config, args.smoke_report)
        complete_cache(config, report)
        frozen_config = output / 'launch_config.json'
        frozen_config.write_text(json.dumps(config, indent=2) + '\n')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, args.gpus)),
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
        env.update(config['runtime'].get('environment', {}))
        python = config['runtime']['python']
        command = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   str(ROOT / 'train.py'), '--config', str(frozen_config), '--resume', 'auto']
        status = {'stage': 'training', 'mode': config['training']['mode'], 'gpus': args.gpus,
                  'command': shlex.join(command), 'started': time.time(),
                  'smoke_report': str(args.smoke_report)}
        (output / 'launch_status.json').write_text(json.dumps(status, indent=2) + '\n')
        print('Launching ' + shlex.join(command), flush=True)
        with (output / 'train.log').open('a', buffering=1) as log:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status.update(stage='finished' if result.returncode == 0 else 'failed',
                      returncode=result.returncode, ended=time.time())
        (output / 'launch_status.json').write_text(json.dumps(status, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{config["training"]["mode"]} exited {result.returncode}; queue stopped')
        if not is_complete():
            raise RuntimeError('Training exited without a valid completion marker; queue stopped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, nargs='+', required=True,
                        help='One or more configurations, executed sequentially on this pair')
    parser.add_argument('--gpus', required=True, help='Physical GPU pair: 0,1 or 2,3')
    parser.add_argument('--smoke-report', type=Path, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--execute', action='store_true', help='Without this, validate and print only')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.config = [path.resolve() for path in args.config]
    args.smoke_report = args.smoke_report.resolve()
    args.gpus = [int(value) for value in args.gpus.split(',')]
    if args.gpus not in ([0, 1], [2, 3]):
        raise SystemExit('Only the user-authorized GPU pairs 0,1 and 2,3 are accepted')
    if args.worker:
        worker(args)
        return
    if not args.session.replace('-', '').replace('_', '').isalnum():
        raise SystemExit('Session name must contain letters, digits, hyphens or underscores')
    for config_path in args.config:
        config = json.loads(config_path.read_text())
        report = validate_report(config, args.smoke_report)
        complete_cache(config, report)
    command = [sys.executable, str(Path(__file__).resolve()), '--config',
               *map(str, args.config), '--gpus', ','.join(map(str, args.gpus)),
               '--smoke-report', str(args.smoke_report), '--session', args.session, '--worker']
    print(shlex.join(command), flush=True)
    if not args.execute:
        print('Validation passed; --execute is required to create the tmux worker')
        return
    subprocess.run(['tmux', 'new-session', '-d', '-s', args.session,
                    shlex.join(command)], check=True)
    print(f'Started tmux session {args.session}; waiting for idle GPUs if necessary', flush=True)


if __name__ == '__main__':
    main()
