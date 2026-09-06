#!/usr/bin/env python3
"""Drain RigidBench gaps with four independent metric groups per idle GPU.

The metric implementations, strict GT, videos and thresholds are unchanged.
All workers use one freshly initialized registry.  Model shards are disjoint
within each group; the evaluator's per-case lock merges different groups.
Completion is audited from finite metric values, never just an exit code.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time

import evaluate_test70_rigidbench_all_methods as E


GPUS = (2, 3, 5, 6, 7)
PYTHON = '/home/gaoya/miniconda3/envs/sam/bin/python'
PUBLISH_PYTHON = '/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python'
PROJECT = Path('/data/gaoya/agent-data/physv_v2v_0819')
SOURCE_PLAN = PROJECT / 'manifests/test70_saved_checkpoint_backfill_tasks.json'
LOG_ROOT = E.DEFAULT_OUTPUT_ROOT / 'logs/backfill_gpu23567_20260906'
PUBLISHER = Path('/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/evaluation/test70/publishing/publish_test70_resume_to_page.py')


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(message):
    print(f'[{stamp()}] {message}', flush=True)


def remaining(models, metrics):
    count = 0
    for model in models:
        for case in model['cases']:
            payload = E.read_json(E.metric_path(E.DEFAULT_OUTPUT_ROOT, model['task_id'], case['case_id']))
            count += sum(E.metric_missing(payload, metric) for metric in metrics)
    return int(count)


def prepare():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    # Rebuild the index, not metric files.  Preserve every old registered model.
    previous = E.read_json(E.DEFAULT_OUTPUT_ROOT / 'registry.json')
    current = E.read_json(E.DEFAULT_DASHBOARD)
    old_ids = {m['task_id'] for m in previous.get('models', [])}
    current_ids = {m['task_id'] for m in current['models']}
    if not old_ids <= current_ids:
        raise RuntimeError('dashboard would drop previously registered checkpoints')
    registry = E.load_registry(E.DEFAULT_DASHBOARD, E.DEFAULT_STRICT_ROOT, E.DEFAULT_OUTPUT_ROOT, write=True)
    tasks = E.read_json(SOURCE_PLAN)['tasks']
    by_id = {m['task_id']: m for m in registry['models']}
    complete = [t for t in tasks if len(by_id[t['task_id']]['cases']) == 70
                and all(c['prediction_exists'] for c in by_id[t['task_id']]['cases'])]
    if not complete:
        raise RuntimeError('no fully generated backfill checkpoints')
    keys = sorted({t['model_key'] for t in complete})
    # Resume older checkpoints of these same methods too; finite existing
    # values are skipped by the original evaluator, never recomputed.
    selected = [m for m in registry['models'] if m['model_key'] in keys
                and len(m['cases']) == 70 and all(c['prediction_exists'] for c in m['cases'])]
    workers = []
    for index, gpu in enumerate(GPUS):
        models = selected[index::len(GPUS)]
        for group, metrics in E.GROUP_METRICS.items():
            pending = remaining(models, metrics)
            command = [PYTHON, str(Path(E.__file__)), '--group', group,
                       '--shard-index', str(index), '--shard-count', str(len(GPUS)),
                       '--gpu-label', f'{gpu}_backfill0906', '--device', 'cuda',
                       '--resume', '--require-complete-model']
            for key in keys:
                command.extend(['--model-key', key])
            workers.append(dict(id=f'gpu{gpu}_{group}', gpu=gpu, group=group,
                                shard=index, task_ids=[m['task_id'] for m in models],
                                initial_missing=pending, command=command))
    # A complete partition per group prevents competing workers from
    # recomputing the same (checkpoint, case, metric) combination.
    for group in E.GROUP_METRICS:
        ids = [i for w in workers if w['group'] == group for i in w['task_ids']]
        assert len(ids) == len(selected) and len(set(ids)) == len(ids)
    plan = dict(created_at_utc=stamp(), gpus=list(GPUS), workers_per_gpu=4,
                source_plan=str(SOURCE_PLAN), complete_backfill_checkpoints=len(complete),
                selected_model_keys=keys, selected_checkpoint_count=len(selected), workers=workers)
    E.atomic_json(LOG_ROOT / 'plan.json', plan)
    log(f'registered {len(registry["models"])} checkpoints; '
        f'{len(complete)} complete backfill checkpoints; '
        f'{sum(w["initial_missing"] for w in workers)} missing metric cells')
    return plan, by_id


def publish():
    with (LOG_ROOT / 'publish.log').open('a') as out:
        result = subprocess.run([PUBLISH_PYTHON, str(PUBLISHER)],
                                stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        log(f'page publisher returned {result.returncode}; see publish.log')


def run(plan, by_id):
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    devices = {int(a): (int(b), int(c)) for a,b,c in (row.split(',') for row in query.splitlines())}
    busy = {g: devices[g] for g in GPUS if devices[g][0] >= 2048 or devices[g][1] >= 10}
    if busy:
        raise RuntimeError(f'allocated GPUs are no longer idle; not launching: {busy}')
    state = dict(controller_pid=os.getpid(), started_at_utc=stamp(), jobs={})
    active = {}
    try:
        for worker in plan['workers']:
            name = worker['id']
            state['jobs'][name] = dict(worker, status='pending')
            if not worker['initial_missing']:
                state['jobs'][name].update(status='complete', remaining=0)
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker['gpu']),
                       PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       NUMEXPR_NUM_THREADS='2', MALLOC_ARENA_MAX='2')
            path = LOG_ROOT / f'{name}.log'
            out = path.open('a')
            proc = subprocess.Popen(worker['command'], env=env, stdout=out, stderr=subprocess.STDOUT)
            active[name] = (proc, out)
            state['jobs'][name].update(status='running', pid=proc.pid, log=str(path), started_at_utc=stamp())
            log(f'{name}: pid={proc.pid}, missing={worker["initial_missing"]}')
        last_publish = 0.0
        while active:
            for name, (proc, out) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                out.close()
                job = state['jobs'][name]
                missing = remaining([by_id[i] for i in job['task_ids']], E.GROUP_METRICS[job['group']])
                job.update(status='complete' if rc == 0 and missing == 0 else 'failed',
                           returncode=rc, remaining=missing, finished_at_utc=stamp())
                log(f'{name}: rc={rc}, actual missing cells={missing}')
                del active[name]
            state['updated_at_utc'] = stamp()
            E.atomic_json(LOG_ROOT / 'state.json', state)
            if time.monotonic() - last_publish >= 60:
                publish()
                last_publish = time.monotonic()
            if active:
                time.sleep(15)
        publish()
        return int(any(j['status'] == 'failed' for j in state['jobs'].values()))
    finally:
        for proc, out in active.values():
            proc.terminate()
            out.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan, by_id = prepare()
    if args.prepare_only:
        return 0
    return run(plan, by_id)


if __name__ == '__main__':
    raise SystemExit(main())
