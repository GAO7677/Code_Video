#!/usr/bin/env python3
"""Register a completed Generic2175 context-noise run for the lineage Test70 page."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


BASE = Path('/data/gaoya/agent-data/physv_v2v_0819')
TASKS_PATH = BASE / 'manifests/test70_lineage_tasks.json'
SETTINGS_PATH = BASE / 'manifests/test70_lineage_watcher.json'
INPUT_LIST = BASE / 'manifests/test70_initial_state_v1/input_list.txt'
MODEL_ID = Path('/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers')
DIT = Path('/data/gaoya/agent-data/weights/physrvg-diffusers-d8caf2/dit/diffusion_pytorch_model.safetensors')
RUN_ROOT = Path('/data/gaoya/agent-data/checkpoints/generic2175_initial0907_context_noise_20260923')
STEPS = (500, 1000, 1500, 2000, 2500)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.context-noise.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda-value', type=float, choices=(0.5, 0.75), required=True)
    parser.add_argument('--apply', action='store_true')
    return parser.parse_args()


def run_name(value: float) -> str:
    return f'lambda{int(round(value * 100)):03d}-initial0907-seed42-' + (
        'gpu23-interrupt' if value == 0.5 else 'gpu01-interrupt'
    )


def model_key(value: float) -> str:
    tag = f'{int(round(value * 100)):03d}'
    return f'full_sa_physrvg_raw_rl_lora_context_noise_lambda{tag}_initial0907_generic_full_2175_clean_context_20260923'


def validate(value: float) -> Path:
    run = RUN_ROOT / run_name(value)
    config = load(run / 'resolved_config.json')
    args = config['arguments']
    if args.get('context_noise_scale') != value:
        raise RuntimeError(f'wrong context_noise_scale in {run}')
    if args.get('caption_field') != 'caption_initial0907' or args.get('seed') != 42:
        raise RuntimeError(f'training configuration mismatch in {run}')
    if len([line for line in INPUT_LIST.read_text().splitlines() if line.strip()]) != 70:
        raise RuntimeError('Test70 input list must contain 70 cases')
    for step in STEPS:
        checkpoint = run / 'checkpoints' / f'step-{step:06d}'
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
    return run


def rows(value: float, run: Path) -> list[dict]:
    key = model_key(value)
    label = f'Generic2175 initial0907 context-noise lambda={value:g} Clean-context eval'
    return [
        {
            'model_key': key,
            'label': label,
            'dataset_group': 'Generic2175 initial0907',
            'color': '#3B7EA1' if value == 0.5 else '#B05A38',
            'runner': 'fastvideo',
            'host': 'local',
            'gpus': [2, 3] if value == 0.5 else [0, 1],
            'result_subdir': key,
            'checkpoint_root': str(run / 'checkpoints'),
            'model_id': str(MODEL_ID),
            'dit_checkpoint': str(DIT),
            'prompt_suffix': '',
            'prompt_suffix_mode': 'never',
            'prompt_embedding_source': 'online_live_umt5',
            'do_cfg': False,
            'negative_prompt': '',
            'context_noise_mode': 'clean',
            'source_input_list': str(INPUT_LIST),
            'input_caption_profile': 'test70_initial_state_v1',
            'inference_entry': 'mixed_clean_context',
            'enforce_training_match': True,
            'context_inference_ablation': 'user_authorized_clean_context',
            'checkpoint_policy': 'saved checkpoints every 500 steps; skip completed outputs',
            'inference_cache_policy': 'test70 initial_state_v1 captions encoded online; no suffix; clean context and zero context timestep',
            'training_run': str(run),
            'training_config': str(run / 'resolved_config.json'),
            'training_sample_count': 2175,
            'training_objective': f'context sigma nearest to {value:g} * future sigma; future latent frames 2-12 in flow loss',
            'tree_relation': f'Generic2175 initial0907 context-noise interpolation; lambda={value:g}; unified Mix5252 initial-state-v1 clean-context Test70 protocol',
            'step': step,
            'task_id': f'{key}__step-{step:06d}',
            'gpu': (index % 2) + (2 if value == 0.5 else 0),
            'result_root': str(BASE / 'outputs/test70' / key / f'step-{step:06d}'),
            'source_checkpoint': str(run / 'checkpoints' / f'step-{step:06d}'),
        }
        for index, step in enumerate(STEPS)
    ]


def main() -> None:
    args = parse_args()
    value = args.lambda_value
    run = validate(value)
    key = model_key(value)
    task_payload = load(TASKS_PATH)
    settings = load(SETTINGS_PATH)
    new_rows = rows(value, run)
    ids = {row['task_id'] for row in new_rows}
    task_payload['tasks'] = [row for row in task_payload.get('tasks', []) if row.get('task_id') not in ids] + new_rows
    methods = {row['model_key']: row for row in settings.get('methods', []) if isinstance(row, dict) and row.get('model_key')}
    methods[key] = {
        'model_key': key,
        'label': new_rows[0]['label'],
        'block_label': '0-29 · RL LoRA (SA / CA / FFN)',
        'cohort': 'context_noise_interpolation',
    }
    settings['methods'] = list(methods.values())
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'applied' if args.apply else 'validated_not_applied', 'model_key': key, 'lambda': value, 'tasks': len(new_rows), 'expected_videos': 350}
    if args.apply:
        suffix = f'.before-generic2175-lambda{int(round(value * 100)):03d}'
        for path in (TASKS_PATH, SETTINGS_PATH):
            backup = path.with_name(path.name + suffix)
            if not backup.exists():
                shutil.copy2(path, backup)
        atomic_json(TASKS_PATH, task_payload)
        atomic_json(SETTINGS_PATH, settings)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
