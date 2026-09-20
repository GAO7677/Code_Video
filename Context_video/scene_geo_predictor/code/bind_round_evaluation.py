"""Bind an isolated round to its completed model and the unchanged constant control."""
import argparse
import json
from pathlib import Path

from prepare_small_trial import ROOT, dump, sha


def bind(round_root):
    train, control = round_root/'train', round_root/'control'
    checkpoint = train/'runs/visual/best.safetensors'
    if not (checkpoint.parent/'complete.json').exists():
        raise ValueError('Training is not complete')
    constant = train/'runs/constant'
    original = ROOT/'small_trial_120/runs/constant'
    if not constant.exists():
        constant.symlink_to(original, target_is_directory=True)
    elif constant.resolve() != original.resolve():
        raise ValueError('Constant baseline changed')
    path = control/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['checkpoints']['visual'] = dict(path=str(checkpoint.resolve()), sha256=sha(checkpoint))
    manifest['round_evaluation'] = str(round_root.resolve())
    dump(path, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=Path, required=True)
    bind(parser.parse_args().round)
