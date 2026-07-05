from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from code_vjepa_vggt.utils.config import load_yaml_config

from .runtime_stage1a_full_token import FullTokenTeacherTrainer


DEFAULT_CONFIG = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "train0706_wan1p3b/config_stage1a_full_token_wan21_13b.yaml"
)


def _resolve_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check the train0706 stage1a full-token checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config.expanduser().resolve())
    trainer = FullTokenTeacherTrainer(cfg, device=_resolve_device())

    resume_checkpoint = args.resume_checkpoint.expanduser().resolve() if args.resume_checkpoint else None
    init_from = args.init_from.expanduser().resolve() if args.init_from else None
    if resume_checkpoint is not None and not resume_checkpoint.exists():
        raise FileNotFoundError(f"resume-checkpoint not found: {resume_checkpoint}")
    if init_from is not None and not init_from.exists():
        raise FileNotFoundError(f"init-from not found: {init_from}")

    load_info = None
    load_source = None
    checkpoint_path = resume_checkpoint or init_from
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            load_info = trainer.load_state_dict(state["model"], strict=False)
            load_source = "model"
        elif isinstance(state, dict):
            load_info = trainer.load_state_dict(state, strict=False)
            load_source = "raw"
        else:
            raise TypeError(f"unsupported checkpoint format: {checkpoint_path}")

    payload = {
        "config": str(args.config.expanduser().resolve()),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "init_from": str(init_from) if init_from else None,
        "load_source": load_source,
        "load_info": None
        if load_info is None
        else {
            "missing_keys": list(getattr(load_info, "missing_keys", [])),
            "unexpected_keys": list(getattr(load_info, "unexpected_keys", [])),
        },
        "inspect": trainer.inspect_one_batch(),
    }

    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output_json)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
