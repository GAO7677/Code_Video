from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from code_vjepa_vggt.AAAinfer.wan_stage1b_context_only_v2v import _load_trainable_state_into_model
from code_vjepa_vggt.object_token_teacher_student.runtime_stage1b_context_only import ContextOnlyInjectionTrainer
from code_vjepa_vggt.utils.config import load_yaml_config


def _resolve_checkpoint(output_dir: Path, step: int) -> Path:
    path = output_dir / f"step_{step:07d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _save_trainable_checkpoint(trainer: ContextOnlyInjectionTrainer, output_dir: Path, step: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"step_{step:07d}.pt"
    state = {
        "step": int(step),
        "model": trainer.export_trainable_state_dict(),
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path


def _load_checkpoint_into_trainer(trainer: ContextOnlyInjectionTrainer, checkpoint_path: Path) -> dict[str, object]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state or not isinstance(state["model"], dict):
        raise RuntimeError(f"unsupported checkpoint format: {checkpoint_path}")
    loaded = trainer.load_state_dict(state["model"], strict=False)
    return {
        "step": int(state.get("step", -1)),
        "loaded_key_count": len(state["model"]),
        "missing_keys": list(loaded.missing_keys),
        "unexpected_keys": list(loaded.unexpected_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for Stage1B context-only train/save/resume/infer pipeline.")
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student/config_stage1b_context_only_smoke.yaml",
    )
    parser.add_argument(
        "--init-from",
        default="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    init_from = Path(args.init_from).expanduser().resolve()
    cfg = load_yaml_config(config_path)
    output_dir = Path(cfg["experiment"]["output_dir"]).expanduser().resolve()
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    trainer = ContextOnlyInjectionTrainer(cfg, build_optimizer=True, device=args.device)
    init_info = _load_checkpoint_into_trainer(trainer, init_from)
    batch = next(iter(trainer.build_dataloader(num_workers=0)))
    prepared_init = trainer._prepare_batch(batch)
    first_ckpt = _save_trainable_checkpoint(trainer, output_dir, step=1)

    resume_trainer = ContextOnlyInjectionTrainer(cfg, build_optimizer=True, device=args.device)
    resume_info = _load_checkpoint_into_trainer(resume_trainer, first_ckpt)
    prepared_resume = resume_trainer._prepare_batch(batch)
    second_ckpt = _save_trainable_checkpoint(resume_trainer, output_dir, step=2)

    infer_trainer = ContextOnlyInjectionTrainer(cfg, build_optimizer=True, device=args.device)
    infer_load_info = _load_trainable_state_into_model(infer_trainer, second_ckpt)
    prepared = infer_trainer._prepare_batch(batch)

    report = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "wan_root": str(cfg["model"]["wan_ckpt_dir"]),
        "frozen_wan_lora_init": str(cfg["model"].get("init_wan_lora_from_checkpoint")),
        "stage1a_init_from": str(init_from),
        "first_checkpoint": str(first_ckpt),
        "second_checkpoint": str(second_ckpt),
        "init_info": init_info,
        "resume_info": resume_info,
        "infer_load_info": infer_load_info,
        "prepared_init_debug_mode": prepared_init["debug"].get("teacher_student_stage1", {}),
        "prepared_resume_debug_mode": prepared_resume["debug"].get("teacher_student_stage1", {}),
        "prepared_debug_mode": prepared["debug"].get("teacher_student_stage1", {}),
        "object_context_shape": list(prepared["object_context"].shape),
        "object_latent_tokens_shape": list(prepared["object_latent_tokens"].shape),
        "trainable_parameter_names": [
            name for name, param in infer_trainer.named_parameters()
            if param.requires_grad
        ],
    }
    report_path = output_dir / "smoke_stage1b_context_only_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
