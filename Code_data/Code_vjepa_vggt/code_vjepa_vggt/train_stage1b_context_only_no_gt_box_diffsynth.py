from __future__ import annotations

import argparse
import json
from pathlib import Path

import accelerate
import torch
from diffsynth.diffusion import ModelLogger

from code_vjepa_vggt.object_token_teacher_student.runtime_stage1b_context_only_no_gt_box import (
    ContextOnlyInjectionNoGTBoxTrainer,
)
from code_vjepa_vggt.train_stage1b_context_only_diffsynth import (
    DEFAULT_COTRACKER_CKPT,
    DEFAULT_JEPA_CKPT,
    DEFAULT_VGGT_ROOT,
    DEFAULT_WAN_ROOT,
    _base_config as _shared_base_config,
    _build_accelerator,
    _init_trackers,
    _load_matching_state_into_model,
    _prepare_args,
    build_parser as _shared_build_parser,
    get_checkpoint_dir,
    resolve_lora_checkpoint_for_resume,
    train_loop,
)
from code_vjepa_vggt.utils.config import load_yaml_config


DEFAULT_CONFIG = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "object_token_teacher_student/config_stage1b_context_only_no_gt_box_template.yaml"
)


class ContextOnlyNoGTBoxDiffSynthModule(ContextOnlyInjectionNoGTBoxTrainer):
    def __init__(self, cfg: dict[str, object], device: str | torch.device) -> None:
        super().__init__(cfg, build_optimizer=True, device=device)
        self.enable_object_branch = True

    def trainable_modules(self):
        return self.trainable_parameters()

    def export_trainable_state_dict(self, state_dict=None, remove_prefix=None):
        if state_dict is None:
            state_dict = self.state_dict()
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        filtered = {
            name: tensor
            for name, tensor in state_dict.items()
            if name in trainable_names
        }
        if remove_prefix is not None:
            filtered = {
                (name[len(remove_prefix):] if name.startswith(remove_prefix) else name): tensor
                for name, tensor in filtered.items()
            }
        return filtered


def _base_config(args: argparse.Namespace) -> dict[str, object]:
    cfg = _shared_base_config(args)
    model_cfg = cfg["model"]

    if args.grounding_device is not None:
        model_cfg["grounding_device"] = str(args.grounding_device)
    if args.grounding_proposal_source is not None:
        model_cfg["grounding_proposal_source"] = str(args.grounding_proposal_source)
    if args.grounding_motion_score_ratio is not None:
        model_cfg["grounding_motion_score_ratio"] = float(args.grounding_motion_score_ratio)
    if args.grounding_text_prompt is not None:
        model_cfg["grounding_text_prompt"] = str(args.grounding_text_prompt)
    if args.grounding_extra_prompt_terms is not None:
        model_cfg["grounding_extra_prompt_terms"] = str(args.grounding_extra_prompt_terms)
    if args.grounding_disable_caption_terms is not None:
        model_cfg["grounding_disable_caption_terms"] = bool(args.grounding_disable_caption_terms)
    if args.grounding_gdino_box_threshold is not None:
        model_cfg["grounding_gdino_box_threshold"] = float(args.grounding_gdino_box_threshold)
    if args.grounding_gdino_text_threshold is not None:
        model_cfg["grounding_gdino_text_threshold"] = float(args.grounding_gdino_text_threshold)
    if args.grounding_prompt_frame_mode is not None:
        model_cfg["grounding_prompt_frame_mode"] = str(args.grounding_prompt_frame_mode)
    if args.grounding_track_dedupe_iou_threshold is not None:
        model_cfg["grounding_track_dedupe_iou_threshold"] = float(args.grounding_track_dedupe_iou_threshold)
    if args.grounding_container_suppress_ratio_threshold is not None:
        model_cfg["grounding_container_suppress_ratio_threshold"] = float(
            args.grounding_container_suppress_ratio_threshold
        )
    if args.grounding_container_suppress_min_contained is not None:
        model_cfg["grounding_container_suppress_min_contained"] = int(
            args.grounding_container_suppress_min_contained
        )
    if args.grounding_container_suppress_min_area_ratio is not None:
        model_cfg["grounding_container_suppress_min_area_ratio"] = float(
            args.grounding_container_suppress_min_area_ratio
        )
    if args.grounding_container_suppress_small_iou_threshold is not None:
        model_cfg["grounding_container_suppress_small_iou_threshold"] = float(
            args.grounding_container_suppress_small_iou_threshold
        )
    return cfg


def _build_model(
    args: argparse.Namespace,
    accelerator: accelerate.Accelerator,
) -> ContextOnlyNoGTBoxDiffSynthModule:
    cfg = _base_config(args)
    return ContextOnlyNoGTBoxDiffSynthModule(cfg, device=accelerator.device)


def build_parser() -> argparse.ArgumentParser:
    parser = _shared_build_parser()
    parser.description = (
        "Train Stage1B context-only no-GT-box branch with "
        "DiffSynth/v_newtrain-style argparse/checkpoint framework."
    )
    parser.set_defaults(
        config=str(DEFAULT_CONFIG),
        wan_root=str(DEFAULT_WAN_ROOT),
        jepa_ckpt_path=str(DEFAULT_JEPA_CKPT),
        vggt_model_path=str(DEFAULT_VGGT_ROOT),
        cotracker_checkpoint=str(DEFAULT_COTRACKER_CKPT),
    )

    parser.add_argument(
        "--grounding_device",
        default=None,
        help="Grounding runtime device. Defaults to the local training device if unset.",
    )
    parser.add_argument("--grounding_proposal_source", default=None)
    parser.add_argument("--grounding_motion_score_ratio", type=float, default=None)
    parser.add_argument("--grounding_text_prompt", default=None)
    parser.add_argument("--grounding_extra_prompt_terms", default=None)
    parser.add_argument(
        "--grounding_disable_caption_terms",
        action="store_true",
        default=None,
        help="Disable caption-derived prompt terms for viewer grounding.",
    )
    parser.add_argument("--grounding_gdino_box_threshold", type=float, default=None)
    parser.add_argument("--grounding_gdino_text_threshold", type=float, default=None)
    parser.add_argument("--grounding_prompt_frame_mode", default=None)
    parser.add_argument("--grounding_track_dedupe_iou_threshold", type=float, default=None)
    parser.add_argument("--grounding_container_suppress_ratio_threshold", type=float, default=None)
    parser.add_argument("--grounding_container_suppress_min_contained", type=int, default=None)
    parser.add_argument("--grounding_container_suppress_min_area_ratio", type=float, default=None)
    parser.add_argument("--grounding_container_suppress_small_iou_threshold", type=float, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = _prepare_args(parser.parse_args())

    template_cfg = load_yaml_config(Path(args.config).expanduser().resolve())
    if args.lora_checkpoint is None:
        args.lora_checkpoint = str(template_cfg["model"]["init_wan_lora_from_checkpoint"])
    if args.output_path is None:
        args.output_path = str(Path(template_cfg["experiment"]["output_dir"]).expanduser().resolve())

    accelerator = _build_accelerator(args)
    _init_trackers(accelerator, args)

    if args.grounding_device is None:
        args.grounding_device = str(accelerator.device)

    model = _build_model(args, accelerator)
    resolved_cfg = _base_config(args)
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "resolved_stage1b_context_only_no_gt_box_config.json").write_text(
        json.dumps(resolved_cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.head_resume_from is not None:
        init_info = _load_matching_state_into_model(model, args.head_resume_from)
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded init checkpoint: "
                f"loaded_count={init_info['loaded_count']}, "
                f"selected_source_keys={init_info['selected_source_keys']}, "
                f"shape_mismatch={len(init_info['skipped_shape_mismatch'])}"
            )

    if args.stage2_resume_from is not None:
        resume_ckpt = resolve_lora_checkpoint_for_resume(args.stage2_resume_from)
        resume_info = _load_matching_state_into_model(model, resume_ckpt)
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded resume checkpoint weights: "
                f"loaded_count={resume_info['loaded_count']}, "
                f"selected_source_keys={resume_info['selected_source_keys']}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )

    model_logger = ModelLogger(
        output_path=get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    progress = train_loop(
        accelerator=accelerator,
        model=model,
        model_logger=model_logger,
        args=args,
    )
    if accelerator.is_main_process:
        (output_path / "train_summary.json").write_text(
            json.dumps({"progress": progress}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
