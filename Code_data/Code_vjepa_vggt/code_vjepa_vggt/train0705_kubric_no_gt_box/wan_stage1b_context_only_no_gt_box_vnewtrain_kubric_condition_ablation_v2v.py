#!/usr/bin/env python3
"""Run the JSON-native Stage1B inference with explicit text/video conditions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as base,
)


VALID_MODES = {"text_video", "text_only", "video_only", "no_object_branch"}


def _pop_condition_mode(argv: list[str]) -> str:
    mode = None
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--condition-mode":
            if index + 1 >= len(argv):
                raise SystemExit("--condition-mode requires a value")
            mode = argv[index + 1]
            del argv[index : index + 2]
            continue
        if token.startswith("--condition-mode="):
            mode = token.split("=", 1)[1]
            del argv[index]
            continue
        index += 1
    mode = "text_video" if mode is None else str(mode).strip().lower()
    if mode not in VALID_MODES:
        raise SystemExit(
            f"unsupported --condition-mode={mode!r}; expected one of {sorted(VALID_MODES)}"
        )
    return mode


def _set_option_value(argv: list[str], option: str, value: str) -> None:
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} requires a value")
            argv[index + 1] = value
            return
        if token.startswith(f"{option}="):
            argv[index] = f"{option}={value}"
            return
    argv.extend([option, value])


CONDITION_MODE = _pop_condition_mode(sys.argv)
_set_option_value(sys.argv, "--method-suffix", CONDITION_MODE)
if CONDITION_MODE == "video_only":
    # Empty positive and negative prompts make both CFG text branches identical.
    _set_option_value(sys.argv, "--negative-prompt", "")
if CONDITION_MODE == "no_object_branch" and "--disable-object-branch" not in sys.argv:
    sys.argv.append("--disable-object-branch")


_original_parse_args = base.parse_args
_original_run_single_case = base._run_single_case_in_process
_original_build_runtime_model = base.infer0705._build_runtime_model


def _build_runtime_model_with_disabled_branch_compat(*args, **kwargs):
    model, model_args, load_info = _original_build_runtime_model(*args, **kwargs)
    if CONDITION_MODE == "no_object_branch" and model.object_adapter is None:
        # The base batch entrypoint configures this attribute unconditionally.
        # The branch remains natively disabled through model.enable_object_branch.
        model.object_adapter = torch.nn.Identity()
    return model, model_args, load_info


def _parse_args_with_condition_mode():
    args = _original_parse_args()
    args.condition_mode = CONDITION_MODE
    return args


def _condition_result(result: dict[str, object], original_caption: str) -> None:
    uses_text = CONDITION_MODE != "video_only"
    uses_video = CONDITION_MODE != "text_only"
    uses_object_context = CONDITION_MODE in {"text_video", "video_only"}
    result["condition_mode"] = CONDITION_MODE
    result["input_caption_original"] = original_caption
    result["conditioning"] = {
        "text_prompt_used": uses_text,
        "negative_prompt_used": uses_text,
        "context_video_used": uses_video,
        "input_image_used": uses_video,
        "object_context_used": uses_object_context,
    }
    if CONDITION_MODE == "text_only":
        result["source_context_preview"] = result.get("input_video")
        result["input_video"] = None
        result["frame_indices"] = []
        result["effective_context_frames"] = 0
        model_args = result.get("model_args")
        if isinstance(model_args, dict):
            model_args["context_frames"] = 0
    elif CONDITION_MODE == "video_only":
        result["input_caption"] = ""
        result["negative_prompt"] = ""


def _run_single_case_with_condition_mode(*args, **kwargs):
    original_caption = str(kwargs.get("input_caption", ""))
    if CONDITION_MODE == "video_only":
        kwargs["input_caption"] = ""
        kwargs["negative_prompt"] = ""

    if CONDITION_MODE not in {"text_only", "no_object_branch"}:
        result, logs = _original_run_single_case(*args, **kwargs)
        _condition_result(result, original_caption)
        logs.append(f"[condition] mode={CONDITION_MODE}")
        return result, logs

    model = kwargs["model"]
    original_object_builder = base.infer0705._build_object_context

    def _skip_object_context(*_args, **_kwargs):
        return None, {
            "enabled": False,
            "skipped": True,
            "reason": (
                "text_only_no_video_condition"
                if CONDITION_MODE == "text_only"
                else "disable_object_branch"
            ),
            "object_valid_count": 0.0,
        }

    base.infer0705._build_object_context = _skip_object_context
    pipe_class = None
    original_pipe_call = None
    if CONDITION_MODE == "text_only":
        pipe_class = type(model.pipe)
        original_pipe_call = pipe_class.__call__

        def _text_only_pipe_call(self, *call_args, **call_kwargs):
            call_kwargs["context_video"] = None
            call_kwargs["input_image"] = None
            call_kwargs.pop("object_context", None)
            return original_pipe_call(self, *call_args, **call_kwargs)

        pipe_class.__call__ = _text_only_pipe_call
    try:
        result, logs = _original_run_single_case(*args, **kwargs)
    finally:
        if pipe_class is not None and original_pipe_call is not None:
            pipe_class.__call__ = original_pipe_call
        base.infer0705._build_object_context = original_object_builder

    _condition_result(result, original_caption)
    if CONDITION_MODE == "text_only":
        logs.append("[condition] mode=text_only context_video=None object_context=None")
    else:
        logs.append(
            "[condition] mode=no_object_branch context_video=enabled "
            "object_context=None enable_object_branch=False"
        )
    return result, logs


def _has_complete_condition_output(output_video: Path, output_json: Path) -> bool:
    if not output_video.is_file() or not output_json.is_file():
        return False
    try:
        with output_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("condition_mode") == CONDITION_MODE


base.parse_args = _parse_args_with_condition_mode
base._run_single_case_in_process = _run_single_case_with_condition_mode
base._has_complete_existing_output = _has_complete_condition_output
base.infer0705._build_runtime_model = _build_runtime_model_with_disabled_branch_compat


if __name__ == "__main__":
    base.main()
