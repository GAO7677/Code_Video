from __future__ import annotations

"""
Kubric stage1b context-only no-object-branch inference wrapper.

This keeps the Kubric runtime stack but forces object-branch inference off, so
the loaded checkpoint is evaluated as a clean no-object-branch ablation.
"""

from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as base,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_base,
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as trainmod,
)

_ORIG_BUILD_MODEL_ARGS = base._build_model_args


def _build_object_context(
    model,
    *,
    context_video_single,
    prompt: str,
    video_path: str,
):
    if not bool(getattr(model, "enable_object_branch", False)):
        return None, {"enabled": False, "reason": "disable_object_branch"}
    return kubric_base._build_object_context(
        model,
        context_video_single=context_video_single,
        prompt=prompt,
        video_path=video_path,
    )


def _build_model_args(args):
    args.disable_object_branch = True
    model_args = _ORIG_BUILD_MODEL_ARGS(args)
    return model_args


def main() -> None:
    base.t0705 = trainmod
    base._build_object_context = _build_object_context
    base._build_model_args = _build_model_args
    base.main()


if __name__ == "__main__":
    main()
