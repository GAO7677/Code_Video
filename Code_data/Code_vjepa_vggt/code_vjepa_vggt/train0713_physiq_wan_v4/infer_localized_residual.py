from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import torch

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_infer,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as trainmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v_queryscheme as queryscheme,
)

from code_vjepa_vggt.train0713_physiq_wan_v4.conditioned_residual import (
    LocalizedConditionedResidual,
    LocalizedResidualConfig,
)


_CONFIG = LocalizedResidualConfig()
_ACTIVE_CONTROLLER: LocalizedConditionedResidual | None = None


def _boxes_from_grounding(grounding: Any) -> torch.Tensor:
    tracks = list(getattr(grounding, "object_tracks", []) or [])
    tubes = [np.asarray(track.boxes_t4, dtype=np.float32) for track in tracks]
    tubes = [tube for tube in tubes if tube.ndim == 2 and tube.shape[0] > 0 and tube.shape[1] == 4]
    if not tubes:
        return torch.empty((0, 0, 4), dtype=torch.float32)
    observed = min(int(tube.shape[0]) for tube in tubes)
    return torch.from_numpy(np.stack([tube[:observed] for tube in tubes], axis=1))


def _masks_from_grounding(grounding: Any) -> torch.Tensor | None:
    tracks = list(getattr(grounding, "object_tracks", []) or [])
    masks = [np.asarray(track.masks_thw, dtype=np.float32) for track in tracks]
    masks = [mask for mask in masks if mask.ndim == 3 and mask.shape[0] > 0]
    if not masks:
        return None
    observed = min(int(mask.shape[0]) for mask in masks)
    image_hw = tuple(int(value) for value in masks[0].shape[-2:])
    masks = [mask for mask in masks if tuple(int(value) for value in mask.shape[-2:]) == image_hw]
    if not masks:
        return None
    return torch.from_numpy(np.stack([mask[:observed] for mask in masks], axis=1))


def _build_object_context_with_localized_residual(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    global _ACTIVE_CONTROLLER
    captured: dict[str, Any] = {}
    original_build_sample = model.viewer_grounding.build_sample

    def capture_grounding(*args, **kwargs):
        grounding = original_build_sample(*args, **kwargs)
        captured["grounding"] = grounding
        return grounding

    model.viewer_grounding.build_sample = capture_grounding
    try:
        object_context, debug = queryscheme._build_object_context_with_query_scheme(
            model=model,
            context_video_single=context_video_single,
            prompt=prompt,
            video_path=video_path,
        )
    finally:
        model.viewer_grounding.build_sample = original_build_sample

    grounding = captured.get("grounding")
    boxes = _boxes_from_grounding(grounding) if grounding is not None else torch.empty((0, 0, 4))
    masks = _masks_from_grounding(grounding) if grounding is not None else None
    if boxes.numel() == 0:
        raise RuntimeError("localized residual requires at least one valid prefix object box tube")
    _ACTIVE_CONTROLLER = LocalizedConditionedResidual(
        boxes,
        image_hw=(int(context_video_single.shape[-2]), int(context_video_single.shape[-1])),
        config=_CONFIG,
        observed_masks=masks,
    )
    model.pipe._conditioned_residual_controller = _ACTIVE_CONTROLLER
    debug["localized_conditioned_residual"] = {
        "config": _ACTIVE_CONTROLLER.summary()["config"],
        "observed_boxes_shape": list(boxes.shape),
        "observed_masks_shape": None if masks is None else list(masks.shape),
    }
    return object_context, debug


def _run_single_case_with_localized_residual(**kwargs):
    global _ACTIVE_CONTROLLER
    model = kwargs["model"]
    model.pipe._conditioned_residual_controller = None
    _ACTIVE_CONTROLLER = None
    try:
        result, logs = queryscheme._run_single_case_with_query_scheme(**kwargs)
        if _ACTIVE_CONTROLLER is None:
            raise RuntimeError("localized residual controller was not installed for the case")
        summary = _ACTIVE_CONTROLLER.summary()
        result["localized_conditioned_residual"] = summary
        logs.append(
            "[localized-residual] "
            f"steps={len(summary['steps'])} config={summary['config']}"
        )
        return result, logs
    finally:
        model.pipe._conditioned_residual_controller = None
        _ACTIVE_CONTROLLER = None


def _install_hooks() -> None:
    batch.infer0705.t0705 = trainmod
    batch.infer0705._build_object_context = _build_object_context_with_localized_residual
    batch.infer0705._build_model_args = kubric_infer._build_model_args


def _parse_experiment_args() -> None:
    global _CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query-scheme", choices=["legacy", "temporal_sam2"], default="temporal_sam2")
    parser.add_argument("--localized-residual-scale", type=float, default=1.0)
    parser.add_argument("--localized-residual-dilation-ratio", type=float, default=0.15)
    parser.add_argument("--localized-residual-step-start", type=float, default=0.0)
    parser.add_argument("--localized-residual-step-end", type=float, default=1.0)
    parser.add_argument("--localized-residual-confidence", type=float, default=1.0)
    known, remaining = parser.parse_known_args(sys.argv[1:])
    queryscheme._SCHEME = str(known.query_scheme)
    _CONFIG = LocalizedResidualConfig(
        scale=float(known.localized_residual_scale),
        dilation_ratio=float(known.localized_residual_dilation_ratio),
        active_step_start=float(known.localized_residual_step_start),
        active_step_end=float(known.localized_residual_step_end),
        condition_confidence=float(known.localized_residual_confidence),
    )
    _CONFIG.validate()
    sys.argv = [sys.argv[0], *remaining]


def main() -> None:
    _parse_experiment_args()
    batch._install_kubric_runtime_hooks = _install_hooks
    batch._run_single_case_in_process = _run_single_case_with_localized_residual
    batch.main()


if __name__ == "__main__":
    main()
