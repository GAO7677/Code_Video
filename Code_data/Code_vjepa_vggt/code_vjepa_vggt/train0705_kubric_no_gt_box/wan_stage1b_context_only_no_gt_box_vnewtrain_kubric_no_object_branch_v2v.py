from __future__ import annotations

"""
Batch v2v inference entrypoint for the Kubric no-object-branch ablation.

This reuses the standard Kubric batch v2v pipeline, but swaps in a dedicated
runtime wrapper that forces object-branch inference off.

Run command example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=0 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_no_object_branch_v2v.py \
  --disable-object-branch \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step1000_no_object_branch \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_no_object_branch_compare_0708 \
  --num-inference-steps 40 \
  --num-frames 49
"""

import sys

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric_no_object_branch as kubric_no_object_infer,
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as base_v2v,
)


def _parse_args_force_no_object_branch():
    args = _ORIG_PARSE_ARGS()
    if "--disable-object-branch" not in sys.argv:
        raise SystemExit(
            "This dedicated no-object-branch entry requires an explicit "
            "--disable-object-branch flag."
        )
    args.disable_object_branch = True
    return args


_ORIG_PARSE_ARGS = base_v2v.parse_args


def main() -> None:
    base_v2v.kubric_infer = kubric_no_object_infer
    base_v2v.parse_args = _parse_args_force_no_object_branch
    try:
        base_v2v.main()
    finally:
        base_v2v.parse_args = _ORIG_PARSE_ARGS


if __name__ == "__main__":
    main()
