from __future__ import annotations

# Run command example:
'''
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_ctx1dupjepa_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_diffsynth_native0705_ctx1dupjepa_0705 \
  --context-frames 1 \
  --num-inference-steps 40

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_ctx1dupjepa_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_vjepa \
  --context-frames 1 \
  --num-inference-steps 40 \
  --vjepa-preset ladder_s20 \
  --vjepa-device cuda:0
'''

"""
Batch V2V wrapper for the single-frame context JEPA workaround.

This reuses the standard train0705 batch V2V script, but swaps in the
single-frame-compatible JEPA path from
`infer_stage1b_context_only_no_gt_box_v_newtrain0705_ctx1dupjepa.py`:

1. If `context_frames == 1`, model args are patched so JEPA is built with
   `fixed_num_context_frames = 2`.
2. Right before `_run_jepa`, the single context frame is duplicated to length 2.

All other batching behavior, outputs, logs, manifests, and argument handling are
kept identical to the original
`wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py`.
"""

from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705_ctx1dupjepa as ctx1dupjepa,
)
from code_vjepa_vggt.train0705 import (
    wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v as base_batch,
)


def main() -> None:
    base_batch.infer0705._build_model_args = ctx1dupjepa._build_model_args
    base_batch.infer0705._build_object_context = ctx1dupjepa._build_object_context
    base_batch.main()


if __name__ == "__main__":
    main()
