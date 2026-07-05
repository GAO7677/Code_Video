from __future__ import annotations

"""Stage1A full-token video visualization entry point for train0706.

This is a thin wrapper around the existing Stage1A visualizer:
  code_vjepa_vggt.object_token_teacher_student.inspect_stage1a_aux_losses

It exists in train0706_wan1p3b so the 1.3B workflow has a local stage1a
inference/visualization entry that matches the new directory layout.

Example:
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  CUDA_VISIBLE_DEVICES=3 \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/infer_stage1a_full_token_v2v.py \
    --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/config_stage1a_full_token_wan21_13b.yaml \
    --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token/step_0003000.pt \
    --indices 0 \
    --output-dir /data/gaoya/AAA_test_video/0623/test/v2v_1p3b/train0706_stage1a_full_token
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from code_vjepa_vggt.object_token_teacher_student.inspect_stage1a_aux_losses import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
