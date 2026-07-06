"""
Frequency-weighted V-JEPA guidance wrapper over wanti2v.py.

Example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v_freqguidance.py \
    --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
    --model-name wan22_official_freqguide_smoke \
    --backend official \
    --size 704*1280 \
    --frame-num 49 \
    --sampling-steps 40 \
    --cfg-scale 5.0 \
    --fps 30 \
    --seed 42 \
    --offload-model \
    --vjepa-preset target_w24_s15_ratio_0025 \
    --vjepa-ckpt /data/gaoya/ckpt/VJEPA2/vith.pt \
    --limit 1
"""

from __future__ import annotations

import sys

from code_vjepa_free.vjepa_guidance.wanti2v import main as wanti2v_main


def _has_flag(flag: str) -> bool:
    return flag in sys.argv


def _inject_flag(flag: str, value: str | None = None) -> None:
    if _has_flag(flag):
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


def main() -> None:
    _inject_flag("--motion-mask-mode", "temporal_union_except_first")
    _inject_flag("--vjepa-use-spectral-guidance")
    _inject_flag("--vjepa-spectral-lowpass-ratio", "0.18")
    _inject_flag("--vjepa-spectral-normalize-percentile", "95.0")
    _inject_flag("--vjepa-spectral-weight-floor", "0.25")
    _inject_flag("--vjepa-spectral-weight-scale", "1.0")
    _inject_flag("--vjepa-spectral-mask-dilation", "5")
    wanti2v_main()


if __name__ == "__main__":
    main()
