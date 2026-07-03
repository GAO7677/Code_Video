#!/usr/bin/env bash
# =============================================================================
# Stage1b DiffSynth-Native 正式训练启动脚本 (train0704)
#
# 用法:
#   bash run_train_stage1b_diffsynth_native0704.sh            # 自动断点续训 (默认)
#   GPU=3 bash run_train_stage1b_diffsynth_native0704.sh      # 指定物理 GPU
#   RESUME=none bash run_train_stage1b_diffsynth_native0704.sh # 强制从头开始训练
#
# 说明:
#   - 默认 RESUME=auto: 自动检测 output_dir 下最新的 checkpoint 继续训练
#     (加载权重 + optimizer 状态 + step, 首次运行无 checkpoint 时自动从头开始)
#   - 每 500 step 保存一次权重 (config: logging.save_every)
#   - 前台运行, 不使用 nohup / & / 后台方式
#   - 禁用 gpu4 (故障), 默认使用物理 GPU7
# =============================================================================
set -euo pipefail

# ---- 可配置项 (环境变量覆盖) ----
GPU="${GPU:-7}"                 # 物理 GPU 编号 (禁止使用 gpu4)
RESUME="${RESUME:-auto}"        # auto=自动续训, none=从头开始, 或指定 checkpoint 目录

PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
SCRIPT=code_vjepa_vggt/train0704/train_stage1b_diffsynth_native.py
CONFIG=code_vjepa_vggt/train0704/config_stage1b_diffsynth_native_test.yaml

if [ "$GPU" = "4" ]; then
  echo "ERROR: gpu4 故障, 禁止使用。请指定其他 GPU。" >&2
  exit 1
fi

cd "$PROJ"

# 通过 CUDA_VISIBLE_DEVICES 映射物理 GPU, 脚本内部固定用 cuda:0
CMD=(env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT" --config "$CONFIG" --gpu 0)

if [ "$RESUME" = "none" ]; then
  echo "[启动] 从头开始训练 (不续训)"
else
  CMD+=(--resume "$RESUME")
  echo "[启动] 断点续训模式: RESUME=$RESUME"
fi

echo "[启动] 物理 GPU=$GPU  配置=$CONFIG"
echo "[启动] 命令: ${CMD[*]}"
exec "${CMD[@]}"
