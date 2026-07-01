#!/usr/bin/env bash
# object_context 敏感性消融实验 — 串行跑7个 ablation mode，全部在 gpu5 上。
#
# 用法：
#   CUDA_VISIBLE_DEVICES=5 bash run_ablation_gpu5.sh
#
# 可按需修改下方变量：
#   WEIGHTS_ROOT   stage1b checkpoint（.pt 文件或包含 step_*.pt 的目录）
#   STAGE1A        stage1a pooler checkpoint
#   INPUT_LIST     测试集 txt（每行一个 json 路径）
#   MODEL_NAME     输出目录前缀；各 mode 自动追加 _<ablation_mode> 后缀
#   GPU            CUDA_VISIBLE_DEVICES（默认 5）
#
# 示例输出目录结构：
#   /data/gaoya/AAA_test_video/0623/test/v2v/<MODEL_NAME>/
#     oracle_cross_attn_step_0001500/          <- baseline
#     oracle_cross_attn_step_0001500_future_zero/
#     oracle_cross_attn_step_0001500_future_noise/
#     ...

set -euo pipefail

# ─── 配置 ────────────────────────────────────────────────────────────────────
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1b_oracle_cross_attn/step_0001500.pt}"
STAGE1A="${STAGE1A:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt}"
INPUT_LIST="${INPUT_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
MODEL_NAME="${MODEL_NAME:-pybullet0629_ablation}"
GPU="${CUDA_VISIBLE_DEVICES:-5}"

SCRIPT="$(dirname "$0")/wan_fulltok_ablation.py"
PYTHONPATH_EXTRA="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main"

MODES=(
    future_zero
    future_noise
    all_zero
    ctx_zero
    future_rand_ctx_frame
    object_context_zero
)
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Ablation experiment"
echo "  WEIGHTS_ROOT : $WEIGHTS_ROOT"
echo "  STAGE1A      : $STAGE1A"
echo "  INPUT_LIST   : $INPUT_LIST"
echo "  MODEL_NAME   : $MODEL_NAME"
echo "  GPU          : $GPU"
echo "  MODES        : ${MODES[*]}"
echo "============================================================"

for MODE in "${MODES[@]}"; do
    echo ""
    echo ">>> mode=$MODE  $(date '+%H:%M:%S')"
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$PYTHONPATH_EXTRA" \
    python3 "$SCRIPT" \
        --weights-root      "$WEIGHTS_ROOT" \
        --stage1a-weights   "$STAGE1A" \
        --input-json-list-path "$INPUT_LIST" \
        --model-name        "$MODEL_NAME" \
        --ablation-mode     "$MODE"
    echo "<<< mode=$MODE done  $(date '+%H:%M:%S')"
done

echo ""
echo "All modes finished."
