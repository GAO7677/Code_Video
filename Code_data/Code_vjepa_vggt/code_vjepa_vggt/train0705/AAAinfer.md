# train0705 推理脚本运行示例

下面只整理 `train0705` 目录下和推理直接相关的脚本，不包含训练脚本，也不包含纯指标汇总脚本。

## 1. 单 case 推理

脚本:
`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705.py`

用途:
单独给一个 `context video + prompt + checkpoint`，直接生成一个结果视频和同名 json。

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705.py \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
  --context-video /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_001460/source_video/context_video_8f.mp4 \
  --prompt "f5 sample 001460 industrial rigid body simulation sphere box" \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/inference_review/step-001000 \
  --sampling-steps 12
```

带 VJEPA guidance 的例子:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705.py \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --context-video /path/to/context_video_8f.mp4 \
  --prompt "your prompt" \
  --output-dir /data/gaoya/agent-data/outputs/train0705_vjepa_demo \
  --sampling-steps 40 \
  --vjepa-preset ladder_s20 \
  --vjepa-device cuda:0
```

## 2. 单帧 context 复制成 2 帧再送 JEPA 的单 case 推理

脚本:
`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705_ctx1dupjepa.py`

用途:
专门处理 `context_frames=1` 的情况。在进入 `_run_jepa` 前把单帧复制成 2 帧，同时把 JEPA adapter 的 `num_frames` 改成 2。

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705_ctx1dupjepa.py \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500/checkpoint.safetensors \
  --context-video /data/gaoya/agent-data/outputs/train0705_ctx1_smoke_20260706/input/physicIQ_0002_ctx01f.mp4 \
  --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement." \
  --output-dir /data/gaoya/agent-data/outputs/train0705_ctx1_smoke_20260706/run_ctx1dupjepa \
  --context-frames 1 \
  --num-frames 9 \
  --sampling-steps 6 \
  --initialize-model-on-cpu
```

## 3. 批量 v2v 推理

脚本:
`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py`

用途:
读取一个 txt 文件，每行一个输入 json，批量跑整套 v2v 推理。内部会复用单 case 推理脚本的 object-conditioning 路径。

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_diffsynth_native0705_step1000 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_test5_compare \
  --height 512 \
  --width 896 \
  --input-cover-crop-width 832 \
  --input-cover-crop-height 480 \
  --num-frames 24 \
  --context-frames 8 \
  --sampling-mode prefix \
  --num-inference-steps 40 \
  --cfg-scale 5.0
```

带 VJEPA guidance 的例子:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=3 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_diffsynth_native0705_step1000_vjepa \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_test5_compare_vjepa \
  --height 512 \
  --width 896 \
  --input-cover-crop-width 832 \
  --input-cover-crop-height 480 \
  --num-frames 24 \
  --context-frames 8 \
  --sampling-mode prefix \
  --num-inference-steps 40 \
  --cfg-scale 5.0 \
  --vjepa-preset ladder_s20 \
  --vjepa-device cuda:0
```

## 4. 单帧 context 复制成 2 帧再送 JEPA 的批量 v2v 推理

脚本:
`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_ctx1dupjepa_v2v.py`

用途:
批量读取 json list，但在推理内部对 `context_frames=1` 做 JEPA 特殊处理：
建模时把 JEPA 的 `num_frames` 改成 2，进入 `_run_jepa` 前把单帧复制成 2 帧。

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_ctx1dupjepa_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
  --model-name train_stage1b_diffsynth_native0705_ctx1_ti2v_step002500 \
  --output-root /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_physicIQ \
  --context-frames 1 \
  --num-inference-steps 40


PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=3 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_ctx1dupjepa_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt \
  --model-name train_stage1b_diffsynth_native0705_ctx1_ti2v_step002500 \
  --output-root /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_morpheus_real_world \
  --context-frames 1 \
  --num-inference-steps 40
```

## 5. pre-pipe 处理流程可视化

脚本:
`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/inspect_stage1b_prepipe_overlay.py`

用途:
对选定 json case 可视化进入 `pipe()` 前的处理流程，包括 viewer grounding boxes、query points、CoTracker 轨迹等 overlay。

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/inspect_stage1b_prepipe_overlay.py \
  --steps step-002500 step-007000 \
  --input-jsons \
    /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_000336_w001.json \
    /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json \
    /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/phyco_kubric_ball_drop_soft_v4_2025-09-05_0144a4.json
```

## 6. source video context 帧数 sweep

脚本:
`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/sweep_source_video_context_frames_train0705.py`

用途:
针对同一批 json，用 source video 的前若干帧构造不同长度的 context video，然后批量调用 `wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py` 跑对比。

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/sweep_source_video_context_frames_train0705.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-007000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_diffsynth_native0705_0705_sourcectx_sweep \
  --context-frames-list 4 16 24 \
  --num-inference-steps 40
```

## 7. 自动监控新权重并触发推理

脚本:
`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/watch_stage1b_context_only_no_gt_box_vnewtrain0705.py`

用途:
前台轮询新的 `step-*` checkpoint。发现新权重后自动调用批量推理脚本，随后再跑 `bench.sh`。

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/watch_stage1b_context_only_no_gt_box_vnewtrain0705.py
```

## 8. 说明

- `wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py` 是批量包装脚本，内部复用 `infer_stage1b_context_only_no_gt_box_v_newtrain0705.py` 的核心推理链。
- 当前 `wan-cu128` 环境建议在命令前加 `PYTHONNOUSERSITE=1`，避免被 `~/.local/lib/python3.10/site-packages` 里的旧版 `huggingface_hub` 元数据污染。
- `wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py` 现在默认对输入 context video 先做 `832x480` 的等比 cover，再做 center crop，最后再缩放到模型输入分辨率 `512x896`。
- `infer_stage1b_context_only_no_gt_box_v_newtrain0705_ctx1dupjepa.py` 是为了单帧 context 单独加的兼容脚本。
- `wan_stage1b_context_only_no_gt_box_vnewtrain0705_ctx1dupjepa_v2v.py` 是对应的批量版本，复用原批量脚本，只替换单帧 JEPA 相关逻辑。
- `collect_stage1b_metric_table.py` 属于结果汇总，不属于推理脚本，这里不收录。

## 9. Physics-IQ Verified 双卡正式跑法

严格按官方 `Physics-IQ Verified` workflow 跑当前 `train0705 native` 权重，推荐直接走下面这个 shell 入口：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.sh
```

它内部调用的 Python wrapper 是：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.py
```

这个 wrapper 会：

- 直接读取官方 `descriptions_base.csv`
- 只取 `take-1` 的 198 个 case
- 直接读取官方 Verified conditioning videos
- 生成阶段自动对齐到底层 Wan 要求的 `num_frames % 4 == 1`
- 生成后自动裁成官方要求的精确 `5.0s`
- 最终生成结果保留在原生目录 `step-*` 或你显式指定的 `--step-output-dir-name`
- 已有 `mp4 + json` 的样本会自动跳过，只继续未完成样本

推荐用和 Kubric 一样的 `GPU_PAIR=主卡,辅卡` 形式。比如主模型放第一张卡，辅助 object stack 放第二张卡：

```bash
GPU_PAIR=0,1 \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
MODEL_NAME=train_stage1b_diffsynth_native0705_step2500_physiq_verified \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/physicsiq/train_stage1b_diffsynth_native0705 \
VERIFIED_ROOT=/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified \
DESCRIPTIONS_FILE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv \
FPS=30 \
HEIGHT=512 \
WIDTH=896 \
INPUT_COVER_CROP_HEIGHT=480 \
INPUT_COVER_CROP_WIDTH=832 \
NUM_FRAMES=150 \
CONTEXT_FRAMES=20 \
SAMPLING_MODE=prefix \
NUM_INFERENCE_STEPS=40 \
CFG_SCALE=5.0 \
SEED=42 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.sh
```

如果要跑带中文 negative prompt 的版本，并把 method / 子目录后缀也区分开：

```bash
GPU_PAIR=0,1 \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
MODEL_NAME=train_stage1b_diffsynth_native0705_step2500_physiq_verified \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/physicsiq/train_stage1b_diffsynth_native0705 \
VERIFIED_ROOT=/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified \
DESCRIPTIONS_FILE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv \
FPS=30 \
HEIGHT=512 \
WIDTH=896 \
INPUT_COVER_CROP_HEIGHT=480 \
INPUT_COVER_CROP_WIDTH=832 \
NUM_FRAMES=150 \
CONTEXT_FRAMES=20 \
SAMPLING_MODE=prefix \
NUM_INFERENCE_STEPS=40 \
CFG_SCALE=5.0 \
SEED=42 \
NEGATIVE_PROMPT='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走' \
STEP_OUTPUT_DIR_NAME=step-002500_withneg \
METHOD_SUFFIX=withneg \
RUN_NAME=train_stage1b_diffsynth_native0705_step2500_physiq_verified_withneg-bpp-run_01 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.sh
```

双卡分工说明：

- 第一张卡：
  - Wan pipe / DiT / VAE
  - trainable object branch 里的 `object_pooler` / `object_aux_heads` / `object_adapter`
- 第二张卡：
  - `JEPA` adapter
  - `CoTracker` adapter
  - `VGGT` adapter
- 当前 shell 入口在双卡模式下会默认额外传：
  - `--inference-devices cuda:0,cuda:1`
  - `--grounding-device cuda:1`
- 这表示 viewer grounding 也尽量放到辅助卡，减轻主卡显存压力

注意：

- `GPU_PAIR=0,1` 指物理卡 0 和 1；脚本会自动把它们映射成进程内的 `cuda:0,cuda:1`
- 如果写成 `GPU_PAIR=0,0`，脚本会自动退化成单卡，不再传 `--inference-devices`
- 禁止使用 `gpu4`
