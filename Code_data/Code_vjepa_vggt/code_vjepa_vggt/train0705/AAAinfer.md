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
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step1000 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708 \
  --num-inference-steps 40 \
  --num-frames 49
```

带 VJEPA guidance 的例子:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=3 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_diffsynth_native0705_0705_vjepa \
  --num-inference-steps 40 \
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
- `infer_stage1b_context_only_no_gt_box_v_newtrain0705_ctx1dupjepa.py` 是为了单帧 context 单独加的兼容脚本。
- `wan_stage1b_context_only_no_gt_box_vnewtrain0705_ctx1dupjepa_v2v.py` 是对应的批量版本，复用原批量脚本，只替换单帧 JEPA 相关逻辑。
- `collect_stage1b_metric_table.py` 属于结果汇总，不属于推理脚本，这里不收录。
