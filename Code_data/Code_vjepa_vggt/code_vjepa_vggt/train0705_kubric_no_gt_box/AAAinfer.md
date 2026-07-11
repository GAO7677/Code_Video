# train0705_kubric_no_gt_box 推理说明

这份文档只整理当前这套 Kubric no-GT-box 推理相关脚本，并以当前仓库里的实际脚本行为为准。优先推荐使用统一 shell 入口，不再推荐手写旧命令拼装。

## 1. 推荐入口

当前推荐入口按用途分 4 类：

1. 统一 batch 推理入口

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

2. 按 txt 自动分 shard、多单卡并行入口

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_parallel_infer_from_txt.sh
```

3. 底层 Python 批量推理脚本

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py
```

4. Physics-IQ Verified 专用 wrapper

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/run_physics_iq_verified_kubric_v2v.py
```

对应 shell wrapper：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/run_physics_iq_verified_kubric_v2v.sh
```

## 2. 统一 shell 入口的实际默认值

统一入口脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

当前脚本里的实际默认值和规则如下：

- 默认 Python 环境：

```text
/home/gaoya/miniconda3/envs/wan-cu128/bin/python
```

- 默认分辨率：
  - `HEIGHT=512`
  - `WIDTH=896`

- 默认输入 cover-crop 尺寸：
  - `INPUT_COVER_CROP_HEIGHT=512`
  - `INPUT_COVER_CROP_WIDTH=896`

- 默认输出帧数：
  - `OUTPUT_FRAMES=49`

- 默认单次 context 帧数：
  - `CTX=8`

- 默认多组 context 列表：

```text
1,2,3,4,6,8,9,12,16,20
```

- 默认采样模式：
  - `SAMPLING_MODE=prefix`

- 默认推理步数：
  - `NUM_INFERENCE_STEPS=40`

- 默认 `cfg scale`：
  - `CFG_SCALE=5.0`

- 默认 `fps`：
  - `FPS=30`

- 默认 `seed`：
  - `SEED=42`

- 默认会注入：
  - `PYTHONNOUSERSITE=1`

- 禁止使用 `gpu4`

- `DISABLE_OBJECT_BRANCH=1` 时，统一入口会切换到底下这个 no-object-branch 版本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_no_object_branch_v2v.py
```

## 3. 统一 shell 入口支持的核心环境变量

最常用的是这些：

- `GPU_PAIR=0,0`
  单次 direct 模式。可以是单卡，也可以是双卡。

- `GPU_PAIRS="1,2 3,5 6,7"`
  sweep 模式。多个 worker 并行跑不同 ctx。

- `TEST_JSON_TXT=/data/.../test_5.txt`
  输入 txt，每行一个 json 路径。

- `WEIGHTS_ROOT=/data/.../checkpoints/step-001000`
  权重目录，要求是 `step-*` 目录。

- `METHOD_NAME=...`
  传给底层 Python 的 `--model-name`。它会参与输出根目录推断，但不是唯一决定最终结果叶子目录名的参数。

- `OUTPUT_ROOT=/data/...`
  结果输出根目录。

- `OUTPUT_FRAMES=49`
  最终输出帧数。

- `CTX=8`
  只跑一个 context 长度。

- `CTX=1,4,8,12,16,20`
  顺序或并行 sweep 多组 context。

- `DISABLE_OBJECT_BRANCH=1`
  使用 no-object-branch 版本。

- `NEGATIVE_PROMPT=...`
  覆盖默认 negative prompt。

- `FORCE=1`
  透传 `--force`。

- `OVERWRITE=1`
  透传 `--overwrite`。

- `LIMIT=10`
  只跑前 10 个样本。

旧变量仍兼容，但不再推荐优先使用：

- `CTX_NUM`
- `CTX_NUMS`
- `CONTEXT_FRAMES`
- `CONTEXT_FRAME_VALUES`
- `VISIBLE_GPU_IDS`
- `INFERENCE_GPU_PAIRS`

## 4. direct 模式和 sweep 模式的区别

### 4.1 direct 模式

典型触发方式：

- 只设置 `GPU_PAIR`
- `CTX` 是单个数字

例如：

```bash
GPU_PAIR=0,0 \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
METHOD_NAME=train_stage1b_kubric0708_step1000 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708 \
OUTPUT_FRAMES=49 \
CTX=8 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

这个模式下，统一入口会向底层 Python 传：

```text
--step-output-dir-name __METHOD_NAME__
```

所以最终结果叶子目录不是简单的 `step-001000`，而是底层 Python 自动解析出来的完整 method 名，通常类似：

```text
train_stage1b_kubric0708_step-001000_steps40_512x896_ctx08_49f_defaultnegprompt
```

### 4.2 sweep 模式

典型触发方式：

- 设置 `GPU_PAIRS`
- 或 `CTX` 是逗号分隔列表

例如：

```bash
GPU_PAIR=6,7 \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
METHOD_NAME=train_stage1b_kubric0708_step1000 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
OUTPUT_FRAMES=49 \
CTX=1,4,8,12,16,20 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

这个模式下，每个 ctx 的输出根目录会先分到：

```text
${OUTPUT_ROOT}/ctx01
${OUTPUT_ROOT}/ctx04
${OUTPUT_ROOT}/ctx08
...
```

而底层 Python 没有再显式传 `--step-output-dir-name`，所以最终叶子目录默认会是：

```text
step-001000
```

也就是典型结果结构会像：

```text
OUTPUT_ROOT/
  ctx01/step-001000/
  ctx04/step-001000/
  ctx08/step-001000/
```

这一点和 direct 模式不同，旧文档里这里写混了。

## 5. 统一 shell 入口的已核对示例

### 5.1 单次单卡 direct 推理

`GPU_PAIR=0,0` 是合法的。脚本内部会去重 `CUDA_VISIBLE_DEVICES`，然后把 `inference_devices` 设为 `none`，因此会走单卡布局。

```bash
GPU_PAIR=0,0 \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
METHOD_NAME=train_stage1b_kubric0708_step1000 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708 \
OUTPUT_FRAMES=49 \
CTX=8 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

### 5.2 单次单卡跑 physicIQ json 列表

```bash
GPU_PAIR=3,3 \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-004500 \
METHOD_NAME=train_stage1b_kubric0708_step4500 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_kubric0708 \
OUTPUT_FRAMES=49 \
CTX=8 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh



GPU_PAIR=3,3 \

GPU_PAIR="7,7" \
AUTO_SPLIT_INPUT=1 \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708_stability_v2_resume3500_20260711T042047Z/checkpoints/step-004000 \
METHOD_NAME=train_stage1b_kubric0708_stability_v2_step4000 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_kubric0708 \
OUTPUT_FRAMES=49 \
CTX=8 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh


GPU_PAIR=7,7 \
TEST_JSON_TXT=/data/gaoya/agent-data/outputs/query_prior_compare_20260710/physicIQ_026_mask_vs_boxuniform/ablllllll/_single_case_input_json.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708_stability_v2_resume3500_20260711T042047Z/checkpoints/step-004000 \
METHOD_NAME=train_stage1b_kubric0708_stability_v2_step4000 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_kubric0708 \
OUTPUT_FRAMES=49 \
CTX=8 \
NUM_INFERENCE_STEPS=40 \
CFG_SCALE=5.0 \
SEED=42 \
OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.15 \
OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
  
```

### 5.3 单个 GPU pair 顺序 sweep 多组 ctx

```bash
GPU_PAIR=6,7 \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
METHOD_NAME=train_stage1b_kubric0708_step1000 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
OUTPUT_FRAMES=49 \
CTX=1,4,8,12,16,20 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

### 5.4 多个 GPU pair 并行 sweep 多组 ctx

```bash
GPU_PAIRS="6,7 7,6 3,5 5,3" \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
METHOD_NAME=train_stage1b_kubric0708_step1000 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
OUTPUT_FRAMES=49 \
CTX=1,4,8,12,16,20 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

这个例子里会按 round-robin 把 ctx 分配给各个 worker。

注意：

- `6,7` 和 `7,6` 是同一对物理卡，只有主卡/辅卡顺序不同
- `3,5` 和 `5,3` 也是同理
- 如果多个 worker 实际共享物理 GPU，仍可能因为显存竞争 OOM
- `gpu4` 故障，不能写进任何 `GPU_PAIR` 或 `GPU_PAIRS`

### 5.5 no-object-branch 版本

```bash
GPU_PAIR=0,0 \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
METHOD_NAME=train_stage1b_kubric0708_step1000_no_object_branch \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_no_object_branch \
OUTPUT_FRAMES=49 \
CTX=8 \
DISABLE_OBJECT_BRANCH=1 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

## 6. 按 txt 自动分 shard 的并行脚本

脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_parallel_infer_from_txt.sh
```

适合场景：

- 只有一批 txt 想均分到多张单卡
- 不想自己手动切 shard
- 每个 shard 单独起一个单卡 worker

已核对命令格式：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_parallel_infer_from_txt.sh \
  --input-txt /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_parallel \
  --method-name train_stage1b_kubric0708_step1000 \
  --gpus 0,1,2,3 \
  --ctx 8 \
  --output-frames 49
```

这个脚本的特点：

- 会把输入 txt 均分成多个 `shard_*.txt`
- shard 临时文件放在：

```text
/data/gaoya/agent-data/outputs
```

- 每个 worker 底层还是调用统一 shell 入口
- `--gpus` 里也不能包含 `4`

## 7. 底层 Python 直调

底层脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py
```

当前已核对的关键参数包括：

- `--weights-root`
- `--input-json-list-path`
- `--model-name`
- `--output-root`
- `--step-output-dir-name`
- `--device`
- `--aux-device`
- `--inference-devices`
- `--height`
- `--width`
- `--input-cover-crop-height`
- `--input-cover-crop-width`
- `--num-frames`
- `--output-num-frames`
- `--context-frames`
- `--sampling-mode`
- `--num-inference-steps`
- `--cfg-scale`
- `--negative-prompt`
- `--disable-object-branch`

### 7.1 单卡直调示例

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step1000 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_py \
  --step-output-dir-name __METHOD_NAME__ \
  --height 512 \
  --width 896 \
  --input-cover-crop-height 512 \
  --input-cover-crop-width 896 \
  --context-frames 8 \
  --sampling-mode prefix \
  --num-inference-steps 40 \
  --cfg-scale 5.0 \
  --seed 42 \
  --output-num-frames 49 \
  --device cuda
```

### 7.2 双设备直调示例

这里 `CUDA_VISIBLE_DEVICES=0,1`，脚本内部看到的是逻辑设备 `cuda:0,cuda:1`。

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=0,1 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step1000 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_py2 \
  --step-output-dir-name __METHOD_NAME__ \
  --height 512 \
  --width 896 \
  --input-cover-crop-height 512 \
  --input-cover-crop-width 896 \
  --context-frames 8 \
  --sampling-mode prefix \
  --num-inference-steps 40 \
  --cfg-scale 5.0 \
  --seed 42 \
  --output-num-frames 49 \
  --device cuda \
  --inference-devices cuda:0,cuda:1
```

## 8. Physics-IQ Verified 正式跑法

这里不要再用旧的 `train0705/run_physics_iq_verified_vnewtrain0705_v2v.py`。当前 Kubric 版本的实际 wrapper 是：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/run_physics_iq_verified_kubric_v2v.py
```

对应 shell wrapper：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/run_physics_iq_verified_kubric_v2v.sh
```

这个 wrapper 当前实际行为：

1. 读取官方 `descriptions_base.csv`
2. 只取 `take-1` 的 198 个 case
3. 读取官方 Verified conditioning videos
4. 先生成 Physics-IQ 专用 json/list 输入
5. 调用 Kubric batch infer 主脚本
6. 为了满足底层 Wan 的 `num_frames % 4 == 1` 约束，必要时会先把生成帧数对齐到最近合法值
7. 生成结束后自动裁回精确 5.0 秒

重要输出路径规则：

- 最终正式结果目录：

```text
${OUTPUT_ROOT}/${RUN_NAME}
```

- 准备阶段的临时输入目录：

```text
${OUTPUT_ROOT}/_physics_iq_inputs/${RUN_NAME}
```

- 如果不加 `--keep-prepared-inputs`，推理完成后临时输入目录会被自动清理

### 8.1 直接跑 shell wrapper

这个 shell wrapper 默认就是双设备布局，最稳妥的写法是显式给两张卡：

```bash
CUDA_VISIBLE_DEVICES_VALUE=5,6 \
INFERENCE_DEVICES=cuda:0,cuda:1 \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-004500 \
MODEL_NAME=train_stage1b_kubric0708_step4500_physiq_verified \
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/physics_iq_verified_v2v \
RUN_NAME=train_stage1b_kubric0708_step4500_physiq_verified-bpp-run_01 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/run_physics_iq_verified_kubric_v2v.sh
```

说明：

- `CUDA_VISIBLE_DEVICES_VALUE=5,6` 表示给 wrapper 两张物理卡
- `INFERENCE_DEVICES=cuda:0,cuda:1` 是相对于 `CUDA_VISIBLE_DEVICES` 重映射后的逻辑设备
- 如果只想单卡跑，不建议走这个 shell wrapper，直接用下面的 Python wrapper 更明确

### 8.2 直接跑 Python wrapper

这是更明确、也更不容易被环境变量误伤的写法：

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/run_physics_iq_verified_kubric_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-004500 \
  --model-name train_stage1b_kubric0708_step4500_physiq_verified \
  --output-root /data/gaoya/agent-data/outputs/physics_iq_verified_v2v \
  --verified-root /data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified \
  --descriptions-file /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv \
  --run-name train_stage1b_kubric0708_step4500_physiq_verified-bpp-run_01 \
  --fps 30 \
  --num-frames 150 \
  --context-frames 20 \
  --sampling-mode prefix \
  --num-inference-steps 40 \
  --cfg-scale 5.0 \
  --seed 42 \
  --device cuda
```

如果要双设备布局，再显式补上：

```text
--inference-devices cuda:0,cuda:1
```

## 9. physicIQ formal compare 可视化一键脚本

如果要把下面整个目录：

```text
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ
```

统一做成可视化页面，并同时生成：

- `physicIQ` 根入口页
- 每个结果叶子目录自己的 grouped gallery
- 跨所有结果目录、按同一 case 聚合的超大总对比页

可以直接运行下面这个一键脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/run_physiciq_compare_portal.sh
```

直接执行：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/run_physiciq_compare_portal.sh
```

这个脚本会自动完成：

- 扫描 `physicIQ` 目录下所有带 `*_input_ctx*.jpg` 的叶子结果目录
- 为每个叶子目录生成一个 `/_case_grouped_gallery/index.html`
- 生成 `physicIQ/index.html` 根入口页
- 生成 `physicIQ/_global_case_compare_gallery/index.html` 超大总对比页
- 最后以前台方式启动本地静态服务

默认前台启动命令：

```bash
python3 -m http.server 8011 --bind 127.0.0.1
```

默认访问地址：

- 根入口页：

```text
http://127.0.0.1:8011/
```

- 超大总对比页：

```text
http://127.0.0.1:8011/_global_case_compare_gallery/
```

关键输出文件：

- 根入口页：

```text
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/index.html
```

- 超大总对比页：

```text
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/_global_case_compare_gallery/index.html
```

如果要改端口或根目录，可以这样执行：

```bash
ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ \
PORT=8022 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/run_physiciq_compare_portal.sh
```

## 10. 核对结论

这次已明确修正的旧文档问题包括：

- 旧文档里把默认输入 cover-crop 写成了 `832x480`，实际当前统一入口默认是 `896x512`
- 旧文档里 Physics-IQ 指向了旧的 `train0705/...` 脚本路径，当前应使用 `code_phys_papers_compare/.../run_physics_iq_verified_kubric_v2v.py`
- 旧文档里的双卡 Python 示例是不完整命令，已经替换成完整可执行版本
- 旧文档没有区分 direct 和 sweep 两种输出目录规则，这里已经按当前脚本真实行为拆开说明

## 11. object branch 彩噪保护

当前 grounding 会额外去除“同 phrase、近中心、高 containment”的嵌套重复框。
这用于处理 GDINO 将同一个物体检测成两个不同大小 proposal 的情况；不要通过提高
`aux_max_objects` 来规避重复 proposal，因为那会保留更多背景或重复条件。

对旧 checkpoint 做正式推理时，建议保留全层 ratio guard，并显式启用异常自动回退：

```bash
GPU_PAIR=7,7 \
TEST_JSON_TXT=/data/gaoya/agent-data/outputs/query_prior_compare_20260710/physicIQ_026_mask_vs_boxuniform/ablllllll/_single_case_input_json.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708_stability_v2_resume3500_20260711T042047Z/checkpoints/step-004000 \
METHOD_NAME=train_stage1b_kubric0708_stability_v2_step4000_dedupe_fallback \
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/query_prior_compare_20260711/stability_v2_step4000_dedupe_fallback \
OUTPUT_FRAMES=49 \
CTX=8 \
NUM_INFERENCE_STEPS=40 \
CFG_SCALE=5.0 \
SEED=42 \
OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.15 \
OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
OBJECT_BRANCH_AUTO_FALLBACK_MAX_ACTIVE_SLOTS=3 \
OBJECT_BRANCH_AUTO_FALLBACK_TRIGGER_COUNT=5 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

自动回退仅在有效对象数大于 3、并且 object residual guard 连续触发时启用。
初次前向达到触发阈值后会立即中止，并使用 grounding 排名前 3 的 slot、相同 seed
重新推理。正常的 1-3 物体 case 不会重跑。

从 stability-v3 开始训练的 checkpoint，推理时还应增加：

```text
COMPACT_OBJECT_CONTEXT_SLOTS=1
OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0
```
