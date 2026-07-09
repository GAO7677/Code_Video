# train0705_kubric_no_gt_box 推理说明

下面只整理 Kubric no-GT-box 这套推理相关脚本，统一推荐使用一个 shell 入口来控制：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

对应的 Python 批量推理脚本是：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py
```

## 1. 核心业务参数

统一入口脚本优先只需要关心下面这些参数：

- `GPU_PAIR=6,7`
  单个 GPU pair，适合单次跑一个 `ctx`，或者在同一对卡上顺序 sweep 多个 `ctx`
- `GPU_PAIRS="6,7 7,6 3,5 5,3"`
  多个 GPU pair，适合把不同 `ctx` 分配到多个 worker 并行跑
- `TEST_JSON_TXT=/data/.../test_5.txt`
  待测样本 txt，每行一个输入 json 路径
- `WEIGHTS_ROOT=/data/.../checkpoints/step-001000`
  推理权重目录
- `METHOD_NAME=train_stage1b_kubric0708_step1000`
  方法名 / model name
- `OUTPUT_ROOT=/data/...`
  输出根目录
- `OUTPUT_FRAMES=49`
  最终输出帧数
- `CTX=8`
  单次只跑一个 context 长度
- `CTX=1,4,8,12,16,20`
  一次跑多组 context 长度
- 旧参数 `CTX_NUM / CTX_NUMS / CONTEXT_FRAMES / CONTEXT_FRAME_VALUES` 仍兼容，但不再推荐

## 2. 默认处理逻辑

- 输入 context video 来自 `input json` 记录的 `source_video`
- 对于每个 case，会取 `source_video` 的前 `ctx` 帧作为实际输入 context
- 实际送进 `pipe` 的所有 context 帧会拼接成一张同名 jpg
- 输出 `json` 的 `input_video` 字段会指向这张 jpg
- 输出 `json` 里会额外保留原始 `source_video`
- 当前默认输入预处理是：
  1. 按比例缩放到至少覆盖 `832x480`
  2. 中心裁剪到 `832x480`
  3. 再缩放到模型输入分辨率 `512x896`
- 当前默认会加：
  - `PYTHONNOUSERSITE=1`
  - 这样可以避免被 `~/.local/lib/python3.10/site-packages` 里的旧版 `huggingface_hub` 元数据污染
- 当前禁止使用 `gpu4`
- 如果某个并行 worker 触发 OOM，该 worker 会直接退出，不继续占用 GPU；其他 worker 继续跑
- `method` 会自动追加后缀：
  - `_ctx{xx}_{yy}f`
  - 例如：`train_stage1b_kubric0708_step-001000_ctx08_49f`

## 3. 统一脚本单次推理

适合：

- 指定一对 GPU
- 指定一个 `ctx`
- 跑一整份 txt

```bash
GPU_PAIR=0,0 \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
METHOD_NAME=train_stage1b_diffsynth_native0705_step2500 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0705 \
OUTPUT_FRAMES=49 \
CTX=8 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh




GPU_PAIR="5,5" \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-003500 \
METHOD_NAME=train_stage1b_kubric0708with_step3500 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_kubric0708 \
OUTPUT_FRAMES=49 \
CTX=8 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh



GPU_PAIR="0,0" \
TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt \
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-002000 \
METHOD_NAME=train_stage1b_kubric0708_step2000 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/morpheus_real_world/train_stage1b_kubric0708 \
OUTPUT_FRAMES=49 \
CTX=8 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh











```

## 3.1 按 GPU 数量自动分 shard 的并行脚本

如果想把一个 `txt` 按 GPU 数量自动均分，再用多张单卡并行启动，可以用：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_parallel_infer_from_txt.sh
```

核心特性：

- `--gpus 0,2,3`
  传几张卡，就自动切几份 shard
- 会把 `txt` 中的 case 按 round-robin 均分到各 shard
- 每个 shard 用一张单卡启动一个 worker
- 默认 `negative_prompt=DEFAULT_NEGATIVE_PROMPT`
- 可以通过命令行覆盖 `ctx / output_frames / negative_prompt`

运行示例：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_parallel_infer_from_txt.sh \
  --input-txt /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-007000 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_diffsynth_native0705_0705 \
  --method-name train_stage1b_diffsynth_native0705_step7000 \
  --gpus 5,6,7 \
  --ctx 8 \
  --output-frames 49 \
  --negative-prompt default
```

其中：

- `--negative-prompt default`
  表示使用脚本内部的 `DEFAULT_NEGATIVE_PROMPT`
- `--negative-prompt empty`
  表示显式传空串 `""`
- `--negative-prompt "some text"`
  表示使用自定义 negative prompt

分片文件会落在：

- `/data/gaoya/agent-data/outputs/<txt_stem>_<step_name>_ctx<ctx>_gpus<gpu_tag>_shards`

如果要明确写出输入预处理参数，也可以加：

```bash
INPUT_COVER_CROP_WIDTH=832 \
INPUT_COVER_CROP_HEIGHT=480
```

## 4. 单个 GPU pair 顺序跑多组 ctx

适合：

- 同一对卡顺序跑多个 `ctx`
- 每个 `ctx` 输出落到单独目录 `ctx01/ctx04/...`

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

## 5. 多个 GPU pair 并行跑多组 ctx

适合：

- 把不同 `ctx` 分给多个 worker 并行跑
- 脚本会按 round-robin 分配 `ctx`

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

按上面这个例子，会分成：

- `6,7` -> `ctx01`, `ctx16`
- `7,6` -> `ctx04`, `ctx20`
- `3,5` -> `ctx08`
- `5,3` -> `ctx12`

注意：

- `6,7` 和 `7,6` 是同一对物理 GPU，只是主卡 / 辅卡顺序互换
- `3,5` 和 `5,3` 也是同理
- 如果多个 worker 共享同一张物理 GPU，仍然可能因为显存竞争触发 OOM

## 6. 旧命令兼容

历史上的 ctx sweep 脚本仍然可以继续用：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_batch_ctx_sweep_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.sh
```

但它现在只是一个兼容包装层，内部会直接转调统一入口脚本：

```text
run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
```

## 7. 底层 Python 脚本直调

如果不想走统一 shell，也可以直接调 Python：

单卡：

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step1000 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708 \
  --height 512 \
  --width 896 \
  --input-cover-crop-width 832 \
  --input-cover-crop-height 480 \
  --context-frames 8 \
  --num-inference-steps 40 \
  --output-num-frames 49
```

双卡：

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=0,1 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \

## 8. Physics-IQ Verified 正式跑法

如果要严格按官方 `Physics-IQ Verified` workflow 跑 `train0705 native` 权重，不要走 Kubric 入口，使用下面这个专用包装器：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.py
```

这个脚本会：

- 直接读取官方 `descriptions_base.csv`
- 只取 `take-1` 的 198 个 case
- 直接读取官方 Verified `conditioning` 视频
- 生成阶段自动对齐到底层 Wan 的 `num_frames % 4 == 1`
- 生成后自动裁成官方要求的精确 `5.0s`
- 最终生成结果保留在原生目录 `step-002500/`，不再额外改名
- 已有 `mp4 + json` 的样本会自动跳过，只有未完成样本继续跑

本次正式跑 `step-002500` 到 `/data/gaoya/AAA_test_video/0623/test/physicsiq` 的命令如下：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
  --model-name train_stage1b_diffsynth_native0705_step2500_physiq_verified \
  --output-root /data/gaoya/AAA_test_video/0623/test/physicsiq \
  --verified-root /data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified \
  --descriptions-file /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv \
  --fps 30 \
  --num-frames 150 \
  --context-frames 20 \
  --sampling-mode prefix \
  --num-inference-steps 40 \
  --cfg-scale 5.0 \
  --seed 42 \
  --device cuda
```

说明：

- 这里显式用了 `CUDA_VISIBLE_DEVICES=2`，避开 `gpu4`，也避免和当前占用较高的 `gpu0/1/6` 冲突
- 生成产物会直接续写到：
  `/data/gaoya/AAA_test_video/0623/test/physicsiq/step-002500`
- wrapper 里的 `run-name` 现在只用于 `_physics_iq_inputs/...` 的准备目录命名，不影响最终输出目录
- 如果后续要补 leaderboard 的多次 run，只需要换不同 `--seed`；如果只是为了区分准备目录，再额外换不同 `--run-name`
