# run_infer_xssc_context_slots.sh 推理指令

只需要修改 3 个输入：

```bash
WEIGHTS_ROOT=/path/to/checkpoints/step-xxxx
INPUT_TXT=/path/to/input_json_list.txt
OUTPUT_ROOT=/path/to/output_root
```

其他参数默认保持和下面结果一致：

```text
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_xSSC/formal_mix49_b2_dropout_metrics_20260719T204359Z_step-001000_steps40_512x896_ctx08_49f_defaultnegprompt
```

## 单卡运行

```bash
cd /home/gaoya

WEIGHTS_ROOT=/path/to/checkpoints/step-xxxx
INPUT_TXT=/path/to/input_json_list.txt
OUTPUT_ROOT=/path/to/output_root

TEST_LIST="${INPUT_TXT}" \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots.sh \
  "${WEIGHTS_ROOT}" \
  0 \
  "${OUTPUT_ROOT}"
```

其中 `0` 是物理 GPU id，可按需改成其他 GPU。

## 双卡分片运行

```bash
cd /home/gaoya

WEIGHTS_ROOT=/path/to/checkpoints/step-xxxx
INPUT_TXT=/path/to/input_json_list.txt
OUTPUT_ROOT=/path/to/output_root

METHOD_NAME="$(basename "$(dirname "$(dirname "${WEIGHTS_ROOT}")")")_$(basename "${WEIGHTS_ROOT}")_steps40_512x896_ctx08_49f_defaultnegprompt"
META_ROOT="${OUTPUT_ROOT}/_run_meta/${METHOD_NAME}"
mkdir -p "${META_ROOT}/shards" "${META_ROOT}/logs" "${META_ROOT}/numeric_traces"

awk 'NF && NR%2==1' "${INPUT_TXT}" > "${META_ROOT}/shards/gpu0.txt"
awk 'NF && NR%2==0' "${INPUT_TXT}" > "${META_ROOT}/shards/gpu1.txt"

TEST_LIST="${META_ROOT}/shards/gpu0.txt" \
STEP_OUTPUT_DIR_NAME="${METHOD_NAME}" \
SHARD_TAG=gpu0 \
TRACE_ROOT="${META_ROOT}/numeric_traces/gpu0" \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots.sh \
  "${WEIGHTS_ROOT}" \
  0 \
  "${OUTPUT_ROOT}" \
  > "${META_ROOT}/logs/gpu0.log" 2>&1

TEST_LIST="${META_ROOT}/shards/gpu1.txt" \
STEP_OUTPUT_DIR_NAME="${METHOD_NAME}" \
SHARD_TAG=gpu1 \
TRACE_ROOT="${META_ROOT}/numeric_traces/gpu1" \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots.sh \
  "${WEIGHTS_ROOT}" \
  1 \
  "${OUTPUT_ROOT}" \
  > "${META_ROOT}/logs/gpu1.log" 2>&1
```

其中 `0/1` 是物理 GPU id，可按需改成其他 GPU。

## step-001000 示例

```bash
cd /home/gaoya

WEIGHTS_ROOT=/data/gaoya/agent-data/checkpoints/train_xssc_context_slots/formal_mix49_b2_dropout_metrics_20260719T204359Z/checkpoints/step-001000
INPUT_TXT=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_xSSC
METHOD_NAME=formal_mix49_b2_dropout_metrics_20260719T204359Z_step-001000_steps40_512x896_ctx08_49f_defaultnegprompt

META_ROOT="${OUTPUT_ROOT}/_run_meta/${METHOD_NAME}"
mkdir -p "${META_ROOT}/shards" "${META_ROOT}/logs" "${META_ROOT}/numeric_traces"

awk 'NF && NR%2==1' "${INPUT_TXT}" > "${META_ROOT}/shards/gpu1.txt"
awk 'NF && NR%2==0' "${INPUT_TXT}" > "${META_ROOT}/shards/gpu4.txt"

TEST_LIST="${META_ROOT}/shards/gpu1.txt" \
STEP_OUTPUT_DIR_NAME="${METHOD_NAME}" \
SHARD_TAG=gpu1 \
TRACE_ROOT="${META_ROOT}/numeric_traces/gpu1" \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots.sh \
  "${WEIGHTS_ROOT}" \
  1 \
  "${OUTPUT_ROOT}" \
  > "${META_ROOT}/logs/gpu1.log" 2>&1

TEST_LIST="${META_ROOT}/shards/gpu4.txt" \
STEP_OUTPUT_DIR_NAME="${METHOD_NAME}" \
SHARD_TAG=gpu4 \
TRACE_ROOT="${META_ROOT}/numeric_traces/gpu4" \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_infer_xssc_context_slots.sh \
  "${WEIGHTS_ROOT}" \
  4 \
  "${OUTPUT_ROOT}" \
  > "${META_ROOT}/logs/gpu4.log" 2>&1
```

## 默认推理参数

由 `run_infer_xssc_context_slots.sh` 固定：

```text
height=512
width=896
num_frames=49
context_frames=8
sampling_mode=prefix
num_inference_steps=40
fps=30
```

默认 xSSC：

```text
XSSC_ROOT=/home/gaoya/Code_Video/xSSC-main
XSSC_CONFIG=/home/gaoya/Code_Video/xSSC-main/config-randsfq/rsfq2_r-ytvis.py
XSSC_CHECKPOINT=/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth
```

默认 object condition：

```text
object_context_shape=[1, 56, 3072]
xssc_slots_shape=[1, 8, 7, 256]
```

默认 negative prompt：

```text
色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走
```
