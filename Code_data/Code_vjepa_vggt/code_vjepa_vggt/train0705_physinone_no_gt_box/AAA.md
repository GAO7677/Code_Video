# PhysInOne Stage1b 训练 / 推理运行指令

## 1. Smoke Train

使用 `gpu3,5,6,7` 跑一次 `stage1b` 的 smoke train，保存 `step-000001` checkpoint：

```bash
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_physinone_no_gt_box/run_smoke_stage1b_context_only_no_gt_box_v_newtrain_physinone_gpu3567.sh
```

对应脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_physinone_no_gt_box/run_smoke_stage1b_context_only_no_gt_box_v_newtrain_physinone_gpu3567.sh
```

默认输出目录：

```text
/data/gaoya/agent-data/checkpoints/stage1b_physinone_no_gt_box_smoke_gpu3567
```

本次实际产出的 smoke checkpoint：

```text
/data/gaoya/agent-data/checkpoints/stage1b_physinone_no_gt_box_smoke_gpu3567/checkpoints/step-000001
```

其中包含：

```text
/data/gaoya/agent-data/checkpoints/stage1b_physinone_no_gt_box_smoke_gpu3567/checkpoints/step-000001/checkpoint.safetensors
/data/gaoya/agent-data/checkpoints/stage1b_physinone_no_gt_box_smoke_gpu3567/checkpoints/step-000001/training_state.pt
```


## 2. 批量推理

对 `/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt` 中的 case 进行批量推理。  
实际执行时使用 batch wrapper，一次加载模型，避免每个 case 反复重载。

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py \
  --weights-root /data/gaoya/agent-data/checkpoints/stage1b_physinone_no_gt_box_smoke_gpu3567/checkpoints/step-000001 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name phisinone_stage1b_smoke_step000001 \
  --output-root /data/gaoya/agent-data/outputs/tmp_stage1b_physinone_smoke_test5_batch \
  --num-inference-steps 12 \
  --device cuda \
  --force
```

推理脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py
```

输入列表：

```text
/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt
```

输出目录：

```text
/data/gaoya/agent-data/outputs/tmp_stage1b_physinone_smoke_test5_batch
```

本次实际结果目录：

```text
/data/gaoya/agent-data/outputs/tmp_stage1b_physinone_smoke_test5_batch/step-000001
```

结果汇总：

```text
/data/gaoya/agent-data/outputs/tmp_stage1b_physinone_smoke_test5_batch/step-000001/result.json
```


## 3. 逐条调用 infer_stage1b 的封装脚本

如果必须显式走 `infer_stage1b_context_only_no_gt_box_v_newtrain0705.py`，可用下面这个封装脚本。  
注意：它会对每个 case 单独拉起一次 Python 进程并重载模型，批量跑时明显更慢。

脚本路径：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_infer_stage1b_context_only_no_gt_box_v_newtrain0705_from_json_list.sh
```

示例命令：

```bash
CHECKPOINT=/data/gaoya/agent-data/checkpoints/stage1b_physinone_no_gt_box_smoke_gpu3567/checkpoints/step-000001 \
INPUT_JSON_LIST_PATH=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/tmp_stage1b_physinone_smoke_test5_direct \
GPU=7 \
SAMPLING_STEPS=12 \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_infer_stage1b_context_only_no_gt_box_v_newtrain0705_from_json_list.sh
```


## 4. 备注

- 禁止使用 `gpu4`。
- 这次 `test_5.txt` 实际共有 `18` 条记录，其中 `1` 条重复，因此最终唯一输出视频数为 `17`。
- 本次批量推理统计结果为：`num_total=18`、`num_success=18`、`num_failed=0`、`num_skipped=0`。
