# train0705 Bench Notes

## 1. 预处理可视化
`no gt box` 跟踪可视化脚本：

`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/inspect_stage1b_prepipe_overlay.py`

## 2. 指标回填
`bench.sh` 会批量回填这些指标：
- `wmreward`
- `physics_iq`
- `physics_iq_with_context`
- `physics_iq_without_context`
- `pmf_with_context`
- `pmf_without_context`
- `videophy2`
- `cosmos_reason1`

脚本行为：
- 每个 metric 结束后读取 `eval_summary_<metric>.json`，输出真实信号：
  - `status=ok`
  - `status=empty`
  - `status=failed`
  - `status=partial`
- 还会打印每个结果文件夹当前的指标进度：`metric=已完成/全部`
- 最终打印 `final_signal=success|empty|failed`
- `status=empty/failed/partial` 时返回非零退出码，不再无条件打印成功

整棵结果树回填：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.sh \
  /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare
```

按数据集分别回填：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.sh \
  /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/morpheus_real_world
```

```bash
CUDA_VISIBLE_DEVICES=0 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.sh \
  /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ
```

单独重跑某一个指标：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py \
  --metric physics_iq_without_context \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/morpheus_real_world \
  --overwrite
```

## 2.1 ti2v / t2v 专用评测
`ti2v/t2v` 现在使用独立脚本：

`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/bench_ti2v_t2v.py`

这版不会再从 `case_json/input_json` 里反推 `input_video/context_video`，只使用结果 JSON 里真实存在的 `input_*` 字段，适合当前 `ti2v=input_image_only` 和 `t2v` 的结果格式。

其中 `cosmos_reason1` 会自动切到 `vphy` 环境执行，不依赖 `wan-cu128` 里的 `transformers` 版本。

默认支持这些指标：
- `wmreward`
- `physics_iq`
- `physics_iq_with_context`
- `pmf_with_context`
- `videophy2`
- `cosmos_reason1`

先对单个目录做 1 个 case smoke：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
CUDA_VISIBLE_DEVICES=0 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/bench_ti2v_t2v.py \
  --metric physics_iq_with_context \
  --result-root /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_morpheus_real_world/openvid_lora_step10000 \
  --limit 1
```

## 2.2 ti2v / t2v 用 gpu0 整批回填所有指标
下面这条命令会按目录依次回填：
- `/data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705`
- `/data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_morpheus_real_world`
- `/data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_physicIQ`
- `/data/gaoya/AAA_test_video/0623/test/t2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_morpheus_real_world`
- `/data/gaoya/AAA_test_video/0623/test/t2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_physicIQ`

默认不加 `--overwrite`，只补缺失指标：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
bash -lc '
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH_PY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/bench_ti2v_t2v.py
METRICS=(
  wmreward
  physics_iq
  physics_iq_with_context
  pmf_with_context
  videophy2
  cosmos_reason1
)
ROOTS=(
  /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705
  /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_morpheus_real_world
  /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_physicIQ
  /data/gaoya/AAA_test_video/0623/test/t2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_morpheus_real_world
  /data/gaoya/AAA_test_video/0623/test/t2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_physicIQ
)
for result_root in "${ROOTS[@]}"; do
  echo "[train0705-ti2v-t2v] start result_root=${result_root}"
  for metric in "${METRICS[@]}"; do
    echo "[train0705-ti2v-t2v] metric=${metric} result_root=${result_root}"
    "${PYTHON_BIN}" "${BENCH_PY}" \
      --metric "${metric}" \
      --result-root "${result_root}"
  done
done
'
```

如需强制重算，把内层命令改成：

```bash
    "${PYTHON_BIN}" "${BENCH_PY}" \
      --metric "${metric}" \
      --result-root "${result_root}" \
      --overwrite
```

## 3. 生成数量达标文件夹汇总
脚本：

`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/summarize_generated_folder_metrics.py`

扫描范围：
- `/data/gaoya/AAA_test_video/0623/test/v2v`
- `/data/gaoya/AAA_test_video/0623/test/ti2v`
- `/data/gaoya/AAA_test_video/0623/test/t2v`

统计数据集：
- `/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt`
- `/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt`

规则：
- 只纳入生成数量等于数据集列表长度的文件夹
- 每个指标输出 `mean` 和 `count`
- 指标未跑满时，`mean=0.0000`，`count` 保留真实完成数
- 均值保留 4 位小数
- CSV 不展示 `pdi`、`proxy*`、`phyground`

运行：

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/summarize_generated_folder_metrics.py
```

输出：
- Markdown: `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/AAAresults/generated_folder_metric_summary.md`
- CSV: `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/AAAresults/generated_folder_metric_summary.csv`

## 4. HTML 指标报告
只统计这两个数据集列表中的 case：
- `morpheus_real_world (121)`
- `physicIQ (67)`

一键回填 + 合并渲染：

```bash
CUDA_VISIBLE_DEVICES=5 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_train0705_formal_compare_report.sh
```

只重渲染报告，跳过指标回填：

```bash
RUN_BENCH=0 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_train0705_formal_compare_report.sh
```

预览合并页面：

```bash
pyport /data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare/combined 8991
```

单独渲染 `morpheus_real_world`：

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/render_v2v_metric_report.py \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/morpheus_real_world \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt \
  --output-dir /data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare/morpheus_real_world
```

```bash
pyport /data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare/morpheus_real_world 8991
```

单独渲染 `physicIQ`：

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/render_v2v_metric_report.py \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
  --output-dir /data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare/physicIQ
```

```bash
pyport /data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare/physicIQ 8992
```
