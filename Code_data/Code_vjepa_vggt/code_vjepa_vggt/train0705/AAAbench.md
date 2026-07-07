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
旧格式 `ti2v/t2v` 结果 JSON 里有一部分没有 `input_json` / `output_video`，不能直接喂给 `bench.py`。专用脚本会先在临时目录下构造一个可评测镜像树，再分别调用 `AAAinfer/bench.sh` 跑 `ti2v` 和 `t2v`。

`ti2v/t2v` 专用评测默认不跑 `_without_context` 指标，只跑：
- `wmreward`
- `physics_iq`
- `physics_iq_with_context`
- `pmf_with_context`
- `videophy2`
- `cosmos_reason1`

适配脚本：

`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/prepare_ti2v_t2v_bench_inputs.py`

总控脚本：

`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_bench_ti2v_t2v.sh`

先对每个子文件夹只跑 1 个 case 的 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
LIMIT_PER_FOLDER=1 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_bench_ti2v_t2v.sh
```

跑完整棵 `ti2v/t2v`：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_bench_ti2v_t2v.sh
```

默认临时输出目录：

`/data/gaoya/agent-data/outputs/train0705_ti2v_t2v_bench_inputs`

如需覆盖 `ti2v/t2v` 专用脚本的指标列表，可以传：

```bash
TI2V_T2V_BENCH_METRICS=wmreward,physics_iq,physics_iq_with_context,pmf_with_context,videophy2,cosmos_reason1 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_bench_ti2v_t2v.sh
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
