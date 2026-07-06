# no gt box版跟踪可视化
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/inspect_stage1b_prepipe_overlay.py


# train0705 结果整批指标回填

## 整棵结果树批量回填
```bash
CUDA_VISIBLE_DEVICES=0 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.sh \
  /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare
```

## 按数据集分别回填
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

## bench.sh 当前会批量回填的指标
- `wmreward`
- `physics_iq`
- `physics_iq_with_context`
- `physics_iq_without_context`
- `pmf_with_context`
- `pmf_without_context`
- `videophy2`
- `phyground`
- `cosmos_reason1`

## 单独重跑某一个指标
默认会跳过结果 JSON 中已经存在的对应指标字段；如需强制重跑，使用 `bench.py --overwrite`。

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py \
  --metric physics_iq_without_context \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/morpheus_real_world \
  --overwrite
```

## train0705 指标报告可视化
```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/render_v2v_metric_report.py \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare \
  --output-dir /data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare
```

```bash
pyport /data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare 8991
```

如果只想渲染单个数据集，也可以单独指定：

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/render_v2v_metric_report.py \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/morpheus_real_world \
  --output-dir /data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare/morpheus_real_world
```

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/render_v2v_metric_report.py \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ \
  --output-dir /data/gaoya/AAA_test_video/0623/test/report/v2v/train0705_formal_compare/physicIQ
```
