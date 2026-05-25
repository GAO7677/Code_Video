# Physics-IQ 完整版

## 目标

做一个一键全量入口，跑完 Physics-IQ 全部 198 个 `take-1` case，输出官方口径分数，并映射到本地端口查看结果。

## 推荐入口

```bash
python run_physics_iq_full.py \
  --method my_method \
  --port 18701
```

访问：

```text
http://127.0.0.1:18701
```

## 输入

- 全量 `take-1` case，共 198 个
- `switch-frames`
- `descriptions.csv`
- 方法权重 / 推理配置

## 输出

- `generated_videos/<method>/*.mp4`
- `eval_outputs/results/<method>.csv`
- `physics_iq_method_summary.csv`
- `viewer/index.html`

## 全量链路

建议一键脚本内部串起 5 步：

1. 读取 `descriptions.csv`，固定使用全部 `take-1`
2. 生成 198 个预测视频
3. 调官方评测流程产出 `<method>.csv`
4. 调 `calculate_iq_score(csv)` 输出最终 Physics-IQ score
5. 启本地 viewer，在端口展示 case 列表、视频和总分

## 现有可复用脚本

- `run_physics_iq_methods.sh`
- `wan22_ti2v_physics_iq_eval_multigpu.py`
- `wan22_tv2v_physics_iq_eval_multigpu.py`
- `export_physics_iq_method_summary.py`

## 分数口径

这个版本输出的才应标成：

```text
official full Physics-IQ score
```

前提是：

- 使用全部 198 个 `take-1` case
- 评测视频覆盖完整
- 结果 CSV 无缺项

## 端口页建议字段

- `sample_id`
- `caption`
- `first_frame`
- `generated_video`
- `future_gt_video`
- `per-case metrics`
- `final Physics-IQ score`

## 备注

- 如果只跑部分 case，必须降级为 demo 版口径。
- 完整版适合正式评测、横向对比和最终汇总。
