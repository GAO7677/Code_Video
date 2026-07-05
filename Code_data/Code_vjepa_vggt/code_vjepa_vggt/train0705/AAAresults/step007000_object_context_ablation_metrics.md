# step-007000 object_context ablation 指标对比

## 基本信息

| 项目 | 值 |
| --- | --- |
| 测试集路径 | `/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt` |
| 方法数量 | `3` |
| 每组样本数 | `17, 17, 17` |

## 指标方向

- `WMReward Surprise`: 越低越好
- `Physics-IQ Approx`: 越高越好
- `VideoPhy2-PC`: 越高越好
- `PhyGround`: 越高越好
- `Cosmos-Reason1`: 越高越好

## 指标表格

说明：加粗表示该指标在这三组里的当前最优值。

| 方法 | 数量 | WMReward Surprise (↓) | Physics-IQ Approx (↑) | VideoPhy2-PC (↑) | PhyGround (↑) | Cosmos-Reason1 (↑) | 结果目录 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 17 | **0.699088** | **54.824118** | 3.647059 | **3.627441** | 2.176471 | `/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705/step-007000` |
| object_context_zero | 17 | 0.700588 | 41.563529 | 3.588235 | 3.411765 | 2.058824 | `/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705_object_context_zero/step-007000` |
| object_context_random | 17 | 0.699565 | 41.011176 | **3.705882** | 3.333329 | **2.352941** | `/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705_object_context_random/step-007000` |
