# Code_try0526 测试数据集整理

说明：这一份只看“测试数据集”，不按指标展开。

## 1. A 组：PDI-Bench 生成视频

- 含义：PDI-Bench 的 GT / Wan / VACE 输出视频，用于方法级比较
- 数据位置：
  - `/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output`
- 常用指标：
  - PDI-Bench 官方分数
  - WMReward
  - Geometry Proxy
  - VideoPhy-2
  - Cosmos-Reason1
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/evaluate_pdi_benchmark_methods.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/run_pdi_official_eval.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/eval_benchmark_dir_metrics.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python eval_benchmark_dir_metrics.py \
  --input-root /data/gaoya/AAA_test_video/Output_try0526/PDI-Bench \
  --metrics pdi wmreward proxy videophy2 cosmos
```

## 2. B1 组：Ball-Block 物理参数集

- 含义：固定外观，只改恢复系数、摩擦、球质量
- 数据位置：
  - `/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block`
- 常用指标：
  - PDI
  - JEPA
  - WMReward
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_ball_block.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/wmreward_batch.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_physv_groups.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim
python eval_ball_block.py --gpu 0 --port 18703
```

## 3. B2 组：JEPA 运动敏感性集

- 含义：固定外观，系统改变速度、质量、重力、碰撞与方向
- 数据位置：
  - `/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/jepa_sensitivity`
- 常用指标：
  - WMReward
  - VideoPhy-2
  - PhyGround
  - Cosmos-Reason1
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_physv_groups.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/pipeline.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python physics_sim/eval_physv_groups.py \
  --groups B2 \
  --metrics wmreward videophy2 phyground cosmos
```

## 4. B3 组：外观敏感性集

- 含义：同一物理轨迹，只改渲染外观与光照
- 数据位置：
  - `/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block_appearance`
- 常用指标：
  - VideoPhy-2
  - PhyGround
  - Cosmos-Reason1
  - WMReward
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_physv_groups.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/pipeline.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python physics_sim/eval_physv_groups.py \
  --groups B3 \
  --metrics wmreward videophy2 phyground cosmos
```

## 5. C 组：帧序打乱 Sanity Check

- 含义：只破坏时序，不改单帧内容，用来检查时序敏感性
- 数据位置：
  - `/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/shuffle_test`
- 常用指标：
  - VideoPhy-2
  - PhyGround
  - Cosmos-Reason1
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_physv_groups.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/pipeline.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python physics_sim/eval_physv_groups.py \
  --groups C \
  --metrics videophy2 phyground cosmos
```

## 6. PhyGround 自带视频目录

- 含义：如果单独跑 PhyGround 官方脚本，通常是一个 `video_dir + prompts_json` 配套目录
- 相关脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/PhyGround/evals/vlm_eval.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526/PhyGround
python -m evals.vlm_eval \
  --backend qwen9b \
  --video_dir ./videos \
  --prompts_json ./data/prompts/phyground.json \
  --save_path ./scores.json
```
