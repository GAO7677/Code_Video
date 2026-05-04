# 1. 基于 Wan 2.2 5B TI2V 训练 V2V

当前文档以以下代码为准：

- 训练入口：[run_train.sh](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_train.sh:1)
- 训练主逻辑：[train.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/train.py:1)
- 数据封装：[dataset.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/dataset.py:1)
- 推理/benchmark：[batch_eval_lora.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/batch_eval_lora.py:1)
- validation + VBench：[run_validation_vbench.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_validation_vbench.py:1)

## 1.1 训练目标

- 模型：`Wan 2.2 TI2V 5B`
- 任务：多帧 context 条件下的 V2V / context-aware video generation
- 训练形式：LoRA 微调

## 1.2 训练数据

训练入口走混合数据配置：

- 配置文件：[dataset_mix_config.json](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/dataset_mix_config.json:1)
- 数据集封装：
  - `OpenVidParquetDataset`
  - `MoviDTFRecordDataset`
  - `GenesisRigidDataset`

当前混合来源与配比：

- OpenVid：`repeat=1`
- MOVI-D train：`repeat=2`
- Genesis rigid train：`repeat=2`

当前运行时样本统计，以最新训练日志为准：

- OpenVid：`65975` 条，effective `65975`
- MOVI-D train：`3083` 条，effective `6166`
- Genesis rigid train：`2438` 条，effective `4876`
- 总 effective 样本数：`77017`

对应原始属性：

- OpenVid：原始分辨率约 `1280x720`，原始帧数 `variable`
- MOVI-D：原始分辨率 `256x256`，原始帧数 `24`
- Genesis rigid：原始分辨率 `960x720`，原始帧数 `13` 或 `16`

预处理策略：

- 统一目标分辨率：`384x672`
- OpenVid：`crop + resize`
- Genesis rigid：`crop + resize`
- MOVI-D：保持宽高比后 `pad to canvas` 到 `384x672`

训练帧数策略：

- 全局目标帧数：`24`
- OpenVid：要求原视频至少 `24` 帧，然后在视频中随机截取连续 `24` 帧
- MOVI-D：直接取 `24` 帧
- Genesis rigid：不补帧，保留原始 `13/16` 帧，训练侧支持 variable `T`

Genesis 的 held-out 规则：

- held-out pool 数量：`8` 个 object-level group
- held-out 测试样本数：`205`
- train 样本数：当前为 `2438`
- 训练不会看到 held-out 对象

## 1.3 Context 训练策略

当前训练使用 `mixed_modes` context 采样，见 [train.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/train.py:289)。

采样概率：

- `prefix`：`55%`
- `first_frame`：`20%`
- `sparse`：`15%`
- `random`：`5%`
- `text_only`：`5%`

参考前缀集合定义在 `49` 帧基准上：

- `context_reference_prefixes = {1, 4, 8, 12, 16}`

在当前 `24` 帧训练下，按 `49 -> 24` 等比例向上取整后，实际集合为：

- `prefix` 可选帧数：`{1, 2, 4, 6, 8}`
- `sparse` 可选帧数：`{2, 4, 6, 8}`
- `random` 可选帧数：`{2, 4, 6, 8}`

各模式含义：

- `prefix`：取前 `K` 帧
- `first_frame`：只取第 `0` 帧
- `sparse`：在整段视频中均匀铺开采样，尽量覆盖尾帧
- `random`：一定包含第 `0` 帧，再随机补若干帧
- `text_only`：不提供 visual context

训练核心逻辑见 [context_wan.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/context_wan.py:111)：

- context 帧对应 latent 保持 clean
- non-context latent 加噪
- loss 只计算 non-context latent

为了适配 Wan 的时间维约束，代码里额外做了保护：

- Wan 内部要求 `num_frames` 满足 `4n+1`
- 训练请求 `24` 帧时，内部会对齐到 `25`
- 如果 raw-frame context 映射到 latent 后覆盖全部时间 latent，会自动保留至少 `1` 个 latent 用于监督，避免训练直接报错

## 1.4 当前训练配置

启动脚本：[run_train.sh](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_train.sh:1)

主配置：

- GPU：`4 x 4090`
- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- per-GPU batch：`1`
- `gradient_accumulation_steps=4`
- effective batch：`16`
- 分辨率：`384x672`
- 目标帧数：`24`
- `learning_rate=1e-4`
- `weight_decay=0.01`
- `max_train_steps=8000`
- `num_epochs=10`
- `save_steps=1000`
- `dataset_num_workers=0`
- `wandb_mode=offline`

LoRA 配置：

- `lora_base_model=dit`
- `lora_target_modules=q,k,v,o,ffn.0,ffn.2`
- `lora_rank=32`

额外条件输入：

- `extra_inputs=input_image`
- 也就是非 `text_only` 情况下会额外给首帧图像条件

输出目录：

- `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora`

## 1.5 Checkpoint 与恢复

当前 checkpoint 命名已经统一：

- 使用零填充格式：`step-001000`、`step-002000`、`step-003000`、...

当前脚本恢复策略：

- `run_train.sh` 会在输出目录存在可恢复状态时自动传入 `--resume_from`
- `train.py` 会在输出目录下自动解析最新可恢复状态
- 若存在 `interrupted-latest`，会优先使用
- 否则使用 `checkpoints/` 下最新 step 的 `training_state.pt`
- 当前数据集为非 cache 读取模式，因此恢复时如果保存点处于 epoch 中间，代码会启用 `resume fast-path`：
  - 恢复 `global_step`
  - 恢复优化器/调度器状态
  - 但不会精确 replay 到 epoch 内旧的 batch 位置，而是从该 epoch 开头继续

注意：

- 早期运行遗留过两套重复命名：`step-1000` 和 `step-001000` 之类
- 当前代码已经修正，后续不会继续生成同一步的两套目录
- 旧目录暂未自动删除，避免误删历史状态

## 1.6 Benchmark 配置

固定 benchmark 每 `1000` steps 运行一次，配置在 [run_train.sh](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_train.sh:42)。

样本列表：

- [benchmark_meta_json_paths_fixed24.txt](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_fixed24.txt:1)

样本组成：

- OpenVid：`12`
- MOVI-D：`6`
- Genesis：`6`
- 总数：`24`

当前 benchmark 生成参数：

- 分辨率：`384x672`
- 输出帧数：`24`
- `context_frames=8`
- `fps=8`
- `num_inference_steps=50`
- `cfg_scale=5.0`
- `seed=42`

内部对齐说明：

- `batch_eval_lora.py` 会把请求的 `24` 帧在 Wan 内部对齐到 `25`
- 最终保存视频时只保留前 `24` 帧
- 因此外部看到的 benchmark 结果仍然是 `24` 帧

## 1.7 Validation 配置

validation 每 `2000` steps 运行一次。

样本列表：

- [benchmark_meta_json_paths_validation100.txt](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_validation100.txt:1)

样本组成：

- OpenVid：`50`
- MOVI-D：`25`
- Genesis：`25`
- 总数：`100`

当前 validation 生成参数：

- 分辨率：`384x672`
- 输出帧数：`24`
- `fps=8`
- `num_inference_steps=50`
- `cfg_scale=5.0`
- context sweep：`0,1,2,4,6,8`

validation 包装脚本：

- [run_validation_vbench.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_validation_vbench.py:1)

当前 validation 输出：

- 每个 context 设置对应的生成结果
- `summary.json`
- `context_curve.csv`
- VBench short 指标
- GT-based 指标：
  - `future_psnr`
  - `future_ssim`
  - `future_lpips`
  - `future_dino`

resize 规则：

- MOVI-D 指标前处理使用 `pad`
- 其他数据集指标前处理使用 `crop`

## 1.8 评测资源使用方式

- benchmark / validation 复用训练同一组 `4` 张卡
- 不额外申请新 GPU
- 在训练过程中的检查点处顺序执行

## 1.9 评测样本来源

当前训练期评测样本来源：

- MOVI-D test：`/data/gaoya/dataset/kubric_tfds_movi-d/mytest`
- Genesis held-out：`/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/mytest`
- OpenVid 训练期评测子集：`/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/mytest_train_eval`

相关列表与准备脚本：

- 全量列表：[benchmark_meta_json_paths_full.txt](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_full.txt:1)
- 训练期小列表准备脚本：[prepare_training_eval_meta_lists.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/prepare_training_eval_meta_lists.py:1)

## 1.10 当前已知运行特性

- 训练请求 `24` 帧时，Wan 内部时间长度仍会对齐到 `25`
- 现在这条对齐提示已经静默，不再刷 `num_frames % 4 != 1 ...`
- validation `summary` 的 flatten/logging bug 已修复
- 当前主要剩余不稳定项是偶发的多卡 `NCCL allreduce timeout`，不是数据格式错误，也不是 validation 指标解析错误

## 1.11 Stage0 benchmark 可视化

相关目录约定：

- 生成视频：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/output`
- 指标结果：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result`
- 可视化脚本与页面：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/tools/visualization`

当前常用页面：

- 指标折线图页：`result/model_metric_linecharts_latest/index.html`
- case 对比页：`tools/visualization/compact_selected_portal/index.html`

重建指标折线图页面：

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_train/train_0419/build_stage0_metric_linecharts.py
```

重建 case 对比页面：

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_train/train_0419/build_stage0_compact_selected_portal.py
```

本地启动静态可视化端口：

```bash
cd /data/gaoya/AAA_test_video/Benchmark/stage0_V2V
python -m http.server 8040 --bind 127.0.0.1
```

打开页面：

- `http://127.0.0.1:8040/result/model_metric_linecharts_latest/index.html`
- `http://127.0.0.1:8040/tools/visualization/compact_selected_portal/index.html`

当前指标页额外标注的信息：

- `sample300_full` 的数据集样本组成
- 不同模型实际输入条件差异
- 任务类型：`TI2V / context-aware / V2V`
- 生成分辨率：Wan 系列 `672x384`，VACE 系列 `720x544`


# 2. 基于 stage1 训练好的 V2V 模型，利用自建数据集的物理状态真值训练 adapter

这一部分当前文档尚未展开，后续以对应训练脚本为准再补。
