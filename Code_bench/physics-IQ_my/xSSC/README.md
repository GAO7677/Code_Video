# xSSC Full-SA No-Object: Physics-IQ Verified V2V

该目录使用现有模型入口：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/
train_xSSC/object_self_attn_lora_experiments/infer_full_sa_no_object_lora.py
```

不修改原推理代码，不修改 Physics-IQ 官方代码或下载的数据集。

所有视频、输入缓存和评分结果写入：

```text
/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified
```

## 官方要求与模型设置

官方硬性要求包括完整 V2V conditioning、198 个 take-1 多视角样本、精确 5 秒生成段、连续 ID、统一 FPS，以及使用 `final_score_view`。分辨率、采样步数、negative prompt 和随机种子属于模型推理设置，不是 benchmark 对所有模型的官方统一规定；本项目从新启动的 P0 横向比较开始，统一采用 `../common/physicsiq_p0_prompt.env` 中与 PhysRVG-72f-adapted 相同的长版 negative prompt。

- Benchmark：Physics-IQ Verified。
- 输入模式：V2V。
- Prompt：官方 `descriptions/best_practice/descriptions_base.csv`，即 base `bpp`。
- 样本：198 个 take-1、多视角样本，ID 连续为 `0001` 至 `0198`。
- Conditioning：官方 3 秒 conditioning 主文件，不使用 `_h264` 副本。
- Conditioning FPS：从官方 30 FPS 无损保留时长地转换为 24 FPS，使其与生成/GT FPS 一致。
- 模型上下文：完整读取转换后的 72 帧 conditioning，`sampling-mode=prefix`，不抽帧、不截短。
- xSSC 分辨率：`512x896`。
- xSSC 推理步数：`40`。
- 原始模型输出：`189 frames @ 24 FPS`。
- Wan 时间 VAE 将完整 72 帧 condition 编码为 18 个 clean prefix latents，对应输出前 69 帧。
- 正式提交视频：删除前 69 个条件帧，保留后续 120 帧，即严格 `5.000 秒 @ 24 FPS`。
- 四次运行 seeds：`42, 43, 44, 45`。
- 输出目录包含 checkpoint SHA-256 短指纹，避免可变 checkpoint 名复用旧结果。
- 推理设置 `PYTHONNOUSERSITE=1`，避免用户目录中的 `peft` 覆盖 wan-cu128 环境依赖。
- Verified 分数：官方输出中的 `final_score_view * 100`。

## 目录结构

```text
physicsiq_verified/
├── inputs/bpp/
│   ├── conditioning/24FPS/
│   ├── jsons/
│   ├── manifest.json
│   └── verified_v2v_bpp_198.txt
├── raw/<model>-<checkpoint>-bpp-run_01/
├── generated_videos_5s/<model>-<checkpoint>-bpp-run_01/
└── evaluation/
    ├── physics-IQ-benchmark-verified/results/
    └── verified_summary.csv
```

## 1. 安装官方 benchmark 环境

当前系统已有 wan-cu128 的 `ffmpeg/ffprobe`，但尚未发现 `uv`。按官方推荐执行：

```bash
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/setup_official_benchmark_env.sh
```

uv 缓存和环境放在 `/data/gaoya/agent-data/cache`，不会在 `/home/gaoya` 存放大缓存。

## 2. 生成一个正式 run

```bash
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_verified_inference.sh \
  /path/to/checkpoint-directory \
  0 \
  1
```

参数依次为 checkpoint、物理 GPU ID、run index。GPU 4 被明确禁止。

该命令会自动：

1. 构建 198 条官方 BPP V2V JSON。
2. 将 conditioning 转为 24 FPS。
3. 将完整 72 帧 condition 送入原 Full-SA No-Object Python 推理入口，并生成 189 帧。
4. 删除 clean V2V prefix 对应的前 69 帧。
5. 验证最终目录包含 198 个、每个 120 帧/24 FPS/5.000 秒的 MP4。

### 单 case smoke test

smoke test 使用完全相同的 condition、189 帧推理和 5 秒后处理参数，但只运行官方 ID `0001`，并隔离写入 `physicsiq_verified/smoke/`：

```bash
CASE_LIMIT=1 bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_verified_inference.sh \
  /path/to/checkpoint-directory \
  3 \
  1
```

smoke 文件夹不能用于官方评分；正式评分脚本仍要求每个 run 恰好包含 198 个视频。

## 3. 四次独立运行

分别执行 `RUN_INDEX=1,2,3,4`。可以选择不同空闲 GPU，但不要使用 GPU 4：

```bash
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_verified_inference.sh CHECKPOINT 0 1
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_verified_inference.sh CHECKPOINT 1 2
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_verified_inference.sh CHECKPOINT 2 3
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_verified_inference.sh CHECKPOINT 3 4
```

每个 run 都是 198 个 189 帧长视频推理，计算和显存开销显著高于原来的 49 帧测试。完整 condition 不允许退回只读取前 8 帧；启动前应确认 GPU 和磁盘资源。

## 4. 官方 Verified 评分

将生成的 1 至 4 个正式目录传给评分脚本：

```bash
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/score_verified_runs.sh \
  /data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/<model>-<checkpoint>-bpp-run_01 \
  /data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/<model>-<checkpoint>-bpp-run_02 \
  /data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/<model>-<checkpoint>-bpp-run_03 \
  /data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/<model>-<checkpoint>-bpp-run_04
```

评分脚本调用官方 `physiq/run_physics_iq.py`，然后调用官方 `aggregate_runs_from_csvs.py --score-type verified`。最终 leaderboard 汇总位于：

```text
/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/verified_summary.csv
```
