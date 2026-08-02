# Physics-IQ Verified 官方评测调用

本目录不复制或修改官方代码。两个包装脚本最终只调用：

- `physiq/run_physics_iq.py`：逐次运行评测。
- `physiq/aggregate_runs_from_csvs.py --score-type verified`：聚合 1 至 4 次运行。

官方仓库：`/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main`

Verified 数据集：`/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified`

## 1. 官方口径

- Physics-IQ Verified 是默认模式，评测时不要添加 `--original_physics_iq`。
- 每次运行必须有 198 个 `.mp4`，按文件名排序后 ID 前缀必须连续为 `0001_` 至 `0198_`。
- 每个视频必须精确为 5 秒，官方容差为 `0.001` 秒。
- 同一次运行内全部视频 FPS 必须一致。
- I2V 输入使用数据集 `switch-frames/` 中 take-1 对应的前 198 张图。
- V2V 输入使用 `split-videos/conditioning/30FPS/`，最终只提交生成的 5 秒，不包含 3 秒条件片段。
- `bpp` 表示模型适配的 best-practice prompt；没有模型专用模板时使用官方 `descriptions/best_practice/descriptions_base.csv`。
- `op` 表示官方 `descriptions/descriptions_original.csv` 原始 prompt。
- Verified 单次分数是 metrics JSON 中的 `final_score_view * 100`，不是 `final_score_stable`，也不是 `final_score_orig`。
- 官方接受单次运行；常规报告推荐 4 次独立运行的均值和标准差，宣称 SOTA 必须报告 4 次运行标准差。

本地数据已经包含官方评测所需的 take-1/take-2 GT：每个 `8FPS`、`16FPS`、`24FPS`、`30FPS` testing 和 real-mask 目录均有 396 个视频。

## 2. 环境

严格按官方推荐在官方仓库执行：

```bash
cd /home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main
uv sync
```

系统还必须提供 `ffprobe`，它随 `ffmpeg` 安装。当前非交互 shell 中尚未发现 `uv` 和 `ffprobe`；安装后再运行评测。

## 3. 生成视频目录

每个独立 run 放一个目录，推荐命名：

```text
<model_name>-bpp-run_01/
├── 0001_....mp4
├── 0002_....mp4
└── ...
```

标准 leaderboard 设置使用 `run_01` 至 `run_04`。官方允许文件名后半部分不同，但四位 ID 前缀必须完整、唯一且连续。

如果需要裁为 5 秒并统一为 24 FPS，可按官方示例执行：

```bash
mkdir -p /data/gaoya/agent-data/outputs/physics-iq-videos/<model_name>-bpp-run_01
for v in /path/to/raw-run-01/*.mp4; do
  ffmpeg -y -i "$v" -t 5 -r 24 \
    "/data/gaoya/agent-data/outputs/physics-iq-videos/<model_name>-bpp-run_01/$(basename "$v")"
done
```

## 4. 运行官方 Verified 评测

默认使用官方 base `bpp` descriptions，并使用串行指标计算：

```bash
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/run_verified_official.sh \
  --output-folder /data/gaoya/agent-data/outputs/physics-iq/<model_name>-bpp \
  /path/to/<model_name>-bpp-run_01 \
  /path/to/<model_name>-bpp-run_02 \
  /path/to/<model_name>-bpp-run_03 \
  /path/to/<model_name>-bpp-run_04
```

若生成时使用了其他 prompt CSV，必须显式传入同一个文件：

```bash
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/run_verified_official.sh \
  --output-folder /data/gaoya/agent-data/outputs/physics-iq/<model_name>-op \
  --descriptions-file /home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main/descriptions/descriptions_original.csv \
  /path/to/<model_name>-op-run_01
```

脚本会打印最终调用的完整官方命令。它会在 `/data/gaoya/agent-data/cache/physics-iq-verified` 创建临时兼容布局：

- GT 视频和 real masks 只通过符号链接读取，不修改原始数据集。
- 生成视频也以符号链接暂存，避免官方重命名逻辑改动原始生成目录。
- 生成 masks 写入临时工作区，退出时自动清理。
- 如需保留中间 masks 进行排错，添加 `--keep-workdir`。
- `--n-process 0` 是官方串行模式；并行 worker 内存开销很高，确认资源后再提高。

输出位于：

```text
<output-folder>/physics-IQ-benchmark-verified/results/
├── <run-name>.csv
├── <run-name>_metrics.json
├── physics_IQ_score_Original_barplot.pdf
└── physics_IQ_score_Verified_barplot.pdf
```

## 5. 聚合 Verified 分数

```bash
bash /home/gaoya/Code_Video/Code_bench/physics-IQ_my/aggregate_verified_official.sh \
  /data/gaoya/agent-data/outputs/physics-iq/<model_name>-bpp/physics-IQ-benchmark-verified/results/<model_name>-bpp-run_01.csv \
  /data/gaoya/agent-data/outputs/physics-iq/<model_name>-bpp/physics-IQ-benchmark-verified/results/<model_name>-bpp-run_02.csv \
  /data/gaoya/agent-data/outputs/physics-iq/<model_name>-bpp/physics-IQ-benchmark-verified/results/<model_name>-bpp-run_03.csv \
  /data/gaoya/agent-data/outputs/physics-iq/<model_name>-bpp/physics-IQ-benchmark-verified/results/<model_name>-bpp-run_04.csv \
  --save-csv /data/gaoya/agent-data/outputs/physics-iq/<model_name>-bpp/verified_summary.csv \
  --model-name '<model_name>'
```

聚合器使用 pandas 的样本标准差，并将 `final_score_view` 转为百分制，输出 leaderboard 所需的 `mean ± std`。

## 6. 重要的复现注意事项

- 同一报告中的 4 次运行必须使用相同模型配置、输入模式、prompt setting、FPS 和后处理，仅随机种子不同。
- 不要混合 `bpp` 与 `op`，也不要混合 I2V 与 V2V 结果。
- 不要直接复用旧 CSV 或旧 generated masks。包装脚本每次创建新临时工作区，避免官方缓存逻辑误用同名 run 的旧 mask。
- 结果投稿或加入 leaderboard 时，明确报告输入类型、prompt setting、运行次数、均值和标准差。
