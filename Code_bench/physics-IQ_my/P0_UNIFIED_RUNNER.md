# Physics-IQ Verified P0 统一运行入口

`run_physicsiq_p0.py` 是本目录后续严格可比评测的统一协议层。它把模型推理
和评测协议分开：模型差异留在 adapter，所有模型共用同一套输入、输出、校验
和官方评分流程。

## 统一固定的协议

除非明确使用 `--allow-noncanonical`（此时结果不得标记为严格可比），runner
会自动拒绝以下任一偏差：

| 项目 | P0 固定值 |
|---|---|
| 输入集合 | 官方 BPP V2V take-1，198 cases；共享 `verified_v2v_bpp_198.txt` |
| 条件视频 | 72 帧、24 FPS、3 秒 |
| 输入模式 | V2V；`512x896` 是模型推理分辨率，条件源视频本身可为4K |
| 模型输出 | 189 帧、24 FPS |
| 条件前缀 | 删除前69帧 |
| 官方提交 | 后120帧纯预测、24 FPS、5 秒、896x512 |
| 120帧编码 | 从同一次推理返回的内存帧执行 `raw[69:189]`，直接 `export_to_video(..., macro_block_size=1)`；不得先解码已编码的 raw MP4 |
| 推理参数 | 40 steps、guidance 5、`do_cfg=false`、seed 42 |
| 条件 mask | `dynamic_effective` |
| negative prompt | `physrvg-72f-adapted-long-v1`，SHA256由 `common/physicsiq_p0_prompt.env` 校验 |
| 提交目录 | 只允许198个官方命名的 `.mp4` 文件；metadata/manifest放目录外 |

runner 还会逐个 probe 198 个条件视频和最终提交视频，检查帧数、FPS、时长、
分辨率、文件名集合和 prompt/input hash。它不会修改官方 benchmark 或原始数据。

## 文件布局

```text
physics-IQ_my/
├── run_physicsiq_p0.py                         # 唯一协议入口
├── P0_UNIFIED_RUNNER.md                         # 本说明
├── common/physicsiq_p0_prompt.env               # canonical negative prompt
├── common/validate_verified_run.py              # 旧版通用最终目录校验
└── adapters/                                    # 只放模型特定的推理入口
    ├── model_adapter_template.sh
    ├── physrvg_full_sa_72f_aligned_p0.sh
    └── xssc_full_sa_no_object_p0.sh
```

大文件、raw 视频、submission 和评测结果仍放在 `/data/gaoya`，不放在代码目录。

## 第一步：预检共享输入

预检不使用 GPU，只核查一次统一输入和条件视频：

```bash
python3 /home/gaoya/Code_Video/Code_bench/physics-IQ_my/run_physicsiq_p0.py \
  preflight \
  --summary-json /data/gaoya/agent-data/outputs/physics-iq-p0/preflight.json
```

成功时会确认：198 个 JSON、198 个 take-1 BPP V2V case、72@24 条件视频、
官方文件名集合和 canonical prompt。预检摘要可作为每次运行的输入证据。

## 第二步：通过 adapter 生成

runner 不猜测模型 checkpoint，也不复制模型的加载代码；它把固定协议通过
`PHYSIQ_*` 环境变量传给 adapter。adapter 只负责加载模型、逐 case 推理，并
把最终 submission 目录写入 `PHYSIQ_RESULT_FILE`。默认还要求 adapter 同时
保留189帧 raw 目录，供协议审计。

### Full-SA latent-mask 72f-aligned

先做 dry-run，确认命令和 GPU；dry-run 不加载模型：

```bash
python3 /home/gaoya/Code_Video/Code_bench/physics-IQ_my/run_physicsiq_p0.py \
  generate \
  --adapter /home/gaoya/Code_Video/Code_bench/physics-IQ_my/adapters/physrvg_full_sa_72f_aligned_p0.sh \
  --model-name physrvg-full-sa-latent-mask-step001000 \
  --gpu 2 \
  --adapter-arg /data/gaoya/agent-data/checkpoints/physrvg_full_sa_latent_mask/full-sa-pybullet-physrvg-latent-mask-b2-gacc2-20260818T052732Z/checkpoints/step-001000 \
  --dry-run
```

确认无误后去掉 `--dry-run` 执行。runner 会在 adapter 返回后自动检查 raw 和
submission；任何一个 case、帧数、FPS、分辨率或目录文件不符合要求都会失败，
不会产生可误标为 P0 的结果。

### Full-SA VJEPA

```bash
python3 /home/gaoya/Code_Video/Code_bench/physics-IQ_my/run_physicsiq_p0.py \
  generate \
  --adapter /home/gaoya/Code_Video/Code_bench/physics-IQ_my/adapters/physrvg_full_sa_72f_aligned_p0.sh \
  --model-name physrvg-full-sa-vjepa-step000500 \
  --gpu 2 \
  --adapter-arg /data/gaoya/agent-data/checkpoints/physrvg_full_sa_vjepa/full-sa-pybullet-physrvg-vjepa-b2-gacc2-ddp-sync-20260817T190000Z/checkpoints/step-000500 \
  --dry-run
```

该 adapter 与 latent-mask 使用同一个 72f-aligned PhysRVG 推理/内存编码
路径，只替换 LoRA checkpoint；模型路径、LoRA路径和编码协议会写入 manifest，
pipeline/运行环境等实现差异仍由 adapter 负责记录，不会被统一脚本偷偷替换。

### xSSC Full-SA no-object

```bash
python3 /home/gaoya/Code_Video/Code_bench/physics-IQ_my/run_physicsiq_p0.py \
  generate \
  --adapter /home/gaoya/Code_Video/Code_bench/physics-IQ_my/adapters/xssc_full_sa_no_object_p0.sh \
  --model-name xssc-full-sa-no-object-step002000 \
  --gpu 2 \
  --adapter-arg /path/to/checkpoint-directory \
  --dry-run
```

这个兼容 adapter 会把既有 xSSC 生成的 MP4 以硬链接方式放入统一 raw/submission
目录，不改变视频字节；中心 runner 仍会重新检查全部198个文件的格式。既有 xSSC
流程的120帧视频来自 raw MP4 后处理，不满足新的“内存切片直接编码”门槛；若只做
历史兼容检查，必须显式添加 `--no-require-direct-encoding`，所得结果不可标记为新的
严格 P0。

## Adapter 接口

新模型从模板复制：

```bash
cp /home/gaoya/Code_Video/Code_bench/physics-IQ_my/adapters/model_adapter_template.sh \
   /home/gaoya/Code_Video/Code_bench/physics-IQ_my/adapters/<model>_p0.sh
```

adapter 必须遵守：

1. 只使用 runner 提供的 `PHYSIQ_INPUT_LIST`、`PHYSIQ_SEED`、
   `PHYSIQ_NEGATIVE_PROMPT` 及其余 `PHYSIQ_*` 协议变量，不自行改成另一套
   198-case 输入或 120 帧格式。
2. 最终视频写入 `PHYSIQ_SUBMISSION_ROOT`，文件名直接使用输入 JSON 的
   `generated_video_name`。
3. raw 视频（如果模型输出 raw）写入 `PHYSIQ_RAW_ROOT`，应为189@24；
   从内存帧切片并编码的模型必须在自己的 adapter 中保持该路径，不能先把
   189帧 MP4解码后再重编码120帧。
4. 最后执行：

   ```bash
   printf '%s\n' "$PHYSIQ_SUBMISSION_ROOT" >"$PHYSIQ_RESULT_FILE"
   ```
5. 在 `PHYSIQ_ENCODING_MANIFEST` 写入 JSON，并声明：

   ```json
   {
     "encoding": {
       "mode": "in_memory_slice_then_export_to_video",
       "slice": "raw[69:189]",
       "macro_block_size": 1,
       "intermediate_decode": false
     }
   }
   ```

runner 会把以下关键变量注入 adapter：

```text
PHYSIQ_INPUT_LIST             统一198-case输入清单
PHYSIQ_RAW_ROOT               raw输出目录
PHYSIQ_SUBMISSION_ROOT        最终MP4目录
PHYSIQ_RAW_FRAMES=189        PHYSIQ_PREFIX_FRAMES=69
PHYSIQ_SUBMISSION_FRAMES=120 PHYSIQ_FPS=24
PHYSIQ_HEIGHT=512             PHYSIQ_WIDTH=896
PHYSIQ_NUM_INFERENCE_STEPS=40 PHYSIQ_GUIDANCE_SCALE=5.0
PHYSIQ_SEED=42                PHYSIQ_DO_CFG=0
PHYSIQ_CONTEXT_MASK_MODE=dynamic_effective
PHYSIQ_NEGATIVE_PROMPT_VERSION / PHYSIQ_NEGATIVE_PROMPT_SHA256
PHYSIQ_RESULT_FILE             PHYSIQ_ENCODING_MANIFEST
```

`PHYSIQ_GPU_ID` 是物理 GPU 号，runner 会设置 `CUDA_VISIBLE_DEVICES`；GPU 4
会被拒绝。不同模型仍可以有不同 base model、DiT、LoRA、pipeline 和运行环境，
但这些差异必须在 adapter 和运行 manifest 中显式记录。

## 单独校验和官方评分

对已有 submission 做严格校验：

```bash
python3 /home/gaoya/Code_Video/Code_bench/physics-IQ_my/run_physicsiq_p0.py \
  validate \
  /data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/<run-name> \
  --summary-json /data/gaoya/agent-data/outputs/physics-iq-p0/<run-name>.validation.json
```

校验通过后再调用官方 Verified evaluator；默认不使用 GPU：

```bash
python3 /home/gaoya/Code_Video/Code_bench/physics-IQ_my/run_physicsiq_p0.py \
  score \
  /data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/<run-name> \
  --n-process 0
```

评分结果写入 `evaluation/`，并由固定的
`aggregate_verified_official.sh` 使用 `--score-type verified` 聚合。

## 严格模式和例外

- `--allow-noncanonical`：允许非官方输入/prompt，仅用于诊断；结果必须标记
  为非 P0，不得进入严格比较表。
- `--allow-external-output`：兼容尚未迁移的历史 adapter；它允许 adapter
  返回非标准目录，但不会放宽视频内容校验。迁移完成后应去掉该选项。
- `--no-require-raw`：只适用于没有 raw 保存能力的旧流程；不建议用于新的
  严格评测，因为无法审计189→120的处理链。
- `--no-require-direct-encoding`：只用于历史兼容；默认会强制检查 adapter
  manifest 中的内存切片、`macro_block_size=1` 和无中间解码声明。关闭后不能
  标记为新的严格 P0。

统一 runner 能消除输入、输出和评分协议漂移，但不能让不同 LoRA、pipeline
或软件环境变成 bitwise identical；这些差异仍应在结果登记文件中保留。
