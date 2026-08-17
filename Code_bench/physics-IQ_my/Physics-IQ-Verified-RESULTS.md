# Physics-IQ-Verified 评测结果登记

最后更新：2026-08-12 UTC

本文档是本项目 Physics-IQ-Verified 评测结果的唯一登记入口。新增评测方案时必须先在本文档登记，生成和官方评分完成后必须补充最终进度、分数与产物路径。不得覆盖、删除或静默修改历史结果的含义。

## 分数解释

- Physics-IQ-Verified 主分数为 `final_score_view * 100`。
- `final_score_origround` 是附加的 Original 聚合分数，不是 Verified 主分数。
- 只有使用 `P0` 协议的结果才标记为“严格可比”。
- 调用官方 evaluator 只是必要条件。Prompt、输入条件、预测时间段和198个文件集合也必须一致。
- 官方结果 CSV 包含66行场景，每行包含 left、center、right 三个视角，总计198个生成视频。

## 评测协议登记

### P0：严格统一的 BPP V2V 协议

后续跨模型比较必须使用该协议。

| 配置项 | 固定值 |
|---|---|
| Benchmark | Physics-IQ-Verified |
| Case | 198个 take-1 case，文件名集合与官方完全一致 |
| Prompt | `descriptions/best_practice/descriptions_base.csv` 中的 BPP Prompt |
| 输入模式 | V2V |
| 条件视频 | 72帧、24 FPS、3秒 |
| 分辨率 | 512x896 |
| 推理步数 | 40 |
| Guidance | 5 |
| Seed | 42 |
| 原始模型输出 | 189帧、24 FPS |
| 干净条件前缀 | 69帧 |
| Evaluator 输入 | 后120帧纯预测视频，24 FPS、5秒 |
| Evaluator | 官方 `physiq/run_physics_iq.py` |
| 聚合方式 | 官方 `physiq/aggregate_runs_from_csvs.py --score-type verified` |

统一输入清单：

`/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/inputs/bpp/verified_v2v_bpp_198.txt`

输入清单 SHA256：

`f0cbcd79cc7d523fd0c30ef6053373163dbc3667da88baa5d10e205def177956`

预期生成文件名集合 SHA256：

`8ee2101106b2acaaecac752ea5175cee89d30b3aab9c602623ff02360eacc071`

官方评测代码来源：

| 文件 | SHA256 |
|---|---|
| `physiq/run_physics_iq.py` | `46348303225316f935873aeed265a5a4a4bb79345aed1f63d0363ec4c5a7c1e5` |
| `physiq/aggregate_runs_from_csvs.py` | `71125009926b1ad2120ae7fdbb80531db620aed63336a328538ba925bc3a8ede` |
| `descriptions/best_practice/descriptions_base.csv` | `20ffd208acc0b0f50d4638d1da69218168e78336e96118244a53d0ae046729c8` |

### P1：旧版 PhysRVG BPP 协议

- 使用198个BPP case。
- 条件视频为90帧、30 FPS。
- 分辨率480x832，16个推理步，Guidance 5。
- 生成149帧并按5秒导出，随后转换为150帧、30 FPS。
- 提交的5秒视频包含重建的条件段，真正的未来预测只有约2秒。
- 使用官方 evaluator，但不能与 `P0` 结果比较。

### P2：旧版 PhysRVG OP 协议

时序配置与 `P1` 相同，但使用 `descriptions/descriptions_original.csv` 中的 OP Prompt，不能与 `P0` 结果比较。

### P3：Wan2.2 OP 最后一帧 I2V 协议

- 使用198个OP case。
- 仅将条件视频最后一帧作为PNG输入。
- 分辨率1248x704。
- 最终输出120帧、24 FPS、5秒。
- 输入是单帧I2V，不是完整72帧V2V条件，不能与 `P0` 结果比较。

## 已完成的官方评测结果

| 状态 | 模型与 Run | 协议 | 视频数 | Verified | Original | 严格可比 |
|---|---|---:|---:|---:|---:|---|
| 已完成 | xSSC Full-SA no-object step-2000 | P0 | 198/198 | **33.8024** | 35.65 | 是 |
| 已完成 | xSSC Full-SA no-object xSSC-loss DINOv3 MOVi-C step-500 | P0 | 198/198 | **33.2976** | 34.45 | 是 |
| 已完成 | PhysRVG-72f-adapted | P0 | 198/198 | **39.9116** | 41.86 | 是 |
| 已完成 | PhysRVG 旧版 BPP | P1 | 198/198 | 28.7738 | 26.92 | 否 |
| 已完成 | PhysRVG 旧版 OP | P2 | 198/198 | 29.5964 | 27.73 | 否 |
| 已完成 | Wan2.2-TI2V-5B OP last-frame baseline | P3 | 198/198 | 28.1540 | 26.43 | 否 |

## xSSC Full-SA no-object step-2000

Run ID：

`full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01`

生成与case准备脚本：

- 通用启动器：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/launch_verified_benchmark.sh`
- 模型适配器：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/adapters/xssc_full_sa_no_object.sh`
- 推理封装：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_verified_inference.sh`
- 输入构建：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/build_verified_v2v_inputs.py`
- 189帧转120帧后处理：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/prepare_verified_outputs.py`

官方评分脚本：

- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/run_verified_official.sh`
- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/aggregate_verified_official.sh`
- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/score_verified_runs.sh`

结果产物：

- 120帧 submission：`/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01`
- 189帧 raw：`/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/raw/full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01`
- Metrics：`/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/physics-IQ-benchmark-verified/results/full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01_metrics.json`
- 官方 CSV：`/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/physics-IQ-benchmark-verified/results/full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01.csv`
- Batch manifest：`/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/raw/batch_manifest_full_sa_no_object_gpu67_resume_step6-step-002000-36890878a58d-bpp-run_01.json`

子指标：

| 指标 | 数值 x100 |
|---|---:|
| Physics-IQ Verified | 33.8024 |
| Spatial view | 31.7480 |
| Spatiotemporal view | 49.6522 |
| Weighted spatial view | 23.9549 |
| MSE view | 29.8545 |

## xSSC Full-SA no-object xSSC-loss DINOv3 MOVi-C step-500

状态：已完成。生成、后处理和官方评分于 `2026-08-12 11:38:51 UTC` 完成。

Run ID：

`full_sa_no_object_xssc_loss_dinov3_movic_step50000-step-000500-2c970f718bcf-bpp-run_01`

Checkpoint：

`/data/gaoya/agent-data/checkpoints/xssc_feature_loss/full_sa_no_object_xssc_loss_dinov3_movic_step50000/formal_gpu01/checkpoints/step-000500`

Checkpoint SHA256：

`2c970f718bcf788ea17901af7d2fd041ecbe4064ce8fa49c8377f980e1223866`

生成入口与脚本：

- 原模型 shell 入口：`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_infer_from_experiment.sh`
- 实际底层模型入口：`infer_xssc_object_self_attn_lora.py`。原 shell 将帧数写死为49、context写死为8，不能直接满足P0；benchmark适配器不修改原代码，而是调用同一底层入口并显式传入P0参数。
- 本次配置：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/feature_loss_step000500_physicsiq_verified_gpu67_remote.env`
- 本次启动脚本：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_feature_loss_step000500_physicsiq_verified_gpu67_remote.sh`
- 通用SSH启动器：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/launch_xssc_experiment_verified_ssh118.sh`
- 通用双卡worker：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_xssc_experiment_verified_gpu67_remote_worker.sh`
- 输入路径映射与协议校验：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/prepare_verified_remote_inputs.py`
- 189帧转120帧后处理：`/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/prepare_verified_outputs.py`

评测协议：`P0`，与 `xSSC Full-SA no-object step-2000` 相同。

- 198个take-1 case，BPP Prompt，完整72帧、24 FPS、3秒V2V条件。
- 512x896，40步，Guidance 5，run_01 seed 42。
- GPU 6和GPU 7各处理99个case；每个case使用相同seed 42。
- 模型原始输出189帧、24 FPS；完整保存raw，并去掉前69帧，提交后120帧、24 FPS、5秒视频。
- 使用官方 `physiq/run_physics_iq.py`、`descriptions/best_practice/descriptions_base.csv` 和 `--score-type verified` 聚合方式。
- 官方评测入口、聚合入口和BPP CSV的SHA256均与本文档P0登记值一致。

SSH 118上的结果产物：

- 189帧 raw：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/raw/full_sa_no_object_xssc_loss_dinov3_movic_step50000-step-000500-2c970f718bcf-bpp-run_01`
- 120帧 submission：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/full_sa_no_object_xssc_loss_dinov3_movic_step50000-step-000500-2c970f718bcf-bpp-run_01`
- Metrics：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/physics-IQ-benchmark-verified/results/full_sa_no_object_xssc_loss_dinov3_movic_step50000-step-000500-2c970f718bcf-bpp-run_01_metrics.json`
- 官方 CSV：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/physics-IQ-benchmark-verified/results/full_sa_no_object_xssc_loss_dinov3_movic_step50000-step-000500-2c970f718bcf-bpp-run_01.csv`
- 汇总 CSV：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/full_sa_no_object_xssc_loss_dinov3_movic_step50000-step-000500-2c970f718bcf-bpp-run_01_verified_summary.csv`
- 日志：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/logs/full_sa_no_object_xssc_loss_dinov3_movic_step50000-step-000500-2c970f718bcf-bpp-run_01`

子指标：

| 指标 | 数值 x100 |
|---|---:|
| Physics-IQ Verified | 33.2976 |
| Spatial view | 30.4234 |
| Spatiotemporal view | 50.4199 |
| Weighted spatial view | 23.0123 |
| MSE view | 29.3349 |
| Physics-IQ Original | 34.45 |

## PhysRVG 旧版 BPP

SSH 118上的生成和评分脚本：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main/generate_physics_iq_verified_official.py`
- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main/prepare_physiq_submission_30fps.py`
- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main/eval_physics_iq_verified_official.sh`

SSH 118上的结果产物：

- Submission：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physRVG-verified-bpp-run_01_30fps_mp4only`
- Manifest：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physRVG-verified-bpp-run_01_manifest.json`
- Metrics：`/home/gaoya/data/agent-data/outputs/physics_iq_verified_eval_physrvg_30fps/physics-IQ-benchmark-verified/results/physRVG-verified-bpp-run_01_30fps_mp4only_metrics.json`
- 官方 CSV：`/home/gaoya/data/agent-data/outputs/physics_iq_verified_eval_physrvg_30fps/physics-IQ-benchmark-verified/results/physRVG-verified-bpp-run_01_30fps_mp4only.csv`

## PhysRVG 旧版 OP

使用与旧版BPP相同的脚本，但Prompt文件改为 `descriptions/descriptions_original.csv`。

SSH 118上的结果产物：

- Submission：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physRVG-verified-op-run_01_30fps_mp4only`
- Manifest：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physRVG-verified-op-run_01_manifest.json`
- Metrics：`/home/gaoya/data/agent-data/outputs/physics_iq_verified_eval_physrvg_op_30fps/physics-IQ-benchmark-verified/results/physRVG-verified-op-run_01_30fps_mp4only_metrics.json`
- 官方 CSV：`/home/gaoya/data/agent-data/outputs/physics_iq_verified_eval_physrvg_op_30fps/physics-IQ-benchmark-verified/results/physRVG-verified-op-run_01_30fps_mp4only.csv`

## Wan2.2-TI2V-5B OP last-frame baseline

Case准备脚本：

`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/wan22_physicsiq_repro/prepare_physicsiq_original_i2v.py`

现有结果元数据没有保留唯一且可确认的生成启动脚本。在补充来源证据之前，不得推测或填写生成入口。

SSH 118上的结果产物：

- Submission：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/wan22_i2v_physicsiq_original_repro/baseline/submission_5s/wan22-ti2v5b-op-conditioning-last-frame-run_01_24fps`
- Metrics：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/wan22_i2v_physicsiq_original_repro/official_eval/baseline/physics-IQ-benchmark-verified/results/wan22-ti2v5b-op-conditioning-last-frame-run_01_24fps_metrics.json`
- 官方 CSV：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/wan22_i2v_physicsiq_original_repro/official_eval/baseline/physics-IQ-benchmark-verified/results/wan22-ti2v5b-op-conditioning-last-frame-run_01_24fps.csv`

## 运行中与未完成方案

本节中的方案在官方评分完成前没有可报告的 Physics-IQ-Verified 分数。

| 最近状态 | 模型与 Run | 协议 | 进度 | 分数 |
|---|---|---:|---:|---|
| 已中断，可续跑 | xSSC slot-dedup step-2000 | P0 | 73/198 raw | 暂无 |
| 未完成 | stage1b step-2500 | 非P0 | 11/198 | 暂无 |
| 未完成 | stage1b step-2500 with negative prompt | 非P0 | 1/198 | 暂无 |
| 未完成 | Wan2.2 BoN16 | P3衍生 | 仅有部分候选结果 | 暂无 |

## PhysRVG-72f-adapted P0 运行

Run ID：

`physrvg-72f-xssc-aligned-bpp-run_01`

该方案使用PhysRVG官方权重和模型类，但通过适配目录中的condition-mask实现，使72帧输入编码得到的全部时序latent成为有效条件。它不是PhysRVG原始5帧条件推理协议，结果必须标记为 `PhysRVG-72f-adapted`。

生成脚本和配置：

- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/PhysRVG/generate_physrvg_verified.py`
- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/PhysRVG/pipeline_wan_v2v_72f.py`
- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/PhysRVG/run_physrvg_verified_remote118_gpu67.sh`
- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/PhysRVG/run_physrvg_verified_gpu67_remote_worker.sh`
- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/PhysRVG/physrvg_verified_remote118_gpu67.env`

SSH 118上的结果产物：

- 120帧 submission：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/generated_videos_5s/physrvg-72f-xssc-aligned-bpp-run_01`
- 续跑后保留的189帧 raw：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/raw_model_outputs/physrvg-72f-xssc-aligned-bpp-run_01`
- Evaluation root：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/physrvg-72f-xssc-aligned-bpp-run_01`
- Pipeline log：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/logs/physrvg-72f-xssc-aligned-bpp-run_01_pipeline.log`

官方评测结果：

- Verified：`39.91156421027763`（主表记录 `39.9116`）
- Original：`0.41853671160880473`，官方 JSON 展示 `41.86`
- CSV：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/physrvg-72f-xssc-aligned-bpp-run_01/physics-IQ-benchmark-verified/results/physrvg-72f-xssc-aligned-bpp-run_01_mp4only.csv`
- Metrics JSON：`/home/gaoya/data/AAA_test_video/0623/test/physicsiq/physicsiq_verified/evaluation/physrvg-72f-xssc-aligned-bpp-run_01/physics-IQ-benchmark-verified/results/physrvg-72f-xssc-aligned-bpp-run_01_mp4only_metrics.json`
- 官方 CSV：66 行场景记录，对应 198 个视频；Verified 子指标：spatiotemporal `0.5129906042588389`，spatial `0.4086476549475409`，weighted spatial `0.33871453322437833`，MSE `0.33610977598034736`。
- 备注：提交目录含 `.json` 边车文件，官方评测器只接受 MP4；评测时使用内容不变的 198 个 MP4 硬链接目录 `physrvg-72f-xssc-aligned-bpp-run_01_mp4only`。

Raw保存说明：

- 暂停前已完成的前121个submission按要求不重新生成，因此没有对应的189帧PhysRVG raw。
- 剩余77个case同时保留真实189帧raw和120帧纯预测submission。

可视化：

- 页面：`http://10.176.42.45:8844/physicsiq-verified-standard/`
- 页面源码：`/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/physicsiq-verified-standard/index.html`
- 增量同步：`/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/physicsiq-verified-standard/sync_from_118.sh`

## xSSC slot-dedup step-2000 P0 运行

Run ID：

`full_sa_slot_dedup_merge_gpu56_resume_wandb23kjvge2-step-002000-b110c1ed7a64-bpp-run_01`

生成脚本和配置：

- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_slot_dedup_step002000_physicsiq_verified_gpu2.sh`
- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/slot_dedup_step002000_physicsiq_verified_gpu2.json`
- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/slot_dedup_step002000_physicsiq_verified_gpu3.json`
- `/home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_verified_inference.sh`

结果产物：

- 部分raw：`/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/raw/full_sa_slot_dedup_merge_gpu56_resume_wandb23kjvge2-step-002000-b110c1ed7a64-bpp-run_01`
- Batch manifest：`/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/raw/batch_manifest_full_sa_slot_dedup_merge_gpu56_resume_wandb23kjvge2-step-002000-b110c1ed7a64-bpp-run_01.json`

## 其他未完成尝试

stage1b生成入口：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.sh`
- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v.py`
- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_physics_iq_verified_vnewtrain0705_v2v_fullctx.py`

现有输出为153帧、30 FPS、5.1秒，不是 `P0` submission。除非重新进行规范化并补充完整来源记录，否则不得按照 `P0` 结果评分或比较。

## 后续新增方案的强制登记模板

新增评测方案在完整生成开始前必须添加以下记录：

```markdown
### <模型名称与 Run ID>

- 状态：计划中 | 运行中 | 已完成 | 失败 | 已中断
- 协议：P0 | P1 | P2 | P3 | 新协议编号
- 模型与checkpoint：
- Case生成入口脚本：
- 模型适配器或推理脚本：
- 配置文件：
- 输入清单及SHA256：
- Prompt文件及SHA256：
- 条件帧数、FPS和时长：
- 分辨率、推理步数、Guidance和Seed：
- Raw输出帧数和FPS：
- 条件前缀删除或其它时序后处理：
- Submission帧数、FPS和时长：
- Submission目录：
- Raw输出目录：
- 官方evaluator及SHA256：
- 官方聚合命令：
- 结果CSV：
- Metrics JSON：
- Verified分数：
- Original附加分数：
- 是否与P0严格可比：是 | 否
- 偏差与注意事项：
```

完成规则：

1. 198个预期文件全部存在并通过格式检查后，才能登记分数。
2. 必须记录实际生成脚本，不能只填写模型名称。
3. Prompt和输入条件协议必须与evaluator来源分别记录。
4. 失败、废弃和被替代的run也要保留为历史记录。
5. 修改模型接口的方案必须明确标注，例如使用 `PhysRVG-72f-adapted`，不能写成 `PhysRVG official`。
6. 官方聚合完成后应立即更新进度、产物路径、Metrics和最终分数。
