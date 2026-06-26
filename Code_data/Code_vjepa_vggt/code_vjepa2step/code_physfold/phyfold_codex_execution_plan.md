# PhyFOLD 执行方案 v2：Failure-Localized Video Supervision + Predictive Distillation

> 目标：实现一个高可行性的局部物理纠错原型，用于验证  
> `student 自生成 long rollout -> 定位物理失败窗口 -> 分情况构造局部训练信号 -> 只在 failure window 上训练`  
> 是否能提升视频生成中的物理一致性。
>
> 本版核心改动：  
> 不再把训练信号完全绑定在 best-of-K positive 上，而是采用 A/B 双分支：
>
> - `A 分支`：当存在具体 positive continuation 时，使用局部强监督
> - `B 分支`：当不存在具体 positive 时，使用 failure-localized V-JEPA predictive distillation
---

## 0. 核心定义

项目名仍记为 `PhyFOLD`：

```text
Physics Failure-Localized On-policy Distillation
```

本版方法定义：

```text
student self-rollout
-> failure onset localization
-> local relational region extraction
-> branch A or branch B training
-> local physical correction
```

与旧版区别：

```text
旧版：
same-prefix corrected continuation 是默认主监督来源

新版：
same-prefix corrected continuation 仍然保留，但只作为 A 分支
当没有 concrete positive 时，转入 B 分支，用 V-JEPA 提供局部未来表征教师
```

第一版只解决四类高置信物理错误：

```text
1. object permanence：物体消失、重现、面积突变、形状严重漂移
2. contact consistency：接触关系突然断裂、悬浮、穿透、接触闪烁
3. collision response：碰撞后目标物体不动、方向错误、速度突变
4. gravity / falling：掉落悬浮、反向上升、倒下后复原
```

暂不覆盖：

```text
液体、布料、破碎、复杂多体堆叠、强相机运动、真实人类动作
```

---

## 1. 新版核心思路

### 1.1 两类训练情形

#### 情况 A：存在具体 positive continuation

来源包括：

```text
1. same-prefix best-of-K continuation 中采到更好样本
2. 更强 teacher video model 在相同 prefix 下生成了更好 continuation
3. 已有高质量 offline continuation 可作为局部正样本
```

训练方式：

```text
只在 failure window × local relational region 上施加强监督
L_video = local flow matching / local SFT / optional local DPO
```

直觉：

```text
有具体正样本时，用视频级强监督最直接，也最稳定。
```

#### 情况 B：不存在具体 positive continuation

此时不强行构造 pseudo positive video，也不做 `V-JEPA -> VAE latent` 反演。

改为：

```text
student failed rollout x_bad
-> mask 掉 failure window × object tube / relational tube
-> V-JEPA 根据 prefix + visible context 预测未来局部正向表征 f_teacher^+
-> student DiT hidden 经过 projector 得到 f_student
-> 在 failure-localized region 上做 predictive distillation
```

基础形式：

```text
L_jepa_pos = align(f_student, f_teacher^+)
```

可选再加入对比项：

```text
f_bad = V-JEPA encode(原失败区域)

让：
f_student 靠近 f_teacher^+
f_student 相对远离 f_bad
```

这条分支命名为：

```text
failure-localized V-JEPA predictive distillation
```

---

## 2. 设计原则

### 2.1 必须坚持的边界

```text
1. 不做 full fine-tuning
2. 不先做 RL
3. 不要求 V-JEPA 直接生成视频或 VAE latent
4. 不把 V-JEPA 当唯一物理 scorer
5. 不在全视频范围内施加对齐损失
6. 不训练低置信 failure case
7. 不在表征空间伪造“硬正样本视频”
```

### 2.2 为什么不用 JEPA-to-VAE 反演

```text
1. V-JEPA feature 不是天然可逆解码目标
2. feature space 的合理未来不对应唯一像素未来
3. 训练 feature -> latent bridge 会引入新研究问题
4. 很容易出现 feature hacking，而不是物理修正
```

因此本版固定采用：

```text
V-JEPA 只做 future representation teacher / scorer / regularizer
真正的视频生成仍由 student video model 负责
```

---

## 3. 最小可行版本范围

### 3.1 Backbone

优先顺序：

```text
Phase 1: Wan-VACE / Wan2.1-VACE 1.3B 级别，验证 pipeline
Phase 2: Wan2.2-5B I2V / VACE，训练 LoRA / adapter
```

第一版训练限制：

```text
- freeze VAE
- freeze text encoder
- freeze most spatial blocks
- train LoRA / temporal blocks / projector / optional adapter only
- 不做 full fine-tuning
```

### 3.2 任务形式

推荐继续使用：

```text
I2V / video continuation
```

输入：

```text
prompt c
prefix video P = student generated x_1:t
```

输出：

```text
continuation video x_t:T
```

### 3.3 第一版验收目标

实现以下闭环即可：

```text
1. 生成 student long rollout
2. 用 perception wrapper 提取 mask / track / depth / flow
3. 构建 object tube 与 local relational region
4. 自动定位 high-confidence failure onset t*
5. A 分支：若能采到 positive，输出 local video supervision pairs
6. B 分支：若无 positive，输出 local JEPA distillation samples
7. 训练数据统一写成 jsonl
8. 可选：接入 local SFT + local JEPA regularization
9. 输出评估报告：base vs A only vs A+B
```

---

## 4. 推荐代码结构

```text
physfold/
  __init__.py

  configs/
    prompt_bank.yaml
    mining.yaml
    correction.yaml
    training.yaml
    eval.yaml
    vjepa.yaml

  prompts/
    build_prompt_bank.py

  rollout/
    generate_rollouts.py
    continuation_backend.py

  perception/
    wrappers.py
    extract_masks.py
    track_points.py
    estimate_depth.py
    estimate_flow.py

  mining/
    data_schema.py
    tubes.py
    region_masks.py
    failure_scores.py
    mine_failures.py

  correction/
    resample_continuations.py
    rank_continuations.py
    build_local_pairs.py

  vjepa/
    wrappers.py
    build_teacher_targets.py
    encode_failed_regions.py
    projector.py
    losses.py

  training/
    build_training_manifest.py
    build_local_dataset.py
    train_local_sft.py
    train_local_ab.py
    loss_masks.py

  eval/
    eval_physics.py
    eval_visual_semantic.py
    eval_jepa_features.py
    summarize_results.py

scripts/
  00_build_prompts.sh
  01_generate_rollouts.sh
  02_extract_perception.sh
  03_mine_failures.sh
  04_resample_continuations.sh
  05_build_training_manifest.sh
  06_train_local_ab.sh
  07_eval.sh

tests/
  test_failure_scores.py
  test_tubes.py
  test_region_masks.py
  test_schema.py
  test_jepa_targets.py
  test_local_dataset.py
```

---

## 5. 统一数据 schema

中间文件优先使用 `jsonl`，数组或 feature 保存为 `.npz` / `.pt`。

### 5.1 RolloutSample

文件：`data/rollouts/rollouts.jsonl`

```json
{
  "sample_id": "rollout_000001",
  "prompt": "A red ball rolls from the left and hits a wooden block on the table.",
  "seed": 123,
  "video_path": "data/rollouts/videos/rollout_000001.mp4",
  "num_frames": 96,
  "fps": 12,
  "height": 480,
  "width": 832,
  "generation_backend": "wan_vace",
  "segment_boundaries": [[0, 31], [32, 63], [64, 95]],
  "metadata": {
    "category": "collision",
    "camera": "static",
    "objects": ["red ball", "wooden block", "table"]
  }
}
```

### 5.2 FailureCase

文件：`data/mining/failures.jsonl`

```json
{
  "failure_id": "failure_000001",
  "sample_id": "rollout_000001",
  "prompt": "A red ball rolls from the left and hits a wooden block on the table.",
  "video_path": "data/rollouts/videos/rollout_000001.mp4",
  "failure_type": "collision_no_response",
  "failure_onset": 44,
  "window_start": 38,
  "window_end": 62,
  "prefix_end": 41,
  "failure_score": 0.81,
  "detector_confidence": 0.88,
  "failure_objects": ["red ball", "wooden block"],
  "tube_path": "data/mining/tubes/rollout_000001.npz",
  "region_mask_path": "data/mining/regions/failure_000001.npz"
}
```

### 5.3 CandidatePairSample

用于 A 分支。

文件：`data/correction/local_pairs.jsonl`

```json
{
  "failure_id": "failure_000001",
  "sample_id": "rollout_000001",
  "prompt": "A red ball rolls from the left and hits a wooden block on the table.",
  "prefix_path": "data/correction/prefix/failure_000001.mp4",
  "negative_path": "data/correction/negative/failure_000001.mp4",
  "positive_path": "data/correction/positive/failure_000001_k2.mp4",
  "score_positive": 0.79,
  "score_negative": 0.41,
  "score_margin": 0.38,
  "window_start": 38,
  "window_end": 62,
  "prefix_end": 41,
  "region_mask_path": "data/mining/regions/failure_000001.npz",
  "training_branch": "A_video_supervision"
}
```

### 5.4 JepaDistillSample

用于 B 分支。

文件：`data/vjepa/distill_samples.jsonl`

```json
{
  "failure_id": "failure_000001",
  "sample_id": "rollout_000001",
  "prompt": "A red ball rolls from the left and hits a wooden block on the table.",
  "video_path": "data/rollouts/videos/rollout_000001.mp4",
  "prefix_end": 41,
  "window_start": 38,
  "window_end": 62,
  "region_mask_path": "data/mining/regions/failure_000001.npz",
  "teacher_target_path": "data/vjepa/teacher_targets/failure_000001.pt",
  "failed_feature_path": "data/vjepa/failed_features/failure_000001.pt",
  "jepa_confidence": 0.73,
  "training_branch": "B_jepa_distill"
}
```

### 5.5 UnifiedTrainingSample

文件：`data/training/training_manifest.jsonl`

```json
{
  "training_id": "train_000001",
  "branch_type": "A_video_supervision",
  "pair_path": "data/correction/local_pairs.jsonl::failure_000001",
  "sample_weight": 0.81
}
```

或：

```json
{
  "training_id": "train_000157",
  "branch_type": "B_jepa_distill",
  "distill_path": "data/vjepa/distill_samples.jsonl::failure_000083",
  "sample_weight": 0.52
}
```

---

## 6. Prompt bank

第一版仍使用受控 prompt bank，按物理类型分组：

```text
1. rolling -> collision
2. falling -> landing
3. sliding -> edge / stop
4. contact -> support / detach
```

要求：

```text
- 优先静态镜头
- 2-3 个主要物体
- 明确初始关系
- 明确运动方向
- 避免抽象、诗意、开放世界描述
```

---

## 7. Rollout 生成

文件：`physfold/rollout/generate_rollouts.py`

第一版继续采用 segment-by-segment continuation，保证更容易出现长程失败。

目标：

```text
1. 让 student 在可控 prompt 上暴露 failure
2. 生成可供同前缀重采样的 rollout
3. 保留 seed、prefix 边界、segment 元信息
```

---

## 8. Perception 与 local relational region

### 8.1 外部 perception wrapper

第一版仍允许使用：

```text
- SAM2 / mask extractor
- CoTracker / point tracking
- Depth Anything
- RAFT / flow
```

### 8.2 Local relational region

不要只用单物体框，而是构建：

```text
failure window × relational region
```

包含：

```text
1. failure object 主体
2. interacting object
3. contact band / support band
4. 在时间上前后扩展的局部窗口
```

第一版规则：

```python
region = dilate(union(mask_failure_obj, mask_interacting_obj), pixels=12)
window = [failure_onset - delta_pre, failure_onset + delta_post]
```

说明：

```text
JEPA distillation 若只打单个局部框，通常看不到足够的物理关系上下文。
```

---

## 9. Failure score 与 onset mining

文件：`physfold/mining/mine_failures.py`

继续保留四类高置信 failure score：

```text
1. object permanence score
2. contact consistency score
3. collision response score
4. gravity / falling score
```

配置建议：

```yaml
mining:
  min_detector_confidence: 0.70
  min_failure_score: 0.65
  consecutive_frames: 2
  ignore_segment_boundary_margin: 2

window:
  delta_pre: 6
  delta_post: 18
  prefix_gap: 3

filters:
  min_visible_frames_before_failure: 8
  min_object_area: 64
  max_camera_motion: null
```

onset 规则：

```python
def find_failure_onset(frame_scores, threshold, consecutive_frames):
    for t in range(len(frame_scores) - consecutive_frames + 1):
        if all(frame_scores[k].total_score >= threshold for k in range(t, t + consecutive_frames)):
            return t
    return None
```

必须：

```text
不要取 argmax。
first crossing 才是 failure onset。
```

---

## 10. A 分支：positive continuation 获取与筛选

文件：

```text
physfold/correction/resample_continuations.py
physfold/correction/rank_continuations.py
physfold/correction/build_local_pairs.py
```

### 10.1 输入

```text
failures.jsonl
原始 rollout video
prefix_end = failure_onset - prefix_gap
```

### 10.2 positive 候选来源

```text
1. same-prefix best-of-K continuation
2. stronger teacher model continuation
3. offline good continuation
```

### 10.3 排序原则

```text
total_score =
  w_phy * physics_score
+ w_sem * semantic_score
+ w_vis * visual_quality_score
+ w_pre * prefix_consistency_score
```

推荐权重：

```yaml
ranking:
  weights:
    physics: 0.50
    semantic: 0.15
    visual_quality: 0.15
    prefix_consistency: 0.20
  min_margin: 0.20
  min_positive_score: 0.60
  max_negative_score: 0.55
```

### 10.4 A 分支选择规则

```python
negative = original_failed_continuation
positive = best candidate

if positive_score >= min_positive_score and positive_score - negative_score >= min_margin:
    emit branch A sample
else:
    no branch A sample
```

注意：

```text
A 分支不是必须存在。
采不到高置信 positive 时，直接转 B 分支，不强行硬造 positive。
```

---

## 11. B 分支：failure-localized V-JEPA predictive distillation

文件：

```text
physfold/vjepa/build_teacher_targets.py
physfold/vjepa/encode_failed_regions.py
physfold/vjepa/projector.py
physfold/vjepa/losses.py
```

### 11.1 B 分支输入

```text
failed rollout video x_bad
failure window
local relational region mask
prefix and visible context
```

### 11.2 teacher 输出

teacher 不输出视频，只输出未来局部正向表征：

```text
f_teacher^+ = V-JEPA predicted future local representation
```

输入给 teacher 的上下文必须尽量避免泄漏：

```text
1. prefix frames
2. failure onset 前上下文
3. failure region 外的可见上下文
4. optional masked future context
```

不建议：

```text
直接把原失败区域完整 future 喂给 teacher，再把其输出当正向目标
```

### 11.3 failed feature

可选编码：

```text
f_bad = V-JEPA encode(原失败区域)
```

用途：

```text
提供相对约束，而不是伪造像素级负样本
```

### 11.4 student 对齐位置

student 仍然是 video generator。

从 student 中取：

```text
1. 高层 temporal blocks hidden
2. 只取 1-2 个层位
3. 经 projector 映射到 JEPA feature space
```

得到：

```text
f_student = projector(h_student_local)
```

不建议：

```text
1. 直接对齐 raw hidden
2. 在所有层全量对齐
3. 在全图 / 全时段广播 loss
```

### 11.5 B 分支损失

正向对齐项：

```text
L_jepa_pos = 1 - cosine(f_student, f_teacher^+)
```

相对约束项，推荐优先于“强制远离 f_bad”：

```text
L_jepa_rel = max(
  0,
  margin - sim(f_student, f_teacher^+) + sim(f_bad, f_teacher^+)
)
```

或更直接的 student-centered 对比式：

```text
L_jepa_ctr = max(
  0,
  margin - sim(f_student, f_teacher^+) + sim(f_student, f_bad)
)
```

第一版建议：

```text
先做 cosine positive alignment
第二步再加 relative / contrastive 项
```

### 11.6 B 分支的定位

```text
B 分支是 predictive representation regularization
不是 positive video 替代品
不是 JEPA-to-VAE inversion
不是单独的视频解码器训练
```

---

## 12. 统一训练目标

### 12.1 情况 A：有 positive

训练主目标：

```text
L_video = local flow matching / local SFT / optional local DPO
```

只在 failure-localized region 生效：

```text
loss mask = time mask × region mask
```

### 12.2 情况 B：无 positive

训练主目标：

```text
L_jepa = lambda_pos * L_jepa_pos + lambda_rel * L_jepa_rel
```

### 12.3 混合训练

统一形式：

```text
L_total = L_base_gen
        + 1[A] * lambda_video * L_video
        + 1[B] * lambda_jepa * L_jepa
        + lambda_boundary * L_boundary_consistency
        + lambda_keep * L_nonfailure_preserve
```

说明：

```text
1. A 和 B 可以按样本分支切换
2. 即使 A 存在，也可小权重叠加 B 作为 regularizer
3. B 绝不替代原始生成目标，只做局部正则
```

### 12.4 非 failure 保持项

为了避免局部对齐损坏全局视频质量，需要加入：

```text
L_nonfailure_preserve
```

目标：

```text
不让 non-failure 区域被局部 loss 带坏
```

可选形式：

```text
1. non-failure mask 上的 reconstruction / consistency
2. 对未遮罩区域的小权重保持约束
3. reference model regularization
```

---

## 13. 训练配置建议

文件：`physfold/configs/training.yaml`

```yaml
training:
  method: "local_ab"
  train_lora: true
  lora_rank: 16
  lora_alpha: 16
  learning_rate: 2.0e-5
  batch_size: 1
  grad_accum_steps: 8
  max_steps: 3000
  mixed_precision: "bf16"
  freeze_vae: true
  freeze_text_encoder: true
  align_layers: ["temporal_block_10", "temporal_block_14"]

loss:
  use_time_mask: true
  use_region_mask: true
  lambda_video: 1.0
  lambda_jepa: 0.2
  lambda_jepa_pos: 1.0
  lambda_jepa_rel: 0.0
  lambda_boundary: 0.1
  lambda_keep: 0.1
  sample_weight_by_margin: true

data_mix:
  branch_a_ratio: 0.50
  branch_b_ratio: 0.30
  generic_video_ratio: 0.20
```

文件：`physfold/configs/vjepa.yaml`

```yaml
vjepa:
  model_name: "vjepa"
  pooling: "masked_spatiotemporal_avg"
  feature_dim: 1024
  projector_hidden_dim: 2048
  projector_out_dim: 1024
  teacher_confidence_threshold: 0.65
  use_relative_loss: false
  stop_gradient_teacher: true
```

---

## 14. 训练阶段划分

### 14.1 Stage 0：只做闭环，不做真实训练

必须实现：

```text
1. prompt bank
2. rollout generation
3. perception extraction
4. failure mining
5. local relational region
6. A/B sample manifest 输出
7. mock-mode 端到端跑通
```

### 14.2 Stage 1：只做 A 分支

目标：

```text
验证有 positive 时，failure-localized video supervision 是否优于 global/random-window
```

### 14.3 Stage 2：只做 JEPA 可分性验证

在真正训练前先做：

```text
1. failed vs less-failed continuation feature separability
2. f_teacher^+ 与 f_bad 的 margin 分析
3. 不同 failure 类型的 feature 判别力统计
```

如果这一步不成立，不进入 B 分支训练。

### 14.4 Stage 3：引入 B 分支

目标：

```text
验证在无 positive 的样本上，B 分支是否能提供额外收益
```

### 14.5 Stage 4：A+B 联合训练

目标：

```text
验证 A+B 是否优于只做 A
```

---

## 15. Evaluation

文件：

```text
physfold/eval/eval_physics.py
physfold/eval/eval_visual_semantic.py
physfold/eval/eval_jepa_features.py
```

### 15.1 必须评估的 baseline

```text
1. Base Wan
2. Random-window local SFT
3. Global SFT on positive continuation
4. A only: failure-window video supervision
5. B only: JEPA predictive distillation
6. A + B: unified local training
```

如果条件允许，再加：

```text
7. Offline SFT on good continuations
8. Strong teacher continuation only
```

### 15.2 物理指标

```text
object_persistence_error_rate
contact_flicker_rate
collision_no_response_rate
collision_wrong_direction_rate
falling_reverse_rate
trajectory_jump_rate
overall_failure_rate
```

### 15.3 视觉 / 语义指标

```text
CLIP text-video similarity
aesthetic score
blur / artifact score
temporal consistency score
```

### 15.4 JEPA 相关诊断

新增：

```text
1. failed vs positive feature margin
2. failed vs teacher-target cosine gap
3. student feature alignment gain after training
4. B 分支样本的 teacher confidence 分布
```

### 15.5 评估注意事项

```text
1. 尽量避免用完全同构的 heuristic 同时做 mining、ranking、final eval
2. 至少保留一小批人工复核样本
3. B 分支需要单独验证其对不同 failure 类型是否都有效
```

---

## 16. Scripts

### 16.1 Build prompt bank

```bash
python -m physfold.prompts.build_prompt_bank \
  --config physfold/configs/prompt_bank.yaml \
  --output data/prompts/physics_prompts.jsonl \
  --num-per-category 200
```

### 16.2 Generate rollouts

```bash
python -m physfold.rollout.generate_rollouts \
  --prompt-jsonl data/prompts/physics_prompts.jsonl \
  --output-dir data/rollouts \
  --backend wan_vace \
  --num-frames 96 \
  --fps 12 \
  --segments 3 \
  --prefix-frames 8 \
  --num-seeds 2
```

### 16.3 Extract perception

```bash
python -m physfold.perception.extract_masks \
  --rollouts data/rollouts/rollouts.jsonl \
  --output-dir data/perception \
  --model sam2

python -m physfold.perception.track_points \
  --perception data/perception/perception.jsonl \
  --output-dir data/perception/tracks \
  --model cotracker

python -m physfold.perception.estimate_depth \
  --rollouts data/rollouts/rollouts.jsonl \
  --output-dir data/perception/depth \
  --model depth_anything

python -m physfold.perception.estimate_flow \
  --rollouts data/rollouts/rollouts.jsonl \
  --output-dir data/perception/flow \
  --model raft
```

### 16.4 Mine failures

```bash
python -m physfold.mining.mine_failures \
  --rollouts data/rollouts/rollouts.jsonl \
  --perception data/perception/perception.jsonl \
  --output data/mining/failures.jsonl \
  --config physfold/configs/mining.yaml
```

### 16.5 Resample continuations for A branch

```bash
python -m physfold.correction.resample_continuations \
  --failures data/mining/failures.jsonl \
  --output-dir data/correction \
  --backend wan_vace \
  --num-candidates 4 \
  --continuation-frames 48 \
  --prefix-frames 8

python -m physfold.correction.rank_continuations \
  --candidates data/correction/candidates.jsonl \
  --output data/correction/local_pairs.jsonl \
  --config physfold/configs/correction.yaml
```

### 16.6 Build V-JEPA teacher targets for B branch

```bash
python -m physfold.vjepa.build_teacher_targets \
  --failures data/mining/failures.jsonl \
  --rollouts data/rollouts/rollouts.jsonl \
  --output-dir data/vjepa/teacher_targets \
  --config physfold/configs/vjepa.yaml

python -m physfold.vjepa.encode_failed_regions \
  --failures data/mining/failures.jsonl \
  --output-dir data/vjepa/failed_features \
  --config physfold/configs/vjepa.yaml
```

### 16.7 Build unified training manifest

```bash
python -m physfold.training.build_training_manifest \
  --pairs data/correction/local_pairs.jsonl \
  --failures data/mining/failures.jsonl \
  --teacher-target-dir data/vjepa/teacher_targets \
  --failed-feature-dir data/vjepa/failed_features \
  --output data/training/training_manifest.jsonl
```

### 16.8 Train A/B model

```bash
python -m physfold.training.train_local_ab \
  --dataset data/training/training_manifest.jsonl \
  --config physfold/configs/training.yaml \
  --output-dir outputs/phyfold_local_ab
```

### 16.9 Eval

```bash
python -m physfold.eval.eval_physics \
  --prompt-jsonl data/prompts/physics_prompts_eval.jsonl \
  --methods base outputs/phyfold_local_ab \
  --output outputs/eval/physics_metrics.json
```

---

## 17. 单元测试要求

必须先实现 tests，不依赖真实视频模型。

### 17.1 `test_failure_scores.py`

synthetic tube case：

```text
1. object area suddenly drops to zero -> permanence score high
2. collision target does not move after contact -> collision score high
3. falling object moves upward after unsupported -> falling score high
4. normal smooth motion -> all scores low
```

### 17.2 `test_tubes.py`

检查：

```text
- center 计算正确
- velocity / acceleration 计算正确
- visible flag 正确
- smoothing 不改变 shape
```

### 17.3 `test_region_masks.py`

检查：

```text
- failure object + interacting object union 正确
- dilation 后区域不越界
- temporal window 映射正确
```

### 17.4 `test_jepa_targets.py`

检查：

```text
- teacher target shape 正确
- failed feature shape 正确
- projector 输出维度正确
- 无信息泄漏的 mock 输入协议正确
```

### 17.5 `test_schema.py`

检查所有 jsonl schema 必须包含必需字段。

### 17.6 `test_local_dataset.py`

检查：

```text
- A/B manifest 路由正确
- local window 从 full video frame index 正确映射到 continuation index
- time mask shape 正确
- region mask shape 正确
- score_margin / confidence filtering 生效
```

运行：

```bash
pytest tests -q
```

---

## 18. 实施优先级

### P0：必须完成

```text
1. prompt bank builder
2. jsonl schema / dataclass
3. object tube builder
4. local relational region builder
5. failure score functions + unit tests
6. mine_failures CLI
7. continuation candidate ranking
8. unified training manifest builder
9. mock-mode end-to-end demo
```

### P1：强烈建议完成

```text
1. rollout backend 抽象
2. perception wrapper mock mode
3. A 分支 resample continuation CLI
4. JEPA target builder mock mode
5. eval_physics 自动评估
```

### P2：模型训练相关

```text
1. A 分支 local SFT / flow matching 接入
2. projector + B 分支 JEPA loss 接入
3. A+B 混合训练
4. non-failure preserve regularization
```

### P3：第二阶段增强

```text
1. B 分支 relative / contrastive loss
2. stronger teacher continuation backend
3. object-aware dynamic masks
4. multi-round on-policy loop
5. optional local DPO
```

---

## 19. 第一版不要做的事

```text
1. 不要做 full fine-tuning
2. 不要先做 RL
3. 不要覆盖液体/布料/破碎
4. 不要依赖人工逐帧标注
5. 不要让 VLM 成为唯一 scorer
6. 不要用 argmax failure score 当 onset
7. 不要训练低置信 failure pair
8. 不要把整个视频都作为 loss 区域
9. 不要做 JEPA-to-VAE latent positive 反演
10. 不要直接对齐 raw DiT hidden
```

---

## 20. 关键可行性假设

本版只需要验证以下假设：

```text
H1. student long rollout 中存在可自动检测的高置信物理失败。
H2. same-prefix continuation 中有一部分样本可作为 A 分支 positive。
H3. 即使没有 concrete positive，V-JEPA 也能对 failure-localized future 提供有判别力的目标表征。
H4. 只在 failure-localized region 上做 A/B 训练，比 global SFT / random-window 更有效。
H5. 局部训练不会显著损伤视觉质量和 prompt adherence。
```

若 H2 不成立：

```text
1. 增大 K
2. 从更早 prefix 重采样
3. 使用 stronger teacher continuation
4. 提高 A 分支阈值，更多样本走 B 分支
```

若 H3 不成立：

```text
1. 先不启用 B 分支训练
2. 保留 JEPA 只做 reranker / diagnostic scorer
3. 重新检查 region mask 与 failure type 的匹配
4. 只在部分 failure 类型上启用 B 分支
```

---

## 21. 最终产物

应交付：

```text
1. 可运行的 physfold Python package
2. scripts/00-07 全流程脚本
3. tests 全部通过
4. sample jsonl 文件
5. 至少一个 mock-mode end-to-end demo
6. 若环境可用，至少 100 条真实 rollout 的 failure mining 结果
7. A/B training manifest 样例
8. eval summary json + markdown report
9. JEPA separability diagnostic report
```

验收命令：

```bash
pytest tests -q

bash scripts/00_build_prompts.sh
bash scripts/01_generate_rollouts.sh
bash scripts/02_extract_perception.sh
bash scripts/03_mine_failures.sh
bash scripts/04_resample_continuations.sh
bash scripts/05_build_training_manifest.sh
bash scripts/07_eval.sh
```

训练命令可单独验收：

```bash
bash scripts/06_train_local_ab.sh
```

如果没有 GPU / 外部模型，必须支持 mock mode 跑通：

```bash
PHYFOLD_MOCK=1 bash scripts/00_build_prompts.sh
PHYFOLD_MOCK=1 bash scripts/01_generate_rollouts.sh
PHYFOLD_MOCK=1 bash scripts/02_extract_perception.sh
PHYFOLD_MOCK=1 bash scripts/03_mine_failures.sh
PHYFOLD_MOCK=1 bash scripts/04_resample_continuations.sh
PHYFOLD_MOCK=1 bash scripts/05_build_training_manifest.sh
PHYFOLD_MOCK=1 bash scripts/07_eval.sh
```
