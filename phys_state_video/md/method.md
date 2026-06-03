# Method

## 版本管理约定

1. 以后每新增一版方案，必须保留之前方案的可复现接口，不能直接删除或覆盖旧方案的关键实现；如果新方案需要大改接口，应复制出新的脚本或代码路径，在新文件上继续修改。
2. 每新增一版方案，都要在本文件中新增一个独立小节，写清楚版本日期、版本名、核心改动、输出目录和可视化入口。
3. 每版方法说明都必须至少包含四项：
   `predictor 输入`、`predictor 输出`、`视频生成模型的 condition`、`condition 进入视频模型的形式`。
4. `condition 进入视频模型的形式` 需要明确写是：
   `token 维拼接 / cross-attention memory / ControlNet-style 空间条件 / adapter 注入 / 其它`，
   不能只写“作为条件输入”。

## 2026-06-02 baseline_v1 显式状态条件基线

### 1. 方法流程

这版基线方法采用显式 object state 作为主条件：给定前 `K` 帧 context 视频及其物体级状态，先将每个样本整理为 `context_frames ∈ R^{B×K×3×H×W}`、`context_states ∈ R^{B×K×N×10}`、`appearance ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt；`Future State Predictor` 直接基于 `context_states + appearance + camera + prompt` 预测未来显式状态 `future_states ∈ R^{B×T×N×10}`，其中 10 维状态为 `center_x, center_y, depth, log_scale, vel_x, vel_y, depth_vel, visibility, existence, confidence`；随后把 `future_states` 转成 `condition_maps ∈ R^{B×T×C_map×H×W}`，包括 heatmap、bbox、depth、visibility、velocity map，同时从 `appearance` 和末帧尺度等信息构造 `memory_tokens`；最后 `State-Conditioned Video Adapter` 接收 `context_frames`、`condition_maps` 和 `memory_tokens` 进行未来视频重建，输出 `generated_frames ∈ R^{B×T×3×H×W}`，训练时主要依赖视频重建损失、显式 state auxiliary loss 和 spatial auxiliary loss，使视频模型尽量服从显式位置、尺度、深度和可见性条件。

### 1.1 Predictor 输入

`predictor` 输入为 `context_states ∈ R^{B×K×N×10}`、`appearance ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt；当前这版不直接把 `context_frames` 输入给 predictor，而是依赖已经抽取好的 object-level state 与外观向量。

### 1.2 Predictor 输出

`predictor` 输出为未来显式状态 `future_states ∈ R^{B×T×N×10}`，10 维分别表示 `center_x, center_y, depth, log_scale, vel_x, vel_y, depth_vel, visibility, existence, confidence`；这版没有额外的 future latent 输出。

### 1.3 视频生成模型的 Condition

视频生成模型接收三类 condition：
`context_frames ∈ R^{B×K×3×H×W}`、显式空间条件 `condition_maps ∈ R^{B×T×C_map×H×W}`、以及物体级 `memory_tokens`；其中 `condition_maps` 由 `future_states` 投影得到，包含 heatmap、bbox、depth、visibility、velocity 等空间图，`memory_tokens` 主要包含 `appearance` 和末帧尺度置信信息。

### 1.4 Condition 进入视频模型的形式

这版是 `空间图 + memory token + adapter` 的形式：`condition_maps` 先经过卷积编码器，以 `ControlNet-style / spatial adapter-style` 的方式和 context latent 做逐帧相加融合；`memory_tokens` 再作为 `cross-attention memory` 输入到 `StateCrossAttentionAdapter` 中，对视频 latent token 做条件注入。换句话说，这版不是把状态直接拼到 token 维度，而是“空间条件走卷积支路，物体身份条件走 cross-attention adapter”。

### 2. 关键实现

- `显式 Future State Predictor`
  路径：[predictor.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor.py)
  关键函数：`FutureStatePredictor.forward()`、`predictor_loss()`

- `State / Box rollout 与端到端生成流程`
  路径：[pipeline.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/pipeline.py)
  关键函数：`rollout_boxes_from_states()`、`StateConditionedGenerationPipeline.predict_states()`、`StateConditionedGenerationPipeline.generate()`

- `显式 condition maps 构造`
  路径：[conditioning.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/conditioning.py)
  关键函数：`build_condition_bundle()`

- `State-Conditioned Video Adapter`
  路径：[adapter.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/adapter.py)
  关键函数：`TinyVideoBackbone.forward()`、`adapter_loss()`

- `Predictor 训练入口`
  路径：[train_predictor.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor.py)
  关键函数：`run_epoch()`、`main()`

- `Adapter 训练入口`
  路径：[train_adapter.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_adapter.py)
  关键函数：`run_epoch()`、`main()`

- `推理入口`
  路径：[run_inference.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference.py)
  关键函数：`main()`

### 3. 输出目录与可视化指令

- `训练输出目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_baseline_v1`

- `checkpoint 目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_baseline_v1/checkpoints`

- `best ckpt case 可视化目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_baseline_v1/viz/trained_cases_v1`

- `best ckpt case 本地端口`
  地址：`http://127.0.0.1:18832`

## 2026-06-03 predictor预测隐式状态，latent接head做监督

### 1. 方法流程

整个方法可以写成一条链：给定前 `K` 帧 context 视频及其物体级状态，先把每个样本整理成 `context_frames ∈ R^{B×K×3×H×W}`、`context_states ∈ R^{B×K×N×S}`、`appearance/physics ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt，其中当前实现里 `S=10`；`Future Latent Predictor` 以 `context_states + appearance/physics + camera + prompt` 为输入，先编码成历史隐状态，再为每个未来时刻、每个物体预测 `future_latents ∈ R^{B×T×N×D}`，当前可设 `D=128`，并从这些 latent 上接显式监督 head，得到 `states ∈ R^{B×T×N×10}` 和 `motion ∈ R^{B×T×N×3}`，其中 `states` 包含 `center/depth/log_scale/visibility` 等可解释变量；随后把显式状态投影成空间条件 `condition_maps ∈ R^{B×T×C_map×H×W}`，例如 heatmap、bbox、depth、visibility、velocity map，同时把 `future_latents` 直接作为 object-temporal memory token；最后 `State/Latent-Conditioned Video Adapter` 接收 `context_frames`、`condition_maps`、`memory_tokens` 以及 `future_latent_tokens`，在内部把像素特征从 `R^{B×K×3×H×W}` 编到时空 latent，再与 `R^{B×T×N×D}` 的未来物体 latent 做 cross-attention 融合，输出未来视频 `generated_frames ∈ R^{B×T×3×H×W}`，训练时同时用视频重建损失和 latent 上各个显式 head 的监督损失约束，使模型既保留可解释的物体状态控制，又能用高带宽 latent 表达接触相位、姿态变化和复杂动力学。

### 1.1 Predictor 输入

`predictor` 输入为 `context_states ∈ R^{B×K×N×10}`、`appearance/physics ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt；其中 `appearance/physics` 里已经编码了物体类别、颜色、尺寸、质量、摩擦等静态属性。

### 1.2 Predictor 输出

`predictor` 主输出为 `future_latents ∈ R^{B×T×N×128}`；同时从这些 latent 上接显式监督 head，输出 `states ∈ R^{B×T×N×10}` 和 `motion ∈ R^{B×T×N×3}`。这里 `future_latents` 是高带宽隐式未来状态，`states/motion` 是可解释监督分支。

### 1.3 视频生成模型的 Condition

视频生成模型接收 `context_frames ∈ R^{B×K×3×H×W}`、显式空间条件 `condition_maps ∈ R^{B×T×7×H×W}`、`memory_tokens`、`future_latent_tokens ∈ R^{B×T×N×128}`、以及由 `context_states` 和 prompt 编码得到的额外 memory token；其中 `condition_maps` 来自显式 `states` 投影，`future_latent_tokens` 来自 predictor 的主输出。

### 1.4 Condition 进入视频模型的形式

这版是 `空间图 + future latent cross-attention + context/prompt adapter` 的形式：`condition_maps` 走卷积条件支路，与 context latent 逐帧相加；`future_latent_tokens` 作为 `cross-attention memory` 按时间步注入到视频 token 中；`context_states` 和 prompt 也会编码成 token，经由同一个 `StateCrossAttentionAdapter` 作为额外 memory 一起参与 attention。也就是说，这版仍然保留 `ControlNet-style / spatial adapter-style` 显式图条件，但新增了一个更强的 `future latent adapter` 支路。

### 2. 关键实现

- `Future Latent Predictor`
  路径：[predictor.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor.py)
  关键函数：`FutureStatePredictor.forward()`、`predictor_loss()`

- `State / Box rollout 与端到端生成流程`
  路径：[pipeline.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/pipeline.py)
  关键函数：`rollout_boxes_from_states()`、`StateConditionedGenerationPipeline.predict_states()`、`StateConditionedGenerationPipeline.generate()`

- `显式 condition maps 构造`
  路径：[conditioning.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/conditioning.py)
  关键函数：`build_condition_bundle()`

- `State / Latent-Conditioned Video Adapter`
  路径：[adapter.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/adapter.py)
  关键函数：`StateCrossAttentionAdapter.forward()`、`TinyVideoBackbone.forward()`、`adapter_loss()`

- `Predictor 训练入口`
  路径：[train_predictor.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor.py)
  关键函数：`run_epoch()`、`main()`

- `Adapter 训练入口`
  路径：[train_adapter.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_adapter.py)
  关键函数：`run_epoch()`、`main()`

- `推理入口`
  路径：[run_inference.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference.py)
  关键函数：`main()`

- `仿真 episode 导出`
  路径：[prepare_sim_episodes.py](/home/gaoya/Code_Video/phys_state_video/scripts/prepare_sim_episodes.py)
  关键函数：`appearance_vector()`、`build_camera_vector()`、`projected_box_from_object()`、`main()`

### 3. 输出目录与可视化指令

- `episode 数据目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6`

- `当前方法训练输出目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1`

- `checkpoint 目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/checkpoints`

- `训练日志目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/logs`

- `评估结果目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/eval`

- `训练中 ckpt 可视化目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/viz/training_ckpts`

- `启动训练指令`

```bash
tmux new-session -d -s phys_state_latent_v1_train \
  'bash -lc "/home/gaoya/Code_Video/phys_state_video/scripts/run_industrial_s1_scale2_latent_v1.sh"'
```

- `启动训练中 ckpt 可视化 watcher 指令`

```bash
tmux new-session -d -s phys_state_latent_v1_viz \
  'bash -lc "/data/gaoya/miniconda3/envs/vjepa2/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/watch_adapter_cases.py \
  --env-py /data/gaoya/miniconda3/envs/vjepa2/bin/python \
  --project-root /home/gaoya/Code_Video/phys_state_video \
  --episode-root /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6 \
  --predictor-checkpoint /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/checkpoints/predictor_best.pt \
  --checkpoint-dir /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/checkpoints \
  --output-root /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v1/viz/training_ckpts \
  --port 18833 \
  --poll-seconds 180 \
  --max-cases 12 \
  --fps 6 \
  --device cuda"'
```

- `训练中 ckpt 可视化本地端口`
  地址：`http://127.0.0.1:18833`

- `查看训练 tmux 输出`

```bash
tmux capture-pane -pt phys_state_latent_v1_train:0 | tail -n 80
tmux capture-pane -pt phys_state_latent_v1_viz:0 | tail -n 80
```

## 2026-06-03 latent_v2 显式状态仅监督，视频生成主条件改为 future latent

### 1. 方法流程

这版方法把显式 object state 从“视频生成主条件”降级为“监督信号”：给定前 `K` 帧 context 视频及其物体级状态，先整理成 `context_frames ∈ R^{B×K×3×H×W}`、`context_states ∈ R^{B×K×N×10}`、`appearance/physics ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt；`Future Latent Predictor` 先基于 `context_states + appearance/physics + camera + prompt` 预测未来物体 latent `future_latents ∈ R^{B×T×N×128}`，并从 latent 上接显式 head 输出 `states ∈ R^{B×T×N×10}` 与 `motion ∈ R^{B×T×N×3}`，这些显式量只用来和 GT 做 predictor 监督；随后仍然可把 `states` 投影成 `condition_maps ∈ R^{B×T×7×H×W}` 用于可视化和 adapter 的 spatial auxiliary target，但在真正喂给视频模型时，会通过 `latent_only` 模式把 future state maps 置零，并把 memory token 中末帧 `log_scale/confidence` 显式分量清掉，只保留 object identity / shape / color / size / mass / friction 等 object memory；最终 `State/Latent-Conditioned Video Adapter` 主要接收 `context_frames`、`future_latent_tokens ∈ R^{B×T×N×128}`、`object memory tokens`、`context state tokens` 和 `prompt tokens` 做 cross-attention 融合，输出 `generated_frames ∈ R^{B×T×3×H×W}`，训练时再通过视频重建损失、adapter state auxiliary loss 和 spatial auxiliary loss 把生成结果拉回到正确的物体轨迹、尺度与可见性上。

### 1.1 Predictor 输入

`predictor` 输入与 `latent_v1` 相同，为 `context_states ∈ R^{B×K×N×10}`、`appearance/physics ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt；这一版没有删除旧 predictor 接口，只改变了视频生成器使用 predictor 输出的方式。

### 1.2 Predictor 输出

`predictor` 主输出仍然是 `future_latents ∈ R^{B×T×N×128}`，同时输出显式监督分支 `states ∈ R^{B×T×N×10}` 与 `motion ∈ R^{B×T×N×3}`；其中 `states/motion` 在这版里主要用于和 GT 做监督，以及做可视化诊断，不再是视频生成器的主条件。

### 1.3 视频生成模型的 Condition

视频生成模型主 condition 为 `context_frames ∈ R^{B×K×3×H×W}`、`future_latent_tokens ∈ R^{B×T×N×128}`、`object memory tokens`、`context state tokens` 和 `prompt tokens`；显式 `condition_maps ∈ R^{B×T×7×H×W}` 只作为辅助监督 target 和调试可视化存在，不再作为主生成条件输入。

### 1.4 Condition 进入视频模型的形式

这版是 `future latent cross-attention + object/context/prompt adapter` 为主的形式：`future_latent_tokens` 按时间步作为 `cross-attention memory` 注入视频 latent token；`object memory tokens`、`context_states` 编码 token、prompt token 也一起作为 `StateCrossAttentionAdapter` 的 memory 参与 attention；显式 `condition_maps` 在 `latent_only` 模式下会被清零，所以不再承担 `ControlNet-style` 的主控制作用。换句话说，这版的生成主链路不是“空间图控制”，而是“future latent token 通过 adapter/cross-attention 控制”。

### 2. 关键实现

- `Future Latent Predictor`
  路径：[predictor.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor.py)
  关键函数：`FutureStatePredictor.forward()`、`predictor_loss()`

- `条件模式切换`
  路径：[experiment.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/experiment.py)
  关键函数：`apply_condition_mode()`

- `端到端生成流程`
  路径：[pipeline.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/pipeline.py)
  关键函数：`StateConditionedGenerationPipeline.generate()`

- `State / Latent-Conditioned Video Adapter`
  路径：[adapter.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/adapter.py)
  关键函数：`StateCrossAttentionAdapter.forward()`、`TinyVideoBackbone.forward()`、`adapter_loss()`

- `Adapter 训练入口`
  路径：[train_adapter.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_adapter.py)
  关键函数：`run_epoch()`、`main()`

- `Adapter 评估入口`
  路径：[evaluate_adapter.py](/home/gaoya/Code_Video/phys_state_video/scripts/evaluate_adapter.py)
  关键函数：`main()`

- `case 导出与可视化`
  路径：[export_trained_cases.py](/home/gaoya/Code_Video/phys_state_video/scripts/export_trained_cases.py)
  关键函数：`main()`、`render_html()`

### 3. 输出目录与可视化指令

- `episode 数据目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6`

- `当前方法训练输出目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2`

- `checkpoint 目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/checkpoints`

- `训练日志目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/logs`

- `评估结果目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/eval`

- `训练中 ckpt 可视化目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/viz/training_ckpts`

- `启动训练指令`

```bash
tmux new-session -d -s phys_state_latent_v2_train \
  'bash -lc "/home/gaoya/Code_Video/phys_state_video/scripts/run_industrial_s1_scale2_latent_v2.sh"'
```

- `启动训练中 ckpt 可视化 watcher 指令`

```bash
tmux new-session -d -s phys_state_latent_v2_viz \
  'bash -lc "/data/gaoya/miniconda3/envs/vjepa2/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/watch_adapter_cases.py \
  --env-py /data/gaoya/miniconda3/envs/vjepa2/bin/python \
  --project-root /home/gaoya/Code_Video/phys_state_video \
  --episode-root /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6 \
  --predictor-checkpoint /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/checkpoints/predictor_best.pt \
  --checkpoint-dir /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/checkpoints \
  --output-root /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_latent_v2/viz/training_ckpts \
  --port 18835 \
  --poll-seconds 180 \
  --max-cases 12 \
  --fps 6 \
  --device cuda"'
```

- `训练中 ckpt 可视化本地端口`
  地址：`http://127.0.0.1:18835`

- `查看训练 tmux 输出`

```bash
tmux capture-pane -pt phys_state_latent_v2_train:0 | tail -n 80
tmux capture-pane -pt phys_state_latent_v2_viz:0 | tail -n 80
```

## 统一可视化入口与跨方法对比

### 1. 页面说明

统一可视化页面用于汇总当前所有方法的 case 入口，并保留一个固定的对比页面，对比各个方法 best ckpt 在同一批代表性 case 上的结果；当前总页面已接入 `baseline_v1` 和 `latent_v1`，其中每个方法都提供各自的 best ckpt case 页面，`latent_v1` 额外保留训练中不同 ckpt 的时间线页面；统一对比页固定展示相同的 context、GT future、generated future 和 predicted conditions，便于直接比较不同方案在同一批样本上的轨迹、尺度和外观行为差异。

### 2. 输出目录

- `统一总入口页目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/method_overview_v1`

- `跨方法 best ckpt 对比页目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/method_overview_v1/compare_best`

- `方法入口软链接目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/method_overview_v1/methods`

### 3. 生成与启动指令

```bash
PORT=18834
ROOT=/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/method_overview_v1
LISTEN_PID=$(ss -ltnp | awk '/:18834 / {print $NF}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1)
if [ -n "$LISTEN_PID" ]; then kill "$LISTEN_PID"; fi
rm -f "$ROOT/http_${PORT}.pid"
/data/gaoya/miniconda3/envs/vjepa2/bin/python \
  /home/gaoya/Code_Video/phys_state_video/scripts/generate_method_overview_page.py \
  --clean \
  --port 18834
```

### 4. 本地端口入口

- `统一方法总入口页`
  地址：`http://127.0.0.1:18834`

- `best ckpt 同批 case 对比页`
  地址：`http://127.0.0.1:18834/compare_best/index.html`

### 5. 当前已接入的方法

- `baseline_v1`
  入口：`http://127.0.0.1:18834/methods/baseline_v1_best/index.html`

- `latent_v1 best ckpt`
  入口：`http://127.0.0.1:18834/methods/latent_v1_best/index.html`

- `latent_v1 训练中 ckpt 时间线`
  入口：`http://127.0.0.1:18834/methods/latent_v1_timeline/index.html`
