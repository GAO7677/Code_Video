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

### 1.1 Predictor 输入输出

`predictor` 输入为 `context_states ∈ R^{B×K×N×10}`、`appearance ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt；当前这版不直接把 `context_frames` 输入给 predictor，而是依赖已经抽取好的 object-level state 与外观向量。`predictor` 输出为未来显式状态 `future_states ∈ R^{B×T×N×10}`，10 维分别表示 `center_x, center_y, depth, log_scale, vel_x, vel_y, depth_vel, visibility, existence, confidence`；这版没有额外的 future latent 输出。

### 1.2 视频生成模型的 Condition + 条件注入

视频生成模型接收三类 condition：`context_frames ∈ R^{B×K×3×H×W}`、显式空间条件 `condition_maps ∈ R^{B×T×C_map×H×W}`、以及物体级 `memory_tokens`；其中 `condition_maps` 由 `future_states` 投影得到，包含 heatmap、bbox、depth、visibility、velocity 等空间图，`memory_tokens` 主要包含 `appearance` 和末帧尺度置信信息。这版的条件注入形式是 `空间图 + memory token + adapter`：`condition_maps` 先经过卷积编码器，以 `ControlNet-style / spatial adapter-style` 的方式和 context latent 做逐帧相加融合；`memory_tokens` 再作为 `cross-attention memory` 输入到 `StateCrossAttentionAdapter` 中，对视频 latent token 做条件注入。换句话说，这版不是把状态直接拼到 token 维度，而是“空间条件走卷积支路，物体身份条件走 cross-attention adapter”。

### 1.3 可训练模块

这版中可训练模块主要包括两部分：`Future State Predictor` 整体可训练，用于从历史 object state 预测未来显式状态；`State-Conditioned Video Adapter` 及其内部的条件编码器、cross-attention adapter、decoder、state/spatial 辅助头可训练，用于在给定显式 condition 的情况下重建未来视频。

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

### 1.1 Predictor 输入输出

`predictor` 输入为 `context_states ∈ R^{B×K×N×10}`、`appearance/physics ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt；其中 `appearance/physics` 里已经编码了物体类别、颜色、尺寸、质量、摩擦等静态属性。`predictor` 主输出为 `future_latents ∈ R^{B×T×N×128}`；同时从这些 latent 上接显式监督 head，输出 `states ∈ R^{B×T×N×10}` 和 `motion ∈ R^{B×T×N×3}`。这里 `future_latents` 是高带宽隐式未来状态，`states/motion` 是可解释监督分支。

### 1.2 视频生成模型的 Condition + 条件注入

视频生成模型接收 `context_frames ∈ R^{B×K×3×H×W}`、显式空间条件 `condition_maps ∈ R^{B×T×7×H×W}`、`memory_tokens`、`future_latent_tokens ∈ R^{B×T×N×128}`、以及由 `context_states` 和 prompt 编码得到的额外 memory token；其中 `condition_maps` 来自显式 `states` 投影，`future_latent_tokens` 来自 predictor 的主输出。这版的条件注入形式是 `空间图 + future latent cross-attention + context/prompt adapter`：`condition_maps` 走卷积条件支路，与 context latent 逐帧相加；`future_latent_tokens` 作为 `cross-attention memory` 按时间步注入到视频 token 中；`context_states` 和 prompt 也会编码成 token，经由同一个 `StateCrossAttentionAdapter` 作为额外 memory 一起参与 attention。也就是说，这版仍然保留 `ControlNet-style / spatial adapter-style` 显式图条件，但新增了一个更强的 `future latent adapter` 支路。

### 1.3 可训练模块

这版中可训练模块主要包括两部分：`Future Latent Predictor` 整体可训练，包括 latent 主干和显式监督 head；`State/Latent-Conditioned Video Adapter` 整体可训练，包括 condition encoder、`StateCrossAttentionAdapter`、decoder、state/spatial 辅助头，以及 context state / prompt 的条件投影分支。

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

### 1.1 Predictor 输入输出

`predictor` 输入与 `latent_v1` 相同，为 `context_states ∈ R^{B×K×N×10}`、`appearance/physics ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt；这一版没有删除旧 predictor 接口，只改变了视频生成器使用 predictor 输出的方式。`predictor` 主输出仍然是 `future_latents ∈ R^{B×T×N×128}`，同时输出显式监督分支 `states ∈ R^{B×T×N×10}` 与 `motion ∈ R^{B×T×N×3}`；其中 `states/motion` 在这版里主要用于和 GT 做监督，以及做可视化诊断，不再是视频生成器的主条件。

### 1.2 视频生成模型的 Condition + 条件注入

视频生成模型主 condition 为 `context_frames ∈ R^{B×K×3×H×W}`、`future_latent_tokens ∈ R^{B×T×N×128}`、`object memory tokens`、`context state tokens` 和 `prompt tokens`；显式 `condition_maps ∈ R^{B×T×7×H×W}` 只作为辅助监督 target 和调试可视化存在，不再作为主生成条件输入。这版的条件注入形式是 `future latent cross-attention + object/context/prompt adapter`：`future_latent_tokens` 按时间步作为 `cross-attention memory` 注入视频 latent token；`object memory tokens`、`context_states` 编码 token、prompt token 也一起作为 `StateCrossAttentionAdapter` 的 memory 参与 attention；显式 `condition_maps` 在 `latent_only` 模式下会被清零，所以不再承担 `ControlNet-style` 的主控制作用。换句话说，这版的生成主链路不是“空间图控制”，而是“future latent token 通过 adapter/cross-attention 控制”。

### 1.3 可训练模块

这版中可训练模块包括：`Future Latent Predictor` 整体可训练，包括 latent 主干和显式监督 head；`State/Latent-Conditioned Video Adapter` 整体可训练，但其主控制信号已经切换为 future latent 分支，因此重点可训练部分是 `StateCrossAttentionAdapter`、future latent 注入相关投影层、decoder，以及 state/spatial 辅助头。

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

## 2026-06-03 visual_context_predictor_v3 视觉上下文驱动 predictor 分支

### 1. 方法流程

这版不是在旧 `predictor.py` 上继续改，而是单独复制出新的 predictor 分支，核心思路是：既然 predictor 的主输出已经变成隐式 `future_latents`，那 predictor 的输入也不应该继续强依赖带误差的 `context_states` 抽取结果，而应该直接吃 `context_frames` 的视觉压缩表示。具体做法是先把 `context_frames ∈ R^{B×K×3×H×W}` 送入一个轻量视觉编码器，得到每帧的压缩特征，再通过一个 VAE-style 压缩器得到时序视觉 latent；随后用时序编码器把这些视觉 latent 聚合成历史上下文表征，并结合 prompt 形成全局条件；再引入可学习的 object slot query，为每个未来物体位置生成一个 slot 隐变量，逐步 rollout 出 `future_latents ∈ R^{B×T×N×128}`，同时从这些 latent 上接显式监督 head，输出 `states ∈ R^{B×T×N×10}` 与 `motion ∈ R^{B×T×N×3}`。这版当前先把 predictor 独立做成新文件和新训练脚本，旧版本 predictor/adapter 链路保持不动，后续如果验证有效，再在新的 adapter 版本中把这条视觉 predictor 分支接进去。

### 1.1 Predictor 输入输出

`predictor` 输入改为以视觉为主：`context_frames ∈ R^{B×K×3×H×W}` 和 prompt 是主输入，当前实现里为了兼容现有 episode 数据格式，batch 中仍然保留 `context_states`、`appearance`、`camera`，但新 predictor 主干并不依赖它们做未来 latent 预测。`predictor` 输出为 `future_latents ∈ R^{B×T×N×128}`，同时保留显式监督分支 `states ∈ R^{B×T×N×10}` 和 `motion ∈ R^{B×T×N×3}`，并额外输出视觉压缩器的 `kl` 正则项，用于约束视觉上下文压缩空间。

### 1.2 视频生成模型的 Condition + 条件注入

这版当前还没有替换现有视频生成模型分支，重点是在 predictor 侧建立“视觉上下文 -> future latent”的新接口，所以视频生成模型的 condition 注入方式暂时不变；后续推荐的接法是：把该 predictor 输出的 `future_latents ∈ R^{B×T×N×128}` 作为主条件，通过 `cross-attention memory / adapter 注入` 的形式送入视频模型，而不是再依赖由 `context_states` 推导出的显式 future state maps。换句话说，这个版本先解决 predictor 输入源的问题，再决定新的 adapter 版本如何接入。

### 1.3 可训练模块

这版目前新增且可训练的模块包括：视觉编码器、VAE-style 压缩器、时序编码器、prompt 投影层、object slot query、future latent 解码器，以及从 latent 到 `states/motion` 的监督 head。旧版 `predictor.py`、`train_predictor.py`、`train_adapter.py` 和现有 `latent_v1/v2` 训练链路都没有被覆盖，仍然保持原状可复现。

### 2. 关键实现

- `视觉上下文 predictor`
  路径：[predictor_visual_v3.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor_visual_v3.py)
  关键函数：`VisualContextLatentPredictorV3.forward()`、`predictor_visual_v3_loss()`

- `predictor 数据 collate 扩展`
  路径：[dataset.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/dataset.py)
  关键函数：`NpzPredictorDataset.__getitem__()`、`collate_predictor_episodes()`

- `视觉 predictor 训练入口`
  路径：[train_predictor_visual_v3.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor_visual_v3.py)
  关键函数：`run_epoch()`、`main()`

### 3. 输出目录与可视化指令

- `episode 数据目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6`

- `当前方法训练输出目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_visualctx_predictor_v3_gpu0123`

- `旧版两卡来源目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_visualctx_predictor_v3_gpu67`

- `checkpoint 目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_visualctx_predictor_v3_gpu0123/checkpoints`

- `训练日志目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_visualctx_predictor_v3_gpu0123/logs`

- `配置导出目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_visualctx_predictor_v3_gpu0123/configs`

- `训练说明`
  这版保持和之前四卡方案一致的 batch 设计，使用 `CUDA_VISIBLE_DEVICES=0,1,2,3`、`--gpu-ids 0,1,2,3`、`batch-size=512`；为了不浪费已经跑出的进度，训练会从旧版两卡实验的 `predictor_last.epoch025.pt` 继续续训 15 个 epoch，在新目录中产出新的 best/last checkpoint。

- `启动训练指令`

```bash
tmux new-session -d -s phys_state_visualctx_v3_gpu0123 \
  'bash -lc "/home/gaoya/Code_Video/phys_state_video/scripts/run_industrial_s1_scale2_visualctx_predictor_v3_gpu0123.sh"'
```

- `查看训练 tmux 输出`

```bash
tmux capture-pane -pt phys_state_visualctx_v3_gpu0123:0 | tail -n 80
```

- `当前状态`
  这版目前仍然是 predictor-only 分支，还没有单独接入新的 adapter 训练脚本、case watcher 和总页面入口；现阶段重点是先验证视觉上下文驱动 predictor 相比旧显式输入 predictor 的收敛和泛化表现。

## 2026-06-03 prefix_infill_v1 context clean latent + future noisy latent 补全版

### 1. 方法流程

这版已经在仓库里落成了 `predictor + Wan` 的 prefix continuation 链路，但需要把“训练时的 predictor 时间轴”和“Wan 真正采样时的 latent 时间轴”分开看。单个 batch 的基础输入来自 episode：`context_frames ∈ R^{B×K×3×H×W}`、`camera ∈ R^{B×K×C_cam}`、prompt，以及监督用的 `context_states ∈ R^{B×K×N×10}`、`future_states ∈ R^{B×T×N×10}`。训练 predictor 时，先用冻结的 `Wan VAE` 对整段 context clip 编码，得到原生 Wan latent `z_ctx_clip ∈ R^{B×C_w×L_ctx_raw×H_w×W_w}`；随后为了保持按原帧数 `K` 监督，会把 latent 沿时间维重采样回 `context_latents ∈ R^{B×K×C_w×H_w×W_w}`，再喂给 `WanStateLatentPredictor`。predictor 输出 `context_state_latents ∈ R^{B×K×D_s}` 和 `future_state_latents ∈ R^{B×T×D_s}`，并通过监督 head 解码出 `context_state_predictions ∈ R^{B×K×N×10}`、`future_state_predictions ∈ R^{B×T×N×10}`。正式推理时，predictor 仍先从 `context_latents` 预测 `future_state_latents`，但视频生成部分不会直接使用训练时那份重采样后的 `context_latents`，而是重新把单个样本的 `context_frames ∈ R^{K×3×H×W}` 按 Wan 目标分辨率 resize 后编码成原生 `clean_prefix_latents ∈ R^{C_w×L_ctx×H'_w×W'_w}`；接着构造总噪声 `noise ∈ R^{C_w×L×H'_w×W'_w}`，并在每个 diffusion step 前后都把前 `L_ctx` 个时间步覆盖回 `clean_prefix_latents`，只对后续 `L_future=L-L_ctx` 个 latent step 做补全。与此同时，`future_state_latents ∈ R^{T×D_s}` 会被重采样到 `R^{L_future×D_s}`，经外部 `Wan state adapter` 编成 `state_context` 注入 Wan DiT；这样整条链路的真正目标就是“固定干净前缀 latent，只生成未来 latent”。

### 1.1 Predictor 输入输出

当前实现里，`predictor` 真正使用的输入只有 `context_latents ∈ R^{B×K×C_w×H_w×W_w}`、`camera ∈ R^{B×K×C_cam}` 和 prompt token；虽然 episode 里还包含 `appearance ∈ R^{B×N×A}` 与 `context_states ∈ R^{B×K×N×10}`，但这两个量目前没有作为 predictor 的前向输入，只在训练 loss 中把预测结果和 GT state 对齐时使用。每帧 `context_latents` 会先通过 `adaptive_avg_pool2d` 压成 `latent_pool_side × latent_pool_side` 的 pooled 特征，再与每帧 latent 的 `mean/std` 拼接，因此单帧视觉特征维度为 `C_w×(latent_pool_side^2+2)`；默认 `latent_pool_side=2`，若 `C_w=16`，则单帧视觉特征维度为 `16×(4+2)=96`。这些帧特征与 `camera`、prompt embedding 拼接后，经 `context encoder` 得到 `context_state_latents ∈ R^{B×K×D_s}`，再由 `future decoder` 产生 `future_state_latents ∈ R^{B×T×D_s}`，默认 `D_s=128`。需要明确一点：这里的 `future_state_latents` 是“每个 future 时间步一个全局 state token”，shape 为 `R^{B×T×D_s}`，不是 `R^{B×T×N×D_s}` 的 per-object latent。监督头 `object_state_head` 再把每个时间步的全局 latent 一次性解码成 `N` 个物体的显式状态，因此输出 `context_state_predictions ∈ R^{B×K×N×10}` 与 `future_state_predictions ∈ R^{B×T×N×10}`；其中最后一维 `10` 仍对应 `center_x, center_y, depth, log_scale, vel_x, vel_y, depth_vel, visibility, existence, confidence`。

### 1.2 视频生成模型的 Condition + 条件注入

当前已经接通的是 `WanI2V` 路线，真实 condition 由三部分组成。第一部分是主视频 latent 分支：对单个样本，`Wan` 接收 `context_frames ∈ R^{K×3×H×W}`，按目标分辨率 resize 成 `R^{K×3×H_out×W_out}` 后整段编码成 `clean_prefix_latents ∈ R^{C_w×L_ctx×H'_w×W'_w}`，其中 `L_ctx` 是 Wan 自身时间压缩后的 prefix latent 步数；采样时先构造总 latent `noise ∈ R^{C_w×L×H'_w×W'_w}`，再在每个去噪 step 前后都直接用 `clean_prefix_latents` 覆盖前 `L_ctx` 个时间步，所以真正保持干净的是 latent 时间轴上的 prefix，而不是只靠 mask。第二部分是 predictor 条件分支：`future_state_latents ∈ R^{T×D_s}` 会先按 Wan 的 future latent 步数重采样成 `R^{L_future×D_s}`，再通过外部 `Wan state adapter` 编成 `state_context`，以 `cross-attention memory / adapter 注入` 的形式送入 Wan DiT；因此 state 条件与 Wan 自身的 latent 时间轴是对齐的。第三部分是 `y`：由于外部 `WanI2V` 代码仍然硬性要求 `y is not None`，当前实现保留了首帧 I2V 的 `y`，即由首帧 `i2v_latent` 和首帧 mask 拼成的兼容条件；但主前缀视频条件已经转移为 `clean_prefix_latents` 覆盖主 latent 序列，`y` 现在不再承担完整 context video 注入职责。这里有一个实现前提必须写清楚：如果要让 state 条件真正生效，需要显式加载 `state_adapter_ckpt`；否则 Wan 侧虽然会按输入 shape 把 adapter 分支搭起来，但权重可能仍接近默认初始化，控制效果不可靠。除此之外，仓库里还保留了一条更通用的 `Wan state_condition` 数据桥接路径：如果暂时没有可直接跑 `clean_prefix_latents + WanI2V` 的本地权重或 `wan_state_v1` predictor checkpoint，可以先把 episode 导出成 `input_image.png + state_condition.npz + prompt.txt`，再走外部 `Wan TI2V` 的官方 `state_condition` 接口做训练或 smoke test；不过要注意，这条导出路径和正式 prefix continuation 链路并不完全等价，因为 `ground_truth` 导出模式提供的是 `predicted_states ∈ R^{T×N×10}`，而正式 prefix continuation 使用的是 predictor 产出的 `state_tokens ∈ R^{L_future×D_s}`。

### 1.3 可训练模块

从当前仓库实现看，可训练模块已经明确分成两类。第一类是 `phys_state_video` 内部的 `WanStateLatentPredictor`：包括 `PromptEncoder`、`prompt_proj`、`context_input_proj`、`context encoder`、`context_state_latent_proj`、`future_queries`、`future_pos_embed`、`future decoder`、`future_state_latent_proj` 和 `object_state_head`；其中 `object_state_head` 只在训练时用于把 `context_state_latents / future_state_latents` 解码成 `N×10` 的物体级状态监督，推理时会被完全丢掉。第二类是 `Wan` 侧的 `state adapter` 分支及其对应的 DiT 注入权重，这部分不在当前 predictor 训练脚本里优化，而是需要走外部 Wan 训练脚本单独训练并产出 `state_adapter_ckpt`，推理时再由 `wan_bridge.py` 显式加载。冻结模块包括：用于提取 `context_latents` 的 `Wan VAE`，以及当前桥接推理链路里直接复用的基础 `Wan` 主干参数、text encoder 等。因此现在的职责划分是：仓库内的 predictor 负责把 context 视频编码成未来状态 token，外部 Wan state adapter 负责把这些 `future_state_latents` 映射成对视频去噪真正有效的条件控制。

### 2. 关键实现

- `当前状态`
  这版已经在仓库里有可运行实现，不再只是设计稿；当前接通的是 `predictor + Wan` 推理链路，`VACE` 仍未接入。需要区分两层“已接通”含义：一层是仓库内部的 `clean prefix latent + future state latent` 逻辑已经实现并有单测覆盖关键 helper；另一层是外部 `Wan` 真正大模型采样能否在本机跑起，后者还受本地 checkpoint 类型和 CUDA 运行时约束。

- `predictor 实现`
  路径：`/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor_wan_state.py`
  关键函数：`WanStateLatentPredictor.forward()`、`wan_state_predictor_loss()`
  作用：把 `context_latents + camera + prompt` 映射成 `context_state_latents / future_state_latents`，并在训练时额外输出 `context_state_predictions / future_state_predictions`。

- `Wan latent 提取与推理桥接`
  路径：`/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_bridge.py`
  关键函数：`WanLatentExtractor.encode_context_frames()`、`WanImageToVideoBackend.generate()`
  作用：前者负责把整段 context clip 编码成 Wan latent 并重采样回 predictor 需要的 `B×K×C_z×H'×W'`；后者负责 `clean_prefix_latents` 覆盖、future `state_tokens` 时间重采样、`state_adapter_ckpt` 加载以及最终 Wan 采样。

- `predictor 训练入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor_wan_state.py`
  关键函数：`encode_context_latents()`、`run_epoch()`、`build_model_config()`、`main()`
  作用：调用 `WanLatentExtractor` 生成训练时的 `context_latents`，并训练 `WanStateLatentPredictor`。

- `Wan 推理入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state.py`
  关键函数：`main()`
  作用：读取单个 episode，跑 `WanStateLatentPredictor` 预测 `future_state_latents`，再把它们送入 `WanImageToVideoBackend.generate()` 做 prefix continuation 推理；当前脚本导出的是 `context_frames`、`predicted_future_states`、`future_state_latents`、`generated_full_video`、`generated_future_frames`，不再假设 predictor dataset 自带 `future_gt_frames`。

- `Wan state_condition 数据导出入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/export_wan_state_condition_dataset.py`
  关键函数：`build_ground_truth_state_condition()`、`build_predictor_state_condition()`、`main()`
  作用：把 `phys_state_video` 的 episode 导出成外部 `Wan` 可直接消费的 bundle，包括 `input_image.png`、`state_condition.npz`、`meta.json`、`prompt.txt`、`manifest.jsonl`；其中既支持直接导出 `future_states -> predicted_states`，也支持未来导出 `wan_state_v1 predictor -> state_tokens`。

- `Wan TI2V smoke test 入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/run_wan_ti2v_state_condition_smoke.py`
  关键函数：`load_state_condition()`、`main()`
  作用：直接读取上一步导出的 bundle，用本地现成的 `Wan2.2-TI2V-5B` 权重验证外部 `Wan` 官方 `state_condition` 接口能否跑通；这条脚本主要用于当前机器环境下的桥接 smoke test，而不是最终的 prefix continuation 正式推理入口。

- `Wan 侧外部依赖`
  路径：`/home/gaoya/Code_Video/Wan2.2-main`
  关键文件：`wan_/image2video.py`、`wan_/textimage2video.py`、`wan_/state_condition.py`、`generate.py`
  作用：
  `wan_/image2video.py`
  对应 `WanI2V`，是当前 `clean_prefix_latents + future noisy latent` 正式桥接逻辑最终复用的主接口。
  `wan_/textimage2video.py`
  对应 `WanTI2V`，是当前本地现成 `Wan2.2-TI2V-5B` 权重可直接验证的 `state_condition` 路线。
  `wan_/state_condition.py`
  定义 `state_tokens / predicted_states / memory_tokens / condition_maps` 的规范化接口和 `WanObjectStateAdapter`。
  `generate.py`
  给出了官方 CLI 路径，也明确说明了“提供了 `state_condition` 但没有 `state_adapter_ckpt` 时，state branch 仍可能接近零门控默认状态”这一关键限制。

### 3. 输出目录与可视化指令

- `当前状态`
  这版已经可以单独训练 predictor、单独做 `predictor + Wan` 推理，也可以把 episode 导出成外部 `Wan state_condition` bundle；但 `Wan state adapter` 的训练和 checkpoint 仍依赖外部 Wan 仓库脚本，不在当前目录里统一管理。

- `建议训练输出目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_prefix_infill_v1`

- `建议可视化目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_prefix_infill_v1/viz/trained_cases_v1`

- `当前定位`
  这版是下一条主线候选方案，目标是替代当前“predictor 先 rollout future state，再由 adapter 猜视频”的结构，改为“context latent 保持干净、future latent 补噪补全”的 prefix-conditioned 生成方式。当前仓库内已经有 3 个应该优先看的入口：`train_predictor_wan_state.py` 负责训练 `wan_state_v1 predictor`，`run_inference_wan_state.py` 负责正式的 prefix continuation 推理链路，`export_wan_state_condition_dataset.py` 负责把现有 episode 对齐到外部 `Wan state adapter` 训练/验证接口；如果只是排查本机环境能否把外部 `Wan` 跑起来，则看 `run_wan_ti2v_state_condition_smoke.py`。

## 2026-06-03 wan_state_v2_latent_time predictor + TI2V state adapter 本地训练版

### 1. 方法流程

这一版是当前仓库里已经落地并通过正式单测的 `v2 predictor` 主线，核心变化是 predictor 不再把 Wan latent 时间维插值回原视频帧数，而是始终停留在 `Wan VAE latent` 的时间轴上训练和推理。dataset 输入仍然来自 episode：`context_frames ∈ R^{B×K×3×H×W}`、`camera ∈ R^{B×K×C_cam}`、`context_states ∈ R^{B×K×N×10}`、`future_states ∈ R^{B×T×N×10}`、prompt。训练 predictor 时，先把 `context_frames` 编码成 `context_latents_raw ∈ R^{B×L_ctx×C_w×H_w×W_w}`；这里 `L_ctx = 1 + floor((K-1)/stride_t)`，`stride_t` 由 Wan VAE 决定，当前 smoke case 中 `stride_t=4`，所以例如 `K=4` 时会得到 `L_ctx=1`。随后把 `camera ∈ R^{B×K×C_cam}` 沿时间维重采样到 `camera_latent ∈ R^{B×L_ctx×C_cam}`，把 `context_states` 也重采样到 `R^{B×L_ctx×N×10}` 仅用于监督；predictor 输出 `context_state_latents ∈ R^{B×L_ctx×D_s}`、`future_state_latents ∈ R^{B×L_future×D_s}`，并通过 grouped state heads 解码出 `context_state_predictions ∈ R^{B×L_ctx×N×10}`、`future_state_predictions ∈ R^{B×L_future×N×10}`。其中 `L_future = compute_future_latent_steps(K, T, stride_t)`，也是直接按 latent 时间轴定义，而不是按原视频 future 帧数定义。训练 schedule 采用三阶段：先 `context_only`，只用 context loss 把 grouped state heads 训稳；再 `future_only`，冻结 state heads，只训练 future latent rollout；最后 `joint_finetune` 小步联合微调。推理时，给定单个 episode，先走同样的 latent-time predictor 得到 `future_state_latents ∈ R^{L_future×D_s}`，然后有两条后续路径：如果只是验证 predictor，本仓库直接用 `run_inference_wan_state_v2.py` 导出 `npz/meta`；如果要接入 Wan，则通过 `export_wan_state_condition_dataset.py` 把它写成 `state_tokens ∈ R^{L_future×D_s}` 的 bundle，再交给本地 `Wan TI2V state adapter` 训练或推理路径。

### 1.1 Predictor 输入输出

当前 `WanStateLatentPredictorV2` 的真实输入是 `context_latents_raw ∈ R^{B×L_ctx×C_w×H_w×W_w}`、`camera_latent ∈ R^{B×L_ctx×C_cam}` 和 prompt token；`context_states/future_states` 不进入 predictor 主干，只在 loss 中作为监督目标。每个 latent step 的视觉特征会先经过空间池化和线性投影，送入 context encoder；decoder 再基于 learned future queries 产生 `future_state_latents ∈ R^{B×L_future×D_s}`。当前实现中的显式状态 head 不是一个单独的大 head，而是按物理语义分组：`geom` 头输出 `4` 维，对应 `center_x, center_y, depth, log_scale`；`motion` 头输出 `3` 维，对应 `vel_x, vel_y, depth_vel`；`vis` 头输出 `3` 维，对应 `visibility, existence, confidence`。三组拼起来后得到 `context_state_predictions ∈ R^{B×L_ctx×N×10}` 和 `future_state_predictions ∈ R^{B×L_future×N×10}`。因此这版 predictor 的主输出仍然是 `future_state_latents`，显式 `10` 维状态是监督分支；和旧 `v1` 不同的是，这里的时间维全部是 `Wan latent` 时间维。当前已经支持两种 latent 来源：`mock latent` 路径使用 `MockLatentExtractor`，输入 `context_frames ∈ R^{B×K×3×H×W}` 后输出 `context_latents_raw ∈ R^{B×L_ctx×C_w×H_w×W_w}`，可以在 CPU 上跑单测和 smoke；`real Wan latent` 路径使用 `WanLatentExtractor.encode_context_frames_raw()`，输出 shape 相同，但需要本地 Wan VAE 和可用 CUDA 运行时。

### 1.2 视频生成模型的 Condition + 条件注入

这版和旧 prefix continuation 的区别是，当前仓库里 predictor 已经升级到 latent-time 版本，但真正接上的视频模型路径首先是 `Wan state_condition / TI2V adapter` 路线，而不是直接把 `v2 predictor` 接回 `WanI2V clean-prefix continuation`。具体来说，`export_wan_state_condition_dataset.py` 会把每个样本导出成一个 bundle：`input_image.png` 是首帧 context 图像，`state_condition.npz` 在 predictor 模式下保存 `state_tokens ∈ R^{L_future×D_s}`，`meta.json` 记录 `episode_path`、`context_latent_steps`、`future_latent_steps`、`temporal_stride` 等信息，`prompt.txt` 保存文本提示。Wan 侧的条件注入形式是 `adapter 注入 + cross-attention memory`：`wan_/state_condition.py` 先把 `state_tokens` 编码成 `state_context`，然后 `WanObjectStateAdapter` 的输出以 `cross-attention memory` 的形式注入 `WanTI2V.model` 的每个 block 内部 `state_adapter_*` 分支。因此当前仓库里真正训练的视频模型条件，不再是旧 baseline 里的 `condition_maps` 或 `memory_tokens`，而是 `state_tokens ∈ R^{L_future×D_s}`。在本地 `TI2V` 训练脚本里，监督视频不是 `K` 帧 context 全拼进去，而是按照当前 dataset 和 `WanTI2V` 的接口对齐为“首帧图像 + future video”：构造 `training_video ∈ R^{F_train×3×H_out×W_out}`，其中第一帧来自 `context_frames[0]`，后面接 `future_frames`，再按 Wan 规则补齐到 `4n+1` 帧。例如若 episode 里 `T=6`，则 `1+T=7`，对齐后训练视频长度会补到 `F_train=9`。训练时 `WanTI2V` 把整段 `training_video` 编码成 `input_latents ∈ R^{C_w×L×H_w×W_w}`，首帧单独编码成 `first_frame_latents ∈ R^{C_w×1×H_w×W_w}`，然后只对 `t>=1` 的 latent step 加噪，保持第一个 latent step 为 clean first-frame condition。

### 1.3 可训练模块

这版实际可训练模块已经明确分成两段。第一段是 `phys_state_video` 内部的 `WanStateLatentPredictorV2`，包括 visual latent encoder、prompt encoder、context encoder、future decoder、future latent projection，以及三组显式状态 heads；训练时通过 `context_only -> future_only -> joint_finetune` 三阶段优化。第二段是本地 `Wan TI2V state adapter` 训练：`train_wan_state_adapter_local.py` 先读取导出的 bundle，再回溯 `meta.json["episode_path"]` 到原始 episode，构造首帧图像加 future 视频的监督样本，然后只训练 `pipeline.state_adapter` 本身和 `WanTI2V.model` 里所有名字包含 `state_adapter_` 的参数；`text_encoder`、`vae`、以及主 DiT 其它权重保持冻结。保存出的 checkpoint 也已经对齐到 Wan 原生格式：`state_adapter_config`、`state_adapter`、`model_state_adapter`，可以直接通过 `WanTI2V.load_state_adapter()` 或 `run_wan_ti2v_state_condition_smoke.py --state-adapter-ckpt` 加载。需要如实说明的是：代码层面这条本地 adapter 训练闭环已经补全，并且 bundle 发现、`4n+1` 对齐、checkpoint 格式判断等逻辑有正式单测；但在当前机器的 `wan` 环境里，真实训练仍被 CUDA 运行时失配阻塞，因此本次还没有完成真实 GPU 优化和保存后再采样的视频 smoke。

### 2. 关键实现

- `v2 predictor 实现`
  路径：`/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor_wan_state_v2.py`
  关键函数：`WanStateLatentPredictorV2.forward()`、`wan_state_predictor_v2_loss()`、`resample_temporal_states()`
  作用：在 Wan latent 时间轴上完成 context 编码、future rollout、grouped state heads 解码和 staged loss 计算。

- `v2 latent helper`
  路径：`/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_state_v2_helpers.py`
  关键函数：`compute_latent_step_count()`、`compute_future_latent_steps()`、`resample_camera_to_latent_steps()`、`MockLatentExtractor.encode_context_frames_raw()`
  作用：统一 latent 时间轴长度计算、camera 对齐和 mock latent smoke 路线。

- `v2 predictor 训练入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor_wan_state_v2.py`
  关键函数：`build_latent_extractor()`、`configure_stage()`、`run_epoch()`、`main()`
  作用：支持 `mock`/`wan` 两种 latent 路径，并按三阶段 schedule 训练 predictor。

- `v2 predictor 推理入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state_v2.py`
  关键函数：`main()`
  作用：加载 `wan_state_v2_latent_time` checkpoint，导出 `context_latents`、`future_state_latents`、`context_state_predictions`、`future_state_predictions` 等结果。

- `state_condition 导出入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/export_wan_state_condition_dataset.py`
  关键函数：`build_predictor_state_condition()`、`main()`
  作用：把 `wan_state_v2_latent_time` predictor 产出的 `future_state_latents` 写成 `state_tokens` bundle，并记录 latent-step metadata。

- `本地 TI2V adapter 训练辅助`
  路径：`/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_adapter_training.py`
  关键函数：`discover_state_condition_bundles()`、`build_ti2v_training_video()`、`select_ti2v_state_adapter_parameters()`、`LocalWanFlowMatchScheduler`
  作用：统一 bundle 发现、episode 回溯、首帧+future 训练视频构造、`4n+1` 对齐、可训练参数选择、以及本地 flow-matching 训练辅助。

- `本地 TI2V adapter 训练入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_local.py`
  关键函数：`prepare_training_sample()`、`run_step()`、`main()`
  作用：读取 bundle 和原始 episode，训练 Wan TI2V 的 state adapter，并导出 `WanTI2V.load_state_adapter()` 兼容 checkpoint。

- `Wan TI2V smoke 推理入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/run_wan_ti2v_state_condition_smoke.py`
  关键函数：`main()`
  作用：直接加载 bundle 和可选 `state_adapter_ckpt`，验证 `state_condition` 路线推理。

### 3. 当前状态与验证

- `正式单测`
  路径：`/home/gaoya/Code_Video/phys_state_video/tests/test_wan_state_predictor.py`
  路径：`/home/gaoya/Code_Video/phys_state_video/tests/test_wan_adapter_training.py`
  当前结果：`17 passed`

- `已跑通的 smoke`
  这版已经跑通过 `toy dataset -> train_predictor_wan_state_v2.py --latent-source mock -> run_inference_wan_state_v2.py -> export_wan_state_condition_dataset.py` 的 CPU smoke，说明 `predictor_v2 -> export state_tokens` 这半条链是通的。

- `当前阻塞`
  本机 `nvidia-smi` 正常，驱动版本为 `570.124.06`，CUDA 版本显示为 `12.8`；但 `wan` 环境中的 PyTorch 是 `torch 2.11.0+cu130`，`torch.version.cuda == 13.0`，因此 `torch.cuda.is_available()` 返回 `False`，并报 `found version 12080` 的 driver/runtime mismatch。结论是：当前 `mock latent` predictor 路线和正式单测都可运行；本地 `Wan TI2V state adapter` 训练脚本已经补齐，但真实 GPU 训练、真实 Wan latent 提取、以及保存 adapter 后再做 Wan 采样，仍然是环境阻塞，而不是当前仓库缺少训练 loop。

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
