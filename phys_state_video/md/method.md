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

这版方案把“先预测状态、再生成视频”的主链路改成“前缀视频 latent 已知、未来视频 latent 补全”：先把完整视频编码成 `z_all ∈ R^{B×(K+T)×C×H'×W'}`，再切成 `context_latents ∈ R^{B×K×C×H'×W'}` 和 `future_latents_gt ∈ R^{B×T×C×H'×W'}`；训练时保持 `context_latents` 干净不加噪，只对 `future_latents_gt` 加噪得到 `future_latents_noisy ∈ R^{B×T×C×H'×W'}`，随后把两段拼成 `sequence_latents ∈ R^{B×(K+T)×C×H'×W'}`，并配套 `future_mask ∈ R^{B×(K+T)}` 标明哪些时间步属于未来段。视频模型主干接收 `context clean + future noisy` 的整段 latent 序列，只对 future 段做噪声预测 / latent 补全；如果保留 predictor，则 predictor 不再主输出绝对显式状态，而是输出 `future_prior_tokens ∈ R^{B×T×N×D}` 或 `future_latent_prior ∈ R^{B×T×M×D}` 作为未来段隐式先验，同时从这些隐式 token 上接 `state/motion` head 做辅助监督。推理时输入真实 `context`，后面 future 段直接 padding 高斯噪声，再让模型在 context 前缀条件下把未来 latent 逐步 denoise 成未来视频。

### 1.1 Predictor 输入输出

这版里 predictor 不是必须模块；如果启用 predictor，它的输入优先改为 `context_frames` 或 `context_latents` 加 prompt，而不是 `context_states`。`predictor` 输出不再以绝对 `future_states ∈ R^{B×T×N×10}` 作为视频生成主条件，而是以 `future_prior_tokens ∈ R^{B×T×N×D}` 或 `future_latent_prior ∈ R^{B×T×M×D}` 作为主输出；同时可以保留辅助显式分支 `states ∈ R^{B×T×N×10}`、`motion ∈ R^{B×T×N×3}`，这些显式量只负责监督和诊断，不再承担主生成条件的职责。

### 1.2 视频生成模型的 Condition + 条件注入

视频生成模型的主输入是整段 latent 序列：`context_latents_clean ∈ R^{B×K×C×H'×W'}` 与 `future_latents_noisy ∈ R^{B×T×C×H'×W'}` 拼接后的 `sequence_latents ∈ R^{B×(K+T)×C×H'×W'}`，外加 `future_mask`、prompt，以及可选的 `future_prior_tokens`。这版的关键是 `future-only condition 注入`：`context` 段 latent 作为真实前缀保留，不再额外注入 object condition；外部条件只作用在 `future` 段 token 上。条件进入视频模型的形式建议是 `full-sequence self-attention + future-only cross-attention / adapter 注入`：整段 token 一起做时序建模，使 future 能看到 context；但 predictor prior、object memory、prompt bias 等条件只通过 `future_mask` gated 的 adapter 或 cross-attention 注入到 future token，而不改写 context token。本质上，这版不是 `ControlNet-style` 空间图主控制，而是 `prefix latent + future-only token condition` 的视频补全结构。

### 1.3 可训练模块

这版的可训练模块主要包括：整段视频 latent 补全主干、future-only adapter / cross-attention 注入层、prompt / prior token 投影层，以及可选 predictor 的 future prior 分支和其上的显式监督 head。如果需要显式物理约束，可以从 future latent token 上接 `state/motion` heads 做辅助监督，但这些 heads 不是主生成链路的一部分。

### 2. 关键实现

- `当前状态`
  这版目前还是方法设计稿，尚未在仓库中落成独立实现文件；下一步应复制出新的版本化文件，而不是直接改坏 `latent_v2` 或 `visual_context_predictor_v3` 的现有接口。

- `建议新增的主干实现`
  路径：`/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/adapter_prefix_infill_v1.py`
  关键函数：`PrefixInfillVideoBackbone.forward()`

- `建议新增的训练入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/train_adapter_prefix_infill_v1.py`
  关键函数：`run_epoch()`、`main()`

- `建议新增的推理入口`
  路径：`/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_prefix_infill_v1.py`
  关键函数：`main()`

### 3. 输出目录与可视化指令

- `当前状态`
  这版尚未开始正式实现和训练，因此还没有固定的运行目录、checkpoint 目录和可视化页面。

- `建议训练输出目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_prefix_infill_v1`

- `建议可视化目录`
  路径：`/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_prefix_infill_v1/viz/trained_cases_v1`

- `当前定位`
  这版是下一条主线候选方案，目标是替代当前“predictor 先 rollout future state，再由 adapter 猜视频”的结构，改为“context latent 保持干净、future latent 补噪补全”的 prefix-conditioned 生成方式。

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
