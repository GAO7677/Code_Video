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

这版现在已经在仓库里落成了可运行的 `Wan` 链路，整体拆成两段：前半段是 `state latent predictor`，后半段是 `Wan` 前缀续写生成器。输入样本先给出 `context_frames ∈ R^{B×K×3×H×W}`、`camera ∈ R^{B×K×C_cam}`、prompt，以及物理真值 `context_states ∈ R^{B×K×N×10}`、`future_states ∈ R^{B×T×N×10}`。视觉侧先用冻结的 `Wan VAE` 对整段 context clip 进行联合编码，得到时间压缩后的 `z_ctx_clip ∈ R^{B×K_lat×C_z×H'×W'}`，其中 `K_lat` 是 Wan 自身时间 stride 下的 latent 步数；为了让 predictor 保持按原视频帧监督，再沿 latent 时间轴把 `z_ctx_clip` 重采样回 `context_latents ∈ R^{B×K×C_z×H'×W'}`。随后 predictor 读取 `context_latents + camera + prompt`，先输出 `context_state_latents ∈ R^{B×K×D_s}`，再通过时序 transformer 逐帧预测 `future_state_latents ∈ R^{B×T×D_s}`。训练时在这两组 state latent 上都接物体级 head，分别得到 `context_state_predictions ∈ R^{B×K×N×10}` 和 `future_state_predictions ∈ R^{B×T×N×10}`，用仿真导出的物理 GT 监督；推理时丢掉这些 head，只保留 `future_state_latents` 作为未来视频条件。

### 1.1 Predictor 输入输出

`predictor` 的显式输入仍然固定为 `context_latents ∈ R^{B×K×C_z×H'×W'}`、`camera ∈ R^{B×K×C_cam}` 和 prompt，但它现在不再只看每帧 latent 的 `mean/std`，而是先对每帧 Wan latent 做轻量空间池化，再拼接 `mean/std` 得到更丰富的视觉特征；默认每帧特征维度是 `C_z×(s^2+2)`，这里 `s=latent_pool_side`。这些特征经过 context encoder 后得到 `context_state_latents ∈ R^{B×K×D_s}`，再由 future decoder 逐帧产生 `future_state_latents ∈ R^{B×T×D_s}`。状态监督头把这两组 latent 分别映射成 `context_state_predictions ∈ R^{B×K×N×10}` 和 `future_state_predictions ∈ R^{B×T×N×10}`，其中最后一维 `10` 对应每个物体的连续物理状态字段。也就是说，`state latent` 是 predictor 内部单独学习出来的物理语义空间，`vae latent` 只是它的视觉输入，不与 `state latent` 共享表征空间。

### 1.2 视频生成模型的 Condition + 条件注入

当前已经接通的是 `Wan` 路线，而且真实 condition 由三部分组成。第一部分是主视频 latent 分支：对单个样本，`Wan` 接收 `context_frames ∈ R^{K×3×H×W}`，按目标分辨率 resize 后整段编码成 `clean_prefix_latents ∈ R^{C_w×L_ctx×H_w×W_w}`，其中 `L_ctx` 是 Wan 时间压缩后的前缀 latent 步数；采样时先构造总 latent `noise ∈ R^{C_w×L×H_w×W_w}`，再在每一步去噪前都直接用 `clean_prefix_latents` 覆盖前 `L_ctx` 个时间步，所以真正保持干净的是 latent 时间轴上的 prefix，而不是仅靠额外 mask。第二部分是 predictor 条件分支：`future_state_latents ∈ R^{T×D_s}` 会先按 Wan 的 future latent 步数重采样成 `R^{L_future×D_s}`，再通过外部 `Wan state adapter` 变成 `state_context` 注入去噪模型，因此 state 条件和 Wan 自身的时间轴是对齐的。第三部分是 `y`：由于外部 `WanI2V` 代码仍然硬性要求 `y is not None`，当前实现保留了首帧 I2V 的 `y` 作为兼容条件，但主前缀视频条件已经转移为 `clean_prefix_latents` 覆盖主 latent 序列，`y` 不再承担完整 context video 注入职责。这里有一个实现前提必须写清楚：如果要让 state 条件真正生效，需要显式加载 `state_adapter_ckpt`；否则分支虽然会按输入 shape 被构建出来，但仍可能接近默认初始化，控制效果不可靠。除此之外，仓库里现在还补了一条更通用的 `Wan state_condition` 数据桥接路径：如果暂时没有可直接跑 `clean_prefix_latents + WanI2V` 的本地权重或 `wan_state_v1` predictor checkpoint，可以先把 episode 导出成 `input_image.png + state_condition.npz + prompt.txt`，再走外部 `Wan TI2V` 的官方 `state_condition` 接口做训练或 smoke test；这条路径虽然不等价于最终的 prefix continuation 推理链路，但已经把 `phys_state_video -> Wan state adapter` 的条件格式完全对齐了。

### 1.3 可训练模块

从当前仓库实现看，已经明确分成两类可训练模块。第一类是 `phys_state_video` 内部的 predictor：包括 `PromptEncoder`、context encoder、future decoder 和物体级状态 head；其中 head 只在训练时用于把 `context_state_latents / future_state_latents` 解码成 `N×10` 的物体级状态监督，推理时会被完全丢掉。第二类是 `Wan` 侧的 state adapter 分支及其对应的 DiT 注入权重，这部分不在当前 predictor 训练脚本里优化，而是需要走外部 Wan 训练脚本单独训练并产出 `state_adapter_ckpt`，推理时再由 `wan_bridge.py` 显式加载。冻结模块则包括 `Wan VAE` 和当前桥接推理链路里直接复用的基础 `Wan` 主干参数；因此现在的职责划分是：predictor 学未来物理状态的隐式时序表征，外部 Wan state adapter 学如何把 `future_state_latents` 映射成对视频去噪真正有效的条件控制。

### 2. 关键实现

- `当前状态`
  这版已经在仓库里有可运行实现，不再只是设计稿；当前接通的是 `predictor + Wan` 推理链路，`VACE` 仍未接入。需要区分两层“已接通”含义：一层是仓库内部的 `prefix latent + future state latent` 逻辑已经实现并可单测；另一层是外部 `Wan` 真正大模型采样能否在本机跑起，后者还受本地 checkpoint 类型和 CUDA 运行时约束。

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
  作用：读取单个 episode，跑 `WanStateLatentPredictor` 预测 `future_state_latents`，再把它们送入 `WanImageToVideoBackend.generate()` 做 prefix continuation 推理。

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
