# Method

## 2026-06-03 predictor预测隐式状态，latent接head做监督

### 1. 方法流程

整个方法可以写成一条链：给定前 `K` 帧 context 视频及其物体级状态，先把每个样本整理成 `context_frames ∈ R^{B×K×3×H×W}`、`context_states ∈ R^{B×K×N×S}`、`appearance/physics ∈ R^{B×N×A}`、`camera ∈ R^{B×K×C}` 和 prompt，其中当前实现里 `S=10`；`Future Latent Predictor` 以 `context_states + appearance/physics + camera + prompt` 为输入，先编码成历史隐状态，再为每个未来时刻、每个物体预测 `future_latents ∈ R^{B×T×N×D}`，当前可设 `D=128`，并从这些 latent 上接显式监督 head，得到 `states ∈ R^{B×T×N×10}` 和 `motion ∈ R^{B×T×N×3}`，其中 `states` 包含 `center/depth/log_scale/visibility` 等可解释变量；随后把显式状态投影成空间条件 `condition_maps ∈ R^{B×T×C_map×H×W}`，例如 heatmap、bbox、depth、visibility、velocity map，同时把 `future_latents` 直接作为 object-temporal memory token；最后 `State/Latent-Conditioned Video Adapter` 接收 `context_frames`、`condition_maps`、`memory_tokens` 以及 `future_latent_tokens`，在内部把像素特征从 `R^{B×K×3×H×W}` 编到时空 latent，再与 `R^{B×T×N×D}` 的未来物体 latent 做 cross-attention 融合，输出未来视频 `generated_frames ∈ R^{B×T×3×H×W}`，训练时同时用视频重建损失和 latent 上各个显式 head 的监督损失约束，使模型既保留可解释的物体状态控制，又能用高带宽 latent 表达接触相位、姿态变化和复杂动力学。

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
