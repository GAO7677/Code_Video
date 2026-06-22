# 可视化服务目录

这份目录只保留会在本地端口起 HTTP 页面的脚本。所有命令默认以前台方式运行。

## 维护约定

- 新增或修改本地端口可视化脚本时，同步更新这份文件。
- 每个条目只保留 3 个信息: 前台启动命令、访问链接、最短用途说明。
- 默认端口冲突时，不复用同一端口；并行运行时显式追加 `--port <新端口>`。

## 通用前缀

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python
```

## 推荐可视化链路

这条链路对应训练里“从 `context_video` 提取对象级条件”的主流程：

`context_video -> 对象先验 -> query points -> tracks -> geometry -> object tokens / fused context`

### Step 1. 看训练实际使用的 context_video

- 服务: `serve_phys_state_training_inputs.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/serve_phys_state_training_inputs.py
```

- 链接: `http://127.0.0.1:8767/index.html`
- 说明: 先确认 `full video` 和 `context video` 是否符合训练输入预期。

### Step 2. 看对象先验是怎么来的

- 主服务: `inspect_sam2_motion_prompt_jsons.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_sam2_motion_prompt_jsons.py
```

- 链接: `http://127.0.0.1:8797/index.html`
- 说明: 看 motion prompt、SAM2 mask 和 prompt 叠加结果。

- 多目标补充: `inspect_groundedsam_vggt_cotracker.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_groundedsam_vggt_cotracker.py
```

- 链接: `http://127.0.0.1:8803/index.html`
- 说明: 看 `GroundingDINO -> SAM2 -> query points` 的多目标版本。

### Step 3. 看 query points 和 tracks

- 主服务: `inspect_vggt_query_points_overlay.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_vggt_query_points_overlay.py
```

- 链接: `http://127.0.0.1:8777/index.html`
- 说明: 看 query points、SAM prior、VGGT tracks 和可选 CoTracker 叠加。

- 监督检查: `eval_vggt_box_supervision.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_vggt_box_supervision.py
```

- 链接: `http://127.0.0.1:8766/index.html`
- 说明: 看 VGGT tracks 和 GT box 的贴合程度。

### Step 4. 看 prior 是否改善 tracking

- 服务: `eval_vggt_sam_prior.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_vggt_sam_prior.py
```

- 链接: `http://127.0.0.1:8771/index.html`
- 说明: 看加入 SAM prior 后，VGGT tracking 是否更稳。

### Step 5. 看多目标 query 分配

- 服务: `eval_vggt_sam_multi_object_viewer.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_vggt_sam_multi_object_viewer.py
```

- 链接: `http://127.0.0.1:8783/index.html`
- 说明: 看多目标检测、query 分配和 VGGT 轨迹。

### Step 6. 看 CoTracker 替代轨迹源

- 对比服务: `inspect_vggt_vs_cotracker_compare.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_vggt_vs_cotracker_compare.py \
  --device cuda:0
```

- 链接: `http://127.0.0.1:8805/index.html`
- 说明: 看 VGGT 和 CoTracker 的 tracks 差异。

### Step 7. 看 geometry 采样结果

- 服务: `inspect_cotracker_vggt_geometry.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_cotracker_vggt_geometry.py
```

- 链接: `http://127.0.0.1:8807/index.html`
- 说明: 看 tracks 采样出来的 `depth / world points` 是否稳定。

### Step 8. 看 object tokens 和 fused context 的最终结构

- 服务: `inspect_object_pipeline.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_object_pipeline.py \
  --serve
```

- 链接: `http://127.0.0.1:8765/index.html`
- 说明: 最后看 `tracks -> object_tokens -> fused_context` 的结构化输出。

## 1. 数据与输入

### PhysState 数据集采样

- 脚本: `code_vjepa_vggt/serve_phys_state_dataset.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/serve_phys_state_dataset.py
```

- 链接: `http://127.0.0.1:8765/index.html`
- 说明: 查看 context 抽帧、box 和样本元数据。

### PhysState 训练输入

- 脚本: `code_vjepa_vggt/serve_phys_state_training_inputs.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/serve_phys_state_training_inputs.py
```

- 链接: `http://127.0.0.1:8767/index.html`
- 说明: 查看训练实际送入模型的 `full video` 和 `context video`。

### Object Pipeline 报告

- 脚本: `code_vjepa_vggt/inspect_object_pipeline.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_object_pipeline.py \
  --serve
```

- 链接: `http://127.0.0.1:8765/index.html`
- 说明: 查看 object pipeline 的 shape、tracks、object tokens 和 fused context。

## 2. 监督与先验

### VGGT 对比 Box 监督

- 脚本: `code_vjepa_vggt/eval_vggt_box_supervision.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_vggt_box_supervision.py
```

- 链接: `http://127.0.0.1:8766/index.html`
- 说明: 看 VGGT query tracks 和 GT box 的贴合程度。

### SAM2 Motion Prompt 对比 Box 监督

- 脚本: `code_vjepa_vggt/eval_sam2_motion_box_supervision.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_sam2_motion_box_supervision.py
```

- 链接: `http://127.0.0.1:8767/index.html`
- 说明: 看 SAM2 motion prompt 轨迹和 box 监督的一致性。

### VGGT + SAM Prior

- 脚本: `code_vjepa_vggt/eval_vggt_sam_prior.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_vggt_sam_prior.py
```

- 链接: `http://127.0.0.1:8771/index.html`
- 说明: 看加入 SAM query prior 后的 VGGT tracking 表现。

### SAM2 Motion Prompt JSON

- 脚本: `code_vjepa_vggt/inspect_sam2_motion_prompt_jsons.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_sam2_motion_prompt_jsons.py
```

- 链接: `http://127.0.0.1:8797/index.html`
- 说明: 看 GT JSON 源视频上的 motion prompt、SAM2 mask 和叠加结果。

## 3. Query 与多目标

### Query Points Overlay

- 脚本: `code_vjepa_vggt/inspect_vggt_query_points_overlay.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_vggt_query_points_overlay.py
```

- 链接: `http://127.0.0.1:8777/index.html`
- 说明: 看 query points、SAM prior、VGGT tracks 和可选 CoTracker 叠加。

### VGGT 多目标 SAM

- 脚本: `code_vjepa_vggt/eval_vggt_sam_multi_object_viewer.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_vggt_sam_multi_object_viewer.py
```

- 链接: `http://127.0.0.1:8783/index.html`
- 说明: 看多目标检测、query 分配和 VGGT 轨迹。

### Grounded-SAM 到 CoTracker

- 脚本: `code_vjepa_vggt/inspect_groundedsam_vggt_cotracker.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_groundedsam_vggt_cotracker.py
```

- 链接: `http://127.0.0.1:8803/index.html`
- 说明: 看 `GroundingDINO -> SAM2 -> query points -> CoTracker` 整条链路。

## 4. VGGT / CoTracker 对比

### VGGT 与 CoTracker 对比

- 脚本: `code_vjepa_vggt/inspect_vggt_vs_cotracker_compare.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_vggt_vs_cotracker_compare.py \
  --device cuda:0
```

- 链接: `http://127.0.0.1:8805/index.html`
- 说明: 并排对比 VGGT 与 CoTracker 的 tracks 和覆盖效果。

### CoTracker 采样 VGGT 几何

- 脚本: `code_vjepa_vggt/inspect_cotracker_vggt_geometry.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_cotracker_vggt_geometry.py
```

- 链接: `http://127.0.0.1:8807/index.html`
- 说明: 用 CoTracker 轨迹采样 VGGT depth / world points，检查几何稳定性。

## 端口冲突

- `8765`: `serve_phys_state_dataset.py`、`inspect_object_pipeline.py`
- `8767`: `serve_phys_state_training_inputs.py`、`eval_sam2_motion_box_supervision.py`

并行运行时，直接在命令后追加例如 `--port 8865`。
