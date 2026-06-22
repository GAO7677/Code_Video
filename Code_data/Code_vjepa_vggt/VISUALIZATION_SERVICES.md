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

这条链路对应总入口 `http://localhost:8810/` 当前保留的最小页面集：

`Step 1 训练输入 -> Step 2-5 对象级条件提取总览`

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

### Step 2-5. 看对象级条件提取总览

- 入口页: `visualization_hub/step2_5_pipeline.html`
- 链接: `http://localhost:8810/visualization_hub/step2_5_pipeline.html`
- 说明: 只保留“进入 object pooler 前的最终 active tracks”主视图，主页面用 `track_source_compare` 展示完整轨迹链路，并在同页补 `vggt_box_eval_viewer` 作为 GT 对照。

## 1. 数据与输入

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

## 端口冲突

- `8765`: `serve_phys_state_dataset.py`、`inspect_object_pipeline.py`
- `8767`: `serve_phys_state_training_inputs.py`、`eval_sam2_motion_box_supervision.py`

并行运行时，直接在命令后追加例如 `--port 8865`。
