# 可视化服务目录

这份目录只收录“会在本地端口起 HTTP 页面”的可视化脚本。所有命令默认以前台方式运行，不使用后台启动。

## 维护约定

- 新增或修改本地端口可视化脚本时，同步更新这份文件。
- 启动命令统一保留前台写法，直接可复制运行。
- 链接统一写成 `http://127.0.0.1:<port>/index.html`。
- 默认端口有冲突时，不并行复用；需要并行运行时显式追加 `--port <新端口>`。

## 通用前缀

下面所有命令都使用同一个 Python 环境和 `PYTHONPATH`：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python
```

## 服务列表

### 1. PhysState 数据集采样查看

- 脚本: `code_vjepa_vggt/serve_phys_state_dataset.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/serve_phys_state_dataset.py
```

- 链接: `http://127.0.0.1:8765/index.html`
- 说明: 查看 `phys_state_0601` 数据集中抽到的 context 帧、对应 box 和元数据。

### 2. PhysState 训练输入查看

- 脚本: `code_vjepa_vggt/serve_phys_state_training_inputs.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/serve_phys_state_training_inputs.py
```

- 链接: `http://127.0.0.1:8767/index.html`
- 说明: 查看训练真正送入模型的 `full video` 和 `context video`。

### 3. Object Pipeline 形状检查

- 脚本: `code_vjepa_vggt/inspect_object_pipeline.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_object_pipeline.py \
  --serve
```

- 链接: `http://127.0.0.1:8765/index.html`
- 说明: 生成并打开 object pipeline 报告，重点看 shape、tracks、object tokens 和 fused context。

### 4. VGGT 与 Box 监督对比

- 脚本: `code_vjepa_vggt/eval_vggt_box_supervision.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_vggt_box_supervision.py
```

- 链接: `http://127.0.0.1:8766/index.html`
- 说明: 对比 VGGT query tracks 与 GT box 的贴合情况。

### 5. SAM2 Motion Prompt 与 Box 监督对比

- 脚本: `code_vjepa_vggt/eval_sam2_motion_box_supervision.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_sam2_motion_box_supervision.py
```

- 链接: `http://127.0.0.1:8767/index.html`
- 说明: 对比 SAM2 motion prompt 生成的轨迹和 box 监督的一致性。

### 6. VGGT + SAM Prior 查看

- 脚本: `code_vjepa_vggt/eval_vggt_sam_prior.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_vggt_sam_prior.py
```

- 链接: `http://127.0.0.1:8771/index.html`
- 说明: 查看加入 SAM query prior 以后，VGGT tracking 在原速和慢放下的表现。

### 7. VGGT 多目标 SAM Viewer

- 脚本: `code_vjepa_vggt/eval_vggt_sam_multi_object_viewer.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/eval_vggt_sam_multi_object_viewer.py
```

- 链接: `http://127.0.0.1:8783/index.html`
- 说明: 查看多目标情况下 GroundingDINO / SAM2 检测、query 分配和 VGGT 轨迹。

### 8. SAM2 Motion Prompt JSON Viewer

- 脚本: `code_vjepa_vggt/inspect_sam2_motion_prompt_jsons.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_sam2_motion_prompt_jsons.py
```

- 链接: `http://127.0.0.1:8797/index.html`
- 说明: 读取 GT JSON 对应源视频，查看 motion prompt、SAM2 mask 和结果叠加。

### 9. Grounded-SAM 到 CoTracker 链路查看

- 脚本: `code_vjepa_vggt/inspect_groundedsam_vggt_cotracker.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_groundedsam_vggt_cotracker.py
```

- 链接: `http://127.0.0.1:8803/index.html`
- 说明: 查看 `GroundingDINO -> SAM2 -> query points -> CoTracker` 的完整链路。

### 10. VGGT 与 CoTracker 对比

- 脚本: `code_vjepa_vggt/inspect_vggt_vs_cotracker_compare.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_vggt_vs_cotracker_compare.py \
  --device cuda:0
```

- 链接: `http://127.0.0.1:8805/index.html`
- 说明: 并排对比 VGGT 与 CoTracker 的 query tracks 和覆盖效果。

### 11. CoTracker 轨迹采样 VGGT 几何

- 脚本: `code_vjepa_vggt/inspect_cotracker_vggt_geometry.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_cotracker_vggt_geometry.py
```

- 链接: `http://127.0.0.1:8807/index.html`
- 说明: 用 CoTracker 的轨迹去采样 VGGT depth / world points，检查几何稳定性。

### 12. Query Points Overlay Viewer

- 脚本: `code_vjepa_vggt/inspect_vggt_query_points_overlay.py`
- 启动命令:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_vggt_query_points_overlay.py
```

- 链接: `http://127.0.0.1:8777/index.html`
- 说明: 查看 query points、SAM prior、VGGT tracks 和可选 CoTracker 叠加结果。

## 默认端口冲突

- `8765`: `serve_phys_state_dataset.py` 和 `inspect_object_pipeline.py`
- `8767`: `serve_phys_state_training_inputs.py` 和 `eval_sam2_motion_box_supervision.py`

需要并行运行时，直接在命令后追加例如 `--port 8865`。
