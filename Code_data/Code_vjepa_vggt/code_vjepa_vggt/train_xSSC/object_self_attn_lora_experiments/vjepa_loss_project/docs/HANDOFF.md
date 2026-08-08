# V-JEPA / Flow Loss 可视化项目交接文档

## 1. 项目目标

这个项目用于把 V-JEPA 和 Flow matching 的局部 loss 可视化到视频帧上，比较以下几类结果：

- `step03463_lora` vs `no_step03463_lora`
- PyBullet 多物体 case
- v2v JSON case
- native rect 输入 `384x672`
- 全视频 V-JEPA 输入
- Flow 高区域加权后的 V-JEPA feature map

当前结果已经整理到本地静态页面，可直接通过浏览器查看。

## 2. 当前状态

- 结果目录：`/data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps/pybullet_multiobject_step03463_compare`
- 当前前台可视化服务：`127.0.0.1:8787`
- 运行中的服务命令：

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -m http.server 8787 --bind 127.0.0.1 --directory /data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps/pybullet_multiobject_step03463_compare
```

- 当前总入口页：[`index.html`](./index.html)
- v2v case 列表：[`v2v_json_case_list.txt`](./v2v_json_case_list.txt)

## 3. 已完成的可视化产物

### PyBullet 路线

- `step03463_lora/`
- `no_step03463_lora/`
- `step03463_lora_full_video_native_rect/`
- `no_step03463_lora_full_video_native_rect/`
- `step03463_lora_full_video_native_rect_flowweighted_vjepa_a2/`
- `no_step03463_lora_full_video_native_rect_flowweighted_vjepa_a2/`
- `step03463_lora_actual_frames/`
- `no_step03463_lora_actual_frames/`

### v2v JSON 路线

- `step03463_lora_v2v_jsons_native_rect_flowweighted_vjepa_a2/`
- `no_step03463_lora_v2v_jsons_native_rect_flowweighted_vjepa_a2/`

### 汇总页面

- `all_losses_one_page_flowweighted_vjepa_a2.html`
- `comparison_step03463_pybullet_multiobject_flowweighted_vjepa_a2.html`
- `comparison_step03463_pybullet_multiobject_all_frames_flowweighted_vjepa_a2.html`
- `comparison_step03463_v2v_jsons_flowweighted_vjepa_a2.html`
- `comparison_step03463_v2v_jsons_all_frames_flowweighted_vjepa_a2.html`

## 4. 关键执行方式

当前脚本入口是：

```bash
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/visualize_vjepa_loss_heatmaps.py
```

典型启动方式：

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/visualize_vjepa_loss_heatmaps.py \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/configs/formal_full_sa_no_object_gpu27_vjepa_loss.json \
  --checkpoint /data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_no_object_gpu01_formal_vjepa_loss/20260805T180305Z/checkpoints/interrupted-latest \
  --output-root /data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps \
  --run-tag no_step03463_lora_v2v_jsons_native_rect_flowweighted_vjepa_a2 \
  --num-cases 11 \
  --seed 3463 \
  --gpu-set 0,1 \
  --vjepa-input-mode native_rect \
  --case-selection json_list \
  --json-case-list /data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps/pybullet_multiobject_step03463_compare/v2v_json_case_list.txt \
  --model-condition no_step03463_lora \
  --compare-model-run-tag step03463_lora_v2v_jsons_native_rect_flowweighted_vjepa_a2
```

说明：

- 这里的 `--gpu-set 0,1` 对应用户要求的 `gpu01`。
- 不要使用 GPU 4。
- `step03463_lora` 会加载 step03463 checkpoint。
- `no_step03463_lora` 不是“保留 0 初始化 LoRA”，而是直接不加载对应 step LoRA。

## 5. 代码改动要点

`visualize_vjepa_loss_heatmaps.py` 已经扩展了以下能力：

- 支持 `--case-selection json_list`
- 支持 `--json-case-list`
- 支持自定义 `--gpu-set`
- 支持从 JSON case 里直接读取 `source_video` 和 `input_video`
- 保留 JSON 列表的顺序和重复项
- Flow 权重相关的 quantile 统计改成了 CPU `np.quantile`，避免大张量 `torch.quantile` 的内存/性能问题

相关的核心计算链路：

- Flow 反推 x0：

```python
pred_x0_raw = latent_xt - sigma * model_output
```

- 条件帧恢复：

```python
pred_x0 = restore_condition_latents(...)
```

## 6. `pred_x0` 为什么看起来像 GT

这个点后续很容易误解，建议直接记住结论：

- `pred_x0` 不是纯自由预测结果
- 它先由 `x_t - sigma * v_pred` 反推出 latent x0
- 然后会把条件帧 / context latent 直接替换回 GT latent
- 最后再 decode 成 `pred_x0.mp4`

因此页面里看到的 `Predicted x0`，更准确地说是：

> `pred_x0_context_restored`

而不是“完全 raw 的模型输出”。

这也是为什么它肉眼会和 `GT` 非常接近，尤其是前面的 context 段。

## 7. 已知实验结论

- Flow / V-JEPA 的可视化页面已经生成并可浏览。
- v2v JSON 这条链路已经补齐，`11` 个 case 都在结果目录里。
- `step03463_lora` 和 `no_step03463_lora` 的比较页都已生成。
- 当前首页已经包含 v2v 比较入口。

从实际视频差异统计看：

- `pred_x0` 和 `GT` 的差异很小，但不是零。
- context 段因为条件恢复，和 GT 更接近。
- future 段仍然存在可测差异。

## 8. 目录约定

### 输入

- 配置文件：`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/configs/formal_full_sa_no_object_gpu27_vjepa_loss.json`
- checkpoint：`/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_no_object_gpu01_formal_vjepa_loss/20260805T180305Z/checkpoints/interrupted-latest`
- v2v case list：`/data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps/pybullet_multiobject_step03463_compare/v2v_json_case_list.txt`

### 输出

- 所有结果都应继续落到：

```text
/data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps/
```

- 不要把数据集、权重、checkpoint 或大体积产物放到 `/home/gaoya`。

## 9. 后续建议

- 如果要继续检查 `pred_x0`，建议把页面拆成两列：
  - `raw pred_x0 before restore`
  - `restored pred_x0 used for V-JEPA loss`
- 如果要继续分析 V-JEPA 和 Flow 的空间关系，优先看：
  - `comparison_step03463_v2v_jsons_flowweighted_vjepa_a2.html`
  - `comparison_step03463_v2v_jsons_all_frames_flowweighted_vjepa_a2.html`
- 如果要重新启动前台静态服务，直接用上面的 `python -m http.server` 命令。

## 10. 相关文件

- `index.html`
- `v2v_json_case_list.txt`
- `comparison_step03463_v2v_jsons_flowweighted_vjepa_a2.html`
- `comparison_step03463_v2v_jsons_all_frames_flowweighted_vjepa_a2.html`
- `step03463_lora_v2v_jsons_native_rect_flowweighted_vjepa_a2/index.html`
- `no_step03463_lora_v2v_jsons_native_rect_flowweighted_vjepa_a2/index.html`
