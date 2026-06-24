# 训练过程可视化说明

## 1. 最终参与辅助 loss 的 `pred track/box/depth` 对比 `GT track/box/depth`

- 脚本  
  - [inspect_train_aux_losses.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_train_aux_losses.py)
- 目的  
  - 直接可视化训练里真正用于计算 `train/loss_track_aux`、`train/loss_box_aux`、`train/loss_depth_aux` 的量
  - 不是额外代理指标，而是直接取自 `ContextVideoTrainer._prepare_batch()` 的 `gt_*` 与 `object_aux_heads` 的 `pred_*`
- 对应训练张量
  - `pred_track_summary = object_aux_out.pred_track_summary`
  - `gt_track_summary = prepared["gt_track_summary"]`
  - `pred_box_xyxy = object_aux_out.pred_box_xyxy`
  - `gt_box_xyxy = prepared["gt_box_xyxy"]`
  - `pred_depth = object_aux_out.pred_depth`
  - `gt_depth = prepared["gt_depth"]`
- 页面展示内容
  - `Track aux overlay`
    - 把 `GT track summary` 和 `Pred track summary` 反解到原始 context frame 上
    - 每个物体显示终点和起点
    - 用于看 `train/loss_track_aux` 到底在监督哪个区域、预测有没有偏
  - `Box aux overlay`
    - 把 `GT box` 和 `Pred box` overlay 到原始 context frame 上
    - 用于看 `train/loss_box_aux` 对应的真实 box 监督是否和物体区域一致
  - `Depth aux panel`
    - 左边是 `GT depth`，右边是 `Pred depth`
    - 只在当前配置真的有 depth target 时显示
- 关键函数
  - `_compute_aux_metrics(...)`
    - 直接按训练公式重算 `track_aux_loss / box_aux_loss / depth_aux_loss`
  - `_render_track_overlay(...)`
    - 负责把 `pred_track_summary` 和 `gt_track_summary` 画到原始视频帧上
  - `_render_box_overlay(...)`
    - 负责把 `pred_box_xyxy` 和 `gt_box_xyxy` 画到原始视频帧上
  - `_render_depth_panel(...)`
    - 负责生成 `GT depth vs Pred depth` 对照视频
  - `_prepare_case(...)`
    - 单个样本的前向、取值、视频导出、结果打包入口
  - `_build_report(...)`
    - 生成本地静态页面 `index.html`

## 2. 当前 object-level 语义

- 当前训练已改成
  - `1 个物体 = 8 个 query 点 + 1 个最终 box`
- 因此这个页面里展示的 `pred track` / `pred box`
  - 都是按物体聚合后的最终监督量
  - 不是单点级别的 32 个 query 分别一套 box
- 具体做法
  - tracker 仍然先输出扁平 query tracks
  - trainer 内部 reshape 成 `[B, T, O, 8, 2]`
  - 再把同一物体的 `8` 个点聚合成一个 object-level summary
  - 然后辅助 head 只对每个物体输出一套 `track summary / box / depth`

## 3. 启动命令

前台启动命令如下：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=0 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_train_aux_losses.py \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_freeze_lora_other_modules_gpu67/step_0000600.pt \
  --indices 0 1 2 \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/aux_loss_vis \
  --port 8810 \
  --fps 30
```

- 本地访问链接
  - `http://localhost:8810/`
- 页面入口文件
  - [index.html](/data/gaoya/AAA_test_video/0623/train/train0624/aux_loss_vis/index.html)
- 汇总数据
  - [metrics.json](/data/gaoya/AAA_test_video/0623/train/train0624/aux_loss_vis/metrics.json)

## 4. 使用建议

- 当 `box_aux` 看起来明显不对时，优先看
  - `GT box` 是否和当前 object slot 对应的是同一个物体
  - `pred track` 是否已经偏离到别的物体上
- 当 `track_aux` 不稳定时，优先看
  - 同一物体的 8 个 query 点是否被错误采到了背景或别的物体
  - `object_valid_mask` 是否把空槽位正确屏蔽
- 当 `depth_aux` 接近 0 或页面没有 depth panel 时，通常表示
  - 当前配置没有有效 depth supervision
  - 或当前样本对应目标 depth 无有效 GT

## 5. v_newtrain 相关推理与监控

- 当前 `old DiffSynth backbone + object branch` 的正式训练分支，不再输出旧格式 `step_0000600.pt`
- 新格式是：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints/step-000600/checkpoint.safetensors`
- 对应推理脚本：
  - [infer_v_newtrain_context_video_wan.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_v_newtrain_context_video_wan.py)
- 对应批量推理脚本：
  - [batch_infer_checkpoints.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/batch_infer_checkpoints.py)
- 对应持续监听脚本：
  - [watch_checkpoint_infer.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/watch_checkpoint_infer.py)
- 三者现在都兼容目录式 checkpoint
- 对 `v_newtrain` 更推荐长期使用：
  - [watch_v_newtrain_batch_infer.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/watch_v_newtrain_batch_infer.py)
  - 原因是它直接复用已经验证稳定的 [batch_infer_checkpoints.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/batch_infer_checkpoints.py)

## 6. v_newtrain 对比版 loss 可视化

- 脚本  
  - [inspect_train_aux_losses_v_newtrain_compare.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_train_aux_losses_v_newtrain_compare.py)
- 目的  
  - 对比多个 `v_newtrain` checkpoint 的辅助 loss 可视化
  - 同一批 case 横向展示，不再混用旧 `ContextVideoTrainer` 入口
  - 额外导出逐帧静态图，方便按帧核查 `track / box` 是否圈到同一物体
- 当前启动命令

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_train_aux_losses_v_newtrain_compare.py \
  --checkpoints \
    /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints/step-000200 \
    /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints/step-001000 \
    /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints/step-001600 \
  --indices 0 1 \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/aux_loss_vis_v_newtrain_compare \
  --port 8814 \
  --fps 30
```

- 本地访问链接
  - `http://localhost:8814/`
- 页面入口文件
  - [index.html](/data/gaoya/AAA_test_video/0623/train/train0624/aux_loss_vis_v_newtrain_compare/index.html)
- 汇总数据
  - [metrics.json](/data/gaoya/AAA_test_video/0623/train/train0624/aux_loss_vis_v_newtrain_compare/metrics.json)
