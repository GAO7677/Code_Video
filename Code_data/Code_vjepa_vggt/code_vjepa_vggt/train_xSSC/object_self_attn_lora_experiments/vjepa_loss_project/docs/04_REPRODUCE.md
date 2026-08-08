# 复现入口

## 1. 训练

训练脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/train_xssc_object_self_attn_lora_vjepa_loss.py
```

前台启动命令：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/run_train_from_config_vjepa_loss.sh \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/configs/formal_full_sa_no_object_gpu27_vjepa_loss.json
```

注意：该命令使用当前的 49/full 配置。历史 step 3463 run 的精确命令保存在：

```text
/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_no_object_gpu01_formal_vjepa_loss/20260805T180305Z/resolved_experiment_config.json
```

## 2. 权重与 checkpoint

| 组件 | 路径 |
|---|---|
| Wan2.2 TI2V-5B | `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B` |
| V-JEPA2.1 ViT-L | `/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt` |
| Tiny VAE | `/data/gaoya/ckpt/taew2_2.pth` |
| 中断 checkpoint | `/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_no_object_gpu01_formal_vjepa_loss/20260805T180305Z/checkpoints/interrupted-latest` |

## 3. 离线特征 MSE

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/compute_vjepa2_feature_mse.py \
  --device cuda:0
```

结果：

```text
/data/gaoya/agent-data/outputs/vjepa2_tinyvae_mse/20260805T163347Z/mse_results.json
```

## 4. 热图与页面

生成脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/visualize_vjepa_loss_heatmaps.py
```

结果目录：

```text
/data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps/pybullet_multiobject_step03463_compare
```

前台静态服务命令：

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -m http.server 8787 \
  --bind 127.0.0.1 \
  --directory /data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps/pybullet_multiobject_step03463_compare
```

完整热图生成参数见 [HANDOFF.md](HANDOFF.md)。禁止使用 GPU 4。

## 5. Ball-block pairwise 特征差异

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/visualize_ball_block_pairwise.py \
  --device cuda:0 \
  --input-dir /data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block \
  --output-dir /data/gaoya/agent-data/outputs/vjepa_ball_block_pairwise/RUN_NAME
```
