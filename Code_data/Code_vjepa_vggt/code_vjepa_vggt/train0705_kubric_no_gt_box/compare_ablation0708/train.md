# compare_ablation0708

当前目录先提供两种 Kubric Stage1B 训练级结构消融：

- `wo_jepa`
  - 从训练结构里移除 JEPA 分支
  - appearance 只保留 latent branch
- `wo_vggt`
  - 从训练结构里移除 VGGT 分支
  - geometry / depth 不再参与 object token 构建

设计原则：

- 不改当前正式训练入口
- 复用当前 Kubric 正式训练的 dataset / ctx 采样 / resume / interrupt checkpoint 逻辑
- 保留当前修好的时间对齐逻辑
- 保留当前训练监控字段
  - `train/sampled_ctx_num_frames`
  - `train/sampled_ctx_last_index`
  - `train/ctx_max_length`
  - `train/jepa_input_frames`
  - `train/jepa_padding_frames`

默认输出根目录：

```text
/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708_structure_ablation
```

默认 wandb project：

```text
vjepa_vggt_wan_structure_ablation0708
```
