# 历史结果索引

以下目录包含早期 V-JEPA 预检、guidance、timestep sweep、重建误差或诊断实验。这里只做索引；未重新审查的目录不视为当前方法结论。

## 1. 当前 Wan/DiT 路线

| 内容 | 目录 |
|---|---|
| Tiny VAE 视频特征 MSE | `/data/gaoya/agent-data/outputs/vjepa2_tinyvae_mse` |
| V-JEPA/Flow 热图 | `/data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps` |
| V-JEPA loss Smoke | `/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_vjepa_loss_smoke` |
| 正式 run 指标与视频 | `/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch` |

## 2. 早期 Wan V-JEPA 探索

```text
/data/gaoya/agent-data/outputs/local_vjepa_diagnostics
/data/gaoya/agent-data/outputs/train0705_vjepa_case025_round1
/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes
/data/gaoya/agent-data/outputs/train0705_vjepa_sweep_case001460_round1
/data/gaoya/agent-data/outputs/vjepa_guidance_case_showcase
/data/gaoya/agent-data/outputs/vjepa_guidance_trace
/data/gaoya/agent-data/outputs/vjepa_mask_union_except_first_case
/data/gaoya/agent-data/outputs/vjepa_phase4_multicase
/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep
/data/gaoya/agent-data/outputs/vjepa_timestep_sweep
/data/gaoya/agent-data/outputs/vjepa_timestep_sweep_1460
/data/gaoya/agent-data/outputs/vjepa_trace_multi
/data/gaoya/agent-data/outputs/vjepa_trace_test5_demo
/data/gaoya/agent-data/outputs/vjepa_wan_precheck
/data/gaoya/agent-data/outputs/vjepa_wan_precheck_dense
/data/gaoya/agent-data/outputs/wanti2v_vjepa_batch_eval
```

## 3. SAVi/V-JEPA 相关分析

已有 Pixel 与 V-JEPA 重建误差分析：

```text
/data/gaoya/agent-data/outputs/AAA_physv/savi_pixel_vjepa_reconstruction_error_val10_physiq4_20260717/analysis_report.md
```

该分析中 V-JEPA dynamic/background 比值约 `0.998`，motion/non-motion proxy 比值约 `1.062`。这说明该特征 MSE 在该 SAVi 设置下对运动区域的区分较弱；其模型、特征维度和任务与当前 Wan 辅助 loss 不同，不能直接外推。

其他相关目录：

```text
/data/gaoya/agent-data/outputs/AAA_physv/savi_vjepa_mask_oom_b16x2_acc2_gpu23_20260717
/data/gaoya/agent-data/outputs/AAA_physv/vjepa_loss_weight_heatmaps_train5_20260717
/data/gaoya/agent-data/outputs/AAA_physv/vjepa_similarity_30f_patch_overlay_physiq025_x0_remaining35_vs01_vs_gt_20260714
/data/gaoya/agent-data/outputs/AAA_physv/vjepa_similarity_physiq025_x0_remaining35_vs01_vs_gt_20260714
/data/gaoya/agent-data/outputs/AAA_physv/vjepa_stable_reconstruction_motion_region_physiq025_20260717
/data/gaoya/agent-data/outputs/AAA_physv/vjepa_stable_reconstruction_pybullet_val10_20260717
/data/gaoya/agent-data/outputs/AAA_physv/vjepa_stable_reconstruction_validation_physiq4_20260717
```

