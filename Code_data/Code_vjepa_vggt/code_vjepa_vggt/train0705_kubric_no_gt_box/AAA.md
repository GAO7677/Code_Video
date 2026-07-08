# Kubric Stage1b 训练 / 推理运行指令

## 1. 相关脚本

训练脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py
```

正式训练启动脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh
```

smoke 启动脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_smoke_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh
```

批量推理脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py
```


## 2. 数据集

当前 Kubric 训练数据根目录：

```text
/data/gaoya/dataset/nnsriram97-phyco_kubric
```

当前适配后的 `train split` 样本数：

```text
114276
```

说明：

- `KubricNoGTBoxDataset` 使用稳定哈希划分 `train/val/test`。
- `--kubric_init_scan_limit` 只用于截断前若干样本做 smoke，不影响正式训练逻辑。


## 3. 正式训练

直接使用正式训练启动脚本：

```bash
GPU=5 \
OUTPUT_DIR=/data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_train_gpu5 \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh
```

如果要从已有状态继续训练：

```bash
GPU=5 \
OUTPUT_DIR=/data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_train_gpu5 \
RESUME=/data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_train_gpu5/checkpoints/step-xxxxx/training_state.pt \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh
```


## 3.1 多卡正式训练

如果要直接用 `gpu3,5,6,7` 做正式多卡训练，可以不走单卡封装脚本，直接用 `accelerate launch --num_processes 4`：

```bash
env \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=3,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py \
  --diffsynth_root /home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --dataset_type kubric_no_gt_box \
  --kubric_root /data/gaoya/dataset/nnsriram97-phyco_kubric \
  --kubric_split train \
  --kubric_cache_root /data/gaoya/agent-data/cache/kubric_no_gt_box_dataset \
  --kubric_sampling_strategy prefix \
  --height 512 \
  --width 896 \
  --num_frames 24 \
  --fixed_num_context_frames 8 \
  --max_train_steps 20000 \
  --num_epochs 100 \
  --dataset_num_workers 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 1 \
  --optimizer_type paged_adamw8bit \
  --max_grad_norm 1.0 \
  --find_unused_parameters \
  --save_steps 500 \
  --max_checkpoints_keep 10 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path /data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_train_gpu3567 \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
  --extra_inputs input_image \
  --enable_object_branch \
  --freeze_non_object_trainables \
  --train_object_adapter \
  --train_object_dit_branch \
  --object_num_queries 8 \
  --aux_max_objects 4 \
  --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth \
  --jepa_input_size 384 \
  --jepa_patch_size 16 \
  --jepa_tubelet_size 2 \
  --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth \
  --cotracker_input_h 384 \
  --cotracker_input_w 512 \
  --cotracker_window_len 60 \
  --vggt_model_path /data/gaoya/ckpt/facebook-VGGT-1B \
  --vggt_input_h 420 \
  --vggt_input_w 728 \
  --object_pooler_latent_dim 16 \
  --cond_proj_dim 4096 \
  --jepa_window_radius 1 \
  --latent_window_radius 1 \
  --object_gate_init 0.1 \
  --lambda_main 1.0 \
  --lambda_track_aux 0.0 \
  --lambda_box_aux 0.0 \
  --lambda_depth_aux 0.0 \
  --stage1a_init_from /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --grounding_proposal_source gdino_only \
  --grounding_motion_score_ratio 0.15 \
  --grounding_text_prompt "box . cube . block . cylinder . capsule . sphere . ball ." \
  --grounding_disable_caption_terms \
  --grounding_gdino_box_threshold 0.20 \
  --grounding_gdino_text_threshold 0.15 \
  --grounding_prompt_frame_mode first \
  --grounding_track_dedupe_iou_threshold 0.75 \
  --grounding_container_suppress_ratio_threshold 0.95 \
  --grounding_container_suppress_min_contained 2 \
  --grounding_container_suppress_min_area_ratio 1.5 \
  --grounding_container_suppress_small_iou_threshold 0.7 \
  --sam2_segment_len 8 \
  --report_to wandb \
  --wandb_project vjepa_vggt_wan \
  --wandb_name stage1b_kubric_no_gt_box_gpu3567
```

如果要在 `gpu3,5,6,7` 上续训：

```bash
env \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=3,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py \
  --diffsynth_root /home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --dataset_type kubric_no_gt_box \
  --kubric_root /data/gaoya/dataset/nnsriram97-phyco_kubric \
  --kubric_split train \
  --kubric_cache_root /data/gaoya/agent-data/cache/kubric_no_gt_box_dataset \
  --kubric_sampling_strategy prefix \
  --height 512 \
  --width 896 \
  --num_frames 24 \
  --fixed_num_context_frames 8 \
  --max_train_steps 20000 \
  --num_epochs 100 \
  --dataset_num_workers 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 1 \
  --optimizer_type paged_adamw8bit \
  --max_grad_norm 1.0 \
  --find_unused_parameters \
  --save_steps 500 \
  --max_checkpoints_keep 10 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path /data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_train_gpu3567 \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
  --extra_inputs input_image \
  --enable_object_branch \
  --freeze_non_object_trainables \
  --train_object_adapter \
  --train_object_dit_branch \
  --object_num_queries 8 \
  --aux_max_objects 4 \
  --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth \
  --jepa_input_size 384 \
  --jepa_patch_size 16 \
  --jepa_tubelet_size 2 \
  --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth \
  --cotracker_input_h 384 \
  --cotracker_input_w 512 \
  --cotracker_window_len 60 \
  --vggt_model_path /data/gaoya/ckpt/facebook-VGGT-1B \
  --vggt_input_h 420 \
  --vggt_input_w 728 \
  --object_pooler_latent_dim 16 \
  --cond_proj_dim 4096 \
  --jepa_window_radius 1 \
  --latent_window_radius 1 \
  --object_gate_init 0.1 \
  --lambda_main 1.0 \
  --lambda_track_aux 0.0 \
  --lambda_box_aux 0.0 \
  --lambda_depth_aux 0.0 \
  --stage1a_init_from /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --grounding_proposal_source gdino_only \
  --grounding_motion_score_ratio 0.15 \
  --grounding_text_prompt "box . cube . block . cylinder . capsule . sphere . ball ." \
  --grounding_disable_caption_terms \
  --grounding_gdino_box_threshold 0.20 \
  --grounding_gdino_text_threshold 0.15 \
  --grounding_prompt_frame_mode first \
  --grounding_track_dedupe_iou_threshold 0.75 \
  --grounding_container_suppress_ratio_threshold 0.95 \
  --grounding_container_suppress_min_contained 2 \
  --grounding_container_suppress_min_area_ratio 1.5 \
  --grounding_container_suppress_small_iou_threshold 0.7 \
  --sam2_segment_len 8 \
  --report_to wandb \
  --wandb_project vjepa_vggt_wan \
  --wandb_name stage1b_kubric_no_gt_box_gpu3567_resume \
  --stage2_resume_from /data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_train_gpu3567/checkpoints/step-xxxxx/training_state.pt
```


## 4. Smoke Train

本次实际跑通的 smoke 命令：

```bash
OUTPUT_DIR=/data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_smoke_gpu5 \
KUBRIC_INIT_SCAN_LIMIT=64 \
GPU=5 \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_smoke_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh
```

本次实际 smoke 输出目录：


## 5. Test5 多 Context 推理 Sweep

使用 `gpu6,7` 对 `test_5.txt` 跑多组 context 长度，其他配置保持一致：

```bash
VISIBLE_GPU_IDS=6,7 \
INFERENCE_DEVICES=cuda:0,cuda:1 \
CONTEXT_FRAME_VALUES=1,4,8,12,16,20 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_batch_ctx_sweep_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.sh
```

如果要按多个 GPU pair 并行分配不同 `ctx` 任务，可直接传：

```bash
INFERENCE_GPU_PAIRS="6,7 7,6 3,5 5,3" \
INFERENCE_DEVICES=cuda:0,cuda:1 \
CONTEXT_FRAME_VALUES=1,4,8,12,16,20 \
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_batch_ctx_sweep_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.sh
```

上述配置会按 round-robin 分配：

- `6,7` -> `ctx01, ctx16`
- `7,6` -> `ctx04, ctx20`
- `3,5` -> `ctx08`
- `5,3` -> `ctx12`

输出根目录：

```text
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn
```

每个 context 长度一个子目录，例如：

```text
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn/ctx01
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn/ctx04
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn/ctx08
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn/ctx12
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn/ctx16
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn/ctx20
```

```text
/data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_smoke_gpu5
```

本次实际落盘的 checkpoint：

```text
/data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_smoke_gpu5/checkpoints/step-000002
/data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_smoke_gpu5/checkpoints/step-000004
/data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_smoke_gpu5/checkpoints/step-000005
```

本次后续推理实际使用的是：

```text
/data/gaoya/agent-data/checkpoints/stage1b_kubric_no_gt_box_smoke_gpu5/checkpoints/step-000005
```


## 5. Kubric 推理测试

这次先构造了一个只包含 3 条 Kubric case 的列表：

```text
/data/gaoya/agent-data/outputs/tmp_stage1b_kubric_smoke_infer/kubric_test3.txt
```

其内容为：

```text
/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/phyco_kubric_ball_drop_soft_v4_2025-09-05_0144a4.json
/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/phyco_kubric_friction_slide_flat_force_v3_2025-10-07_003c2c.json
/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/phyco_kubric_pool_table_force_2025-09-27_fef01f.json
```

本次实际跑通的批量推理命令：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/stage1b_kubric_no_gt_box_train_gpu3567_stop1067_20260706/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name kubric_stage1b_train_step001000 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare \
  --num-inference-steps 40 \
  --device cuda 
```

推理输出根目录：

```text
/data/gaoya/agent-data/outputs/tmp_stage1b_kubric_smoke_infer
```

本次实际结果目录：

```text
/data/gaoya/agent-data/outputs/tmp_stage1b_kubric_smoke_infer/step-000005
```

结果汇总：

```text
/data/gaoya/agent-data/outputs/tmp_stage1b_kubric_smoke_infer/step-000005/result.json
```

本次推理统计结果：

```text
num_total=3
num_success=3
num_failed=0
num_skipped=0
```


## 6. 辅助检查脚本区别

这 4 个脚本可以按“介入训练链路的深浅”来理解：

- `sample_kubric_overlay_inputs.py`
  作用：只负责从 Kubric/PhyCo 数据集里抽样，生成后续检查脚本可直接使用的输入 json。
  特点：不建模，不跑 viewer grounding，不跑 CoTracker / JEPA / VGGT，也不跑训练前向或 loss。

- `visualize_kubric_actual_train_samples.py`
  作用：查看“训练真正会吃到什么输入”。
  特点：会按训练配置重新采样实际 context 帧，展示 full video、train clip、20-frame context pool、实际 sampled context、grounding prompt boxes、query points、CoTracker overlay 和 metadata。
  限制：只到输入和先验层，不进入完整训练前向，不算 `loss_main`，也不检查 aux 预测框 / 预测轨迹。

- `inspect_kubric_train_forward_aux_overlay.py`
  作用：对训练样本真实跑一次模型前向，检查 aux 分支输出。
  特点：会真实经过 `_build_object_query_priors`、`_run_cotracker`、`_run_vggt`、`_run_jepa`、`object_pooler`、`object_aux_heads`、`object_adapter`，并计算 `loss_main`、`loss_object_context_reg`、`loss_total`。
  输出重点：input overlay、reference box vs aux predicted box、reference track summary vs aux predicted track summary、以及真实前向得到的 metrics。

- `inspect_kubric_actual_train_forward_aux_overlay.py`
  作用：在“真实训练时的 context 采样方式”下，再跑一次前向和 aux overlay。
  特点：先对数据集样本按训练配置重新做实际 context 采样，再复用 `inspect_kubric_train_forward_aux_overlay.py` 的前向和可视化能力。
  额外输出：`context_sampling_mode`、`raw_context_pool_indices`、`actual_context_local_indices`、`actual_context_source_indices`、`sampled_train_frame_indices`、`context_pool.mp4` 等，更贴近训练现场。

怎么选：

- 只想快速造一批检查输入：用 `sample_kubric_overlay_inputs.py`
- 只想看训练输入长什么样：用 `visualize_kubric_actual_train_samples.py`
- 想看真实前向和 aux 监督结果：用 `inspect_kubric_train_forward_aux_overlay.py`
- 想看真实训练采样下的真实前向结果：优先用 `inspect_kubric_actual_train_forward_aux_overlay.py`


## 7. 推理版 ctx sweep 中间可视化脚本

新增脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/inspect_kubric_infer_forward_ctx_sweep.py
```

作用：

- 走 Kubric `stage1b context-only no-GT-box` 的真实推理链路，不走训练 dataset 路径。
- 对同一批推理 case，分别用 `ctx=1,4,8,12,16,20` 做 sweep。
- 每个 `ctx` 都会真实执行：
  `source_video -> context 取帧 -> viewer grounding / CoTracker / VGGT / JEPA / object pooler / object adapter -> pipe() 生成`
- 同时导出“中间前向可视化”与“最终生成视频”，方便对齐：
  实际喂给 `pipe()` 的 context、prompt、可选 `input_image`、前向 overlay、最终生成结果。

运行示例：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2,3 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/inspect_kubric_infer_forward_ctx_sweep.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name train_stage1b_kubric0708_step1000_ctx_sweep \
  --output-root /data/gaoya/agent-data/outputs/kubric_infer_forward_ctx_sweep \
  --inference-devices cuda:0,cuda:1 \
  --num-inference-steps 40 \
  --num-frames 49 \
  --ctx-values 1,4,8,12,16,20
```

输出结构：

- 根目录按 `step-xxxxx` 再分层。
- 每个样本再按 `ctx01/ctx04/ctx08/ctx12/ctx16/ctx20` 分子目录。
- 每个 `ctx` 子目录至少包含：
  `generated_video.mp4`、`context_video.mp4`、`context_sheet.jpg`、`prompt_text.png`、
  `prompt_preview.png`、`input_prepipe_overlay.mp4`、`aux_pred_box_overlay.mp4`、
  `aux_pred_track_overlay.mp4`、`source_full_video.browser.mp4`、`result.json`、`index.html`。
- 若输入 json 里有 `input_image` 或 `first_frame_path`，会额外导出 `input_image.png`。

说明：

- `source_video` 优先作为 context 源；不存在时才回落到 `input_video`。
- 如果源视频帧数小于请求的 `ctx`，会自动降到实际可用帧数，不报错。
- `result.json` 里会记录 `requested_context_frames`、`effective_context_frames`、`frame_indices`、
  prompt / source video / 可选 input image 路径、forward metrics、以及各可视化文件名。
- 这个脚本是“推理版 actual forward overlay”，和
  `inspect_kubric_actual_train_forward_aux_overlay.py` 的区别在于：
  前者输入来自推理 case json，后者输入来自训练 dataset sample。


## 8. 备注

- 禁止使用 `gpu4`。
- `run_smoke_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh` 默认是单卡单进程。
- 如果只想快速验证链路，优先改 `KUBRIC_INIT_SCAN_LIMIT` 和 `OUTPUT_DIR`，不要改原始训练脚本。
