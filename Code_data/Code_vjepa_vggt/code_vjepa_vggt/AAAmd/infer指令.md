## 0624 train object token gate cross-attn
#### 权重目录

##### Wan 官方底座：冻结
```json
/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
```
##### 0613 LoRA：冻结 
- 说明：通用+仿真视频数据集训练v2v，训练说明在/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/AAAinfer/AAA.md  
- 权重目录：
```json
/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
```
- 运行指令：
```json
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan_no_object_branch.py \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_freeze_lora_other_modules_gpu67/step_0000800.pt \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml \
  --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 \
  --prompt "industrial rigid body simulation sphere" \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_test/wan_lora_no_object_branch \
  --output-video /data/gaoya/AAA_test_video/0623/train/train0624/infer_test/wan_lora_no_object_branch/prediction.mp4 \
  --num-frames 24 \
  --sampling-mode prefix \
  --sampling-steps 40 \
  --fps 30 \
  --seed 42
```

##### object-conditioned 相关模块：训练并保存在 step_*.pt
1. pybullet0624_freeze_lora_other_modules_gpu67
    - 说明：
    每个视频固定8个query point，所以当视频中只有一个物体的时候，最终用来算boxloss的有8个box（GT：1个box）
    - 权重目录：
    ```json
    /data/gaoya/AAA_test_video/0623/train/train0624/infer_test/pybullet0624_freeze_lora_other_modules_gpu67 
    ```
    - 运行指令
    ```json
    PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
    CUDA_VISIBLE_DEVICES=5 \
    /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
    /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/batch_infer_checkpoints.py \
    --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_freeze_lora_other_modules_gpu67 \
    --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml \
    --infer-script /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py \
    --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 \
    --prompt "industrial rigid body simulation sphere" \
    --output-root /data/gaoya/AAA_test_video/0623/train/train0624/infer_test \
    --gpu 5 \
    --num-frames 24 \
    --sampling-mode prefix \
    --sampling-steps 40 \
    --fps 30 \
    --seed 42
    ```
2. 
