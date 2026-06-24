## 0624 train object token gate cross-attn
#### 权重目录

/data/gaoya/AAA_test_video/0623/train/train0624/infer_test/pybullet0624_freeze_lora_other_modules_gpu67 
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