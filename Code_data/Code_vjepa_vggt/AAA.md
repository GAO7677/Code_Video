cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt

/data/gaoya/miniconda3/envs/vphy/bin/python -m code_vjepa_vggt.serve_phys_state_dataset \
    --root /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6 \
    --split train \
    --index 0 \
    --output-dir /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/phys_state_dataset_viewer \
    --port 8777



cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python -m code_vjepa_vggt.eval_vggt_sam_multi_object_viewer \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml \
  --ball-block-root /data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block \
  --num-single 0 \
  --num-multi 1 \
  --num-queries 8 \
  --min-queries-per-object 2 \
  --prompt-frame-mode first \
  --gdino-device cpu \
  --sam2-device cpu \
  --ball-num-frames 16 \
  --ball-num-context-frames 16 \
  --output-dir /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/vggt_sam_multi_case \
  --port 8814