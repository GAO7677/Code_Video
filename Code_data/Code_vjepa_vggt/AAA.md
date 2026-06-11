cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt

/data/gaoya/miniconda3/envs/vphy/bin/python -m code_vjepa_vggt.serve_phys_state_dataset \
    --root /data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6 \
    --split train \
    --index 0 \
    --output-dir /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/phys_state_dataset_viewer \
    --port 8777