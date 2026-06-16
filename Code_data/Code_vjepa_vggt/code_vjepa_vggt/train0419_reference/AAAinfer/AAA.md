0. 通用数据集微调 V2V
- 数据集是 dataset_mix_config.json，里面混了 openvid、movi_d 和 genesis_rigid
- 脚本
    - /home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_train.sh
- 权重
    - /data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors  

1. （0613pybullet数据微调）
- 脚本  
    - /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/run_train_phys_state_lora_continue.sh  
- 权重
    - /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
- 推理脚本
    - /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer.sh