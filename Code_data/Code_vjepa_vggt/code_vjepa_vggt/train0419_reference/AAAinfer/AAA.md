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

2. （0624 基于0613 LoRA 冻结LoRA，只训练其他模块）
- 目的
    - 先加载 0613 phys-state 微调得到的 Wan LoRA
    - 冻结 Wan LoRA，不再更新 LoRA 参数
    - 只训练其他可训练模块：object-conditioned 分支、object_pooler、object_adapter、object_aux_heads
- 初始化 LoRA 权重
    - /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
- 配置
    - /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml
- 启动脚本
    - /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/run_train_0624_freeze_lora_other_modules_gpu67.sh
- 输出目录
    - /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_freeze_lora_other_modules_gpu67
- wandb
    - project: vjepa_vggt_wan
    - run_name: pybullet0624_freeze_lora_other_modules_gpu67
- 关键配置说明
    - `init_wan_lora_from_checkpoint` 指向上述 0613 LoRA safetensors
    - `freeze_wan_lora: true`
    - `freeze_wan_dit: true`
    - 含义是：Wan 主干冻结、LoRA 冻结，但 object 条件相关分支和外部条件模块继续训练
