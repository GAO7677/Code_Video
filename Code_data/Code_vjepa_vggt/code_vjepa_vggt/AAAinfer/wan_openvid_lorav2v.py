'''

PYTHONPATH=/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py \
    --input_json_list_path /data/gaoya/AAA_test_video/0623/testjsons/test_100.txt \
    --output_root /data/gaoya/AAA_test_video/0623/test/v2v/loramodel/wan_openvid_lorav2v_step10000 \
    --runtime_root /data/gaoya/AAA_test_video/0623/test/v2v/loramodel/wan_openvid_lorav2v_step10000_runtime \
    --lora_path /data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors \
    --model_name wan_openvid_lorav2v_step10000 \
    --device cuda
'''