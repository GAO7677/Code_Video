torchrun --nnodes=1 --nproc_per_node=8 \
    fastvideo/train_wan_v2v.py \
    --exp_name v2v \
    --model_id models/Wan2.2-TI2V-5B-Diffusers \
    --data_json_path /robby/share/MM/zhangqiyuan/code/PhysRVG/data/data.jsonl \
    --data_repeat 1000 \
    --output_dir exp \
    --seed 42 \
    --fps 15 \
    --train_batch_size 1 \
    --gradient_checkpointing \
    --learning_rate 1e-5 \
    --weight_decay 0.0001 \
    --lr_warmup_steps 100 \
    --dataloader_num_workers 4 \
    --train_epoch 999999 \
    --max_train_steps 999999 \
    --checkpoints_total_limit 4 \
    --max_grad_norm 1.0 \
    --height 480 \
    --width 832 \
    --num_frames 49 \
    --gradient_accumulation_steps 1 \
    --sample_step 16 \
    --checkpointing_steps 10 \

