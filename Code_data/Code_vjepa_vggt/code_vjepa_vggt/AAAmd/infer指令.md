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

## 0624 v_newtrain old DiffSynth backbone + object branch

### 1. 单个 checkpoint 推理

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_v_newtrain_context_video_wan.py \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints/step-000400 \
  --context-video /data/gaoya/dataset/physics-iq-benchmark/full-videos/take-1/30FPS/0002_full-videos_30FPS_perspective-center_take-1_trimmed-ball-and-block-fall.mp4 \
  --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement." \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/ \
  --output-video /data/gaoya/AAA_test_video/0623/train/train0624//prediction.mp4
```

- 说明
  - `--checkpoint` 可以直接传 `step-000400` 目录
  - 脚本会自动解析其中的 `checkpoint.safetensors`
  - 输入是固定 `8` 帧 context video
  - 输出是 `24` 帧、`512x896`、`fps=30` 的生成视频

### 2. 批量跑当前所有 checkpoint

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/batch_infer_checkpoints.py \
  --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml \
  --infer-script /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_v_newtrain_context_video_wan.py \
  --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 \
  --prompt "industrial rigid body simulation sphere" \
  --output-root /data/gaoya/AAA_test_video/0623/train/train0624/infer_v_newtrain_batch \
  --gpu 5 \
  --num-frames 24 \
  --sampling-mode prefix \
  --sampling-steps 40 \
  --fps 30
```

- 输出结构
  - 例如 `step-000400` 会生成：
    - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_v_newtrain_batch/checkpoints/step-000400.mp4`
    - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_v_newtrain_batch/checkpoints/step-000400.json`
    - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_v_newtrain_batch/checkpoints/step-000400/result.json`
    - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_v_newtrain_batch/checkpoints/step-000400/infer.log`

### 3. 持续监听新 checkpoint 自动推理

前台启动命令：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=2 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/watch_v_newtrain_batch_infer.py \
  --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints \
  --output-root /data/gaoya/AAA_test_video/0623/train/train0624/infer_v_newtrain_batch \
  --infer-script /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_v_newtrain_context_video_wan.py \
  --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 \
  --prompt "industrial rigid body simulation sphere" \
  --gpu 2 \
  --num-frames 24 \
  --sampling-mode prefix \
  --sampling-steps 40 \
  --fps 30
```

- 推荐原因
  - 这个脚本内部复用 `batch_infer_checkpoints.py`
  - 当前已经真实验证 `batch_infer_checkpoints.py` 可稳定跑通 `step-000200 / 000400 / 000600 / 000800 / 001000`
  - 更适合 `v_newtrain` 这条 old DiffSynth backbone 训练线长期维护
  - 运行状态会写到：
    - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_v_newtrain_batch/watch_v_newtrain_status.json`

### 4. 旧 watcher 说明

前台启动命令：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/watch_checkpoint_infer.py \
  --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_diffsynth_object_v_newtrain_gpu67/checkpoints \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml \
  --infer-script /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_v_newtrain_context_video_wan.py \
  --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 \
  --prompt "industrial rigid body simulation sphere" \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_v_newtrain_watch \
  --gpu 5 \
  --num-frames 24 \
  --sampling-mode prefix \
  --sampling-steps 40 \
  --fps 30 \
  --process-existing
```

- 说明
  - 这个 watcher 现在同时支持两种 checkpoint 格式：
    - `step_0000600.pt`
    - `step-000600/checkpoint.safetensors`
  - `--process-existing` 会先把当前目录里已经存在的 checkpoint 也补跑一遍
  - 如果只想监听后续新 checkpoint，不加 `--process-existing`
  - 但对于 `v_newtrain`，当前实测存在 CUDA OOM / illegal memory access 风险，不建议作为主链路
