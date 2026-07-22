# xSSC-VACE Condition Training

本目录先复制 DiffSynth 官方 Wan VACE 训练/验证/推理脚本，再基于官方训练入口做 xSSC 条件源适配。

## 复制的官方脚本

官方 VACE 脚本放在：

```text
upstream_vace_scripts/
```

包含：

- `model_training/train.py`
- `model_training/full/*VACE*.sh`
- `model_training/lora/*VACE*.sh`
- `model_training/validate_full/*VACE*.py`
- `model_training/validate_lora/*VACE*.py`
- `model_inference/*VACE*.py`
- `model_inference_low_vram/*VACE*.py`
- `diffsynth_models/wan_video_vace.py`
- `state_dict_converters/wan_video_vace.py`

这些文件是参考副本，不直接修改。

## 本实验新增文件

- `xssc_vace_condition.py`
- `train_xssc_vace_condition.py`
- `run_train_xssc_vace_wan21_13b.sh`

## 方法流程

第一版只做条件层面的适配，不改官方 VACE 的 layer mapping，也不加自定义 Wan layer hook。

```text
training video [49 frames]
  -> take ctx first 8 frames
  -> frozen official xSSC / RandSFQ2
  -> slots [B,8,7,256]
  -> Wan VAE temporal grouping:
       frame 0 -> latent condition 0
       frame 1..4 mean -> latent condition 1
       frame 5..7 mean -> latent condition 2
       future latent steps repeat last ctx condition
  -> learned coordinate query competes over 7 slots
  -> dense VACE context [B,96,Tz,Hvae,Wvae]
  -> official VaceWanModel
  -> official Wan VACE residual hint injection
```

`xSSC`、Wan DiT、VAE、text encoder 均冻结；默认训练：

```text
pipe.vace
xssc_conditioner.slot_norm
xssc_conditioner.slot_key
xssc_conditioner.slot_value
xssc_conditioner.coord_query
xssc_conditioner.output_norm
```

官方 xSSC 本身不会训练。

## 与官方 VACE 的差异

官方 VACE 条件输入是：

```text
vace_video / vace_video_mask / vace_reference_image
-> VAE latents + mask latents
-> vace_context [B,96,Tz,Hvae,Wvae]
```

本实验改成：

```text
video ctx frames
-> frozen xSSC slots
-> learned slots-to-dense condition
-> vace_context [B,96,Tz,Hvae,Wvae]
```

VACE 分支结构和 Wan block residual 注入方式保持官方版本。

## 一键运行

默认脚本：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/wan_controlnet_xSSC/vace_train/run_train_xssc_vace_wan21_13b.sh
```

常用覆盖变量：

```bash
DATASET_BASE_PATH=/path/to/dataset \
DATASET_METADATA_PATH=/path/to/metadata.csv \
OUTPUT_PATH=/path/to/output \
CUDA_VISIBLE_DEVICES=0 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/wan_controlnet_xSSC/vace_train/run_train_xssc_vace_wan21_13b.sh
```

## 重要注意

1. 当前脚本使用官方 `UnifiedDataset`，要求 metadata 至少能提供 `video` 和 `prompt`。
2. 当前不使用 `vace_reference_image`，避免 reference latents 改变时间轴。
3. 默认 `xSSC_CONDITION_FRAMES=8`，避免未来泄漏。
4. 默认保存完整训练模块 trainable keys；不要使用官方 VACE 脚本里的 `--remove_prefix_in_ckpt pipe.vace.`，否则 xSSC conditioner 权重会和 VACE 权重加载逻辑不一致。
5. 第一版条件 map 是 learned coordinate query over slots，不是每个 Wan layer 用 hidden query 重新算 assignment。

