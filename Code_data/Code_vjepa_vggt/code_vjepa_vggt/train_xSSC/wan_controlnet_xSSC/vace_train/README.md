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
  -> ctx first 8 frames are padded to 9 frames
  -> VACE reference video = padded ctx frames, encoded as one short video
  -> frozen official xSSC / RandSFQ2 runs on the full 49-frame training video
  -> slots [B,49,7,256]
  -> raw-frame mask: ctx frames 0..7 visible, future frames masked
  -> Wan VAE-style temporal grouping:
       reference ctx clip 9 frames -> 3 reference latent steps
       target video 49 frames -> 13 video latent steps
       future slot content is replaced by a placeholder before the official inactive/reactive split
  -> learned coordinate query competes over 7 slots
  -> official VACE split with vace_video replaced by vace_slot:
       inactive = vace_slot * (1 - vace_slot_mask)
       reactive = vace_slot * vace_slot_mask
  -> inactive/reactive xSSC condition [B,32,Tz+3,Hvae,Wvae]
  -> mask channels [B,64,Tz+3,Hvae,Wvae], reference mask = 0, ctx visible, future masked
  -> dense VACE context [B,96,Tz+3,Hvae,Wvae]
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
xssc_conditioner.video_norm
xssc_conditioner.future_placeholder
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
full video
-> frozen xSSC slots
-> ctx-visible/future-masked slots-to-dense xSSC VACE condition
-> vace_context [B,96,Tz+3,Hvae,Wvae]
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
2. 当前使用 `ctx frames 0..7` 构造 reference video，并 repeat-pad 到 9 帧后整段 VAE 编码；49 帧训练时会得到 3 个 reference latent。
3. 默认 `xSSC_CONDITION_FRAMES=8` 表示 ctx 可见帧数；xSSC 会抽取完整 49 帧 slots，但 future slot 内容在 VACE 条件中会被 mask 成 placeholder，避免把未来状态泄漏给 VACE。
4. 默认保存完整训练模块 trainable keys；不要使用官方 VACE 脚本里的 `--remove_prefix_in_ckpt pipe.vace.`，否则 xSSC conditioner 权重会和 VACE 权重加载逻辑不一致。
5. 第一版条件 map 是 learned coordinate query over slots，不是每个 Wan layer 用 hidden query 重新算 assignment。
6. xSSC-VACE 的 `vace_video` 不是 RGB/深度/softedge 视频，而是由 xSSC slots 生成的 inactive/reactive 32 通道时序条件；后 64 通道保持 VACE mask 语义。
