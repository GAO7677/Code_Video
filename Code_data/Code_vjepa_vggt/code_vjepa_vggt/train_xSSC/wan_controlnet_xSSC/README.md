# Wan ControlNet xSSC Prototype

这个目录用于探索方案 B：用 frozen xSSC slots 构造 Wan 的 ControlNet-style residual condition。

## 官方仓库核查结论

DiffSynth-Studio 官方仓库中有 Wan2.1 / Wan2.2 的通用训练脚本：

- `examples/wanvideo/model_training/train.py`
- `examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B.sh`
- `examples/wanvideo/model_training/full/Wan2.2-TI2V-5B.sh`

官方也有 `Wan-Fun-Control`、`VACE`、camera/motion controller 等控制类模型脚本，但没有发现一个直接面向原生 `Wan2.1-T2V-1.3B` 或 `Wan2.2-TI2V-5B` 的独立 ControlNet 训练脚本。最接近我们需要的结构是 VACE：它先产生 block-wise hints，然后在 Wan denoiser block loop 中做 residual injection：

```text
x = x + current_vace_hint * vace_scale
```

因此本目录把官方 Wan 训练入口复制过来，并以 VACE 的 residual hook 作为方案 B 的实现参照。

## 已复制文件

- `diffsynth_wan_train.py`：来自官方 `examples/wanvideo/model_training/train.py`
- `run_official_wan21_13b_full.sh`：来自官方 Wan2.1-1.3B full training 示例
- `run_official_wan22_ti2v_5b_full.sh`：来自官方 Wan2.2-TI2V-5B full training 示例
- `reference_wan_video_vace.py`：来自官方 `diffsynth/models/wan_video_vace.py`，仅作为 residual hint 结构参考

## 方案 B 当前设计

对于每个训练视频：

```text
video frames
  -> frozen xSSC
  -> slots [B, T_slot, K=7, 256]
  -> temporal align to Wan latent time [B, T_lat, 7, 256]
```

在 Wan denoiser 中，latent 已经被 patchify 成：

```text
latent query tokens [B, T_lat * H_lat * W_lat, C_wan]
```

对每一个 latent 时刻、每一个空间位置，用该位置 hidden state 作为 query，在同一 latent 时刻的 7 个 xSSC slots 之间做 softmax 竞争归属：

```text
assignment = softmax(Q @ K_slot, dim=slot)
dense_cond = sum_slot assignment * V_slot
```

然后通过 zero-init residual projection 生成每个注入层的 hint：

```text
hints[layer] = zero_linear(dense_cond) * gate[layer]
x = x + hints[layer]
```

第一版建议只注入少数层，例如之前可视化中比较有代表性的 `layer11` 和 `layer29`。

## 已实现文件

- `xssc_slot_control_adapter.py`

其中 `XSSCSlotAssignmentControlAdapter` 已实现：

- xSSC slots `[B,T,K,256]` 到 latent time 的 `linear` / `window_mean` 对齐
- latent query 与同一时刻 slots 的 slot-dim softmax 竞争
- dense condition map/token 构造
- zero-init per-layer residual hints
- gate 参数，初始为 0

## 还没有贸然执行的部分

还没有直接修改 DiffSynth-Studio 官方源码，也没有启动训练。原因是这一步涉及模型 forward hook，需要确认我们采用哪种集成方式：

1. 本地 monkey patch `model_fn_wan_video`，只影响本实验目录。
2. 复制 `wan_video.py` 中 `model_fn_wan_video` 到本目录，加入 `xssc_control_adapter/xssc_slots` 参数后在训练脚本中调用。
3. 直接 patch DiffSynth-Studio 源码。这个最方便但污染官方仓库，不建议作为第一版。

我建议采用第 2 种：复制 `model_fn_wan_video` 的相关 forward 到本目录，最小修改 block loop，训练入口只在本实验目录引用它。

## 需要确认

- 这版方案 B 是基于 Wan2.2-TI2V-5B 继续做，还是先用 Wan2.1-T2V-1.3B 降低工程和显存压力？
- xSSC slots 使用完整 49 帧 oracle，还是先使用 ctx 8 帧对齐到 latent time？
- 注入层第一版是否固定为 `11,29`？

如果不额外指定，我建议第一版使用：

```text
Wan2.2-TI2V-5B
49-frame oracle xSSC slots
temporal_align=linear
control_layers=11,29
Wan/xSSC frozen
trainable: slot projection, query/key/value projection, zero-linear hints, gates
```

