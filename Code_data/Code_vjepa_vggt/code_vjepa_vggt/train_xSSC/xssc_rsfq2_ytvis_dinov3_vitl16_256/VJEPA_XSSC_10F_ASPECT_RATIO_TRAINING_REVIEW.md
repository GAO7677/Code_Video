# V-JEPA xSSC：10帧、非因果、原宽高比续训方案与严格审查

## 1. 实验结论与边界

本实验从已完成的非因果 YTVIS-HQ step-10000 xSSC 权重继续，训练一个新的
10原始帧、5个 V-JEPA tubelet 时间步、宽高比分桶版本，目标 step 为20000。
这是一次 **model-weight transfer**，不是原训练状态的无缝 resume：模型可兼容的
权重被继承，optimizer 和 LR schedule 为新阶段重新创建。实验只训练 xSSC；
V-JEPA2.1 ViT-L encoder 仍然冻结。

训练数据是 YTVIS-HQ。MOVi-C 原数据为 256×256，因此“保留原始宽高比”在
MOVi-C 上没有变量意义；本阶段先解决 YTVIS 的时间粒度和非正方形输入问题。

## 2. 已冻结的关键配置

| 项目 | 配置 | 逻辑 |
|---|---:|---|
| source checkpoint | noncausal YTVIS step-10000 | 使用当前最新完整训练结果 |
| source xSSC 时间步 | 3 | 6帧经 tubelet=2 得到3步 |
| target 原始帧 | 10 | 与 DINOv3 的5个帧级监督步对齐 |
| target xSSC 时间步 | 5 | V-JEPA tubelet size 固定为2 |
| 标签帧 | 1,3,5,7,9（零起始） | 每个 tubelet 使用第二帧 mask |
| temporal mode | noncausal | V-JEPA encoder 可在10帧内双向注意 |
| slots | 7 | 与 source 完全一致 |
| slot dim | 512 | 与 source 完全一致 |
| frozen backbone | V-JEPA2.1 ViT-L/16 | 避免小数据集破坏预训练特征 |
| trainable params | 80,779,264 | xSSC encoder-project、slot、transition、decoder |
| GPU | 5、6 | 不使用 GPU4 |
| batch/GPU | 64 | GPU7 实测完整 backward 峰值32.12 GiB |
| accumulation | 3 | 保持 effective global batch=384 |
| optimizer steps | 10000→20000 | 新阶段共10000次更新 |
| validation | 每500 step | 计算完整 val loss 与 slot segmentation 指标 |
| checkpoint | 每1000 step | 保存模型，并维护 `resume-latest.pth` |
| precision | bfloat16 | 不使用 loss scaler |

## 3. 数据读取与时间采样

1. `YTVIS` 从 `train.lmdb` 读取一个视频样本。LMDB 内视频帧是压缩 RGB 图像，
   segmentation 是逐帧的 index mask。
2. 训练时 `StridedRandomSliceSequence(size=10)` 从视频中选择连续10帧。
   长度不足10的11个训练视频重复最后一帧补到10；它们占1678个样本的0.66%。
3. 图像解码后：
   - `video: [10,3,H0,W0]`
   - `segment: [10,H0,W0]`
4. segmentation 取 `[1,3,5,7,9]`，形成：
   - `segment: [5,H,W,S_gt]`
   这与 V-JEPA 每两个原始帧产生一个 tubelet token 的时间语义严格对齐。
5. validation 不随机切10帧，而是保留完整视频；奇数帧重复最后一帧补偶数，
   再按第二帧策略取 mask。因此 validation 仍衡量完整视频的时序稳定性。

审查结论：不存在 feature 为5步而 label 为10步、或 label 错移到 tubelet 第一帧的
off-by-one 问题。

## 4. 原始宽高比与分桶

不能把不同空间尺寸直接 default-collate，也不能简单 pad 后把 padding 当背景：后者
会同时污染 V-JEPA feature、MSE reconstruction loss、slot attention 和 ARI/mBO。
因此采用无 padding 的 shape-homogeneous batch。

训练 buckets（高×宽）为：

- 336×192：portrait，21×12=252 patches；
- 256×256：square，16×16=256 patches；
- 224×288：4:3附近，14×18=252 patches；
- 192×336：16:9附近，12×21=252 patches；
- 144×448：ultrawide，9×28=252 patches。

bucket 选择最小化输入宽高比与 bucket 宽高比的 log-distance。YTVIS-HQ train 的
实测分布与 DDP padding 如下：

| bucket | 原样本 | 分桶后每epoch抽样位 | 补齐重复 |
|---|---:|---:|---:|
| 336×192 | 19 | 32 | 13 |
| 256×256 | 0 | 0 | 0 |
| 224×288 | 66 | 96 | 30 |
| 192×336 | 1589 | 1600 | 11 |
| 144×448 | 4 | 32 | 28 |

以上是最初 batch=16/GPU 的分桶统计。正式训练按要求改为 batch=64/GPU 后，
global batch 为128；各 bucket 需补齐到128的倍数，总抽样位变为2048，新增重复位
370个，相对原数据量为22.05%。宽高比相对误差均值
1.72%，P99为4.76%；极少数约2.13:1视频的最大误差为21.25%。ultrawide 样本被
有意轻度过采样，这是用固定 global batch 保证 DDP 梯度等权的代价，必须在解释
实验时保留该 caveat。

每个 epoch 两个 rank 使用同一 bucket 顺序；每个 global microbatch 有32个样本，
rank0/rank1 各取16个。这样两个 rank 在每次 DDP forward 使用相同 H/W，但样本
不重叠。单元测试已经逐 batch 验证两个 rank 的 bucket 完全一致。

## 5. 空间预处理顺序

训练：

1. 按原视频宽高比选择 bucket；
2. RGB 用 bilinear resize，segment 用 nearest-exact resize；
3. 以0.5概率水平翻转 RGB 与 segment；
4. RGB 用 ImageNet mean/std normalize；
5. segment 取 tubelet 第二帧。

旧版 `RandomCrop + square resize` 被移除，因为它先改变视野和宽高比，再强制拉伸
为正方形，与本实验目的冲突。新方案保留完整帧内容，但这也构成一次明确的数据分布
变化，因此使用新阶段 LR warmup，而不是把 optimizer 当作完全相同的数据流续跑。

## 6. V-JEPA encoder

collate 后典型输入为：

`video [B=16,T_raw=10,C=3,H=192,W=336]`

adapter 检查 H/W 均可被 patch size 16 整除，然后转换为官方输入：

`[16,3,10,192,336]`

V-JEPA2.1 ViT-L 使用 tubelet=2、patch=16 和 3D RoPE，输出：

`tokens [16,5×12×21,1024]`

adapter reshape 为：

`feature [16,5,1024,12,21]`

官方 encoder 的实际 `T/H/W patches` 被传入每个 RoPE attention block，因此矩形
输入不是伪支持。底层已实测 `10帧×224×384 → 1680 tokens`，五个训练 bucket 也
全部做过 forward shape 验证。

encoder 参数 `requires_grad=False`、始终 `eval()`；xSSC forward 对 feature 再执行
`detach()`。所以 reconstruction loss 不会更新 V-JEPA。

## 7. xSSC encoder、slot 与 transition

1. 每个 tubelet feature 从 `[1024,h,w]` 展平为 `[h×w,1024]`；
2. 经过 pre-LN MLP project，维度仍为1024；
3. 第0个时间步由 `NormalShared` 采样7个512维 slot query；
4. 第0步 Slot Attention 迭代3次；后续每步使用 transition query，只迭代1次；
5. `RSFQTransit` 使用过去 slots 与截至当前的 feature tokens 产生下一步 query；
6. transition 的最大时间窗口 `dt` 从3扩为5。

source 的 `m.transit.te.weight` 是 `[3,512]`，target 为 `[5,512]`。迁移规则：

- row 0..2 从 step-10000 精确复制；
- row 3..4 保持 target 模型的新初始化；
- 其他 shape mismatch、unexpected key、非允许 missing key 均直接报错。

真实 checkpoint 审计结果：82个 key 匹配、0 unexpected、0 shape mismatch、
0 disallowed missing；前三行逐元素相同，新增两行未被 source 覆盖。

## 8. decoder、动态位置编码与 reconstruction loss

decoder 使用4层 TransformerDecoder：

- query：V-JEPA spatial feature token；
- memory：7个 slot；
- learned spatial PE：source 的16×16参数网格；
- target PE：按当前 `(h,w)` 用 bilinear 插值，再展平到 `h×w`。

PE 参数本身仍是 `[1,256,1024]`，因此可完整继承 source checkpoint；只改变 forward
时的采样网格。插值路径已验证可反向传播且 gradient finite。

训练 decoder 时会随机 mask/shuffle spatial tokens，并以 `decoder_dt=1` 随机选择
当前或前一时间步的动态分量。decoder 返回 `fsti` 后，feature target 使用相同时间
索引 gather，确保 recon 与 target 对齐。训练目标只有：

`L_recon = mean((recon - stop_grad(V-JEPA feature[fsti]))²)`

slot segmentation 指标不参与梯度。它来自 decoder cross-attention argmax，按当前
bucket 的真实 H/W resize，而不再写死256×256。

## 9. 指标

train 每个 optimizer step 聚合：

- reconstruction MSE；
- mBO；
- gradient norm、clip coefficient、LR、峰值显存。

validation 每500 optimizer step 在280个 val 视频上计算：

- reconstruction MSE；
- ARI；
- foreground ARI；
- mBO；
- mIoU。

绝对 reconstruction MSE 只可在同一个 V-JEPA feature target 内纵向比较，不能与
DINOv3 reconstruction MSE 横向比较。ARI/mBO/mIoU 可以比较，但本实验同时改变了
时间步和空间预处理，因此不能把改善完全归因于单一变量。

## 10. batch、梯度与 DDP

`64 samples/GPU × 2 GPUs × 3 accumulation = 384 samples/update`。

一个 sampler epoch 有16个 microbatches；为保证每次更新正好累积3个 microbatch，
每个 epoch 使用前15个、随机丢弃尾部1个。batch 顺序每个 epoch重洗，所以不是永久
丢弃固定 bucket。该策略与旧训练“丢弃不足一次 accumulation 的 epoch 尾部”一致。

使用 bf16 autocast；因 bf16 不需要 GradScaler。每次更新前执行 global gradient
norm clip，阈值0.05；非有限梯度或被跳过的 optimizer step 会立刻终止训练。

两卡一真实 optimizer-step 集成测试结果：

- train loss 1.6033；
- train mBO 0.1793；
- pre-clip gradient norm 13.5756；
- peak reserved memory 10.88 GiB/rank；
- validation loss/ARI/ARI-FG/mBO/mIoU 全部成功；
- checkpoint bundle 与 metadata 成功生成。

大 gradient norm 是新 temporal rows 和新空间数据分布下的预期风险，已由 clip 和
phase warmup 控制；正式训练前几百步必须监控 clip coefficient，而不能只看 loss。

## 11. optimizer 与 LR

不能恢复 source optimizer：`transit.te` shape 已改变，而且数据分布也改变。因此创建
新的 Adam optimizer，但 optimizer step 仍从10000计数，便于 checkpoint 和 W&B
对齐。

LR schedule 以全局 step-10000 为 phase step 0：

- 10000→10500：0 线性 warmup 到 `5e-5`；
- 10500→20000：cosine decay 到 `5e-8`。

这避免了把全局 step=10000 直接索引旧20k cosine 的中段，从第一步突然使用约
`2.7e-5`。第一步 LR=0 是有意的完整数值/梯度探测；随后逐步增加。

## 12. checkpoint、resume 与 W&B

首次启动使用 `--ckpt-file step-010000.pth`，触发严格 model transfer。每1000 step
保存：

- `step-XXXXXX.pth`：不含冻结 V-JEPA backbone；
- `step-XXXXXX.metadata.json`：variant、step、epoch、world size、effective batch；
- `resume-latest.pth`：optimizer、scaler、随机数状态和下一 sampler epoch。

中断后必须使用 `RESUME_FILE=.../resume-latest.pth`；resume 会检查 target variant，
恢复 optimizer 和 RNG，并从下一个完整 sampler epoch开始，避免 replay 半个 epoch。

W&B project：`xssc_vjepa2_1_video_10f_ar`。日志横轴是全局 optimizer step，起点
为10000，包含 train step metrics、每500 step validation metrics 和系统信息。

## 13. 严格审查结果

已修复的 blocker：

1. adapter 写死256×256、输出写死16×16；
2. decoder learned PE 只支持256个一维 token；
3. segmentation callback 写死 resize 到256×256；
4. variable-size 样本无法 default-collate；
5. DDP 两个 rank 可能在同一步得到不同 H/W；
6. temporal embedding 3→5无法 strict-load；
7. 新阶段 LR 会错误索引全局 schedule；
8. launcher 缺少 target-run resume 路径。
9. launcher 的冷启动数据根一度指向易失的 `/dev/shm`，现已改为持久数据根
   `/data/gaoya/dataset`，并在启动前检查两个实际 LMDB 文件。

验证通过：

- Python compile、shell syntax、4个CPU单元测试；
- 两个 rank 的 bucket order/shape 一致；
- 真实 YTVIS batch shape `[16,10,3,192,336] → [16,5,...]`；
- 五个 spatial bucket forward；
- source checkpoint 严格迁移；
- 单卡真实 batch=16 loss、mBO、backward 和 finite gradients；
- GPU5/6 DDP 一次完整 optimizer step、validation、checkpoint。

保留风险：

- 本实验同时改变时间步与空间预处理，不是单变量 ablation；
- batch=64/GPU 使分桶补齐重复位达到原数据量的22.05%，ultrawide 被明显过采样；
- 前期 gradient clip 可能频繁生效；
- 10帧使 frozen V-JEPA 的时空 attention 成本高于6帧版本；
- 训练 step 不能等价为相同 epoch，因为新 sampler 每个 epoch 的有效样本流不同。

审查裁决：在明确保留上述 caveat、并在正式启动后核验前3个 optimizer step、首个
validation 与 W&B 的前提下，方案可启动；没有剩余 correctness blocker。

## 14. 启动与恢复

首次训练：

```bash
bash run_train_vjepa2_1_video_ytvis_10f_ar_transfer10000_gpu56.sh
```

中断恢复：

```bash
RESUME_FILE=/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_noncausal_ytvis_hq_10f_ar_steps20000/rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-transfer10000/42/resume-latest.pth \
bash run_train_vjepa2_1_video_ytvis_10f_ar_transfer10000_gpu56.sh
```
