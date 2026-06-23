# 0624 训练链路排查记录

## 当前目标

- 让 object 条件分支真正参与主去噪损失回传，而不是只靠 `object_aux_heads`
- 在正式训练前，把每一步问题、原因、修复和验证记录清楚

## 阶段 1：先把 smoke test 跑通

### 问题

- 初始 smoke test 卡在 `construct_trainer`
- 之后又遇到 `meta tensor`、分布式初始化、dataset 冷启动过慢等问题

### 原因

- WAN 新增 object 分支参数不在原 checkpoint 中，`from_pretrained(..., low_cpu_mem_usage=True)` 时会残留 `meta`
- smoke 脚本误调用 `trainer.train()`，触发 `accelerate` 的分布式入口
- `PhysStateEpisodeDataset` 初始化时会扫描全部 3600 个样本做 context 过滤，单次 smoke 冷启动过重

### 解决

- `WanModel.from_pretrained` 改成 `low_cpu_mem_usage=False`
- smoke 脚本改成只切 `train mode`，不走真正的训练 launcher
- 增加 `init_scan_limit`，只在 smoke 时限制 dataset 初始化扫描范围

### 证据

- `/data/gaoya/AAA_test_video/0623/train/smoke_test/wan_init_profile.json`
- `/data/gaoya/AAA_test_video/0623/train/smoke_test/smoke_report.json`

## 阶段 2：确认 object 分支没有形成有效主损失梯度

### 现象

- smoke 可以完整跑到 `forward/backward`
- `object_pooler` 和 `object_aux_heads` 有梯度
- `object_adapter` 的梯度统计为 0

### 初步怀疑

- `object_gate` 初值为 0，object 分支被门控完全关死
- WAN 内新增 object 分支参数虽然被标记为 trainable，但实际没有进入主损失路径

### 处理

- 给 `object_gate` 设置小的非零初值 `0.1`
- 将 `object_cross_attn / norm4 / object_gate / object_embedding` 从冻结主干中单独解冻

### 结果

- WAN object 分支参数被纳入 trainable 集合
- 但 `object_adapter` 仍然没有拿到有效主损失梯度

## 阶段 3：把断点从“猜测”收敛成证据

### 诊断 1：最小梯度链路

#### 观测

- `object_latent_tokens.grad_abs_sum > 0`
- `object_context.grad_abs_sum == 0`

#### 结论

- 梯度不是丢在 `object_pooler`
- 梯度断在 `object_latent_tokens -> object_adapter -> object_context -> WAN object branch` 的后半段

### 诊断 2：WAN object 分支参数梯度

#### 观测

- `object_cross_attn.q/k/v`、`norm4`、`object_embedding` 梯度几乎全 0
- 只有 `object_cross_attn.o.bias` 有明显非零梯度

#### 结论

- 主损失确实进入了 object 分支末端残差位置
- 但 object attention 主体没有形成正常前向输出

### 诊断 3：前向数值检查

#### 观测

- `object_embedding_in_absmax ≈ 4.5`
- `object_embedding_out_absmax = 0.0`
- `block0_q_absmax ≈ 5.7e-31`
- `block0_k_absmax = 0.0`
- `block0_v_absmax = 0.0`
- `block0_object_delta_absmax = 0.0`

#### 结论

- 不是 `object_context` 没信息
- 是 `object_embedding` 前向直接输出了全 0，导致整个 object attention 塌掉

### 诊断 4：参数本体检查

#### 观测

- `object_embedding.0.weight/bias = 0`
- `object_embedding.2.weight/bias = 0`
- `blocks.0.object_cross_attn.q/k/v/o.base_layer.weight` 量级约 `1e-36`

#### 结论

- 问题不是 dtype 把正常值压没
- 是“checkpoint 中不存在的新增 object 分支参数”在加载后几乎变成全 0

## 当前根因判断

- WAN 主 checkpoint 不包含 object 分支
- 这些缺失参数在当前 `from_pretrained` 路径里没有得到可靠初始化
- 结果是：
  - `object_embedding` 前向输出全 0
  - `object_cross_attn` 的 `q/k/v` 全 0
  - 主损失对 object 分支只有极弱的末端偏置梯度
  - `object_context` 无法形成有效梯度回传

## 当前修复方案

- 在加载 WAN checkpoint 之后，显式重初始化缺失的 object 分支参数：
  - `object_embedding`
  - 每个 block 的 `object_cross_attn`
  - 每个 block 的 `norm4`
  - 每个 block 的 `object_gate`

### 初始化策略

- `object_embedding`：按 upstream `text_embedding/object_embedding` 逻辑，用 `normal_(std=0.02)`
- `object_cross_attn.q/k/v/o`：按 upstream 线性层逻辑，用 `xavier_uniform_`
- `norm4.weight=1, norm4.bias=0`
- `object_gate=0.1`

## 下一步验证项

- 重新跑 smoke，确认：
  - `object_embedding_out_absmax > 0`
  - `block0_object_delta_absmax > 0`
  - `object_context.grad_abs_sum > 0`
  - `object_adapter` 梯度不再为 0
- 如果通过，再上 `gpu6,7` 正式训练
