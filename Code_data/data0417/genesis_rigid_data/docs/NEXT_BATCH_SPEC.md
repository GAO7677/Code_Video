# 下一批 Genesis Rigid 数据生成规范

## 目标

这一版规范的目标不是“继续多生成一些样本”，而是把数据生产链拆成两条明确的路线：

- `Stage1-Simple`
  - 用于 state-conditioned adapter / 简单运动建模
  - 重点是干净、可切窗、状态与视频对应关系清晰
- `Stage2-Full`
  - 用于一般物理视频预测 / 复杂交互建模
  - 重点是覆盖复杂碰撞、遮挡、环境交互、多样性

不要再用同一套 case 协议同时服务这两个目标。

---

## 一、数据集拆分原则

### 1. Stage1-Simple

允许的样本类型：

- `count_01 / no_collision`
- `count_01 / env_only`
- `count_02 / no_collision`

不允许：

- window 内任何 `obj-obj contact`
- 初始帧已经存在 `sustained_contact`
- 物体一开始就互相贴住
- 严重遮挡导致主物体状态歧义

说明：

- `count_02 / env_only` 不建议保留，除非两个物体始终分离，只是其中一个和环境接触。
- 如果两个物体最终会发生碰撞，但切出的 window 完全位于碰撞前，且 window 内没有任何 obj-obj contact，可以作为 `count_02 / no_collision` 候选。

### 2. Stage2-Full

允许的样本类型：

- `count_02 / obj_obj_only_c1`
- `count_02 / obj_obj_only_c2plus`
- `count_02 / mixed_c1`
- `count_02 / mixed_c2plus`
- `count_03_04 / mixed_*`

目标：

- 保留复杂接触、碰撞、多次反弹、环境交互
- 不要求每条样本都能切出干净的 adapter window

---

## 二、初始条件硬约束

这是下一批数据最重要的改动。

### 1. `count_02` 初始间距

生成时必须保证：

- 两个物体在 `frame 0` 不接触
- 两个物体在前 `N_init_free` 帧不接触
- 建议 `N_init_free >= 8`

推荐同时做两种检查：

- 几何检查
  - 初始 AABB / 包围球 / 凸包最小距离大于阈值
- 动力学检查
  - 前 `N_init_free` 帧 contact graph 全 0

### 2. 物体与环境初始状态

区分两种情况：

- 合法静置：
  - 物体稳定放在地面上
  - 可以有静态支撑关系
- 非法初始穿插 / 压入：
  - 出生时就发生异常接触
  - 删掉

对于 `Stage1-Simple / count_01 / env_only`：

- 允许物体一开始放在地面上
- 但如果目标是“下落再碰撞”，则初始帧不能已接触地面

### 3. striker 类

如果有 striker：

- striker 初始位置不能贴着主物体
- striker 到主物体的最小初始距离要大于阈值
- 需要至少留出一段清晰的接近阶段

---

## 三、case 设计重组

不要再依赖“少量 case + 随机扰动”自然长出想要的 bucket。

### 1. `count_01`

建议基础 case：

- `drop_clean`
  - 单物体竖直下落
- `projectile_clean`
  - 单物体抛物线
- `slide_entry`
  - 单物体从侧边进入
- `ground_bounce_once`
  - 单物体落地弹跳一次

### 2. `count_02`

建议基础 case：

- `pair_parallel_nocollision`
  - 两物体共视野，同向或异向运动，但轨迹分离
- `pair_staggered_nocollision`
  - 两物体先后经过，但不接触
- `pair_single_hit`
  - 明确的一次碰撞
- `pair_multi_hit`
  - 多次碰撞或碰后再碰环境

### 3. 不建议继续沿用的坏模式

- 初始就接触的 `static_center/static_left/static_right`
- 视觉上像静止堆叠，实际从前几帧就有持续接触
- 用这种 case 去切 Stage1 window，产率会一直很差

---

## 四、相机与构图规范

相机不要继续“靠感觉微调”，改成模板化。

### 1. 必须满足的画面约束

- 初始帧所有关键物体完整在画面内
- 地面必须可见
- 主物体面积占比在合理区间
- `count_02` 时两个物体都可见

推荐阈值：

- 主物体初始面积占比：`8% ~ 30%`
- 任一关键物体最小面积占比：`>= 4%`
- 初始帧边界留白：bbox 到图像边缘至少 `3%` 宽度

### 2. 相机模板

建议固定三套：

- `cam_single_clean`
  - 单物体，偏正视，轻微俯视
- `cam_pair_wide`
  - 双物体共视野，保证两者完整入框
- `cam_pair_collision`
  - 双物体碰撞细节，更紧，但仍要保证初始完整

不要为每个 case 随机大量改相机参数。

---

## 五、scale 与物理一致性

优先顺序必须改成：

1. 调相机距离 / FOV
2. 调物体初始深度
3. 最后才少量调物体 scale

不要默认依赖 scale 放大物体。

### 1. scale 规则

- 默认保持物体原始物理尺寸
- 只有极小物体才允许放大
- 放大倍率需要记录到 metadata

### 2. metadata 必记字段

- `scale_factor_applied`
- `camera_template_id`
- `min_bbox_area_ratio`
- `max_bbox_area_ratio`

---

## 六、速度与时长

当前一批数据一个明显问题是视觉速度不稳定。

### 1. 速度分桶建议

如果保留 striker：

- `< 4 m/s`: 40%
- `4 ~ 5 m/s`: 40%
- `> 5 m/s`: 20%

### 2. 真实时长

不要通过“重复帧慢放”伪造物理时长。

建议：

- 仿真步数足够密
- 导出时尽量用真实时间采样
- 可以做轻度 dense sampling
- 但不要让运动看起来像不连续抽帧或假慢放

### 3. 尾部留白

如果运动在 2 秒内结束，不要硬导出 3 秒大量静止尾帧。

建议：

- 自动裁掉长静止尾部
- 只保留少量恢复稳定后的冗余帧

---

## 七、QA 与过滤字段

下一批样本必须在生成后自动记录 QA 标签，后续切子集时直接用。

建议每个样本至少写入：

- `initial_contact_free`
- `obj_obj_contact_in_first_k_frames`
- `env_contact_in_first_k_frames`
- `has_obj_obj_contact_full`
- `has_env_contact_full`
- `all_objects_visible_in_first_k_frames`
- `main_object_visible_ratio`
- `min_bbox_area_ratio`
- `max_bbox_area_ratio`
- `camera_template_id`
- `scale_factor_applied`
- `invalid_reason`

对于 window：

- `has_obj_obj_contact_in_window`
- `has_env_contact_in_window`
- `first_new_collision_onset`
- `window_is_precollision_clean`

---

## 八、推荐的数据量配额

### 1. Stage1-Simple

建议目标配额：

- `count_01 / no_collision`: 40%
- `count_01 / env_only`: 40%
- `count_02 / no_collision`: 20%

前提：

- `count_02` 必须是真正无接触的双物体共视野

### 2. Stage2-Full

建议目标配额：

- `count_02 / single collision`: 25%
- `count_02 / multi collision`: 25%
- `count_02 / mixed`: 25%
- `count_03_04 / mixed`: 25%

---

## 九、当前 version0515zoom 的问题总结

当前这批：

- 几乎全是 `count_02`
- 大量样本前几帧就存在 `sustained_contact`
- 无法切出合格的 Stage1 window
- 所以 `stage1adapter_simple_train` 最终为 `0`

这批数据更适合作为：

- 小规模 `Stage2-Full` 原型数据

不适合作为：

- `Stage1-Simple` adapter 数据源

---

## 十、后续落地顺序

建议按这个顺序改代码和数据生产链：

1. 先加初始接触 QA
   - 保证 `count_02` 前 `N` 帧无 obj-obj contact
2. 固定相机模板
   - 先稳定构图，再谈多样性
3. 重写 `count_01 / count_02` case 设计
   - 明确生成 `no_collision / env_only / single_hit / multi_hit`
4. 生成后写全 QA 标签
   - 后续切子集直接筛，不再手工追视频
5. 最后再补渲染真实感
   - 材质、灯光、背景优化放后面

---

## 十一、最短执行版本

如果只允许做最小改动，优先做这三件事：

- `count_02` 前 `8` 帧禁止 obj-obj contact
- 固定 `pair_wide` 相机模板
- 单独生成一批 `count_01 no_collision/env_only + count_02 no_collision`

只做这三件事，下一批数据质量就会明显比当前版本高。
