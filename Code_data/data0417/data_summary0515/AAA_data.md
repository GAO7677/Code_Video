# data_summary0515 说明

## 当前主目录保留项

### 1. `version0515zoom_genesis_rigid_full`

路径：

`/home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version0515zoom_genesis_rigid_full`

数据量：

- total: `46`
- train: `36`
- test: `5`
- val: `5`

分层统计：

```text
train
  rigid
    count_02
      obj_obj_only_c2plus: 1
      mixed_c1: 8
      mixed_c2plus: 27

test
  rigid
    count_02
      obj_obj_only_c2plus: 1
      mixed_c1: 1
      mixed_c2plus: 3

val
  rigid
    count_02
      obj_obj_only_c2plus: 1
      mixed_c1: 1
      mixed_c2plus: 3
```

说明：

- 这套整理对应的数据源是：
  `/data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid/train/rigid`
- 这是基于现存 raw 数据重建的总目录 summary。
- split 是直接从 raw `train/rigid` 做 heldout。

### 2. `version0515zoom_genesis_rigid_stage1adapter_simple_train`

路径：

`/home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version0515zoom_genesis_rigid_stage1adapter_simple_train`

数据量：

- total: `0`
- train: `0`

说明：

- 这套整理对应的数据源是：
  `/data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid/`
- 当前按最终筛选标准，这批数据没有合格的 stage1adapter 样本。
- 规则：
  - `count_01` 保持旧规则；
  - `count_02` 允许两个物体共视野；
  - 但保留下来的 window 内不能出现任何物体-物体接触；
  - 可以有物体-环境碰撞；
  - 初始就存在的 `sustained_contact` 也视为不合格。
- 因为这批 `version0515zoom` 的 `count_02` raw 样本都包含物体-物体接触，所以最终全部被过滤。

## 已归档项

- `version_1_genesis_rigid_data_all_cases_sum0504_like`
  已移到：
  `/home/gaoya/Code_Video/Code_data/data0417/data_summary0515/archive_invalid/version_1_genesis_rigid_data_all_cases_sum0504_like`
- 原因：
  其对应原始数据根目录已被删除，记录的样本路径全部失效，不再保留在主目录下。

## 可视化重建

### 1. 重建 `version0515zoom_genesis_rigid_full` 页面

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --output_root /home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_full_portal \
  --summary_root /home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version0515zoom_genesis_rigid_full \
  --portal_title "version0515zoom_genesis_rigid Full Portal" \
  --prefer_gif
```

### 2. 重建 `version0515zoom_genesis_rigid_stage1adapter_simple_train` 页面

```bash
/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --output_root /home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_stage1adapter_simple_train_portal \
  --summary_root /home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version0515zoom_genesis_rigid_stage1adapter_simple_train \
  --portal_title "version0515zoom_genesis_rigid Stage1Adapter Simple Train Portal" \
  --prefer_gif
```

## 启动本地服务

```bash
cd /
python3 -m http.server 8672 --bind localhost
```

## 页面地址

- `version0515zoom_genesis_rigid_full`
  `http://localhost:8672/home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_full_portal/index.html`
- `version0515zoom_genesis_rigid_stage1adapter_simple_train`
  `http://localhost:8672/home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_stage1adapter_simple_train_portal/index.html`

说明：

- `full` 页面有样本内容。
- `stage1adapter` 页面当前会显示“无样本”提示，不再指向任何不存在的路径。
