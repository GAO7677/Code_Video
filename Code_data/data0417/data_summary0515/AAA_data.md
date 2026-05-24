# data_summary0515 说明

## 当前有效目录

### 1. `version_1_genesis_rigid_data_all_cases_sum0504_like`

路径：

`/home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version_1_genesis_rigid_data_all_cases_sum0504_like`

数据量：

- total: `1064`
- train: `850`
- test: `107`
- val: `107`

分层统计：

```text
train
  rigid
    count_02
      env_only: 28
      obj_obj_only_c1: 1
      obj_obj_only_c2plus: 1
      mixed_c1: 324
      mixed_c2plus: 141
    count_03_04
      obj_obj_only_c2plus: 1
      mixed_c1: 22
      mixed_c2plus: 332

test
  rigid
    count_02
      env_only: 4
      obj_obj_only_c2plus: 1
      mixed_c1: 40
      mixed_c2plus: 18
    count_03_04
      mixed_c1: 3
      mixed_c2plus: 41

val
  rigid
    count_02
      env_only: 4
      obj_obj_only_c2plus: 1
      mixed_c1: 40
      mixed_c2plus: 18
    count_03_04
      mixed_c1: 3
      mixed_c2plus: 41
```

说明：

- 这是只基于 `count_02` 和 `count_03_04` 的 Genesis rigid summary。
- `count_01` 当前为空，因此不单列。

### 2. `version0515zoom_genesis_rigid_stage1adapter_simple_train`

路径：

`/home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version0515zoom_genesis_rigid_stage1adapter_simple_train`

数据量：

- total: `0`
- train: `0`

说明：

- 来源是 `version0515zoom_genesis_rigid/stage1adapter/train/genesis/rigid`。
- 当前按最终筛选标准，这批数据没有合格样本。
- 规则：
  - `count_01` 保持旧规则；
  - `count_02` 允许两个物体共视野；
  - 但保留下来的 window 内不能出现任何物体-物体接触；
  - 可以有物体-环境碰撞；
  - 初始就存在的 `sustained_contact` 也视为不合格。
- 因为这批 `version0515zoom` 的 `count_02` raw 样本都包含物体-物体接触，所以最终全部被过滤。

## 可视化重建

### 1. 重建 `version_1_genesis_rigid_data_all_cases_sum0504_like` 页面

```bash
python /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --output_root /home/gaoya/portal_hub_sim/data_summary0515_sum0504_like_portal \
  --summary_root /home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version_1_genesis_rigid_data_all_cases_sum0504_like \
  --portal_title "data_summary0515 sum0504-like Portal" \
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
python3 -m http.server 8668 --bind localhost
```

## 页面地址

- `version_1_genesis_rigid_data_all_cases_sum0504_like`
  `http://localhost:8668/home/gaoya/portal_hub_sim/data_summary0515_sum0504_like_portal/index.html`
- `version0515zoom_genesis_rigid_stage1adapter_simple_train`
  `http://localhost:8668/home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_stage1adapter_simple_train_portal/index.html`

说明：

- 首页每个最小类别最多展示 `10` 条。
- 首页主卡片只放主视频，其他内容放详情页。
