# data_summary 可视化指令

## 数据目录与数量

### 1. `sum0504`

路径：

`/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504`

数量：

- total: 2468
- train: 1974
- test: 247
- val: 247

分层统计：

```text
train: 1974
  count_01
    no_collision: 348
    env_only: 776
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

test: 247
  count_01
    no_collision: 43
    env_only: 97
  count_02
    env_only: 4
    obj_obj_only_c2plus: 1
    mixed_c1: 40
    mixed_c2plus: 18
  count_03_04
    mixed_c1: 3
    mixed_c2plus: 41

val: 247
  count_01
    no_collision: 43
    env_only: 97
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

- `test/val` 直接从 Genesis raw  
  `/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid`
  做 heldout。

### 2. `version0515zoom_genesis_rigid`

路径：

`/home/gaoya/Code_Video/Code_data/data0417/data_summary/version0515zoom_genesis_rigid`

数量：

- total: 46
- train: 36
- test: 5
- val: 5

分层统计：

```text
train: 36
  count_02
    obj_obj_only_c2plus: 1
    mixed_c1: 8
    mixed_c2plus: 27

test: 5
  count_02
    obj_obj_only_c2plus: 1
    mixed_c1: 1
    mixed_c2plus: 3

val: 5
  count_02
    obj_obj_only_c2plus: 1
    mixed_c1: 1
    mixed_c2plus: 3
```

说明：

- 只记录路径，不移动原始数据。
- split 来自 Genesis raw `train/rigid` 的 heldout。
- 当前只保留非空分类目录。

### 3. `version0515zoom_genesis_rigid_stage1adapter_simple_window`

路径：

`/home/gaoya/Code_Video/Code_data/data0417/data_summary/version0515zoom_genesis_rigid_stage1adapter_simple_window`

数量：

- total: 38
- train: 31
- test: 4
- val: 3

分层统计：

```text
train: 31
  count_02
    no_collision: 31

test: 4
  count_02
    no_collision: 4

val: 3
  count_02
    no_collision: 3
```

说明：

- 这里记录的是 `version0515zoom_genesis_rigid/stage1adapter_simple_window/train/genesis` 的 window 样本。
- split 继承自对应 raw sample 在 `version0515zoom_genesis_rigid/raw_split_assignments.json` 中的 heldout。
- 当前这批 window 都被切成了 `no_collision`。

## 可视化重建

### 1. 重建 `sum0504` 页面

```bash
python \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --output_root /home/gaoya/portal_hub_sim/sum0504_portal \
  --summary_root /home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504 \
  --portal_title "sum0504 Portal" \
  --prefer_gif
```

### 2. 重建 `version0515zoom_genesis_rigid` 页面

```bash
python \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --output_root /home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_portal \
  --summary_root /home/gaoya/Code_Video/Code_data/data0417/data_summary/version0515zoom_genesis_rigid \
  --portal_title "version0515zoom_genesis_rigid Portal" \
  --prefer_gif
```

### 3. 重建 `version0515zoom_genesis_rigid_stage1adapter_simple_window` 页面

```bash
python \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --output_root /home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_stage1adapter_portal \
  --summary_root /home/gaoya/Code_Video/Code_data/data0417/data_summary/version0515zoom_genesis_rigid_stage1adapter_simple_window \
  --portal_title "version0515zoom_genesis_rigid Stage1Adapter Portal" \
  --prefer_gif
```

### 4. 启动本地静态服务

```bash
cd /
python3 -m http.server 8061 --bind localhost
```

## 页面地址

- `sum0504`:
  `http://localhost:8061/home/gaoya/portal_hub_sim/sum0504_portal/index.html`
- `version0515zoom_genesis_rigid`:
  `http://localhost:8061/home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_portal/index.html`
- `version0515zoom_genesis_rigid_stage1adapter_simple_window`:
  `http://localhost:8061/home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_stage1adapter_portal/index.html`

说明：

- 首页每个最小类别最多展示 `10` 条。
- 当前首页和详情页都优先展示 `gif`。
