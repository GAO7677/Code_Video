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

### 2. `stage1adapter_simple_window`

路径：

`/home/gaoya/Code_Video/Code_data/data0417/data_summary/stage1adapter_simple_window`

数量：

- total: 1142
- train: 942
- test: 106
- val: 94

分层统计：

```text
train: 942
  count_01
    no_collision: 504
    env_only: 410
  count_02
    env_only: 28

test: 106
  count_01
    no_collision: 62
    env_only: 40
  count_02
    env_only: 4

val: 94
  count_01
    no_collision: 46
    env_only: 48
```

说明：

- 只保留 `no_collision + env_only` 的 Genesis window 样本。
- split 继承自对应 raw source sample 在 `sum0504` 中的 heldout 结果。

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

### 2. 重建 `stage1adapter_simple_window` 页面

```bash
python \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --output_root /home/gaoya/portal_hub_sim/stage1adapter_simple_window_portal \
  --summary_root /home/gaoya/Code_Video/Code_data/data0417/data_summary/stage1adapter_simple_window \
  --portal_title "stage1adapter_simple_window Portal" \
  --prefer_gif
```

### 3. 启动本地静态服务

```bash
cd /
python3 -m http.server 8049 --bind localhost
```

## 页面地址

- `sum0504`:
  `http://localhost:8049/home/gaoya/portal_hub_sim/sum0504_portal/index.html`
- `stage1adapter_simple_window`:
  `http://localhost:8049/home/gaoya/portal_hub_sim/stage1adapter_simple_window_portal/index.html`

说明：

- 首页每个最小类别最多展示 `10` 条。
- 当前首页和详情页都优先展示 `gif`。
