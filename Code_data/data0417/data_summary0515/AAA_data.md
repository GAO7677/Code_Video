# data_summary0515 可视化指令

## 数据目录与数量

### 1. `version0515zoom_genesis_rigid_stage1adapter_simple_train`

路径：

`/home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version0515zoom_genesis_rigid_stage1adapter_simple_train`

数量：

- train: 40

分层统计：

```text
train: 40
  interaction_pair_plus_dynamic
    count_02
      none
        simple: 11
        static: 27
    invalid_by_qa
      none
        simple: 1
        static: 1
```

说明：

- 这里只记录路径，不移动原始数据。
- 来源是 `/data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid/stage1adapter/train/genesis/rigid`。
- 叶子分组按 `scene_composition / count_bucket_path / collision_bucket / motion_complexity`。
- `invalid_by_qa` 路径会被原样保留，不会并入正常 `count_02`。

## 可视化重建

### 1. 重建 `version0515zoom_genesis_rigid_stage1adapter_simple_train` 页面

```bash
python \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --output_root /home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_stage1adapter_simple_train_portal \
  --summary_root /home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version0515zoom_genesis_rigid_stage1adapter_simple_train \
  --portal_title "version0515zoom_genesis_rigid Stage1Adapter Simple Train Portal" \
  --prefer_gif
```

### 2. 启动本地静态服务

```bash
cd /
python3 -m http.server 8062 --bind localhost
```

## 页面地址

- `version0515zoom_genesis_rigid_stage1adapter_simple_train`:
  `http://localhost:8062/home/gaoya/portal_hub_sim/version0515zoom_genesis_rigid_stage1adapter_simple_train_portal/index.html`

说明：

- 首页每个最小类别最多展示 `10` 条。
- 当前首页和详情页都优先展示 `gif`。

## 构建命令

### 1. raw train -> stage1adapter simple train

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/repair/build_stage1adapter_simple_dataset.py \
  --raw_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid \
  --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid/stage1adapter \
  --overwrite
```

### 2. stage1adapter simple train -> data summary0515

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/repair/build_stage1adapter_simple_path_summary.py \
  --dataset_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid/stage1adapter \
  --output_root /home/gaoya/Code_Video/Code_data/data0417/data_summary0515/version0515zoom_genesis_rigid_stage1adapter_simple_train \
  --overwrite
```
