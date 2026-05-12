# data0417 常用命令

以下命令已按当前目录结构核对。

## 1. 重建 `sum0504` 路径索引

功能：按 `train/test/val + rigid + count bucket + collision bucket` 重写  
`/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504`

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/repair/rebuild_sum0504_index.py
```

## 2. 重建 `sum0504` 本地可视化页面

功能：读取 `sum0504` 下的 `samples.txt`，生成首页和样本详情页。

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py
```

只重建首页：

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --index_only
```

只看 `no_collision`：

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py \
  --output_root /home/gaoya/portal_hub_sim/sum0504_nocollision_portal \
  --collision_bucket no_collision
```

## 3. 启动本地静态服务

功能：把本地门户页挂到浏览器端口。

```bash
cd /
python3 -m http.server 8049 --bind localhost
```

常用地址：

- 完整版：`http://localhost:8049/home/gaoya/portal_hub_sim/sum0504_portal/index.html`
- 无碰撞：`http://localhost:8049/home/gaoya/portal_hub_sim/sum0504_nocollision_portal/index.html`

## 4. 生成 `train/rigid`

功能：从 PhysXNet 生成当前多物体 train 样本。

快速 smoke test：

```bash
bash /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/runs/run0417.sh
```

直接生成：

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generators/generate_physxnet_train_rigid_multi.py \
  --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/physxnet_train_rigid_multi \
  --num_samples 64 \
  --seed 20260419
```

## 5. 生成 `stage1adapter` window 子集

功能：从完整 Genesis 数据集中切出 Stage-1 window，输出到  
`preprocess_v1/stage1_subsets_v1`

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_train/train_0419/state_adapter/build_stage1_subsets.py \
  --dataset_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases \
  --out_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/stage1_subsets_v1 \
  --count_buckets count_01,count_02
```

## 6. 重建 `stage1adapter` 的 test / val 包装目录

功能：从现有 `stage1adapter/train/genesis/...` window 样本中，重建：

- `stage1adapter/test/genesis`
- `stage1adapter/benchmark/fixed24/genesis`
- `stage1adapter/benchmark/validation100/genesis`

并同步更新 manifest。

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/repair/rebuild_stage1adapter_genesis_eval_splits.py
```

## 7. 生成 held-out benchmark 并顺手切 stage1 子集

功能：生成 `stage1_heldout` benchmark；默认会调用上面的  
`state_adapter/build_stage1_subsets.py`

```bash
bash /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/runs/run_stage1_count01_benchmark.sh
```

说明：

- 旧文档里提到的 `build_by_raw_window_portal.py` 已不存在，不再使用。
- 现在推荐的可视化入口是 `rebuild_sum0504_portal_with_sample_pages.py`。
