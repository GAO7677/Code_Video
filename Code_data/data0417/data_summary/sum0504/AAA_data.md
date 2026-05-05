# sum0504

目录结构：`<split>/<simulator_type>/<object_count_bucket>/<collision_bucket>/samples.txt`

当前仅整理可稳定映射到以下规则的样本：
- split: `train / val / test`
- simulator_type: `rigid`
- object_count_bucket: `count_01 / count_02 / count_03_04`
- collision_bucket: `no_collision / env_only / obj_obj_only_c1 / obj_obj_only_c2plus / mixed_c1 / mixed_c2plus`
  - no_collision：没有碰撞
  - env_only：只有物体和环境碰撞
  - obj_obj_only_c1：只有物体和物体碰撞，且碰撞 1 次
  - obj_obj_only_c2plus：只有物体和物体碰撞，且碰撞 2 次及以上
  - mixed_c1：既有物体-环境碰撞，也有物体-物体碰撞，总碰撞次数为 1
  - mixed_c2plus：既有物体-环境碰撞，也有物体-物体碰撞，总碰撞次数为 2 次及以上
说明：
- 不移动真实样本文件夹，仅记录绝对路径。
- 每个叶子目录只保留 `samples.txt` 和 `summary.json`。
- 根目录和 split 目录下提供汇总 `summary.json`。
- 无法稳定映射到这套规则的样本不会被纳入，会记录在根目录 `summary.json` 的 `excluded_breakdown` 中。
- Genesis 自建样本的样本级元数据文件统一使用 `meta.json`；本目录下的 `samples.txt` 只记录样本文件夹路径，不直接记录 `meta.json / metadata.json` 文件名。

## 可视化与重建指令

重建 `sum0504` 路径索引：

```bash
source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan
python /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/rebuild_sum0504_index.py
```

重建 `sum0504` 总览与样本详情页：

```bash
kport 8048
python /home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py
```

页面输出位置：

`/home/gaoya/portal_hub_sim/sum0504_portal/index.html`

本地访问地址：

`http://127.0.0.1:8048/home/gaoya/portal_hub_sim/sum0504_portal/index.html`

如果本地静态文件服务未启动，可执行：

```bash
cd /
python3 -m http.server 8048
```


