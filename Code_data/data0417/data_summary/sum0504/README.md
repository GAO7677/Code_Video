# sum0504

目录结构：`<split>/<simulator_type>/<object_count_bucket>/<collision_bucket>/samples.txt`

当前仅整理可稳定映射到以下规则的样本：
- split: `train / val / test`
- simulator_type: `rigid`
- object_count_bucket: `count_01 / count_02 / count_03_04`
- collision_bucket: `no_collision / env_only / obj_obj_only_c1 / obj_obj_only_c2plus / mixed_c1 / mixed_c2plus`

说明：
- 不移动真实样本文件夹，仅记录绝对路径。
- 每个叶子目录只保留 `samples.txt` 和 `summary.json`。
- 根目录和 split 目录下提供汇总 `summary.json`。
- 无法稳定映射到这套规则的样本不会被纳入，会记录在根目录 `summary.json` 的 `excluded_breakdown` 中。
