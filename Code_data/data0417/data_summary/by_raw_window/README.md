# by_raw_window

这个目录记录按 `raw/window -> train/test/benchmark -> 复杂度叶子类别` 重建后的样本路径清单。

分类依据：
- 第一层按当前 metadata 的 `view_type` 划分：`raw` / `window`。
- 第二层优先按真实路径组织划分：`train` / `test` / `benchmark`。
- 第三层按 `物体数量分组 + 简化碰撞类型` 划分。
- `物体数量分组` 统一映射为：`single_1` / `pair_2` / `few_3_4` / `many_5plus` / `unknown`。
- `简化碰撞类型` 统一映射为：`none` / `env_only` / `collision` / `unknown`。
- 其中 `pair_2`、`few_3_4`、`many_5plus` 不再区分 `obj-obj` 与 `obj-env`，所有已知碰撞统一并到 `*_collision`。
- `single_1` 仍保留 `single_1_env_only`，因为单物体与环境接触是主要碰撞形式。
- 对 window 样本，如果自身缺少 `collision_type_bucket`，会优先回看 `source_sample_dir` 对应 raw 样本的 metadata。
- 本次重建不复用旧类别文件名中的历史判定，只使用当前 `meta.json` / `metadata.json` 和 source raw metadata。

总样本数：`11212`

按视图统计：
- `raw`：6034
- `window`：5178

按目录统计：
- `raw/train`：6034
- `window/benchmark/fixed24`：24
- `window/benchmark/validation100`：100
- `window/test`：1620
- `window/train`：3434

按叶子类别统计：
- `raw/train/few_3_4_collision`：924
- `raw/train/many_5plus_collision`：2893
- `raw/train/pair_2_collision`：1072
- `raw/train/pair_2_unknown`：33
- `raw/train/single_1_env_only`：546
- `raw/train/single_1_none`：560
- `raw/train/single_1_unknown`：6
- `window/benchmark/fixed24/few_3_4_collision`：1
- `window/benchmark/fixed24/many_5plus_unknown`：6
- `window/benchmark/fixed24/pair_2_collision`：3
- `window/benchmark/fixed24/single_1_none`：2
- `window/benchmark/fixed24/unknown_unknown`：12
- `window/benchmark/validation100/few_3_4_collision`：12
- `window/benchmark/validation100/many_5plus_unknown`：25
- `window/benchmark/validation100/pair_2_collision`：8
- `window/benchmark/validation100/single_1_env_only`：3
- `window/benchmark/validation100/single_1_none`：2
- `window/benchmark/validation100/unknown_unknown`：50
- `window/test/few_3_4_unknown`：12
- `window/test/many_5plus_unknown`：1580
- `window/test/single_1_env_only`：7
- `window/test/single_1_none`：17
- `window/test/single_1_unknown`：4
- `window/train/few_3_4_collision`：505
- `window/train/many_5plus_collision`：3
- `window/train/pair_2_collision`：694
- `window/train/pair_2_unknown`：66
- `window/train/single_1_env_only`：1040
- `window/train/single_1_none`：1114
- `window/train/single_1_unknown`：12

说明：
- 每个 `json/txt` 文件都只保存样本文件夹绝对路径。
- `_all_samples` 表示该目录下的全量路径合集。
