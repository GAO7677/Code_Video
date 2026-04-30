# organized_view_split_complexity_v1

这个目录是当前主用的路径清单版本，只记录样本文件夹路径。

分类依据：
- 第一层按视图形式划分：`raw` / `window`
- 第二层按用途划分：`train` / `test` / `benchmark`
- 第三层按运动复杂类型划分：`物体数量分组 + 碰撞类型`
- 对 `pair_2`、`few_3_4`、`many_5plus` 做了简化：不再区分 `obj-obj`、`obj-env`、`mixed`
  - 所有有碰撞的样本统一并到 `*_collision`
  - 无碰撞保留为 `*_none`
  - 无法判定的保留为 `*_unknown`
  - 例如 `pair_2_collision` 表示 2 个物体，窗口内存在碰撞
  - 例如 `single_1_none` 表示单物体，且窗口内无碰撞

总数据量：
- 全部样本：11212 条

各大类数据量：
- `raw/train`：6034 条
- `raw/test`：807 条
- `window/train`：3434 条
- `window/test`：813 条
- `window/benchmark/fixed24`：24 条
- `window/benchmark/validation100`：100 条

说明：
- 每个 `txt/json` 文件里都只保存样本文件夹路径
- `_all_samples.txt` / `_all_samples.json` 表示该目录下的全量路径合集
