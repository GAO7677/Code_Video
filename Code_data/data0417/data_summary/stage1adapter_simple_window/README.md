# stage1adapter_simple_window

- 这里只记录路径，不移动原始数据。
- 来源是 Genesis stage1adapter/train 下的 window 样本。
- 只保留 `no_collision` 和 `env_only`。
- split 继承自对应 raw source sample 在 `sum0504/raw_split_assignments.json` 中的 heldout 结果。
- 当前只保留非空分类目录。

## 数量

- train: 942
- test: 106
- val: 94
- total: 1142

## 叶子分类

- train/rigid/count_01/no_collision: 504
- train/rigid/count_01/env_only: 410
- train/rigid/count_02/env_only: 28
- test/rigid/count_01/no_collision: 62
- test/rigid/count_01/env_only: 40
- test/rigid/count_02/env_only: 4
- val/rigid/count_01/no_collision: 46
- val/rigid/count_01/env_only: 48
