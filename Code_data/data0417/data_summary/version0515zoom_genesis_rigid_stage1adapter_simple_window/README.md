# version0515zoom_genesis_rigid_stage1adapter_simple_window

- 这里只记录路径，不移动原始 window 数据。
- 来源是 `version0515zoom_genesis_rigid/stage1adapter_simple_window/train/genesis`。
- split 继承自对应 raw source sample 在 `version0515zoom_genesis_rigid/raw_split_assignments.json` 中的 heldout 结果。
- collision bucket 读取 window 自身的 `pair_meta.json -> window_interactions.future_window.collision_type_bucket`。
- 当前只保留非空分类目录。

## 数量

- train: 31
- test: 4
- val: 3
- total: 38

## 叶子分类

- train/rigid/count_02/no_collision: 31
- test/rigid/count_02/no_collision: 4
- val/rigid/count_02/no_collision: 3
