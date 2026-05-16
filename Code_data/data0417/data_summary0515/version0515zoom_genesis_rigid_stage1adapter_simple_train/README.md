# version0515zoom_genesis_rigid_stage1adapter_simple_train

- 这里只记录路径，不移动原始 stage1adapter 数据。
- 来源是 `version0515zoom_genesis_rigid/stage1adapter/train/genesis/rigid`。
- 叶子分组按 `scene_composition / count_bucket_path / collision_bucket / motion_complexity`。
- `invalid_by_qa` 路径会被原样保留，不会并入正常 count bucket。

## 数量

- train: 40

## 叶子分类

- train/rigid/interaction_pair_plus_dynamic/count_02/none/simple: 11
- train/rigid/interaction_pair_plus_dynamic/count_02/none/static: 27
- train/rigid/interaction_pair_plus_dynamic/invalid_by_qa/none/simple: 1
- train/rigid/interaction_pair_plus_dynamic/invalid_by_qa/none/static: 1
