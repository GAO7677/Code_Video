# genesis_rigid_data 整理提案

## 当前问题

- 所有脚本几乎都平铺在根目录，生成、修复、可视化、portal、wrapper 混在一起。
- 一部分脚本是稳定入口，一部分是底层 backend，一部分是一次性修复脚本，目前边界不清晰。
- `__pycache__` 也在目录里，噪声较大。
- 部分旧实验脚本仍保留在根目录，容易和当前主流程混淆。

## 建议目录结构

建议按“入口 / backend / 修复 / 可视化 / wrapper / 配置”拆分：

```text
genesis_rigid_data/
  docs/
    README.md
    AAA_data0417.md
    REORG_PROPOSAL.md
  configs/
    physinone_benchmark_taxonomy.json
    try1_physxnet_articulation_mpm0417.json
  core/
    utils_io.py
    scene_templates.py
    sample_bucket_labels.py
    physinone_benchmark_taxonomy.py
  generators/
    generate_physxnet_train_rigid_multi.py
    generate_rigid_benchmark.py
    generate_video_captions.py
    try1_physxnet_articulation_mpm0417.py
    try1_physxnet_articulation_mpm0417_textured_rt.py
    try1_physxnet_benchmark.py
  repair/
    backfill_*.py
    fix_environment_collision_onsets.py
    regenerate_affected_benchmarks.py
    rebuild_sum0504_index.py
    rebuild_stage1adapter_*.py
    filter_single_object_motion_cases.py
    list_zero_gravity_counterfactual_tasks.py
    audit_benchmark_inertial_origins.py
  inspect/
    inspect_physics_sample.py
    batch_inspect_physics_samples.py
    visualize_sample.py
    validate_saved_dataset_states.py
    validate_energy_cases.py
  portals/
    export_rigid_init_scene_html.py
  runs/
    *.sh
  legacy/
    仅保留确认还需要的旧实验脚本
```

## 当前稳定主入口

这些脚本我建议优先保留在清晰位置，并保证引用路径不变或同步修复：

- `generate_physxnet_train_rigid_multi.py`
- `generate_rigid_benchmark.py`
- `generate_video_captions.py`
- `rebuild_sum0504_index.py`
- `rebuild_stage1adapter_genesis_eval_splits.py`
- `rebuild_stage1adapter_path_summary.py`
- `validate_saved_dataset_states.py`
- `visualize_sample.py`

## 当前共享模块

这些脚本已经被多处复用，建议单独归到 `core/`：

- `utils_io.py`
- `scene_templates.py`
- `sample_bucket_labels.py`
- `physinone_benchmark_taxonomy.py`

## 候选删除项

以下内容我暂时没有删除，等你确认后再执行：

### 可以直接删除

- `__pycache__/`
  - 纯缓存文件，不应进入整理后的代码目录。

### 候选归档或保留

这些脚本目前在目录内没有被其他脚本直接 import，但其中有些是“历史主入口”而不是垃圾文件；我建议优先归到 `legacy/`，不要直接删除：

- `try3_dataset_3_rigid_genesis0417.py`
  - 这是早期 Genesis rigid 数据集的重要生成入口，不应直接删除。
  - 目前它更像“历史主生成器”，建议移动到 `legacy/` 或 `generators/legacy/`，并在 README 中明确它对应旧版数据流。
- 以下 4 个低优先级一次性工具已确认删除：
  - `build_physxnet_mpm_gallery.py`
  - `watch_multiobject_preview_autobuild.py`
  - `build_spotcheck_validation_wrappers.py`
  - `generate_counterfactual_gallery_gifs.py`
- 以下 `build_*portal.py / build_*gallery.py / build_*compare.py` 已确认删除：
  - `build_case_topdown_compare.py`
  - `build_collision_complexity_bucket_portal.py`
  - `build_collision_trajectory_compare.py`
  - `build_counterfactual_rgb_gallery.py`
  - `build_object_case_rgb_gallery.py`
  - `build_recent_sample_motion_bucket_portal.py`
  - `build_simple_collision_event_portal.py`
- 以下一次性 scene3d / collision summary 工具已确认删除：
  - `build_main_collision_summary.py`
  - `build_physxnet_case_scene3d.py`
  - `build_physxnet_object_scene3d.py`
- `export_rigid_init_scene_html.py` 暂保留：
  - 仍被 `batch_inspect_physics_samples.py` 直接调用，不能直接删除。

## 我建议的执行顺序

### 第一阶段：低风险，已适合直接做

- 给所有代码文件补首部用途说明。
- 保留现有文件路径，先补文档与分组方案。

### 第二阶段：需要你确认后再做

- 删除 `__pycache__/`
- 把 `.sh` 移到 `runs/`
- 把共享模块移到 `core/`
- 把生成入口移到 `generators/`
- 保留仍有价值的 portal / scene 可视化脚本，其余已删
- 把修复与回填脚本移到 `repair/`
- 把旧主入口 / 旧实验脚本移到 `legacy/`
- 同步修正 import、README 和运行命令

## 当前建议

如果你同意，我下一步会做：

1. 非破坏性重组
   - 新建上述子目录
   - 先移动 `docs/ configs/ runs/`
   - 再移动 `core/ generators/ repair/ inspect/ portals/`
   - 同步修复所有 import 与 README

2. 删除确认
   - 先只删 `__pycache__/`
   - 其余候选脚本等你逐项确认
